/*
 * @file serve.rs
 * @brief Minimal OpenAI-compatible HTTP server for the TinyGPT demo
 * @author Kevin Thomas
 * @date 2025
 *
 * MIT License
 *
 * Copyright (c) 2025 Kevin Thomas
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//! FILE: serve.rs
//!
//! DESCRIPTION:
//! Minimal OpenAI-compatible HTTP server for the TinyGPT demo.
//!
//! BRIEF:
//! Serves POST /v1/chat/completions (and GET /health) with the standard
//! OpenAI chat.completion JSON shape, backed by the trained model + vocab.
//! Uses only std::net, no web framework or async runtime.

use crate::core::tiny_gpt::TinyGPT;
use crate::core::vocab::Vocab;
use serde::Deserialize;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Process boot epoch, recorded when the server starts.
static BOOT_EPOCH: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Incoming OpenAI chat completion request body.
#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: Option<String>,
    content: String,
}

/// Segments and encodes a question, skipping tokens not in the vocabulary.
fn encode_prompt(vocab: &Vocab, content: &str) -> Vec<usize> {
    let mut buf = String::new();
    let mut tokens: Vec<String> = vec![];
    for ch in content.chars() {
        if ch.is_whitespace() {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            continue;
        }
        if ch.is_ascii() {
            buf.push(ch);
        } else {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(ch.to_string());
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
        .iter()
        .filter_map(|t| vocab.w2i.get(t).copied())
        .collect()
}

/// Builds an OpenAI chat.completion response for the given content.
fn completion_json(
    vocab: &Vocab,
    model: &TinyGPT<crate::core::linear::Linear>,
    content: &str,
    max_tokens: usize,
    learner: &std::sync::Mutex<crate::core::learn::Learner>,
) -> String {
    let prompt = build_prompt(vocab, content);
    let prompt_tokens = prompt.len();
    let end = vocab.encode("<END>");
    let text = answer_text(vocab, model, &prompt, content, max_tokens, end, learner);
    let completion_tokens = text.split_whitespace().count();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = serde_json::json!({
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "created": created,
        "model": "demo_chat",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    });
    body.to_string()
}

/// Pure routing decision via the unified RouterStack.
///
/// Always returns an answer: the first confident router, otherwise a polite
/// "don't know" fallback. On a miss the query's unknown words are auto-learned
/// into the shared learner (the system updates its own parameters).
fn decide_path(content: &str, learner: &std::sync::Mutex<crate::core::learn::Learner>) -> String {
    // /tool:* commands are handled by the tool dispatcher (ast-grep etc.)
    if let Some(tool_ans) = crate::core::codex::run_tool(content) {
        return tool_ans;
    }
    let fallback = "（这个问题我暂时还不会，但我会学习这个词，下次试试。可以换种问法，如：我需要打开文件 / 泛型是什么 / 如何定义函数）";
    let stack = crate::core::router::RouterStack::new();
    match stack.route(content) {
        Some(a) => a.text,
        None => {
            if let Ok(mut l) = learner.lock() {
                l.learn_from_miss(content);
                l.save();
            }
            fallback.to_string()
        }
    }
}

/// Produces the answer through the unified routing stack.
///
/// The model path is intentionally unused from the web chat: a memorizer-only
/// model answers unknown questions with echoes/garbage, so the knowledge-first
/// router (with honest fallbacks) serves instead.
fn answer_text(
    vocab: &Vocab,
    model: &TinyGPT<crate::core::linear::Linear>,
    prompt: &[usize],
    content: &str,
    max_tokens: usize,
    end: usize,
    learner: &std::sync::Mutex<crate::core::learn::Learner>,
) -> String {
    let _ = (vocab, model, prompt, max_tokens, end);
    decide_path(content, learner)
}

#[cfg(test)]
mod tests {
    use super::decide_path;

    /// Action questions must return code, not dictionary explanations.
    #[test]
    fn test_action_gets_code() {
        assert!(
            decide_path("怎么删除文件", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("remove_file"),
            "怎么删除文件 应给代码"
        );
        assert!(
            decide_path("如何定义结构体", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("struct S"),
            "如何定义结构体 应给代码"
        );
        assert!(
            decide_path("声明函数", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("fn f"),
            "声明函数 应给 fn 代码（当前误入释义）"
        );
        assert!(
            decide_path("我需要打开文件", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("File::open"),
            "我需要打开文件 应给代码"
        );
        assert!(
            decide_path("怎么生成随机数", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("rand"),
            "怎么生成随机数 应给 cargo add rand"
        );
    }

    /// Explanation questions must return the keyword's meaning.
    #[test]
    fn test_tool_ast_grep_routes() {
        let ans = decide_path(
            "/tool:ast-grep serde_json 帮我找找怎么解析json",
            &std::sync::Mutex::new(crate::core::learn::Learner::default()),
        );
        assert!(ans.contains("ast-grep serde_json"), "got {}", ans);
        assert!(
            ans.contains("from_str") || ans.contains("parse") || ans.contains("to_string"),
            "got {}",
            ans
        );
    }

    /// Explanation questions must return the keyword's meaning.
    #[test]
    fn test_explain_gets_meaning() {
        assert!(
            decide_path("泛型是什么", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("类型参数化"),
            "泛型是什么 应给释义"
        );
        assert!(
            decide_path("结构体是什么", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("struct"),
            "结构体是什么 应给释义"
        );
    }

    /// Bare keywords get the dictionary meaning, not a code answer.
    #[test]
    fn test_bare_keyword_explains() {
        assert!(
            decide_path("泛型", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("类型参数化"),
            "泛型 应给释义"
        );
        assert!(
            decide_path("字符串", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("一串字符"),
            "字符串 应给释义"
        );
    }

    /// Unknown / out-of-domain queries fall back to the model (None).
    #[test]
    fn test_unknown_falls_back() {
        assert!(decide_path("中午吃什么", &std::sync::Mutex::new(crate::core::learn::Learner::default())).contains("暂时"));
        
    }
}

/// Builds the model prompt: `问 : <content>` char-encoded (matching training).
fn build_prompt(vocab: &Vocab, content: &str) -> Vec<usize> {
    let mut prompt = encode_prompt(vocab, content);
    if prompt.first().map_or(true, |&t| vocab.i2w[&t] != "问") {
        let mut q = vec![vocab.encode("问"), vocab.encode(":")];
        q.extend(prompt);
        prompt = q;
    }
    prompt
}

/// Streams an OpenAI-style SSE chat completion: one `data:` chunk per token.
///
/// # Details
/// The answer is generated greedily then emitted token-by-token with a small
/// delay and explicit flush, so the browser renders words progressively.
fn stream_completion(
    stream: &mut TcpStream,
    vocab: &Vocab,
    model: &TinyGPT<crate::core::linear::Linear>,
    content: &str,
    max_tokens: usize,
    learner: &std::sync::Mutex<crate::core::learn::Learner>,
) {
    let prompt = build_prompt(vocab, content);
    let end = vocab.encode("<END>");
    let text = answer_text(vocab, model, &prompt, content, max_tokens, end, learner);
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();

    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nCache-Control: no-cache\r\n\r\n";
    let _ = stream.write_all(header.as_bytes());

    let mut emit = |s: &mut TcpStream, data: String| {
        let _ = s.write_all(format!("data: {data}\n\n").as_bytes());
        let _ = s.flush();
    };
    emit(
        stream,
        serde_json::json!({"choices":[{"delta":{"role":"assistant"},"index":0}]}).to_string(),
    );
    for word in &words {
        let chunk = serde_json::json!({
            "choices":[{"delta":{"content": word },"index":0}]
        })
        .to_string();
        emit(stream, chunk);
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
    emit(
        stream,
        serde_json::json!({"choices":[{"delta":{},"finish_reason":"stop","index":0}]}).to_string(),
    );
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}

/// The embedded chat web page (streams via fetch + SSE).
const CHAT_HTML: &str = include_str!("chat.html");

/// Reads an HTTP request and returns its path and body.
fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut head = Vec::new();
    let mut buf = [0u8; 4096];
    let mut content_length = 0usize;
    let mut path = String::new();
    loop {
        let n = stream.read(&mut buf).unwrap_or(0);
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
        if let Some(pos) = head.windows(4).position(|w| w == b"\r\n\r\n") {
            let header = String::from_utf8_lossy(&head[..pos]).to_string();
            for line in header.lines() {
                if let Some(rest) = line.strip_prefix("GET ") {
                    path = rest.split_whitespace().next().unwrap_or("").to_string();
                } else if let Some(rest) = line.strip_prefix("POST ") {
                    path = rest.split_whitespace().next().unwrap_or("").to_string();
                }
                if let Some(rest) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = rest.trim().parse().unwrap_or(0);
                }
            }
            let body_start = pos + 4;
            let mut body = head[body_start..].to_vec();
            while body.len() < content_length {
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&buf[..n]);
            }
            body.truncate(content_length);
            return (path, body);
        }
        if head.len() > 1 << 16 {
            break;
        }
    }
    (path, vec![])
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn handle(
    mut stream: TcpStream,
    model: &TinyGPT<crate::core::linear::Linear>,
    vocab: &Vocab,
    learner: &std::sync::Mutex<crate::core::learn::Learner>,
) {
    let (path, body) = read_request(&mut stream);
    match path.as_str() {
        "/" | "/index.html" => {
            respond(&mut stream, "200 OK", "text/html", CHAT_HTML.as_bytes())
        }
        "/health" => respond(&mut stream, "200 OK", "text/plain", b"ok"),
        "/time" => {
            // 时间接口：当前时间 + 启动历史
            let now = crate::core::time_log::now_epoch();
            let uptime = now.saturating_sub(*BOOT_EPOCH.get().unwrap_or(&now));
            let log = crate::core::time_log::default_log_path();
            let startups = crate::core::time_log::read_startups(&log);
            let body = serde_json::json!({
                "now_epoch": now,
                "now": crate::core::time_log::format_iso(now),
                "uptime_secs": uptime,
                "startups": startups,
            })
            .to_string();
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/toolchain" => {
            // 工具链状态接口：cargo/rustc 版本、最新稳定版、落后警告、启动历史
            let mut st = crate::core::time_log::toolchain_status();
            let now = crate::core::time_log::now_epoch();
            st.uptime_secs = now.saturating_sub(*BOOT_EPOCH.get().unwrap_or(&now));
            let body = serde_json::to_string(&st).unwrap_or_else(|_| "{}".to_string());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/v1/chat/completions" => {
            let req: ChatRequest = serde_json::from_slice(&body).unwrap_or(ChatRequest {
                model: None,
                messages: vec![],
                max_tokens: None,
                stream: None,
            });
            let content = req
                .messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let max_tokens = req.max_tokens.unwrap_or(40);
            if req.stream.unwrap_or(false) {
                stream_completion(&mut stream, vocab, model, &content, max_tokens, learner);
            } else {
                let json = completion_json(vocab, model, &content, max_tokens, learner);
                respond(&mut stream, "200 OK", "application/json", json.as_bytes());
            }
        }
        _ => respond(&mut stream, "404 Not Found", "text/plain", b"not found"),
    }
}

/// Starts the OpenAI-compatible server on the given port.
///
/// # Details
/// Blocks forever. Each connection is handled on a new thread; the model is
/// shared by reference and the vocabulary via `Arc`.
///
/// # Arguments
/// * `model` - Trained model to serve
/// * `vocab` - Shared vocabulary
/// * `port` - TCP port to bind
pub fn serve(model: TinyGPT<crate::core::linear::Linear>, vocab: Arc<Vocab>, port: u16) -> anyhow::Result<()> {
    let _ = BOOT_EPOCH.set(crate::core::time_log::now_epoch());
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    // shared auto-learner: unanswered queries teach new words
    let learner = std::sync::Arc::new(std::sync::Mutex::new(crate::core::learn::Learner::load()));
    println!("colibri-style OpenAI server on http://127.0.0.1:{port}  (POST /v1/chat/completions)");
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let model = model.clone();
        let vocab = Arc::clone(&vocab);
        let learner = std::sync::Arc::clone(&learner);
        std::thread::spawn(move || handle(stream, &model, &vocab, &learner));
    }
    Ok(())
}
