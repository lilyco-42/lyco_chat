/*
 * @file router.rs
 * @brief Unified routing stack: one interface over all knowledge routers
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

//! FILE: router.rs
//!
//! DESCRIPTION:
//! Unified routing stack over every knowledge source in the project.
//!
//! BRIEF:
//! DictRouter (dictionary meanings), ConceptRouter (canonical code),
//! WordBlockRouter (action blocks) and SemanticMoE (paraphrase intents) all
//! implement one `Router` trait. A prioritized `RouterStack` tries them in
//! order and returns the first confident answer, replacing the ad-hoc
//! hand-written decision chain with a pluggable, testable pipeline.

use crate::core::block_router::WordBlockRouter;
use crate::core::concept::ConceptRouter;
use crate::core::dict::DictRouter;
use crate::core::semantic::SemanticMoE;

/// One routed answer and where it came from.
#[derive(Debug, Clone)]
pub struct Answer {
    pub text: String,
    pub source: &'static str,
}

impl Answer {
    fn new(source: &'static str, text: String) -> Self {
        Self { text, source }
    }
}

/// Any knowledge router: maps a query to an answer it confidently knows.
pub trait Router {
    fn name(&self) -> &'static str;
    fn route(&self, query: &str) -> Option<Answer>;
}

/// Action verbs that signal a "do X" request (should get code, not meaning).
const ACTION_WORDS: [&str; 32] = [
    "需要", "怎么", "如何", "帮我", "想要", "想", "请", "编写", "创建", "删除",
    "打开", "读取", "写入", "生成", "发送", "连接", "定义", "遍历", "排序", "使用",
    "声明", "实现", "构建", "新建", "获取", "查找", "搜索", "调用", "移除", "格式化",
    "解析", "添加",
];

/// Whether a query asks for an action (code answer) rather than a meaning.
pub fn is_action_query(query: &str) -> bool {
    ACTION_WORDS.iter().any(|m| query.contains(m))
}

/// Dictionary meanings: explanation questions and bare keywords.
pub struct ExplainRouter {
    inner: DictRouter,
}

impl ExplainRouter {
    pub fn new() -> Self {
        Self {
            inner: DictRouter::new(),
        }
    }
}

impl Default for ExplainRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl Router for ExplainRouter {
    fn name(&self) -> &'static str {
        "词典释义"
    }
    fn route(&self, query: &str) -> Option<Answer> {
        // only claim explanation questions and bare keywords; action queries
        // (我需要打开文件 / 声明函数) must fall through to the code routers
        let is_explain = crate::core::dict::DictRouter::is_explain(query);
        if !is_explain && is_action_query(query) {
            // format-file questions ("怎么写 cargo.toml") still get the format
            if !query.to_lowercase().contains("cargo.toml") {
                return None;
            }
        }
        if is_explain {
            return self
                .inner
                .explain(query)
                .map(|text| Answer::new(self.name(), text));
        }
        // bare keyword: only accept when the matched dict term is actually
        // present in the query (智能合约 must NOT match 智能指针)
        let hit = self.inner.route(query).into_iter().next();
        if let Some(h) = hit {
            if query.contains(&h.definition.term) {
                let mut text = format!("{}：{}", h.definition.term, h.definition.meanings.join("；"));
                if !h.definition.synonyms.is_empty() {
                    text.push_str(&format!("（近义：{}）", h.definition.synonyms.join("、")));
                }
                return Some(Answer::new(self.name(), text));
            }
        }
        None
    }
}

/// Concept canonical code answers for action questions.
pub struct ConceptAnswerRouter {
    inner: ConceptRouter,
}

impl ConceptAnswerRouter {
    pub fn new() -> Self {
        Self {
            inner: ConceptRouter::new(),
        }
    }
}

impl Default for ConceptAnswerRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl Router for ConceptAnswerRouter {
    fn name(&self) -> &'static str {
        "概念代码"
    }
    fn route(&self, query: &str) -> Option<Answer> {
        // async queries must reach the block router (async fn block), not the
        // sync concept answer
        if query.contains("异步") || query.to_lowercase().contains("async") {
            return None;
        }
        self.inner
            .route(query)
            .map(|(i, _)| Answer::new(self.name(), self.inner.concepts[i].answer.to_string()))
    }
}

/// Word-level attention + action blocks (fn declaration vs call blocks).
pub struct BlockRouterAdapter {
    inner: WordBlockRouter,
}

impl BlockRouterAdapter {
    pub fn new() -> Self {
        Self {
            inner: WordBlockRouter::new(),
        }
    }
}

impl Default for BlockRouterAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Router for BlockRouterAdapter {
    fn name(&self) -> &'static str {
        "动作块"
    }
    fn route(&self, query: &str) -> Option<Answer> {
        self.inner
            .route(query)
            .map(|o| Answer::new(self.name(), o.code))
    }
}

/// Paraphrase intent routing to a canonical answer.
pub struct IntentRouter {
    inner: SemanticMoE,
}

impl IntentRouter {
    pub fn new() -> Self {
        Self {
            inner: SemanticMoE::new(),
        }
    }
}

impl Default for IntentRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl Router for IntentRouter {
    fn name(&self) -> &'static str {
        "近义意图"
    }
    fn route(&self, query: &str) -> Option<Answer> {
        self.inner
            .route(query)
            .map(|(i, _)| Answer::new(self.name(), self.inner.experts[i].answer.to_string()))
    }
}

/// Crate recommendation via transformer-style intent attention.
pub struct RecommendRouter {
    intent: crate::core::intent::IntentAttention,
}

impl RecommendRouter {
    pub fn new() -> Self {
        Self {
            intent: crate::core::intent::IntentAttention::new(),
        }
    }
}

impl Default for RecommendRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl Router for RecommendRouter {
    fn name(&self) -> &'static str {
        "专家推荐"
    }
    fn route(&self, query: &str) -> Option<Answer> {
        // conceptual questions (X 原理 / 机制 / 概念) ask for knowledge, not a
        // crate recommendation — defer to the honest fallback instead
        const CONCEPT_WORDS: &[&str] = &["原理", "机制", "概念", "本质", "工作方式", "怎么工作", "内部", "流程"];
        if CONCEPT_WORDS.iter().any(|w| query.contains(w)) {
            return None;
        }
        // 1) 标签神经元激活（crate 真实标签 + 中英同义词，最精确）
        let ta = crate::core::codex::TagActivation::load();
        if let Some((c, cat, act, fired)) = ta.best(query) {
            if act >= 1.0 {
                return Some(Answer::new(
                    self.name(),
                    format!("推荐: cargo add {c}（分类：{cat}；激活标签：{}）", fired.join("/")),
                ));
            }
        }
        // 2) transformer 意图注意力推断
        if let Some(it) = self.intent.infer(query) {
            let mut text = format!("推荐: cargo add {}（分类：{}）", it.crate_name, it.category);
            if !it.traits.is_empty() {
                text.push_str(&format!("；特质：{}", it.traits.join("/")));
            }
            text.push_str(&format!("；{}", it.description));
            return Some(Answer::new(self.name(), text));
        }
        // 3) 未知词 → 转译英文 → 实时 crates.io 搜索 → 学新领域
        // 先整句、再逐词、再双字组合（jieba 可能过度切分：区块链→区块/链）
        let jieba = jieba_rs::Jieba::new();
        let words: Vec<String> = jieba
            .cut(query, false)
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| s.chars().count() >= 2)
            .collect();
        let mut candidates: Vec<String> = vec![query.trim().to_string()];
        candidates.extend(words.iter().cloned());
        for pair in words.windows(2) {
            candidates.push(pair.join(""));
        }
        for c in candidates {
            if let Some((crate_name, desc, _)) = crate::core::codex::learn_and_recommend(&c) {
                let mut text = format!("推荐: cargo add {crate_name}（已学习新词 {c}）");
                if !desc.is_empty() {
                    text.push_str(&format!("；{desc}"));
                }
                return Some(Answer::new(self.name(), text));
            }
        }
        None
    }
}

/// Merged crate knowledge: name -> (description, tags) from the most-common
/// ecosystem crates (data/crates_top.json) plus the curated tag neurons
/// (data/crate_tags.json, incl. axum/bevy/symphonia which aren't in the
/// top downloads list).
fn crate_knowledge() -> &'static std::collections::HashMap<String, (String, Vec<String>)> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static K: OnceLock<HashMap<String, (String, Vec<String>)>> = OnceLock::new();
    K.get_or_init(|| {
        let mut m: HashMap<String, (String, Vec<String>)> = HashMap::new();
        if let Ok(top) = serde_json::from_str::<HashMap<String, serde_json::Value>>(
            include_str!("../../data/crates_top.json"),
        ) {
            for (name, v) in top {
                let desc = v
                    .get("desc")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let tags: Vec<String> = v
                    .get("tags")
                    .and_then(|x| x.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                m.insert(name, (desc, tags));
            }
        }
        if let Ok(tags) = serde_json::from_str::<HashMap<String, Vec<String>>>(
            include_str!("../../data/crate_tags.json"),
        ) {
            for (name, t) in tags {
                let e = m.entry(name).or_insert_with(|| (String::new(), vec![]));
                e.1 = t;
            }
        }
        m
    })
}

/// Crate-name detection shared by the crate experts: jieba a query, find a
/// token that names a known ecosystem crate. Generic descriptor words that are
/// also crate names (http/web/json/toml...) only count when the query clearly
/// talks about a crate (crate / cargo add / 里面的 / 怎么用...).
///
/// # Returns
/// * `Option<(crate_name, ask)>` - ask = query with the crate token removed
pub fn crate_name_in_query(query: &str) -> Option<(String, String)> {
    const GENERIC: &[&str] = &[
        "http", "web", "json", "data", "image", "time", "audio", "video", "media",
        "log", "async", "sql", "crypto", "cli", "arg", "parser", "parse", "server",
        "client", "file", "io", "regex", "graph", "net", "bytes", "random",
        "request", "model", "tensor", "fast", "game", "toml", "yaml", "xml", "csv",
        "channel", "queue", "cache", "map", "path", "string", "vec", "hash", "error",
    ];
    const CTX: &[&str] = &[
        "crate", "cargo add", "里面的", "里的", "怎么用", "用什么", "框架", "生态",
        "这个库", "那个库", "use ", "import ", "库有哪些", "怎么实现",
    ];
    let ql = query.to_lowercase();
    let has_ctx = CTX.iter().any(|m| ql.contains(m));
    let jieba = jieba_rs::Jieba::new();
    let tokens: Vec<String> = jieba
        .cut(query, false)
        .into_iter()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let knowledge = crate_knowledge();
    let mut best: Option<(usize, String)> = None;
    for (name, _) in knowledge {
        if !has_ctx && GENERIC.contains(&name.as_str()) {
            continue;
        }
        if let Some(pos) = tokens.iter().position(|t| t == name) {
            let score = 1000usize.saturating_sub(pos * 10);
            if best
                .as_ref()
                .map(|(s, _)| score > *s)
                .unwrap_or(true)
            {
                best = Some((score, name.clone()));
            }
        }
    }
    let (_, name) = best?;
    // ask = query without the crate token
    let ask = tokens
        .iter()
        .filter(|t| *t != &name)
        .cloned()
        .collect::<Vec<String>>()
        .join(" ");
    Some((name, ask))
}

/// A crate question carries real intent (怎么/新建/路由/请求...) vs a pure
/// description ask (是什么/里面的/有什么). Real intent delegates to the
/// dynamic ast-grep expert; pure description is answered from the knowledge.
fn has_crate_intent(ask: &str) -> bool {
    const STRONG: &[&str] = &[
        "怎么", "如何", "新建", "创建", "实现", "定义", "找找", "找", "查", "看看",
        "看", "路由", "请求", "发送", "解析", "读取", "写入", "中间件", "配置",
        "测试", "生成", "返回", "启动", "监听", "连接", "异步", "线程", "错误",
        "日志", "用", "使用", "处理", "构建", "声明", "获取", "添加", "移除",
        "遍历", "排序", "调用", "设置", "修改", "删除", "打开", "关闭",
    ];
    STRONG.iter().any(|w| ask.contains(w))
}

/// Crate expert: when a query names a specific ecosystem crate (axum, tokio,
/// reqwest, rand...), answer about THAT crate — describe it from the knowledge
/// base; if the ask carries real intent, defer to the dynamic ast-grep expert.
pub struct CrateExpertRouter;

impl Router for CrateExpertRouter {
    fn name(&self) -> &'static str {
        "crate专家"
    }
    fn route(&self, query: &str) -> Option<Answer> {
        let (name, ask) = crate_name_in_query(query)?;
        // real intent (怎么新建路由 / 如何发请求) -> delegate to the dynamic
        // ast-grep expert, which reads the crate's actual source; the static
        // experts only know std-library knowledge and would miss axum/tokio...
        if has_crate_intent(&ask) {
            return AstGrepExpertRouter.route(query);
        }
        let knowledge = crate_knowledge();
        let (desc, tags) = &knowledge[&name];
        let mut text = format!("crate {name}");
        if !tags.is_empty() {
            text.push_str(&format!("（{}）", tags.join("/")));
        }
        if !desc.is_empty() {
            text.push_str(&format!("：{desc}"));
        }
        if let Some(dir) = crate::core::codex::find_crate_src(&name) {
            let apis = crate::core::codex::analyze_crate_api(&dir);
            if !apis.is_empty() {
                text.push_str(&format!(
                    "\n本地 {name} 源码分析（{} 个公开函数）：",
                    apis.len()
                ));
                for a in apis.iter().take(3) {
                    text.push_str(&format!("\n  · {}", a.signature));
                }
            }
        }
        Some(Answer::new(self.name(), text))
    }
}

/// Dynamic ast-grep expert: the stack's last resort for crate-specific
/// questions. When every static expert (which only knows std-library
/// knowledge) finds no axum/tokio/... answer, this layer is delegated to and
/// ast-greps the crate's actual source, surfacing pub fns + docs that match
/// the intent.
pub struct AstGrepExpertRouter;

impl Router for AstGrepExpertRouter {
    fn name(&self) -> &'static str {
        "动态ast-grep专家"
    }
    fn route(&self, query: &str) -> Option<Answer> {
        let (name, ask) = crate_name_in_query(query)?;
        if !has_crate_intent(&ask) {
            return None;
        }
        Some(Answer::new(
            self.name(),
            crate::core::codex::ast_grep_tool(&name, &ask),
        ))
    }
}

/// The unified, prioritized routing stack.
pub struct RouterStack {
    pub routers: Vec<Box<dyn Router>>,
}

impl RouterStack {
    /// Builds the default pipeline: meanings -> concept code -> action blocks
    /// -> paraphrase intents -> crate recommendations.
    pub fn new() -> Self {
        Self {
            routers: vec![
                Box::new(CrateExpertRouter),
                Box::new(ExplainRouter::new()),
                Box::new(ConceptAnswerRouter::new()),
                Box::new(BlockRouterAdapter::new()),
                Box::new(RecommendRouter::new()),
                Box::new(IntentRouter::new()),
                Box::new(AstGrepExpertRouter),
            ],
        }
    }

    /// Routes a query through the stack, returning the first confident answer.
    pub fn route(&self, query: &str) -> Option<Answer> {
        for r in &self.routers {
            if let Some(a) = r.route(query) {
                return Some(a);
            }
        }
        None
    }

    /// Convenience: text of the first answer, if any.
    pub fn answer(&self, query: &str) -> Option<String> {
        self.route(query).map(|a| a.text)
    }
}

impl Default for RouterStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Explanation question -> dictionary meaning.
    #[test]
    fn test_explain_via_stack() {
        let stack = RouterStack::new();
        let a = stack.route("泛型是什么").expect("should route");
        assert!(a.text.contains("类型参数化"), "got {}", a.text);
    }

    /// Action question -> concept code.
    #[test]
    fn test_action_via_stack() {
        let stack = RouterStack::new();
        let a = stack.route("我需要打开文件").expect("should route");
        assert!(a.text.contains("File::open"), "got {}", a.text);
    }

    /// Declaration -> fn block.
    #[test]
    fn test_block_via_stack() {
        let stack = RouterStack::new();
        let a = stack.route("声明函数").expect("should route");
        assert!(a.text.contains("fn f"), "got {}", a.text);
    }

    /// Unknown query -> None (fallback handled by caller).
    #[test]
    fn test_unknown_none() {
        let stack = RouterStack::new();
        assert!(stack.route("中午吃什么").is_none());
    }

    /// A named ecosystem crate is answered about that crate.
    #[test]
    fn test_crate_expert() {
        let stack = RouterStack::new();
        let a = stack.route("我需要的是axum里面的").expect("axum should route");
        assert!(a.text.contains("axum"), "got {}", a.text);
        assert!(a.text.contains("http"), "got {}", a.text);
    }

    /// Naming a crate beats the generic word "crate" explanation.
    #[test]
    fn test_crate_expert_beats_crate_word() {
        let stack = RouterStack::new();
        let a = stack
            .route("你自动cargo add axum 看crate")
            .expect("axum should route");
        assert!(a.text.contains("axum"), "got {}", a.text);
    }

    /// A generic descriptor word ("http") must NOT hijack a normal request —
    /// it should still be recommended a client, not answered about `http` crate.
    #[test]
    fn test_generic_crate_word_does_not_hijack() {
        let stack = RouterStack::new();
        let a = stack.route("如何新建一个http 请求").expect("should route");
        assert!(!a.text.contains("crate http"), "got {}", a.text);
        assert!(a.text.contains("cargo add"), "got {}", a.text);
    }

    /// Real crate intent is delegated to the dynamic ast-grep expert.
    #[test]
    fn test_crate_intent_delegates_to_ast_grep() {
        let stack = RouterStack::new();
        let a = stack
            .route("axum 帮我找找怎么新建路由")
            .expect("should delegate to ast-grep");
        assert!(a.text.contains("源码分析"), "got {}", a.text);
        assert!(a.text.contains("pub fn"), "got {}", a.text);
    }
}
