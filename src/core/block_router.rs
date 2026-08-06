/*
 * @file block_router.rs
 * @brief Word-level multi-head attention routing with action blocks
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

//! FILE: block_router.rs
//!
//! DESCRIPTION:
//! Word-level multi-head attention routing with per-concept action blocks.
//!
//! BRIEF:
//! jieba segments the query into words; every word is a "head" that votes on
//! each concept via near-synonym analysis (函数/方法/function -> function
//! concept). The concept with the highest aggregated attention is activated.
//! The action word (声明/定义/调用...) then routes to a specific *block*
//! inside that concept — e.g. 声明函数 -> the fn-declaration block.

use jieba_rs::Jieba;

/// One action block inside a concept (e.g. the fn-declaration block).
pub struct Block {
    pub action: &'static str,
    pub code: &'static str,
}

/// A concept: noun synonyms for attention routing + action blocks.
pub struct BlockConcept {
    pub name: &'static str,
    pub nouns: &'static [&'static str],
    pub blocks: Vec<Block>,
}

/// The outcome of word-level attention routing.
#[derive(Debug, Clone)]
pub struct RouteOutcome {
    pub words: Vec<String>,
    pub concept_index: usize,
    pub concept_name: String,
    pub action: String,
    pub code: String,
    pub attention: Vec<f32>,
}

/// Word-level multi-head attention router.
pub struct WordBlockRouter {
    jieba: Jieba,
    pub concepts: Vec<BlockConcept>,
    pub threshold: f32,
}

const BLOCK: fn(&'static str, &'static str) -> Block = |action, code| Block { action, code };

impl WordBlockRouter {
    pub fn new() -> Self {
        Self {
            jieba: Jieba::new(),
            threshold: 0.5,
            concepts: vec![
                BlockConcept {
                    name: "文件操作",
                    nouns: &["文件", "file", "档案"],
                    blocks: vec![
                        BLOCK("打开", "use std::fs::File; let mut f = File::open(\"a.txt\")?;"),
                        BLOCK("删除", "std::fs::remove_file(\"a.txt\")?;"),
                        BLOCK("读取", "let s = std::fs::read_to_string(\"a.txt\")?;"),
                        BLOCK("写入", "std::fs::write(\"a.txt\", \"hello\")?;"),
                        BLOCK("创建", "let mut f = File::create(\"a.txt\")?;"),
                    ],
                },
                BlockConcept {
                    name: "函数",
                    nouns: &["函数", "方法", "子程序", "function", "fn"],
                    blocks: vec![
                        BLOCK("异步", "async fn f() -> i32 { 1 }"),
                        BLOCK("声明", "fn f(x: i32) -> i32 { x + 1 }"),
                        BLOCK("定义", "fn f(x: i32) -> i32 { x + 1 }"),
                        BLOCK("实现", "fn f(x: i32) -> i32 { x + 1 }"),
                        BLOCK("调用", "let r = f(1);"),
                        BLOCK("编写", "fn f(x: i32) -> i32 { x + 1 }"),
                    ],
                },
                BlockConcept {
                    name: "结构体",
                    nouns: &["结构体", "struct"],
                    blocks: vec![
                        BLOCK("定义", "struct S { x: T }"),
                        BLOCK("声明", "struct S { x: T }"),
                        BLOCK("创建", "let s = S { x: 1 };"),
                    ],
                },
                BlockConcept {
                    name: "随机数",
                    nouns: &["随机数", "随机", "random"],
                    blocks: vec![BLOCK("生成", "cargo add rand"), BLOCK("获取", "rand::thread_rng()")],
                },
                BlockConcept {
                    name: "HTTP请求",
                    nouns: &["HTTP", "请求", "接口", "网络", "http"],
                    blocks: vec![BLOCK("发送", "cargo add reqwest"), BLOCK("发起", "cargo add reqwest")],
                },
                BlockConcept {
                    name: "数据库",
                    nouns: &["数据库", "db", "SQL"],
                    blocks: vec![BLOCK("连接", "cargo add sqlx"), BLOCK("使用", "cargo add sqlx")],
                },
                BlockConcept {
                    name: "文件夹",
                    nouns: &["文件夹", "目录", "folder"],
                    blocks: vec![BLOCK("创建", "std::fs::create_dir(\"d\")?;"), BLOCK("新建", "std::fs::create_dir(\"d\")?;")],
                },
                BlockConcept {
                    name: "数组",
                    nouns: &["数组", "vector", "vec"],
                    blocks: vec![
                        BLOCK("遍历", "for x in v.iter() { println!(\"{}\", x); }"),
                        BLOCK("排序", "v.sort();"),
                        BLOCK("创建", "let v = vec![1, 2, 3];"),
                    ],
                },
                BlockConcept {
                    name: "字符串",
                    nouns: &["字符串", "string", "str"],
                    blocks: vec![
                        BLOCK("格式化", "let s = format!(\"{}-{}\", 1, 2);"),
                        BLOCK("拼接", "s.push_str(\" world\");"),
                    ],
                },
                BlockConcept {
                    name: "泛型",
                    nouns: &["泛型", "generic", "类型参数"],
                    blocks: vec![BLOCK("定义", "fn identity<T>(x: T) -> T { x }")],
                },
            ],
        }
    }

    /// jieba-segments the query into words.
    pub fn cut(&self, query: &str) -> Vec<String> {
        self.jieba.cut(query, false).into_iter().map(|s| s.to_string()).collect()
    }

    /// Routes a query with word-level multi-head attention.
    ///
    /// # Details
    /// Each jieba word is a head voting on every concept via near-synonym
    /// noun matching. The concept with the most attention is activated, then
    /// the action word (声明/定义/调用...) selects the specific block inside
    /// it. Returns `None` when no concept clears the threshold.
    pub fn route(&self, query: &str) -> Option<RouteOutcome> {
        let words = self.cut(query);
        let n = self.concepts.len();
        let mut attention = vec![0f32; n];
        for w in &words {
            for (ci, c) in self.concepts.iter().enumerate() {
                // word-level vote: the segmented word IS a concept noun or
                // embeds one (jieba may keep "调用函数" as a single token)
                if c.nouns.iter().any(|s| w == s || w.contains(*s)) {
                    attention[ci] += 1.0;
                }
            }
        }
        // near-synonym fallback: jieba can over-split compounds (泛型 -> 泛/型,
        // 结构体 -> 结构/体), so a noun present in the query also votes once.
        for (ci, c) in self.concepts.iter().enumerate() {
            if attention[ci] == 0.0
                && c.nouns
                    .iter()
                    .any(|s| s.len() >= 2 && query.contains(*s))
            {
                attention[ci] += 1.0;
            }
        }
        // tie-break: prefer the earliest concept with the same attention
        let mut best = 0usize;
        for (ci, a) in attention.iter().enumerate() {
            if a > &attention[best] {
                best = ci;
            }
        }
        if attention[best] < self.threshold {
            return None;
        }
        let concept = &self.concepts[best];
        // action word selects a block; if the query has an action word that
        // does not match ANY block, this is a mismatched action (e.g. "解析
        // exe 文件" vs 文件打开) -> decline, let a recommender answer.
        const ACTIONS: [&str; 24] = [
            "声明", "定义", "调用", "打开", "删除", "读取", "写入", "创建", "生成", "遍历",
            "排序", "格式化", "拼接", "新建", "获取", "发送", "连接", "使用", "实现", "编写",
            "解析", "处理", "编译", "安装",
        ];
        let mut block_idx = 0usize;
        let mut matched = false;
        for (i, b) in concept.blocks.iter().enumerate() {
            if words.iter().any(|w| w == b.action) || query.contains(b.action) {
                block_idx = i;
                matched = true;
                break;
            }
        }
        if !matched && ACTIONS.iter().any(|a| words.iter().any(|w| w == *a) || query.contains(*a)) {
            return None;
        }
        let block = &concept.blocks[block_idx];
        Some(RouteOutcome {
            words,
            concept_index: best,
            concept_name: concept.name.to_string(),
            action: block.action.to_string(),
            code: block.code.to_string(),
            attention,
        })
    }

    /// Convenience: returns just the routed code block, if any.
    pub fn answer(&self, query: &str) -> Option<String> {
        self.route(query).map(|o| o.code)
    }
}

impl Default for WordBlockRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "声明函数": 函数 noun routes to the function concept, 声明 action
    /// selects the fn-declaration block.
    #[test]
    fn test_declare_function_uses_fn_block() {
        let r = WordBlockRouter::new();
        let o = r.route("声明函数").expect("should route");
        assert_eq!(o.concept_name, "函数");
        assert_eq!(o.action, "声明");
        assert!(o.code.contains("fn f"), "声明函数 should hit the fn declaration block: {}", o.code);
    }

    /// Different actions pick different blocks inside the same concept.
    #[test]
    fn test_action_selects_block() {
        let r = WordBlockRouter::new();
        let declare = r.route("如何定义函数").unwrap();
        let call = r.route("调用函数").unwrap();
        assert_eq!(declare.concept_name, "函数");
        assert_eq!(call.concept_name, "函数");
        assert!(declare.code.contains("fn f"));
        assert!(call.code.contains("let r = f(1);"), "调用函数 should hit the call block");
    }

    /// File domain: open/delete select different blocks.
    #[test]
    fn test_file_blocks() {
        let r = WordBlockRouter::new();
        assert!(r.answer("怎么打开文件").unwrap().contains("File::open"));
        assert!(r.answer("我需要删除文件").unwrap().contains("remove_file"));
        assert!(r.answer("读取整个文件").unwrap().contains("read_to_string"));
    }

    /// Word-level attention: 声明函数 votes 函数 highest.
    #[test]
    fn test_attention_hits_function() {
        let r = WordBlockRouter::new();
        let o = r.route("声明函数").unwrap();
        let fn_idx = r.concepts.iter().position(|c| c.name == "函数").unwrap();
        assert!(o.attention[fn_idx] > 0.0);
        assert_eq!(o.concept_index, fn_idx);
    }

    /// Unknown queries do not route.
    #[test]
    fn test_unknown() {
        let r = WordBlockRouter::new();
        assert!(r.answer("今天天气怎么样").is_none());
    }
}
