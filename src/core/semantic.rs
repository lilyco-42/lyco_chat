/*
 * @file semantic.rs
 * @brief Semantic intent MoE: paraphrase clusters routed to canonical answers
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

//! FILE: semantic.rs
//!
//! DESCRIPTION:
//! Semantic intent MoE for TinyGPT.
//!
//! BRIEF:
//! Each expert is a cluster of near-synonym paraphrases expressing one
//! meaning (e.g. "打开文件 / 我想打开文件 / 怎么打开文件" all mean open-file).
//! A dependency-free router scores a query by character + bigram overlap
//! with each cluster and routes it to the best matching meaning, whose
//! expert returns the canonical answer. This keeps the classification job
//! tiny while every expert only ever needs to reproduce one answer.

use std::collections::HashSet;

/// A semantic expert: one meaning expressed by many paraphrases.
#[derive(Debug, Clone)]
pub struct SemanticExpert {
    /// Meaning name, e.g. "打开文件".
    pub name: String,
    /// Canonical answer for this meaning.
    pub answer: String,
    /// Paraphrase sentences that express this meaning.
    pub paraphrases: Vec<String>,
    /// Union of all characters across paraphrases.
    chars: HashSet<char>,
    /// Union of all character bigrams across paraphrases.
    bigrams: HashSet<(char, char)>,
}

impl SemanticExpert {
    /// Builds an expert from its paraphrase cluster.
    pub fn new(name: &str, answer: &str, paraphrases: &[&str]) -> Self {
        let mut chars = HashSet::new();
        let mut bigrams = HashSet::new();
        for p in paraphrases {
            for ch in p.chars() {
                if !ch.is_whitespace() {
                    chars.insert(ch);
                }
            }
            let c: Vec<char> = p.chars().filter(|c| !c.is_whitespace()).collect();
            for w in c.windows(2) {
                bigrams.insert((w[0], w[1]));
            }
        }
        Self {
            name: name.to_string(),
            answer: answer.to_string(),
            paraphrases: paraphrases.iter().map(|s| s.to_string()).collect(),
            chars,
            bigrams,
        }
    }

    /// Overlap score of a query against this expert's paraphrase cluster.
    ///
    /// # Details
    /// Weighted fraction of the query's characters and bigrams covered by
    /// the cluster. Returns `(char_score, bigram_score)` in 0..=1.
    fn score(&self, qchars: &HashSet<char>, qbigrams: &HashSet<(char, char)>) -> (f32, f32) {
        let c = if qchars.is_empty() {
            0.0
        } else {
            qchars.iter().filter(|c| self.chars.contains(c)).count() as f32 / qchars.len() as f32
        };
        let b = if qbigrams.is_empty() {
            0.0
        } else {
            qbigrams
                .iter()
                .filter(|b| self.bigrams.contains(b))
                .count() as f32
                / qbigrams.len() as f32
        };
        (c, b)
    }
}

/// Semantic intent MoE: paraphrases route to canonical answers.
pub struct SemanticMoE {
    pub experts: Vec<SemanticExpert>,
    /// Minimum combined score to accept a routing.
    pub threshold: f32,
}

fn query_sets(query: &str) -> (HashSet<char>, HashSet<(char, char)>) {
    let chars: Vec<char> = query.chars().filter(|c| !c.is_whitespace()).collect();
    let mut cs = HashSet::new();
    for &c in &chars {
        cs.insert(c);
    }
    let mut bs = HashSet::new();
    for w in chars.windows(2) {
        bs.insert((w[0], w[1]));
    }
    (cs, bs)
}

impl SemanticMoE {
    /// Creates the router with a built-in intent table.
    ///
    /// # Details
    /// Intents and paraphrases mirror the bilingual QA corpus. Each intent
    /// has several near-synonym phrasings that must route to the same answer.
    pub fn new() -> Self {
        let experts = vec![
            SemanticExpert::new(
                "打开文件",
                "use std::fs::File; let mut f = File::open(\"a.txt\")?;",
                &[
                    "打开文件", "我想打开文件", "怎么打开文件", "请帮我打开文件",
                    "请问如何打开文件", "我需要打开一个文件", "把文件打开", "打开一个文件",
                ],
            ),
            SemanticExpert::new(
                "删除文件",
                "std::fs::remove_file(\"a.txt\")?;",
                &[
                    "删除文件", "我想删除文件", "怎么删除文件", "帮我删除文件",
                    "如何删除一个文件", "把文件删掉",
                ],
            ),
            SemanticExpert::new(
                "读取文件",
                "let s = std::fs::read_to_string(\"a.txt\")?;",
                &[
                    "读取文件", "读取文件内容", "我想读取文件", "怎么读取文件",
                    "读取整个文件的内容",
                ],
            ),
            SemanticExpert::new(
                "写入文件",
                "std::fs::write(\"a.txt\", \"hello\")?;",
                &[
                    "写入文件", "向文件写入内容", "怎么写入文件", "把内容写进文件",
                ],
            ),
            SemanticExpert::new(
                "创建文件夹",
                "std::fs::create_dir(\"d\")?;",
                &[
                    "创建文件夹", "创建目录", "新建目录", "怎么创建文件夹", "创建一个文件夹",
                ],
            ),
            SemanticExpert::new(
                "定义函数",
                "fn f(x: i32) -> i32 { x + 1 }",
                &[
                    "定义函数", "我想定义一个函数", "怎么定义函数", "如何写一个函数",
                    "编写一个函数",
                ],
            ),
            SemanticExpert::new(
                "定义结构体",
                "struct S { x: T }",
                &["定义结构体", "怎么定义结构体", "定义一个结构体", "新建一个结构体"],
            ),
            SemanticExpert::new(
                "生成随机数",
                "cargo add rand",
                &["生成随机数", "随机数", "怎么生成随机数", "需要一个随机数", "获取随机数"],
            ),
            SemanticExpert::new(
                "发送HTTP请求",
                "cargo add reqwest",
                &[
                    "发送HTTP请求", "HTTP请求", "怎么发HTTP请求", "发起网络请求",
                    "调用一个接口",
                ],
            ),
            SemanticExpert::new(
                "连接数据库",
                "cargo add sqlx",
                &["连接数据库", "使用数据库", "怎么连接数据库", "操作数据库"],
            ),
        ];
        Self {
            experts,
            threshold: 0.4,
        }
    }

    /// Routes a query to the best matching meaning.
    ///
    /// # Details
    /// Score = 0.35 * char coverage + 0.65 * bigram coverage (bigrams carry
    /// word order). Returns `(expert_id, score)` if the best match clears the
    /// threshold, otherwise `None`.
    pub fn route(&self, query: &str) -> Option<(usize, f32)> {
        let (qc, qb) = query_sets(query);
        let mut best: Option<(usize, f32)> = None;
        for (i, e) in self.experts.iter().enumerate() {
            let (c, b) = e.score(&qc, &qb);
            let s = 0.35 * c + 0.65 * b;
            if best.map_or(true, |(_, bs)| s > bs) {
                best = Some((i, s));
            }
        }
        match best {
            Some((i, s)) if s >= self.threshold => Some((i, s)),
            _ => None,
        }
    }

    /// Returns the canonical answer for a query, if routed.
    pub fn answer(&self, query: &str) -> Option<&str> {
        self.route(query).map(|(i, _)| self.experts[i].answer.as_str())
    }
}

impl Default for SemanticMoE {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests near-synonym phrasings all route to the same meaning.
    #[test]
    fn test_synonyms_route_to_same_expert() {
        let moe = SemanticMoE::new();
        let target = moe.experts.iter().position(|e| e.name == "打开文件").unwrap();
        for q in ["打开文件", "我想打开文件", "怎么打开文件", "请帮我打开文件", "请问如何打开文件", "把文件打开"] {
            let (i, s) = moe.route(q).expect("should route");
            assert_eq!(i, target, "query {q:?} should route to 打开文件, score {s:.2}");
        }
    }

    /// Tests different meanings route to different experts.
    #[test]
    fn test_distinct_meanings() {
        let moe = SemanticMoE::new();
        let (open, _) = moe.route("我想打开文件").unwrap();
        let (del, _) = moe.route("帮我删除文件").unwrap();
        let (read, _) = moe.route("怎么读取文件内容").unwrap();
        assert_ne!(open, del);
        assert_ne!(open, read);
        assert_ne!(del, read);
    }

    /// Tests answers are the canonical code for the routed meaning.
    #[test]
    fn test_answer_canonical() {
        let moe = SemanticMoE::new();
        assert_eq!(
            moe.answer("我想打开文件").unwrap(),
            "use std::fs::File; let mut f = File::open(\"a.txt\")?;"
        );
        assert_eq!(moe.answer("怎么删除文件").unwrap(), "std::fs::remove_file(\"a.txt\")?;");
        assert_eq!(moe.answer("如何写一个函数").unwrap(), "fn f(x: i32) -> i32 { x + 1 }");
    }

    /// Tests unknown queries fall back to None below the threshold.
    #[test]
    fn test_unknown_query_falls_back() {
        let moe = SemanticMoE::new();
        assert!(moe.answer("今天天气怎么样").is_none());
        assert!(moe.answer("中午吃什么").is_none());
    }
}
