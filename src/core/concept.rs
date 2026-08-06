/*
 * @file concept.rs
 * @brief jieba-segmented Rust concept router -> neural answer activation
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

//! FILE: concept.rs
//!
//! DESCRIPTION:
//! Rust common-concepts router driven by jieba-rs word segmentation.
//!
//! BRIEF:
//! Splits a Chinese query into words with jieba-rs, then checks which Rust
//! concept's keywords appear. A high-confidence concept match activates the
//! corresponding neural/expert answer. Word boundaries make this far more
//! precise than character-overlap heuristics ("打开" is one token, never
//! confused with other "打-*" words).

use crate::core::linear::Linear;
use crate::core::math::softmax1d;
use crate::core::tiny_gpt::TinyGPT;
use crate::core::vocab::Vocab;
use jieba_rs::Jieba;

/// Char-level encode of a query, matching the corpus tokenization.
fn char_encode(vocab: &Vocab, content: &str) -> Vec<usize> {
    let mut out: Vec<String> = vec![];
    let mut buf = String::new();
    for ch in content.chars() {
        if ch.is_whitespace() {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            continue;
        }
        if ch.is_ascii() {
            buf.push(ch);
        } else {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(ch.to_string());
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out.iter().filter_map(|t| vocab.w2i.get(t).copied()).collect()
}

/// A keyword with its activation weight.
///
/// # Details
/// Domain anchors (e.g. 文件) carry a high weight, action words (e.g. 打开)
/// a medium one. A concept activates when the weighted sum of matched
/// keywords crosses its threshold — so "如何打开一个文件" (文件 high + 打开
/// medium) hits 打开文件, while "文件" alone stays ambiguous.
#[derive(Clone, Copy, Debug)]
pub struct Keyword {
    pub word: &'static str,
    pub weight: f32,
}

/// One Rust common concept and the weighted keywords that activate it.
pub struct Concept {
    pub name: &'static str,
    pub answer: &'static str,
    pub keywords: Vec<Keyword>,
    /// Minimum weighted-match ratio to activate this concept.
    pub threshold: f32,
}

/// Routes segmented queries to Rust concepts.
pub struct ConceptRouter {
    jieba: Jieba,
    pub concepts: Vec<Concept>,
}

/// Const-constructs a keyword (allows `&'static [Keyword]` array literals).
const fn kw(word: &'static str, weight: f32) -> Keyword {
    Keyword { word, weight }
}

impl ConceptRouter {
    /// Creates the router with a built-in Rust concepts table.
    pub fn new() -> Self {
        Self {
            jieba: Jieba::new(),
            concepts: vec![
                Concept {
                    name: "打开文件",
                    answer: "use std::fs::File; let mut f = File::open(\"a.txt\")?;",
                    keywords: vec![kw("打开", 0.5), kw("开启", 0.4), kw("文件", 0.8)],
                    threshold: 0.5,
                },
                Concept {
                    name: "删除文件",
                    answer: "std::fs::remove_file(\"a.txt\")?;",
                    keywords: vec![kw("删除", 0.5), kw("移除", 0.4), kw("删掉", 0.4), kw("文件", 0.8)],
                    threshold: 0.5,
                },
                Concept {
                    name: "读取文件",
                    answer: "let s = std::fs::read_to_string(\"a.txt\")?;",
                    keywords: vec![kw("读取", 0.5), kw("读入", 0.4), kw("文件", 0.8)],
                    threshold: 0.5,
                },
                Concept {
                    name: "写入文件",
                    answer: "std::fs::write(\"a.txt\", \"hello\")?;",
                    keywords: vec![kw("写入", 0.5), kw("写进", 0.4), kw("文件", 0.8)],
                    threshold: 0.5,
                },
                Concept {
                    name: "创建文件夹",
                    answer: "std::fs::create_dir(\"d\")?;",
                    keywords: vec![kw("创建", 0.5), kw("新建", 0.4), kw("文件夹", 0.8), kw("目录", 0.6)],
                    threshold: 0.5,
                },
                Concept {
                    name: "定义函数",
                    answer: "fn f(x: i32) -> i32 { x + 1 }",
                    keywords: vec![kw("定义", 0.5), kw("函数", 0.8), kw("编写", 0.3)],
                    threshold: 0.5,
                },
                Concept {
                    name: "定义结构体",
                    answer: "struct S { x: T }",
                    keywords: vec![kw("结构体", 0.8), kw("结构", 0.4)],
                    threshold: 0.5,
                },
                Concept {
                    name: "泛型",
                    answer: "fn identity<T>(x: T) -> T { x }",
                    keywords: vec![kw("泛型", 0.8), kw("类型参数", 0.5), kw("模板", 0.4)],
                    threshold: 0.4,
                },
                Concept {
                    name: "生成随机数",
                    answer: "cargo add rand",
                    keywords: vec![kw("随机数", 0.8), kw("随机", 0.5)],
                    threshold: 0.5,
                },
                Concept {
                    name: "发送HTTP请求",
                    answer: "cargo add reqwest",
                    keywords: vec![kw("HTTP", 0.8), kw("请求", 0.5), kw("接口", 0.4), kw("网络", 0.4)],
                    threshold: 0.5,
                },
                Concept {
                    name: "连接数据库",
                    answer: "cargo add sqlx",
                    keywords: vec![kw("数据库", 0.8), kw("连接", 0.5), kw("SQL", 0.5)],
                    threshold: 0.5,
                },
                Concept {
                    name: "格式化字符串",
                    answer: "let s = format!(\"{}-{}\", 1, 2);",
                    keywords: vec![kw("格式化", 0.6), kw("字符串", 0.8), kw("format", 0.4)],
                    threshold: 0.5,
                },
                Concept {
                    name: "遍历数组",
                    answer: "for x in v.iter() { println!(\"{}\", x); }",
                    keywords: vec![kw("遍历", 0.6), kw("数组", 0.8), kw("迭代", 0.4)],
                    threshold: 0.5,
                },
                Concept {
                    name: "排序数组",
                    answer: "v.sort();",
                    keywords: vec![kw("排序", 0.8), kw("排列", 0.5)],
                    threshold: 0.5,
                },
            ],
        }
    }

    /// Segments a query into words with jieba.
    ///
    /// # Returns
    /// * `Vec<String>` - Owned segmented words
    pub fn cut(&self, query: &str) -> Vec<String> {
        self.jieba.cut(query, false).into_iter().map(|s| s.to_string()).collect()
    }

    /// Routes a query to the concept with the highest weighted keyword match.
    ///
    /// # Details
    /// A keyword matches if it appears among the segmented words OR as a
    /// substring of the query (covers terms jieba's default dictionary does
    /// not know, e.g. "泛型"). Score = matched weight / total weight of the
    /// concept's keywords; the concept activates only when its own threshold
    /// is cleared. High-weight domain anchors (文件) plus medium-weight
    /// actions (打开) jointly hit 打开文件, while a bare 文件 stays ambiguous.
    ///
    /// # Arguments
    /// * `query` - User question
    ///
    /// # Returns
    /// * `Option<(usize, f32)>` - (concept index, weighted score)
    pub fn route(&self, query: &str) -> Option<(usize, f32)> {
        let words = self.cut(query);
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.concepts.iter().enumerate() {
            let total: f32 = c.keywords.iter().map(|k| k.weight).sum();
            let matched: f32 = c
                .keywords
                .iter()
                .filter(|k| words.iter().any(|w| w == k.word) || query.contains(k.word))
                .map(|k| k.weight)
                .sum();
            if matched <= 0.0 {
                continue;
            }
            let score = matched / total;
            if score >= c.threshold && best.map_or(true, |(_, bs)| score > bs) {
                best = Some((i, score));
            }
        }
        best
    }

    /// Returns the activated neural answer for a query, if routed.
    pub fn answer(&self, query: &str) -> Option<&'static str> {
        self.route(query).map(|(i, _)| self.concepts[i].answer)
    }

    /// Neural re-ranking: which concept's answer is most pertinent.
    ///
    /// # Details
    /// A concept maps 1:1 to its answer; the neural network then judges which
    /// answer best fits the query. Each candidate answer's tokens are scored
    /// by the model's average negative log-likelihood when continuing
    /// `问 : <query> 答 : <answer>` — a lower loss means the model finds that
    /// continuation more natural, i.e. more 贴切. This turns free generation
    /// (which hallucinates) into a constrained choice among known answers.
    ///
    /// # Arguments
    /// * `model` - Trained model (any `LinearLike` backend)
    /// * `vocab` - Vocabulary matching the model's corpus
    /// * `query` - User question
    ///
    /// # Returns
    /// * `Option<(usize, Vec<(usize, f32, f32)>)>` - (best concept index,
    ///   per-candidate (concept id, avg loss, answer token count))
    pub fn neural_rank<L: crate::core::linear::LinearLike + Sync>(
        &self,
        model: &TinyGPT<L>,
        vocab: &Vocab,
        query: &str,
    ) -> Option<(usize, Vec<(usize, f32, f32)>)> {
        let q = char_encode(vocab, query);
        if q.is_empty() {
            return None;
        }
        let mut scored: Vec<(usize, f32, f32)> = vec![];
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.concepts.iter().enumerate() {
            let ans_tokens: Vec<usize> = c
                .answer
                .split_whitespace()
                .filter_map(|t| vocab.w2i.get(t).copied())
                .collect();
            if ans_tokens.is_empty() {
                continue;
            }
            let mut seq = vec![vocab.encode("问"), vocab.encode(":")];
            seq.extend(&q);
            seq.push(vocab.encode("答"));
            seq.push(vocab.encode(":"));
            seq.extend(&ans_tokens);
            let logits = model.forward(&seq);
            let base_ans = q.len() + 4; // first answer token's row in logits
            let mut loss = 0.0;
            for (k, &t) in ans_tokens.iter().enumerate() {
                let probs = softmax1d(&logits.row(base_ans + k));
                loss += (-probs[t].ln()).max(0.0);
            }
            let avg = loss / ans_tokens.len() as f32;
            scored.push((i, avg, ans_tokens.len() as f32));
            if best.map_or(true, |(_, bl)| avg < bl) {
                best = Some((i, avg));
            }
        }
        best.map(|(id, _)| (id, scored))
    }
}

impl Default for ConceptRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests jieba segments Rust-relevant Chinese into words.
    #[test]
    fn test_jieba_segmentation() {
        let r = ConceptRouter::new();
        let words = r.cut("我想打开文件");
        assert!(words.contains(&"打开".to_string()));
        assert!(words.contains(&"文件".to_string()));
    }

    /// Tests near-synonym phrasings all activate the same concept.
    #[test]
    fn test_synonyms_activate_same_concept() {
        let r = ConceptRouter::new();
        let target = r.concepts.iter().position(|c| c.name == "打开文件").unwrap();
        for q in ["我想打开文件", "怎么打开文件", "请帮我打开文件", "打开文件", "把文件打开"] {
            let (i, s) = r.route(q).expect("should route");
            assert_eq!(i, target, "query {q:?} should activate 打开文件, score {s:.2}");
        }
    }

    /// Tests distinct concepts do not cross-activate.
    #[test]
    fn test_concepts_distinct() {
        let r = ConceptRouter::new();
        let a = r.answer("帮我删除文件").unwrap();
        assert_eq!(a, "std::fs::remove_file(\"a.txt\")?;");
        let b = r.answer("如何写一个函数").unwrap();
        assert_eq!(b, "fn f(x: i32) -> i32 { x + 1 }");
        assert_ne!(a, b);
    }

    /// Tests a term jieba does not split still activates via substring match.
    #[test]
    fn test_generics_substring_fallback() {
        let r = ConceptRouter::new();
        let (i, s) = r.route("泛型的作用").expect("泛型 should route");
        assert_eq!(r.concepts[i].name, "泛型", "score {s:.2}");
    }

    /// Tests unknown queries do not activate anything.
    #[test]
    fn test_unknown_no_activation() {
        let r = ConceptRouter::new();
        assert!(r.answer("今天天气怎么样").is_none());
        assert!(r.answer("中午吃什么").is_none());
    }
}
