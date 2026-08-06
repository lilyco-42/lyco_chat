/*
 * @file intent.rs
 * @brief Transformer-style labeled attention for intent inference
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

//! FILE: intent.rs
//!
//! DESCRIPTION:
//! Guesses a user's intent by tagging each jieba word with a role
//! (adjective / programming term / verb / particle) and running a
//! transformer-style multi-head attention over candidate intents.
//!
//! BRIEF:
//! Every word is a "head" carrying a role label: term words attend to the
//! domain (category), adjective words attend to crate traits (fast -> fastrand),
//! verb words attend to actions. Aggregated attention picks the winning
//! intent, which routes to the right expert (category -> crate).

use crate::core::codex::CategoryIndex;
use jieba_rs::Jieba;

/// The role of a segmented word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordRole {
    /// 形容词：快速/轻量/安全/异步 ...
    Adj,
    /// 编程术语：随机数/函数/数据库/HTTP ...
    Term,
    /// 动词/动作：生成/创建/删除/声明 ...
    Verb,
    /// 虚词/语气：我/的/怎么/一个 ...
    Particle,
    /// 其他
    Other,
}

/// A jieba-segmented word with its inferred role.
#[derive(Clone, Debug)]
pub struct TaggedWord {
    pub word: String,
    pub role: WordRole,
}

const ADJ: &[&str] = &[
    "快速", "快", "更快", "超快", "极快", "高性能", "高速", "性能", "轻量", "轻",
    "简单", "易用", "简洁", "安全", "加密", "异步", "并发", "同步", "大", "小",
    "内存", "稳定", "跨平台", "小", "快",
];
const TERM: &[&str] = &[
    "随机数", "随机", "函数", "结构体", "数组", "字符串", "文件", "文件夹", "目录",
    "数据库", "HTTP", "JSON", "压缩", "排序", "解析", "遍历", "日志", "测试", "密码",
    "图像", "游戏", "机器学习", "网络", "请求", "客户端", "服务", "正则", "线程",
    "并行", "缓存", "环境变量", "命令行", "时间", "日期", "邮件", "图形界面", "界面",
    "音视频", "多媒体", "音频", "视频", "神经", "神经网络", "exe", "二进制",
];
const VERB: &[&str] = &[
    "生成", "创建", "删除", "打开", "读取", "写入", "声明", "定义", "连接", "发送",
    "调用", "格式化", "获取", "查找", "搜索", "遍历", "排序", "解析", "压缩", "加密",
    "测试", "构建", "编译", "添加", "使用", "实现", "调用", "下载", "处理",
];
const PARTICLE: &[&str] = &[
    "我", "你", "想", "要", "的", "了", "吗", "呢", "怎么", "如何", "为什么",
    "一个", "请", "帮", "这", "那", "是", "在", "和", "或", "给", "把", "用",
    "下", "里", "进行", "用来", "一下", "可以", "希望", "想要", "给我",
];

/// Adjective -> crate trait (fast -> fastrand).
fn adj_to_trait(adj: &str) -> Option<&'static str> {
    match adj {
        "快速" | "快" | "更快" | "超快" | "极快" | "高性能" | "高速" | "性能" => Some("fast"),
        "轻量" | "轻" | "小" => Some("lightweight"),
        "简单" | "易用" | "简洁" => Some("simple"),
        "安全" | "加密" => Some("secure"),
        "异步" | "并发" => Some("async"),
        "同步" => Some("sync"),
        "内存" => Some("memory"),
        _ => None,
    }
}

/// jieba-segments a query and tags each word with a role.
///
/// # Details
/// A word gets the most specific role it matches: Term > Adj > Verb > Particle.
/// jieba-segments a query and tags each word with a role.
///
/// # Details
/// A word gets the most specific role it matches: Term > Adj > Verb > Particle.
/// `learned` terms (auto-learned from past misses) are treated as Term words.
pub fn tag_words_with(query: &str, learned: &[String]) -> Vec<TaggedWord> {
    let jieba = Jieba::new();
    jieba
        .cut(query, false)
        .into_iter()
        .filter(|w| !w.trim().is_empty())
        .map(|w| {
            let w = w.to_string();
            let role = if TERM.contains(&w.as_str()) || learned.iter().any(|l| l == &w) {
                WordRole::Term
            } else if ADJ.contains(&w.as_str()) {
                WordRole::Adj
            } else if VERB.contains(&w.as_str()) {
                WordRole::Verb
            } else if PARTICLE.contains(&w.as_str()) {
                WordRole::Particle
            } else {
                WordRole::Other
            };
            TaggedWord { word: w, role }
        })
        .collect()
}

/// jieba-segments a query and tags each word (no learned terms).
pub fn tag_words(query: &str) -> Vec<TaggedWord> {
    tag_words_with(query, &[])
}

/// The inferred intent: winning category + crate + traits + attention scores.
#[derive(Clone, Debug)]
pub struct Intent {
    pub category: String,
    pub crate_name: String,
    pub description: String,
    pub traits: Vec<String>,
    /// Per-category attention (category name, weight) after softmax.
    pub attention: Vec<(String, f32)>,
}

/// Transformer-style labeled attention intent inference.
pub struct IntentAttention {
    categories: CategoryIndex,
    traits: std::collections::HashMap<&'static str, Vec<&'static str>>,
    /// Terms auto-learned from past unanswered queries.
    learned: Vec<String>,
}

impl IntentAttention {
    pub fn new() -> Self {
        Self {
            categories: CategoryIndex::from_data(),
            traits: crate::core::codex::CategoryIndex::crate_traits(),
            learned: crate::core::learn::Learner::load().term_words(),
        }
    }

    /// Infers the intent from role-tagged words attending to categories.
    ///
    /// # Details
    /// Each word is a head: Term words vote 3x for a matching category,
    /// Adj words vote 2x for categories holding a trait-matching crate,
    /// Verb words vote 1x for action categories. Scores are softmaxed into
    /// attention, and the winner's crate is picked by adjective traits.
    pub fn infer(&self, query: &str) -> Option<Intent> {
        let words = tag_words_with(query, &self.learned);
        let cats = &self.categories.categories;
        let mut scores = vec![0f32; cats.len()];

        for tw in &words {
            let q = tw.word.to_lowercase();
            for (ci, c) in cats.iter().enumerate() {
                let in_kw = |k: &String| {
                    let k = k.to_lowercase();
                    q == k || tw.word.to_lowercase().contains(&k)
                };
                match tw.role {
                    WordRole::Term => {
                        if c.keywords.iter().any(|k| in_kw(k)) {
                            scores[ci] += 3.0;
                        }
                    }
                    WordRole::Adj => {
                        if let Some(t) = adj_to_trait(&tw.word) {
                            // category holds a crate with this trait?
                            let has = c.crates.iter().any(|cr| {
                                self.traits
                                    .get(cr.as_str())
                                    .map_or(false, |ts| ts.contains(&t))
                            });
                            if has {
                                scores[ci] += 2.0;
                            }
                        }
                    }
                    WordRole::Verb => {
                        // verbs attend to any category whose keywords describe the action
                        if c.keywords.iter().any(|k| in_kw(k)) {
                            scores[ci] += 1.0;
                        }
                    }
                    _ => {}
                }
            }
        }

        let total: f32 = scores.iter().sum();
        if total <= 0.0 {
            return None;
        }
        // softmax attention
        let attention: Vec<(String, f32)> = cats
            .iter()
            .zip(scores.iter())
            .filter(|(_, s)| **s > 0.0)
            .map(|(c, s)| (c.name.clone(), s / total))
            .collect();
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)?;
        let cat = &cats[best_idx];

        // crate pick by adjective traits (adj -> trait -> crate)
        let quals: Vec<&str> = words
            .iter()
            .filter(|w| w.role == WordRole::Adj)
            .filter_map(|w| adj_to_trait(&w.word))
            .collect();
        let (crate_name, traits): (String, Vec<String>) = if quals.is_empty() {
            (cat.crates[0].clone(), vec![])
        } else {
            let best = cat
                .crates
                .iter()
                .map(|c| {
                    let t = self.traits.get(c.as_str()).cloned().unwrap_or_default();
                    let score = t.iter().filter(|t| quals.contains(t)).count();
                    (c, t, score)
                })
                .max_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(b.0)));
            match best {
                Some((c, t, _)) => (c.clone(), t.iter().map(|s| s.to_string()).collect()),
                None => (cat.crates[0].clone(), vec![]),
            }
        };

        Some(Intent {
            category: cat.name.clone(),
            crate_name,
            description: cat.description.clone(),
            traits,
            attention,
        })
    }
}

impl Default for IntentAttention {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Terms + adjectives tag correctly; 更快 + 随机数 -> 算法/fastrand.
    #[test]
    fn test_intent_fast_random() {
        let ia = IntentAttention::new();
        let it = ia.infer("我想要更快的随机数").expect("should infer");
        assert_eq!(it.category, "算法");
        assert_eq!(it.crate_name, "fastrand", "更快 -> fast -> fastrand");
        assert!(it.traits.contains(&"fast".to_string()));
    }

    /// Term words drive the domain.
    #[test]
    fn test_intent_parse_json() {
        let ia = IntentAttention::new();
        let it = ia.infer("怎么解析 JSON 数据").expect("should infer");
        assert_eq!(it.category, "编码数据");
        assert_eq!(it.crate_name, "serde_json");
    }

    /// Role tagging assigns the right labels.
    #[test]
    fn test_tag_roles() {
        let words = tag_words("我想要更快的随机数");
        assert!(
            words.iter().any(|w| w.role == WordRole::Adj && (w.word == "更快" || w.word == "快")),
            "fast adjective should be tagged Adj: {words:?}"
        );
        assert!(words.iter().any(|w| w.role == WordRole::Term && w.word == "随机数"));
        assert!(words.iter().any(|w| w.role == WordRole::Particle));
    }
}
