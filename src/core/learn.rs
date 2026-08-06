/*
 * @file learn.rs
 * @brief Auto-parameter update: the system learns from unanswered queries
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

//! FILE: learn.rs
//!
//! DESCRIPTION:
//! Auto-parameter update: when a query is unanswered (falls back to "don't
//! know"), the system jieba-segments it, extracts the unknown meaningful
//! words, and persists them so the next startup's vocabularies and category
//! keywords include them. The system literally updates its own parameters
//! from its misses.

use crate::core::codex::{Category, CategoryIndex};
use serde::{Deserialize, Serialize};

/// A word auto-learned from an unanswered query, tagged to a category.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LearnedTerm {
    pub term: String,
    pub category: String,
}

/// Words already known (particles / existing lexicon), never re-learned.
const KNOWN: &[&str] = &[
    "我", "你", "想", "要", "的", "了", "吗", "呢", "怎么", "如何", "为什么", "一个",
    "请", "帮", "这", "那", "是", "在", "和", "或", "给", "把", "用", "下", "里",
    "进行", "用来", "一下", "可以", "希望", "想要", "给我", "文件", "函数", "数据",
    "需要", "得到", "看看", "有没有", "什么", "处理", "实现", "使用", "应该", "哪个",
    "快", "快速", "轻量", "安全", "异步", "简单", "更快", "性能", "随机", "请求",
];

/// The auto-learning state, persisted to `data/learned.json`.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Learner {
    pub terms: Vec<LearnedTerm>,
}

impl Learner {
    /// Default persistence path.
    pub fn path() -> String {
        "data/learned.json".to_string()
    }

    /// Loads learned parameters (empty when absent).
    pub fn load() -> Self {
        std::fs::read_to_string(Self::path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persists the learned parameters.
    pub fn save(&self) {
        if let Some(parent) = std::path::Path::new(&Self::path()).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(Self::path(), json);
        }
    }

    /// Number of learned terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether nothing has been learned yet.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Extracts unknown meaningful words from an unanswered query and records
    /// them against the best-guess category.
    ///
    /// # Details
    /// jieba-segments the query; any word not already known (and not already
    /// learned) becomes a new term for the category that best matches the
    /// query. This is the "auto-update parameters from a miss" step.
    pub fn learn_from_miss(&mut self, query: &str) {
        let jieba = jieba_rs::Jieba::new();
        let cats = CategoryIndex::from_data();
        let cat = cats.recommend(query);
        for w in jieba.cut(query, false) {
            let w = w.trim().to_string();
            if w.chars().count() < 2 {
                continue;
            }
            if KNOWN.contains(&w.as_str()) {
                continue;
            }
            if self.terms.iter().any(|t| t.term == w) {
                continue;
            }
            let category = cat.map(|c| c.name.clone()).unwrap_or_else(|| "通用".to_string());
            self.terms.push(LearnedTerm { term: w, category });
        }
    }

    /// Applies learned terms to a category list: each term is appended to its
    /// category's keywords so future matching is stronger.
    pub fn apply_to(&self, categories: &mut Vec<Category>) {
        for lt in &self.terms {
            if let Some(c) = categories.iter_mut().find(|c| c.name == lt.category) {
                if !c.keywords.iter().any(|k| k == &lt.term) {
                    c.keywords.push(lt.term.clone());
                }
            }
        }
    }

    /// The learned term words (for intent term-tagging).
    pub fn term_words(&self) -> Vec<String> {
        self.terms.iter().map(|t| t.term.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unanswered query teaches new words.
    #[test]
    fn test_learn_from_miss() {
        let mut l = Learner::default();
        l.learn_from_miss("如何处理 Rust 的错误处理链");
        // it should have learned at least one meaningful new word
        assert!(l.len() >= 1, "should learn new terms, got {l:?}");
        // round-trip via serde on a temp path
        let tmp = std::env::temp_dir().join(format!("learned_test_{}.json", std::process::id()));
        let p = tmp.to_str().unwrap().to_string();
        std::fs::write(&p, serde_json::to_string(&l).unwrap()).unwrap();
        let raw = std::fs::read_to_string(&p).unwrap();
        let loaded: Learner = serde_json::from_str(&raw).unwrap();
        assert_eq!(loaded.len(), l.len());
        let _ = std::fs::remove_file(&p);
    }

    /// Applied learned terms extend the target category.
    #[test]
    fn test_apply_to() {
        let mut l = Learner::default();
        l.terms.push(LearnedTerm {
            term: "错误处理".to_string(),
            category: "日志调试".to_string(),
        });
        let mut cats = CategoryIndex::from_data().categories;
        l.apply_to(&mut cats);
        let c = cats.iter().find(|c| c.name == "日志调试").unwrap();
        assert!(c.keywords.iter().any(|k| k == "错误处理"));
    }
}
