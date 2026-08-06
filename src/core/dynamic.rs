/*
 * @file dynamic.rs
 * @brief Dynamic expert registry: learn crate APIs after routing
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

//! FILE: dynamic.rs
//!
//! DESCRIPTION:
//! Dynamic expert registry for crate-level to specific-API abstraction.
//!
//! BRIEF:
//! When a query is routed to a crate-level answer (`cargo add rand`), the
//! system "acquires" that crate's key API (the functions/interfaces a
//! developer learns after adding the crate) and registers a new, more
//! specific expert. Subsequent identical questions route straight to the
//! concrete API — the expert system grows at runtime.

use jieba_rs::Jieba;

/// A runtime-acquired expert: one concept's concrete API for a crate.
#[derive(Clone, Debug)]
pub struct DynamicExpert {
    pub crate_name: String,
    pub concept: String,
    pub keywords: Vec<String>,
    pub api_answer: String,
}

/// Built-in crate API profiles, simulating what one learns from crate docs
/// (in production these would be fetched from docs.rs after `cargo add`).
struct CrateApiProfile {
    crate_name: &'static str,
    concept: &'static str,
    keywords: &'static [&'static str],
    api: &'static str,
}

const PROFILES: &[CrateApiProfile] = &[
    CrateApiProfile {
        crate_name: "rand",
        concept: "生成随机数",
        keywords: &["随机数", "随机", "rand"],
        api: "use rand::Rng; let mut rng = rand::thread_rng(); let n: u32 = rng.gen_range(1..=100);",
    },
    CrateApiProfile {
        crate_name: "reqwest",
        concept: "发送HTTP请求",
        keywords: &["HTTP", "请求", "reqwest", "网络"],
        api: "let client = reqwest::Client::new(); let resp = client.get(\"https://example.com\").send().await?;",
    },
    CrateApiProfile {
        crate_name: "sqlx",
        concept: "连接数据库",
        keywords: &["数据库", "sqlx", "连接"],
        api: "let pool = sqlx::SqlitePool::connect(\"sqlite::memory:\").await?;",
    },
    CrateApiProfile {
        crate_name: "clap",
        concept: "解析命令行参数",
        keywords: &["命令行", "参数", "clap", "解析"],
        api: "#[derive(clap::Parser)] struct Args { name: String } let args = Args::parse();",
    },
    CrateApiProfile {
        crate_name: "regex",
        concept: "正则表达式",
        keywords: &["正则", "匹配", "regex"],
        api: "let re = regex::Regex::new(r\"^\\d+$\")?; re.is_match(\"123\");",
    },
    CrateApiProfile {
        crate_name: "serde_json",
        concept: "解析JSON",
        keywords: &["JSON", "json", "解析", "serde"],
        api: "let v: serde_json::Value = serde_json::from_str(data)?;",
    },
];

/// Dynamic, growing expert registry.
pub struct DynamicRegistry {
    jieba: Jieba,
    pub experts: Vec<DynamicExpert>,
}

impl DynamicRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            jieba: Jieba::new(),
            experts: Vec::new(),
        }
    }

    /// Learns a new expert at runtime.
    ///
    /// # Arguments
    /// * `crate_name` - Crate the API belongs to
    /// * `concept` - Concept the API answers
    /// * `keywords` - Words that activate this expert
    /// * `api_answer` - Concrete API code
    pub fn learn(&mut self, crate_name: &str, concept: &str, keywords: &[&str], api_answer: &str) {
        self.experts.push(DynamicExpert {
            crate_name: crate_name.to_string(),
            concept: concept.to_string(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            api_answer: api_answer.to_string(),
        });
    }

    /// Acquires a crate's key API after it has been routed to.
    ///
    /// # Details
    /// Called when a query routes to `cargo add <crate>`. Looks up the
    /// built-in profile and registers the crate's concrete API as a new,
    /// more specific expert. Returns the learned API answer if any.
    ///
    /// # Arguments
    /// * `crate_name` - e.g. "rand"
    ///
    /// # Returns
    /// * `Option<String>` - The newly learned API answer
    pub fn acquire(&mut self, crate_name: &str) -> Option<String> {
        if self.experts.iter().any(|e| e.crate_name == crate_name) {
            return self
                .experts
                .iter()
                .find(|e| e.crate_name == crate_name)
                .map(|e| e.api_answer.clone());
        }
        let p = PROFILES.iter().find(|p| p.crate_name == crate_name)?;
        self.learn(p.crate_name, p.concept, p.keywords, p.api);
        Some(p.api.to_string())
    }

    /// Learns a crate's hot API directly from its source via ast-grep.
    ///
    /// # Details
    /// This is the weighted code-block method: the crate's `function_item`s
    /// are extracted with ast-grep and ranked by internal reference weight;
    /// the hottest function becomes the specific API answer. The expert is
    /// registered so the same question now routes to the concrete API.
    ///
    /// # Arguments
    /// * `crate_name` - e.g. "rand"
    /// * `src_dir` - Crate source directory
    ///
    /// # Returns
    /// * `Option<(String, String)>` - (hottest function name, its signature)
    pub fn acquire_from_source(&mut self, crate_name: &str, src_dir: &str) -> Option<(String, String)> {
        if let Some(e) = self.experts.iter().find(|e| e.crate_name == crate_name) {
            return Some((e.concept.clone(), e.api_answer.clone()));
        }
        let ranked = crate::core::codex::analyze_crate_api(src_dir);
        let hot = ranked.into_iter().next()?;
        let (name, signature) = (hot.name.clone(), hot.signature.clone());
        let keywords: Vec<String> = vec![crate_name.to_string(), name.clone()];
        self.experts.push(DynamicExpert {
            crate_name: crate_name.to_string(),
            concept: name.clone(),
            keywords,
            api_answer: signature.clone(),
        });
        Some((name, signature))
    }

    /// Routes a query to the most specific learned expert.
    ///
    /// # Details
    /// jieba-segments the query and matches expert keywords (word or
    /// substring). Returns the expert with the most weighted keyword hits.
    ///
    /// # Arguments
    /// * `query` - User question
    ///
    /// # Returns
    /// * `Option<&DynamicExpert>` - Best matching learned expert
    pub fn route(&self, query: &str) -> Option<&DynamicExpert> {
        let words: Vec<String> = self
            .jieba
            .cut(query, false)
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let mut best: Option<(&DynamicExpert, usize)> = None;
        for e in &self.experts {
            let hits = e
                .keywords
                .iter()
                .filter(|k| words.iter().any(|w| w == *k) || query.contains(k.as_str()))
                .count();
            if hits > 0 && best.map_or(true, |(_, bh)| hits > bh) {
                best = Some((e, hits));
            }
        }
        best.map(|(e, _)| e)
    }

    /// Number of learned experts.
    pub fn len(&self) -> usize {
        self.experts.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }
}

impl Default for DynamicRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests acquiring a crate learns its concrete API.
    #[test]
    fn test_acquire_learns_api() {
        let mut r = DynamicRegistry::new();
        assert!(r.is_empty());
        let api = r.acquire("rand").expect("rand profile exists");
        assert!(api.contains("gen_range"), "rand API should mention gen_range");
        assert_eq!(r.len(), 1);
    }

    /// Tests a second acquire does not duplicate the expert.
    #[test]
    fn test_acquire_idempotent() {
        let mut r = DynamicRegistry::new();
        r.acquire("rand");
        r.acquire("rand");
        assert_eq!(r.len(), 1);
    }

    /// Tests the learned expert routes the same question to the specific API.
    #[test]
    fn test_route_to_specific_api() {
        let mut r = DynamicRegistry::new();
        r.acquire("rand");
        let e = r.route("怎么生成随机数").expect("should route to rand expert");
        assert_eq!(e.crate_name, "rand");
        assert!(e.api_answer.contains("gen_range"));
    }

    /// Tests unknown crates acquire nothing and unrelated queries route nowhere.
    #[test]
    fn test_unknown() {
        let mut r = DynamicRegistry::new();
        assert!(r.acquire("not_a_crate").is_none());
        let e = r.route("今天天气怎么样");
        assert!(e.is_none());
    }

    /// Tests learning from a crate's source registers the hot API.
    #[test]
    fn test_acquire_from_source() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("dyn_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("lib.rs")).unwrap();
        f.write_all(
            b"pub fn hot() -> u8 { 1 }\npub fn cold() -> u8 { hot() }\n",
        )
        .unwrap();
        let mut r = DynamicRegistry::new();
        let learned = r.acquire_from_source("demo_crate", &dir.to_str().unwrap());
        assert!(learned.is_some(), "should learn the hot API");
        let (name, _) = learned.unwrap();
        assert_eq!(name, "hot");
        assert_eq!(r.len(), 1);
    }
}
