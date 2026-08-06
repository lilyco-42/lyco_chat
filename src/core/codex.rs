/*
 * @file codex.rs
 * @brief ast-grep based crate API analysis (weighted code-block extraction)
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

//! FILE: codex.rs
//!
//! DESCRIPTION:
//! ast-grep based structural analysis of a crate's public API.
//!
//! BRIEF:
//! Walks a crate's Rust sources, extracts every `function_item` via
//! ast-grep, and scores each by how often its name is referenced across the
//! crate's own source (internal usage frequency = code-block weight). This
//! answers "which API is hot inside this crate" — the weighted method used
//! to pick a concrete API after a query routes to `cargo add <crate>`.

use ast_grep_core::matcher::KindMatcher;
use ast_grep_core::tree_sitter::StrDoc;
use ast_grep_core::AstGrep;
use serde::Deserialize;
use std::collections::HashMap;

/// Translates a Chinese term to English for crate search.
///
/// # Details
/// Uses a tech glossary; unseen terms return None (cannot search).
pub fn translate_term(zh: &str) -> Option<String> {
    const GLOSSARY: &[(&str, &str)] = &[
        ("区块链", "blockchain"), ("智能合约", "smart-contract"), ("合约", "contract"),
        ("打包", "package"), ("发布", "release"), ("部署", "deploy"), ("交易", "transaction"),
        ("钱包", "wallet"), ("挖矿", "mining"), ("代币", "token"), ("去中心化", "decentralized"),
        ("文件", "file"), ("图像", "image"), ("音频", "audio"), ("视频", "video"),
        ("数据库", "database"), ("网络", "network"), ("游戏", "game"), ("网页", "web"),
        ("爬虫", "crawler"), ("神经网络", "neural-network"), ("加密", "encryption"),
        ("哈希", "hash"), ("压缩", "compression"), ("排序", "sort"), ("解析", "parse"),
        ("日志", "log"), ("测试", "test"), ("命令行", "cli"), ("图形", "graphics"),
        ("聊天", "chat"), ("机器人", "bot"), ("支付", "payment"), ("流式", "stream"),
        ("网关", "gateway"), ("缓存", "cache"), ("队列", "queue"), ("消息", "message"),
        ("终端", "terminal"), ("渲染", "rendering"), ("仿真", "simulation"),
        ("地理", "geographic"), ("地图", "map"), ("时间", "time"), ("日期", "date"),
    ];
    GLOSSARY
        .iter()
        .find(|(z, _)| zh.contains(z) || *z == zh)
        .map(|(_, e)| e.to_string())
}

/// Searches crates.io for crates matching an English term.
///
/// # Returns
/// * `Vec<(name, description, keywords)>` - top results
pub fn search_crates(q: &str) -> Vec<(String, String, Vec<String>)> {
    let url = format!("https://crates.io/api/v1/crates?q={q}&per_page=5");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build();
    let Ok(client) = client else { return vec![] };
    let resp = client
        .get(&url)
        .header("User-Agent", "demo_chat/1.0 (crate tag learning)")
        .send();
    let Ok(resp) = resp else { return vec![] };
    let Ok(body) = resp.text() else { return vec![] };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else { return vec![] };
    let mut out = vec![];
    if let Some(crates) = v.get("crates").and_then(|c| c.as_array()) {
        for c in crates.iter().take(5) {
            let name = c.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
            let desc = c.get("description").and_then(|n| n.as_str()).unwrap_or("").to_string();
            let keywords: Vec<String> = c
                .get("keywords")
                .and_then(|k| k.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            if !name.is_empty() {
                out.push((name, desc, keywords));
            }
        }
    }
    out
}

/// Merges searched crates + tags into `data/crate_tags.json` (auto-update).
pub fn merge_crate_tags(found: &[(String, String, Vec<String>)]) {
    let path = std::env::current_dir()
        .unwrap_or_default()
        .join("data/crate_tags.json");
    let mut tags: HashMap<String, Vec<String>> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    for (name, _, keywords) in found {
        if !name.is_empty() && !tags.contains_key(name) {
            tags.insert(name.clone(), keywords.clone());
        }
    }
    if let Ok(json) = serde_json::to_string(&tags) {
        let _ = std::fs::write(&path, json);
    }
}

/// Learns a whole new domain: translate an unseen Chinese term, search
/// crates.io, persist the results, and return the top crate.
///
/// # Returns
/// * `Option<(crate, description, keywords)>`
pub fn learn_and_recommend(zh: &str) -> Option<(String, String, Vec<String>)> {
    let en = translate_term(zh)?;
    let found = search_crates(&en);
    if found.is_empty() {
        return None;
    }
    merge_crate_tags(&found);
    Some(found[0].clone())
}

/// One lib.rs category: description + keywords + recommended crates.
#[derive(Deserialize)]
struct CategoryRaw {
    category: String,
    description: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    crates: Vec<String>,
}

/// lib.rs category index: maps an intent to a recommended crate.
///
/// # Details
/// Uses the curated lib.rs categories (312K crates grouped by purpose) to
/// generalize the dynamic expert beyond a fixed concept table: any query is
/// segmented with jieba and matched against category keywords/descriptions,
/// returning the best crate + its description.
#[derive(Clone)]
pub struct CategoryIndex {
    pub categories: Vec<Category>,
}

#[derive(Clone, Debug)]
pub struct Category {
    pub name: String,
    pub description: String,
    pub keywords: Vec<String>,
    pub crates: Vec<String>,
}

impl CategoryIndex {
    /// Builds the index from `data/librs_categories.json`, then auto-merges
    /// any terms the system has learned from past unanswered queries.
    pub fn from_data() -> Self {
        let raw: Vec<CategoryRaw> =
            serde_json::from_str(include_str!("../../data/librs_categories.json"))
                .unwrap_or_default();
        let mut categories: Vec<Category> = raw
            .into_iter()
            .map(|c| Category {
                name: c.category,
                description: c.description,
                keywords: c.keywords,
                crates: c.crates,
            })
            .collect();
        // auto-update parameters: merge learned terms into their categories
        crate::core::learn::Learner::load().apply_to(&mut categories);
        Self { categories }
    }

    /// jieba-segments and matches an intent to the best category.
    ///
    /// # Details
    /// A keyword that is a full jieba *word* of the query is a strong signal
    /// (3x weight, longer keywords count more); a bare substring is weak.
    /// Case-insensitive, so "JSON" hits the `json` keyword.
    ///
    /// # Returns
    /// * `Option<&Category>` - Best matching category
    pub fn recommend(&self, query: &str) -> Option<&Category> {
        let jieba = jieba_rs::Jieba::new();
        let words: Vec<String> = jieba
            .cut(query, false)
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        let q = query.to_lowercase();
        let mut best: Option<(&Category, i32)> = None;
        for c in &self.categories {
            let mut score = 0i32;
            // category NAME words as jieba words are a strong signal
            // ("高性能的 HTTP 服务" -> 服务 -> HTTP 服务 beats http 客户端)
            for nw in c.name.split_whitespace() {
                let nw = nw.to_lowercase();
                if words.iter().any(|w| *w == nw) {
                    score += 4 * nw.chars().count() as i32;
                }
            }
            for k in &c.keywords {
                let k = k.to_lowercase();
                if words.iter().any(|w| *w == k) {
                    score += 3 * k.chars().count() as i32; // strong word match
                } else if q.contains(&k) {
                    score += k.chars().count() as i32; // weak substring
                }
            }
            if score > 0 && best.map_or(true, |(_, bs)| score > bs) {
                best = Some((c, score));
            }
        }
        best.map(|(c, _)| c)
    }

    /// Qualifier words in a query mapped to crate traits.
    ///
    /// # Details
    /// jieba-segments the query and matches each segmented word against an
    /// adjective -> trait table (快速/快/更快 -> fast -> fastrand).
    fn query_qualifiers(query: &str) -> Vec<&'static str> {
        let jieba = jieba_rs::Jieba::new();
        let words: Vec<String> = jieba
            .cut(query, false)
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let mut out: Vec<&'static str> = vec![];
        for (w, t) in [
            ("快速", "fast"), ("快", "fast"), ("更快", "fast"), ("超快", "fast"), ("极快", "fast"),
            ("高性能", "fast"), ("高速", "fast"), ("性能", "fast"),
            ("轻量", "lightweight"), ("轻", "lightweight"),
            ("简单", "simple"), ("易用", "simple"), ("简洁", "simple"),
            ("安全", "secure"), ("加密", "secure"),
            ("异步", "async"), ("async", "async"), ("并发", "async"),
            ("同步", "sync"), ("小", "small"), ("内存", "memory"),
        ] {
            // match as a segmented word (exact) or the word embeds it
            if words.iter().any(|word| *word == w || word.contains(w)) && !out.contains(&t) {
                out.push(t);
            }
        }
        out
    }

    /// Crate -> traits (curated from blessed.rs / knowledge of the ecosystem).
    pub fn crate_traits() -> HashMap<&'static str, Vec<&'static str>> {
        let m: &[(&str, &[&str])] = &[
            ("rand", &["general"]), ("fastrand", &["fast", "lightweight", "simple"]),
            ("ureq", &["lightweight", "simple", "sync"]), ("reqwest", &["full", "async"]),
            ("axum", &["ergonomic", "minimal", "async"]), ("actix-web", &["fast", "performance"]),
            ("rocket", &["ergonomic"]), ("hyper", &["low-level", "fast"]),
            ("sqlx", &["async", "compile-time"]), ("rusqlite", &["simple", "sync"]),
            ("diesel", &["sync", "strict"]), ("sea-orm", &["async", "orm"]),
            ("redis", &["sync"]), ("mongodb", &["async"]),
            ("flate2", &["general"]), ("zip", &["general"]), ("lz4_flex", &["fast", "lightweight"]),
            ("miniz_oxide", &["pure-rust", "lightweight"]),
            ("clap", &["full"]), ("pico-args", &["lightweight", "simple"]),
            ("lexopt", &["lightweight", "fast"]), ("bpaf", &["lightweight"]),
            ("serde_json", &["fast", "full"]), ("simd-json", &["fast"]),
            ("walkdir", &["simple"]), ("ignore", &["fast"]),
            ("tokio", &["async", "full"]), ("smol", &["lightweight", "async"]),
            ("rayon", &["parallel", "fast"]), ("crossbeam-channel", &["general"]),
            ("egui", &["simple", "immediate"]), ("iced", &["retained", "type-safe"]),
            ("slint", &["declarative"]), ("gtk4", &["native"]),
            ("chrono", &["general"]), ("jiff", &["fast", "nice-api"]), ("time", &["focused"]),
            ("image", &["full"]), ("qrcode", &["specialized"]),
        ];
        m.iter().map(|(k, v)| (*k, v.to_vec())).collect()
    }

    /// Personalized crate recommendation: category + query qualifiers.
    ///
    /// # Details
    /// Picks the crate in the category whose traits best match the query's
    /// qualifiers ("快速的随机数" -> fastfastrand), instead of always the first.
    ///
    /// # Returns
    /// * `Option<(String, String, Vec<String>)>` - (crate, category description,
    ///   matched traits)
    pub fn recommend_crate(&self, query: &str) -> Option<(String, String, Vec<String>)> {
        let cat = self.recommend(query)?;
        let quals = Self::query_qualifiers(query);
        if quals.is_empty() {
            return Some((cat.crates[0].clone(), cat.description.clone(), vec![]));
        }
        let traits = Self::crate_traits();
        let best = cat
            .crates
            .iter()
            .map(|c| {
                let t = traits
                    .get(c.as_str())
                    .cloned()
                    .unwrap_or_default();
                let score = t.iter().filter(|t| quals.contains(t)).count() as i32;
                (c, t, score)
            })
            .max_by(|a, b| {
                a.2.cmp(&b.2)
                    .then_with(|| a.1.len().cmp(&b.1.len()))
                    .then_with(|| a.0.cmp(b.0))
            })
            .map(|(c, t, _)| {
                (
                    c.clone(),
                    cat.description.clone(),
                    t.into_iter().map(|s| s.to_string()).collect(),
                )
            })?;
        Some(best)
    }
}

impl Default for CategoryIndex {
    fn default() -> Self {
        Self::from_data()
    }
}

/// Dynamic tag-neuron activation: each crate's crates.io tags are neurons;
/// a query activates the tags it matches and the most-activated crate wins,
/// like the brain firing the region whose semantic field is hit.
pub struct TagActivation {
    /// crate -> its crates.io keywords (neurons).
    pub crate_tags: HashMap<String, Vec<String>>,
    /// crate -> category name.
    pub category_of: HashMap<String, String>,
}

impl TagActivation {
    /// Loads tags (`data/crate_tags.json`) and category membership.
    pub fn load() -> Self {
        let tags: HashMap<String, Vec<String>> =
            serde_json::from_str(include_str!("../../data/crate_tags.json"))
                .unwrap_or_default();
        let mut category_of = HashMap::new();
        for c in CategoryIndex::from_data().categories {
            for cr in &c.crates {
                category_of.insert(cr.clone(), c.name.clone());
            }
        }
        Self {
            crate_tags: tags,
            category_of,
        }
    }

    /// Chinese / semantic words that fire specific English crate tags.
    fn word_tags() -> &'static [(&'static str, &'static [&'static str])] {
        &[
            ("随机", &["random", "rng", "rand"]), ("随机数", &["random", "rng"]),
            ("猜", &["random", "rand", "game"]), ("猜数字", &["random", "rand"]),
            ("快", &["fast"]), ("快速", &["fast"]), ("更快", &["fast"]), ("高速", &["fast"]),
            ("json", &["json", "serde", "serialization"]), ("解析", &["parse", "parser", "decoder"]),
            ("音视频", &["audio", "video", "multimedia", "media", "codec"]),
            ("音频", &["audio"]), ("视频", &["video"]), ("媒体", &["media", "multimedia"]),
            ("神经", &["machine-learning", "tensor", "ai"]), ("模型", &["tensor", "onnx", "machine-learning"]),
            ("http", &["http", "web", "routing"]), ("服务", &["web", "routing", "server"]),
            ("请求", &["http", "request"]), ("客户端", &["client", "http"]),
            ("数据库", &["database", "sql", "db"]), ("sql", &["sql", "database"]), ("数据", &["database", "data"]),
            ("压缩", &["compression", "compress", "gzip"]), ("加密", &["crypto", "cryptography"]),
            ("安全", &["crypto", "security"]), ("密码", &["crypto", "password", "hash"]),
            ("日志", &["log", "logging"]), ("测试", &["testing", "test", "mock"]),
            ("线程", &["concurrency", "thread"]), ("并行", &["parallel", "concurrency"]),
            ("异步", &["async", "tokio"]), ("并发", &["concurrency", "async"]),
            ("图形", &["gpu", "graphics", "rendering"]), ("游戏", &["game", "bevy"]),
            ("文件", &["file", "fs", "io"]), ("文件系统", &["file", "fs"]),
            ("图像", &["image", "png", "jpeg"]), ("图片", &["image"]),
            ("命令行", &["cli", "arg"]), ("参数", &["cli", "arg"]), ("终端", &["cli", "tui"]),
            ("时间", &["time", "datetime"]), ("日期", &["datetime", "date", "time"]),
            ("邮件", &["email", "smtp", "mail"]), ("压缩", &["compression", "compress"]),
        ]
    }

    /// Activates tag neurons with a query's words.
    ///
    /// # Details
    /// Each jieba word fires a crate's tag when it matches exactly, the tag is
    /// contained in the word, the word is contained in the tag (case-insensitive),
    /// or a Chinese synonym (WORD_TAGS) maps the word to English tags.
    /// Generic words (rust/cargo/语言/写...) are skipped — they should not
    /// fire crate tags for concept questions.
    ///
    /// # Returns
    /// * `Vec<(crate, activation, fired_tags)>` - descending by activation
    pub fn activate(&self, query: &str) -> Vec<(String, f32, Vec<String>)> {
        const STOP: &[&str] = &[
            "rust", "cargo", "语言", "编程", "代码", "一个", "用", "写", "怎么", "如何",
            "的", "什么", "是", "有", "吗", "呢", "请", "帮", "需要", "想要", "我想",
            "能", "可以", "该", "要", "在", "和", "或", "给", "把", "下", "里",
        ];
        let jieba = jieba_rs::Jieba::new();
        let words: Vec<String> = jieba
            .cut(query, false)
            .into_iter()
            .map(|s| s.to_lowercase())
            .filter(|w| !STOP.contains(&w.as_str()))
            .collect();
        let mut out: Vec<(String, f32, Vec<String>)> = vec![];
        for (crate_name, tags) in &self.crate_tags {
            let mut fired: Vec<String> = vec![];
            let mut act = 0.0f32;
            for tag in tags {
                let t = tag.to_lowercase();
                // direct match: exact, or substring only for tags/words of
                // length >= 3 ("ml" must not fire from inside "toml")
                let direct = words.iter().any(|w| {
                    *w == t
                        || (t.len() >= 3 && w.contains(&t))
                        || (w.len() >= 3 && t.contains(w))
                });
                let syn = Self::word_tags().iter().any(|(w, ts)| {
                    ts.iter().any(|x| *x == t)
                        && words.iter().any(|qw| *qw == *w || qw.contains(w) || w.contains(qw))
                });
                if direct {
                    act += 2.0; // direct word-tag match is a stronger signal
                    fired.push(tag.clone());
                } else if syn {
                    act += 1.0; // Chinese synonym -> English tag
                    fired.push(tag.clone());
                }
            }
            if act > 0.0 {
                out.push((crate_name.clone(), act, fired));
            }
        }
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap()
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    /// Best activated crate: fires neurons, prefers the recommended category.
    ///
    /// # Returns
    /// * `Option<(crate, category, activation, fired_tags)>`
    pub fn best(&self, query: &str) -> Option<(String, String, f32, Vec<String>)> {
        let mut ranked = self.activate(query);
        if ranked.is_empty() {
            return None;
        }
        // prefer a crate in the intent's recommended category
        let cat = CategoryIndex::from_data().recommend(query).map(|c| c.name.clone());
        if let Some(cat_name) = cat {
            if let Some(pos) = ranked.iter().position(|(c, _, _)| {
                self.category_of.get(c).map(|x| x == &cat_name).unwrap_or(false)
            }) {
                let (c, a, f) = ranked.remove(pos);
                return Some((c, cat_name, a, f));
            }
        }
        let (c, a, f) = ranked.remove(0);
        let cat_name = self
            .category_of
            .get(&c)
            .cloned()
            .unwrap_or_else(|| "通用".to_string());
        Some((c, cat_name, a, f))
    }
}

impl Default for TagActivation {
    fn default() -> Self {
        Self::load()
    }
}

/// A public function discovered in a crate, with its usage weight.
#[derive(Debug, Clone)]
pub struct ApiInfo {
    pub name: String,
    /// Number of times the name appears across the crate's own sources.
    pub weight: usize,
    /// Signature text (everything before the function body).
    pub signature: String,
    /// First `///` doc-comment line above the function (the documented answer).
    pub doc: String,
}

fn name_from_text(text: &str) -> String {
    text.split_whitespace()
        .skip_while(|w| *w != "fn")
        .nth(1)
        .unwrap_or_default()
        .split(['<', '(', ':', ' '])
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Analyzes every `.rs` file under `dir` and returns public functions ranked
/// by internal reference weight (highest first).
///
/// # Arguments
/// * `dir` - Crate source directory
///
/// # Returns
/// * `Vec<ApiInfo>` - Functions with weight, hot first
pub fn analyze_crate_api(dir: &str) -> Vec<ApiInfo> {
    // gather all .rs sources
    let mut files: Vec<String> = vec![];
    let mut stack = vec![dir.to_string()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p.to_string_lossy().to_string());
                } else if p.extension().map_or(false, |x| x == "rs") {
                    files.push(p.to_string_lossy().to_string());
                }
            }
        }
    }

    // 1) extract every function_item (ignores generics/attributes)
    let mut sigs: Vec<(String, String, String)> = vec![]; // (name, signature, doc)
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap_or_default();
        let Ok(doc) = StrDoc::try_new(&src, ast_grep_language::Rust) else {
            continue;
        };
        let root = AstGrep::doc(doc);
        let km = KindMatcher::new("function_item", ast_grep_language::Rust);
        for m in root.root().find_all(&km) {
            let text = m.text().to_string();
            // only truly public functions: `pub fn` (reject pub(crate)/pub(super) and private fn)
            if !text.starts_with("pub ") {
                continue;
            }
            let name = name_from_text(&text);
            if name.is_empty() || name.starts_with('_') {
                continue;
            }
            let signature = text
                .split('{')
                .next()
                .unwrap_or(&text)
                .trim()
                .to_string();
            // doc comment: the `///` lines above the function, skipping
            // `#[...]` attributes and blank lines in between
            let start = m.range().start.min(src.len());
            let mut docs: Vec<String> = vec![];
            for l in src[..start].lines().rev() {
                let t = l.trim_start();
                if t.starts_with("///") {
                    docs.push(t.trim_start_matches("///").trim().to_string());
                } else if t.starts_with('#') || t.is_empty() {
                    continue; // attribute or blank line
                } else {
                    break;
                }
            }
            let doc = docs.into_iter().rev().collect::<Vec<_>>().join(" ");
            sigs.push((name, signature, doc));
        }
    }

    // 2) weight = references across all sources
    let mut all_src = String::new();
    for f in &files {
        all_src.push_str(&std::fs::read_to_string(f).unwrap_or_default());
    }
    let mut weight: HashMap<String, usize> = HashMap::new();
    for (name, _, _) in &sigs {
        *weight.entry(name.clone()).or_insert(0) += all_src.matches(name.as_str()).count();
    }

    // 3) rank, dedup by name, hottest first
    let mut seen: HashMap<String, String> = HashMap::new();
    let mut ranked: Vec<ApiInfo> = sigs
        .into_iter()
        .map(|(name, signature, doc)| ApiInfo {
            weight: weight[&name],
            name: name.clone(),
            signature,
            doc,
        })
        .filter(|a| seen.insert(a.name.clone(), a.signature.clone()).is_none())
        .collect();
    ranked.sort_by(|a, b| b.weight.cmp(&a.weight).then_with(|| a.name.cmp(&b.name)));
    ranked
}

/// Locates a crate's extracted source dir inside the local cargo registry,
/// cross-platform.
///
/// # Details
/// Cargo home is resolved the official cross-platform way (`home::cargo_home`:
/// `$CARGO_HOME`, else `$HOME/.cargo` on Unix / `%USERPROFILE%\.cargo` on
/// Windows). The registry `src` tree is `registry/src/<hash>/<crate>-<ver>`,
/// so we walk the hash dirs and match `<crate>-` where the character right
/// after the dash is a digit (a version), which distinguishes real crates
/// from names like `tokio-util`.
///
/// # Arguments
/// * `crate_name` - e.g. "rand"
///
/// # Returns
/// * `Option<String>` - Path to the crate's `src` directory
pub fn find_crate_src(crate_name: &str) -> Option<String> {
    let home = home::cargo_home().ok()?;
    let registry = home.join("registry").join("src");
    let needle = format!("{crate_name}-");
    let needle_len = needle.len();

    let mut dirs: Vec<std::path::PathBuf> = vec![];
    let Ok(top) = std::fs::read_dir(&registry) else {
        return None;
    };
    for e in top.flatten() {
        let d = e.path();
        if !d.is_dir() {
            continue;
        }
        // crates live under registry/src/<hash>/<crate>-<version>
        if let Ok(inner) = std::fs::read_dir(&d) {
            for e2 in inner.flatten() {
                if e2.path().is_dir() {
                    dirs.push(e2.path());
                }
            }
        }
    }
    for d in dirs {
        let name = d.file_name().map(|n| n.to_string_lossy().to_string())?;
        if !name.starts_with(&needle) {
            continue;
        }
        let after = name[needle_len.min(name.len())..].chars().next();
        let has_version = after.map_or(false, |c| c.is_ascii_digit());
        if !has_version {
            continue;
        }
        let src = d.join("src");
        if src.is_dir() {
            return Some(src.to_string_lossy().to_string());
        }
    }
    None
}

/// Lists all crates found in the local cargo registry (cross-platform).
///
/// # Returns
/// * `Vec<(String, String)>` - (crate name, extracted src dir)
pub fn list_registry_crates() -> Vec<(String, String)> {
    let mut out = vec![];
    let Ok(home) = home::cargo_home() else {
        return out;
    };
    let registry = home.join("registry").join("src");
    let Ok(top) = std::fs::read_dir(&registry) else {
        return out;
    };
    for e in top.flatten() {
        let d = e.path();
        if !d.is_dir() {
            continue;
        }
        if let Ok(inner) = std::fs::read_dir(&d) {
            for e2 in inner.flatten() {
                let p = e2.path();
                if p.is_dir() {
                    let name = p
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let src = p.join("src");
                    if src.is_dir() {
                        out.push((name, src.to_string_lossy().to_string()));
                    }
                }
            }
        }
    }
    out
}

/// Resolves a crate's extracted source: local registry first, then download
/// from crates.io into `/tmp/opencode/cratesrc` (no project dependency added).
///
/// # Returns
/// * `Option<String>` - Path to the crate's `src` directory
pub fn crate_src_dir(crate_name: &str) -> Option<String> {
    if let Some(local) = find_crate_src(crate_name) {
        return Some(local);
    }
    download_crate_src(crate_name)
}

/// Downloads + extracts a crate's source from crates.io into a temp cache.
///
/// # Returns
/// * `Option<String>` - Path to the extracted crate's `src` directory
pub fn download_crate_src(crate_name: &str) -> Option<String> {
    let cache_root = std::path::PathBuf::from("/tmp/opencode/cratesrc");
    let crate_file = cache_root.join(format!("{crate_name}.crate"));
    // find an already-extracted <name>-<ver> dir
    if let Ok(rd) = std::fs::read_dir(&cache_root) {
        let mut extracted: Vec<_> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .filter(|p| {
                let n = p.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
                n.starts_with(&format!("{crate_name}-")) && n.len() > crate_name.len()
            })
            .collect();
        extracted.sort();
        if let Some(dir) = extracted.pop() {
            let src = dir.join("src");
            if src.is_dir() {
                return Some(src.to_string_lossy().to_string());
            }
        }
    }
    std::fs::create_dir_all(&cache_root).ok()?;
    let client = reqwest::blocking::Client::builder()
        .user_agent("rust-learning-demo (ast-grep-tool)")
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .ok()?;
    // resolve the latest version first
    let meta = client
        .get(format!("https://crates.io/api/v1/crates/{crate_name}"))
        .send()
        .ok()?;
    let version = serde_json::from_slice::<serde_json::Value>(&meta.bytes().ok()?)
        .ok()?
        .get("crate")
        .and_then(|c| c.get("max_version"))
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    if version.is_empty() {
        return None;
    }
    let url = format!("https://crates.io/api/v1/crates/{crate_name}/{version}/download");
    let resp = client.get(&url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let bytes = resp.bytes().ok()?;
    std::fs::write(&crate_file, &bytes).ok()?;
    // extract tar.gz via system tar
    let out = std::process::Command::new("tar")
        .arg("xzf")
        .arg(&crate_file)
        .current_dir(&cache_root)
        .status()
        .ok()?;
    if !out.success() {
        return None;
    }
    let mut extracted: Vec<String> = vec![];
    if let Ok(rd) = std::fs::read_dir(&cache_root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                let n = p.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
                if n.starts_with(&format!("{crate_name}-")) && n.len() > crate_name.len() {
                    extracted.push(p.to_string_lossy().to_string());
                }
            }
        }
    }
    extracted.sort();
    let dir = extracted.pop()?;
    let src = std::path::PathBuf::from(dir).join("src");
    if src.is_dir() {
        Some(src.to_string_lossy().to_string())
    } else {
        None
    }
}

/// Chinese domain words mapped to the English API keywords they imply.
fn domain_keywords(ask: &str) -> Vec<String> {
    let mut kws: Vec<String> = vec![];
    let jieba = jieba_rs::Jieba::new();
    for w in jieba.cut(ask, false) {
        let w = w.trim();
        if w.chars().count() < 2 {
            continue;
        }
        let lower = w.to_lowercase();
        kws.push(lower.clone());
        // Chinese -> English domain words (if it looks like one we map)
        for (zh, en) in [
            ("路由", "route"), ("中间件", "middleware"), ("请求", "request"),
            ("响应", "response"), ("处理", "handler"), ("新建", "new"),
            ("创建", "new"), ("配置", "config"), ("服务", "server"),
            ("启动", "serve"), ("监听", "listen"), ("读取", "read"),
            ("写入", "write"), ("发送", "send"), ("连接", "connect"),
            ("查询", "query"), ("解析", "parse"), ("解析", "from"), ("格式化", "format"),
            ("测试", "test"), ("错误", "error"), ("日志", "log"),
            ("生成", "gen"), ("返回", "into"), ("状态", "status"),
            ("定义", "fn"), ("结构", "struct"), ("特征", "trait"),
            ("异常", "error"), ("线程", "spawn"), ("异步", "async"),
        ] {
            if w == zh || w.contains(zh) || zh.contains(w) {
                kws.push(en.to_string());
            }
        }
    }
    kws
}

/// `/tool:ast-grep <crate> <ask>` — ast-grep the crate's pub API and surface
/// the functions whose names / signatures / docs match the ask's keywords.
///
/// # Returns
/// * `String` - matched pub functions (or the hottest ones) with their docs
pub fn ast_grep_tool(crate_name: &str, ask: &str) -> String {
    let Some(src) = crate_src_dir(crate_name) else {
        return format!("找不到 {crate_name} 的源码（本地 registry 与 crates.io 都没有）");
    };
    let apis = analyze_crate_api(&src);
    if apis.is_empty() {
        return format!("ast-grep 扫描 {crate_name} 未发现公开函数");
    }
    let kws = domain_keywords(ask);
    // score each pub fn: name hits strongest, doc next, signature lightest
    // score per dimension; sort by fn-NAME match first so `route`/`new`
    // (the actual answer) beat functions whose docs merely mention the words
    let scored: Vec<((usize, usize, usize), &ApiInfo)> = apis
        .iter()
        .map(|a| {
            let n = a.name.to_lowercase();
            let sig = a.signature.to_lowercase();
            let doc = a.doc.to_lowercase();
            let mut name_s = 0usize;
            let mut doc_s = 0usize;
            let mut sig_s = 0usize;
            for k in &kws {
                if n == *k {
                    name_s += 4;
                } else if n.starts_with(k.as_str())
                    || n.split(['_', ' ']).any(|p| p == k.as_str())
                {
                    name_s += 2;
                }
                if doc.contains(k) {
                    doc_s += 2;
                }
                if sig.contains(k) {
                    sig_s += 1;
                }
            }
            ((name_s, doc_s, sig_s), a)
        })
        .collect();
    let mut ranked = scored;
    ranked.sort_by(|a, b| {
        (b.0).0
            .cmp(&(a.0).0)
            .then_with(|| (b.0).1.cmp(&(a.0).1))
            .then_with(|| (b.0).2.cmp(&(a.0).2))
            .then_with(|| b.1.weight.cmp(&a.1.weight))
    });
    let best_name = ranked[0].0 .0;
    let hits: Vec<&ApiInfo> = if best_name > 0 {
        ranked
            .iter()
            .filter(|((n, _, _), _)| *n > 0)
            .take(6)
            .map(|(_, a)| *a)
            .collect()
    } else {
        ranked.iter().take(5).map(|(_, a)| *a).collect()
    };
    let mut out = format!(
        "ast-grep {} 源码分析（共 {} 个公开函数，命中“{}”）:",
        crate_name,
        apis.len(),
        if ask.is_empty() { "热门" } else { ask }
    );
    for a in hits {
        let doc = if a.doc.is_empty() { String::new() } else {
            format!("  · {}", a.doc.chars().take(110).collect::<String>())
        };
        out.push_str(&format!("\n· {}\n{}", a.signature.chars().take(160).collect::<String>(), doc));
    }
    out
}

/// Dispatches a `/tool:*` command. Returns the tool's answer if the query is
/// a tool invocation, else `None` (normal routing continues).
pub fn run_tool(query: &str) -> Option<String> {
    let rest = query.trim().strip_prefix("/tool:")?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let tool = parts.next().unwrap_or_default().trim();
    let after = parts.next().unwrap_or_default().trim();
    match tool {
        "ast-grep" => {
            if after.is_empty() {
                return Some("用法: /tool:ast-grep <crate> <问>，例如 /tool:ast-grep axum 帮我找找怎么新建路由".into());
            }
            let (name, ask) = match after.find(char::is_whitespace) {
                Some(i) => (&after[..i], after[i..].trim()),
                None => (after, ""),
            };
            Some(ast_grep_tool(name, ask))
        }
        other => Some(format!("未知工具 {other}（目前支持: ast-grep）")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_src(content: &str) -> String {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lib.rs");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        dir.to_string_lossy().to_string()
    }

    /// Tests extraction keeps only public functions and ranks by reference weight.
    #[test]
    fn test_analyze_filters_and_ranks() {
        let dir = temp_src(
            "pub fn alpha() -> u8 { 1 }\n\
             pub(crate) fn hidden() {}\n\
             fn private() {}\n\
             pub fn beta() -> u8 { alpha() + alpha() + alpha() }\n",
        );
        let ranked = analyze_crate_api(&dir);
        let names: Vec<&str> = ranked.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"alpha"), "public fn should be kept: {names:?}");
        assert!(names.contains(&"beta"));
        assert!(!names.contains(&"hidden"), "pub(crate) must be filtered");
        assert!(!names.contains(&"private"), "private fn must be filtered");
        // alpha referenced 3 times inside beta -> hotter than beta
        let alpha = ranked.iter().find(|a| a.name == "alpha").unwrap();
        let beta = ranked.iter().find(|a| a.name == "beta").unwrap();
        assert!(alpha.weight > beta.weight, "alpha should outrank beta");
    }

    /// Tests registry listing returns real crates, not tokio-util style noise.
    #[test]
    fn test_list_registry() {
        let crates = list_registry_crates();
        let names: Vec<&str> = crates.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("rand-")), "rand should be listed: {names:?}");
        // version digit check: no crate named with a non-numeric suffix after `rand-`
        assert!(!names.iter().any(|n| n.starts_with("rand-") && !n[5..].starts_with(char::is_numeric)));
    }
}

#[cfg(test)]
mod codex_tests {
    use super::*;

    /// Personalized: "快速的随机数" recommends fastrand over rand.
    #[test]
    fn test_recommend_personalized_fast() {
        let idx = CategoryIndex::from_data();
        let (crate_name, _, traits) = idx.recommend_crate("我想要快速的随机数").expect("should recommend");
        assert_eq!(crate_name, "fastrand", "fast random should pick fastrand");
        assert!(traits.iter().any(|t| t == "fast"));
    }

    /// Default (no qualifier) recommends the de facto standard first crate.
    #[test]
    fn test_recommend_default() {
        let idx = CategoryIndex::from_data();
        let (crate_name, _, _) = idx.recommend_crate("怎么生成随机数").expect("should recommend");
        assert_eq!(crate_name, "rand");
    }

    /// Category name words matter: "HTTP 服务" beats "http 客户端" for servers.
    #[test]
    fn test_recommend_server_vs_client() {
        let idx = CategoryIndex::from_data();
        let (crate_name, _, _) = idx.recommend_crate("高性能的 HTTP 服务").expect("should recommend");
        let cat = idx.recommend("高性能的 HTTP 服务").unwrap();
        assert_eq!(cat.name, "HTTP 服务", "server intent should pick HTTP 服务, got {}", cat.name);
        assert!(!crate_name.is_empty());
    }

    /// 快速 / 快 / 更快 all route to the "fast" intent -> fastrand.
    #[test]
    fn test_recommend_fast_adjectives() {
        let idx = CategoryIndex::from_data();
        for q in ["我想要更快的随机数", "我想要快速的随机数", "我想要快的随机数"] {
            let (c, _, traits) = idx.recommend_crate(q).expect("should recommend");
            assert_eq!(c, "fastrand", "query {q} should pick fastrand");
            assert!(traits.contains(&"fast".to_string()));
        }
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;

    /// A query fires the tags of matching crates; fastrand wins for "fast random".
    #[test]
    fn test_activate_fast_random() {
        let ta = TagActivation::load();
        let ranked = ta.activate("我想要更快的随机数");
        assert!(!ranked.is_empty());
        let top = &ranked[0];
        assert_eq!(top.0, "fastrand", "fast random should activate fastrand, got {:?}", ranked.iter().map(|r| &r.0).take(5).collect::<Vec<_>>());
        assert!(top.2.contains(&"fast".to_string()), "fast tag should fire: {:?}", top.2);
    }

    /// Tag neurons fire for the right domain (category-aware best()).
    #[test]
    fn test_activate_json() {
        let ta = TagActivation::load();
        let b = ta.best("怎么解析 JSON").expect("should activate");
        assert_eq!(b.0, "serde_json", "json should best-activate serde_json, got {}", b.0);
        assert!(b.3.iter().any(|t| t == "json"), "json tag should fire: {:?}", b.3);
    }

    /// best() prefers the recommended category's crate.
    #[test]
    fn test_best() {
        let ta = TagActivation::load();
        let b = ta.best("怎么处理音视频").expect("should activate");
        assert!(b.2 > 0.0);
        assert!(!b.3.is_empty(), "should have fired tags");
    }
}
