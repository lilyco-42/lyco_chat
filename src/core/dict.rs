/*
 * @file dict.rs
 * @brief Generic dictionary routing expert over four specialist dictionaries
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

//! FILE: dict.rs
//!
//! DESCRIPTION:
//! Generic routing expert that unifies four specialist dictionaries.
//!
//! BRIEF:
//! English, Chinese, Chinese->English translation, and programming-term
//! dictionaries all implement one `Dictionary` trait. A generic `DictRouter`
//! scores a query against every specialist's entries and returns the best
//! hits across sources, so "one router, many dictionaries".

use crate::core::semantic::SemanticMoE;
use serde::Deserialize;
use std::collections::HashSet;

/// One entry of the high-quality dictionary preprocessed from the corpora.
#[derive(Deserialize)]
struct DictEntryRaw {
    term: String,
    kind: String,
    en: String,
    zh: String,
    code: String,
    synonyms: Vec<String>,
}

/// Builds Chinese / ZhEn / Program definitions from `data/dictionary.json`.
///
/// # Details
/// The dictionary is preprocessed from the collected corpora (std docs +
/// cheats.rs + blessed.rs + areweguiyet): 146 bilingual entries with code and
/// near-synonym question clusters, far beyond the hand-written seeds.
fn corpus_definitions() -> (Vec<Definition>, Vec<Definition>, Vec<Definition>) {
    let mut zh = Vec::new();
    let mut zhen = Vec::new();
    let mut prog = Vec::new();
    let raw: Vec<DictEntryRaw> =
        serde_json::from_str(include_str!("../../data/dictionary.json")).unwrap_or_default();
    for e in raw {
        zh.push(Definition {
            term: e.term.clone(),
            meanings: vec![e.zh.clone()],
            synonyms: e.synonyms.clone(),
            origin: "语料",
            priority: 1,
        });
        zhen.push(Definition {
            term: e.term.clone(),
            meanings: vec![e.en.clone()],
            synonyms: vec![e.code.clone()],
            origin: "语料",
            priority: 1,
        });
        prog.push(Definition {
            term: e.term.clone(),
            meanings: vec![e.zh.clone()],
            synonyms: vec![e.code.clone()],
            origin: "语料",
            priority: 1,
        });
    }
    (zh, zhen, prog)
}

/// One dictionary lookup result.
#[derive(Clone, Debug)]
pub struct Definition {
    pub term: String,
    pub meanings: Vec<String>,
    pub synonyms: Vec<String>,
    /// Knowledge source of the answer, e.g. "标准库" / "blessed.rs" / "cheats.rs".
    pub origin: &'static str,
    /// Answer priority: curated ecosystem sources rank above the std library.
    pub priority: u32,
}

/// Any specialist dictionary: given a query, returns its best entry + score.
pub trait Dictionary: Send + Sync {
    /// Human-readable dictionary name, e.g. "英文字典".
    fn name(&self) -> &'static str;
    /// Best matching entry for the query, if it beats the dictionary threshold.
    fn best_match(&self, query: &str) -> Option<(Definition, f32)>;
}

/// A hit from one dictionary, unified by the generic router.
#[derive(Clone, Debug)]
pub struct RouteHit {
    pub source: &'static str,
    pub definition: Definition,
    pub score: f32,
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

/// Char + bigram overlap score of a query against one term, in 0..=1.
///
/// # Details
/// Uses bidirectional coverage: how much of the query the term covers AND
/// how much of the term the query contains. The term-coverage side makes a
/// query that merely *contains* a dictionary term (e.g. "泛型的作用" vs
/// "泛型") match strongly instead of being diluted by extra characters.
fn overlap_score(query: &str, term: &str) -> f32 {
    let (qc, qb) = query_sets(query);
    let (tc, tb) = query_sets(term);
    let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f32 / b as f32 };
    let cq = ratio(qc.iter().filter(|c| tc.contains(c)).count(), qc.len());
    let ct = ratio(tc.iter().filter(|c| qc.contains(c)).count(), tc.len());
    let bq = ratio(qb.iter().filter(|b| tb.contains(b)).count(), qb.len());
    let bt = ratio(tb.iter().filter(|b| qb.contains(b)).count(), tb.len());
    0.4 * (0.5 * cq + 0.5 * ct) + 0.6 * (0.5 * bq + 0.5 * bt)
}

fn best_from(
    entries: &[Definition],
    query: &str,
    threshold: f32,
) -> Option<(Definition, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, e) in entries.iter().enumerate() {
        let s = overlap_score(query, &e.term);
        if best.map_or(true, |(_, bs)| s > bs) {
            best = Some((i, s));
        }
    }
    match best {
        Some((i, s)) if s >= threshold => Some((entries[i].clone(), s)),
        _ => None,
    }
}

/// English dictionary: English word -> English definition + synonyms.
pub struct EnglishDict {
    entries: Vec<Definition>,
    threshold: f32,
}

impl EnglishDict {
    pub fn new() -> Self {
        Self {
            entries: vec![
                def("open", &["to move something so it is no longer closed", "to start or make something available"], &["unlock", "unclose", "start", "launch"]),
                def("close", &["to move something so it blocks an opening", "to end"], &["shut", "seal", "finish"]),
                def("delete", &["to remove something", "to erase"], &["remove", "erase", "wipe"]),
                def("remove", &["to take something away", "to get rid of"], &["delete", "erase", "discard"]),
                def("read", &["to look at and understand written words", "to take in data"], &["scan", "browse"]),
                def("write", &["to mark letters or words on a surface", "to store data"], &["record", "note"]),
                def("create", &["to make something new"], &["make", "build", "produce", "generate"]),
                def("make", &["to create something", "to cause to exist"], &["create", "build", "produce"]),
                def("build", &["to construct something"], &["construct", "assemble", "create"]),
                def("function", &["a block of reusable code", "a purpose or role"], &["routine", "method", "procedure"]),
                def("method", &["a way of doing something", "a function on a type"], &["function", "procedure", "way"]),
                def("random", &["happening without order or pattern"], &["arbitrary", "unpredictable", "chance"]),
                def("connect", &["to join things together", "to link"], &["link", "join", "attach"]),
                def("network", &["a system of connected things"], &["web", "grid"]),
                def("request", &["to ask for something"], &["ask", "demand", "call"]),
                def("database", &["an organized collection of data"], &["data store", "repository"]),
                def("print", &["to produce text on paper or screen"], &["output", "display"]),
                def("format", &["to arrange in a particular form"], &["arrange", "style"]),
                def("parse", &["to analyze a string into parts"], &["analyze", "interpret"]),
                def("sort", &["to arrange in order"], &["order", "arrange", "classify"]),
                def("file", &["a container for data on a computer"], &["document", "record"]),
                def("folder", &["a container for files"], &["directory", "directory"]),
                def("directory", &["a list of files or folders", "a folder"], &["folder", "index"]),
                def("path", &["a route or location of a file"], &["route", "location"]),
                def("search", &["to look for something"], &["find", "look", "seek"]),
                def("find", &["to discover something"], &["locate", "search"]),
                def("copy", &["to make a duplicate"], &["duplicate", "clone"]),
                def("move", &["to change position"], &["transfer", "shift"]),
                def("rename", &["to give a new name"], &["re-title"]),
                def("string", &["a sequence of characters"], &["text", "text"]),
                def("array", &["an ordered collection of elements"], &["list", "vector"]),
                def("list", &["a number of items in order"], &["array", "sequence"]),
                def("generic", &["relating to a whole class rather than one specific thing", "a type that can stand in for many types"], &["template", "parameterized"]),
            ],
            threshold: 0.35,
        }
    }
}

impl Default for EnglishDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Dictionary for EnglishDict {
    fn name(&self) -> &'static str {
        "英文字典"
    }
    fn best_match(&self, query: &str) -> Option<(Definition, f32)> {
        best_from(&self.entries, query, self.threshold)
    }
}

/// Chinese dictionary: Chinese word -> Chinese definition + synonyms.
pub struct ChineseDict {
    entries: Vec<Definition>,
    threshold: f32,
}

impl ChineseDict {
    pub fn new() -> Self {
        let (corpus_zh, _, _) = corpus_definitions();
        let mut entries = vec![
                def("打开", &["把关闭的东西开启", "展开；使开始"], &["开启", "翻开", "启动"]),
                def("关闭", &["使合拢；停止运行"], &["关上", "关闭", "结束"]),
                def("删除", &["去掉；抹去"], &["移除", "清除", "擦除"]),
                def("移除", &["拿开；去掉"], &["删除", "搬走"]),
                def("读取", &["阅读；把数据取入"], &["读入", "阅读"]),
                def("写入", &["把内容记下；存储数据"], &["记录", "存写"]),
                def("创建", &["建立新的事物"], &["建立", "新建", "制造"]),
                def("制作", &["制造；创作"], &["创建", "制造"]),
                def("构建", &["建造；组装"], &["构造", "建立"]),
                def("函数", &["一段可复用的代码块", "功能或作用"], &["子程序", "方法", "过程"]),
                def("方法", &["做事情的方式", "类型上的函数"], &["函数", "途径", "方式"]),
                def("随机", &["没有固定顺序或规律"], &["任意", "偶然"]),
                def("连接", &["把事物接在一起"], &["链接", "接通", "联结"]),
                def("网络", &["相互连接的系统"], &["网", "网际"]),
                def("请求", &["向对方提出要求"], &["要求", "索取", "调用"]),
                def("数据库", &["有组织的数据集合"], &["数据仓库", "数据存储"]),
                def("打印", &["把文字输出到纸面或屏幕"], &["输出", "显示"]),
                def("格式化", &["按特定形式整理"], &["整理", "排布"]),
                def("解析", &["把字符串拆解分析"], &["分析", "解读"]),
                def("排序", &["按顺序排列"], &["排列", "分类"]),
                def("文件", &["计算机中存放数据的容器"], &["文档", "档案"]),
                def("文件夹", &["存放文件的容器"], &["目录", "目录"]),
                def("目录", &["文件或文件夹的列表", "文件夹"], &["文件夹", "索引"]),
                def("路径", &["文件所在的位置或路线"], &["位置", "路线"]),
                def("搜索", &["寻找"], &["查找", "搜寻"]),
                def("查找", &["寻找目标"], &["搜索", "搜寻"]),
                def("复制", &["制作一份副本"], &["拷贝", "克隆"]),
                def("移动", &["改变位置"], &["转移", "挪动"]),
                def("重命名", &["给予新名字"], &["改名"]),
                def("字符串", &["一串字符"], &["文本", "串"]),
                def("数组", &["有序的元素集合"], &["列表", "向量"]),
                def("泛型", &["类型参数化，让同一份代码适用于多种类型"], &["类型参数", "模板"]),
                def("结构体", &["用 struct 定义的数据类型，可包含多个字段"], &["结构", "自定义类型"]),
                def("字符串转换", &["把字符串转成切片、数字或其他类型"], &["as_str", "parse", "转换"]),
                def("当前目录", &["进程当前所在的工作目录"], &["current_dir", "工作目录"]),
                def("异步", &["不阻塞等待结果，用 async/await 和 Future 的编程方式"], &["async", "future"]),
        ];
        entries.extend(corpus_zh);
        Self {
            entries,
            threshold: 0.35,
        }
    }
}

impl Default for ChineseDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Dictionary for ChineseDict {
    fn name(&self) -> &'static str {
        "中文字典"
    }
    fn best_match(&self, query: &str) -> Option<(Definition, f32)> {
        best_from(&self.entries, query, self.threshold)
    }
}

/// Chinese-to-English translation dictionary.
pub struct ZhEnDict {
    entries: Vec<Definition>,
    threshold: f32,
}

impl ZhEnDict {
    pub fn new() -> Self {
        let (_, corpus_zhen, _) = corpus_definitions();
        let mut entries = vec![
                def("打开", &["open", "to open"], &["open", "switch on"]),
                def("关闭", &["close", "to close"], &["close", "shut"]),
                def("删除", &["delete", "to delete"], &["delete", "remove"]),
                def("读取", &["read", "to read"], &["read"]),
                def("写入", &["write", "to write"], &["write", "store"]),
                def("创建", &["create", "to create"], &["create", "make"]),
                def("函数", &["function"], &["function", "routine"]),
                def("方法", &["method"], &["method", "function"]),
                def("随机", &["random"], &["random", "arbitrary"]),
                def("连接", &["connect", "to connect"], &["connect", "link"]),
                def("网络", &["network"], &["network", "web"]),
                def("请求", &["request"], &["request", "ask"]),
                def("数据库", &["database"], &["database"]),
                def("打印", &["print"], &["print", "output"]),
                def("解析", &["parse"], &["parse", "analyze"]),
                def("排序", &["sort"], &["sort", "order"]),
                def("文件", &["file"], &["file", "document"]),
                def("文件夹", &["folder", "directory"], &["folder", "directory"]),
                def("目录", &["directory"], &["directory", "folder"]),
                def("路径", &["path"], &["path"]),
                def("搜索", &["search"], &["search", "find"]),
                def("复制", &["copy"], &["copy", "duplicate"]),
                def("移动", &["move"], &["move", "transfer"]),
                def("字符串", &["string"], &["string", "text"]),
                def("数组", &["array"], &["array", "vector"]),
                def("泛型", &["generic", "generics"], &["generic", "parameterized"]),
                def("结构体", &["struct"], &["struct"]),
                def("异步", &["async"], &["async"]),
        ];
        entries.extend(corpus_zhen);
        Self {
            entries,
            threshold: 0.4,
        }
    }
}

impl Default for ZhEnDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Dictionary for ZhEnDict {
    fn name(&self) -> &'static str {
        "汉语翻译英文"
    }
    fn best_match(&self, query: &str) -> Option<(Definition, f32)> {
        best_from(&self.entries, query, self.threshold)
    }
}

/// Programming-terms dictionary: term -> canonical code / semantic answer.
pub struct ProgramDict {
    entries: Vec<Definition>,
    threshold: f32,
}

impl ProgramDict {
    pub fn new() -> Self {
        let (_, _, corpus_prog) = corpus_definitions();
        let mut entries = vec![
                def("打开文件", &["以只读模式打开文件"], &["use std::fs::File; let mut f = File::open(\"a.txt\")?;"]),
                def("删除文件", &["从文件系统删除文件"], &["std::fs::remove_file(\"a.txt\")?;"]),
                def("读取文件", &["把文件全部内容读成字符串"], &["let s = std::fs::read_to_string(\"a.txt\")?;"]),
                def("写入文件", &["把字节写入文件"], &["std::fs::write(\"a.txt\", \"hello\")?;"]),
                def("创建文件夹", &["创建新的空目录"], &["std::fs::create_dir(\"d\")?;"]),
                def("定义结构体", &["用 struct 定义结构体"], &["struct S { x: T }"]),
                def_high("blessed.rs", "生成随机数", &["使用 rand 库生成随机数"], &["cargo add rand"]),
                def_high("blessed.rs", "发送HTTP请求", &["使用 reqwest 发起 HTTP 请求"], &["cargo add reqwest"]),
                def_high("blessed.rs", "连接数据库", &["使用 sqlx 连接数据库"], &["cargo add sqlx"]),
                def_high("blessed.rs", "解析命令行参数", &["使用 clap 解析参数"], &["cargo add clap"]),
                def_high("cheats.rs", "定义函数", &["用 fn 定义函数"], &["fn f(x: i32) -> i32 { x + 1 }"]),
                def("当前运行目录", &["返回当前工作目录路径"], &["let d = std::env::current_dir()?;"]),
                def("字符串转换切片", &["把 String 转成 &str 切片"], &["let s: &str = s.as_str();"]),
                def("异步", &["用 async fn 定义异步函数"], &["async fn f() -> i32 { 1 }"]),
                def("cargo.toml", &["Cargo 清单文件，声明包元数据和依赖"], &["[package]\nname = \"demo\"\nversion = \"0.1.0\"\n[dependencies]\nserde = \"1\""]),
                def("格式化字符串", &["用 format! 插值生成字符串"], &["let s = format!(\"{}-{}\", 1, 2);"]),
                def("遍历数组", &["用 for 循环遍历迭代器"], &["for x in v.iter() { println!(\"{}\", x); }"]),
                def("排序数组", &["调用 sort 排序切片"], &["v.sort();"]),
                def("解析命令行参数", &["使用 clap 解析参数"], &["cargo add clap"]),
                def("泛型", &["类型参数化，一份代码适用多种类型"], &["fn identity<T>(x: T) -> T { x }"]),
        ];
        entries.extend(corpus_prog);
        Self {
            entries,
            threshold: 0.4,
        }
    }
}

impl Default for ProgramDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Dictionary for ProgramDict {
    fn name(&self) -> &'static str {
        "编程术语字典"
    }
    fn best_match(&self, query: &str) -> Option<(Definition, f32)> {
        best_from(&self.entries, query, self.threshold)
    }
}

fn def(term: &str, meanings: &[&str], synonyms: &[&str]) -> Definition {
    Definition {
        term: term.to_string(),
        meanings: meanings.iter().map(|s| s.to_string()).collect(),
        synonyms: synonyms.iter().map(|s| s.to_string()).collect(),
        origin: "标准库",
        priority: 1,
    }
}

/// Builds a high-value answer from a curated ecosystem source.
fn def_high(
    origin: &'static str,
    term: &str,
    meanings: &[&str],
    synonyms: &[&str],
) -> Definition {
    let mut d = def(term, meanings, synonyms);
    d.origin = origin;
    d.priority = 3;
    d
}

/// The generic routing expert: unifies all four dictionaries + intent router.
pub struct DictRouter {
    pub dicts: Vec<Box<dyn Dictionary>>,
    pub intents: SemanticMoE,
}

impl DictRouter {
    /// Creates the router with the four specialist dictionaries.
    pub fn new() -> Self {
        Self {
            dicts: vec![
                Box::new(EnglishDict::new()),
                Box::new(ChineseDict::new()),
                Box::new(ZhEnDict::new()),
                Box::new(ProgramDict::new()),
            ],
            intents: SemanticMoE::new(),
        }
    }

    /// Routes a query to the best dictionaries, sorted by score.
    ///
    /// # Details
    /// Generic across any number of `Dictionary` specialists: every dict
    /// scores the query, and all hits above the per-dict threshold come back
    /// ranked. The semantic intent router is consulted in addition.
    ///
    /// # Arguments
    /// * `query` - Term or paraphrase to look up
    ///
    /// # Returns
    /// * `Vec<RouteHit>` - Matches across sources, best first
    pub fn route(&self, query: &str) -> Vec<RouteHit> {
        let mut hits: Vec<RouteHit> = self
            .dicts
            .iter()
            .filter_map(|d| {
                d.best_match(query)
                    .map(|(definition, score)| RouteHit {
                        source: d.name(),
                        definition,
                        score,
                    })
            })
            .collect();
        // High-value curated answers (blessed.rs / cheats.rs / lib.rs) rank
        // above std-library answers; ties break by match score.
        hits.sort_by(|a, b| {
            b.definition
                .priority
                .cmp(&a.definition.priority)
                .then_with(|| b.score.partial_cmp(&a.score).unwrap())
        });
        hits
    }

    /// Routes and also reports the semantic intent, if any.
    pub fn route_with_intent(&self, query: &str) -> (Vec<RouteHit>, Option<(usize, f32)>) {
        let hits = self.route(query);
        let intent = self.intents.route(query);
        (hits, intent)
    }

    /// Whether a query asks for an *explanation* ("X 是什么") rather than code.
    ///
    /// # Details
    /// Explanation questions should answer with the keyword's meaning from the
    /// dictionaries, not with a code continuation the model happens to memorize.
    pub fn is_explain(query: &str) -> bool {
        ["是什么", "什么是", "解释", "说明", "含义", "介绍", "讲讲", "了解一下", "定义是什么"]
            .iter()
            .any(|m| query.contains(m))
    }

    /// Answers an explanation question with the keyword's dictionary meaning.
    ///
    /// # Details
    /// For "X 是什么" the standard-library / Chinese / translation dictionaries
    /// carry the definition; the programming-terms code answer is deprioritized.
    /// Returns `Some(definition)` when a meaning dictionary matches.
    ///
    /// # Arguments
    /// * `query` - e.g. "泛型是什么"
    ///
    /// # Returns
    /// * `Option<String>` - A natural definition sentence
    pub fn explain(&self, query: &str) -> Option<String> {
        let hits = self.route(query);
        // prefer meaning dictionaries over the code (编程术语) dictionary
        let h = hits.iter().find(|h| h.source != "编程术语字典")?;
        let mut s = format!("{}：{}", h.definition.term, h.definition.meanings.join("；"));
        if !h.definition.synonyms.is_empty() {
            s.push_str(&format!("（近义：{}）", h.definition.synonyms.join("、")));
        }
        Some(s)
    }
}

impl Default for DictRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests an English word routes to the English dictionary.
    #[test]
    fn test_english_routing() {
        let r = DictRouter::new();
        let hits = r.route("delete");
        assert!(!hits.is_empty(), "should get a hit");
        assert_eq!(hits[0].source, "英文字典");
        assert!(hits[0].definition.synonyms.contains(&"remove".to_string()));
    }

    /// Tests a Chinese word routes to the Chinese and translation dicts.
    #[test]
    fn test_chinese_routing() {
        let r = DictRouter::new();
        let hits = r.route("删除文件");
        let sources: Vec<_> = hits.iter().map(|h| h.source).collect();
        assert!(sources.contains(&"编程术语字典"), "file ops are programming terms");
        // at least one Chinese source should also hit
        assert!(hits.iter().any(|h| h.source == "中文字典" || h.source == "汉语翻译英文"));
    }

    /// Tests near-synonym queries hit the same intent.
    #[test]
    fn test_synonym_route_to_same_program_answer() {
        let r = DictRouter::new();
        for q in ["我想打开文件", "怎么打开文件", "请帮我打开文件"] {
            let a = r.intents.answer(q).expect("should route intent");
            assert_eq!(a, "use std::fs::File; let mut f = File::open(\"a.txt\")?;");
        }
    }

    /// Tests the router is generic: adding a dictionary is transparent.
    #[test]
    fn test_router_generic_over_dicts() {
        let r = DictRouter::new();
        let en_hits = r.route("read");
        let en = en_hits.iter().find(|h| h.source == "英文字典");
        assert!(en.is_some());
        let zh_hits = r.route("读取");
        let zh = zh_hits.iter().find(|h| h.source == "汉语翻译英文");
        assert!(zh.is_some());
    }
}
