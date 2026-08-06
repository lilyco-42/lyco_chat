//! Analyze a crate's public API by ast-grep: extract every function_item and
//! score each by internal reference weight, then print the hottest APIs.
//!
//! Usage: cargo run --release --bin codex -- <crate-src-dir> [top_n]

use ast_grep_core::matcher::KindMatcher;
use ast_grep_core::tree_sitter::StrDoc;
use ast_grep_core::AstGrep;
use std::collections::HashMap;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: codex <crate-src-dir> [top_n]");
    let top_n = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(12);

    let mut files: Vec<String> = vec![];
    let mut stack = vec![dir.clone()];
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

    let mut sigs: Vec<(String, String)> = vec![]; // (name, signature)
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap_or_default();
        let Ok(doc) = StrDoc::try_new(&src, ast_grep_language::Rust) else {
            continue;
        };
        let root = AstGrep::doc(doc);
        let km = KindMatcher::new("function_item", ast_grep_language::Rust);
        for m in root.root().find_all(&km) {
            let text = m.text().to_string();
            if !text.starts_with("pub ") {
                continue;
            }
            let name = text
                .split_whitespace()
                .skip_while(|w| *w != "fn")
                .nth(1)
                .unwrap_or_default()
                .split(['<', '(', ':', ' '])
                .next()
                .unwrap_or_default()
                .to_string();
            if name.is_empty() || name.starts_with('_') {
                continue;
            }
            let signature = text.split('{').next().unwrap_or(&text).trim().to_string();
            sigs.push((name, signature));
        }
    }

    let mut all_src = String::new();
    for f in &files {
        all_src.push_str(&std::fs::read_to_string(f).unwrap_or_default());
    }
    let mut weight: HashMap<String, usize> = HashMap::new();
    for (name, _) in &sigs {
        *weight.entry(name.clone()).or_insert(0) += all_src.matches(name.as_str()).count();
    }

    let mut seen: HashMap<String, String> = HashMap::new();
    let mut ranked: Vec<(String, usize, String)> = sigs
        .into_iter()
        .map(|(n, s)| (n.clone(), weight[&n], s))
        .filter(|(n, _, _)| seen.insert(n.clone(), String::new()).is_none())
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    println!("# {} 个 pub fn, 按内部引用权重排序 (top {top_n})", ranked.len());
    for (name, w, sig) in ranked.into_iter().take(top_n) {
        println!("{w:>5}  {name}");
        println!("        {sig}");
    }
}
