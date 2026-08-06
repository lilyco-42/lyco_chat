//! jieba-rs frequency lexicon over the collected bilingual corpus.
//!
//! Reads corpus.json (Chinese single-char segmented + English/code), rebuilds
//! continuous Chinese from the CJK tokens, runs jieba-rs word segmentation,
//! and prints the most frequent words (Chinese words + English/code words).
//!
//! Usage: cargo run --release --bin jieba_freq -- [top_n]

use jieba_rs::Jieba;
use std::collections::HashMap;

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

fn main() {
    let corpus: Vec<String> = serde_json::from_str(include_str!("../../corpus.json")).unwrap();
    let jieba = Jieba::new();
    let mut freq: HashMap<String, usize> = HashMap::new();

    let mut zh_buf = String::new();
    let mut flush = |zh_buf: &mut String, freq: &mut HashMap<String, usize>| {
        if zh_buf.is_empty() {
            return;
        }
        for w in jieba.cut(zh_buf, false) {
            let w = w.to_string();
            if w.chars().count() >= 2 {
                *freq.entry(w).or_insert(0) += 1;
            }
        }
        zh_buf.clear();
    };

    for line in &corpus {
        for tok in line.split_whitespace() {
            if tok.chars().all(is_cjk) {
                // accumulate char-segmented Chinese into a continuous run
                zh_buf.push_str(tok);
                continue;
            }
            flush(&mut zh_buf, &mut freq);
            // English / code words: count maximal alphanumeric runs
            let mut cur = String::new();
            for c in tok.chars() {
                if c.is_alphanumeric() {
                    cur.push(c);
                } else if !cur.is_empty() {
                    if cur.chars().count() >= 2 {
                        *freq.entry(std::mem::take(&mut cur)).or_insert(0) += 1;
                    } else {
                        cur.clear();
                    }
                }
            }
            if cur.chars().count() >= 2 {
                *freq.entry(cur).or_insert(0) += 1;
            }
        }
        flush(&mut zh_buf, &mut freq);
    }

    let mut v: Vec<(String, usize)> = freq.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let top = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);

    println!("# 语料: {} 行, 词表大小: {} 词, 输出前 {top}", corpus.len(), v.len());
    for (w, n) in v.into_iter().take(top) {
        println!("{n}\t{w}");
    }
}
