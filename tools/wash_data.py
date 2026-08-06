#!/usr/bin/env python3
"""自动清洗 Rust 高质量语料.

源:
  1. rustwiki 中文书      https://rustwiki.org/zh-CN/book/print.html
  2. rust-by-example 中文  https://rustwiki.org/zh-CN/rust-by-example/print.html
  3. rustwiki 标准库      https://rustwiki.org/zh-CN/std/   (结构同 std-cn)
  4. lib.rs              https://lib.rs/
  5. blessed.rs/crates    https://blessed.rs/crates
  6. areweguiyet          https://areweguiyet.com/#ecosystem
  7. cheats.rs            https://cheats.rs/

清洗: 下载(带缓存) -> 提取正文 -> 去导航/样板/版权 -> 中文段与代码块分列 -> 去重 -> 输出.
输出: data/washed/<name>.zh.txt  (中文文本) 和 <name>.code.txt (代码块)
"""
import os
import re
import sys
import requests
from bs4 import BeautifulSoup

CACHE = "/tmp/opencode/wash"
OUT = "/workspace/data/washed"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# 样板/噪音行：翻译注、版权、导航、目录等
BOILER = [
    "翻译注", "中文翻译", "授权协议", "版权", "版权所有", "MIT License", "Apache",
    "https://", "http://", "rustwiki", "Contributing", "编辑本章", "目录",
    "下一章", "上一章", "Rust 中文", "简体中文", "Rust 社区",
]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}


def fetch(name, url):
    """下载并缓存到本地文件（原始字节，避免编码损坏）."""
    path = os.path.join(CACHE, name + ".html")
    if not os.path.exists(path):
        print(f"[fetch] {name} <- {url}")
        r = requests.get(url, timeout=120, headers=HEADERS)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    return open(path, "rb").read()


def is_boiler(line: str) -> bool:
    return any(b in line for b in BOILER) or len(line.strip()) < 4


def clean_rustwiki(name, url):
    """rustwiki 系：提取中文段落 + Rust 代码块，分列输出."""
    html = fetch(name, url)
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    main = soup.find("main") or soup.find("body")
    zh, code = [], []
    for p in main.find_all(["p", "li"]):
        t = p.get_text(" ", strip=True)
        if t and not is_boiler(t) and re.search(r"[\u4e00-\u9fff]", t):
            zh.append(t)
    for pre in main.find_all("pre"):
        t = pre.get_text(strip=True)
        if t and len(t) > 4:
            code.append(t)
    write(name, zh, code)


def clean_std(name, url):
    """标准库首页：抓各模块简介."""
    html = fetch(name, url)
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    main = soup.find("main") or soup.find("body")
    zh, code = [], []
    for p in main.find_all(["p", "li"]):
        t = p.get_text(" ", strip=True)
        if t and not is_boiler(t) and re.search(r"[\u4e00-\u9fff]", t):
            zh.append(t)
    write(name, zh, code)


def clean_librs(name, url):
    """lib.rs 被封；改用 crates.io API 抓 top 下载 crate 名 + 描述."""
    zh, code = [], []
    try:
        r = requests.get(
            "https://crates.io/api/v1/crates?page=1&per_page=100&sort=downloads",
            timeout=60, headers={"User-Agent": "wash_data/1.0 (data cleaning)"},
        )
        data = r.json()
        for c in data.get("crates", []):
            nm = c.get("name", "")
            desc = c.get("description", "")
            if nm:
                code.append(f"cargo add {nm}")
            if desc and 10 < len(desc) < 200:
                zh.append(desc)
    except Exception as e:
        print(f"[warn] librs: {e}")
    write(name, zh, code)


def clean_generic(name, url):
    """通用站点（可能纯英文）：提取正文文本，英文也保留."""
    html = fetch(name, url)
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    zh, code = [], []
    for p in soup.find_all(["p", "li", "h2", "h3"]):
        t = p.get_text(" ", strip=True)
        if t and not is_boiler(t) and len(t) >= 12:
            zh.append(t)
    for pre in soup.find_all("pre"):
        t = pre.get_text(strip=True)
        if t and len(t) > 4:
            code.append(t)
    write(name, zh, code)


def write(name, zh, code):
    zh = list(dict.fromkeys(zh))
    code = list(dict.fromkeys(code))
    with open(os.path.join(OUT, f"{name}.zh.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(zh))
    with open(os.path.join(OUT, f"{name}.code.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(code))
    tokens = sum(len(t.split()) for t in zh) + sum(len(t.split()) for t in code)
    print(f"[ok] {name}: 中文 {len(zh)} 条, 代码 {len(code)} 条, tokens {tokens}")


def clean_rustwiki_dict(name, url, out_path):
    """从 rustwiki 书/实例按『标题(概念) → 中文解释 → 代码』抽取字典条目."""
    html = fetch(name, url)
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    main = soup.find("main") or soup.find("body")
    entries = []
    term, text_buf, code_buf = None, [], []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "pre"]):
        tag = el.name
        if tag in ("h1", "h2", "h3", "h4"):
            # flush previous entry
            if term and text_buf:
                entries.append({
                    "term": term, "kind": "book",
                    "zh": " ".join(text_buf)[:400],
                    "en": "", "code": "\n".join(code_buf)[:400],
                    "synonyms": [],
                })
            term = el.get_text(" ", strip=True)[:30]
            text_buf, code_buf = [], []
        elif tag == "p":
            t = el.get_text(" ", strip=True)
            if t and not is_boiler(t) and re.search(r"[\u4e00-\u9fff]", t):
                text_buf.append(t)
        elif tag == "pre":
            t = el.get_text(strip=True)
            if t and len(t) > 4:
                code_buf.append(t)
    if term and text_buf:
        entries.append({
            "term": term, "kind": "book",
            "zh": " ".join(text_buf)[:400],
            "en": "", "code": "\n".join(code_buf)[:400],
            "synonyms": [],
        })
    import json as _json
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[ok] {name} 字典条目: {len(entries)}")


def main():
    clean_rustwiki("book", "https://rustwiki.org/zh-CN/book/print.html")
    clean_rustwiki("rust_by_example", "https://rustwiki.org/zh-CN/rust-by-example/print.html")
    clean_std("std_zh", "https://rustwiki.org/zh-CN/std/")
    clean_librs("librs", "https://lib.rs/")
    clean_generic("blessed", "https://blessed.rs/crates")
    clean_generic("areweguiyet", "https://areweguiyet.com/#ecosystem")
    clean_generic("cheats", "https://cheats.rs/")
    clean_rustwiki_dict("book_dict", "https://rustwiki.org/zh-CN/book/print.html",
                        "/workspace/data/washed/dict_book.json")
    clean_rustwiki_dict("rbe_dict", "https://rustwiki.org/zh-CN/rust-by-example/print.html",
                        "/workspace/data/washed/dict_rbe.json")


if __name__ == "__main__":
    main()
