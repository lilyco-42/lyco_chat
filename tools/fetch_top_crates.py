#!/usr/bin/env python3
"""Store the most-common Rust ecosystem crates + their keywords.

Merges:
  1. top crates.io downloads (name, description, downloads)
  2. existing data/crate_tags.json tags (hand-curated)
  3. per-crate detail API keywords/categories for the rest
Output: data/crates_top.json  name -> {desc, tags, categories, downloads}
"""
import json
import os
import time
import urllib.error
import urllib.request

UA = "rust-learning-demo (crate-keyword-miner)"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "crates_top.json")
TAGS = os.path.join(ROOT, "data", "crate_tags.json")
CACHE = "/tmp/opencode/crates_top_cache.json"
DETAIL = "/tmp/opencode/crates_top_detail.json"


def get(url: str, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=20))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def main():
    if os.path.exists(CACHE):
        data = json.load(open(CACHE))
        crates = data.get("crates", [])
        if len(crates) >= 100:
            data2 = get("https://crates.io/api/v1/crates?page=2&per_page=100&sort=downloads")
            crates += data2.get("crates", [])
    else:
        data = get("https://crates.io/api/v1/crates?page=1&per_page=100&sort=downloads")
        crates = data.get("crates", [])
        time.sleep(1)
        data2 = get("https://crates.io/api/v1/crates?page=2&per_page=100&sort=downloads")
        crates += data2.get("crates", [])
        with open(CACHE, "w") as f:
            json.dump({"crates": crates}, f, ensure_ascii=False)

    existing = {}
    if os.path.exists(TAGS):
        existing = json.load(open(TAGS))
    detail = {}
    if os.path.exists(DETAIL):
        detail = json.load(open(DETAIL))

    merged = {}
    for c in crates:
        name = c["name"]
        tags = existing.get(name) or []
        cats = []
        if not tags and name not in detail:
            try:
                d = get(f"https://crates.io/api/v1/crates/{name}?include=keywords,categories")
                detail[name] = {
                    "tags": [k["keyword"] for k in (d.get("keywords") or [])],
                    "categories": [c2["slug"] for c2 in (d.get("categories") or [])],
                }
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
                detail[name] = {"tags": [], "categories": []}
            time.sleep(0.15)
        if name in detail:
            tags = tags or detail[name]["tags"]
            cats = detail[name]["categories"]
        merged[name] = {
            "desc": c.get("description") or "",
            "tags": tags,
            "categories": cats,
            "downloads": c.get("downloads", 0),
            "version": c.get("max_version", ""),
        }

    with open(DETAIL, "w") as f:
        json.dump(detail, f, ensure_ascii=False)
    with open(OUT, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    tagged = sum(1 for v in merged.values() if v["tags"])
    print(f"saved {len(merged)} crates -> {OUT}  ({tagged} with tags)")
    for name in ["axum", "tokio", "serde", "reqwest", "clap"]:
        if name in merged:
            print(name, merged[name]["tags"], "|", merged[name]["desc"][:45])
    if "axum" not in merged:
        print("note: axum not in top downloads; it lives in data/crate_tags.json")


if __name__ == "__main__":
    main()
