#!/usr/bin/env python3
"""从 crates.io 抓取每个 crate 自带的 keywords/tags，存到 data/crate_tags.json."""
import json
import time
import requests

CATS = "/workspace/data/librs_categories.json"
OUT = "/workspace/data/crate_tags.json"
UA = {"User-Agent": "demo_chat-tags/1.0 (crate tag index)"}

def main():
    data = json.load(open(CATS))
    crates = []
    for c in data:
        for cr in c.get("crates", []):
            if cr not in crates:
                crates.append(cr)
    tags = {}
    for i, name in enumerate(crates):
        try:
            r = requests.get(f"https://crates.io/api/v1/crates/{name}", timeout=20, headers=UA)
            if r.status_code == 200:
                j = r.json()
                crate = j.get("crate", {})
                tags[name] = crate.get("keywords", [])
            else:
                tags[name] = []
        except Exception as e:
            print(f"[warn] {name}: {e}")
        if i % 10 == 0:
            print(f"  {i}/{len(crates)}: {name} -> {tags.get(name, [])[:3]}")
            time.sleep(0.3)
    json.dump(tags, open(OUT, "w"), ensure_ascii=False, indent=1)
    n = sum(1 for v in tags.values() if v)
    print(f"OK: {len(crates)} crates, {n} with tags -> {OUT}")

if __name__ == "__main__":
    main()
