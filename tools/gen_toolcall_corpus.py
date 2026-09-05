#!/usr/bin/env python3
"""Generate toolcall_corpus.json — teach lyco_chat the standard AI tool-call
input/output formats, sourced from lilyco tool schemas.

Input : lilyco registry manifest JSON (any lilyco app: `app --schema`,
        multi-command: `multi --schema` → RegisteredCommand[]).
Output: toolcall_corpus.json  — Vec<String>, same "问 : … 答 : …" format as
        corpus.json (whitespace tokenized), lines teach:
          1) user intent → OpenAI tool_call JSON (spaced, parseable)
          2) tool result JSON → short final answer
        tools_openai.json — the OpenAI tools array (runtime system prompt).

Design constraints (lyco_chat tokenizer):
  - whitespace split → emitted JSON MUST be space-separated
  - values MUST NOT contain spaces (keeps tokens aligned)

Usage:
  python tools/gen_toolcall_corpus.py --schema manifest.json --per-command 24
  python tools/gen_toolcall_corpus.py --demo          # embedded demo schema
"""
import argparse
import json
import random

DEMO_MANIFEST = [
    {
        "name": "img-compress",
        "schema": {
            "name": "img-compress",
            "about": "Compress image files",
            "args": [
                {"name": "input", "about": "Input image file", "kind": {"type": "path", "must_exist": True}, "required": True, "default": None},
                {"name": "quality", "about": "Quality 1-100", "kind": {"type": "number", "min": 1.0, "max": 100.0}, "required": False, "default": 75},
                {"name": "format", "about": "Output format", "kind": {"type": "enum", "values": ["jpeg", "png", "webp"]}, "required": False, "default": "jpeg"},
                {"name": "dry_run", "about": "Dry run", "kind": {"type": "flag"}, "required": False, "default": None},
            ],
            "subcommands": [],
        },
        "aliases": ["imgc"],
        "hidden": False,
    },
    {
        "name": "ping",
        "schema": {
            "name": "ping",
            "about": "向某人问好",
            "args": [
                {"name": "name", "about": "问谁", "kind": {"type": "text"}, "required": False, "default": "world"},
            ],
            "subcommands": [],
        },
        "aliases": [],
        "hidden": False,
    },
]

ZH_TEMPLATES = [
    "帮我用 {tool} {args_phrase}",
    "{tool} {args_phrase}",
    "我要 {tool} {args_phrase}",
    "请执行 {tool} {args_phrase}",
]
EN_TEMPLATES = [
    "run {tool} with {args_phrase}",
    "{tool} {args_phrase}",
    "please call {tool} {args_phrase}",
]


def arg_sample(arg, rng):
    """Sample a space-free JSON value for one arg (respects schema constraints)."""
    kind = arg["kind"]
    t = kind.get("type")
    if t == "flag":
        return rng.choice([True, False])
    if t == "number":
        lo = kind.get("min") or 0.0
        hi = kind.get("max") or 100.0
        v = round(rng.uniform(lo, hi))
        return int(v)
    if t == "enum":
        return rng.choice(kind["values"])
    if t == "path":
        base = ["data/in", "assets/pic", "tmp/f", "out/o"]
        return rng.choice(base) + str(rng.randint(1, 9)) + rng.choice([".png", ".jpg", ".mp4"])
    # text / fallback
    if "default" in arg and isinstance(arg.get("default"), str):
        return rng.choice([arg["default"], arg["default"] + str(rng.randint(1, 99))])
    return rng.choice(["v1", "v2", "demo"])


def sample_args(args, rng):
    """Return (arguments dict, args phrase for the question side)."""
    out, phrase = {}, []
    for a in args:
        required = a.get("required")
        has_default = a.get("default") is not None
        if not required and not has_default and rng.random() < 0.5:
            continue  # optional-without-default: sometimes omitted
        if not required and has_default and rng.random() < 0.3:
            continue  # defaulted: usually omitted
        v = arg_sample(a, rng)
        out[a["name"]] = v
        phrase.append(f"{a['name']} {v}")
    return out, " ".join(phrase)


def seg(text):
    """Mirror lyco_chat src/main.rs seg_tokens: ASCII runs stay whole words,
    non-ASCII chars are split one per token (corpus is pre-segmented this way)."""
    out, buf = [], ""
    for ch in text:
        if ch.isspace():
            if buf:
                out.append(buf)
                buf = ""
        elif ch.isascii():
            buf += ch
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
    if buf:
        out.append(buf)
    return " ".join(out)


def spaced(obj):
    """Fully space-separated JSON so the whitespace tokenizer learns every token
    (braces/commas as independent tokens; json.loads still parses the result)."""
    if isinstance(obj, dict):
        items = [f'"{k}" : {spaced(v)}' for k, v in obj.items()]
        return "{ " + " , ".join(items) + " }"
    if isinstance(obj, list):
        return "[ " + " , ".join(spaced(v) for v in obj) + " ]"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if obj is None:
        return "null"
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def gen_lines(manifest, per_command, rng):
    lines = []
    for cmd in manifest:
        if cmd.get("hidden"):
            continue
        schema = cmd["schema"]
        name = schema["name"]
        args = schema.get("args", [])
        for _ in range(per_command):
            arguments, phrase = sample_args(args, rng)
            lang = rng.random()
            if lang < 0.6:
                q = rng.choice(ZH_TEMPLATES).format(tool=name, args_phrase=phrase or "默认参数")
            else:
                q = rng.choice(EN_TEMPLATES).format(tool=name, args_phrase=phrase or "defaults")
            call = {"tool_call": {"name": name, "arguments": arguments}}
            lines.append(f"问 : {seg(q)} 答 : {spaced(call)}")
            # 工具结果 → 最终答复（教模型消费 tool result）
            result = {"ok": True, "tool": name, "duration_ms": rng.randint(1, 999)}
            ans = f"{name} 完成，耗时 {result['duration_ms']}ms"
            lines.append(f"问 : 工具结果 {spaced(result)} 答 : {seg(ans)}")
    return lines


def tools_openai(manifest):
    """The OpenAI tools array (Chat Completions shape) for the runtime prompt."""
    tools = []
    for cmd in manifest:
        if cmd.get("hidden"):
            continue
        schema = cmd["schema"]
        params = {"type": "object", "properties": {}, "required": []}
        for a in schema.get("args", []):
            k = a["kind"]
            t = k.get("type")
            if t == "flag":
                prop = {"type": "boolean", "description": a["about"]}
            elif t == "number":
                prop = {"type": "number", "description": a["about"]}
                if k.get("min") is not None:
                    prop["minimum"] = k["min"]
                if k.get("max") is not None:
                    prop["maximum"] = k["max"]
            elif t == "enum":
                prop = {"type": "string", "enum": k["values"], "description": a["about"]}
            else:
                prop = {"type": "string", "description": a["about"]}
            params["properties"][a["name"]] = prop
            if a.get("required"):
                params["required"].append(a["name"])
        tools.append({"type": "function", "function": {"name": schema["name"], "description": schema["about"], "parameters": params}})
    return tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", help="lilyco manifest JSON (app --schema)")
    ap.add_argument("--demo", action="store_true", help="use embedded demo schema")
    ap.add_argument("--per-command", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="toolcall_corpus.json")
    ap.add_argument("--tools-out", default="tools_openai.json")
    a = ap.parse_args()

    if a.schema:
        with open(a.schema, encoding="utf-8") as f:
            manifest = json.load(f)
    elif a.demo:
        manifest = DEMO_MANIFEST
    else:
        ap.error("需要 --schema 或 --demo")

    rng = random.Random(a.seed)
    lines = gen_lines(manifest, a.per_command, rng)
    rng.shuffle(lines)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=1)
    with open(a.tools_out, "w", encoding="utf-8") as f:
        json.dump(tools_openai(manifest), f, ensure_ascii=False, indent=1)

    # 自检：每个 答 : 之后的 JSON 必须可解析（协议格式学习的前提）
    bad = 0
    for ln in lines:
        if " 答 : {" not in ln:
            continue
        q, payload = ln.split(" 答 : ", 1)
        q_body = q[3:] if q.startswith("问 : ") else q
        if '"' in q_body:
            bad += 1  # 问侧引号会破坏 seg_tokens/词表一致性
        try:
            json.loads(payload)
        except json.JSONDecodeError:
            bad += 1
    print(f"[ok] {len(lines)} 行语料 → {a.out}（自检失败 {bad}）; tools → {a.tools_out}")


if __name__ == "__main__":
    main()
