#!/usr/bin/env python3
"""Generate corpus.json from English std doc sentences + Chinese std-cn translations.

Sources:
  - English: real std library doc sentences (curated from local rust-src).
  - Chinese: Chinese translations extracted from rust-lang-cn/std-cn HTML docs.
  - Code: short valid Rust examples, reused in `问 : ... 答 : ...` QA lines.
"""
import json
import os
import re
import sys

STD_CN = "/tmp/opencode/std-cn/doc/std"


def load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def resolve(path):
    """Follow rustdoc HTML redirects."""
    h = load(path)
    m = re.search(r'content="0;URL=([^"]+)"', h)
    if m:
        return os.path.normpath(os.path.join(os.path.dirname(path), m.group(1)))
    return path


def zh_summary(path, anchor=None):
    """First sentence of the Chinese docblock for a page or a method anchor."""
    path = resolve(path)
    h = load(path)
    if anchor:
        i = -1
        for pat in (f'id="method.{anchor}"', f'id="tymethod.{anchor}"'):
            i = h.find(pat)
            if i >= 0:
                break
        if i < 0:
            return None
        h = h[i:]
    m = re.search(r'<div class="docblock">(.*?)</div>', h, re.S)
    if not m:
        return None
    txt = re.sub(r"<[^>]+>", " ", m.group(1))
    txt = re.sub(r"\s+", " ", txt).strip()
    return re.split(r"(?<=。)", txt)[0]


def seg_zh(s):
    """Split Chinese into character-level tokens; keep ASCII runs intact."""
    out, buf = [], ""
    for ch in s:
        if ch in " \t\r\n":
            continue
        if ch.isascii() and ch != "。" and ch != "，":
            buf += ch
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
    if buf:
        out.append(buf)
    return out


def seg_en(s):
    return s.split()


def dict_entry(qas, en, zh, code, kind, term=None):
    """Builds one dictionary entry from a corpus item."""
    clean = lambda s: "".join(ch for ch in s if not ch.isspace())
    zhc = clean(zh)
    if term is None:
        # prefer the first paraphrase question, else a keyword from the zh doc
        term = clean(qas[0]) if qas else zhc[:4]
    return {
        "term": term,
        "kind": kind,
        "en": " ".join(seg_en(en)),
        "zh": " ".join(seg_zh(zh)),
        "code": code,
        "synonyms": [clean(q) for q in qas],
    }


# (html_page, method_anchor or None, english_doc, code, [chinese_questions])
ENTRIES = [    ("fs/struct.File.html", None,
     "A reference to an open file on the filesystem.",
     "use std::fs::File;", []),
    ("fs/struct.File.html", "open",
     "Attempts to open a file in read-only mode.",
     'use std::fs::File; let mut f = File::open("a.txt")?;',
     ["我需要打开文件", "我需要打开一个文件", "如何打开文件", "请给我打开文件的代码", "我想打开文件"]),
    ("fs/struct.File.html", "create",
     "Opens a file in write-only mode.",
     'let mut f = File::create("a.txt")?;',
     ["我需要创建文件", "如何创建新文件", "给我创建文件的代码"]),
    ("fs/fn.read_to_string.html", None,
     "Reads the entire contents of a file into a string.",
     'let s = std::fs::read_to_string("a.txt")?;',
     ["我需要读取文件内容", "如何读取整个文件", "读取文件内容的代码"]),
    ("fs/fn.read.html", None,
     "Reads the entire contents of a file into a bytes vector.",
     'let b = std::fs::read("a.png")?;', []),
    ("fs/fn.write.html", None,
     "Writes a slice of bytes as the entire contents of a file.",
     'std::fs::write("a.txt", "hello")?;',
     ["我需要写入文件", "如何向文件写入内容", "写入文件的代码"]),
    ("fs/fn.remove_file.html", None,
     "Removes a file from the filesystem.",
     'std::fs::remove_file("a.txt")?;',
     ["我需要删除文件", "如何删除一个文件"]),
    ("fs/fn.metadata.html", None,
     "Given a path, query the file system to get information about a file, directory, etc.",
     'let m = std::fs::metadata("a.txt")?;', []),
    ("fs/fn.rename.html", None,
     "Rename a file or directory.",
     'std::fs::rename("a.txt", "b.txt")?;', []),
    ("fs/fn.copy.html", None,
     "Copies the contents of one file to another.",
     'std::fs::copy("a.txt", "b.txt")?;', []),
    ("fs/fn.create_dir.html", None,
     "Creates a new, empty directory at the provided path.",
     'std::fs::create_dir("d")?;',
     ["我需要创建文件夹", "如何创建目录"]),
    ("fs/fn.create_dir_all.html", None,
     "Recursively create a directory and all of its parent components.",
     'std::fs::create_dir_all("a/b/c")?;', []),
    ("fs/fn.remove_dir.html", None,
     "Removes an existing, empty directory.",
     'std::fs::remove_dir("d")?;', []),
    ("fs/fn.try_exists.html", None,
     "Returns Ok(true) if the path points at an existing entity.",
     'std::fs::try_exists("a.txt")?;',
     ["如何判断文件是否存在"]),
    ("fs/struct.OpenOptions.html", "new",
     "Options and flags which can be used to configure how a file is opened.",
     'let f = std::fs::OpenOptions::new().append(true).open("log.txt")?;',
     ["如何追加内容到文件"]),
    ("io/struct.Stdin.html", None,
     "A handle to the standard input stream of a process.",
     'let s = std::io::stdin();', []),
    ("io/struct.Stdout.html", None,
     "A handle to the global standard output stream of the current process.",
     'let s = std::io::stdout();', []),
    ("io/trait.Read.html", "read_to_string",
     "Read all bytes until EOF in this source, placing them into buf.",
     'let mut s = String::new(); f.read_to_string(&mut s)?;', []),
    ("io/trait.Write.html", "write_all",
     "Attempts to write an entire buffer into this writer.",
     'f.write_all(b"hello")?;', []),
    ("io/trait.BufRead.html", "lines",
     "Returns an iterator over the lines of this reader.",
     'for l in std::io::BufReader::new(f).lines() { println!("{}", l?); }',
     ["如何按行读取文件"]),
    ("io/struct.BufReader.html", "new",
     "Creates a new BufReader with a default buffer capacity.",
     'let r = std::io::BufReader::new(file);', []),
    ("io/struct.BufWriter.html", "new",
     "Creates a new BufWriter with a default buffer capacity.",
     'let w = std::io::BufWriter::new(file);', []),
    ("string/struct.String.html", "new",
     "Creates a new empty String.",
     "let s = String::new();", []),
    ("string/struct.String.html", "from",
     "Constructs a new String from a string literal.",
     'let s = String::from("hi");', []),
    ("string/struct.String.html", "push",
     "Appends the given char to the end of this String.",
     "let mut s = String::from(\"a\"); s.push('b');", []),
    ("string/struct.String.html", "push_str",
     "Appends a given string slice onto the end of this String.",
     's.push_str(" world");',
     ["如何拼接字符串"]),
    ("string/struct.String.html", "len",
     "Returns the length of this String, in bytes.",
     "let n = s.len();", []),
    ("string/struct.String.html", "is_empty",
     "Returns true if this String has a length of zero, and false otherwise.",
     "s.is_empty();", []),
    ("string/struct.String.html", "capacity",
     "Returns this String's capacity, in bytes.",
     "let c = s.capacity();", []),
    ("string/struct.String.html", "clear",
     "Truncates this String, removing all contents.",
     "s.clear();", []),
    ("string/struct.String.html", "contains",
     "Returns true if this String contains a pattern.",
     's.contains("hi");', []),
    ("string/struct.String.html", "replace",
     "Replaces all matches of a pattern with another string.",
     's.replace("a", "b");', []),
    ("string/struct.String.html", "as_str",
     "Extracts a string slice containing the entire String.",
     "let s: &str = s.as_str();", []),
    ("primitive.str.html", "split_whitespace",
     "Splits a string slice by whitespace.",
     "for w in s.split_whitespace() { println!(\"{}\", w); }", []),
    ("string/trait.ToString.html", "to_string",
     "Converts the given value to a String.",
     "let s = 42.to_string();", []),
    ("primitive.str.html", "parse",
     "Parses this string slice into another type.",
     "let n: i32 = s.parse()?;", []),
    ("vec/struct.Vec.html", "new",
     "Constructs a new, empty Vec.",
     "let v: Vec<i32> = Vec::new();", []),
    ("vec/struct.Vec.html", "with_capacity",
     "Constructs a new, empty Vec with the specified capacity.",
     "let v: Vec<i32> = Vec::with_capacity(10);", []),
    ("vec/struct.Vec.html", "push",
     "Appends an element to the back of a collection.",
     "v.push(1);",
     ["如何向数组添加元素", "如何添加元素到向量"]),
    ("vec/struct.Vec.html", "pop",
     "Removes the last element from a vector and returns it.",
     "let x = v.pop();", []),
    ("vec/struct.Vec.html", "len",
     "Returns the number of elements in the vector, also referred to as its length.",
     "let n = v.len();", []),
    ("vec/struct.Vec.html", "is_empty",
     "Returns true if the vector contains no elements.",
     "v.is_empty();", []),
    ("vec/struct.Vec.html", "clear",
     "Clears the vector, removing all values.",
     "v.clear();", []),
    ("vec/struct.Vec.html", "get",
     "Returns a reference to an element or subslice depending on the type of index.",
     "if let Some(x) = v.get(0) { println!(\"{}\", x); }", []),
    ("vec/struct.Vec.html", "insert",
     "Inserts an element at position index within the vector, shifting all elements after it to the right.",
     "v.insert(0, 9);", []),
    ("vec/struct.Vec.html", "remove",
     "Removes and returns the element at position index within the vector.",
     "let x = v.remove(0);", []),
    ("vec/struct.Vec.html", "contains",
     "Returns true if the slice contains an element with the given value.",
     "v.contains(&2);",
     ["如何判断数组是否包含某个值"]),
    ("vec/struct.Vec.html", "sort",
     "Sorts the slice, but might not preserve the order of equal elements.",
     "v.sort();",
     ["如何排序数组"]),
    ("vec/struct.Vec.html", "iter",
     "Returns an iterator over the slice.",
     "for x in v.iter() { println!(\"{}\", x); }",
     ["如何遍历数组"]),
    ("collections/hash/map/struct.HashMap.html", "new",
     "Creates an empty HashMap.",
     "let mut m = std::collections::HashMap::new();", []),
    ("collections/hash/map/struct.HashMap.html", "insert",
     "Inserts a key-value pair into the map.",
     'm.insert("k", 1);',
     ["如何存储键值对", "如何插入数据到字典"]),
    ("collections/hash/map/struct.HashMap.html", "get",
     "Returns a reference to the value corresponding to the key.",
     'if let Some(v) = m.get("k") { println!("{}", v); }', []),
    ("collections/hash/map/struct.HashMap.html", "remove",
     "Removes a key from the map, returning the value at the key if the key was previously in the map.",
     'm.remove("k");', []),
    ("collections/hash/map/struct.HashMap.html", "contains_key",
     "Returns true if the map contains a value for the specified key.",
     'm.contains_key("k");', []),
    ("collections/hash/map/struct.HashMap.html", "len",
     "Returns the number of elements in the map.",
     "let n = m.len();", []),
    ("macro.println.html", None,
     "Prints to the standard output, with a newline.",
     'println!("hello");',
     ["如何打印一行", "打印到控制台的代码"]),
    ("macro.print.html", None,
     "Prints to the standard output.",
     'print!("hi");', []),
    ("macro.eprintln.html", None,
     "Prints to the standard error, with a newline.",
     'eprintln!("error!");', []),
    ("macro.format.html", None,
     "Creates a String using interpolation of runtime expressions.",
     'let s = format!("{}+{}={}", 1, 2, 3);',
     ["如何格式化字符串"]),
    ("env/fn.var.html", None,
     "Fetches the environment variable key from the current process.",
     'let v = std::env::var("PATH")?;',
     ["如何读取环境变量"]),
    ("env/fn.args.html", None,
     "Returns the arguments which this program was started with.",
     "for a in std::env::args() { println!(\"{}\", a); }", []),
    ("env/fn.set_var.html", None,
     "Sets the environment variable key to the value for the currently running process.",
     'std::env::set_var("K", "v");', []),
    ("env/fn.current_dir.html", None,
     "Returns the full filesystem path of the current working directory.",
     "let d = std::env::current_dir()?;",
     ["如何获取当前目录"]),
    ("path/struct.Path.html", "new",
     "Directly wraps a string slice as a Path slice.",
     'let p = std::path::Path::new("a.txt");', []),
    ("path/struct.Path.html", "join",
     "Creates an owned PathBuf with path adjoined to self.",
     'let p = p.join("b.txt");', []),
    ("path/struct.Path.html", "exists",
     "Returns true if the path points at an existing entity.",
     "if p.exists() { println!(\"yes\"); }", []),
    ("path/struct.Path.html", "is_file",
     "Returns true if the path points at an existing file.",
     "if p.is_file() { println!(\"file\"); }", []),
    ("path/struct.Path.html", "is_dir",
     "Returns true if the path points at an existing directory.",
     "if p.is_dir() { println!(\"dir\"); }", []),
    ("process/struct.Command.html", "new",
     "Constructs a new Command for launching the program at path.",
     'let mut c = std::process::Command::new("ls");',
     ["如何运行外部命令"]),
    ("process/struct.Command.html", "status",
     "Executes the command as a child process, waiting for it to finish and collecting its status.",
     "c.status()?;", []),
    ("process/struct.Command.html", "output",
     "Executes the command as a child process, waiting for it to finish and collecting all of its output.",
     "let o = c.output()?;", []),
    ("option/enum.Option.html", None,
     "The Option type, representing either the absence of a value or a value.",
     "let x: Option<i32> = Some(42);", []),
    ("option/enum.Option.html", "unwrap",
     "Returns the contained Some value, consuming the self value.",
     "let x = opt.unwrap();", []),
    ("option/enum.Option.html", "expect",
     "Returns the contained Some value, consuming the self value.",
     'let x = opt.expect("should have a value");', []),
    ("macro.panic.html", None,
     "Panics the current thread.",
     'panic!("boom");', []),
    ("primitive.i32.html", None,
     "The 32-bit signed integer type.",
     "let x: i32 = 42;", []),
    ("primitive.f64.html", None,
     "A 64-bit floating point type.",
     "let x: f64 = 3.14;", []),
    ("primitive.bool.html", None,
     "The boolean type.",
     "let flag: bool = true;", []),
]


# (crate_name, english_description, chinese_translation, [chinese_questions])
# Curated from https://blessed.rs/crates, https://areweguiyet.com/#ecosystem,
# https://www.arewewebyet.org/topics/. Code is `cargo add <crate>`.
ECOSYSTEM_ENTRIES = [
    ("rand", "De facto standard random number generation library split out from the standard library",
     "从标准库中拆分出来的事实上的标准随机数生成库",
     ["如何生成随机数", "随机数用什么库"]),
    ("regex", "De facto standard regex library. Very fast.",
     "事实上的标准正则表达式库，速度很快。",
     ["如何使用正则表达式", "正则表达式用什么库"]),
    ("uuid", "Implements generating and parsing UUIDs and a number of utility functions",
     "实现 UUID 的生成、解析以及大量工具函数",
     ["如何生成 UUID"]),
    ("tempfile", "Supports both temporary files and temporary directories",
     "同时支持临时文件和临时目录",
     ["如何创建临时文件"]),
    ("flate2", "Uses a pure-Rust implementation by default",
     "默认使用纯 Rust 实现",
     ["如何压缩和解压数据"]),
    ("indexmap", "A HashMap that separately keeps track of insertion order",
     "一个能单独记住插入顺序的 HashMap",
     []),
    ("reqwest", "Full-fat HTTP client. Can be used in both synchronous and asynchronous code",
     "功能完整的 HTTP 客户端，可在同步和异步代码中使用",
     ["如何发送 HTTP 请求", "HTTP 客户端用什么库"]),
    ("ureq", "Minimal synchronous HTTP client focused on simplicity",
     "极简的同步 HTTP 客户端，注重简单",
     ["如何发送简单的 HTTP 请求"]),
    ("anyhow", "Provides a boxed error type that can hold any error",
     "提供可容纳任意错误的装箱错误类型",
     ["如何处理应用错误"]),
    ("thiserror", "Helps with generating boilerplate for enum-style error types",
     "帮助为枚举风格错误类型生成样板代码",
     ["如何定义错误类型"]),
    ("tracing", "The go-to crate for logging",
     "日志记录的首选库",
     ["如何记录日志"]),
    ("serde", "De facto standard serialization library",
     "事实上的标准序列化库",
     ["如何序列化和反序列化数据"]),
    ("serde_json", "A fast and heavily optimized JSON encoder and decoder",
     "一个快速且经过高度优化的 JSON 编解码器",
     ["如何解析 JSON", "如何处理 JSON 数据"]),
    ("toml", "Serde compatible encoder for the TOML configuration language",
     "与 Serde 兼容的 TOML 配置语言编解码器",
     ["如何解析配置文件"]),
    ("itertools", "A bunch of useful methods on iterators that aren't in the stdlib",
     "标准库里没有的一堆有用的迭代器方法",
     ["如何更方便地处理迭代器"]),
    ("bitflags", "Strongly typed bitflag types",
     "强类型的位标志类型",
     ["如何定义位标志"]),
    ("libc", "Bindings for directly calling libc functions",
     "直接调用 libc 函数的绑定",
     []),
    ("num-bigint", "Big integers. It's not the fastest, but it's part of the trusted num library",
     "大整数支持，虽然不是最快的，但是可信的 num 库的一部分",
     ["如何处理大整数"]),
    ("nalgebra", "General-purpose linear algebra library with statically or dynamically sized matrices",
     "通用线性代数库，支持静态或动态尺寸的矩阵",
     ["如何做矩阵运算"]),
    ("ndarray", "Supports arbitrarily dimensioned arrays",
     "支持任意维度数组",
     ["如何做数组运算"]),
    ("pyo3", "Supports both calling python code from Rust and exposing Rust code to Python",
     "既支持从 Rust 调用 Python 代码，也支持向 Python 暴露 Rust 代码",
     ["如何在 Rust 中调用 Python"]),
    ("clap", "Ergonomic, battle-tested command line argument parser",
     "符合人体工学的、久经考验的命令行参数解析器",
     ["如何解析命令行参数"]),
    ("walkdir", "Basic recursive filesystem walking",
     "基础的递归文件系统遍历",
     ["如何遍历目录"]),
    ("notify", "Watch files or directories and execute a function when they change",
     "监视文件或目录，并在其变化时执行函数",
     ["如何监视文件变化"]),
    ("ratatui", "A high-level TUI library with widgets, layout, etc",
     "一个带控件、布局等的高级终端 UI 库",
     ["如何构建终端界面"]),
    ("rayon", "Convert sequential computation into parallel computation with one call",
     "一次调用即可把串行计算转换为并行计算",
     ["如何并行计算"]),
    ("tokio", "The most widely supported async runtime in the Rust ecosystem",
     "Rust 生态中最广泛支持的异步运行时",
     ["如何使用异步运行时", "如何编写异步代码"]),
    ("axum", "A minimal and ergonomic HTTP server framework",
     "一个极简且符合人体工学的 HTTP 服务框架",
     ["如何写一个 HTTP 服务", "HTTP 服务器用什么库"]),
    ("actix-web", "A performance focused HTTP framework",
     "一个以性能为导向的 HTTP 框架",
     ["如何构建高性能 HTTP 服务"]),
    ("hyper", "A low-level HTTP implementation, both client and server",
     "一个底层 HTTP 实现，既是客户端也是服务端",
     []),
    ("tonic", "gRPC over HTTP/2 with full support for asynchronous code",
     "基于 HTTP/2 的 gRPC，完整支持异步代码",
     ["如何使用 gRPC"]),
    ("sqlx", "Works with Postgres, MySQL, SQLite, and MS SQL with compile time checked queries",
     "支持 Postgres、MySQL、SQLite 和 MS SQL，并支持编译期检查查询",
     ["如何连接数据库"]),
    ("rusqlite", "Provides a sync API to SQLite",
     "为 SQLite 提供同步 API",
     ["如何使用 SQLite"]),
    ("redis", "Redis client library",
     "Redis 客户端库",
     ["如何连接 Redis"]),
    ("mongodb", "MongoDB driver",
     "MongoDB 驱动",
     ["如何连接 MongoDB"]),
    ("bevy", "An ECS based game engine, good for 3D but also capable of 2D",
     "基于 ECS 的游戏引擎，擅长 3D 也能做 2D",
     ["如何做游戏开发"]),
    ("glam", "Fast math library optimised for game development",
     "为游戏开发优化的快速数学库",
     ["如何做 3D 数学"]),
    ("winit", "The de facto standard window creation library",
     "事实上的标准窗口创建库",
     ["如何创建窗口"]),
    ("egui", "An easy-to-use immediate mode GUI that runs on both web and native",
     "一个易用的即时模式 GUI，可在 Web 和原生平台运行",
     ["如何写图形界面", "GUI 用什么库"]),
    ("iced", "A renderer-agnostic GUI library focused on simplicity and type-safety",
     "一个渲染器无关的 GUI 库，注重简单和类型安全",
     ["如何构建桌面界面"]),
    ("slint", "A toolkit to efficiently develop fluid graphical user interfaces",
     "一个高效开发流畅图形用户界面的工具包",
     []),
    ("tauri", "A framework for building tiny, blazing fast desktop binaries",
     "一个构建小巧且极速桌面程序的框架",
     ["如何构建桌面应用"]),
    ("leptos", "A full-stack, isomorphic Rust web framework",
     "一个全栈、同构的 Rust Web 框架",
     ["如何写前端界面"]),
    ("yew", "A framework for creating reliable and efficient web applications",
     "一个创建可靠高效 Web 应用的框架",
     []),
    ("dioxus", "Elegant React-like library for building user interfaces",
     "一个优雅的、类似 React 的界面构建库",
     []),
    ("gtk4", "Rust bindings of the GTK 4 library",
     "GTK 4 库的 Rust 绑定",
     []),
    ("fltk", "A cross-platform lightweight GUI library",
     "一个跨平台的轻量级 GUI 库",
     []),
    ("gpui", "A hybrid immediate and retained mode, GPU accelerated, UI framework",
     "一个混合即时与保留模式、GPU 加速的界面框架",
     []),
]

# (example_code, english_explanation, chinese_translation, [chinese_questions])
# Language constructs from https://cheats.rs/.
LANG_ENTRIES = [
    ("struct S { x: T }", "Define a struct with named fields.",
     "定义一个带命名字段的结构体",
     ["如何定义一个结构体", "我想定义一个结构体"]),
    ("enum E { A, B }", "Define an enum, an algebraic data type.",
     "定义一个枚举，一种代数数据类型",
     ["如何定义一个枚举"]),
    ("fn f(x: i32) -> i32 { x + 1 }", "Definition of a function.",
     "定义一个函数",
     ["如何定义一个函数", "我想定义一个函数"]),
    ("trait T { fn m(&self); }", "Define a trait; common behavior types can adhere to.",
     "定义一个特质，类型可以遵守的共同行为",
     ["如何定义一个 trait"]),
    ("impl T for S {}", "Implement trait T for type S.",
     "为类型 S 实现特质 T",
     ["如何实现一个 trait"]),
    ("let x: i32 = 5;", "Allocate a value bound as x, assignable once, not mutable.",
     "声明一个绑定为 x 的值，只能赋值一次，不可变",
     ["如何声明一个变量"]),
    ("let mut x = 5;", "Like let, but allow for mutability.",
     "和 let 类似，但允许可变",
     ["如何声明可变变量"]),
    ("|x| x + 1", "A closure that borrows its captures.",
     "一个借用其捕获变量的闭包",
     ["如何定义闭包"]),
    ("loop { break; }", "Loop indefinitely until break.",
     "无限循环直到 break",
     ["如何写无限循环"]),
    ("while x < 10 {}", "Loop while expression x is true.",
     "当表达式 x 为真时循环",
     ["如何写 while 循环"]),
    ("for x in collection {}", "Syntactic sugar to loop over iterators.",
     "遍历迭代器的语法糖",
     ["如何遍历集合"]),
    ("if x {} else {}", "Conditional branch if expression is true.",
     "当表达式为真时的条件分支",
     ["如何写条件判断"]),
    ("match x { Some(y) => y, None => 0 }", "Pattern match on a value.",
     "对值进行模式匹配",
     ["如何使用 match 匹配"]),
    ("&s", "Shared borrow of a value.",
     "对值的共享借用",
     ["如何借用值"]),
    ("&mut s", "Exclusive borrow that allows mutability.",
     "允许修改的独占借用",
     ["如何可变借用"]),
    ("let v = vec![1, 2, 3];", "Create a Vec from a list of elements.",
     "从元素列表创建向量",
     ["如何创建数组"]),
    ("panic!(\"boom\");", "Panics the current thread.",
     "让当前线程恐慌",
     ["如何触发 panic"]),
    ("let n: i32 = s.parse()?;", "Parse a string slice into another type, propagating errors.",
     "把字符串解析为其他类型，并传播错误",
     ["如何解析字符串"]),
    ("let s = format!(\"{}-{}\", 1, 2);", "Creates a String using interpolation of expressions.",
     "使用表达式插值创建字符串",
     ["如何格式化字符串"]),
    ("async fn f() -> i32 { 1 }", "Async function that returns a Future.",
     "返回 Future 的异步函数",
     ["如何写异步函数"]),
]


def main():
    en_lines, zh_lines, code_lines = [], [], []
    qa_std, qa_high = [], []
    dict_entries = []
    missing = []
    for page, anchor, en, code, qas in ENTRIES:
        path = os.path.join(STD_CN, page)
        if not os.path.exists(path):
            missing.append((page, anchor, "page"))
            continue
        zh = zh_summary(path, anchor)
        if zh is None:
            missing.append((page, anchor, "docblock"))
            continue
        # English doc line
        en_lines.append(" ".join(seg_en(en)))
        # Chinese doc line
        zh_lines.append(" ".join(seg_zh(zh)))
        # code line
        code_lines.append(code)
        # QA lines (standard library: baseline weight)
        for q in qas:
            qa_std.append("问 : " + " ".join(seg_zh(q)) + " 答 : " + code)
        dict_entries.append(dict_entry(qas, en, zh, code, kind="std"))

    # Curated ecosystem crates (blessed.rs / areweguiyet.com / arewewebyet.org)
    for crate, en, zh, qas in ECOSYSTEM_ENTRIES:
        en_lines.append(" ".join(seg_en(en)))
        zh_lines.append(" ".join(seg_zh(zh)))
        code = f"cargo add {crate}"
        code_lines.append(code)
        for q in qas:
            qa_high.append("问 : " + " ".join(seg_zh(q)) + " 答 : " + code)
        dict_entries.append(dict_entry(qas, en, zh, code, kind="crate", term=crate))

    # Language constructs (cheats.rs)
    for code, en, zh, qas in LANG_ENTRIES:
        en_lines.append(" ".join(seg_en(en)))
        zh_lines.append(" ".join(seg_zh(zh)))
        code_lines.append(code)
        for q in qas:
            qa_high.append("问 : " + " ".join(seg_zh(q)) + " 答 : " + code)
        dict_entries.append(dict_entry(qas, en, zh, code, kind="lang"))

    # Emphasize the QA pattern (the demo's core capability) so the small
    # model can memorize the question -> code mapping reliably.
    # High-value curated sources (cheats.rs / blessed.rs / areweguiyet.com /
    # lib.rs) are oversampled relative to the std library baseline.
    HIGH_QA_REPEAT, STD_QA_REPEAT = 10, 4
    lines = (
        en_lines
        + zh_lines
        + code_lines
        + qa_std * STD_QA_REPEAT
        + qa_high * HIGH_QA_REPEAT
    )

    # The key requirement: "我问我需要文件打开,给我对应代码实现".
    # The exact prompt must dominate so the tiny model answers it reliably.
    file_open_answer = 'use std::fs::File; let mut f = File::open("a.txt")?;'
    file_open_variants = [
        "我需要打开文件", "我需要打开一个文件", "我想打开文件", "帮我打开文件",
        "请给我打开文件的代码", "请问如何打开文件",
    ]
    file_open_lines = []
    for q in file_open_variants:
        for _ in range(8):
            file_open_lines.append("问 : " + " ".join(seg_zh(q)) + " 答 : " + file_open_answer)
    lines += file_open_lines * 10

    # High-quality dictionary: term + en/zh meanings + code + near-synonym
    # clusters, derived from the same collected corpora.
    import json as _json
    # merge the washed book / rust-by-example entries (concept -> meaning -> code)
    seen_terms = {e["term"] for e in dict_entries}
    for src in ["/workspace/data/washed/dict_book.json", "/workspace/data/washed/dict_rbe.json"]:
        try:
            with open(src, encoding="utf-8") as f:
                for e in _json.load(f):
                    t = (e.get("term") or "").strip()
                    zh = (e.get("zh") or "").strip()
                    code = (e.get("code") or "").strip()
                    if not t or not zh or t in seen_terms:
                        continue
                    # skip front-matter boilerplate headings
                    if any(b in t for b in ["前言", "Rust 程序", "通过例子", "本文档", "目录", "附录"]):
                        continue
                    if len(zh) < 10:
                        continue
                    seen_terms.add(t)
                    dict_entries.append({
                        "term": t[:24], "kind": "book",
                        "en": "", "zh": zh[:300],
                        "code": code[:300],
                        "synonyms": [],
                    })
        except FileNotFoundError:
            pass
    with open("/workspace/data/dictionary.json", "w", encoding="utf-8") as f:
        _json.dump(dict_entries, f, ensure_ascii=False, indent=2)
    print(f"dictionary.json: {len(dict_entries)} entries")

    for page, anchor, why in missing:
        print(f"[warn] {page}#{anchor}: {why}", file=sys.stderr)

    with open("/workspace/corpus.json", "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=4)

    # stats
    tokens = " ".join(lines).split()
    uniq = set(tokens)
    print(f"lines={len(lines)}  tokens={len(tokens)}  vocab={len(uniq)}")
    for t in ["问", "答", ":", "<END>"]:
        if t in uniq:
            print(f"  marker {t!r} in vocab")


if __name__ == "__main__":
    main()
