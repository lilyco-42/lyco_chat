# 微型GPT — Rust 标准库双语学习助手

一个从零实现的 Rust 学习助手（微型 GPT）：**纯 Rust** 实现的 Transformer + BitNet 2-bit 量化 + MoE 混合专家，配上知识路由决策链（词典释义 / 概念代码 / 动作块 / crate 推荐 / 动态 ast-grep 专家），能回答「我需要打开文件」「怎么生成随机数」「axum 怎么新建路由」这类 Rust 问题。

> 基座参考 [mytechnotalent/TinyGPT](https://github.com/mytechnotalent/TinyGPT) 的教程实现，本仓库在此基础上扩展为可用的中文问答系统。

---

## 特性

- **从零实现的 Transformer**：只用 `ndarray` 做张量运算（CPU），`candle` 做 CUDA 训练/推理，不依赖深度学习框架。
- **BitNet**：1.58bit 三元权重量化（absmean 按通道、2-bit 打包），与 microsoft/BitNet i2_s 逐位对齐。
- **Colibrì 风格 MoE**：`PackedBitLinear` / `PackedExpert` / `ExpertStore` / `SparseBitMoE` + LRU 专家缓存（热度 + 预取）。
- **DeepSeek 风格 SharedSparseMoE**：共享专家 + top-k 路由，`TinyGPTMoE::from_dense` 一键从稠密基模升级。
- **9 层知识路由决策链**（`src/core/router.rs`），模型不兜底生成，诚实说「不会」并自动学习新词：
  ```
  crate专家 → 词典释义 → 概念代码 → 动作块 → 标签激活(crate真实标签)
  → 意图注意力 → crates.io 实时搜索 → 动态ast-grep专家 → 兜底+自学习
  ```
- **动态 ast-grep 专家**：本地 registry 或自动从 crates.io 下载源码，jieba 抓意图关键词，扫 `pub fn` + 文档，按（名字命中 / 文档 / 签名 / 引用权重）排序回答。
- **Web 对话**：OpenAI 兼容 `/v1/chat/completions` + SSE 流式，内置聊天页（nvim 风格工具链对话框）。
- **双语数据管道**：Rust 官方书 / Rust by Example / std 中文文档 / blessed / lib.rs 分类等 7 个来源自动清洗成中英对照语料。

---

## 架构

```
┌────────────────────────────────────────────────────────────┐
│  TinyGPT<L>  L: LinearLike (Linear | BitLinear)            │
│  Token Embedding + Position Embedding → Block×N → LayerNorm │
│  Block = MultiHeadAttention → FeedForward                   │
│  │                                                          │
│  ├── BitNet  量化权重 (absmean 三元 → 2-bit 打包)            │
│  ├── Colibrì MoE  PackedExpert / ExpertStore / SparseBitMoE │
│  └── SharedSparseMoE  shared + top-k (DeepSeek)             │
├────────────────────────────────────────────────────────────┤
│  candle_trainer  CandleGPT (CUDA 训练 / 推理, 权重双向转换)   │
├────────────────────────────────────────────────────────────┤
│  RouterStack  9 层决策链 + Learner 自动学习                  │
│  serve         OpenAI SSE / chat.html / /tool:ast-grep      │
└────────────────────────────────────────────────────────────┘
```

**模块速览**（`src/core/`）：

| 模块 | 作用 |
|---|---|
| `tiny_gpt.rs` / `linear.rs` / `head.rs` / `attention.rs` / `feed_forward.rs` | 泛型化 Transformer（`LinearLike` 双后端） |
| `bitnet.rs` | BitNet 1.58bit 量化 + 整数 LUT 矩阵乘 |
| `moe.rs` | Colibrì 风格打包 MoE + 专家缓存 |
| `moe_model.rs` | SharedSparseMoE 共享专家 + top-k 路由 |
| `candle_trainer.rs` | CandleGPT CUDA 训练 / 推理 / CandleMoERouter |
| `router.rs` | 9 层路由栈 + crate 专家 + 动态 ast-grep 专家 |
| `codex.rs` | ast-grep API 分析、crate 标签激活、crates.io 搜索学习 |
| `intent.rs` / `semantic.rs` | 词性标注 + transformer 风格意图注意力 |
| `dict.rs` / `concept.rs` / `block_router.rs` | 词典释义 / 概念代码 / 动作块 |
| `learn.rs` | 答不上就学：Learner 自动记忆新词 |
| `serve.rs` | OpenAI 兼容流式对话服务 |
| `time_log.rs` | 时间记录 + 工具链版本滞后检查 |

---

## 快速开始

```bash
# 1. 构建（release）
cargo build --release -j 8

# 2. 训练基模（CPU，约 15000 步）
#    也可用 GPU：tools 里跑 candle_trainer（需 CUDA，candle-core 已开 cuda feature）

# 3. 命令行对话
./target/release/demo_chat --config demo_chat.toml

# 4. 启动 Web 服务（http://127.0.0.1:8080）
./target/release/demo_chat --config server.toml
```

**配置文件**（TOML，`--config` 指定，默认 `demo_chat.toml`）：

| 字段 | 说明 |
|---|---|
| `mode` | `chat` / `serve` / `semantic` / `dict` / `concept` / `dynamic` / `moe` / `blocks` / `intent` / `activate` / `moe-chat` / `find-crate` |
| `model_path` | 基模权重 JSON（`model/base_gpu.json`，用 `model/` 下已训练模型） |
| `question` | chat 模式的提问 |
| `port` | serve 模式的端口（默认 8080） |

---

## 安装与发布

### crates.io 安装（源码编译）

```bash
cargo install lyco_chat
# 二进制为 demo_chat；GPU 版：cargo install lyco_chat --features cuda（需本地 CUDA）
```

### cargo-binstall（预编译二进制，秒装）

```bash
cargo binstall lyco_chat
```

打 `v*` tag 后，GitHub Actions 会把各平台预编译产物上传到 Release（命名
`lyco_chat-{tag}-{target}.tar.gz/.zip`，与 `[package.metadata.binstall]` 对齐）。

### 交叉编译目标

| 平台 | target | 构建方式 |
|---|---|---|
| Linux x86_64 / aarch64 | `x86_64-unknown-linux-gnu` / `aarch64-unknown-linux-gnu` | cargo-zigbuild |
| Android（Termux 用 aarch64） | `aarch64-linux-android` / `armv7-linux-androideabi` / `x86_64-linux-android` | cargo-zigbuild |
| Windows | `x86_64-pc-windows-msvc` / `aarch64-pc-windows-msvc` | 本机 MSVC |
| macOS | `aarch64-apple-darwin` / `x86_64-apple-darwin` / universal(lipo) | Xcode |
| iOS | `aarch64-apple-ios` / 模拟器 universal | Xcode |

Termux 用户直接下载 **aarch64-linux-android** 的 `lyco_chat-*.tar.gz` 即可运行，
或在 Termux 里 `pkg install zig` 后用 `cargo install lyco_chat` 原生编译。

### 发布流程（GitHub Actions，见 `.github/workflows/release.yml`）

1. `git tag v0.1.0 && git push --tags`
2. workflow 自动：`create-gh-release-action` 建 Release → `upload-rust-binary-action`
   交叉编译 12 个目标并上传（含 sha256）→ `cargo publish` 发 crates.io
3. 首次发布前：在仓库 Secrets 配置 `CARGO_REGISTRY_TOKEN`（需 crates.io 账号的
   `publish` 权限 token），并把默认分支暴露给 `environment: crates-io`

### 本地 GPU（CUDA）构建

发布版默认是纯 CPU（crates.io 的 candle）。本地 GPU 训练/推理：

```bash
# .cargo/config.toml 已把 candle 指回 git 提交（含 CUDA 13 兼容补丁）
cargo build --release --features cuda
```

> `.cargo/` 与 `model/`、`data/washed/` 不入库，因此 CI / crates.io 安装
> 用的是 crates.io candle（CPU），交叉编译也完全不依赖 CUDA。

---

## 试问

| 提问 | 回答 |
|---|---|
| 我需要打开文件 | `use std::fs::File; File::open("a.txt")?;` |
| 怎么生成随机数 | `cargo add rand` |
| 泛型是什么 | 词典释义：类型参数化 |
| 如何定义函数 | `fn f(...)` 动作块 |
| 怎么写爬虫 | 转译 crawler → crates.io 搜索 → 推荐并学习 |
| 我需要的是axum里面的 | crate axum 描述（http/routing/web） |
| axum 帮我找找怎么新建路由 | 委派动态 ast-grep 专家：`new` / `route` 等 pub fn + 文档 |
| `/tool:ast-grep tokio 怎么创建异步任务` | 斜杠工具，直接 ast-grep 源码 |

---

## 数据与工具

| 路径 | 说明 |
|---|---|
| `corpus.json` | 训练语料（1738 行中英对照，词表 1230） |
| `data/dictionary.json` | 830 条中英程序词典 |
| `data/crates_top.json` | crates.io 下载量 Top-205 crate（描述 + 真实关键词） |
| `data/crate_tags.json` | 196 个 crate 的标签神经元 |
| `data/librs_categories.json` | lib.rs 41+ 分类（含类别关键词） |
| `data/learned.json` | Learner 自动学习的新词 |
| `tools/wash_data.py` | 7 个来源的语料清洗（book / rust_by_example / std_zh / librs / blessed / areweguiyet / cheats） |
| `tools/gen_corpus.py` | 汇总语料 + 词典 |
| `tools/fetch_crate_tags.py` | 抓取 crate 标签 |
| `tools/fetch_top_crates.py` | 抓取最常用 crate 关键词 |

> 训练基模 `model/base_gpu.json` 与原始清洗语料 `data/washed/` 体积较大，未入库；可用上述工具重新生成。

---

## 测试

```bash
cargo test --bin demo_chat -j 8
```

覆盖：Transformer 前向/梯度、BitNet 量化对齐、MoE 两阶段训练、知识路由（词典/概念/动作块/推荐/意图/动态专家）、Learner 自学习、Web 决策链。详见 [TEST_REPORT.md](TEST_REPORT.md)。

---

## 环境要求

- Rust 2024 edition（1.85+）
- 可选 CUDA 12+/13.0（GPU 训练，`candle-core` 已开 `cuda` feature）
- 联网（crates.io 实时搜索、动态 ast-grep 源码下载）

## License

MIT
