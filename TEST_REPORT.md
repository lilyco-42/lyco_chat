# 测试报告

**项目**: 微型GPT — Rust 标准库双语学习助手（`demo_chat`）
**日期**: 2026-08-06
**命令**: `cargo test --bin demo_chat -j 8`
**结果**: ✅ **220 通过 / 0 失败 / 0 忽略**（约 46 秒）

---

## 汇总

| 指标 | 值 |
|---|---|
| 测试总数 | 220 |
| 通过 | 220 |
| 失败 | 0 |
| 覆盖模块 | 29 |
| 编译警告 | 少量预存（unused import / variable，不影响正确性） |

## 按模块分布

| 模块 | 测试数 | 覆盖内容 |
|---|---:|---|
| `core::moe` | 16 | Colibrì 打包 MoE、专家缓存（LRU 热度 / 预取 / 统计） |
| `core::bitnet` | 16 | BitNet 1.58bit 量化、与 microsoft/BitNet i2_s 逐位对齐、2-bit 打包 |
| `core::tiny_gpt` | 14 | 泛型 Transformer 前向 / 生成（`LinearLike` 双后端） |
| `core::math` | 13 | 张量 / 数值运算 |
| `core::trainer` | 12 | 训练前向、梯度、loss 有限性 |
| `core::vocab` | 10 | 词表构建与确定性排序 |
| `core::time_log` | 9 | ISO 时间、工具链版本滞后检查 |
| `core::embedding` | 9 | token / 位置嵌入 |
| `core::config` | 9 | 模型超参数配置 |
| `core::codex` | 9 | ast-grep API 提取、crate 标签激活、动态专家 |
| `core::router` | 8 | 9 层路由栈、crate 专家、动态 ast-grep 委派 |
| `core::layer_norm` | 8 | LayerNorm |
| `core::head` | 8 | 输出头 |
| `core::feed_forward` | 8 | FFN |
| `core::block` | 8 | Transformer Block |
| `core::attention` | 8 | 多头注意力 |
| `core::linear` | 7 | `LinearLike` trait / Linear 实现 |
| `core::serve` | 5 | Web 决策链、`/tool:ast-grep` 路由 |
| `core::sampling` | 5 | 采样策略 |
| `core::dynamic` | 5 | 动态专家演示 |
| `core::data` | 5 | 语料数据 |
| `core::concept` | 5 | 概念路由（加权关键词 + 阈值） |
| `core::block_router` | 5 | 动作块路由（jieba 词级多头注意力） |
| `core::semantic` | 4 | 近义意图路由 |
| `core::dict` | 4 | 中英程序词典释义 |
| `core::moe_model` | 3 | SharedSparseMoE、稠密→MoE 升级 |
| `core::intent` | 3 | 词性标注 + 意图注意力 |
| `core::learn` | 2 | Learner 自动学习 |
| `core::candle_trainer` | 2 | CandleMoERouter 两阶段训练（loss 下降 + 路由正确性） |

## 关键场景回归（serve 决策链）

| 提问 | 期望 | 结果 |
|---|---|---|
| 怎么删除文件 / 如何定义结构体 / 声明函数 / 我需要打开文件 | 返回代码（非释义） | ✅ |
| 怎么生成随机数 | `cargo add rand` | ✅ |
| 泛型是什么 / 结构体是什么 | 词典释义 | ✅ |
| 中午吃什么 | 诚实「不会」兜底 | ✅ |
| 我需要的是axum里面的 | crate 描述（不解释"crate"单词） | ✅ |
| 你自动cargo add axum 看crate | 委派动态 ast-grep 专家 | ✅ |
| 如何新建一个http 请求 | 推荐 HTTP 客户端（泛词门控不劫持） | ✅ |
| `/tool:ast-grep serde_json 怎么解析json` | ast-grep 源码分析 → `from_str` 等 | ✅ |

## 已知说明

- `candle_trainer::test_moe_two_phase_training` 在**全量并行**下偶发超时（GPU 争用），单独运行 3/3 稳定通过；已改为断言「loss 下降 + 路由仍正确」，去除随机初始化带来的 flaky。
- 动态 ast-grep 测试依赖网络（首次从 crates.io 下载源码），已缓存于 `/tmp/opencode/cratesrc`，断网环境会走本地 registry。
