// mod ui;
// use ui::流式输出;
// fn main() {
//     let text = "Hello, 终端流式输出！";
//     流式输出(text);
//     // axum 或 tiny_http 提供api服务
// }

mod core;

use core::bitnet::quantize_model;
use core::trainer::Trainer;
use core::vocab::Vocab;
use rand::rng;

/// Main application entry point.
///
/// # Details
/// Trains on a bilingual Rust std library corpus and answers code questions.
/// Reads an optional question from the command line, defaulting to asking
/// how to open a file. Also compares full precision and BitNet b1.58
/// quantized inference.
///
/// # Returns
/// * `()` - Returns nothing.
/// Segments a question the same way the corpus generator does:
/// CJK characters become individual tokens, ASCII runs stay whole.
fn seg_tokens(q: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut buf = String::new();
    for ch in q.chars() {
        if ch.is_whitespace() {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            continue;
        }
        if ch.is_ascii() {
            buf.push(ch);
        } else {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(ch.to_string());
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Runs the dynamic tag-neuron activation demo (brain-region style).
fn activate_demo(query: &str) {
    use core::codex::TagActivation;
    let ta = TagActivation::load();
    println!("\n== 标签神经元动态激活（脑区式）==");
    println!("query: {query}");
    let ranked = ta.activate(query);
    println!("激活的脑区（crate ← 点亮的标签）:");
    for (c, act, fired) in ranked.iter().take(6) {
        let cat = ta.category_of.get(c).map(|s| s.as_str()).unwrap_or("?");
        println!("   {act:.0}  {c} [{cat}]  ← {}", fired.join(" / "));
    }
    if let Some((c, cat, act, fired)) = ta.best(query) {
        println!("\n→ 最强激活 [{cat}] {c}（激活值 {act:.0}，标签: {}）", fired.join(" / "));
    } else {
        println!("\n→ 无标签激活");
    }
}

/// Runs transformer-style labeled-attention intent inference.
fn intent_demo(query: &str) {
    use core::intent::{IntentAttention, tag_words};
    let ia = IntentAttention::new();
    println!("\n== transformer 标注注意力：意图推断 ==");
    println!("query: {query}");
    let tagged = tag_words(query);
    let roles: Vec<String> = tagged
        .iter()
        .map(|w| format!("{}({:?})", w.word, w.role))
        .collect();
    println!("jieba 词性标注: {}", roles.join("  "));
    match ia.infer(query) {
        Some(it) => {
            println!("注意力分布:");
            let mut att = it.attention.clone();
            att.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (name, w) in att.into_iter().take(5) {
                println!("   → {}: {:.2}", name, w);
            }
            println!("→ 推断意图 [{category}] → cargo add {crate_name}", category = it.category, crate_name = it.crate_name);
            if !it.traits.is_empty() {
                println!("   匹配特质: {}", it.traits.join(" / "));
            }
            println!("   说明: {}", it.description);
        }
        None => println!("→ 未能推断意图"),
    }
}

/// Runs the quantized MoE transformer (upgrades the dense base model).
fn moe_chat(question: &str, model_path: &str) {
    use core::moe_model::TinyGPTMoE;
    println!("\n== 量化 MoE Transformer (BitNet 专家 + DeepSeek shared) ==");
    let corpus: Vec<String> = serde_json::from_str(include_str!("../corpus.json")).unwrap();
    let vocab = Vocab::from_corpus(&corpus);
    let dense = core::tiny_gpt::TinyGPT::load(model_path).expect("需要已训练基模");
    let moe = TinyGPTMoE::from_dense(&dense, 4, 2);
    println!("from_dense 完成：每块 shared 1 + 路由专家 {}，top_k 2（专家 2-bit 量化）", 4);

    let tokens = seg_tokens(question);
    let prompt: Vec<usize> = tokens.iter().map(|t| vocab.encode(t)).collect();
    println!("question: {}", tokens.join(" "));
    let end = vocab.encode("<END>");
    let out = moe.generate_greedy_from(&prompt, 40);
    let answer: Vec<usize> = out.iter().copied().take_while(|&t| t != end).collect();
    println!("answer:\n{}", vocab.decode(&answer));
}

/// Runs the word-level multi-head attention block routing demo.
fn blocks_demo(query: &str) {
    use core::block_router::WordBlockRouter;
    let r = WordBlockRouter::new();
    println!("\n== 词级多头注意路由 + 动作块 ==");
    println!("query: {query}");
    match r.route(query) {
        Some(o) => {
            println!("jieba 切词: {}", o.words.join(" / "));
            for (ci, c) in r.concepts.iter().enumerate() {
                if o.attention[ci] > 0.0 {
                    println!("  注意力 → [{ci}] {}: {:.1}", c.name, o.attention[ci]);
                }
            }
            println!("→ 激活概念 [{}] {}", o.concept_index, o.concept_name);
            println!("→ 动作词路由到 [{action}] 块", action = o.action);
            println!("  代码块: {}", o.code);
        }
        None => println!("→ 未激活任何概念"),
    }
}

/// Runs the dynamic abstraction demo: crate-level -> learn hot API -> specific.
fn dynamic_demo(query: &str) {
    use core::concept::ConceptRouter;
    use core::dynamic::DynamicRegistry;

    println!("\n== 动态抽象：分类推荐 → 库内权重分析 → 具体 API ==");
    println!("query: {query}");

    // 1) 分类索引泛化推荐：lib.rs 分类 → 个性化推荐 crate（按修饰语挑）
    let categories = core::codex::CategoryIndex::from_data();
    let personalized = categories.recommend_crate(query);
    let concept_crate = {
        let concepts = ConceptRouter::new();
        concepts.route(query).map(|(i, _)| {
            concepts.concepts[i]
                .answer
                .split_whitespace()
                .last()
                .unwrap_or("")
                .to_string()
        })
    };
    let (crate_name, description, category_name, matched_traits) = match (&personalized, &concept_crate) {
        (Some((c, desc, traits)), _) => (
            c.clone(),
            desc.clone(),
            categories.recommend(query).map(|x| x.name.clone()),
            traits.clone(),
        ),
        (None, Some(c)) => (c.clone(), String::new(), None, vec![]),
        (None, None) => {
            println!("（无法推荐 crate）");
            return;
        }
    };
    match &category_name {
        Some(n) => {
            println!("① 分类推荐 [{n}] → cargo add {crate_name}");
            if !matched_traits.is_empty() {
                println!("   按修饰语匹配特质: {}", matched_traits.join(" / "));
            }
        }
        None => println!("① 概念路由 → cargo add {crate_name}"),
    }
    if !description.is_empty() {
        println!("   分类说明: {description}");
    }

    // 2) 库内代码块权重分析 → 热 API
    println!("② 分析 {crate_name} 库内代码块权重...");
    let mut registry = DynamicRegistry::new();
    let src = core::codex::find_crate_src(&crate_name).expect("本地应有 crate 源码");
    let ranked = core::codex::analyze_crate_api(&src);
    println!("   提取 {} 个 pub fn", ranked.len());

    // 3) 意图关键词匹配具体函数：name 命中权重高，doc 命中辅助，大小写不敏感
    let mut intent_keywords: Vec<String> = vec![];
    if let Some(c) = categories.recommend(query) {
        intent_keywords.extend(c.keywords.iter().cloned());
        for w in c.description.split_whitespace() {
            let w = w.trim_matches(|x: char| !x.is_alphanumeric());
            if w.len() >= 3 {
                intent_keywords.push(w.to_lowercase());
            }
        }
    }
    if let Some(concept) = concept_crate.as_ref() {
        let concepts = ConceptRouter::new();
        if let Some((i, _)) = concepts.route(query) {
            for k in &concepts.concepts[i].keywords {
                intent_keywords.push(k.word.to_string());
            }
        }
    }
    let best = ranked
        .iter()
        .map(|a| {
            let name_score = intent_keywords
                .iter()
                .filter(|k| a.name.to_lowercase().contains(&k.to_lowercase()))
                .count() as i32;
            let doc_score = intent_keywords
                .iter()
                .filter(|k| a.doc.to_lowercase().contains(&k.to_lowercase()))
                .count() as i32;
            (a, 3 * name_score + doc_score)
        })
        .filter(|(_, s)| *s > 0)
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(a, _)| a)
        .or_else(|| ranked.first())
        .expect("应分析出 API");
    println!("   意图匹配函数: {}(权重 {})", best.name, best.weight);
    println!("   签名: {}", best.signature);

    // 4) 文档中的答案：doc 注释（含用法）
    let answer_text = if best.doc.is_empty() {
        format!("{}", best.signature)
    } else {
        format!("文档: {}\n签名: {}", best.doc, best.signature)
    };
    let concept_name = category_name.clone().unwrap_or_else(|| crate_name.clone());
    let learn_keywords: Vec<&str> = intent_keywords.iter().map(|s| s.as_str()).collect();
    registry.learn(&crate_name, &concept_name, &learn_keywords, &answer_text);

    // 5) 同问再路由 → 具体 API
    let learned = registry.route(query).expect("应路由到已学专家");
    let cn = learned.crate_name.clone();
    let cp = learned.concept.clone();
    println!("③ 再问 → 路由到具体专家 [{cn}::{cp}]");
    println!("   给出文档答案:");
    println!("   {}", learned.api_answer);
}

/// Runs the jieba concept router demo: high-concept words activate an answer.
fn concept_demo(query: &str) {
    use core::concept::ConceptRouter;
    let r = ConceptRouter::new();
    println!("\n== jieba 分词 → 概念激活 ==");
    println!("query: {query}");
    let words = r.cut(query);
    println!("jieba 切词: {}", words.join(" / "));
    match r.route(query) {
        Some((id, score)) => {
            let c = &r.concepts[id];
            println!("→ 激活概念 [{id}] {name} (score {score:.2})", name = c.name, score = score);
            let kws: Vec<String> = c
                .keywords
                .iter()
                .map(|k| format!("{}:{}", k.word, k.weight))
                .collect();
            println!("  加权关键词: {}", kws.join("  "));
            println!("  神经回答: {}", c.answer);
        }
        None => println!("→ 未激活任何概念"),
    }
}

/// Runs the generic dictionary routing expert demo.
fn dict_demo(query: &str) {
    use core::dict::DictRouter;
    let router = DictRouter::new();
    println!("\n== 通用路由专家：四本字典 ==");
    println!("query: {query}");
    let (hits, intent) = router.route_with_intent(query);
    if hits.is_empty() {
        println!("（无命中）");
    }
    for h in &hits {
        println!(
            "→ [{:?}] 词条: {}  (score {:.2}, 来源: {}, 优先级 {})",
            h.source,
            h.definition.term,
            h.score,
            h.definition.origin,
            h.definition.priority
        );
        for m in &h.definition.meanings {
            println!("    释义: {m}");
        }
        if !h.definition.synonyms.is_empty() {
            println!("    近义: {}", h.definition.synonyms.join(" / "));
        }
    }
    if let Some((id, score)) = intent {
        let e = &router.intents.experts[id];
        println!(
            "→ 意图路由: {} (score {score:.2}) → 标准答案: {}",
            e.name, e.answer
        );
    }
}

/// Runs the semantic intent MoE demo: paraphrases route to one meaning.
fn semantic_demo(query: &str) {
    use core::semantic::SemanticMoE;
    let moe = SemanticMoE::new();
    println!("\n== 语义意图 MoE：近义句路由到同一含义 ==");
    println!("query: {query}");
    match moe.route(query) {
        Some((id, score)) => {
            let e = &moe.experts[id];
            println!("→ 路由到专家 [{id}] {name} (score {score:.2})", name = e.name, score = score);
            println!("  近义句簇: {}", e.paraphrases.join(" / "));
            println!("  标准答案: {}", e.answer);
        }
        None => println!("→ 未命中任何已知含义（低于阈值 {:.2}）", moe.threshold),
    }
}

/// Runs the MoE memory-hierarchy demo (Colibrì-style tiers) and prints metrics.
fn moe_demo() {
    use core::feed_forward::FeedForward;
    use core::linear::Linear;
    use core::moe::{ExpertCache, ExpertStore};
    use ndarray::Array2;

    let d = 64usize;
    let n_experts = 16usize;
    let top_k = 2usize;
    let budget = 4usize;

    let dense: Vec<FeedForward<Linear>> =
        (0..n_experts).map(|_| FeedForward::<Linear>::new(d)).collect();
    let store = ExpertStore::from_dense(&dense);
    let router = Linear::new(d, n_experts);
    let mut cache = ExpertCache::new(store, router, budget);

    println!("\n== MoE 存储层次 (Colibrì 思路) ==");
    println!("专家数={n_experts}  top_k={top_k}  常驻预算={budget}");
    let (store_b, fp32_b) = (cache.store_bytes(), cache.store_fp32_bytes());
    println!(
        "存储层(2-bit打包): {} B    fp32 等价: {} B    压缩 {:.1}x",
        store_b,
        fp32_b,
        fp32_b as f64 / store_b as f64
    );

    let x = Array2::from_elem((8, d), 1.0f32);
    for round in 0..5 {
        let gates = cache.gates(&x);
        cache.prefetch_next(&gates, top_k + 2); // router lookahead
        let out = cache.forward(&x);
        assert!(out.iter().all(|v| v.is_finite()));
        let (hits, misses) = cache.stats();
        let (pl, pu) = cache.prefetch_stats();
        println!(
            "round {round}: hits={hits} misses={misses} prefetch_loads={pl} prefetch_uses={pu} resident={} B",
            cache.resident_bytes()
        );
    }
}

/// Startup configuration loaded from a TOML file (see demo_chat.toml).
#[derive(serde::Deserialize)]
#[serde(default)]
struct CliConfig {
    model_path: Option<String>,
    mode: String,
    train_backend: String,
    steps: usize,
    infer_backend: String,
    question: String,
    port: u16,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            mode: "chat".into(),
            train_backend: "cpu".into(),
            steps: 0,
            infer_backend: "cpu".into(),
            question: "问 : 我需要打开文件".into(),
            port: 8080,
        }
    }
}

fn main() {
    let mut config_path = "demo_chat.toml".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "--config" {
            if let Some(p) = it.next() {
                config_path = p;
            }
        }
    }
    let cfg: CliConfig = if std::path::Path::new(&config_path).exists() {
        let raw = std::fs::read_to_string(&config_path).expect("read config");
        toml::from_str(&raw).expect("parse config toml")
    } else {
        println!("（未找到 {config_path}，使用默认配置）");
        CliConfig::default()
    };

    match cfg.mode.as_str() {
        "semantic" => {
            semantic_demo(&cfg.question);
            return;
        }        "dict" => {
            dict_demo(&cfg.question);
            return;
        }
        "concept" => {
            concept_demo(&cfg.question);
            return;
        }
        "dynamic" => {
            dynamic_demo(&cfg.question);
            return;
        }
        "blocks" => {
            blocks_demo(&cfg.question);
            return;
        }
        "intent" => {
            intent_demo(&cfg.question);
            return;
        }
        "activate" => {
            activate_demo(&cfg.question);
            return;
        }
        "moe-chat" => {
            moe_chat(&cfg.question, &cfg.model_path.clone().unwrap_or_default());
            return;
        }
        "moe" => {
            moe_demo();
            return;
        }
        "find-crate" => {
            let mut found = false;
            for (n, src) in core::codex::list_registry_crates() {
                let short = n.split('-').next().unwrap_or(&n);
                if short == cfg.question.trim() {
                    println!("{n}\n  -> {src}");
                    found = true;
                }
            }
            if !found {
                println!("（未在 registry 找到 {}）", cfg.question.trim());
            }
            return;
        }
        _ => {}
    }

    let model_path = cfg.model_path.clone();
    let question = cfg.question.clone();
    let use_candle = cfg.train_backend == "candle";
    let steps = cfg.steps;
    let serve_port = if cfg.mode == "serve" { cfg.port } else { 0 };
    let infer_backend = cfg.infer_backend.clone();

    println!("微型GPT (Rust 标准库双语基模)");

    // 时间接口：每次启动追加时间数据
    let t0 = std::time::Instant::now();
    let boot = core::time_log::record_startup(&core::time_log::default_log_path());
    println!("[info] 启动时间: {boot}");
    if let Some(w) = core::time_log::check_toolchain_warning() {
        println!("{w}");
    }
    // 版本落后检查：Rust 稳定版每 6 周一个 minor，落后太多就警告
    if let Some(w) = core::time_log::check_version_lag() {
        println!("{w}");
    }

    println!("[info] 读取语料 corpus.json ...");
    let corpus: Vec<String> = serde_json::from_str(include_str!("../corpus.json")).unwrap();
    println!("[info] 构建词表（{} 行语料）...", corpus.len());
    let vocab = Vocab::from_corpus(&corpus);
    println!("[info] 词表大小: {}，训练数据 {} tokens", vocab.size, vocab.data.len());
    let mut rng = rng();

    let mut trainer = None;
    if let Some(p) = &model_path {
        if std::path::Path::new(p).exists() {
            println!("[info] 加载基模 {p} ...");
            let t0 = std::time::Instant::now();
            trainer = Some(Trainer {
                model: core::tiny_gpt::TinyGPT::load(p).unwrap(),
                loader: core::data::DataLoader::new(&vocab.data),
            });
            println!("[info] 基模加载完成（{:.2}s）", t0.elapsed().as_secs_f32());
        } else {
            println!("[info] 未找到基模 {p}，将训练");
        }
    }
    if trainer.is_none() {
        let backend = if use_candle { "GPU(candle)" } else { "CPU(ndarray)" };
        let n = if steps > 0 { steps } else { core::config::CFG.epochs };
        println!("[info] 开始训练基模（{backend}，{n} 步）...");
        let model = if use_candle {
            core::candle_trainer::train_candle(
                vocab.size,
                &vocab.data,
                core::candle_trainer::pick_device().unwrap(),
                steps,
                &mut rng,
            )
            .unwrap()
        } else {
            let mut t = Trainer::new(vocab.size, &vocab.data);
            if steps > 0 {
                t.train_steps(steps, &mut rng);
            } else {
                t.train(&mut rng);
            }
            t.model
        };
        if let Some(p) = &model_path {
            model.save(p).unwrap();
            println!("[info] 基模已保存到 {p}");
        }
        trainer = Some(Trainer {
            model,
            loader: core::data::DataLoader::new(&vocab.data),
        });
    }
    let trainer = trainer.unwrap();

    if serve_port > 0 {
        let model = trainer.model.clone();
        let vocab = std::sync::Arc::new(vocab);
        println!("[info] 启动服务 ...");
        core::serve::serve(model, vocab, serve_port).unwrap();
        return;
    }

    let tokens = seg_tokens(&question);
    let prompt: Vec<usize> = tokens.iter().map(|t| vocab.encode(t)).collect();
    println!("\nquestion: {}", tokens.join(" "));
    let end = vocab.encode("<END>");

    // 客户端自选推理后端：CPU (ndarray) 或 GPU (candle CUDA)
    let gpu = infer_backend == "gpu" || (infer_backend == "auto" && core::candle_trainer::cuda_available());
    println!("[info] 推理后端: {}", if gpu { "GPU (candle/cuda)" } else { "CPU (ndarray)" });
    println!("[info] 生成回答中 ...");
    let gen_tokens = if gpu {
        let device = core::candle_trainer::pick_device().unwrap();
        let cmodel = core::candle_trainer::CandleGPT::from_ndarray(&trainer.model, &device).unwrap();
        cmodel.generate_greedy_from(&prompt, 40, &device).unwrap()
    } else {
        trainer.model.generate_greedy_from(&prompt, 40)
    };
    let answer: Vec<usize> = gen_tokens.iter().copied().take_while(|&t| t != end).collect();
    println!("answer:\n{}", vocab.decode(&answer));

    let qmodel = quantize_model(&trainer.model);
    let qout = qmodel.generate_greedy_from(&prompt, 40);
    let qanswer: Vec<usize> = qout.iter().copied().take_while(|&t| t != end).collect();
    println!("\nquantized (BitNet b1.58) answer:\n{}", vocab.decode(&qanswer));
}
