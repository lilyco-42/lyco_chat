// mod ui;
// use ui::流式输出;
// fn main() {
//     let text = "Hello, 终端流式输出！";
//     流式输出(text);
//     // axum 或 tiny_http 提供api服务
// }

mod core;

use core::trainer::Trainer;
use core::vocab::Vocab;
use rand::rng;

/// Main application entry point.
///
/// # Details
/// Initializes the TinyGPT model and runs the training loop.
/// Loads corpus from JSON and generates text after training.
///
/// # Returns
/// * `()` - Returns nothing.
fn main() {
    println!("微型GPT");
    let mut rng = rng();
    let corpus: Vec<String> = serde_json::from_str(include_str!("../corpus.json")).unwrap();
    let vocab = Vocab::from_corpus(&corpus);
    let mut trainer = Trainer::new(vocab.size, &vocab.data);
    trainer.train(&mut rng);
    let out = trainer.generate(vocab.encode("我"), 15, &mut rng);
    println!("\ngenerated text:\n{}", vocab.decode(&out));
}
