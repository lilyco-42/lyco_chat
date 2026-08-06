/*
 * @file tiny_gpt.rs
 * @brief TinyGPT model implementation
 * @author Kevin Thomas
 * @date 2025
 *
 * MIT License
 *
 * Copyright (c) 2025 Kevin Thomas
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//! FILE: tiny_gpt.rs
//!
//! DESCRIPTION:
//! TinyGPT Model Implementation.
//!
//! BRIEF:
//! Provides complete GPT architecture with transformer blocks.
//! Implements forward pass, backpropagation, and text generation.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::core::block::Block;
use crate::core::config::CFG;
use crate::core::embedding::Embedding;
use crate::core::layer_norm::LayerNorm;
use crate::core::linear::{Linear, LinearLike};
use crate::core::math::softmax1d;
use crate::core::sampling::sample;
use ndarray::{Array1, Array2, Array3, s};
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};

/// TinyGPT language model.
///
/// # Details
/// Complete GPT architecture with embeddings, transformer blocks, and output head.
/// Supports training and autoregressive text generation.
/// Generic over the linear layer type: `Linear` for full precision,
/// `BitLinear` for ternary quantized inference.
///
/// # Fields
/// * `tok` - Token embedding layer
/// * `pos` - Position embedding layer
/// * `blocks` - Stack of transformer blocks
/// * `ln` - Final layer normalization
/// * `head` - Output projection to vocabulary
/// * `vocab` - Vocabulary size
#[derive(Clone, Serialize, Deserialize)]
pub struct TinyGPT<L: LinearLike = Linear> {
    pub(crate) tok: Embedding,
    pub(crate) pos: Embedding,
    pub(crate) blocks: Vec<Block<L>>,
    pub(crate) ln: LayerNorm,
    pub(crate) head: L,
    pub(crate) vocab: usize,
}

impl<L: LinearLike + Sync> TinyGPT<L> {
    /// Creates new TinyGPT model.
    ///
    /// # Details
    /// Initializes all model components based on configuration.
    ///
    /// # Arguments
    /// * `v` - Vocabulary size
    ///
    /// # Returns
    /// * `Self` - TinyGPT instance
    pub fn new(v: usize) -> Self {
        Self {
            tok: Embedding::new(v, CFG.embed_dim),
            pos: Embedding::new(CFG.block_size, CFG.embed_dim),
            blocks: (0..CFG.n_layers)
                .map(|_| Block::<L>::new(CFG.embed_dim, CFG.n_heads))
                .collect(),
            ln: LayerNorm::new(CFG.embed_dim),
            head: L::new(CFG.embed_dim, v),
            vocab: v,
        }
    }

    /// Computes forward pass for single sequence.
    ///
    /// # Details
    /// Runs token through embeddings, transformer blocks, and output head.
    ///
    /// # Arguments
    /// * `idx` - Token indices
    ///
    /// # Returns
    /// * `Array2<f32>` - Logits for each position
    pub fn forward(&self, idx: &[usize]) -> Array2<f32> {
        self.forward_hidden_logits(idx).1
    }

    /// Computes hidden states and logits in a single pass.
    ///
    /// # Details
    /// Returns the transformer output before the final projection together
    /// with the logits, sharing one forward pass instead of two.
    ///
    /// # Arguments
    /// * `seq` - Token indices
    ///
    /// # Returns
    /// * `(Array2<f32>, Array2<f32>)` - (hidden states, logits)
    fn forward_hidden_logits(&self, seq: &[usize]) -> (Array2<f32>, Array2<f32>) {
        let mut x = &self.tok.forward(seq) + &self.pos.forward(&(0..seq.len()).collect::<Vec<_>>());
        for b in &self.blocks {
            x = b.forward(&x);
        }
        let h = self.ln.forward(&x);
        let logits = self.head.forward(&h);
        (h, logits)
    }

    /// Computes forward pass for batch.
    ///
    /// # Details
    /// Processes multiple sequences in parallel across CPU cores.
    ///
    /// # Arguments
    /// * `batch` - Batch of token indices
    ///
    /// # Returns
    /// * `Array3<f32>` - Logits for each batch and position
    pub fn forward_batch(&self, batch: &Array2<usize>) -> Array3<f32> {
        use rayon::prelude::*;
        let (b, t, v) = (batch.shape()[0], batch.shape()[1], self.vocab);
        let rows: Vec<Array2<f32>> = (0..b)
            .into_par_iter()
            .map(|i| self.forward(&batch.row(i).to_vec()))
            .collect();
        let mut out = Array3::zeros((b, t, v));
        for (i, row) in rows.into_iter().enumerate() {
            out.slice_mut(s![i, .., ..]).assign(&row);
        }
        out
    }

    /// Computes hidden states before output head.
    ///
    /// # Details
    /// Returns transformer output before final projection.
    ///
    /// # Arguments
    /// * `seq` - Token indices
    ///
    /// # Returns
    /// * `Array2<f32>` - Hidden states
    fn hidden(&self, seq: &[usize]) -> Array2<f32> {
        self.forward_hidden_logits(seq).0
    }

    /// Generates text autoregressively.
    ///
    /// # Details
    /// Samples tokens one at a time from model predictions.
    /// Uses context window limited by block size.
    ///
    /// # Arguments
    /// * `start` - Starting token index
    /// * `n` - Number of tokens to generate
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * `Vec<usize>` - Generated token sequence
    pub fn generate(&self, start: usize, n: usize, rng: &mut ThreadRng) -> Vec<usize> {
        self.generate_from(&[start], n, rng)
    }

    /// Generates text conditioned on a prompt.
    ///
    /// # Details
    /// Starts from the full prompt tokens and samples `n` more tokens.
    /// Only the last `block_size` tokens are used as context.
    ///
    /// # Arguments
    /// * `prompt` - Initial token sequence
    /// * `n` - Number of tokens to generate
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * `Vec<usize>` - Prompt followed by generated tokens
    pub fn generate_from(&self, prompt: &[usize], n: usize, rng: &mut ThreadRng) -> Vec<usize> {
        let mut out = prompt.to_vec();
        for _ in 0..n {
            let ctx: Vec<_> = out
                .iter()
                .rev()
                .take(CFG.block_size)
                .rev()
                .copied()
                .collect();
            let logits = self.forward(&ctx);
            out.push(sample(&softmax1d(&logits.row(logits.shape()[0] - 1)), rng));
        }
        out
    }

    /// Generates text greedily (argmax) from a prompt.
    ///
    /// # Details
    /// Deterministically picks the most likely token at every step, which is
    /// better suited for code answers than random sampling.
    ///
    /// # Arguments
    /// * `prompt` - Initial token sequence
    /// * `n` - Number of tokens to generate
    ///
    /// # Returns
    /// * `Vec<usize>` - Prompt followed by generated tokens
    pub fn generate_greedy_from(&self, prompt: &[usize], n: usize) -> Vec<usize> {
        let mut out = prompt.to_vec();
        for _ in 0..n {
            let ctx: Vec<_> = out
                .iter()
                .rev()
                .take(CFG.block_size)
                .rev()
                .copied()
                .collect();
            let logits = self.forward(&ctx);
            let row = logits.row(logits.shape()[0] - 1);
            let mut best = 0;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            out.push(best);
        }
        out
    }

    /// Zeros all gradient accumulators.
    ///
    /// # Details
    /// Resets gradients for embeddings, blocks, and output head.
    pub fn zero_grad(&mut self) {
        self.tok.zero_grad();
        self.pos.zero_grad();
        self.head.zero_grad();
        self.blocks.iter_mut().for_each(|b| b.zero_grad());
    }

    /// Updates all parameters with gradient descent.
    ///
    /// # Details
    /// Applies gradient update to embeddings, blocks, and output head.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.tok.step(lr);
        self.pos.step(lr);
        self.head.step(lr);
        self.blocks.iter_mut().for_each(|b| b.step(lr));
    }
}

impl TinyGPT<Linear> {
    /// Saves the model to a JSON file.
    ///
    /// # Arguments
    /// * `path` - Output file path
    ///
    /// # Returns
    /// * `anyhow::Result<()>` - Ok on success
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Loads a model from a JSON file.
    ///
    /// # Arguments
    /// * `path` - Input file path
    ///
    /// # Returns
    /// * `anyhow::Result<TinyGPT<Linear>>` - The loaded model
    pub fn load(path: &str) -> anyhow::Result<TinyGPT<Linear>> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Computes gradients for training.
    ///
    /// # Details
    /// Performs simplified backpropagation through output head and embeddings.
    /// Uses cross-entropy loss gradient. Batch items are processed in
    /// parallel across CPU cores, with per-item gradients merged afterwards.
    /// Each item's forward pass is computed only once.
    ///
    /// # Arguments
    /// * `batch` - Input token indices
    /// * `targets` - Target token indices
    pub fn backward(&mut self, batch: &Array2<usize>, targets: &Array2<usize>) {
        use rayon::prelude::*;
        let (b, t, v) = (batch.shape()[0], batch.shape()[1], self.vocab);
        let norm = (b * t) as f32;
        let items: Vec<(Array2<f32>, Array2<f32>)> = (0..b)
            .into_par_iter()
            .map(|i| self.forward_hidden_logits(&batch.row(i).to_vec()))
            .collect();
        let contribs: Vec<(
            Array2<f32>,
            Array1<f32>,
            Array2<f32>,
            Array2<f32>,
        )> = (0..b)
            .into_par_iter()
            .map(|i| {
                let (h, logits) = &items[i];
                let seq = batch.row(i).to_vec();
                let mut head_dw = Array2::<f32>::zeros((CFG.embed_dim, v));
                let mut head_db = Array1::<f32>::zeros(v);
                let mut tok_dw = Array2::<f32>::zeros((v, CFG.embed_dim));
                let mut pos_dw = Array2::<f32>::zeros((CFG.block_size, CFG.embed_dim));
                for j in 0..t {
                    let mut grad = softmax1d(&logits.row(j));
                    grad[targets[[i, j]]] -= 1.0;
                    grad /= norm;
                    let hj = h.row(j);
                    for k in 0..v {
                        for l in 0..CFG.embed_dim {
                            head_dw[[l, k]] += hj[l] * grad[k];
                        }
                        head_db[k] += grad[k];
                    }
                    let gh = self.head.w.dot(&grad);
                    for l in 0..CFG.embed_dim {
                        tok_dw[[seq[j], l]] += gh[l];
                        pos_dw[[j, l]] += gh[l];
                    }
                }
                (head_dw, head_db, tok_dw, pos_dw)
            })
            .collect();

        self.head.dw.fill(0.0);
        self.head.db.fill(0.0);
        self.tok.dw.fill(0.0);
        self.pos.dw.fill(0.0);
        for (head_dw, head_db, tok_dw, pos_dw) in contribs {
            self.head.dw += &head_dw;
            self.head.db += &head_db;
            self.tok.dw += &tok_dw;
            self.pos.dw += &pos_dw;
        }
    }
}

/// Computes cross-entropy loss.
///
/// # Details
/// Calculates average negative log probability of targets.
///
/// # Arguments
/// * `logits` - Model output logits
/// * `tgt` - Target token indices
///
/// # Returns
/// * `f32` - Average cross-entropy loss
pub fn cross_entropy(logits: &Array3<f32>, tgt: &Array2<usize>) -> f32 {
    let (b, t) = (logits.shape()[0], logits.shape()[1]);
    let mut loss = 0.0;
    for i in 0..b {
        for j in 0..t {
            loss -= softmax1d(&logits.slice(s![i, j, ..]))[tgt[[i, j]]].ln();
        }
    }
    loss / (b * t) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};
    use rand::rng;

    /// Tests TinyGPT creation.
    #[test]
    fn test_tiny_gpt_new() {
        let model = TinyGPT::<Linear>::new(100);
        assert_eq!(model.head.w.shape()[1], 100);
    }

    /// Tests forward pass shape.
    #[test]
    fn test_tiny_gpt_forward_shape() {
        let model = TinyGPT::<Linear>::new(50);
        let result = model.forward(&[0, 1, 2]);
        assert_eq!(result.shape(), &[3, 50]);
    }

    /// Tests forward pass produces finite values.
    #[test]
    fn test_tiny_gpt_forward_finite() {
        let model = TinyGPT::<Linear>::new(20);
        let result = model.forward(&[0, 1]);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests forward_batch shape.
    #[test]
    fn test_tiny_gpt_forward_batch_shape() {
        let model = TinyGPT::<Linear>::new(30);
        let batch = arr2(&[[0, 1, 2], [3, 4, 5]]);
        let result = model.forward_batch(&batch);
        assert_eq!(result.shape(), &[2, 3, 30]);
    }

    /// Tests generate output length.
    #[test]
    fn test_tiny_gpt_generate_length() {
        let model = TinyGPT::<Linear>::new(20);
        let mut rng = rng();
        let result = model.generate(0, 5, &mut rng);
        assert_eq!(result.len(), 6); // start + 5 generated
    }

    /// Tests generate produces valid tokens.
    #[test]
    fn test_tiny_gpt_generate_valid() {
        let model = TinyGPT::<Linear>::new(15);
        let mut rng = rng();
        let result = model.generate(0, 3, &mut rng);
        assert!(result.iter().all(|&t| t < 15));
    }

    /// Tests zero_grad clears gradients.
    #[test]
    fn test_tiny_gpt_zero_grad() {
        let mut model = TinyGPT::<Linear>::new(10);
        model.head.dw.fill(1.0);
        model.tok.dw.fill(1.0);
        model.zero_grad();
        assert!(model.head.dw.iter().all(|&x| x == 0.0));
        assert!(model.tok.dw.iter().all(|&x| x == 0.0));
    }

    /// Tests step updates parameters.
    #[test]
    fn test_tiny_gpt_step() {
        let mut model = TinyGPT::<Linear>::new(10);
        let before = model.head.w[[0, 0]];
        model.head.dw.fill(1.0);
        model.step(0.1);
        assert!((model.head.w[[0, 0]] - (before - 0.1)).abs() < 1e-5);
    }

    /// Tests backward computes gradients.
    #[test]
    fn test_tiny_gpt_backward() {
        let mut model = TinyGPT::<Linear>::new(10);
        let batch = arr2(&[[0, 1], [2, 3]]);
        let targets = arr2(&[[1, 2], [3, 4]]);
        model.zero_grad();
        model.backward(&batch, &targets);
        // At least some gradient should be non-zero
        let has_grad = model.head.dw.iter().any(|&x| x != 0.0);
        assert!(has_grad);
    }

    /// Tests cross_entropy is positive.
    #[test]
    fn test_cross_entropy_positive() {
        let logits = arr3(&[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        let tgt = arr2(&[[0, 1]]);
        let loss = cross_entropy(&logits, &tgt);
        assert!(loss > 0.0);
    }

    /// Tests cross_entropy decreases with correct predictions.
    #[test]
    fn test_cross_entropy_correct() {
        let logits_good = arr3(&[[[10.0, 0.0, 0.0]]]);
        let logits_bad = arr3(&[[[0.0, 0.0, 10.0]]]);
        let tgt = arr2(&[[0]]);
        let loss_good = cross_entropy(&logits_good, &tgt);
        let loss_bad = cross_entropy(&logits_bad, &tgt);
        assert!(loss_good < loss_bad);
    }

    /// Tests cross_entropy is finite.
    #[test]
    fn test_cross_entropy_finite() {
        let logits = arr3(&[[[1.0, 2.0], [3.0, 4.0]]]);
        let tgt = arr2(&[[0, 1]]);
        let loss = cross_entropy(&logits, &tgt);
        assert!(loss.is_finite());
    }

    /// Tests save/load round-trips the model exactly.
    #[test]
    fn test_save_load_roundtrip() {
        let model = TinyGPT::<Linear>::new(20);
        let x = [1usize, 2, 3, 4];
        let before = model.forward(&x);
        let path = std::env::temp_dir().join("tinygpt_roundtrip.json");
        let p = path.to_str().unwrap();
        model.save(p).unwrap();
        let loaded = TinyGPT::<Linear>::load(p).unwrap();
        let after = loaded.forward(&x);
        let _ = std::fs::remove_file(p);
        assert_eq!(before.shape(), after.shape());
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-4, "roundtrip mismatch: {a} vs {b}");
        }
    }

    /// Tests single token forward.
    #[test]
    fn test_tiny_gpt_single_token() {
        let model = TinyGPT::<Linear>::new(10);
        let result = model.forward(&[0]);
        assert_eq!(result.shape(), &[1, 10]);
    }
}
