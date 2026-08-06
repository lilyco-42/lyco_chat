/*
 * @file moe_model.rs
 * @brief Quantized MoE transformer: MoE FFN integrated into TinyGPT
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

//! FILE: moe_model.rs
//!
//! DESCRIPTION:
//! A transformer whose feed-forward is a quantized Mixture-of-Experts
//! (DeepSeek-style shared expert + top-k routed experts).
//!
//! BRIEF:
//! `from_dense` upgrades a trained `TinyGPT<Linear>` into `TinyGPTMoE`:
//! attention / norms / embeddings stay fp32, each block's FFN becomes a
//! `SharedSparseMoE` with 2-bit quantized experts (the shared expert reuses
//! the original FFN so the model's knowledge is preserved). Runs greedy
//! generation end-to-end.

use crate::core::attention::MultiHeadAttention;
use crate::core::config::CFG;
use crate::core::embedding::Embedding;
use crate::core::feed_forward::FeedForward;
use crate::core::layer_norm::LayerNorm;
use crate::core::linear::Linear;
use crate::core::moe::SharedSparseMoE;
use crate::core::tiny_gpt::TinyGPT;
use ndarray::Array2;

/// A transformer block with a MoE feed-forward network.
#[derive(Clone)]
pub struct MoEBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention<Linear>,
    pub ln2: LayerNorm,
    pub moe: SharedSparseMoE,
}

impl MoEBlock {
    fn from_dense(b: &crate::core::block::Block<Linear>, n_routed: usize, top_k: usize) -> Self {
        // routed experts start as copies of the original FFN; the shared
        // expert reuses it, preserving the trained mapping
        let routed: Vec<FeedForward<Linear>> =
            (0..n_routed).map(|_| b.ffn.clone()).collect();
        Self {
            ln1: b.ln1.clone(),
            attn: b.sa.clone(),
            ln2: b.ln2.clone(),
            moe: SharedSparseMoE::from_dense(&b.ffn, &routed, top_k),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let x = x + &self.attn.forward(&self.ln1.forward(x));
        &x + &self.moe.forward(&self.ln2.forward(&x))
    }
}

/// A TinyGPT whose FFNs are quantized MoEs.
#[derive(Clone)]
pub struct TinyGPTMoE {
    pub tok: Embedding,
    pub pos: Embedding,
    pub blocks: Vec<MoEBlock>,
    pub ln: LayerNorm,
    pub head: Linear,
    pub vocab: usize,
}

impl TinyGPTMoE {
    /// Upgrades a trained dense model into a quantized MoE transformer.
    ///
    /// # Arguments
    /// * `model` - Trained `TinyGPT<Linear>`
    /// * `n_routed` - Routed experts per block
    /// * `top_k` - Active routed experts per token
    ///
    /// # Returns
    /// * `Self` - MoE transformer
    pub fn from_dense(model: &TinyGPT<Linear>, n_routed: usize, top_k: usize) -> Self {
        Self {
            tok: model.tok.clone(),
            pos: model.pos.clone(),
            blocks: model
                .blocks
                .iter()
                .map(|b| MoEBlock::from_dense(b, n_routed, top_k))
                .collect(),
            ln: model.ln.clone(),
            head: model.head.clone(),
            vocab: model.vocab,
        }
    }

    /// Computes forward logits for a sequence.
    pub fn forward(&self, idx: &[usize]) -> Array2<f32> {
        let mut x = &self.tok.forward(idx)
            + &self.pos.forward(&(0..idx.len()).collect::<Vec<_>>());
        for b in &self.blocks {
            x = b.forward(&x);
        }
        self.head.forward(&self.ln.forward(&x))
    }

    /// Greedy generation, like `TinyGPT::generate_greedy_from`.
    pub fn generate_greedy_from(&self, prompt: &[usize], n: usize) -> Vec<usize> {
        let mut out = prompt.to_vec();
        for _ in 0..n {
            let ctx: Vec<usize> = out
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// from_dense builds a MoE transformer with quantized experts.
    #[test]
    fn test_from_dense_shapes() {
        let dense = TinyGPT::<Linear>::new(30);
        let moe = TinyGPTMoE::from_dense(&dense, 4, 2);
        assert_eq!(moe.blocks.len(), CFG.n_layers);
        assert_eq!(moe.blocks[0].moe.store.len(), 4);
        assert_eq!(moe.vocab, 30);
    }

    /// Forward produces finite logits of the right shape.
    #[test]
    fn test_forward_finite() {
        let dense = TinyGPT::<Linear>::new(30);
        let moe = TinyGPTMoE::from_dense(&dense, 4, 2);
        let out = moe.forward(&[0usize, 1, 2]);
        assert_eq!(out.shape(), &[3, 30]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Generation produces valid tokens.
    #[test]
    fn test_generate() {
        let dense = TinyGPT::<Linear>::new(20);
        let moe = TinyGPTMoE::from_dense(&dense, 4, 2);
        let out = moe.generate_greedy_from(&[0usize], 5);
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|&t| t < 20));
    }
}
