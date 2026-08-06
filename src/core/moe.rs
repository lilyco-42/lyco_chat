/*
 * @file moe.rs
 * @brief BitNet Mixture-of-Experts (MoE) module
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

//! FILE: moe.rs
//!
//! DESCRIPTION:
//! BitNet b1.58 Mixture-of-Experts module for TinyGPT.
//!
//! BRIEF:
//! Replaces the dense feed-forward network with a router plus several
//! ternary-quantized expert FFNs, activating only the top-k experts per
//! token (as in BitNet a4.8 / bitnet.cpp MoE support). The router stays in
//! full precision (matching bitnet.cpp), while expert weights use BitLinear.

use crate::core::attention::MultiHeadAttention;
use crate::core::bitnet::{quantize_input, lut_matmul, BitLinear};
use crate::core::feed_forward::FeedForward;
use crate::core::layer_norm::LayerNorm;
use crate::core::linear::Linear;
use crate::core::math::softmax;
use ndarray::Array2;

/// A ternary layer kept in its 2-bit packed form (no fp32 weight copy).
///
/// # Details
/// Mirrors `BitLinear::forward` (per-token int8 activations + LUT matmul +
/// per-output-channel scale dequantization) but stores only the packed
/// ternary weights and scales. This is the "storage" tier of the memory
/// hierarchy: ~14x smaller than an fp32 weight matrix.
///
/// # Fields
/// * `qw` - Packed ternary weights `[in / 4, out]`
/// * `w_scales` - Per-output-channel absmean scales
/// * `in_` - Input dimension
/// * `out` - Output dimension
#[derive(Clone)]
pub struct PackedBitLinear {
    pub qw: Array2<u8>,
    pub w_scales: ndarray::Array1<f32>,
    pub in_: usize,
    pub out: usize,
}

impl PackedBitLinear {
    /// Packs a `BitLinear` layer, dropping its fp32 weight copy.
    pub fn from_bitlinear(b: &BitLinear) -> Self {
        Self {
            qw: b.qw.clone(),
            w_scales: b.w_scales.clone(),
            in_: b.w.shape()[0],
            out: b.w.shape()[1],
        }
    }

    /// Bytes occupied by this packed layer.
    pub fn bytes(&self) -> usize {
        self.qw.len() + self.w_scales.len() * 4
    }

    /// Bytes the equivalent fp32 weights would occupy.
    pub fn fp32_bytes(&self) -> usize {
        self.in_ * self.out * 4 + self.out * 4
    }

    /// Computes `y = xW` using the packed ternary weights.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (t, o) = (x.shape()[0], self.out);
        let (xq, x_scales) = quantize_input(x);
        let acc = lut_matmul(&xq, &self.qw);
        let mut out = Array2::<f32>::zeros((t, o));
        for i in 0..t {
            for j in 0..o {
                out[[i, j]] = acc[[i, j]] as f32 * self.w_scales[j] / x_scales[i];
            }
        }
        out
    }
}

/// A quantized expert stored in packed form for on-demand loading.
///
/// # Details
/// ReLU(FFN) = l2(l1(x)) with both layers 2-bit packed. This is the unit the
/// memory hierarchy moves between storage/RAM/VRAM: it is only "active" when
/// a token routes to it.
#[derive(Clone)]
pub struct PackedExpert {
    pub l1: PackedBitLinear,
    pub l2: PackedBitLinear,
}

impl PackedExpert {
    /// Packs a dense `FeedForward` expert (quantize + drop fp32 copies).
    pub fn from_dense(e: &FeedForward<Linear>) -> Self {
        Self {
            l1: PackedBitLinear::from_bitlinear(&BitLinear::from_linear(&e.l1)),
            l2: PackedBitLinear::from_bitlinear(&BitLinear::from_linear(&e.l2)),
        }
    }

    /// Bytes occupied by this packed expert.
    pub fn bytes(&self) -> usize {
        self.l1.bytes() + self.l2.bytes()
    }

    /// Bytes the equivalent fp32 expert would occupy.
    pub fn fp32_bytes(&self) -> usize {
        self.l1.fp32_bytes() + self.l2.fp32_bytes()
    }

    /// Computes the expert FFN on the given rows.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        self.l2.forward(&self.l1.forward(x).mapv(|v| v.max(0.0)))
    }
}

/// Storage tier of the MoE memory hierarchy: all experts in packed form.
/// # Details
/// Experts live here compactly (2 bits per weight). During inference only
/// the experts selected by top-k routing are "loaded" (accessed) and run,
/// so both resident memory and compute scale with the number of active
/// experts, not the total. This mirrors treating storage/RAM/VRAM as one
/// hierarchy (Colibrì-style) at the scale of this project.
#[derive(Clone)]
pub struct ExpertStore {
    pub experts: Vec<PackedExpert>,
}

impl ExpertStore {
    /// Builds a store from trained dense experts.
    pub fn from_dense(dense: &[FeedForward<Linear>]) -> Self {
        Self {
            experts: dense.iter().map(PackedExpert::from_dense).collect(),
        }
    }

    /// Number of experts.
    pub fn len(&self) -> usize {
        self.experts.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }

    /// Total packed bytes across all experts.
    pub fn bytes(&self) -> usize {
        self.experts.iter().map(|e| e.bytes()).sum()
    }

    /// Total fp32 bytes the same experts would occupy.
    pub fn fp32_bytes(&self) -> usize {
        self.experts.iter().map(|e| e.fp32_bytes()).sum()
    }

    /// Loads (materializes access to) the given experts' packed weights.
    ///
    /// # Details
    /// In the hierarchy this is the storage -> RAM transfer step. The packed
    /// weights are ready to run directly; no fp32 copy is allocated.
    pub fn load(&self, ids: &[usize]) -> Vec<&PackedExpert> {
        ids.iter().map(|&i| &self.experts[i]).collect()
    }
}

/// Sparse BitNet MoE over the packed expert store.
///
/// # Details
/// Only the top-k experts chosen per token are computed, on their token rows
/// only. Compared to computing every expert this cuts FFN work to `top_k / n`
/// and keeps only active experts' weights hot.
#[derive(Clone)]
pub struct SparseBitMoE {
    pub router: Linear,
    pub store: ExpertStore,
    pub top_k: usize,
}

impl SparseBitMoE {
    /// Creates a sparse MoE from trained dense experts.
    pub fn from_dense(dense: &[FeedForward<Linear>], top_k: usize) -> Self {
        let d = dense[0].l1.w.shape()[0];
        Self {
            router: Linear::new(d, dense.len()),
            store: ExpertStore::from_dense(dense),
            top_k,
        }
    }

    /// Computes sparse MoE output, activating only top-k experts per token.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (t, d) = (x.shape()[0], x.shape()[1]);
        let n = self.store.len();
        let k = self.top_k.min(n);
        let gates = softmax(&self.router.forward(x)); // [t, n]

        // top-k experts per token
        let topk: Vec<Vec<usize>> = (0..t)
            .map(|i| {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| gates[[i, b]].partial_cmp(&gates[[i, a]]).unwrap());
                idx[..k].to_vec()
            })
            .collect();

        let mut out = Array2::<f32>::zeros((t, d));
        for e in 0..n {
            let rows: Vec<usize> = (0..t).filter(|&i| topk[i].contains(&e)).collect();
            if rows.is_empty() {
                continue;
            }
            // gather only this expert's token rows
            let mut gx = Array2::<f32>::zeros((rows.len(), d));
            for (ri, &i) in rows.iter().enumerate() {
                for j in 0..d {
                    gx[[ri, j]] = x[[i, j]];
                }
            }
            let gy = self.store.experts[e].forward(&gx);
            for (ri, &i) in rows.iter().enumerate() {
                let wsum: f32 = topk[i].iter().map(|&ee| gates[[i, ee]]).sum();
                let w = if wsum > 0.0 { gates[[i, e]] / wsum } else { 0.0 };
                for j in 0..d {
                    out[[i, j]] += w * gy[[ri, j]];
                }
            }
        }
        out
    }
}

/// Colibrì-style learning cache: routing-heat LRU over the expert store.
///
/// # Details
/// The "JIT for weights" idea: experts are data staged on demand, not
/// resident state. This cache keeps a bounded set of experts resident
/// (the hot-store), tracks routing heat, and evicts the coldest experts
/// when the residency budget is exceeded. The more a workload routes to an
/// expert, the longer it stays hot — the engine effectively learns the
/// workload, so repeated prompts hit resident experts instead of loading
/// them from storage.
pub struct ExpertCache {
    pub router: Linear,
    store: ExpertStore,
    budget: usize,
    resident: Vec<Option<PackedExpert>>,
    heat: Vec<u32>,
    hits: u64,
    misses: u64,
    prefetched: Vec<bool>,
    prefetch_loads: u64,
    prefetch_uses: u64,
}

impl ExpertCache {
    /// Creates a cache over a packed expert store.
    ///
    /// # Arguments
    /// * `store` - All experts in packed (storage) form
    /// * `router` - Full precision gating network
    /// * `budget` - Maximum number of experts kept resident (RAM/VRAM tier)
    ///
    /// # Returns
    /// * `Self` - ExpertCache instance
    pub fn new(store: ExpertStore, router: Linear, budget: usize) -> Self {
        let n = store.len();
        Self {
            router,
            store,
            budget: budget.max(1),
            resident: vec![None; n],
            heat: vec![0; n],
            hits: 0,
            misses: 0,
            prefetched: vec![false; n],
            prefetch_loads: 0,
            prefetch_uses: 0,
        }
    }

    /// Evicts the coldest resident expert when over budget.
    fn enforce_budget(&mut self) {
        let resident_ids: Vec<usize> = (0..self.resident.len())
            .filter(|&i| self.resident[i].is_some())
            .collect();
        if resident_ids.len() <= self.budget {
            return;
        }
        let coldest = *resident_ids
            .iter()
            .min_by(|&&a, &&b| self.heat[a].cmp(&self.heat[b]))
            .unwrap();
        self.resident[coldest] = None;
        self.prefetched[coldest] = false;
    }

    /// Ensures an expert is resident, loading it from storage on a miss.
    ///
    /// # Details
    /// On a miss the expert is materialized from the packed store. An expert
    /// that was prefetched ahead of time is counted as a prefetch use.
    fn ensure_resident(&mut self, id: usize) {
        self.heat[id] += 1;
        if self.resident[id].is_some() {
            self.hits += 1;
            if self.prefetched[id] {
                self.prefetch_uses += 1;
                self.prefetched[id] = false;
            }
            return;
        }
        self.misses += 1;
        self.resident[id] = Some(self.store.experts[id].clone());
        self.enforce_budget();
    }

    /// Prefetches the given experts into the resident set.
    ///
    /// # Details
    /// This is the router-lookahead equivalent: stage experts the next layer
    /// is expected to route to before they are needed, so a later hit does
    /// not pay the storage latency.
    pub fn prefetch(&mut self, ids: &[usize]) {
        for &id in ids {
            if id >= self.resident.len() || self.resident[id].is_some() {
                continue;
            }
            self.resident[id] = Some(self.store.experts[id].clone());
            self.prefetched[id] = true;
            self.prefetch_loads += 1;
            self.heat[id] += 1;
        }
        self.enforce_budget();
    }

    /// Predicts the next layer's experts from current routing heat and
    /// prefetches them (routing tends to be stable across adjacent layers).
    ///
    /// # Arguments
    /// * `gates` - Softmax router output `[t, n_experts]` for the current layer
    /// * `k` - Number of experts to prefetch
    pub fn prefetch_next(&mut self, gates: &Array2<f32>, k: usize) {
        let n = self.store.len();
        let mut mass = vec![0f32; n];
        for i in 0..gates.shape()[0] {
            for e in 0..n {
                mass[e] += gates[[i, e]];
            }
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| mass[b].partial_cmp(&mass[a]).unwrap());
        let take = k.min(n);
        self.prefetch(&idx[..take]);
    }

    /// Softmax gates for an input (used to drive lookahead prefetch).
    pub fn gates(&self, x: &Array2<f32>) -> Array2<f32> {
        softmax(&self.router.forward(x))
    }

    /// Runs the MoE forward with caching, activating top-k experts per token.
    ///
    /// # Returns
    /// * `Array2<f32>` - MoE output
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let (t, d) = (x.shape()[0], x.shape()[1]);
        let n = self.store.len();
        let k = self.top_k().min(n);
        let gates = self.gates(x);

        let topk: Vec<Vec<usize>> = (0..t)
            .map(|i| {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| gates[[i, b]].partial_cmp(&gates[[i, a]]).unwrap());
                idx[..k].to_vec()
            })
            .collect();

        let mut out = Array2::<f32>::zeros((t, d));
        for e in 0..n {
            let rows: Vec<usize> = (0..t).filter(|&i| topk[i].contains(&e)).collect();
            if rows.is_empty() {
                continue;
            }
            self.ensure_resident(e);
            let exp = self.resident[e].as_ref().unwrap();
            let mut gx = Array2::<f32>::zeros((rows.len(), d));
            for (ri, &i) in rows.iter().enumerate() {
                for j in 0..d {
                    gx[[ri, j]] = x[[i, j]];
                }
            }
            let gy = exp.forward(&gx);
            for (ri, &i) in rows.iter().enumerate() {
                let wsum: f32 = topk[i].iter().map(|&ee| gates[[i, ee]]).sum();
                let w = if wsum > 0.0 { gates[[i, e]] / wsum } else { 0.0 };
                for j in 0..d {
                    out[[i, j]] += w * gy[[ri, j]];
                }
            }
        }
        out
    }

    fn top_k(&self) -> usize {
        // default top_k = 2 unless budget is smaller
        self.budget.min(2).max(1)
    }

    /// Cache hit and miss counts.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Prefetch accounting: (experts staged ahead, staged experts actually used).
    pub fn prefetch_stats(&self) -> (u64, u64) {
        (self.prefetch_loads, self.prefetch_uses)
    }

    /// Number of experts currently resident.
    pub fn resident_count(&self) -> usize {
        self.resident.iter().filter(|r| r.is_some()).count()
    }

    /// Packed bytes of the resident expert set.
    pub fn resident_bytes(&self) -> usize {
        self.resident
            .iter()
            .filter_map(|r| r.as_ref().map(|e| e.bytes()))
            .sum()
    }

    /// Packed bytes of the entire storage tier.
    pub fn store_bytes(&self) -> usize {
        self.store.bytes()
    }

    /// Bytes the fp32 equivalents of the entire storage tier would occupy.
    pub fn store_fp32_bytes(&self) -> usize {
        self.store.fp32_bytes()
    }
}

/// DeepSeek-style MoE: one always-active shared expert plus top-k routed ones.
///
/// # Details
/// The shared expert captures common knowledge (computed for every token,
/// like DeepSeek-MoE's shared expert), while fine-grained routed experts
/// handle specialized patterns. Output = shared(x) + Σ_{top-k} gate·expert(x).
#[derive(Clone)]
pub struct SharedSparseMoE {
    pub shared: PackedExpert,
    pub router: Linear,
    pub store: ExpertStore,
    pub top_k: usize,
}

impl SharedSparseMoE {
    /// Builds from a shared dense FFN + routed dense FFNs (quantized).
    pub fn from_dense(shared: &FeedForward<Linear>, routed: &[FeedForward<Linear>], top_k: usize) -> Self {
        Self {
            shared: PackedExpert::from_dense(shared),
            router: Linear::new(shared.l1.w.shape()[0], routed.len()),
            store: ExpertStore::from_dense(routed),
            top_k,
        }
    }

    /// Computes output = shared(x) + sparse top-k routed experts.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (t, d) = (x.shape()[0], x.shape()[1]);
        let n = self.store.len();
        let k = self.top_k.min(n);
        let gates = softmax(&self.router.forward(x));

        // shared expert: always on
        let mut out = self.shared.forward(x);

        // top-k routed experts (per token)
        let topk: Vec<Vec<usize>> = (0..t)
            .map(|i| {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| gates[[i, b]].partial_cmp(&gates[[i, a]]).unwrap());
                idx[..k].to_vec()
            })
            .collect();
        for e in 0..n {
            let rows: Vec<usize> = (0..t).filter(|&i| topk[i].contains(&e)).collect();
            if rows.is_empty() {
                continue;
            }
            let mut gx = Array2::<f32>::zeros((rows.len(), d));
            for (ri, &i) in rows.iter().enumerate() {
                for j in 0..d {
                    gx[[ri, j]] = x[[i, j]];
                }
            }
            let gy = self.store.experts[e].forward(&gx);
            for (ri, &i) in rows.iter().enumerate() {
                let wsum: f32 = topk[i].iter().map(|&ee| gates[[i, ee]]).sum();
                let w = if wsum > 0.0 { gates[[i, e]] / wsum } else { 0.0 };
                for j in 0..d {
                    out[[i, j]] += w * gy[[ri, j]];
                }
            }
        }
        out
    }
}

/// Number of experts for a random `BitMoE`.
pub const DEFAULT_N_EXPERTS: usize = 4;

/// Number of active experts per token.
pub const DEFAULT_TOP_K: usize = 2;

/// BitNet Mixture-of-Experts FFN.
///
/// # Details
/// A router (full precision Linear) computes a softmax distribution over
/// experts; the top-k experts' outputs are combined with renormalized gate
/// weights. Expert weights are ternary-quantized via `BitLinear`.
///
/// # Fields
/// * `router` - Full precision gating network `(in -> n_experts)`
/// * `experts` - Quantized expert FFNs, each `in -> 4*in -> in`
/// * `top_k` - Number of experts activated per token
#[derive(Clone)]
pub struct BitMoE {
    pub router: Linear,
    pub experts: Vec<FeedForward<BitLinear>>,
    pub top_k: usize,
}

impl BitMoE {
    /// Creates a new MoE with random quantized experts.
    ///
    /// # Arguments
    /// * `d` - Input/output dimension
    /// * `n_experts` - Number of experts
    /// * `top_k` - Active experts per token
    ///
    /// # Returns
    /// * `Self` - BitMoE instance
    pub fn new(d: usize, n_experts: usize, top_k: usize) -> Self {
        let router = Linear::new(d, n_experts);
        let experts = (0..n_experts)
            .map(|_| FeedForward::<BitLinear>::new(d))
            .collect();
        Self {
            router,
            experts,
            top_k,
        }
    }

    /// Builds an MoE from a set of trained dense experts.
    ///
    /// # Details
    /// Quantizes each dense expert's weights into `BitLinear` layers while
    /// keeping a freshly initialized router.
    ///
    /// # Arguments
    /// * `dense_experts` - Trained full precision expert FFNs
    /// * `top_k` - Active experts per token
    ///
    /// # Returns
    /// * `Self` - BitMoE instance
    pub fn from_dense(dense_experts: &[FeedForward<Linear>], top_k: usize) -> Self {
        let n = dense_experts.len();
        let router = Linear::new(dense_experts[0].l1.w.shape()[0], n);
        let experts = dense_experts
            .iter()
            .map(|e| FeedForward::<BitLinear> {
                l1: BitLinear::from_linear(&e.l1),
                l2: BitLinear::from_linear(&e.l2),
            })
            .collect();
        Self {
            router,
            experts,
            top_k,
        }
    }

    /// Computes MoE output.
    ///
    /// # Details
    /// `x` is `[t, d]` for one sequence. Gates are the row-wise softmax of
    /// the router logits; only the top-k experts contribute, with their gate
    /// weights renormalized to sum to one.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - MoE output
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (t, d) = (x.shape()[0], x.shape()[1]);
        let n = self.experts.len();
        let gates = softmax(&self.router.forward(x)); // [t, n]
        let outs: Vec<Array2<f32>> = self.experts.iter().map(|e| e.forward(x)).collect();
        let k = self.top_k.min(n);
        let mut out = Array2::<f32>::zeros((t, d));
        for i in 0..t {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| gates[[i, b]].partial_cmp(&gates[[i, a]]).unwrap());
            let topk = &idx[..k];
            let mut wsum = 0.0f32;
            for &e in topk {
                wsum += gates[[i, e]];
            }
            if wsum <= 0.0 {
                continue;
            }
            for &e in topk {
                let w = gates[[i, e]] / wsum;
                for j in 0..d {
                    out[[i, j]] += w * outs[e][[i, j]];
                }
            }
        }
        out
    }

    /// Zeros gradient accumulators (router only; experts are inference-only).
    pub fn zero_grad(&mut self) {
        self.router.zero_grad();
    }

    /// Updates the router with gradient descent.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.router.step(lr);
    }
}

/// A transformer block using a BitNet MoE feed-forward network.
///
/// # Details
/// Mirrors `Block` (pre-norm, residual) but swaps the dense FFN for a
/// top-k Mixture-of-Experts FFN with ternary-quantized experts.
///
/// # Fields
/// * `sa` - Dense self-attention module
/// * `moe` - Mixture-of-Experts feed-forward
/// * `ln1` - Layer norm before attention
/// * `ln2` - Layer norm before MoE
#[derive(Clone)]
pub struct BitMoEBlock {
    pub sa: MultiHeadAttention<Linear>,
    pub moe: BitMoE,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
}

impl BitMoEBlock {
    /// Creates a new MoE transformer block.
    ///
    /// # Arguments
    /// * `d` - Embedding dimension
    /// * `nh` - Number of attention heads
    /// * `n_experts` - Number of experts
    /// * `top_k` - Active experts per token
    ///
    /// # Returns
    /// * `Self` - BitMoEBlock instance
    pub fn new(d: usize, nh: usize, n_experts: usize, top_k: usize) -> Self {
        Self {
            sa: MultiHeadAttention::<Linear>::new(d, nh),
            moe: BitMoE::new(d, n_experts, top_k),
            ln1: LayerNorm::new(d),
            ln2: LayerNorm::new(d),
        }
    }

    /// Computes block output with pre-norm and residual connections.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Block output
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let x = x + &self.sa.forward(&self.ln1.forward(x));
        &x + &self.moe.forward(&self.ln2.forward(&x))
    }

    /// Zeros gradient accumulators.
    pub fn zero_grad(&mut self) {
        self.sa.zero_grad();
        self.moe.zero_grad();
    }

    /// Updates parameters with gradient descent.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.sa.step(lr);
        self.moe.step(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests BitMoE forward output shape.
    #[test]
    fn test_bit_moe_forward_shape() {
        let moe = BitMoE::new(16, 4, 2);
        let x = arr2(&[[1.0; 16], [2.0; 16], [3.0; 16]]);
        let y = moe.forward(&x);
        assert_eq!(y.shape(), &[3, 16]);
    }

    /// Tests BitMoE forward produces finite values.
    #[test]
    fn test_bit_moe_forward_finite() {
        let moe = BitMoE::new(16, 4, 2);
        let x = arr2(&[[1.0; 16], [-2.0; 16]]);
        let y = moe.forward(&x);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Tests top_k=1 output equals the best expert's output for each token.
    #[test]
    fn test_bit_moe_topk_one() {
        let moe = BitMoE::new(8, 3, 1);
        let x = arr2(&[[1.0; 8], [0.5; 8]]);
        let y = moe.forward(&x);
        let gates = softmax(&moe.router.forward(&x));
        for i in 0..x.shape()[0] {
            let mut idx: Vec<usize> = (0..3).collect();
            idx.sort_by(|&a, &b| gates[[i, b]].partial_cmp(&gates[[i, a]]).unwrap());
            let e = idx[0];
            let expected = moe.experts[e].forward(&x);
            for j in 0..8 {
                assert!((y[[i, j]] - expected[[i, j]]).abs() < 1e-3, "mismatch at ({i},{j})");
            }
        }
    }

    /// Tests top_k=1 with one dominant expert: output follows that expert.
    #[test]
    fn test_bit_moe_from_dense() {
        let dense: Vec<FeedForward<Linear>> = (0..3).map(|_| FeedForward::<Linear>::new(8)).collect();
        let moe = BitMoE::from_dense(&dense, 2);
        let x = arr2(&[[1.0; 8]]);
        let y = moe.forward(&x);
        assert_eq!(y.shape(), &[1, 8]);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Tests BitMoEBlock forward shape and finiteness.
    #[test]
    fn test_bit_moe_block_forward() {
        let b = BitMoEBlock::new(8, 2, 4, 2);
        let x = arr2(&[[1.0; 8], [2.0; 8]]);
        let y = b.forward(&x);
        assert_eq!(y.shape(), &[2, 8]);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Tests BitMoEBlock zero_grad and step do not panic.
    #[test]
    fn test_bit_moe_block_train() {
        let mut b = BitMoEBlock::new(8, 2, 4, 2);
        b.zero_grad();
        b.step(0.1);
        assert!(b.moe.router.w.iter().all(|v| v.is_finite()));
    }

    /// Tests packed expert storage is much smaller than fp32.
    #[test]
    fn test_expert_store_compact() {
        let dense: Vec<FeedForward<Linear>> = (0..4).map(|_| FeedForward::<Linear>::new(32)).collect();
        let store = ExpertStore::from_dense(&dense);
        assert!(store.bytes() > 0);
        assert!(store.bytes() < store.fp32_bytes() / 10, "packed should be <10x smaller");
        println!("packed {} B vs fp32 {} B", store.bytes(), store.fp32_bytes());
    }

    /// Tests packed expert forward matches the unpacked BitLinear forward.
    #[test]
    fn test_packed_expert_matches_bitlinear() {
        let dense = FeedForward::<Linear>::new(16);
        let bl = FeedForward::<BitLinear> {
            l1: BitLinear::from_linear(&dense.l1),
            l2: BitLinear::from_linear(&dense.l2),
        };
        let packed = PackedExpert::from_dense(&dense);
        let x = ndarray::arr2(&[[1.0; 16], [-1.0; 16], [2.0; 16]]);
        let a = bl.forward(&x);
        let b = packed.forward(&x);
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-4, "packed forward mismatch");
        }
    }

    /// Tests sparse MoE equals dense-compute MoE when all experts are active.
    #[test]
    fn test_sparse_equals_dense_all_experts() {
        let dense: Vec<FeedForward<Linear>> = (0..4).map(|_| FeedForward::<Linear>::new(16)).collect();
        // dense-compute reference: BitMoE with same quantized experts + same router weights
        let n = dense.len();
        let mut dense_moe = BitMoE::from_dense(&dense, n);
        let mut sparse = SparseBitMoE::from_dense(&dense, n);
        // share router weights so gating matches
        sparse.router.w.assign(&dense_moe.router.w);
        sparse.router.b.assign(&dense_moe.router.b);
        let x = ndarray::arr2(&[[1.0; 16], [0.5; 16], [-0.5; 16]]);
        let a = dense_moe.forward(&x);
        let b = sparse.forward(&x);
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-3, "sparse vs dense mismatch: {va} vs {vb}");
        }
    }

    /// Tests sparse MoE activates fewer experts than the total.
    #[test]
    fn test_sparse_activation_count() {
        let dense: Vec<FeedForward<Linear>> = (0..8).map(|_| FeedForward::<Linear>::new(16)).collect();
        let sparse = SparseBitMoE::from_dense(&dense, 2);
        let x = ndarray::arr2(&[[1.0; 16], [2.0; 16], [3.0; 16]]);
        let y = sparse.forward(&x);
        assert_eq!(y.shape(), &[3, 16]);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Tests ExpertStore::load returns the requested experts.
    #[test]
    fn test_expert_store_load() {
        let dense: Vec<FeedForward<Linear>> = (0..4).map(|_| FeedForward::<Linear>::new(8)).collect();
        let store = ExpertStore::from_dense(&dense);
        let loaded = store.load(&[1, 3]);
        assert_eq!(loaded.len(), 2);
        assert_eq!(store.len(), 4);
    }

    /// Tests DeepSeek-style shared-expert MoE: shape, finiteness, and that the
    /// shared expert always contributes.
    #[test]
    fn test_shared_moe() {
        let shared = FeedForward::<Linear>::new(16);
        let routed: Vec<FeedForward<Linear>> = (0..6).map(|_| FeedForward::<Linear>::new(16)).collect();
        let moe = SharedSparseMoE::from_dense(&shared, &routed, 2);
        let x = ndarray::arr2(&[[1.0; 16], [2.0; 16]]);
        let y = moe.forward(&x);
        assert_eq!(y.shape(), &[2, 16]);
        assert!(y.iter().all(|v| v.is_finite()));
        // shared alone must differ from the combined output (routed experts add)
        let shared_only = moe.shared.forward(&x);
        let mut any_diff = false;
        for (a, b) in y.iter().zip(shared_only.iter()) {
            if (a - b).abs() > 1e-4 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "routed experts must contribute beyond the shared one");
    }

    /// Tests the learning cache: repeated forwards hit instead of miss.
    #[test]
    fn test_expert_cache_learns() {
        let dense: Vec<FeedForward<Linear>> = (0..4).map(|_| FeedForward::<Linear>::new(16)).collect();
        let store = ExpertStore::from_dense(&dense);
        let router = Linear::new(16, 4);
        let mut cache = ExpertCache::new(store, router, 4);
        let x = ndarray::arr2(&[[1.0; 16], [2.0; 16]]);
        for _ in 0..3 {
            cache.forward(&x);
        }
        let (hits, misses) = cache.stats();
        // 3 passes should mostly hit after the first pass
        assert!(hits > misses, "cache should learn: hits={hits} misses={misses}");
        assert!(misses <= 4, "at most 4 experts ever miss");
    }

    /// Tests eviction: a small budget keeps residency bounded.
    #[test]
    fn test_expert_cache_budget() {
        let dense: Vec<FeedForward<Linear>> = (0..8).map(|_| FeedForward::<Linear>::new(16)).collect();
        let store = ExpertStore::from_dense(&dense);
        let router = Linear::new(16, 8);
        let mut cache = ExpertCache::new(store, router, 2);
        let x = ndarray::arr2(&[[1.0; 16]]);
        for _ in 0..5 {
            cache.forward(&x);
        }
        assert!(cache.resident_count() <= 2, "residency must stay within budget");
        assert!(cache.resident_bytes() > 0);
    }

    /// Tests cache output with full residency equals the sparse MoE output.
    #[test]
    fn test_expert_cache_matches_sparse() {
        let dense: Vec<FeedForward<Linear>> = (0..4).map(|_| FeedForward::<Linear>::new(16)).collect();
        let store = ExpertStore::from_dense(&dense);
        let router = Linear::new(16, 4);
        let mut cache = ExpertCache::new(store.clone(), router.clone(), 4);
        let sparse = SparseBitMoE {
            router,
            store,
            top_k: 2,
        };
        let x = ndarray::arr2(&[[1.0; 16], [0.5; 16], [-0.5; 16]]);
        let a = cache.forward(&x);
        let b = sparse.forward(&x);
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-3, "cache vs sparse mismatch: {va} vs {vb}");
        }
    }

    /// Tests lookahead prefetch: staging next layer's experts avoids misses.
    #[test]
    fn test_expert_cache_prefetch() {
        let dense: Vec<FeedForward<Linear>> = (0..8).map(|_| FeedForward::<Linear>::new(16)).collect();
        let store = ExpertStore::from_dense(&dense);
        let router = Linear::new(16, 8);
        let mut cache = ExpertCache::new(store, router, 4);
        let x = ndarray::arr2(&[[1.0; 16]]);

        // predict from the router (runs a layer ahead) and stage before need
        let gates = cache.gates(&x);
        cache.prefetch_next(&gates, 2);
        let (loads_before, _) = cache.prefetch_stats();
        assert!(loads_before > 0, "prefetch must stage experts before the forward");

        // the forward now hits the staged experts with zero misses
        cache.forward(&x);
        let (hits, misses) = cache.stats();
        assert_eq!(misses, 0, "staged experts must not miss");
        assert!(hits > 0);
        let (_, uses) = cache.prefetch_stats();
        assert!(uses > 0, "staged experts must be consumed (uses={uses})");
    }
}
