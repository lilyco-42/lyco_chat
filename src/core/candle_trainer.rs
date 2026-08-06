/*
 * @file candle_trainer.rs
 * @brief Multi-backend (CPU/CUDA) training via candle
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

//! FILE: candle_trainer.rs
//!
//! DESCRIPTION:
//! Multi-backend (CPU / CUDA) training for TinyGPT using candle.
//!
//! BRIEF:
//! Mirrors the ndarray transformer architecture exactly so trained weights
//! can be transferred to `TinyGPT<Linear>` for CPU inference and BitNet
//! quantization. Device is chosen at runtime (`Device::Cuda` when available,
//! otherwise `Device::Cpu`), giving a single code path for both backends.

use crate::core::attention::MultiHeadAttention;
use crate::core::block::Block;
use crate::core::config::CFG;
use crate::core::data::DataLoader;
use crate::core::embedding::Embedding;
use crate::core::feed_forward::FeedForward;
use crate::core::head::Head;
use crate::core::layer_norm::LayerNorm;
use crate::core::linear::Linear;
use crate::core::tiny_gpt::TinyGPT;
use candle_core::{Device, IndexOp, Tensor, Var};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::softmax;
use ndarray::{Array1, Array2};
use rand::rngs::ThreadRng;

/// A linear layer with weights shaped `[in, out]`, matching the ndarray core.
struct CLinear {
    w: Var,
    b: Var,
}

impl CLinear {
    fn new(i: usize, o: usize, dev: &Device) -> anyhow::Result<Self> {
        let w = Var::randn::<_, f32>(0.0, 0.02, (i, o), dev)?;
        let b = Var::zeros((o,), candle_core::DType::F32, dev)?;
        Ok(Self { w, b })
    }

    /// `x` is `[b, t, in]`; returns `[b, t, out]`.
    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let x = x.reshape((b * t, self.w.dims()[0]))?;
        let y = x.matmul(&self.w)?.broadcast_add(&self.b)?;
        Ok(y.reshape((b, t, self.w.dims()[1]))?)
    }
}

/// Layer norm matching the ndarray `layer_norm` exactly (eps = 1e-5).
struct CLayerNorm {
    g: Var,
    b: Var,
    dim: usize,
}

impl CLayerNorm {
    fn new(dim: usize, dev: &Device) -> anyhow::Result<Self> {
        let g = Var::ones((dim,), candle_core::DType::F32, dev)?;
        let b = Var::zeros((dim,), candle_core::DType::F32, dev)?;
        Ok(Self { g, b, dim })
    }

    /// Normalizes the last dimension: (x - mean) / sqrt(var + 1e-5) * g + b.
    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x2 = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let var = x2.sub(&mean.sqr()?)?;
        let centered = x.sub(&mean.broadcast_as(x.dims())?)?;
        let denom = var.affine(1.0, 1e-5)?.sqrt()?;
        let norm = centered.div(&denom.broadcast_as(x.dims())?)?;
        Ok(norm.broadcast_mul(&self.g)?.broadcast_add(&self.b)?)
    }
}

/// Per-head attention with causal masking, matching `Head::forward`.
fn attn_head(q: &Tensor, k: &Tensor, v: &Tensor, hs: usize, dev: &Device) -> anyhow::Result<Tensor> {
    let scale = 1.0 / (hs as f64).sqrt();
    let t = q.dim(1)?;
    let mut scores = q.matmul(&k.t()?)?.affine(scale, 0.)?;
    let mut mask = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            mask[i * t + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(mask, (t, t), dev)?.unsqueeze(0)?;
    scores = scores.broadcast_add(&mask)?;
    let sm = softmax(&scores, 2)?;
    Ok(sm.matmul(v)?)
}

/// One transformer block, mirroring the ndarray `Block`.
struct CBlock {
    ln1: CLayerNorm,
    q: Vec<CLinear>,
    k: Vec<CLinear>,
    v: Vec<CLinear>,
    proj: CLinear,
    ln2: CLayerNorm,
    l1: CLinear,
    l2: CLinear,
    d: usize,
    nh: usize,
}

impl CBlock {
    fn new(d: usize, nh: usize, dev: &Device) -> anyhow::Result<Self> {
        let hs = d / nh;
        let mut q = Vec::with_capacity(nh);
        let mut k = Vec::with_capacity(nh);
        let mut v = Vec::with_capacity(nh);
        for _ in 0..nh {
            q.push(CLinear::new(d, hs, dev)?);
            k.push(CLinear::new(d, hs, dev)?);
            v.push(CLinear::new(d, hs, dev)?);
        }
        Ok(Self {
            ln1: CLayerNorm::new(d, dev)?,
            q,
            k,
            v,
            proj: CLinear::new(d, d, dev)?,
            ln2: CLayerNorm::new(d, dev)?,
            l1: CLinear::new(d, 4 * d, dev)?,
            l2: CLinear::new(4 * d, d, dev)?,
            d,
            nh,
        })
    }

    fn attention(&self, x: &Tensor, dev: &Device) -> anyhow::Result<Tensor> {
        let hs = self.d / self.nh;
        let mut heads = Vec::with_capacity(self.nh);
        for i in 0..self.nh {
            let q = self.q[i].forward(x)?;
            let k = self.k[i].forward(x)?;
            let v = self.v[i].forward(x)?;
            heads.push(attn_head(&q, &k, &v, hs, dev)?);
        }
        let h = Tensor::cat(&heads, 2)?;
        self.proj.forward(&h)
    }

    fn forward(&self, x: &Tensor, dev: &Device) -> anyhow::Result<Tensor> {
        let x = x.add(&self.attention(&self.ln1.forward(x)?, dev)?)?;
        let y = self.ln2.forward(&x)?;
        let y = self.l1.forward(&y)?.relu()?;
        let f = self.l2.forward(&y)?;
        Ok(x.add(&f)?)
    }
}

/// The full model, mirroring `TinyGPT`.
pub struct CandleGPT {
    tok: Var,
    pos: Var,
    blocks: Vec<CBlock>,
    ln: CLayerNorm,
    head: CLinear,
    d: usize,
    nh: usize,
    vocab: usize,
    block_size: usize,
}

impl CandleGPT {
    fn new(vocab: usize, dev: &Device) -> anyhow::Result<Self> {
        let d = CFG.embed_dim;
        let nh = CFG.n_heads;
        let tok = Var::randn::<_, f32>(0.0, 0.02, (vocab, d), dev)?;
        let pos = Var::randn::<_, f32>(0.0, 0.02, (CFG.block_size, d), dev)?;
        let blocks = (0..CFG.n_layers)
            .map(|_| CBlock::new(d, nh, dev))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            tok,
            pos,
            blocks,
            ln: CLayerNorm::new(d, dev)?,
            head: CLinear::new(d, vocab, dev)?,
            d,
            nh,
            vocab,
            block_size: CFG.block_size,
        })
    }

    fn embed(w: &Var, ids: &Tensor, hidden: usize) -> anyhow::Result<Tensor> {
        let mut final_dims = ids.dims().to_vec();
        final_dims.push(hidden);
        let flat = ids.flatten_all()?;
        let values = w.as_tensor().index_select(&flat, 0)?;
        Ok(values.reshape(final_dims)?)
    }

    fn forward(&self, tok_ids: &Tensor, dev: &Device) -> anyhow::Result<Tensor> {
        let (_, t) = tok_ids.dims2()?;
        let x = Self::embed(&self.tok, tok_ids, self.d)?;
        let pos_ids = Tensor::arange(0u32, t as u32, dev)?;
        let pos = Self::embed(&self.pos, &pos_ids, self.d)?.unsqueeze(0)?;
        let mut x = x.broadcast_add(&pos)?;
        for block in &self.blocks {
            x = block.forward(&x, dev)?;
        }
        let x = self.ln.forward(&x)?;
        self.head.forward(&x)
    }

    fn params(&self) -> Vec<&Var> {
        let mut out = vec![&self.tok, &self.pos];
        for b in &self.blocks {
            out.push(&b.ln1.g);
            out.push(&b.ln1.b);
            for l in &b.q {
                out.push(&l.w);
                out.push(&l.b);
            }
            for l in &b.k {
                out.push(&l.w);
                out.push(&l.b);
            }
            for l in &b.v {
                out.push(&l.w);
                out.push(&l.b);
            }
            out.push(&b.proj.w);
            out.push(&b.proj.b);
            out.push(&b.ln2.g);
            out.push(&b.ln2.b);
            out.push(&b.l1.w);
            out.push(&b.l1.b);
            out.push(&b.l2.w);
            out.push(&b.l2.b);
        }
        out.push(&self.ln.g);
        out.push(&self.ln.b);
        out.push(&self.head.w);
        out.push(&self.head.b);
        out
    }

    fn step(&self, loss: &Tensor, lr: f64) -> anyhow::Result<()> {
        let grads = loss.backward()?;
        for var in self.params() {
            if let Some(g) = grads.get(var.as_tensor()) {
                let delta = g.affine(lr, 0.)?;
                let new_val = var.as_tensor().sub(&delta)?;
                var.set(&new_val)?;
            }
        }
        Ok(())
    }

    /// Transfers weights into the ndarray `TinyGPT<Linear>` for CPU inference.
    fn to_ndarray(&self) -> anyhow::Result<TinyGPT<Linear>> {
        fn arr2(v: &Var) -> anyhow::Result<Array2<f32>> {
            let rows = v.as_tensor().to_vec2::<f32>()?;
            let c = if rows.is_empty() { 0 } else { rows[0].len() };
            Ok(Array2::from_shape_vec((rows.len(), c), rows.into_iter().flatten().collect())?)
        }
        fn arr1(v: &Var) -> anyhow::Result<Array1<f32>> {
            Ok(Array1::from_vec(v.as_tensor().to_vec1::<f32>()?))
        }
        fn lin(l: &CLinear) -> anyhow::Result<Linear> {
            let (r, c) = (l.w.dims()[0], l.w.dims()[1]);
            Ok(Linear {
                w: arr2(&l.w)?,
                b: arr1(&l.b)?,
                dw: Array2::zeros((r, c)),
                db: Array1::zeros(c),
            })
        }

        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let hs = self.d / self.nh;
            let mut heads = Vec::with_capacity(self.nh);
            for i in 0..self.nh {
                heads.push(Head {
                    key: lin(&b.k[i])?,
                    query: lin(&b.q[i])?,
                    value: lin(&b.v[i])?,
                    hs,
                });
            }
            blocks.push(Block {
                sa: MultiHeadAttention {
                    heads,
                    proj: lin(&b.proj)?,
                },
                ffn: FeedForward {
                    l1: lin(&b.l1)?,
                    l2: lin(&b.l2)?,
                },
                ln1: LayerNorm { g: arr1(&b.ln1.g)?, b: arr1(&b.ln1.b)? },
                ln2: LayerNorm { g: arr1(&b.ln2.g)?, b: arr1(&b.ln2.b)? },
            });
        }
        Ok(TinyGPT {
            tok: Embedding { w: arr2(&self.tok)?, dw: Array2::zeros((self.vocab, self.d)) },
            pos: Embedding { w: arr2(&self.pos)?, dw: Array2::zeros((self.block_size, self.d)) },
            blocks,
            ln: LayerNorm { g: arr1(&self.ln.g)?, b: arr1(&self.ln.b)? },
            head: lin(&self.head)?,
            vocab: self.vocab,
        })
    }

    /// Loads a trained ndarray model onto the given device for GPU/CPU inference.
    ///
    /// # Details
    /// The reverse of `to_ndarray`: every weight becomes a candle `Var` on the
    /// chosen backend (CUDA or CPU), so inference runs where the client asks.
    ///
    /// # Arguments
    /// * `model` - Trained `TinyGPT<Linear>`
    /// * `device` - Target device (CUDA or CPU)
    ///
    /// # Returns
    /// * `anyhow::Result<CandleGPT>` - Model on the target device
    pub fn from_ndarray(model: &TinyGPT<Linear>, device: &Device) -> anyhow::Result<CandleGPT> {
        fn var(a: &Array2<f32>, dev: &Device) -> anyhow::Result<Var> {
            let (r, c) = a.dim();
            Ok(Var::from_vec(a.iter().copied().collect::<Vec<f32>>(), (r, c), dev)?)
        }
        fn var1(a: &Array1<f32>, dev: &Device) -> anyhow::Result<Var> {
            Ok(Var::from_vec(a.iter().copied().collect::<Vec<f32>>(), a.len(), dev)?)
        }
        fn lin(l: &Linear, dev: &Device) -> anyhow::Result<CLinear> {
            Ok(CLinear {
                w: var(&l.w, dev)?,
                b: var1(&l.b, dev)?,
            })
        }
        let d = CFG.embed_dim;
        let nh = CFG.n_heads;
        let mut blocks = Vec::with_capacity(model.blocks.len());
        for b in &model.blocks {
            let hs = d / nh;
            let mut q = Vec::with_capacity(nh);
            let mut k = Vec::with_capacity(nh);
            let mut v = Vec::with_capacity(nh);
            for h in &b.sa.heads {
                q.push(lin(&h.query, device)?);
                k.push(lin(&h.key, device)?);
                v.push(lin(&h.value, device)?);
            }
            blocks.push(CBlock {
                ln1: CLayerNorm {
                    g: var1(&b.ln1.g, device)?,
                    b: var1(&b.ln1.b, device)?,
                    dim: d,
                },
                q,
                k,
                v,
                proj: lin(&b.sa.proj, device)?,
                ln2: CLayerNorm {
                    g: var1(&b.ln2.g, device)?,
                    b: var1(&b.ln2.b, device)?,
                    dim: d,
                },
                l1: lin(&b.ffn.l1, device)?,
                l2: lin(&b.ffn.l2, device)?,
                d,
                nh,
            });
        }
        Ok(CandleGPT {
            tok: var(&model.tok.w, device)?,
            pos: var(&model.pos.w, device)?,
            blocks,
            ln: CLayerNorm {
                g: var1(&model.ln.g, device)?,
                b: var1(&model.ln.b, device)?,
                dim: d,
            },
            head: lin(&model.head, device)?,
            d,
            nh,
            vocab: model.vocab,
            block_size: CFG.block_size,
        })
    }

    /// Greedily generates tokens on the model's device (GPU/CPU inference).
    ///
    /// # Arguments
    /// * `prompt` - Token ids to condition on
    /// * `n` - Number of tokens to generate
    /// * `device` - Device the model lives on
    ///
    /// # Returns
    /// * `anyhow::Result<Vec<usize>>` - Prompt followed by generated tokens
    pub fn generate_greedy_from(
        &self,
        prompt: &[usize],
        n: usize,
        device: &Device,
    ) -> anyhow::Result<Vec<usize>> {
        let mut out = prompt.to_vec();
        for _ in 0..n {
            let ctx: Vec<usize> = out
                .iter()
                .rev()
                .take(self.block_size)
                .rev()
                .copied()
                .collect();
            let t = ctx.len();
            let ids = Tensor::from_vec(
                ctx.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                (1, t),
                device,
            )?;
            let logits = self.forward(&ids, device)?; // [1, t, vocab]
            let last = logits.i((0, t - 1))?; // [vocab]
            let best = last.argmax(0)?;
            out.push(best.to_scalar::<u32>()? as usize);
        }
        Ok(out)
    }
}

fn to_tensor_u32(a: &ndarray::Array2<usize>, dev: &Device) -> anyhow::Result<Tensor> {
    let (r, c) = a.dim();
    let v: Vec<u32> = a.iter().map(|&x| x as u32).collect();
    Ok(Tensor::from_vec(v, (r, c), dev)?)
}

/// Trains on the given backend and returns a CPU-inference `TinyGPT<Linear>`.
///
/// # Arguments
/// * `vocab_size` - Vocabulary size
/// * `data` - Tokenized training data
/// * `device` - CPU or CUDA device (multi-backend)
/// * `steps` - Number of training steps (0 uses the configured epoch count)
/// * `rng` - Random number generator
///
/// # Returns
/// * `anyhow::Result<TinyGPT<Linear>>` - Trained model ready for ndarray inference
pub fn train_candle(
    vocab_size: usize,
    data: &[usize],
    device: Device,
    steps: usize,
    rng: &mut ThreadRng,
) -> anyhow::Result<TinyGPT<Linear>> {
    let loader = DataLoader::new(data);
    let model = CandleGPT::new(vocab_size, &device)?;
    let lr = CFG.lr as f64;
    let steps = if steps == 0 { CFG.epochs } else { steps };
    for step in 0..steps {
        let (xb, yb) = loader.get_batch(rng);
        let xt = to_tensor_u32(&xb, &device)?;
        let yt = to_tensor_u32(&yb, &device)?;
        let logits = model.forward(&xt, &device)?;
        let (b, t, v) = logits.dims3()?;
        let logits = logits.reshape((b * t, v))?;
        let targets = yt.flatten_all()?;
        let loss = cross_entropy(&logits, &targets)?;
        model.step(&loss, lr)?;
        if step % 300 == 0 {
            println!(
                "candle {} step {step}, loss={:.4}",
                if matches!(device, Device::Cuda(_)) { "cuda" } else { "cpu" },
                loss.to_scalar::<f32>()?
            );
        }
    }
    model.to_ndarray()
}

/// Picks the training device: CUDA when compiled in and available, else CPU.
pub fn pick_device() -> anyhow::Result<Device> {    let dev = Device::cuda_if_available(0)?;
    match &dev {
        Device::Cuda(_) => println!("candle device: cuda(0)"),
        Device::Cpu => println!("candle device: cpu (cuda feature not enabled)"),
        _ => println!("candle device: cpu"),
    }
    Ok(dev)
}

/// Whether a CUDA device is compiled in and usable.
pub fn cuda_available() -> bool {
    matches!(pick_device().ok(), Some(Device::Cuda(_)))
}

/// A minimal trainable MoE router (DeepSeek-style): experts are frozen,
/// only the gating network learns which expert handles which input.
///
/// # Details
/// `n_experts` fixed single-layer experts (frozen `Tensor`s); a `Var` router
/// produces a softmax gate, and the output is the gated mixture. Training
/// runs the router's `Var` through candle autograd so only the gate updates.
pub struct CandleMoERouter {
    router: Var, // [in, n_experts]
    experts: Vec<[Var; 2]>, // per expert: [in, hidden], [hidden, out]
    out: usize,
}

impl CandleMoERouter {
    /// Creates a router with random experts on `dev`.
    pub fn new(in_: usize, hidden: usize, out: usize, n: usize, dev: &Device) -> anyhow::Result<Self> {
        let router = Var::randn::<_, f32>(0.0, 0.2, (in_, n), dev)?;
        let mut experts = Vec::with_capacity(n);
        for _ in 0..n {
            let w1 = Var::randn::<_, f32>(0.0, 0.5, (in_, hidden), dev)?;
            let w2 = Var::randn::<_, f32>(0.0, 0.5, (hidden, out), dev)?;
            experts.push([w1, w2]);
        }
        Ok(Self { router, experts, out })
    }

    /// Computes the gated mixture output: x@router -> softmax -> weighted
    /// sum of expert outputs.
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let n = self.experts.len();
        let logits = x.matmul(&self.router)?.affine(1.0, 0.)?; // [b, n]
        let gates = softmax(&logits, 1)?; // [b, n]
        let mut out: Option<Tensor> = None;
        for (i, [w1, w2]) in self.experts.iter().enumerate() {
            let h = x.matmul(w1.as_tensor())?.relu()?;
            let e = h.matmul(w2.as_tensor())?; // [b, out]
            let g = gates.narrow(1, i, 1)?; // [b, 1]
            let contrib = e.broadcast_mul(&g)?;
            out = Some(match out {
                Some(o) => o.add(&contrib)?,
                None => contrib,
            });
        }
        Ok(out.unwrap_or(Tensor::zeros((x.dim(0)?, self.out), candle_core::DType::F32, x.device())?))
    }

    /// One training step: MSE between gated output and target, backprop,
    /// SGD update on the router only.
    pub fn train_step(&self, x: &Tensor, target: &Tensor, lr: f64) -> anyhow::Result<f32> {
        let out = self.forward(x)?;
        let loss = out.sub(target)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        if let Some(g) = grads.get(self.router.as_tensor()) {
            let delta = g.affine(lr, 0.)?;
            let new = self.router.as_tensor().sub(&delta)?;
            self.router.set(&new)?;
        }
        Ok(loss.to_scalar::<f32>()?)
    }

    /// The gate probabilities for an input (which expert it routes to).
    pub fn gates(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        Ok(softmax(&x.matmul(&self.router)?, 1)?)
    }

    /// Trains the router directly: cross-entropy between the gate and the
    /// target expert index. This is the pure "learn to route" objective.
    pub fn train_gate(&self, x: &Tensor, expert_idx: &Tensor, lr: f64) -> anyhow::Result<f32> {
        let gates = self.gates(x)?;
        let loss = cross_entropy(&gates, expert_idx)?;
        let grads = loss.backward()?;
        if let Some(g) = grads.get(self.router.as_tensor()) {
            let delta = g.affine(lr, 0.)?;
            let new = self.router.as_tensor().sub(&delta)?;
            self.router.set(&new)?;
        }
        Ok(loss.to_scalar::<f32>()?)
    }

    /// Output of a single expert (used after routing to specialize it).
    pub fn expert_forward(&self, idx: usize, x: &Tensor) -> anyhow::Result<Tensor> {
        let [w1, w2] = &self.experts[idx];
        let h = x.matmul(w1.as_tensor())?.relu()?;
        Ok(h.matmul(w2.as_tensor())?)
    }

    /// Phase-2 training: specialize one expert on the data routed to it.
    ///
    /// # Details
    /// After the router has assigned inputs to experts, each expert is trained
    /// (MSE) on only its own inputs, so the experts diverge and specialize.
    pub fn train_expert(&self, idx: usize, x: &Tensor, target: &Tensor, lr: f64) -> anyhow::Result<f32> {
        let [w1, w2] = &self.experts[idx];
        let e = self.expert_forward(idx, x)?;
        let loss = e.sub(target)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        for w in [w1, w2] {
            if let Some(g) = grads.get(w.as_tensor()) {
                let new = w.as_tensor().sub(&g.affine(lr, 0.)?)?;
                w.set(&new)?;
            }
        }
        Ok(loss.to_scalar::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The MoE router learns to route each input class to its expert.
    #[test]
    fn test_moe_router_learns() {
        let dev = Device::Cpu;
        let r = CandleMoERouter::new(2, 4, 2, 2, &dev).unwrap();
        // class 0 inputs ~ [1,0], class 1 inputs ~ [0,1]
        let x0 = Tensor::new(&[[1.0f32, 0.0], [0.9, 0.1], [1.0, 0.0]], &dev).unwrap();
        let x1 = Tensor::new(&[[0.0f32, 1.0], [0.1, 0.9], [0.0, 1.0]], &dev).unwrap();
        let y0 = Tensor::new(&[0u32, 0, 0], &dev).unwrap();
        let y1 = Tensor::new(&[1u32, 1, 1], &dev).unwrap();
        for _ in 0..300 {
            r.train_gate(&x0, &y0, 0.05).unwrap();
            r.train_gate(&x1, &y1, 0.05).unwrap();
        }
        let g0 = r.gates(&x0).unwrap().to_vec2::<f32>().unwrap();
        let g1 = r.gates(&x1).unwrap().to_vec2::<f32>().unwrap();
        for row in &g0 {
            assert!(row[0] > row[1], "class0 should route to expert0: {row:?}");
        }
        for row in &g1 {
            assert!(row[1] > row[0], "class1 should route to expert1: {row:?}");
        }
    }
}

/// Two-phase MoE training: router first, then experts specialize on their
/// routed data.
#[test]
fn test_moe_two_phase_training() {
    use candle_core::Device;
    let dev = Device::Cpu;
    let r = CandleMoERouter::new(2, 4, 1, 2, &dev).unwrap();
    // class 0 -> expert 0 (target x[0]); class 1 -> expert 1 (target x[1])
    let x0 = Tensor::new(&[[1.0f32, 0.0], [0.9, 0.1], [1.0, 0.0]], &dev).unwrap();
    let x1 = Tensor::new(&[[0.0f32, 1.0], [0.1, 0.9], [0.0, 1.0]], &dev).unwrap();
    let y0 = Tensor::new(&[0u32, 0, 0], &dev).unwrap();
    let y1 = Tensor::new(&[1u32, 1, 1], &dev).unwrap();
    // phase 1: router learns to route
    for _ in 0..400 {
        r.train_gate(&x0, &y0, 0.05).unwrap();
        r.train_gate(&x1, &y1, 0.05).unwrap();
    }
    // phase 2: each expert specializes on its own class's data
    // (expert0 learns to output ~1 for its class, expert1 ~0 for its class)
    let t0 = Tensor::new(&[[1.0f32], [1.0], [1.0]], &dev).unwrap();
    let t1 = Tensor::new(&[[0.0f32], [0.0], [0.0]], &dev).unwrap();
    let first_loss = r.train_expert(0, &x0, &t0, 0.05).unwrap();
    r.train_expert(1, &x1, &t1, 0.05).unwrap();
    for _ in 0..500 {
        r.train_expert(0, &x0, &t0, 0.05).unwrap();
        r.train_expert(1, &x1, &t1, 0.05).unwrap();
    }
    let last_loss = r.train_expert(0, &x0, &t0, 0.05).unwrap();
    // expert training made progress (loss decreased), robust to random init
    assert!(
        last_loss < first_loss,
        "expert training should reduce loss: {last_loss} vs {first_loss}"
    );
    // the router must still route correctly after expert training
    let g0 = r.gates(&x0).unwrap().to_vec2::<f32>().unwrap();
    let g1 = r.gates(&x1).unwrap().to_vec2::<f32>().unwrap();
    for row in &g0 {
        assert!(row[0] > row[1], "router broke for class0: {row:?}");
    }
    for row in &g1 {
        assert!(row[1] > row[0], "router broke for class1: {row:?}");
    }
}
