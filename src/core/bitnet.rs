/*
 * @file bitnet.rs
 * @brief BitNet b1.58 ternary quantization and LUT matmul
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

//! FILE: bitnet.rs
//!
//! DESCRIPTION:
//! BitNet b1.58 Quantization and LUT Matmul for TinyGPT.
//!
//! BRIEF:
//! Ports the bitnet.cpp i2_s scheme: ternary {-1, 0, +1} weights quantized
//! per-output-channel with absmean scale, int8 activations quantized per-token
//! with absmax scale, and matrix multiplication via a 256-entry lookup table.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::core::attention::MultiHeadAttention;
use crate::core::block::Block;
use crate::core::feed_forward::FeedForward;
use crate::core::head::Head;
use crate::core::linear::{Linear, LinearLike};
use crate::core::math::randn;
use crate::core::tiny_gpt::TinyGPT;
use ndarray::{Array1, Array2};

/// Number of ternary weights packed into a single byte (2 bits each).
pub const GROUP_SIZE: usize = 4;

/// Number of LUT entries per input group (one per possible byte pattern).
pub const LUT_SIZE: usize = 256;

/// Decodes a 2-bit ternary code into its real value.
///
/// # Details
/// Encoding follows bitnet.cpp i2_s: `0 => -1`, `1 => 0`, `2 => +1`.
///
/// # Arguments
/// * `code` - 2-bit ternary code
///
/// # Returns
/// * `f32` - Decoded ternary value
fn ternary_value(code: u8) -> f32 {
    match code {
        0 => -1.0,
        1 => 0.0,
        _ => 1.0,
    }
}

/// Quantizes weights to ternary values with per-channel absmean scale.
///
/// # Details
/// Applies absmean quantization `Wq = clamp(round(W / gamma), -1, 1)` where
/// `gamma` is the mean absolute value of each output channel (column).
/// Resulting ternary weights are packed 4-per-byte along the input dimension.
///
/// # Arguments
/// * `w` - Weight matrix of shape `(in_features, out_features)`
///
/// # Returns
/// * `(Array2<u8>, Array1<f32>)` - Packed ternary weights of shape
///   `(in_features / GROUP_SIZE, out_features)` and per-channel scales
pub fn quantize_weights(w: &Array2<f32>) -> (Array2<u8>, Array1<f32>) {
    let (i, o) = (w.shape()[0], w.shape()[1]);
    assert!(i % GROUP_SIZE == 0, "in_features must be divisible by GROUP_SIZE");
    let mut qw = Array2::<u8>::zeros((i / GROUP_SIZE, o));
    let mut scales = Array1::<f32>::zeros(o);
    for j in 0..o {
        let gamma = w.column(j).mapv(|v| v.abs()).mean().unwrap().max(1e-6);
        scales[j] = gamma;
        for g in 0..i / GROUP_SIZE {
            let mut byte = 0u8;
            for k in 0..GROUP_SIZE {
                let v = w[[g * GROUP_SIZE + k, j]] / gamma;
                let code = if v.round() >= 1.0 {
                    2
                } else if v.round() <= -1.0 {
                    0
                } else {
                    1
                };
                byte |= code << (2 * k);
            }
            qw[[g, j]] = byte;
        }
    }
    (qw, scales)
}

/// Quantizes activations to int8 with per-token absmax scale.
///
/// # Details
/// Computes scale `s = 127 / max(|x|)` per row (token), rounds to int8.
/// Reconstruction is `x ~= xq / s`.
///
/// # Arguments
/// * `x` - Activation matrix of shape `(batch, features)`
///
/// # Returns
/// * `(Array2<i8>, Array1<f32>)` - Quantized int8 activations and per-token scales
pub fn quantize_input(x: &Array2<f32>) -> (Array2<i8>, Array1<f32>) {
    let (b, i) = (x.shape()[0], x.shape()[1]);
    let mut xq = Array2::<i8>::zeros((b, i));
    let mut scales = Array1::<f32>::zeros(b);
    for t in 0..b {
        let m = x.row(t).fold(0.0f32, |acc, &v| acc.max(v.abs())).max(1e-5);
        let s = 127.0 / m;
        scales[t] = s;
        for k in 0..i {
            xq[[t, k]] = (x[[t, k]] * s).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (xq, scales)
}

/// Multiplies quantized int8 activations by packed ternary weights via LUT.
///
/// # Details
/// Builds a 256-entry lookup table per input group. Entry `LUT[g][byte]` is the
/// integer dot product of the ternary weights encoded by `byte` against the
/// corresponding int8 activations. Each output element is a gather:
/// `out = sum over groups of LUT[g][qw[g, j]]`. No floating point multiply is
/// needed inside the accumulation, mirroring the bitnet.cpp LUT kernels.
///
/// # Arguments
/// * `xq` - Quantized activations of shape `(batch, in_features)`
/// * `qw` - Packed ternary weights of shape `(in_features / GROUP_SIZE, out_features)`
///
/// # Returns
/// * `Array2<i32>` - Integer accumulation of shape `(batch, out_features)`
pub fn lut_matmul(xq: &Array2<i8>, qw: &Array2<u8>) -> Array2<i32> {
    let (b, i) = (xq.shape()[0], xq.shape()[1]);
    let (ng, o) = (qw.shape()[0], qw.shape()[1]);
    assert_eq!(i, ng * GROUP_SIZE, "activation and weight input dims must match");
    let mut out = Array2::<i32>::zeros((b, o));
    let mut lut = vec![vec![0i32; LUT_SIZE]; ng];
    for t in 0..b {
        for g in 0..ng {
            for byte in 0..LUT_SIZE {
                let mut acc = 0i32;
                for k in 0..GROUP_SIZE {
                    let code = ((byte >> (2 * k)) & 0x3) as i8;
                    let wv: i32 = match code {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    };
                    acc += wv * xq[[t, g * GROUP_SIZE + k]] as i32;
                }
                lut[g][byte] = acc;
            }
        }
        for j in 0..o {
            let mut acc = 0i32;
            for g in 0..ng {
                acc += lut[g][qw[[g, j]] as usize];
            }
            out[[t, j]] = acc;
        }
    }
    out
}

/// BitLinear layer: ternary weights with LUT matmul.
///
/// # Details
/// Drop-in analog of `Linear` using BitNet b1.58 quantization.
/// Weights are stored in full precision and quantized on demand; inference
/// runs through `quantize_input` + `lut_matmul` + scale dequantization.
///
/// # Fields
/// * `w` - Full precision weights of shape `(in_features, out_features)`
/// * `qw` - Packed ternary weights of shape `(in_features / GROUP_SIZE, out_features)`
/// * `w_scales` - Per-output-channel absmean scales
#[derive(Clone)]
pub struct BitLinear {
    pub w: Array2<f32>,
    pub qw: Array2<u8>,
    pub w_scales: Array1<f32>,
}

impl BitLinear {
    /// Creates new bit linear layer.
    ///
    /// # Details
    /// Initializes weights with random normal distribution and quantizes them.
    ///
    /// # Arguments
    /// * `i` - Input dimension
    /// * `o` - Output dimension
    ///
    /// # Returns
    /// * `Self` - BitLinear instance
    pub fn new(i: usize, o: usize) -> Self {
        let w = randn(i, o);
        let (qw, w_scales) = quantize_weights(&w);
        Self { w, qw, w_scales }
    }

    /// Re-quantizes weights from the current full precision matrix.
    pub fn quantize(&mut self) {
        let (qw, w_scales) = quantize_weights(&self.w);
        self.qw = qw;
        self.w_scales = w_scales;
    }

    /// Computes quantized forward pass.
    ///
    /// # Details
    /// Runs `y = xW` with per-token int8 activations and ternary weights,
    /// dequantizing with `w_scales[j] / x_scale[t]`.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Output tensor
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (b, o) = (x.shape()[0], self.w.shape()[1]);
        let (xq, x_scales) = quantize_input(x);
        let acc = lut_matmul(&xq, &self.qw);
        let mut out = Array2::<f32>::zeros((b, o));
        for t in 0..b {
            for j in 0..o {
                out[[t, j]] = acc[[t, j]] as f32 * self.w_scales[j] / x_scales[t];
            }
        }
        out
    }

    /// Creates a quantized layer from an existing full precision layer.
    ///
    /// # Details
    /// Copies the full precision weights and computes packed ternary weights
    /// plus per-output-channel scales.
    ///
    /// # Arguments
    /// * `l` - Full precision linear layer
    ///
    /// # Returns
    /// * `Self` - BitLinear instance
    pub fn from_linear(l: &Linear) -> Self {
        let (qw, w_scales) = quantize_weights(&l.w);
        Self {
            w: l.w.clone(),
            qw,
            w_scales,
        }
    }
}

impl LinearLike for BitLinear {
    fn new(i: usize, o: usize) -> Self {
        BitLinear::new(i, o)
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        BitLinear::forward(self, x)
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self, _lr: f32) {}
}

/// Quantizes a full precision model into a ternary BitNet model.
///
/// # Details
/// Converts every projection layer in `TinyGPT<Linear>` to `BitLinear` while
/// preserving embeddings, layer norms, and configuration. The result is an
/// inference-only model (gradients are no-ops).
///
/// # Arguments
/// * `model` - Trained full precision model
///
/// # Returns
/// * `TinyGPT<BitLinear>` - Quantized inference model
pub fn quantize_model(model: &TinyGPT<Linear>) -> TinyGPT<BitLinear> {
    TinyGPT {
        tok: model.tok.clone(),
        pos: model.pos.clone(),
        blocks: model
            .blocks
            .iter()
            .map(|b| Block {
                sa: MultiHeadAttention {
                    heads: b
                        .sa
                        .heads
                        .iter()
                        .map(|h| Head {
                            key: BitLinear::from_linear(&h.key),
                            query: BitLinear::from_linear(&h.query),
                            value: BitLinear::from_linear(&h.value),
                            hs: h.hs,
                        })
                        .collect(),
                    proj: BitLinear::from_linear(&b.sa.proj),
                },
                ffn: FeedForward {
                    l1: BitLinear::from_linear(&b.ffn.l1),
                    l2: BitLinear::from_linear(&b.ffn.l2),
                },
                ln1: b.ln1.clone(),
                ln2: b.ln2.clone(),
            })
            .collect(),
        ln: model.ln.clone(),
        head: BitLinear::from_linear(&model.head),
        vocab: model.vocab,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests ternary value decoding.
    #[test]
    fn test_ternary_value() {
        assert_eq!(ternary_value(0), -1.0);
        assert_eq!(ternary_value(1), 0.0);
        assert_eq!(ternary_value(2), 1.0);
        assert_eq!(ternary_value(3), 1.0);
    }

    /// Tests weight quantization produces only ternary codes.
    #[test]
    fn test_quantize_weights_ternary_codes() {
        let w = arr2(&[
            [1.0, -2.0, 0.5, 0.0],
            [-1.0, 0.5, 2.0, -0.5],
            [0.2, 1.5, -1.5, 0.1],
            [2.0, -1.0, 0.3, 1.0],
        ]);
        let (qw, scales) = quantize_weights(&w);
        assert_eq!(qw.shape(), &[1, 4]);
        assert_eq!(scales.shape(), &[4]);
        for &byte in qw.iter() {
            for k in 0..GROUP_SIZE {
                let code = (byte >> (2 * k)) & 0x3;
                assert!(code <= 2, "code must be 0..=2, got {code}");
            }
        }
    }

    /// Tests weight scales are positive and match absmean per channel.
    #[test]
    fn test_quantize_weights_scales() {
        let w = arr2(&[
            [1.0, -2.0],
            [3.0, 4.0],
            [1.0, -2.0],
            [3.0, 4.0],
        ]);
        let (_, scales) = quantize_weights(&w);
        assert!((scales[0] - 2.0).abs() < 1e-5); // mean(|1|,|3|,|1|,|3|) = 2.0
        assert!((scales[1] - 3.0).abs() < 1e-5); // mean(|-2|,|4|,|-2|,|4|) = 3.0
    }

    /// Tests weight quantization round-trips sign for dominant values.
    #[test]
    fn test_quantize_weights_sign() {
        let w = arr2(&[
            [10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
        ]);
        let (qw, scales) = quantize_weights(&w);
        // column 0: all +10.0 => +1
        for &byte in qw.column(0).iter() {
            assert_eq!((byte >> 0) & 0x3, 2);
        }
        // column 1: all -10.0 => -1
        for &byte in qw.column(1).iter() {
            assert_eq!((byte >> 0) & 0x3, 0);
        }
        // column 2: all 0.0 => 0
        for &byte in qw.column(2).iter() {
            assert_eq!((byte >> 0) & 0x3, 1);
        }
        assert!(scales.iter().all(|&s| s > 0.0));
    }

    /// Tests input quantization preserves bounds of int8.
    #[test]
    fn test_quantize_input_bounds() {
        let x = arr2(&[[1.0, -2.0, 0.5], [3.0, -1.0, 2.0]]);
        let (xq, scales) = quantize_input(&x);
        assert_eq!(xq.shape(), &[2, 3]);
        assert_eq!(scales.shape(), &[2]);
        for &v in xq.iter() {
            assert!((v as i16).abs() <= 127);
        }
        // max value maps to 127
        assert_eq!(xq[[1, 0]], 127);
        assert_eq!(xq[[0, 1]], -127);
    }

    /// Tests input scale formula.
    #[test]
    fn test_quantize_input_scales() {
        let x = arr2(&[[2.0, -4.0]]);
        let (_, scales) = quantize_input(&x);
        assert!((scales[0] - 127.0 / 4.0).abs() < 1e-5);
    }

    /// Tests LUT matmul matches direct integer dot product.
    #[test]
    fn test_lut_matmul_matches_direct() {
        let x = arr2(&[[1.0, -2.0, 0.5, 1.5], [-1.0, 2.0, 1.0, -0.5]]);
        let w = arr2(&[
            [1.0, -2.0, 0.5, 0.0],
            [-1.0, 0.5, 2.0, -0.5],
            [0.2, 1.5, -1.5, 0.1],
            [2.0, -1.0, 0.3, 1.0],
        ]);
        let (xq, _) = quantize_input(&x);
        let (qw, scales) = quantize_weights(&w);
        let acc = lut_matmul(&xq, &qw);

        // direct integer reference: sum over k of ternary * xq
        let (i, o) = (qw.shape()[0] * GROUP_SIZE, qw.shape()[1]);
        for t in 0..xq.shape()[0] {
            for j in 0..o {
                let mut direct = 0i32;
                for g in 0..qw.shape()[0] {
                    let byte = qw[[g, j]];
                    for k in 0..GROUP_SIZE {
                        let code = (byte >> (2 * k)) & 0x3;
                        let wv: i32 = match code {
                            0 => -1,
                            1 => 0,
                            _ => 1,
                        };
                        direct += wv * xq[[t, g * GROUP_SIZE + k]] as i32;
                    }
                }
                assert_eq!(acc[[t, j]], direct, "LUT mismatch at ({t},{j})");
                let _ = i;
            }
        }
        assert!(scales.len() == o);
    }

    /// Tests BitLinear forward preserves output shape.
    #[test]
    fn test_bit_linear_forward_shape() {
        let l = BitLinear::new(8, 16);
        let x = arr2(&[[1.0; 8], [2.0; 8]]);
        let y = l.forward(&x);
        assert_eq!(y.shape(), &[2, 16]);
    }

    /// Tests BitLinear forward produces finite values.
    #[test]
    fn test_bit_linear_forward_finite() {
        let l = BitLinear::new(16, 8);
        let x = arr2(&[[1.0; 16], [-2.0; 16]]);
        let y = l.forward(&x);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Tests BitLinear forward matches the quantized reference computation.
    #[test]
    fn test_bit_linear_forward_matches_reference() {
        let l = BitLinear::new(16, 8);
        let x = arr2(&[[1.0; 16], [2.0; 16], [-1.0; 16]]);
        let y = l.forward(&x);

        // reference: quantize input, dequantize weights, float matmul, scale back
        let (xq, x_scales) = quantize_input(&x);
        let (b, i) = (xq.shape()[0], xq.shape()[1]);
        let (o,) = (l.w.shape()[1],);
        for t in 0..b {
            for j in 0..o {
                let mut acc = 0.0f32;
                for k in 0..i {
                    let g = k / GROUP_SIZE;
                    let pos = k % GROUP_SIZE;
                    let code = (l.qw[[g, j]] >> (2 * pos)) & 0x3;
                    acc += ternary_value(code) * xq[[t, k]] as f32;
                }
                let expected = acc * l.w_scales[j] / x_scales[t];
                assert!((y[[t, j]] - expected).abs() < 1e-3, "mismatch at ({t},{j})");
            }
        }
    }

    /// Tests quantize() re-computes packed weights.
    #[test]
    fn test_bit_linear_requantize() {
        let mut l = BitLinear::new(4, 4);
        let before = l.qw.clone();
        l.w = arr2(&[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]);
        l.quantize();
        assert_eq!(before.shape(), l.qw.shape());
        // uniform positive weights => all codes +1
        for &byte in l.qw.iter() {
            assert_eq!(byte, 0b10_10_10_10);
        }
    }

    /// Tests quantization error is bounded for smooth weights.
    #[test]
    fn test_bit_linear_quantization_error() {
        let w = arr2(&[
            [1.0, -1.0, 0.0, 1.5, -0.5, 2.0, -2.0, 0.0],
            [1.0, -1.0, 0.0, 1.5, -0.5, 2.0, -2.0, 0.0],
            [1.0, -1.0, 0.0, 1.5, -0.5, 2.0, -2.0, 0.0],
            [1.0, -1.0, 0.0, 1.5, -0.5, 2.0, -2.0, 0.0],
        ]);
        let (qw, _) = quantize_weights(&w);
        // channel 0: all +1.0 => +1
        let c0 = qw.column(0);
        for &byte in c0.iter() {
            assert_eq!(byte, 0b10_10_10_10);
        }
        // channel 1: all -1.0 => -1
        let c1 = qw.column(1);
        for &byte in c1.iter() {
            assert_eq!(byte, 0b00_00_00_00);
        }
        // channel 2: all 0.0 => 0
        let c2 = qw.column(2);
        for &byte in c2.iter() {
            assert_eq!(byte, 0b01_01_01_01);
        }
        // channel 7: all 0.0 => 0
        let c7 = qw.column(7);
        for &byte in c7.iter() {
            assert_eq!(byte, 0b01_01_01_01);
        }
    }

    /// Tests BitLinear clone.
    #[test]
    fn test_bit_linear_clone() {
        let l1 = BitLinear::new(8, 4);
        let l2 = l1.clone();
        assert_eq!(l1.w.shape(), l2.w.shape());
        assert_eq!(l1.qw.shape(), l2.qw.shape());
        assert_eq!(l1.w_scales.shape(), l2.w_scales.shape());
    }

    /// Tests from_linear copies weights and quantizes.
    #[test]
    fn test_from_linear() {
        let l = Linear::new(8, 4);
        let b = BitLinear::from_linear(&l);
        assert_eq!(b.w, l.w);
        assert_eq!(b.qw.shape(), &[2, 4]);
        assert_eq!(b.w_scales.shape(), &[4]);
    }

    /// Tests from_linear forward matches original Linear output correlation.
    #[test]
    fn test_from_linear_forward_correlated() {
        let l = Linear::new(32, 16);
        let b = BitLinear::from_linear(&l);
        let x = randn(4, 32);
        let fp = l.forward(&x);
        let q = b.forward(&x);
        assert_eq!(fp.shape(), q.shape());
        let n = fp.len() as f32;
        let mf = fp.iter().sum::<f32>() / n;
        let mq = q.iter().sum::<f32>() / n;
        let (mut num, mut df, mut dq) = (0.0, 0.0, 0.0);
        for (f, qv) in fp.iter().zip(q.iter()) {
            let (a, b) = (f - mf, qv - mq);
            num += a * b;
            df += a * a;
            dq += b * b;
        }
        let corr = num / (df * dq).sqrt().max(1e-8);
        assert!(corr > 0.0, "quantized single layer should correlate with fp");
    }

    /// Tests end-to-end quantized model matches full precision model.
    #[test]
    fn test_quantized_model_end_to_end() {
        use crate::core::trainer::Trainer;
        use crate::core::vocab::Vocab;
        use rand::rng;

        let corpus = vec![
            "the quick brown fox jumps over the lazy dog".to_string(),
            "the slow red fox sleeps under the big tree".to_string(),
            "a quick dog runs after a brown cat today".to_string(),
            "the lazy cat sleeps near the warm fire today".to_string(),
            "every quick brown fox jumps over every lazy dog".to_string(),
            "the big tree grows over the lazy red dog today".to_string(),
        ];
        let vocab = Vocab::from_corpus(&corpus);
        let mut trainer = Trainer::new(vocab.size, &vocab.data);
        let mut rng = rng();
        trainer.train_steps(50, &mut rng);

        let idx = [0usize, 1, 2, 3, 4];
        let fp = trainer.model.forward(&idx);
        let qmodel = quantize_model(&trainer.model);
        let q = qmodel.forward(&idx);
        assert_eq!(fp.shape(), q.shape());
        assert!(q.iter().all(|v| v.is_finite()));

        let n = fp.len() as f32;
        let mf = fp.iter().sum::<f32>() / n;
        let mq = q.iter().sum::<f32>() / n;
        let (mut num, mut df, mut dq) = (0.0, 0.0, 0.0);
        for (f, qv) in fp.iter().zip(q.iter()) {
            let (a, b) = (f - mf, qv - mq);
            num += a * b;
            df += a * a;
            dq += b * b;
        }
        let corr = num / (df * dq).sqrt().max(1e-8);
        println!("quantized model correlation: {corr:.3}");
        assert!(
            corr > 0.3,
            "quantized model should stay correlated with fp, corr={corr}"
        );
    }
}
