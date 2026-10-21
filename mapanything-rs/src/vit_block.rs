//! The transformer block shared by the DINOv2 encoder and the alternating-attention
//! transformer: pre-norm attention and SwiGLU MLP, each scaled by a LayerScale.

use candle_core::{D, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

/// Every LayerNorm in the model.
const LAYER_NORM_EPS: f64 = 1e-6;
const HEAD_DIM: usize = 64;

pub fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    candle_nn::layer_norm(dim, LAYER_NORM_EPS, vb)
}

/// [batch, tokens, heads * HEAD_DIM] -> [batch, heads, tokens, HEAD_DIM]
fn split_heads(xs: &Tensor) -> Result<Tensor> {
    let (batch, token_count, dim) = xs.dims3()?;
    xs.reshape((batch, token_count, dim / HEAD_DIM, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()
}

/// [batch, heads, tokens, HEAD_DIM] -> [batch, tokens, heads * HEAD_DIM]
fn merge_heads(xs: &Tensor) -> Result<Tensor> {
    let (batch, heads, token_count, head_dim) = xs.dims4()?;
    xs.transpose(1, 2)?.reshape((batch, token_count, heads * head_dim))
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    // The fused kernel only exists on Metal; it never materializes the score matrix, which
    // across all views of a long capture would not fit in memory.
    if q.device().is_metal() {
        return candle_nn::ops::sdpa(q, k, v, None, false, scale, 1.0);
    }
    let scores = (q.matmul(&k.t()?)? * scale as f64)?;
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)
}

/// SwiGLU hidden width: 2/3 of 4 x dim, rounded up to a multiple of 8 (DINOv2's
/// SwiGLUFFNFused). 4096 for dim 1536.
pub fn swiglu_hidden_dim(dim: usize) -> usize {
    (dim * 4 * 2 / 3).div_ceil(8) * 8
}

/// w3(silu(x1) * x2), with [x1, x2] = w12(x). Also the Giant Depth Anything 3 backbone's
/// feed-forward (depthanything3-rs).
pub struct SwiGlu {
    w12: Linear,
    w3: Linear,
}

impl SwiGlu {
    /// `vb` at the block's `mlp`.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = swiglu_hidden_dim(dim);
        Ok(Self {
            w12: candle_nn::linear(dim, 2 * hidden, vb.pp("w12"))?,
            w3: candle_nn::linear(hidden, dim, vb.pp("w3"))?,
        })
    }
}

impl Module for SwiGlu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x12 = self.w12.forward(xs)?;
        let hidden = x12.dim(D::Minus1)? / 2;
        let gated = (x12.narrow(D::Minus1, 0, hidden)?.silu()? * x12.narrow(D::Minus1, hidden, hidden)?)?;
        self.w3.forward(&gated)
    }
}

pub struct VitBlock {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    layer_scale1: Tensor,
    norm2: LayerNorm,
    mlp: SwiGlu,
    layer_scale2: Tensor,
}

impl VitBlock {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(dim, vb.pp("norm1"))?,
            qkv: candle_nn::linear(dim, 3 * dim, vb.pp("attn.qkv"))?,
            proj: candle_nn::linear(dim, dim, vb.pp("attn.proj"))?,
            layer_scale1: vb.get(dim, "ls1.gamma")?,
            norm2: layer_norm(dim, vb.pp("norm2"))?,
            mlp: SwiGlu::new(dim, vb.pp("mlp"))?,
            layer_scale2: vb.get(dim, "ls2.gamma")?,
        })
    }

    /// `xs`: [sequences, tokens, dim]; every sequence attends only within itself.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dim = xs.dim(2)?;
        let qkv = self.qkv.forward(&self.norm1.forward(xs)?)?;
        let q = split_heads(&qkv.narrow(2, 0, dim)?)?;
        let k = split_heads(&qkv.narrow(2, dim, dim)?)?;
        let v = split_heads(&qkv.narrow(2, 2 * dim, dim)?)?;
        let attended = self.proj.forward(&merge_heads(&attention(&q, &k, &v)?)?)?;
        let xs = (xs + attended.broadcast_mul(&self.layer_scale1)?)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs + mlp.broadcast_mul(&self.layer_scale2)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swiglu_width_matches_dinov2_giant() {
        assert_eq!(swiglu_hidden_dim(1536), 4096);
    }
}
