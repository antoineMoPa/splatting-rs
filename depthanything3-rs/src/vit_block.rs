//! The DINOv2 transformer block of the DA3 backbone: pre-norm attention and a feed-forward
//! (GELU MLP, or SwiGLU in the Giant), each scaled by a LayerScale. From the variant's
//! `alt_start` on, the blocks also LayerNorm queries and keys per head (`qk_norm`) and
//! rotate them with 2D RoPE.

use candle_core::{D, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use mapanything_rs::vit_block::SwiGlu;

use crate::rope::RopeTables;
use crate::variant::{FeedForward, HEAD_DIM};

/// The blocks' norm1 and norm2 (upstream's `ln_eps`).
const BLOCK_NORM_EPS: f64 = 1e-6;
/// PyTorch's default, used by the q/k norms, the backbone's final norm and the head's norms.
pub const TORCH_LAYER_NORM_EPS: f64 = 1e-5;
const MLP_RATIO: usize = 4;

pub fn layer_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    candle_nn::layer_norm(dim, eps, vb)
}

/// LayerNorm whose affine parameters are absent from the checkpoint and left at PyTorch's
/// initialization (weight 1, bias 0), on `vb`'s device and dtype.
pub fn layer_norm_without_affine(dim: usize, eps: f64, vb: &VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        Tensor::ones(dim, vb.dtype(), vb.device())?,
        Tensor::zeros(dim, vb.dtype(), vb.device())?,
        eps,
    ))
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

/// `q`, `k`, `v`: [batch, heads, tokens, HEAD_DIM].
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    // The fused kernel only exists on Metal; it never materializes the score matrix, which
    // across all views of a long capture would not fit in memory.
    if q.device().is_metal() {
        return candle_nn::ops::sdpa(q, k, v, None, false, scale, 1.0);
    }
    // On the CPU one head at a time: the score matrix of all heads at once (6 views at
    // 378x504 attending across views: 6 x 5838 x 5838 floats) is 800 MB, which a wasm32
    // browser tab cannot afford.
    let heads = q.dim(1)?;
    let attended = (0..heads)
        .map(|head| {
            let scores = (q.narrow(1, head, 1)?.matmul(&k.narrow(1, head, 1)?.t()?)? * scale as f64)?;
            candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.narrow(1, head, 1)?)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&attended, 1)
}

/// fc2(gelu(fc1(x))), exact (erf) GELU as `nn.GELU`.
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.gelu_erf()?)
    }
}

/// The block's feed-forward, as the variant's `FeedForward`.
enum FeedForwardLayer {
    Mlp(Mlp),
    SwiGlu(SwiGlu),
}

impl FeedForwardLayer {
    /// `vb` at the block's `mlp`.
    fn new(kind: FeedForward, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(match kind {
            FeedForward::Mlp => Self::Mlp(Mlp {
                fc1: candle_nn::linear(dim, dim * MLP_RATIO, vb.pp("fc1"))?,
                fc2: candle_nn::linear(dim * MLP_RATIO, dim, vb.pp("fc2"))?,
            }),
            FeedForward::SwiGlu => Self::SwiGlu(SwiGlu::new(dim, vb)?),
        })
    }
}

impl Module for FeedForwardLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::SwiGlu(swiglu) => swiglu.forward(xs),
        }
    }
}

/// Per-head LayerNorm of the queries and keys, [batch, heads, tokens, HEAD_DIM].
struct QkNorm {
    q_norm: LayerNorm,
    k_norm: LayerNorm,
}

pub struct VitBlock {
    norm1: LayerNorm,
    qkv: Linear,
    qk_norm: Option<QkNorm>,
    proj: Linear,
    layer_scale1: Tensor,
    norm2: LayerNorm,
    mlp: FeedForwardLayer,
    layer_scale2: Tensor,
}

impl VitBlock {
    pub fn new(dim: usize, qk_norm: bool, feed_forward: FeedForward, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(dim, BLOCK_NORM_EPS, vb.pp("norm1"))?,
            qkv: candle_nn::linear(dim, 3 * dim, vb.pp("attn.qkv"))?,
            qk_norm: qk_norm
                .then(|| {
                    Ok::<_, candle_core::Error>(QkNorm {
                        q_norm: layer_norm(HEAD_DIM, TORCH_LAYER_NORM_EPS, vb.pp("attn.q_norm"))?,
                        k_norm: layer_norm(HEAD_DIM, TORCH_LAYER_NORM_EPS, vb.pp("attn.k_norm"))?,
                    })
                })
                .transpose()?,
            proj: candle_nn::linear(dim, dim, vb.pp("attn.proj"))?,
            layer_scale1: vb.get(dim, "ls1.gamma")?,
            norm2: layer_norm(dim, BLOCK_NORM_EPS, vb.pp("norm2"))?,
            mlp: FeedForwardLayer::new(feed_forward, dim, vb.pp("mlp"))?,
            layer_scale2: vb.get(dim, "ls2.gamma")?,
        })
    }

    /// `xs`: [sequences, tokens, dim]; every sequence attends only within itself. `rope`
    /// (one entry per token) is applied to the queries and keys after their norm.
    pub fn forward(&self, xs: &Tensor, rope: Option<&RopeTables>) -> Result<Tensor> {
        let dim = xs.dim(D::Minus1)?;
        let qkv = self.qkv.forward(&self.norm1.forward(xs)?)?;
        let mut q = split_heads(&qkv.narrow(2, 0, dim)?)?;
        let mut k = split_heads(&qkv.narrow(2, dim, dim)?)?;
        let v = split_heads(&qkv.narrow(2, 2 * dim, dim)?)?;
        if let Some(norms) = &self.qk_norm {
            q = norms.q_norm.forward(&q)?;
            k = norms.k_norm.forward(&k)?;
        }
        if let Some(rope) = rope {
            q = rope.apply(&q)?.contiguous()?;
            k = rope.apply(&k)?.contiguous()?;
        }
        let attended = self.proj.forward(&merge_heads(&attention(&q, &k, &v)?)?)?;
        let xs = (xs + attended.broadcast_mul(&self.layer_scale1)?)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs + mlp.broadcast_mul(&self.layer_scale2)?
    }
}
