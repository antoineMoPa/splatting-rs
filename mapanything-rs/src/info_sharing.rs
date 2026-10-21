//! MapAnything's multi-view alternating-attention transformer: blocks alternate between
//! attention over every token of every view (even blocks) and attention within each view
//! (odd blocks). No positional encoding beyond a learned embedding marking view 0.

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

use crate::dinov2::DIM;
use crate::vit_block::VitBlock;
use crate::vit_block::layer_norm;

const DEPTH: usize = 16;
/// Blocks after which the (normed) sequence feeds the dense head, besides the last.
const INTERMEDIATE_BLOCKS: [usize; 2] = [7, 11];

/// One view's input: fused patch tokens [1, patches, DIM] and its raw encoder class token
/// [1, 1, DIM].
pub struct ViewTokens {
    pub patches: Tensor,
    pub class: Tensor,
}

/// Normed patch tokens [1, patches, DIM] of each view, after blocks 7, 11 and 15.
pub struct SharedFeatures {
    pub per_view: Vec<[Tensor; 3]>,
    /// Final normed global scale token [1, DIM].
    pub scale_token: Tensor,
}

pub struct AlternatingAttention {
    blocks: Vec<VitBlock>,
    norm: LayerNorm,
    /// [DIM], added to every token of view 0, the reference view.
    reference_view_embedding: Tensor,
}

impl AlternatingAttention {
    /// `vb` at `info_sharing`.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            blocks: (0..DEPTH)
                .map(|index| VitBlock::new(DIM, vb.pp(format!("self_attention_blocks.{index}"))))
                .collect::<Result<_>>()?,
            norm: layer_norm(DIM, vb.pp("norm"))?,
            reference_view_embedding: vb.get((1, DIM), "view_pos_table")?.squeeze(0)?,
        })
    }

    /// Every view must have the same patch count. `scale_token`: [1, 1, DIM].
    pub fn forward(&self, views: &[ViewTokens], scale_token: &Tensor) -> Result<SharedFeatures> {
        let view_count = views.len();
        let patch_count = views[0].patches.dim(1)?;
        assert!(
            views.iter().all(|view| view.patches.dim(1).unwrap() == patch_count),
            "every view must have the same patch grid"
        );
        // Per view [patches, class], view 0 marked as the reference.
        let tokens_per_view = patch_count + 1;
        let sequences: Vec<Tensor> = views
            .iter()
            .enumerate()
            .map(|(index, view)| {
                let sequence = Tensor::cat(&[&view.patches, &view.class], 1)?;
                match index {
                    0 => sequence.broadcast_add(&self.reference_view_embedding),
                    _ => Ok(sequence),
                }
            })
            .collect::<Result<_>>()?;
        let view_tokens = view_count * tokens_per_view;
        let mut xs = Tensor::cat(&[Tensor::cat(&sequences, 1)?, scale_token.clone()], 1)?;

        let mut stored = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            xs = if index % 2 == 0 {
                block.forward(&xs)?
            } else {
                // Each view attends within itself; the scale token waits.
                let per_view = xs.narrow(1, 0, view_tokens)?.reshape((view_count, tokens_per_view, DIM))?;
                let attended = block.forward(&per_view)?.reshape((1, view_tokens, DIM))?;
                Tensor::cat(&[attended, xs.narrow(1, view_tokens, 1)?], 1)?
            };
            if INTERMEDIATE_BLOCKS.contains(&index) || index == DEPTH - 1 {
                stored.push(self.norm.forward(&xs)?);
            }
        }

        let patches_of = |sequence: &Tensor, view: usize| sequence.narrow(1, view * tokens_per_view, patch_count);
        let per_view = (0..view_count)
            .map(|view| {
                Ok([
                    patches_of(&stored[0], view)?,
                    patches_of(&stored[1], view)?,
                    patches_of(&stored[2], view)?,
                ])
            })
            .collect::<Result<_>>()?;
        Ok(SharedFeatures {
            per_view,
            scale_token: stored[2].narrow(1, view_tokens, 1)?.squeeze(1)?,
        })
    }
}
