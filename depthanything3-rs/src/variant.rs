//! The Depth Anything 3 variants with Apache 2.0 weights, and what differs between them:
//! only widths. The Large, Giant and Nested checkpoints are CC BY-NC and not supported.

use clap::ValueEnum;

/// DINOv2 patch size, every variant.
pub const PATCH_SIZE: usize = 14;
/// Transformer blocks, every variant (ViT-S and ViT-B are both 12 deep).
pub const DEPTH: usize = 12;
/// Attention head width, every variant.
pub const HEAD_DIM: usize = 64;
/// The blocks whose outputs feed the dense head and (the last one) the camera head.
pub const OUT_LAYERS: [usize; 4] = [5, 7, 9, 11];
/// From this block on: the class token is replaced by a camera token, every odd block
/// attends across all views (even blocks within each view), queries and keys are
/// LayerNorm-ed per head, and 2D rotary position embeddings are applied.
pub const ALT_START: usize = 4;
/// Longest image side the model is run at, as upstream's `process_res`.
pub const PROCESS_RESOLUTION: u32 = 504;
/// Weights file in each Hugging Face repository.
pub const WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Variant {
    Small,
    Base,
}

pub struct VariantSpec {
    pub weights_repo: &'static str,
    /// Transformer width.
    pub dim: usize,
    pub heads: usize,
    /// Width of the dense head's fusion features.
    pub head_features: usize,
    /// Channel count of each of the 4 hooked token maps after its 1x1 projection.
    pub head_stage_dims: [usize; 4],
}

const SMALL: VariantSpec = VariantSpec {
    weights_repo: "depth-anything/DA3-SMALL",
    dim: 384,
    heads: 6,
    head_features: 64,
    head_stage_dims: [48, 96, 192, 384],
};

const BASE: VariantSpec = VariantSpec {
    weights_repo: "depth-anything/DA3-BASE",
    dim: 768,
    heads: 12,
    head_features: 128,
    head_stage_dims: [96, 192, 384, 768],
};

const VARIANTS: [(Variant, &VariantSpec); 2] = [(Variant::Small, &SMALL), (Variant::Base, &BASE)];

impl Variant {
    pub fn spec(self) -> &'static VariantSpec {
        VARIANTS
            .iter()
            .find(|(variant, _)| *variant == self)
            .map(|(_, spec)| *spec)
            .expect("every variant has a spec")
    }
}

impl VariantSpec {
    /// Width of the tokens the heads see: the last local block's output concatenated with
    /// the current block's.
    pub fn head_input_dim(&self) -> usize {
        2 * self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_width_is_a_multiple_of_the_head_dim() {
        for (_, spec) in VARIANTS {
            assert_eq!(spec.dim, spec.heads * HEAD_DIM);
        }
    }
}
