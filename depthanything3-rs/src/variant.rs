//! The Depth Anything 3 variants and what differs between them: widths, depth, where the
//! alternating attention starts and (the Giant) the feed-forward. Small and Base are Apache
//! 2.0; Large and Giant are CC BY-NC 4.0 (no commercial use). The Nested checkpoints are not
//! supported.

use clap::ValueEnum;

/// DINOv2 patch size, every variant.
pub const PATCH_SIZE: usize = 14;
/// Attention head width, every variant.
pub const HEAD_DIM: usize = 64;
/// Longest image side the model is run at, as upstream's `process_res`.
pub const PROCESS_RESOLUTION: u32 = 504;
/// Weights file in each Hugging Face repository.
pub const WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Variant {
    Small,
    Base,
    Large,
    Giant,
}

/// The transformer blocks' feed-forward (DINOv2's `ffn_layer`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedForward {
    /// fc2(gelu(fc1(x))), 4 x dim wide.
    Mlp,
    /// w3(silu(x1) * x2) with [x1, x2] = w12(x) (DINOv2's `swiglufused`), 2/3 of 4 x dim
    /// wide rounded up to a multiple of 8.
    SwiGlu,
}

pub struct VariantSpec {
    /// For people: "Depth Anything 3 Small".
    pub label: &'static str,
    pub weights_repo: &'static str,
    /// Size of `WEIGHTS_FILE` in the repository, in MB.
    pub weights_megabytes: u32,
    /// Transformer width.
    pub dim: usize,
    pub heads: usize,
    /// Transformer blocks.
    pub depth: usize,
    pub feed_forward: FeedForward,
    /// From this block on: the class token is replaced by a camera token, every odd block
    /// attends across all views (even blocks within each view), queries and keys are
    /// LayerNorm-ed per head, and 2D rotary position embeddings are applied.
    pub alt_start: usize,
    /// The blocks whose outputs feed the dense head and (the last one) the camera head.
    pub out_layers: [usize; 4],
    /// Width of the dense head's fusion features.
    pub head_features: usize,
    /// Channel count of each of the 4 hooked token maps after its 1x1 projection.
    pub head_stage_dims: [usize; 4],
}

const SMALL: VariantSpec = VariantSpec {
    label: "Depth Anything 3 Small",
    weights_repo: "depth-anything/DA3-SMALL",
    weights_megabytes: 137,
    dim: 384,
    heads: 6,
    depth: 12,
    feed_forward: FeedForward::Mlp,
    alt_start: 4,
    out_layers: [5, 7, 9, 11],
    head_features: 64,
    head_stage_dims: [48, 96, 192, 384],
};

const BASE: VariantSpec = VariantSpec {
    label: "Depth Anything 3 Base",
    weights_repo: "depth-anything/DA3-BASE",
    weights_megabytes: 542,
    dim: 768,
    heads: 12,
    depth: 12,
    feed_forward: FeedForward::Mlp,
    alt_start: 4,
    out_layers: [5, 7, 9, 11],
    head_features: 128,
    head_stage_dims: [96, 192, 384, 768],
};

const LARGE: VariantSpec = VariantSpec {
    label: "Depth Anything 3 Large",
    weights_repo: "depth-anything/DA3-LARGE",
    weights_megabytes: 1644,
    dim: 1024,
    heads: 16,
    depth: 24,
    feed_forward: FeedForward::Mlp,
    alt_start: 8,
    out_layers: [11, 15, 19, 23],
    head_features: 256,
    head_stage_dims: [256, 512, 1024, 1024],
};

const GIANT: VariantSpec = VariantSpec {
    label: "Depth Anything 3 Giant",
    weights_repo: "depth-anything/DA3-GIANT",
    weights_megabytes: 5423,
    dim: 1536,
    heads: 24,
    depth: 40,
    feed_forward: FeedForward::SwiGlu,
    alt_start: 13,
    out_layers: [19, 27, 33, 39],
    head_features: 256,
    head_stage_dims: [256, 512, 1024, 1024],
};

const VARIANTS: [(Variant, &VariantSpec); 4] = [
    (Variant::Small, &SMALL),
    (Variant::Base, &BASE),
    (Variant::Large, &LARGE),
    (Variant::Giant, &GIANT),
];

impl Variant {
    /// Every variant, smallest first.
    pub const ALL: [Variant; 4] = [Variant::Small, Variant::Base, Variant::Large, Variant::Giant];

    pub fn label(self) -> &'static str {
        self.spec().label
    }

    pub fn hf_repo(self) -> &'static str {
        self.spec().weights_repo
    }

    pub fn weights_megabytes(self) -> u32 {
        self.spec().weights_megabytes
    }

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
    fn all_lists_every_variant_once() {
        assert_eq!(Variant::ALL.len(), VARIANTS.len());
        for (variant, _) in VARIANTS {
            assert_eq!(Variant::ALL.iter().filter(|v| **v == variant).count(), 1);
        }
    }

    #[test]
    fn head_width_is_a_multiple_of_the_head_dim() {
        for (_, spec) in VARIANTS {
            assert_eq!(spec.dim, spec.heads * HEAD_DIM);
        }
    }

    #[test]
    fn output_layers_rise_through_the_alternating_blocks_to_the_last() {
        for (variant, spec) in VARIANTS {
            assert!(spec.alt_start < spec.out_layers[0], "{variant:?}");
            assert!(spec.out_layers.windows(2).all(|pair| pair[0] < pair[1]), "{variant:?}");
            assert_eq!(spec.out_layers[3], spec.depth - 1, "{variant:?}");
        }
    }
}
