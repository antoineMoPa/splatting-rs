//! The DA3 backbone: a DINOv2 ViT over all views at once. Blocks 0 to `alt_start - 1` are
//! plain DINOv2 blocks, one view at a time. From block `alt_start` on, the class token is
//! replaced by a learned camera token (one for the reference view, one for the others), odd
//! blocks attend across all views' tokens and even blocks within each view, with per-head
//! q/k norms and 2D RoPE (`rope.rs`). The outputs of the spec's `out_layers` feed the heads, each as the last within-view block's output concatenated with the block's
//! own (`cat_token`), the latter after the final LayerNorm.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, VarBuilder};
use mapanything_rs::dinov2::resize_grid;

use crate::model::{Progress, Stage};
use crate::reference_view::{
    MIN_VIEWS_FOR_SELECTION, ReferenceViewStrategy, order_with_reference_first, positions_of_views, select_reference_view,
};
use crate::rope::rope_tables;
use crate::variant::{PATCH_SIZE, VariantSpec};
use crate::vit_block::{TORCH_LAYER_NORM_EPS, VitBlock, layer_norm};

/// The positional embedding is trained on a TRAINED_GRID x TRAINED_GRID patch grid (518 px).
const TRAINED_GRID: usize = 37;
/// DINOv2 resizes its positional embedding with scale factor (out + this) / TRAINED_GRID,
/// not with the output size (a historical guard against floating point error).
const INTERPOLATE_OFFSET: f64 = 0.1;
/// Index of the camera (formerly class) token in each view's sequence.
const CAMERA_TOKEN: usize = 0;

pub struct BackboneOutput {
    /// One per `out_layers`: the patch tokens [views, patches, 2 * dim], in input order.
    pub layers: [Tensor; 4],
    /// The camera tokens of the last output layer, [views, 2 * dim], before any norm.
    pub camera_tokens: Tensor,
    /// The view the poses are relative to.
    pub reference_view: usize,
}

pub struct Backbone {
    dim: usize,
    alt_start: usize,
    out_layers: [usize; 4],
    patch_embed: Conv2d,
    cls_token: Tensor,
    /// [1, 2, dim]: the reference view's camera token, then the other views'.
    camera_token: Tensor,
    /// [1, 1 + TRAINED_GRID^2, dim], on the CPU in f32 for resizing.
    pos_embed: Tensor,
    blocks: Vec<VitBlock>,
    norm: LayerNorm,
}

/// RoPE position of every token of one view: camera token at (0, 0); patches at
/// (row + 1, column + 1) within a view, all at (1, 1) across views.
fn rope_positions(rows: usize, columns: usize, across_views: bool) -> Vec<[u32; 2]> {
    let mut positions = vec![[0, 0]];
    for row in 0..rows {
        for column in 0..columns {
            positions.push(if across_views {
                [1, 1]
            } else {
                [row as u32 + 1, column as u32 + 1]
            });
        }
    }
    positions
}

impl Backbone {
    /// `vb` at the DINOv2 model (`backbone.pretrained`).
    pub fn new(spec: &VariantSpec, vb: VarBuilder) -> Result<Self> {
        let dim = spec.dim;
        let patch_config = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };
        Ok(Self {
            dim,
            alt_start: spec.alt_start,
            out_layers: spec.out_layers,
            patch_embed: candle_nn::conv2d(3, dim, PATCH_SIZE, patch_config, vb.pp("patch_embed.proj"))?,
            cls_token: vb.get((1, 1, dim), "cls_token")?,
            camera_token: vb.get((1, 2, dim), "camera_token")?,
            pos_embed: vb
                .get((1, 1 + TRAINED_GRID * TRAINED_GRID, dim), "pos_embed")?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?,
            blocks: (0..spec.depth)
                .map(|index| VitBlock::new(dim, index >= spec.alt_start, spec.feed_forward, vb.pp(format!("blocks.{index}"))))
                .collect::<Result<_>>()?,
            norm: layer_norm(dim, TORCH_LAYER_NORM_EPS, vb.pp("norm"))?,
        })
    }

    /// [1, 1 + rows * columns, dim]: the class embedding, then the patch grid's.
    fn positional_embedding(&self, rows: usize, columns: usize) -> Result<Tensor> {
        if (rows, columns) == (TRAINED_GRID, TRAINED_GRID) {
            return Ok(self.pos_embed.clone());
        }
        let class = self.pos_embed.narrow(1, 0, 1)?;
        let grid = self
            .pos_embed
            .narrow(1, 1, TRAINED_GRID * TRAINED_GRID)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let scale = |out: usize| (out as f64 + INTERPOLATE_OFFSET) / TRAINED_GRID as f64;
        let resized = resize_grid(
            &grid,
            (TRAINED_GRID, TRAINED_GRID, self.dim),
            (rows, columns),
            (scale(rows), scale(columns)),
        );
        Tensor::cat(&[class, Tensor::from_vec(resized, (1, rows * columns, self.dim), &Device::Cpu)?], 1)
    }

    /// The camera tokens of `views` views in reference-first order: [views, 1, dim].
    fn camera_tokens(&self, views: usize) -> Result<Tensor> {
        let reference = self.camera_token.narrow(1, 0, 1)?;
        let source = self.camera_token.narrow(1, 1, 1)?;
        Tensor::cat(
            &std::iter::once(&reference)
                .chain(std::iter::repeat_n(&source, views - 1))
                .collect::<Vec<_>>(),
            0,
        )
    }

    /// `images`: [views, 3, H, W], ImageNet-normalized, H and W multiples of PATCH_SIZE.
    /// `progress` hears of every block done.
    pub fn forward(&self, images: &Tensor, strategy: ReferenceViewStrategy, progress: Progress) -> Result<BackboneOutput> {
        let (views, _, height, width) = images.dims4()?;
        assert!(views >= 1, "no views");
        assert!(
            height % PATCH_SIZE == 0 && width % PATCH_SIZE == 0,
            "image {width}x{height} is not a multiple of the {PATCH_SIZE} px patch"
        );
        let (rows, columns) = (height / PATCH_SIZE, width / PATCH_SIZE);
        let tokens_per_view = 1 + rows * columns;
        let device = images.device();
        let dtype = images.dtype();

        let patches = self.patch_embed.forward(images)?.flatten_from(2)?.transpose(1, 2)?;
        let positional = self.positional_embedding(rows, columns)?.to_dtype(dtype)?.to_device(device)?;
        let cls = self.cls_token.broadcast_as((views, 1, self.dim))?;
        let mut xs = Tensor::cat(&[&cls, &patches], 1)?.broadcast_add(&positional)?;

        let within_view_rope = rope_tables(&rope_positions(rows, columns, false), device, dtype)?;
        let across_views_rope = rope_tables(&rope_positions(rows, columns, true).repeat(views), device, dtype)?;

        // Views are kept in input order until the reference is chosen; `order[position]`
        // is then the input view at each position of `xs`.
        let mut order: Vec<usize> = (0..views).collect();
        let mut reference_view = 0;
        let mut within_view_output = xs.clone();
        let mut layers = Vec::with_capacity(self.out_layers.len());
        let mut camera_tokens = None;
        for (index, block) in self.blocks.iter().enumerate() {
            progress(Stage::Block(index));
            if index + 1 == self.alt_start && views >= MIN_VIEWS_FOR_SELECTION {
                let class_tokens = xs.narrow(1, CAMERA_TOKEN, 1)?.squeeze(1)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                reference_view = select_reference_view(&class_tokens, strategy);
                order = order_with_reference_first(views, reference_view);
                let indices = Tensor::new(order.iter().map(|&v| v as u32).collect::<Vec<_>>(), device)?;
                xs = xs.index_select(&indices, 0)?;
            }
            if index == self.alt_start {
                xs = Tensor::cat(&[self.camera_tokens(views)?, xs.narrow(1, 1, tokens_per_view - 1)?], 1)?;
            }
            let across_views = index >= self.alt_start && index % 2 == 1;
            if across_views {
                xs = block
                    .forward(&xs.reshape((1, views * tokens_per_view, self.dim))?, Some(&across_views_rope))?
                    .reshape((views, tokens_per_view, self.dim))?;
            } else {
                let rope = (index >= self.alt_start).then_some(&within_view_rope);
                xs = block.forward(&xs, rope)?;
                within_view_output = xs.clone();
            }
            if self.out_layers.contains(&index) {
                let mut out = Tensor::cat(&[&within_view_output, &xs], D::Minus1)?;
                if order.iter().enumerate().any(|(position, &view)| position != view) {
                    let positions = positions_of_views(&order);
                    let indices = Tensor::new(positions.iter().map(|&p| p as u32).collect::<Vec<_>>(), device)?;
                    out = out.index_select(&indices, 0)?;
                }
                // Contiguous: Metal's matmul rejects the strided view.
                camera_tokens = Some(out.narrow(1, CAMERA_TOKEN, 1)?.squeeze(1)?.contiguous()?);
                let within = out.narrow(2, 0, self.dim)?;
                let across = self.norm.forward(&out.narrow(2, self.dim, self.dim)?)?;
                layers.push(Tensor::cat(&[within, across], D::Minus1)?.narrow(1, 1, tokens_per_view - 1)?);
            }
        }
        Ok(BackboneOutput {
            layers: layers.try_into().expect("4 output layers"),
            camera_tokens: camera_tokens.expect("at least one output layer"),
            reference_view,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_positions_follow_the_grid() {
        let within = rope_positions(2, 3, false);
        assert_eq!(within[0], [0, 0]);
        assert_eq!(within[1], [1, 1]);
        assert_eq!(within[3], [1, 3]);
        assert_eq!(within[4], [2, 1]);
        let across = rope_positions(2, 3, true);
        assert!(across[1..].iter().all(|&p| p == [1, 1]));
    }
}
