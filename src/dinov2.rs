//! DINOv2 ViT-giant image encoder, truncated to its first ENCODER_DEPTH blocks.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use crate::vit_block::VitBlock;

pub const PATCH_SIZE: usize = 14;
pub const DIM: usize = 1536;
const ENCODER_DEPTH: usize = 24;
/// The positional embedding is trained on a TRAINED_GRID x TRAINED_GRID patch grid (518 px).
const TRAINED_GRID: usize = 37;
/// DINOv2 resizes its positional embedding with scale factor (out + this) / TRAINED_GRID,
/// not with the output size (a historical guard against floating point error).
const INTERPOLATE_OFFSET: f64 = 0.1;
/// PyTorch's bicubic convolution coefficient.
const BICUBIC_A: f64 = -0.75;

/// PyTorch's 4 bicubic weights for a sample at fraction `t` past its floor.
fn bicubic_weights(t: f64) -> [f64; 4] {
    let a = BICUBIC_A;
    let near = |x: f64| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let far = |x: f64| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

/// For each output index along one axis: the 4 clamped input indices and their weights,
/// as `F.interpolate(mode="bicubic", align_corners=False, scale_factor=scale)`.
fn bicubic_taps(input: usize, output: usize, scale: f64) -> Vec<([usize; 4], [f64; 4])> {
    (0..output)
        .map(|out| {
            let source = (out as f64 + 0.5) / scale - 0.5;
            let floor = source.floor();
            let indices = [-1i64, 0, 1, 2]
                .map(|offset| (floor as i64 + offset).clamp(0, input as i64 - 1) as usize);
            (indices, bicubic_weights(source - floor))
        })
        .collect()
}

/// Bicubic resize of a row-major [rows, columns, channels] grid.
fn resize_grid(
    grid: &[f32],
    (rows, columns, channels): (usize, usize, usize),
    (out_rows, out_columns): (usize, usize),
    (row_scale, column_scale): (f64, f64),
) -> Vec<f32> {
    let column_taps = bicubic_taps(columns, out_columns, column_scale);
    let mut horizontal = vec![0f64; rows * out_columns * channels];
    for row in 0..rows {
        for (out_column, (indices, weights)) in column_taps.iter().enumerate() {
            let target = (row * out_columns + out_column) * channels;
            for (&column, &weight) in indices.iter().zip(weights) {
                let source = (row * columns + column) * channels;
                for channel in 0..channels {
                    horizontal[target + channel] += weight * grid[source + channel] as f64;
                }
            }
        }
    }
    let row_taps = bicubic_taps(rows, out_rows, row_scale);
    let mut resized = vec![0f32; out_rows * out_columns * channels];
    for (out_row, (indices, weights)) in row_taps.iter().enumerate() {
        for out_column in 0..out_columns {
            let target = (out_row * out_columns + out_column) * channels;
            for channel in 0..channels {
                resized[target + channel] = indices
                    .iter()
                    .zip(weights)
                    .map(|(&row, &weight)| {
                        weight * horizontal[(row * out_columns + out_column) * channels + channel]
                    })
                    .sum::<f64>() as f32;
            }
        }
    }
    resized
}

pub struct Dinov2 {
    patch_embed: Conv2d,
    cls_token: Tensor,
    /// [1, 1 + TRAINED_GRID^2, DIM], on the CPU in f32 for resizing.
    pos_embed: Tensor,
    blocks: Vec<VitBlock>,
}

impl Dinov2 {
    /// `vb` at the DINOv2 model (`encoder.model`).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let patch_config = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };
        Ok(Self {
            patch_embed: candle_nn::conv2d(
                3,
                DIM,
                PATCH_SIZE,
                patch_config,
                vb.pp("patch_embed.proj"),
            )?,
            cls_token: vb.get((1, 1, DIM), "cls_token")?,
            pos_embed: vb
                .get((1, 1 + TRAINED_GRID * TRAINED_GRID, DIM), "pos_embed")?
                .to_dtype(candle_core::DType::F32)?
                .to_device(&Device::Cpu)?,
            blocks: (0..ENCODER_DEPTH)
                .map(|index| VitBlock::new(DIM, vb.pp(format!("blocks.{index}"))))
                .collect::<Result<_>>()?,
        })
    }

    /// [1, 1 + rows * columns, DIM]: the class embedding, then the patch grid's.
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
            (TRAINED_GRID, TRAINED_GRID, DIM),
            (rows, columns),
            (scale(rows), scale(columns)),
        );
        Tensor::cat(
            &[
                class,
                Tensor::from_vec(resized, (1, rows * columns, DIM), &Device::Cpu)?,
            ],
            1,
        )
    }

    /// `image`: [1, 3, H, W], DINOv2-normalized, H and W multiples of PATCH_SIZE.
    /// Returns [1, 1 + patches, DIM] after the last block, without a final norm.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let (_, _, height, width) = image.dims4()?;
        assert!(
            height % PATCH_SIZE == 0 && width % PATCH_SIZE == 0,
            "image {width}x{height} is not a multiple of the {PATCH_SIZE} px patch"
        );
        let patches = self
            .patch_embed
            .forward(image)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let positional = self
            .positional_embedding(height / PATCH_SIZE, width / PATCH_SIZE)?
            .to_dtype(image.dtype())?
            .to_device(image.device())?;
        let mut xs = (Tensor::cat(&[&self.cls_token, &patches], 1)? + positional)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_weights_sum_to_one() {
        for t in [0.0, 0.25, 0.5, 0.9] {
            assert!((bicubic_weights(t).iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert_eq!(bicubic_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn resize_to_same_size_with_unit_scale_is_identity() {
        let grid: Vec<f32> = (0..2 * 3 * 2).map(|v| v as f32).collect();
        let resized = resize_grid(&grid, (2, 3, 2), (2, 3), (1.0, 1.0));
        assert_eq!(resized, grid);
    }
}
