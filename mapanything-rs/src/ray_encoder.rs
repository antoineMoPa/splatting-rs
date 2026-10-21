//! Encoder of known camera rays (UniCeption's DenseRepresentationEncoder, no positional
//! encoding): MapAnything adds its features to the image features when the intrinsics
//! are known, instead of guessing the camera.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, VarBuilder};

use crate::dinov2::{DIM, PATCH_SIZE};
use crate::vit_block::layer_norm;

/// 3 ray channels x 14 x 14 pixels per patch, then the residual blocks' widths.
const WIDTHS: [usize; 3] = [3 * PATCH_SIZE * PATCH_SIZE, 768, 1024];

fn conv(in_dim: usize, out_dim: usize, kernel: usize, vb: VarBuilder) -> Result<Conv2d> {
    let config = Conv2dConfig {
        padding: kernel / 2,
        ..Default::default()
    };
    candle_nn::conv2d(in_dim, out_dim, kernel, config, vb)
}

/// gelu(conv2(gelu(conv1(x))) + shortcut(x)), 3x3 convs, 1x1 shortcut.
struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    shortcut: Conv2d,
}

impl ResidualBlock {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: conv(in_dim, out_dim, 3, vb.pp("conv1"))?,
            conv2: conv(out_dim, out_dim, 3, vb.pp("conv2"))?,
            shortcut: conv(in_dim, out_dim, 1, vb.pp("shortcut"))?,
        })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv2.forward(&self.conv1.forward(xs)?.gelu_erf()?)?;
        (out + self.shortcut.forward(xs)?)?.gelu_erf()
    }
}

/// [batch, channels, h * r, w * r] -> [batch, channels * r * r, h, w], channel-major then
/// row then column within the block, as torch's PixelUnshuffle.
fn pixel_unshuffle(xs: &Tensor, factor: usize) -> Result<Tensor> {
    let (batch, channels, height, width) = xs.dims4()?;
    let (h, w) = (height / factor, width / factor);
    xs.reshape((batch, channels, h, factor, w, factor))?
        .permute((0, 1, 3, 5, 2, 4))?
        .reshape((batch, channels * factor * factor, h, w))
}

pub struct RayEncoder {
    conv_in: Conv2d,
    blocks: [ResidualBlock; 2],
    conv_out: Conv2d,
    norm: LayerNorm,
}

impl RayEncoder {
    /// `vb` at `ray_dirs_encoder`.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv_in: conv(WIDTHS[0], WIDTHS[0], 3, vb.pp("conv_in"))?,
            blocks: [
                ResidualBlock::new(WIDTHS[0], WIDTHS[1], vb.pp("encoder.0"))?,
                ResidualBlock::new(WIDTHS[1], WIDTHS[2], vb.pp("encoder.1"))?,
            ],
            conv_out: conv(WIDTHS[2], DIM, 1, vb.pp("encoder.2"))?,
            norm: layer_norm(DIM, vb.pp("norm_layer"))?,
        })
    }

    /// `rays`: [1, 3, H, W] unit ray directions. Returns patch features [1, patches, DIM].
    pub fn forward(&self, rays: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv_in.forward(&pixel_unshuffle(rays, PATCH_SIZE)?)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let xs = self.conv_out.forward(&xs)?.flatten_from(2)?.transpose(1, 2)?;
        self.norm.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn pixel_unshuffle_orders_channel_then_row_then_column() {
        // One channel, 2x4 image, factor 2: two 2x2 blocks.
        let xs = Tensor::arange(0f32, 8.0, &Device::Cpu).unwrap().reshape((1, 1, 2, 4)).unwrap();
        let out = pixel_unshuffle(&xs, 2).unwrap().squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        // Output channel (row i, column j within the block) holds that pixel of each block:
        // block 0 is [[0, 1], [4, 5]], block 1 is [[2, 3], [6, 7]].
        let expected = [[0.0, 2.0], [1.0, 3.0], [4.0, 6.0], [5.0, 7.0]];
        let expected: Vec<Vec<Vec<f32>>> = expected.iter().map(|pair| vec![pair.to_vec()]).collect();
        assert_eq!(out, expected);
    }
}
