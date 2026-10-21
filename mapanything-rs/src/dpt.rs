//! DPT (Dense Prediction Transformer) head: fuses 4 token maps into a map of raw (before
//! activation) outputs at input resolution: ray direction (3), depth along the ray (1),
//! confidence (1), mask logit (1).

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder};

use crate::dinov2::{DIM, PATCH_SIZE};

pub const OUTPUT_CHANNELS: usize = 6;
const FEATURE_DIM: usize = 256;
const HEAD_HIDDEN_DIM: usize = FEATURE_DIM / 2;
/// Channel count of each hooked token map after its 1x1 projection.
const LAYER_DIMS: [usize; 4] = [96, 192, 384, 768];

fn conv(in_dim: usize, out_dim: usize, kernel: usize, padding: usize, vb: VarBuilder) -> Result<Conv2d> {
    let config = Conv2dConfig {
        padding,
        ..Default::default()
    };
    candle_nn::conv2d(in_dim, out_dim, kernel, config, vb)
}

fn upsample_2x(xs: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = xs.dims4()?;
    xs.upsample_bilinear2d(2 * h, 2 * w, true)
}

/// Brings one hooked token map to its pyramid scale: 4x up, 2x up, 1x, 2x down.
enum Resample {
    Up(ConvTranspose2d),
    Identity,
    Down(Conv2d),
}

/// 1x1 projection, resampling, then a 3x3 projection to FEATURE_DIM without bias.
struct HookAdapter {
    project: Conv2d,
    resample: Resample,
    to_features: Conv2d,
}

impl HookAdapter {
    /// `vb` at `input_process.{hook}`.
    fn new(hook: usize, vb: VarBuilder) -> Result<Self> {
        let dim = LAYER_DIMS[hook];
        let resample_vb = vb.pp("0.1");
        let up = |factor: usize| -> Result<Resample> {
            let config = ConvTranspose2dConfig {
                stride: factor,
                ..Default::default()
            };
            let layer = candle_nn::conv_transpose2d(dim, dim, factor, config, resample_vb.clone())?;
            Ok(Resample::Up(layer))
        };
        let resample = match hook {
            0 => up(4)?,
            1 => up(2)?,
            2 => Resample::Identity,
            3 => {
                let config = Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                };
                Resample::Down(candle_nn::conv2d(dim, dim, 3, config, resample_vb)?)
            }
            _ => panic!("DPT has 4 hooks, got hook {hook}"),
        };
        let no_bias_3x3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            project: conv(DIM, dim, 1, 0, vb.pp("0.0"))?,
            resample,
            to_features: candle_nn::conv2d_no_bias(dim, FEATURE_DIM, 3, no_bias_3x3, vb.pp("1"))?,
        })
    }
}

impl Module for HookAdapter {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.project.forward(xs)?;
        let xs = match &self.resample {
            Resample::Up(layer) => layer.forward(&xs)?,
            Resample::Identity => xs,
            Resample::Down(layer) => layer.forward(&xs)?,
        };
        self.to_features.forward(&xs)
    }
}

struct ResidualConvUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualConvUnit {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: conv(FEATURE_DIM, FEATURE_DIM, 3, 1, vb.pp("conv1"))?,
            conv2: conv(FEATURE_DIM, FEATURE_DIM, 3, 1, vb.pp("conv2"))?,
        })
    }
}

impl Module for ResidualConvUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv1.forward(&xs.relu()?)?;
        let out = self.conv2.forward(&out.relu()?)?;
        out + xs
    }
}

struct FeatureFusionBlock {
    /// None for the coarsest level, which has no skip input.
    skip_unit: Option<ResidualConvUnit>,
    unit: ResidualConvUnit,
    out_conv: Conv2d,
}

impl FeatureFusionBlock {
    fn new(has_skip: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            skip_unit: has_skip.then(|| ResidualConvUnit::new(vb.pp("resConfUnit1"))).transpose()?,
            unit: ResidualConvUnit::new(vb.pp("resConfUnit2"))?,
            out_conv: conv(FEATURE_DIM, FEATURE_DIM, 1, 0, vb.pp("out_conv"))?,
        })
    }

    fn forward(&self, xs: &Tensor, skip: Option<&Tensor>) -> Result<Tensor> {
        let xs = match (skip, &self.skip_unit) {
            (Some(skip), Some(unit)) => (xs + unit.forward(skip)?)?,
            (None, None) => xs.clone(),
            _ => panic!("skip input and skip unit must come together"),
        };
        self.out_conv.forward(&upsample_2x(&self.unit.forward(&xs)?)?)
    }
}

pub struct DptHead {
    hook_adapters: Vec<HookAdapter>,
    /// Index 0 is the finest level.
    refinenets: Vec<FeatureFusionBlock>,
    head_conv1: Conv2d,
    head_conv2: Conv2d,
    head_conv3: Conv2d,
}

impl DptHead {
    /// `vb` at `dense_head`.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let features = vb.pp("0");
        let regressor = vb.pp("1");
        Ok(Self {
            hook_adapters: (0..4)
                .map(|hook| HookAdapter::new(hook, features.pp(format!("input_process.{hook}"))))
                .collect::<Result<_>>()?,
            refinenets: (1..=4)
                .map(|level| {
                    let vb = features.pp(format!("scratch.refinenet{level}"));
                    FeatureFusionBlock::new(level < 4, vb)
                })
                .collect::<Result<_>>()?,
            head_conv1: conv(FEATURE_DIM, HEAD_HIDDEN_DIM, 3, 1, regressor.pp("conv1"))?,
            head_conv2: conv(HEAD_HIDDEN_DIM, HEAD_HIDDEN_DIM, 3, 1, regressor.pp("conv2.0"))?,
            head_conv3: conv(HEAD_HIDDEN_DIM, OUTPUT_CHANNELS, 1, 0, regressor.pp("conv2.2"))?,
        })
    }

    /// `features`: 4 token maps [batch, patch_h * patch_w, DIM].
    /// Returns [batch, OUTPUT_CHANNELS, patch_h * PATCH_SIZE, patch_w * PATCH_SIZE].
    pub fn forward(&self, features: &[Tensor; 4], patch_h: usize, patch_w: usize) -> Result<Tensor> {
        let pyramid: Vec<Tensor> = features
            .iter()
            .zip(&self.hook_adapters)
            .map(|(tokens, adapter)| {
                let (batch, _, dim) = tokens.dims3()?;
                let spatial = tokens.transpose(1, 2)?.reshape((batch, dim, patch_h, patch_w))?;
                adapter.forward(&spatial)
            })
            .collect::<Result<_>>()?;

        // Coarsest level first; an odd patch grid makes the 2x-upsampled map one row/col too big.
        let mut path = self.refinenets[3].forward(&pyramid[3], None)?;
        for level in (0..3).rev() {
            let (_, _, h, w) = pyramid[level].dims4()?;
            path = path.narrow(2, 0, h)?.narrow(3, 0, w)?;
            path = self.refinenets[level].forward(&path, Some(&pyramid[level]))?;
        }

        // Bilinear to the input resolution (8 x 14 / 8 = 1.75x), then the regressor convs.
        let out = self
            .head_conv1
            .forward(&path)?
            .upsample_bilinear2d(patch_h * PATCH_SIZE, patch_w * PATCH_SIZE, true)?;
        self.head_conv3.forward(&self.head_conv2.forward(&out)?.relu()?)
    }
}
