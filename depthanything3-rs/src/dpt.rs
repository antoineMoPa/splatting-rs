//! The dense head (upstream's `DualDPT`): the 4 hooked token maps are projected, resampled
//! to a pyramid (4x, 2x, 1x, 1/2x of the patch grid) and fused top-down twice, by two
//! independent DPT fusion chains sharing the pyramid. The depth chain ends at the image
//! resolution with 2 channels: log depth and a confidence logit. The ray chain ends at 8x
//! the patch grid with 7 channels: a camera ray per pixel (direction and origin, 6) and a
//! confidence logit, which upstream's optional "ray pose" fits a camera to by RANSAC; this
//! port returns them raw and takes the cameras from the camera head instead.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm, VarBuilder};

use crate::uv_embed::uv_embedding;
use crate::variant::{PATCH_SIZE, VariantSpec};
use crate::vit_block::{TORCH_LAYER_NORM_EPS, layer_norm, layer_norm_without_affine};

/// Channels of both chains' last hidden layer (upstream's `head_features_2`).
const OUT_HIDDEN_DIM: usize = 32;
/// Log depth and confidence logit.
pub const DEPTH_CHANNELS: usize = 2;
/// Ray (6) and confidence logit.
pub const RAY_CHANNELS: usize = 7;
/// The ray chain's output is at this multiple of the patch grid.
pub const RAY_SCALE: usize = 8;
/// Pyramid scale of each hooked map relative to the patch grid, as (numerator, denominator).
const PYRAMID_SCALES: [(usize, usize); 4] = [(4, 1), (2, 1), (1, 1), (1, 2)];
/// The ray chain's pre-output stack: 5 convolutions alternating between features / 2 and
/// features, without activations.
const RAY_PRE_OUTPUT_CONVS: usize = 5;

fn conv(in_dim: usize, out_dim: usize, kernel: usize, padding: usize, vb: VarBuilder) -> Result<Conv2d> {
    let config = Conv2dConfig {
        padding,
        ..Default::default()
    };
    candle_nn::conv2d(in_dim, out_dim, kernel, config, vb)
}

/// `F.interpolate(mode="bilinear", align_corners=True)`.
fn resize_bilinear(xs: &Tensor, (rows, columns): (usize, usize)) -> Result<Tensor> {
    xs.upsample_bilinear2d(rows, columns, true)
}

fn spatial_size(xs: &Tensor) -> Result<(usize, usize)> {
    let (_, _, rows, columns) = xs.dims4()?;
    Ok((rows, columns))
}

/// Brings one hooked token map to its pyramid scale.
enum Resample {
    Up(ConvTranspose2d),
    Identity,
    Down(Conv2d),
}

impl Resample {
    fn new(hook: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(match PYRAMID_SCALES[hook] {
            (factor, 1) if factor > 1 => {
                let config = ConvTranspose2dConfig {
                    stride: factor,
                    ..Default::default()
                };
                Resample::Up(candle_nn::conv_transpose2d(dim, dim, factor, config, vb)?)
            }
            (1, 1) => Resample::Identity,
            (1, 2) => {
                let config = Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                };
                Resample::Down(candle_nn::conv2d(dim, dim, 3, config, vb)?)
            }
            scale => panic!("unsupported pyramid scale {scale:?}"),
        })
    }
}

impl Module for Resample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Resample::Up(layer) => layer.forward(xs),
            Resample::Identity => Ok(xs.clone()),
            Resample::Down(layer) => layer.forward(xs),
        }
    }
}

/// conv2(relu(conv1(relu(x)))) + x
struct ResidualConvUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualConvUnit {
    fn new(features: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: conv(features, features, 3, 1, vb.pp("conv1"))?,
            conv2: conv(features, features, 3, 1, vb.pp("conv2"))?,
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
    fn new(features: usize, has_skip: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            skip_unit: has_skip
                .then(|| ResidualConvUnit::new(features, vb.pp("resConfUnit1")))
                .transpose()?,
            unit: ResidualConvUnit::new(features, vb.pp("resConfUnit2"))?,
            out_conv: conv(features, features, 1, 0, vb.pp("out_conv"))?,
        })
    }

    fn forward(&self, xs: &Tensor, skip: Option<&Tensor>, out_size: (usize, usize)) -> Result<Tensor> {
        let xs = match (skip, &self.skip_unit) {
            (Some(skip), Some(unit)) => (xs + unit.forward(skip)?)?,
            (None, None) => xs.clone(),
            _ => panic!("skip input and skip unit must come together"),
        };
        self.out_conv
            .forward(&resize_bilinear(&self.unit.forward(&xs)?, out_size)?)
    }
}

/// Top-down fusion of the 4-level pyramid; index 0 is the finest level (`refinenet1`).
struct FusionChain {
    refinenets: Vec<FeatureFusionBlock>,
}

impl FusionChain {
    /// `vb` at `scratch`; `suffix` is "" for the depth chain, "_aux" for the ray chain.
    fn new(features: usize, suffix: &str, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            refinenets: (1..=4)
                .map(|level| {
                    FeatureFusionBlock::new(features, level < 4, vb.pp(format!("refinenet{level}{suffix}")))
                })
                .collect::<Result<_>>()?,
        })
    }

    /// `pyramid`: finest first. Returns a map at twice the finest level's size.
    fn forward(&self, pyramid: &[Tensor; 4]) -> Result<Tensor> {
        let mut path = self.refinenets[3].forward(&pyramid[3], None, spatial_size(&pyramid[2])?)?;
        for level in (1..3).rev() {
            let out_size = spatial_size(&pyramid[level - 1])?;
            path = self.refinenets[level].forward(&path, Some(&pyramid[level]), out_size)?;
        }
        let (rows, columns) = spatial_size(&pyramid[0])?;
        self.refinenets[0].forward(&path, Some(&pyramid[0]), (2 * rows, 2 * columns))
    }
}

pub struct DenseOutput {
    /// [views, DEPTH_CHANNELS, H, W]: log depth, confidence logit.
    pub depth_logits: Tensor,
    /// [views, RAY_CHANNELS, RAY_SCALE * rows, RAY_SCALE * columns]: ray, confidence logit.
    pub ray_logits: Tensor,
}

pub struct DualDptHead {
    norm: LayerNorm,
    projections: Vec<Conv2d>,
    resamplers: Vec<Resample>,
    to_features: Vec<Conv2d>,
    depth_chain: FusionChain,
    depth_pre_output: Conv2d,
    depth_output: [Conv2d; 2],
    ray_chain: FusionChain,
    ray_pre_output: Vec<Conv2d>,
    ray_output_conv: Conv2d,
    ray_output_norm: LayerNorm,
    ray_output: Conv2d,
}

impl DualDptHead {
    /// `vb` at `head`.
    pub fn new(spec: &VariantSpec, vb: VarBuilder) -> Result<Self> {
        let input_dim = spec.head_input_dim();
        let features = spec.head_features;
        let scratch = vb.pp("scratch");
        let no_bias_3x3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let ray_pre_output = (0..RAY_PRE_OUTPUT_CONVS)
            .map(|index| {
                let (in_dim, out_dim) = if index % 2 == 0 {
                    (features, features / 2)
                } else {
                    (features / 2, features)
                };
                conv(in_dim, out_dim, 3, 1, scratch.pp(format!("output_conv1_aux.3.{index}")))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            norm: layer_norm(input_dim, TORCH_LAYER_NORM_EPS, vb.pp("norm"))?,
            projections: (0..4)
                .map(|hook| conv(input_dim, spec.head_stage_dims[hook], 1, 0, vb.pp(format!("projects.{hook}"))))
                .collect::<Result<_>>()?,
            resamplers: (0..4)
                .map(|hook| Resample::new(hook, spec.head_stage_dims[hook], vb.pp(format!("resize_layers.{hook}"))))
                .collect::<Result<_>>()?,
            to_features: (0..4)
                .map(|hook| {
                    candle_nn::conv2d_no_bias(
                        spec.head_stage_dims[hook],
                        features,
                        3,
                        no_bias_3x3,
                        scratch.pp(format!("layer{}_rn", hook + 1)),
                    )
                })
                .collect::<Result<_>>()?,
            depth_chain: FusionChain::new(features, "", scratch.clone())?,
            depth_pre_output: conv(features, features / 2, 3, 1, scratch.pp("output_conv1"))?,
            depth_output: [
                conv(features / 2, OUT_HIDDEN_DIM, 3, 1, scratch.pp("output_conv2.0"))?,
                conv(OUT_HIDDEN_DIM, DEPTH_CHANNELS, 1, 0, scratch.pp("output_conv2.2"))?,
            ],
            ray_chain: FusionChain::new(features, "_aux", scratch.clone())?,
            ray_pre_output,
            ray_output_conv: conv(features / 2, OUT_HIDDEN_DIM, 3, 1, scratch.pp("output_conv2_aux.3.0"))?,
            // The checkpoint has no weight or bias for this LayerNorm (only the unused
            // level 0's), so upstream runs it at PyTorch's initialization.
            ray_output_norm: layer_norm_without_affine(OUT_HIDDEN_DIM, TORCH_LAYER_NORM_EPS, &scratch)?,
            ray_output: conv(OUT_HIDDEN_DIM, RAY_CHANNELS, 1, 0, scratch.pp("output_conv2_aux.3.5"))?,
        })
    }

    /// `xs` + the sinusoidal embedding of its pixels' positions on the `(height, width)` image.
    fn add_uv_embedding(xs: &Tensor, (height, width): (usize, usize)) -> Result<Tensor> {
        let (_, channels, rows, columns) = xs.dims4()?;
        let embedding = uv_embedding(
            channels,
            (rows, columns),
            width as f64 / height as f64,
            xs.device(),
            xs.dtype(),
        )?;
        xs.broadcast_add(&embedding)
    }

    /// `layers`: the backbone's 4 output layers [views, rows * columns, 2 * dim], for an
    /// image of `rows` x `columns` patches.
    /// The head over every view, one view at a time: each view's maps are independent of the
    /// others', and the full-resolution activations of all views at once (gigabytes on the
    /// Large and Giant variants) do not fit in wasm32's 4 GB memory.
    pub fn forward(&self, layers: &[Tensor; 4], grid: (usize, usize)) -> Result<DenseOutput> {
        let views = layers[0].dim(0)?;
        let (mut depth, mut rays) = (Vec::with_capacity(views), Vec::with_capacity(views));
        for view in 0..views {
            let one_view: Vec<Tensor> = layers.iter().map(|layer| layer.narrow(0, view, 1)).collect::<Result<_>>()?;
            let output = self.forward_views(&one_view.try_into().expect("4 layers"), grid)?;
            depth.push(output.depth_logits);
            rays.push(output.ray_logits);
        }
        Ok(DenseOutput {
            depth_logits: Tensor::cat(&depth, 0)?,
            ray_logits: Tensor::cat(&rays, 0)?,
        })
    }

    fn forward_views(&self, layers: &[Tensor; 4], (rows, columns): (usize, usize)) -> Result<DenseOutput> {
        let image_size = (rows * PATCH_SIZE, columns * PATCH_SIZE);
        let mut pyramid = Vec::with_capacity(4);
        for (hook, tokens) in layers.iter().enumerate() {
            let (views, patches, dim) = tokens.dims3()?;
            assert_eq!(patches, rows * columns, "layer {hook} token count");
            let spatial = self
                .norm
                .forward(tokens)?
                .transpose(1, 2)?
                .reshape((views, dim, rows, columns))?;
            let projected = Self::add_uv_embedding(&self.projections[hook].forward(&spatial)?, image_size)?;
            pyramid.push(self.to_features[hook].forward(&self.resamplers[hook].forward(&projected)?)?);
        }
        let pyramid: [Tensor; 4] = pyramid.try_into().expect("4 levels");

        let fused = self.depth_pre_output.forward(&self.depth_chain.forward(&pyramid)?)?;
        let fused = Self::add_uv_embedding(&resize_bilinear(&fused, image_size)?, image_size)?;
        let depth_logits = self
            .depth_output[1]
            .forward(&self.depth_output[0].forward(&fused)?.relu()?)?;

        let mut rays = self.ray_chain.forward(&pyramid)?;
        for layer in &self.ray_pre_output {
            rays = layer.forward(&rays)?;
        }
        let rays = self
            .ray_output_conv
            .forward(&Self::add_uv_embedding(&rays, image_size)?)?;
        // The LayerNorm is over the channels: [views, C, h, w] -> [views, h, w, C] and back.
        let rays = self
            .ray_output_norm
            .forward(&rays.permute((0, 2, 3, 1))?.contiguous()?)?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        let ray_logits = self.ray_output.forward(&rays.relu()?)?;
        Ok(DenseOutput {
            depth_logits,
            ray_logits,
        })
    }
}
