//! MapAnything's camera pose head and metric scale head.

use candle_core::{D, Module, Result, Tensor};
use candle_nn::{Conv2d, Linear, VarBuilder};

use crate::dinov2::{DIM, PATCH_SIZE};

/// Pose head width: 4 x 14^2.
const POSE_DIM: usize = 4 * PATCH_SIZE * PATCH_SIZE;
const POSE_RESIDUAL_BLOCKS: usize = 2;
/// Scale head width: 14^2.
const SCALE_DIM: usize = PATCH_SIZE * PATCH_SIZE;
const SCALE_HIDDEN_LAYERS: usize = 2;
const QUATERNION_MIN_NORM: f64 = 1e-8;
const MIN_SCALE: f64 = 1e-8;

fn conv1x1(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Conv2d> {
    candle_nn::conv2d(in_dim, out_dim, 1, Default::default(), vb)
}

/// relu(conv3(relu(conv2(relu(conv1(x)))))) + x, all 1x1.
struct ResidualBlock {
    convs: [Conv2d; 3],
}

impl Module for ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut residual = xs.clone();
        for conv in &self.convs {
            residual = conv.forward(&residual)?.relu()?;
        }
        xs + residual
    }
}

/// Camera-to-world pose of one view, OpenCV convention, before metric scaling.
pub struct RawPose {
    /// [1, 3]
    pub translation: Tensor,
    /// [1, 4] unit quaternion (x, y, z, w).
    pub quaternion: Tensor,
}

pub struct PoseHead {
    projection: Conv2d,
    residual_blocks: Vec<ResidualBlock>,
    mlp: [Linear; 2],
    translation: Linear,
    rotation: Linear,
}

impl PoseHead {
    /// `vb` at `pose_head`.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let residual_blocks = (0..POSE_RESIDUAL_BLOCKS)
            .map(|block| {
                let vb = vb.pp(format!("res_conv.{block}"));
                Ok(ResidualBlock {
                    convs: [
                        conv1x1(POSE_DIM, POSE_DIM, vb.pp("res_conv1"))?,
                        conv1x1(POSE_DIM, POSE_DIM, vb.pp("res_conv2"))?,
                        conv1x1(POSE_DIM, POSE_DIM, vb.pp("res_conv3"))?,
                    ],
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            projection: conv1x1(DIM, POSE_DIM, vb.pp("proj"))?,
            residual_blocks,
            mlp: [
                candle_nn::linear(POSE_DIM, POSE_DIM, vb.pp("more_mlps.0"))?,
                candle_nn::linear(POSE_DIM, POSE_DIM, vb.pp("more_mlps.2"))?,
            ],
            translation: candle_nn::linear(POSE_DIM, 3, vb.pp("fc_t"))?,
            rotation: candle_nn::linear(POSE_DIM, 4, vb.pp("fc_rot"))?,
        })
    }

    /// `features`: the view's final shared features [1, DIM, h, w].
    pub fn forward(&self, features: &Tensor) -> Result<RawPose> {
        let mut xs = self.projection.forward(features)?;
        for block in &self.residual_blocks {
            xs = block.forward(&xs)?;
        }
        let mut pooled = xs.flatten_from(2)?.mean(D::Minus1)?;
        for layer in &self.mlp {
            pooled = layer.forward(&pooled)?.relu()?;
        }
        let quaternion = self.rotation.forward(&pooled)?;
        let norm = quaternion
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(QUATERNION_MIN_NORM, f64::INFINITY)?;
        Ok(RawPose {
            translation: self.translation.forward(&pooled)?,
            quaternion: quaternion.broadcast_div(&norm)?,
        })
    }
}

/// Metric scale shared by every view, from the global scale token.
pub struct ScaleHead {
    projection: Linear,
    hidden: Vec<Linear>,
    output: Linear,
}

impl ScaleHead {
    /// `vb` at `scale_head`.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            projection: candle_nn::linear(DIM, SCALE_DIM, vb.pp("proj"))?,
            hidden: (0..SCALE_HIDDEN_LAYERS)
                .map(|layer| candle_nn::linear(SCALE_DIM, SCALE_DIM, vb.pp(format!("mlp.{layer}.0"))))
                .collect::<Result<_>>()?,
            output: candle_nn::linear(SCALE_DIM, 1, vb.pp("output_proj"))?,
        })
    }

    /// `scale_token`: [1, DIM]. Returns the scale as a [1] tensor, exp of the output.
    pub fn forward(&self, scale_token: &Tensor) -> Result<Tensor> {
        // No activation after the projection.
        let mut xs = self.projection.forward(scale_token)?;
        for layer in &self.hidden {
            xs = layer.forward(&xs)?.relu()?;
        }
        self.output.forward(&xs)?.exp()?.clamp(MIN_SCALE, f64::INFINITY)?.flatten_all()
    }
}
