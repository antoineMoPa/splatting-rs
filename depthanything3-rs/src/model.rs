//! The full Depth Anything 3 model: images in; per view a depth map, its confidence, a
//! camera-to-world pose and intrinsics out. The backbone (`backbone.rs`) sees all views at
//! once, the dense head (`dpt.rs`) predicts depth and confidence per view, the camera head
//! (`camera_head.rs`) the pose and field of view from each view's camera token.
//!
//! Depth is relative, not metric: upstream scales it to the median camera distance of
//! given extrinsics, and this port has no such input. Poses are relative to the reference
//! view (`reference_view.rs`), whose pose comes out near the identity.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use nalgebra::{Matrix3, Vector3};

use crate::backbone::{Backbone, BackboneOutput};
use crate::camera_head::{CameraEstimate, CameraHead, POSE_ENCODING_DIM};
use crate::dpt::{DEPTH_CHANNELS, DualDptHead};
use crate::reference_view::ReferenceViewStrategy;
use crate::variant::{PATCH_SIZE, Variant};

/// confidence = CONF_MIN + exp(logit)
const CONF_MIN: f32 = 1.0;

/// A stage of the forward pass, reported to a `Progress` callback as it starts (the browser
/// shows the stage running now; on the larger variants the depth head takes minutes on one
/// CPU thread, so it must be announced before it runs, not after).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Backbone block `0..depth` of the variant.
    Block(usize),
    DenseHead,
    CameraHead,
}

pub type Progress<'a> = &'a mut dyn FnMut(Stage);

/// Every tensor the model predicts, on the device, for comparison with upstream.
pub struct RawPrediction {
    /// [views, H, W], exp of the logit.
    pub depth: Tensor,
    /// [views, H, W], 1 + exp of the logit.
    pub depth_conf: Tensor,
    /// [views, RAY_CHANNELS, 8 rows, 8 columns] raw ray-head output.
    pub ray_logits: Tensor,
    /// [views, POSE_ENCODING_DIM], f32.
    pub pose_encoding: Tensor,
    pub reference_view: usize,
    /// What the heads saw.
    pub backbone: BackboneOutput,
}

/// One view's prediction, on the CPU.
pub struct ViewOutput {
    pub height: usize,
    pub width: usize,
    /// Row-major depth (z in the camera frame) per pixel.
    pub depth: Vec<f32>,
    /// Row-major, 1 + exp(logit), always > 1.
    pub conf: Vec<f32>,
    /// Camera to world, the world being the reference view's camera frame (OpenCV: x
    /// right, y down, z forward).
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// (fx, fy) in pixels of the model input.
    pub focal: [f64; 2],
    /// (cx, cy): the image center, pixel (i, j) at integer coordinates as upstream.
    pub principal_point: [f64; 2],
}

pub struct Prediction {
    pub views: Vec<ViewOutput>,
    /// The view whose camera frame is the world of every pose.
    pub reference_view: usize,
}

pub struct DepthAnything3 {
    variant: Variant,
    backbone: Backbone,
    dense_head: DualDptHead,
    camera_head: CameraHead,
    reference_view_strategy: ReferenceViewStrategy,
}

impl DepthAnything3 {
    /// `vb` over the checkpoint at its `model.` prefix.
    pub fn new(variant: Variant, reference_view_strategy: ReferenceViewStrategy, vb: VarBuilder) -> Result<Self> {
        let spec = variant.spec();
        Ok(Self {
            variant,
            backbone: Backbone::new(spec, vb.pp("backbone.pretrained"))?,
            dense_head: DualDptHead::new(spec, vb.pp("head"))?,
            camera_head: CameraHead::new(spec, vb.pp("cam_dec"))?,
            reference_view_strategy,
        })
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// `images`: [views, 3, H, W], ImageNet-normalized, H and W multiples of PATCH_SIZE.
    pub fn forward_raw(&self, images: &Tensor) -> Result<RawPrediction> {
        self.forward_raw_with_progress(images, &mut |_| {})
    }

    pub fn forward_raw_with_progress(&self, images: &Tensor, progress: Progress) -> Result<RawPrediction> {
        let (_, _, height, width) = images.dims4()?;
        let grid = (height / PATCH_SIZE, width / PATCH_SIZE);
        let features = self.backbone.forward(images, self.reference_view_strategy, progress)?;
        progress(Stage::DenseHead);
        let dense = self.dense_head.forward(&features.layers, grid)?;
        progress(Stage::CameraHead);
        let pose_encoding = self.camera_head.forward(&features.camera_tokens)?;
        let logits = dense.depth_logits.to_dtype(DType::F32)?;
        assert_eq!(logits.dim(1)?, DEPTH_CHANNELS);
        Ok(RawPrediction {
            depth: logits.narrow(1, 0, 1)?.squeeze(1)?.exp()?,
            depth_conf: (logits.narrow(1, 1, 1)?.squeeze(1)?.exp()? + CONF_MIN as f64)?,
            ray_logits: dense.ray_logits,
            pose_encoding,
            reference_view: features.reference_view,
            backbone: features,
        })
    }

    pub fn forward(&self, images: &Tensor) -> Result<Prediction> {
        self.forward_with_progress(images, &mut |_| {})
    }

    pub fn forward_with_progress(&self, images: &Tensor, progress: Progress) -> Result<Prediction> {
        let (view_count, _, height, width) = images.dims4()?;
        let raw = self.forward_raw_with_progress(images, progress)?;
        let encodings = raw.pose_encoding.to_vec2::<f32>()?;
        assert_eq!(encodings.len(), view_count);
        let views = (0..view_count)
            .map(|view| {
                let camera = CameraEstimate::from_pose_encoding(&encodings[view][..POSE_ENCODING_DIM]);
                let depth: Vec<f32> = raw.depth.get(view)?.flatten_all()?.to_vec1()?;
                let conf: Vec<f32> = raw.depth_conf.get(view)?.flatten_all()?.to_vec1()?;
                assert!(
                    depth.iter().chain(&conf).all(|v| v.is_finite()) && depth.iter().all(|&d| d > 0.0),
                    "view {view}: depth or confidence is not finite and positive"
                );
                assert!(
                    camera.translation.iter().all(|v| v.is_finite()) && camera.fov.iter().all(|v| v.is_finite()),
                    "view {view}: camera is not finite"
                );
                Ok(ViewOutput {
                    height,
                    width,
                    depth,
                    conf,
                    rotation: camera.rotation,
                    translation: camera.translation,
                    focal: camera.focal((height, width)),
                    principal_point: CameraEstimate::principal_point((height, width)),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Prediction {
            views,
            reference_view: raw.reference_view,
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::reference_view::ReferenceViewStrategy;
    use crate::weights::load_model;

    /// Loads the DA3-SMALL checkpoint (every tensor's shape is checked as it is read) and
    /// runs 3 synthetic views on the CPU: finite depth, confidence and cameras, with the
    /// principal point at the image center. Downloads the 137 MB weights on first run.
    #[test]
    fn small_weights_load_and_a_synthetic_forward_is_finite() {
        let device = Device::Cpu;
        let loaded =
            load_model(Variant::Small, ReferenceViewStrategy::SaddleBalanced, None, DType::F32, &device).expect("DA3-SMALL weights");
        let (views, height, width) = (3, 4 * PATCH_SIZE, 3 * PATCH_SIZE);
        // A smooth gradient per view, shifted between views, in ImageNet-normalized range.
        let mut pixels = vec![0f32; views * 3 * height * width];
        for (index, value) in pixels.iter_mut().enumerate() {
            let view = index / (3 * height * width);
            let x = index % width;
            *value = (x + view) as f32 / width as f32 * 2.0 - 1.0;
        }
        let images = Tensor::from_vec(pixels, (views, 3, height, width), &device).unwrap();
        let prediction = loaded.model.forward(&images).expect("forward");
        assert_eq!(prediction.views.len(), views);
        for view in &prediction.views {
            assert_eq!(view.depth.len(), height * width);
            assert!(view.conf.iter().all(|&c| c >= CONF_MIN));
            assert_eq!(view.principal_point, [width as f64 / 2.0, height as f64 / 2.0]);
            assert!(view.focal.iter().all(|&f| f > 0.0));
            assert!((view.rotation.determinant() - 1.0).abs() < 1e-9);
        }
        // The reference view's pose is near the identity.
        let reference = &prediction.views[prediction.reference_view];
        assert!(reference.translation.norm() < 0.1, "{}", reference.translation);
    }
}
