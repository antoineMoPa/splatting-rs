//! The full MapAnything model. Images and, when known, camera intrinsics in; per view a
//! metric point map, confidence, ambiguity mask and camera pose out. The optional depth and
//! pose inputs are not supported.

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

use crate::dinov2::{DIM, Dinov2, PATCH_SIZE};
use crate::dpt::DptHead;
use crate::heads::{PoseHead, ScaleHead};
use crate::image_input::Intrinsics;
use crate::info_sharing::{AlternatingAttention, ViewTokens};
use crate::ray_encoder::RayEncoder;
use crate::vit_block::layer_norm;

const RAY_MIN_NORM: f32 = 1e-8;
/// confidence = CONF_MIN + exp(raw)
const CONF_MIN: f32 = 1.0;

/// One view's reconstruction, metric (scale applied), on the CPU.
pub struct ViewOutput {
    pub height: usize,
    pub width: usize,
    /// Row-major, in the view's camera frame (OpenCV: x right, y down, z forward).
    pub points_in_camera: Vec<[f32; 3]>,
    /// 1 + exp(raw), always > 1.
    pub conf: Vec<f32>,
    /// Pixels the model considers unambiguous (sigmoid(mask logit) > 0.5).
    pub unambiguous: Vec<bool>,
    /// Camera to world, the world being roughly view 0's camera frame.
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

/// [1, 3, height, width] unit ray directions of a pinhole camera, pixel centers at integer
/// coordinates, as upstream's `get_rays_in_camera_frame`.
fn ray_map(intrinsics: &Intrinsics, height: usize, width: usize, like: &Tensor) -> Result<Tensor> {
    let mut rays = vec![0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let direction = [
                (x as f64 - intrinsics.principal_point[0]) / intrinsics.focal[0],
                (y as f64 - intrinsics.principal_point[1]) / intrinsics.focal[1],
                1.0,
            ];
            let norm = direction.iter().map(|c| c * c).sum::<f64>().sqrt();
            for (channel, component) in direction.iter().enumerate() {
                rays[(channel * height + y) * width + x] = (component / norm) as f32;
            }
        }
    }
    Tensor::from_vec(rays, (1, 3, height, width), like.device())?.to_dtype(like.dtype())
}

pub struct MapAnything {
    encoder: Dinov2,
    ray_encoder: RayEncoder,
    fusion_norm: LayerNorm,
    scale_token: Tensor,
    info_sharing: AlternatingAttention,
    dense_head: DptHead,
    pose_head: PoseHead,
    scale_head: ScaleHead,
}

impl MapAnything {
    /// `vb` over the facebook/map-anything-apache checkpoint.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: Dinov2::new(vb.pp("encoder.model"))?,
            // In f32, as upstream: its features overflow f16.
            ray_encoder: RayEncoder::new(vb.pp("ray_dirs_encoder").set_dtype(DType::F32))?,
            fusion_norm: layer_norm(DIM, vb.pp("fusion_norm_layer"))?,
            scale_token: vb.get(DIM, "scale_token")?,
            info_sharing: AlternatingAttention::new(vb.pp("info_sharing"))?,
            dense_head: DptHead::new(vb.pp("dense_head"))?,
            pose_head: PoseHead::new(vb.pp("pose_head"))?,
            scale_head: ScaleHead::new(vb.pp("scale_head"))?,
        })
    }

    /// `images`: each [1, 3, H, W], ImageNet-normalized, all the same size, H and W
    /// multiples of PATCH_SIZE. View 0 is the reference view. `intrinsics`: one per image,
    /// known cameras are given to the model instead of letting it guess.
    pub fn forward(&self, images: &[Tensor], intrinsics: &[Option<Intrinsics>]) -> Result<Vec<ViewOutput>> {
        assert_eq!(images.len(), intrinsics.len(), "one intrinsics entry per image");
        let (_, _, height, width) = images[0].dims4()?;
        let (patch_h, patch_w) = (height / PATCH_SIZE, width / PATCH_SIZE);
        let patch_count = patch_h * patch_w;

        let mut views = Vec::with_capacity(images.len());
        for (image, intrinsics) in images.iter().zip(intrinsics) {
            let tokens = self.encoder.forward(image)?;
            let image_patches = tokens.narrow(1, 1, patch_count)?;
            // Known cameras are added to the image features; unknown ones add nothing.
            let patches = match intrinsics {
                Some(intrinsics) => {
                    let rays = ray_map(intrinsics, height, width, image)?.to_dtype(DType::F32)?;
                    let ray_features = self.ray_encoder.forward(&rays)?;
                    (image_patches + ray_features.to_dtype(image.dtype())?)?
                }
                None => image_patches,
            };
            views.push(ViewTokens {
                patches: self.fusion_norm.forward(&patches)?,
                // The class token joins the transformer without the fusion norm.
                class: tokens.narrow(1, 0, 1)?,
            });
        }
        let scale_token = self.scale_token.reshape((1, 1, DIM))?;
        let shared = self.info_sharing.forward(&views, &scale_token)?;
        let scale = self
            .scale_head
            .forward(&shared.scale_token)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?[0];

        views
            .iter()
            .zip(&shared.per_view)
            .map(|(view, [after_7, after_11, last])| {
                let raw = self
                    .dense_head
                    .forward(
                        &[view.patches.clone(), after_7.clone(), after_11.clone(), last.clone()],
                        patch_h,
                        patch_w,
                    )?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?;
                let spatial = last.transpose(1, 2)?.reshape((1, DIM, patch_h, patch_w))?;
                let pose = self.pose_head.forward(&spatial)?;
                let translation = pose.translation.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let quaternion = pose.quaternion.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let [x, y, z, w] = [0, 1, 2, 3].map(|axis| quaternion[axis] as f64);
                let rotation = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z))
                    .to_rotation_matrix()
                    .into_inner();
                let translation = Vector3::from_fn(|axis, _| translation[axis] as f64 * scale as f64);
                dense_output(&raw, scale, rotation, translation)
            })
            .collect()
    }
}

/// Turns the dense head's [6, H, W] raw output into metric camera-frame points.
fn dense_output(raw: &Tensor, scale: f32, rotation: Matrix3<f64>, translation: Vector3<f64>) -> Result<ViewOutput> {
    let (_, height, width) = raw.dims3()?;
    let channel = |index: usize| -> Result<Vec<f32>> { raw.get(index)?.flatten_all()?.to_vec1() };
    let rays: Vec<Vec<f32>> = (0..3).map(channel).collect::<Result<_>>()?;
    let log_depth = channel(3)?;
    let raw_conf = channel(4)?;
    let mask_logit = channel(5)?;
    let points_in_camera = (0..height * width)
        .map(|pixel| {
            let ray = [rays[0][pixel], rays[1][pixel], rays[2][pixel]];
            let norm = ray.iter().map(|c| c * c).sum::<f32>().sqrt().max(RAY_MIN_NORM);
            let depth = log_depth[pixel].exp() * scale;
            ray.map(|c| c / norm * depth)
        })
        .collect();
    Ok(ViewOutput {
        height,
        width,
        points_in_camera,
        conf: raw_conf.iter().map(|c| CONF_MIN + c.exp()).collect(),
        unambiguous: mask_logit.iter().map(|&logit| logit > 0.0).collect(),
        rotation,
        translation,
    })
}
