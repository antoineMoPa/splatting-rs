//! The camera head (upstream's `CameraDec`): from each view's camera token, a 9-number pose
//! encoding: translation (3), rotation quaternion (x, y, z, w; not unit, 4), and the
//! vertical and horizontal fields of view (2, ReLU-ed). Rotation and translation are
//! camera-to-world, the world being the reference view's camera frame (OpenCV: x right, y
//! down, z forward). The principal point is the image center.

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

use crate::variant::VariantSpec;

pub const POSE_ENCODING_DIM: usize = 9;
/// Upstream clamps tan(fov / 2) at this before dividing by it.
const MIN_TAN_HALF_FOV: f64 = 1e-6;

/// Camera-to-world pose and intrinsics of one view.
#[derive(Clone, Debug)]
pub struct CameraEstimate {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// Fields of view in radians: (vertical, horizontal).
    pub fov: [f64; 2],
}

impl CameraEstimate {
    /// From upstream's 9-number encoding.
    pub fn from_pose_encoding(encoding: &[f32]) -> Self {
        assert_eq!(encoding.len(), POSE_ENCODING_DIM, "pose encoding length");
        let [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w] = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| encoding[i] as f64);
        Self {
            rotation: UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz))
                .to_rotation_matrix()
                .into_inner(),
            translation: Vector3::new(tx, ty, tz),
            fov: [fov_h, fov_w],
        }
    }

    /// Focal lengths (fx, fy) in pixels of a `width` x `height` image.
    pub fn focal(&self, (height, width): (usize, usize)) -> [f64; 2] {
        let [fov_h, fov_w] = self.fov;
        [
            width as f64 / 2.0 / (fov_w / 2.0).tan().max(MIN_TAN_HALF_FOV),
            height as f64 / 2.0 / (fov_h / 2.0).tan().max(MIN_TAN_HALF_FOV),
        ]
    }

    /// Upstream's principal point: the image center in continuous coordinates (W / 2,
    /// H / 2), which it also uses when unprojecting integer pixel coordinates.
    pub fn principal_point((height, width): (usize, usize)) -> [f64; 2] {
        [width as f64 / 2.0, height as f64 / 2.0]
    }
}

pub struct CameraHead {
    hidden: [Linear; 2],
    translation: Linear,
    quaternion: Linear,
    fov: Linear,
}

impl CameraHead {
    /// `vb` at `cam_dec`.
    pub fn new(spec: &VariantSpec, vb: VarBuilder) -> Result<Self> {
        let dim = spec.head_input_dim();
        // Upstream runs the output layers in f32 whatever the backbone's precision.
        let output = vb.clone().set_dtype(DType::F32);
        Ok(Self {
            hidden: [
                candle_nn::linear(dim, dim, vb.pp("backbone.0"))?,
                candle_nn::linear(dim, dim, vb.pp("backbone.2"))?,
            ],
            translation: candle_nn::linear(dim, 3, output.pp("fc_t"))?,
            quaternion: candle_nn::linear(dim, 4, output.pp("fc_qvec"))?,
            fov: candle_nn::linear(dim, 2, output.pp("fc_fov.0"))?,
        })
    }

    /// `camera_tokens`: [views, 2 * dim]. Returns [views, POSE_ENCODING_DIM] in f32.
    pub fn forward(&self, camera_tokens: &Tensor) -> Result<Tensor> {
        let mut xs = camera_tokens.clone();
        for layer in &self.hidden {
            xs = layer.forward(&xs)?.relu()?;
        }
        let xs = xs.to_dtype(DType::F32)?;
        let translation = self.translation.forward(&xs)?;
        let quaternion = self.quaternion.forward(&xs)?;
        let fov = self.fov.forward(&xs)?.relu()?;
        Tensor::cat(&[translation, quaternion, fov], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_encoding_is_the_identity_pose() {
        let camera = CameraEstimate::from_pose_encoding(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert!((camera.rotation - Matrix3::identity()).norm() < 1e-12);
        assert_eq!(camera.translation, Vector3::zeros());
        // fov 1 rad on a 200 px wide image: f = 100 / tan(0.5).
        let [fx, _] = camera.focal((100, 200));
        assert!((fx - 100.0 / 0.5f64.tan()).abs() < 1e-9);
    }

    #[test]
    fn non_unit_quaternion_rotates_like_its_direction() {
        // 2 * (0, 0, sin 45, cos 45): 90 degrees about z.
        let s = 2.0 * (std::f64::consts::FRAC_PI_4).sin() as f32;
        let camera = CameraEstimate::from_pose_encoding(&[0.0, 0.0, 0.0, 0.0, 0.0, s, s, 1.0, 1.0]);
        let rotated = camera.rotation * Vector3::x();
        assert!((rotated - Vector3::y()).norm() < 1e-6, "{rotated}");
    }
}
