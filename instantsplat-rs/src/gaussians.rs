//! The optimized parameters of a set of 3D Gaussians, in the 3DGS parametrization: the
//! stored values are what the PLY holds and what Adam updates (log scales, opacity logits,
//! unnormalized quaternions, SH degree-0 coefficients).

use nalgebra::Vector3;

use crate::shaders::PARAM_STRIDE;

/// Degree-0 real spherical harmonic basis: color = SH_C0 * sh_dc + 0.5.
pub const SH_C0: f32 = 0.282_094_79;

/// Row-major flat vectors, one Gaussian after another.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianCloud {
    /// [N * 3]
    pub positions: Vec<f32>,
    /// [N * 3], log of the per-axis standard deviation.
    pub log_scales: Vec<f32>,
    /// [N * 4], (w, x, y, z), not necessarily unit length.
    pub rotations: Vec<f32>,
    /// [N], opacity = sigmoid(logit).
    pub opacity_logits: Vec<f32>,
    /// [N * 3]
    pub sh_dc: Vec<f32>,
}

impl GaussianCloud {
    pub fn len(&self) -> usize {
        let count = self.positions.len() / 3;
        assert_eq!(self.positions.len(), count * 3);
        assert_eq!(self.log_scales.len(), count * 3);
        assert_eq!(self.rotations.len(), count * 4);
        assert_eq!(self.opacity_logits.len(), count);
        assert_eq!(self.sh_dc.len(), count * 3);
        count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Mean position.
    pub fn centroid(&self) -> Vector3<f64> {
        let count = self.len();
        assert!(count > 0, "centroid of no Gaussians");
        self.positions
            .chunks_exact(3)
            .fold(Vector3::zeros(), |sum, p| sum + Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64))
            / count as f64
    }

    /// The layout the kernels read: per Gaussian, position, log scale, rotation, opacity
    /// logit, SH dc (`PARAM_STRIDE` floats).
    pub fn to_packed(&self) -> Vec<f32> {
        let count = self.len();
        let mut packed = Vec::with_capacity(count * PARAM_STRIDE);
        for g in 0..count {
            packed.extend_from_slice(&self.positions[g * 3..g * 3 + 3]);
            packed.extend_from_slice(&self.log_scales[g * 3..g * 3 + 3]);
            packed.extend_from_slice(&self.rotations[g * 4..g * 4 + 4]);
            packed.push(self.opacity_logits[g]);
            packed.extend_from_slice(&self.sh_dc[g * 3..g * 3 + 3]);
        }
        packed
    }

    pub fn from_packed(packed: &[f32]) -> Self {
        assert_eq!(packed.len() % PARAM_STRIDE, 0, "packed length {}", packed.len());
        let mut cloud = Self {
            positions: Vec::new(),
            log_scales: Vec::new(),
            rotations: Vec::new(),
            opacity_logits: Vec::new(),
            sh_dc: Vec::new(),
        };
        for gaussian in packed.chunks_exact(PARAM_STRIDE) {
            cloud.positions.extend_from_slice(&gaussian[0..3]);
            cloud.log_scales.extend_from_slice(&gaussian[3..6]);
            cloud.rotations.extend_from_slice(&gaussian[6..10]);
            cloud.opacity_logits.push(gaussian[10]);
            cloud.sh_dc.extend_from_slice(&gaussian[11..14]);
        }
        cloud
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse of `sigmoid`, for 0 < p < 1.
pub fn logit(p: f32) -> f32 {
    assert!(p > 0.0 && p < 1.0, "logit of {p}");
    (p / (1.0 - p)).ln()
}

/// From sRGB in [0, 1] to the SH degree-0 coefficient that renders it.
pub fn rgb_to_sh_dc(rgb: f32) -> f32 {
    (rgb - 0.5) / SH_C0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_inverts_sigmoid() {
        for p in [0.01, 0.1, 0.5, 0.9] {
            assert!((sigmoid(logit(p)) - p).abs() < 1e-6);
        }
    }

    #[test]
    fn packed_layout_round_trips() {
        let cloud = GaussianCloud {
            positions: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            log_scales: vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            rotations: vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            opacity_logits: vec![0.1, 0.2],
            sh_dc: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        };
        let packed = cloud.to_packed();
        assert_eq!(packed.len(), 2 * PARAM_STRIDE);
        assert_eq!(&packed[14..17], &[4.0, 5.0, 6.0]);
        assert_eq!(packed[10], 0.1);
        assert_eq!(GaussianCloud::from_packed(&packed), cloud);
    }

    #[test]
    fn centroid_is_the_mean_position() {
        let cloud = GaussianCloud {
            positions: vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            log_scales: vec![0.0; 6],
            rotations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            opacity_logits: vec![0.0; 2],
            sh_dc: vec![0.0; 6],
        };
        assert_eq!(cloud.centroid(), Vector3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn sh_dc_round_trips_color() {
        let sh = rgb_to_sh_dc(0.8);
        assert!((SH_C0 * sh + 0.5 - 0.8).abs() < 1e-6);
    }
}
