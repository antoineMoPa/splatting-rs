//! One Gaussian per (kept) dense point, as InstantSplat: the position is the point, the
//! color its pixel, the scale the spacing to the neighboring points, isotropic and
//! axis-aligned, with a low starting opacity. Training may densify and prune them
//! (`densify.rs`).

use crate::dense_view::DenseView;
use crate::gaussians::{GaussianCloud, logit, rgb_to_sh_dc};

#[derive(Clone, Copy, Debug)]
pub struct InitSettings {
    /// Keep one pixel in `stride` along each axis.
    pub stride: usize,
    /// Drop points with confidence below this (rejected points have 0).
    pub min_conf: f32,
    /// Starting opacity of every Gaussian.
    pub opacity: f32,
}

/// 3DGS starts every Gaussian at this opacity.
pub const INIT_OPACITY: f32 = 0.1;

impl DenseView {
    fn keeps(&self, x: usize, y: usize, min_conf: f32) -> bool {
        let i = y * self.width + x;
        self.foreground[i] && self.conf[i] >= min_conf
    }

    fn distance(&self, a: (usize, usize), b: (usize, usize)) -> f32 {
        let p = self.points[a.1 * self.width + a.0];
        let q = self.points[b.1 * self.width + b.0];
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }

    /// Distance to the nearest kept neighbor `stride` pixels away along an axis; None
    /// when the pixel has no kept neighbor. The nearest, so that a depth discontinuity
    /// on one side does not turn the point into a huge Gaussian.
    fn neighbor_spacing(&self, x: usize, y: usize, settings: InitSettings) -> Option<f32> {
        let stride = settings.stride as isize;
        [(stride, 0), (-stride, 0), (0, stride), (0, -stride)]
            .into_iter()
            .filter_map(|(dx, dy)| {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                let inside = nx >= 0 && ny >= 0 && (nx as usize) < self.width && (ny as usize) < self.height;
                let neighbor = (nx as usize, ny as usize);
                (inside && self.keeps(neighbor.0, neighbor.1, settings.min_conf))
                    .then(|| self.distance((x, y), neighbor))
            })
            .filter(|spacing| *spacing > 0.0)
            .min_by(|a, b| a.partial_cmp(b).expect("finite spacing"))
    }
}

pub fn gaussians_from_dense_views(views: &[DenseView], settings: InitSettings) -> GaussianCloud {
    assert!(settings.stride >= 1, "stride {}", settings.stride);
    let mut cloud = GaussianCloud {
        positions: Vec::new(),
        log_scales: Vec::new(),
        rotations: Vec::new(),
        opacity_logits: Vec::new(),
        sh_dc: Vec::new(),
    };
    let opacity_logit = logit(settings.opacity);
    for view in views {
        view.check();
        for y in (0..view.height).step_by(settings.stride) {
            for x in (0..view.width).step_by(settings.stride) {
                if !view.keeps(x, y, settings.min_conf) {
                    continue;
                }
                let Some(spacing) = view.neighbor_spacing(x, y, settings) else {
                    continue;
                };
                let i = y * view.width + x;
                cloud.positions.extend(view.points[i]);
                cloud.log_scales.extend([spacing.ln(); 3]);
                cloud.rotations.extend([1.0, 0.0, 0.0, 0.0]);
                cloud.opacity_logits.push(opacity_logit);
                cloud
                    .sh_dc
                    .extend((0..3).map(|c| rgb_to_sh_dc(view.rgb[i * 3 + c])));
            }
        }
    }
    assert!(!cloud.is_empty(), "no point kept: every pixel is background or below min_conf");
    cloud
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use nalgebra::{Matrix3, Vector3};

    /// A 4x3 view of a plane at z = 2 with 0.5 units between pixels, pixel (3, 0) rejected.
    fn plane_view() -> DenseView {
        let (width, height) = (4, 3);
        let mut points = Vec::new();
        for y in 0..height {
            for x in 0..width {
                points.push([x as f32 * 0.5, y as f32 * 0.5, 2.0]);
            }
        }
        let mut conf = vec![2.0; width * height];
        conf[3] = 0.0;
        DenseView {
            width,
            height,
            points,
            conf,
            foreground: vec![true; width * height],
            rgb: (0..width * height * 3).map(|i| (i % 3) as f32 * 0.25).collect(),
            camera: PinholeCamera {
                rotation: Matrix3::identity(),
                translation: Vector3::zeros(),
                focal: [1.0, 1.0],
                principal_point: [0.0, 0.0],
                width: width as u32,
                height: height as u32,
            },
        }
    }

    #[test]
    fn one_gaussian_per_kept_point_with_neighbor_spacing() {
        let cloud = gaussians_from_dense_views(
            &[plane_view()],
            InitSettings {
                stride: 1,
                min_conf: 1.0,
                opacity: 0.1,
            },
        );
        assert_eq!(cloud.len(), 11);
        assert_eq!(&cloud.positions[0..3], &[0.0, 0.0, 2.0]);
        assert!(cloud.log_scales.iter().all(|s| (s - 0.5f32.ln()).abs() < 1e-6));
        assert!((crate::gaussians::sigmoid(cloud.opacity_logits[0]) - 0.1).abs() < 1e-6);
        assert!((crate::gaussians::SH_C0 * cloud.sh_dc[1] + 0.5 - 0.25).abs() < 1e-6);
    }

    #[test]
    fn stride_keeps_every_other_pixel_and_scales_spacing() {
        let cloud = gaussians_from_dense_views(
            &[plane_view()],
            InitSettings {
                stride: 2,
                min_conf: 1.0,
                opacity: 0.1,
            },
        );
        // Pixels (0,0), (2,0), (0,2), (2,2).
        assert_eq!(cloud.len(), 4);
        assert!(cloud.log_scales.iter().all(|s| (s - 1.0f32.ln()).abs() < 1e-6));
    }
}
