//! The training loss: what the render is compared against and the weights of its terms.
//! The kernels that compute it are `shaders/ssim_forward.wgsl` and `shaders/loss.wgsl`,
//! run by the rasterizer; the ground truth they read is built by `ground_truth_rgba`.
//!
//! The render is compared over nothing (premultiplied by its coverage) against the photo
//! on the foreground pixels only, and its coverage against the foreground mask on every
//! pixel. A constant background color in the loss (the first version) rewarded a Gaussian
//! of that color floating in empty space; comparing color over black on every pixel, as
//! Brush does, was tried next and turned Gaussians that any view projects onto background
//! black as well as transparent, which with MapAnything's not quite consistent per-view
//! geometry hit many right ones (black fragments and holes in the training views). So a
//! background pixel acts through the alpha term alone: a misprojected Gaussian only
//! loses coverage, which the views that see it right can give back.

use bytemuck::{Pod, Zeroable};

use crate::dense_view::DenseView;
use crate::shaders::{SSIM_RADIUS, SSIM_SIGMA};

#[derive(Clone, Copy, Debug)]
pub struct LossWeights {
    /// L1 of the premultiplied rgb, averaged over channels and pixels.
    pub l1: f32,
    /// 1 - SSIM of the premultiplied rgb, averaged over channels and pixels.
    pub dssim: f32,
    /// L1 of the coverage against the foreground mask, averaged over pixels.
    pub alpha: f32,
}

/// 3DGS: (1 - 0.2) L1 + 0.2 (1 - SSIM); plus the alpha term at 1.0 (the Brush
/// `--match-alpha-weight 1` that gave the solid garment in `brush_cutout_alpha`).
pub const LOSS_WEIGHTS: LossWeights = LossWeights {
    l1: 0.8,
    dssim: 0.2,
    alpha: 1.0,
};

/// Means over the pixels of one view's loss terms, unweighted, and the weighted total.
#[derive(Clone, Copy, Debug, Default)]
pub struct LossTerms {
    pub l1: f32,
    pub alpha: f32,
    pub dssim: f32,
    pub total: f32,
}

impl LossTerms {
    /// From the per-pixel terms the loss kernel writes (4 floats per pixel).
    pub fn from_pixel_loss(pixel_loss: &[f32]) -> Self {
        assert!(!pixel_loss.is_empty() && pixel_loss.len().is_multiple_of(4), "pixel loss length {}", pixel_loss.len());
        let pixels = (pixel_loss.len() / 4) as f64;
        let mean = |term: usize| (pixel_loss.chunks_exact(4).map(|p| p[term] as f64).sum::<f64>() / pixels) as f32;
        Self {
            l1: mean(0),
            alpha: mean(1),
            dssim: mean(2),
            total: mean(3),
        }
    }
}

impl std::fmt::Display for LossTerms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "loss {:.4} (L1 {:.4}, alpha {:.4}, D-SSIM {:.4})",
            self.total, self.l1, self.alpha, self.dssim
        )
    }
}

/// The loss kernels' uniform: one per ground-truth view.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LossView {
    pub pixel_count: u32,
    pub width: u32,
    pub height: u32,
    /// First float of the view in the ground truth buffer.
    pub ground_truth_offset: u32,
    pub l1_weight: f32,
    pub dssim_weight: f32,
    pub alpha_weight: f32,
    pub _pad: f32,
}

/// Floats per pixel of the ground truth buffer: rgb (zero on background), then the mask.
pub const GROUND_TRUTH_STRIDE: usize = 4;

/// One view's ground truth as the loss kernels read it.
pub fn ground_truth_rgba(view: &DenseView) -> Vec<f32> {
    view.check();
    let mut rgba = Vec::with_capacity(view.pixel_count() * GROUND_TRUTH_STRIDE);
    for (pixel, &foreground) in view.foreground.iter().enumerate() {
        let mask = if foreground { 1.0 } else { 0.0 };
        rgba.extend(view.rgb[pixel * 3..pixel * 3 + 3].iter().map(|c| c * mask));
        rgba.push(mask);
    }
    rgba
}

/// The SSIM window's weights along one axis, summing to 1, as `ssim_taps` in the shaders.
pub fn ssim_taps() -> [f32; 2 * SSIM_RADIUS + 1] {
    let mut taps = [0.0f32; 2 * SSIM_RADIUS + 1];
    for (k, tap) in taps.iter_mut().enumerate() {
        let d = k as f32 - SSIM_RADIUS as f32;
        *tap = (-d * d / (2.0 * SSIM_SIGMA * SSIM_SIGMA)).exp();
    }
    let sum: f32 = taps.iter().sum();
    taps.map(|tap| tap / sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssim_taps_are_the_3dgs_window() {
        let taps = ssim_taps();
        assert!((taps.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // gaussian(11, 1.5) of the reference implementation: 1 / (1 + 2 sum exp(-k^2 / 4.5)).
        assert!((taps[5] - 0.266_01).abs() < 1e-5, "center tap {}", taps[5]);
        assert!((taps[0] - 0.001_028_4).abs() < 1e-6, "edge tap {}", taps[0]);
        assert_eq!(taps[1], taps[9]);
    }

    #[test]
    fn mean_loss_terms() {
        let terms = LossTerms::from_pixel_loss(&[0.2, 1.0, 0.5, 2.0, 0.4, 0.0, 0.1, 1.0]);
        assert!((terms.l1 - 0.3).abs() < 1e-6);
        assert!((terms.alpha - 0.5).abs() < 1e-6);
        assert!((terms.dssim - 0.3).abs() < 1e-6);
        assert!((terms.total - 1.5).abs() < 1e-6);
    }
}
