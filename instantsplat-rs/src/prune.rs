//! Limits on the Gaussians' scales, applied after every optimization step
//! (`shaders/clamp_scales.wgsl`) and at export, and the pruning at export of Gaussians
//! that contribute nothing: nearly transparent ones and (as a safety net behind the
//! clamp) oversized ones.
//!
//! Why: a Gaussian seen from a single view is free along that view's ray, so without a
//! cap it stretches into a spike that only shows from other viewpoints; a haze of faint
//! Gaussians is left behind by the low starting opacity.

use crate::gaussians::{GaussianCloud, sigmoid};

#[derive(Clone, Copy, Debug)]
pub struct ScaleLimits {
    /// Largest over smallest axis of one Gaussian.
    pub max_anisotropy: f32,
    /// Largest axis, as a fraction of the camera extent (`cameras_extent`).
    pub max_scale_extent_fraction: f32,
}

/// 3DGS prunes Gaussians above 0.1 of the extent when densifying; the spikes to remove
/// here are far thinner than that, so the cap is tighter: 2% of the extent is about 4 cm
/// on the fleece scene, 10-20 times the starting scale of a Gaussian.
pub const SCALE_LIMITS: ScaleLimits = ScaleLimits {
    max_anisotropy: 10.0,
    max_scale_extent_fraction: 0.02,
};

/// Gaussians below this opacity are dropped at export (3DGS's densification prunes at 0.005
/// but resets opacities; nothing resets them here).
pub const PRUNE_OPACITY: f32 = 0.05;

impl ScaleLimits {
    pub fn max_log_scale(&self, extent: f32) -> f32 {
        assert!(extent > 0.0, "camera extent {extent}");
        (self.max_scale_extent_fraction * extent).ln()
    }

    pub fn max_log_anisotropy(&self) -> f32 {
        self.max_anisotropy.ln()
    }

    /// What `clamp_scales.wgsl` does after each step, on the CPU.
    pub fn clamp(&self, cloud: &mut GaussianCloud, extent: f32) {
        let max_log_scale = self.max_log_scale(extent);
        let max_log_anisotropy = self.max_log_anisotropy();
        for log_scale in cloud.log_scales.chunks_exact_mut(3) {
            for axis in log_scale.iter_mut() {
                *axis = axis.min(max_log_scale);
            }
            let smallest = log_scale.iter().copied().fold(f32::INFINITY, f32::min);
            for axis in log_scale.iter_mut() {
                *axis = axis.min(smallest + max_log_anisotropy);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PruneReport {
    pub kept: usize,
    /// Opacity below `PRUNE_OPACITY`.
    pub dropped_transparent: usize,
    /// Largest axis over the scale cap (and opacity fine).
    pub dropped_oversized: usize,
}

impl std::fmt::Display for PruneReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "kept {} Gaussians, dropped {} with opacity below {PRUNE_OPACITY} and {} with a scale over {} of the camera extent",
            self.kept, self.dropped_transparent, self.dropped_oversized, SCALE_LIMITS.max_scale_extent_fraction
        )
    }
}

/// The Gaussians worth exporting. Panics when nothing is left.
pub fn prune(cloud: &GaussianCloud, extent: f32) -> (GaussianCloud, PruneReport) {
    let max_log_scale = SCALE_LIMITS.max_log_scale(extent);
    let mut kept = GaussianCloud {
        positions: Vec::new(),
        log_scales: Vec::new(),
        rotations: Vec::new(),
        opacity_logits: Vec::new(),
        sh_dc: Vec::new(),
    };
    let mut report = PruneReport::default();
    for g in 0..cloud.len() {
        if sigmoid(cloud.opacity_logits[g]) < PRUNE_OPACITY {
            report.dropped_transparent += 1;
            continue;
        }
        if cloud.log_scales[g * 3..g * 3 + 3].iter().any(|&s| s > max_log_scale) {
            report.dropped_oversized += 1;
            continue;
        }
        kept.positions.extend_from_slice(&cloud.positions[g * 3..g * 3 + 3]);
        kept.log_scales.extend_from_slice(&cloud.log_scales[g * 3..g * 3 + 3]);
        kept.rotations.extend_from_slice(&cloud.rotations[g * 4..g * 4 + 4]);
        kept.opacity_logits.push(cloud.opacity_logits[g]);
        kept.sh_dc.extend_from_slice(&cloud.sh_dc[g * 3..g * 3 + 3]);
    }
    report.kept = kept.len();
    assert!(report.kept > 0, "pruning dropped every Gaussian: {report}");
    (kept, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussians::logit;

    fn three_gaussians() -> GaussianCloud {
        GaussianCloud {
            positions: vec![0.0; 9],
            // Isotropic; z far above x and y; anisotropic by e^4 and transparent.
            log_scales: vec![-3.0, -3.0, -3.0, -3.0, -3.0, 0.0, -5.0, -5.0, -1.0],
            rotations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            opacity_logits: vec![logit(0.5), logit(0.5), logit(0.01)],
            sh_dc: vec![0.0; 9],
        }
    }

    #[test]
    fn clamp_caps_absolute_scale_and_anisotropy() {
        let mut cloud = three_gaussians();
        // Extent 1: the absolute cap is ln(0.02) = -3.9, under every x and y, which the
        // anisotropy cap then leaves alone.
        SCALE_LIMITS.clamp(&mut cloud, 1.0);
        let cap = 0.02f32.ln();
        assert_eq!(&cloud.log_scales[0..3], &[cap, cap, cap]);
        assert_eq!(&cloud.log_scales[3..6], &[cap, cap, cap]);
        // z = -1 hits the absolute cap, which is below -5 + ln(10) = -2.7.
        assert_eq!(&cloud.log_scales[6..9], &[-5.0, -5.0, cap]);
        // Extent 100: the absolute cap is ln(2), over everything; only the anisotropy acts.
        let mut cloud = three_gaussians();
        SCALE_LIMITS.clamp(&mut cloud, 100.0);
        assert_eq!(&cloud.log_scales[0..3], &[-3.0, -3.0, -3.0]);
        assert_eq!(&cloud.log_scales[3..5], &[-3.0, -3.0]);
        assert!((cloud.log_scales[5] - (-3.0 + 10f32.ln())).abs() < 1e-6);
    }

    #[test]
    fn prune_drops_transparent_then_oversized() {
        let cloud = three_gaussians();
        // Extent 10: cap at ln(0.2) = -1.6, so the second is oversized on z and the third
        // (z = -1, also oversized) counts as transparent first.
        let (kept, report) = prune(&cloud, 10.0);
        assert_eq!(
            report,
            PruneReport {
                kept: 1,
                dropped_transparent: 1,
                dropped_oversized: 1
            }
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(&kept.log_scales, &[-3.0, -3.0, -3.0]);
    }

    #[test]
    #[should_panic(expected = "pruning dropped every Gaussian")]
    fn prune_refuses_an_empty_result() {
        let mut cloud = three_gaussians();
        cloud.opacity_logits = vec![logit(0.01); 3];
        prune(&cloud, 10.0);
    }
}
