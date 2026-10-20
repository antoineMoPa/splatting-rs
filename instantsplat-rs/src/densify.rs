//! Adaptive density control during training, after 3DGS: every few steps, Gaussians whose
//! screen-space mean gradient (`rasterizer::DensifyStats`, summed on the GPU by every
//! backward pass) is high on average over the passes they were visible in are cloned
//! (small ones: a copy where they are) or split (large ones: two smaller Gaussians sampled
//! from the original), and nearly transparent or oversized ones are pruned. The packed
//! parameters and Adam's moments are read back, rebuilt here on the CPU, and the trainer
//! rebuilds its buffers at the new count (`Trainer::train_step_async`).
//!
//! Why: one Gaussian per point of every view's depth map puts one copy of the object per
//! view where the depth maps disagree; the gradient is high where the copies fight, and
//! new Gaussians there (with the transparent leftovers pruned) let the training settle on
//! the shape all photos agree on.

use crate::gaussians::{logit, sigmoid};
use crate::optimizer::AdamMoments;
use crate::rasterizer::DensifyStats;
use crate::shaders::PARAM_STRIDE;

/// When and how strongly the Gaussians are densified and pruned.
#[derive(Clone, Copy, Debug)]
pub struct DensifySettings {
    /// Steps between densifications.
    pub interval: usize,
    /// The first densification follows this many steps.
    pub start: usize,
    /// None after this fraction of the run, so that the last Gaussians settle.
    pub stop_fraction: f32,
    /// Mean screen-space gradient (normalized device coordinates) above which a Gaussian
    /// is cloned or split.
    pub gradient_threshold: f32,
    /// Gaussians whose largest axis is above this fraction of the camera extent are split,
    /// the others cloned.
    pub split_scale_fraction: f32,
    /// Gaussians below this opacity are pruned.
    pub prune_opacity: f32,
    /// Gaussians whose screen radius reached this many pixels are pruned.
    pub max_screen_radius: f32,
    /// The count never grows past this multiple of the count training started with: the
    /// highest gradients are densified first.
    pub max_growth: f32,
    /// At the densifications after a multiple of this many steps, every opacity above
    /// `RESET_OPACITY` is lowered to it (3DGS: every 3000 steps): the Gaussians the photos
    /// need regain opacity, the redundant ones fade and get pruned.
    pub opacity_reset_interval: Option<usize>,
}

/// What an opacity reset lowers the opacities to, as 3DGS.
pub const RESET_OPACITY: f32 = 0.01;

/// For the apps' short runs (300 steps over a handful of photos): 3DGS densifies every 100
/// steps from 500 to 15k with a threshold of 2e-4 and splits above 1% of the extent; here
/// every 50 steps over the first half of the run. Measured on the six duck photos through
/// Depth Anything 3 Base, 300 steps in the browser: loss 0.0789 -> 0.0764 with refined
/// poses (0.0802 -> 0.0777 fixed), 265k -> 296k Gaussians, 16.4 s -> 18-20 s; leaving one
/// photo out, its loss 0.358 -> 0.351. A threshold of 1e-4 (394k Gaussians) lowers the
/// training loss further but not the held-out one; an opacity reset at step 100 does not
/// recover in time (training loss 0.099); a sparser init (stride 3 or 4) with densification
/// is better on held-out photos (0.336) but blurrier (training loss 0.10-0.12).
pub const DENSIFY_SHORT_RUN: DensifySettings = DensifySettings {
    interval: 50,
    start: 50,
    stop_fraction: 0.5,
    gradient_threshold: 2e-4,
    split_scale_fraction: 0.01,
    prune_opacity: 0.005,
    max_screen_radius: 100.0,
    max_growth: 2.0,
    opacity_reset_interval: None,
};

/// A split Gaussian becomes this many, each with its scales divided by
/// `SPLIT_SCALE_DIVISOR`, as 3DGS.
const SPLIT_CHILDREN: usize = 2;
const SPLIT_SCALE_DIVISOR: f32 = 1.6;

impl DensifySettings {
    /// Whether the Gaussians are densified after `done` (1-based) of `steps` steps.
    pub fn densifies_after(&self, done: usize, steps: usize) -> bool {
        assert!(self.interval > 0, "densify interval");
        assert!(done >= 1 && done <= steps, "step {done} of {steps}");
        done >= self.start && done % self.interval == 0 && done as f32 <= self.stop_fraction * steps as f32
    }

    /// Whether the densification after `done` steps also resets the opacities.
    pub fn resets_opacity_after(&self, done: usize) -> bool {
        self.opacity_reset_interval.is_some_and(|interval| {
            assert!(interval > 0 && interval % self.interval == 0, "opacity reset interval {interval}: a multiple of the densify interval {}", self.interval);
            done % interval == 0
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DensifyReport {
    pub before: usize,
    pub cloned: usize,
    pub split: usize,
    /// Opacity below `prune_opacity`.
    pub pruned_transparent: usize,
    /// Screen radius reached `max_screen_radius` (and opacity fine).
    pub pruned_oversized: usize,
    /// Whether the opacities were reset (`RESET_OPACITY`).
    pub opacity_reset: bool,
    /// Above the gradient threshold but past `max_growth`: left as they are.
    pub over_budget: usize,
    pub after: usize,
}

impl std::fmt::Display for DensifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {} Gaussians: cloned {}, split {}, pruned {} transparent and {} oversized",
            self.before, self.after, self.cloned, self.split, self.pruned_transparent, self.pruned_oversized
        )?;
        if self.over_budget > 0 {
            write!(f, ", {} more over the growth cap", self.over_budget)?;
        }
        if self.opacity_reset {
            write!(f, ", opacities reset to {RESET_OPACITY}")?;
        }
        Ok(())
    }
}

/// The Gaussians and Adam's moments after one densification: the kept Gaussians keep their
/// moments, clones and split children start from zero (as 3DGS).
pub struct Densified {
    pub params: Vec<f32>,
    pub moments: AdamMoments,
    pub report: DensifyReport,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fate {
    Kept,
    Cloned,
    Split,
    PrunedTransparent,
    PrunedOversized,
}

/// Densifies and prunes `params` (packed, `PARAM_STRIDE` per Gaussian) by `stats`, with
/// `extent` the camera extent and at most `max_count` Gaussians after; `seed` picks the
/// split children's samples. With `reset_opacity`, every opacity is then lowered to
/// `RESET_OPACITY` at most, and the opacities' moments zeroed.
#[allow(clippy::too_many_arguments)]
pub fn densify(
    params: &[f32],
    moments: &AdamMoments,
    stats: &[DensifyStats],
    extent: f32,
    max_count: usize,
    settings: &DensifySettings,
    reset_opacity: bool,
    seed: u64,
) -> Densified {
    let count = stats.len();
    assert_eq!(params.len(), count * PARAM_STRIDE, "params of {count} Gaussians");
    assert_eq!(moments.first.len(), params.len(), "first moment length");
    assert_eq!(moments.second.len(), params.len(), "second moment length");
    assert!(extent > 0.0, "camera extent {extent}");
    let split_log_scale = (settings.split_scale_fraction * extent).ln();

    let mut fates: Vec<Fate> = (0..count)
        .map(|g| {
            let gaussian = &params[g * PARAM_STRIDE..(g + 1) * PARAM_STRIDE];
            if sigmoid(gaussian[10]) < settings.prune_opacity {
                Fate::PrunedTransparent
            } else if stats[g].max_radius >= settings.max_screen_radius {
                Fate::PrunedOversized
            } else {
                Fate::Kept
            }
        })
        .collect();
    let mean_gradient = |g: usize| {
        let stat = stats[g];
        if stat.visible_count > 0.0 { stat.mean_gradient_sum / stat.visible_count } else { 0.0 }
    };
    let mut candidates: Vec<usize> = (0..count)
        .filter(|&g| fates[g] == Fate::Kept && mean_gradient(g) > settings.gradient_threshold)
        .collect();
    candidates.sort_by(|&a, &b| mean_gradient(b).total_cmp(&mean_gradient(a)));
    // Clones and splits each add one Gaussian.
    let kept = fates.iter().filter(|&&fate| fate == Fate::Kept).count();
    let budget = max_count.saturating_sub(kept).min(candidates.len());
    for &g in &candidates[..budget] {
        let largest = params[g * PARAM_STRIDE + 3..g * PARAM_STRIDE + 6].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        fates[g] = if largest > split_log_scale { Fate::Split } else { Fate::Cloned };
    }

    let mut out = Densified {
        params: Vec::with_capacity((kept + budget) * PARAM_STRIDE),
        moments: AdamMoments {
            first: Vec::with_capacity((kept + budget) * PARAM_STRIDE),
            second: Vec::with_capacity((kept + budget) * PARAM_STRIDE),
        },
        report: DensifyReport {
            before: count,
            over_budget: candidates.len() - budget,
            ..Default::default()
        },
    };
    let mut new_gaussians = Vec::new();
    let mut rng = SplitMix64(seed);
    for (g, fate) in fates.iter().enumerate() {
        let range = g * PARAM_STRIDE..(g + 1) * PARAM_STRIDE;
        let gaussian = &params[range.clone()];
        match fate {
            Fate::PrunedTransparent => out.report.pruned_transparent += 1,
            Fate::PrunedOversized => out.report.pruned_oversized += 1,
            Fate::Kept | Fate::Cloned => {
                out.params.extend_from_slice(gaussian);
                out.moments.first.extend_from_slice(&moments.first[range.clone()]);
                out.moments.second.extend_from_slice(&moments.second[range]);
                if *fate == Fate::Cloned {
                    out.report.cloned += 1;
                    new_gaussians.extend_from_slice(gaussian);
                }
            }
            Fate::Split => {
                out.report.split += 1;
                for _ in 0..SPLIT_CHILDREN {
                    new_gaussians.extend(split_child(gaussian, &mut rng));
                }
            }
        }
    }
    out.params.extend_from_slice(&new_gaussians);
    out.moments.first.resize(out.params.len(), 0.0);
    out.moments.second.resize(out.params.len(), 0.0);
    out.report.after = out.params.len() / PARAM_STRIDE;
    if reset_opacity {
        out.report.opacity_reset = true;
        let reset_logit = logit(RESET_OPACITY);
        for g in 0..out.report.after {
            let slot = g * PARAM_STRIDE + 10;
            out.params[slot] = out.params[slot].min(reset_logit);
            out.moments.first[slot] = 0.0;
            out.moments.second[slot] = 0.0;
        }
    }
    assert!(out.report.after > 0, "densification pruned every Gaussian: {}", out.report);
    out
}

/// One of the Gaussians a split replaces `gaussian` (packed) with: its center sampled from
/// the original Gaussian, its scales divided by `SPLIT_SCALE_DIVISOR`, the rest as is.
fn split_child(gaussian: &[f32], rng: &mut SplitMix64) -> [f32; PARAM_STRIDE] {
    let scale = [3, 4, 5].map(|k| gaussian[k].exp());
    let local = [0, 1, 2].map(|k| scale[k] * rng.standard_normal());
    let rotation = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(gaussian[6], gaussian[7], gaussian[8], gaussian[9]));
    let offset = rotation * nalgebra::Vector3::from(local);
    let mut child = [0.0; PARAM_STRIDE];
    child.copy_from_slice(gaussian);
    for k in 0..3 {
        child[k] += offset[k];
        child[3 + k] -= SPLIT_SCALE_DIVISOR.ln();
    }
    child
}

/// A small seeded generator for the split samples (rand is a dev-dependency only).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 1.0) / (1u64 << 24) as f32
    }

    /// Box-Muller.
    fn standard_normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussians::logit;

    /// Extent 10: split above ln(0.1). Gaussian 0 is small with a high gradient, 1 large with a
    /// high gradient, 2 transparent, 3 oversized on screen, 4 small with a low gradient, 5
    /// never visible.
    fn six_gaussians() -> (Vec<f32>, AdamMoments, Vec<DensifyStats>) {
        let log_scales = [-4.0, -1.0, -4.0, -4.0, -4.0, -4.0];
        let opacities = [0.5, 0.5, 0.001, 0.5, 0.5, 0.5];
        let mut params = Vec::new();
        for g in 0..6 {
            params.extend([g as f32, 0.0, 5.0]);
            params.extend([log_scales[g], log_scales[g] - 1.0, log_scales[g] - 2.0]);
            params.extend([1.0, 0.0, 0.0, 0.0, logit(opacities[g])]);
            params.extend([0.1 * g as f32, 0.2, 0.3]);
        }
        let first: Vec<f32> = (0..params.len()).map(|i| i as f32 + 1.0).collect();
        let second: Vec<f32> = first.iter().map(|m| m * 10.0).collect();
        let stat = |sum: f32, visible: f32, max_radius: f32| DensifyStats {
            mean_gradient_sum: sum,
            visible_count: visible,
            max_radius,
            ..Default::default()
        };
        let stats = vec![
            stat(3e-3, 4.0, 5.0),
            stat(2e-3, 4.0, 30.0),
            stat(1.0, 4.0, 5.0),
            stat(1.0, 4.0, 200.0),
            stat(1e-4, 4.0, 5.0),
            stat(0.0, 0.0, 0.0),
        ];
        (params, AdamMoments { first, second }, stats)
    }

    #[test]
    fn clones_small_splits_large_and_prunes() {
        let (params, moments, stats) = six_gaussians();
        let out = densify(&params, &moments, &stats, 10.0, 100, &DENSIFY_SHORT_RUN, false, 1);
        assert_eq!(
            out.report,
            DensifyReport {
                before: 6,
                cloned: 1,
                split: 1,
                pruned_transparent: 1,
                pruned_oversized: 1,
                over_budget: 0,
                opacity_reset: false,
                after: 6,
            }
        );
        let gaussian = |g: usize| &out.params[g * PARAM_STRIDE..(g + 1) * PARAM_STRIDE];
        // Kept in order: 0, 4, 5 with their moments; then the clone of 0 and the two
        // children of 1, with zero moments.
        assert_eq!(gaussian(0), &params[0..PARAM_STRIDE]);
        assert_eq!(gaussian(1)[0], 4.0);
        assert_eq!(gaussian(2)[0], 5.0);
        assert_eq!(&out.moments.first[PARAM_STRIDE..2 * PARAM_STRIDE], &moments.first[4 * PARAM_STRIDE..5 * PARAM_STRIDE]);
        assert_eq!(&out.moments.second[0..PARAM_STRIDE], &moments.second[0..PARAM_STRIDE]);
        assert_eq!(gaussian(3), &params[0..PARAM_STRIDE]);
        assert!(out.moments.first[3 * PARAM_STRIDE..].iter().all(|&m| m == 0.0));
        assert!(out.moments.second[3 * PARAM_STRIDE..].iter().all(|&m| m == 0.0));
        for child in [gaussian(4), gaussian(5)] {
            let original = &params[PARAM_STRIDE..2 * PARAM_STRIDE];
            for k in 3..6 {
                assert!((child[k] - (original[k] - 1.6f32.ln())).abs() < 1e-6, "child log scale {}", child[k]);
            }
            assert_eq!(&child[6..], &original[6..]);
            // Within 4 sigma (e^-1 along x) of the original center.
            let offset = (0..3).map(|k| (child[k] - original[k]).powi(2)).sum::<f32>().sqrt();
            assert!(offset > 0.0 && offset < 4.0 * (-1.0f32).exp(), "child offset {offset}");
        }
        assert_ne!(gaussian(4)[0..3], gaussian(5)[0..3]);
    }

    #[test]
    fn the_growth_cap_densifies_the_highest_gradients_first() {
        let (params, moments, stats) = six_gaussians();
        // 4 kept after pruning; one more allowed: Gaussian 0 (3e-3 / 4) beats 1 (2e-3 / 4).
        let out = densify(&params, &moments, &stats, 10.0, 5, &DENSIFY_SHORT_RUN, false, 1);
        assert_eq!((out.report.cloned, out.report.split, out.report.over_budget, out.report.after), (1, 0, 1, 5));
    }

    #[test]
    fn split_samples_spread_like_the_gaussian() {
        // A Gaussian with sigma (0.1, 0.01, 0.01), turned 90 degrees about z: its long axis is y.
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let mut gaussian = [0.0f32; PARAM_STRIDE];
        gaussian[3..6].copy_from_slice(&[0.1f32.ln(), 0.01f32.ln(), 0.01f32.ln()]);
        gaussian[6..10].copy_from_slice(&[half, 0.0, 0.0, half]);
        let mut rng = SplitMix64(9);
        let samples: Vec<[f32; PARAM_STRIDE]> = (0..4000).map(|_| split_child(&gaussian, &mut rng)).collect();
        let deviation = |k: usize| (samples.iter().map(|s| s[k] * s[k]).sum::<f32>() / samples.len() as f32).sqrt();
        assert!((deviation(1) - 0.1).abs() < 0.005, "y deviation {}", deviation(1));
        assert!((deviation(0) - 0.01).abs() < 0.0005, "x deviation {}", deviation(0));
        assert!((deviation(2) - 0.01).abs() < 0.0005, "z deviation {}", deviation(2));
    }

    #[test]
    fn an_opacity_reset_lowers_every_opacity_and_its_moments() {
        let (params, moments, stats) = six_gaussians();
        let out = densify(&params, &moments, &stats, 10.0, 100, &DENSIFY_SHORT_RUN, true, 1);
        assert!(out.report.opacity_reset);
        for g in 0..out.report.after {
            let slot = g * PARAM_STRIDE + 10;
            assert!((sigmoid(out.params[slot]) - RESET_OPACITY).abs() < 1e-6, "opacity {}", sigmoid(out.params[slot]));
            assert_eq!((out.moments.first[slot], out.moments.second[slot]), (0.0, 0.0));
        }
        // Other moments are kept.
        assert_eq!(out.moments.first[0], moments.first[0]);
        let resets = DensifySettings {
            opacity_reset_interval: Some(100),
            ..DENSIFY_SHORT_RUN
        };
        assert_eq!((resets.resets_opacity_after(50), resets.resets_opacity_after(100)), (false, true));
    }

    #[test]
    fn densifies_every_interval_over_the_first_half() {
        let after: Vec<usize> = (1..=300).filter(|&done| DENSIFY_SHORT_RUN.densifies_after(done, 300)).collect();
        assert_eq!(after, [50, 100, 150]);
    }

    #[test]
    #[should_panic(expected = "densification pruned every Gaussian")]
    fn pruning_everything_is_refused() {
        let (mut params, moments, stats) = six_gaussians();
        for g in 0..6 {
            params[g * PARAM_STRIDE + 10] = logit(0.001);
        }
        densify(&params, &moments, &stats, 10.0, 100, &DENSIFY_SHORT_RUN, false, 1);
    }
}
