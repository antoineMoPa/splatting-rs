//! GPU tests of the loss kernels (`ssim_forward.wgsl`, `loss.wgsl`) on images written
//! straight into the rasterizer's image buffer: identical images, the SSIM term against a
//! CPU reference, and the loss gradient against central finite differences of the loss.
//! With `rasterizer_tests.rs` (the backward pass against finite differences of the render,
//! for the rgb and the transmittance channels) this covers the whole chain from the loss
//! to the Gaussian parameters: the two meet at the per-pixel loss gradient buffer.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::gpu::Gpu;
use crate::loss::{GROUND_TRUTH_STRIDE, LOSS_WEIGHTS, LossTerms, ssim_taps};
use crate::rasterizer::Rasterizer;
use crate::shaders::{PARAM_STRIDE, SSIM_RADIUS};

struct LossScene {
    gpu: Gpu,
    rasterizer: Rasterizer,
}

impl LossScene {
    /// One ground-truth view of `width` x `height` (rgba, `GROUND_TRUTH_STRIDE` per pixel).
    fn new(width: u32, height: u32, ground_truth: &[f32]) -> Self {
        assert_eq!(ground_truth.len(), (width * height) as usize * GROUND_TRUTH_STRIDE);
        let gpu = Gpu::request();
        let params = gpu.storage_buffer("params", (PARAM_STRIDE * 4) as u64);
        let ground_truth = gpu.storage_buffer_from("ground truth", ground_truth);
        let rasterizer = Rasterizer::new(&gpu, &params, 1, width, height, &ground_truth, 1);
        Self { gpu, rasterizer }
    }

    /// The loss of `rgbt` (premultiplied rgb and transmittance per pixel): the per-pixel
    /// terms and the gradient with respect to `rgbt`.
    fn loss_of(&self, rgbt: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.rasterizer.write_image(&self.gpu, rgbt);
        let mut encoder = self.gpu.encoder();
        self.rasterizer.encode_loss(&mut encoder, 0);
        self.gpu.submit(encoder);
        (
            self.rasterizer.read_pixel_loss(&self.gpu),
            self.rasterizer.read_loss_gradient(&self.gpu),
        )
    }
}

/// Summed in f64 so that small differences between two evaluations survive.
fn total_loss(pixel_loss: &[f32]) -> f64 {
    pixel_loss.chunks_exact(4).map(|p| p[3] as f64).sum::<f64>() / (pixel_loss.len() / 4) as f64
}

/// Random photo and mask, and a random render.
fn random_pair(width: u32, height: u32, rng: &mut StdRng) -> (Vec<f32>, Vec<f32>) {
    let pixels = (width * height) as usize;
    let mut ground_truth = Vec::with_capacity(pixels * GROUND_TRUTH_STRIDE);
    let mut rgbt = Vec::with_capacity(pixels * 4);
    for _ in 0..pixels {
        let mask = if rng.random_bool(0.6) { 1.0 } else { 0.0 };
        ground_truth.extend((0..3).map(|_| rng.random_range(0.0f32..1.0) * mask));
        ground_truth.push(mask);
        rgbt.extend((0..3).map(|_| rng.random_range(0.0f32..1.0)));
        rgbt.push(rng.random_range(0.1f32..0.9));
    }
    (ground_truth, rgbt)
}

#[test]
fn identical_images_have_zero_loss_and_gradient() {
    let (width, height) = (24, 20);
    let mut rng = StdRng::seed_from_u64(11);
    let (ground_truth, _) = random_pair(width, height, &mut rng);
    // The render that matches: premultiplied rgb, transmittance 1 - mask.
    let rgbt: Vec<f32> = ground_truth
        .chunks_exact(GROUND_TRUTH_STRIDE)
        .flat_map(|p| [p[0], p[1], p[2], 1.0 - p[3]])
        .collect();
    let scene = LossScene::new(width, height, &ground_truth);
    let (pixel_loss, gradient) = scene.loss_of(&rgbt);
    let terms = LossTerms::from_pixel_loss(&pixel_loss);
    assert_eq!(terms.l1, 0.0);
    assert_eq!(terms.alpha, 0.0);
    assert!(terms.dssim.abs() < 1e-6, "D-SSIM {}", terms.dssim);
    assert!(terms.total.abs() < 1e-6, "total {}", terms.total);
    let worst = gradient.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    assert!(worst < 1e-7, "largest gradient {worst}");
}

/// 1 - SSIM as `ssim()` of the 3DGS reference (11x11 Gaussian window, zero padding), on
/// the foreground pixels, averaged over all.
fn reference_dssim(width: u32, height: u32, rgbt: &[f32], ground_truth: &[f32]) -> f64 {
    let taps = ssim_taps().map(|t| t as f64);
    let (w, h) = (width as i64, height as i64);
    let radius = SSIM_RADIUS as i64;
    let mut sum = 0.0;
    for py in 0..h {
        for px in 0..w {
            for channel in 0..3 {
                let (mut mx, mut my, mut exx, mut eyy, mut exy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let (qx, qy) = (px + dx, py + dy);
                        if qx < 0 || qy < 0 || qx >= w || qy >= h {
                            continue;
                        }
                        let q = (qy * w + qx) as usize;
                        let weight = taps[(dx + radius) as usize] * taps[(dy + radius) as usize];
                        let x = rgbt[q * 4 + channel] as f64;
                        let y = ground_truth[q * GROUND_TRUTH_STRIDE + channel] as f64;
                        mx += weight * x;
                        my += weight * y;
                        exx += weight * x * x;
                        eyy += weight * y * y;
                        exy += weight * x * y;
                    }
                }
                let (sxx, syy, sxy) = (exx - mx * mx, eyy - my * my, exy - mx * my);
                let ssim = ((2.0 * mx * my + 1e-4) * (2.0 * sxy + 9e-4)) / ((mx * mx + my * my + 1e-4) * (sxx + syy + 9e-4));
                sum += ground_truth[(py * w + px) as usize * GROUND_TRUTH_STRIDE + 3] as f64 * (1.0 - ssim);
            }
        }
    }
    sum / (3 * w * h) as f64
}

#[test]
fn loss_terms_match_the_cpu_reference() {
    let (width, height) = (40, 24);
    let mut rng = StdRng::seed_from_u64(5);
    let (ground_truth, rgbt) = random_pair(width, height, &mut rng);
    let scene = LossScene::new(width, height, &ground_truth);
    let (pixel_loss, _) = scene.loss_of(&rgbt);
    let terms = LossTerms::from_pixel_loss(&pixel_loss);
    let pixels = (width * height) as usize;
    let mask = |p: usize| ground_truth[p * GROUND_TRUTH_STRIDE + 3] as f64;
    let l1: f64 = (0..pixels)
        .flat_map(|p| (0..3).map(move |c| (p, c)))
        .map(|(p, c)| mask(p) * (rgbt[p * 4 + c] - ground_truth[p * GROUND_TRUTH_STRIDE + c]).abs() as f64)
        .sum::<f64>()
        / (3 * pixels) as f64;
    let alpha: f64 = (0..pixels)
        .map(|p| (1.0 - rgbt[p * 4 + 3] - ground_truth[p * GROUND_TRUTH_STRIDE + 3]).abs() as f64)
        .sum::<f64>()
        / pixels as f64;
    let dssim = reference_dssim(width, height, &rgbt, &ground_truth);
    assert!((terms.l1 as f64 - l1).abs() < 1e-5, "L1 {} vs {l1}", terms.l1);
    assert!((terms.alpha as f64 - alpha).abs() < 1e-5, "alpha {} vs {alpha}", terms.alpha);
    assert!((terms.dssim as f64 - dssim).abs() < 1e-4, "D-SSIM {} vs {dssim}", terms.dssim);
    let total = LOSS_WEIGHTS.l1 as f64 * l1 + LOSS_WEIGHTS.alpha as f64 * alpha + LOSS_WEIGHTS.dssim as f64 * dssim;
    assert!((terms.total as f64 - total).abs() < 1e-4, "total {} vs {total}", terms.total);
    assert!(dssim > 0.3, "the random pair is too similar ({dssim}) to test anything");
}

#[test]
fn loss_gradient_matches_finite_differences() {
    let (width, height) = (32, 20);
    let mut rng = StdRng::seed_from_u64(17);
    let (ground_truth, rgbt) = random_pair(width, height, &mut rng);
    let scene = LossScene::new(width, height, &ground_truth);
    let (_, gradient) = scene.loss_of(&rgbt);
    let epsilon = 1e-2f32;
    let mut failures = Vec::new();
    let mut checked = 0;
    let mut background_checked = 0;
    while checked < 60 {
        let pixel = rng.random_range(0..(width * height) as usize);
        let channel = rng.random_range(0..4);
        let index = pixel * 4 + channel;
        // The L1 terms have a kink where the render meets the photo (the transmittance
        // channel never does: the mask is 0 or 1, the transmittance in [0.1, 0.9]).
        let difference = rgbt[index] - ground_truth[pixel * GROUND_TRUTH_STRIDE + channel];
        if channel < 3 && difference.abs() < 3.0 * epsilon {
            continue;
        }
        if channel < 3 && ground_truth[pixel * GROUND_TRUTH_STRIDE + 3] == 0.0 {
            background_checked += 1;
        }
        checked += 1;
        let loss_at = |delta: f32| {
            let mut moved = rgbt.clone();
            moved[index] += delta;
            total_loss(&scene.loss_of(&moved).0)
        };
        let numeric = (loss_at(epsilon) - loss_at(-epsilon)) / (2.0 * epsilon as f64);
        let analytic = gradient[index] as f64;
        let tolerance = 1e-6 + 1e-2 * numeric.abs().max(analytic.abs());
        if (analytic - numeric).abs() > tolerance {
            failures.push(format!(
                "pixel {pixel} channel {channel}: analytic {analytic:.3e} vs finite difference {numeric:.3e}"
            ));
        }
    }
    assert!(failures.is_empty(), "gradient mismatches:\n{}", failures.join("\n"));
    assert!(
        background_checked >= 4,
        "only {background_checked} background color entries checked"
    );
}
