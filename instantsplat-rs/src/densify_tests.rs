//! GPU tests of densification: the statistics `project_backward.wgsl` sums against a CPU
//! reference of the 2D mean gradient (the trainer densifying is tested in `training.rs`).

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::gaussians::GaussianCloud;
use crate::rasterizer::DensifyStats;
use crate::rasterizer_tests::{ReferenceGaussian, Scene, camera_at_origin, random_cloud, reference_project};
use crate::shaders::TILE;

/// dL/d(2D mean) of every Gaussian, for L = sum of `weights` times the rendered (rgb,
/// transmittance), by the same compositing rules as the shaders (alpha capped at 0.99 with
/// no gradient through the cap, skipped below 1/255, stop before the transmittance falls
/// under 1e-4).
fn reference_mean_gradients(gaussians: &[ReferenceGaussian], width: u32, height: u32, weights: &[f32]) -> Vec<[f64; 2]> {
    let mut order: Vec<usize> = (0..gaussians.len()).collect();
    order.sort_by(|&a, &b| gaussians[a].depth.total_cmp(&gaussians[b].depth));
    let mut gradients = vec![[0.0f64; 2]; gaussians.len()];
    for py in 0..height {
        for px in 0..width {
            let (tile_x, tile_y) = (px / TILE, py / TILE);
            let pixel = (py * width + px) as usize;
            let weight = [0, 1, 2, 3].map(|c| weights[pixel * 4 + c] as f64);
            // Front to back: (Gaussian, alpha, d alpha / d power or 0 under the cap, offset).
            let mut contributions = Vec::new();
            let mut transmittance = 1.0f64;
            for &g in &order {
                let gaussian = &gaussians[g];
                let in_tiles = tile_x >= gaussian.tile_min[0]
                    && tile_x < gaussian.tile_max[0]
                    && tile_y >= gaussian.tile_min[1]
                    && tile_y < gaussian.tile_max[1];
                if !in_tiles {
                    continue;
                }
                let d = [gaussian.mean[0] as f64 - px as f64, gaussian.mean[1] as f64 - py as f64];
                let conic = gaussian.conic.map(|v| v as f64);
                let power = -0.5 * (conic[0] * d[0] * d[0] + conic[2] * d[1] * d[1]) - conic[1] * d[0] * d[1];
                if power > 0.0 {
                    continue;
                }
                let unclamped = gaussian.opacity as f64 * power.exp();
                let alpha = unclamped.min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let next = transmittance * (1.0 - alpha);
                if next < 1e-4 {
                    break;
                }
                let d_alpha_d_power = if unclamped < 0.99 { unclamped } else { 0.0 };
                contributions.push((g, alpha, d_alpha_d_power, d, transmittance));
                transmittance = next;
            }
            // Back to front: what is composited behind each Gaussian, as (rgb, transmittance).
            let mut behind = [0.0, 0.0, 0.0, transmittance];
            for &(g, alpha, d_alpha_d_power, d, in_front) in contributions.iter().rev() {
                let color = [
                    gaussians[g].color[0] as f64,
                    gaussians[g].color[1] as f64,
                    gaussians[g].color[2] as f64,
                    0.0,
                ];
                let d_alpha: f64 = (0..4).map(|c| weight[c] * (color[c] * in_front - behind[c] / (1.0 - alpha))).sum();
                for c in 0..4 {
                    behind[c] += color[c] * alpha * in_front;
                }
                let conic = gaussians[g].conic.map(|v| v as f64);
                let d_power = d_alpha * d_alpha_d_power;
                gradients[g][0] += -d_power * (conic[0] * d[0] + conic[1] * d[1]);
                gradients[g][1] += -d_power * (conic[1] * d[0] + conic[2] * d[1]);
            }
        }
    }
    gradients
}

#[test]
fn densify_stats_match_the_cpu_mean_gradient() {
    let mut rng = StdRng::seed_from_u64(11);
    let cloud: GaussianCloud = random_cloud(40, &mut rng);
    let camera = camera_at_origin(48, 32, 40.0);
    let scene = Scene::new(&cloud, camera.clone());
    let gaussians = reference_project(&cloud, &camera);
    assert_eq!(gaussians.len(), cloud.len(), "every Gaussian is in front of the camera");
    let pixel_count = (camera.width * camera.height) as usize;
    let weights: Vec<f32> = (0..pixel_count * 4).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    // Two backward passes of the same render and loss gradient: the sums double.
    for _ in 0..2 {
        scene.render();
        scene.rasterizer.write_loss_gradient(&scene.gpu, &weights);
        let mut encoder = scene.gpu.encoder();
        scene.rasterizer.encode_backward(&mut encoder);
        scene.gpu.submit(encoder);
    }
    let stats: Vec<DensifyStats> = scene.rasterizer.read_densify_stats(&scene.gpu);
    let reference = reference_mean_gradients(&gaussians, camera.width, camera.height, &weights);
    let half = [camera.width as f64 / 2.0, camera.height as f64 / 2.0];
    let mut visible = 0;
    for (g, (stat, gradient)) in stats.iter().zip(&reference).enumerate() {
        let on_screen = gaussians[g].radius > 0.0
            && gaussians[g].tile_max[0] > gaussians[g].tile_min[0]
            && gaussians[g].tile_max[1] > gaussians[g].tile_min[1];
        if !on_screen {
            assert_eq!(*stat, DensifyStats::default(), "Gaussian {g} is off screen");
            continue;
        }
        visible += 1;
        assert_eq!(stat.visible_count, 2.0, "Gaussian {g}");
        assert_eq!(stat.max_radius, gaussians[g].radius, "Gaussian {g}");
        let expected = 2.0 * (gradient[0] * half[0]).hypot(gradient[1] * half[1]);
        let actual = stat.mean_gradient_sum as f64;
        assert!(
            (actual - expected).abs() <= 1e-3 + 2e-3 * expected,
            "Gaussian {g}: GPU {actual:.5} vs CPU {expected:.5}"
        );
    }
    assert!(visible > 20, "only {visible} Gaussians on screen");
}
