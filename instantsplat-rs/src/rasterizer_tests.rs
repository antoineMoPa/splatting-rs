//! GPU tests of the rasterizer: a known Gaussian, agreement with a CPU reference on a
//! random scene (which exercises binning and the sort), and the gradient check of the
//! backward pass against finite differences, for the color and the transmittance
//! channels. The loss kernels are tested in `loss_tests.rs`.

use nalgebra::{Matrix3, Vector3};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::camera::PinholeCamera;
use crate::gaussians::{GaussianCloud, SH_C0, logit, rgb_to_sh_dc, sigmoid};
use crate::gpu::Gpu;
use crate::loss::GROUND_TRUTH_STRIDE;
use crate::rasterizer::{BACKGROUND, Rasterizer, Rendered};
use crate::shaders::{PARAM_STRIDE, TILE};

pub(crate) fn camera_at_origin(width: u32, height: u32, focal: f64) -> PinholeCamera {
    PinholeCamera {
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
        focal: [focal, focal],
        principal_point: [(width / 2) as f64, (height / 2) as f64],
        width,
        height,
    }
}

fn single_gaussian(position: [f32; 3], sigma: f32, opacity: f32, rgb: [f32; 3]) -> GaussianCloud {
    GaussianCloud {
        positions: position.to_vec(),
        log_scales: vec![sigma.ln(); 3],
        rotations: vec![1.0, 0.0, 0.0, 0.0],
        opacity_logits: vec![logit(opacity)],
        sh_dc: rgb.map(rgb_to_sh_dc).to_vec(),
    }
}

pub(crate) struct Scene {
    pub(crate) gpu: Gpu,
    pub(crate) params: wgpu::Buffer,
    pub(crate) rasterizer: Rasterizer,
    pub(crate) camera: PinholeCamera,
}

impl Scene {
    pub(crate) fn new(cloud: &GaussianCloud, camera: PinholeCamera) -> Self {
        let gpu = Gpu::request();
        let params = gpu.storage_buffer_from("params", &cloud.to_packed());
        let ground_truth = gpu.storage_buffer(
            "ground truth",
            (camera.width * camera.height) as u64 * GROUND_TRUTH_STRIDE as u64 * 4,
        );
        let rasterizer = Rasterizer::new(
            &gpu,
            &params,
            cloud.len(),
            camera.width,
            camera.height,
            &ground_truth,
            1,
        );
        Self {
            gpu,
            params,
            rasterizer,
            camera,
        }
    }

    pub(crate) fn render(&self) -> Rendered {
        self.rasterizer.render(&self.gpu, &self.camera)
    }

    fn render_with(&self, packed: &[f32]) -> Rendered {
        self.gpu.write(&self.params, packed);
        self.render()
    }
}

fn premultiply(alpha: f32, rgb: [f32; 3]) -> [f32; 3] {
    rgb.map(|c| alpha * c)
}

fn assert_rgb_close(actual: [f32; 3], expected: [f32; 3], tolerance: f32, what: &str) {
    for c in 0..3 {
        assert!(
            (actual[c] - expected[c]).abs() <= tolerance,
            "{what}: channel {c} is {} but should be {}",
            actual[c],
            expected[c]
        );
    }
}

#[test]
fn renders_a_known_gaussian() {
    // 3D sigma 0.1 at depth 5 with focal 100: 2 px on screen, variance 4 + 0.3 of blur.
    let (width, height) = (32, 32);
    let cloud = single_gaussian([0.0, 0.0, 5.0], 0.1, 0.8, [1.0, 0.0, 0.0]);
    let scene = Scene::new(&cloud, camera_at_origin(width, height, 100.0));
    let rendered = scene.render();
    assert_rgb_close(rendered.premultiplied(16, 16), premultiply(0.8, [1.0, 0.0, 0.0]), 1e-3, "center");
    let alpha_2px = 0.8 * (-0.5 * 4.0 / 4.3f32).exp();
    assert_rgb_close(rendered.premultiplied(18, 16), premultiply(alpha_2px, [1.0, 0.0, 0.0]), 1e-3, "2 px right");
    assert!((rendered.alpha(18, 16) - alpha_2px).abs() < 1e-3, "alpha 2 px right {}", rendered.alpha(18, 16));
    assert_rgb_close(rendered.premultiplied(16, 13), premultiply(0.8 * (-0.5 * 9.0 / 4.3f32).exp(), [1.0, 0.0, 0.0]), 1e-3, "3 px up");
    assert_rgb_close(rendered.premultiplied(0, 0), [0.0; 3], 0.0, "corner");
    assert_rgb_close(rendered.over_background(0, 0), BACKGROUND, 0.0, "corner over the background");
    assert_rgb_close(rendered.premultiplied(31, 31), [0.0; 3], 0.0, "opposite corner");
    let transmittance = rendered.transmittance(16, 16);
    assert!((transmittance - 0.2).abs() < 1e-3, "transmittance {transmittance}");
    let over = rendered.over_background(16, 16);
    assert_rgb_close(over, [0.8 + 0.2 * BACKGROUND[0], 0.2 * BACKGROUND[1], 0.2 * BACKGROUND[2]], 1e-3, "center over the background");
    let rgba = rendered.to_rgba8();
    assert_eq!(rgba.get_pixel(16, 16).0, [255, 0, 0, 204]);
    assert_eq!(rgba.get_pixel(0, 0).0, [0, 0, 0, 0]);
}

#[test]
fn a_gaussian_behind_the_camera_is_culled() {
    let cloud = single_gaussian([0.0, 0.0, -5.0], 0.1, 0.8, [1.0, 0.0, 0.0]);
    let scene = Scene::new(&cloud, camera_at_origin(32, 32, 100.0));
    let rendered = scene.render();
    assert!(rendered.rgbt.chunks(4).all(|p| p[..3] == [0.0; 3] && p[3] == 1.0));
}

/// A Gaussian as the CPU reference sees it, with the same conventions as the shaders.
pub(crate) struct ReferenceGaussian {
    pub(crate) mean: [f32; 2],
    pub(crate) conic: [f32; 3],
    pub(crate) depth: f32,
    pub(crate) color: [f32; 3],
    pub(crate) opacity: f32,
    pub(crate) radius: f32,
    pub(crate) tile_min: [u32; 2],
    pub(crate) tile_max: [u32; 2],
}

pub(crate) fn reference_project(cloud: &GaussianCloud, camera: &PinholeCamera) -> Vec<ReferenceGaussian> {
    let (w, t) = camera.world_to_camera();
    (0..cloud.len())
        .filter_map(|g| {
            let p = w * Vector3::new(
                cloud.positions[g * 3] as f64,
                cloud.positions[g * 3 + 1] as f64,
                cloud.positions[g * 3 + 2] as f64,
            ) + t;
            if p.z < 0.01 {
                return None;
            }
            let scale = Vector3::from_fn(|k, _| (cloud.log_scales[g * 3 + k] as f64).exp());
            let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                cloud.rotations[g * 4] as f64,
                cloud.rotations[g * 4 + 1] as f64,
                cloud.rotations[g * 4 + 2] as f64,
                cloud.rotations[g * 4 + 3] as f64,
            ));
            let m = q.to_rotation_matrix().into_inner() * Matrix3::from_diagonal(&scale);
            let sigma = m * m.transpose();
            let [fx, fy] = camera.focal;
            let limit_x = 1.3 * camera.width as f64 / (2.0 * fx);
            let limit_y = 1.3 * camera.height as f64 / (2.0 * fy);
            let tx = (p.x / p.z).clamp(-limit_x, limit_x) * p.z;
            let ty = (p.y / p.z).clamp(-limit_y, limit_y) * p.z;
            let j = nalgebra::Matrix2x3::new(
                fx / p.z,
                0.0,
                -fx * tx / (p.z * p.z),
                0.0,
                fy / p.z,
                -fy * ty / (p.z * p.z),
            );
            let tm = j * w;
            let cov = tm * sigma * tm.transpose();
            let a = cov[(0, 0)] + 0.3;
            let b = cov[(0, 1)];
            let c = cov[(1, 1)] + 0.3;
            let det = a * c - b * b;
            let mid = 0.5 * (a + c);
            let radius = (3.0 * (mid + (mid * mid - det).max(0.1).sqrt()).sqrt()).ceil();
            let mean = [
                fx * p.x / p.z + camera.principal_point[0],
                fy * p.y / p.z + camera.principal_point[1],
            ];
            let tiles = [camera.width.div_ceil(TILE) as f64, camera.height.div_ceil(TILE) as f64];
            let tile_min = [0, 1].map(|k| ((mean[k] - radius) / TILE as f64).floor().clamp(0.0, tiles[k]) as u32);
            let tile_max = [0, 1].map(|k| (((mean[k] + radius) / TILE as f64).floor() + 1.0).clamp(0.0, tiles[k]) as u32);
            Some(ReferenceGaussian {
                mean: mean.map(|v| v as f32),
                conic: [c / det, -b / det, a / det].map(|v| v as f32),
                depth: p.z as f32,
                color: [0, 1, 2].map(|k| (SH_C0 * cloud.sh_dc[g * 3 + k] + 0.5).max(0.0)),
                opacity: sigmoid(cloud.opacity_logits[g]),
                radius: radius as f32,
                tile_min,
                tile_max,
            })
        })
        .collect()
}

fn reference_render(cloud: &GaussianCloud, camera: &PinholeCamera) -> Vec<f32> {
    let mut gaussians = reference_project(cloud, camera);
    gaussians.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    let mut image = Vec::with_capacity((camera.width * camera.height * 4) as usize);
    for py in 0..camera.height {
        for px in 0..camera.width {
            let (tile_x, tile_y) = (px / TILE, py / TILE);
            let mut transmittance = 1.0f32;
            let mut color = [0.0f32; 3];
            for g in &gaussians {
                let in_tiles = tile_x >= g.tile_min[0]
                    && tile_x < g.tile_max[0]
                    && tile_y >= g.tile_min[1]
                    && tile_y < g.tile_max[1];
                if !in_tiles {
                    continue;
                }
                let dx = g.mean[0] - px as f32;
                let dy = g.mean[1] - py as f32;
                let power = -0.5 * (g.conic[0] * dx * dx + g.conic[2] * dy * dy) - g.conic[1] * dx * dy;
                if power > 0.0 {
                    continue;
                }
                let alpha = (g.opacity * power.exp()).min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let next = transmittance * (1.0 - alpha);
                if next < 1e-4 {
                    break;
                }
                for (channel, value) in color.iter_mut().zip(g.color) {
                    *channel += value * alpha * transmittance;
                }
                transmittance = next;
            }
            image.extend(color);
            image.push(transmittance);
        }
    }
    image
}

/// Random Gaussians in front of the camera, with distinct depths far enough apart that
/// the 20-bit depth keys order them like the CPU does.
pub(crate) fn random_cloud(count: usize, rng: &mut StdRng) -> GaussianCloud {
    let mut depths: Vec<f32> = (0..count).map(|i| 3.0 + i as f32 * 0.01).collect();
    depths.shuffle(rng);
    let mut cloud = GaussianCloud {
        positions: Vec::new(),
        log_scales: Vec::new(),
        rotations: Vec::new(),
        opacity_logits: Vec::new(),
        sh_dc: Vec::new(),
    };
    for depth in depths {
        let spread = 0.4 * depth;
        cloud.positions.extend([
            rng.random_range(-spread..spread),
            rng.random_range(-spread..spread),
            depth,
        ]);
        cloud.log_scales.extend((0..3).map(|_| rng.random_range(-3.5f32..-1.5)));
        cloud.rotations.extend((0..4).map(|_| rng.random_range(-1.0f32..1.0)));
        cloud.opacity_logits.push(rng.random_range(-2.0f32..3.0));
        cloud.sh_dc.extend((0..3).map(|_| rng.random_range(-1.5f32..1.5)));
    }
    cloud
}

#[test]
fn forward_matches_the_cpu_reference() {
    let mut rng = StdRng::seed_from_u64(7);
    let cloud = random_cloud(600, &mut rng);
    let camera = camera_at_origin(80, 48, 60.0);
    let scene = Scene::new(&cloud, camera.clone());
    let rendered = scene.render();
    let reference = reference_render(&cloud, &camera);
    let intersections = scene.rasterizer.read_intersections(&scene.gpu);
    assert!(intersections.count > 600, "only {} intersections", intersections.count);
    let worst = rendered
        .rgbt
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 2e-4, "largest pixel difference {worst}");
}

/// Loss = sum of per-pixel weights times the rendered (rgb, transmittance); the backward
/// pass must agree with central finite differences of the forward pass on every parameter.
///
/// The rasterizer skips Gaussians below 1/255 of alpha and outside their 3 sigma tiles, so
/// a pixel's value is only piecewise smooth in the parameters, and a difference straddling
/// a jump is off by jump / (2 epsilon). Pixels where the estimates at two step sizes
/// disagree are left out of the comparison, on both sides: the loss is linear in the
/// weights, so the backward pass is rerun with those pixels' weights zeroed.
fn gradient_check(camera: PinholeCamera) {
    let mut rng = StdRng::seed_from_u64(3);
    let mut cloud = GaussianCloud {
        positions: Vec::new(),
        log_scales: Vec::new(),
        rotations: Vec::new(),
        opacity_logits: Vec::new(),
        sh_dc: Vec::new(),
    };
    // Gaussians in front of the camera, along its optical axis.
    for g in 0..4 {
        let in_camera = Vector3::new(
            rng.random_range(-0.3f64..0.3),
            rng.random_range(-0.3f64..0.3),
            3.0 + g as f64 * 0.5,
        );
        let in_world = camera.rotation * in_camera + camera.translation;
        cloud.positions.extend([in_world.x as f32, in_world.y as f32, in_world.z as f32]);
        cloud.log_scales.extend((0..3).map(|_| rng.random_range(-2.5f32..-1.0)));
        cloud.rotations.extend((0..4).map(|_| rng.random_range(-1.0f32..1.0)));
        cloud.opacity_logits.push(rng.random_range(-0.5f32..1.5));
        cloud.sh_dc.extend((0..3).map(|_| rng.random_range(-1.0f32..1.0)));
    }
    let scene = Scene::new(&cloud, camera.clone());
    let pixel_count = (camera.width * camera.height) as usize;
    let weights: Vec<f32> = (0..pixel_count * 4)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let packed = cloud.to_packed();
    let base_render = scene.render();
    assert!(base_render.rgbt.chunks(4).any(|p| p[3] < 0.5), "scene too transparent");

    let analytic_with = |weights: &[f32]| -> Vec<f32> {
        scene.render_with(&packed);
        scene.rasterizer.write_loss_gradient(&scene.gpu, weights);
        let mut encoder = scene.gpu.encoder();
        scene.rasterizer.encode_backward(&mut encoder);
        scene.gpu.submit(encoder);
        scene.rasterizer.read_param_gradients(&scene.gpu)
    };
    let slot_names = [
        "x", "y", "z", "log_sx", "log_sy", "log_sz", "qw", "qx", "qy", "qz", "opacity", "sh_r", "sh_g", "sh_b",
    ];
    let epsilon = 1e-3f32;
    let mut failures = Vec::new();
    let mut nonzero = 0;
    for (index, &value) in packed.iter().enumerate() {
        let name = format!("gaussian {} {}", index / PARAM_STRIDE, slot_names[index % PARAM_STRIDE]);
        let render_at = |delta: f32| {
            let mut moved = packed.clone();
            moved[index] = value + delta;
            scene.render_with(&moved).rgbt
        };
        let samples = [-2.0, -1.0, 1.0, 2.0].map(|k| render_at(k * epsilon));
        let mut smooth_weights = weights.clone();
        let mut numeric = 0.0f64;
        let mut excluded = 0;
        for pixel in 0..pixel_count {
            let channel = |sample: usize, c: usize| samples[sample][pixel * 4 + c] as f64;
            let narrow: Vec<f64> = (0..4).map(|c| (channel(2, c) - channel(1, c)) / (2.0 * epsilon as f64)).collect();
            let wide: Vec<f64> = (0..4).map(|c| (channel(3, c) - channel(0, c)) / (4.0 * epsilon as f64)).collect();
            let jumps = (0..4).any(|c| (narrow[c] - wide[c]).abs() > 2e-2 + 5e-2 * narrow[c].abs());
            if jumps {
                excluded += 1;
                for c in 0..4 {
                    smooth_weights[pixel * 4 + c] = 0.0;
                }
            } else {
                for c in 0..4 {
                    numeric += weights[pixel * 4 + c] as f64 * narrow[c];
                }
            }
        }
        assert!(excluded * 20 < pixel_count, "{name}: {excluded} of {pixel_count} pixels jump");
        let a = analytic_with(&smooth_weights)[index] as f64;
        if a.abs() > 1e-3 {
            nonzero += 1;
        }
        let tolerance = 3e-3 + 1e-2 * numeric.abs().max(a.abs());
        if (a - numeric).abs() > tolerance {
            failures.push(format!("{name}: analytic {a:.5} vs finite difference {numeric:.5}"));
        }
    }
    assert!(nonzero > 40, "only {nonzero} non-zero gradients: the scene does not exercise the backward pass");
    assert!(failures.is_empty(), "gradient mismatches:\n{}", failures.join("\n"));
}

#[test]
fn backward_matches_finite_differences() {
    gradient_check(camera_at_origin(32, 32, 40.0));
}

#[test]
fn backward_matches_finite_differences_for_a_moved_camera() {
    let camera = PinholeCamera {
        rotation: nalgebra::Rotation3::from_euler_angles(0.4, -0.7, 1.1).into_inner(),
        translation: Vector3::new(1.5, -2.0, 0.7),
        focal: [40.0, 40.0],
        principal_point: [16.0, 16.0],
        width: 32,
        height: 32,
    };
    gradient_check(camera);
}
