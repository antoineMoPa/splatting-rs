//! GPU tests of the camera pose refinement: the camera gradient of the backward pass,
//! chained to the pose parameters (`camera_pose.rs`), against finite differences of the
//! render over several views; and a synthetic scene whose perturbed cameras are recovered
//! by the trainer with the Gaussians held fixed.

use nalgebra::{Matrix3, Rotation3, Vector3};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::camera::PinholeCamera;
use crate::camera_pose::{PoseCorrection, corrected_camera, pose_distance, pose_gradient};
use crate::dense_view::DenseView;
use crate::gaussians::GaussianCloud;
use crate::gpu::Gpu;
use crate::optimizer::LearningRates;
use crate::rasterizer_tests::Scene;
use crate::training::{CameraPoses, Trainer};

/// Three cameras of a 32x32 view at different places, each looking at the Gaussians.
fn test_cameras() -> Vec<PinholeCamera> {
    [
        (Rotation3::identity(), Vector3::zeros()),
        (Rotation3::from_euler_angles(0.4, -0.7, 1.1), Vector3::new(1.5, -2.0, 0.7)),
        (Rotation3::from_euler_angles(-0.2, 0.3, -0.5), Vector3::new(-0.8, 0.4, -1.2)),
    ]
    .into_iter()
    .map(|(rotation, translation)| PinholeCamera {
        rotation: rotation.into_inner(),
        translation,
        focal: [40.0, 42.0],
        principal_point: [16.0, 16.0],
        width: 32,
        height: 32,
    })
    .collect()
}

/// Four anisotropic Gaussians in front of `camera`, as the rasterizer's gradient check.
fn gaussians_in_front_of(camera: &PinholeCamera, rng: &mut StdRng) -> GaussianCloud {
    let mut cloud = GaussianCloud {
        positions: Vec::new(),
        log_scales: Vec::new(),
        rotations: Vec::new(),
        opacity_logits: Vec::new(),
        sh_dc: Vec::new(),
    };
    for g in 0..4 {
        let in_camera = Vector3::new(rng.random_range(-0.3f64..0.3), rng.random_range(-0.3f64..0.3), 3.0 + g as f64 * 0.5);
        let in_world = camera.rotation * in_camera + camera.translation;
        cloud.positions.extend([in_world.x as f32, in_world.y as f32, in_world.z as f32]);
        cloud.log_scales.extend((0..3).map(|_| rng.random_range(-2.5f32..-1.0)));
        cloud.rotations.extend((0..4).map(|_| rng.random_range(-1.0f32..1.0)));
        cloud.opacity_logits.push(rng.random_range(-0.5f32..1.5));
        cloud.sh_dc.extend((0..3).map(|_| rng.random_range(-1.0f32..1.0)));
    }
    cloud
}

const POSE_PARAMETERS: [&str; 7] = ["rot_x", "rot_y", "rot_z", "t_x", "t_y", "t_z", "log_focal"];

fn with_parameter(correction: &PoseCorrection, focal_log_scale: f64, index: usize, delta: f64) -> (PoseCorrection, f64) {
    let mut moved = *correction;
    let mut focal = focal_log_scale;
    match index {
        0..=2 => moved.rotation[index] += delta,
        3..=5 => moved.translation[index - 3] += delta,
        6 => focal += delta,
        _ => unreachable!("pose parameter {index}"),
    }
    (moved, focal)
}

/// Loss = sum of per-pixel weights times the rendered (rgb, transmittance), as the
/// rasterizer's gradient check, differentiated with respect to the 7 pose parameters of a
/// camera that already carries a correction (so the rotation vector's Jacobian is not the
/// identity). The pixels where the two finite-difference step sizes disagree (the
/// rasterizer's cut-offs) are left out on both sides.
#[test]
fn pose_gradient_matches_finite_differences_on_every_view() {
    let mut rng = StdRng::seed_from_u64(11);
    let cameras = test_cameras();
    let correction = PoseCorrection {
        rotation: Vector3::new(0.1, -0.05, 0.08),
        translation: Vector3::new(0.02, -0.03, 0.01),
    };
    let focal_log_scale = 0.03;
    let mut failures = Vec::new();
    let mut nonzero = 0;
    for (view, initial) in cameras.iter().enumerate() {
        let camera = corrected_camera(initial, &correction, focal_log_scale);
        let cloud = gaussians_in_front_of(&camera, &mut rng);
        let scene = Scene::new(&cloud, camera.clone());
        let pixel_count = (camera.width * camera.height) as usize;
        let weights: Vec<f32> = (0..pixel_count * 4).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        assert!(scene.render().rgbt.chunks(4).any(|p| p[3] < 0.5), "view {view}: scene too transparent");
        let render_at = |correction: &PoseCorrection, focal_log_scale: f64| {
            scene.rasterizer.render(&scene.gpu, &corrected_camera(initial, correction, focal_log_scale)).rgbt
        };
        let analytic_with = |weights: &[f32]| {
            scene.rasterizer.set_camera(&scene.gpu, &camera);
            let mut encoder = scene.gpu.encoder();
            scene.rasterizer.encode_forward(&mut encoder);
            scene.gpu.submit(encoder);
            scene.rasterizer.write_loss_gradient(&scene.gpu, weights);
            let mut encoder = scene.gpu.encoder();
            scene.rasterizer.encode_backward(&mut encoder);
            scene.gpu.submit(encoder);
            let gradient = scene.rasterizer.read_camera_gradient(&scene.gpu);
            let pose = pose_gradient(initial, &correction, focal_log_scale, &gradient);
            [
                pose.rotation.x,
                pose.rotation.y,
                pose.rotation.z,
                pose.translation.x,
                pose.translation.y,
                pose.translation.z,
                pose.focal_log_scale,
            ]
        };
        // Smaller than the Gaussian parameters' 1e-3: a rotation moves every Gaussian at
        // once, and at 1e-3 rad one view's step straddled a cut-off both step sizes missed.
        let epsilon = 2.5e-4f64;
        for (index, name) in POSE_PARAMETERS.iter().enumerate() {
            let name = format!("view {view} {name}");
            let samples = [-2.0, -1.0, 1.0, 2.0].map(|k| {
                let (moved, focal) = with_parameter(&correction, focal_log_scale, index, k * epsilon);
                render_at(&moved, focal)
            });
            let mut smooth_weights = weights.clone();
            let mut numeric = 0.0f64;
            let mut excluded = 0;
            for pixel in 0..pixel_count {
                let channel = |sample: usize, c: usize| samples[sample][pixel * 4 + c] as f64;
                let narrow: Vec<f64> = (0..4).map(|c| (channel(2, c) - channel(1, c)) / (2.0 * epsilon)).collect();
                let wide: Vec<f64> = (0..4).map(|c| (channel(3, c) - channel(0, c)) / (4.0 * epsilon)).collect();
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
            let analytic = analytic_with(&smooth_weights)[index];
            if analytic.abs() > 1e-3 {
                nonzero += 1;
            }
            let tolerance = 3e-3 + 1e-2 * numeric.abs().max(analytic.abs());
            if (analytic - numeric).abs() > tolerance {
                failures.push(format!("{name}: analytic {analytic:.5} vs finite difference {numeric:.5}"));
            }
        }
    }
    assert!(nonzero >= 18, "only {nonzero} non-zero pose gradients of 21");
    assert!(failures.is_empty(), "pose gradient mismatches:\n{}", failures.join("\n"));
}

/// A view whose photo is the scene rendered from `true_camera`, posed at `camera`.
fn synthetic_view(scene: &Scene, true_camera: &PinholeCamera, camera: PinholeCamera) -> DenseView {
    let rendered = scene.rasterizer.render(&scene.gpu, true_camera);
    let (width, height) = (true_camera.width as usize, true_camera.height as usize);
    let mut rgb = Vec::with_capacity(width * height * 3);
    let mut foreground = Vec::with_capacity(width * height);
    for pixel in rendered.rgbt.chunks_exact(4) {
        rgb.extend_from_slice(&pixel[..3]);
        foreground.push(1.0 - pixel[3] > 0.5);
    }
    let view = DenseView {
        width,
        height,
        points: vec![[0.0; 3]; width * height],
        conf: vec![0.0; width * height],
        foreground,
        rgb,
        camera,
    };
    view.check();
    view
}

/// 400 opaque Gaussians of 1-2 px in a box at depth 2-3, under the scale cap of the
/// cameras' extent (2% of 1.1) as a real scene's Gaussians are, so that `clamp_scales`
/// leaves them alone.
fn opaque_cloud(rng: &mut StdRng) -> GaussianCloud {
    let mut cloud = GaussianCloud {
        positions: Vec::new(),
        log_scales: Vec::new(),
        rotations: Vec::new(),
        opacity_logits: Vec::new(),
        sh_dc: Vec::new(),
    };
    for _ in 0..400 {
        cloud.positions.extend([rng.random_range(-0.3f32..0.3), rng.random_range(-0.3f32..0.3), rng.random_range(2.0f32..3.0)]);
        cloud.log_scales.extend((0..3).map(|_| rng.random_range(-4.6f32..-3.9)));
        cloud.rotations.extend((0..4).map(|_| rng.random_range(-1.0f32..1.0)));
        cloud.opacity_logits.push(rng.random_range(3.0f32..5.0));
        cloud.sh_dc.extend((0..3).map(|_| rng.random_range(-1.5f32..1.5)));
    }
    cloud
}

/// Three views of `opaque_cloud` from x = -1, 0, 1 turned towards it, the photos rendered
/// from the true cameras; views 1 and 2 start 1.5 degrees and 0.03 (3% of the extent)
/// away from the truth and must come back with the Gaussians held fixed (their learning
/// rates zero).
#[test]
fn perturbed_cameras_are_recovered() {
    let mut rng = StdRng::seed_from_u64(5);
    let cloud = opaque_cloud(&mut rng);
    let true_cameras: Vec<PinholeCamera> = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|x: f64| PinholeCamera {
            rotation: Rotation3::from_euler_angles(0.0, -(x / 2.5).atan(), 0.0).into_inner(),
            translation: Vector3::new(x, 0.0, 0.0),
            focal: [150.0, 150.0],
            principal_point: [32.0, 32.0],
            width: 64,
            height: 64,
        })
        .collect();
    let scene = Scene::new(&cloud, true_cameras[0].clone());
    let perturbed: Vec<PinholeCamera> = true_cameras
        .iter()
        .enumerate()
        .map(|(view, camera)| {
            if view == 0 {
                return camera.clone();
            }
            let axis = Vector3::new(rng.random_range(-1.0f64..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)).normalize();
            let direction = Vector3::new(rng.random_range(-1.0f64..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)).normalize();
            corrected_camera(
                camera,
                &PoseCorrection {
                    rotation: axis * 1.5f64.to_radians(),
                    translation: direction * 0.03,
                },
                0.0,
            )
        })
        .collect();
    let views: Vec<DenseView> = true_cameras
        .iter()
        .zip(&perturbed)
        .map(|(true_camera, camera)| synthetic_view(&scene, true_camera, camera.clone()))
        .collect();
    assert!(views[0].foreground.iter().filter(|&&f| f).count() > 500, "the scene covers too little of the view");

    // The Gaussians held still; InstantSplat's pose rates, undecayed here (`step` with
    // factor 1), and the focal free so that its gradient is exercised too.
    let gaussians_fixed = LearningRates {
        position: 0.0,
        log_scale: 0.0,
        rotation: 0.0,
        opacity: 0.0,
        sh_dc: 0.0,
        camera_rotation: 1e-3,
        camera_translation: 1e-3,
        camera_focal: 1e-3,
    };
    let gpu = Gpu::request();
    let mut trainer = Trainer::new(&gpu, &cloud, &views, gaussians_fixed, CameraPoses::Refined);
    let errors = |trainer: &Trainer| -> Vec<(f64, f64)> {
        (1..3).map(|view| pose_distance(&trainer.camera(view), &true_cameras[view])).collect()
    };
    let before = errors(&trainer);
    let loss_before = trainer.loss_over_views().total;
    for step in 0..600 {
        trainer.step(step % 3, 1.0);
    }
    let after = errors(&trainer);
    let loss_after = trainer.loss_over_views().total;
    println!("loss {loss_before:.5} -> {loss_after:.5}, pose errors (deg, distance) {before:?} -> {after:?}");
    assert_eq!(trainer.camera(0).rotation, Matrix3::from(true_cameras[0].rotation), "view 0 is fixed");
    for view in 0..2 {
        assert!((before[view].0 - 1.5).abs() < 1e-9 && (before[view].1 - 0.03).abs() < 1e-9, "{before:?}");
        assert!(after[view].0 < before[view].0 / 10.0, "view {}: angle {:.4} -> {:.4} deg", view + 1, before[view].0, after[view].0);
        assert!(after[view].1 < before[view].1 / 10.0, "view {}: center {:.5} -> {:.5}", view + 1, before[view].1, after[view].1);
    }
    // What remains is the alpha term on partially covered pixels, the same at every pose.
    assert!(loss_after < loss_before / 2.0, "loss {loss_before} -> {loss_after}");
    assert!((trainer.focal_scale() - 1.0).abs() < 0.01, "focal scale {}", trainer.focal_scale());
}
