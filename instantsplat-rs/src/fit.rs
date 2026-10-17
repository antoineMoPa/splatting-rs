//! The `fit` command: a network (`--network`) on a few photos, one Gaussian per point, a short
//! optimization against those photos with the camera poses refined along (unless turned
//! off), outputs next to the PLY (the training views as RGBA PNGs, mask as alpha, so that
//! the optimization can be continued from them, as the browser demo does); optionally
//! judged against a COLMAP model of the capture (`evaluate_against_colmap.rs`), which
//! plays no part in the training.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, ensure};
use candle_core::DType;

use crate::camera::PinholeCamera;
use crate::cameras_json::write_cameras_json;
use crate::evaluate_against_colmap::{EvaluationSettings, evaluate_against_colmap};
use crate::gpu::Gpu;
use crate::init_from_points::{INIT_OPACITY, InitSettings, gaussians_from_dense_views};
use crate::network::{Network, NetworkSettings, dense_views_from_images, image_name};
use crate::optimizer::{LEARNING_RATES_3DGS, LearningRates};
use crate::run_output::{CAMERAS_FILE, SPLAT_FILE, output_dir};
use crate::splat_ply::write_splat_ply;
use crate::training::{CameraPoses, TrainSettings, Trainer};

pub struct FitSettings {
    pub images: Vec<PathBuf>,
    pub out: PathBuf,
    pub steps: usize,
    pub network: Network,
    pub focal_35mm: Option<f64>,
    pub stride: usize,
    pub min_conf: f32,
    pub dtype: DType,
    pub weights: Option<PathBuf>,
    pub cpu: bool,
    pub log_every: usize,
    pub camera_poses: CameraPoses,
    /// `LEARNING_RATES_3DGS` with the camera rates possibly overridden.
    pub learning_rates: LearningRates,
    pub evaluation: Option<EvaluationSettings>,
}

pub fn run_fit(settings: &FitSettings) -> Result<()> {
    assert!(settings.steps > 0, "steps must be at least 1");
    for image in &settings.images {
        let extension = image.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        ensure!(
            matches!(extension.as_deref(), Some("jpg" | "jpeg" | "png")),
            "{} is not a JPEG or PNG image (the output goes after --out)",
            image.display()
        );
        ensure!(image.is_file(), "image {} does not exist", image.display());
    }
    let out_dir = output_dir(&settings.out)?;

    let (views, network_timing) = dense_views_from_images(
        &settings.images,
        &NetworkSettings {
            network: settings.network,
            focal_35mm: settings.focal_35mm,
            dtype: settings.dtype,
            weights: settings.weights.clone(),
            cpu: settings.cpu,
        },
    )?;
    println!(
        "{}: {} views of {}x{} in {:.2?} (+{:.2?} loading weights)",
        settings.network.label(),
        views.len(),
        views[0].width,
        views[0].height,
        network_timing.inference,
        network_timing.load_weights
    );

    let started = Instant::now();
    let cloud = gaussians_from_dense_views(
        &views,
        InitSettings {
            stride: settings.stride,
            min_conf: settings.min_conf,
            opacity: INIT_OPACITY,
        },
    );
    println!("initialized {} Gaussians in {:.2?}", cloud.len(), started.elapsed());

    let gpu = Gpu::request();
    let report = Trainer::new(&gpu, &cloud, &views, settings.learning_rates, settings.camera_poses).optimize(&TrainSettings {
        steps: settings.steps,
        log_every: settings.log_every,
        learning_rates: settings.learning_rates,
    });
    println!(
        "optimized {} Gaussians for {} steps in {:.2?} ({:.1} ms per step): final L1 {:.4}",
        report.cloud.len(),
        settings.steps,
        report.elapsed,
        report.elapsed.as_secs_f64() * 1000.0 / settings.steps as f64,
        report.last_mean_error
    );

    let splat_path = out_dir.join(SPLAT_FILE);
    write_splat_ply(&splat_path, &report.cloud)?;
    println!("wrote {} ({} bytes)", splat_path.display(), std::fs::metadata(&splat_path)?.len());
    let named = |cameras: &[PinholeCamera]| -> Vec<(String, PinholeCamera)> {
        settings
            .images
            .iter()
            .zip(cameras)
            .map(|(path, camera)| (image_name(path), camera.clone()))
            .collect()
    };
    let initial_cameras = named(&views.iter().map(|view| view.camera.clone()).collect::<Vec<_>>());
    let cameras = named(&report.cameras);
    let cameras_path = out_dir.join(CAMERAS_FILE);
    write_cameras_json(&cameras_path, &cameras)?;
    println!("wrote {} ({} cameras, focal scale {:.4})", cameras_path.display(), cameras.len(), report.focal_scale);

    let trainer = Trainer::new(&gpu, &report.cloud, &views, LEARNING_RATES_3DGS, CameraPoses::Fixed);
    for (index, (view, camera)) in views.iter().zip(&report.cameras).enumerate() {
        trainer.render_camera(camera).to_rgb8().save(out_dir.join(format!("render_view{index}.png")))?;
        view.to_rgba8().save(out_dir.join(format!("input_view{index}.png")))?;
    }
    println!("wrote {} renders and inputs to {}", views.len(), out_dir.display());

    if let Some(evaluation) = &settings.evaluation {
        let initial = (settings.camera_poses == CameraPoses::Refined).then_some(initial_cameras.as_slice());
        evaluate_against_colmap(&gpu, &report.cloud, initial, &cameras, evaluation, &out_dir)?;
    }
    Ok(())
}
