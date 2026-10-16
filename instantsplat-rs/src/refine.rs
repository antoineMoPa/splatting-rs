//! The `refine` command: continue optimizing a splat against many more photos whose
//! poses come from a COLMAP model (`posed_frames.rs`: the COLMAP world is aligned to the
//! splat's frame by the cameras the two have in common, by frame name); every k-th frame
//! is held out and its L1 before and after the optimization is what judges the result.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};

use crate::camera::PinholeCamera;
use crate::cameras_json::{read_cameras_json, write_cameras_json};
use crate::colmap_model::read_colmap_model;
use crate::gpu::Gpu;
use crate::optimizer::LEARNING_RATES_3DGS;
use crate::posed_frames::{align_colmap_to_cameras, is_held_out, load_views, posed_frames, without_stray_cameras};
use crate::run_output::{CAMERAS_FILE, SPLAT_FILE, output_dir, side_by_side};
use crate::splat_ply::{read_splat_ply, write_splat_ply};
use crate::training::{CameraPoses, TrainSettings, Trainer};
use crate::training_frames::training_size;
use crate::view_error::{ViewError, mean_error, render_all, view_error};

pub struct RefineSettings {
    /// A `fit` run's output directory: its `splat.ply` and `cameras.json`.
    pub run: PathBuf,
    pub colmap: PathBuf,
    pub frames: PathBuf,
    pub out: PathBuf,
    pub steps: usize,
    /// Every this-many-th frame (by name) is held out; 0 holds out none.
    pub hold_out_every: usize,
    /// Training height = the splat's training height times this.
    pub resolution_scale: f64,
    pub log_every: usize,
}

pub fn run_refine(settings: &RefineSettings) -> Result<()> {
    assert!(settings.steps > 0, "steps must be at least 1");
    let out_dir = output_dir(&settings.out)?;

    let splat_path = settings.run.join(SPLAT_FILE);
    let cloud = read_splat_ply(&splat_path).with_context(|| format!("reading {}", splat_path.display()))?;
    let cameras_json_path = settings.run.join(CAMERAS_FILE);
    let splat_cameras = read_cameras_json(&cameras_json_path)?;
    ensure!(!splat_cameras.is_empty(), "{} has no cameras", cameras_json_path.display());
    let training_height = splat_cameras[0].1.height;
    for (name, camera) in &splat_cameras {
        ensure!(camera.height == training_height, "{name}: height {} differs from the first camera's {training_height}", camera.height);
    }
    println!(
        "{}: {} Gaussians, trained on {} views {} px high",
        splat_path.display(),
        cloud.len(),
        splat_cameras.len(),
        training_height
    );
    let colmap = without_stray_cameras(&read_colmap_model(&settings.colmap)?);
    let (similarity, _) = align_colmap_to_cameras(&colmap, &splat_cameras, "splat's cameras");
    let frames = posed_frames(&colmap, &settings.frames, &similarity);

    let size = training_size(&frames[0].camera, training_height, settings.resolution_scale);
    let started = Instant::now();
    let views = load_views(&frames, size)?;
    println!("loaded {} frames at {}x{} in {:.2?}", views.len(), size.0, size.1, started.elapsed());
    let (mut train_views, mut train_names, mut held_views, mut held_names) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (index, (view, frame)) in views.into_iter().zip(&frames).enumerate() {
        if is_held_out(index, settings.hold_out_every) {
            held_views.push(view);
            held_names.push(frame.name.clone());
        } else {
            train_views.push(view);
            train_names.push(frame.name.clone());
        }
    }
    println!("{} training views, {} held out: {}", train_views.len(), held_views.len(), held_names.join(" "));

    let gpu = Gpu::request();
    let renders_before = render_all(&gpu, &cloud, &held_views);
    let errors_before: Vec<ViewError> = renders_before.iter().zip(&held_views).map(|(r, v)| view_error(r, v)).collect();
    if !held_views.is_empty() {
        let mean = mean_error(&errors_before);
        println!("held-out L1 before: {:.4} (foreground {:.4})", mean.all, mean.foreground);
    }

    let report = Trainer::new(&gpu, &cloud, &train_views, LEARNING_RATES_3DGS, CameraPoses::Fixed).optimize(&TrainSettings {
        steps: settings.steps,
        log_every: settings.log_every,
        learning_rates: LEARNING_RATES_3DGS,
    });
    println!(
        "optimized {} Gaussians for {} steps in {:.2?} ({:.1} ms per step): last training L1 {:.4}",
        report.cloud.len(),
        settings.steps,
        report.elapsed,
        report.elapsed.as_secs_f64() * 1000.0 / settings.steps as f64,
        report.last_mean_error
    );

    let splat_path = out_dir.join(SPLAT_FILE);
    write_splat_ply(&splat_path, &report.cloud)?;
    println!("wrote {} ({} bytes)", splat_path.display(), std::fs::metadata(&splat_path)?.len());
    let mut cameras: Vec<(String, PinholeCamera)> = train_names
        .iter()
        .zip(&train_views)
        .chain(held_names.iter().zip(&held_views))
        .map(|(name, view)| (name.clone(), view.camera.clone()))
        .collect();
    cameras.sort_by(|a, b| a.0.cmp(&b.0));
    let cameras_path = out_dir.join(CAMERAS_FILE);
    write_cameras_json(&cameras_path, &cameras)?;
    println!("wrote {} ({} cameras)", cameras_path.display(), cameras.len());

    if held_views.is_empty() {
        return Ok(());
    }
    let renders_after = render_all(&gpu, &report.cloud, &held_views);
    let errors_after: Vec<ViewError> = renders_after.iter().zip(&held_views).map(|(r, v)| view_error(r, v)).collect();
    println!("held-out view: L1 before -> after (foreground only), photo | before | after:");
    for (index, name) in held_names.iter().enumerate() {
        let comparison_path = out_dir.join(format!("heldout_{}.png", name.trim_end_matches(".png")));
        side_by_side(&[held_views[index].to_rgb8(), renders_before[index].to_rgb8(), renders_after[index].to_rgb8()])
            .save(&comparison_path)?;
        println!(
            "  {name}: {:.4} -> {:.4} ({:.4} -> {:.4}) {}",
            errors_before[index].all,
            errors_after[index].all,
            errors_before[index].foreground,
            errors_after[index].foreground,
            comparison_path.display()
        );
    }
    let (before, after) = (mean_error(&errors_before), mean_error(&errors_after));
    println!(
        "held-out L1 mean over {} views: {:.4} -> {:.4} (foreground only {:.4} -> {:.4})",
        held_views.len(),
        before.all,
        after.all,
        before.foreground,
        after.foreground
    );
    Ok(())
}
