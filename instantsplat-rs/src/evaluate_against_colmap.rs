//! Judging a splat and its cameras without COLMAP in the training: COLMAP's model of the
//! whole capture is similarity-aligned onto the splat's (optimized) training cameras by
//! the frames they share, the residuals say how far those cameras still are from
//! COLMAP's, and every k-th frame of the model (the same frames `refine` holds out) is
//! rendered through its aligned COLMAP camera and compared with its photo.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::camera::PinholeCamera;
use crate::colmap_model::read_colmap_model;
use crate::gaussians::GaussianCloud;
use crate::gpu::Gpu;
use crate::posed_frames::{PosedFrame, align_colmap_to_cameras, is_held_out, load_views, posed_frames, without_stray_cameras};
use crate::run_output::side_by_side;
use crate::training_frames::training_size;
use crate::view_error::{ViewError, mean_error, render_all, view_error};

pub struct EvaluationSettings {
    /// Directory with COLMAP's `cameras.txt` and `images.txt`.
    pub colmap: PathBuf,
    /// Directory with the photos the COLMAP model names.
    pub frames: PathBuf,
    /// Every this-many-th frame of the model is rendered and compared.
    pub hold_out_every: usize,
}

pub struct EvaluationReport {
    /// Per held-out frame, by name.
    pub held_out: Vec<(String, ViewError)>,
    pub mean: ViewError,
}

/// `cameras` are the splat's training cameras (by frame name) after the optimization,
/// `initial_cameras` the ones it started from (their residuals are printed first, for the
/// comparison). Writes `heldout_<frame>.png` (photo | render) per held-out frame into
/// `out_dir`.
pub fn evaluate_against_colmap(
    gpu: &Gpu,
    cloud: &GaussianCloud,
    initial_cameras: Option<&[(String, PinholeCamera)]>,
    cameras: &[(String, PinholeCamera)],
    settings: &EvaluationSettings,
    out_dir: &Path,
) -> Result<EvaluationReport> {
    assert!(settings.hold_out_every >= 2, "hold out every {} frames", settings.hold_out_every);
    assert!(!cameras.is_empty(), "no cameras to evaluate");
    let training_height = cameras[0].1.height;
    let colmap = without_stray_cameras(&read_colmap_model(&settings.colmap)?);
    if let Some(initial) = initial_cameras {
        align_colmap_to_cameras(&colmap, initial, "initial cameras");
    }
    let (similarity, _) = align_colmap_to_cameras(&colmap, cameras, "optimized cameras");
    let frames = posed_frames(&colmap, &settings.frames, &similarity);
    let frame_count = frames.len();
    let held_out: Vec<PosedFrame> = frames
        .into_iter()
        .enumerate()
        .filter(|(index, _)| is_held_out(*index, settings.hold_out_every))
        .map(|(_, frame)| frame)
        .collect();
    assert!(!held_out.is_empty(), "no frame held out of {frame_count}");
    let size = training_size(&held_out[0].camera, training_height, 1.0);
    let views = load_views(&held_out, size)?;
    println!(
        "{} of the {frame_count} COLMAP frames held out (every {}th), rendered at {}x{}",
        held_out.len(),
        settings.hold_out_every,
        size.0,
        size.1
    );
    let renders = render_all(gpu, cloud, &views);
    println!("held-out view through its aligned COLMAP camera: L1 (foreground only), photo | render:");
    let mut errors = Vec::new();
    for ((frame, view), render) in held_out.iter().zip(&views).zip(&renders) {
        let error = view_error(render, view);
        let comparison_path = out_dir.join(format!("heldout_{}.png", frame.name.trim_end_matches(".png")));
        side_by_side(&[view.to_rgb8(), render.to_rgb8()]).save(&comparison_path)?;
        println!("  {}: {:.4} ({:.4}) {}", frame.name, error.all, error.foreground, comparison_path.display());
        errors.push((frame.name.clone(), error));
    }
    let mean = mean_error(&errors.iter().map(|(_, error)| *error).collect::<Vec<_>>());
    println!(
        "held-out L1 mean over {} views: {:.4} (foreground only {:.4})",
        errors.len(),
        mean.all,
        mean.foreground
    );
    Ok(EvaluationReport { held_out: errors, mean })
}
