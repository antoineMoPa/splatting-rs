//! Renders a splat from a novel viewpoint: one of the cameras of a run's `cameras.json`,
//! swung about the Gaussians' centroid (yaw about the vertical axis, the scene frame's y,
//! camera 0's down; pitch about the camera's right axis), to a PNG over the display
//! background and an RGBA PNG.
//!
//! cargo run --release --example novel_view -- runs/scene/splat.ply --view 0 --yaw-degrees 30 --out runs/scene/novel_view0_yaw30.png

use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::Parser;

use instantsplat_rs::cameras_json::read_cameras_json;
use instantsplat_rs::gpu::Gpu;
use instantsplat_rs::loss::GROUND_TRUTH_STRIDE;
use instantsplat_rs::rasterizer::Rasterizer;
use instantsplat_rs::run_output::CAMERAS_FILE;
use instantsplat_rs::splat_ply::read_splat_ply;
use instantsplat_rs::turntable::Turntable;

#[derive(Parser)]
struct Args {
    /// A splat written by instantsplat-rs, with its `cameras.json` next to it.
    splat: PathBuf,
    /// Index of the camera in `cameras.json` to orbit.
    #[arg(long, default_value_t = 0)]
    view: usize,
    /// Rotation about the vertical axis through the Gaussians' centroid, in degrees.
    #[arg(long, default_value_t = 30.0)]
    yaw_degrees: f64,
    /// Rotation about the camera's right axis through the centroid, in degrees.
    #[arg(long, default_value_t = 0.0)]
    pitch_degrees: f64,
    /// Output PNG; an `_rgba.png` sibling with the coverage as alpha is written too.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cloud = read_splat_ply(&args.splat)?;
    let cameras_path = args.splat.parent().expect("splat path has a parent").join(CAMERAS_FILE);
    let cameras = read_cameras_json(&cameras_path)?;
    ensure!(
        args.view < cameras.len(),
        "{}: {} cameras, no view {}",
        cameras_path.display(),
        cameras.len(),
        args.view
    );
    let (name, camera) = &cameras[args.view];
    let count = cloud.len();
    let centroid = cloud.centroid();
    let all: Vec<_> = cameras.iter().map(|(_, camera)| camera.clone()).collect();
    let novel = Turntable::from_cameras(&all, centroid).view(camera, args.yaw_degrees.to_radians(), args.pitch_degrees.to_radians());
    println!(
        "{count} Gaussians, centroid ({:.3}, {:.3}, {:.3}); view {} ({name}) at ({:.3}, {:.3}, {:.3}) orbited {} deg yaw, {} deg pitch to ({:.3}, {:.3}, {:.3})",
        centroid.x,
        centroid.y,
        centroid.z,
        args.view,
        camera.translation.x,
        camera.translation.y,
        camera.translation.z,
        args.yaw_degrees,
        args.pitch_degrees,
        novel.translation.x,
        novel.translation.y,
        novel.translation.z
    );

    let gpu = Gpu::request();
    let params = gpu.storage_buffer_from("params", &cloud.to_packed());
    let ground_truth = gpu.storage_buffer("ground truth", (novel.width * novel.height) as u64 * GROUND_TRUTH_STRIDE as u64 * 4);
    let rasterizer = Rasterizer::new(&gpu, &params, count, novel.width, novel.height, &ground_truth, 1);
    let rendered = rasterizer.render(&gpu, &novel);
    rendered.to_rgb8().save(&args.out)?;
    let rgba_path = args
        .out
        .with_file_name(format!("{}_rgba.png", args.out.file_stem().expect("file stem").to_string_lossy()));
    rendered.to_rgba8().save(&rgba_path)?;
    println!("wrote {} and {}", args.out.display(), rgba_path.display());
    Ok(())
}
