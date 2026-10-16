//! Renders a splat from a novel viewpoint: one of the cameras of a run's `cameras.json`,
//! orbited about the vertical axis (the scene frame's y, camera 0's down) through the
//! Gaussians' centroid, to a PNG over the display background and an RGBA PNG.
//!
//! cargo run --release --example novel_view -- runs/scene/splat.ply --view 0 --yaw-degrees 30 --out runs/scene/novel_view0_yaw30.png

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use nalgebra::{Matrix3, Rotation3, Vector3};

use instantsplat_rs::camera::PinholeCamera;
use instantsplat_rs::gpu::Gpu;
use instantsplat_rs::loss::GROUND_TRUTH_STRIDE;
use instantsplat_rs::rasterizer::Rasterizer;
use instantsplat_rs::splat_ply::read_splat_ply;

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
    /// Output PNG; an `_rgba.png` sibling with the coverage as alpha is written too.
    #[arg(long)]
    out: PathBuf,
}

/// The value after `"key": ` in `object`: the numbers inside its brackets (to the matching
/// close), or the single number up to the next comma or brace.
fn numbers_after(object: &str, key: &str) -> Result<Vec<f64>> {
    let marker = format!("\"{key}\": ");
    let start = object
        .find(&marker)
        .with_context(|| format!("no \"{key}\" in {object}"))?
        + marker.len();
    let rest = &object[start..];
    let end = if rest.starts_with('[') {
        let mut depth = 0i32;
        rest.char_indices()
            .find_map(|(i, c)| {
                depth += match c {
                    '[' => 1,
                    ']' => -1,
                    _ => 0,
                };
                (depth == 0).then_some(i + 1)
            })
            .with_context(|| format!("\"{key}\": unclosed bracket"))?
    } else {
        rest.find([',', '}']).with_context(|| format!("\"{key}\": no value end"))?
    };
    rest[..end]
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E'))
        .filter(|word| !word.is_empty())
        .map(|word| word.parse::<f64>().with_context(|| format!("{word:?} after \"{key}\"")))
        .collect()
}

/// Camera `view` of a `cameras.json` written by `cameras_json::write_cameras_json`.
fn read_camera(path: &std::path::Path, view: usize) -> Result<PinholeCamera> {
    let text = std::fs::read_to_string(path)?;
    let objects: Vec<&str> = text.split('{').skip(1).collect();
    ensure!(view < objects.len(), "{}: {} cameras, no view {view}", path.display(), objects.len());
    let object = objects[view];
    let one = |key: &str| -> Result<f64> {
        let values = numbers_after(object, key)?;
        ensure!(values.len() == 1, "\"{key}\" has {} values", values.len());
        Ok(values[0])
    };
    let position = numbers_after(object, "position")?;
    let rotation = numbers_after(object, "rotation")?;
    ensure!(position.len() == 3 && rotation.len() == 9, "camera {view}: malformed position or rotation");
    Ok(PinholeCamera {
        rotation: Matrix3::from_row_slice(&rotation),
        translation: Vector3::from_column_slice(&position),
        focal: [one("fx")?, one("fy")?],
        principal_point: [(one("width")? - 1.0) / 2.0, (one("height")? - 1.0) / 2.0],
        width: one("width")? as u32,
        height: one("height")? as u32,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cloud = read_splat_ply(&args.splat)?;
    let cameras_path = args.splat.parent().expect("splat path has a parent").join("cameras.json");
    let camera = read_camera(&cameras_path, args.view)?;
    let count = cloud.len();
    let centroid = (0..count).fold(Vector3::zeros(), |sum, g| {
        sum + Vector3::new(cloud.positions[g * 3] as f64, cloud.positions[g * 3 + 1] as f64, cloud.positions[g * 3 + 2] as f64)
    }) / count as f64;
    let orbit = Rotation3::from_axis_angle(&Vector3::y_axis(), args.yaw_degrees.to_radians()).into_inner();
    let novel = PinholeCamera {
        rotation: orbit * camera.rotation,
        translation: centroid + orbit * (camera.translation - centroid),
        ..camera
    };
    println!(
        "{count} Gaussians, centroid ({:.3}, {:.3}, {:.3}); view {} at ({:.3}, {:.3}, {:.3}) orbited {}° to ({:.3}, {:.3}, {:.3})",
        centroid.x, centroid.y, centroid.z, args.view,
        camera.translation.x, camera.translation.y, camera.translation.z,
        args.yaw_degrees,
        novel.translation.x, novel.translation.y, novel.translation.z
    );

    let gpu = Gpu::request();
    let params = gpu.storage_buffer_from("params", &cloud.to_packed());
    let ground_truth = gpu.storage_buffer(
        "ground truth",
        (novel.width * novel.height) as u64 * GROUND_TRUTH_STRIDE as u64 * 4,
    );
    let rasterizer = Rasterizer::new(&gpu, &params, count, novel.width, novel.height, &ground_truth, 1);
    let rendered = rasterizer.render(&gpu, &novel);
    rendered.to_rgb8().save(&args.out)?;
    let rgba_path = args.out.with_file_name(format!(
        "{}_rgba.png",
        args.out.file_stem().expect("file stem").to_string_lossy()
    ));
    rendered.to_rgba8().save(&rgba_path)?;
    println!("wrote {} and {}", args.out.display(), rgba_path.display());
    Ok(())
}
