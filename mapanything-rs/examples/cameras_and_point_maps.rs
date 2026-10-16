//! What the network predicts per photo, before it is merged into one point cloud: the
//! camera (position in the first photo's frame, focal length in pixels) and the dense point
//! map (one point per pixel, with a confidence and a foreground flag). Writes one PLY per
//! photo so the views can be compared in a viewer.
//!
//! cargo run --release --example cameras_and_point_maps -- --focal-35mm 26 frame_0001.png frame_0010.png

use std::path::PathBuf;

use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use mapanything_rs::model::MapAnything;
use mapanything_rs::pipeline::reconstruct;
use mapanything_rs::ply::{confident_points, write_ply};
use mapanything_rs::preprocess::load_images;

const WEIGHTS_REPO: &str = "facebook/map-anything-apache";
const WEIGHTS_FILE: &str = "model.safetensors";
const MIN_CONF: f32 = 1.0;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let focal_35mm = match args.iter().position(|arg| arg == "--focal-35mm") {
        Some(index) => {
            ensure!(index + 1 < args.len(), "--focal-35mm needs a value in mm");
            let value: f64 = args[index + 1].parse()?;
            args.drain(index..index + 2);
            Some(value)
        }
        None => None,
    };
    let photos: Vec<PathBuf> = args.into_iter().map(PathBuf::from).collect();
    ensure!(
        photos.len() >= 2,
        "usage: cameras_and_point_maps [--focal-35mm MM] <photo> <photo> [photo...]"
    );

    let device = Device::new_metal(0)?;
    let weights = hf_hub::api::sync::Api::new()?
        .model(WEIGHTS_REPO.to_string())
        .get(WEIGHTS_FILE)?;
    // SAFETY: the checkpoint file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F16, &device)? };
    let model = MapAnything::new(vb)?;

    let images = load_images(&photos, focal_35mm, DType::F16, &device)?;
    let reconstruction = reconstruct(&model, &images, &device)?;

    for (index, ((point_map, camera), image)) in reconstruction
        .point_maps
        .iter()
        .zip(&reconstruction.cameras)
        .zip(&images)
        .enumerate()
    {
        let position = camera.translation;
        let kept = point_map.conf.iter().filter(|&&conf| conf >= MIN_CONF).count();
        println!(
            "photo {index} {}: camera at ({:.3}, {:.3}, {:.3}), focal {:.1} px, point map {}x{}, {} of {} pixels confident",
            photos[index].display(),
            position.x,
            position.y,
            position.z,
            camera.focal,
            point_map.width,
            point_map.height,
            kept,
            point_map.width * point_map.height
        );
        let out = format!("view{index}.ply");
        write_ply(out.as_ref(), &confident_points(point_map, &image.rgb, MIN_CONF)?)?;
        println!("  wrote {out}");
    }
    Ok(())
}
