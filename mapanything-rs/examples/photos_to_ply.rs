//! The library in four calls: load the weights, load the photos, reconstruct, write a PLY.
//!
//! cargo run --release --example photos_to_ply -- photo1.jpg photo2.jpg photo3.jpg

use std::path::{Path, PathBuf};

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
const OUT: &str = "scene.ply";

fn main() -> Result<()> {
    let photos: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    ensure!(photos.len() >= 2, "usage: photos_to_ply <photo> <photo> [photo...]");

    let device = Device::new_metal(0)?;
    let weights = hf_hub::api::sync::Api::new()?.model(WEIGHTS_REPO.to_string()).get(WEIGHTS_FILE)?;
    // SAFETY: the checkpoint file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F16, &device)? };
    let model = MapAnything::new(vb)?;

    // Focal length from EXIF when the photos have it; pass Some(mm) for video frames.
    let images = load_images(&photos, None, DType::F16, &device)?;
    let reconstruction = reconstruct(&model, &images, &device)?;

    let mut points = Vec::new();
    for (point_map, image) in reconstruction.point_maps.iter().zip(&images) {
        points.extend(confident_points(point_map, &image.rgb, MIN_CONF)?);
    }
    write_ply(Path::new(OUT), &points)?;
    println!("wrote {OUT}: {} points from {} photos", points.len(), photos.len());
    Ok(())
}
