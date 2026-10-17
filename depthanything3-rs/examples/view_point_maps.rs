//! The library used from Rust: photos in, one PLY per photo out (so the views can be
//! compared in a viewer), with each camera's predicted focal length and position printed.
//!
//! cargo run --release --example view_point_maps -- photo1.jpg photo2.jpg photo3.jpg

use std::path::PathBuf;

use anyhow::{Result, ensure};
use candle_core::{DType, Device};

use depthanything3_rs::pipeline::reconstruct;
use depthanything3_rs::preprocess::load_images;
use depthanything3_rs::reference_view::ReferenceViewStrategy;
use depthanything3_rs::variant::Variant;
use depthanything3_rs::weights::load_model;
use mapanything_rs::ply::{confident_points, write_ply};

const MIN_CONF: f32 = 1.0;

fn main() -> Result<()> {
    let photos: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    ensure!(!photos.is_empty(), "usage: photos_to_ply <photo> [photo...]");

    let device = Device::new_metal(0)?;
    let loaded = load_model(Variant::Small, ReferenceViewStrategy::SaddleBalanced, None, DType::F32, &device)?;
    let images = load_images(&photos, None, DType::F32, &device)?;
    let reconstruction = reconstruct(&loaded.model, &images, &device)?;

    for (index, (view, image)) in reconstruction.views.iter().zip(&images).enumerate() {
        let camera = &view.camera;
        let kept = view.point_map.conf.iter().filter(|&&conf| conf >= MIN_CONF).count();
        println!(
            "photo {index} {}: camera at ({:.3}, {:.3}, {:.3}), focal {:.1} {:.1} px, {} of {} pixels confident",
            photos[index].display(),
            camera.translation.x,
            camera.translation.y,
            camera.translation.z,
            camera.focal[0],
            camera.focal[1],
            kept,
            view.point_map.width * view.point_map.height
        );
        let out = format!("view{index}.ply");
        write_ply(out.as_ref(), &confident_points(&view.point_map, &image.rgb, MIN_CONF)?)?;
        println!("  wrote {out}");
    }
    Ok(())
}
