use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};

use mapanything_rs::model::MapAnything;
use mapanything_rs::pipeline::reconstruct;
use mapanything_rs::ply::{confident_points, write_ply};
use mapanything_rs::point_map::PointMap;
use mapanything_rs::preprocess::load_images;

/// MapAnything, Apache 2.0 weights.
const WEIGHTS_REPO: &str = "facebook/map-anything-apache";
const WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    F16,
    Bf16,
}

impl From<Precision> for DType {
    fn from(precision: Precision) -> Self {
        match precision {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        }
    }
}

/// Reconstruct a colored 3D point cloud from two or more images of the same scene, with
/// MapAnything: all images in one forward pass.
#[derive(Parser)]
struct Args {
    /// Two or more JPEG or PNG images, view 0 first. Transparent pixels of PNGs are
    /// background: cut the background out for turntable captures (fixed camera, rotating
    /// object).
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,
    /// Output point cloud (y up), in the camera frame of the first image.
    #[arg(long, default_value = "reconstruction.ply")]
    out: PathBuf,
    /// Keep points with confidence >= this. Confidence is 1 + exp(raw), so always > 1, or 0
    /// for points rejected as ambiguous or inconsistent with other views.
    #[arg(long, default_value_t = 1.0)]
    min_conf: f32,
    /// 35 mm-equivalent focal length of the camera, in mm, for every image; overrides EXIF.
    /// Known focal lengths are given to the model instead of letting it guess, which it does
    /// poorly on object-centric captures. Video frames have no EXIF: iPhone main cameras are
    /// about 24-26.
    #[arg(long)]
    focal_35mm: Option<f64>,
    #[arg(long, value_enum, default_value = "f16")]
    precision: Precision,
    /// MapAnything safetensors checkpoint; downloaded from Hugging Face when omitted.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Also save the dense point maps (image{i}.pts3d [H, W, 3], image{i}.conf [H, W],
    /// image{i}.foreground [H, W] u8) and cameras (image{i}.cam_to_world [3, 4],
    /// image{i}.focal in pixels) to this safetensors file, OpenCV camera convention.
    #[arg(long)]
    raw_out: Option<PathBuf>,
    /// Run on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,
}

fn point_map_tensors(index: usize, point_map: &PointMap) -> Result<[(String, Tensor); 3]> {
    let (height, width) = (point_map.height, point_map.width);
    let points: Vec<f32> = point_map.points.iter().flatten().copied().collect();
    let foreground: Vec<u8> = point_map.foreground.iter().map(|&f| f as u8).collect();
    Ok([
        (
            format!("image{index}.pts3d"),
            Tensor::from_vec(points, (height, width, 3), &Device::Cpu)?,
        ),
        (
            format!("image{index}.conf"),
            Tensor::from_slice(&point_map.conf, (height, width), &Device::Cpu)?,
        ),
        (
            format!("image{index}.foreground"),
            Tensor::from_vec(foreground, (height, width), &Device::Cpu)?,
        ),
    ])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    let dtype = DType::from(args.precision);

    let weights = match &args.weights {
        Some(path) => path.clone(),
        None => hf_hub::api::sync::Api::new()?
            .model(WEIGHTS_REPO.to_string())
            .get(WEIGHTS_FILE)?,
    };
    let started = Instant::now();
    // SAFETY: the checkpoint file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device)? };
    let model = MapAnything::new(vb)?;
    println!("loaded {} in {:.2?}", weights.display(), started.elapsed());

    let images = load_images(&args.images, args.focal_35mm, dtype, &device)?;
    let known = images
        .iter()
        .filter(|image| image.intrinsics.is_some())
        .count();
    println!(
        "camera intrinsics known for {known} of {} images",
        images.len()
    );
    let reconstruction = reconstruct(&model, &images, &device)?;

    if let Some(raw_out) = &args.raw_out {
        let mut tensors: HashMap<String, Tensor> = reconstruction
            .point_maps
            .iter()
            .enumerate()
            .map(|(index, point_map)| point_map_tensors(index, point_map))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        for (index, camera) in reconstruction.cameras.iter().enumerate() {
            let cam_to_world: Vec<f32> = (0..3)
                .flat_map(|row| {
                    [
                        camera.rotation[(row, 0)],
                        camera.rotation[(row, 1)],
                        camera.rotation[(row, 2)],
                        camera.translation[row],
                    ]
                })
                .map(|value| value as f32)
                .collect();
            tensors.insert(
                format!("image{index}.cam_to_world"),
                Tensor::from_vec(cam_to_world, (3, 4), &Device::Cpu)?,
            );
            tensors.insert(
                format!("image{index}.focal"),
                Tensor::new(camera.focal as f32, &Device::Cpu)?,
            );
        }
        candle_core::safetensors::save(&tensors, raw_out)?;
        println!("wrote {}", raw_out.display());
    }

    let mut points = Vec::new();
    for (point_map, image) in reconstruction.point_maps.iter().zip(&images) {
        points.extend(confident_points(point_map, &image.rgb, args.min_conf)?);
    }
    write_ply(&args.out, &points)?;
    println!("wrote {} points to {}", points.len(), args.out.display());
    Ok(())
}
