use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::{Parser, ValueEnum};

use depthanything3_rs::depth_image::{conf_to_gray, depth_to_gray};
use depthanything3_rs::pipeline::{ReconstructedView, Reconstruction, reconstruct};
use depthanything3_rs::preprocess::load_images;
use depthanything3_rs::reference_view::ReferenceViewStrategy;
use depthanything3_rs::variant::Variant;
use depthanything3_rs::weights::load_model;
use mapanything_rs::ply::{confident_points, write_ply};

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

/// Depth, confidence and cameras for a set of images of the same scene with Depth
/// Anything 3 (one forward pass over all images), and a colored point cloud from them.
#[derive(Parser)]
struct Args {
    /// One or more JPEG or PNG images, view 0 first. Transparent pixels of PNGs are
    /// background: cut the background out for turntable captures (fixed camera, rotating
    /// object).
    #[arg(required = true, num_args = 1..)]
    images: Vec<PathBuf>,
    /// Output point cloud (y up), in the camera frame of the first image.
    #[arg(long, default_value = "reconstruction.ply")]
    out: PathBuf,
    /// Keep points with confidence >= this. Confidence is 1 + exp(logit), so always > 1,
    /// or 0 for points inconsistent with other views.
    #[arg(long, default_value_t = 1.0)]
    min_conf: f32,
    /// 35 mm-equivalent focal length of the camera, in mm, for every image; overrides
    /// EXIF. The model predicts its own intrinsics; a known focal length is only printed
    /// next to the prediction.
    #[arg(long)]
    focal_35mm: Option<f64>,
    #[arg(long, value_enum, default_value = "small")]
    variant: Variant,
    /// Which view the poses are predicted relative to (all outputs are then moved to view
    /// 0's frame); upstream's default picks it from the image features.
    #[arg(long, value_enum, default_value = "saddle-balanced")]
    reference_view: ReferenceViewStrategy,
    #[arg(long, value_enum, default_value = "f32")]
    precision: Precision,
    /// Safetensors checkpoint; downloaded from Hugging Face when omitted.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Also save the dense outputs (image{i}.depth [H, W], image{i}.conf [H, W],
    /// image{i}.pts3d [H, W, 3] in view 0's frame, image{i}.foreground [H, W] u8) and
    /// cameras (image{i}.cam_to_world [3, 4], image{i}.intrinsics [3, 3]) to this
    /// safetensors file, OpenCV camera convention.
    #[arg(long)]
    raw_out: Option<PathBuf>,
    /// Also write each view's depth (near is bright) and confidence as grayscale PNGs
    /// (`view{i}_depth.png`, `view{i}_conf.png`) into this directory.
    #[arg(long)]
    debug_dir: Option<PathBuf>,
    /// Run on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,
}

/// Where the cameras look: the centroid of the confident foreground points of every view.
fn scene_center(reconstruction: &Reconstruction, min_conf: f32) -> nalgebra::Vector3<f64> {
    let mut sum = nalgebra::Vector3::zeros();
    let mut count = 0usize;
    for view in &reconstruction.views {
        let map = &view.point_map;
        for ((point, &conf), &foreground) in map.points.iter().zip(&map.conf).zip(&map.foreground) {
            if foreground && conf >= min_conf {
                sum += nalgebra::Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);
                count += 1;
            }
        }
    }
    assert!(count > 0, "no confident foreground point");
    sum / count as f64
}

fn view_tensors(index: usize, view: &ReconstructedView) -> Result<Vec<(String, Tensor)>> {
    let map = &view.point_map;
    let (height, width) = (map.height, map.width);
    let points: Vec<f32> = map.points.iter().flatten().copied().collect();
    let foreground: Vec<u8> = map.foreground.iter().map(|&f| f as u8).collect();
    let camera = &view.camera;
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
    let [fx, fy] = camera.focal.map(|f| f as f32);
    let [cx, cy] = camera.principal_point.map(|c| c as f32);
    let intrinsics = vec![fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0];
    let cpu = &Device::Cpu;
    Ok(vec![
        (format!("image{index}.depth"), Tensor::from_slice(&view.depth, (height, width), cpu)?),
        (format!("image{index}.conf"), Tensor::from_slice(&map.conf, (height, width), cpu)?),
        (format!("image{index}.pts3d"), Tensor::from_vec(points, (height, width, 3), cpu)?),
        (format!("image{index}.foreground"), Tensor::from_vec(foreground, (height, width), cpu)?),
        (format!("image{index}.cam_to_world"), Tensor::from_vec(cam_to_world, (3, 4), cpu)?),
        (format!("image{index}.intrinsics"), Tensor::from_vec(intrinsics, (3, 3), cpu)?),
    ])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu { Device::Cpu } else { Device::new_metal(0)? };
    let dtype = DType::from(args.precision);

    let loaded = load_model(args.variant, args.reference_view, args.weights.as_ref(), dtype, &device)?;
    println!("loaded {} in {:.2?}", loaded.weights_path.display(), loaded.load_time);

    let images = load_images(&args.images, args.focal_35mm, dtype, &device)?;
    let reconstruction = reconstruct(&loaded.model, &images, &device)?;

    let center = scene_center(&reconstruction, args.min_conf);
    for (index, (view, image)) in reconstruction.views.iter().zip(&images).enumerate() {
        let camera = &view.camera;
        let given = match image.intrinsics {
            Some(intrinsics) => format!(", given {:.1} {:.1} px", intrinsics.focal[0], intrinsics.focal[1]),
            None => String::new(),
        };
        // Angle between the optical axis and the direction to the scene center: small
        // when the camera looks at the object.
        let to_center = center - camera.translation;
        let off_axis = camera.rotation.column(2).dot(&to_center.normalize()).clamp(-1.0, 1.0).acos().to_degrees();
        let (min_depth, max_depth) = view
            .depth
            .iter()
            .fold((f32::INFINITY, 0f32), |(min, max), &d| (min.min(d), max.max(d)));
        println!(
            "view {index} {}x{}: predicted focal {:.1} {:.1} px{given}, camera at ({:.3}, {:.3}, {:.3}), \
             {:.3} from the scene center, {off_axis:.1} deg off axis, depth {min_depth:.3} to {max_depth:.3}",
            view.point_map.width,
            view.point_map.height,
            camera.focal[0],
            camera.focal[1],
            camera.translation.x,
            camera.translation.y,
            camera.translation.z,
            to_center.norm()
        );
        if let Some(dir) = &args.debug_dir {
            std::fs::create_dir_all(dir)?;
            let (width, height) = (view.point_map.width, view.point_map.height);
            depth_to_gray(&view.depth, width, height).save(dir.join(format!("view{index}_depth.png")))?;
            conf_to_gray(&view.point_map.conf, width, height).save(dir.join(format!("view{index}_conf.png")))?;
            image.rgb.save(dir.join(format!("view{index}_input.png")))?;
        }
    }

    if let Some(raw_out) = &args.raw_out {
        let tensors: HashMap<String, Tensor> = reconstruction
            .views
            .iter()
            .enumerate()
            .map(|(index, view)| view_tensors(index, view))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        candle_core::safetensors::save(&tensors, raw_out)?;
        println!("wrote {}", raw_out.display());
    }

    let mut points = Vec::new();
    for (view, image) in reconstruction.views.iter().zip(&images) {
        points.extend(confident_points(&view.point_map, &image.rgb, args.min_conf)?);
    }
    write_ply(&args.out, &points)?;
    println!("wrote {} points to {}", points.len(), args.out.display());
    Ok(())
}
