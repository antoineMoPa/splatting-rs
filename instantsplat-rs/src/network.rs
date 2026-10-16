//! Stage 1: MapAnything (via mapanything-rs, candle on Metal) turns the input images into
//! dense point maps and cameras. Requires the `mapanything` feature; candle has no WebGPU
//! backend, so this stage does not build for wasm32.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use mapanything_rs::model::MapAnything;
use mapanything_rs::pipeline::reconstruct;
use mapanything_rs::preprocess::load_images;

use crate::camera::PinholeCamera;
use crate::dense_view::DenseView;
use crate::pinhole_fit::fit_pinhole;
use crate::replace_rays::replace_rays;

const WEIGHTS_REPO: &str = "facebook/map-anything-apache";
const WEIGHTS_FILE: &str = "model.safetensors";

pub struct NetworkSettings {
    /// 35 mm-equivalent focal length for every image, overriding EXIF.
    pub focal_35mm: Option<f64>,
    pub dtype: DType,
    /// Local checkpoint; downloaded from Hugging Face when None.
    pub weights: Option<PathBuf>,
    pub cpu: bool,
}

pub struct NetworkTiming {
    pub load_weights: std::time::Duration,
    pub inference: std::time::Duration,
}

/// Runs MapAnything on `images` (view 0 first). The point maps are in view 0's camera
/// frame. Each view's intrinsics are the given ones (`focal_35mm` or EXIF, scaled by
/// mapanything-rs to its input size, principal point at the photo's center) when known,
/// and the points are moved onto the given rays (`replace_rays`, keeping the predicted
/// distance along each ray); otherwise the intrinsics are fit to the view's own point
/// map. Either way rendering the points through the camera lands on their pixels.
pub fn dense_views_from_images(
    images: &[PathBuf],
    settings: &NetworkSettings,
) -> Result<(Vec<DenseView>, NetworkTiming)> {
    let device = if settings.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    let weights = match &settings.weights {
        Some(path) => path.clone(),
        None => hf_hub::api::sync::Api::new()?
            .model(WEIGHTS_REPO.to_string())
            .get(WEIGHTS_FILE)?,
    };
    let started = Instant::now();
    // SAFETY: the checkpoint file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], settings.dtype, &device)? };
    let model = MapAnything::new(vb)?;
    let load_weights = started.elapsed();
    println!("loaded {} in {load_weights:.2?}", weights.display());

    let inputs = load_images(images, settings.focal_35mm, settings.dtype, &device)?;
    let known = inputs.iter().filter(|image| image.intrinsics.is_some()).count();
    println!("camera intrinsics known for {known} of {} images", inputs.len());
    let started = Instant::now();
    let reconstruction = reconstruct(&model, &inputs, &device)?;
    let inference = started.elapsed();

    let views = reconstruction
        .point_maps
        .iter()
        .zip(&reconstruction.cameras)
        .zip(&inputs)
        .map(|((point_map, camera), input)| {
            let pose = PinholeCamera {
                rotation: camera.rotation,
                translation: camera.translation,
                focal: [camera.focal; 2],
                principal_point: [0.0; 2],
                width: point_map.width as u32,
                height: point_map.height as u32,
            };
            let mut view = DenseView {
                width: point_map.width,
                height: point_map.height,
                points: point_map.points.clone(),
                conf: point_map.conf.clone(),
                foreground: point_map.foreground.clone(),
                rgb: input.rgb.as_raw().iter().map(|&c| c as f32 / 255.0).collect(),
                camera: pose,
            };
            match input.intrinsics {
                Some(given) => {
                    println!(
                        "view {}x{}: given focal {:.1} {:.1} px (mapanything fit {:.1}), principal point {:.1} {:.1}",
                        view.width, view.height, given.focal[0], given.focal[1], camera.focal, given.principal_point[0], given.principal_point[1]
                    );
                    view.camera.focal = given.focal;
                    view.camera.principal_point = given.principal_point;
                    // The network predicts its own rays regardless; put the points on
                    // the given ones so that they render onto their pixels.
                    replace_rays(&mut view);
                }
                None => {
                    let fit = fit_pinhole(&view);
                    println!(
                        "view {}x{}: fit focal {:.1} {:.1} px (mapanything {:.1}), principal point {:.1} {:.1}",
                        view.width, view.height, fit.focal[0], fit.focal[1], camera.focal, fit.principal_point[0], fit.principal_point[1]
                    );
                    view.camera.focal = fit.focal;
                    view.camera.principal_point = fit.principal_point;
                }
            }
            view.check();
            view
        })
        .collect();
    Ok((
        views,
        NetworkTiming {
            load_weights,
            inference,
        },
    ))
}

pub fn image_name(path: &Path) -> String {
    path.file_name()
        .expect("image path has a file name")
        .to_string_lossy()
        .into_owned()
}
