//! MapAnything as stage 1 (via mapanything-rs). Without a known focal length the
//! intrinsics are fit to the view's own point map (`pinhole_fit`), since MapAnything
//! predicts rays, not a camera.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle_nn::VarBuilder;
use mapanything_rs::model::MapAnything;
use mapanything_rs::pipeline::reconstruct;
use mapanything_rs::preprocess::load_images;

use crate::camera::PinholeCamera;
use crate::dense_view::DenseView;
use crate::network::{NetworkSettings, NetworkTiming, device, use_given_intrinsics};
use crate::pinhole_fit::fit_pinhole;

pub const WEIGHTS_REPO: &str = "facebook/map-anything-apache";
pub const WEIGHTS_FILE: &str = "model.safetensors";

pub fn dense_views(images: &[PathBuf], settings: &NetworkSettings) -> Result<(Vec<DenseView>, NetworkTiming)> {
    let device = device(settings)?;
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
            let mut view = DenseView {
                width: point_map.width,
                height: point_map.height,
                points: point_map.points.clone(),
                conf: point_map.conf.clone(),
                foreground: point_map.foreground.clone(),
                rgb: input.rgb.as_raw().iter().map(|&c| c as f32 / 255.0).collect(),
                camera: PinholeCamera {
                    rotation: camera.rotation,
                    translation: camera.translation,
                    focal: [camera.focal; 2],
                    principal_point: [0.0; 2],
                    width: point_map.width as u32,
                    height: point_map.height as u32,
                },
            };
            match input.intrinsics {
                Some(given) => use_given_intrinsics(&mut view, given, [camera.focal; 2]),
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
    Ok((views, NetworkTiming { load_weights, inference }))
}
