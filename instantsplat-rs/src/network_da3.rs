//! Depth Anything 3 as stage 1 (via depthanything3-rs). DA3 predicts a depth map and a
//! camera (pose and focal length) per view and does not take intrinsics as input, so a
//! known focal length cannot be given to it: as for MapAnything, the given intrinsics
//! replace the predicted ones afterwards and the points move onto the given rays. Without
//! one, DA3's own camera is used as is (its depth was unprojected through it).
//!
//! DA3's confidence is 1 + exp(logit) like MapAnything's (0 for points mapanything-rs's
//! multi-view cleaning rejected), so `--min-conf` means the same thing.
//!
//! `dense_views_from_reconstruction` is the part the browser demo shares (feature `da3`);
//! `dense_views` runs the whole thing from photo files (feature `networks`).

#[cfg(feature = "networks")]
use std::path::PathBuf;

#[cfg(feature = "networks")]
use anyhow::Result;
use depthanything3_rs::pipeline::Reconstruction;
#[cfg(feature = "networks")]
use depthanything3_rs::reference_view::ReferenceViewStrategy;
#[cfg(feature = "networks")]
use depthanything3_rs::variant::Variant;
use mapanything_rs::image_input::InputImage;

use crate::camera::PinholeCamera;
use crate::dense_view::DenseView;
use crate::network::use_given_intrinsics;
#[cfg(feature = "networks")]
use crate::network::{NetworkSettings, NetworkTiming, device};

/// Upstream's default; every pose is moved to view 0's frame afterwards anyway.
#[cfg(feature = "networks")]
const REFERENCE_VIEW: ReferenceViewStrategy = ReferenceViewStrategy::SaddleBalanced;

#[cfg(feature = "networks")]
pub fn dense_views(images: &[PathBuf], settings: &NetworkSettings, variant: Variant) -> Result<(Vec<DenseView>, NetworkTiming)> {
    use depthanything3_rs::pipeline::reconstruct;
    use depthanything3_rs::preprocess::load_images;
    use depthanything3_rs::weights::load_model;

    let device = device(settings)?;
    let loaded = load_model(variant, REFERENCE_VIEW, settings.weights.as_ref(), settings.dtype, &device)?;
    println!("loaded {} in {:.2?}", loaded.weights_path.display(), loaded.load_time);

    let inputs = load_images(images, settings.focal_35mm, settings.dtype, &device)?;
    let known = inputs.iter().filter(|image| image.intrinsics.is_some()).count();
    println!("camera intrinsics known for {known} of {} images", inputs.len());
    let started = std::time::Instant::now();
    let reconstruction = reconstruct(&loaded.model, &inputs, &device)?;
    let inference = started.elapsed();

    let views = dense_views_from_reconstruction(&reconstruction, &inputs);
    Ok((
        views,
        NetworkTiming {
            load_weights: loaded.load_time,
            inference,
        },
    ))
}

/// DA3's reconstruction of `inputs` (the images it ran on, in the same order) as views:
/// the predicted camera, or the input's known intrinsics with the points moved onto their
/// rays.
pub fn dense_views_from_reconstruction(reconstruction: &Reconstruction, inputs: &[InputImage]) -> Vec<DenseView> {
    assert_eq!(reconstruction.views.len(), inputs.len(), "one reconstructed view per input image");
    reconstruction
        .views
        .iter()
        .zip(inputs)
        .map(|(predicted, input)| {
            let map = &predicted.point_map;
            let camera = &predicted.camera;
            let mut view = DenseView {
                width: map.width,
                height: map.height,
                points: map.points.clone(),
                conf: map.conf.clone(),
                foreground: map.foreground.clone(),
                rgb: input.rgb.as_raw().iter().map(|&c| c as f32 / 255.0).collect(),
                camera: PinholeCamera {
                    rotation: camera.rotation,
                    translation: camera.translation,
                    focal: camera.focal,
                    principal_point: camera.principal_point,
                    width: map.width as u32,
                    height: map.height as u32,
                },
            };
            match input.intrinsics {
                Some(given) => use_given_intrinsics(&mut view, given, camera.focal),
                None => println!(
                    "view {}x{}: DA3 focal {:.1} {:.1} px, principal point {:.1} {:.1}",
                    view.width, view.height, camera.focal[0], camera.focal[1], camera.principal_point[0], camera.principal_point[1]
                ),
            }
            view.check();
            view
        })
        .collect()
}
