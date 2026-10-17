//! Depth Anything 3 reconstruction: one forward pass over all images, each view's depth
//! unprojected through its predicted intrinsics into a point map placed by its predicted
//! pose, everything re-expressed in view 0's camera frame, then mapanything-rs's
//! multi-view cleaning.

use anyhow::Result;
use candle_core::{Device, Tensor};
use mapanything_rs::cleaning::{DepthView, zero_conf_in_front_of_other_views};
use mapanything_rs::image_input::InputImage;
use mapanything_rs::point_map::PointMap;
use mapanything_rs::reconstruction::relative_to_first;
use nalgebra::{Matrix3, Vector3};
use web_time::Instant;

use crate::model::{DepthAnything3, Progress, ViewOutput};

/// Camera of one view, in view 0's camera frame.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Camera to scene frame.
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// (fx, fy) in pixels of the model's input image.
    pub focal: [f64; 2],
    /// Pixel (i, j) at integer coordinates.
    pub principal_point: [f64; 2],
}

pub struct ReconstructedView {
    pub point_map: PointMap,
    /// Row-major depth (z in the view's camera frame) per pixel.
    pub depth: Vec<f32>,
    pub camera: Camera,
}

pub struct Reconstruction {
    pub views: Vec<ReconstructedView>,
    /// The view the network expressed its poses in before everything was moved to view
    /// 0's frame.
    pub reference_view: usize,
}

/// Pixel (x, y) at `depth` along z, through the view's intrinsics, in its camera frame.
fn unproject(view: &ViewOutput, x: usize, y: usize, depth: f32) -> Vector3<f64> {
    let depth = depth as f64;
    Vector3::new(
        (x as f64 - view.principal_point[0]) / view.focal[0] * depth,
        (y as f64 - view.principal_point[1]) / view.focal[1] * depth,
        depth,
    )
}

pub fn reconstruct(model: &DepthAnything3, images: &[InputImage], device: &Device) -> Result<Reconstruction> {
    reconstruct_with_progress(model, images, device, &mut |_| {})
}

/// `reconstruct`, reporting each stage of the forward pass to `progress`.
pub fn reconstruct_with_progress(
    model: &DepthAnything3,
    images: &[InputImage],
    device: &Device,
    progress: Progress,
) -> Result<Reconstruction> {
    let started = Instant::now();
    let batch = Tensor::cat(&images.iter().map(|image| image.tensor.clone()).collect::<Vec<_>>(), 0)?;
    let prediction = model.forward_with_progress(&batch, progress)?;
    device.synchronize()?;
    println!(
        "ran Depth Anything 3 on {} images in {:.2?} (reference view {})",
        images.len(),
        started.elapsed(),
        prediction.reference_view
    );

    let poses = relative_to_first(
        &prediction
            .views
            .iter()
            .map(|view| (view.rotation, view.translation))
            .collect::<Vec<_>>(),
    );
    let mut views: Vec<ReconstructedView> = prediction
        .views
        .iter()
        .zip(&poses)
        .zip(images)
        .map(|((view, (rotation, translation)), image)| {
            let points = (0..view.height)
                .flat_map(|y| (0..view.width).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let world = rotation * unproject(view, x, y, view.depth[y * view.width + x]) + translation;
                    [world.x as f32, world.y as f32, world.z as f32]
                })
                .collect();
            ReconstructedView {
                point_map: PointMap {
                    height: view.height,
                    width: view.width,
                    points,
                    conf: view.conf.clone(),
                    foreground: image.foreground.clone(),
                },
                depth: view.depth.clone(),
                camera: Camera {
                    rotation: *rotation,
                    translation: *translation,
                    focal: view.focal,
                    principal_point: view.principal_point,
                },
            }
        })
        .collect();

    let depth_views: Vec<DepthView> = views
        .iter()
        .map(|view| DepthView {
            rotation: view.camera.rotation,
            translation: view.camera.translation,
            focal: (view.camera.focal[0] + view.camera.focal[1]) / 2.0,
            principal_point: view.camera.principal_point,
            depth: &view.depth,
        })
        .collect();
    let mut point_maps: Vec<PointMap> = views.iter().map(|view| view.point_map.clone()).collect();
    zero_conf_in_front_of_other_views(&mut point_maps, &depth_views);
    drop(depth_views);
    for (view, point_map) in views.iter_mut().zip(point_maps) {
        view.point_map = point_map;
    }
    Ok(Reconstruction {
        views,
        reference_view: prediction.reference_view,
    })
}
