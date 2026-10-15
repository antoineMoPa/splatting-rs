//! MapAnything reconstruction: one forward pass over all images, then multi-view cleaning.

use std::time::Instant;

use anyhow::Result;
use candle_core::Device;
use nalgebra::{Matrix2, Vector2, Vector3};

use crate::cleaning::{DepthView, zero_conf_in_front_of_other_views};
use crate::image_input::InputImage;
use crate::model::{MapAnything, ViewOutput};
use crate::point_map::PointMap;
use crate::reconstruction::{Camera, Reconstruction, relative_to_first};

/// Intrinsics are fit on every (side / this)-th pixel along each side, as upstream.
const INTRINSICS_SAMPLES_PER_SIDE: usize = 50;

/// Pinhole intrinsics from the camera-frame points: per axis, the least-squares (center,
/// focal) of pixel = center + focal * (coordinate / z), as upstream's
/// `recover_pinhole_intrinsics_from_ray_directions`.
fn fit_intrinsics(view: &ViewOutput) -> ([f64; 2], [f64; 2]) {
    let step_y = (view.height / INTRINSICS_SAMPLES_PER_SIDE).max(1);
    let step_x = (view.width / INTRINSICS_SAMPLES_PER_SIDE).max(1);
    let mut normal = [Matrix2::<f64>::zeros(); 2];
    let mut rhs = [Vector2::<f64>::zeros(); 2];
    for y in (0..view.height).step_by(step_y) {
        for x in (0..view.width).step_by(step_x) {
            let [px, py, pz] = view.points_in_camera[y * view.width + x].map(|c| c as f64);
            for (axis, (pixel, coordinate)) in
                [(x as f64, px), (y as f64, py)].into_iter().enumerate()
            {
                let ratio = coordinate / pz;
                normal[axis] += Matrix2::new(1.0, ratio, ratio, ratio * ratio);
                rhs[axis] += Vector2::new(pixel, ratio * pixel);
            }
        }
    }
    let solve = |axis: usize| -> Vector2<f64> {
        normal[axis]
            .lu()
            .solve(&rhs[axis])
            .expect("rays must vary across the image")
    };
    let (x, y) = (solve(0), solve(1));
    ([x[0], y[0]], [x[1], y[1]])
}

pub fn reconstruct(
    model: &MapAnything,
    images: &[InputImage],
    device: &Device,
) -> Result<Reconstruction> {
    let started = Instant::now();
    let tensors: Vec<_> = images.iter().map(|image| image.tensor.clone()).collect();
    let intrinsics: Vec<_> = images.iter().map(|image| image.intrinsics).collect();
    let views = model.forward(&tensors, &intrinsics)?;
    device.synchronize()?;
    println!(
        "ran MapAnything on {} images in {:.2?}",
        images.len(),
        started.elapsed()
    );

    let poses = relative_to_first(
        &views
            .iter()
            .map(|view| (view.rotation, view.translation))
            .collect::<Vec<_>>(),
    );
    let fitted: Vec<([f64; 2], [f64; 2])> = views.iter().map(fit_intrinsics).collect();
    let mut point_maps: Vec<PointMap> = views
        .iter()
        .zip(&poses)
        .zip(images)
        .map(|((view, (rotation, translation)), image)| PointMap {
            height: view.height,
            width: view.width,
            points: view
                .points_in_camera
                .iter()
                .map(|point| {
                    let world =
                        rotation * Vector3::from_fn(|axis, _| point[axis] as f64) + translation;
                    [world.x as f32, world.y as f32, world.z as f32]
                })
                .collect(),
            // Pixels the model calls ambiguous are rejected, like inconsistent ones.
            conf: view
                .conf
                .iter()
                .zip(&view.unambiguous)
                .map(|(&conf, &unambiguous)| if unambiguous { conf } else { 0.0 })
                .collect(),
            foreground: image.foreground.clone(),
        })
        .collect();

    let depths: Vec<Vec<f32>> = views
        .iter()
        .map(|view| view.points_in_camera.iter().map(|point| point[2]).collect())
        .collect();
    let depth_views: Vec<DepthView> = poses
        .iter()
        .zip(&fitted)
        .zip(&depths)
        .map(
            |(((rotation, translation), (center, focal)), depth)| DepthView {
                rotation: *rotation,
                translation: *translation,
                focal: (focal[0] + focal[1]) / 2.0,
                principal_point: *center,
                depth,
            },
        )
        .collect();
    zero_conf_in_front_of_other_views(&mut point_maps, &depth_views);

    let cameras = poses
        .into_iter()
        .zip(&fitted)
        .map(|((rotation, translation), (_, focal))| Camera {
            rotation,
            translation,
            focal: (focal[0] + focal[1]) / 2.0,
        })
        .collect();
    Ok(Reconstruction {
        point_maps,
        cameras,
    })
}
