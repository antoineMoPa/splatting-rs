//! A COLMAP model's frames put into the splat's frame: the similarity that takes COLMAP's
//! cameras onto the splat's (by the frames they share, residuals printed), the stray
//! cameras COLMAP sometimes places hundreds of units away, and the posed photos loaded as
//! training views. Shared by `refine` (trains on them) and `evaluate_against_colmap`
//! (holds them out).

use std::path::{Path, PathBuf};

use anyhow::Result;
use nalgebra::Vector3;

use crate::camera::{PinholeCamera, cameras_extent};
use crate::colmap_model::ColmapImage;
use crate::dense_view::DenseView;
use crate::similarity_align::{AlignmentResidual, CameraPair, Similarity, align_cameras};
use crate::training_frames::load_training_view;

/// A COLMAP frame posed in the splat's frame, with its photo.
pub struct PosedFrame {
    pub name: String,
    pub path: PathBuf,
    pub camera: PinholeCamera,
}

/// COLMAP world to the frame of `cameras` (named by frame), from the frames both have;
/// prints every residual under `label`, and returns them.
pub fn align_colmap_to_cameras(
    colmap: &[ColmapImage],
    cameras: &[(String, PinholeCamera)],
    label: &str,
) -> (Similarity, Vec<AlignmentResidual>) {
    let pairs: Vec<CameraPair> = cameras
        .iter()
        .filter_map(|(name, to)| {
            colmap
                .iter()
                .find(|image| &image.name == name)
                .map(|image| CameraPair { name, from: &image.camera, to })
        })
        .collect();
    println!(
        "alignment of COLMAP onto the {label}: {} of the {} cameras are in the COLMAP model ({} images)",
        pairs.len(),
        cameras.len(),
        colmap.len()
    );
    let (similarity, residuals) = align_cameras(&pairs);
    let extent = cameras_extent(&cameras.iter().map(|(_, camera)| camera.clone()).collect::<Vec<_>>());
    println!("COLMAP -> {label}: scale {:.4}, camera extent {extent:.3}", similarity.scale);
    for residual in &residuals {
        println!(
            "  {}: center off by {:.4} ({:.1}% of the extent), viewing direction off by {:.2} deg",
            residual.name,
            residual.center_error,
            100.0 * residual.center_error / extent,
            residual.direction_error_degrees
        );
    }
    let count = residuals.len() as f64;
    let mean_center = residuals.iter().map(|r| r.center_error).sum::<f64>() / count;
    let mean_direction = residuals.iter().map(|r| r.direction_error_degrees).sum::<f64>() / count;
    let max_center = residuals.iter().map(|r| r.center_error).fold(0.0, f64::max);
    let max_direction = residuals.iter().map(|r| r.direction_error_degrees).fold(0.0, f64::max);
    println!(
        "  mean: center {:.4} ({:.1}% of the extent), direction {:.2} deg; worst: center {:.4} ({:.1}%), direction {:.2} deg",
        mean_center,
        100.0 * mean_center / extent,
        mean_direction,
        max_center,
        100.0 * max_center / extent,
        max_direction
    );
    (similarity, residuals)
}

/// A camera farther than this times the median camera distance from the median camera
/// center is a mis-registered frame (COLMAP occasionally places one hundreds of units
/// away) and is left out: it would blow up the camera extent that scales the position
/// learning rate.
pub const MAX_CENTER_DISTANCE_OVER_MEDIAN: f64 = 3.0;

fn median(mut values: Vec<f64>) -> f64 {
    assert!(!values.is_empty(), "median of nothing");
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
    values[values.len() / 2]
}

/// The COLMAP images without the stray ones; each dropped image is printed.
pub fn without_stray_cameras(colmap: &[ColmapImage]) -> Vec<ColmapImage> {
    let median_center = Vector3::from_fn(|axis, _| median(colmap.iter().map(|image| image.camera.translation[axis]).collect()));
    let distances: Vec<f64> = colmap.iter().map(|image| (image.camera.translation - median_center).norm()).collect();
    let limit = MAX_CENTER_DISTANCE_OVER_MEDIAN * median(distances.clone());
    let kept: Vec<ColmapImage> = colmap
        .iter()
        .zip(&distances)
        .filter(|(image, distance)| {
            let distance = **distance;
            if distance > limit {
                println!(
                    "dropping {}: its camera is {distance:.1} from the median camera center, over {limit:.1} ({MAX_CENTER_DISTANCE_OVER_MEDIAN} x the median distance)",
                    image.name
                );
            }
            distance <= limit
        })
        .map(|(image, _)| image.clone())
        .collect();
    assert!(kept.len() >= 3, "only {} COLMAP cameras left after dropping the stray ones", kept.len());
    kept
}

/// The COLMAP frames whose photo exists in `frames`, posed in the splat's frame, in the
/// model's (name) order.
pub fn posed_frames(colmap: &[ColmapImage], frames: &Path, similarity: &Similarity) -> Vec<PosedFrame> {
    let (mut posed, mut missing) = (Vec::new(), Vec::new());
    for image in colmap {
        let path = frames.join(&image.name);
        if path.is_file() {
            posed.push(PosedFrame {
                name: image.name.clone(),
                path,
                camera: similarity.apply_camera(&image.camera),
            });
        } else {
            missing.push(image.name.as_str());
        }
    }
    assert!(!posed.is_empty(), "none of the {} COLMAP images is in {}", colmap.len(), frames.display());
    if !missing.is_empty() {
        println!("{} COLMAP images have no photo in {}: {}", missing.len(), frames.display(), missing.join(" "));
    }
    posed
}

/// Whether frame `index` (in `posed_frames` order) is held out: every `hold_out_every`-th
/// counting from the `hold_out_every`-th, so that the first frame always trains; 0 holds
/// out none.
pub fn is_held_out(index: usize, hold_out_every: usize) -> bool {
    assert!(hold_out_every != 1, "holding out every frame leaves nothing to train on");
    hold_out_every > 0 && (index + 1).is_multiple_of(hold_out_every)
}

/// Loads the frames' photos as views of `size`, on every core.
pub fn load_views(frames: &[PosedFrame], size: (u32, u32)) -> Result<Vec<DenseView>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = frames.len().div_ceil(threads).max(1);
    let loaded: Vec<Result<Vec<DenseView>>> = std::thread::scope(|scope| {
        let workers: Vec<_> = frames
            .chunks(chunk)
            .map(|frames| {
                scope.spawn(move || {
                    frames
                        .iter()
                        .map(|frame| load_training_view(&frame.path, &frame.camera, size))
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect();
        workers.into_iter().map(|worker| worker.join().expect("loader thread")).collect()
    });
    Ok(loaded.into_iter().collect::<Result<Vec<_>>>()?.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tenth_frame_from_the_tenth_is_held_out() {
        let held: Vec<usize> = (0..25).filter(|&index| is_held_out(index, 10)).collect();
        assert_eq!(held, vec![9, 19]);
        assert!((0..25).all(|index| !is_held_out(index, 0)));
    }

    #[test]
    #[should_panic(expected = "every frame")]
    fn holding_out_every_frame_is_refused() {
        is_held_out(0, 1);
    }
}
