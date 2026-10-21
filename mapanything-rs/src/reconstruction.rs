//! The reconstruction: one dense point map and one camera per image, all in
//! the camera frame of image 0 (OpenCV convention: x right, y down, z forward).

use nalgebra::{Matrix3, Vector3};

use crate::point_map::PointMap;

pub struct Camera {
    /// Camera to scene frame.
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// In pixels of the model's input image.
    pub focal: f64,
}

pub struct Reconstruction {
    pub point_maps: Vec<PointMap>,
    pub cameras: Vec<Camera>,
}

/// Re-expresses camera-to-world poses relative to the first camera.
pub fn relative_to_first(poses: &[(Matrix3<f64>, Vector3<f64>)]) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
    let (first_rotation, first_translation) = poses[0];
    poses
        .iter()
        .map(|(rotation, translation)| {
            (
                first_rotation.transpose() * rotation,
                first_rotation.transpose() * (translation - first_translation),
            )
        })
        .collect()
}
