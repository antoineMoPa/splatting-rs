//! Pinhole cameras in the OpenCV convention (x right, y down, z forward), with pixel centers
//! at integer coordinates: pixel (i, j) of the image is at (u, v) = (i, j).

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix3, Rotation3, UnitVector3, Vector3};

#[derive(Clone, Debug)]
pub struct PinholeCamera {
    /// Camera to world.
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// In pixels of the image this camera renders.
    pub focal: [f64; 2],
    pub principal_point: [f64; 2],
    pub width: u32,
    pub height: u32,
}

/// What the projection shaders read: world-to-camera as a 3x4 (row-major, padded to vec4
/// rows) and the intrinsics.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub world_to_camera_rows: [[f32; 4]; 3],
    pub focal: [f32; 2],
    pub principal_point: [f32; 2],
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
}

/// What the backward pass sums over the Gaussians of one view (`project_backward.wgsl`,
/// `reduce_camera_gradient.wgsl`): the loss gradient with respect to the entries of the
/// world-to-camera rotation (rows, as `CameraUniform`), the world-to-camera translation
/// and the focal lengths. `camera_pose::pose_gradient` chains it to the pose parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CameraGradient {
    pub world_to_camera_rotation_rows: [[f32; 3]; 3],
    pub world_to_camera_translation: [f32; 3],
    pub focal: [f32; 2],
}

impl PinholeCamera {
    pub fn world_to_camera(&self) -> (Matrix3<f64>, Vector3<f64>) {
        let rotation = self.rotation.transpose();
        (rotation, -(rotation * self.translation))
    }

    pub fn uniform(&self, tile_size: u32) -> CameraUniform {
        let (rotation, translation) = self.world_to_camera();
        let row = |r: usize| {
            [
                rotation[(r, 0)] as f32,
                rotation[(r, 1)] as f32,
                rotation[(r, 2)] as f32,
                translation[r] as f32,
            ]
        };
        CameraUniform {
            world_to_camera_rows: [row(0), row(1), row(2)],
            focal: [self.focal[0] as f32, self.focal[1] as f32],
            principal_point: [self.principal_point[0] as f32, self.principal_point[1] as f32],
            width: self.width,
            height: self.height,
            tiles_x: self.width.div_ceil(tile_size),
            tiles_y: self.height.div_ceil(tile_size),
        }
    }

    pub fn position(&self) -> Vector3<f64> {
        self.translation
    }

    /// This camera rendering a `width` x `height` image instead: same pose and focal
    /// lengths, the old image's center at the new one's, so a wider image shows more of the
    /// scene at the sides rather than a stretched view.
    pub fn with_image_size(&self, width: u32, height: u32) -> PinholeCamera {
        let shift = |new: u32, old: u32| (new as f64 - old as f64) / 2.0;
        PinholeCamera {
            principal_point: [
                self.principal_point[0] + shift(width, self.width),
                self.principal_point[1] + shift(height, self.height),
            ],
            width,
            height,
            ..self.clone()
        }
    }

    /// This camera swung about `center` (a novel viewpoint): `pitch` radians about its own
    /// right axis, then `yaw` radians about the scene's vertical axis (y, camera 0's
    /// down); it keeps looking at what it looked at relative to `center`.
    pub fn orbited(&self, center: Vector3<f64>, yaw: f64, pitch: f64) -> PinholeCamera {
        let right = UnitVector3::new_normalize(self.rotation.column(0).into_owned());
        let orbit = (Rotation3::from_axis_angle(&Vector3::y_axis(), yaw)
            * Rotation3::from_axis_angle(&right, pitch))
        .into_inner();
        PinholeCamera {
            rotation: orbit * self.rotation,
            translation: center + orbit * (self.translation - center),
            ..self.clone()
        }
    }
}

/// Radius of the smallest sphere centered on the mean camera position that contains every
/// camera, times 1.1, as the reference 3DGS `cameras_extent`. Scales the position
/// learning rate.
pub fn cameras_extent(cameras: &[PinholeCamera]) -> f64 {
    assert!(!cameras.is_empty(), "no cameras");
    let center = cameras
        .iter()
        .map(PinholeCamera::position)
        .sum::<Vector3<f64>>()
        / cameras.len() as f64;
    let radius = cameras
        .iter()
        .map(|camera| (camera.position() - center).norm())
        .fold(0.0, f64::max);
    radius * 1.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_camera_inverts_pose() {
        let rotation = nalgebra::Rotation3::from_euler_angles(0.3, -0.2, 0.9).into_inner();
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let camera = PinholeCamera {
            rotation,
            translation,
            focal: [100.0, 100.0],
            principal_point: [10.0, 10.0],
            width: 20,
            height: 20,
        };
        let (w_rotation, w_translation) = camera.world_to_camera();
        let point = Vector3::new(0.5, -1.0, 4.0);
        let in_camera = w_rotation * point + w_translation;
        let back = rotation * in_camera + translation;
        assert!((back - point).norm() < 1e-12);
    }

    #[test]
    fn orbit_keeps_the_distance_to_the_center_and_the_view_of_it() {
        let camera = PinholeCamera {
            rotation: nalgebra::Rotation3::from_euler_angles(0.1, -0.4, 0.2).into_inner(),
            translation: Vector3::new(0.5, -0.3, -2.0),
            focal: [100.0, 100.0],
            principal_point: [10.0, 10.0],
            width: 20,
            height: 20,
        };
        let center = Vector3::new(0.1, 0.2, 1.0);
        let orbited = camera.orbited(center, 0.7, -0.3);
        assert!(((orbited.translation - center).norm() - (camera.translation - center).norm()).abs() < 1e-12);
        // The center lands on the same pixel: same camera-frame coordinates before and after.
        let in_camera = |camera: &PinholeCamera| {
            let (rotation, translation) = camera.world_to_camera();
            rotation * center + translation
        };
        assert!((in_camera(&orbited) - in_camera(&camera)).norm() < 1e-12);
        assert_eq!(camera.orbited(center, 0.0, 0.0).translation, camera.translation);
    }

    #[test]
    fn a_wider_image_keeps_the_center_and_the_scale() {
        let camera = PinholeCamera {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
            focal: [100.0, 100.0],
            principal_point: [9.5, 14.5],
            width: 20,
            height: 30,
        };
        let wider = camera.with_image_size(40, 30);
        assert_eq!(wider.principal_point, [19.5, 14.5]);
        assert_eq!((wider.width, wider.height), (40, 30));
        assert_eq!(wider.focal, camera.focal);
        // A point one unit right at depth one lands 100 pixels right of the center in both.
        let pixel = |camera: &PinholeCamera| camera.principal_point[0] + camera.focal[0] * 1.0;
        assert_eq!(pixel(&wider) - wider.principal_point[0], pixel(&camera) - camera.principal_point[0]);
    }

    #[test]
    fn extent_is_1_1_times_farthest_camera() {
        let at = |x: f64| PinholeCamera {
            rotation: Matrix3::identity(),
            translation: Vector3::new(x, 0.0, 0.0),
            focal: [1.0, 1.0],
            principal_point: [0.0, 0.0],
            width: 1,
            height: 1,
        };
        let extent = cameras_extent(&[at(-1.0), at(1.0), at(3.0)]);
        assert!((extent - 2.2).abs() < 1e-12);
    }
}
