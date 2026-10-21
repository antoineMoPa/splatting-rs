//! The 3D views' turntable: the camera circles what the photos look at, above it, looking
//! at it, with no roll. Both are found from the photos' cameras, not from the network's
//! frame: its axes are the first photo's, tilted however that photo was held, and the
//! Gaussians' centroid sits wherever most points are (usually the floor).
//!
//! - The center is the point closest to every camera's optical axis (least squares): the
//!   object the photos were taken of.
//! - The vertical is the normal of the plane the cameras lie on (photos taken around an
//!   object are on a ring), on the side the cameras' own "down" points to. Cameras along a
//!   line have no such plane; their mean "down" is the vertical then.

use nalgebra::{Matrix3, SymmetricEigen, UnitVector3, Vector3};

use crate::camera::PinholeCamera;

/// The camera stays between these elevations above the horizontal plane through the
/// center: never under the object (the center is the object's middle, so lower than
/// about 15 degrees puts the camera in the floor it stands on), never straight down on it.
pub const MIN_ELEVATION: f64 = 15.0 * std::f64::consts::PI / 180.0;
pub const MAX_ELEVATION: f64 = 85.0 * std::f64::consts::PI / 180.0;
/// Camera positions whose second spread (covariance eigenvalue) is under this fraction of
/// their first lie along a line rather than a ring: their plane says nothing about up.
const MIN_RING_SPREAD: f64 = 0.05;
/// Optical axes whose crossing is this badly determined (smallest eigenvalue of the mean
/// of `I - d dᵀ`; 0 for parallel axes, 2/3 for axes in every direction) cross nowhere in
/// particular: the center is the caller's fallback then.
const MIN_AXES_SPREAD: f64 = 0.01;

#[derive(Clone, Copy, Debug)]
pub struct Turntable {
    pub center: Vector3<f64>,
    /// The scene's vertical, pointing down (OpenCV cameras: y down).
    pub down: UnitVector3<f64>,
}

impl Turntable {
    /// From the photos' cameras; `fallback_center` (say the Gaussians' centroid) when the
    /// cameras' axes do not cross, as for photos all taken facing the same way.
    pub fn from_cameras(cameras: &[PinholeCamera], fallback_center: Vector3<f64>) -> Turntable {
        assert!(!cameras.is_empty(), "no cameras");
        Turntable {
            center: axes_crossing(cameras).unwrap_or(fallback_center),
            down: vertical(cameras),
        }
    }

    /// `camera` moved onto the turntable: its distance to the center kept, its azimuth about
    /// the vertical plus `yaw` radians, its elevation plus `pitch` radians clamped to
    /// `MIN_ELEVATION..=MAX_ELEVATION`; looking at the center with no roll.
    pub fn view(&self, camera: &PinholeCamera, yaw: f64, pitch: f64) -> PinholeCamera {
        let down = self.down.into_inner();
        let offset = camera.translation - self.center;
        let distance = offset.norm();
        assert!(distance > 0.0, "the camera stands at the turntable's center");
        let height = -offset.dot(&down);
        let elevation = ((height / distance).asin() + pitch).clamp(MIN_ELEVATION, MAX_ELEVATION);
        // The camera's direction in the horizontal plane; any one when it is straight above.
        let horizontal = offset + height * down;
        let across = if horizontal.norm() > 1e-9 * distance {
            horizontal
        } else {
            down.cross(&Vector3::x()).cross(&down)
        };
        let across = across.normalize();
        let side = down.cross(&across);
        let azimuth = across * yaw.cos() + side * yaw.sin();
        let position = self.center + distance * (elevation.cos() * azimuth - elevation.sin() * down);
        let forward = UnitVector3::new_normalize(self.center - position);
        let right = UnitVector3::new_normalize(down.cross(&forward));
        PinholeCamera {
            rotation: Matrix3::from_columns(&[right.into_inner(), forward.cross(&right), forward.into_inner()]),
            translation: position,
            ..camera.clone()
        }
    }
}

/// The point closest to every camera's optical axis, unless the axes are too close to
/// parallel to cross anywhere (`MIN_AXES_SPREAD`).
fn axes_crossing(cameras: &[PinholeCamera]) -> Option<Vector3<f64>> {
    let (mut normal_matrix, mut right_side) = (Matrix3::zeros(), Vector3::zeros());
    for camera in cameras {
        let axis = camera.rotation.column(2).normalize();
        let off_axis = Matrix3::identity() - axis * axis.transpose();
        normal_matrix += off_axis;
        right_side += off_axis * camera.translation;
    }
    let smallest = SymmetricEigen::new(normal_matrix / cameras.len() as f64).eigenvalues.min();
    (smallest >= MIN_AXES_SPREAD).then(|| normal_matrix.lu().solve(&right_side).expect("a well-conditioned 3x3 system"))
}

/// The normal of the cameras' plane, on their "down" side, when they spread as a ring
/// (`MIN_RING_SPREAD`); else their mean "down".
fn vertical(cameras: &[PinholeCamera]) -> UnitVector3<f64> {
    let mean_down: Vector3<f64> = cameras.iter().map(|camera| camera.rotation.column(1).into_owned()).sum();
    assert!(mean_down.norm() > 0.0, "the cameras' down directions cancel out");
    let mean_position = cameras.iter().map(|camera| camera.translation).sum::<Vector3<f64>>() / cameras.len() as f64;
    let covariance: Matrix3<f64> = cameras
        .iter()
        .map(|camera| {
            let offset = camera.translation - mean_position;
            offset * offset.transpose()
        })
        .sum();
    let eigen = SymmetricEigen::new(covariance);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));
    let [first, second, third] = order.map(|index| eigen.eigenvalues[index]);
    let is_ring = cameras.len() >= 3 && first > 0.0 && second / first >= MIN_RING_SPREAD && third < second;
    if !is_ring {
        return UnitVector3::new_normalize(mean_down);
    }
    let normal = eigen.eigenvectors.column(order[2]).into_owned();
    UnitVector3::new_normalize(if normal.dot(&mean_down) >= 0.0 { normal } else { -normal })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    /// A camera at `position` looking at `target`, its right axis perpendicular to `down`.
    fn looking_at(position: Vector3<f64>, target: Vector3<f64>, down: Vector3<f64>) -> PinholeCamera {
        let forward = (target - position).normalize();
        let right = down.cross(&forward).normalize();
        PinholeCamera {
            rotation: Matrix3::from_columns(&[right, forward.cross(&right), forward]),
            translation: position,
            focal: [100.0, 100.0],
            principal_point: [10.0, 10.0],
            width: 20,
            height: 20,
        }
    }

    /// Six cameras on a ring 1 above `object` (along `-down`), radius 2, looking at it,
    /// each tilted by its own small roll so no camera's "down" is the true vertical.
    fn ring(object: Vector3<f64>, down: Vector3<f64>) -> Vec<PinholeCamera> {
        let across = down.cross(&Vector3::new(0.3, 0.1, 0.9)).normalize();
        let side = down.cross(&across);
        (0..6)
            .map(|index| {
                let angle = index as f64 * 0.9;
                let position = object + 2.0 * (across * angle.cos() + side * angle.sin()) - down;
                let camera = looking_at(position, object, down);
                let roll = Rotation3::from_axis_angle(
                    &UnitVector3::new_normalize(camera.rotation.column(2).into_owned()),
                    0.2 * (index as f64 - 2.5),
                );
                PinholeCamera {
                    rotation: roll.into_inner() * camera.rotation,
                    ..camera
                }
            })
            .collect()
    }

    fn elevation(turntable: &Turntable, camera: &PinholeCamera) -> f64 {
        let offset = camera.translation - turntable.center;
        (-offset.dot(&turntable.down) / offset.norm()).asin()
    }

    fn assert_looks_at_center_without_roll(turntable: &Turntable, camera: &PinholeCamera) {
        let (rotation, translation) = camera.world_to_camera();
        let in_camera = rotation * turntable.center + translation;
        assert!(
            in_camera.x.abs() < 1e-9 && in_camera.y.abs() < 1e-9 && in_camera.z > 0.0,
            "{in_camera}"
        );
        assert!(camera.rotation.column(0).dot(&turntable.down).abs() < 1e-9, "rolled");
        assert!((camera.rotation.determinant() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn finds_the_object_and_the_vertical_of_a_tilted_ring() {
        let object = Vector3::new(0.4, -0.2, 3.0);
        let down = Vector3::new(0.2, 0.9, 0.3).normalize();
        let turntable = Turntable::from_cameras(&ring(object, down), Vector3::zeros());
        assert!((turntable.center - object).norm() < 1e-9, "{}", turntable.center);
        assert!(
            (turntable.down.into_inner() - down).norm() < 1e-9,
            "{}",
            turntable.down.into_inner()
        );
    }

    #[test]
    fn views_keep_the_distance_look_at_the_center_and_stay_above() {
        let object = Vector3::new(0.4, -0.2, 3.0);
        let down = Vector3::new(0.2, 0.9, 0.3).normalize();
        let cameras = ring(object, down);
        let turntable = Turntable::from_cameras(&cameras, Vector3::zeros());
        let camera = &cameras[1];
        let distance = (camera.translation - object).norm();
        let start = turntable.view(camera, 0.0, 0.0);
        assert!(
            (start.translation - camera.translation).norm() < 1e-9,
            "yaw 0, pitch 0 keeps the position"
        );
        for (yaw, pitch) in [(0.7, 0.3), (-2.0, 0.0), (1.0, -3.0), (0.0, 3.0)] {
            let view = turntable.view(camera, yaw, pitch);
            assert!(((view.translation - object).norm() - distance).abs() < 1e-9);
            let elevation = elevation(&turntable, &view);
            assert!((MIN_ELEVATION - 1e-9..=MAX_ELEVATION + 1e-9).contains(&elevation), "{elevation}");
            assert_looks_at_center_without_roll(&turntable, &view);
        }
        assert!((elevation(&turntable, &turntable.view(camera, 0.0, -3.0)) - MIN_ELEVATION).abs() < 1e-9);
        let around = turntable.view(camera, std::f64::consts::TAU, 0.0);
        assert!((around.translation - start.translation).norm() < 1e-9, "a full turn comes back");
    }

    #[test]
    fn parallel_axes_take_the_fallback_center_and_the_mean_down() {
        let down = Vector3::new(0.0, 1.0, 0.0);
        let cameras: Vec<_> = (0..3)
            .map(|index| looking_at(Vector3::new(index as f64, 0.0, 0.0), Vector3::new(index as f64, 0.0, 5.0), down))
            .collect();
        let fallback = Vector3::new(1.0, 0.0, 5.0);
        let turntable = Turntable::from_cameras(&cameras, fallback);
        assert_eq!(turntable.center, fallback);
        assert!((turntable.down.into_inner() - down).norm() < 1e-12);
    }
}
