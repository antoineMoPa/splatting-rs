//! Camera pose refinement parameters: per view a 6-DoF correction on top of the initial
//! camera-to-world pose (a rotation vector applied to the initial rotation, a translation
//! of the camera center) and one focal correction shared by every view; and the chain rule
//! from what the backward pass sums over the Gaussians (`CameraGradient`: the loss gradient
//! with respect to the world-to-camera entries and the focal) to those parameters.
//!
//! Camera-to-world rotation R = exp([omega]) R0, center C = C0 + t, focal f = f0 exp(k);
//! what the shaders see is W = R^T and t_wc = -W C.

use nalgebra::{Matrix3, Vector3};

use crate::camera::{CameraGradient, PinholeCamera};

/// What one view's pose is corrected by.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PoseCorrection {
    /// Rotation vector (axis times angle, radians), applied in the world frame.
    pub rotation: Vector3<f64>,
    /// Added to the camera center.
    pub translation: Vector3<f64>,
}

/// dL/d(rotation vector), dL/d(translation), dL/d(log focal scale) of one view.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PoseGradient {
    pub rotation: Vector3<f64>,
    pub translation: Vector3<f64>,
    pub focal_log_scale: f64,
}

/// Below this angle the series expansions of `so3_exp` and `so3_left_jacobian` are used.
const SMALL_ANGLE: f64 = 1e-6;

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// The rotation matrix of a rotation vector (Rodrigues).
pub fn so3_exp(omega: &Vector3<f64>) -> Matrix3<f64> {
    let angle = omega.norm();
    let k = skew(omega);
    if angle < SMALL_ANGLE {
        return Matrix3::identity() + k + 0.5 * k * k;
    }
    Matrix3::identity() + angle.sin() / angle * k + (1.0 - angle.cos()) / (angle * angle) * k * k
}

/// J_l(omega): exp([omega + delta]) = exp([J_l(omega) delta]) exp([omega]) to first order.
pub fn so3_left_jacobian(omega: &Vector3<f64>) -> Matrix3<f64> {
    let angle = omega.norm();
    let k = skew(omega);
    if angle < SMALL_ANGLE {
        return Matrix3::identity() + 0.5 * k + k * k / 6.0;
    }
    let angle2 = angle * angle;
    Matrix3::identity() + (1.0 - angle.cos()) / angle2 * k + (angle - angle.sin()) / (angle2 * angle) * k * k
}

/// The initial camera with the correction and the shared focal scale applied.
pub fn corrected_camera(initial: &PinholeCamera, correction: &PoseCorrection, focal_log_scale: f64) -> PinholeCamera {
    let focal_scale = focal_log_scale.exp();
    PinholeCamera {
        rotation: so3_exp(&correction.rotation) * initial.rotation,
        translation: initial.translation + correction.translation,
        focal: [initial.focal[0] * focal_scale, initial.focal[1] * focal_scale],
        ..initial.clone()
    }
}

/// The vector v of a skew-symmetric matrix [v].
fn vee(skew: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(skew[(2, 1)], skew[(0, 2)], skew[(1, 0)])
}

/// Chains `gradient`, summed at the camera `corrected_camera(initial, correction, k)`, to
/// the pose parameters. For a left perturbation R' = exp([phi]) R of the camera-to-world
/// rotation, W' = W exp(-[phi]) and t_wc' = -W' C, so that
/// dL/dphi = vee(A^T - A) + C x (W^T dL/dt_wc) with A = W^T dL/dW, and the rotation
/// vector's gradient is J_l(omega)^T dL/dphi. The center: dL/dC = -W^T dL/dt_wc. The focal:
/// dL/dk = fx dL/dfx + fy dL/dfy.
pub fn pose_gradient(
    initial: &PinholeCamera,
    correction: &PoseCorrection,
    focal_log_scale: f64,
    gradient: &CameraGradient,
) -> PoseGradient {
    let camera = corrected_camera(initial, correction, focal_log_scale);
    let (w, _) = camera.world_to_camera();
    let d_w = Matrix3::from_fn(|row, col| gradient.world_to_camera_rotation_rows[row][col] as f64);
    let d_t_wc = Vector3::from_fn(|i, _| gradient.world_to_camera_translation[i] as f64);
    let a = w.transpose() * d_w;
    let d_center = -(w.transpose() * d_t_wc);
    let d_phi = vee(&(a.transpose() - a)) - camera.translation.cross(&d_center);
    PoseGradient {
        rotation: so3_left_jacobian(&correction.rotation).transpose() * d_phi,
        translation: d_center,
        focal_log_scale: camera.focal[0] * gradient.focal[0] as f64 + camera.focal[1] * gradient.focal[1] as f64,
    }
}

/// Angle in degrees between the two cameras' rotations and the distance between their
/// centers: how far `camera` is from `reference`.
pub fn pose_distance(camera: &PinholeCamera, reference: &PinholeCamera) -> (f64, f64) {
    let relative = camera.rotation * reference.rotation.transpose();
    let cosine = ((relative.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    (cosine.acos().to_degrees(), (camera.translation - reference.translation).norm())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    #[test]
    fn exp_matches_axis_angle() {
        let omega = Vector3::new(0.3, -0.5, 0.8);
        let expected = Rotation3::from_scaled_axis(omega).into_inner();
        assert!((so3_exp(&omega) - expected).norm() < 1e-12);
        assert!((so3_exp(&Vector3::new(1e-9, 0.0, 0.0)) - Matrix3::identity()).norm() < 1e-8);
    }

    #[test]
    fn left_jacobian_matches_finite_differences_of_exp() {
        for omega in [Vector3::new(0.3, -0.5, 0.8), Vector3::new(1e-8, 2e-8, 0.0)] {
            let jacobian = so3_left_jacobian(&omega);
            let epsilon = 1e-6;
            for axis in 0..3 {
                let mut delta = Vector3::zeros();
                delta[axis] = epsilon;
                // exp([omega + delta]) exp([omega])^T is exp([J_l delta]) = I + [J_l delta].
                let left = so3_exp(&(omega + delta)) * so3_exp(&omega).transpose();
                let phi = vee(&(left - Matrix3::identity())) / epsilon;
                let expected = jacobian.column(axis);
                assert!((phi - expected).norm() < 1e-5, "axis {axis}: {phi} vs {expected}");
            }
        }
    }

    fn camera() -> PinholeCamera {
        PinholeCamera {
            rotation: Rotation3::from_euler_angles(0.4, -0.7, 1.1).into_inner(),
            translation: Vector3::new(1.5, -2.0, 0.7),
            focal: [40.0, 42.0],
            principal_point: [16.0, 16.0],
            width: 32,
            height: 32,
        }
    }

    /// A loss linear in the camera entries the shaders read, with the given coefficients
    /// as its exact gradient; finite differences over the pose parameters check the chain.
    #[test]
    fn pose_gradient_matches_finite_differences_of_a_linear_loss() {
        let initial = camera();
        let gradient = CameraGradient {
            world_to_camera_rotation_rows: [[0.3, -1.2, 0.7], [2.0, 0.1, -0.4], [-0.9, 0.5, 1.3]],
            world_to_camera_translation: [0.8, -0.6, 1.1],
            focal: [0.02, -0.05],
        };
        let loss = |correction: &PoseCorrection, focal_log_scale: f64| -> f64 {
            let camera = corrected_camera(&initial, correction, focal_log_scale);
            let (w, t_wc) = camera.world_to_camera();
            let mut sum = 0.0;
            for row in 0..3 {
                for col in 0..3 {
                    sum += gradient.world_to_camera_rotation_rows[row][col] as f64 * w[(row, col)];
                }
                sum += gradient.world_to_camera_translation[row] as f64 * t_wc[row];
            }
            sum + gradient.focal[0] as f64 * camera.focal[0] + gradient.focal[1] as f64 * camera.focal[1]
        };
        let correction = PoseCorrection {
            rotation: Vector3::new(0.2, -0.1, 0.3),
            translation: Vector3::new(-0.3, 0.4, 0.1),
        };
        let focal_log_scale = 0.05;
        let analytic = pose_gradient(&initial, &correction, focal_log_scale, &gradient);
        let epsilon = 1e-6;
        for axis in 0..3 {
            let mut delta = Vector3::zeros();
            delta[axis] = epsilon;
            let rotated = |sign: f64| PoseCorrection {
                rotation: correction.rotation + sign * delta,
                ..correction
            };
            let numeric = (loss(&rotated(1.0), focal_log_scale) - loss(&rotated(-1.0), focal_log_scale)) / (2.0 * epsilon);
            assert!(
                (numeric - analytic.rotation[axis]).abs() < 1e-6,
                "rotation {axis}: {numeric} vs {}",
                analytic.rotation[axis]
            );
            let moved = |sign: f64| PoseCorrection {
                translation: correction.translation + sign * delta,
                ..correction
            };
            let numeric = (loss(&moved(1.0), focal_log_scale) - loss(&moved(-1.0), focal_log_scale)) / (2.0 * epsilon);
            assert!(
                (numeric - analytic.translation[axis]).abs() < 1e-6,
                "translation {axis}: {numeric} vs {}",
                analytic.translation[axis]
            );
        }
        let numeric = (loss(&correction, focal_log_scale + epsilon) - loss(&correction, focal_log_scale - epsilon)) / (2.0 * epsilon);
        assert!(
            (numeric - analytic.focal_log_scale).abs() < 1e-6,
            "focal: {numeric} vs {}",
            analytic.focal_log_scale
        );
    }

    #[test]
    fn pose_distance_reads_angle_and_center() {
        let reference = camera();
        let moved = corrected_camera(
            &reference,
            &PoseCorrection {
                rotation: Vector3::new(0.0, 0.1, 0.0),
                translation: Vector3::new(0.3, 0.0, 0.4),
            },
            0.0,
        );
        let (angle, distance) = pose_distance(&moved, &reference);
        assert!((angle - 0.1f64.to_degrees()).abs() < 1e-9, "{angle}");
        assert!((distance - 0.5).abs() < 1e-12, "{distance}");
    }
}
