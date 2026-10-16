//! The similarity (scale, rotation, translation) that takes one set of cameras onto
//! another, by frame name: Umeyama / Kabsch on the camera centers, with the cameras'
//! viewing directions added to the rotation's cross-covariance so that a few nearly
//! collinear centers still pin the rotation down. Used to put a COLMAP model into the
//! splat's frame.

use nalgebra::{Matrix3, Vector3};

use crate::camera::PinholeCamera;

/// x_to = scale * rotation * x_from + translation.
#[derive(Clone, Debug)]
pub struct Similarity {
    pub scale: f64,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

/// How far a camera lands from its target after the alignment.
#[derive(Clone, Debug)]
pub struct AlignmentResidual {
    pub name: String,
    /// Distance between the mapped center and the target center, in target units.
    pub center_error: f64,
    /// Angle between the mapped viewing direction and the target's, in degrees.
    pub direction_error_degrees: f64,
}

/// Centers of two cameras of the same frame: `from` and `to`.
pub struct CameraPair<'a> {
    pub name: &'a str,
    pub from: &'a PinholeCamera,
    pub to: &'a PinholeCamera,
}

pub const MIN_PAIRS: usize = 3;

impl Similarity {
    pub fn apply_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.scale * (self.rotation * point) + self.translation
    }

    /// The same camera, posed in the target frame (intrinsics unchanged).
    pub fn apply_camera(&self, camera: &PinholeCamera) -> PinholeCamera {
        PinholeCamera {
            rotation: self.rotation * camera.rotation,
            translation: self.apply_point(&camera.translation),
            ..camera.clone()
        }
    }
}

/// The optical axis in world coordinates (OpenCV: camera z).
fn viewing_direction(camera: &PinholeCamera) -> Vector3<f64> {
    camera.rotation.column(2).into_owned()
}

fn mean(points: &[Vector3<f64>]) -> Vector3<f64> {
    points.iter().sum::<Vector3<f64>>() / points.len() as f64
}

/// The rotation maximizing trace(R * cross_covariance), with reflections rejected.
fn rotation_from_cross_covariance(cross_covariance: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = cross_covariance.svd(true, true);
    let (u, v_t) = (svd.u.expect("svd u"), svd.v_t.expect("svd v_t"));
    let mut correction = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    u * correction * v_t
}

/// Panics with fewer than `MIN_PAIRS` pairs or when the `from` centers all coincide.
pub fn align_cameras(pairs: &[CameraPair]) -> (Similarity, Vec<AlignmentResidual>) {
    assert!(
        pairs.len() >= MIN_PAIRS,
        "{} camera pairs to align, need at least {MIN_PAIRS}",
        pairs.len()
    );
    let from: Vec<Vector3<f64>> = pairs.iter().map(|pair| pair.from.translation).collect();
    let to: Vec<Vector3<f64>> = pairs.iter().map(|pair| pair.to.translation).collect();
    let (from_mean, to_mean) = (mean(&from), mean(&to));
    let rms = |points: &[Vector3<f64>], center: &Vector3<f64>| {
        (points.iter().map(|p| (p - center).norm_squared()).sum::<f64>() / points.len() as f64).sqrt()
    };
    let (from_rms, to_rms) = (rms(&from, &from_mean), rms(&to, &to_mean));
    assert!(from_rms > 0.0, "all source camera centers coincide");
    assert!(to_rms > 0.0, "all target camera centers coincide");

    // Centers scaled to unit RMS radius and unit viewing directions, so the two kinds of
    // correspondence weigh the same in the rotation.
    let mut cross_covariance = Matrix3::zeros();
    for (pair, (p, q)) in pairs.iter().zip(from.iter().zip(&to)) {
        cross_covariance += ((q - to_mean) / to_rms) * ((p - from_mean) / from_rms).transpose();
        cross_covariance += viewing_direction(pair.to) * viewing_direction(pair.from).transpose();
    }
    let rotation = rotation_from_cross_covariance(&cross_covariance);
    // Least-squares scale for this rotation.
    let scale = from
        .iter()
        .zip(&to)
        .map(|(p, q)| (q - to_mean).dot(&(rotation * (p - from_mean))))
        .sum::<f64>()
        / from.iter().map(|p| (p - from_mean).norm_squared()).sum::<f64>();
    assert!(scale > 0.0, "negative alignment scale {scale}");
    let similarity = Similarity {
        scale,
        rotation,
        translation: to_mean - scale * (rotation * from_mean),
    };

    let residuals = pairs
        .iter()
        .map(|pair| {
            let mapped = similarity.apply_camera(pair.from);
            let cosine = viewing_direction(&mapped).dot(&viewing_direction(pair.to)).clamp(-1.0, 1.0);
            AlignmentResidual {
                name: pair.name.to_string(),
                center_error: (mapped.translation - pair.to.translation).norm(),
                direction_error_degrees: cosine.acos().to_degrees(),
            }
        })
        .collect();
    (similarity, residuals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn camera(rotation: Matrix3<f64>, translation: Vector3<f64>) -> PinholeCamera {
        PinholeCamera {
            rotation,
            translation,
            focal: [100.0; 2],
            principal_point: [10.0; 2],
            width: 20,
            height: 20,
        }
    }

    /// Cameras on an orbit looking at the origin, as a walk-around capture.
    fn orbit(count: usize) -> Vec<PinholeCamera> {
        (0..count)
            .map(|i| {
                let angle = i as f64 / count as f64 * std::f64::consts::TAU;
                let center = Vector3::new(3.0 * angle.cos(), 0.2 * (i as f64).sin(), 3.0 * angle.sin());
                let forward = (-center).normalize();
                let right = Vector3::y().cross(&forward).normalize();
                let down = forward.cross(&right);
                camera(Matrix3::from_columns(&[right, down, forward]), center)
            })
            .collect()
    }

    #[test]
    fn recovers_a_synthetic_similarity() {
        let truth = Similarity {
            scale: 2.5,
            rotation: Rotation3::from_euler_angles(0.4, -1.1, 2.0).into_inner(),
            translation: Vector3::new(-1.0, 4.0, 0.5),
        };
        let from = orbit(5);
        let to: Vec<PinholeCamera> = from.iter().map(|c| truth.apply_camera(c)).collect();
        let names: Vec<String> = (0..from.len()).map(|i| format!("frame_{i}")).collect();
        let pairs: Vec<CameraPair> = names
            .iter()
            .zip(from.iter().zip(&to))
            .map(|(name, (from, to))| CameraPair { name, from, to })
            .collect();
        let (found, residuals) = align_cameras(&pairs);
        assert!((found.scale - truth.scale).abs() < 1e-9, "scale {}", found.scale);
        assert!((found.rotation - truth.rotation).norm() < 1e-9, "rotation {}", found.rotation);
        assert!((found.translation - truth.translation).norm() < 1e-9, "translation {}", found.translation);
        for residual in &residuals {
            assert!(residual.center_error < 1e-9, "{residual:?}");
            assert!(residual.direction_error_degrees < 1e-7, "{residual:?}");
        }
    }

    #[test]
    fn three_collinear_centers_still_fix_the_rotation_through_the_directions() {
        let truth = Similarity {
            scale: 0.5,
            rotation: Rotation3::from_euler_angles(-0.3, 0.7, 0.1).into_inner(),
            translation: Vector3::new(2.0, 0.0, -3.0),
        };
        let looking = Rotation3::from_euler_angles(0.2, 0.1, 0.0).into_inner();
        let from: Vec<PinholeCamera> = (0..3)
            .map(|i| camera(looking, Vector3::new(i as f64, 0.0, 0.0)))
            .collect();
        let to: Vec<PinholeCamera> = from.iter().map(|c| truth.apply_camera(c)).collect();
        let pairs: Vec<CameraPair> = from
            .iter()
            .zip(&to)
            .map(|(from, to)| CameraPair { name: "x", from, to })
            .collect();
        let (found, residuals) = align_cameras(&pairs);
        assert!((found.rotation - truth.rotation).norm() < 1e-9, "rotation {}", found.rotation);
        assert!((found.scale - truth.scale).abs() < 1e-9);
        assert!(residuals.iter().all(|r| r.center_error < 1e-9 && r.direction_error_degrees < 1e-7));
    }

    #[test]
    #[should_panic(expected = "need at least 3")]
    fn refuses_two_pairs() {
        let cameras = orbit(2);
        let pairs: Vec<CameraPair> = cameras
            .iter()
            .map(|c| CameraPair { name: "x", from: c, to: c })
            .collect();
        align_cameras(&pairs);
    }
}
