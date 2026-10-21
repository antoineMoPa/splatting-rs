//! Pinhole intrinsics of a dense view from its own points: per axis, the least-squares
//! (principal point, focal) of pixel = principal_point + focal * (coordinate / depth) over
//! the kept points, as mapanything-rs fits them (which keeps only the focal). Rendering
//! the points through the fit lands them on their pixels.

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::dense_view::DenseView;

/// Every (side / this)-th pixel along each side is used.
const SAMPLES_PER_SIDE: usize = 50;

pub struct PinholeFit {
    pub focal: [f64; 2],
    pub principal_point: [f64; 2],
}

pub fn fit_pinhole(view: &DenseView) -> PinholeFit {
    let (rotation, translation) = view.camera.world_to_camera();
    let step_y = (view.height / SAMPLES_PER_SIDE).max(1);
    let step_x = (view.width / SAMPLES_PER_SIDE).max(1);
    let mut normal = [Matrix2::<f64>::zeros(); 2];
    let mut rhs = [Vector2::<f64>::zeros(); 2];
    let mut used = 0;
    for y in (0..view.height).step_by(step_y) {
        for x in (0..view.width).step_by(step_x) {
            let i = y * view.width + x;
            if view.conf[i] <= 0.0 || !view.foreground[i] {
                continue;
            }
            let p = view.points[i];
            let in_camera = rotation * Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64) + translation;
            if in_camera.z <= 0.0 {
                continue;
            }
            used += 1;
            for (axis, (pixel, coordinate)) in [(x as f64, in_camera.x), (y as f64, in_camera.y)].into_iter().enumerate() {
                let ratio = coordinate / in_camera.z;
                normal[axis] += Matrix2::new(1.0, ratio, ratio, ratio * ratio);
                rhs[axis] += Vector2::new(pixel, ratio * pixel);
            }
        }
    }
    assert!(used >= 16, "only {used} kept points to fit the camera");
    let solve = |axis: usize| -> Vector2<f64> { normal[axis].lu().solve(&rhs[axis]).expect("rays must vary across the image") };
    let (x, y) = (solve(0), solve(1));
    assert!(x[1] > 0.0 && y[1] > 0.0, "negative focal length {} {}", x[1], y[1]);
    PinholeFit {
        focal: [x[1], y[1]],
        principal_point: [x[0], y[0]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use nalgebra::Vector3;

    #[test]
    fn recovers_the_intrinsics_of_a_synthetic_view() {
        let (width, height) = (64usize, 48usize);
        let rotation = nalgebra::Rotation3::from_euler_angles(0.2, 0.5, -0.3).into_inner();
        let translation = Vector3::new(0.3, -0.2, 1.0);
        let (focal, principal_point) = ([70.0, 75.0], [31.0, 24.5]);
        let mut points = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let depth = 2.0 + 0.01 * (x + y) as f64;
                let in_camera = Vector3::new(
                    (x as f64 - principal_point[0]) / focal[0] * depth,
                    (y as f64 - principal_point[1]) / focal[1] * depth,
                    depth,
                );
                let world = rotation * in_camera + translation;
                points.push([world.x as f32, world.y as f32, world.z as f32]);
            }
        }
        let view = DenseView {
            width,
            height,
            points,
            conf: vec![1.5; width * height],
            foreground: vec![true; width * height],
            rgb: vec![0.0; width * height * 3],
            camera: PinholeCamera {
                rotation,
                translation,
                focal: [1.0; 2],
                principal_point: [0.0; 2],
                width: width as u32,
                height: height as u32,
            },
        };
        let fit = fit_pinhole(&view);
        for axis in 0..2 {
            assert!((fit.focal[axis] - focal[axis]).abs() < 1e-2, "focal {:?}", fit.focal);
            assert!(
                (fit.principal_point[axis] - principal_point[axis]).abs() < 1e-2,
                "principal point {:?}",
                fit.principal_point
            );
        }
    }
}
