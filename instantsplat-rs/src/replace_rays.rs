//! MapAnything predicts its own ray directions even when told the intrinsics (upstream
//! keeps the predicted rays too; on the cutout run it predicts a 470 px focal when given
//! 412 px). Rendering those points through the given intrinsics then lands them on the
//! wrong pixels. This keeps each point's predicted distance along its ray and puts it on
//! the ray of the view's own pinhole camera instead, so that pixel (u, v)'s point
//! projects to (u, v) again.

use nalgebra::Vector3;

use crate::dense_view::DenseView;

/// Replaces `view.points` in place; `view.camera` must already hold the intrinsics to use.
pub fn replace_rays(view: &mut DenseView) {
    let (world_to_camera_rotation, world_to_camera_translation) = view.camera.world_to_camera();
    let camera_to_world_rotation = view.camera.rotation;
    let camera_to_world_translation = view.camera.translation;
    let [fx, fy] = view.camera.focal;
    let [cx, cy] = view.camera.principal_point;
    for y in 0..view.height {
        for x in 0..view.width {
            let i = y * view.width + x;
            let p = view.points[i];
            let in_camera = world_to_camera_rotation * Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64)
                + world_to_camera_translation;
            let distance = in_camera.norm();
            let ray = Vector3::new((x as f64 - cx) / fx, (y as f64 - cy) / fy, 1.0).normalize();
            let world = camera_to_world_rotation * (ray * distance) + camera_to_world_translation;
            view.points[i] = [world.x as f32, world.y as f32, world.z as f32];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use crate::pinhole_fit::fit_pinhole;

    #[test]
    fn points_follow_the_cameras_intrinsics_afterwards() {
        let (width, height) = (64usize, 48usize);
        let rotation = nalgebra::Rotation3::from_euler_angles(0.2, 0.5, -0.3).into_inner();
        let translation = Vector3::new(0.3, -0.2, 1.0);
        // Points predicted along a 70 px focal, camera told 100 px.
        let predicted_focal = 70.0;
        let mut points = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let depth = 2.0 + 0.01 * (x + y) as f64;
                let in_camera = Vector3::new(
                    (x as f64 - 31.5) / predicted_focal * depth,
                    (y as f64 - 23.5) / predicted_focal * depth,
                    depth,
                );
                let world = rotation * in_camera + translation;
                points.push([world.x as f32, world.y as f32, world.z as f32]);
            }
        }
        let mut view = DenseView {
            width,
            height,
            points,
            conf: vec![1.5; width * height],
            foreground: vec![true; width * height],
            rgb: vec![0.0; width * height * 3],
            camera: PinholeCamera {
                rotation,
                translation,
                focal: [100.0, 100.0],
                principal_point: [31.5, 23.5],
                width: width as u32,
                height: height as u32,
            },
        };
        let before = fit_pinhole(&view);
        assert!((before.focal[0] - predicted_focal).abs() < 1e-2, "{:?}", before.focal);
        replace_rays(&mut view);
        let after = fit_pinhole(&view);
        assert!((after.focal[0] - 100.0).abs() < 1e-2 && (after.focal[1] - 100.0).abs() < 1e-2, "{:?}", after.focal);
        assert!((after.principal_point[0] - 31.5).abs() < 1e-2, "{:?}", after.principal_point);
        // The distance along the ray is kept: the center pixel's point did not move.
        let center = view.points[24 * width + 32];
        let expected = rotation * Vector3::new(0.5 / 70.0 * 2.56, 0.5 / 70.0 * 2.56, 2.56) + translation;
        let kept_distance = (Vector3::new(center[0] as f64, center[1] as f64, center[2] as f64) - translation).norm();
        assert!((kept_distance - (expected - translation).norm()).abs() < 1e-4);
    }
}
