//! Multi-view consistency cleaning: a point that lands in front of what another camera sees
//! there, and is less confident than that camera's point, would hide a surface the other
//! camera saw. Its confidence is set to 0.

use nalgebra::{Matrix3, Vector3};

use crate::point_map::PointMap;

/// A point is in front when its depth in the other camera is below (1 - this) times that
/// camera's depth at the pixel it projects to. The views come from one forward pass, not a
/// joint optimization, and disagree by centimeters: at 0.1%, cleaning removed most of the
/// least confident views (the back of an object) instead of outliers.
const DEPTH_TOLERANCE: f64 = 0.02;

/// One image's camera and its depth at every pixel, as placed in the point cloud.
pub struct DepthView<'a> {
    /// Camera to point-cloud frame.
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub focal: f64,
    pub principal_point: [f64; 2],
    /// Row-major camera-frame depth of each pixel.
    pub depth: &'a [f32],
}

/// Sets to 0 the confidence of every foreground point in front of another view's
/// foreground. Confidences are compared before any is changed.
pub fn zero_conf_in_front_of_other_views(point_maps: &mut [PointMap], views: &[DepthView]) {
    let original_conf: Vec<Vec<f32>> = point_maps.iter().map(|map| map.conf.clone()).collect();
    for image in 0..point_maps.len() {
        for (other, view) in views.iter().enumerate() {
            if other == image {
                continue;
            }
            let other_map = &point_maps[other];
            let (other_width, other_height) = (other_map.width, other_map.height);
            let other_foreground = other_map.foreground.clone();
            let world_to_other = view.rotation.transpose();
            let map = &mut point_maps[image];
            for pixel in 0..map.points.len() {
                if !map.foreground[pixel] {
                    continue;
                }
                let [x, y, z] = map.points[pixel].map(|c| c as f64);
                let in_other = world_to_other * (Vector3::new(x, y, z) - view.translation);
                if in_other.z <= 0.0 {
                    continue;
                }
                let u = (view.focal * in_other.x / in_other.z + view.principal_point[0]).round();
                let v = (view.focal * in_other.y / in_other.z + view.principal_point[1]).round();
                if u < 0.0 || v < 0.0 || u >= other_width as f64 || v >= other_height as f64 {
                    continue;
                }
                let other_pixel = v as usize * other_width + u as usize;
                if other_foreground[other_pixel]
                    && in_other.z < (1.0 - DEPTH_TOLERANCE) * view.depth[other_pixel] as f64
                    && original_conf[image][pixel] < original_conf[other][other_pixel]
                {
                    map.conf[pixel] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_map(depth: f32, conf: f32) -> PointMap {
        PointMap {
            height: 1,
            width: 1,
            points: vec![[0.0, 0.0, depth]],
            conf: vec![conf],
            foreground: vec![true],
        }
    }

    #[test]
    fn zeroes_less_confident_point_in_front() {
        // Two cameras at the origin looking down z: image 0 sees a point at depth 1,
        // image 1 sees one at depth 2, behind it, more confidently.
        let mut maps = vec![point_map(1.0, 2.0), point_map(2.0, 3.0)];
        let depths = [vec![1.0f32], vec![2.0f32]];
        let views: Vec<DepthView> = depths
            .iter()
            .map(|depth| DepthView {
                rotation: Matrix3::identity(),
                translation: Vector3::zeros(),
                focal: 1.0,
                principal_point: [0.0, 0.0],
                depth,
            })
            .collect();
        zero_conf_in_front_of_other_views(&mut maps, &views);
        assert_eq!(maps[0].conf, vec![0.0]);
        assert_eq!(maps[1].conf, vec![3.0]);
    }
}
