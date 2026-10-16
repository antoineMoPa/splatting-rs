//! COLMAP's TXT sparse model (`cameras.txt`, `images.txt`) as this crate's cameras:
//! camera-to-world, OpenCV axes, pixel centers at integer coordinates (COLMAP puts the
//! top-left pixel's center at (0.5, 0.5), so its principal point is shifted by half a
//! pixel). Only undistorted models are accepted: the rasterizer is a pinhole.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

use crate::camera::PinholeCamera;

/// Where each COLMAP camera model keeps its focal lengths, principal point and distortion
/// in its PARAMS[] list (indices). Single-focal models repeat the focal index.
struct CameraModelLayout {
    name: &'static str,
    param_count: usize,
    focal: [usize; 2],
    principal_point: [usize; 2],
    distortion: &'static [usize],
}

const CAMERA_MODELS: [CameraModelLayout; 7] = [
    CameraModelLayout { name: "SIMPLE_PINHOLE", param_count: 3, focal: [0, 0], principal_point: [1, 2], distortion: &[] },
    CameraModelLayout { name: "PINHOLE", param_count: 4, focal: [0, 1], principal_point: [2, 3], distortion: &[] },
    CameraModelLayout { name: "SIMPLE_RADIAL", param_count: 4, focal: [0, 0], principal_point: [1, 2], distortion: &[3] },
    CameraModelLayout { name: "RADIAL", param_count: 5, focal: [0, 0], principal_point: [1, 2], distortion: &[3, 4] },
    CameraModelLayout { name: "OPENCV", param_count: 8, focal: [0, 1], principal_point: [2, 3], distortion: &[4, 5, 6, 7] },
    CameraModelLayout { name: "OPENCV_FISHEYE", param_count: 8, focal: [0, 1], principal_point: [2, 3], distortion: &[4, 5, 6, 7] },
    CameraModelLayout { name: "FULL_OPENCV", param_count: 12, focal: [0, 1], principal_point: [2, 3], distortion: &[4, 5, 6, 7, 8, 9, 10, 11] },
];

/// Intrinsics of one `cameras.txt` row, in this crate's pixel convention.
#[derive(Clone, Debug)]
struct ColmapCamera {
    width: u32,
    height: u32,
    focal: [f64; 2],
    principal_point: [f64; 2],
}

/// One registered image: its file name and its camera, camera-to-world.
#[derive(Clone, Debug)]
pub struct ColmapImage {
    pub name: String,
    pub camera: PinholeCamera,
}

fn data_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

fn parse_cameras(text: &str) -> Result<BTreeMap<u32, ColmapCamera>> {
    let mut cameras = BTreeMap::new();
    for line in data_lines(text) {
        let words: Vec<&str> = line.split_whitespace().collect();
        ensure!(words.len() >= 4, "cameras.txt: short row {line:?}");
        let id: u32 = words[0].parse().with_context(|| format!("cameras.txt: camera id in {line:?}"))?;
        let layout = CAMERA_MODELS
            .iter()
            .find(|layout| layout.name == words[1])
            .with_context(|| format!("cameras.txt: camera model {} is not in the model table", words[1]))?;
        let params: Vec<f64> = words[4..]
            .iter()
            .map(|word| word.parse::<f64>().with_context(|| format!("cameras.txt: parameter {word:?}")))
            .collect::<Result<_>>()?;
        ensure!(
            params.len() == layout.param_count,
            "cameras.txt: {} takes {} parameters, row has {}",
            layout.name,
            layout.param_count,
            params.len()
        );
        let distortion: Vec<f64> = layout.distortion.iter().map(|&i| params[i]).collect();
        if distortion.iter().any(|&k| k != 0.0) {
            bail!(
                "cameras.txt: camera {id} ({}) has distortion {distortion:?}; the rasterizer is a pinhole, undistort the model first (colmap image_undistorter)",
                layout.name
            );
        }
        let previous = cameras.insert(
            id,
            ColmapCamera {
                width: words[2].parse()?,
                height: words[3].parse()?,
                focal: layout.focal.map(|i| params[i]),
                principal_point: layout.principal_point.map(|i| params[i] - 0.5),
            },
        );
        ensure!(previous.is_none(), "cameras.txt: camera {id} listed twice");
    }
    ensure!(!cameras.is_empty(), "cameras.txt: no cameras");
    Ok(cameras)
}

/// `images.txt` alternates a pose row (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME,
/// world-to-camera) and a row of 2D points; the point rows are skipped.
fn parse_images(text: &str, cameras: &BTreeMap<u32, ColmapCamera>) -> Result<Vec<ColmapImage>> {
    let mut images = Vec::new();
    // Not `data_lines`: the 2D points row of an image without observations is empty and
    // must still be consumed.
    let mut rows = text.lines().map(str::trim).filter(|line| !line.starts_with('#'));
    while let Some(line) = rows.next() {
        if line.is_empty() {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        ensure!(words.len() == 10, "images.txt: expected 10 fields in {line:?}");
        let values: Vec<f64> = words[1..8]
            .iter()
            .map(|word| word.parse::<f64>().with_context(|| format!("images.txt: number {word:?}")))
            .collect::<Result<_>>()?;
        let camera_id: u32 = words[8].parse()?;
        let intrinsics = cameras
            .get(&camera_id)
            .with_context(|| format!("images.txt: {} uses camera {camera_id}, not in cameras.txt", words[9]))?;
        let world_to_camera = UnitQuaternion::from_quaternion(Quaternion::new(values[0], values[1], values[2], values[3]));
        let world_to_camera_translation = Vector3::new(values[4], values[5], values[6]);
        let rotation: Matrix3<f64> = world_to_camera.to_rotation_matrix().into_inner().transpose();
        images.push(ColmapImage {
            name: words[9].to_string(),
            camera: PinholeCamera {
                rotation,
                translation: -(rotation * world_to_camera_translation),
                focal: intrinsics.focal,
                principal_point: intrinsics.principal_point,
                width: intrinsics.width,
                height: intrinsics.height,
            },
        });
        ensure!(rows.next().is_some(), "images.txt: {} has no 2D points row", words[9]);
    }
    ensure!(!images.is_empty(), "images.txt: no images");
    images.sort_by(|a, b| a.name.cmp(&b.name));
    for pair in images.windows(2) {
        ensure!(pair[0].name != pair[1].name, "images.txt: {} registered twice", pair[0].name);
    }
    Ok(images)
}

/// Reads `dir/cameras.txt` and `dir/images.txt`; images sorted by name.
pub fn read_colmap_model(dir: &Path) -> Result<Vec<ColmapImage>> {
    let read = |file: &str| {
        std::fs::read_to_string(dir.join(file)).with_context(|| format!("reading {}", dir.join(file).display()))
    };
    let cameras = parse_cameras(&read("cameras.txt")?)?;
    parse_images(&read("images.txt")?, &cameras)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAMERAS: &str = "# comment\n1 PINHOLE 100 80 50.0 52.0 50.5 40.5\n";
    // A 90 degree rotation about y (world-to-camera), translation (1, 2, 3).
    const IMAGES: &str = "# comment\n\
        2 0.7071067811865476 0 0.7071067811865476 0 1 2 3 1 b.png\n\
        10.5 20.5 -1 30 40 7\n\
        1 1 0 0 0 0 0 0 1 a.png\n\
        \n";

    #[test]
    fn reads_a_tiny_model() {
        let cameras = parse_cameras(CAMERAS).unwrap();
        let images = parse_images(IMAGES, &cameras).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].name, "a.png");
        assert_eq!(images[1].name, "b.png");
        let a = &images[0].camera;
        assert_eq!((a.width, a.height), (100, 80));
        assert_eq!(a.focal, [50.0, 52.0]);
        assert_eq!(a.principal_point, [50.0, 40.0]);
        assert_eq!(a.rotation, Matrix3::identity());
        assert_eq!(a.translation, Vector3::zeros());

        let b = &images[1].camera;
        // Camera center C = -R^T t: the world-to-camera rotation sends world x to -z.
        let (w_rotation, w_translation) = b.world_to_camera();
        let in_camera = w_rotation * Vector3::new(1.0, 0.0, 0.0);
        assert!((in_camera - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-12, "{in_camera}");
        assert!((w_translation - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-12, "{w_translation}");
        let world = w_rotation.transpose() * (Vector3::zeros() - w_translation);
        assert!((b.translation - world).norm() < 1e-12);
    }

    #[test]
    fn refuses_distortion_and_unknown_models() {
        assert!(parse_cameras("1 SIMPLE_RADIAL 10 10 5 5 5 0.01\n").is_err());
        assert!(parse_cameras("1 SIMPLE_RADIAL 10 10 5 5 5 0\n").is_ok());
        assert!(parse_cameras("1 THIN_PRISM_FISHEYE 10 10 5 5 5 0 0 0 0 0 0 0 0\n").is_err());
        assert!(parse_cameras("1 PINHOLE 10 10 5 5 5\n").is_err());
    }
}
