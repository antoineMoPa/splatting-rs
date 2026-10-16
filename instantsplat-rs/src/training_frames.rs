//! Training views from posed photos (the `refine` stage): each RGBA cutout frame is
//! composited on the rasterizer's background, shrunk to the optimization resolution, and
//! paired with its camera scaled the same way. Alpha marks the foreground.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageReader, RgbaImage};

use crate::camera::PinholeCamera;
use crate::dense_view::DenseView;

/// What transparent pixels are blended onto: `rasterizer::BACKGROUND` (128 / 255, what
/// the renders are composited over), the same mid-gray mapanything-rs puts under cutouts.
pub const BACKGROUND_RGB8: [u8; 3] = [128, 128, 128];
/// Pixels with at least this alpha (after resizing) are foreground, as mapanything-rs.
pub const FOREGROUND_MIN_ALPHA: u8 = 128;

/// `camera` is scaled by the same per-axis factors as the image: pixel (i, j) of the
/// result sees what the camera's `(i + 0.5) / factor - 0.5` saw.
fn scale_camera(camera: &PinholeCamera, width: u32, height: u32) -> PinholeCamera {
    let factor = [width as f64 / camera.width as f64, height as f64 / camera.height as f64];
    PinholeCamera {
        focal: [camera.focal[0] * factor[0], camera.focal[1] * factor[1]],
        principal_point: [0, 1].map(|axis| (camera.principal_point[axis] + 0.5) * factor[axis] - 0.5),
        width,
        height,
        ..camera.clone()
    }
}

/// Blends the color onto the background by alpha, keeping alpha; done before resizing
/// so that the filter never mixes in the color hidden under transparent pixels.
fn composite_on_background(mut rgba: RgbaImage) -> RgbaImage {
    for pixel in rgba.pixels_mut() {
        let alpha = pixel[3] as u32;
        for (channel, background) in pixel.0.iter_mut().zip(BACKGROUND_RGB8) {
            *channel = ((*channel as u32 * alpha + background as u32 * (255 - alpha)) / 255) as u8;
        }
    }
    rgba
}

/// Height of the training views: the splat's training height times `resolution_scale`;
/// the width follows the frame's aspect ratio (uniform scale, rounded).
pub fn training_size(camera: &PinholeCamera, training_height: u32, resolution_scale: f64) -> (u32, u32) {
    assert!(resolution_scale > 0.0, "resolution scale {resolution_scale}");
    let height = (training_height as f64 * resolution_scale).round() as u32;
    let width = (camera.width as f64 * height as f64 / camera.height as f64).round() as u32;
    assert!(width >= 16 && height >= 16, "training size {width}x{height} is too small");
    (width, height)
}

/// Loads `path`, which must be `camera.width` x `camera.height` (the image COLMAP
/// registered), and builds the view at `width` x `height`. Images without alpha are all
/// foreground. `points` and `conf` are not used by the optimization and are left at zero.
pub fn load_training_view(path: &Path, camera: &PinholeCamera, (width, height): (u32, u32)) -> Result<DenseView> {
    let image = ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("decoding {}", path.display()))?;
    ensure!(
        image.dimensions() == (camera.width, camera.height),
        "{} is {}x{}, its camera is {}x{}",
        path.display(),
        image.width(),
        image.height(),
        camera.width,
        camera.height
    );
    let composited = composite_on_background(image.to_rgba8());
    let filter = if width < camera.width { FilterType::Lanczos3 } else { FilterType::CatmullRom };
    let resized = DynamicImage::ImageRgba8(composited).resize_exact(width, height, filter).to_rgba8();
    let foreground: Vec<bool> = resized.pixels().map(|pixel| pixel[3] >= FOREGROUND_MIN_ALPHA).collect();
    ensure!(foreground.contains(&true), "{} has no foreground pixel", path.display());
    let rgb: Vec<f32> = resized
        .pixels()
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]].map(|c| c as f32 / 255.0))
        .collect();
    let pixel_count = (width * height) as usize;
    let view = DenseView {
        width: width as usize,
        height: height as usize,
        points: vec![[0.0; 3]; pixel_count],
        conf: vec![0.0; pixel_count],
        foreground,
        rgb,
        camera: scale_camera(camera, width, height),
    };
    view.check();
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;
    use nalgebra::{Matrix3, Vector3};

    #[test]
    fn composites_shrinks_and_masks() {
        // 40x20 frame: the left half opaque red, the right half transparent with a
        // hidden blue underneath.
        let mut rgba = RgbaImage::from_pixel(40, 20, Rgba([0, 0, 255, 0]));
        for y in 0..20 {
            for x in 0..20 {
                rgba.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        let dir = std::env::temp_dir().join(format!("instantsplat-frames-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.png");
        rgba.save(&path).unwrap();
        let camera = PinholeCamera {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
            focal: [100.0, 100.0],
            principal_point: [19.5, 9.5],
            width: 40,
            height: 20,
        };
        let view = load_training_view(&path, &camera, (20, 10)).unwrap();
        assert_eq!((view.width, view.height), (20, 10));
        assert_eq!(view.camera.focal, [50.0, 50.0]);
        assert_eq!(view.camera.principal_point, [9.5, 4.5]);
        // Interior pixels of each half: foreground red, background gray (never blue).
        let at = |x: usize, y: usize| &view.rgb[(y * 20 + x) * 3..(y * 20 + x) * 3 + 3];
        assert_eq!(at(3, 5), &[1.0, 0.0, 0.0]);
        assert!(view.foreground[5 * 20 + 3]);
        assert!(!view.foreground[5 * 20 + 16]);
        let gray = 128.0 / 255.0;
        for c in at(16, 5) {
            assert!((c - gray).abs() < 1e-6, "{:?}", at(16, 5));
        }
        assert_eq!(view.foreground.iter().filter(|&&f| f).count(), 100);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn background_is_the_rasterizers() {
        for c in 0..3 {
            assert!((BACKGROUND_RGB8[c] as f32 / 255.0 - crate::rasterizer::BACKGROUND[c]).abs() < 1e-5);
        }
    }

    #[test]
    fn training_size_keeps_the_aspect() {
        let camera = PinholeCamera {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
            focal: [1.0; 2],
            principal_point: [0.0; 2],
            width: 1061,
            height: 1891,
        };
        assert_eq!(training_size(&camera, 518, 1.0), (291, 518));
        assert_eq!(training_size(&camera, 518, 2.0), (581, 1036));
    }
}
