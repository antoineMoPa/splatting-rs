//! MapAnything's image preprocessing (`load_images`, resize_mode "fixed_mapping", 518):
//! every image is resized to cover one shared size, picked by mean aspect ratio, and
//! center-cropped to it.

use std::path::Path;

use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};

use crate::image_input::{InputImage, PixelMapping, load_cutout, to_model_input};

/// Aspect ratio (width / height) to input (width, height), all multiples of 14, for a 518 px
/// resolution set. Sorted by aspect ratio.
const SIZE_BY_ASPECT: [(f64, (u32, u32)); 10] = [
    (0.486, (252, 518)),
    (0.567, (294, 518)),
    (0.649, (336, 518)),
    (0.757, (392, 518)),
    (1.000, (518, 518)),
    (1.321, (518, 392)),
    (1.542, (518, 336)),
    (1.762, (518, 294)),
    (2.056, (518, 252)),
    (3.083, (518, 168)),
];
/// Upstream adds this to the cover scale so that flooring never falls one pixel short.
const COVER_SCALE_EPS: f64 = 1e-8;

/// The table entry closest to the mean aspect ratio; ties go to the smaller ratio.
fn input_size(images: &[DynamicImage]) -> (u32, u32) {
    let mean_aspect = images.iter().map(|image| image.width() as f64 / image.height() as f64).sum::<f64>() / images.len() as f64;
    let mut best = SIZE_BY_ASPECT[0];
    for entry in SIZE_BY_ASPECT {
        if (entry.0 - mean_aspect).abs() < (best.0 - mean_aspect).abs() {
            best = entry;
        }
    }
    best.1
}

/// Resizes to cover (width, height), Lanczos to shrink and bicubic to enlarge as PIL in the
/// reference, then crops the center. Extends `mapping` (onto `image`) to the result.
fn cover_and_crop(image: &DynamicImage, mapping: PixelMapping, (width, height): (u32, u32)) -> (DynamicImage, PixelMapping) {
    let scale = (width as f64 / image.width() as f64).max(height as f64 / image.height() as f64) + COVER_SCALE_EPS;
    let filter = if scale < 1.0 {
        FilterType::Lanczos3
    } else {
        FilterType::CatmullRom
    };
    let resized = image.resize_exact(
        (image.width() as f64 * scale).floor() as u32,
        (image.height() as f64 * scale).floor() as u32,
        filter,
    );
    let left = (resized.width() - width) / 2;
    let top = (resized.height() - height) / 2;
    let mapping = mapping
        .then_resize(image.dimensions(), resized.dimensions())
        .then_crop(left as f64, top as f64);
    (resized.crop_imm(left, top, width, height), mapping)
}

/// `focal_35mm` overrides every photo's EXIF focal length (35 mm equivalent, in mm).
pub fn load_images(paths: &[impl AsRef<Path>], focal_35mm: Option<f64>, dtype: DType, device: &Device) -> Result<Vec<InputImage>> {
    ensure!(!paths.is_empty(), "no images");
    let cutouts = paths.iter().map(|path| load_cutout(path.as_ref())).collect::<Result<Vec<_>>>()?;
    let images: Vec<DynamicImage> = cutouts.iter().map(|cutout| cutout.image.clone()).collect();
    let size = input_size(&images);
    cutouts
        .iter()
        .zip(paths)
        .map(|(cutout, path)| {
            let (resized, mapping) = cover_and_crop(&cutout.image, cutout.mapping, size);
            let intrinsics = focal_35mm.or(cutout.focal_35mm).map(|focal| cutout.intrinsics(mapping, focal));
            let rgba = resized.to_rgba8();
            to_model_input(rgba, intrinsics, path.as_ref(), dtype, device)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_size_by_mean_aspect() {
        let portrait = DynamicImage::new_rgb8(3024, 4032);
        assert_eq!(input_size(&[portrait]), (392, 518));
        let landscape = DynamicImage::new_rgb8(1920, 1080);
        assert_eq!(input_size(&[landscape]), (518, 294));
    }

    #[test]
    fn covers_and_crops_to_size() {
        let image = DynamicImage::new_rgb8(1000, 700);
        let (cropped, _) = cover_and_crop(&image, PixelMapping::IDENTITY, (518, 392));
        assert_eq!((cropped.width(), cropped.height()), (518, 392));
    }
}
