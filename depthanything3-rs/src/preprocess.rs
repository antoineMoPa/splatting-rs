//! Upstream's image preprocessing (`InputProcessor`, `process_res` 504,
//! `upper_bound_resize`): each image is resized so its longer side is PROCESS_RESOLUTION,
//! then each side to the nearest multiple of the patch size, and all images are center
//! cropped to the smallest size among them. Image loading, EXIF orientation and focal
//! length, and background cutouts are mapanything-rs's (`image_input`).

use std::path::Path;

use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use image::DynamicImage;
use image::imageops::FilterType;
use mapanything_rs::image_input::{Cutout, InputImage, PixelMapping, load_cutout, to_model_input};

use crate::variant::{PATCH_SIZE, PROCESS_RESOLUTION};

/// Upstream shrinks with OpenCV's INTER_AREA (a box filter over the source pixels) and
/// enlarges with INTER_CUBIC; the nearest of the `image` crate's filters.
const SHRINK_FILTER: FilterType = FilterType::Triangle;
const ENLARGE_FILTER: FilterType = FilterType::CatmullRom;

/// Round half up, as Python's `int(round())` for the positive values here (ties are rare).
fn round(value: f64) -> u32 {
    (value + 0.5).floor().max(1.0) as u32
}

/// The nearest multiple of `step`, ties going up, as upstream's `nearest_multiple`.
fn nearest_multiple(value: u32, step: u32) -> u32 {
    let down = value / step * step;
    let up = down + step;
    if up - value <= value - down { up } else { down }.max(step)
}

/// (width, height) after the longest side is brought to PROCESS_RESOLUTION, then after
/// each side is rounded to a multiple of PATCH_SIZE.
pub fn model_input_size((width, height): (u32, u32)) -> ((u32, u32), (u32, u32)) {
    let scale = PROCESS_RESOLUTION as f64 / width.max(height) as f64;
    let bounded = if width.max(height) == PROCESS_RESOLUTION {
        (width, height)
    } else {
        (round(width as f64 * scale), round(height as f64 * scale))
    };
    let patch = PATCH_SIZE as u32;
    (bounded, (nearest_multiple(bounded.0, patch), nearest_multiple(bounded.1, patch)))
}

fn resize_exact(image: &DynamicImage, (width, height): (u32, u32)) -> DynamicImage {
    if (image.width(), image.height()) == (width, height) {
        return image.clone();
    }
    let filter = if width > image.width() || height > image.height() {
        ENLARGE_FILTER
    } else {
        SHRINK_FILTER
    };
    image.resize_exact(width, height, filter)
}

/// Resizes as upstream, extending `mapping` (onto `image`) to the result.
fn resize_for_model(image: &DynamicImage, mapping: PixelMapping) -> (DynamicImage, PixelMapping) {
    let (bounded, divisible) = model_input_size((image.width(), image.height()));
    let resized = resize_exact(&resize_exact(image, bounded), divisible);
    (resized, mapping.then_resize((image.width(), image.height()), divisible))
}

/// Crops the center window of `size` (width, height), extending `mapping`.
fn center_crop(image: &DynamicImage, mapping: PixelMapping, (width, height): (u32, u32)) -> (DynamicImage, PixelMapping) {
    assert!(width <= image.width() && height <= image.height(), "crop larger than the image");
    let left = (image.width() - width) / 2;
    let top = (image.height() - height) / 2;
    (
        image.crop_imm(left, top, width, height),
        mapping.then_crop(left as f64, top as f64),
    )
}

/// `focal_35mm` overrides every photo's EXIF focal length (35 mm equivalent, in mm); the
/// intrinsics are only known for images with one or the other.
pub fn load_images(
    paths: &[impl AsRef<Path>],
    focal_35mm: Option<f64>,
    dtype: DType,
    device: &Device,
) -> Result<Vec<InputImage>> {
    ensure!(!paths.is_empty(), "no images");
    let cutouts: Vec<(String, Cutout)> = paths
        .iter()
        .map(|path| Ok((path.as_ref().display().to_string(), load_cutout(path.as_ref())?)))
        .collect::<Result<_>>()?;
    model_inputs_from_cutouts(&cutouts, focal_35mm, dtype, device)
}

/// `load_images` from already decoded photos (`image_input::cutout_from_bytes` in the
/// browser), each with the name that labels it in errors.
pub fn model_inputs_from_cutouts(
    cutouts: &[(String, Cutout)],
    focal_35mm: Option<f64>,
    dtype: DType,
    device: &Device,
) -> Result<Vec<InputImage>> {
    ensure!(!cutouts.is_empty(), "no images");
    let resized: Vec<(DynamicImage, PixelMapping)> = cutouts
        .iter()
        .map(|(_, cutout)| resize_for_model(&cutout.image, cutout.mapping))
        .collect();
    let common = resized.iter().fold((u32::MAX, u32::MAX), |(width, height), (image, _)| {
        (width.min(image.width()), height.min(image.height()))
    });
    if resized.iter().any(|(image, _)| (image.width(), image.height()) != common) {
        println!(
            "images differ in size after resizing; center cropping all to {}x{}",
            common.0, common.1
        );
    }
    cutouts
        .iter()
        .zip(resized)
        .map(|((name, cutout), (image, mapping))| {
            let (cropped, mapping) = center_crop(&image, mapping, common);
            let intrinsics = focal_35mm
                .or(cutout.focal_35mm)
                .map(|focal| cutout.intrinsics(mapping, focal));
            to_model_input(cropped.to_rgba8(), intrinsics, Path::new(name), dtype, device)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iphone_portrait_photo_becomes_378_by_504() {
        assert_eq!(model_input_size((3024, 4032)), ((378, 504), (378, 504)));
    }

    #[test]
    fn sides_round_to_the_nearest_patch_multiple() {
        // 1920x1080 -> 504x284 (283.5 rounds up) -> 504x280.
        assert_eq!(model_input_size((1920, 1080)), ((504, 284), (504, 280)));
        assert_eq!(nearest_multiple(377, 14), 378);
        assert_eq!(nearest_multiple(371, 14), 378);
        assert_eq!(nearest_multiple(370, 14), 364);
    }

    #[test]
    fn center_crop_keeps_the_mapping_consistent() {
        let image = DynamicImage::new_rgb8(100, 60);
        let (cropped, mapping) = center_crop(&image, PixelMapping::IDENTITY, (80, 60));
        assert_eq!((cropped.width(), cropped.height()), (80, 60));
        // The photo center (50, 30) lands 10 px left of where it was, in pixel-center
        // coordinates.
        let cutout = Cutout {
            image: DynamicImage::new_rgb8(1, 1),
            original_size: (100, 60),
            mapping: PixelMapping::IDENTITY,
            focal_35mm: None,
        };
        assert_eq!(cutout.intrinsics(mapping, 24.0).principal_point, [39.5, 29.5]);
    }
}
