//! Image loading: EXIF orientation and focal length, background cutout, and conversion to
//! model input. Resizing to the model's input size is in `preprocess`.
//!
//! Transparent pixels (alpha < 50%) are background: the model sees them as a flat
//! `BACKDROP_RGB` backdrop, and they are left out of the point cloud.
//! Images with transparency are first cropped to their foreground, so the object fills
//! the model's input. Cutting the background out is what makes turntable captures (fixed
//! camera, rotating object) work, since a static background otherwise tells the model the
//! camera never moved.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage, Rgba, RgbaImage};

/// What transparent pixels are replaced with before the model sees them: mid-gray, about 0
/// after normalization, so the backdrop carries no signal.
const BACKDROP_RGB: [u8; 3] = [128, 128, 128];
/// Pixels with at least this alpha are foreground.
const FOREGROUND_MIN_ALPHA: u8 = 128;
/// Margin kept around the foreground bounding box, as a fraction of its size, per side.
const FOREGROUND_MARGIN: f64 = 0.05;

pub struct InputImage {
    /// The cropped image the model sees, used to color the point cloud.
    pub rgb: RgbImage,
    /// Row-major, one entry per pixel of `rgb`. All true for images without alpha.
    pub foreground: Vec<bool>,
    /// [1, 3, H, W], ImageNet-normalized.
    pub tensor: Tensor,
    /// Pinhole intrinsics of `rgb`, when the photo's focal length is known.
    pub intrinsics: Option<Intrinsics>,
}

/// Diagonal of a 36 x 24 mm frame, the reference of 35 mm-equivalent focal lengths.
const FULL_FRAME_DIAGONAL_MM: f64 = 43.266_615;

/// In pixels of the model input, pixel centers at integer coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub focal: [f64; 2],
    pub principal_point: [f64; 2],
}

/// How model-input coordinates relate to the upright original photo, per axis:
/// input = original * scale - offset, in continuous coordinates (0 at the outer corner of
/// the top-left pixel).
#[derive(Clone, Copy, Debug)]
pub struct PixelMapping {
    scale: [f64; 2],
    offset: [f64; 2],
}

impl PixelMapping {
    pub const IDENTITY: Self = Self {
        scale: [1.0; 2],
        offset: [0.0; 2],
    };

    /// Then keep the window starting at (left, top).
    pub fn then_crop(self, left: f64, top: f64) -> Self {
        Self {
            offset: [self.offset[0] + left, self.offset[1] + top],
            ..self
        }
    }

    /// Then resize the whole image from `from` to `to` (width, height).
    pub fn then_resize(self, from: (u32, u32), to: (u32, u32)) -> Self {
        let factor = [to.0 as f64 / from.0 as f64, to.1 as f64 / from.1 as f64];
        Self {
            scale: [self.scale[0] * factor[0], self.scale[1] * factor[1]],
            offset: [self.offset[0] * factor[0], self.offset[1] * factor[1]],
        }
    }
}

/// An upright photo cropped to its foreground and composited on the backdrop.
pub struct Cutout {
    pub image: DynamicImage,
    /// Upright original (width, height).
    pub original_size: (u32, u32),
    /// From the original photo to `image`.
    pub mapping: PixelMapping,
    /// 35 mm-equivalent focal length from EXIF, in mm.
    pub focal_35mm: Option<f64>,
}

impl Cutout {
    /// Intrinsics of the image after `mapping` (which must start from the original photo),
    /// for a principal point at the photo's center.
    pub fn intrinsics(&self, mapping: PixelMapping, focal_35mm: f64) -> Intrinsics {
        let (width, height) = (self.original_size.0 as f64, self.original_size.1 as f64);
        let focal = focal_35mm * width.hypot(height) / FULL_FRAME_DIAGONAL_MM;
        let center = [width / 2.0, height / 2.0];
        Intrinsics {
            focal: [focal * mapping.scale[0], focal * mapping.scale[1]],
            principal_point: [0, 1]
                .map(|axis| center[axis] * mapping.scale[axis] - mapping.offset[axis] - 0.5),
        }
    }
}

/// FocalLengthIn35mmFilm, when the file has EXIF and the tag.
fn exif_focal_35mm(path: &Path) -> Option<f64> {
    let file = std::fs::File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .ok()?;
    let field = exif.get_field(exif::Tag::FocalLengthIn35mmFilm, exif::In::PRIMARY)?;
    field.value.get_uint(0).map(|mm| mm as f64)
}

/// ImageNet mean and std, the normalization DINOv2 is trained with.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn read_upright(path: &Path) -> Result<DynamicImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("decoding {}", path.display()))?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// Crops to the foreground bounding box plus margin, widened to the image's own aspect
/// ratio. Parts of the crop outside the image are transparent. Fully opaque images are
/// returned whole. Also returns the crop's (left, top) in `rgba`.
fn crop_to_foreground(rgba: RgbaImage) -> Result<(RgbaImage, [i64; 2])> {
    if rgba.pixels().all(|pixel| pixel[3] == u8::MAX) {
        return Ok((rgba, [0, 0]));
    }
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        if pixel[3] >= FOREGROUND_MIN_ALPHA {
            (min_x, min_y) = (min_x.min(x), min_y.min(y));
            (max_x, max_y) = (max_x.max(x), max_y.max(y));
        }
    }
    ensure!(min_x <= max_x, "image has no foreground pixel");
    let box_w = (max_x - min_x + 1) as f64 * (1.0 + 2.0 * FOREGROUND_MARGIN);
    let box_h = (max_y - min_y + 1) as f64 * (1.0 + 2.0 * FOREGROUND_MARGIN);
    let aspect = rgba.width() as f64 / rgba.height() as f64;
    let (crop_w, crop_h) = if box_w / box_h > aspect {
        (box_w, box_w / aspect)
    } else {
        (box_h * aspect, box_h)
    };
    let left = ((min_x + max_x + 1) as f64 - crop_w) / 2.0;
    let top = ((min_y + max_y + 1) as f64 - crop_h) / 2.0;
    let mut cropped = RgbaImage::from_pixel(
        crop_w.round() as u32,
        crop_h.round() as u32,
        Rgba([0, 0, 0, 0]),
    );
    let corner = [left.round() as i64, top.round() as i64];
    image::imageops::replace(&mut cropped, &rgba, -corner[0], -corner[1]);
    Ok((cropped, corner))
}

/// Blends color onto the backdrop by alpha, keeping alpha. Done at full resolution so
/// the resize filters never mix in the color hidden under transparent pixels.
fn composite_on_backdrop(mut rgba: RgbaImage) -> DynamicImage {
    for pixel in rgba.pixels_mut() {
        let alpha = pixel[3] as u32;
        for (channel, backdrop) in pixel.0.iter_mut().zip(BACKDROP_RGB) {
            *channel = ((*channel as u32 * alpha + backdrop as u32 * (255 - alpha)) / 255) as u8;
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

/// The upright image, cropped to its foreground and composited on the backdrop, at full
/// resolution. Alpha is kept: it marks the foreground.
pub fn load_cutout(path: &Path) -> Result<Cutout> {
    let upright = read_upright(path)?;
    let original_size = (upright.width(), upright.height());
    let (foreground_crop, corner) = crop_to_foreground(upright.to_rgba8())
        .with_context(|| format!("cropping {}", path.display()))?;
    Ok(Cutout {
        image: composite_on_backdrop(foreground_crop),
        original_size,
        mapping: PixelMapping::IDENTITY.then_crop(corner[0] as f64, corner[1] as f64),
        focal_35mm: exif_focal_35mm(path),
    })
}

/// The final, already resized and cropped image as model input.
pub fn to_model_input(
    rgba: RgbaImage,
    intrinsics: Option<Intrinsics>,
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<InputImage> {
    let foreground: Vec<bool> = rgba
        .pixels()
        .map(|pixel| pixel[3] >= FOREGROUND_MIN_ALPHA)
        .collect();
    ensure!(
        foreground.contains(&true),
        "{} is fully transparent after cropping",
        path.display()
    );
    let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let mean = Tensor::new(&IMAGENET_MEAN, device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&IMAGENET_STD, device)?.reshape((3, 1, 1))?;
    let tensor = Tensor::from_vec(rgb.as_raw().clone(), (height, width, 3), device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(1.0 / 255.0, 0.0)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?
        .unsqueeze(0)?
        .to_dtype(dtype)?;
    Ok(InputImage {
        rgb,
        foreground,
        tensor,
        intrinsics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_pixels_become_backdrop() {
        let mut rgba = RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255]));
        rgba.put_pixel(1, 0, Rgba([10, 20, 30, 0]));
        let composited = composite_on_backdrop(rgba).to_rgba8();
        assert_eq!(composited.get_pixel(0, 0).0, [10, 20, 30, 255]);
        assert_eq!(composited.get_pixel(1, 0).0, [128, 128, 128, 0]);
    }

    #[test]
    fn crops_to_foreground_at_image_aspect() {
        // 200x100 image with a 20x20 opaque square at (90, 40).
        let mut rgba = RgbaImage::from_pixel(200, 100, Rgba([0, 0, 0, 0]));
        for y in 40..60 {
            for x in 90..110 {
                rgba.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        let (cropped, corner) = crop_to_foreground(rgba).unwrap();
        // 20px + 5% margin per side = 22px tall, widened to 2:1, centered on (100, 50).
        assert_eq!(cropped.dimensions(), (44, 22));
        assert_eq!(corner, [78, 39]);
        assert_eq!(cropped.get_pixel(22, 11).0, [255, 0, 0, 255]);
        assert_eq!(cropped.get_pixel(1, 11).0[3], 0);
    }

    #[test]
    fn keeps_opaque_images_whole() {
        let rgba = RgbaImage::from_pixel(30, 20, Rgba([1, 2, 3, 255]));
        let (cropped, corner) = crop_to_foreground(rgba).unwrap();
        assert_eq!((cropped.dimensions(), corner), ((30, 20), [0, 0]));
    }

    #[test]
    fn intrinsics_follow_crop_and_resize() {
        // A 4000x3000 photo at 24 mm equivalent: focal = 24 * 5000 / 43.27 = 2773 px.
        let cutout = Cutout {
            image: DynamicImage::new_rgb8(1, 1),
            original_size: (4000, 3000),
            mapping: PixelMapping::IDENTITY,
            focal_35mm: None,
        };
        // Keep the 2000x1500 window at (1000, 750), then halve it.
        let mapping = PixelMapping::IDENTITY
            .then_crop(1000.0, 750.0)
            .then_resize((2000, 1500), (1000, 750));
        let intrinsics = cutout.intrinsics(mapping, 24.0);
        let expected_focal = 24.0 * 5000.0 / FULL_FRAME_DIAGONAL_MM / 2.0;
        assert!((intrinsics.focal[0] - expected_focal).abs() < 1e-9);
        // The photo center (2000, 1500) is the window center: (500, 375), minus the half
        // pixel to pixel-center coordinates.
        assert_eq!(intrinsics.principal_point, [499.5, 374.5]);
    }
}
