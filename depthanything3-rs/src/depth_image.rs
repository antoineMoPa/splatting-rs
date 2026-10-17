//! Depth and confidence maps as grayscale images, to look at next to the photos.

use image::GrayImage;

/// Near is bright: inverse depth, scaled so that the nearest pixel is white and the
/// farthest black.
pub fn depth_to_gray(depth: &[f32], width: usize, height: usize) -> GrayImage {
    assert_eq!(depth.len(), width * height, "depth map size");
    let inverse: Vec<f32> = depth.iter().map(|&d| 1.0 / d).collect();
    scaled_to_gray(&inverse, width, height)
}

/// The least confident pixel black, the most confident white.
pub fn conf_to_gray(conf: &[f32], width: usize, height: usize) -> GrayImage {
    assert_eq!(conf.len(), width * height, "confidence map size");
    scaled_to_gray(conf, width, height)
}

fn scaled_to_gray(values: &[f32], width: usize, height: usize) -> GrayImage {
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(min.is_finite() && max.is_finite(), "values are not finite");
    let range = (max - min).max(f32::MIN_POSITIVE);
    GrayImage::from_fn(width as u32, height as u32, |x, y| {
        let value = values[y as usize * width + x as usize];
        image::Luma([((value - min) / range * 255.0).round() as u8])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_is_white_and_farthest_black() {
        let gray = depth_to_gray(&[1.0, 2.0, 4.0, 1.0], 2, 2);
        assert_eq!(gray.get_pixel(0, 0).0, [255]);
        assert_eq!(gray.get_pixel(0, 1).0, [0]);
        // 1/2 is a third of the way from 1/4 to 1.
        assert_eq!(gray.get_pixel(1, 0).0, [85]);
    }
}
