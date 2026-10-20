//! What the feed-forward network stage hands to the Gaussian stage: per input image, a
//! dense point map in the scene frame, the image it was predicted from, and its camera.

use crate::camera::PinholeCamera;

#[derive(Clone)]
pub struct DenseView {
    pub width: usize,
    pub height: usize,
    /// Row-major, one point per pixel, in the scene frame (camera 0's frame, OpenCV).
    pub points: Vec<[f32; 3]>,
    /// Row-major; 0 for pixels the network rejected.
    pub conf: Vec<f32>,
    /// Row-major; false for background pixels of cutout images.
    pub foreground: Vec<bool>,
    /// Row-major rgb in [0, 1], `width * height * 3`.
    pub rgb: Vec<f32>,
    /// Pixels of this view's image.
    pub camera: PinholeCamera,
}

impl DenseView {
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    pub fn to_rgb8(&self) -> image::RgbImage {
        image::RgbImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            let i = (y as usize * self.width + x as usize) * 3;
            image::Rgb([0, 1, 2].map(|c| (self.rgb[i + c] * 255.0).round() as u8))
        })
    }

    /// The colors with the foreground mask as alpha: `training_frames::training_view_from_image`
    /// reads it back as the same view.
    pub fn to_rgba8(&self) -> image::RgbaImage {
        image::RgbaImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            let pixel = y as usize * self.width + x as usize;
            let rgb = [0, 1, 2].map(|c| (self.rgb[pixel * 3 + c] * 255.0).round() as u8);
            image::Rgba([rgb[0], rgb[1], rgb[2], if self.foreground[pixel] { 255 } else { 0 }])
        })
    }

    /// Panics on inconsistent sizes.
    pub fn check(&self) {
        let count = self.pixel_count();
        assert_eq!(self.points.len(), count, "points");
        assert_eq!(self.conf.len(), count, "conf");
        assert_eq!(self.foreground.len(), count, "foreground");
        assert_eq!(self.rgb.len(), count * 3, "rgb");
        assert_eq!(
            (self.camera.width as usize, self.camera.height as usize),
            (self.width, self.height),
            "camera size"
        );
    }
}
