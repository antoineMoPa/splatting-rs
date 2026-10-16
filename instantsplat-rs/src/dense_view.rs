//! What the feed-forward network stage hands to the Gaussian stage: per input image, a
//! dense point map in the scene frame, the image it was predicted from, and its camera.

use crate::camera::PinholeCamera;

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
