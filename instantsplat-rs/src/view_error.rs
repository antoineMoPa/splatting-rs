//! How far a render is from its photo: the mean absolute error over all pixels and over
//! the foreground pixels only, and the renders of a splat on a set of views.

use crate::dense_view::DenseView;
use crate::gaussians::GaussianCloud;
use crate::gpu::Gpu;
use crate::optimizer::LEARNING_RATES_3DGS;
use crate::rasterizer::Rendered;
use crate::training::{CameraPoses, Trainer};

/// Mean absolute error of a render against its view, over all pixels and over the
/// foreground pixels only.
#[derive(Clone, Copy, Debug)]
pub struct ViewError {
    pub all: f32,
    pub foreground: f32,
}

pub fn view_error(render: &Rendered, view: &DenseView) -> ViewError {
    assert_eq!((render.width as usize, render.height as usize), (view.width, view.height));
    let (mut sum_all, mut sum_foreground, mut foreground_count) = (0.0f64, 0.0f64, 0usize);
    for (i, &foreground) in view.foreground.iter().enumerate() {
        let rendered = render.over_background((i % view.width) as u32, (i / view.width) as u32);
        let error = (0..3).map(|c| (rendered[c] - view.rgb[i * 3 + c]).abs() as f64).sum::<f64>() / 3.0;
        sum_all += error;
        if foreground {
            sum_foreground += error;
            foreground_count += 1;
        }
    }
    ViewError {
        all: (sum_all / view.pixel_count() as f64) as f32,
        foreground: (sum_foreground / foreground_count.max(1) as f64) as f32,
    }
}

pub fn mean_error(errors: &[ViewError]) -> ViewError {
    assert!(!errors.is_empty(), "mean of no errors");
    ViewError {
        all: errors.iter().map(|e| e.all).sum::<f32>() / errors.len() as f32,
        foreground: errors.iter().map(|e| e.foreground).sum::<f32>() / errors.len() as f32,
    }
}

/// `cloud` rendered through every view's camera.
pub fn render_all(gpu: &Gpu, cloud: &GaussianCloud, views: &[DenseView]) -> Vec<Rendered> {
    if views.is_empty() {
        return Vec::new();
    }
    let trainer = Trainer::new(gpu, cloud, views, LEARNING_RATES_3DGS, CameraPoses::Fixed);
    (0..views.len()).map(|index| trainer.render(index)).collect()
}
