//! Stage 1: a feed-forward network (candle on Metal) turns the input images into dense
//! point maps and cameras, as `DenseView`s. Two networks: MapAnything (`../mapanything-rs`,
//! the default) and Depth Anything 3 (`../depthanything3-rs`, 35x smaller). Running them
//! from photo files needs the `networks` feature (native); the `da3` feature alone gives
//! what the browser demo needs, Depth Anything 3's prediction turned into `DenseView`s.
//!
//! Either way each view's intrinsics are the given ones (`focal_35mm` or EXIF, scaled to
//! the network's input size, principal point at the photo's center) when known, and the
//! points are moved onto the given rays (`replace_rays`, keeping the predicted distance
//! along each ray), so that rendering the points through the camera lands on their pixels.
//! Without a known focal length, each network's own intrinsics are used.

use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(feature = "networks")]
use anyhow::Result;
#[cfg(feature = "networks")]
use candle_core::Device;
use candle_core::DType;
use clap::ValueEnum;
use mapanything_rs::image_input::Intrinsics;

use crate::dense_view::DenseView;
use crate::replace_rays::replace_rays;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Network {
    /// Meta's MapAnything (1.2 B parameters, 4.9 GB): takes the known intrinsics as input.
    Mapanything,
    /// ByteDance's Depth Anything 3 Small (34 M parameters, 137 MB): predicts its own
    /// intrinsics.
    Da3Small,
    /// Depth Anything 3 Base (135 M parameters, 541 MB).
    Da3Base,
}

const NETWORK_LABELS: [(Network, &str); 3] = [
    (Network::Mapanything, "MapAnything"),
    (Network::Da3Small, "Depth Anything 3 Small"),
    (Network::Da3Base, "Depth Anything 3 Base"),
];

impl Network {
    pub fn label(self) -> &'static str {
        NETWORK_LABELS
            .iter()
            .find(|(network, _)| *network == self)
            .map(|(_, label)| *label)
            .expect("every network has a label")
    }
}

pub struct NetworkSettings {
    pub network: Network,
    /// 35 mm-equivalent focal length for every image, overriding EXIF.
    pub focal_35mm: Option<f64>,
    pub dtype: DType,
    /// Local checkpoint; downloaded from Hugging Face when None.
    pub weights: Option<PathBuf>,
    pub cpu: bool,
}

pub struct NetworkTiming {
    pub load_weights: Duration,
    pub inference: Duration,
}

#[cfg(feature = "networks")]
pub fn device(settings: &NetworkSettings) -> Result<Device> {
    Ok(if settings.cpu { Device::Cpu } else { Device::new_metal(0)? })
}

/// Runs the network on `images` (view 0 first). The point maps are in view 0's camera
/// frame.
#[cfg(feature = "networks")]
pub fn dense_views_from_images(images: &[PathBuf], settings: &NetworkSettings) -> Result<(Vec<DenseView>, NetworkTiming)> {
    match settings.network {
        Network::Mapanything => crate::network_mapanything::dense_views(images, settings),
        Network::Da3Small => crate::network_da3::dense_views(images, settings, depthanything3_rs::variant::Variant::Small),
        Network::Da3Base => crate::network_da3::dense_views(images, settings, depthanything3_rs::variant::Variant::Base),
    }
}

/// Gives `view` the known intrinsics and puts its points on their rays; `predicted` (the
/// network's own focal) is only printed next to them.
pub fn use_given_intrinsics(view: &mut DenseView, given: Intrinsics, predicted: [f64; 2]) {
    println!(
        "view {}x{}: given focal {:.1} {:.1} px (network {:.1} {:.1}), principal point {:.1} {:.1}",
        view.width,
        view.height,
        given.focal[0],
        given.focal[1],
        predicted[0],
        predicted[1],
        given.principal_point[0],
        given.principal_point[1]
    );
    view.camera.focal = given.focal;
    view.camera.principal_point = given.principal_point;
    replace_rays(view);
}

pub fn image_name(path: &Path) -> String {
    path.file_name()
        .expect("image path has a file name")
        .to_string_lossy()
        .into_owned()
}
