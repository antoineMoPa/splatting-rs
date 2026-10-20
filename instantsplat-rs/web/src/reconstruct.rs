//! The worker half of the demo (`worker.js`): the photos through Depth Anything 3 (Small
//! or Base, the worker's choice) on candle's CPU backend (wasm32, one thread), the points
//! cleaned across views, the views' point maps put onto each other (cameras along), one
//! Gaussian per point, and the result packed as a `fit` run
//! (`splat.ply`, `cameras.json`, the input views as RGBA PNGs) for `lib.rs` to load on the
//! main thread. No GPU here: the
//! network must not freeze the tab, and a worker shares nothing with the page but
//! messages.

use std::cell::RefCell;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use depthanything3_rs::model::{DepthAnything3, Stage};
use depthanything3_rs::pipeline::reconstruct_with_progress;
use depthanything3_rs::preprocess::model_inputs_from_cutouts;
use depthanything3_rs::reference_view::ReferenceViewStrategy;
use depthanything3_rs::variant::Variant;
use depthanything3_rs::weights::model_from_safetensors_bytes;
use instantsplat_rs::cameras_json::write_cameras_json_to;
use instantsplat_rs::dense_view::DenseView;
use instantsplat_rs::init_from_points::{INIT_OPACITY, InitSettings, gaussians_from_dense_views};
use instantsplat_rs::network::Network;
use instantsplat_rs::network_da3::dense_views_from_reconstruction;
use instantsplat_rs::point_map_alignment::align_point_maps;
use instantsplat_rs::splat_ply::write_splat_ply_to;
use mapanything_rs::image_input::{Cutout, cutout_from_bytes};
use wasm_bindgen::prelude::*;
use web_time::Instant;

/// The networks `worker.js` can load, by the name it sends (the CLI's `--network` names).
/// Not the Giant: 5.4 GB of f32 weights do not fit wasm32's 4 GB of memory.
const WORKER_NETWORKS: [(&str, Network, Variant); 3] = [
    ("da3-small", Network::Da3Small, Variant::Small),
    ("da3-base", Network::Da3Base, Variant::Base),
    ("da3-large", Network::Da3Large, Variant::Large),
];
/// Upstream's default; every pose is moved to view 0's frame afterwards anyway.
const REFERENCE_VIEW: ReferenceViewStrategy = ReferenceViewStrategy::SaddleBalanced;
/// As `fit`'s defaults: one Gaussian per 2x2 pixels, every point the cleaning kept.
const INIT: InitSettings = InitSettings {
    stride: 2,
    min_conf: 1.0,
    opacity: INIT_OPACITY,
};
/// Depth Anything 3 needs at least this many photos to place cameras.
const MIN_PHOTOS: usize = 2;

thread_local! {
    /// The loaded network and its model.
    static MODEL: RefCell<Option<(Network, DepthAnything3)>> = const { RefCell::new(None) };
    /// The photos added since the last `reconstruct`, decoded and cropped to their
    /// foreground, with their file names.
    static PHOTOS: RefCell<Vec<(String, Cutout)>> = const { RefCell::new(Vec::new()) };
}

fn js_error(error: anyhow::Error) -> JsError {
    JsError::new(&format!("{error:#}"))
}

/// Builds the model of `network` (a `WORKER_NETWORKS` name, anything else panics) from
/// the bytes of its `model.safetensors` (f32), replacing the model loaded before, if any.
#[wasm_bindgen]
pub fn load_da3(network: &str, safetensors: Vec<u8>) -> Result<(), JsError> {
    let (_, network, variant) = *WORKER_NETWORKS
        .iter()
        .find(|(name, _, _)| *name == network)
        .unwrap_or_else(|| panic!("load_da3: unknown network {network:?}"));
    // The old model is dropped first: the new one's tensors fit in its memory.
    MODEL.with(|slot| *slot.borrow_mut() = None);
    let model = model_from_safetensors_bytes(variant, REFERENCE_VIEW, safetensors, DType::F32, &Device::Cpu)
        .with_context(|| format!("loading the {} weights", network.label()))
        .map_err(js_error)?;
    MODEL.with(|slot| *slot.borrow_mut() = Some((network, model)));
    Ok(())
}

/// Decodes one photo (JPEG or PNG; transparent pixels are background) for the next
/// `reconstruct`; the number of photos so far.
#[wasm_bindgen]
pub fn add_input_photo(name: &str, encoded: &[u8]) -> Result<usize, JsError> {
    let cutout = cutout_from_bytes(encoded, name).map_err(js_error)?;
    PHOTOS.with(|photos| {
        photos.borrow_mut().push((name.to_string(), cutout));
        Ok(photos.borrow().len())
    })
}

/// A reconstruction packed as a `fit` run, for `load_scene` and `add_photo`.
#[wasm_bindgen]
pub struct Reconstructed {
    splat_ply: Vec<u8>,
    cameras_json: String,
    /// One RGBA PNG per view, the foreground as alpha, at the cameras' size.
    input_view_pngs: Vec<Vec<u8>>,
    gaussian_count: usize,
    report: String,
}

#[wasm_bindgen]
impl Reconstructed {
    pub fn splat_ply(&self) -> Vec<u8> {
        self.splat_ply.clone()
    }

    pub fn cameras_json(&self) -> String {
        self.cameras_json.clone()
    }

    pub fn view_count(&self) -> usize {
        self.input_view_pngs.len()
    }

    pub fn input_view_png(&self, view: usize) -> Vec<u8> {
        self.input_view_pngs[view].clone()
    }

    pub fn gaussian_count(&self) -> usize {
        self.gaussian_count
    }

    /// The stages' timings, one line.
    pub fn report(&self) -> String {
        self.report.clone()
    }
}

fn stage_label(stage: Stage, depth: usize) -> String {
    match stage {
        Stage::Block(index) => format!("backbone block {}/{depth}", index + 1),
        Stage::DenseHead => "depth head".to_string(),
        Stage::CameraHead => "camera head".to_string(),
    }
}

fn png_bytes(view: &DenseView) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    view.to_rgba8().write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)?;
    Ok(bytes)
}

/// Runs the network on the photos added (view 0 first), with `focal_35mm` (35 mm
/// equivalent, in mm) as every photo's focal length when given, else each photo's EXIF
/// focal length, else the network's own prediction. `progress` hears each stage's label.
/// Takes the photos: add them again for another run.
#[wasm_bindgen]
pub fn reconstruct(focal_35mm: Option<f64>, progress: &js_sys::Function) -> Result<Reconstructed, JsError> {
    let photos = PHOTOS.with(|photos| std::mem::take(&mut *photos.borrow_mut()));
    MODEL.with(|model| {
        let model = model.borrow();
        let (network, model) = model.as_ref().expect("load_da3 first");
        reconstruct_photos(*network, model, &photos, focal_35mm, &mut |label| {
            progress.call1(&JsValue::NULL, &JsValue::from_str(label)).expect("progress callback");
        })
    })
    .map_err(js_error)
}

fn reconstruct_photos(
    network: Network,
    model: &DepthAnything3,
    photos: &[(String, Cutout)],
    focal_35mm: Option<f64>,
    progress: &mut dyn FnMut(&str),
) -> Result<Reconstructed> {
    ensure!(photos.len() >= MIN_PHOTOS, "{} photos: reconstruction needs at least {MIN_PHOTOS}", photos.len());
    let device = Device::Cpu;
    let started = Instant::now();
    let inputs = model_inputs_from_cutouts(photos, focal_35mm, DType::F32, &device)?;
    let known = inputs.iter().filter(|image| image.intrinsics.is_some()).count();
    let preprocessing = started.elapsed();
    progress(&format!("preprocessed {} photos ({known} with a known focal length)", inputs.len()));

    let started = Instant::now();
    let depth = model.variant().spec().depth;
    let reconstruction =
        reconstruct_with_progress(model, &inputs, &device, &mut |stage| progress(&stage_label(stage, depth)))?;
    let inference = started.elapsed();

    let started = Instant::now();
    let mut views = dense_views_from_reconstruction(&reconstruction, &inputs);
    let alignments = align_point_maps(&mut views);
    for alignment in &alignments {
        progress(&alignment.to_string());
    }
    let alignment = started.elapsed();

    let started = Instant::now();
    let cloud = gaussians_from_dense_views(&views, INIT);
    let mut splat_ply = Vec::new();
    write_splat_ply_to(&mut splat_ply, &cloud)?;
    let cameras: Vec<_> = photos
        .iter()
        .zip(&views)
        .map(|((name, _), view)| (Path::new(name).display().to_string(), view.camera.clone()))
        .collect();
    let mut cameras_json = Vec::new();
    write_cameras_json_to(&mut cameras_json, &cameras)?;
    let input_view_pngs = views.iter().map(png_bytes).collect::<Result<_>>()?;
    let packing = started.elapsed();

    Ok(Reconstructed {
        splat_ply,
        cameras_json: String::from_utf8(cameras_json).expect("cameras.json is UTF-8"),
        input_view_pngs,
        gaussian_count: cloud.len(),
        report: format!(
            "{} views of {}x{}: preprocessing {:.1?}, {} {:.1?}, alignment {:.1?}, {} Gaussians + packing {:.1?}",
            views.len(),
            views[0].width,
            views[0].height,
            preprocessing,
            network.label(),
            inference,
            alignment,
            cloud.len(),
            packing
        ),
    })
}
