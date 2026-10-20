//! The browser demo: the WebGPU device, one splat with its cameras, renders of it as
//! `ImageData`, and its optimization against photos of the cameras, driven from
//! `index.html` through wasm-bindgen. One GPU job at a time: a call made while another
//! awaits the GPU panics. `reconstruct.rs` is the other half, run by `worker.js` in a Web
//! Worker: photos through Depth Anything 3 (Small or Base) to a splat for this half to load.

pub mod reconstruct;

use std::cell::RefCell;
use std::io::Cursor;

use anyhow::{Context, Result, ensure};
use image::GenericImageView;
use instantsplat_rs::camera::PinholeCamera;
use instantsplat_rs::camera_pose::pose_distance;
use instantsplat_rs::cameras_json::parse_cameras_json;
use instantsplat_rs::dense_view::DenseView;
use instantsplat_rs::gaussians::GaussianCloud;
use instantsplat_rs::gpu::Gpu;
use instantsplat_rs::loss::GROUND_TRUTH_STRIDE;
use instantsplat_rs::optimizer::{LEARNING_RATES_SHORT_RUN, position_rate_factor};
use instantsplat_rs::rasterizer::{Rasterizer, Rendered};
use instantsplat_rs::splat_ply::{parse_splat_ply, write_splat_ply_to};
use instantsplat_rs::training::{CameraPoses, Trainer};
use instantsplat_rs::training_frames::{scale_camera, training_view_from_image};
use instantsplat_rs::turntable::Turntable;
use wasm_bindgen::prelude::*;
use web_sys::ImageData;

/// Photos whose aspect ratio is further than this from their camera's are refused.
const MAX_ASPECT_RATIO_ERROR: f64 = 0.02;
/// The preview is rendered at the photos' height and at least this wide for it (4:3):
/// portrait photos get a landscape view that shows more of the scene at the sides.
const PREVIEW_MIN_ASPECT_RATIO: f64 = 4.0 / 3.0;

thread_local! {
    /// Requested once per page; leaked so that the trainer can borrow it for 'static.
    static GPU: RefCell<Option<&'static Gpu>> = const { RefCell::new(None) };
    /// Taken out while a GPU job awaits, so that a second job finds nothing and panics.
    static SCENE: RefCell<Option<Scene>> = const { RefCell::new(None) };
}

struct Scene {
    cloud: GaussianCloud,
    /// By frame name, as `cameras.json`; all of one image size.
    cameras: Vec<(String, PinholeCamera)>,
    /// Views of the cameras with a photo (`add_photo`), in the order added.
    views: Vec<DenseView>,
    /// The camera name of each of `views`.
    view_names: Vec<String>,
    renderer: Renderer,
}

enum Renderer {
    /// The splat as loaded (or as last trained), through its own rasterizer (whose bind
    /// groups keep the parameter buffer alive).
    Splat(Rasterizer),
    /// Between `start_training` and `finish_training`: the trainer's parameters, which
    /// `preview` (at the preview size; the trainer's own rasterizer is at the photos')
    /// renders live.
    Training {
        trainer: Trainer<'static>,
        preview: Rasterizer,
        steps: usize,
        step: usize,
    },
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

fn js_error(error: anyhow::Error) -> JsError {
    JsError::new(&format!("{error:#}"))
}

fn gpu() -> &'static Gpu {
    GPU.with(|gpu| gpu.borrow().expect("init_gpu first"))
}

fn take_scene() -> Scene {
    SCENE.with(|scene| scene.borrow_mut().take()).expect("no scene: none loaded, or a GPU job is still running")
}

fn put_scene(scene: Scene) {
    SCENE.with(|slot| *slot.borrow_mut() = Some(scene));
}

/// The preview's image size for cameras of `camera`'s: see `PREVIEW_MIN_ASPECT_RATIO`.
fn preview_size(camera: &PinholeCamera) -> (u32, u32) {
    let min_width = (camera.height as f64 * PREVIEW_MIN_ASPECT_RATIO).round() as u32;
    (camera.width.max(min_width), camera.height)
}

/// A rasterizer over `params` (`gaussian_count` packed Gaussians) at the preview size.
fn preview_rasterizer(params: &wgpu::Buffer, gaussian_count: usize, (width, height): (u32, u32)) -> Rasterizer {
    let gpu = gpu();
    // The rasterizer wants a ground truth to take losses against; none is taken here.
    let ground_truth = gpu.storage_buffer("ground truth", (width * height) as u64 * GROUND_TRUTH_STRIDE as u64 * 4);
    Rasterizer::new(gpu, params, gaussian_count, width, height, &ground_truth, 1)
}

fn splat_renderer(cloud: &GaussianCloud, preview: (u32, u32)) -> Renderer {
    let params = gpu().storage_buffer_from("params", &cloud.to_packed());
    Renderer::Splat(preview_rasterizer(&params, cloud.len(), preview))
}

/// Requests the WebGPU adapter and device; the adapter's name.
#[wasm_bindgen]
pub async fn init_gpu() -> String {
    assert!(GPU.with(|gpu| gpu.borrow().is_none()), "init_gpu called twice");
    let gpu: &'static Gpu = Box::leak(Box::new(Gpu::request_async().await));
    GPU.with(|slot| *slot.borrow_mut() = Some(gpu));
    format!("{} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend)
}

/// Loads a 3DGS `splat.ply` with its `cameras.json`, replacing any scene; the number of
/// Gaussians.
#[wasm_bindgen]
pub fn load_scene(ply: &[u8], cameras_json: &str) -> Result<usize, JsError> {
    SCENE.with(|scene| scene.borrow_mut().take());
    let cloud = parse_splat_ply(Cursor::new(ply), "splat.ply").map_err(js_error)?;
    let cameras = parse_cameras_json(cameras_json, "cameras.json").map_err(js_error)?;
    assert!(!cameras.is_empty(), "cameras.json has no camera");
    let (width, height) = (cameras[0].1.width, cameras[0].1.height);
    for (name, camera) in &cameras {
        assert_eq!((camera.width, camera.height), (width, height), "{name}: every camera must have the same image size");
    }
    let renderer = splat_renderer(&cloud, preview_size(&cameras[0].1));
    let count = cloud.len();
    put_scene(Scene {
        cloud,
        cameras,
        views: Vec::new(),
        view_names: Vec::new(),
        renderer,
    });
    Ok(count)
}

/// The cameras' frame names, in `cameras.json` order.
#[wasm_bindgen]
pub fn camera_names() -> Vec<String> {
    SCENE.with(|scene| {
        let scene = scene.borrow();
        scene.as_ref().expect("no scene").cameras.iter().map(|(name, _)| name.clone()).collect()
    })
}

fn rgba_over_background(rendered: &Rendered) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((rendered.width * rendered.height * 4) as usize);
    for y in 0..rendered.height {
        for x in 0..rendered.width {
            let rgb = rendered.over_background(x, y);
            rgba.extend(rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8));
            rgba.push(255);
        }
    }
    rgba
}

/// Camera `view` moved on the turntable of the scene's cameras (`Turntable`), rendered
/// at the preview size over the display background.
#[wasm_bindgen]
pub async fn render(view: usize, yaw_degrees: f64, pitch_degrees: f64) -> Result<ImageData, JsError> {
    let scene = take_scene();
    assert!(view < scene.cameras.len(), "view {view} of {}", scene.cameras.len());
    let camera = &scene.cameras[view].1;
    let (width, height) = preview_size(camera);
    let cameras: Vec<PinholeCamera> = scene.cameras.iter().map(|(_, camera)| camera.clone()).collect();
    let turntable = Turntable::from_cameras(&cameras, scene.cloud.centroid());
    let camera = turntable
        .view(camera, yaw_degrees.to_radians(), pitch_degrees.to_radians())
        .with_image_size(width, height);
    let rasterizer = match &scene.renderer {
        Renderer::Splat(rasterizer) => rasterizer,
        Renderer::Training { preview, .. } => preview,
    };
    let rendered = rasterizer.render_async(gpu(), &camera).await;
    put_scene(scene);
    let rgba = rgba_over_background(&rendered);
    ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&rgba), rendered.width, rendered.height)
        .map_err(|error| JsError::new(&format!("ImageData: {error:?}")))
}

/// The photo of camera `name` (JPEG or PNG; transparent pixels are background), at any
/// resolution with the camera's aspect ratio, as a training view. The number of views.
#[wasm_bindgen]
pub fn add_photo(name: &str, encoded: &[u8]) -> Result<usize, JsError> {
    let mut scene = take_scene();
    let result = (|| -> Result<usize> {
        assert!(!matches!(scene.renderer, Renderer::Training { .. }), "add photos before training");
        let (_, camera) = scene
            .cameras
            .iter()
            .find(|(camera_name, _)| camera_name == name)
            .with_context(|| format!("no camera named {name} in cameras.json"))?;
        let image = image::load_from_memory(encoded).with_context(|| format!("decoding {name}"))?;
        let (width, height) = image.dimensions();
        let aspect_ratio_error = (width as f64 / height as f64) / (camera.width as f64 / camera.height as f64) - 1.0;
        ensure!(
            aspect_ratio_error.abs() <= MAX_ASPECT_RATIO_ERROR,
            "{name} is {width}x{height}, not the aspect ratio of its camera's {}x{}",
            camera.width,
            camera.height
        );
        let view = training_view_from_image(image, name, &scale_camera(camera, width, height), (camera.width, camera.height))?;
        scene.views.push(view);
        scene.view_names.push(name.to_string());
        Ok(scene.views.len())
    })();
    put_scene(scene);
    result.map_err(js_error)
}

/// Starts optimizing the splat against the photos added, for `steps` steps run by
/// `train_steps`; with `refine_poses`, the poses and a shared focal scale are optimized
/// along with it.
#[wasm_bindgen]
pub fn start_training(steps: usize, refine_poses: bool) {
    let mut scene = take_scene();
    assert!(steps > 0, "steps");
    assert!(scene.views.len() >= 2, "{} photos: training needs at least 2", scene.views.len());
    assert!(!matches!(scene.renderer, Renderer::Training { .. }), "already training");
    let camera_poses = if refine_poses { CameraPoses::Refined } else { CameraPoses::Fixed };
    let trainer = Trainer::new(gpu(), &scene.cloud, &scene.views, LEARNING_RATES_SHORT_RUN, camera_poses);
    let preview = preview_rasterizer(trainer.params(), trainer.gaussian_count(), preview_size(&scene.cameras[0].1));
    scene.renderer = Renderer::Training {
        trainer,
        preview,
        steps,
        step: 0,
    };
    put_scene(scene);
}

/// Up to `count` more steps (fewer at the end of the run), then one logged step: the
/// status line "step <done>/<steps>: <loss>", for the page.
#[wasm_bindgen]
pub async fn train_steps(count: usize) -> String {
    let mut scene = take_scene();
    let Renderer::Training { trainer, steps, step, .. } = &mut scene.renderer else {
        panic!("not training: start_training first");
    };
    assert!(count > 0, "count");
    assert!(*step < *steps, "every step has been taken; finish_training");
    let last = (*step + count).min(*steps);
    let view_count = trainer.view_count();
    let mut view = 0;
    while *step < last {
        view = *step % view_count;
        trainer.step_async(view, position_rate_factor(*step, *steps)).await;
        *step += 1;
    }
    let loss = trainer.log_step_async(*step - 1, view).await;
    let status = format!("step {}/{}: {loss}", *step, *steps);
    put_scene(scene);
    status
}

/// Ends the training: the Gaussians read back and pruned replace the scene's. A summary.
#[wasm_bindgen]
pub async fn finish_training() -> String {
    let mut scene = take_scene();
    let Renderer::Training { steps, step, .. } = scene.renderer else {
        panic!("not training");
    };
    assert_eq!(step, steps, "{step} of {steps} steps taken");
    let preview = preview_size(&scene.cameras[0].1);
    let placeholder = splat_renderer(&scene.cloud, preview);
    let Renderer::Training { trainer, .. } = std::mem::replace(&mut scene.renderer, placeholder) else {
        unreachable!();
    };
    let report = trainer.finish_async().await;
    let largest_turn = scene
        .views
        .iter()
        .zip(&report.cameras)
        .map(|(view, refined)| pose_distance(refined, &view.camera).0)
        .fold(0.0, f64::max);
    let summary = format!(
        "{} steps in {:.1?}: {}; {}; cameras turned up to {largest_turn:.2}°",
        steps,
        report.elapsed,
        report.final_loss,
        report.prune
    );
    // The refined cameras replace the photos' (so that training again starts from them)
    // and, at the size of cameras.json, the scene's (render, the turntable).
    for ((view, name), refined) in scene.views.iter_mut().zip(&scene.view_names).zip(&report.cameras) {
        view.camera = refined.clone();
        let (_, camera) = scene
            .cameras
            .iter_mut()
            .find(|(camera_name, _)| camera_name == name)
            .expect("every view has a camera in cameras.json");
        *camera = scale_camera(refined, camera.width, camera.height);
    }
    scene.cloud = report.cloud;
    scene.renderer = splat_renderer(&scene.cloud, preview);
    put_scene(scene);
    summary
}

/// The scene's Gaussians as a 3DGS `splat.ply`.
#[wasm_bindgen]
pub fn splat_ply() -> Vec<u8> {
    SCENE.with(|scene| {
        let scene = scene.borrow();
        let mut bytes = Vec::new();
        write_splat_ply_to(&mut bytes, &scene.as_ref().expect("no scene").cloud).expect("writing to memory");
        bytes
    })
}
