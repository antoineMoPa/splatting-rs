//! The photos → splat chain on one background thread that owns the GPU: the network on
//! candle/Metal, one Gaussian per point, `TRAIN_STEPS` steps of the wgpu trainer with the
//! preview rendered live from its parameters, then the trained cloud kept for `Save`. The
//! window talks to it through `Worker`: commands in, `Event`s out, and the orbit the
//! preview is rendered from in a mutex, so that a drag never waits for the GPU. The same
//! sequence as the browser demo (`../web/src/lib.rs`, `reconstruct.rs`). Nothing happens
//! before the first `Reconstruct`: the thread sleeps on its channel, the GPU device and
//! the weights are requested by that command.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use candle_core::DType;
use hf_hub::api::Progress;
use instantsplat_rs::camera::PinholeCamera;
use instantsplat_rs::gaussians::GaussianCloud;
use instantsplat_rs::gpu::Gpu;
use instantsplat_rs::init_from_points::{INIT_OPACITY, InitSettings, gaussians_from_dense_views};
use instantsplat_rs::loss::GROUND_TRUTH_STRIDE;
use instantsplat_rs::network::{Network, NetworkSettings, dense_views_from_images};
use instantsplat_rs::optimizer::{LEARNING_RATES_3DGS, position_rate_factor};
use instantsplat_rs::rasterizer::Rasterizer;
use instantsplat_rs::splat_ply::write_splat_ply;
use instantsplat_rs::training::{CameraPoses, Trainer};
use instantsplat_rs::turntable::Turntable;

use crate::weights::{ModelChoice, model_choice};

/// Steps run on every reconstruction before the result is shown (the browser demo's
/// `TRAIN_STEPS`).
pub const TRAIN_STEPS: usize = 300;
/// The preview is rendered from the trainer's live parameters every this many steps.
const PREVIEW_EVERY: usize = 10;
/// Photos the chain needs at least (the networks place cameras from two views up).
pub const MIN_PHOTOS: usize = 2;
/// The photo files the chain reads, by extension (lowercased).
pub const PHOTO_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];
/// As `fit`'s defaults and the browser demo: one Gaussian per 2x2 pixels, every point
/// the cleaning kept.
const INIT: InitSettings = InitSettings {
    stride: 2,
    min_conf: 1.0,
    opacity: INIT_OPACITY,
};
/// The preview is rendered at the photos' height and at least this wide for it (4:3):
/// portrait photos get a landscape view that shows more of the scene at the sides.
const PREVIEW_MIN_ASPECT_RATIO: f64 = 4.0 / 3.0;
/// The network's precision on Metal, as the command line's default.
const NETWORK_DTYPE: DType = DType::F16;

/// Progress bar shares: the model download is its own step; a reconstruction is the
/// network, then training.
#[derive(Clone, Copy)]
pub enum Stage {
    Download,
    Reconstruct,
    Train,
}

const STAGE_SHARES: [(Stage, f32, f32); 3] = [
    (Stage::Download, 0.0, 1.0),
    (Stage::Reconstruct, 0.0, 0.5),
    (Stage::Train, 0.5, 1.0),
];

impl Stage {
    /// Where `fraction` of this stage lands on the whole bar, in [0, 1].
    pub fn progress(self, fraction: f32) -> f32 {
        let (_, from, to) = STAGE_SHARES[self as usize];
        from + (to - from) * fraction.clamp(0.0, 1.0)
    }
}

/// Where the preview camera stands: the first photo's camera moved on the turntable
/// (`instantsplat_rs::turntable`), in degrees.
#[derive(Clone, Copy, Default)]
pub struct Orbit {
    pub yaw_degrees: f64,
    pub pitch_degrees: f64,
}

pub enum Command {
    /// Runs the whole chain on these photo files with this network.
    Reconstruct { photos: Vec<PathBuf>, network: Network },
    /// Renders the preview again from the current `Orbit`.
    Render,
    /// Writes the trained splat as a 3DGS PLY.
    Save(PathBuf),
}

/// The preview as RGB bytes (over the display background), for an egui texture.
pub struct PreviewFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

pub enum Event {
    Status(String),
    /// The bar's fill in [0, 1], or hidden.
    Progress(Option<f32>),
    Frame(PreviewFrame),
    /// A reconstruction ended, with a splat to look at and save when `ok`.
    Finished { ok: bool },
}

/// The window's end of the worker thread.
pub struct Worker {
    commands: Sender<Command>,
    events: Receiver<Event>,
    orbit: Arc<Mutex<Orbit>>,
}

impl Worker {
    /// Starts the thread; `ctx` is repainted whenever an event is sent.
    pub fn spawn(ctx: egui::Context) -> Self {
        let (commands, command_receiver) = channel();
        let (event_sender, events) = channel();
        let orbit = Arc::new(Mutex::new(Orbit::default()));
        let thread_orbit = Arc::clone(&orbit);
        thread::Builder::new()
            .name("pipeline".to_string())
            .spawn(move || {
                let mut thread = PipelineThread {
                    gpu: None,
                    commands: command_receiver,
                    events: Events { sender: event_sender, ctx },
                    orbit: thread_orbit,
                    scene: None,
                };
                thread.run();
            })
            .expect("spawning the pipeline thread");
        Self { commands, events, orbit }
    }

    pub fn reconstruct(&self, photos: Vec<PathBuf>, network: Network) {
        self.commands.send(Command::Reconstruct { photos, network }).expect("the pipeline thread is gone");
    }

    /// Moves the preview camera; the next frame is rendered from here.
    pub fn look(&self, orbit: Orbit) {
        *self.orbit.lock().expect("orbit lock") = orbit;
        self.commands.send(Command::Render).expect("the pipeline thread is gone");
    }

    pub fn save(&self, path: PathBuf) {
        self.commands.send(Command::Save(path)).expect("the pipeline thread is gone");
    }

    /// The next event, if one is waiting. Panics when the thread has stopped (a panic
    /// over there, printed to the terminal).
    pub fn next_event(&self) -> Option<Event> {
        match self.events.try_recv() {
            Ok(event) => Some(event),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => panic!("the pipeline thread stopped: see the terminal"),
        }
    }
}

/// The thread's end of the event channel, waking the window on every send.
struct Events {
    sender: Sender<Event>,
    ctx: egui::Context,
}

impl Events {
    fn send(&self, event: Event) {
        // The window may have closed while the chain runs: nobody to tell, no harm.
        let _ = self.sender.send(event);
        self.ctx.request_repaint();
    }

    fn status(&self, text: impl Into<String>) {
        let text = text.into();
        println!("{text}");
        self.send(Event::Status(text));
    }

    fn progress(&self, stage: Stage, fraction: f32) {
        self.send(Event::Progress(Some(stage.progress(fraction))));
    }

    fn hide_progress(&self) {
        self.send(Event::Progress(None));
    }
}

/// A splat to look at: the first camera to orbit, the turntable it moves on, and a
/// rasterizer over the parameters (the trainer's while training, the trained cloud's
/// after).
struct Scene {
    camera: PinholeCamera,
    turntable: Turntable,
    /// Kept alongside the rasterizer that binds it.
    _params: wgpu::Buffer,
    rasterizer: Rasterizer,
    cloud: GaussianCloud,
}

struct PipelineThread {
    /// Requested by the first reconstruction.
    gpu: Option<Gpu>,
    commands: Receiver<Command>,
    events: Events,
    orbit: Arc<Mutex<Orbit>>,
    scene: Option<Scene>,
}

/// The preview's image size for cameras of `camera`'s: see `PREVIEW_MIN_ASPECT_RATIO`.
fn preview_size(camera: &PinholeCamera) -> (u32, u32) {
    let min_width = (camera.height as f64 * PREVIEW_MIN_ASPECT_RATIO).round() as u32;
    (camera.width.max(min_width), camera.height)
}

/// A rasterizer over `params` (`gaussian_count` packed Gaussians) at the preview size.
fn preview_rasterizer(gpu: &Gpu, params: &wgpu::Buffer, gaussian_count: usize, (width, height): (u32, u32)) -> Rasterizer {
    // The rasterizer wants a ground truth to take losses against; none is taken here.
    let ground_truth = gpu.storage_buffer("preview ground truth", (width * height) as u64 * GROUND_TRUTH_STRIDE as u64 * 4);
    Rasterizer::new(gpu, params, gaussian_count, width, height, &ground_truth, 1)
}

/// `camera` moved by `orbit` on `turntable`, rendered through `rasterizer` at the
/// preview size over the display background.
fn render_preview(gpu: &Gpu, rasterizer: &Rasterizer, camera: &PinholeCamera, turntable: &Turntable, orbit: Orbit) -> PreviewFrame {
    let (width, height) = preview_size(camera);
    let camera = turntable
        .view(camera, orbit.yaw_degrees.to_radians(), orbit.pitch_degrees.to_radians())
        .with_image_size(width, height);
    let rendered = rasterizer.render(gpu, &camera);
    PreviewFrame {
        width: rendered.width as usize,
        height: rendered.height as usize,
        rgb: rendered.to_rgb8().into_raw(),
    }
}

/// Hugging Face's download progress as status lines and the bar.
struct DownloadProgress<'a> {
    events: &'a Events,
    total: usize,
    downloaded: usize,
    last_percent: usize,
}

impl Progress for DownloadProgress<'_> {
    fn init(&mut self, size: usize, _filename: &str) {
        self.total = size;
        self.events.status(format!("downloading the model ({} MB, once): 0%", size / 1_000_000));
        self.events.progress(Stage::Download, 0.0);
    }

    fn update(&mut self, size: usize) {
        self.downloaded += size;
        let fraction = self.downloaded as f32 / self.total.max(1) as f32;
        let percent = (fraction * 100.0) as usize;
        if percent > self.last_percent {
            self.last_percent = percent;
            self.events.status(format!("downloading the model ({} MB, once): {percent}%", self.total / 1_000_000));
            self.events.progress(Stage::Download, fraction);
        }
    }

    fn finish(&mut self) {
        self.events.hide_progress();
    }
}

/// The network's checkpoint on disk: Hugging Face's cache, or downloaded into it now.
fn fetch_weights(source: &ModelChoice, events: &Events) -> Result<PathBuf> {
    if let Some(path) = source.cached() {
        return Ok(path);
    }
    events.status(format!("downloading the model ({}, once)…", source.size_label));
    let progress = DownloadProgress {
        events,
        total: 0,
        downloaded: 0,
        last_percent: 0,
    };
    hf_hub::api::sync::Api::new()?
        .model(source.repo.to_string())
        .download_with_progress(source.file, progress)
        .with_context(|| format!("downloading {}/{}", source.repo, source.file))
}

fn check_photo(path: &Path) -> Result<()> {
    let extension = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    ensure!(
        extension.as_deref().is_some_and(|e| PHOTO_EXTENSIONS.contains(&e)),
        "{} is not a JPEG or PNG photo",
        path.display()
    );
    ensure!(path.is_file(), "{} does not exist", path.display());
    Ok(())
}

impl PipelineThread {
    fn run(&mut self) {
        while let Ok(command) = self.commands.recv() {
            match command {
                Command::Reconstruct { photos, network } => self.reconstruct(photos, network),
                Command::Render => {
                    self.drain_renders();
                    self.render();
                }
                Command::Save(path) => self.save(&path),
            }
        }
    }

    /// Takes every `Render` waiting (one render shows the latest orbit for all of them).
    /// Nothing else may be waiting: the window sends no other command while busy.
    fn drain_renders(&self) {
        while let Ok(command) = self.commands.try_recv() {
            assert!(matches!(command, Command::Render), "a command other than Render while busy");
        }
    }

    fn orbit(&self) -> Orbit {
        *self.orbit.lock().expect("orbit lock")
    }

    /// The GPU device, once a reconstruction has requested it.
    fn gpu(&self) -> &Gpu {
        self.gpu.as_ref().expect("the GPU is requested by the first reconstruction")
    }

    fn render(&self) {
        let Some(scene) = &self.scene else {
            return;
        };
        let frame = render_preview(self.gpu(), &scene.rasterizer, &scene.camera, &scene.turntable, self.orbit());
        self.events.send(Event::Frame(frame));
    }

    fn save(&self, path: &Path) {
        let scene = self.scene.as_ref().expect("Save before any reconstruction");
        match write_splat_ply(path, &scene.cloud) {
            Ok(()) => self.events.status(format!(
                "saved {} ({} Gaussians, {:.1} MB). Drag to look around.",
                path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string()),
                scene.cloud.len(),
                std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f64 / 1e6
            )),
            Err(error) => self.events.status(format!("error: writing {}: {error:#}", path.display())),
        }
    }

    fn reconstruct(&mut self, photos: Vec<PathBuf>, network: Network) {
        let ok = match self.run_chain(&photos, network) {
            Ok(()) => true,
            Err(error) => {
                self.events.hide_progress();
                self.events.status(format!("error: {error:#}"));
                false
            }
        };
        self.events.send(Event::Finished { ok });
    }

    /// Network → Gaussians → `TRAIN_STEPS` steps, previews along the way; the trained
    /// splat becomes the scene.
    fn run_chain(&mut self, photos: &[PathBuf], network: Network) -> Result<()> {
        ensure!(photos.len() >= MIN_PHOTOS, "{} photos: pick at least {MIN_PHOTOS}", photos.len());
        for photo in photos {
            check_photo(photo)?;
        }
        *self.orbit.lock().expect("orbit lock") = Orbit::default();
        let weights = fetch_weights(&model_choice(network), &self.events)?;
        if self.gpu.is_none() {
            self.gpu = Some(Gpu::request());
        }
        let gpu = self.gpu();

        let started = Instant::now();
        let seconds = || started.elapsed().as_secs_f64().round() as u64;
        self.events.progress(Stage::Reconstruct, 0.0);
        self.events.status(format!("reconstructing {} photos with {}…", photos.len(), network.label()));
        let (views, timing) = dense_views_from_images(
            photos,
            &NetworkSettings {
                network,
                focal_35mm: None,
                dtype: NETWORK_DTYPE,
                weights: Some(weights),
                cpu: false,
            },
        )?;
        self.events.progress(Stage::Reconstruct, 0.9);
        self.events.status(format!(
            "reconstructed: {} views of {}x{} in {:.1?} (+{:.1?} loading the weights)",
            views.len(),
            views[0].width,
            views[0].height,
            timing.inference,
            timing.load_weights
        ));

        let cloud = gaussians_from_dense_views(&views, INIT);
        let camera = views[0].camera.clone();
        let cameras: Vec<PinholeCamera> = views.iter().map(|view| view.camera.clone()).collect();
        let turntable = Turntable::from_cameras(&cameras, cloud.centroid());
        let preview = preview_size(&camera);
        let mut trainer = Trainer::new(gpu, &cloud, &views, LEARNING_RATES_3DGS, CameraPoses::Fixed);
        let live = preview_rasterizer(gpu, trainer.params(), trainer.gaussian_count(), preview);
        self.events.send(Event::Frame(render_preview(gpu, &live, &camera, &turntable, self.orbit())));
        self.events.progress(Stage::Train, 0.0);

        let view_count = trainer.view_count();
        for step in 0..TRAIN_STEPS {
            let view = step % view_count;
            trainer.step(view, position_rate_factor(step, TRAIN_STEPS));
            let done = step + 1;
            if done % PREVIEW_EVERY == 0 || done == TRAIN_STEPS {
                pollster::block_on(trainer.log_step_async(step, view));
                self.drain_renders();
                self.events.send(Event::Frame(render_preview(gpu, &live, &camera, &turntable, self.orbit())));
                self.events.progress(Stage::Train, done as f32 / TRAIN_STEPS as f32);
                self.events.status(format!("refining: step {done} of {TRAIN_STEPS} ({} s)", seconds()));
            }
        }
        drop(live);
        let report = pollster::block_on(trainer.finish_async());
        println!("trained: {TRAIN_STEPS} steps in {:.1?}: {}; {}", report.elapsed, report.final_loss, report.prune);

        let params = gpu.storage_buffer_from("splat params", &report.cloud.to_packed());
        let rasterizer = preview_rasterizer(gpu, &params, report.cloud.len(), preview);
        self.scene = Some(Scene {
            camera,
            turntable,
            _params: params,
            rasterizer,
            cloud: report.cloud,
        });
        self.render();
        self.events.hide_progress();
        self.events.status(format!(
            "done in {} s: {} Gaussians from {} photos with {}. Drag to look around.",
            seconds(),
            self.scene.as_ref().expect("scene just set").cloud.len(),
            photos.len(),
            network.label()
        ));
        Ok(())
    }
}
