//! The photos → splat chain on one background thread that owns the GPU: the network on
//! candle/Metal, one Gaussian per point, `Training::steps` steps of the wgpu trainer with
//! the preview rendered live from its parameters, then the trained cloud kept for `Save`
//! and for more training (`Train`). A `fit` run loads as such a scene (`LoadRun`). The
//! window talks to it through `Worker`: commands in, `Event`s out, and the orbit the
//! preview is rendered from in a mutex, so that a drag never waits for the GPU. The same
//! sequence as the browser demo (`../web/src/lib.rs`, `reconstruct.rs`). Nothing happens
//! before the first `Reconstruct` or `LoadRun`: the thread sleeps on its channel, the GPU
//! device (and the weights) are requested by that command.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use candle_core::DType;
use hf_hub::api::Progress;
use instantsplat_rs::camera::PinholeCamera;
use instantsplat_rs::camera_pose::pose_distance;
use instantsplat_rs::cameras_json::read_cameras_json;
use instantsplat_rs::dense_view::DenseView;
use instantsplat_rs::gaussians::GaussianCloud;
use instantsplat_rs::gpu::Gpu;
use instantsplat_rs::init_from_points::{INIT_OPACITY, InitSettings, gaussians_from_dense_views};
use instantsplat_rs::loss::GROUND_TRUTH_STRIDE;
use instantsplat_rs::network::{Network, NetworkSettings, dense_views_from_images};
use instantsplat_rs::densify::DENSIFY_SHORT_RUN;
use instantsplat_rs::optimizer::LEARNING_RATES_SHORT_RUN;
use instantsplat_rs::point_map_alignment::align_point_maps;
use instantsplat_rs::rasterizer::Rasterizer;
use instantsplat_rs::splat_ply::{read_splat_ply, write_splat_ply};
use instantsplat_rs::training::{CameraPoses, Trainer};
use instantsplat_rs::training_frames::load_training_view;
use instantsplat_rs::turntable::Turntable;

use crate::weights::{ModelChoice, model_choice};

/// Steps run on every reconstruction before the result is shown, until "Advanced" changes
/// them (the browser demo's `DEFAULT_TRAIN_STEPS`).
pub const DEFAULT_TRAIN_STEPS: usize = 300;
/// A `fit` run's files, as `fit` writes them: the splat, its cameras, and the photo of
/// camera i (`cameras.json` order) as `input_view{i}.png`.
const RUN_SPLAT: &str = "splat.ply";
const RUN_CAMERAS: &str = "cameras.json";
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
/// network, then training; more training on a splat is its own step.
#[derive(Clone, Copy)]
pub enum Stage {
    Download,
    Reconstruct,
    Train,
    TrainMore,
}

const STAGE_SHARES: [(Stage, f32, f32); 4] = [
    (Stage::Download, 0.0, 1.0),
    (Stage::Reconstruct, 0.0, 0.5),
    (Stage::Train, 0.5, 1.0),
    (Stage::TrainMore, 0.0, 1.0),
];

impl Stage {
    /// Where `fraction` of this stage lands on the whole bar, in [0, 1].
    pub fn progress(self, fraction: f32) -> f32 {
        let (_, from, to) = STAGE_SHARES[self as usize];
        from + (to - from) * fraction.clamp(0.0, 1.0)
    }
}

/// Where the preview camera stands: the camera of photo `view` moved on the turntable
/// (`instantsplat_rs::turntable`), in degrees.
#[derive(Clone, Copy, Default)]
pub struct Orbit {
    pub view: usize,
    pub yaw_degrees: f64,
    pub pitch_degrees: f64,
}

/// How the splat is trained against the photos: the "Training" row of "Advanced".
#[derive(Clone, Copy)]
pub struct Training {
    /// 0 keeps a reconstruction's Gaussians untrained.
    pub steps: usize,
    pub camera_poses: CameraPoses,
}

pub enum Command {
    /// Runs the whole chain on these photo files with this network, then `training`.
    Reconstruct {
        photos: Vec<PathBuf>,
        network: Network,
        training: Training,
    },
    /// Loads the `fit` run in this directory as the scene.
    LoadRun(PathBuf),
    /// Trains the scene's splat more against its photos.
    Train(Training),
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
    /// A new scene (reconstructed or loaded): its photos' names, the orbit's views; the
    /// orbit is back at view 0.
    Photos(Vec<String>),
    /// A reconstruction, a run load or training ended, with a splat to look at, save and
    /// train when `ok`.
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

    pub fn reconstruct(&self, photos: Vec<PathBuf>, network: Network, training: Training) {
        self.send(Command::Reconstruct { photos, network, training });
    }

    pub fn load_run(&self, dir: PathBuf) {
        self.send(Command::LoadRun(dir));
    }

    pub fn train(&self, training: Training) {
        self.send(Command::Train(training));
    }

    /// Moves the preview camera; the next frame is rendered from here.
    pub fn look(&self, orbit: Orbit) {
        *self.orbit.lock().expect("orbit lock") = orbit;
        self.send(Command::Render);
    }

    pub fn save(&self, path: PathBuf) {
        self.send(Command::Save(path));
    }

    fn send(&self, command: Command) {
        self.commands.send(command).expect("the pipeline thread is gone");
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

/// A splat to look at and to train: the photos as training views (their cameras the
/// refined ones once trained with `CameraPoses::Refined`), and a rasterizer over the
/// cloud.
struct Scene {
    /// One per view: the file names of the photos, or a run's `cameras.json` names.
    names: Vec<String>,
    views: Vec<DenseView>,
    cloud: GaussianCloud,
    /// Kept alongside the rasterizer that binds it.
    _params: wgpu::Buffer,
    rasterizer: Rasterizer,
}

impl Scene {
    fn new(gpu: &Gpu, names: Vec<String>, views: Vec<DenseView>, cloud: GaussianCloud) -> Scene {
        assert_eq!(names.len(), views.len(), "one name per view");
        let params = gpu.storage_buffer_from("splat params", &cloud.to_packed());
        let rasterizer = preview_rasterizer(gpu, &params, cloud.len(), preview_size(&views[0].camera));
        Scene {
            names,
            views,
            cloud,
            _params: params,
            rasterizer,
        }
    }

    fn cameras(&self) -> Vec<PinholeCamera> {
        self.views.iter().map(|view| view.camera.clone()).collect()
    }

    /// The turntable the views' cameras orbit on.
    fn turntable(&self) -> Turntable {
        Turntable::from_cameras(&self.cameras(), self.cloud.centroid())
    }
}

struct PipelineThread {
    /// Requested by the first reconstruction or run load.
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

/// The file name of `path`, for the status line.
fn file_name(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
}

impl PipelineThread {
    fn run(&mut self) {
        while let Ok(command) = self.commands.recv() {
            match command {
                Command::Reconstruct { photos, network, training } => {
                    let result = self.reconstruct(&photos, network, training);
                    self.finish(result);
                }
                Command::LoadRun(dir) => {
                    let result = self.load_run(&dir);
                    self.finish(result);
                }
                Command::Train(training) => {
                    let started = Instant::now();
                    self.train(training, Stage::TrainMore);
                    self.events.hide_progress();
                    self.events.status(format!(
                        "trained {} more steps in {} s. Drag to look around.",
                        training.steps,
                        started.elapsed().as_secs_f64().round() as u64
                    ));
                    self.finish(Ok(()));
                }
                Command::Render => {
                    self.drain_renders();
                    self.render();
                }
                Command::Save(path) => self.save(&path),
            }
        }
    }

    /// Tells the window a command is over, after its error when it failed.
    fn finish(&self, result: Result<()>) {
        if let Err(error) = &result {
            self.events.hide_progress();
            self.events.status(format!("error: {error:#}"));
        }
        self.events.send(Event::Finished { ok: result.is_ok() });
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

    /// The GPU device, requested on first use.
    fn gpu(&mut self) -> &Gpu {
        self.gpu.get_or_insert_with(Gpu::request)
    }

    /// The scene from `orbit`.
    fn render(&self) {
        let Some(scene) = &self.scene else {
            return;
        };
        let gpu = self.gpu.as_ref().expect("a scene has a GPU");
        let orbit = self.orbit();
        let frame = render_preview(gpu, &scene.rasterizer, &scene.views[orbit.view].camera, &scene.turntable(), orbit);
        self.events.send(Event::Frame(frame));
    }

    /// Makes `scene` the one to look at, from its first photo.
    fn show(&mut self, scene: Scene) {
        *self.orbit.lock().expect("orbit lock") = Orbit::default();
        self.events.send(Event::Photos(scene.names.clone()));
        self.scene = Some(scene);
        self.render();
    }

    fn save(&self, path: &Path) {
        let scene = self.scene.as_ref().expect("Save before any reconstruction");
        match write_splat_ply(path, &scene.cloud) {
            Ok(()) => self.events.status(format!(
                "saved {} ({} Gaussians, {:.1} MB). Drag to look around.",
                file_name(path),
                scene.cloud.len(),
                std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f64 / 1e6
            )),
            Err(error) => self.events.status(format!("error: writing {}: {error:#}", path.display())),
        }
    }

    /// Network → Gaussians → `training`, previews along the way; the trained splat becomes
    /// the scene.
    fn reconstruct(&mut self, photos: &[PathBuf], network: Network, training: Training) -> Result<()> {
        ensure!(photos.len() >= MIN_PHOTOS, "{} photos: pick at least {MIN_PHOTOS}", photos.len());
        for photo in photos {
            check_photo(photo)?;
        }
        let weights = fetch_weights(&model_choice(network), &self.events)?;

        let started = Instant::now();
        let seconds = || started.elapsed().as_secs_f64().round() as u64;
        self.events.progress(Stage::Reconstruct, 0.0);
        self.events.status(format!("reconstructing {} photos with {}…", photos.len(), network.label()));
        let (mut views, timing) = dense_views_from_images(
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

        for alignment in align_point_maps(&mut views) {
            println!("{alignment}");
        }
        let cloud = gaussians_from_dense_views(&views, INIT);
        let names = photos.iter().map(|photo| file_name(photo)).collect();
        let scene = Scene::new(self.gpu(), names, views, cloud);
        self.show(scene);
        if training.steps > 0 {
            self.train(training, Stage::Train);
        }
        self.events.hide_progress();
        self.events.status(format!(
            "done in {} s: {} Gaussians from {} photos with {}{}. Drag to look around.",
            seconds(),
            self.scene.as_ref().expect("scene just set").cloud.len(),
            photos.len(),
            network.label(),
            if training.steps > 0 { "" } else { ", untrained" }
        ));
        Ok(())
    }

    /// The `fit` run in `dir` (`RUN_SPLAT`, `RUN_CAMERAS` and every camera's
    /// `input_view{i}.png`) as the scene.
    fn load_run(&mut self, dir: &Path) -> Result<()> {
        self.events.status(format!("loading {}…", dir.display()));
        let cloud = read_splat_ply(&dir.join(RUN_SPLAT))?;
        let cameras = read_cameras_json(&dir.join(RUN_CAMERAS))?;
        ensure!(cameras.len() >= MIN_PHOTOS, "{}: {} cameras, training needs at least {MIN_PHOTOS}", RUN_CAMERAS, cameras.len());
        let views = cameras
            .iter()
            .enumerate()
            .map(|(index, (_, camera))| load_training_view(&dir.join(format!("input_view{index}.png")), camera, (camera.width, camera.height)))
            .collect::<Result<Vec<_>>>()?;
        let names = cameras.into_iter().map(|(name, _)| name).collect();
        let scene = Scene::new(self.gpu(), names, views, cloud);
        self.show(scene);
        let scene = self.scene.as_ref().expect("scene just set");
        self.events.status(format!(
            "loaded {} Gaussians and {} photos from {}. Drag to look around.",
            scene.cloud.len(),
            scene.views.len(),
            file_name(dir)
        ));
        Ok(())
    }

    /// `training` on the scene, previews along the way, the bar filling `stage`'s share;
    /// the trained cloud and refined cameras replace the scene's.
    fn train(&mut self, training: Training, stage: Stage) {
        let Training { steps, camera_poses } = training;
        assert!(steps > 0, "training 0 steps");
        let scene = self.scene.take().expect("training without a scene");
        let gpu = self.gpu.as_ref().expect("a scene has a GPU");
        let started = Instant::now();
        let turntable = scene.turntable();
        let mut trainer =
            Trainer::new(gpu, &scene.cloud, &scene.views, LEARNING_RATES_SHORT_RUN, camera_poses).with_densification(DENSIFY_SHORT_RUN);
        let live_rasterizer = |trainer: &Trainer| preview_rasterizer(gpu, trainer.params(), trainer.gaussian_count(), preview_size(&scene.views[0].camera));
        let render_live = |live: &Rasterizer, orbit: Orbit| render_preview(gpu, live, &scene.views[orbit.view].camera, &turntable, orbit);
        let mut live = live_rasterizer(&trainer);
        self.events.send(Event::Frame(render_live(&live, self.orbit())));
        self.events.progress(stage, 0.0);

        let view_count = trainer.view_count();
        for step in 0..steps {
            if trainer.train_step(step, steps).is_some() {
                // Densified: the parameters are in a new buffer.
                live = live_rasterizer(&trainer);
            }
            let done = step + 1;
            if done % PREVIEW_EVERY == 0 || done == steps {
                pollster::block_on(trainer.log_step_async(step, step % view_count));
                self.drain_renders();
                self.events.send(Event::Frame(render_live(&live, self.orbit())));
                self.events.progress(stage, done as f32 / steps as f32);
                self.events.status(format!(
                    "refining: step {done} of {steps} ({} s)",
                    started.elapsed().as_secs_f64().round() as u64
                ));
            }
        }
        drop(live);
        let report = pollster::block_on(trainer.finish_async());
        let largest_turn = scene
            .views
            .iter()
            .zip(&report.cameras)
            .map(|(view, refined)| pose_distance(refined, &view.camera).0)
            .fold(0.0, f64::max);
        println!(
            "trained: {steps} steps in {:.1?}: {}; {}; {}; cameras turned up to {largest_turn:.2}°",
            report.elapsed,
            report.final_loss,
            report.densification_summary(),
            report.prune
        );
        // The refined cameras replace the views' (view 0 is the fixed one), so that more
        // training starts from them.
        let Scene { names, mut views, .. } = scene;
        for (view, refined) in views.iter_mut().zip(report.cameras) {
            view.camera = refined;
        }
        self.scene = Some(Scene::new(gpu, names, views, report.cloud));
        self.render();
    }
}
