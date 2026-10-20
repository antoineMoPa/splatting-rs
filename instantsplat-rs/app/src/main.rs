//! A desktop window for instantsplat-rs's photos → Gaussian splat chain: the browser
//! demo's page (`../web/index.html`). Pick photos (or give them on the command line),
//! watch the network and the training on the status line and the live 3D view, drag to
//! look around, save the `.ply`; "Advanced" (`advanced.rs`) has the same controls as the
//! page's: the photo to view from, the training, more training, loading a `fit` run. The
//! chain runs on its own
//! thread with its own GPU device (`pipeline.rs`), and only once the start button is
//! clicked (it names the model download when the model is not cached yet); this
//! file is the window.

mod advanced;
mod look;
mod pipeline;
mod weights;

use std::path::PathBuf;

use clap::Parser;
use egui::{
    CentralPanel, ColorImage, ComboBox, CornerRadius, CursorIcon, Frame, Image, RichText, ScrollArea, Sense, TextureHandle, TextureOptions, Vec2,
};
use instantsplat_rs::network::Network;
use instantsplat_rs::training::CameraPoses;

use crate::look::{BODY_SIZE, FOOTER_SIZE, HEADING_SIZE, SMALL_SIZE, VIEW_RADIUS, VIEW_SHADOW};
use crate::pipeline::{DEFAULT_TRAIN_STEPS, Event, MIN_PHOTOS, Orbit, PHOTO_EXTENSIONS, Training, Worker};
use crate::weights::{ModelChoice, model_choices};

/// Photos to a Gaussian splat, in a window: a network predicts points and cameras, one
/// Gaussian per point is trained against the photos, drag to look around, save the PLY.
/// Nothing downloads or runs until the start button is clicked (or `--start`).
#[derive(Parser)]
struct Args {
    /// Photos to offer in the window (2 or more JPEG or PNG), with a button to start.
    photos: Vec<PathBuf>,
    /// Start on the photos given right away, as if the button were clicked.
    #[arg(long, requires = "photos")]
    start: bool,
    /// The model the window starts with (the dropdown changes it): a Depth Anything 3
    /// variant, or MapAnything.
    #[arg(long, value_enum, default_value = "da3-small")]
    network: Network,
    /// The training steps run on every reconstruction ("Advanced" changes them); 0 shows the
    /// network's Gaussians untrained.
    #[arg(long, default_value_t = DEFAULT_TRAIN_STEPS)]
    steps: usize,
    /// Train with the cameras fixed where the network put them ("Advanced" has the box).
    #[arg(long)]
    fixed_poses: bool,
    /// Load this `fit` run (splat.ply, cameras.json, input_view{i}.png) at start, as
    /// "Advanced"'s "Load a run" would.
    #[arg(long, conflicts_with_all = ["photos", "start"])]
    run: Option<PathBuf>,
    /// Write a PNG of the window here once it is idle (after the run with --start), for
    /// tests.
    #[arg(long)]
    screenshot: Option<PathBuf>,
    /// Write the splat here once the --start run is done, as the Save button would, for
    /// tests.
    #[arg(long, requires = "start")]
    save_to: Option<PathBuf>,
}

const WINDOW_TITLE: &str = "Photos to Gaussian Splat";
const WINDOW_SIZE: [f32; 2] = [800.0, 960.0];
const WINDOW_MIN_SIZE: [f32; 2] = [520.0, 640.0];
const WINDOW_POSITION: [f32; 2] = [40.0, 40.0];
/// Degrees of yaw or pitch per point dragged, as the page.
const DRAG_DEGREES_PER_POINT: f64 = 0.3;
/// The view takes at most this much of the window's height (the page caps it at 70vh; less
/// here so that the whole page fits a laptop screen).
const VIEW_MAX_HEIGHT_FRACTION: f32 = 0.42;
/// The view before anything is rendered: 4:3, as a portrait photo's preview.
const PLACEHOLDER_SIZE: [usize; 2] = [4, 3];
const PREVIEW_TEXTURE: &str = "preview";
const READY_STATUS: &str = "ready: choose your photos.";
/// A `--screenshot` waits for this many painted frames once the window is idle, so that
/// the last status and preview are on screen.
const SCREENSHOT_SETTLE_FRAMES: u64 = 3;

fn main() -> eframe::Result {
    let args = Args::parse();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(WINDOW_TITLE)
            .with_inner_size(WINDOW_SIZE)
            .with_min_inner_size(WINDOW_MIN_SIZE)
            .with_position(WINDOW_POSITION),
        ..Default::default()
    };
    eframe::run_native(WINDOW_TITLE, options, Box::new(move |cc| Ok(Box::new(App::new(cc, args)))))
}

struct App {
    worker: Worker,
    /// The dropdown's entries.
    models: Vec<ModelChoice>,
    /// The dropdown's choice, the next run's network.
    network: Network,
    status: String,
    /// The bar's fill, or hidden.
    progress: Option<f32>,
    /// A reconstruction, a run load or training is running: no picking, no saving.
    busy: bool,
    /// A splat is there to save and to train.
    has_scene: bool,
    /// The scene's photos, by name: the views to look from.
    photo_names: Vec<String>,
    /// "Advanced"'s training steps and box.
    train_steps: usize,
    refine_poses: bool,
    /// Photos from the command line, waiting for the start button.
    chosen: Vec<PathBuf>,
    preview: Option<TextureHandle>,
    /// Of `preview`, or the placeholder's aspect ratio.
    preview_size: [usize; 2],
    orbit: Orbit,
    /// Frames painted so far.
    frames: u64,
    /// `--screenshot`: where, and not before this many frames.
    screenshot: Option<(PathBuf, u64)>,
    /// `--save-to`, until the run is done.
    save_to: Option<PathBuf>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, args: Args) -> Self {
        look::apply(&cc.egui_ctx);
        let mut app = Self {
            worker: Worker::spawn(cc.egui_ctx.clone()),
            models: model_choices(),
            network: args.network,
            status: READY_STATUS.to_string(),
            progress: None,
            busy: false,
            has_scene: false,
            photo_names: Vec::new(),
            train_steps: args.steps,
            refine_poses: !args.fixed_poses,
            chosen: Vec::new(),
            preview: None,
            preview_size: PLACEHOLDER_SIZE,
            orbit: Orbit::default(),
            frames: 0,
            screenshot: args.screenshot.map(|path| (path, SCREENSHOT_SETTLE_FRAMES)),
            save_to: args.save_to,
        };
        if let Some(dir) = args.run {
            app.load_run(dir);
        } else if args.start {
            app.start(args.photos);
        } else if !args.photos.is_empty() {
            app.choose(args.photos);
        }
        app
    }

    /// Keeps `photos` for the start button: choosing photos downloads and runs nothing.
    fn choose(&mut self, photos: Vec<PathBuf>) {
        if photos.len() < MIN_PHOTOS {
            self.status = format!("pick at least {MIN_PHOTOS} photos");
            return;
        }
        self.status = format!("{} photos chosen.", photos.len());
        self.chosen = photos;
    }

    /// The start button's text: it says when it also downloads the selected model.
    fn start_label(&self) -> String {
        let choice = self.models.iter().find(|choice| choice.network == self.network).expect("the chosen network is a model choice");
        match choice.cached() {
            Some(_) => "Make the splat".to_string(),
            None => format!("Download the {} model and make the splat", choice.size_label),
        }
    }

    /// Runs the chain on `photos`, unless there are too few.
    fn start(&mut self, photos: Vec<PathBuf>) {
        assert!(!self.busy, "start while busy");
        if photos.len() < MIN_PHOTOS {
            self.status = format!("pick at least {MIN_PHOTOS} photos");
            return;
        }
        println!("photos: {}", photos.iter().map(|photo| photo.display().to_string()).collect::<Vec<_>>().join(" "));
        self.busy = true;
        // Kept after the run: another model can be tried on the same photos.
        self.chosen = photos.clone();
        self.orbit = Orbit::default();
        self.status = format!("reconstructing {} photos…", photos.len());
        self.progress = Some(0.0);
        self.worker.reconstruct(photos, self.network, self.training());
    }

    /// "Advanced"'s training settings.
    fn training(&self) -> Training {
        Training {
            steps: self.train_steps,
            camera_poses: if self.refine_poses { CameraPoses::Refined } else { CameraPoses::Fixed },
        }
    }

    fn load_run(&mut self, dir: PathBuf) {
        assert!(!self.busy, "load_run while busy");
        self.busy = true;
        self.worker.load_run(dir);
    }

    /// More training on the scene, "Advanced"'s steps of it.
    fn train_more(&mut self) {
        assert!(!self.busy && self.has_scene && self.train_steps > 0, "train_more without a scene or steps, or while busy");
        self.busy = true;
        self.progress = Some(0.0);
        self.worker.train(self.training());
    }

    /// The small model dropdown under its label, disabled while a run uses the model chosen.
    fn model_dropdown(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            look::muted(ui, "model", SMALL_SIZE);
            let selected = &self.models.iter().find(|choice| choice.network == self.network).expect("the chosen network is a model choice").label;
            ui.add_enabled_ui(!self.busy, |ui| {
                ComboBox::from_id_salt("model")
                    .selected_text(RichText::new(selected).size(SMALL_SIZE).color(look::MUTED))
                    .show_ui(ui, |ui| {
                        for choice in &self.models {
                            ui.selectable_value(&mut self.network, choice.network, RichText::new(&choice.label).size(SMALL_SIZE));
                        }
                    });
            });
        });
    }

    fn take_events(&mut self, ctx: &egui::Context) {
        while let Some(event) = self.worker.next_event() {
            match event {
                Event::Status(text) => self.status = text,
                Event::Progress(fraction) => self.progress = fraction,
                Event::Frame(frame) => {
                    let image = ColorImage::from_rgb([frame.width, frame.height], &frame.rgb);
                    self.preview_size = [frame.width, frame.height];
                    match &mut self.preview {
                        Some(texture) => texture.set(image, TextureOptions::LINEAR),
                        None => self.preview = Some(ctx.load_texture(PREVIEW_TEXTURE, image, TextureOptions::LINEAR)),
                    }
                }
                Event::Photos(names) => {
                    self.photo_names = names;
                    self.orbit = Orbit::default();
                }
                Event::Finished { ok } => {
                    self.busy = false;
                    self.has_scene |= ok;
                    if let Some((_, not_before)) = &mut self.screenshot {
                        *not_before = self.frames + SCREENSHOT_SETTLE_FRAMES;
                        // A covered window is not painted (and so never shot): bring it up.
                        ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                    }
                    if let Some(path) = self.save_to.take()
                        && ok
                    {
                        self.worker.save(path);
                    }
                }
            }
        }
    }

    /// Writes the `--screenshot` once the window is idle and settled.
    fn take_screenshot(&mut self, ctx: &egui::Context) {
        let Some((_, not_before)) = &self.screenshot else {
            return;
        };
        if self.busy || self.frames < *not_before {
            // Frames are painted on demand: keep them coming until the settle count is reached.
            ctx.request_repaint();
            return;
        }
        let (path, _) = self.screenshot.take().expect("checked above");
        ctx.request_screenshot(move |image| {
            let [width, height] = image.size;
            let rgba = image::RgbaImage::from_raw(width as u32, height as u32, image.as_raw().to_vec()).expect("screenshot bytes");
            rgba.save(&path).expect("writing the screenshot");
            println!("screenshot written to {}", path.display());
        });
    }

    fn pick_photos() -> Option<Vec<PathBuf>> {
        rfd::FileDialog::new()
            .set_title("Choose photos")
            .add_filter("Photos (JPEG, PNG)", &PHOTO_EXTENSIONS)
            .pick_files()
    }

    fn pick_save_path() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Save the splat")
            .set_file_name("splat.ply")
            .add_filter("Gaussian splat", &["ply"])
            .save_file()
    }

    fn photos_card(&mut self, ui: &mut egui::Ui, hovering_files: bool) {
        look::card(ui, |ui| {
            // The model first: it is chosen before the photos start the run.
            self.model_dropdown(ui);
            ui.add_space(12.0 - ui.spacing().item_spacing.y);
            let chosen_title = format!("{} photos chosen", self.chosen.len());
            let (title, subtitle) = if self.chosen.is_empty() {
                (look::DROP_TITLE, look::DROP_SUBTITLE)
            } else {
                (chosen_title.as_str(), "or choose other photos")
            };
            if look::drop_zone(ui, title, subtitle, hovering_files, !self.busy).clicked()
                && let Some(photos) = Self::pick_photos()
            {
                self.choose(photos);
            }
            if !self.chosen.is_empty() {
                ui.add_space(16.0 - ui.spacing().item_spacing.y);
                ui.vertical_centered(|ui| {
                    if look::primary_button(ui, &self.start_label(), !self.busy) {
                        self.start(self.chosen.clone());
                    }
                });
            }
            if let Some(fraction) = self.progress {
                ui.add_space(16.0 - ui.spacing().item_spacing.y);
                look::progress_bar(ui, fraction);
            }
            ui.add_space(10.0 - ui.spacing().item_spacing.y);
            look::muted(ui, &self.status, SMALL_SIZE);
        });
    }

    /// The preview (or its placeholder) at `size`; dragging it orbits the camera.
    fn view(&mut self, ui: &mut egui::Ui, size: Vec2) {
        let (rect, response) = ui.allocate_exact_size(size, Sense::drag());
        ui.painter().add(VIEW_SHADOW.as_shape(rect, CornerRadius::same(VIEW_RADIUS)));
        match &self.preview {
            Some(texture) => Image::new(texture).corner_radius(CornerRadius::same(VIEW_RADIUS)).paint_at(ui, rect),
            None => {
                ui.painter().rect_filled(rect, f32::from(VIEW_RADIUS), look::VIEW_BG);
            }
        }
        if self.preview.is_none() {
            return;
        }
        let dragged = response.dragged();
        if dragged {
            let delta = response.drag_delta();
            self.orbit.yaw_degrees += f64::from(delta.x) * DRAG_DEGREES_PER_POINT;
            self.orbit.pitch_degrees += f64::from(delta.y) * DRAG_DEGREES_PER_POINT;
            self.worker.look(self.orbit);
        }
        response.on_hover_cursor(if dragged { CursorIcon::Grabbing } else { CursorIcon::Grab });
    }

    fn view_card(&mut self, ui: &mut egui::Ui, window_height: f32) {
        look::card(ui, |ui| {
            ui.vertical_centered(|ui| {
                let [width, height] = self.preview_size.map(|n| n as f32);
                let scale = (ui.available_width() / width).min(window_height * VIEW_MAX_HEIGHT_FRACTION / height);
                self.view(ui, Vec2::new(width * scale, height * scale));
                ui.add_space(12.0 - ui.spacing().item_spacing.y);
                look::muted(ui, "drag to look around", SMALL_SIZE);
                ui.add_space(12.0 - ui.spacing().item_spacing.y);
                if look::primary_button(ui, "Save .ply", self.has_scene && !self.busy)
                    && let Some(path) = Self::pick_save_path()
                {
                    self.worker.save(path);
                }
            });
        });
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.take_events(ctx);
        let dropped: Vec<PathBuf> = ctx.input(|input| input.raw.dropped_files.iter().map(|file| file.path().to_path_buf()).collect());
        if !dropped.is_empty() && !self.busy {
            self.choose(dropped);
        }
        self.take_screenshot(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.frames += 1;
        let hovering_files = ui.input(|input| !input.raw.hovered_files.is_empty());
        let window_height = ui.ctx().content_rect().height();
        CentralPanel::default()
            .frame(Frame::new().fill(look::BG).inner_margin(look::PAGE_MARGIN))
            .show(ui, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    look::page_column(ui, |ui| {
                        look::ink(ui, "Photos to Gaussian Splat", HEADING_SIZE);
                        ui.add_space(4.0 - ui.spacing().item_spacing.y);
                        look::muted(ui, "Turn object photos into a Gaussian splat on your machine.", BODY_SIZE);
                        ui.add_space(24.0 - ui.spacing().item_spacing.y);
                        self.photos_card(ui, hovering_files);
                        ui.add_space(look::CARD_GAP - ui.spacing().item_spacing.y);
                        self.view_card(ui, window_height);
                        ui.add_space(look::CARD_GAP - ui.spacing().item_spacing.y);
                        self.advanced_card(ui);
                        ui.add_space(24.0 - ui.spacing().item_spacing.y);
                        look::muted(
                            ui,
                            "Nothing leaves your machine: the photos stay here and the network is downloaded once.",
                            FOOTER_SIZE,
                        );
                    });
                });
            });
    }
}
