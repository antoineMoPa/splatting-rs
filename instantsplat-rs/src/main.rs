use std::path::PathBuf;

use anyhow::Result;
use candle_core::DType;
use clap::{Parser, Subcommand, ValueEnum};

use instantsplat_rs::evaluate_against_colmap::EvaluationSettings;
use instantsplat_rs::fit::{FitSettings, run_fit};
use instantsplat_rs::network::Network;
use instantsplat_rs::optimizer::{LEARNING_RATES_3DGS, LearningRates};
use instantsplat_rs::refine::{RefineSettings, run_refine};
use instantsplat_rs::training::CameraPoses;

#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    F16,
    Bf16,
}

impl From<Precision> for DType {
    fn from(precision: Precision) -> Self {
        match precision {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        }
    }
}

/// Gaussian splats from a few photos (`fit`), refined against many posed photos
/// (`refine`).
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// A network (MapAnything or Depth Anything 3) predicts dense points and cameras for a
    /// few photos; one Gaussian per point is optimized against them with a wgpu
    /// rasterizer, the camera poses along.
    Fit(FitArgs),
    /// Continue optimizing a splat from `fit` against every frame of a COLMAP model,
    /// aligned to the splat by the frames they share; every k-th frame is held out and
    /// its L1 before and after is reported.
    Refine(RefineArgs),
}

#[derive(Parser)]
struct FitArgs {
    /// Two or more JPEG or PNG images of the scene, view 0 first. Transparent pixels of
    /// PNGs are background and get no Gaussians.
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,
    /// Output directory: `splat.ply` (3DGS PLY, in the first image's camera frame, OpenCV:
    /// x right, y down, z forward), `cameras.json` and one render per view.
    #[arg(long)]
    out: PathBuf,
    /// Optimization steps; each renders one view.
    #[arg(long, default_value_t = 1000)]
    steps: usize,
    /// The network predicting points and cameras: MapAnything (4.9 GB of weights, takes
    /// the focal length as input) or Depth Anything 3 (Small 137 MB, Base 542 MB, Large
    /// 1.6 GB, Giant 5.4 GB; predicts its own focal length, a given one replaces it
    /// afterwards).
    #[arg(long, value_enum, default_value = "mapanything")]
    network: Network,
    /// 35 mm-equivalent focal length of the camera, in mm, for every image; overrides EXIF.
    /// Video frames have none: iPhone video is about 26. Images cropped after capture
    /// need the value for the crop, not the sensor.
    #[arg(long)]
    focal_35mm: Option<f64>,
    /// Keep one dense point in this many along each image axis as a Gaussian.
    #[arg(long, default_value_t = 2)]
    stride: usize,
    /// Drop points below this network confidence (1 + exp(raw), so always > 1, or 0
    /// when rejected).
    #[arg(long, default_value_t = 1.0)]
    min_conf: f32,
    #[arg(long, value_enum, default_value = "f16")]
    precision: Precision,
    /// The network's safetensors checkpoint; downloaded from Hugging Face when omitted.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Run the network on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,
    /// Print the loss every this many steps.
    #[arg(long, default_value_t = 50)]
    log_every: usize,
    /// Keep the cameras where the network put them instead of optimizing their poses (and
    /// a shared focal scale) along with the Gaussians.
    #[arg(long)]
    no_pose_refinement: bool,
    /// Seed the Gaussians from the network's point maps as they come, without putting the
    /// views onto each other first (`point_map_alignment.rs`); for comparison.
    #[arg(long)]
    unaligned: bool,
    /// Judge the result afterwards against a COLMAP TXT model of the whole capture
    /// (`cameras.txt`, `images.txt`, naming the input frames): COLMAP's cameras are aligned
    /// onto the optimized ones, the residuals printed, and every `--hold-out-every`-th frame
    /// rendered and compared with its photo. Not used for training.
    #[arg(long, requires = "evaluate_frames")]
    evaluate_colmap: Option<PathBuf>,
    /// Directory with the photos the COLMAP model names (RGBA PNG cutouts).
    #[arg(long, requires = "evaluate_colmap")]
    evaluate_frames: Option<PathBuf>,
    /// Which frames of the COLMAP model `--evaluate-colmap` renders: every this-many-th.
    #[arg(long, default_value_t = 10)]
    hold_out_every: usize,
    /// Learning rate of the cameras' rotation vectors, radians per step before the decay
    /// (default: `LEARNING_RATES_3DGS`).
    #[arg(long)]
    camera_rotation_lr: Option<f32>,
    /// Learning rate of the camera centers, in camera extents per step before the decay.
    #[arg(long)]
    camera_translation_lr: Option<f32>,
    /// Learning rate of the shared log focal scale.
    #[arg(long)]
    camera_focal_lr: Option<f32>,
}

#[derive(Parser)]
struct RefineArgs {
    /// A `fit` run's output directory (its `splat.ply` and `cameras.json`).
    run: PathBuf,
    /// Directory with COLMAP's `cameras.txt` and `images.txt` (TXT model), posing the
    /// frames; must include the frames of the `fit` run by the same names.
    #[arg(long)]
    colmap: PathBuf,
    /// Directory with the photos the COLMAP model names (RGBA PNG cutouts: alpha 0 is
    /// background).
    #[arg(long)]
    frames: PathBuf,
    /// Output directory: `splat.ply`, a `cameras.json` of every view and one
    /// `heldout_<frame>.png` (photo | render before | render after) per held-out frame.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 3000)]
    steps: usize,
    /// Hold every this-many-th frame (by name) out of the training set; 0 holds out none.
    #[arg(long, default_value_t = 10)]
    hold_out_every: usize,
    /// Training height = the splat's training height times this (width follows the
    /// frames' aspect ratio). 2 quadruples the pixels and the GPU memory for the photos.
    #[arg(long, default_value_t = 1.0)]
    resolution_scale: f64,
    /// Print the loss every this many steps.
    #[arg(long, default_value_t = 100)]
    log_every: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Fit(args) => run_fit(&FitSettings {
            images: args.images,
            out: args.out,
            steps: args.steps,
            network: args.network,
            focal_35mm: args.focal_35mm,
            stride: args.stride,
            min_conf: args.min_conf,
            dtype: DType::from(args.precision),
            weights: args.weights,
            cpu: args.cpu,
            log_every: args.log_every,
            align_views: !args.unaligned,
            camera_poses: if args.no_pose_refinement { CameraPoses::Fixed } else { CameraPoses::Refined },
            learning_rates: LearningRates {
                camera_rotation: args.camera_rotation_lr.unwrap_or(LEARNING_RATES_3DGS.camera_rotation),
                camera_translation: args.camera_translation_lr.unwrap_or(LEARNING_RATES_3DGS.camera_translation),
                camera_focal: args.camera_focal_lr.unwrap_or(LEARNING_RATES_3DGS.camera_focal),
                ..LEARNING_RATES_3DGS
            },
            evaluation: args.evaluate_colmap.zip(args.evaluate_frames).map(|(colmap, frames)| EvaluationSettings {
                colmap,
                frames,
                hold_out_every: args.hold_out_every,
            }),
        }),
        Command::Refine(args) => run_refine(&RefineSettings {
            run: args.run,
            colmap: args.colmap,
            frames: args.frames,
            out: args.out,
            steps: args.steps,
            hold_out_every: args.hold_out_every,
            resolution_scale: args.resolution_scale,
            log_every: args.log_every,
        }),
    }
}
