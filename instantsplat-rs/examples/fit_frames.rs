//! The `fit` command from Rust: frames in, splat out, with the defaults of the CLI.
//!
//! cargo run --release --example fit_frames -- 26 frame_0001.png frame_0009.png frame_0017.png
//!
//! The first argument is the 35 mm-equivalent focal length (video frames have no EXIF).
//! Writes `runs/example_fit/`: `splat.ply`, `cameras.json` and one render per frame.

use std::path::PathBuf;

use anyhow::{Result, ensure};
use candle_core::DType;

use instantsplat_rs::fit::{FitSettings, run_fit};
use instantsplat_rs::network::Network;
use instantsplat_rs::optimizer::LEARNING_RATES_3DGS;
use instantsplat_rs::training::CameraPoses;

const OUT: &str = "runs/example_fit";
const STEPS: usize = 1000;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() >= 3, "usage: fit_frames <focal 35 mm> <frame> <frame> [frame...]");
    let focal_35mm: f64 = args[0].parse()?;
    let images: Vec<PathBuf> = args[1..].iter().map(PathBuf::from).collect();

    run_fit(&FitSettings {
        images,
        out: PathBuf::from(OUT),
        steps: STEPS,
        network: Network::Mapanything,
        focal_35mm: Some(focal_35mm),
        stride: 2,
        min_conf: 1.0,
        dtype: DType::F16,
        weights: None,
        cpu: false,
        log_every: 100,
        camera_poses: CameraPoses::Refined,
        learning_rates: LEARNING_RATES_3DGS,
        evaluation: None,
    })
}
