//! The `refine` command from Rust: continue a fitted splat against every frame of the
//! capture, with camera poses from a COLMAP TXT model, holding every 10th frame out to
//! measure the result on frames the optimisation never saw.
//!
//! cargo run --release --example refine_with_colmap -- runs/scene capture/sparse_txt capture/images
//!
//! Writes `runs/example_refine/`: `splat.ply`, `cameras.json` and `heldout_<frame>.png`
//! (photo | before | after).

use std::path::PathBuf;

use anyhow::{Result, ensure};

use instantsplat_rs::refine::{RefineSettings, run_refine};

const OUT: &str = "runs/example_refine";
const STEPS: usize = 3000;
const HOLD_OUT_EVERY: usize = 10;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: refine_with_colmap <fit run dir> <colmap sparse_txt dir> <frames dir>"
    );

    run_refine(&RefineSettings {
        run: PathBuf::from(&args[0]),
        colmap: PathBuf::from(&args[1]),
        frames: PathBuf::from(&args[2]),
        out: PathBuf::from(OUT),
        steps: STEPS,
        hold_out_every: HOLD_OUT_EVERY,
        resolution_scale: 1.0,
        log_every: 100,
    })
}
