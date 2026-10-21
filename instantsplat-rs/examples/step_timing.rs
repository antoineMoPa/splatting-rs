//! Times training steps natively on a `fit` run (splat.ply, cameras.json and the
//! input_view{i}.png it was trained on), the same scene the browser demo loads by
//! default, to compare with the browser's steps: fixed poses (nothing read back until
//! the end) and refined poses (the camera gradient read back every step).
//!
//! cargo run --release --example step_timing -- web/example

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use instantsplat_rs::cameras_json::read_cameras_json;
use instantsplat_rs::gpu::Gpu;
use instantsplat_rs::optimizer::{LEARNING_RATES_SHORT_RUN, position_rate_factor};
use instantsplat_rs::run_output::CAMERAS_FILE;
use instantsplat_rs::splat_ply::read_splat_ply;
use instantsplat_rs::training::{CameraPoses, Trainer};
use instantsplat_rs::training_frames::load_training_view;

const POSES: [(&str, CameraPoses); 2] = [("fixed", CameraPoses::Fixed), ("refined", CameraPoses::Refined)];

#[derive(Parser)]
struct Args {
    run: PathBuf,
    #[arg(long, default_value_t = 300)]
    steps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cloud = read_splat_ply(&args.run.join("splat.ply"))?;
    let cameras = read_cameras_json(&args.run.join(CAMERAS_FILE))?;
    let views = cameras
        .iter()
        .enumerate()
        .map(|(i, (_, camera))| load_training_view(&args.run.join(format!("input_view{i}.png")), camera, (camera.width, camera.height)))
        .collect::<Result<Vec<_>>>()?;
    println!(
        "{} Gaussians, {} views of {}x{}",
        cloud.len(),
        views.len(),
        views[0].width,
        views[0].height
    );
    let gpu = Gpu::request();
    for (label, poses) in POSES {
        let mut trainer = Trainer::new(&gpu, &cloud, &views, LEARNING_RATES_SHORT_RUN, poses);
        // Pipelines and first-use allocations out of the way.
        for step in 0..10 {
            trainer.step(step % views.len(), 1.0);
        }
        trainer.loss(0);
        let started = Instant::now();
        let mut submitted = std::time::Duration::ZERO;
        for step in 0..args.steps {
            trainer.step(step % views.len(), position_rate_factor(step, args.steps));
            if step + 1 == args.steps {
                submitted = started.elapsed();
            }
        }
        trainer.loss(0);
        let elapsed = started.elapsed();
        println!(
            "{label}: {} steps in {:.2?} ({:.1} ms/step; all submitted after {:.2?})",
            args.steps,
            elapsed,
            elapsed.as_secs_f64() * 1000.0 / args.steps as f64,
            submitted
        );
    }
    Ok(())
}
