//! Training with and without densification (`densify.rs`) on the same photos, judged on what
//! the training loss cannot see:
//! - leave one out: per `--hold-out` view, the Gaussians are seeded from and trained on the
//!   other views only, then the held-out photo's loss (as the trainer computes it) and mean
//!   absolute error over its foreground are taken through its network camera (poses are
//!   not refined, so that the held-out camera stays in the same frame);
//! - between views: trained on every view, the splat rendered from cameras interpolated
//!   between views `--between` (rotation slerped, center lerped), side by side in one PNG
//!   per variant: an object that slides or doubles between viewpoints shows there.
//!
//! The network runs once, the views' point maps are aligned as the apps do, then each
//! variant trains with the apps' learning rates for `--steps`.
//!
//! cargo run --release --example densify_eval -- --network da3-base --out runs/densify_eval ../.example/*.png

use std::path::PathBuf;

use anyhow::{Result, ensure};
use candle_core::DType;
use clap::Parser;
use nalgebra::UnitQuaternion;

use instantsplat_rs::camera::PinholeCamera;
use instantsplat_rs::dense_view::DenseView;
use instantsplat_rs::densify::{DENSIFY_SHORT_RUN, DensifySettings};
use instantsplat_rs::gpu::Gpu;
use instantsplat_rs::init_from_points::{INIT_OPACITY, InitSettings, gaussians_from_dense_views};
use instantsplat_rs::network::{Network, NetworkSettings, dense_views_from_images};
use instantsplat_rs::optimizer::{LEARNING_RATES_3DGS, LEARNING_RATES_SHORT_RUN};
use instantsplat_rs::point_map_alignment::align_point_maps;
use instantsplat_rs::training::{CameraPoses, TrainReport, TrainSettings, Trainer};
use instantsplat_rs::view_error::view_error;

/// Interpolation parameters of the cameras between the two `--between` views.
const BETWEEN_STEPS: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = Network::Da3Base)]
    network: Network,
    #[arg(long, default_value_t = 300)]
    steps: usize,
    /// Init stride without densification, as the apps.
    #[arg(long, default_value_t = 2)]
    stride: usize,
    /// Init stride with densification.
    #[arg(long, default_value_t = 2)]
    densify_stride: usize,
    /// Views to leave out one at a time (none: only the between-views renders).
    #[arg(long, value_delimiter = ',')]
    hold_out: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_values_t = [2, 3])]
    between: Vec<usize>,
    /// Overrides of `DENSIFY_SHORT_RUN`.
    #[arg(long)]
    gradient_threshold: Option<f32>,
    #[arg(long)]
    interval: Option<usize>,
    #[arg(long)]
    start: Option<usize>,
    #[arg(long)]
    stop_fraction: Option<f32>,
    #[arg(long)]
    max_growth: Option<f32>,
    #[arg(long)]
    split_scale_fraction: Option<f32>,
    #[arg(long)]
    opacity_reset_interval: Option<usize>,
    #[arg(long)]
    prune_opacity: Option<f32>,
    /// Train the densified variant only (the fixed count's numbers do not change).
    #[arg(long)]
    densified_only: bool,
    #[arg(long)]
    out: PathBuf,
    images: Vec<PathBuf>,
}

/// `views` seeded at `stride` and trained `steps` steps with fixed poses, densified by
/// `densify` if any.
fn train(gpu: &Gpu, views: &[DenseView], stride: usize, steps: usize, densify: Option<DensifySettings>) -> TrainReport {
    let cloud = gaussians_from_dense_views(
        views,
        InitSettings {
            stride,
            min_conf: 1.0,
            opacity: INIT_OPACITY,
        },
    );
    let trainer = Trainer::new(gpu, &cloud, views, LEARNING_RATES_SHORT_RUN, CameraPoses::Fixed);
    let trainer = match densify {
        Some(settings) => trainer.with_densification(settings),
        None => trainer,
    };
    trainer.optimize(&TrainSettings {
        steps,
        log_every: steps,
        learning_rates: LEARNING_RATES_SHORT_RUN,
    })
}

/// `a` turned and moved a fraction `t` of the way to `b`.
fn interpolate(a: &PinholeCamera, b: &PinholeCamera, t: f64) -> PinholeCamera {
    let ra = UnitQuaternion::from_matrix(&a.rotation);
    let rb = UnitQuaternion::from_matrix(&b.rotation);
    PinholeCamera {
        rotation: ra.slerp(&rb, t).to_rotation_matrix().into_inner(),
        translation: a.translation.lerp(&b.translation, t),
        ..a.clone()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.between.len() == 2, "--between takes two views");
    std::fs::create_dir_all(&args.out)?;
    let (mut views, _) = dense_views_from_images(
        &args.images,
        &NetworkSettings {
            network: args.network,
            focal_35mm: None,
            dtype: DType::F16,
            weights: None,
            cpu: false,
        },
    )?;
    align_point_maps(&mut views);
    for &view in args.hold_out.iter().chain(&args.between) {
        ensure!(view < views.len(), "view {view} of {}", views.len());
    }
    let densify = DensifySettings {
        gradient_threshold: args.gradient_threshold.unwrap_or(DENSIFY_SHORT_RUN.gradient_threshold),
        interval: args.interval.unwrap_or(DENSIFY_SHORT_RUN.interval),
        start: args.start.unwrap_or(DENSIFY_SHORT_RUN.start),
        stop_fraction: args.stop_fraction.unwrap_or(DENSIFY_SHORT_RUN.stop_fraction),
        max_growth: args.max_growth.unwrap_or(DENSIFY_SHORT_RUN.max_growth),
        split_scale_fraction: args.split_scale_fraction.unwrap_or(DENSIFY_SHORT_RUN.split_scale_fraction),
        opacity_reset_interval: args.opacity_reset_interval.or(DENSIFY_SHORT_RUN.opacity_reset_interval),
        prune_opacity: args.prune_opacity.unwrap_or(DENSIFY_SHORT_RUN.prune_opacity),
        ..DENSIFY_SHORT_RUN
    };
    println!("{densify:?}");
    let all_variants = [
        ("fixed count", args.stride, None),
        ("densified", args.densify_stride, Some(densify)),
    ];
    let variants: Vec<(&str, usize, Option<DensifySettings>)> = all_variants
        .into_iter()
        .filter(|(_, _, settings)| settings.is_some() || !args.densified_only)
        .collect();
    let gpu = Gpu::request();

    let mut held_out_rows = Vec::new();
    for &held_out in &args.hold_out {
        let training: Vec<DenseView> = views
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != held_out)
            .map(|(_, v)| v.clone())
            .collect();
        for &(name, stride, settings) in &variants {
            let report = train(&gpu, &training, stride, args.steps, settings);
            let judge = Trainer::new(&gpu, &report.cloud, &views, LEARNING_RATES_3DGS, CameraPoses::Fixed);
            let loss = judge.loss(held_out);
            let error = view_error(&judge.render(held_out), &views[held_out]);
            held_out_rows.push((
                held_out,
                name,
                report.final_loss.total,
                loss.total,
                error.foreground,
                report.cloud.len(),
            ));
            judge
                .render(held_out)
                .to_rgb8()
                .save(args.out.join(format!("heldout{held_out}_{}.png", name.replace(' ', "_"))))?;
        }
    }

    let [a, b] = [args.between[0], args.between[1]];
    let mut between_rows = Vec::new();
    for &(name, stride, settings) in &variants {
        let report = train(&gpu, &views, stride, args.steps, settings);
        let judge = Trainer::new(&gpu, &report.cloud, &views, LEARNING_RATES_3DGS, CameraPoses::Fixed);
        let (width, height) = (views[a].width as u32, views[a].height as u32);
        let mut strip = image::RgbImage::new(width * BETWEEN_STEPS.len() as u32, height);
        for (k, &t) in BETWEEN_STEPS.iter().enumerate() {
            let render = judge.render_camera(&interpolate(&views[a].camera, &views[b].camera, t)).to_rgb8();
            image::imageops::replace(&mut strip, &render, (k as u32 * width) as i64, 0);
        }
        let path = args.out.join(format!("between{a}_{b}_{}.png", name.replace(' ', "_")));
        strip.save(&path)?;
        between_rows.push((
            name,
            report.final_loss.total,
            report.densification_summary(),
            report.prune.kept,
            report.elapsed,
            path,
        ));
    }

    println!("\nleave one out ({} steps):", args.steps);
    for (held_out, name, train_loss, loss, error, count) in &held_out_rows {
        println!(
            "  view {held_out} {name:>11}: training loss {train_loss:.4}, held-out loss {loss:.4}, foreground error {error:.4}, {count} Gaussians"
        );
    }
    for (name, ..) in &variants {
        let rows: Vec<_> = held_out_rows.iter().filter(|row| row.1 == *name).collect();
        if rows.is_empty() {
            continue;
        }
        let mean = |pick: fn(&&(usize, &str, f32, f32, f32, usize)) -> f32| rows.iter().map(pick).sum::<f32>() / rows.len() as f32;
        println!(
            "  mean {name:>11}: training loss {:.4}, held-out loss {:.4}, foreground error {:.4}",
            mean(|row| row.2),
            mean(|row| row.3),
            mean(|row| row.4)
        );
    }
    println!("\nall views:");
    for (name, loss, summary, kept, elapsed, path) in between_rows {
        println!(
            "  {name:>11}: loss {loss:.4}, {summary}, kept {kept}, {elapsed:.1?}; {}",
            path.display()
        );
    }
    Ok(())
}
