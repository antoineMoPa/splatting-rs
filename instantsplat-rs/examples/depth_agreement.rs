//! How well the network's per-view point maps agree with each other: every foreground
//! point of view a is projected into view b and its depth there compared with the depth
//! of b's own point at that pixel. Printed per pair as percentiles of (a's depth - b's
//! depth), in percent of the object's size (the diagonal of the foreground points' bounding
//! box): negative means a puts the surface in front of where b sees it. Each view's points
//! seed their own Gaussians, so a disagreement here is a second copy of the surface. The
//! last column, the median ratio of a's depth to b's, would sit away from 1 for a view
//! whose depth is off by a scale. `--align` measures after `point_map_alignment`.
//!
//! cargo run --release --example depth_agreement -- --network da3-base [--align] ../.example/*.png

use std::path::PathBuf;

use anyhow::Result;
use candle_core::DType;
use clap::Parser;
use nalgebra::Vector3;

use instantsplat_rs::dense_view::DenseView;
use instantsplat_rs::network::{Network, NetworkSettings, dense_views_from_images};
use instantsplat_rs::point_map_alignment::align_point_maps;

/// Every this many pixels along each axis, as the apps' Gaussian init.
const STRIDE: usize = 2;
const PERCENTILES: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = Network::Da3Base)]
    network: Network,
    /// Align the views' point maps first, as the apps do.
    #[arg(long)]
    align: bool,
    images: Vec<PathBuf>,
}

fn kept(view: &DenseView, index: usize) -> bool {
    view.foreground[index] && view.conf[index] > 0.0
}

fn object_size(views: &[DenseView]) -> f64 {
    let mut low = Vector3::repeat(f64::INFINITY);
    let mut high = Vector3::repeat(f64::NEG_INFINITY);
    for view in views {
        for (index, point) in view.points.iter().enumerate() {
            if kept(view, index) {
                let point = Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);
                low = low.inf(&point);
                high = high.sup(&point);
            }
        }
    }
    (high - low).norm()
}

/// (a's depth, b's depth) in b's camera for a's points that land on b's foreground.
fn depth_pairs(a: &DenseView, b: &DenseView) -> Vec<(f64, f64)> {
    let (rotation, translation) = b.camera.world_to_camera();
    let in_b = |point: &[f32; 3]| rotation * Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64) + translation;
    let mut pairs = Vec::new();
    for y in (0..a.height).step_by(STRIDE) {
        for x in (0..a.width).step_by(STRIDE) {
            let index = y * a.width + x;
            if !kept(a, index) {
                continue;
            }
            let point = in_b(&a.points[index]);
            if point.z <= 0.0 {
                continue;
            }
            let u = (b.camera.focal[0] * point.x / point.z + b.camera.principal_point[0]).round();
            let v = (b.camera.focal[1] * point.y / point.z + b.camera.principal_point[1]).round();
            if u < 0.0 || v < 0.0 || u >= b.width as f64 || v >= b.height as f64 {
                continue;
            }
            let b_index = v as usize * b.width + u as usize;
            if kept(b, b_index) {
                pairs.push((point.z, in_b(&b.points[b_index]).z));
            }
        }
    }
    pairs
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|p, q| p.partial_cmp(q).expect("finite depth"));
    values
}

fn main() -> Result<()> {
    let args = Args::parse();
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
    if args.align {
        for alignment in align_point_maps(&mut views) {
            println!("{alignment}");
        }
    }
    let size = object_size(&views);
    println!("{} views, object size {size:.4} (scene units)", views.len());
    println!("pair   points  p10 p25 p50 p75 p90 of (a - b) depth, % of object size; median a / b");
    for (a_index, a) in views.iter().enumerate() {
        for (b_index, b) in views.iter().enumerate() {
            if a_index == b_index {
                continue;
            }
            let pairs = depth_pairs(a, b);
            if pairs.is_empty() {
                println!("{a_index}->{b_index}  no overlap");
                continue;
            }
            let differences = sorted(pairs.iter().map(|(a, b)| a - b).collect());
            let ratios = sorted(pairs.iter().map(|(a, b)| a / b).collect());
            let percentiles: Vec<String> = PERCENTILES
                .iter()
                .map(|p| format!("{:+5.1}", 100.0 * differences[(p * (differences.len() - 1) as f64) as usize] / size))
                .collect();
            println!(
                "{a_index}->{b_index}  {:>7}  {}  {:.4}",
                differences.len(),
                percentiles.join(" "),
                ratios[ratios.len() / 2]
            );
        }
    }
    Ok(())
}
