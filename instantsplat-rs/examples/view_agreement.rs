//! How well the views' point maps agree once projected into each other's photos, in the
//! two ways that a splat seeded from them would show: every kept point of view a is
//! projected into view b and
//! - its color compared with b's pixel there (mean absolute difference over rgb, in 0-255
//!   steps): a camera turned or shifted against the others puts the floor's texture in
//!   the wrong place;
//! - for the object's points (picked by color, by default the yellow duck of `.example/`),
//!   away from its outline in a (`EDGE_PIXELS`: the network blends the depths of the
//!   object and of the floor behind it there), the percentage that lands outside b's
//!   object pixels (grown by `MASK_MARGIN` pixels):
//!   every point of the object projects inside its silhouette in every photo, whatever
//!   side of it the photo sees, so points outside are a second copy of the object, off to
//!   the side.
//!
//! Printed per pair, then averaged over the pairs within `--group` and across it (the views
//! in it against the others). Last, the rigid transform (`point_map_alignment`'s ICP)
//! that takes the object points of `--group` onto those of the other views: the move of
//! their centroid, in percent of the object's size (the diagonal of the other views'
//! object points' bounding box). `--align` measures after `point_map_alignment`, as the apps
//! seed their Gaussians.
//!
//! cargo run --release --example view_agreement -- --network da3-base [--align] ../.example/*.png

use std::path::PathBuf;

use anyhow::Result;
use candle_core::DType;
use clap::Parser;
use nalgebra::Vector3;

use instantsplat_rs::dense_view::DenseView;
use instantsplat_rs::network::{Network, NetworkSettings, dense_views_from_images};
use instantsplat_rs::point_map_alignment::{align_point_maps, rigid_transform_between_groups};

/// Every this many pixels along each axis, as the apps' Gaussian init.
const STRIDE: usize = 2;
/// The object's pixels grow by this many pixels, for its edge pixels the color test
/// misses.
const MASK_MARGIN: usize = 2;
/// The ICP's match radii are relative to the object's size times this: the views' object
/// shapes differ by more than a percent of its size (the network flattens it differently
/// from each side), and the last radius (`point_map_alignment` is tuned on the scene depth)
/// would leave too few matches.
const ICP_EXTENT_OVER_OBJECT_SIZE: f64 = 3.0;
/// Object points this close to the object's outline in their own photo are left out.
const EDGE_PIXELS: usize = 4;

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = Network::Da3Base)]
    network: Network,
    /// Align the views' point maps first, as the apps do.
    #[arg(long)]
    align: bool,
    /// Views that Depth Anything 3 places apart from the others (the duck's two high
    /// shots).
    #[arg(long, value_delimiter = ',', default_values_t = [3, 4])]
    group: Vec<usize>,
    images: Vec<PathBuf>,
}

/// The duck: r > 0.6, g > 0.5, b < 0.35.
fn is_object(view: &DenseView, index: usize) -> bool {
    let rgb = &view.rgb[index * 3..index * 3 + 3];
    rgb[0] > 0.6 && rgb[1] > 0.5 && rgb[2] < 0.35
}

fn kept(view: &DenseView, index: usize) -> bool {
    view.foreground[index] && view.conf[index] > 0.0
}

/// The pixel of b that a world point lands on, if inside b's image and in front of it.
fn project(b: &DenseView, point: &[f32; 3]) -> Option<usize> {
    let (rotation, translation) = b.camera.world_to_camera();
    let p = rotation * Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64) + translation;
    if p.z <= 0.0 {
        return None;
    }
    let u = (b.camera.focal[0] * p.x / p.z + b.camera.principal_point[0]).round();
    let v = (b.camera.focal[1] * p.y / p.z + b.camera.principal_point[1]).round();
    (u >= 0.0 && v >= 0.0 && u < b.width as f64 && v < b.height as f64).then(|| v as usize * b.width + u as usize)
}

/// The pixels within `margin` (a square) of pixel `index`.
fn around(view: &DenseView, index: usize, margin: usize) -> impl Iterator<Item = usize> + '_ {
    let (x, y) = (index % view.width, index / view.width);
    (y.saturating_sub(margin)..(y + margin + 1).min(view.height))
        .flat_map(move |ny| (x.saturating_sub(margin)..(x + margin + 1).min(view.width)).map(move |nx| ny * view.width + nx))
}

fn object_pixel(view: &DenseView, index: usize) -> bool {
    kept(view, index) && is_object(view, index)
}

/// Per pixel of the view, whether an object pixel lies within `MASK_MARGIN`.
fn object_mask(view: &DenseView) -> Vec<bool> {
    let mut mask = vec![false; view.pixel_count()];
    for index in (0..view.pixel_count()).filter(|&i| object_pixel(view, i)) {
        for near in around(view, index, MASK_MARGIN) {
            mask[near] = true;
        }
    }
    mask
}

/// An object pixel at least `EDGE_PIXELS` from anything else.
fn inside_object(view: &DenseView, index: usize) -> bool {
    around(view, index, EDGE_PIXELS).all(|near| object_pixel(view, near))
}

struct PairAgreement {
    color_error: f64,
    /// Percent of a's object points landing outside b's object.
    object_outside: f64,
}

fn agreement(a: &DenseView, b: &DenseView) -> PairAgreement {
    let mut color_error = (0.0, 0usize);
    let mask = object_mask(b);
    let mut outside = (0usize, 0usize);
    for y in (0..a.height).step_by(STRIDE) {
        for x in (0..a.width).step_by(STRIDE) {
            let index = y * a.width + x;
            if !kept(a, index) {
                continue;
            }
            let Some(b_index) = project(b, &a.points[index]) else {
                continue;
            };
            let difference: f32 = (0..3).map(|c| (a.rgb[index * 3 + c] - b.rgb[b_index * 3 + c]).abs()).sum();
            color_error = (color_error.0 + difference as f64 * 255.0 / 3.0, color_error.1 + 1);
            if inside_object(a, index) {
                outside = (outside.0 + !mask[b_index] as usize, outside.1 + 1);
            }
        }
    }
    assert!(outside.1 > 0, "none of a's object points lands in b");
    PairAgreement {
        color_error: color_error.0 / color_error.1 as f64,
        object_outside: 100.0 * outside.0 as f64 / outside.1 as f64,
    }
}

/// The view with every pixel but the object's dropped.
fn object_only(view: &DenseView) -> DenseView {
    DenseView {
        foreground: (0..view.pixel_count()).map(|i| object_pixel(view, i)).collect(),
        points: view.points.clone(),
        conf: view.conf.clone(),
        rgb: view.rgb.clone(),
        camera: view.camera.clone(),
        ..*view
    }
}

fn object_points(views: &[&DenseView]) -> Vec<Vector3<f64>> {
    views
        .iter()
        .flat_map(|view| {
            (0..view.pixel_count())
                .filter(|&i| view.foreground[i])
                .map(|i| Vector3::new(view.points[i][0] as f64, view.points[i][1] as f64, view.points[i][2] as f64))
        })
        .collect()
}

/// The move of the group's object onto the other views' object, as `main`'s doc says.
fn print_object_offset(views: &[DenseView], group: &[usize]) {
    let objects: Vec<DenseView> = views.iter().map(object_only).collect();
    let (from, to): (Vec<(usize, &DenseView)>, Vec<(usize, &DenseView)>) =
        objects.iter().enumerate().partition(|(index, _)| group.contains(index));
    let (from, to): (Vec<&DenseView>, Vec<&DenseView>) = (from.into_iter().map(|(_, v)| v).collect(), to.into_iter().map(|(_, v)| v).collect());
    let to_points = object_points(&to);
    let low = to_points.iter().fold(Vector3::repeat(f64::INFINITY), |low, p| low.inf(p));
    let high = to_points.iter().fold(Vector3::repeat(f64::NEG_INFINITY), |high, p| high.sup(p));
    let size = (high - low).norm();
    let from_points = object_points(&from);
    let centroid = from_points.iter().sum::<Vector3<f64>>() / from_points.len() as f64;
    let transform = rigid_transform_between_groups(&from, &to, ICP_EXTENT_OVER_OBJECT_SIZE * size);
    let moved = transform.apply_point(&centroid) - centroid;
    println!(
        "object of views {group:?} onto the others' (ICP): centroid moved {:.2}% of the object's size, rotated {:.2}°",
        100.0 * moved.norm() / size,
        transform.rotation_degrees()
    );
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
    println!("pair  color error (0-255)  object points outside its silhouette (%)");
    let mut within = Vec::new();
    let mut across = Vec::new();
    for (a_index, a) in views.iter().enumerate() {
        for (b_index, b) in views.iter().enumerate() {
            if a_index == b_index {
                continue;
            }
            let pair = agreement(a, b);
            println!("{a_index}->{b_index}  {:6.2}  {:6.2}", pair.color_error, pair.object_outside);
            if args.group.contains(&a_index) == args.group.contains(&b_index) {
                within.push(pair);
            } else {
                across.push(pair);
            }
        }
    }
    let mean = |pairs: &[PairAgreement], value: fn(&PairAgreement) -> f64| pairs.iter().map(value).sum::<f64>() / pairs.len() as f64;
    for (label, pairs) in [("within groups", &within), ("across groups", &across)] {
        println!(
            "{label} ({} pairs): color error {:.2}, object points outside its silhouette {:.2}%",
            pairs.len(),
            mean(pairs, |p| p.color_error),
            mean(pairs, |p| p.object_outside)
        );
    }
    print_object_offset(&views, &args.group);
    Ok(())
}
