//! Puts the views' point maps onto each other before they seed Gaussians. The network's
//! per-view point maps do not quite agree (Depth Anything 3 on the duck photos: two groups
//! of views a few percent of the object apart), and each view seeds its own Gaussians, so
//! the splat would start with two copies of the surface and in-between viewpoints would
//! blend them.
//!
//! View 0 stays put (it defines the scene frame). Every other view gets the rigid
//! transform that takes its points onto the other views' surface, and its camera moves
//! with them, so its points still project onto the same pixels of its photo. Two passes:
//! first each view onto the views before it (so that a group of views that agree with
//! each other but not with view 0 cannot hold itself in place), then each view onto all
//! the others. A pass that finds no surface shared with the others (fewer than
//! `MIN_MATCHES` matches: photos of an object's front and back, or point maps the network
//! put far apart) leaves the view where it was, the network's pose if both passes do.
//!
//! The transform is point-to-plane ICP: a sample of each view's confident points, each
//! matched to the nearest sampled point of the other views (a voxel grid, within a radius
//! that shrinks round by round, relative to the scene depth) and pulled onto that point's
//! tangent plane (its normal from the neighboring pixels of its point map). Point-to-point
//! matching would leave a view sliding along a table: the table's points would hold it
//! in place, and only the object's few points would pull. Each step is also pulled back
//! toward the network's pose (`ROTATION_PRIOR_WEIGHT`, `TRANSLATION_PRIOR_WEIGHT`), so
//! that the motions the surface does not pin down stay where the network put them.

use std::collections::HashMap;
use std::fmt;

use nalgebra::{Matrix6, Rotation3, Vector3, Vector6};

use crate::dense_view::DenseView;
use crate::similarity_align::Similarity;

/// About this many points of each view take part in the matching.
const SAMPLES_PER_VIEW: usize = 8_000;
/// The match radius, as a fraction of the scene depth (`scene_depth`), shrinks
/// geometrically from the first to the last over `ROUNDS` rounds.
const FIRST_RADIUS: f64 = 0.08;
const LAST_RADIUS: f64 = 0.01;
const ROUNDS: usize = 4;
const ITERATIONS_PER_ROUND: usize = 4;
/// A point's normal is the cross product of the point map's differences this many pixels
/// to either side, along x and along y.
const NORMAL_PIXELS: usize = 2;
/// How hard each step pulls the rotation and the translation done so far back toward
/// none, relative to how hard the matches pull on the same block on average: the matched
/// surface decides where it pins the view down, the network's pose stays where it does
/// not. A turn about the vertical axis of a roundish object on a floor moves no point off
/// the surface. Measured on the six duck photos (DA3-Base): with both weights at 0.01 the
/// views turn by up to 3.2°, with the rotation held at 1.0 by at most 0.9° (it still
/// levels a tilted floor) and the training loss after 300 steps is lowest (0.0372 against
/// 0.0377, unaligned 0.0391); the translation (the duck's sideways copies) stays free.
/// The price: about half of a rotation the surface does pin down is left undone too.
const ROTATION_PRIOR_WEIGHT: f64 = 1.0;
const TRANSLATION_PRIOR_WEIGHT: f64 = 0.01;
/// Fewer matches than this in some iteration means the view shares no surface with the
/// others.
const MIN_MATCHES: usize = 100;

/// An ICP iteration that found fewer than `MIN_MATCHES` matches.
#[derive(Clone, Debug)]
pub struct NoSharedSurface {
    pub matches: usize,
    pub points: usize,
    /// The match radius, in scene units.
    pub radius: f64,
}

impl fmt::Display for NoSharedSurface {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} of {} points within {:.4} of the target: the point sets share no surface",
            self.matches, self.points, self.radius
        )
    }
}

/// A sampled point and its surface normal (unit, either orientation).
#[derive(Clone, Copy, Debug)]
struct Surfel {
    point: Vector3<f64>,
    normal: Vector3<f64>,
}

impl Surfel {
    fn moved(&self, transform: &Similarity) -> Surfel {
        Surfel {
            point: transform.apply_point(&self.point),
            normal: transform.rotation * self.normal,
        }
    }
}

/// What the alignment did to one view (not view 0).
#[derive(Clone, Debug)]
pub struct ViewAlignment {
    pub view: usize,
    /// Applied to the view's points and camera.
    pub transform: Similarity,
    /// How far the camera center moved, as a fraction of the scene depth.
    pub camera_moved: f64,
    /// Median distance from the view's sampled points to the nearest sampled point of the
    /// other (aligned) views, as a fraction of the scene depth, before and after the
    /// view's own transform; points with no neighbor within `FIRST_RADIUS` are left out,
    /// None when no point has one.
    pub median_distance_before: Option<f64>,
    pub median_distance_after: Option<f64>,
    /// Points of the view counted in `median_distance_after`.
    pub matches: usize,
    /// The passes (of two) that left the view where it was, finding no shared surface.
    pub passes_without_shared_surface: Vec<NoSharedSurface>,
}

fn percent_or_none(fraction: Option<f64>) -> String {
    fraction.map_or("none".to_string(), |fraction| format!("{:.2}%", 100.0 * fraction))
}

impl fmt::Display for ViewAlignment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "view {}: rotated {:.2}°, camera moved {:.2}% of the scene depth, median match distance {} -> {} ({} matches)",
            self.view,
            self.transform.rotation_degrees(),
            100.0 * self.camera_moved,
            percent_or_none(self.median_distance_before),
            percent_or_none(self.median_distance_after),
            self.matches
        )?;
        for skipped in &self.passes_without_shared_surface {
            write!(f, "; a pass left it where it was: {skipped}")?;
        }
        Ok(())
    }
}

/// The surfels of a set, bucketed in cubes of twice the match radius: the ball of that
/// radius around any point lies within the 2x2x2 cubes nearest to it.
struct VoxelGrid<'a> {
    surfels: &'a [Surfel],
    cell: f64,
    cells: HashMap<[i32; 3], Vec<u32>>,
}

impl<'a> VoxelGrid<'a> {
    fn new(surfels: &'a [Surfel], radius: f64) -> VoxelGrid<'a> {
        assert!(radius > 0.0, "match radius {radius}");
        let cell = 2.0 * radius;
        let mut cells: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (index, surfel) in surfels.iter().enumerate() {
            cells
                .entry([0, 1, 2].map(|axis| (surfel.point[axis] / cell).floor() as i32))
                .or_default()
                .push(index as u32);
        }
        VoxelGrid { surfels, cell, cells }
    }

    /// The nearest surfel within the radius and its distance.
    fn nearest(&self, point: &Vector3<f64>) -> Option<(&Surfel, f64)> {
        let radius_squared = (self.cell / 2.0).powi(2);
        // Per axis, the cube holding the point and the neighbor on the point's side.
        let spans = [0, 1, 2].map(|axis| {
            let position = point[axis] / self.cell;
            let cube = position.floor();
            let neighbor = if position - cube < 0.5 { cube - 1.0 } else { cube + 1.0 };
            [cube as i32, neighbor as i32]
        });
        let mut best: Option<(usize, f64)> = None;
        for x in spans[0] {
            for y in spans[1] {
                for z in spans[2] {
                    let Some(indices) = self.cells.get(&[x, y, z]) else {
                        continue;
                    };
                    for &index in indices {
                        let distance_squared = (self.surfels[index as usize].point - point).norm_squared();
                        if distance_squared <= radius_squared && best.is_none_or(|(_, d)| distance_squared < d) {
                            best = Some((index as usize, distance_squared));
                        }
                    }
                }
            }
        }
        best.map(|(index, distance_squared)| (&self.surfels[index], distance_squared.sqrt()))
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    assert!(!values.is_empty(), "median of nothing");
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite value"));
    values[values.len() / 2]
}

/// The rigid motion so far, as what the prior pulls back to zero: the rotation vector and
/// the move of the points' centroid.
fn motion_so_far(transform: &Similarity, centroid: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let rotation = Rotation3::from_matrix_unchecked(transform.rotation).scaled_axis();
    (rotation, transform.apply_point(centroid) - centroid)
}

/// The next small rigid motion, a rotation about `center` (the matched points' centroid)
/// and a translation, that best puts the matched points onto their targets' tangent
/// planes (to first order), with a pull back toward no motion at all (the prior weights)
/// (`so_far` is the motion already done, `motion_so_far`).
fn point_to_plane_step(matches: &[(Vector3<f64>, &Surfel)], center: &Vector3<f64>, so_far: (Vector3<f64>, Vector3<f64>)) -> Similarity {
    let mut normal_matrix = Matrix6::zeros();
    let mut right_side = Vector6::zeros();
    for (point, target) in matches {
        let n = target.normal;
        let row = Vector6::from_iterator((point - center).cross(&n).iter().chain(n.iter()).copied());
        normal_matrix += row * row.transpose();
        right_side += row * -(point - target.point).dot(&n);
    }
    // The prior, per block (rotation, translation), relative to that block's mean curvature.
    let (rotation_so_far, move_so_far) = so_far;
    for (block, so_far, prior_weight) in [
        (0, rotation_so_far, ROTATION_PRIOR_WEIGHT),
        (3, move_so_far, TRANSLATION_PRIOR_WEIGHT),
    ] {
        let weight = prior_weight * (0..3).map(|i| normal_matrix[(block + i, block + i)]).sum::<f64>() / 3.0;
        for i in 0..3 {
            normal_matrix[(block + i, block + i)] += weight;
            right_side[block + i] -= weight * so_far[i];
        }
    }
    let solution = normal_matrix
        .cholesky()
        .expect("positive definite with the prior")
        .solve(&right_side);
    let rotation = Rotation3::new(Vector3::new(solution[0], solution[1], solution[2])).into_inner();
    Similarity {
        scale: 1.0,
        rotation,
        translation: center + Vector3::new(solution[3], solution[4], solution[5]) - rotation * center,
    }
}

/// The rigid transform taking the points of `from` onto the surface of `to`, starting
/// from `start`: point-to-plane ICP with the match radius shrinking from `FIRST_RADIUS`
/// to `LAST_RADIUS` times `extent`, pulled toward no motion (the prior weights). Fails
/// when an iteration finds fewer than `MIN_MATCHES` matches.
fn point_to_plane_icp(from: &[Surfel], to: &[Surfel], start: &Similarity, extent: f64) -> Result<Similarity, NoSharedSurface> {
    assert!(extent > 0.0, "extent {extent}");
    let centroid = from.iter().map(|surfel| surfel.point).sum::<Vector3<f64>>() / from.len() as f64;
    let mut transform = start.clone();
    for round in 0..ROUNDS {
        let radius = extent * FIRST_RADIUS * (LAST_RADIUS / FIRST_RADIUS).powf(round as f64 / (ROUNDS - 1) as f64);
        let grid = VoxelGrid::new(to, radius);
        for _ in 0..ITERATIONS_PER_ROUND {
            let matches: Vec<(Vector3<f64>, &Surfel)> = from
                .iter()
                .filter_map(|surfel| {
                    let point = transform.apply_point(&surfel.point);
                    grid.nearest(&point).map(|(target, _)| (point, target))
                })
                .collect();
            if matches.len() < MIN_MATCHES {
                return Err(NoSharedSurface {
                    matches: matches.len(),
                    points: from.len(),
                    radius,
                });
            }
            let center = matches.iter().map(|(point, _)| point).sum::<Vector3<f64>>() / matches.len() as f64;
            let step = point_to_plane_step(&matches, &center, motion_so_far(&transform, &centroid));
            transform = step.after(&transform);
        }
    }
    Ok(transform)
}

/// Distances from `from` (moved by `transform`) to their nearest neighbor in `grid`, for
/// those that have one.
fn match_distances(from: &[Surfel], transform: &Similarity, grid: &VoxelGrid) -> Vec<f64> {
    from.iter()
        .filter_map(|surfel| grid.nearest(&transform.apply_point(&surfel.point)).map(|(_, distance)| distance))
        .collect()
}

impl DenseView {
    fn point(&self, x: usize, y: usize) -> Vector3<f64> {
        let [px, py, pz] = self.points[y * self.width + x];
        Vector3::new(px as f64, py as f64, pz as f64)
    }

    fn has_point(&self, x: usize, y: usize) -> bool {
        let i = y * self.width + x;
        self.foreground[i] && self.conf[i] > 0.0
    }

    /// The normal from the point map around pixel (x, y); None near the edges of the
    /// image or of the kept points.
    fn normal(&self, x: usize, y: usize) -> Option<Vector3<f64>> {
        let d = NORMAL_PIXELS;
        if x < d || y < d || x + d >= self.width || y + d >= self.height {
            return None;
        }
        let neighbors = [(x - d, y), (x + d, y), (x, y - d), (x, y + d)];
        if !neighbors.iter().all(|&(nx, ny)| self.has_point(nx, ny)) {
            return None;
        }
        let along_x = self.point(x + d, y) - self.point(x - d, y);
        let along_y = self.point(x, y + d) - self.point(x, y - d);
        along_x.cross(&along_y).try_normalize(1e-12)
    }

    /// About `SAMPLES_PER_VIEW` of the view's points at or above its median confidence,
    /// on a regular pixel grid, with their normals.
    fn surfels(&self) -> Vec<Surfel> {
        let confidences: Vec<f64> = (0..self.pixel_count())
            .filter(|&i| self.foreground[i] && self.conf[i] > 0.0)
            .map(|i| self.conf[i] as f64)
            .collect();
        assert!(!confidences.is_empty(), "a view without a single point");
        let kept_count = confidences.len();
        let min_conf = median(confidences) as f32;
        let stride = ((kept_count as f64 / 2.0 / SAMPLES_PER_VIEW as f64).sqrt().round() as usize).max(1);
        let mut surfels = Vec::new();
        for y in (0..self.height).step_by(stride) {
            for x in (0..self.width).step_by(stride) {
                if !self.has_point(x, y) || self.conf[y * self.width + x] < min_conf {
                    continue;
                }
                if let Some(normal) = self.normal(x, y) {
                    surfels.push(Surfel {
                        point: self.point(x, y),
                        normal,
                    });
                }
            }
        }
        surfels
    }
}

/// The median depth of `surfels` in `view`'s camera.
fn scene_depth(view: &DenseView, surfels: &[Surfel]) -> f64 {
    let (rotation, translation) = view.camera.world_to_camera();
    let depth = median(surfels.iter().map(|s| (rotation * s.point + translation).z).collect());
    assert!(depth > 0.0, "median depth {depth} in view 0");
    depth
}

/// The surfels of `views`, each moved by its view's transform.
fn moved_surfels(surfels: &[Vec<Surfel>], transforms: &[Similarity], views: impl Iterator<Item = usize>) -> Vec<Surfel> {
    views
        .flat_map(|view| surfels[view].iter().map(move |surfel| surfel.moved(&transforms[view])))
        .collect()
}

/// The rigid transform that takes the points of the views `from` (together) onto the
/// surface of the views `to`, the match radii relative to `extent` (scene units). For
/// measuring how far apart two groups of views put a surface.
pub fn rigid_transform_between_groups(from: &[&DenseView], to: &[&DenseView], extent: f64) -> Result<Similarity, NoSharedSurface> {
    let from: Vec<Surfel> = from.iter().flat_map(|view| view.surfels()).collect();
    let to: Vec<Surfel> = to.iter().flat_map(|view| view.surfels()).collect();
    point_to_plane_icp(&from, &to, &Similarity::identity(), extent)
}

/// Moves every view but view 0 (points and camera) onto the others; what was done, per
/// view. A pass that finds no surface shared with the others leaves the view where it
/// was. Panics with fewer than two views.
pub fn align_point_maps(views: &mut [DenseView]) -> Vec<ViewAlignment> {
    assert!(views.len() >= 2, "{} views to align, need at least 2", views.len());
    let surfels: Vec<Vec<Surfel>> = views.iter().map(DenseView::surfels).collect();
    let depth = scene_depth(&views[0], &surfels[0]);
    let mut transforms = vec![Similarity::identity(); views.len()];
    let mut passes_without_shared_surface: Vec<Vec<NoSharedSurface>> = vec![Vec::new(); views.len()];
    let others = |view: usize| (0..views.len()).filter(move |&other| other != view);
    for pass in 0..2 {
        for view in 1..views.len() {
            let target = match pass {
                0 => moved_surfels(&surfels, &transforms, 0..view),
                _ => moved_surfels(&surfels, &transforms, others(view)),
            };
            match point_to_plane_icp(&surfels[view], &target, &transforms[view], depth) {
                Ok(transform) => transforms[view] = transform,
                Err(no_shared_surface) => passes_without_shared_surface[view].push(no_shared_surface),
            }
        }
    }

    let alignments = (1..views.len())
        .map(|view| {
            let target = moved_surfels(&surfels, &transforms, others(view));
            let grid = VoxelGrid::new(&target, FIRST_RADIUS * depth);
            let before = match_distances(&surfels[view], &Similarity::identity(), &grid);
            let after = match_distances(&surfels[view], &transforms[view], &grid);
            let center = views[view].camera.translation;
            let median_of_depth = |distances: Vec<f64>| (!distances.is_empty()).then(|| median(distances) / depth);
            ViewAlignment {
                view,
                transform: transforms[view].clone(),
                camera_moved: (transforms[view].apply_point(&center) - center).norm() / depth,
                median_distance_before: median_of_depth(before),
                matches: after.len(),
                median_distance_after: median_of_depth(after),
                passes_without_shared_surface: std::mem::take(&mut passes_without_shared_surface[view]),
            }
        })
        .collect();

    for (view, transform) in views.iter_mut().zip(&transforms).skip(1) {
        for point in &mut view.points {
            let moved = transform.apply_point(&Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64));
            *point = [moved.x as f32, moved.y as f32, moved.z as f32];
        }
        view.camera = transform.apply_camera(&view.camera);
    }
    alignments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use nalgebra::Matrix3;

    /// A table at depth 2 seen from above with two bumps on it (so that no rotation or
    /// sideways move maps it onto itself), sampled on a grid of `size` points offset by
    /// `offset` grid steps.
    fn surface_view(size: usize, offset: f64) -> DenseView {
        let height = |x: f64, y: f64| {
            0.4 * (-((x - 0.2).powi(2) + y.powi(2)) / 0.05).exp() + 0.25 * (-((x + 0.4).powi(2) + (y - 0.4).powi(2)) / 0.02).exp()
        };
        let step = 2.0 / size as f64;
        let mut points = Vec::new();
        for j in 0..size {
            for i in 0..size {
                let (x, y) = (-1.0 + (i as f64 + offset) * step, -1.0 + (j as f64 + offset) * step);
                points.push([x as f32, y as f32, (2.0 - height(x, y)) as f32]);
            }
        }
        let count = size * size;
        DenseView {
            width: size,
            height: size,
            points,
            conf: vec![2.0; count],
            foreground: vec![true; count],
            rgb: vec![0.5; count * 3],
            camera: PinholeCamera {
                rotation: Matrix3::identity(),
                translation: Vector3::zeros(),
                focal: [size as f64; 2],
                principal_point: [size as f64 / 2.0; 2],
                width: size as u32,
                height: size as u32,
            },
        }
    }

    fn moved(view: DenseView, transform: &Similarity) -> DenseView {
        let points = view
            .points
            .iter()
            .map(|p| {
                let q = transform.apply_point(&Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64));
                [q.x as f32, q.y as f32, q.z as f32]
            })
            .collect();
        DenseView {
            points,
            camera: transform.apply_camera(&view.camera),
            ..view
        }
    }

    #[test]
    fn recovers_a_known_shift_of_one_view() {
        // A shift of 4% of the depth and a slight turn (0.17°), which the rotation prior
        // mostly leaves.
        let perturbation = Similarity {
            scale: 1.0,
            rotation: Rotation3::from_euler_angles(0.002, -0.001, 0.002).into_inner(),
            translation: Vector3::new(0.06, -0.04, 0.03),
        };
        let mut views = vec![
            surface_view(120, 0.0),
            moved(surface_view(120, 0.5), &perturbation),
            surface_view(120, 0.25),
        ];
        let view_0_points = views[0].points.clone();
        let alignments = align_point_maps(&mut views);

        assert_eq!(views[0].points, view_0_points, "view 0 defines the frame");
        let perturbed = &alignments[0];
        assert!(perturbed.median_distance_before.expect("matches before") > 0.01, "{perturbed}");
        // The samples of two views are different points: half a grid step (0.6% of the
        // depth) apart at best.
        assert!(perturbed.median_distance_after.expect("matches after") < 0.006, "{perturbed}");
        assert!(perturbed.passes_without_shared_surface.is_empty(), "{perturbed}");
        // Undone: the camera is back where it was, with its points.
        let camera = &views[1].camera;
        let rotation_error = Rotation3::from_matrix(&camera.rotation).angle().to_degrees();
        assert!(
            rotation_error < 0.2,
            "camera rotated {rotation_error}° after alignment ({perturbed})"
        );
        // 0.078 off before; what is left goes with the turn left.
        assert!(camera.translation.norm() < 0.01, "camera at {} after alignment", camera.translation);
        // The unperturbed view stays where it was.
        assert!(alignments[1].camera_moved < 0.001, "{}", alignments[1]);
    }

    #[test]
    fn the_points_still_project_onto_their_pixels() {
        let perturbation = Similarity {
            scale: 1.0,
            rotation: Rotation3::from_euler_angles(0.0, 0.02, 0.0).into_inner(),
            translation: Vector3::new(0.03, 0.0, 0.0),
        };
        let mut views = vec![surface_view(80, 0.0), moved(surface_view(80, 0.5), &perturbation)];
        let project = |view: &DenseView, index: usize| {
            let (rotation, translation) = view.camera.world_to_camera();
            let p = view.points[index];
            let q = rotation * Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64) + translation;
            [q.x / q.z, q.y / q.z]
        };
        let before: Vec<[f64; 2]> = (0..views[1].pixel_count()).map(|i| project(&views[1], i)).collect();
        align_point_maps(&mut views);
        for (index, [x, y]) in before.iter().enumerate() {
            let [ax, ay] = project(&views[1], index);
            assert!((x - ax).abs() < 1e-5 && (y - ay).abs() < 1e-5, "pixel {index} moved");
        }
    }

    #[test]
    fn leaves_views_far_apart_where_they_are() {
        let far = Similarity {
            translation: Vector3::new(5.0, 0.0, 0.0),
            ..Similarity::identity()
        };
        let mut views = vec![surface_view(40, 0.0), moved(surface_view(40, 0.0), &far)];
        let far_points = views[1].points.clone();
        let alignments = align_point_maps(&mut views);

        assert_eq!(views[1].points, far_points);
        let far_view = &alignments[0];
        assert_eq!(far_view.passes_without_shared_surface.len(), 2, "{far_view}");
        assert_eq!(
            (far_view.median_distance_before, far_view.median_distance_after, far_view.matches),
            (None, None, 0)
        );
    }
}
