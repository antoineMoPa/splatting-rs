//! Stage 3: the joint optimization of all Gaussians against the input images. Each step
//! renders one view, takes the loss against its image and foreground mask (`loss.rs`),
//! runs the backward pass and one Adam update with the scale limits, on the GPU; views
//! are cycled in order. With `CameraPoses::Refined` the same backward pass gives the
//! camera gradient, read back each step for one Adam update of the view's pose
//! (`pose_refinement.rs`). With `with_densification`, the Gaussians are cloned, split and
//! pruned every few steps (`densify.rs`): the parameter, Adam and rasterizer buffers are
//! rebuilt at the new count, so `params` changes. The Gaussians are pruned (`prune.rs`)
//! when the run ends.
//!
//! Everything that reads the GPU back is async so that a browser can await it
//! (`optimize_async`, `step_async`, ...); the blocking `optimize`, `step`, ... are
//! native-only `pollster::block_on` wrappers.

use std::time::Duration;
use web_time::Instant;

use crate::camera::{PinholeCamera, cameras_extent};
use crate::dense_view::DenseView;
use crate::densify::{DensifyReport, DensifySettings, densify};
use crate::gaussians::GaussianCloud;
use crate::gpu::Gpu;
use crate::loss::{LossTerms, ground_truth_rgba};
use crate::optimizer::{Adam, LearningRates, position_rate_factor};
use crate::pose_refinement::{FIXED_VIEW, PoseRefinement};
use crate::prune::{PruneReport, prune};
use crate::rasterizer::{Rasterizer, Rendered};
use crate::shaders::PARAM_STRIDE;

pub struct TrainSettings {
    pub steps: usize,
    /// Print the loss every this many steps (and check the intersection list).
    pub log_every: usize,
    pub learning_rates: LearningRates,
}

/// Whether the views' cameras are optimized along with the Gaussians.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraPoses {
    Fixed,
    /// Every view's pose but view `FIXED_VIEW`, and one focal scale shared by all.
    Refined,
}

pub struct TrainReport {
    /// After pruning.
    pub cloud: GaussianCloud,
    /// The views' cameras at the end: as given for `CameraPoses::Fixed`.
    pub cameras: Vec<PinholeCamera>,
    /// What every focal length was multiplied by (1 for `CameraPoses::Fixed`).
    pub focal_scale: f64,
    pub gaussians_before_pruning: usize,
    pub prune: PruneReport,
    /// One per densification, in order.
    pub densifications: Vec<DensifyReport>,
    /// Since the trainer was created.
    pub elapsed: Duration,
    /// Mean absolute color error of the last logged step's view, in [0, 1] (premultiplied
    /// rgb against the photo on foreground pixels, over all pixels); `last_loss.l1`.
    pub last_mean_error: f32,
    pub last_loss: LossTerms,
    /// Means over every view of the loss terms of the trained Gaussians, before pruning.
    pub final_loss: LossTerms,
    /// (Gaussian, tile) pairs of the last logged step.
    pub last_intersections: u32,
}

impl TrainReport {
    /// The Gaussian count through the densifications, e.g. "densified 120000 -> 150000 ->
    /// 170000 Gaussians", or "not densified".
    pub fn densification_summary(&self) -> String {
        match self.densifications.first() {
            None => "not densified".to_string(),
            Some(first) => {
                let counts: Vec<String> = std::iter::once(first.before)
                    .chain(self.densifications.iter().map(|report| report.after))
                    .map(|count| count.to_string())
                    .collect();
                format!("densified {} Gaussians", counts.join(" -> "))
            }
        }
    }
}

/// The optimizer's state for one scene: parameters on the GPU, the rasterizer sized for
/// the views (it holds their ground truth, so the views themselves are not kept), and
/// Adam; all three are rebuilt when densification changes the count.
pub struct Trainer<'a> {
    gpu: &'a Gpu,
    /// The views' cameras as given.
    initial_cameras: Vec<PinholeCamera>,
    params: wgpu::Buffer,
    ground_truth: wgpu::Buffer,
    rasterizer: Rasterizer,
    adam: Adam,
    poses: Option<PoseRefinement>,
    gaussian_count: usize,
    densify: Option<DensifySettings>,
    /// What training started with: `DensifySettings::max_growth` multiplies it.
    initial_gaussian_count: usize,
    densifications: Vec<DensifyReport>,
    extent: f32,
    learning_rates: LearningRates,
    started: Instant,
    /// Of the last `read_step_async`.
    last_loss: LossTerms,
    last_intersections: u32,
}

impl<'a> Trainer<'a> {
    pub fn new(gpu: &'a Gpu, cloud: &GaussianCloud, views: &[DenseView], learning_rates: LearningRates, camera_poses: CameraPoses) -> Self {
        assert!(!views.is_empty(), "no views");
        let (width, height) = (views[0].width, views[0].height);
        for view in views {
            view.check();
            assert_eq!((view.width, view.height), (width, height), "every view must have the same size");
        }
        let ground_truth: Vec<f32> = views.iter().flat_map(ground_truth_rgba).collect();
        let ground_truth = gpu.storage_buffer_from("ground truth", &ground_truth);
        let params = gpu.storage_buffer_from("params", &cloud.to_packed());
        let rasterizer = Rasterizer::new(gpu, &params, cloud.len(), width as u32, height as u32, &ground_truth, views.len());
        let cameras: Vec<_> = views.iter().map(|view| view.camera.clone()).collect();
        let extent = cameras_extent(&cameras) as f32;
        assert!(
            extent > 0.0,
            "camera extent {extent}: the views' cameras are all at the same position"
        );
        let poses = match camera_poses {
            CameraPoses::Fixed => None,
            CameraPoses::Refined => Some(PoseRefinement::new(&cameras, &learning_rates, extent)),
        };
        let adam = Adam::new(gpu, &params, rasterizer.param_gradients(), cloud.len(), learning_rates, extent);
        Self {
            gpu,
            initial_cameras: cameras,
            params,
            ground_truth,
            rasterizer,
            adam,
            poses,
            gaussian_count: cloud.len(),
            densify: None,
            initial_gaussian_count: cloud.len(),
            densifications: Vec::new(),
            extent,
            learning_rates,
            started: Instant::now(),
            last_loss: LossTerms::default(),
            last_intersections: 0,
        }
    }

    /// Densify and prune by `settings` during `train_step_async` and `optimize_async`.
    pub fn with_densification(mut self, settings: DensifySettings) -> Self {
        self.densify = Some(settings);
        self
    }

    /// Camera extent of the views (`cameras_extent`).
    pub fn extent(&self) -> f32 {
        self.extent
    }

    /// The Gaussians' parameters as they are being optimized (`gaussian_count` of them,
    /// packed as `GaussianCloud::to_packed`): another rasterizer can render them live, and
    /// must be rebuilt over the new buffer after a densification.
    pub fn params(&self) -> &wgpu::Buffer {
        &self.params
    }

    pub fn gaussian_count(&self) -> usize {
        self.gaussian_count
    }

    pub fn view_count(&self) -> usize {
        self.initial_cameras.len()
    }

    /// The camera `view` is rendered with now.
    pub fn camera(&self, view: usize) -> PinholeCamera {
        match &self.poses {
            Some(poses) => poses.camera(view),
            None => self.initial_cameras[view].clone(),
        }
    }

    pub fn cameras(&self) -> Vec<PinholeCamera> {
        (0..self.view_count()).map(|view| self.camera(view)).collect()
    }

    pub fn focal_scale(&self) -> f64 {
        self.poses.as_ref().map_or(1.0, PoseRefinement::focal_scale)
    }

    /// One optimization step on `view`, with the position and camera pose learning rates
    /// at `rate_factor` of their initial values.
    pub async fn step_async(&mut self, view: usize, rate_factor: f32) {
        self.rasterizer.set_camera(self.gpu, &self.camera(view));
        let mut encoder = self.gpu.encoder();
        self.rasterizer.encode_forward(&mut encoder);
        self.rasterizer.encode_loss(&mut encoder, view);
        self.rasterizer.encode_backward(&mut encoder);
        self.adam.encode_step(self.gpu, &mut encoder, rate_factor);
        self.gpu.submit(encoder);
        if let Some(poses) = &mut self.poses {
            let gradient = self.rasterizer.read_camera_gradient_async(self.gpu).await;
            poses.step(view, &gradient, rate_factor);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn step(&mut self, view: usize, rate_factor: f32) {
        pollster::block_on(self.step_async(view, rate_factor))
    }

    /// Step `step` (0-based) of a run of `steps`: the densification due after the previous
    /// step if any, then the step, views cycled in order, the position rate decayed
    /// (`position_rate_factor`). Densifying before the step leaves the step's loss and
    /// intersections in the rasterizer for `read_step_async`. The densification's report
    /// when there was one: `params` is then a new buffer.
    pub async fn train_step_async(&mut self, step: usize, steps: usize) -> Option<DensifyReport> {
        let due = self.densify.filter(|settings| step > 0 && settings.densifies_after(step, steps));
        let report = match due {
            Some(settings) => Some(self.densify_async(&settings).await),
            None => None,
        };
        self.step_async(step % self.view_count(), position_rate_factor(step, steps)).await;
        report
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn train_step(&mut self, step: usize, steps: usize) -> Option<DensifyReport> {
        pollster::block_on(self.train_step_async(step, steps))
    }

    /// Reads the Gaussians, Adam's moments and the densification statistics back,
    /// densifies, and rebuilds the buffers at the new count (the statistics start over).
    async fn densify_async(&mut self, settings: &DensifySettings) -> DensifyReport {
        let params = self
            .gpu
            .read_back_async::<f32>(&self.params, self.gaussian_count * PARAM_STRIDE)
            .await;
        let moments = self.adam.read_moments_async(self.gpu).await;
        let stats = self.rasterizer.read_densify_stats_async(self.gpu).await;
        let max_count = (self.initial_gaussian_count as f32 * settings.max_growth) as usize;
        let seed = self.densifications.len() as u64;
        let reset_opacity = settings.resets_opacity_after(self.adam.steps_taken() as usize);
        let densified = densify(&params, &moments, &stats, self.extent, max_count, settings, reset_opacity, seed);
        let count = densified.report.after;
        self.params = self.gpu.storage_buffer_from("params", &densified.params);
        self.rasterizer = Rasterizer::new(
            self.gpu,
            &self.params,
            count,
            self.rasterizer.width,
            self.rasterizer.height,
            &self.ground_truth,
            self.view_count(),
        );
        self.adam = self
            .adam
            .resumed(self.gpu, &self.params, self.rasterizer.param_gradients(), count, &densified.moments);
        self.gaussian_count = count;
        self.densifications.push(densified.report);
        densified.report
    }

    /// After a step, its loss and intersection list still in the buffers: reads them back,
    /// keeps them for the report, and returns the loss.
    pub async fn read_step_async(&mut self) -> LossTerms {
        self.last_intersections = self.rasterizer.read_intersections_async(self.gpu).await.count;
        self.last_loss = self.rasterizer.read_loss_async(self.gpu).await;
        self.last_loss
    }

    /// The camera extent and the learning rates that depend on it.
    fn print_settings(&self) {
        let rates = &self.learning_rates;
        println!(
            "camera extent {:.3}: position learning rate {:.2e}",
            self.extent,
            rates.position * self.extent
        );
        if self.poses.is_some() {
            println!(
                "refining the poses of {} views (view {FIXED_VIEW} fixed): rotation rate {:.1e}, translation rate {:.2e}, focal rate {:.1e}",
                self.view_count(),
                rates.camera_rotation,
                rates.camera_translation * self.extent,
                rates.camera_focal
            );
        }
    }

    /// After step `step` (0-based) of `view`: `read_step_async`, printed with the pose
    /// gradient.
    async fn print_step_async(&mut self, step: usize, view: usize) {
        let loss = self.read_step_async().await;
        println!(
            "step {:>5} view {view}: {loss}, {} gaussian-tile pairs, {:.1?}",
            step + 1,
            self.last_intersections,
            self.started.elapsed()
        );
        self.print_pose_step_async(view).await;
    }

    /// After a logged step of `view`: the view's pose gradient against the Gaussians'
    /// mean position gradient (whether the pose gradient is comparable, not swamped),
    /// and where the view's pose stands.
    async fn print_pose_step_async(&self, view: usize) {
        let Some(poses) = &self.poses else {
            return;
        };
        let gradients = self.rasterizer.read_param_gradients_async(self.gpu).await;
        let position_gradient = gradients
            .chunks_exact(PARAM_STRIDE)
            .map(|g| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt() as f64)
            .sum::<f64>()
            / self.gaussian_count as f64;
        let gradient = poses.last_gradient();
        let correction = poses.correction(view);
        println!(
            "      pose gradient: rotation {:.3e}, translation {:.3e}, focal {:.3e} (mean Gaussian position gradient {:.3e}); view {view} rotated {:.2} deg, center moved {:.4}, focal scale {:.4}",
            gradient.rotation.norm(),
            gradient.translation.norm(),
            gradient.focal_log_scale,
            position_gradient,
            correction.rotation.norm().to_degrees(),
            correction.translation.norm(),
            poses.focal_scale()
        );
    }

    /// Per view, how far its pose moved from the initial one, and the focal scale.
    fn print_pose_corrections(&self) {
        let Some(poses) = &self.poses else {
            return;
        };
        for view in 0..self.view_count() {
            let correction = poses.correction(view);
            println!(
                "  view {view}: rotated {:.2} deg, center moved {:.4} ({:.1}% of the extent)",
                correction.rotation.norm().to_degrees(),
                correction.translation.norm(),
                100.0 * correction.translation.norm() / self.extent as f64
            );
        }
        println!("  focal scale {:.4}", poses.focal_scale());
    }

    pub async fn render_async(&self, view: usize) -> Rendered {
        self.rasterizer.render_async(self.gpu, &self.camera(view)).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(&self, view: usize) -> Rendered {
        pollster::block_on(self.render_async(view))
    }

    /// Any camera of the views' image size, for novel viewpoints.
    pub async fn render_camera_async(&self, camera: &PinholeCamera) -> Rendered {
        self.rasterizer.render_async(self.gpu, camera).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_camera(&self, camera: &PinholeCamera) -> Rendered {
        pollster::block_on(self.render_camera_async(camera))
    }

    /// The loss of `view` with the current Gaussians, without a step.
    pub async fn loss_async(&self, view: usize) -> LossTerms {
        self.rasterizer.set_camera(self.gpu, &self.camera(view));
        let mut encoder = self.gpu.encoder();
        self.rasterizer.encode_forward(&mut encoder);
        self.rasterizer.encode_loss(&mut encoder, view);
        self.gpu.submit(encoder);
        self.rasterizer.read_loss_async(self.gpu).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn loss(&self, view: usize) -> LossTerms {
        pollster::block_on(self.loss_async(view))
    }

    /// Means over every view of the loss terms, with the current Gaussians.
    pub async fn loss_over_views_async(&self) -> LossTerms {
        let mut sum = LossTerms::default();
        for view in 0..self.view_count() {
            let terms = self.loss_async(view).await;
            sum.l1 += terms.l1;
            sum.alpha += terms.alpha;
            sum.dssim += terms.dssim;
            sum.total += terms.total;
        }
        let count = self.view_count() as f32;
        LossTerms {
            l1: sum.l1 / count,
            alpha: sum.alpha / count,
            dssim: sum.dssim / count,
            total: sum.total / count,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn loss_over_views(&self) -> LossTerms {
        pollster::block_on(self.loss_over_views_async())
    }

    pub async fn cloud_async(&self) -> GaussianCloud {
        GaussianCloud::from_packed(
            &self
                .gpu
                .read_back_async::<f32>(&self.params, self.gaussian_count * PARAM_STRIDE)
                .await,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn cloud(&self) -> GaussianCloud {
        pollster::block_on(self.cloud_async())
    }

    /// `settings.steps` steps cycling the views, printed every `settings.log_every`, then
    /// `finish_async`, its report printed.
    pub async fn optimize_async(mut self, settings: &TrainSettings) -> TrainReport {
        assert!(settings.log_every > 0, "log_every");
        self.print_settings();
        for step in 0..settings.steps {
            if let Some(densified) = self.train_step_async(step, settings.steps).await {
                println!("densified after step {step}: {densified}");
            }
            if (step + 1) % settings.log_every == 0 || step + 1 == settings.steps {
                self.print_step_async(step, step % self.view_count()).await;
            }
        }
        assert_eq!(self.adam.steps_taken() as usize, settings.steps);
        self.print_pose_corrections();
        let report = self.finish_async().await;
        println!("mean over the {} views: {}", report.cameras.len(), report.final_loss);
        println!("pruned: {}", report.prune);
        report
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn optimize(self, settings: &TrainSettings) -> TrainReport {
        pollster::block_on(self.optimize_async(settings))
    }

    /// After the steps: the loss over every view, the Gaussians read back and pruned.
    pub async fn finish_async(self) -> TrainReport {
        assert!(self.adam.steps_taken() > 0, "no step taken");
        let final_loss = self.loss_over_views_async().await;
        let trained = self.cloud_async().await;
        let (cloud, prune_report) = prune(&trained, self.extent);
        TrainReport {
            cloud,
            cameras: self.cameras(),
            focal_scale: self.focal_scale(),
            gaussians_before_pruning: trained.len(),
            prune: prune_report,
            densifications: self.densifications,
            elapsed: self.started.elapsed(),
            last_mean_error: self.last_loss.l1,
            last_loss: self.last_loss,
            final_loss,
            last_intersections: self.last_intersections,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::densify::DENSIFY_SHORT_RUN;
    use crate::init_from_points::{INIT_OPACITY, InitSettings, gaussians_from_dense_views};
    use crate::optimizer::LEARNING_RATES_3DGS;
    use nalgebra::{Matrix3, Vector3};

    /// A 48x48 view, from a camera at x = `camera_x` looking down z, of the plane z = 2
    /// whose foreground is a disc of patterned colors; the dense points are the plane, so
    /// the Gaussians start where they should be.
    fn disc_view(camera_x: f32) -> DenseView {
        let (width, height) = (48usize, 48usize);
        let focal = 60.0f32;
        let mut points = Vec::new();
        let mut foreground = Vec::new();
        let mut rgb = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let z = 2.0f32;
                let point = [(x as f32 - 24.0) * z / focal + camera_x, (y as f32 - 24.0) * z / focal, z];
                points.push(point);
                let inside = point[0].powi(2) + point[1].powi(2) < (14.0 * z / focal).powi(2);
                foreground.push(inside);
                let mask = if inside { 1.0 } else { 0.0 };
                let (u, v) = ((point[0] * 30.0).floor() as i64, (point[1] * 30.0).floor() as i64);
                rgb.extend([(u.rem_euclid(3)) as f32 / 2.0 * mask, 0.6 * mask, (v.rem_euclid(2)) as f32 * mask]);
            }
        }
        DenseView {
            width,
            height,
            points,
            conf: vec![2.0; width * height],
            foreground,
            rgb,
            camera: PinholeCamera {
                rotation: Matrix3::identity(),
                translation: Vector3::new(camera_x as f64, 0.0, 0.0),
                focal: [focal as f64, focal as f64],
                principal_point: [24.0, 24.0],
                width: width as u32,
                height: height as u32,
            },
        }
    }

    #[test]
    fn optimization_lowers_every_loss_term_and_prunes() {
        let views = [disc_view(0.0), disc_view(0.3)];
        let cloud = gaussians_from_dense_views(
            &views,
            InitSettings {
                stride: 1,
                min_conf: 1.0,
                opacity: INIT_OPACITY,
            },
        );
        let gpu = Gpu::request();
        let trainer = Trainer::new(&gpu, &cloud, &views, LEARNING_RATES_3DGS, CameraPoses::Fixed);
        let before = trainer.loss_over_views();
        let report = trainer.optimize(&TrainSettings {
            steps: 150,
            log_every: 50,
            learning_rates: LEARNING_RATES_3DGS,
        });
        let after = report.final_loss;
        assert!(after.total < 0.5 * before.total, "total {} -> {}", before.total, after.total);
        assert!(after.alpha < 0.5 * before.alpha, "alpha {} -> {}", before.alpha, after.alpha);
        assert!(after.l1 < before.l1, "L1 {} -> {}", before.l1, after.l1);
        assert!(after.dssim < before.dssim, "D-SSIM {} -> {}", before.dssim, after.dssim);
        assert_eq!(report.gaussians_before_pruning, cloud.len());
        assert_eq!(report.prune.dropped_oversized, 0, "the clamp keeps every scale under the cap");
        assert!(
            report.cloud.len() > cloud.len() / 2,
            "pruned to {} of {}",
            report.cloud.len(),
            cloud.len()
        );
        assert_eq!(report.cameras[1].translation, views[1].camera.translation, "fixed poses stay");
        assert_eq!(report.focal_scale, 1.0);
    }

    #[test]
    #[should_panic(expected = "camera extent 0")]
    fn cameras_at_one_position_are_refused() {
        let views = [disc_view(0.0), disc_view(0.0)];
        let cloud = gaussians_from_dense_views(
            &views,
            InitSettings {
                stride: 1,
                min_conf: 1.0,
                opacity: INIT_OPACITY,
            },
        );
        let gpu = Gpu::request();
        Trainer::new(&gpu, &cloud, &views, LEARNING_RATES_3DGS, CameraPoses::Fixed);
    }

    #[test]
    fn training_with_densification_changes_the_count_and_lowers_the_loss() {
        let views = [disc_view(0.0), disc_view(0.3)];
        let cloud = gaussians_from_dense_views(
            &views,
            InitSettings {
                stride: 1,
                min_conf: 1.0,
                opacity: INIT_OPACITY,
            },
        );
        let gpu = Gpu::request();
        let trainer = Trainer::new(&gpu, &cloud, &views, LEARNING_RATES_3DGS, CameraPoses::Fixed).with_densification(DENSIFY_SHORT_RUN);
        let before = trainer.loss_over_views();
        let steps = 150;
        let report = trainer.optimize(&TrainSettings {
            steps,
            log_every: 50,
            learning_rates: LEARNING_RATES_3DGS,
        });
        assert_eq!(report.densifications.len(), 1, "one densification at step 50 of 150");
        let densification = report.densifications[0];
        assert_eq!(densification.before, cloud.len());
        assert!(densification.cloned + densification.split > 0, "nothing densified: {densification}");
        assert_eq!(report.gaussians_before_pruning, densification.after);
        assert!(
            report.final_loss.total < 0.5 * before.total,
            "total {} -> {}",
            before.total,
            report.final_loss.total
        );
    }
}
