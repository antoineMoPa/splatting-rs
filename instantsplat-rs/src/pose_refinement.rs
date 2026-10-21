//! The optimizer's state for the camera poses: per view a `PoseCorrection` with its own
//! Adam moments (updated only on the steps that render the view), one shared focal
//! correction updated every step, view 0 kept fixed to pin the gauge. The pose parameters
//! are few (7 per view), so they live on the CPU in f64: each step reads the view's
//! `CameraGradient` back from the GPU (one round trip per step) and chains it here.

use nalgebra::Vector3;

use crate::camera::{CameraGradient, PinholeCamera};
use crate::camera_pose::{PoseCorrection, PoseGradient, corrected_camera, pose_gradient};
use crate::optimizer::{ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, LearningRates};

/// Which view pins the gauge: its pose never moves.
pub const FIXED_VIEW: usize = 0;

/// Adam's moments over a few parameters.
struct AdamMoments {
    first: Vec<f64>,
    second: Vec<f64>,
    steps: u32,
}

impl AdamMoments {
    fn new(count: usize) -> Self {
        Self {
            first: vec![0.0; count],
            second: vec![0.0; count],
            steps: 0,
        }
    }

    /// One Adam update of `params` from `gradient`, with a learning rate per parameter.
    fn update(&mut self, params: &mut [f64], gradient: &[f64], learning_rates: &[f64]) {
        assert_eq!(params.len(), self.first.len());
        assert_eq!(gradient.len(), self.first.len());
        assert_eq!(learning_rates.len(), self.first.len());
        self.steps += 1;
        let (beta1, beta2) = (ADAM_BETA1 as f64, ADAM_BETA2 as f64);
        let bias1 = 1.0 - beta1.powi(self.steps as i32);
        let bias2 = 1.0 - beta2.powi(self.steps as i32);
        for i in 0..params.len() {
            self.first[i] = beta1 * self.first[i] + (1.0 - beta1) * gradient[i];
            self.second[i] = beta2 * self.second[i] + (1.0 - beta2) * gradient[i] * gradient[i];
            let first_hat = self.first[i] / bias1;
            let second_hat = self.second[i] / bias2;
            params[i] -= learning_rates[i] * first_hat / (second_hat.sqrt() + ADAM_EPSILON as f64);
        }
    }
}

pub struct PoseRefinement {
    initial: Vec<PinholeCamera>,
    corrections: Vec<PoseCorrection>,
    focal_log_scale: f64,
    /// Of the last `step`, for the log.
    last_gradient: PoseGradient,
    moments: Vec<AdamMoments>,
    focal_moments: AdamMoments,
    rotation_rate: f64,
    translation_rate: f64,
    focal_rate: f64,
}

impl PoseRefinement {
    /// `extent` (the camera extent) multiplies the translation rate, as it does the
    /// Gaussians' position rate.
    pub fn new(initial: &[PinholeCamera], learning_rates: &LearningRates, extent: f32) -> Self {
        assert!(initial.len() > FIXED_VIEW, "no views");
        assert!(extent > 0.0, "camera extent {extent}");
        Self {
            initial: initial.to_vec(),
            corrections: vec![PoseCorrection::default(); initial.len()],
            focal_log_scale: 0.0,
            last_gradient: PoseGradient::default(),
            moments: (0..initial.len()).map(|_| AdamMoments::new(6)).collect(),
            focal_moments: AdamMoments::new(1),
            rotation_rate: learning_rates.camera_rotation as f64,
            translation_rate: learning_rates.camera_translation as f64 * extent as f64,
            focal_rate: learning_rates.camera_focal as f64,
        }
    }

    pub fn camera(&self, view: usize) -> PinholeCamera {
        corrected_camera(&self.initial[view], &self.corrections[view], self.focal_log_scale)
    }

    pub fn cameras(&self) -> Vec<PinholeCamera> {
        (0..self.initial.len()).map(|view| self.camera(view)).collect()
    }

    pub fn correction(&self, view: usize) -> &PoseCorrection {
        &self.corrections[view]
    }

    /// What every focal length is multiplied by.
    pub fn focal_scale(&self) -> f64 {
        self.focal_log_scale.exp()
    }

    /// The pose gradient of the last `step`.
    pub fn last_gradient(&self) -> &PoseGradient {
        &self.last_gradient
    }

    /// The gradient of `view`'s pose parameters from the camera gradient of a backward
    /// pass at `self.camera(view)`.
    pub fn gradient(&self, view: usize, gradient: &CameraGradient) -> PoseGradient {
        pose_gradient(&self.initial[view], &self.corrections[view], self.focal_log_scale, gradient)
    }

    /// One Adam step of `view`'s pose (none for `FIXED_VIEW`) and of the shared focal,
    /// with the rates multiplied by `rate_factor` (the position rate's decay).
    pub fn step(&mut self, view: usize, gradient: &CameraGradient, rate_factor: f32) {
        assert!(rate_factor > 0.0 && rate_factor <= 1.0, "rate factor {rate_factor}");
        let pose_gradient = self.gradient(view, gradient);
        assert!(
            pose_gradient
                .rotation
                .iter()
                .chain(pose_gradient.translation.iter())
                .all(|g| g.is_finite())
                && pose_gradient.focal_log_scale.is_finite(),
            "view {view}: pose gradient is not finite: {pose_gradient:?}"
        );
        self.last_gradient = pose_gradient;
        let rate_factor = rate_factor as f64;
        if view != FIXED_VIEW {
            let correction = &mut self.corrections[view];
            let mut params = [
                correction.rotation.x,
                correction.rotation.y,
                correction.rotation.z,
                correction.translation.x,
                correction.translation.y,
                correction.translation.z,
            ];
            let gradient = [
                pose_gradient.rotation.x,
                pose_gradient.rotation.y,
                pose_gradient.rotation.z,
                pose_gradient.translation.x,
                pose_gradient.translation.y,
                pose_gradient.translation.z,
            ];
            let rotation_rate = self.rotation_rate * rate_factor;
            let translation_rate = self.translation_rate * rate_factor;
            let rates = [
                rotation_rate,
                rotation_rate,
                rotation_rate,
                translation_rate,
                translation_rate,
                translation_rate,
            ];
            self.moments[view].update(&mut params, &gradient, &rates);
            correction.rotation = Vector3::new(params[0], params[1], params[2]);
            correction.translation = Vector3::new(params[3], params[4], params[5]);
        }
        let mut focal = [self.focal_log_scale];
        self.focal_moments
            .update(&mut focal, &[pose_gradient.focal_log_scale], &[self.focal_rate * rate_factor]);
        self.focal_log_scale = focal[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::LEARNING_RATES_3DGS;
    use nalgebra::Matrix3;

    fn cameras() -> Vec<PinholeCamera> {
        (0..2)
            .map(|i| PinholeCamera {
                rotation: Matrix3::identity(),
                translation: Vector3::new(i as f64, 0.0, 0.0),
                focal: [100.0, 100.0],
                principal_point: [10.0, 10.0],
                width: 20,
                height: 20,
            })
            .collect()
    }

    #[test]
    fn a_step_moves_the_view_against_its_gradient_and_never_the_fixed_view() {
        let mut refinement = PoseRefinement::new(&cameras(), &LEARNING_RATES_3DGS, 2.0);
        // dL/dt_wc = (1, 0, 0) with W = I: dL/dC = (-1, 0, 0), so the center moves +x.
        let gradient = CameraGradient {
            world_to_camera_translation: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        refinement.step(1, &gradient, 1.0);
        let moved = refinement.camera(1).translation;
        let expected_step = LEARNING_RATES_3DGS.camera_translation as f64 * 2.0;
        assert!((moved.x - (1.0 + expected_step)).abs() < 1e-12, "{moved}");
        assert_eq!(refinement.camera(1).rotation, Matrix3::identity());
        refinement.step(0, &gradient, 1.0);
        assert_eq!(refinement.camera(0).translation, Vector3::zeros());
        assert_eq!(refinement.correction(0), &PoseCorrection::default());
        // No focal gradient: the focal stays.
        assert_eq!(refinement.focal_scale(), 1.0);
    }

    #[test]
    fn focal_follows_its_gradient_on_every_view() {
        let rates = LearningRates {
            camera_focal: 1e-3,
            ..LEARNING_RATES_3DGS
        };
        let mut refinement = PoseRefinement::new(&cameras(), &rates, 1.0);
        let gradient = CameraGradient {
            focal: [-1.0, -1.0],
            ..Default::default()
        };
        refinement.step(0, &gradient, 1.0);
        let scale = refinement.focal_scale();
        assert!((scale.ln() - rates.camera_focal as f64).abs() < 1e-12, "{scale}");
        assert_eq!(refinement.camera(1).focal[0], 100.0 * scale);
    }
}
