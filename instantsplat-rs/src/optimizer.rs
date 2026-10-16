//! Adam over the packed Gaussian parameters, on the GPU, with a learning rate per
//! parameter group, the position rate decaying over the run, and the scale limits
//! applied after every step.

use bytemuck::{Pod, Zeroable};

use crate::gpu::{Gpu, dispatch, workgroups};
use crate::prune::SCALE_LIMITS;
use crate::shaders::{self, PARAM_STRIDE};

/// Per group, as the reference 3DGS implementation; `position` and `camera_translation`
/// are multiplied by the scene's camera extent. The `camera_*` rates are the pose
/// refinement's (`pose_refinement.rs`): rotation vector in radians, camera center,
/// log of the shared focal scale.
#[derive(Clone, Copy, Debug)]
pub struct LearningRates {
    pub position: f32,
    pub log_scale: f32,
    pub rotation: f32,
    pub opacity: f32,
    pub sh_dc: f32,
    pub camera_rotation: f32,
    pub camera_translation: f32,
    pub camera_focal: f32,
}

/// 3DGS's Gaussian rates. Camera pose: 10x InstantSplat's 1e-3 (decayed like the position
/// rate), because at 1e-3 the poses can travel about a degree over a 1000-step run and
/// MapAnything's are several degrees off; the shared focal is not moved by default (it
/// drifted away from the camera's focal, see NOTES). `fit --camera-*-lr` override them.
pub const LEARNING_RATES_3DGS: LearningRates = LearningRates {
    position: 1.6e-4,
    log_scale: 5e-3,
    rotation: 1e-3,
    opacity: 5e-2,
    sh_dc: 2.5e-3,
    camera_rotation: 1e-2,
    camera_translation: 1e-2,
    camera_focal: 0.0,
};

/// The position rate at the last step over the first, as 3DGS (1.6e-4 to 1.6e-6) and
/// InstantSplat over its 1000 steps, decayed exponentially in between. The camera pose
/// rates decay by the same factor.
pub const POSITION_RATE_FINAL_OVER_INITIAL: f32 = 0.01;

/// Multiplies the position (and camera pose) learning rates at `step` (0-based) of `steps`.
pub fn position_rate_factor(step: usize, steps: usize) -> f32 {
    assert!(steps > 0 && step < steps, "step {step} of {steps}");
    POSITION_RATE_FINAL_OVER_INITIAL.powf(step as f32 / (steps - 1).max(1) as f32)
}

pub const ADAM_BETA1: f32 = 0.9;
pub const ADAM_BETA2: f32 = 0.999;
pub const ADAM_EPSILON: f32 = 1e-15;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AdamStep {
    count: u32,
    step: u32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    _pad: [f32; 3],
    learning_rate_by_slot: [[f32; 4]; 4],
}

impl LearningRates {
    /// One learning rate per packed parameter slot, as 4 vec4s.
    fn by_slot(&self, position_scale: f32) -> [[f32; 4]; 4] {
        let mut slots = [0.0f32; 16];
        for (slot, rate) in slots.iter_mut().enumerate().take(PARAM_STRIDE) {
            *rate = match slot {
                0..=2 => self.position * position_scale,
                3..=5 => self.log_scale,
                6..=9 => self.rotation,
                10 => self.opacity,
                11..=13 => self.sh_dc,
                _ => unreachable!("slot {slot}"),
            };
        }
        [
            [slots[0], slots[1], slots[2], slots[3]],
            [slots[4], slots[5], slots[6], slots[7]],
            [slots[8], slots[9], slots[10], slots[11]],
            [slots[12], slots[13], slots[14], slots[15]],
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ScaleLimitsUniform {
    count: u32,
    max_log_scale: f32,
    max_log_anisotropy: f32,
    _pad: f32,
}

pub struct Adam {
    count: usize,
    gaussian_count: usize,
    step: u32,
    settings: AdamStep,
    learning_rates: LearningRates,
    position_scale: f32,
    uniform: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    clamp_scales: wgpu::ComputePipeline,
    clamp_scales_group: wgpu::BindGroup,
}

impl Adam {
    /// `position_scale` is the camera extent: it multiplies the position learning rate
    /// and sets the absolute scale cap (`SCALE_LIMITS`).
    pub fn new(
        gpu: &Gpu,
        params: &wgpu::Buffer,
        gradients: &wgpu::Buffer,
        gaussian_count: usize,
        learning_rates: LearningRates,
        position_scale: f32,
    ) -> Self {
        let count = gaussian_count * PARAM_STRIDE;
        assert_eq!(params.size(), (count * 4) as u64, "params buffer size");
        assert_eq!(gradients.size(), (count * 4) as u64, "gradients buffer size");
        let first_moment = gpu.storage_buffer("adam first moment", (count * 4) as u64);
        let second_moment = gpu.storage_buffer("adam second moment", (count * 4) as u64);
        let settings = AdamStep {
            count: count as u32,
            step: 0,
            beta1: ADAM_BETA1,
            beta2: ADAM_BETA2,
            epsilon: ADAM_EPSILON,
            _pad: [0.0; 3],
            learning_rate_by_slot: learning_rates.by_slot(position_scale),
        };
        let uniform = gpu.uniform_buffer("adam step", &settings);
        let pipeline = gpu.compute_pipeline("adam", &shaders::adam(), "main");
        let bind_group = gpu.bind_group(
            &pipeline,
            &[params, gradients, &first_moment, &second_moment, &uniform],
        );
        let limits = gpu.uniform_buffer(
            "scale limits",
            &ScaleLimitsUniform {
                count: gaussian_count as u32,
                max_log_scale: SCALE_LIMITS.max_log_scale(position_scale),
                max_log_anisotropy: SCALE_LIMITS.max_log_anisotropy(),
                _pad: 0.0,
            },
        );
        let clamp_scales = gpu.compute_pipeline("clamp scales", &shaders::clamp_scales(), "main");
        let clamp_scales_group = gpu.bind_group(&clamp_scales, &[params, &limits]);
        Self {
            count,
            gaussian_count,
            step: 0,
            settings,
            learning_rates,
            position_scale,
            uniform,
            pipeline,
            bind_group,
            clamp_scales,
            clamp_scales_group,
        }
    }

    /// One update from the gradients buffer, with the position learning rate multiplied
    /// by `position_rate_factor`, then the scale limits. Each step must be submitted
    /// before the next is encoded: the step counter travels in a uniform.
    pub fn encode_step(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, position_rate_factor: f32) {
        assert!(position_rate_factor > 0.0 && position_rate_factor <= 1.0, "position rate factor {position_rate_factor}");
        self.step += 1;
        self.settings.step = self.step;
        self.settings.learning_rate_by_slot = self.learning_rates.by_slot(self.position_scale * position_rate_factor);
        gpu.write(&self.uniform, &[self.settings]);
        dispatch(
            encoder,
            &self.pipeline,
            &self.bind_group,
            workgroups(self.count, 256),
        );
        dispatch(
            encoder,
            &self.clamp_scales,
            &self.clamp_scales_group,
            workgroups(self.gaussian_count, 256),
        );
    }

    pub fn steps_taken(&self) -> u32 {
        self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_get_their_group_rate() {
        let by_slot = LEARNING_RATES_3DGS.by_slot(2.0);
        assert_eq!(by_slot[0][0], 3.2e-4);
        assert_eq!(by_slot[0][3], 5e-3);
        assert_eq!(by_slot[1][2], 1e-3);
        assert_eq!(by_slot[2][2], 5e-2);
        assert_eq!(by_slot[3][1], 2.5e-3);
        assert_eq!(by_slot[3][2], 0.0);
    }

    #[test]
    fn position_rate_decays_to_one_percent() {
        assert_eq!(position_rate_factor(0, 1000), 1.0);
        assert!((position_rate_factor(999, 1000) - 0.01).abs() < 1e-6);
        assert!((position_rate_factor(500, 1001) - 0.1).abs() < 1e-6);
        assert_eq!(position_rate_factor(0, 1), 1.0);
    }
}
