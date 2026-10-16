//! The WGSL kernels, each prefixed with the shared declarations of `common.wgsl`.

const COMMON: &str = include_str!("shaders/common.wgsl");

macro_rules! kernel {
    ($name:ident, $file:literal) => {
        pub fn $name() -> String {
            format!("{COMMON}\n{}", include_str!(concat!("shaders/", $file)))
        }
    };
}

kernel!(project_forward, "project_forward.wgsl");
kernel!(scan, "scan.wgsl");
kernel!(emit_tile_keys, "emit_tile_keys.wgsl");
kernel!(sort_histogram, "sort_histogram.wgsl");
kernel!(sort_scatter, "sort_scatter.wgsl");
kernel!(tile_ranges, "tile_ranges.wgsl");
kernel!(rasterize_forward, "rasterize_forward.wgsl");
kernel!(ssim_forward, "ssim_forward.wgsl");
kernel!(loss, "loss.wgsl");
kernel!(rasterize_backward, "rasterize_backward.wgsl");
kernel!(project_backward, "project_backward.wgsl");
kernel!(reduce_camera_gradient, "reduce_camera_gradient.wgsl");
kernel!(adam, "adam.wgsl");
kernel!(clamp_scales, "clamp_scales.wgsl");

/// Pixels per tile side, as `TILE` in the shaders.
pub const TILE: u32 = 16;
/// Floats per Gaussian in the packed parameter buffer, as `PARAM_STRIDE` in the shaders.
pub const PARAM_STRIDE: usize = 14;
/// Floats per (Gaussian, tile) gradient slot, as `GRAD_STRIDE` in the shaders.
pub const GRAD_STRIDE: usize = 10;
/// Floats of one view's camera gradient (`camera::CameraGradient`), as `CAMERA_GRAD_STRIDE`
/// in the shaders.
pub const CAMERA_GRAD_STRIDE: usize = 14;
/// Bits of the sort key below the tile index, as `DEPTH_BITS` in the shaders.
pub const DEPTH_BITS: u32 = 20;
/// Keys per radix sort block, as `SORT_BLOCK` in the shaders.
pub const SORT_BLOCK: usize = 1024;
/// Bins per radix sort pass, as `SORT_BINS` in the shaders.
pub const SORT_BINS: usize = 16;
/// Bytes of one `Projected` struct in the shaders.
pub const PROJECTED_BYTES: u64 = 80;
/// Half side of the SSIM window, as `SSIM_RADIUS` in the shaders (11x11).
pub const SSIM_RADIUS: usize = 5;
/// Standard deviation of the SSIM window, as `SSIM_SIGMA` in the shaders.
pub const SSIM_SIGMA: f32 = 1.5;

#[cfg(test)]
mod tests {
    #[test]
    fn constants_match_the_shaders() {
        let common = super::COMMON;
        assert!(common.contains("const TILE: u32 = 16u;"));
        assert!(common.contains("const PARAM_STRIDE: u32 = 14u;"));
        assert!(common.contains("const GRAD_STRIDE: u32 = 10u;"));
        assert!(common.contains("const CAMERA_GRAD_STRIDE: u32 = 14u;"));
        assert_eq!(std::mem::size_of::<crate::camera::CameraGradient>(), super::CAMERA_GRAD_STRIDE * 4);
        assert!(common.contains("const DEPTH_BITS: u32 = 20u;"));
        assert!(common.contains("const SSIM_RADIUS: i32 = 5;"));
        assert!(common.contains("const SSIM_SIGMA: f32 = 1.5;"));
    }
}
