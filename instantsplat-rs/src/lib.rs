//! InstantSplat-style pipeline: dense points and cameras from MapAnything or Depth Anything
//! 3, one Gaussian per point, then a short optimization of the Gaussians against the input images with a wgpu
//! rasterizer (forward and backward passes in WGSL).
//!
//! Builds for wasm32 too (`--no-default-features`, plus `--features da3` for Depth
//! Anything 3 on candle's CPU backend): the kernels, the optimization and the PLY /
//! `cameras.json` parsers; the commands that read photos and COLMAP models from disk
//! (`fit`, `refine`, `evaluate_against_colmap`, `posed_frames`) are native only.

pub mod camera;
pub mod camera_pose;
pub mod cameras_json;
pub mod colmap_model;
pub mod dense_view;
#[cfg(not(target_arch = "wasm32"))]
pub mod evaluate_against_colmap;
#[cfg(feature = "networks")]
pub mod fit;
pub mod gaussians;
pub mod gpu;
pub mod init_from_points;
pub mod loss;
#[cfg(feature = "da3")]
pub mod network;
#[cfg(feature = "da3")]
pub mod network_da3;
#[cfg(feature = "networks")]
pub mod network_mapanything;
pub mod optimizer;
pub mod pinhole_fit;
pub mod pose_refinement;
#[cfg(not(target_arch = "wasm32"))]
pub mod posed_frames;
pub mod prune;
pub mod rasterizer;
#[cfg(not(target_arch = "wasm32"))]
pub mod refine;
pub mod replace_rays;
pub mod run_output;
pub mod shaders;
pub mod similarity_align;
pub mod sort;
pub mod splat_ply;
pub mod training;
pub mod training_frames;
pub mod view_error;

#[cfg(test)]
mod loss_tests;
#[cfg(test)]
mod pose_tests;
#[cfg(test)]
mod rasterizer_tests;
