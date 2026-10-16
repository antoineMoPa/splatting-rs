//! InstantSplat-style pipeline: dense points and cameras from MapAnything, one Gaussian per
//! point, then a short optimization of the Gaussians against the input images with a wgpu
//! rasterizer (forward and backward passes in WGSL).

pub mod camera;
pub mod camera_pose;
pub mod cameras_json;
pub mod colmap_model;
pub mod dense_view;
pub mod evaluate_against_colmap;
#[cfg(feature = "mapanything")]
pub mod fit;
pub mod gaussians;
pub mod gpu;
pub mod init_from_points;
pub mod loss;
#[cfg(feature = "mapanything")]
pub mod network;
pub mod optimizer;
pub mod pinhole_fit;
pub mod pose_refinement;
pub mod posed_frames;
pub mod prune;
pub mod rasterizer;
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
