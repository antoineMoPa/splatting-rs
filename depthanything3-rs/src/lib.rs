//! Depth Anything 3 (depth-anything/DA3-SMALL, DA3-BASE) on candle: any number of images
//! in; per view a depth map, its confidence, a camera-to-world pose and intrinsics out, and
//! from those a point cloud in the first image's camera frame.

pub mod backbone;
pub mod camera_head;
pub mod depth_image;
pub mod dpt;
pub mod model;
pub mod pipeline;
pub mod preprocess;
pub mod reference_view;
pub mod rope;
pub mod uv_embed;
pub mod variant;
pub mod vit_block;
pub mod weights;
