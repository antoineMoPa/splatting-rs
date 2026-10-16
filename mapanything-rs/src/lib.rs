//! MapAnything (facebook/map-anything-apache) 3D reconstruction on candle: any number of
//! images in, one dense colored point cloud out.

pub mod cleaning;
pub mod dinov2;
pub mod dpt;
pub mod heads;
pub mod image_input;
pub mod info_sharing;
pub mod model;
pub mod pipeline;
pub mod ply;
pub mod point_map;
pub mod preprocess;
pub mod ray_encoder;
pub mod reconstruction;
pub mod vit_block;
