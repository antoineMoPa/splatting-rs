//! One image's dense 3D points in the scene frame.

/// Row-major, on the CPU.
#[derive(Clone)]
pub struct PointMap {
    pub height: usize,
    pub width: usize,
    pub points: Vec<[f32; 3]>,
    pub conf: Vec<f32>,
    /// Pixels of the object being reconstructed, from the input image's alpha.
    pub foreground: Vec<bool>,
}
