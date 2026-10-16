// Shared declarations, prepended to every kernel (WGSL has no include).
//
// Conventions: cameras are OpenCV (x right, y down, z forward) and pixel (i, j) has its
// center at image coordinates (i, j). Gaussian parameters are packed per Gaussian with
// PARAM_STRIDE floats: position xyz, log scale xyz, rotation wxyz, opacity logit, SH dc rgb.

const TILE: u32 = 16u;
const PARAM_STRIDE: u32 = 14u;
const SLOT_POSITION: u32 = 0u;
const SLOT_LOG_SCALE: u32 = 3u;
const SLOT_ROTATION: u32 = 6u;
const SLOT_OPACITY: u32 = 10u;
const SLOT_SH_DC: u32 = 11u;
// Per-intersection gradient: mean xy, conic abc, color rgb, opacity, one unused.
const GRAD_STRIDE: u32 = 10u;
// Per-view camera gradient: the 9 entries of the world-to-camera rotation (row-major),
// the world-to-camera translation, the two focal lengths (`camera::CameraGradient`).
const CAMERA_GRAD_STRIDE: u32 = 14u;
// Sort keys: tile index in the high bits, the top bits of the (positive) depth below.
const DEPTH_BITS: u32 = 20u;
const SH_C0: f32 = 0.28209479;
// Added to the 2D covariance diagonal so that every Gaussian covers at least a pixel.
const COVARIANCE_BLUR: f32 = 0.3;
const NEAR_Z: f32 = 0.01;
// 3DGS limits on x/z and y/z, as a multiple of the half field of view, before projecting
// the covariance; keeps the Jacobian finite for Gaussians far outside the frame.
const VIEW_LIMIT_FACTOR: f32 = 1.3;
const MIN_ALPHA: f32 = 1.0 / 255.0;
const MAX_ALPHA: f32 = 0.99;
const MIN_TRANSMITTANCE: f32 = 1e-4;
// SSIM as 3DGS: an 11x11 Gaussian window of sigma 1.5, zero padded at the image border,
// stabilizers C1 = (0.01 L)^2 and C2 = (0.03 L)^2 for a dynamic range L of 1.
const SSIM_RADIUS: i32 = 5;
const SSIM_SIGMA: f32 = 1.5;
const SSIM_C1: f32 = 0.0001;
const SSIM_C2: f32 = 0.0009;

struct Camera {
    // World to camera, 3 rows of (r0, r1, r2, t).
    rows: array<vec4<f32>, 3>,
    focal: vec2<f32>,
    principal_point: vec2<f32>,
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
};

// One Gaussian seen from one camera.
struct Projected {
    mean: vec2<f32>,
    conic: vec3<f32>,
    depth: f32,
    color: vec3<f32>,
    opacity: f32,
    // 0 when culled.
    radius: u32,
    tile_min: vec2<u32>,
    // Exclusive.
    tile_max: vec2<u32>,
};

// The list of (Gaussian, tile) pairs of one render.
struct IntersectionMeta {
    count: u32,
    sort_blocks: u32,
    histogram_length: u32,
    overflow: u32,
    capacity: u32,
};

fn world_to_camera(camera: Camera, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(camera.rows[0], h), dot(camera.rows[1], h), dot(camera.rows[2], h));
}

// The rotation part of world_to_camera.
fn world_to_camera_rotation(camera: Camera) -> mat3x3<f32> {
    // mat3x3 is column-major: columns are the rows' first three entries transposed.
    return mat3x3<f32>(
        vec3<f32>(camera.rows[0].x, camera.rows[1].x, camera.rows[2].x),
        vec3<f32>(camera.rows[0].y, camera.rows[1].y, camera.rows[2].y),
        vec3<f32>(camera.rows[0].z, camera.rows[1].z, camera.rows[2].z),
    );
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Rotation matrix of a unit quaternion (w, x, y, z).
fn quaternion_to_matrix(q: vec4<f32>) -> mat3x3<f32> {
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    // Columns.
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
        vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
        vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y)),
    );
}

fn tile_of_key(key: u32) -> u32 {
    return key >> DEPTH_BITS;
}

// The SSIM window's weights along one axis, summing to 1.
fn ssim_taps() -> array<f32, 11> {
    var taps: array<f32, 11>;
    var sum = 0.0;
    for (var k = 0; k <= 2 * SSIM_RADIUS; k++) {
        let d = f32(k - SSIM_RADIUS);
        taps[k] = exp(-d * d / (2.0 * SSIM_SIGMA * SSIM_SIGMA));
        sum += taps[k];
    }
    for (var k = 0; k <= 2 * SSIM_RADIUS; k++) {
        taps[k] /= sum;
    }
    return taps;
}
