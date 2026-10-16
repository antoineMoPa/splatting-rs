// Projects every Gaussian's log scales back inside the scale limits after an Adam step:
// no axis above the absolute cap, and no axis more than the anisotropy cap above the
// smallest axis. One thread per Gaussian.

struct ScaleLimits {
    count: u32,
    max_log_scale: f32,
    max_log_anisotropy: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<uniform> limits: ScaleLimits;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = id.x;
    if g >= limits.count {
        return;
    }
    let base = g * PARAM_STRIDE + SLOT_LOG_SCALE;
    var log_scale = vec3<f32>(params[base], params[base + 1u], params[base + 2u]);
    log_scale = min(log_scale, vec3<f32>(limits.max_log_scale));
    let smallest = min(log_scale.x, min(log_scale.y, log_scale.z));
    log_scale = min(log_scale, vec3<f32>(smallest + limits.max_log_anisotropy));
    params[base] = log_scale.x;
    params[base + 1u] = log_scale.y;
    params[base + 2u] = log_scale.z;
}
