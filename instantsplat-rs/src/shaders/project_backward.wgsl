// Backward of project_forward: sums each Gaussian's per-tile gradients (2D mean, conic,
// color, opacity) and chains them back to its 3D parameters, and to the camera: the
// world-to-camera rotation and translation and the focal lengths. One thread per Gaussian,
// writing all PARAM_STRIDE gradients (zeros when culled); the camera gradients are summed
// over the workgroup's 256 Gaussians into one partial per workgroup
// (reduce_camera_gradient.wgsl adds the partials up), no atomics. Each visible Gaussian
// also adds its screen-space mean gradient to its densification statistics.

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> projected: array<Projected>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> intersection_gradients: array<f32>;
@group(0) @binding(5) var<storage, read_write> param_gradients: array<f32>;
// One CAMERA_GRAD_STRIDE partial per workgroup.
@group(0) @binding(6) var<storage, read_write> camera_partials: array<f32>;
// Per Gaussian, summed over the backward passes since they were last cleared
// (`densify.rs`): the norm of the 2D mean gradient in normalized device coordinates (the
// pixel gradient times half the image size, as 3DGS thresholds it), the passes it was
// visible in, and the largest screen radius in pixels.
@group(0) @binding(7) var<storage, read_write> densify_stats: array<vec4<f32>>;

struct Gradients {
    params: array<f32, PARAM_STRIDE>,
    camera: array<f32, CAMERA_GRAD_STRIDE>,
};

var<workgroup> partial: array<array<f32, CAMERA_GRAD_STRIDE>, 256>;

fn gradients_of(g: u32) -> Gradients {
    // Zero-initialized: what culled Gaussians return.
    var grads: Gradients;
    let p = projected[g];
    if p.radius == 0u {
        return grads;
    }

    // Sum over the tiles this Gaussian touched.
    let count = (p.tile_max.x - p.tile_min.x) * (p.tile_max.y - p.tile_min.y);
    let capacity = arrayLength(&intersection_gradients) / GRAD_STRIDE;
    var d_mean = vec2<f32>(0.0);
    var d_conic = vec3<f32>(0.0);
    var d_color = vec3<f32>(0.0);
    var d_opacity = 0.0;
    for (var k = 0u; k < count; k++) {
        let slot = tile_offsets[g] + k;
        if slot >= capacity {
            break;
        }
        let base = slot * GRAD_STRIDE;
        d_mean += vec2<f32>(intersection_gradients[base], intersection_gradients[base + 1u]);
        d_conic += vec3<f32>(intersection_gradients[base + 2u], intersection_gradients[base + 3u], intersection_gradients[base + 4u]);
        d_color += vec3<f32>(intersection_gradients[base + 5u], intersection_gradients[base + 6u], intersection_gradients[base + 7u]);
        d_opacity += intersection_gradients[base + 8u];
    }
    let half_image = 0.5 * vec2<f32>(f32(camera.width), f32(camera.height));
    let stats = densify_stats[g];
    densify_stats[g] = vec4<f32>(
        stats.x + length(d_mean * half_image),
        stats.y + 1.0,
        max(stats.z, f32(p.radius)),
        0.0,
    );

    // Recompute the forward quantities.
    let base = g * PARAM_STRIDE;
    let position = vec3<f32>(params[base], params[base + 1u], params[base + 2u]);
    let pc = world_to_camera(camera, position);
    let scale = exp(vec3<f32>(params[base + 3u], params[base + 4u], params[base + 5u]));
    let q_raw = vec4<f32>(params[base + 6u], params[base + 7u], params[base + 8u], params[base + 9u]);
    let q = normalize(q_raw);
    let r = quaternion_to_matrix(q);
    let m = mat3x3<f32>(r[0] * scale.x, r[1] * scale.y, r[2] * scale.z);
    let sigma = m * transpose(m);
    let limit = VIEW_LIMIT_FACTOR * vec2<f32>(f32(camera.width), f32(camera.height)) / (2.0 * camera.focal);
    let xz = pc.x / pc.z;
    let yz = pc.y / pc.z;
    let x_clamped = abs(xz) > limit.x;
    let y_clamped = abs(yz) > limit.y;
    let tx = clamp(xz, -limit.x, limit.x) * pc.z;
    let ty = clamp(yz, -limit.y, limit.y) * pc.z;
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let z2 = pc.z * pc.z;
    let z3 = z2 * pc.z;
    let j = mat3x2<f32>(
        vec2<f32>(fx / pc.z, 0.0),
        vec2<f32>(0.0, fy / pc.z),
        vec2<f32>(-fx * tx / z2, -fy * ty / z2),
    );
    let w = world_to_camera_rotation(camera);
    let t = j * w;
    let cov2d = t * sigma * transpose(t);
    let a = cov2d[0][0] + COVARIANCE_BLUR;
    let b = cov2d[1][0];
    let c = cov2d[1][1] + COVARIANCE_BLUR;
    let det = a * c - b * b;
    // det > 0 held in the forward pass, as radius > 0; keep the compiler from folding.
    if det <= 0.0 {
        return grads;
    }

    // Opacity and color.
    let opacity = p.opacity;
    grads.params[SLOT_OPACITY] = d_opacity * opacity * (1.0 - opacity);
    let color_raw = SH_C0 * vec3<f32>(params[base + 11u], params[base + 12u], params[base + 13u]) + 0.5;
    let d_sh = SH_C0 * d_color * vec3<f32>(color_raw > vec3<f32>(0.0));
    grads.params[SLOT_SH_DC] = d_sh.x;
    grads.params[SLOT_SH_DC + 1u] = d_sh.y;
    grads.params[SLOT_SH_DC + 2u] = d_sh.z;

    // conic = inverse(cov2d): dL/dcov2d = -conic dL/dconic conic, with the single
    // off-diagonal parameter split over both entries.
    let conic_matrix = mat2x2<f32>(vec2<f32>(p.conic.x, p.conic.y), vec2<f32>(p.conic.y, p.conic.z));
    let d_conic_matrix = mat2x2<f32>(vec2<f32>(d_conic.x, 0.5 * d_conic.y), vec2<f32>(0.5 * d_conic.y, d_conic.z));
    let d_cov2d_full = -1.0 * (conic_matrix * d_conic_matrix * conic_matrix);
    // As the symmetric matrix gradient (off-diagonal halved) used for the chain below.
    let d_cov2d = mat2x2<f32>(
        vec2<f32>(d_cov2d_full[0][0], 0.5 * (d_cov2d_full[0][1] + d_cov2d_full[1][0])),
        vec2<f32>(0.5 * (d_cov2d_full[0][1] + d_cov2d_full[1][0]), d_cov2d_full[1][1]),
    );

    // cov2d = T Sigma T^T: dL/dSigma = T^T G T, dL/dT = 2 G T Sigma.
    let d_sigma = transpose(t) * d_cov2d * t;
    let d_t = 2.0 * (d_cov2d * t * sigma);
    // T = J W: dL/dJ = dL/dT W^T.
    let d_j = d_t * transpose(w);
    // J's dependence on the camera-space point (tx, ty constant where clamped).
    var d_pc = vec3<f32>(0.0);
    d_pc.x += select(-fx / z2 * d_j[2][0], 0.0, x_clamped);
    d_pc.y += select(-fy / z2 * d_j[2][1], 0.0, y_clamped);
    d_pc.z += -fx / z2 * d_j[0][0] - fy / z2 * d_j[1][1]
        + 2.0 * fx * tx / z3 * d_j[2][0] + 2.0 * fy * ty / z3 * d_j[2][1];
    // The 2D mean.
    d_pc.x += d_mean.x * fx / pc.z;
    d_pc.y += d_mean.y * fy / pc.z;
    d_pc.z += -d_mean.x * fx * pc.x / z2 - d_mean.y * fy * pc.y / z2;
    let d_position = transpose(w) * d_pc;
    grads.params[SLOT_POSITION] = d_position.x;
    grads.params[SLOT_POSITION + 1u] = d_position.y;
    grads.params[SLOT_POSITION + 2u] = d_position.z;

    // The camera: pc = W x + t and T = J W, so dL/dW = d_pc x^T + J^T dL/dT and
    // dL/dt = d_pc (column-major: d_w[col][row]).
    let d_w = mat3x3<f32>(d_pc * position.x, d_pc * position.y, d_pc * position.z) + transpose(j) * d_t;
    for (var row = 0u; row < 3u; row++) {
        for (var col = 0u; col < 3u; col++) {
            grads.camera[row * 3u + col] = d_w[col][row];
        }
    }
    grads.camera[9] = d_pc.x;
    grads.camera[10] = d_pc.y;
    grads.camera[11] = d_pc.z;
    // The focal: through the mean (f pc.xy / pc.z) and J; where tx is clamped, J's third
    // column is -VIEW_LIMIT_FACTOR width / (2 z), independent of the focal.
    grads.camera[12] = d_mean.x * pc.x / pc.z + d_j[0][0] / pc.z + select(-tx / z2 * d_j[2][0], 0.0, x_clamped);
    grads.camera[13] = d_mean.y * pc.y / pc.z + d_j[1][1] / pc.z + select(-ty / z2 * d_j[2][1], 0.0, y_clamped);

    // Sigma = M M^T: dL/dM = 2 dL/dSigma M (dL/dSigma is symmetric).
    let d_m = 2.0 * (d_sigma * m);
    // M = R diag(scale): dL/dscale_k = column k of R dot column k of dL/dM, times scale for the log.
    let d_log_scale = vec3<f32>(dot(r[0], d_m[0]), dot(r[1], d_m[1]), dot(r[2], d_m[2])) * scale;
    grads.params[SLOT_LOG_SCALE] = d_log_scale.x;
    grads.params[SLOT_LOG_SCALE + 1u] = d_log_scale.y;
    grads.params[SLOT_LOG_SCALE + 2u] = d_log_scale.z;
    // dL/dR = dL/dM diag(scale), then through the quaternion (w, x, y, z) and its normalization.
    let d_r = mat3x3<f32>(d_m[0] * scale.x, d_m[1] * scale.y, d_m[2] * scale.z);
    // d_r[col][row] = dL/dR_{row, col}.
    let r00 = d_r[0][0];
    let r01 = d_r[1][0];
    let r02 = d_r[2][0];
    let r10 = d_r[0][1];
    let r11 = d_r[1][1];
    let r12 = d_r[2][1];
    let r20 = d_r[0][2];
    let r21 = d_r[1][2];
    let r22 = d_r[2][2];
    let qw = q.x;
    let qx = q.y;
    let qy = q.z;
    let qz = q.w;
    let d_unit = vec4<f32>(
        2.0 * (qz * (r10 - r01) + qy * (r02 - r20) + qx * (r21 - r12)),
        2.0 * (qy * (r01 + r10) + qz * (r02 + r20) + qw * (r21 - r12)) - 4.0 * qx * (r11 + r22),
        2.0 * (qx * (r01 + r10) + qw * (r02 - r20) + qz * (r12 + r21)) - 4.0 * qy * (r00 + r22),
        2.0 * (qw * (r10 - r01) + qx * (r02 + r20) + qy * (r12 + r21)) - 4.0 * qz * (r00 + r11),
    );
    let d_q = (d_unit - q * dot(q, d_unit)) / length(q_raw);
    grads.params[SLOT_ROTATION] = d_q.x;
    grads.params[SLOT_ROTATION + 1u] = d_q.y;
    grads.params[SLOT_ROTATION + 2u] = d_q.z;
    grads.params[SLOT_ROTATION + 3u] = d_q.w;
    return grads;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    let g = id.x;
    // Threads past the last Gaussian contribute zeros and stay for the reduction's barriers.
    var grads: Gradients;
    if g < arrayLength(&tile_offsets) {
        grads = gradients_of(g);
        for (var k = 0u; k < PARAM_STRIDE; k++) {
            param_gradients[g * PARAM_STRIDE + k] = grads.params[k];
        }
    }
    // Sum the camera gradients over the workgroup.
    partial[thread] = grads.camera;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half >>= 1u) {
        if thread < half {
            for (var c = 0u; c < CAMERA_GRAD_STRIDE; c++) {
                partial[thread][c] += partial[thread + half][c];
            }
        }
        workgroupBarrier();
    }
    if thread == 0u {
        for (var c = 0u; c < CAMERA_GRAD_STRIDE; c++) {
            camera_partials[group.x * CAMERA_GRAD_STRIDE + c] = partial[0][c];
        }
    }
}
