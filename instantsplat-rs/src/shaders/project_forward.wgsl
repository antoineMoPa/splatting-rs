// Projects every Gaussian into one camera: 2D mean, conic (inverse 2D covariance), radius,
// the tiles it touches, color and opacity. One thread per Gaussian.

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> projected: array<Projected>;
@group(0) @binding(3) var<storage, read_write> tile_counts: array<u32>;

fn culled(g: u32) {
    var out: Projected;
    out.radius = 0u;
    out.tile_min = vec2<u32>(0u);
    out.tile_max = vec2<u32>(0u);
    out.depth = 0.0;
    projected[g] = out;
    tile_counts[g] = 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = id.x;
    if g >= arrayLength(&tile_counts) {
        return;
    }
    let base = g * PARAM_STRIDE;
    let position = vec3<f32>(params[base], params[base + 1u], params[base + 2u]);
    let p = world_to_camera(camera, position);
    if p.z < NEAR_Z {
        culled(g);
        return;
    }

    // 3D covariance = M M^T with M = R diag(scale).
    let scale = exp(vec3<f32>(params[base + 3u], params[base + 4u], params[base + 5u]));
    let q = normalize(vec4<f32>(params[base + 6u], params[base + 7u], params[base + 8u], params[base + 9u]));
    let r = quaternion_to_matrix(q);
    let m = mat3x3<f32>(r[0] * scale.x, r[1] * scale.y, r[2] * scale.z);
    let sigma = m * transpose(m);

    // Jacobian of the projection at the (clamped) point, times the world-to-camera rotation.
    let limit = VIEW_LIMIT_FACTOR * vec2<f32>(f32(camera.width), f32(camera.height)) / (2.0 * camera.focal);
    let tx = clamp(p.x / p.z, -limit.x, limit.x) * p.z;
    let ty = clamp(p.y / p.z, -limit.y, limit.y) * p.z;
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let z2 = p.z * p.z;
    // Columns of the 2x3 Jacobian.
    let j = mat3x2<f32>(
        vec2<f32>(fx / p.z, 0.0),
        vec2<f32>(0.0, fy / p.z),
        vec2<f32>(-fx * tx / z2, -fy * ty / z2),
    );
    let t = j * world_to_camera_rotation(camera);
    // 2x2 covariance = T Sigma T^T, as (a, b, c) of [[a, b], [b, c]].
    let cov2d = t * sigma * transpose(t);
    let a = cov2d[0][0] + COVARIANCE_BLUR;
    let b = cov2d[1][0];
    let c = cov2d[1][1] + COVARIANCE_BLUR;
    let det = a * c - b * b;
    if det <= 0.0 {
        culled(g);
        return;
    }
    let conic = vec3<f32>(c, -b, a) / det;
    let mid = 0.5 * (a + c);
    let largest_eigenvalue = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(largest_eigenvalue));
    let mean = camera.focal * p.xy / p.z + camera.principal_point;

    let tiles = vec2<f32>(f32(camera.tiles_x), f32(camera.tiles_y));
    let tile_min = clamp(floor((mean - radius) / f32(TILE)), vec2<f32>(0.0), tiles);
    let tile_max = clamp(floor((mean + radius) / f32(TILE)) + 1.0, vec2<f32>(0.0), tiles);
    let count = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if count <= 0.0 {
        culled(g);
        return;
    }

    var out: Projected;
    out.mean = mean;
    out.conic = conic;
    out.depth = p.z;
    out.color = max(
        vec3<f32>(0.0),
        SH_C0 * vec3<f32>(params[base + 11u], params[base + 12u], params[base + 13u]) + 0.5,
    );
    out.opacity = sigmoid(params[base + 10u]);
    out.radius = u32(radius);
    out.tile_min = vec2<u32>(tile_min);
    out.tile_max = vec2<u32>(tile_max);
    projected[g] = out;
    tile_counts[g] = u32(count);
}
