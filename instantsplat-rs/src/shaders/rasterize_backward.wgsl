// Backward of rasterize_forward: walks each tile's Gaussians back to front, recomputing
// each pixel's transmittance in front of every Gaussian, and accumulates the loss gradient
// with respect to the Gaussian's 2D mean, conic, color and opacity. The loss gradient has
// four channels per pixel: rgb and the remaining transmittance, which is composited like a
// fourth color channel that every Gaussian has 0 of and the empty space behind has 1 of. Every thread's
// contribution is summed over the workgroup, so each (Gaussian, tile) pair gets one write
// into its own gradient slot: no atomics, deterministic.

const BATCH: u32 = 64u;

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> sorted_values: array<u32>;
@group(0) @binding(3) var<storage, read> intersection_gaussian: array<u32>;
@group(0) @binding(4) var<storage, read> ranges: array<u32>;
// rgb and final transmittance, from the forward pass.
@group(0) @binding(5) var<storage, read> image: array<f32>;
@group(0) @binding(6) var<storage, read> contributions: array<u32>;
// dL/d(rgb, transmittance) per pixel.
@group(0) @binding(7) var<storage, read> loss_gradient: array<f32>;
@group(0) @binding(8) var<storage, read_write> intersection_gradients: array<f32>;

var<workgroup> batch_mean: array<vec2<f32>, BATCH>;
var<workgroup> batch_conic: array<vec3<f32>, BATCH>;
var<workgroup> batch_color: array<vec3<f32>, BATCH>;
var<workgroup> batch_opacity: array<f32, BATCH>;
var<workgroup> batch_slot: array<u32, BATCH>;
var<workgroup> partial: array<array<f32, GRAD_STRIDE>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    let tile = group.y * camera.tiles_x + group.x;
    let start = ranges[2u * tile];
    let end = ranges[2u * tile + 1u];
    let px = group.x * TILE + local.x;
    let py = group.y * TILE + local.y;
    let inside = px < camera.width && py < camera.height;
    let pixel = vec2<f32>(f32(px), f32(py));

    var transmittance = 0.0;
    var contributed = 0u;
    var gradient = vec4<f32>(0.0);
    if inside {
        let index = py * camera.width + px;
        transmittance = image[index * 4u + 3u];
        contributed = contributions[index];
        gradient = vec4<f32>(
            loss_gradient[index * 4u],
            loss_gradient[index * 4u + 1u],
            loss_gradient[index * 4u + 2u],
            loss_gradient[index * 4u + 3u],
        );
    }
    // (Color, transmittance) composited behind the current Gaussian, times its transmittance.
    var behind = vec4<f32>(0.0, 0.0, 0.0, transmittance);

    let count = end - start;
    let batches = (count + BATCH - 1u) / BATCH;
    for (var bb = 0u; bb < batches; bb++) {
        let b = batches - 1u - bb;
        workgroupBarrier();
        let j = b * BATCH + thread;
        if thread < BATCH && j < count {
            let slot = sorted_values[start + j];
            let p = projected[intersection_gaussian[slot]];
            batch_mean[thread] = p.mean;
            batch_conic[thread] = p.conic;
            batch_color[thread] = p.color;
            batch_opacity[thread] = p.opacity;
            batch_slot[thread] = slot;
        }
        workgroupBarrier();
        let in_batch = min(BATCH, count - b * BATCH);
        for (var kk = 0u; kk < in_batch; kk++) {
            let k = in_batch - 1u - kk;
            let j = b * BATCH + k;
            var grad = array<f32, GRAD_STRIDE>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            if inside && j < contributed {
                let d = batch_mean[k] - pixel;
                let conic = batch_conic[k];
                let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
                if power <= 0.0 {
                    let gaussian = exp(power);
                    let opacity = batch_opacity[k];
                    let alpha_unclamped = opacity * gaussian;
                    let alpha = min(MAX_ALPHA, alpha_unclamped);
                    if alpha >= MIN_ALPHA {
                        // Transmittance in front of this Gaussian.
                        transmittance = transmittance / (1.0 - alpha);
                        let color = vec4<f32>(batch_color[k], 0.0);
                        let d_color = gradient.xyz * alpha * transmittance;
                        let d_alpha = dot(gradient, color * transmittance - behind / (1.0 - alpha));
                        behind += color * alpha * transmittance;
                        grad[5] = d_color.x;
                        grad[6] = d_color.y;
                        grad[7] = d_color.z;
                        if alpha_unclamped < MAX_ALPHA {
                            grad[8] = d_alpha * gaussian;
                            let d_power = d_alpha * opacity * gaussian;
                            grad[0] = -d_power * (conic.x * d.x + conic.y * d.y);
                            grad[1] = -d_power * (conic.y * d.x + conic.z * d.y);
                            grad[2] = -0.5 * d_power * d.x * d.x;
                            grad[3] = -d_power * d.x * d.y;
                            grad[4] = -0.5 * d_power * d.y * d.y;
                        }
                    }
                }
            }
            // Sum over the tile's 256 threads.
            partial[thread] = grad;
            workgroupBarrier();
            for (var half = 128u; half > 0u; half >>= 1u) {
                if thread < half {
                    for (var c = 0u; c < GRAD_STRIDE; c++) {
                        partial[thread][c] += partial[thread + half][c];
                    }
                }
                workgroupBarrier();
            }
            if thread == 0u {
                let base = batch_slot[k] * GRAD_STRIDE;
                for (var c = 0u; c < GRAD_STRIDE; c++) {
                    intersection_gradients[base + c] = partial[0][c];
                }
            }
            workgroupBarrier();
        }
    }
}
