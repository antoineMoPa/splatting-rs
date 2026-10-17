// Alpha-composites the depth-sorted Gaussians of each tile, front to back, one workgroup
// per tile and one thread per pixel. Gaussians are staged in workgroup memory 256 at a time.
// Outputs per pixel: the Gaussians' color (rgb, over nothing: premultiplied by the pixel's
// coverage), the remaining transmittance in the 4th channel (coverage = 1 - transmittance),
// and how many of the tile's Gaussians contributed (the last one included), for the backward.

const BATCH: u32 = 256u;

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> sorted_values: array<u32>;
@group(0) @binding(3) var<storage, read> intersection_gaussian: array<u32>;
@group(0) @binding(4) var<storage, read> ranges: array<u32>;
@group(0) @binding(5) var<storage, read_write> image: array<f32>;
@group(0) @binding(6) var<storage, read_write> contributions: array<u32>;

var<workgroup> batch_mean: array<vec2<f32>, BATCH>;
var<workgroup> batch_conic: array<vec3<f32>, BATCH>;
var<workgroup> batch_color: array<vec3<f32>, BATCH>;
var<workgroup> batch_opacity: array<f32, BATCH>;

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

    var transmittance = 1.0;
    var color = vec3<f32>(0.0);
    var contributed = 0u;
    var done = !inside;

    let count = end - start;
    let batches = (count + BATCH - 1u) / BATCH;
    for (var b = 0u; b < batches; b++) {
        workgroupBarrier();
        let j = b * BATCH + thread;
        if j < count {
            let g = intersection_gaussian[sorted_values[start + j]];
            let p = projected[g];
            batch_mean[thread] = p.mean;
            batch_conic[thread] = p.conic;
            batch_color[thread] = p.color;
            batch_opacity[thread] = p.opacity;
        }
        workgroupBarrier();
        // No break or continue on a per-pixel condition: WebGPU's uniformity analysis
        // would then refuse the barriers of the next batch. Finished pixels idle instead.
        let in_batch = min(BATCH, count - b * BATCH);
        for (var k = 0u; k < in_batch; k++) {
            if !done {
                let d = batch_mean[k] - pixel;
                let conic = batch_conic[k];
                let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
                if power <= 0.0 {
                    let alpha = min(MAX_ALPHA, batch_opacity[k] * exp(power));
                    if alpha >= MIN_ALPHA {
                        let next_transmittance = transmittance * (1.0 - alpha);
                        if next_transmittance < MIN_TRANSMITTANCE {
                            done = true;
                        } else {
                            color += batch_color[k] * alpha * transmittance;
                            transmittance = next_transmittance;
                            contributed = b * BATCH + k + 1u;
                        }
                    }
                }
            }
        }
    }

    if inside {
        let index = py * camera.width + px;
        image[index * 4u] = color.x;
        image[index * 4u + 1u] = color.y;
        image[index * 4u + 2u] = color.z;
        image[index * 4u + 3u] = transmittance;
        contributions[index] = contributed;
    }
}
