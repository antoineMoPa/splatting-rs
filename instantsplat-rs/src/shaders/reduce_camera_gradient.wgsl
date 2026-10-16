// Adds up the per-workgroup camera gradient partials of project_backward into the one
// CAMERA_GRAD_STRIDE gradient of the view: one workgroup, each thread summing every
// 256th partial, then a tree reduction. Deterministic, no atomics.

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> gradient: array<f32>;

var<workgroup> partial: array<array<f32, CAMERA_GRAD_STRIDE>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) thread: u32) {
    let count = arrayLength(&partials) / CAMERA_GRAD_STRIDE;
    var sum: array<f32, CAMERA_GRAD_STRIDE>;
    for (var i = thread; i < count; i += 256u) {
        for (var c = 0u; c < CAMERA_GRAD_STRIDE; c++) {
            sum[c] += partials[i * CAMERA_GRAD_STRIDE + c];
        }
    }
    partial[thread] = sum;
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
            gradient[c] = partial[0][c];
        }
    }
}
