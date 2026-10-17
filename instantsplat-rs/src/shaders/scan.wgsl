// Exclusive prefix sum of a u32 array, by one workgroup looping over 256-element chunks.
// Two entry points: over the per-Gaussian tile counts (which also sizes the intersection
// list and the indirect dispatches that follow), and over the sort histograms.

const SORT_BLOCK: u32 = 1024u;
const SORT_BINS: u32 = 16u;

struct ScanControl {
    length: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> control: ScanControl;
@group(0) @binding(3) var<storage, read_write> intersections: IntersectionMeta;
// Workgroup counts: [0..3) the sort blocks, [3..6) 256-thread groups over the intersections.
@group(0) @binding(4) var<storage, read_write> indirect: array<u32>;

var<workgroup> temp: array<u32, 256>;
var<workgroup> carry: u32;
// The length comes from a read-write buffer, which WebGPU's uniformity analysis takes as
// possibly differing between threads; the loop over the chunks, with its barriers, needs
// a uniform one, so it goes through workgroup memory and `workgroupUniformLoad`.
var<workgroup> shared_length: u32;

fn exclusive_scan(length_from_buffer: u32, thread: u32) -> u32 {
    if thread == 0u {
        carry = 0u;
        shared_length = length_from_buffer;
    }
    // Includes a workgroup barrier.
    let length = workgroupUniformLoad(&shared_length);
    let chunks = (length + 255u) / 256u;
    for (var chunk = 0u; chunk < chunks; chunk++) {
        let i = chunk * 256u + thread;
        var value = 0u;
        if i < length {
            value = input[i];
        }
        temp[thread] = value;
        workgroupBarrier();
        for (var offset = 1u; offset < 256u; offset <<= 1u) {
            var add = 0u;
            if thread >= offset {
                add = temp[thread - offset];
            }
            workgroupBarrier();
            temp[thread] += add;
            workgroupBarrier();
        }
        if i < length {
            output[i] = carry + temp[thread] - value;
        }
        workgroupBarrier();
        if thread == 255u {
            carry += temp[255u];
        }
        workgroupBarrier();
    }
    return carry;
}

@compute @workgroup_size(256)
fn scan_tile_counts(@builtin(local_invocation_id) local: vec3<u32>) {
    let total = exclusive_scan(control.length, local.x);
    if local.x == 0u {
        control.total = total;
        let count = min(total, intersections.capacity);
        intersections.count = count;
        intersections.overflow = u32(total > intersections.capacity);
        intersections.sort_blocks = (count + SORT_BLOCK - 1u) / SORT_BLOCK;
        intersections.histogram_length = SORT_BINS * intersections.sort_blocks;
        indirect[0] = intersections.sort_blocks;
        indirect[1] = 1u;
        indirect[2] = 1u;
        indirect[3] = (count + 255u) / 256u;
        indirect[4] = 1u;
        indirect[5] = 1u;
    }
}

@compute @workgroup_size(256)
fn scan_sort_histograms(@builtin(local_invocation_id) local: vec3<u32>) {
    let total = exclusive_scan(intersections.histogram_length, local.x);
    if local.x == 0u {
        control.total = total;
    }
}
