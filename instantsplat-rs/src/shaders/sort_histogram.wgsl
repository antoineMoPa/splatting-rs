// Radix sort, 4 bits per pass: per block of SORT_BLOCK keys, how many fall in each of the
// 16 bins of the current digit. Written bin-major (bin * blocks + block) so that an
// exclusive scan over the whole array gives every (bin, block) its first output slot.

const SORT_BLOCK: u32 = 1024u;
const SORT_BINS: u32 = 16u;
const KEYS_PER_THREAD: u32 = SORT_BLOCK / 256u;

struct SortPass {
    shift: u32,
};

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> intersections: IntersectionMeta;
@group(0) @binding(2) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(3) var<uniform> pass_info: SortPass;

var<workgroup> local_counts: array<atomic<u32>, SORT_BINS>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let block = group.x;
    let thread = local.x;
    if thread < SORT_BINS {
        atomicStore(&local_counts[thread], 0u);
    }
    workgroupBarrier();
    for (var k = 0u; k < KEYS_PER_THREAD; k++) {
        let i = block * SORT_BLOCK + k * 256u + thread;
        if i < intersections.count {
            let digit = (keys[i] >> pass_info.shift) & (SORT_BINS - 1u);
            atomicAdd(&local_counts[digit], 1u);
        }
    }
    workgroupBarrier();
    if thread < SORT_BINS {
        histograms[thread * intersections.sort_blocks + block] = atomicLoad(&local_counts[thread]);
    }
}
