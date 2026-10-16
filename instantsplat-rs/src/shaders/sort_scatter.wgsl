// Radix sort, 4 bits per pass: moves each key (and its value) of a block to its slot,
// stably. Keys are taken 256 at a time; within such a wave, a key's rank among the keys
// with the same digit comes from a prefix sum over 16 per-thread counters, packed as 16-bit
// lanes in 8 u32s. Earlier waves of the block add their per-digit totals.

const SORT_BLOCK: u32 = 1024u;
const SORT_BINS: u32 = 16u;
const WAVES: u32 = SORT_BLOCK / 256u;

struct SortPass {
    shift: u32,
};

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<storage, read> intersections: IntersectionMeta;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(6) var<uniform> pass_info: SortPass;

var<workgroup> packed: array<array<u32, 8>, 256>;
var<workgroup> bin_base: array<u32, SORT_BINS>;
var<workgroup> bin_done: array<u32, SORT_BINS>;

fn lane(counters: array<u32, 8>, digit: u32) -> u32 {
    return (counters[digit / 2u] >> ((digit % 2u) * 16u)) & 0xFFFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let block = group.x;
    let thread = local.x;
    if thread < SORT_BINS {
        bin_base[thread] = bin_offsets[thread * intersections.sort_blocks + block];
        bin_done[thread] = 0u;
    }
    workgroupBarrier();

    for (var wave = 0u; wave < WAVES; wave++) {
        let i = block * SORT_BLOCK + wave * 256u + thread;
        let valid = i < intersections.count;
        var key = 0u;
        if valid {
            key = keys_in[i];
        }
        let digit = (key >> pass_info.shift) & (SORT_BINS - 1u);
        var counters = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        if valid {
            counters[digit / 2u] = 1u << ((digit % 2u) * 16u);
        }
        packed[thread] = counters;
        workgroupBarrier();
        // Inclusive prefix sum over the 256 threads, all 16 lanes at once.
        for (var offset = 1u; offset < 256u; offset <<= 1u) {
            var add = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
            if thread >= offset {
                add = packed[thread - offset];
            }
            workgroupBarrier();
            for (var k = 0u; k < 8u; k++) {
                packed[thread][k] += add[k];
            }
            workgroupBarrier();
        }
        if valid {
            let rank = lane(packed[thread], digit) - 1u;
            let destination = bin_base[digit] + bin_done[digit] + rank;
            keys_out[destination] = key;
            values_out[destination] = values_in[i];
        }
        workgroupBarrier();
        if thread < SORT_BINS {
            bin_done[thread] += lane(packed[255u], thread);
        }
        workgroupBarrier();
    }
}
