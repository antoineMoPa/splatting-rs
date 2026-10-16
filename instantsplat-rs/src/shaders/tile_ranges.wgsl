// After the sort, where each tile's run of intersections starts and ends (exclusive), as
// ranges[2 * tile] and ranges[2 * tile + 1]. The buffer is zeroed before, so tiles with no
// Gaussian keep an empty range.

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> intersections: IntersectionMeta;
@group(0) @binding(2) var<storage, read_write> ranges: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= intersections.count {
        return;
    }
    let tile = tile_of_key(keys[i]);
    if i == 0u || tile_of_key(keys[i - 1u]) != tile {
        ranges[2u * tile] = i;
    }
    if i + 1u == intersections.count || tile_of_key(keys[i + 1u]) != tile {
        ranges[2u * tile + 1u] = i + 1u;
    }
}
