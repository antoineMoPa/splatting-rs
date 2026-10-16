// Writes one (tile, depth) key per tile each Gaussian touches, at the Gaussian's offset in
// the intersection list. The value sorted along with the key is the entry's own index, so
// that after sorting, each entry still finds its Gaussian and its gradient slot.

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> intersections: IntersectionMeta;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> values: array<u32>;
@group(0) @binding(6) var<storage, read_write> intersection_gaussian: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = id.x;
    if g >= arrayLength(&tile_offsets) {
        return;
    }
    let p = projected[g];
    if p.radius == 0u {
        return;
    }
    // Positive depths sort like their bit patterns.
    let depth_bits = bitcast<u32>(p.depth) >> (32u - DEPTH_BITS);
    var index = tile_offsets[g];
    for (var ty = p.tile_min.y; ty < p.tile_max.y; ty++) {
        for (var tx = p.tile_min.x; tx < p.tile_max.x; tx++) {
            if index < intersections.capacity {
                keys[index] = ((ty * camera.tiles_x + tx) << DEPTH_BITS) | depth_bits;
                values[index] = index;
                intersection_gaussian[index] = g;
            }
            index++;
        }
    }
}
