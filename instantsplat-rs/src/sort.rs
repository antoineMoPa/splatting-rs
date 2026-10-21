//! GPU radix sort of the (tile, depth) keys of one render, 4 bits per pass, 8 passes over
//! the 32-bit keys. Keys and values ping-pong between two buffers each; after the even
//! number of passes the result is back in the first buffers.

use bytemuck::{Pod, Zeroable};

use crate::gpu::{Gpu, dispatch, dispatch_indirect};
use crate::shaders;

const PASSES: usize = 8;
const BITS_PER_PASS: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortPass {
    shift: u32,
    _pad: [u32; 3],
}

pub struct RadixSort {
    histogram: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    histogram_groups: Vec<wgpu::BindGroup>,
    scatter_groups: Vec<wgpu::BindGroup>,
    scan_group: wgpu::BindGroup,
}

impl RadixSort {
    /// `capacity` keys at most; `meta` is the `IntersectionMeta` the scan over the tile
    /// counts fills in, `control` a `ScanControl`.
    pub fn new(
        gpu: &Gpu,
        keys: [&wgpu::Buffer; 2],
        values: [&wgpu::Buffer; 2],
        meta: &wgpu::Buffer,
        control: &wgpu::Buffer,
        capacity: usize,
    ) -> Self {
        let blocks = capacity.div_ceil(shaders::SORT_BLOCK);
        let histogram_entries = (shaders::SORT_BINS * blocks * 4) as u64;
        let histograms = gpu.storage_buffer("sort histograms", histogram_entries);
        let bin_offsets = gpu.storage_buffer("sort bin offsets", histogram_entries);

        let histogram = gpu.compute_pipeline("sort histogram", &shaders::sort_histogram(), "main");
        let scatter = gpu.compute_pipeline("sort scatter", &shaders::sort_scatter(), "main");
        let scan = gpu.compute_pipeline("sort scan", &shaders::scan(), "scan_sort_histograms");

        let pass_uniforms: Vec<wgpu::Buffer> = (0..PASSES)
            .map(|pass| {
                gpu.uniform_buffer(
                    "sort pass",
                    &SortPass {
                        shift: pass as u32 * BITS_PER_PASS,
                        _pad: [0; 3],
                    },
                )
            })
            .collect();
        let histogram_groups = (0..PASSES)
            .map(|pass| gpu.bind_group(&histogram, &[keys[pass % 2], meta, &histograms, &pass_uniforms[pass]]))
            .collect();
        let scatter_groups = (0..PASSES)
            .map(|pass| {
                let (from, to) = (pass % 2, (pass + 1) % 2);
                gpu.bind_group(
                    &scatter,
                    &[
                        keys[from],
                        values[from],
                        meta,
                        &bin_offsets,
                        keys[to],
                        values[to],
                        &pass_uniforms[pass],
                    ],
                )
            })
            .collect();
        let scan_group = gpu.bind_group(&scan, &[&histograms, &bin_offsets, control, meta]);
        Self {
            histogram,
            scatter,
            scan,
            histogram_groups,
            scatter_groups,
            scan_group,
        }
    }

    /// `indirect` holds the sort block count as its first dispatch triple.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, indirect: &wgpu::Buffer) {
        for pass in 0..PASSES {
            dispatch_indirect(encoder, &self.histogram, &self.histogram_groups[pass], indirect, 0);
            dispatch(encoder, &self.scan, &self.scan_group, 1);
            dispatch_indirect(encoder, &self.scatter, &self.scatter_groups[pass], indirect, 0);
        }
    }
}
