//! wgpu device setup and the few buffer helpers every kernel needs. Compute only: Metal
//! natively, WebGPU in a browser, with the same WGSL. Read backs are async because a
//! browser cannot block on the GPU; natively `read_back` wraps them in `pollster::block_on`.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use bytemuck::Pod;
use wgpu::util::DeviceExt;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

/// Largest buffer the kernels may allocate; the adapter must allow it.
const MAX_BUFFER_BYTES: u64 = 1 << 30;

/// Whether wgpu adds its own bounds checks to every buffer access of the kernels. On Metal
/// they made a training step 4x slower (158 ms against 36 ms on an M5, `examples/step_timing`),
/// so native release builds go without; the crate's tests and debug builds keep them so
/// that an out-of-bounds index still shows. The browser applies its own checks either way.
const SHADER_RUNTIME_CHECKS: bool = cfg!(any(test, debug_assertions, target_arch = "wasm32"));

impl Gpu {
    /// Requests the adapter and device, blocking; in a browser use `request_async`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn request() -> Self {
        pollster::block_on(Self::request_async())
    }

    pub async fn request_async() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no wgpu adapter");
        let adapter_limits = adapter.limits();
        // WebGPU's baseline limits, except that buffers may be as large as the adapter allows.
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size.min(MAX_BUFFER_BYTES),
            max_storage_buffer_binding_size: adapter_limits
                .max_storage_buffer_binding_size
                .min(MAX_BUFFER_BYTES),
            ..wgpu::Limits::defaults()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("instantsplat"),
                required_limits,
                ..Default::default()
            })
            .await
            .expect("wgpu device");
        let adapter_info = adapter.get_info();
        println!("wgpu adapter: {} ({:?})", adapter_info.name, adapter_info.backend);
        Self {
            device,
            queue,
            adapter_info,
        }
    }

    /// Zeroed storage buffer of `bytes` bytes, readable and writable by kernels and copies.
    pub fn storage_buffer(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        assert!(bytes > 0, "{label}: empty buffer");
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn storage_buffer_from<T: Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        assert!(!data.is_empty(), "{label}: empty buffer");
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    pub fn uniform_buffer<T: Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Dispatch arguments written by kernels: `u32` triples.
    pub fn indirect_buffer(&self, label: &str, triples: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (triples * 3 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn write<T: Pod>(&self, buffer: &wgpu::Buffer, data: &[T]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    /// Compiles `wgsl` and builds the pipeline of its entry point `entry` (bind group layout
    /// inferred from the shader).
    pub fn compute_pipeline(&self, label: &str, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
        let descriptor = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        };
        let module = if SHADER_RUNTIME_CHECKS {
            self.device.create_shader_module(descriptor)
        } else {
            // SAFETY: the kernels are ours and index within the buffers they are given; the
            // tests and debug builds run them with the checks (`SHADER_RUNTIME_CHECKS`).
            unsafe { self.device.create_shader_module_trusted(descriptor, wgpu::ShaderRuntimeChecks::unchecked()) }
        };
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Bind group 0 of `pipeline`, `buffers` in binding order.
    pub fn bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit([encoder.finish()]);
    }

    /// Copies the first `count` elements of `buffer` to the CPU, waiting for all submitted work.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_back<T: Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
        pollster::block_on(self.read_back_async(buffer, count))
    }

    pub async fn read_back_async<T: Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
        let bytes = (count * std::mem::size_of::<T>()) as u64;
        assert!(bytes <= buffer.size(), "read past the end of a buffer");
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.encoder();
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.submit(encoder);
        let mapped = MapReady::default();
        let callback = mapped.clone();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| callback.set(result));
        // Natively this returns once the GPU is done and the map callback has run; in a
        // browser the device is polled by the page's event loop and this returns at once.
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the GPU");
        mapped.await.expect("mapping the read back buffer");
        let data = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped")).to_vec();
        staging.unmap();
        data
    }
}

/// Resolves once a `map_async` callback has run: the callback stores the result and wakes
/// whoever awaits.
#[derive(Clone, Default)]
struct MapReady(Arc<Mutex<MapState>>);

#[derive(Default)]
struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

impl MapReady {
    fn set(&self, result: Result<(), wgpu::BufferAsyncError>) {
        let mut state = self.0.lock().expect("map state");
        assert!(state.result.is_none(), "map callback ran twice");
        state.result = Some(result);
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }
}

impl Future for MapReady {
    type Output = Result<(), wgpu::BufferAsyncError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.0.lock().expect("map state");
        match state.result.take() {
            Some(result) => Poll::Ready(result),
            None => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Workgroups for `count` items at `workgroup_size` each.
pub fn workgroups(count: usize, workgroup_size: u32) -> u32 {
    let groups = (count as u64).div_ceil(workgroup_size as u64);
    assert!(groups <= 65_535, "{count} items need {groups} workgroups, over the 65535 limit");
    groups as u32
}

/// Runs one compute pass of `pipeline` with `bind_group`.
pub fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Like `dispatch`, with the workgroup count read from `indirect` at `triple_index`.
pub fn dispatch_indirect(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    indirect: &wgpu::Buffer,
    triple_index: u64,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups_indirect(indirect, triple_index * 12);
}
