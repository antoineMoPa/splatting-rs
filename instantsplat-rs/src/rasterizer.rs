//! Differentiable tile rasterizer of 3D Gaussians on wgpu: projection, tile binning, depth
//! sort, alpha compositing over nothing, the loss against a ground-truth view (`loss.rs`),
//! and the backward pass of all of it, down to gradients of the packed Gaussian parameters.
//! One instance renders one image size against one set of ground-truth views. The
//! backward pass also sums the gradient with respect to the camera (`CameraGradient`), and
//! adds up per Gaussian what densification decides on (`DensifyStats`).
//!
//! The read backs (`read_*_async`, `render_async`) are async so that a browser can await
//! them; the blocking `read_*` and `render` are native-only `pollster::block_on` wrappers.

use bytemuck::{Pod, Zeroable};

use crate::camera::{CameraGradient, PinholeCamera};
use crate::gpu::{Gpu, dispatch, dispatch_indirect, workgroups};
use crate::loss::{GROUND_TRUTH_STRIDE, LOSS_WEIGHTS, LossTerms, LossView};
use crate::shaders::{self, CAMERA_GRAD_STRIDE, GRAD_STRIDE, PARAM_STRIDE, PROJECTED_BYTES, TILE};
use crate::sort::RadixSort;

/// Composited under the renders for display (`Rendered::to_rgb8`): the mid-gray
/// mapanything-rs puts under cutouts, so that render and input PNGs compare. The loss
/// never sees it.
pub const BACKGROUND: [f32; 3] = [0.50196; 3];

/// Tile keys are tile index << DEPTH_BITS, so this many tiles at most.
const MAX_TILES: u32 = 1 << (32 - shaders::DEPTH_BITS);
/// (Gaussian, tile) pairs the intersection list can hold, per Gaussian.
const INTERSECTIONS_PER_GAUSSIAN: usize = 24;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct IntersectionMeta {
    pub count: u32,
    pub sort_blocks: u32,
    pub histogram_length: u32,
    pub overflow: u32,
    pub capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ScanControl {
    length: u32,
    total: u32,
}

/// Mirror of the shaders' `Projected`, for tests and diagnostics.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Projected {
    pub mean: [f32; 2],
    _pad0: [f32; 2],
    pub conic: [f32; 3],
    pub depth: f32,
    pub color: [f32; 3],
    pub opacity: f32,
    pub radius: u32,
    _pad1: u32,
    pub tile_min: [u32; 2],
    pub tile_max: [u32; 2],
    _pad2: [u32; 2],
}

/// Mirror of one Gaussian's `densify_stats` in `project_backward.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct DensifyStats {
    /// Sum over the backward passes of the norm of the 2D mean gradient in normalized
    /// device coordinates (pixel gradient times half the image size).
    pub mean_gradient_sum: f32,
    /// Backward passes the Gaussian was visible in.
    pub visible_count: f32,
    /// Largest screen radius in pixels over those passes.
    pub max_radius: f32,
    pub(crate) _pad: f32,
}

/// One rendered view: per pixel, the Gaussians' color premultiplied by the pixel's
/// coverage (over nothing) and the remaining transmittance.
pub struct Rendered {
    pub width: u32,
    pub height: u32,
    /// [height * width * 4]
    pub rgbt: Vec<f32>,
}

impl Rendered {
    pub fn premultiplied(&self, x: u32, y: u32) -> [f32; 3] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgbt[i], self.rgbt[i + 1], self.rgbt[i + 2]]
    }

    pub fn transmittance(&self, x: u32, y: u32) -> f32 {
        self.rgbt[((y * self.width + x) * 4 + 3) as usize]
    }

    /// Coverage, 1 - transmittance.
    pub fn alpha(&self, x: u32, y: u32) -> f32 {
        1.0 - self.transmittance(x, y)
    }

    pub fn over_background(&self, x: u32, y: u32) -> [f32; 3] {
        let premultiplied = self.premultiplied(x, y);
        let transmittance = self.transmittance(x, y);
        [0, 1, 2].map(|c| premultiplied[c] + transmittance * BACKGROUND[c])
    }

    fn to_u8(value: f32) -> u8 {
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    /// Over `BACKGROUND`.
    pub fn to_rgb8(&self) -> image::RgbImage {
        image::RgbImage::from_fn(self.width, self.height, |x, y| {
            image::Rgb(self.over_background(x, y).map(Self::to_u8))
        })
    }

    /// Straight (not premultiplied) color with the coverage as alpha.
    pub fn to_rgba8(&self) -> image::RgbaImage {
        image::RgbaImage::from_fn(self.width, self.height, |x, y| {
            let alpha = self.alpha(x, y);
            let premultiplied = self.premultiplied(x, y);
            let straight = if alpha > 0.0 { premultiplied.map(|c| c / alpha) } else { [0.0; 3] };
            image::Rgba([
                Self::to_u8(straight[0]),
                Self::to_u8(straight[1]),
                Self::to_u8(straight[2]),
                Self::to_u8(alpha),
            ])
        })
    }
}

pub struct Rasterizer {
    pub width: u32,
    pub height: u32,
    pub gaussian_count: usize,
    pub intersection_capacity: usize,
    camera: wgpu::Buffer,
    meta: wgpu::Buffer,
    indirect: wgpu::Buffer,
    ranges: wgpu::Buffer,
    image: wgpu::Buffer,
    loss_gradient: wgpu::Buffer,
    pixel_loss: wgpu::Buffer,
    param_gradients: wgpu::Buffer,
    camera_gradient: wgpu::Buffer,
    densify_stats: wgpu::Buffer,
    project_forward: wgpu::ComputePipeline,
    scan_tile_counts: wgpu::ComputePipeline,
    emit_tile_keys: wgpu::ComputePipeline,
    tile_ranges: wgpu::ComputePipeline,
    rasterize_forward: wgpu::ComputePipeline,
    ssim_forward: wgpu::ComputePipeline,
    loss: wgpu::ComputePipeline,
    rasterize_backward: wgpu::ComputePipeline,
    project_backward: wgpu::ComputePipeline,
    reduce_camera_gradient: wgpu::ComputePipeline,
    project_forward_group: wgpu::BindGroup,
    scan_tile_counts_group: wgpu::BindGroup,
    emit_tile_keys_group: wgpu::BindGroup,
    tile_ranges_group: wgpu::BindGroup,
    rasterize_forward_group: wgpu::BindGroup,
    /// Per ground-truth view: the SSIM partials kernel's and the loss kernel's.
    loss_groups: Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    rasterize_backward_group: wgpu::BindGroup,
    project_backward_group: wgpu::BindGroup,
    reduce_camera_gradient_group: wgpu::BindGroup,
    sort: RadixSort,
}

impl Rasterizer {
    /// `params` holds `gaussian_count * PARAM_STRIDE` floats; `ground_truth` holds
    /// `view_count` images of `width * height * GROUND_TRUTH_STRIDE` floats
    /// (`loss::ground_truth_rgba`).
    pub fn new(
        gpu: &Gpu,
        params: &wgpu::Buffer,
        gaussian_count: usize,
        width: u32,
        height: u32,
        ground_truth: &wgpu::Buffer,
        view_count: usize,
    ) -> Self {
        assert!(gaussian_count > 0, "no Gaussians");
        let tiles_x = width.div_ceil(TILE);
        let tiles_y = height.div_ceil(TILE);
        assert!(
            tiles_x * tiles_y <= MAX_TILES,
            "{width}x{height} has {} tiles, over the {MAX_TILES} the sort keys allow",
            tiles_x * tiles_y
        );
        assert_eq!(
            params.size(),
            (gaussian_count * PARAM_STRIDE * 4) as u64,
            "params buffer size"
        );
        let pixel_count = (width * height) as usize;
        assert_eq!(
            ground_truth.size(),
            (view_count * pixel_count * GROUND_TRUTH_STRIDE * 4) as u64,
            "ground truth buffer size"
        );
        let intersection_capacity = gaussian_count * INTERSECTIONS_PER_GAUSSIAN;

        let camera = gpu.uniform_buffer(
            "camera",
            &PinholeCamera {
                rotation: nalgebra::Matrix3::identity(),
                translation: nalgebra::Vector3::zeros(),
                focal: [1.0, 1.0],
                principal_point: [0.0, 0.0],
                width,
                height,
            }
            .uniform(TILE),
        );
        let projected = gpu.storage_buffer("projected", gaussian_count as u64 * PROJECTED_BYTES);
        let tile_counts = gpu.storage_buffer("tile counts", gaussian_count as u64 * 4);
        let tile_offsets = gpu.storage_buffer("tile offsets", gaussian_count as u64 * 4);
        let scan_control = gpu.storage_buffer_from(
            "scan control",
            &[ScanControl {
                length: gaussian_count as u32,
                total: 0,
            }],
        );
        let meta = gpu.storage_buffer_from(
            "intersection meta",
            &[IntersectionMeta {
                capacity: intersection_capacity as u32,
                ..Default::default()
            }],
        );
        let indirect = gpu.indirect_buffer("indirect dispatches", 2);
        let keys = [
            gpu.storage_buffer("keys a", intersection_capacity as u64 * 4),
            gpu.storage_buffer("keys b", intersection_capacity as u64 * 4),
        ];
        let values = [
            gpu.storage_buffer("values a", intersection_capacity as u64 * 4),
            gpu.storage_buffer("values b", intersection_capacity as u64 * 4),
        ];
        let intersection_gaussian =
            gpu.storage_buffer("intersection gaussian", intersection_capacity as u64 * 4);
        let ranges = gpu.storage_buffer("tile ranges", (tiles_x * tiles_y) as u64 * 2 * 4);
        let image = gpu.storage_buffer("image", pixel_count as u64 * 4 * 4);
        let contributions = gpu.storage_buffer("contributions", pixel_count as u64 * 4);
        let ssim_partials = gpu.storage_buffer("ssim partials", pixel_count as u64 * 3 * 4 * 4);
        let loss_gradient = gpu.storage_buffer("loss gradient", pixel_count as u64 * 4 * 4);
        let pixel_loss = gpu.storage_buffer("pixel loss", pixel_count as u64 * 4 * 4);
        let intersection_gradients = gpu.storage_buffer(
            "intersection gradients",
            (intersection_capacity * GRAD_STRIDE * 4) as u64,
        );
        let param_gradients =
            gpu.storage_buffer("param gradients", (gaussian_count * PARAM_STRIDE * 4) as u64);
        let camera_partials = gpu.storage_buffer(
            "camera gradient partials",
            workgroups(gaussian_count, 256) as u64 * CAMERA_GRAD_STRIDE as u64 * 4,
        );
        let camera_gradient = gpu.storage_buffer("camera gradient", (CAMERA_GRAD_STRIDE * 4) as u64);
        let densify_stats = gpu.storage_buffer("densify stats", gaussian_count as u64 * 4 * 4);

        let project_forward =
            gpu.compute_pipeline("project forward", &shaders::project_forward(), "main");
        let scan_tile_counts =
            gpu.compute_pipeline("scan tile counts", &shaders::scan(), "scan_tile_counts");
        let emit_tile_keys = gpu.compute_pipeline("emit tile keys", &shaders::emit_tile_keys(), "main");
        let tile_ranges = gpu.compute_pipeline("tile ranges", &shaders::tile_ranges(), "main");
        let rasterize_forward =
            gpu.compute_pipeline("rasterize forward", &shaders::rasterize_forward(), "main");
        let ssim_forward = gpu.compute_pipeline("ssim forward", &shaders::ssim_forward(), "main");
        let loss = gpu.compute_pipeline("loss", &shaders::loss(), "main");
        let rasterize_backward =
            gpu.compute_pipeline("rasterize backward", &shaders::rasterize_backward(), "main");
        let project_backward =
            gpu.compute_pipeline("project backward", &shaders::project_backward(), "main");
        let reduce_camera_gradient =
            gpu.compute_pipeline("reduce camera gradient", &shaders::reduce_camera_gradient(), "main");

        let project_forward_group =
            gpu.bind_group(&project_forward, &[&camera, params, &projected, &tile_counts]);
        let scan_tile_counts_group = gpu.bind_group(
            &scan_tile_counts,
            &[&tile_counts, &tile_offsets, &scan_control, &meta, &indirect],
        );
        let emit_tile_keys_group = gpu.bind_group(
            &emit_tile_keys,
            &[
                &camera,
                &projected,
                &tile_offsets,
                &meta,
                &keys[0],
                &values[0],
                &intersection_gaussian,
            ],
        );
        let tile_ranges_group = gpu.bind_group(&tile_ranges, &[&keys[0], &meta, &ranges]);
        let rasterize_forward_group = gpu.bind_group(
            &rasterize_forward,
            &[
                &camera,
                &projected,
                &values[0],
                &intersection_gaussian,
                &ranges,
                &image,
                &contributions,
            ],
        );
        let loss_groups = (0..view_count)
            .map(|view| {
                let uniform = gpu.uniform_buffer(
                    "loss view",
                    &LossView {
                        pixel_count: pixel_count as u32,
                        width,
                        height,
                        ground_truth_offset: (view * pixel_count * GROUND_TRUTH_STRIDE) as u32,
                        l1_weight: LOSS_WEIGHTS.l1,
                        dssim_weight: LOSS_WEIGHTS.dssim,
                        alpha_weight: LOSS_WEIGHTS.alpha,
                        _pad: 0.0,
                    },
                );
                (
                    gpu.bind_group(&ssim_forward, &[&image, ground_truth, &ssim_partials, &uniform]),
                    gpu.bind_group(
                        &loss,
                        &[&image, ground_truth, &ssim_partials, &loss_gradient, &pixel_loss, &uniform],
                    ),
                )
            })
            .collect();
        let rasterize_backward_group = gpu.bind_group(
            &rasterize_backward,
            &[
                &camera,
                &projected,
                &values[0],
                &intersection_gaussian,
                &ranges,
                &image,
                &contributions,
                &loss_gradient,
                &intersection_gradients,
            ],
        );
        let project_backward_group = gpu.bind_group(
            &project_backward,
            &[
                &camera,
                params,
                &projected,
                &tile_offsets,
                &intersection_gradients,
                &param_gradients,
                &camera_partials,
                &densify_stats,
            ],
        );
        let reduce_camera_gradient_group =
            gpu.bind_group(&reduce_camera_gradient, &[&camera_partials, &camera_gradient]);
        let sort = RadixSort::new(
            gpu,
            [&keys[0], &keys[1]],
            [&values[0], &values[1]],
            &meta,
            &scan_control,
            intersection_capacity,
        );

        Self {
            width,
            height,
            gaussian_count,
            intersection_capacity,
            camera,
            meta,
            indirect,
            ranges,
            image,
            loss_gradient,
            pixel_loss,
            param_gradients,
            camera_gradient,
            densify_stats,
            project_forward,
            scan_tile_counts,
            emit_tile_keys,
            tile_ranges,
            rasterize_forward,
            ssim_forward,
            loss,
            rasterize_backward,
            project_backward,
            reduce_camera_gradient,
            project_forward_group,
            scan_tile_counts_group,
            emit_tile_keys_group,
            tile_ranges_group,
            rasterize_forward_group,
            loss_groups,
            rasterize_backward_group,
            project_backward_group,
            reduce_camera_gradient_group,
            sort,
        }
    }

    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// The camera of the next forward and backward passes. Takes effect at the next submit.
    pub fn set_camera(&self, gpu: &Gpu, camera: &PinholeCamera) {
        assert_eq!(
            (camera.width, camera.height),
            (self.width, self.height),
            "camera image size"
        );
        gpu.write(&self.camera, &[camera.uniform(TILE)]);
    }

    /// Projection, binning, sort and compositing into the image buffer.
    pub fn encode_forward(&self, encoder: &mut wgpu::CommandEncoder) {
        let gaussian_groups = workgroups(self.gaussian_count, 256);
        dispatch(encoder, &self.project_forward, &self.project_forward_group, gaussian_groups);
        dispatch(encoder, &self.scan_tile_counts, &self.scan_tile_counts_group, 1);
        dispatch(encoder, &self.emit_tile_keys, &self.emit_tile_keys_group, gaussian_groups);
        self.sort.encode(encoder, &self.indirect);
        encoder.clear_buffer(&self.ranges, 0, None);
        dispatch_indirect(encoder, &self.tile_ranges, &self.tile_ranges_group, &self.indirect, 1);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.rasterize_forward);
        pass.set_bind_group(0, &self.rasterize_forward_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(TILE), self.height.div_ceil(TILE), 1);
    }

    /// Loss of the image buffer against ground-truth view `view` (`LOSS_WEIGHTS`): fills
    /// the loss gradient (dL/d(rgb, transmittance) per pixel) and per-pixel loss buffers.
    pub fn encode_loss(&self, encoder: &mut wgpu::CommandEncoder, view: usize) {
        let (ssim_group, loss_group) = &self.loss_groups[view];
        let groups = workgroups(self.pixel_count(), 256);
        dispatch(encoder, &self.ssim_forward, ssim_group, groups);
        dispatch(encoder, &self.loss, loss_group, groups);
    }

    /// Gradient of the loss with respect to the packed parameters and to the camera, from
    /// the loss gradient buffer and the state of the last forward pass.
    pub fn encode_backward(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.rasterize_backward);
        pass.set_bind_group(0, &self.rasterize_backward_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(TILE), self.height.div_ceil(TILE), 1);
        drop(pass);
        dispatch(
            encoder,
            &self.project_backward,
            &self.project_backward_group,
            workgroups(self.gaussian_count, 256),
        );
        dispatch(encoder, &self.reduce_camera_gradient, &self.reduce_camera_gradient_group, 1);
    }

    /// dL/d(rgb, transmittance) per pixel, [height * width * 4], in place of what
    /// `encode_loss` would write.
    pub fn write_loss_gradient(&self, gpu: &Gpu, gradient: &[f32]) {
        assert_eq!(gradient.len(), self.pixel_count() * 4);
        gpu.write(&self.loss_gradient, gradient);
    }

    /// Premultiplied rgb and transmittance per pixel, [height * width * 4], in place of
    /// what `encode_forward` would render (for testing the loss kernels).
    pub fn write_image(&self, gpu: &Gpu, rgbt: &[f32]) {
        assert_eq!(rgbt.len(), self.pixel_count() * 4);
        gpu.write(&self.image, rgbt);
    }

    pub async fn read_loss_gradient_async(&self, gpu: &Gpu) -> Vec<f32> {
        gpu.read_back_async(&self.loss_gradient, self.pixel_count() * 4).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_loss_gradient(&self, gpu: &Gpu) -> Vec<f32> {
        pollster::block_on(self.read_loss_gradient_async(gpu))
    }

    pub fn param_gradients(&self) -> &wgpu::Buffer {
        &self.param_gradients
    }

    pub async fn read_param_gradients_async(&self, gpu: &Gpu) -> Vec<f32> {
        gpu.read_back_async(&self.param_gradients, self.gaussian_count * PARAM_STRIDE).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_param_gradients(&self, gpu: &Gpu) -> Vec<f32> {
        pollster::block_on(self.read_param_gradients_async(gpu))
    }

    /// Per Gaussian, the densification statistics summed by every `encode_backward` since
    /// the rasterizer was made: `DensifyStats`.
    pub async fn read_densify_stats_async(&self, gpu: &Gpu) -> Vec<DensifyStats> {
        gpu.read_back_async(&self.densify_stats, self.gaussian_count).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_densify_stats(&self, gpu: &Gpu) -> Vec<DensifyStats> {
        pollster::block_on(self.read_densify_stats_async(gpu))
    }

    /// The camera gradient of the last `encode_backward`.
    pub async fn read_camera_gradient_async(&self, gpu: &Gpu) -> CameraGradient {
        gpu.read_back_async::<CameraGradient>(&self.camera_gradient, 1).await[0]
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_camera_gradient(&self, gpu: &Gpu) -> CameraGradient {
        pollster::block_on(self.read_camera_gradient_async(gpu))
    }

    pub async fn read_image_async(&self, gpu: &Gpu) -> Rendered {
        Rendered {
            width: self.width,
            height: self.height,
            rgbt: gpu.read_back_async(&self.image, self.pixel_count() * 4).await,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_image(&self, gpu: &Gpu) -> Rendered {
        pollster::block_on(self.read_image_async(gpu))
    }

    /// Per pixel, the loss terms of the last `encode_loss`: L1 rgb, alpha error, 1 - SSIM,
    /// weighted total (unweighted means are the loss).
    pub async fn read_pixel_loss_async(&self, gpu: &Gpu) -> Vec<f32> {
        gpu.read_back_async(&self.pixel_loss, self.pixel_count() * 4).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_pixel_loss(&self, gpu: &Gpu) -> Vec<f32> {
        pollster::block_on(self.read_pixel_loss_async(gpu))
    }

    /// Means over the pixels of the loss terms of the last `encode_loss`.
    pub async fn read_loss_async(&self, gpu: &Gpu) -> LossTerms {
        LossTerms::from_pixel_loss(&self.read_pixel_loss_async(gpu).await)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_loss(&self, gpu: &Gpu) -> LossTerms {
        pollster::block_on(self.read_loss_async(gpu))
    }

    /// Panics when the last forward pass had more (Gaussian, tile) pairs than the list holds.
    pub async fn read_intersections_async(&self, gpu: &Gpu) -> IntersectionMeta {
        let meta: Vec<IntersectionMeta> = gpu.read_back_async(&self.meta, 1).await;
        assert_eq!(
            meta[0].overflow, 0,
            "intersection list overflow: more than {} (Gaussian, tile) pairs; Gaussians are too large for the image",
            self.intersection_capacity
        );
        meta[0]
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_intersections(&self, gpu: &Gpu) -> IntersectionMeta {
        pollster::block_on(self.read_intersections_async(gpu))
    }

    /// Renders `camera` and reads the image back.
    pub async fn render_async(&self, gpu: &Gpu, camera: &PinholeCamera) -> Rendered {
        self.set_camera(gpu, camera);
        let mut encoder = gpu.encoder();
        self.encode_forward(&mut encoder);
        gpu.submit(encoder);
        self.read_intersections_async(gpu).await;
        self.read_image_async(gpu).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(&self, gpu: &Gpu, camera: &PinholeCamera) -> Rendered {
        pollster::block_on(self.render_async(gpu, camera))
    }
}
