// The loss of the render against one ground-truth view and its gradient per pixel:
//   l1_weight * L1(rgb) + dssim_weight * (1 - SSIM(rgb)) + alpha_weight * L1(coverage, mask)
// with the rgb terms over the three channels of the premultiplied render against the
// photo, on foreground pixels only (zero elsewhere), each averaged over channels and all
// pixels, and coverage = 1 - transmittance against the mask, averaged over pixels: a
// background pixel acts on the Gaussians through the alpha term alone. The gradient is
// with respect to what the forward pass stores: rgb and the transmittance (so the
// coverage gradient changes sign). The SSIM part convolves the (already masked) partials
// of ssim_forward.wgsl with the same window. Per pixel, the loss terms are written
// unweighted for reporting: L1 rgb, alpha error, 1 - SSIM, and their weighted sum.

struct LossView {
    pixel_count: u32,
    width: u32,
    height: u32,
    // First float of this view in `ground_truth`.
    ground_truth_offset: u32,
    l1_weight: f32,
    dssim_weight: f32,
    alpha_weight: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> image: array<f32>;
// Per pixel: rgb premultiplied by the foreground mask, then the mask.
@group(0) @binding(1) var<storage, read> ground_truth: array<f32>;
// Per (pixel, channel): SSIM, dSSIM/dmu_x, dSSIM/dE[x^2], dSSIM/dE[xy].
@group(0) @binding(2) var<storage, read> ssim_partials: array<f32>;
// Per pixel: dL/d(r, g, b, transmittance).
@group(0) @binding(3) var<storage, read_write> loss_gradient: array<f32>;
// Per pixel: L1 rgb, alpha error, 1 - SSIM, weighted total.
@group(0) @binding(4) var<storage, read_write> pixel_loss: array<f32>;
@group(0) @binding(5) var<uniform> view: LossView;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= view.pixel_count {
        return;
    }
    let pixels = f32(view.pixel_count);
    let gt = view.ground_truth_offset + i * 4u;
    let x = vec3<f32>(image[i * 4u], image[i * 4u + 1u], image[i * 4u + 2u]);
    let y = vec3<f32>(ground_truth[gt], ground_truth[gt + 1u], ground_truth[gt + 2u]);

    // SSIM: sum the partials of every pixel whose window contains this one.
    let taps = ssim_taps();
    let px = i32(i % view.width);
    let py = i32(i / view.width);
    var d_mu = vec3<f32>(0.0);
    var d_exx = vec3<f32>(0.0);
    var d_exy = vec3<f32>(0.0);
    for (var dy = -SSIM_RADIUS; dy <= SSIM_RADIUS; dy++) {
        let qy = py + dy;
        if qy < 0 || qy >= i32(view.height) {
            continue;
        }
        for (var dx = -SSIM_RADIUS; dx <= SSIM_RADIUS; dx++) {
            let qx = px + dx;
            if qx < 0 || qx >= i32(view.width) {
                continue;
            }
            let w = taps[dx + SSIM_RADIUS] * taps[dy + SSIM_RADIUS];
            let q = u32(qy) * view.width + u32(qx);
            for (var channel = 0u; channel < 3u; channel++) {
                let base = (q * 3u + channel) * 4u;
                d_mu[channel] += w * ssim_partials[base + 1u];
                d_exx[channel] += w * ssim_partials[base + 2u];
                d_exy[channel] += w * ssim_partials[base + 3u];
            }
        }
    }
    let d_ssim = d_mu + 2.0 * x * d_exx + y * d_exy;
    let mask = ground_truth[gt + 3u];
    var dssim = 0.0;
    for (var channel = 0u; channel < 3u; channel++) {
        dssim += mask * (1.0 - ssim_partials[(i * 3u + channel) * 4u]);
    }

    let difference = x - y;
    let d_rgb = (view.l1_weight * mask * sign(difference) - view.dssim_weight * d_ssim) / (3.0 * pixels);
    loss_gradient[i * 4u] = d_rgb.x;
    loss_gradient[i * 4u + 1u] = d_rgb.y;
    loss_gradient[i * 4u + 2u] = d_rgb.z;

    let coverage = 1.0 - image[i * 4u + 3u];
    let alpha_difference = coverage - ground_truth[gt + 3u];
    // d coverage / d transmittance = -1.
    loss_gradient[i * 4u + 3u] = -view.alpha_weight * sign(alpha_difference) / pixels;

    let l1 = mask * (abs(difference.x) + abs(difference.y) + abs(difference.z)) / 3.0;
    let alpha_error = abs(alpha_difference);
    pixel_loss[i * 4u] = l1;
    pixel_loss[i * 4u + 1u] = alpha_error;
    pixel_loss[i * 4u + 2u] = dssim / 3.0;
    pixel_loss[i * 4u + 3u] = view.l1_weight * l1 + view.alpha_weight * alpha_error + view.dssim_weight * dssim / 3.0;
}
