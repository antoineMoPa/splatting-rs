// SSIM of the render against one ground-truth view, per pixel and color channel, with the
// partial derivatives the loss kernel convolves for the backward. One thread per pixel,
// 121 taps of the 11x11 window read straight from the two images.
//
// With x the render and y the ground truth, the window's means mu_x, mu_y and raw moments
// E[x^2], E[y^2], E[xy]: sigma_x^2 = E[x^2] - mu_x^2, sigma_xy = E[xy] - mu_x mu_y, and
//   SSIM = (C D) / (A B) with A = mu_x^2 + mu_y^2 + C1, B = sigma_x^2 + sigma_y^2 + C2,
//   C = 2 mu_x mu_y + C1, D = 2 sigma_xy + C2.
// Written per (pixel, channel): SSIM, then dSSIM/dmu_x (E[x^2] and E[xy] held fixed),
// dSSIM/dE[x^2], dSSIM/dE[xy], the three times the pixel's foreground mask, since only
// foreground pixels' SSIM enters the loss. Each is a smooth function of the pixel's
// window, so the derivative with respect to a render pixel x_q is the window-weighted sum
// of these over the pixels whose window contains q: loss.wgsl does that convolution.

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
@group(0) @binding(2) var<storage, read_write> ssim_partials: array<f32>;
@group(0) @binding(3) var<uniform> view: LossView;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= view.pixel_count {
        return;
    }
    let taps = ssim_taps();
    let px = i32(i % view.width);
    let py = i32(i / view.width);
    var mu_x = vec3<f32>(0.0);
    var mu_y = vec3<f32>(0.0);
    var e_xx = vec3<f32>(0.0);
    var e_yy = vec3<f32>(0.0);
    var e_xy = vec3<f32>(0.0);
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
            let x = vec3<f32>(image[q * 4u], image[q * 4u + 1u], image[q * 4u + 2u]);
            let gt = view.ground_truth_offset + q * 4u;
            let y = vec3<f32>(ground_truth[gt], ground_truth[gt + 1u], ground_truth[gt + 2u]);
            mu_x += w * x;
            mu_y += w * y;
            e_xx += w * x * x;
            e_yy += w * y * y;
            e_xy += w * x * y;
        }
    }
    let sigma_xx = e_xx - mu_x * mu_x;
    let sigma_yy = e_yy - mu_y * mu_y;
    let sigma_xy = e_xy - mu_x * mu_y;
    let a = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
    let b = sigma_xx + sigma_yy + SSIM_C2;
    let c = 2.0 * mu_x * mu_y + SSIM_C1;
    let d = 2.0 * sigma_xy + SSIM_C2;
    let ab = a * b;
    let ssim = c * d / ab;
    // d/dmu_x: C by 2 mu_y, D by -2 mu_y, A by 2 mu_x, B by -2 mu_x.
    let mask = ground_truth[view.ground_truth_offset + i * 4u + 3u];
    let d_mu = mask * (2.0 * mu_y * (d - c) / ab - 2.0 * mu_x * c * d * (b - a) / (ab * ab));
    let d_exx = mask * (-c * d / (ab * b));
    let d_exy = mask * (2.0 * c / ab);
    for (var channel = 0u; channel < 3u; channel++) {
        let base = (i * 3u + channel) * 4u;
        ssim_partials[base] = ssim[channel];
        ssim_partials[base + 1u] = d_mu[channel];
        ssim_partials[base + 2u] = d_exx[channel];
        ssim_partials[base + 3u] = d_exy[channel];
    }
}
