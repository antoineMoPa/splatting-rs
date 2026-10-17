# instantsplat-rs

A Gaussian splat from a few photos, in seconds instead of the COLMAP + training half hour,
after [InstantSplat](https://instantsplat.github.io): MapAnything (via `../mapanything-rs`,
candle on Metal) predicts dense points and cameras, one Gaussian per point is optimized
against the photos with a wgpu rasterizer written in WGSL.

## Setup

A stable Rust toolchain and Xcode's command line tools (Metal). `../mapanything-rs` must
be checked out next to this folder; its weights (4.9 GB) download from Hugging Face on
first run.

```
cargo build --release
```

## Make a splat

```
cargo run --release -p instantsplat-rs -- fit photos/*.jpg --out runs/thing
```

10 to 20 JPEG or PNG photos with small steps between them, the first one straight on. 11
photos take about 3 minutes on an M5 (plus the one-time weights download). Everything goes
in `runs/thing/`: `splat.ply` (3DGS PLY), `cameras.json`, and one `render_view{i}.png` /
`input_view{i}.png` pair per photo (the input view as the network saw it, cropped to the
foreground, with the mask as alpha: the training view, which the browser demo reloads).

The focal length comes from the photos' EXIF. Video frames have none: pass `--focal-35mm`
(about 26 for iPhone video). `--steps` (default 1000) and `--stride` (default 2: one
Gaussian per 2x2 pixels) trade time for quality.

`--network da3-small` replaces MapAnything with Depth Anything 3 Small
(`../depthanything3-rs`, 137 MB of weights instead of 4.9 GB, 2 s instead of 12 s for 6
photos): it predicts depth and cameras, including the focal length; a known focal length
(EXIF or `--focal-35mm`) replaces the predicted one afterwards, as for MapAnything. Its
cameras are less accurate (see `../README.md`).

The camera poses are optimized along with the Gaussians (as InstantSplat): per view a
rotation and a translation on top of MapAnything's pose, view 0 fixed, plus one focal
scale shared by every view; `cameras.json` holds the refined cameras and the run prints
how far each moved. `--no-pose-refinement` keeps MapAnything's cameras.

With a COLMAP model of the whole capture (see below) the result can be judged without
COLMAP taking part in the training: `--evaluate-colmap capture/sparse_txt
--evaluate-frames capture/images` aligns COLMAP's cameras onto the optimized ones, prints
the residual per camera (how far the poses still are from COLMAP's), renders every
`--hold-out-every`-th frame (default 10) through its aligned COLMAP camera and prints the
L1 against the photo, with `heldout_<frame>.png` (photo | render) in the output directory.

## Refine it with COLMAP poses

Seen from anywhere but its few input views, the splat is spiky and torn. With a COLMAP
model of the whole capture (TXT `cameras.txt` + `images.txt`, undistorted, naming the
same frames), `refine` continues the optimization against every frame:

```
cargo run --release -p instantsplat-rs -- refine runs/thing \
    --colmap capture/sparse_txt --frames capture/images --out runs/thing_refined
```

COLMAP's world is aligned to the splat by the frames both posed (at least 3; the
residual per frame is printed, a bad alignment shows there), every 10th frame is held out
(`--hold-out-every`), and the held-out L1 before and after is printed with one
`heldout_<frame>.png` (photo | before | after) per held-out frame in the output directory.
`--resolution-scale 2` trains at twice the splat's resolution. 98 frames and 3000 steps
take about 5 minutes.

## In the browser

The rasterizer and the trainer also run on WebGPU, same kernels: `web/` has a
wasm-bindgen entry point and a page that renders a run's `splat.ply` from its cameras
(drag to orbit) and continues its optimization against the run's input views; `web/README.md`
has the build and serve commands. `cargo check --target wasm32-unknown-unknown
--no-default-features --lib` keeps the crate building for wasm32 (the commands that read
photos and COLMAP models from disk are native only).

## Other things in here

- `src/shaders/`: the WGSL kernels (projection, tile binning, radix sort, rasterization,
  their backward passes, Adam).
- `src/rasterizer_tests.rs`: gradient check of the backward pass against finite
  differences, and the forward pass against a CPU reference (`cargo test --release`);
  `src/pose_tests.rs`: the same for the camera pose gradient, and a synthetic scene whose
  perturbed cameras are recovered.

## Acknowledgements

- InstantSplat (Fan et al., 2024, [arXiv:2403.20309](https://arxiv.org/abs/2403.20309)): the recipe
- [MapAnything](https://github.com/facebookresearch/map-anything) by Meta, via `../mapanything-rs`
- 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023, [paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)): the rasterizer maths
- [candle](https://github.com/huggingface/candle) by Hugging Face
- [wgpu](https://github.com/gfx-rs/wgpu)
