# depthanything3-rs

Photos in; per photo a depth map, a confidence map, a camera pose and a focal length out,
and a coloured point cloud from them. A Rust port of ByteDance's
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Small, Base,
Large and Giant) on [candle](https://github.com/huggingface/candle): all photos go
through the network in one pass. 137 MB of weights for Small, against MapAnything's
4.9 GB: the browser-sized stage 1 of this workspace. Runs on Apple Silicon (Metal) or the
CPU.

## Setup

A stable Rust toolchain and Xcode's command line tools (Metal). The weights download from
Hugging Face on first run (`depth-anything/DA3-SMALL`, 137 MB; `--variant base|large|giant`
for `DA3-BASE`, 542 MB, `DA3-LARGE`, 1.6 GB, and `DA3-GIANT`, 5.4 GB), no account needed.

```
cargo build --release
```

## Make a point cloud

```
cargo run --release -p depthanything3-rs -- photo1.jpg photo2.jpg photo3.jpg --out scene.ply
```

6 photos take 2 s on an M5 after a 30 ms weight load. The point cloud is in the first
photo's camera frame, at an arbitrary scale (the depth is not metric).

Other options: `--min-conf` (drop uncertain points), `--variant small|base|large|giant`,
`--reference-view`, `--precision f32|f16|bf16`, `--weights <file>`, `--raw-out <file>`
(dense depth, confidence, points and cameras as safetensors), `--debug-dir <dir>` (depth
and confidence maps as PNGs), `--cpu`. `--focal-35mm` prints the known focal length next
to the predicted one; the network does not take it as input.

Background cutouts (transparent PNGs) work as in `../mapanything-rs`: transparent pixels
get no points and the image is cropped to its foreground first.

## Other things in here

- `examples/view_point_maps.rs`: the library used from Rust, one PLY per photo.
- `src/`: one file per stage: `preprocess.rs` (upstream's resizing: longest side 504 px,
  sides rounded to multiples of 14), the network (`backbone.rs` with `vit_block.rs`,
  `rope.rs`, `reference_view.rs`; `dpt.rs` with `uv_embed.rs`; `camera_head.rs`;
  `model.rs`), `pipeline.rs` (depth to points in the first camera's frame, multi-view
  cleaning from mapanything-rs), `variant.rs` (what differs between the variants).

## Limits

- Not compared numerically against the PyTorch reference: checked by its results (the
  duck and the turntable captures in `../instantsplat-rs`) and by sanity checks (finite
  outputs, cameras on an arc around the object, focal length within 7 % of EXIF).
- Upstream's optional inputs (known cameras through its camera encoder), its ray-based
  pose option (RANSAC on the ray head's output; the rays are computed and dropped) and its
  3DGS head (not in these checkpoints) are not ported.
- Resolution is the model's: 504 px on the long side.

## License

Apache-2.0 ([LICENSE](LICENSE)), as Depth Anything 3 and its DA3-SMALL and DA3-BASE
weights. The DA3-LARGE and DA3-GIANT weights are CC BY-NC 4.0 (no commercial use); the
Nested checkpoints are not supported.

## Acknowledgements

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance Seed
  (Lin et al., 2025, [arXiv:2511.10647](https://arxiv.org/abs/2511.10647))
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta
- [candle](https://github.com/huggingface/candle) by Hugging Face
