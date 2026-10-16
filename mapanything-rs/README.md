# mapanything-rs

Photos in, coloured 3D point cloud out. A Rust port of Meta's
[MapAnything](https://github.com/facebookresearch/map-anything) network on
[candle](https://github.com/huggingface/candle): all photos go through the network in one
pass, which predicts the points and the cameras. No camera poses needed. Runs on Apple
Silicon (Metal) or the CPU. Code and weights are Apache 2.0.

## Setup

A stable Rust toolchain and Xcode's command line tools (Metal). The weights (4.9 GB)
download from Hugging Face on first run, no account needed.

```
cargo build --release
```

## Make a point cloud

```
cargo run --release -- photo1.jpg photo2.jpg photo3.jpg --out scene.ply
```

8 photos take about 35 s on an M5, plus 15 s to load the weights. The point cloud is in
the first photo's camera frame.

Video frames have no EXIF, so pass the focal length: `--focal-35mm 26` for iPhone video,
24 for iPhone photos. For an object on a turntable, cut the background out first (see
below), or the model thinks the camera never moved.

Other options: `--min-conf` (drop uncertain points), `--precision f32|f16|bf16`,
`--weights <file>`, `--raw-out <file>` (dense per-photo points, confidence and cameras
as safetensors), `--cpu`.

## Background cutouts and video

`tools/cutout.swift` (macOS 14+, Apple Vision) makes transparent-background PNGs; transparent
pixels get no points:

```
swiftc -O tools/cutout.swift -o cutout
./cutout photo.jpg photo.png
```

`tools/video_frames.sh` extracts evenly spaced frames from a video, with `--cutout` to cut
their background out:

```
tools/video_frames.sh turn.mov 40 frames --cutout
cargo run --release -- frames/*.png --focal-35mm 26 --out object.ply
```

Move slowly and light well: motion blur costs more than extra frames gain.

## Other things in here

- `examples/`: the library used from Rust (`cargo run --release --example <name>`).
- `src/`: one file per stage, from image loading (`image_input.rs`, `preprocess.rs`) through
  the network (`dinov2.rs`, `info_sharing.rs`, `dpt.rs`, `heads.rs`, `model.rs`) to the
  point cloud (`pipeline.rs`, `cleaning.rs`, `ply.rs`).

## Limits

- Not compared numerically against the PyTorch reference; checked on real captures only.
- Upstream's edge masks and optional depth/pose inputs are not ported.
- Resolution is the model's: 518 px on the long side.

## License

Apache-2.0 ([LICENSE](LICENSE)), as MapAnything. The alternating-attention transformer, DPT
head and ray encoder are ported from [UniCeption](https://github.com/castacks/UniCeption),
BSD 3-Clause ([LICENSE-UniCeption](LICENSE-UniCeption)). The weights,
[facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache), are
Apache 2.0.

## Acknowledgements

- [MapAnything](https://github.com/facebookresearch/map-anything) by Meta
- [UniCeption](https://github.com/castacks/UniCeption) by AirLab
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta
- [candle](https://github.com/huggingface/candle) by Hugging Face
