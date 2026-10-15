# mapanything-rs

3D reconstruction from photos with [MapAnything](https://github.com/facebookresearch/map-anything)
(Meta), in Rust on [candle](https://github.com/huggingface/candle).

It takes two or more photos (or video frames) of a scene and writes a colored point cloud
(`.ply`), with every image in a single forward pass. It runs on Apple Silicon GPUs (Metal) or
on the CPU. Code and weights are Apache 2.0, so commercial use is allowed.

## Usage

Requires a recent stable Rust toolchain. Building on macOS needs Metal, which comes with Xcode
or the Command Line Tools.

```sh
cargo run --release -- photo1.jpg photo2.jpg photo3.jpg --out scene.ply
```

On first run, the weights
([facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache), 4.9 GB)
download from Hugging Face into `~/.cache/huggingface`. No account is needed.

| option | default | |
|---|---|---|
| `--out` | `reconstruction.ply` | binary PLY, xyz + rgb, y up, in the first image's camera frame |
| `--focal-35mm` | from EXIF | 35 mm-equivalent focal length of the camera, for every image |
| `--min-conf` | `1` | drop points below this confidence (always > 1, or 0 for rejected points) |
| `--precision` | `f16` | `f32`, `f16` or `bf16` |
| `--weights` | Hugging Face download | local `.safetensors` checkpoint |
| `--raw-out` | none | also save dense per-image points, confidence, mask and cameras as `.safetensors` |
| `--cpu` | off | run on the CPU instead of Metal |

To view the result, open the `.ply` in [MeshLab](https://www.meshlab.net),
[CloudCompare](https://www.cloudcompare.org) or Blender.

### Give it the focal length

MapAnything can take the camera's intrinsics as input, and object-centric captures need
them. On an 8-photo turntable of a mannequin it guessed a 610 px focal length where the real
one was about 400 px, which put every view too far from its camera. The focal length comes
from each photo's EXIF (35 mm equivalent) when present. Video frames have none, so pass
`--focal-35mm` (iPhone main cameras: 24 mm for photos, about 26 mm for stabilized video). The
crop and resize applied to each image are accounted for.

### Objects on a turntable

For a capture where the camera stays put and the object turns, cut the background out
first. Otherwise the static background tells the model the camera never moved. Transparent
pixels of a PNG are background: the image is cropped to the object, the model sees a gray
backdrop there, and those pixels are left out of the point cloud. On macOS 14+,
`tools/cutout.swift` does this with Apple's Vision foreground segmentation:

```sh
swiftc -O tools/cutout.swift -o cutout
for f in photos/*.jpg; do ./cutout "$f" "cutouts/$(basename "${f%.jpg}").png"; done
cargo run --release -- cutouts/*.png --focal-35mm 24 --out object.ply
```

The cutout PNGs carry no EXIF, so pass the focal length.

### From a video

`tools/video_frames.sh` extracts evenly spaced frames, in order, and with `--cutout` cuts
their background out (frames where no foreground is found are dropped):

```sh
tools/video_frames.sh turn.mov 40 frames --cutout
cargo run --release -- frames/*.png --focal-35mm 26 --out object.ply
```

Move slowly and light well: motion blur costs more than extra frames gain. 37 frames of a
28 s video take about 2.5 minutes at f16 on an M5.

## How it works

1. All images are resized to cover one shared size from upstream's aspect-ratio table
   (518 px on the long side, multiples of 14) and center-cropped.
2. Each image goes through DINOv2 ViT-giant (its first 24 blocks). When the image's
   intrinsics are known, a ray-direction encoder's features are added. Then a LayerNorm.
3. 16 alternating-attention blocks: even blocks attend across every token of every image,
   odd blocks within each image. View 0 is marked as the reference.
4. Per image, a DPT head predicts ray directions, depth, confidence and an ambiguity mask,
   and a pose head predicts the camera; a shared scale head makes everything metric.
5. Points are expressed in camera 0's frame and ambiguous pixels are rejected. A point that
   lands more than 2% in front of what another camera sees there, with lower confidence,
   gets confidence 0. The views come from one forward pass and disagree by centimeters: a
   tighter tolerance removed most of the least confident views (an object's back) instead of
   outliers.

8 photos take about 35 s at f16 on an M5, plus about 15 s to load the weights. The attention
across all images grows with the square of the image count.

| file | |
|---|---|
| `src/image_input.rs` | loading, EXIF orientation and focal length, background cutout |
| `src/preprocess.rs` | shared input size, resize, crop, intrinsics |
| `src/dinov2.rs`, `src/vit_block.rs` | DINOv2 ViT-g/14 encoder |
| `src/ray_encoder.rs` | known-intrinsics input |
| `src/info_sharing.rs` | alternating-attention transformer |
| `src/dpt.rs` | dense head |
| `src/heads.rs`, `src/model.rs` | pose and scale heads, dense outputs |
| `src/pipeline.rs`, `src/cleaning.rs` | cameras, point maps, multi-view cleaning |
| `src/ply.rs` | point cloud export |

## Accuracy

**Not yet compared numerically against the PyTorch reference.** Checked on real captures only:
f16 and f32 agree, the cameras of a turntable capture land on a ring around the object, and
recovered focal lengths are plausible.

## Limitations

- Upstream's edge masks (depth and normal discontinuities, which remove flying points along
  silhouettes) are not ported, nor the optional depth and pose inputs.
- On one 8-photo turntable capture, one back view got a mirrored pose (front/back confusion).
- Resolution is the model's: 518 px on the long side, at most about 200k points per image.

## License

The code in this repository is licensed under [Apache-2.0](LICENSE), the license of
MapAnything, from which it is ported. Parts of it (the alternating-attention transformer,
DPT head and ray encoder) are ported from [UniCeption](https://github.com/castacks/UniCeption),
BSD 3-Clause License, Copyright (c) 2024, AirLab Stacks (see
[LICENSE-UniCeption](LICENSE-UniCeption)).

The default weights, [facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache),
are Apache 2.0. The similarly named `facebook/map-anything` is CC BY-NC 4.0 (non-commercial).

## Acknowledgements

- [MapAnything](https://github.com/facebookresearch/map-anything) by Meta
- [UniCeption](https://github.com/castacks/UniCeption) by AirLab
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta
- [candle](https://github.com/huggingface/candle) by Hugging Face
