# splatting-rs

![Six photos of a rubber duck, and the Gaussian splat made from them](docs/duck_splat.png)

3D reconstruction from photos, in Rust, on a Mac. Two crates:

- `mapanything-rs/`: photos → coloured point cloud, with Meta's MapAnything network on
  candle (Metal). One forward pass, no camera poses needed.
- `instantsplat-rs/`: photos → Gaussian splat, after InstantSplat. Uses `mapanything-rs` for
  the points and cameras, then optimises Gaussians with its own wgpu/WGSL rasterizer.

## Setup

A stable Rust toolchain and Xcode's command line tools (Metal). The MapAnything weights
(4.9 GB) download from Hugging Face on first run.

```
cargo build --release
```

## Run

```
cargo run --release -p mapanything-rs -- photo1.jpg photo2.jpg photo3.jpg --out scene.ply
cargo run --release -p instantsplat-rs -- fit photos/*.jpg --out runs/thing
```

Each crate's README has the details. Examples: `cargo run --release -p <crate> --example <name>`;
see `mapanything-rs/examples/` and `instantsplat-rs/examples/`.
