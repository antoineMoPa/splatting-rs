# instantsplat-rs in the browser

Live at https://antoinemopa.github.io/splatting-rs/.

Photos in, a trained Gaussian splat out, nothing but the browser: Depth Anything 3 Small
(`../../depthanything3-rs`, 34 M parameters) predicts depth and cameras from the photos
on candle's CPU backend compiled to wasm, one Gaussian per point (`init_from_points`,
stride 2), then the rasterizer and the trainer of `..` optimize them on WebGPU, with the
same WGSL kernels as the native build. The page first asks for one click to download the
137 MB model (skipped on later visits, the browser keeps it); then picking photos runs the
whole chain, 300 training steps included; then drag to look around and download
`splat.ply`. The view is rendered at the photos' height and at least 4:3 wide (a wider
frame at the same focal length, so portrait photos still give a landscape view).
"Advanced" has the camera choice, more training and loading a `fit` run.

![The page after reconstructing the six duck photos](demo.png)

The network runs in a Web Worker (`worker.js`), so the page stays responsive; it reports
each backbone block. The weights (`model.safetensors` of
[depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL)) are fetched
from Hugging Face behind the download button and kept in the browser's Cache API; the
page asks the worker whether they are cached and, if so, loads them without asking. The
focal length comes from the photos' EXIF when present, else from the network's own
prediction.

Measured in headless Chrome on an M5 for the six 3024x4032 duck photos (`.example/`,
inputs at 378x504): weights download 7 s (then cached), preprocessing 0.14 s, Depth
Anything 3 Small 23.5 s on one thread with SIMD (0.75 s per within-view backbone block,
2.3 s per across-views block, 8 s for the depth head), 260 k Gaussians in 0.04 s, training
26 steps/s (300 steps in 11.5 s). Natively the network takes 2 s on Metal.

The page also loads a `fit` run (`splat.ply`, `cameras.json` and the `input_view{i}.png`
it was trained on) from files (under "Advanced") or `?run=<directory URL>`; the example
run is the default scene.

## Build

`rustup target add wasm32-unknown-unknown`, `cargo install wasm-bindgen-cli --version 0.2.129`:

```
cd web
cargo build --release
wasm-bindgen --target web --out-dir pkg ../../target/wasm32-unknown-unknown/release/instantsplat_web.wasm
```

`.cargo/config.toml` turns on `simd128`; `Cargo.toml` patches candle to the upstream commit
that makes its SIMD kernels compile for wasm32 (the native workspace keeps the release).
Serve the crate directory with any static file server and open the page in Chrome (WebGPU
needs `localhost` or https).

## As a site

`build_site.sh` builds everything into `site/` with `example/` (a `fit` run of the six duck
photos) as the default scene:

```
npm install
npm run build
npm start
open http://localhost:8787/
```

(`package.json` at the repository root.)

`.github/workflows/pages.yml` at the repository root runs the same script on every push
to `main` and deploys `site/` to GitHub Pages (enable Pages with source "GitHub Actions"
in the repository settings once).

The headless test: `web_da3_shot.mjs` in the Playwright scratch directory loads the page,
picks the photos through the file input, waits for the reconstruction, screenshots the
render, trains and screenshots again; Chrome for Testing needs `--enable-unsafe-webgpu`
and a page served over `localhost` (WebGPU is absent on `about:blank`).
