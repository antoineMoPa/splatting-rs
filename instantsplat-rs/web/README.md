# instantsplat-rs in the browser

Live at https://antoinemopa.github.io/splatting-rs/.

Photos in, a trained Gaussian splat out, nothing but the browser: Depth Anything 3
(`../../depthanything3-rs`; Small, Base or Large, chosen in a dropdown)
predicts depth and cameras from the photos on candle's CPU backend compiled to wasm, one
Gaussian per point (`init_from_points`, stride 2), then the rasterizer and the trainer of
`..` optimize them on WebGPU, with the same WGSL kernels as the native build. The page
first asks for one click to download the chosen model, 137 MB for Small, 542 MB for Base or 1644 MB for Large
(skipped on later visits, the browser keeps each one it downloaded); then picking photos
runs the whole chain, 300 training steps included; then drag to look around and download
`splat.ply`. The view is rendered at the photos' height and at least 4:3 wide (a wider
frame at the same focal length, so portrait photos still give a landscape view).
"Advanced" has the photo to view from, the training steps (300, run on every
reconstruction; 0 shows the network's Gaussians untrained) with a "refine camera poses" box
(on by default), "Train N more steps" on the current splat, and loading a `fit` run. The
desktop app (`../app`) has the same section. `?steps=N` presets the steps.

![The page after reconstructing the six duck photos](demo.png)

The network runs in a Web Worker (`worker.js`), so the page stays responsive; it reports
each backbone block. The weights (`model.safetensors` of
[depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) or
[depth-anything/DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE)) are fetched
from Hugging Face behind the download button and kept in the browser's Cache API, one
entry per model; the page asks the worker whether the selected model is cached and, if
so, loads it without asking, so switching models is instant once both are downloaded.
Giant (5.4 GB) is native only: its weights alone exceed wasm32's 4 GB memory. The depth
head runs one view at a time so Large fits. Large takes about 7 minutes for the six duck
photos on one CPU thread. Every worker answer names the model it refers to and the page checks it against the
dropdown. The focal length comes from the photos' EXIF when present, else from the
network's own prediction.

Measured in headless Chrome on an M5 for the six 3024x4032 duck photos (`.example/`,
inputs at 378x504): weights download 7 s (then cached), preprocessing 0.14 s, Depth
Anything 3 Small 23.5 s on one thread with SIMD (0.75 s per within-view backbone block,
2.3 s per across-views block, 8 s for the depth head), 260 k Gaussians in 0.04 s, training
26 steps/s (300 steps in 11.5 s). Natively the network takes 2 s on Metal.

The page also loads a `fit` run (`splat.ply`, `cameras.json` and the `input_view{i}.png`
it was trained on, all picked together under "Advanced") or `?run=<directory URL>`; the
example run is the default scene.

## Build and serve

`rustup target add wasm32-unknown-unknown`, `cargo install wasm-bindgen-cli --version 0.2.129`,
then from the repository root (`package.json` there):

```
npm install
npm run build
npm start
open http://localhost:8787/
```

`build.sh` compiles the wasm module into `pkg/`; the page is this directory as is, with
`example/` (a `fit` run of the six duck photos) as the default scene. `.cargo/config.toml`
turns on `simd128`; `Cargo.toml` patches candle to the upstream commit that makes its SIMD
kernels compile for wasm32 (the native workspace keeps the release). WebGPU needs
`localhost` or https.

`.github/workflows/pages.yml` at the repository root runs `build.sh` on every push to
`main` and deploys this directory to GitHub Pages (enable Pages with source "GitHub
Actions" in the repository settings once).

The headless test: `web_da3_shot.mjs` in the Playwright scratch directory loads the page,
picks the photos through the file input, waits for the reconstruction, screenshots the
render, trains and screenshots again; Chrome for Testing needs `--enable-unsafe-webgpu`
and a page served over `localhost` (WebGPU is absent on `about:blank`).
