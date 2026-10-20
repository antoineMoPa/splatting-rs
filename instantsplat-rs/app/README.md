# instantsplat-rs on the desktop

The browser demo (`../web`) as a native window: photos in, a trained Gaussian splat out,
on the Mac's GPU. A model dropdown offers every Depth Anything 3 variant
(`../../depthanything3-rs`, Small by default, 137 MB downloaded from Hugging Face once)
and MapAnything; the network predicts depth and cameras on candle's Metal backend, one
Gaussian per point (`init_from_points`, stride 2), then the rasterizer and the trainer of
`..` optimize them on wgpu for 300 steps while the view updates live. Choosing or dropping
photos only fills the window; nothing downloads or runs until the start button is clicked
("Download the N MB model and make the splat" when the model is not cached yet); then drag to look around (a turntable above
the object) and save `splat.ply`. "Advanced" is the page's: the photo to view from, the
training steps and the "refine camera poses" box, "Train N more steps", and loading a `fit`
run from its folder.

```
cargo run --release -p instantsplat-app
```

`../../install-app.sh` installs it with `cargo install` and adds a `splatting-rs.app`
launcher to Applications; rerun it to upgrade.

Photos on the command line fill the window the same way, and `--start`
runs them right away: `cargo run --release -p instantsplat-app -- photos/*.jpg --start`.
`--network da3-base` (or `mapanything`) picks the dropdown's initial model, `--steps N` and
`--fixed-poses` the training, `--run <dir>` loads a `fit` run at start; `--screenshot
<png>` and `--save-to <ply>` drive a `--start` run from a script. The chain runs on one
background thread with its own GPU device (`src/pipeline.rs`); the window (`src/main.rs`,
`src/advanced.rs`, `src/look.rs`, the models in `src/weights.rs`) gets status lines and frames over a channel.
