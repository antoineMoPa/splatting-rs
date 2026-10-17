#!/usr/bin/env bash
# Builds the browser demo into web/site/: the wasm module, the page and the example run.
# Serve that directory as is (python3 -m http.server -d site 8787) or deploy it to GitHub
# Pages (.github/workflows/pages.yml at the repository root does).
#
# Needs: rustup target wasm32-unknown-unknown, wasm-bindgen-cli matching Cargo.toml's
# wasm-bindgen version.
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release
rm -rf site
mkdir -p site
wasm-bindgen --target web --out-dir site/pkg ../../target/wasm32-unknown-unknown/release/instantsplat_web.wasm
cp index.html worker.js site/
cp -R example site/example
echo "site/: $(du -sh site | cut -f1), example run $(du -sh site/example | cut -f1)"
