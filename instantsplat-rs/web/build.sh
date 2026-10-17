#!/usr/bin/env bash
# Builds the wasm module of the browser demo into web/pkg/. The page is then this
# directory as is: serve it (npm start at the repository root) or deploy it to GitHub
# Pages (.github/workflows/pages.yml at the repository root does).
#
# Needs: rustup target wasm32-unknown-unknown, wasm-bindgen-cli matching Cargo.toml's
# wasm-bindgen version.
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release
rm -rf pkg
wasm-bindgen --target web --out-dir pkg ../../target/wasm32-unknown-unknown/release/instantsplat_web.wasm
echo "pkg/: $(du -sh pkg | cut -f1)"
