#!/usr/bin/env bash
# Builds the browser demo into web/site/: the wasm module, the page and the example run.
# Serve that directory as is (npm install && npm start at the repository root) or deploy it to GitHub
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
# The worker URL carries the build time: browsers keep worker scripts past a reload, and
# an old worker with a new page breaks the messages between them.
sed "s|\\./worker\\.js\"|./worker.js?build=$(date +%s)\"|" index.html > site/index.html
cp worker.js site/
cp -R example site/example
echo "site/: $(du -sh site | cut -f1), example run $(du -sh site/example | cut -f1)"
