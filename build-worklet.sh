#!/usr/bin/env bash
# Build the web app with the off-main-thread AudioWorklet backend (`worklet`
# feature). This requires a wasm atomics/threads build (nightly + build-std)
# and, at runtime, cross-origin isolation (serve with ./serve.py).
#
# The default (ScriptProcessorNode) build does NOT need any of this; use
#   wasm-pack build --target web --features web --no-default-features
# instead.
#
# Prereqs:
#   rustup toolchain install nightly
#   rustup component add rust-src --toolchain nightly
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.106   # must match Cargo.lock
set -euo pipefail

PROFILE="${1:-release}"
OUT_DIR="pkg"
CRATE="pitch_corrector"

# Atomics + shared memory for the AudioWorklet host (it shares the wasm
# module+memory with the render thread). wasm-bindgen 0.2.x does NOT share the
# memory itself, so lld must: `--shared-memory` (+ `--import-memory` and
# `--max-memory`). `+atomics` enables the atomic instructions.
#
# TROUBLESHOOTING: if wasm-bindgen fails with "failed to find __wasm_init_tls",
# your rustc/lld and wasm-bindgen-cli versions disagree on the wasm-threads TLS
# ABI. Use a matched pair known-good for wasm threads (align the nightly to the
# wasm-bindgen-cli version in Cargo.lock, e.g. via `rustup override`). Without
# `--shared-memory` the build links but Atomics.* fail at runtime with
# "invalid array type for the operation" (memory is a plain ArrayBuffer).
export RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals \
-C link-arg=--shared-memory \
-C link-arg=--import-memory \
-C link-arg=--max-memory=1073741824"

FLAG=""
[ "$PROFILE" = "release" ] && FLAG="--release"

echo "==> cargo +nightly build ($PROFILE, atomics + build-std)"
cargo +nightly build $FLAG \
  --target wasm32-unknown-unknown \
  --no-default-features --features web,worklet \
  -Z build-std=std,panic_abort

WASM="target/wasm32-unknown-unknown/$PROFILE/${CRATE}.wasm"

echo "==> wasm-bindgen --target web"
wasm-bindgen "$WASM" --out-dir "$OUT_DIR" --target web --split-linked-modules

echo "==> done. Serve with cross-origin isolation:  python3 serve.py 8888"
