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
# module+memory with the render thread). Required pieces:
#   * +atomics,+bulk-memory,+mutable-globals  -> atomic instructions
#   * --shared-memory --import-memory --max-memory -> the wasm memory is a
#     shared, imported SharedArrayBuffer (else Atomics.* throw "invalid array
#     type" at runtime).
#   * --export=__wasm_init_tls,__tls_size,__tls_align,__tls_base -> lld emits
#     these TLS symbols but does not export them by default; wasm-bindgen's
#     threads transform needs them exported (else "failed to find
#     __wasm_init_tls"). Exporting them works on both current nightlies, so no
#     specific nightly pin is required.
export RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals \
-C link-arg=--shared-memory \
-C link-arg=--import-memory \
-C link-arg=--max-memory=1073741824 \
-C link-arg=--export=__wasm_init_tls \
-C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align \
-C link-arg=--export=__tls_base"

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
