# Pitch Corrector

Real-time pitch correction using YIN pitch detection and phase vocoder synthesis.

## Prerequisites

This project depends on a fork of [`cpal`](https://github.com/WalterSmuts/cpal)
with additional WebAudio support, referenced as a path dependency at `../cpal`.
Clone it as a sibling of this repository before building:

```bash
# from the parent directory that contains this repo
git clone https://github.com/WalterSmuts/cpal.git
```

Your layout should be:

```
parent/
├── pitch-corrector/   (this repo)
└── cpal/              (the fork)
```

> Note: the web build relies on WebAudio commits in the fork
> (`build_input_stream_raw`, input-device enumeration, ScriptProcessorNode
> output). Ensure the branch you check out contains them. CI clones this fork
> automatically (see `.github/workflows/ci.yml`).

## Build

### Native (terminal UI)

```bash
cargo build
cargo run -- pitch-corrector
```

### Web (WASM)

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --features web --no-default-features
python3 -m http.server 8888
```

Then open http://localhost:8888

### Web (off-main-thread AudioWorklet, experimental)

The default web build runs the audio callback (and therefore the pitch
correction DSP) on the main thread via `ScriptProcessorNode`, which competes
with canvas drawing and causes stutter during voiced audio. The optional
`worklet` feature runs the DSP on the audio render thread instead, using
cpal's AudioWorklet host.

This needs a wasm atomics/threads build and cross-origin isolation:

```bash
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
cargo install wasm-bindgen-cli --version 0.2.106   # match Cargo.lock

./build-worklet.sh          # nightly + -Zbuild-std + atomics, then wasm-bindgen
python3 serve.py 8888       # serves with COOP/COEP so crossOriginIsolated=true
```

Then open http://localhost:8888. Requires a browser with `AudioWorklet` and
SharedArrayBuffer (cross-origin isolated). The `worklet` feature must be built
this way — a normal `wasm-pack` build will not link the threads runtime.

#### Worklet troubleshooting

wasm threads are notoriously toolchain-version-sensitive (see the wasm-bindgen
threads docs and StackBlitz's "Destroyer of Threads" post). Known failure modes:

- **`wasm-bindgen: failed to find __wasm_init_tls`** — lld emits the wasm
  threads TLS symbols (`__wasm_init_tls`, `__tls_size`, `__tls_align`,
  `__tls_base`) but does not export them by default, and wasm-bindgen's threads
  transform needs them exported. `build-worklet.sh` passes
  `-C link-arg=--export=__wasm_init_tls` (and the other three) to fix this; it
  works on current nightlies without pinning.
- **`Atomics.waitAsync: invalid array type`** at runtime — the wasm memory is
  not shared (a plain `ArrayBuffer`). Ensure the build passes
  `--shared-memory --import-memory --max-memory` (build-worklet.sh does) and
  that the page is served cross-origin isolated (serve.py, `crossOriginIsolated
  === true`).
- **`TextDecoder is not defined`** — fixed in the cpal fork by polyfilling
  TextEncoder/TextDecoder in the worklet scope.

### Tests

```bash
cargo test --lib
```

### Performance tuning

Tests prefixed with `perf_` measure system quality with hard thresholds.
Run them with `--nocapture` to get a report, then tighten thresholds as
the system improves:

```bash
cargo test --lib perf_ -- --nocapture --test-threads=1 2>&1 | grep '\[PERF\]'
```

Each line shows the metric, its current value, and the assertion threshold.
To tighten a threshold, find the corresponding `assert!` in the test and
adjust the constant.
