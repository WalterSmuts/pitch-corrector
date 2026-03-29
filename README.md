# Pitch Corrector

Real-time pitch correction using YIN pitch detection and phase vocoder synthesis.

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
