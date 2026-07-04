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
