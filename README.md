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
