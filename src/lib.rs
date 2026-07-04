// The native (terminal UI + cpal host) and web (WASM) backends pull in
// mutually incompatible dependencies and cfg paths. Building both at once
// is always a configuration mistake — fail early with a clear message.
#[cfg(all(feature = "native", feature = "web"))]
compile_error!(
    "features `native` and `web` are mutually exclusive; enable exactly one \
     (native is the default; for web use --no-default-features --features web)"
);

#[cfg(test)]
#[global_allocator]
static A: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;

pub mod complex_interpolation;
pub mod interpolation;
pub mod music;
pub mod pitch_correction;
pub mod signal_processing;

#[cfg(feature = "native")]
pub mod display;
#[cfg(feature = "native")]
pub mod hardware;

#[cfg(feature = "web")]
pub mod web;
