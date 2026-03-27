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
