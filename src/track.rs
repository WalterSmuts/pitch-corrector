//! A timeline-aligned lane of samples shared between an audio callback and
//! the UI thread, plus the UI-side mirror that heavy analysis reads from.
//!
//! The app is a two-track editor (captured input, produced output) and every
//! track needs the same set of behaviors: non-blocking append from the audio
//! callback, brief locks from the UI, truncate-at-playhead for re-processing,
//! and a private UI-side copy to run long analysis on without holding the
//! shared lock. This module gives those one home; `web.rs` holds two of them.
//!
//! Rewrites (seek, re-record, load) are recorded as an explicit low-water
//! mark under the same lock as the samples, and [`Mirror::catch_up`]
//! consumes it. The previous protocol inferred rewrites by comparing
//! lengths, which misses a rewrite whenever the audio callback re-appends
//! past the mirror's length before the next UI sync — the mark can't.
//!
//! Locking protocol:
//! - Audio callbacks use [`Track::writer`] (try-lock; drops samples on
//!   contention rather than blocking the render thread).
//! - The UI thread uses [`Track::locked`] and friends, which spin instead of
//!   blocking — `Mutex::lock` can park via `Atomics.wait`, which throws on
//!   the wasm main thread. Contention windows are all sub-millisecond.

use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::TryLockError;

/// Spin until the lock is acquired. See the module docs for why the UI
/// thread must not block-park.
pub fn spin_lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    loop {
        match m.try_lock() {
            Ok(g) => return g,
            Err(TryLockError::WouldBlock) => std::hint::spin_loop(),
            Err(TryLockError::Poisoned(p)) => return p.into_inner(),
        }
    }
}

#[derive(Default)]
struct TrackState {
    samples: Vec<f32>,
    /// Lowest position rewritten since the last [`Mirror::catch_up`];
    /// everything the mirror holds from here on is stale.
    rewritten: Option<usize>,
}

impl TrackState {
    fn mark_rewritten(&mut self, pos: usize) {
        self.rewritten = Some(self.rewritten.map_or(pos, |p| p.min(pos)));
    }
}

/// A shared, timeline-aligned sample lane.
#[derive(Default)]
pub struct Track {
    state: Mutex<TrackState>,
}

/// Write access to a track's samples (append-only use in audio callbacks).
pub struct TrackGuard<'a>(MutexGuard<'a, TrackState>);

impl Deref for TrackGuard<'_> {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        &self.0.samples
    }
}

impl DerefMut for TrackGuard<'_> {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        &mut self.0.samples
    }
}

impl Track {
    pub fn new() -> Self {
        Self::default()
    }

    /// Non-blocking writer for audio callbacks. `None` when the UI holds the
    /// lock — the callback skips capture for that quantum instead of glitching.
    pub fn writer(&self) -> Option<TrackGuard<'_>> {
        self.state.try_lock().ok().map(TrackGuard)
    }

    /// Spin-locked access for the UI thread (brief operations only).
    pub fn locked(&self) -> TrackGuard<'_> {
        TrackGuard(spin_lock(&self.state))
    }

    pub fn len(&self) -> usize {
        self.locked().len()
    }

    pub fn is_empty(&self) -> bool {
        self.locked().is_empty()
    }

    pub fn snapshot(&self) -> Vec<f32> {
        self.locked().clone()
    }

    pub fn clear(&self) {
        let mut s = spin_lock(&self.state);
        s.samples.clear();
        s.mark_rewritten(0);
    }

    pub fn replace(&self, samples: &[f32]) {
        let mut s = spin_lock(&self.state);
        s.samples = samples.to_vec();
        s.mark_rewritten(0);
    }

    /// Prepare the lane to be overwritten from `pos`: drop everything after
    /// it and pad with silence up to it, so subsequent appends land exactly
    /// at `pos` on the timeline.
    pub fn rewrite_from(&self, pos: usize) {
        let mut s = spin_lock(&self.state);
        s.samples.truncate(pos);
        s.samples.resize(pos, 0.0);
        s.mark_rewritten(pos);
    }
}

/// A UI-thread copy of a [`Track`] that heavy analysis reads from, so the
/// shared lock is only ever held long enough to copy out new samples.
#[derive(Default)]
pub struct Mirror {
    samples: Vec<f32>,
}

impl Mirror {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Forget everything and adopt the track's current state as the new
    /// baseline: consumes any pending rewrite mark, so a rewrite that
    /// happened *before* this reset cannot retroactively truncate entries
    /// appended after it (a session reset already discards everything the
    /// mark was protecting against).
    pub fn resync(&mut self, track: &Track) {
        spin_lock(&track.state).rewritten = None;
        self.samples.clear();
    }

    /// Pull new samples from the track (one brief lock). If the track's
    /// timeline was rewritten since the last catch-up, the mirror discards
    /// its stale tail first and the rewrite position is returned so the
    /// caller can invalidate anything derived from it. `None` means the
    /// track only grew.
    pub fn catch_up(&mut self, track: &Track) -> Option<usize> {
        let mut src = spin_lock(&track.state);
        let rewritten = src.rewritten.take();
        if let Some(pos) = rewritten {
            self.samples.truncate(pos);
        }
        let n = self.samples.len();
        self.samples.extend_from_slice(&src.samples[n..]);
        rewritten
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_appends_incrementally() {
        let track = Track::new();
        let mut mirror = Mirror::new();
        track.writer().unwrap().extend_from_slice(&[1.0, 2.0]);
        assert_eq!(mirror.catch_up(&track), None);
        assert_eq!(mirror.samples(), &[1.0, 2.0]);
        track.writer().unwrap().push(3.0);
        assert_eq!(mirror.catch_up(&track), None);
        assert_eq!(mirror.samples(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn mirror_reports_timeline_rewrite() {
        let track = Track::new();
        let mut mirror = Mirror::new();
        track
            .writer()
            .unwrap()
            .extend_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mirror.catch_up(&track), None);

        // Seek back to 2 and overwrite the tail.
        track.rewrite_from(2);
        track.writer().unwrap().push(9.0);
        assert_eq!(mirror.catch_up(&track), Some(2));
        assert_eq!(mirror.samples(), &[1.0, 2.0, 9.0]);
    }

    #[test]
    fn rewrite_is_not_missed_when_track_outgrows_mirror() {
        // The failure mode of length-comparison sync: rewrite low, then
        // re-append past the mirror's length before the next catch-up.
        let track = Track::new();
        let mut mirror = Mirror::new();
        track.writer().unwrap().extend_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(mirror.catch_up(&track), None);

        track.rewrite_from(1);
        track
            .writer()
            .unwrap()
            .extend_from_slice(&[7.0, 8.0, 9.0, 10.0]);
        // Track is now longer than the mirror; a length comparison would
        // conclude "no rewrite" and keep the stale 2.0, 3.0.
        assert_eq!(mirror.catch_up(&track), Some(1));
        assert_eq!(mirror.samples(), &[1.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn multiple_rewrites_coalesce_to_the_lowest() {
        let track = Track::new();
        let mut mirror = Mirror::new();
        track.writer().unwrap().extend_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(mirror.catch_up(&track), None);

        track.rewrite_from(2);
        track.rewrite_from(1);
        assert_eq!(mirror.catch_up(&track), Some(1));
        assert_eq!(mirror.samples(), &[1.0]);
    }

    #[test]
    fn resync_consumes_a_pending_rewrite() {
        // A rewrite followed by a mirror reset must not truncate data
        // appended after the rewrite (the web upload path: load_recording
        // rewrites the tracks, Analysis::reset resyncs, then offline
        // processing appends — the first catch_up used to consume the
        // stale mark and wipe the fresh entries).
        let track = Track::new();
        let mut mirror = Mirror::new();
        track.replace(&[1.0, 2.0]);
        mirror.resync(&track);
        track.writer().unwrap().extend_from_slice(&[3.0, 4.0]);
        assert_eq!(mirror.catch_up(&track), None, "stale mark must be gone");
        assert_eq!(mirror.samples(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rewrite_from_pads_to_position() {
        let track = Track::new();
        track.replace(&[1.0]);
        track.rewrite_from(3);
        assert_eq!(track.snapshot(), &[1.0, 0.0, 0.0]);
        track.rewrite_from(0);
        assert!(track.is_empty());
    }

    #[test]
    fn writer_is_denied_while_ui_holds_the_lock() {
        let track = Track::new();
        let ui = track.locked();
        assert!(track.writer().is_none());
        drop(ui);
        assert!(track.writer().is_some());
    }
}
