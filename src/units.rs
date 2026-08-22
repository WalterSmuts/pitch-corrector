//! Time-axis index types.
//!
//! The project juggles several time units — raw sample positions, vocoder
//! hops (`HOP_SIZE` samples), pitch-analysis hops (`PITCH_HOP` samples) —
//! and mixing them silently has already caused real bugs (a contour mapped
//! by linear rescaling instead of its hop grid, a cursor restarted on the
//! wrong axis). These types make the unit part of the value: converting
//! between grids requires naming the hop size, and two different grids are
//! two different types.

/// An index on the sample timeline.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SampleIdx(pub usize);

/// An index on a hop grid with `H` samples per hop. The const parameter
/// keeps different grids (e.g. the 512-sample vocoder grid and the
/// 1024-sample pitch-analysis grid) as distinct, non-mixable types.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct HopIdx<const H: usize>(pub usize);

impl<const H: usize> HopIdx<H> {
    /// The hop containing this sample (floor).
    pub fn containing(sample: SampleIdx) -> Self {
        Self(sample.0 / H)
    }

    /// The first sample of this hop.
    pub fn start(self) -> SampleIdx {
        SampleIdx(self.0 * H)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_grids_convert_explicitly() {
        let s = SampleIdx(1500);
        let vocoder: HopIdx<512> = HopIdx::containing(s);
        let analysis: HopIdx<1024> = HopIdx::containing(s);
        assert_eq!(vocoder, HopIdx::<512>(2));
        assert_eq!(analysis, HopIdx::<1024>(1));
        assert_eq!(vocoder.start(), SampleIdx(1024));
        assert_eq!(analysis.start(), SampleIdx(1024));
        // let wrong: HopIdx<512> = analysis; // does not compile: distinct grids
    }
}
