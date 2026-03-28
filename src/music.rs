/// Frequency of C0 in Hz.
const C0_FREQ: f32 = 16.3516;

/// A pitch class (note name without octave).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Note {
    C = 0,
    CS = 1,
    D = 2,
    DS = 3,
    E = 4,
    F = 5,
    FS = 6,
    G = 7,
    GS = 8,
    A = 9,
    AS = 10,
    B = 11,
}

impl Note {
    pub const ALL: [Note; 12] = [
        Note::C,
        Note::CS,
        Note::D,
        Note::DS,
        Note::E,
        Note::F,
        Note::FS,
        Note::G,
        Note::GS,
        Note::A,
        Note::AS,
        Note::B,
    ];

    fn bit(self) -> u16 {
        1 << (self as u16)
    }
}

/// An absolute pitch: a note class plus an octave.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pitch {
    pub note: Note,
    pub octave: i8,
}

impl Pitch {
    pub fn new(note: Note, octave: i8) -> Self {
        Self { note, octave }
    }

    /// MIDI-style semitone number relative to C0.
    pub fn semitones_from_c0(self) -> f32 {
        self.octave as f32 * 12.0 + self.note as u8 as f32
    }

    pub fn to_freq(self) -> f32 {
        C0_FREQ * (2.0f32).powf(self.semitones_from_c0() / 12.0)
    }

    pub fn from_freq(freq: f32) -> Self {
        let semitones = 12.0 * (freq / C0_FREQ).log2();
        let rounded = semitones.round() as i32;
        let octave = rounded.div_euclid(12) as i8;
        let note_idx = rounded.rem_euclid(12) as u8;
        Self {
            note: Note::ALL[note_idx as usize],
            octave,
        }
    }
}

/// A set of note classes forming a scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale(u16);

impl Scale {
    pub fn empty() -> Self {
        Self(0)
    }

    pub fn chromatic() -> Self {
        Self(0x0FFF)
    }

    pub fn from_intervals(intervals: &[Interval], root: Note) -> Self {
        let root_idx = root as u8;
        let mut bits = 0u16;
        for interval in intervals {
            bits |= Note::ALL
                [((root_idx + interval.semitones().rem_euclid(12) as u8) % 12) as usize]
                .bit();
        }
        Self(bits)
    }

    pub fn major(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MAJOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MAJOR_SIXTH,
                Interval::MAJOR_SEVENTH,
            ],
            root,
        )
    }

    pub fn minor(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MINOR_SIXTH,
                Interval::MINOR_SEVENTH,
            ],
            root,
        )
    }

    pub fn pentatonic(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MAJOR_THIRD,
                Interval::PERFECT_FIFTH,
                Interval::MAJOR_SIXTH,
            ],
            root,
        )
    }

    pub fn contains(self, note: Note) -> bool {
        self.0 & note.bit() != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn bits(self) -> u16 {
        self.0
    }

    pub fn from_bits(bits: u16) -> Self {
        Self(bits & 0x0FFF)
    }

    /// Snap a frequency to the nearest note in this scale.
    /// Returns the nearest chromatic pitch if the scale is empty.
    pub fn nearest_pitch(self, freq: f32) -> Pitch {
        let semitones_from_c0 = 12.0 * (freq / C0_FREQ).log2();
        let octave = (semitones_from_c0 / 12.0).floor();
        let semitone_in_octave = semitones_from_c0 - octave * 12.0;

        if self.is_empty() {
            // No scale — snap to nearest chromatic note
            let rounded = semitone_in_octave.round() as i32;
            let note_idx = rounded.rem_euclid(12) as usize;
            return Pitch::new(Note::ALL[note_idx], octave as i8);
        }

        let mut best_offset = 0.0f32;
        let mut best_dist = f32::MAX;
        for note in Note::ALL {
            if !self.contains(note) {
                continue;
            }
            let note_f = note as u8 as f32;
            for &candidate in &[note_f, note_f + 12.0, note_f - 12.0] {
                let dist = (semitone_in_octave - candidate).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_offset = candidate;
                }
            }
        }

        let target_semitones = octave * 12.0 + best_offset;
        let rounded = target_semitones.round() as i32;
        Pitch::new(
            Note::ALL[rounded.rem_euclid(12) as usize],
            rounded.div_euclid(12) as i8,
        )
    }
}

impl std::ops::BitOr<Note> for Scale {
    type Output = Self;
    fn bitor(self, rhs: Note) -> Self {
        Self(self.0 | rhs.bit())
    }
}

impl std::ops::BitOr for Note {
    type Output = Scale;
    fn bitor(self, rhs: Note) -> Scale {
        Scale(self.bit() | rhs.bit())
    }
}

/// The quality of a simple interval within one octave.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SimpleInterval {
    Unison = 0,
    MinorSecond = 1,
    MajorSecond = 2,
    MinorThird = 3,
    MajorThird = 4,
    PerfectFourth = 5,
    Tritone = 6,
    PerfectFifth = 7,
    MinorSixth = 8,
    MajorSixth = 9,
    MinorSeventh = 10,
    MajorSeventh = 11,
}

impl SimpleInterval {
    pub const ALL: [SimpleInterval; 12] = [
        SimpleInterval::Unison,
        SimpleInterval::MinorSecond,
        SimpleInterval::MajorSecond,
        SimpleInterval::MinorThird,
        SimpleInterval::MajorThird,
        SimpleInterval::PerfectFourth,
        SimpleInterval::Tritone,
        SimpleInterval::PerfectFifth,
        SimpleInterval::MinorSixth,
        SimpleInterval::MajorSixth,
        SimpleInterval::MinorSeventh,
        SimpleInterval::MajorSeventh,
    ];
}

/// A musical interval: a simple interval plus signed octave offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Interval {
    pub simple: SimpleInterval,
    pub octaves: i8,
}

impl Interval {
    pub const UNISON: Self = Self::simple(SimpleInterval::Unison);
    pub const MINOR_SECOND: Self = Self::simple(SimpleInterval::MinorSecond);
    pub const MAJOR_SECOND: Self = Self::simple(SimpleInterval::MajorSecond);
    pub const MINOR_THIRD: Self = Self::simple(SimpleInterval::MinorThird);
    pub const MAJOR_THIRD: Self = Self::simple(SimpleInterval::MajorThird);
    pub const PERFECT_FOURTH: Self = Self::simple(SimpleInterval::PerfectFourth);
    pub const TRITONE: Self = Self::simple(SimpleInterval::Tritone);
    pub const PERFECT_FIFTH: Self = Self::simple(SimpleInterval::PerfectFifth);
    pub const MINOR_SIXTH: Self = Self::simple(SimpleInterval::MinorSixth);
    pub const MAJOR_SIXTH: Self = Self::simple(SimpleInterval::MajorSixth);
    pub const MINOR_SEVENTH: Self = Self::simple(SimpleInterval::MinorSeventh);
    pub const MAJOR_SEVENTH: Self = Self::simple(SimpleInterval::MajorSeventh);
    pub const OCTAVE: Self = Self {
        simple: SimpleInterval::Unison,
        octaves: 1,
    };

    const fn simple(s: SimpleInterval) -> Self {
        Self {
            simple: s,
            octaves: 0,
        }
    }

    pub const fn compound(simple: SimpleInterval, octaves: i8) -> Self {
        Self { simple, octaves }
    }

    pub fn semitones(self) -> i32 {
        self.simple as i32 + self.octaves as i32 * 12
    }

    pub fn to_ratio(self) -> f32 {
        (2.0f32).powf(self.semitones() as f32 / 12.0)
    }

    pub fn negate(self) -> Self {
        Self {
            simple: self.simple,
            octaves: -self.octaves - if self.simple as u8 > 0 { 1 } else { 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chromatic_has_all() {
        assert_eq!(Scale::chromatic().bits().count_ones(), 12);
    }

    #[test]
    fn major_c() {
        let s = Scale::major(Note::C);
        assert!(s.contains(Note::C));
        assert!(!s.contains(Note::CS));
        assert!(s.contains(Note::D));
        assert!(!s.contains(Note::DS));
        assert!(s.contains(Note::E));
        assert!(s.contains(Note::F));
        assert!(!s.contains(Note::FS));
        assert!(s.contains(Note::G));
        assert!(!s.contains(Note::GS));
        assert!(s.contains(Note::A));
        assert!(!s.contains(Note::AS));
        assert!(s.contains(Note::B));
    }

    #[test]
    fn major_g() {
        let s = Scale::major(Note::G);
        assert!(s.contains(Note::G));
        assert!(s.contains(Note::A));
        assert!(s.contains(Note::B));
        assert!(s.contains(Note::C));
        assert!(s.contains(Note::D));
        assert!(s.contains(Note::E));
        assert!(s.contains(Note::FS));
        assert!(!s.contains(Note::F));
    }

    #[test]
    fn empty_snaps_to_chromatic() {
        let p = Scale::empty().nearest_pitch(445.0);
        assert_eq!(p.note, Note::A);
        assert_eq!(p.octave, 4);
    }

    #[test]
    fn chromatic_snaps_to_a() {
        let corrected = Scale::chromatic().nearest_pitch(445.0).to_freq();
        approx::assert_abs_diff_eq!(corrected, 440.0, epsilon = 2.0);
    }

    #[test]
    fn pentatonic_c() {
        let corrected = Scale::pentatonic(Note::C).nearest_pitch(349.23).to_freq();
        assert!(
            (corrected - 329.63).abs() < 2.0 || (corrected - 392.0).abs() < 2.0,
            "Expected E4 or G4, got {corrected}"
        );
    }

    #[test]
    fn custom_note_set() {
        let s = Note::C | Note::G;
        let corrected = s.nearest_pitch(349.23).to_freq();
        assert!(
            (corrected - 261.63).abs() < 2.0 || (corrected - 392.0).abs() < 2.0,
            "Expected C4 or G4, got {corrected}"
        );
    }

    #[test]
    fn pitch_round_trip() {
        let p = Pitch::new(Note::A, 4);
        approx::assert_abs_diff_eq!(p.to_freq(), 440.0, epsilon = 0.1);
        let p2 = Pitch::from_freq(440.0);
        assert_eq!(p2, p);
    }

    #[test]
    fn nearest_pitch_returns_correct_note() {
        let p = Scale::chromatic().nearest_pitch(445.0);
        assert_eq!(p.note, Note::A);
        assert_eq!(p.octave, 4);
    }

    #[test]
    fn compound_interval() {
        let tenth = Interval::compound(SimpleInterval::MinorThird, 1);
        assert_eq!(tenth.semitones(), 15);
    }

    #[test]
    fn octave_interval() {
        assert_eq!(Interval::OCTAVE.semitones(), 12);
        approx::assert_abs_diff_eq!(Interval::OCTAVE.to_ratio(), 2.0, epsilon = 0.001);
    }
}
