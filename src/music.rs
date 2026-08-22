/// A4 concert pitch in Hz.
const A4_FREQ: f32 = 440.0;
/// A4 is 57 semitones above C0.
const A4_SEMITONES: f32 = 4.0 * 12.0 + 9.0;

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

    /// Conventional name using sharps (e.g. A#).
    pub fn name_sharp(self) -> &'static str {
        [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ][self as usize]
    }

    /// Conventional name using flats (e.g. Bb). Preferred when spelling
    /// flat keys, where "A#" would more correctly read "Bb".
    pub fn name_flat(self) -> &'static str {
        [
            "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
        ][self as usize]
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
        A4_FREQ * (2.0f32).powf((self.semitones_from_c0() - A4_SEMITONES) / 12.0)
    }

    pub fn from_freq(freq: f32) -> Self {
        // Guard the log2 domain: a non-positive or non-finite frequency
        // would yield NaN/-inf semitones and a garbage pitch. Detected
        // frequencies are always positive, so treat anything else as the
        // lowest representable pitch rather than propagating nonsense.
        if !freq.is_finite() || freq <= 0.0 {
            return Self::new(Note::C, 0);
        }
        let semitones = A4_SEMITONES + 12.0 * (freq / A4_FREQ).log2();
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

    /// Minor pentatonic (1 b3 4 5 b7).
    pub fn minor_pentatonic(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MINOR_SEVENTH,
            ],
            root,
        )
    }

    /// Blues scale: minor pentatonic plus the b5 "blue" note.
    pub fn blues(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::TRITONE,
                Interval::PERFECT_FIFTH,
                Interval::MINOR_SEVENTH,
            ],
            root,
        )
    }

    /// Harmonic minor (natural minor with a raised 7th).
    pub fn harmonic_minor(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MINOR_SIXTH,
                Interval::MAJOR_SEVENTH,
            ],
            root,
        )
    }

    /// Ascending melodic minor (raised 6th and 7th).
    pub fn melodic_minor(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MAJOR_SIXTH,
                Interval::MAJOR_SEVENTH,
            ],
            root,
        )
    }

    pub fn dorian(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MAJOR_SIXTH,
                Interval::MINOR_SEVENTH,
            ],
            root,
        )
    }

    pub fn phrygian(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MINOR_SECOND,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MINOR_SIXTH,
                Interval::MINOR_SEVENTH,
            ],
            root,
        )
    }

    pub fn lydian(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MAJOR_THIRD,
                Interval::TRITONE,
                Interval::PERFECT_FIFTH,
                Interval::MAJOR_SIXTH,
                Interval::MAJOR_SEVENTH,
            ],
            root,
        )
    }

    pub fn mixolydian(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MAJOR_SECOND,
                Interval::MAJOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::PERFECT_FIFTH,
                Interval::MAJOR_SIXTH,
                Interval::MINOR_SEVENTH,
            ],
            root,
        )
    }

    pub fn locrian(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::UNISON,
                Interval::MINOR_SECOND,
                Interval::MINOR_THIRD,
                Interval::PERFECT_FOURTH,
                Interval::TRITONE,
                Interval::MINOR_SIXTH,
                Interval::MINOR_SEVENTH,
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

    /// The pitch `steps` scale degrees above `pitch`: the steps-th in-scale
    /// note strictly above it, crossing octaves as needed. A diatonic third
    /// is 2 steps, a fifth 4 steps. Alloc-free (runs on the audio thread).
    /// Returns `pitch` unchanged if the scale is empty.
    pub fn degree_above(self, pitch: Pitch, steps: usize) -> Pitch {
        if self.is_empty() || steps == 0 {
            return pitch;
        }
        let mut idx = pitch.note as usize;
        let mut octave = pitch.octave as i32;
        let mut remaining = steps;
        while remaining > 0 {
            idx += 1;
            if idx >= 12 {
                idx -= 12;
                octave += 1;
            }
            if self.contains(Note::ALL[idx]) {
                remaining -= 1;
            }
        }
        Pitch::new(Note::ALL[idx], octave as i8)
    }

    /// Snap a frequency to the nearest note in this scale.
    /// Returns the nearest chromatic pitch if the scale is empty.
    pub fn nearest_pitch(self, freq: f32) -> Pitch {
        // See Pitch::from_freq: guard the log2 domain against
        // non-positive / non-finite input.
        if !freq.is_finite() || freq <= 0.0 {
            return Pitch::new(Note::C, 0);
        }
        let semitones_from_c0 = A4_SEMITONES + 12.0 * (freq / A4_FREQ).log2();
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
        // Negate the total semitone span, then re-decompose into a
        // canonical (simple in 0..12, signed octaves) pair. The previous
        // implementation only borrowed the octave and left `simple`
        // unchanged, so e.g. a major third (+4) negated to -8 instead of -4.
        let total = -self.semitones();
        Self {
            simple: SimpleInterval::ALL[total.rem_euclid(12) as usize],
            octaves: total.div_euclid(12) as i8,
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn degree_above_walks_the_scale() {
        let cmaj = Scale::major(Note::C);
        // Diatonic third above C4 is E4 (major); above D4 is F4 (minor).
        assert_eq!(
            cmaj.degree_above(Pitch::new(Note::C, 4), 2),
            Pitch::new(Note::E, 4)
        );
        assert_eq!(
            cmaj.degree_above(Pitch::new(Note::D, 4), 2),
            Pitch::new(Note::F, 4)
        );
        // Crossing the octave: third above B3 is D4.
        assert_eq!(
            cmaj.degree_above(Pitch::new(Note::B, 3), 2),
            Pitch::new(Note::D, 4)
        );
        // Diatonic fifth above F4 is C5.
        assert_eq!(
            cmaj.degree_above(Pitch::new(Note::F, 4), 4),
            Pitch::new(Note::C, 5)
        );
        // 7 steps = a full octave in a 7-note scale.
        assert_eq!(
            cmaj.degree_above(Pitch::new(Note::G, 4), 7),
            Pitch::new(Note::G, 5)
        );

        // Pentatonic C (C D E G A): "third" (2 steps) above A3 is D4.
        let pent = Scale::pentatonic(Note::C);
        assert_eq!(
            pent.degree_above(Pitch::new(Note::A, 3), 2),
            Pitch::new(Note::D, 4)
        );

        // Out-of-scale start walks to in-scale targets all the same.
        assert_eq!(
            cmaj.degree_above(Pitch::new(Note::CS, 4), 2),
            Pitch::new(Note::E, 4)
        );
        // Empty scale: unchanged.
        assert_eq!(
            Scale::empty().degree_above(Pitch::new(Note::A, 4), 2),
            Pitch::new(Note::A, 4)
        );
    }
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

    #[test]
    fn non_positive_freq_is_guarded() {
        // These must not panic nor produce NaN/inf frequencies.
        for bad in [0.0f32, -100.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let p = Pitch::from_freq(bad);
            assert!(p.to_freq().is_finite(), "from_freq({bad}) -> non-finite");
            let q = Scale::chromatic().nearest_pitch(bad);
            assert!(
                q.to_freq().is_finite(),
                "nearest_pitch({bad}) -> non-finite"
            );
            let r = Scale::major(Note::C).nearest_pitch(bad);
            assert!(r.to_freq().is_finite());
        }
    }

    #[test]
    fn new_scales_have_expected_notes() {
        // A minor pentatonic: A C D E G
        let am = Scale::minor_pentatonic(Note::A);
        for n in [Note::A, Note::C, Note::D, Note::E, Note::G] {
            assert!(am.contains(n), "A minor pentatonic missing {n:?}");
        }
        assert!(!am.contains(Note::B));

        // A blues adds the b5 (Eb/DS).
        assert!(Scale::blues(Note::A).contains(Note::DS));

        // C harmonic minor: C D Eb F G Ab B (raised 7th).
        let chm = Scale::harmonic_minor(Note::C);
        assert!(chm.contains(Note::B));
        assert!(chm.contains(Note::DS)); // Eb
        assert!(chm.contains(Note::GS)); // Ab

        // Modes of C should all contain the tonic and have 7 notes.
        for s in [
            Scale::dorian(Note::C),
            Scale::phrygian(Note::C),
            Scale::lydian(Note::C),
            Scale::mixolydian(Note::C),
            Scale::locrian(Note::C),
            Scale::melodic_minor(Note::C),
        ] {
            assert!(s.contains(Note::C));
            assert_eq!(s.bits().count_ones(), 7);
        }

        // C lydian raises the 4th (F#), C mixolydian lowers the 7th (Bb/AS).
        assert!(Scale::lydian(Note::C).contains(Note::FS));
        assert!(Scale::mixolydian(Note::C).contains(Note::AS));
    }

    #[test]
    fn note_flat_and_sharp_names() {
        assert_eq!(Note::AS.name_sharp(), "A#");
        assert_eq!(Note::AS.name_flat(), "Bb");
        assert_eq!(Note::DS.name_flat(), "Eb");
        assert_eq!(Note::C.name_flat(), "C");
        assert_eq!(Note::C.name_sharp(), "C");
    }

    #[test]
    fn negate_simple_intervals() {
        assert_eq!(Interval::UNISON.negate().semitones(), 0);
        assert_eq!(Interval::MAJOR_THIRD.negate().semitones(), -4);
        assert_eq!(Interval::PERFECT_FIFTH.negate().semitones(), -7);
        assert_eq!(Interval::MAJOR_SEVENTH.negate().semitones(), -11);
        assert_eq!(Interval::OCTAVE.negate().semitones(), -12);
    }

    #[test]
    fn negate_is_an_involution() {
        for simple in SimpleInterval::ALL {
            for octaves in -2..=2 {
                let iv = Interval::compound(simple, octaves);
                assert_eq!(
                    iv.negate().semitones(),
                    -iv.semitones(),
                    "negate({iv:?}) had wrong span",
                );
                assert_eq!(
                    iv.negate().negate().semitones(),
                    iv.semitones(),
                    "double negate({iv:?}) did not round-trip",
                );
            }
        }
    }

    #[test]
    fn negate_down_fifth_ratio() {
        // A perfect fifth down is the 2:3 ratio (~0.667), not 3:4 (~0.749).
        approx::assert_abs_diff_eq!(
            Interval::PERFECT_FIFTH.negate().to_ratio(),
            2.0 / 3.0,
            epsilon = 0.001
        );
    }
}
