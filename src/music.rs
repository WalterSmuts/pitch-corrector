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

/// A set of notes forming a scale.
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
            bits |= Note::ALL[((root_idx + interval.semitones() as u8) % 12) as usize].bit();
        }
        Self(bits)
    }

    pub fn major(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::Unison,
                Interval::MajorSecond,
                Interval::MajorThird,
                Interval::PerfectFourth,
                Interval::PerfectFifth,
                Interval::MajorSixth,
                Interval::MajorSeventh,
            ],
            root,
        )
    }

    pub fn minor(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::Unison,
                Interval::MajorSecond,
                Interval::MinorThird,
                Interval::PerfectFourth,
                Interval::PerfectFifth,
                Interval::MinorSixth,
                Interval::MinorSeventh,
            ],
            root,
        )
    }

    pub fn pentatonic(root: Note) -> Self {
        Self::from_intervals(
            &[
                Interval::Unison,
                Interval::MajorSecond,
                Interval::MajorThird,
                Interval::PerfectFifth,
                Interval::MajorSixth,
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

    pub fn nearest_note(self, freq: f32) -> f32 {
        if self.is_empty() {
            return freq;
        }
        let semitones_from_c0 = 12.0 * (freq / 16.3516).log2();
        let octave = (semitones_from_c0 / 12.0).floor();
        let semitone_in_octave = semitones_from_c0 - octave * 12.0;

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
        16.3516 * (2.0f32).powf(target_semitones / 12.0)
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

/// A musical interval measured in semitones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i8)]
pub enum Interval {
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
    Octave = 12,
}

impl Interval {
    pub fn semitones(self) -> i8 {
        self as i8
    }

    pub fn to_ratio(self) -> f32 {
        (2.0f32).powf(self.semitones() as f32 / 12.0)
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
    fn empty_returns_input() {
        assert_eq!(Scale::empty().nearest_note(445.0), 445.0);
    }

    #[test]
    fn chromatic_snaps_to_a() {
        let corrected = Scale::chromatic().nearest_note(445.0);
        approx::assert_abs_diff_eq!(corrected, 440.0, epsilon = 2.0);
    }

    #[test]
    fn pentatonic_c() {
        let corrected = Scale::pentatonic(Note::C).nearest_note(349.23);
        assert!(
            (corrected - 329.63).abs() < 2.0 || (corrected - 392.0).abs() < 2.0,
            "Expected E4 or G4, got {corrected}"
        );
    }

    #[test]
    fn custom_note_set() {
        let s = Note::C | Note::G;
        let corrected = s.nearest_note(349.23);
        assert!(
            (corrected - 261.63).abs() < 2.0 || (corrected - 392.0).abs() < 2.0,
            "Expected C4 or G4, got {corrected}"
        );
    }
}
