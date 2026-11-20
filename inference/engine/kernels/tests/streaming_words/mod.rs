//! The kernel tests' encoders of streaming_words.seismic's u64 steps: each
//! as two i32 words, the low word first.

use magnitude_kernels::constants;

/// A step's words: its pair (STEP_LOW, STEP_HIGH), and no front row.
pub fn step_words(q: u64) -> Vec<i32> {
    magnitude_kernels::step_words(q, 0).to_vec()
}

/// The map rows `[N, MAP_WORDS]` of (record, from, until) entries.
pub fn map_words(rows: &[(i32, u64, u64)]) -> Vec<i32> {
    let mut words = vec![0; rows.len() * constants::MAP_WORDS as usize];
    for (row, &(record, from, until)) in words.chunks_exact_mut(constants::MAP_WORDS as usize).zip(rows) {
        row[constants::MAP_RECORD as usize] = record;
        for (at, value) in [(constants::MAP_FROM, from), (constants::MAP_UNTIL, until)] {
            row[at as usize] = value as u32 as i32;
            row[at as usize + 1] = (value >> 32) as u32 as i32;
        }
    }
    words
}
