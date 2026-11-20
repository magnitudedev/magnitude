//! Checked, generated bindings for the engine-owned numerical kernel catalog.
//!
//! The build script links `kernels` with `seismic-std`, checks the whole
//! source module once, and emits this typed surface.  Engine code uses these
//! entry modules exclusively; it never assembles compiler or runtime bindings.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));

/// The threadgroups of a decode landing launch (`routed_land`) as the grid
/// word its route writes while it lands (`resolve`'s `land_groups`):
/// streaming_words.seismic's LAND_THREADGROUPS, which every backend's
/// declaration calls, where it binds staging records (`staged`, V > 0: the
/// bank places) or none (the host tier places).
pub fn land_groups(staged: bool) -> i32 {
    const fn word(staged: bool) -> i32 {
        let groups = constants::LAND_THREADGROUPS(staged as u64);
        assert!(groups <= i32::MAX as u64, "a grid word holds the landing groups");
        groups as i32
    }
    const GROUPS: [i32; 2] = [word(false), word(true)];
    GROUPS[staged as usize]
}

/// A decode step's words (`routed_step`'s `step` and `sequence`): the u64
/// `sequence` as its pair (STEP_LOW, STEP_HIGH), and the rows of the step
/// that are front rows as STEP_FRONT's bits (bit r: row r).
pub fn step_words(sequence: u64, front: u32) -> [i32; constants::STEP_WORDS as usize] {
    let mut words = [0; constants::STEP_WORDS as usize];
    words[constants::STEP_LOW as usize] = sequence as u32 as i32;
    words[constants::STEP_HIGH as usize] = (sequence >> 32) as u32 as i32;
    words[constants::STEP_FRONT as usize] = front as i32;
    words
}

/// Front row `row`'s field of a step's STEP_GUESSES word (`routed_step`;
/// a ring's RING_GUESSES, streaming_words.seismic's STEP_GUESS_BITS from
/// STEP_GUESS_SHIFT(row)): the kind word of the guess it ran on
/// (guess.seismic's GUESS_CHAINED, GUESS_HEAD), 0 for a row that is no
/// front row.
pub fn guess_kind(word: i32, row: usize) -> i32 {
    assert!(row < constants::VERIFY_ROWS as usize, "a front row is below VERIFY_ROWS");
    (word >> constants::STEP_GUESS_SHIFT(row as u64)) & ((1 << constants::STEP_GUESS_BITS) - 1)
}

/// The STEP_GUESSES word of front rows `rows`, each a (row, kind word) of a
/// front row and the kind of the guess it runs on: what `routed_step` writes
/// of them. Panics on a row at or past VERIFY_ROWS, row 0 (no front row
/// precedes its exact row), a duplicate row, or a word no guess kind's, which
/// no field holds.
pub fn step_guesses(rows: impl IntoIterator<Item = (usize, i32)>) -> i32 {
    rows.into_iter().fold(0, |word, (row, kind)| {
        assert!(row > 0 && row < constants::VERIFY_ROWS as usize, "front row {row} is past row 0 and below VERIFY_ROWS");
        assert!(
            [constants::GUESS_CHAINED, constants::GUESS_HEAD].contains(&(kind as u64)),
            "{kind} is no guess kind's word"
        );
        assert!(
            guess_kind(word, row) == 0,
            "front row {row} appears more than once in step_guesses"
        );
        word | kind << constants::STEP_GUESS_SHIFT(row as u64)
    })
}

/// The blocks of a list's whole launch grid (streaming_words.seismic's
/// LIST_BLOCKS, every backend's `routed_copy` bound): a copy within device
/// memory's, which its host writes as the grid's first word
/// (`routed_list_grid`). A copy over the link takes fewer.
pub const LIST_BLOCKS: u32 = {
    let blocks = constants::LIST_BLOCKS();
    assert!(blocks <= i32::MAX as u64, "a grid word holds a list's blocks");
    blocks as u32
};

/// The words of a decode step's sources `[SOURCE_ROWS, M, K]` (the
/// resolve's snapshot) of the choices (row-major, M K of them): each
/// choice's record for the GPU (`records`: a bank record, -2 - t for a tier
/// record read in place, -1 for none) and its tier record for the CPU
/// (`sites`, -1 for none).
pub fn source_words(records: &[i32], sites: &[i32]) -> Vec<i32> {
    assert_eq!(records.len(), sites.len(), "a source word for each choice of each row");
    let count = records.len();
    let mut words = vec![0; constants::SOURCE_ROWS as usize * count];
    words[constants::SOURCE_RECORD as usize * count..][..count].copy_from_slice(records);
    words[constants::SOURCE_SITE as usize * count..][..count].copy_from_slice(sites);
    words
}

// Each kind's payload begins at SLOT_PAYLOAD with its count (a scratch
// window's round-0 lane: SCRATCH_STAGED, then SCRATCH_FOLLOWED), and a
// scratch lane is SCRATCH_SLOT words: the slot's words and its payload, even.
const _: () = {
    assert!(constants::STEP_GUESS_SHIFT(constants::VERIFY_ROWS) <= 32, "a step's STEP_GUESSES word holds a kind field per front row");
    assert!(
        constants::GUESS_CHAINED < 1 << constants::STEP_GUESS_BITS && constants::GUESS_HEAD < 1 << constants::STEP_GUESS_BITS,
        "every guess kind's word fits a STEP_GUESSES field"
    );
    assert!(constants::SITE_COUNT == constants::SLOT_PAYLOAD, "a site request's count is its payload's first word");
    assert!(constants::FILE_COUNT == constants::SLOT_PAYLOAD, "a file request's count is its payload's first word");
    assert!(constants::SCRATCH_STAGED == constants::SLOT_PAYLOAD, "a scratch window's count is its payload's first word");
    assert!(constants::SCRATCH_FOLLOWED == constants::SCRATCH_STAGED + 1, "a scratch window's fill order follows its count");
    assert!(
        constants::LANDING_HEAD % 2 == 0 && constants::LANDING_ENTRIES == constants::LANDING_HEAD + 2,
        "a landing list's head is one u64 at an even word, its entries after it"
    );
    assert!(
        constants::LAND_WINDOW_AT == 4 * constants::LAND_GROUP_WORDS && constants::LAND_WINDOW_AT % 16 == 0,
        "a landing launch's window follows its group words, 16-byte aligned"
    );
    assert!(
        (constants::SLOT_PAYLOAD + SCRATCH_PAYLOAD).next_multiple_of(2) == constants::SCRATCH_SLOT,
        "a scratch lane is SCRATCH_SLOT words"
    );
};

/// A scratch lane's payload: SCRATCH_STAGED and SCRATCH_FOLLOWED.
const SCRATCH_PAYLOAD: u64 = constants::SCRATCH_FOLLOWED + 1 - constants::SLOT_PAYLOAD;

/// The layout of one table of the exchange (streaming_words.seismic's
/// SLOT_*), the one statement of it the host and every graph read: `slots`
/// slots (step q's the slot q mod `slots`) of `lanes` lanes each, a lane
/// `width` i32 words, the table's rows its slots, `[slots, lanes width]`.
/// A lane's width is even by construction, so every lane, and with it its
/// SLOT_REQUEST and SLOT_ANSWER, begins at a whole u64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotTable {
    slots: u64,
    lanes: u64,
    width: u64,
}

impl SlotTable {
    fn new(slots: u64, lanes: u64, payload: u64) -> Self {
        assert!(slots > 0 && lanes > 0, "a slot table of {slots} slots of {lanes} lanes");
        Self { slots, lanes, width: (constants::SLOT_PAYLOAD + payload).next_multiple_of(2) }
    }

    /// The site kind's table (`site[FLIGHT, R]`), its requests of at most
    /// `choices` entries: SITE_COUNT, then the entries.
    pub fn site(slots: u64, choices: u64) -> Self {
        Self::new(slots, 1, constants::SITE_ENTRY(choices) - constants::SLOT_PAYLOAD)
    }

    /// The file kind's table (`file[FLIGHT, FW]`) of `choices` choices:
    /// FILE_COUNT, then each choice's entry, its word and its record.
    pub fn file(slots: u64, choices: u64) -> Self {
        Self::new(slots, 1, constants::FILE_ENTRY(choices) - constants::SLOT_PAYLOAD)
    }

    /// The scratch kind's table (`scratch[FLIGHT, SCRATCH_SLOT LANES]`) of
    /// `rounds` round lanes across the windows (`P * RW`): the request's lane
    /// (SCRATCH_ASK), then each window's round lanes.
    pub fn scratch(slots: u64, rounds: u64) -> Self {
        Self::new(slots, constants::ROUND_LANE(rounds), SCRATCH_PAYLOAD)
    }

    /// Its slots, `FLIGHT`.
    pub fn slots(self) -> u64 {
        self.slots
    }

    /// A slot's lanes (`LANES` of the scratch table, 1 of the others).
    pub fn lanes(self) -> u64 {
        self.lanes
    }

    /// A lane's words: `R`, `FW` or SCRATCH_SLOT.
    pub fn width(self) -> u64 {
        self.width
    }

    /// The table's extents, `[slots, lanes width]`.
    pub fn extents(self) -> [u64; 2] {
        [self.slots, self.lanes * self.width]
    }

    /// Its words.
    pub fn words(self) -> u64 {
        self.slots * self.lanes * self.width
    }

    /// The first word of lane `lane` of step `step`'s slot.
    pub fn lane(self, step: u64, lane: u64) -> u64 {
        assert!(lane < self.lanes, "lane {lane} of {}", self.lanes);
        ((step % self.slots) * self.lanes + lane) * self.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_guesses_packs_distinct_front_rows() {
        let chained = constants::GUESS_CHAINED as i32;
        let head = constants::GUESS_HEAD as i32;
        let word = step_guesses([(1, chained), (2, head)]);
        assert_eq!(guess_kind(word, 0), 0);
        assert_eq!(guess_kind(word, 1), chained);
        assert_eq!(guess_kind(word, 2), head);
    }

    #[test]
    #[should_panic(expected = "more than once")]
    fn step_guesses_panics_on_duplicate_row() {
        let chained = constants::GUESS_CHAINED as i32;
        let head = constants::GUESS_HEAD as i32;
        let _ = step_guesses([(1, chained), (1, head)]);
    }
}
