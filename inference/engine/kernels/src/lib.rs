//! Checked, generated bindings for the engine-owned numerical kernel catalog.
//!
//! The build script links `kernels` with `seismic-std`, checks the whole
//! source module once, and emits this typed surface.  Engine code uses these
//! entry modules exclusively; it never assembles compiler or runtime bindings.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));

/// Words of an expert-streaming list of `entries` entries (`routed_copy`,
/// `routed_convert`, `routed_seed_commit`): the sequence and count,
/// then each entry's from and to words. Entry `entries` begins there.
pub fn list_extent(entries: u64) -> u64 {
    constants::LIST_ENTRIES + constants::LIST_ENTRY_WORDS * entries
}

/// The blocks of a list's whole launch grid, LIST_GROUPS by LIST_ROWS
/// (streaming_words.seismic's list grid): a copy within device memory's,
/// which its host writes as the grid's first word (`routed_list_grid`). A
/// copy over the link takes fewer.
pub const LIST_BLOCKS: u32 = {
    let blocks = constants::LIST_GROUPS * constants::LIST_ROWS;
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

// Each kind's payload begins at SLOT_PAYLOAD with its count (the scratch
// kind's, SCRATCH_STAGED), and a scratch lane is SCRATCH_SLOT words: the
// slot's words and its payload, even.
const _: () = {
    assert!(constants::SITE_COUNT == constants::SLOT_PAYLOAD, "a site request's count is its payload's first word");
    assert!(constants::FILE_COUNT == constants::SLOT_PAYLOAD, "a file request's count is its payload's first word");
    assert!(constants::SCRATCH_STAGED == constants::SLOT_PAYLOAD, "a scratch answer's count is its payload's first word");
    assert!(
        (constants::SLOT_PAYLOAD + SCRATCH_PAYLOAD).next_multiple_of(2) == constants::SCRATCH_SLOT,
        "a scratch lane is SCRATCH_SLOT words"
    );
};

/// A scratch lane's payload: SCRATCH_STAGED.
const SCRATCH_PAYLOAD: u64 = constants::SCRATCH_STAGED + 1 - constants::SLOT_PAYLOAD;

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
        Self::new(slots, 1, constants::SITE_ENTRIES - constants::SLOT_PAYLOAD + constants::SITE_ENTRY_WORDS * choices)
    }

    /// The file kind's table (`file[FLIGHT, FW]`) of `choices` choices:
    /// FILE_COUNT, then each choice's entry, its word and its record.
    pub fn file(slots: u64, choices: u64) -> Self {
        Self::new(slots, 1, constants::FILE_ENTRIES - constants::SLOT_PAYLOAD + constants::FILE_ENTRY_WORDS * choices)
    }

    /// The first word of choice `choice`'s entry in a file request.
    pub fn file_entry(choice: u64) -> u64 {
        constants::FILE_ENTRIES + constants::FILE_ENTRY_WORDS * choice
    }

    /// The scratch kind's table (`scratch[FLIGHT, SCRATCH_SLOT LANES]`) of
    /// `rounds` rounds: the request's lane (SCRATCH_ASK), then each round's.
    pub fn scratch(slots: u64, rounds: u64) -> Self {
        Self::new(slots, constants::SCRATCH_ROUND + rounds, SCRATCH_PAYLOAD)
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
