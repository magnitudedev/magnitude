use crate::{ClassError, ClassLimits, Demand, LaunchClass};
use std::{fmt, sync::Arc};

pub const HISTORY_WIDTH: usize = 64;
pub const SHAPING_WIDTH: usize = 8;

/// One sampler invocation. The packed representation is the six-word draw
/// record consumed by selection kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Draw {
    pub kind: DrawKind,
    pub seed: u64,
    pub position: u64,
    pub domain: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DrawKind {
    Greedy = 0,
    Categorical = 1,
}

/// The eight f32 values consumed by the shaping kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shaping {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub flags: u32,
}

impl Default for Shaping {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            flags: 0,
        }
    }
}

impl Shaping {
    /// The validated control values a selection uploads for `shape_rows`.
    pub fn params(self) -> Result<[f32; SHAPING_WIDTH], PackError> {
        Ok(self.words()?.map(f32::from_bits))
    }

    fn words(self) -> Result<[u32; SHAPING_WIDTH], PackError> {
        let finite = [
            self.temperature,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        ]
        .into_iter()
        .all(f32::is_finite);
        // top_k and flags are represented exactly in an f32 control tensor.
        const MAX_EXACT_F32_INTEGER: u32 = 1 << 24;
        let ranges_valid = self.temperature >= 0.0
            && 0.0 < self.top_p
            && self.top_p <= 1.0
            && (0.0..=1.0).contains(&self.min_p)
            && self.repetition_penalty > 0.0;
        if !(finite && ranges_valid)
            || self.top_k > MAX_EXACT_F32_INTEGER
            || self.flags > MAX_EXACT_F32_INTEGER
        {
            return Err(PackError::InvalidShaping);
        }
        Ok([
            self.temperature.to_bits(),
            (self.top_k as f32).to_bits(),
            self.top_p.to_bits(),
            self.min_p.to_bits(),
            self.repetition_penalty.to_bits(),
            self.presence_penalty.to_bits(),
            self.frequency_penalty.to_bits(),
            (self.flags as f32).to_bits(),
        ])
    }
}

/// Selection controls attached to a row carrying `Demand::SELECT`.
#[derive(Clone, Debug, PartialEq)]
pub struct Select {
    pub draw: Draw,
    /// One packed vocabulary mask, or `None` for an unconstrained row. The
    /// mask is shared with its producer, never copied into the batch.
    pub mask: Option<Arc<[u32]>>,
    pub shaping: Shaping,
    /// Recent accepted tokens, oldest to newest. Missing entries are padded
    /// with -1 on the right.
    pub history: Vec<i32>,
}

/// A logical model row. Rows remain in the order supplied inside their slot.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub token: i32,
    pub coordinates: [i32; 4],
    /// One entry per history domain of the slot's store, in domain order.
    /// Every row of a batch has the same number.
    pub histories: Vec<RowHistory>,
    pub demand: Demand,
    pub select: Option<Select>,
}

/// A row's history in one history domain.
#[derive(Clone, Debug, PartialEq)]
pub struct RowHistory {
    /// Accepted-history ranges of the domain this row reads. Each range stays
    /// within one slab.
    pub visible: Vec<[i32; 2]>,
    /// The first row of its slot, by offset in the slot, whose fresh history
    /// this row reads: 0 unless the domain's window starts inside the slot.
    pub fresh_start: i32,
    /// For a media row attending its whole span bidirectionally, the offset
    /// in its slot after the span's last row: it reads the fresh rows up to
    /// there. A causal row reads the fresh rows through itself.
    pub bidirectional_end: Option<i32>,
    /// The domain row this row's history is written to, or -1.
    pub destination: i32,
}

/// One history domain's row tables: per row its visible ranges, its fresh
/// batch rows `[start, end)` and its destination.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryTables {
    pub visible: Vec<Vec<[i32; 2]>>,
    pub fresh: Vec<[i32; 2]>,
    pub destinations: Vec<i32>,
}

/// A request-local run of rows. Slots are packed in scheduler order.
/// Recurrent entries read the slot's state from version (`bank`,
/// `previous_tape`): the bank's state advanced by that many of its tape rows.
/// They publish the state after the first `stop` rows to bank
/// `following_bank` and record the rows after `stop` on its tape. A
/// successor is never the zero seed (bank 0), the slot's own bank, or another
/// slot's successor.
#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub rows: Vec<Row>,
    pub bank: i32,
    pub previous_tape: i32,
    pub following_bank: i32,
    pub stop: i32,
}

/// Fully padded logical row tables.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedRowTables {
    pub class: LaunchClass,
    pub actual_rows: usize,
    pub actual_slots: usize,
    pub slots: usize,
    pub mask_words: usize,
    pub mask_count: usize,
    pub tokens: Vec<i32>,
    pub coordinates: Vec<[i32; 4]>,
    /// One table set per history domain, in domain order.
    pub histories: Vec<HistoryTables>,
    pub row_slots: Vec<i32>,
    pub demand: Vec<u32>,
    pub segments: Vec<[i32; 2]>,
    pub bank: Vec<i32>,
    pub previous_tape: Vec<i32>,
    pub following_bank: Vec<i32>,
    pub stop: Vec<i32>,
    pub plane_base: Vec<i32>,
    pub out_rows: Vec<i32>,
    pub select_rows: Vec<i32>,
    pub draws: Vec<[u32; 6]>,
    /// Per selected row, its index in `masks`, or -1 when unconstrained.
    pub mask_rows: Vec<i32>,
    pub masks: Vec<Arc<[u32]>>,
    pub shaping: Vec<[f32; SHAPING_WIDTH]>,
    pub history: Vec<[i32; HISTORY_WIDTH]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackError {
    Class(ClassError),
    EmptySlot {
        slot: usize,
    },
    IntegerOverflow(&'static str),
    InvalidBank {
        slot: usize,
        bank: i32,
    },
    InvalidFollowingBank {
        slot: usize,
        bank: i32,
        following_bank: i32,
    },
    /// A negative read tape, or a publication point outside the slot's rows.
    InvalidRecurrentVersion {
        slot: usize,
        previous_tape: i32,
        stop: i32,
    },
    InvalidToken {
        row: usize,
        token: i32,
    },
    InvalidCoordinates {
        row: usize,
        coordinates: [i32; 4],
    },
    InvalidDestination {
        row: usize,
        destination: i32,
    },
    /// A row whose history domain count differs from the batch's first row.
    HistoryDomains {
        row: usize,
        expected: usize,
        actual: usize,
    },
    /// A fresh span that does not hold the row itself within its slot.
    InvalidFreshSpan {
        row: usize,
        fresh: [i32; 2],
    },
    InvalidVisibleRange {
        row: usize,
        range: [i32; 2],
    },
    OverlappingVisibleRanges {
        row: usize,
    },
    SelectMismatch {
        row: usize,
    },
    InvalidDrawDomain {
        row: usize,
        domain: u32,
    },
    InvalidMaskWidth {
        row: usize,
        expected: usize,
        actual: usize,
    },
    HistoryTooLong {
        row: usize,
        actual: usize,
    },
    InvalidShaping,
    InvalidHeadDemand {
        row: usize,
        demand: Demand,
    },
    /// A head slot whose proposal or chained row count differs from the
    /// batch's step count.
    HeadSteps {
        slot: usize,
    },
    EmptyVocabulary,
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Class(error) => error.fmt(f),
            Self::EmptySlot { slot } => write!(f, "slot {slot} has no rows"),
            Self::IntegerOverflow(field) => {
                write!(f, "{field} does not fit the row-table representation")
            }
            Self::InvalidBank { slot, bank } => write!(f, "slot {slot} has invalid bank {bank}"),
            Self::InvalidFollowingBank {
                slot,
                bank,
                following_bank,
            } => write!(
                f,
                "slot {slot} successor bank {following_bank} is the zero seed, its own bank {bank}, or another slot's successor"
            ),
            Self::InvalidRecurrentVersion {
                slot,
                previous_tape,
                stop,
            } => write!(
                f,
                "slot {slot} reads tape rows {previous_tape} or publishes after {stop} rows outside the slot"
            ),
            Self::InvalidToken { row, token } => write!(f, "row {row} has invalid token {token}"),
            Self::InvalidCoordinates { row, coordinates } => {
                write!(f, "row {row} has invalid coordinates {coordinates:?}")
            }
            Self::InvalidDestination { row, destination } => {
                write!(f, "row {row} has invalid destination {destination}")
            }
            Self::HistoryDomains {
                row,
                expected,
                actual,
            } => write!(
                f,
                "row {row} has {actual} history domains; the batch has {expected}"
            ),
            Self::InvalidFreshSpan { row, fresh } => {
                write!(f, "row {row} has invalid fresh span [{}, {})", fresh[0], fresh[1])
            }
            Self::InvalidVisibleRange { row, range } => write!(
                f,
                "row {row} has invalid visible range [{}, {})",
                range[0], range[1]
            ),
            Self::OverlappingVisibleRanges { row } => {
                write!(f, "row {row} has overlapping visible ranges")
            }
            Self::SelectMismatch { row } => write!(
                f,
                "row {row} selection controls do not match its SELECT demand"
            ),
            Self::InvalidDrawDomain { row, domain } => {
                write!(f, "row {row} has invalid draw domain {domain}")
            }
            Self::InvalidMaskWidth {
                row,
                expected,
                actual,
            } => write!(f, "row {row} mask has {actual} words; expected {expected}"),
            Self::HistoryTooLong { row, actual } => write!(
                f,
                "row {row} history has {actual} tokens; limit is {HISTORY_WIDTH}"
            ),
            Self::InvalidShaping => f.write_str("invalid shaping parameters"),
            Self::InvalidHeadDemand { row, demand } => write!(
                f,
                "head row {row} has incompatible demand bits {}",
                demand.bits()
            ),
            Self::HeadSteps { slot } => write!(
                f,
                "head slot {slot} has a proposal or chained row count other than the batch's steps"
            ),
            Self::EmptyVocabulary => f.write_str("vocabulary must contain at least one token"),
        }
    }
}

impl std::error::Error for PackError {}

impl From<ClassError> for PackError {
    fn from(value: ClassError) -> Self {
        Self::Class(value)
    }
}

impl PackedRowTables {
    /// Pack scheduler-ordered slots into the fixed row-table contract.
    pub fn pack(
        slots: &[Slot],
        vocabulary_size: usize,
        limits: ClassLimits,
    ) -> Result<Self, PackError> {
        Self::pack_covering(slots, vocabulary_size, limits, 1)
    }

    /// Pack into a launch class covering at least `segments` history ranges
    /// per row: the class a reservation sized from a bound on its rows'
    /// ranges.
    pub fn pack_covering(
        slots: &[Slot],
        vocabulary_size: usize,
        limits: ClassLimits,
        segments: usize,
    ) -> Result<Self, PackError> {
        if vocabulary_size == 0 {
            return Err(PackError::EmptyVocabulary);
        }
        let mask_words = vocabulary_size
            .checked_add(31)
            .ok_or(PackError::IntegerOverflow("vocabulary size"))?
            / 32;
        let actual_rows = slots.iter().try_fold(0usize, |count, slot| {
            count
                .checked_add(slot.rows.len())
                .ok_or(PackError::IntegerOverflow("row count"))
        })?;
        let actual_slots = slots.len();
        let padded_slots = actual_slots
            .checked_next_power_of_two()
            .ok_or(PackError::IntegerOverflow("slot count"))?;

        let domains = slots
            .first()
            .and_then(|slot| slot.rows.first())
            .map_or(0, |row| row.histories.len());
        let mut index = 0usize;
        let mut max_segments = segments.max(1);
        let mut union = Demand::NONE;
        for (slot_index, slot) in slots.iter().enumerate() {
            if slot.rows.is_empty() {
                return Err(PackError::EmptySlot { slot: slot_index });
            }
            if slot.bank < 0 {
                return Err(PackError::InvalidBank {
                    slot: slot_index,
                    bank: slot.bank,
                });
            }
            if slot.following_bank <= 0
                || slot.following_bank == slot.bank
                || slots[..slot_index]
                    .iter()
                    .any(|other| other.following_bank == slot.following_bank)
            {
                return Err(PackError::InvalidFollowingBank {
                    slot: slot_index,
                    bank: slot.bank,
                    following_bank: slot.following_bank,
                });
            }
            if slot.previous_tape < 0
                || slot.stop < 1
                || usize::try_from(slot.stop).is_ok_and(|stop| stop > slot.rows.len())
            {
                return Err(PackError::InvalidRecurrentVersion {
                    slot: slot_index,
                    previous_tape: slot.previous_tape,
                    stop: slot.stop,
                });
            }
            for (offset, row) in slot.rows.iter().enumerate() {
                if row.token < 0 {
                    return Err(PackError::InvalidToken {
                        row: index,
                        token: row.token,
                    });
                }
                if row.coordinates.iter().any(|coordinate| *coordinate < 0) {
                    return Err(PackError::InvalidCoordinates {
                        row: index,
                        coordinates: row.coordinates,
                    });
                }
                if row.histories.len() != domains {
                    return Err(PackError::HistoryDomains {
                        row: index,
                        expected: domains,
                        actual: row.histories.len(),
                    });
                }
                for history in &row.histories {
                    if history.destination < -1 {
                        return Err(PackError::InvalidDestination {
                            row: index,
                            destination: history.destination,
                        });
                    }
                    let fresh_end = fresh_end(history, offset)?;
                    let start = usize::try_from(history.fresh_start);
                    let end = usize::try_from(fresh_end);
                    if !matches!(
                        (start, end),
                        (Ok(start), Ok(end)) if start <= offset && offset < end && end <= slot.rows.len()
                    ) {
                        return Err(PackError::InvalidFreshSpan {
                            row: index,
                            fresh: [history.fresh_start, fresh_end],
                        });
                    }
                    validate_visible(index, &history.visible)?;
                    max_segments = max_segments.max(history.visible.len());
                }
                let has_select = row.select.is_some();
                if has_select != row.demand.contains(Demand::SELECT) {
                    return Err(PackError::SelectMismatch { row: index });
                }
                union |= row.demand;
                index += 1;
            }
        }
        let class = LaunchClass::covering(actual_rows, max_segments, union, limits)?;
        let m = class.rows();
        let r = class.segments();
        let m_i32 = i32::try_from(m).map_err(|_| PackError::IntegerOverflow("class rows"))?;
        let b_i32 =
            i32::try_from(padded_slots).map_err(|_| PackError::IntegerOverflow("class slots"))?;

        let mut packed = Self {
            class,
            actual_rows,
            actual_slots,
            slots: padded_slots,
            mask_words,
            mask_count: 0,
            tokens: vec![0; m],
            coordinates: vec![[0; 4]; m],
            histories: vec![
                HistoryTables {
                    visible: vec![vec![[0; 2]; r]; m],
                    fresh: vec![[0; 2]; m],
                    destinations: vec![-1; m],
                };
                domains
            ],
            row_slots: vec![b_i32; m],
            demand: vec![0; m],
            segments: vec![[m_i32, m_i32]; padded_slots + 1],
            bank: vec![-1; padded_slots + 1],
            previous_tape: vec![0; padded_slots + 1],
            following_bank: vec![-1; padded_slots + 1],
            stop: vec![0; padded_slots + 1],
            plane_base: vec![0; padded_slots + 1],
            out_rows: Vec::new(),
            select_rows: Vec::new(),
            draws: Vec::new(),
            mask_rows: Vec::new(),
            masks: Vec::new(),
            shaping: Vec::new(),
            history: Vec::new(),
        };

        let mut row_index = 0;
        for (slot_index, slot) in slots.iter().enumerate() {
            let lo = row_index;
            let hi = lo + slot.rows.len();
            packed.segments[slot_index] =
                [as_i32(lo, "segment start")?, as_i32(hi, "segment end")?];
            packed.bank[slot_index] = slot.bank;
            packed.previous_tape[slot_index] = slot.previous_tape;
            packed.following_bank[slot_index] = slot.following_bank;
            packed.stop[slot_index] = slot.stop;
            for (offset, row) in slot.rows.iter().enumerate() {
                packed.tokens[row_index] = row.token;
                packed.coordinates[row_index] = row.coordinates;
                for (tables, history) in packed.histories.iter_mut().zip(&row.histories) {
                    tables.visible[row_index][..history.visible.len()]
                        .copy_from_slice(&history.visible);
                    // Validated nonnegative and within the slot above.
                    tables.fresh[row_index] = [
                        as_i32(lo + history.fresh_start as usize, "fresh start")?,
                        as_i32(lo + fresh_end(history, offset)? as usize, "fresh end")?,
                    ];
                    tables.destinations[row_index] = history.destination;
                }
                packed.row_slots[row_index] = as_i32(slot_index, "row slot")?;
                packed.demand[row_index] = row.demand.bits();

                if row
                    .demand
                    .intersects(Demand::LOGITS | Demand::FEATURES | Demand::SELECT)
                {
                    let out_index = packed.out_rows.len();
                    packed.out_rows.push(as_i32(row_index, "output row")?);
                    if let Some(select) = &row.select {
                        packed.select_rows.push(as_i32(out_index, "select row")?);
                        append_select(&mut packed, row_index, select)?;
                    }
                }
                row_index += 1;
            }
        }
        packed.mask_count = packed.masks.len();
        Ok(packed)
    }
}

/// Accept the producer's slab boundaries in logical history order and
/// reject ranges that share a row. Arena placement does not follow logical
/// order, so a later range may lie at a lower address than an earlier one.
/// The offset in its slot after the last fresh row the row at `offset`
/// reads (`RowHistory::bidirectional_end`).
fn fresh_end(history: &RowHistory, offset: usize) -> Result<i32, PackError> {
    match history.bidirectional_end {
        Some(end) => Ok(end),
        None => as_i32(offset + 1, "fresh end"),
    }
}

fn validate_visible(row: usize, ranges: &[[i32; 2]]) -> Result<(), PackError> {
    for &range in ranges {
        if range[0] < 0 || range[0] >= range[1] {
            return Err(PackError::InvalidVisibleRange { row, range });
        }
    }
    let mut ordered = ranges.to_vec();
    ordered.sort_unstable();
    if ordered.windows(2).any(|pair| pair[1][0] < pair[0][1]) {
        return Err(PackError::OverlappingVisibleRanges { row });
    }
    Ok(())
}

fn append_select(
    packed: &mut PackedRowTables,
    row: usize,
    select: &Select,
) -> Result<(), PackError> {
    if select.draw.domain > 3 {
        return Err(PackError::InvalidDrawDomain {
            row,
            domain: select.draw.domain,
        });
    }
    let (seed_lo, seed_hi) = split_u64(select.draw.seed);
    let (position_lo, position_hi) = split_u64(select.draw.position);
    packed.draws.push([
        select.draw.kind as u32,
        seed_lo,
        seed_hi,
        position_lo,
        position_hi,
        select.draw.domain,
    ]);
    if let Some(mask) = &select.mask {
        if mask.len() != packed.mask_words {
            return Err(PackError::InvalidMaskWidth {
                row,
                expected: packed.mask_words,
                actual: mask.len(),
            });
        }
        packed
            .mask_rows
            .push(as_i32(packed.masks.len(), "mask row")?);
        packed.masks.push(Arc::clone(mask));
    } else {
        packed.mask_rows.push(-1);
    }
    packed.shaping.push(select.shaping.params()?);
    if select.history.len() > HISTORY_WIDTH {
        return Err(PackError::HistoryTooLong {
            row,
            actual: select.history.len(),
        });
    }
    let mut history = [-1; HISTORY_WIDTH];
    history[..select.history.len()].copy_from_slice(&select.history);
    packed.history.push(history);
    Ok(())
}

fn split_u64(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

fn as_i32(value: usize, field: &'static str) -> Result<i32, PackError> {
    i32::try_from(value).map_err(|_| PackError::IntegerOverflow(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: ClassLimits = ClassLimits {
        rows: 512,
        segments: 63,
    };

    fn row(token: i32, demand: Demand) -> Row {
        Row {
            token,
            coordinates: [token, token, token, 0],
            histories: vec![RowHistory {
                visible: vec![[10, 12], [12, 15], [20, 21]],
                fresh_start: 0,
                bidirectional_end: None,
                destination: token + 100,
            }],
            demand,
            select: None,
        }
    }

    fn selection(mask: Option<Vec<u32>>) -> Select {
        let mask = mask.map(Arc::from);
        Select {
            draw: Draw {
                kind: DrawKind::Categorical,
                seed: 0x1122_3344_5566_7788,
                position: 0x99aa_bbcc_ddee_ff00,
                domain: 0,
            },
            mask,
            shaping: Shaping::default(),
            history: vec![7, 8],
        }
    }

    #[test]
    fn packs_mixed_width_slots_contiguously_and_pads() {
        let slots = vec![
            Slot {
                rows: vec![
                    row(1, Demand::NONE),
                    row(2, Demand::FEATURES),
                    row(3, Demand::NONE),
                ],
                bank: 7,
                previous_tape: 2,
                following_bank: 8,
                stop: 1,
            },
            Slot {
                rows: vec![row(4, Demand::LOGITS)],
                bank: 9,
                previous_tape: 0,
                following_bank: 10,
                stop: 1,
            },
            Slot {
                rows: vec![row(5, Demand::NONE), row(6, Demand::NONE)],
                bank: 11,
                previous_tape: 0,
                following_bank: 12,
                stop: 2,
            },
        ];
        let packed = PackedRowTables::pack(&slots, 33, LIMITS).unwrap();
        assert_eq!((packed.actual_rows, packed.class.rows()), (6, 8));
        assert_eq!((packed.actual_slots, packed.slots), (3, 4));
        assert_eq!(
            packed.segments,
            vec![[0, 3], [3, 4], [4, 6], [8, 8], [8, 8]]
        );
        assert_eq!(packed.bank, vec![7, 9, 11, -1, -1]);
        assert_eq!(packed.previous_tape, vec![2, 0, 0, 0, 0]);
        assert_eq!(packed.following_bank, vec![8, 10, 12, -1, -1]);
        assert_eq!(packed.stop, vec![1, 1, 2, 0, 0]);
        assert_eq!(&packed.row_slots[..6], &[0, 0, 0, 1, 2, 2]);
        assert_eq!(&packed.row_slots[6..], &[4, 4]);
        let history = &packed.histories[0];
        assert_eq!(
            &history.fresh[..6],
            &[[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [4, 6]]
        );
        assert_eq!(&packed.tokens[6..], &[0, 0]);
        assert_eq!(&history.destinations[6..], &[-1, -1]);
        assert_eq!(&history.fresh[6..], &[[0, 0], [0, 0]]);
        assert!(history.visible[6..]
            .iter()
            .flatten()
            .all(|range| *range == [0, 0]));
        assert_eq!(packed.class.segments(), 4);
        assert_eq!(
            history.visible[0],
            vec![[10, 12], [12, 15], [20, 21], [0, 0]]
        );
    }

    /// Each history domain has its own tables; a window domain's rows may
    /// read the slot's fresh rows from a later start.
    #[test]
    fn packs_one_table_set_per_history_domain() {
        let windowed = |token: i32, offset: i32| Row {
            histories: vec![
                RowHistory {
                    visible: vec![[0, 8]],
                    fresh_start: 0,
                    bidirectional_end: None,
                    destination: 8 + offset,
                },
                RowHistory {
                    visible: if offset == 0 { vec![[40, 42]] } else { vec![] },
                    fresh_start: (offset - 1).max(0),
                    bidirectional_end: None,
                    destination: 42 + offset,
                },
            ],
            ..row(token, Demand::NONE)
        };
        let slots = [Slot {
            rows: vec![windowed(1, 0), windowed(2, 1), windowed(3, 2)],
            bank: 0,
            previous_tape: 0,
            following_bank: 1,
            stop: 3,
        }];
        let packed = PackedRowTables::pack(&slots, 33, LIMITS).unwrap();
        assert_eq!(packed.histories.len(), 2);
        assert_eq!(&packed.histories[0].fresh[..3], &[[0, 1], [0, 2], [0, 3]]);
        assert_eq!(&packed.histories[1].fresh[..3], &[[0, 1], [0, 2], [1, 3]]);
        assert_eq!(&packed.histories[1].destinations[..3], &[42, 43, 44]);
        assert_eq!(packed.histories[1].visible[1], vec![[0, 0]]);
        let mut mismatched = slots.clone();
        mismatched[0].rows[1].histories.pop();
        assert!(matches!(
            PackedRowTables::pack(&mismatched, 33, LIMITS),
            Err(PackError::HistoryDomains { row: 1, .. })
        ));
        let mut early = slots.clone();
        early[0].rows[1].histories[1].fresh_start = 2;
        assert!(matches!(
            PackedRowTables::pack(&early, 33, LIMITS),
            Err(PackError::InvalidFreshSpan { row: 1, .. })
        ));
        // A media span's rows read every fresh row of the span; a span end
        // beyond the slot is rejected.
        let mut media = slots.clone();
        for row in &mut media[0].rows {
            row.histories[0].bidirectional_end = Some(3);
        }
        let packed = PackedRowTables::pack(&media, 33, LIMITS).unwrap();
        assert_eq!(&packed.histories[0].fresh[..3], &[[0, 3], [0, 3], [0, 3]]);
        media[0].rows[0].histories[0].bidirectional_end = Some(4);
        assert!(matches!(
            PackedRowTables::pack(&media, 33, LIMITS),
            Err(PackError::InvalidFreshSpan { row: 0, .. })
        ));
    }

    #[test]
    fn demand_readout_and_selection_indices_have_distinct_domains() {
        let mut selected = row(12, Demand::SELECT);
        selected.select = Some(selection(Some(vec![u32::MAX, 1])));
        let mut both = row(13, Demand::SELECT | Demand::FEATURES);
        both.select = Some(selection(None));
        let packed = PackedRowTables::pack(
            &[Slot {
                rows: vec![
                    row(10, Demand::NONE),
                    row(11, Demand::LOGITS),
                    selected,
                    both,
                ],
                bank: 0,
                previous_tape: 0,
                following_bank: 1,
                stop: 1,
            }],
            33,
            LIMITS,
        )
        .unwrap();
        assert_eq!(packed.out_rows, vec![1, 2, 3]);
        assert_eq!(packed.select_rows, vec![1, 2]);
        assert_eq!(packed.mask_rows, vec![0, -1]);
        assert_eq!(packed.masks, vec![Arc::from(vec![u32::MAX, 1])]);
        assert_eq!(
            packed.draws[0],
            [1, 0x5566_7788, 0x1122_3344, 0xddee_ff00, 0x99aa_bbcc, 0]
        );
        assert_eq!(&packed.history[0][..4], &[7, 8, -1, -1]);
        assert_eq!(packed.shaping[0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn invalid_shapes_and_ranges_are_rejected() {
        assert!(matches!(
            PackedRowTables::pack(&[], 32, LIMITS),
            Err(PackError::Class(ClassError::EmptyRows))
        ));
        assert!(matches!(
            PackedRowTables::pack(
                &[Slot {
                    rows: vec![],
                    bank: 0,
                    previous_tape: 0,
                    following_bank: 1,
                    stop: 1,
                }],
                32,
                LIMITS
            ),
            Err(PackError::EmptySlot { .. })
        ));

        let mut bad = row(1, Demand::SELECT);
        assert!(matches!(
            PackedRowTables::pack(
                &[Slot {
                    rows: vec![bad.clone()],
                    bank: 0,
                    previous_tape: 0,
                    following_bank: 1,
                    stop: 1,
                }],
                32,
                LIMITS
            ),
            Err(PackError::SelectMismatch { .. })
        ));
        bad.select = Some(selection(Some(vec![1, 2])));
        assert!(matches!(
            PackedRowTables::pack(
                &[Slot {
                    rows: vec![bad],
                    bank: 0,
                    previous_tape: 0,
                    following_bank: 1,
                    stop: 1,
                }],
                32,
                LIMITS
            ),
            Err(PackError::InvalidMaskWidth { .. })
        ));

        for visible in [vec![[5, 8], [7, 9]], vec![[20, 30], [5, 8], [7, 9]]] {
            let mut bad_range = row(1, Demand::NONE);
            bad_range.histories[0].visible = visible;
            assert!(matches!(
                PackedRowTables::pack(
                    &[Slot {
                        rows: vec![bad_range],
                        bank: 0,
                        previous_tape: 0,
                        following_bank: 1,
                        stop: 1,
                    }],
                    32,
                    LIMITS
                ),
                Err(PackError::OverlappingVisibleRanges { .. })
            ));
        }
    }

    #[test]
    fn visible_ranges_keep_logical_order_and_adjacent_slab_boundaries() {
        let mut moved = row(1, Demand::NONE);
        moved.histories[0].visible = vec![[40, 44], [44, 46], [8, 12], [60, 61]];
        let packed = PackedRowTables::pack(
            &[Slot {
                rows: vec![moved],
                bank: 0,
                previous_tape: 0,
                following_bank: 1,
                stop: 1,
            }],
            32,
            LIMITS,
        )
        .unwrap();
        assert_eq!(packed.class.segments(), 4);
        assert_eq!(
            packed.histories[0].visible[0],
            vec![[40, 44], [44, 46], [8, 12], [60, 61]]
        );
    }

    #[test]
    fn successor_banks_are_validated_per_slot() {
        let pack = |slots: &[(i32, i32)]| {
            PackedRowTables::pack(
                &slots
                    .iter()
                    .map(|&(bank, following_bank)| Slot {
                        rows: vec![row(1, Demand::NONE)],
                        bank,
                        previous_tape: 0,
                        following_bank,
                        stop: 1,
                    })
                    .collect::<Vec<_>>(),
                32,
                LIMITS,
            )
        };
        assert!(pack(&[(0, 3), (0, 4)]).is_ok());
        for (slots, slot) in [
            (&[(0, 0)][..], 0),
            (&[(2, 2)][..], 0),
            (&[(0, -1)][..], 0),
            (&[(0, 3), (5, 3)][..], 1),
        ] {
            assert!(matches!(
                pack(slots),
                Err(PackError::InvalidFollowingBank { slot: actual, .. }) if actual == slot
            ));
        }
    }

    #[test]
    fn recurrent_versions_are_validated_per_slot() {
        let pack = |previous_tape: i32, stop: i32| {
            PackedRowTables::pack(
                &[Slot {
                    rows: vec![row(1, Demand::NONE), row(2, Demand::NONE)],
                    bank: 0,
                    previous_tape,
                    following_bank: 1,
                    stop,
                }],
                32,
                LIMITS,
            )
        };
        assert!(pack(3, 1).is_ok());
        assert!(pack(0, 2).is_ok());
        for (previous_tape, stop) in [(-1, 1), (0, 0), (0, 3)] {
            assert!(matches!(
                pack(previous_tape, stop),
                Err(PackError::InvalidRecurrentVersion { slot: 0, .. })
            ));
        }
    }
}
