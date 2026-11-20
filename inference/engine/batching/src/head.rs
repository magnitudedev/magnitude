//! Device-independent draft-head row semantics.
//!
//! A head batch is one entry pass followed by `steps - 1` chained passes.
//! The entry pass carries every slot's committed rows; its last row per slot
//! produces the slot's first output feature (and, when drafting, its first
//! selection). Each chained pass carries exactly one speculative row per
//! slot, whose token and conditioning are the previous pass's selection and
//! output feature on the device. A front batch is one entry pass alone,
//! every row of which selects.

use crate::{
    ClassLimits, Demand, LaunchClass, PackError, Row, Select, Slot, TargetBatchUpload,
    ValidatedTargetBatch,
};

/// One request's head rows.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadSlot {
    /// The committed entry rows. No row selects or demands outputs; the
    /// batch derives the last row's demand from the proposal count.
    pub entry: Slot,
    /// One speculative row per chained step. Tokens are ignored (the device
    /// feeds them); rows past the request's own proposals append nowhere.
    pub chain: Vec<Row>,
    /// One selection per step: the entry pass's last row, then each chained
    /// row. Empty for a causal-only head.
    pub proposals: Vec<Select>,
}

/// One request's rows for a separate draft (DFlash, DSpark): the entry rows
/// inject the target's conditioning into the draft's history, then one
/// block of rows drafts every proposal in a single non-causal pass.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockSlot {
    /// The committed entry rows. No row selects or demands outputs.
    pub entry: Slot,
    /// The block rows (every slot the same count). A row that proposes a
    /// token carries its selection, in proposal order; the rest carry none.
    /// Block rows append nowhere.
    pub block: Vec<Row>,
}

/// One request's rows of a front launch, the MTP head's launch behind a
/// selecting target step of two-row decode: entry row `r` pairs the step's
/// selection `r` rows past the slot's first with its conditioning, at the
/// head position one past that selection's row.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontSlot {
    /// The entry rows: the committed one (the exact selection's), then the
    /// scratch rows, its `front` rows, which run on the step's front rows'
    /// selections and are never accepted. No row selects or demands
    /// outputs; the batch derives every row's demand. Tokens are ignored
    /// (the device feeds them).
    pub entry: Slot,
    /// One selection per entry row, in row order.
    pub selects: Vec<Select>,
}

/// A token row of the step queued behind a front launch that continues one
/// of its slots: the slot, and its depth there, 0 for the exact row and `j`
/// for front row `j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontContinuation {
    pub slot: usize,
    pub depth: usize,
}

/// How a head batch drafts past its entry pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPasses {
    /// One chained pass per further proposal (MTP).
    Chained,
    /// One block pass drafting every proposal (DFlash, DSpark).
    Block,
    /// None: every entry row selects (a front launch, [`FrontSlot`]).
    Front,
}

/// What follows a head batch's entry pass, per [`HeadPasses`].
#[derive(Clone, Debug)]
enum Passes {
    /// The chained passes, in step order.
    Chained(Vec<ValidatedTargetBatch>),
    /// The one block pass, of a batch that drafts.
    Block(Option<Box<ValidatedTargetBatch>>),
    /// No pass: per token row of the step queued behind the front launch,
    /// the slot it continues ([`FrontContinuation`]), none where it
    /// continues no slot.
    Front(Vec<Option<FrontContinuation>>),
}

#[derive(Clone, Debug)]
pub struct ValidatedHeadBatch {
    entry: ValidatedTargetBatch,
    passes: Passes,
}

impl ValidatedHeadBatch {
    /// Every slot must carry the same number of steps (the domain pads a
    /// slot's surplus steps); a causal-only batch has no proposals at all.
    pub fn from_slots(
        slots: &[HeadSlot],
        vocabulary_size: usize,
        limits: ClassLimits,
    ) -> Result<Self, PackError> {
        let steps = slots.first().map_or(0, |slot| slot.proposals.len());
        for (index, slot) in slots.iter().enumerate() {
            if slot.proposals.len() != steps || slot.chain.len() != steps.saturating_sub(1) {
                return Err(PackError::HeadSteps { slot: index });
            }
            for (row, entry) in slot.entry.rows.iter().chain(&slot.chain).enumerate() {
                if entry.demand != Demand::NONE || entry.select.is_some() {
                    return Err(PackError::InvalidHeadDemand {
                        row,
                        demand: entry.demand,
                    });
                }
            }
        }
        let selecting = Demand::FEATURES | Demand::SELECT;
        let entry = slots
            .iter()
            .map(|slot| {
                let mut entry = slot.entry.clone();
                let last = entry
                    .rows
                    .last_mut()
                    .ok_or(PackError::EmptySlot { slot: 0 })?;
                match slot.proposals.first() {
                    Some(select) => {
                        last.demand = selecting;
                        last.select = Some(select.clone());
                    }
                    None => last.demand = Demand::FEATURES,
                }
                Ok(entry)
            })
            .collect::<Result<Vec<_>, PackError>>()?;
        let entry = ValidatedTargetBatch::from_slots(&entry, vocabulary_size, limits)?;
        let chain = (1..steps)
            .map(|step| {
                let rows = slots
                    .iter()
                    .map(|slot| {
                        let mut row = slot.chain[step - 1].clone();
                        row.token = 0;
                        row.demand = selecting;
                        row.select = Some(slot.proposals[step].clone());
                        Slot {
                            rows: vec![row],
                            bank: slot.entry.bank,
                            previous_tape: slot.entry.previous_tape,
                            following_bank: slot.entry.following_bank,
                            stop: 1,
                            front: 0,
                        }
                    })
                    .collect::<Vec<_>>();
                ValidatedTargetBatch::from_slots(&rows, vocabulary_size, limits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            entry,
            passes: Passes::Chained(chain),
        })
    }

    /// A separate draft's batch: the entry pass, then the block pass when
    /// the slots draft (every slot the same block rows and proposals). A
    /// batch of empty blocks only injects its entry rows.
    pub fn from_block_slots(
        slots: &[BlockSlot],
        vocabulary_size: usize,
        limits: ClassLimits,
    ) -> Result<Self, PackError> {
        let width = slots.first().map_or(0, |slot| slot.block.len());
        let proposals = |slot: &BlockSlot| slot.block.iter().filter(|row| row.select.is_some()).count();
        let steps = slots.first().map_or(0, proposals);
        for (index, slot) in slots.iter().enumerate() {
            if slot.block.len() != width || proposals(slot) != steps || (width > 0) != (steps > 0) {
                return Err(PackError::HeadSteps { slot: index });
            }
            for (row, entry) in slot.entry.rows.iter().enumerate() {
                if entry.demand != Demand::NONE || entry.select.is_some() {
                    return Err(PackError::InvalidHeadDemand {
                        row,
                        demand: entry.demand,
                    });
                }
            }
            for (row, block) in slot.block.iter().enumerate() {
                if block.demand != Demand::NONE {
                    return Err(PackError::InvalidHeadDemand {
                        row,
                        demand: block.demand,
                    });
                }
                if let Some(history) = block.histories.iter().find(|history| history.destination != -1) {
                    return Err(PackError::InvalidDestination {
                        row,
                        destination: history.destination,
                    });
                }
            }
        }
        let entry = slots.iter().map(|slot| slot.entry.clone()).collect::<Vec<_>>();
        let entry = ValidatedTargetBatch::from_slots(&entry, vocabulary_size, limits)?;
        let block = if width == 0 {
            None
        } else {
            let stop =
                i32::try_from(width).map_err(|_| PackError::IntegerOverflow("block rows"))?;
            let rows = slots
                .iter()
                .map(|slot| Slot {
                    rows: slot
                        .block
                        .iter()
                        .map(|row| {
                            let mut row = row.clone();
                            if row.select.is_some() {
                                row.demand = Demand::FEATURES | Demand::SELECT;
                            }
                            row
                        })
                        .collect(),
                    bank: slot.entry.bank,
                    previous_tape: slot.entry.previous_tape,
                    following_bank: slot.entry.following_bank,
                    stop,
                    front: 0,
                })
                .collect::<Vec<_>>();
            Some(Box::new(ValidatedTargetBatch::from_slots(
                &rows,
                vocabulary_size,
                limits,
            )?))
        };
        Ok(Self {
            entry,
            passes: Passes::Block(block),
        })
    }

    /// A front launch's batch: each slot's entry rows, every one demanding
    /// its features and selecting by its own selection. `continuing` names,
    /// per token row of the step queued behind (none where it continues no
    /// slot), the slot that row continues and its depth.
    pub fn from_front_slots(
        slots: &[FrontSlot],
        continuing: &[Option<FrontContinuation>],
        vocabulary_size: usize,
        limits: ClassLimits,
    ) -> Result<Self, PackError> {
        for (index, slot) in slots.iter().enumerate() {
            if slot.selects.len() != slot.entry.rows.len()
                || slot.entry.front.checked_add(1) != Some(slot.entry.rows.len())
            {
                return Err(PackError::FrontSlot { slot: index });
            }
            for (row, entry) in slot.entry.rows.iter().enumerate() {
                if entry.demand != Demand::NONE || entry.select.is_some() {
                    return Err(PackError::InvalidHeadDemand {
                        row,
                        demand: entry.demand,
                    });
                }
            }
        }
        if let Some(row) = continuing.iter().position(|row| {
            row.is_some_and(|row| {
                slots
                    .get(row.slot)
                    .is_none_or(|slot| row.depth >= slot.entry.rows.len())
            })
        }) {
            return Err(PackError::FrontContinuation { row });
        }
        let entry = slots
            .iter()
            .map(|slot| {
                let mut entry = slot.entry.clone();
                for (row, select) in entry.rows.iter_mut().zip(&slot.selects) {
                    row.demand = Demand::FEATURES | Demand::SELECT;
                    row.select = Some(select.clone());
                }
                entry
            })
            .collect::<Vec<_>>();
        Ok(Self {
            entry: ValidatedTargetBatch::from_slots(&entry, vocabulary_size, limits)?,
            passes: Passes::Front(continuing.to_vec()),
        })
    }

    /// How the batch drafts past its entry pass.
    pub fn passes(&self) -> HeadPasses {
        match self.passes {
            Passes::Chained(_) => HeadPasses::Chained,
            Passes::Block(_) => HeadPasses::Block,
            Passes::Front(_) => HeadPasses::Front,
        }
    }

    /// The passes behind the entry pass: the chained passes, or the one
    /// block pass.
    fn following(&self) -> &[ValidatedTargetBatch] {
        match &self.passes {
            Passes::Chained(chain) => chain,
            Passes::Block(block) => block.as_deref().map_or(&[], std::slice::from_ref),
            Passes::Front(_) => &[],
        }
    }

    /// The entry pass's class.
    pub fn class(&self) -> LaunchClass {
        self.entry.class()
    }

    /// Selections per slot: 0 for a causal-only head or an injection-only
    /// draft batch.
    pub fn steps(&self) -> usize {
        match &self.passes {
            Passes::Chained(_) if self.entry.upload().select_rows.is_empty() => 0,
            Passes::Chained(chain) => chain.len() + 1,
            Passes::Block(block) => block.as_ref().map_or(0, |block| {
                block.upload().select_rows.len() / block.actual_slots()
            }),
            Passes::Front(_) => 1,
        }
    }

    /// Of a front batch, per token row of the step queued behind, the slot
    /// it continues and its depth there; none of any other batch.
    pub fn continuing(&self) -> Option<&[Option<FrontContinuation>]> {
        match &self.passes {
            Passes::Front(continuing) => Some(continuing),
            Passes::Chained(_) | Passes::Block(_) => None,
        }
    }

    pub fn actual_rows(&self) -> usize {
        self.entry.actual_rows()
    }

    pub fn actual_slots(&self) -> usize {
        self.entry.actual_slots()
    }

    /// The most visible history spans of any entry or chained row.
    pub fn segments(&self) -> usize {
        self.following()
            .iter()
            .map(|pass| pass.class().segments())
            .fold(self.entry.class().segments(), usize::max)
    }

    pub fn upload(&self) -> TargetBatchUpload<'_> {
        self.entry.upload()
    }

    /// The chained passes' tables, in step order (step 1 first), or the one
    /// block pass.
    pub fn chain(&self) -> impl Iterator<Item = TargetBatchUpload<'_>> {
        self.following().iter().map(ValidatedTargetBatch::upload)
    }

    pub fn slots(&self) -> impl Iterator<Item = crate::TargetBatchSlot<'_>> {
        self.entry.slots()
    }

    /// Destinations of each slot's chained rows in history domain `domain`,
    /// in step order. Block rows append nowhere; a front batch chains none.
    pub fn chain_destinations(&self, slot: usize, domain: usize) -> Vec<i32> {
        match &self.passes {
            Passes::Chained(chain) => chain
                .iter()
                .map(|pass| pass.upload().histories[domain].destinations[slot])
                .collect(),
            Passes::Block(_) | Passes::Front(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Draw, DrawKind, RowHistory, Shaping};

    const LIMITS: ClassLimits = ClassLimits {
        rows: 64,
        segments: 63,
        slots: 64,
    };

    fn row(token: i32, destination: i32, select: Option<Select>) -> Row {
        Row {
            token,
            coordinates: [token, token, token, 0],
            histories: vec![RowHistory {
                visible: Vec::new(),
                fresh_start: 0,
                bidirectional_end: None,
                destination,
            }],
            demand: Demand::NONE,
            select,
        }
    }

    fn select(position: u64) -> Select {
        Select {
            draw: Draw {
                kind: DrawKind::Greedy,
                seed: 0,
                position,
                domain: 0,
            },
            mask: None,
            shaping: Shaping::default(),
            history: Vec::new(),
        }
    }

    fn slot(bank: i32, entry: usize, proposals: usize) -> BlockSlot {
        BlockSlot {
            entry: Slot {
                rows: (0..entry).map(|index| row(index as i32, index as i32, None)).collect(),
                bank,
                previous_tape: 0,
                following_bank: bank + 1,
                stop: entry as i32,
                front: 0,
            },
            // A mask-slot block: row 0 is the anchor, rows 1.. propose.
            block: (0..=proposals)
                .map(|index| row(7, -1, (index > 0).then(|| select(index as u64))))
                .collect(),
        }
    }

    #[test]
    fn a_block_batch_drafts_every_proposal_in_one_pass() {
        let batch =
            ValidatedHeadBatch::from_block_slots(&[slot(2, 3, 4), slot(4, 1, 4)], 16, LIMITS)
                .unwrap();
        assert_eq!(batch.passes(), HeadPasses::Block);
        assert_eq!((batch.actual_slots(), batch.actual_rows(), batch.steps()), (2, 4, 4));
        let block = batch.chain().collect::<Vec<_>>();
        let [block] = block.as_slice() else {
            panic!("one block pass");
        };
        assert_eq!(block.actual_rows, 10);
        // Selections follow the proposing rows, slot by slot.
        assert_eq!(block.select_rows.len(), 8);
        assert_eq!(&block.out_rows[..4], &[1, 2, 3, 4]);
        assert!(batch.chain_destinations(0, 0).is_empty());
    }

    #[test]
    fn an_empty_block_only_injects() {
        // A block without proposals has no rows at all.
        assert!(matches!(
            ValidatedHeadBatch::from_block_slots(&[slot(2, 3, 0)], 16, LIMITS),
            Err(PackError::HeadSteps { slot: 0 })
        ));
        let mut empty = slot(2, 3, 0);
        empty.block.clear();
        let batch = ValidatedHeadBatch::from_block_slots(&[empty], 16, LIMITS).unwrap();
        assert_eq!((batch.steps(), batch.chain().count()), (0, 0));
    }

    #[test]
    fn block_rows_that_append_or_unequal_blocks_are_refused() {
        let mut appending = slot(2, 3, 2);
        appending.block[1].histories[0].destination = 9;
        assert!(matches!(
            ValidatedHeadBatch::from_block_slots(&[appending], 16, LIMITS),
            Err(PackError::InvalidDestination { .. })
        ));
        assert!(matches!(
            ValidatedHeadBatch::from_block_slots(&[slot(2, 3, 2), slot(4, 1, 3)], 16, LIMITS),
            Err(PackError::HeadSteps { slot: 1 })
        ));
    }

    #[test]
    fn front_slots_reject_overflowing_front_count_and_out_of_range_continuation_depth() {
        let front_slot = |front: usize| FrontSlot {
            entry: Slot {
                rows: vec![row(1, 0, None), row(2, 1, None)],
                bank: 0,
                previous_tape: 0,
                following_bank: 1,
                stop: 1,
                front,
            },
            selects: vec![select(0), select(1)],
        };
        assert!(matches!(
            ValidatedHeadBatch::from_front_slots(&[front_slot(usize::MAX)], &[], 16, LIMITS),
            Err(PackError::FrontSlot { slot: 0 })
        ));
        assert!(matches!(
            ValidatedHeadBatch::from_front_slots(
                &[front_slot(1)],
                &[Some(FrontContinuation { slot: 0, depth: 1 })],
                16,
                LIMITS,
            ),
            Ok(_)
        ));
        assert!(matches!(
            ValidatedHeadBatch::from_front_slots(
                &[front_slot(1)],
                &[Some(FrontContinuation { slot: 0, depth: 2 })],
                16,
                LIMITS,
            ),
            Err(PackError::FrontContinuation { row: 0 })
        ));
    }
}
