//! Device-independent draft-head row semantics.
//!
//! A head batch is one entry pass followed by `steps - 1` chained passes.
//! The entry pass carries every slot's committed rows; its last row per slot
//! produces the slot's first output feature (and, when drafting, its first
//! selection). Each chained pass carries exactly one speculative row per
//! slot, whose token and conditioning are the previous pass's selection and
//! output feature on the device.

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

/// How a head batch drafts past its entry pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPasses {
    /// One chained pass per further proposal (MTP).
    Chained,
    /// One block pass drafting every proposal (DFlash, DSpark).
    Block,
}

#[derive(Clone, Debug)]
pub struct ValidatedHeadBatch {
    entry: ValidatedTargetBatch,
    /// The chained passes, or the one block pass.
    chain: Vec<ValidatedTargetBatch>,
    passes: HeadPasses,
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
                        }
                    })
                    .collect::<Vec<_>>();
                ValidatedTargetBatch::from_slots(&rows, vocabulary_size, limits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            entry,
            chain,
            passes: HeadPasses::Chained,
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
        let chain = if width == 0 {
            Vec::new()
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
                })
                .collect::<Vec<_>>();
            vec![ValidatedTargetBatch::from_slots(
                &rows,
                vocabulary_size,
                limits,
            )?]
        };
        Ok(Self {
            entry,
            chain,
            passes: HeadPasses::Block,
        })
    }

    /// How the batch drafts past its entry pass.
    pub fn passes(&self) -> HeadPasses {
        self.passes
    }

    /// The entry pass's class.
    pub fn class(&self) -> LaunchClass {
        self.entry.class()
    }

    /// Selections per slot: 0 for a causal-only head or an injection-only
    /// draft batch.
    pub fn steps(&self) -> usize {
        match self.passes {
            HeadPasses::Chained if self.entry.upload().select_rows.is_empty() => 0,
            HeadPasses::Chained => self.chain.len() + 1,
            HeadPasses::Block => self.chain.first().map_or(0, |block| {
                block.upload().select_rows.len() / block.actual_slots()
            }),
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
        self.chain
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
        self.chain.iter().map(ValidatedTargetBatch::upload)
    }

    pub fn slots(&self) -> impl Iterator<Item = crate::TargetBatchSlot<'_>> {
        self.entry.slots()
    }

    /// Destinations of each slot's chained rows in history domain `domain`,
    /// in step order. Block rows append nowhere.
    pub fn chain_destinations(&self, slot: usize, domain: usize) -> Vec<i32> {
        if self.passes == HeadPasses::Block {
            return Vec::new();
        }
        self.chain
            .iter()
            .map(|pass| pass.upload().histories[domain].destinations[slot])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Draw, DrawKind, RowHistory, Shaping};

    const LIMITS: ClassLimits = ClassLimits {
        rows: 64,
        segments: 63,
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
}
