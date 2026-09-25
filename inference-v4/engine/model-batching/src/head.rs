//! Device-independent draft-head row semantics.
//!
//! A head batch is one entry pass followed by `steps - 1` chained passes.
//! The entry pass carries every slot's committed rows; its last row per slot
//! produces the slot's first output feature (and, when drafting, its first
//! selection). Each chained pass carries exactly one speculative row per
//! slot, whose token and conditioning are the previous pass's selection and
//! output feature on the device.

use crate::{
    Demand, LaunchClass, PackError, Row, Select, Slot, TargetBatchUpload, ValidatedTargetBatch,
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

#[derive(Clone, Debug)]
pub struct ValidatedHeadBatch {
    entry: ValidatedTargetBatch,
    chain: Vec<ValidatedTargetBatch>,
}

impl ValidatedHeadBatch {
    /// Every slot must carry the same number of steps (the domain pads a
    /// slot's surplus steps); a causal-only batch has no proposals at all.
    pub fn from_slots(
        slots: &[HeadSlot],
        vocabulary_size: usize,
        row_limit: usize,
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
        let entry = ValidatedTargetBatch::from_slots(&entry, vocabulary_size, row_limit)?;
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
                ValidatedTargetBatch::from_slots(&rows, vocabulary_size, row_limit)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { entry, chain })
    }

    /// The entry pass's class.
    pub fn class(&self) -> LaunchClass {
        self.entry.class()
    }

    /// Selections per slot: 0 for a causal-only head.
    pub fn steps(&self) -> usize {
        if self.entry.upload().select_rows.is_empty() {
            0
        } else {
            self.chain.len() + 1
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

    /// The chained passes' tables, in step order (step 1 first).
    pub fn chain(&self) -> impl Iterator<Item = TargetBatchUpload<'_>> {
        self.chain.iter().map(ValidatedTargetBatch::upload)
    }

    pub fn slots(&self) -> impl Iterator<Item = crate::TargetBatchSlot<'_>> {
        self.entry.slots()
    }

    /// Destinations of each slot's chained rows, in step order.
    pub fn chain_destinations(&self, slot: usize) -> Vec<i32> {
        self.chain
            .iter()
            .map(|pass| pass.upload().destinations[slot])
            .collect()
    }
}
