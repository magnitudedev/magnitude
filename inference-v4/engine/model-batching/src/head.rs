//! Device-independent head row semantics.

use crate::{Demand, LaunchClass, PackError, Slot, TargetBatchUpload, ValidatedTargetBatch};

/// Head rows can produce only a final feature readout. Projection owns logits
/// and selection, so those demands cannot enter a head program.
#[derive(Clone, Debug)]
pub struct ValidatedHeadBatch {
    rows: ValidatedTargetBatch,
}

impl ValidatedHeadBatch {
    pub fn from_slots(
        slots: &[Slot],
        vocabulary_size: usize,
        row_limit: usize,
    ) -> Result<Self, PackError> {
        let mut global_row = 0;
        for slot in slots {
            for (index, row) in slot.rows.iter().enumerate() {
                let allowed = if index + 1 == slot.rows.len() {
                    Demand::FEATURES
                } else {
                    Demand::NONE
                };
                if row.demand.bits() & !allowed.bits() != 0 || row.select.is_some() {
                    return Err(PackError::InvalidHeadDemand {
                        row: global_row,
                        demand: row.demand,
                    });
                }
                global_row += 1;
            }
        }
        Ok(Self {
            rows: ValidatedTargetBatch::from_slots(slots, vocabulary_size, row_limit)?,
        })
    }

    pub fn class(&self) -> LaunchClass {
        self.rows.class()
    }

    pub fn actual_rows(&self) -> usize {
        self.rows.actual_rows()
    }

    pub fn actual_slots(&self) -> usize {
        self.rows.actual_slots()
    }

    pub fn upload(&self) -> TargetBatchUpload<'_> {
        self.rows.upload()
    }

    pub fn slots(&self) -> impl Iterator<Item = crate::TargetBatchSlot<'_>> {
        self.rows.slots()
    }
}
