//! Opaque, device-independent target row semantics.

use crate::{HistoryTables, LaunchClass, PackError, PackedRowTables, Slot};
use std::sync::Arc;

/// The packed controls and their row mapping are validated together once.
/// Device leases and state transactions are joined by the executor later.
#[derive(Clone, Debug)]
pub struct ValidatedTargetBatch {
    packed: PackedRowTables,
}

impl ValidatedTargetBatch {
    pub fn from_slots(
        slots: &[Slot],
        vocabulary_size: usize,
        row_limit: usize,
    ) -> Result<Self, PackError> {
        Ok(Self {
            packed: PackedRowTables::pack(slots, vocabulary_size, row_limit)?,
        })
    }

    /// Validate into a launch class covering at least `segments` history
    /// ranges per row (`PackedRowTables::pack_covering`).
    pub fn covering(
        slots: &[Slot],
        vocabulary_size: usize,
        row_limit: usize,
        segments: usize,
    ) -> Result<Self, PackError> {
        Ok(Self {
            packed: PackedRowTables::pack_covering(slots, vocabulary_size, row_limit, segments)?,
        })
    }

    pub fn class(&self) -> LaunchClass {
        self.packed.class
    }

    pub fn actual_rows(&self) -> usize {
        self.packed.actual_rows
    }

    pub fn actual_slots(&self) -> usize {
        self.packed.actual_slots
    }

    pub fn demand_bits(&self) -> &[u32] {
        &self.packed.demand[..self.packed.actual_rows]
    }

    /// The validated physical row tables. These fields are borrowed from the
    /// same packed batch so programs cannot recombine unrelated row plans.
    pub fn upload(&self) -> TargetBatchUpload<'_> {
        let packed = &self.packed;
        TargetBatchUpload {
            class: packed.class,
            actual_rows: packed.actual_rows,
            actual_slots: packed.actual_slots,
            mask_words: packed.mask_words,
            mask_count: packed.mask_count,
            tokens: &packed.tokens,
            coordinates: &packed.coordinates,
            histories: &packed.histories,
            row_slots: &packed.row_slots,
            demand: &packed.demand,
            segments: &packed.segments,
            bank: &packed.bank,
            previous_tape: &packed.previous_tape,
            following_bank: &packed.following_bank,
            stop: &packed.stop,
            plane_base: &packed.plane_base,
            out_rows: &packed.out_rows,
            select_rows: &packed.select_rows,
            draws: &packed.draws,
            mask_rows: &packed.mask_rows,
            masks: &packed.masks,
            shaping: &packed.shaping,
            history: &packed.history,
        }
    }

    pub fn slot(&self, index: usize) -> Option<TargetBatchSlot<'_>> {
        if index >= self.packed.actual_slots {
            return None;
        }
        let [start, end] = self.packed.segments[index];
        let start = usize::try_from(start).ok()?;
        let end = usize::try_from(end).ok()?;
        Some(TargetBatchSlot {
            bank: self.packed.bank[index],
            following_bank: self.packed.following_bank[index],
            start,
            end,
            histories: &self.packed.histories,
        })
    }

    pub fn slots(&self) -> impl Iterator<Item = TargetBatchSlot<'_>> {
        (0..self.actual_slots()).map(|index| {
            self.slot(index)
                .expect("validated batch has one segment per actual slot")
        })
    }
}

pub struct TargetBatchUpload<'a> {
    pub class: LaunchClass,
    pub actual_rows: usize,
    pub actual_slots: usize,
    pub mask_words: usize,
    pub mask_count: usize,
    pub tokens: &'a [i32],
    pub coordinates: &'a [[i32; 4]],
    /// One table set per history domain of the store, in domain order.
    pub histories: &'a [HistoryTables],
    pub row_slots: &'a [i32],
    pub demand: &'a [u32],
    pub segments: &'a [[i32; 2]],
    pub bank: &'a [i32],
    pub previous_tape: &'a [i32],
    pub following_bank: &'a [i32],
    pub stop: &'a [i32],
    pub plane_base: &'a [i32],
    pub out_rows: &'a [i32],
    pub select_rows: &'a [i32],
    pub draws: &'a [[u32; 6]],
    pub mask_rows: &'a [i32],
    pub masks: &'a [Arc<[u32]>],
    pub shaping: &'a [[f32; crate::SHAPING_WIDTH]],
    pub history: &'a [[i32; crate::HISTORY_WIDTH]],
}

pub struct TargetBatchSlot<'a> {
    bank: i32,
    following_bank: i32,
    start: usize,
    end: usize,
    histories: &'a [HistoryTables],
}

/// One slot's rows in one history domain.
pub struct SlotHistory<'a> {
    pub destinations: &'a [i32],
    pub visible: &'a [Vec<[i32; 2]>],
    pub fresh: &'a [[i32; 2]],
}

impl<'a> TargetBatchSlot<'a> {
    pub fn bank(&self) -> i32 {
        self.bank
    }

    pub fn following_bank(&self) -> i32 {
        self.following_bank
    }

    pub fn rows(&self) -> usize {
        self.end - self.start
    }

    /// History domains of the slot's store.
    pub fn history_domains(&self) -> usize {
        self.histories.len()
    }

    /// The slot's rows in history domain `domain`.
    pub fn history(&self, domain: usize) -> Option<SlotHistory<'a>> {
        let tables = self.histories.get(domain)?;
        Some(SlotHistory {
            destinations: &tables.destinations[self.start..self.end],
            visible: &tables.visible[self.start..self.end],
            fresh: &tables.fresh[self.start..self.end],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Demand, Row, RowHistory};

    #[test]
    fn batch_exposes_the_validated_slot_and_upload_together() {
        let batch = ValidatedTargetBatch::from_slots(
            &[Slot {
                bank: 2,
                previous_tape: 0,
                following_bank: 3,
                stop: 1,
                rows: vec![Row {
                    token: 7,
                    coordinates: [1, 1, 1, 0],
                    histories: vec![RowHistory {
                        visible: vec![],
                        fresh_start: 0,
                        bidirectional_end: None,
                        destination: 3,
                    }],
                    demand: Demand::NONE,
                    select: None,
                }],
            }],
            16,
            8,
        )
        .unwrap();
        assert_eq!(batch.actual_rows(), 1);
        assert_eq!(batch.actual_slots(), 1);
        let slot = batch.slot(0).unwrap();
        assert_eq!(slot.bank(), 2);
        assert_eq!(slot.following_bank(), 3);
        assert_eq!(slot.rows(), 1);
        assert_eq!(slot.history(0).unwrap().destinations, &[3]);
        assert!(slot.history(1).is_none());
        assert!(batch.slot(1).is_none());
    }
}
