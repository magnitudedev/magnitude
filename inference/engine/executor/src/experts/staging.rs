//! Staging records: the landing pool a file read lands in, as the model
//! file stores it, before it is converted into its tier record
//! (`landing`), by the host, or where the device is priced sooner, by a
//! decode barrier's convert, which reads it in place. Every record is free,
//! read into, or taken by a request until the device is past it. A free
//! record holds nothing: where an expert is has one answer, the directory's
//! (`Directory::located`).
//!
//! The pool has its own lock, beside residency's: the request service's
//! look takes a request's records before residency's lock
//! (`watcher::decode`), and reads ahead take theirs under it; a record is
//! freed where its landing settles.

use std::sync::{Mutex, MutexGuard};

/// A staged expert: a streamed layer (an index of the bank's layers) and
/// one of its experts.
pub(super) type Staged = (usize, usize);

/// The staging records, each free or taken, under their own lock.
pub(super) struct StagingPool {
    /// The free records.
    free: Mutex<Vec<u32>>,
    /// Its records.
    records: usize,
}

impl StagingPool {
    /// `records` records, every one free.
    pub fn new(records: u32) -> Self {
        Self { free: Mutex::new((0..records).collect()), records: records as usize }
    }

    fn held(&self) -> MutexGuard<'_, Vec<u32>> {
        self.free.lock().expect("the staging pool is never left by a panic")
    }

    /// Its records.
    pub fn records(&self) -> usize {
        self.records
    }

    /// Its records free now.
    pub fn free(&self) -> usize {
        self.held().len()
    }

    /// Take a free record, of more than `keep` free: none where `keep` or
    /// fewer are. Prefers the highest free index so cross-layer hints consume
    /// the upper hint band (`demand..staging`) and leave `0..demand` available
    /// for draft-only pools sized to `demand_records()`.
    #[cfg(test)]
    pub fn take_beyond(&self, keep: usize) -> Option<u32> {
        self.take_beyond_within(keep, u32::MAX)
    }

    /// Take a free record `< limit`, of more than `keep` free: none where
    /// `keep` or fewer are free or no free record is `< limit`.
    pub fn take_beyond_within(&self, keep: usize, limit: u32) -> Option<u32> {
        let mut free = self.held();
        if free.len() <= keep {
            return None;
        }
        let pos = free
            .iter()
            .enumerate()
            .filter(|&(_, &record)| record < limit)
            .max_by_key(|&(_, &record)| record)
            .map(|(index, _)| index)?;
        Some(free.swap_remove(pos))
    }

    /// Take up to `count` free records, as many as are free.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn take_up_to(&self, count: usize) -> Vec<u32> {
        self.take_up_to_within(count, u32::MAX)
    }

    /// Take up to `count` free records `< limit`, preferring the highest
    /// eligible indices so target demand misses preserve `0..demand` for
    /// draft-only pools when upper hint slots are free.
    pub fn take_up_to_within(&self, count: usize, limit: u32) -> Vec<u32> {
        let mut free = self.held();
        let mut taken = Vec::with_capacity(count.min(free.len()));
        for _ in 0..count {
            let Some(pos) = free
                .iter()
                .enumerate()
                .filter(|&(_, &record)| record < limit)
                .max_by_key(|&(_, &record)| record)
                .map(|(index, _)| index)
            else {
                break;
            };
            taken.push(free.swap_remove(pos));
        }
        taken
    }

    /// Free `record`.
    pub fn release(&self, record: u32) {
        let mut free = self.held();
        assert!((record as usize) < self.records && !free.contains(&record), "staging record {record} freed twice or out of {}", self.records);
        free.push(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records are taken while free, beyond what a taker leaves, and taken
    /// again once released.
    #[test]
    fn records_are_taken_while_free() {
        let staging = StagingPool::new(3);
        let taken = staging.take_up_to(2);
        assert_eq!((taken.len(), staging.free()), (2, 1));
        assert_eq!(staging.take_beyond(1), None, "one left to keep");
        let last = staging.take_beyond(0).unwrap();
        assert_eq!(staging.take_up_to(2), Vec::<u32>::new(), "none free");
        staging.release(last);
        taken.into_iter().for_each(|record| staging.release(record));
        assert_eq!((staging.free(), staging.records()), (3, 3));
    }

    /// A draft layer whose staging pool is sized to `demand_records` (e.g. 24)
    /// while target pools hold `demand + hints` (e.g. 121) must only ever
    /// receive indices `< 24` from `take_beyond_within`, even when background
    /// cover/read-ahead runs with `vram_slots > 24`.
    #[test]
    fn take_beyond_within_never_returns_out_of_range_draft_staging_record() {
        let staging = StagingPool::new(121);
        let draft_limit = 24u32;
        let mut draft_taken = Vec::new();
        while let Some(record) = staging.take_beyond_within(0, draft_limit) {
            assert!(
                record < draft_limit,
                "draft staging record {record} must be < {draft_limit}"
            );
            draft_taken.push(record);
        }
        assert_eq!(draft_taken.len(), draft_limit as usize);
        // Upper target-hint records remain free for target layers.
        let target_record = staging.take_beyond_within(0, 121).expect("target upper band still free");
        assert!(target_record >= draft_limit && target_record < 121);
        staging.release(target_record);
        for record in draft_taken {
            staging.release(record);
        }
        assert_eq!(staging.free(), 121);
    }
}
