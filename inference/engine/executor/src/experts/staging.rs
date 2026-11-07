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
    /// fewer are.
    pub fn take_beyond(&self, keep: usize) -> Option<u32> {
        let mut free = self.held();
        if free.len() <= keep {
            return None;
        }
        free.pop()
    }

    /// Take up to `count` free records, as many as are free.
    pub fn take_up_to(&self, count: usize) -> Vec<u32> {
        let mut free = self.held();
        let left = free.len().saturating_sub(count);
        free.split_off(left)
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
}
