//! Staging records that keep their bytes: a
//! free record keeps the expert it last held whole until it is taken for
//! another, an empty one first, then the one keeping the least reused
//! expert (`Reuse`, the directory's one ranking), so a later demand for
//! that expert takes the record back without a read. Every record is free,
//! read into, or taken by a request until the device is past it; only a
//! free record is kept. Staging holds experts as the model file stores
//! them, the landing pool each is converted into its tier record from
//! (`landing`), by the host, or where the device is priced sooner, by a
//! decode barrier's convert, which reads it in place.

use super::reuse::Score;
use std::collections::HashMap;

/// A staged expert: a streamed layer (an index of the bank's layers) and
/// one of its experts.
pub(super) type Staged = (usize, usize);

pub(super) struct Staging {
    /// The free records.
    free: Vec<u32>,
    /// What each record last held whole.
    holds: Vec<Option<Staged>>,
    /// The free record keeping each staged expert.
    kept: HashMap<Staged, u32>,
}

impl Staging {
    /// `records` records, every one free and empty.
    pub fn new(records: u32) -> Self {
        Self {
            free: (0..records).collect(),
            holds: vec![None; records as usize],
            kept: HashMap::new(),
        }
    }

    /// Its records.
    pub fn records(&self) -> usize {
        self.holds.len()
    }

    pub fn free(&self) -> usize {
        self.free.len()
    }

    /// Whether a free record keeps `staged`.
    pub fn keeps(&self, staged: Staged) -> bool {
        self.kept.contains_key(&staged)
    }

    /// Every expert a free record keeps.
    #[cfg(test)]
    pub fn kept(&self) -> std::collections::BTreeSet<Staged> {
        self.kept.keys().copied().collect()
    }

    /// Take a free record, forgetting what it kept: an empty one first,
    /// then one keeping an expert `worth` finds worthless (none: the store
    /// serves it), then the one keeping the expert `worth` ranks least, a
    /// tie by the record.
    pub fn take(&mut self, worth: impl Fn(Staged) -> Option<Score>) -> Option<u32> {
        let at = (0..self.free.len()).min_by_key(|&at| {
            let record = self.free[at];
            (self.holds[record as usize].map(&worth), record)
        })?;
        let record = self.free.swap_remove(at);
        if let Some(staged) = self.holds[record as usize].take() {
            self.kept.remove(&staged);
        }
        Some(record)
    }

    /// Take `count` records as [`Self::take`] does; none when fewer are
    /// free.
    pub fn take_many(&mut self, count: usize, worth: impl Fn(Staged) -> Option<Score>) -> Option<Vec<u32>> {
        (self.free.len() >= count).then(|| (0..count).map(|_| self.take(&worth).expect("a free record remains for each taken")).collect())
    }

    /// Take the free record keeping `staged` back, its bytes whole: no read.
    pub fn revive(&mut self, staged: Staged) -> Option<u32> {
        let record = self.kept.remove(&staged)?;
        let at = self
            .free
            .iter()
            .position(|free| *free == record)
            .expect("a kept record is free");
        self.free.swap_remove(at);
        Some(record)
    }

    /// Free `record`, keeping `held` when its bytes are that expert's whole
    /// (a read that landed). A copy of the same expert kept elsewhere is
    /// forgotten: one record keeps an expert.
    pub fn release(&mut self, record: u32, held: Option<Staged>) {
        let previous = held.and_then(|staged| self.kept.insert(staged, record));
        if let Some(previous) = previous.filter(|previous| *previous != record) {
            self.holds[previous as usize] = None;
        }
        self.holds[record as usize] = held;
        self.free.push(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A free record keeps its expert until it is taken, empty records
    /// first, then one keeping an expert the store serves, then the one
    /// keeping the least reused; revived, it is taken back without a read.
    #[test]
    fn free_records_keep_their_experts_the_least_reused_taken_first() {
        let reuse = |(_, expert): Staged| Some(Score::at(expert as u64));
        let mut staging = Staging::new(3);
        let [first, second, third] = [0, 1, 2].map(|_| staging.take(reuse).unwrap());
        assert_eq!(staging.take(reuse), None);
        staging.release(first, Some((0, 7)));
        staging.release(second, Some((0, 3)));
        staging.release(third, Some((1, 9)));
        assert!(staging.keeps((0, 7)) && staging.keeps((1, 9)));
        assert_eq!(staging.revive((1, 9)), Some(third));
        assert_eq!(staging.revive((1, 9)), None);
        staging.release(third, None);
        // The empty record first, then the least reused, (0, 3).
        assert_eq!(staging.take(reuse), Some(third));
        assert_eq!(staging.take(reuse), Some(second));
        assert!(!staging.keeps((0, 3)) && staging.keeps((0, 7)));
        assert_eq!(staging.take_many(2, reuse), None);
        staging.release(third, Some((2, 1)));
        staging.release(second, Some((2, 1)));
        assert_eq!(staging.revive((2, 1)), Some(second), "the newer copy is kept");
        assert_eq!(staging.take_many(2, reuse), Some(vec![third, first]));
        staging.release(third, Some((0, 2)));
        staging.release(first, Some((0, 8)));
        let stored = |staged: Staged| (staged != (0, 8)).then(|| Score::at(staged.1 as u64));
        assert_eq!(staging.take(stored), Some(first), "the store serves (0, 8): its copy is worth nothing");
    }
}
