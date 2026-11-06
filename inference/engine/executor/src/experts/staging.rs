//! Staging records: the landing pool a file read lands in, as the model
//! file stores it, before it is converted into its tier record
//! (`landing`), by the host, or where the device is priced sooner, by a
//! decode barrier's convert, which reads it in place. Every record is free,
//! read into, or taken by a request until the device is past it. A free
//! record holds nothing: where an expert is has one answer, the directory's
//! (`Directory::located`).

/// A staged expert: a streamed layer (an index of the bank's layers) and
/// one of its experts.
pub(super) type Staged = (usize, usize);

pub(super) struct Staging {
    /// The free records.
    free: Vec<u32>,
    /// Its records.
    records: usize,
}

impl Staging {
    /// `records` records, every one free.
    pub fn new(records: u32) -> Self {
        Self { free: (0..records).collect(), records: records as usize }
    }

    /// Its records.
    pub fn records(&self) -> usize {
        self.records
    }

    pub fn free(&self) -> usize {
        self.free.len()
    }

    /// Take a free record.
    pub fn take(&mut self) -> Option<u32> {
        self.free.pop()
    }

    /// Take `count` free records; none when fewer are free.
    pub fn take_many(&mut self, count: usize) -> Option<Vec<u32>> {
        let left = self.free.len().checked_sub(count)?;
        Some(self.free.split_off(left))
    }

    /// Free `record`.
    pub fn release(&mut self, record: u32) {
        assert!((record as usize) < self.records && !self.free.contains(&record), "staging record {record} freed twice or out of {}", self.records);
        self.free.push(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records are taken while free and taken again once released.
    #[test]
    fn records_are_taken_while_free() {
        let mut staging = Staging::new(3);
        let taken = staging.take_many(2).unwrap();
        assert_eq!(staging.free(), 1);
        assert_eq!(staging.take_many(2), None, "too few free");
        let last = staging.take().unwrap();
        assert_eq!(staging.take(), None);
        staging.release(last);
        taken.into_iter().for_each(|record| staging.release(record));
        assert_eq!((staging.free(), staging.records()), (3, 3));
    }
}
