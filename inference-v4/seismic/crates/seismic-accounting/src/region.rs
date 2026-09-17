//! Exact unions of half-open byte regions on identified backing allocations.
//!
//! The caller resolves views and representation planes to backing byte offsets.
//! Allocation capacity is a safety bound, never a substitute for accessed bytes.

use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Regions {
    // Disjoint, nonadjacent half-open intervals, sorted by start.
    intervals: BTreeMap<u64, u64>,
}

impl Regions {
    pub fn insert(&mut self, range: Range<u64>) -> Result<(), String> {
        if range.start > range.end {
            return Err("a byte region cannot have a negative extent".into());
        }
        if range.is_empty() {
            return Ok(());
        }
        let (mut start, mut end) = (range.start, range.end);
        if let Some((&left, &right)) = self.intervals.range(..=start).next_back() {
            if right >= start {
                start = left;
                end = end.max(right);
                self.intervals.remove(&left);
            }
        }
        loop {
            let next = self.intervals.range(start..).next().map(|(&a, &b)| (a, b));
            match next {
                Some((a, b)) if a <= end => {
                    end = end.max(b);
                    self.intervals.remove(&a);
                }
                _ => break,
            }
        }
        self.intervals.insert(start, end);
        Ok(())
    }

    pub fn union(&mut self, other: &Self) {
        for range in other.ranges() {
            // All stored intervals have already been validated.
            self.insert(range).expect("validated region");
        }
    }

    pub fn bytes(&self) -> u64 {
        // Disjoint intervals within [0, u64::MAX) cannot overflow their union size.
        self.intervals.iter().map(|(start, end)| end - start).sum()
    }

    pub fn ranges(&self) -> impl Iterator<Item = Range<u64>> + '_ {
        self.intervals.iter().map(|(&start, &end)| start..end)
    }
}

/// Logical interface obligations, before a memory path or residency is chosen.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Access {
    pub reads: Regions,
    pub writes: Regions,
}

/// Distinct source parameters can resolve to the same backing and share regions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Accesses {
    backings: BTreeMap<String, Access>,
}

impl Accesses {
    pub fn backing(&mut self, identity: impl Into<String>) -> &mut Access {
        self.backings.entry(identity.into()).or_default()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Access)> {
        self.backings
            .iter()
            .map(|(id, access)| (id.as_str(), access))
    }

    pub fn union(&mut self, other: &Self) {
        for (id, access) in other.iter() {
            let target = self.backing(id);
            target.reads.union(&access.reads);
            target.writes.union(&access.writes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_aliases_do_not_count_twice() {
        let mut a = Accesses::default();
        a.backing("allocation").reads.insert(100..200).unwrap();
        let mut b = Accesses::default();
        b.backing("allocation").reads.insert(150..250).unwrap();
        b.backing("different allocation")
            .reads
            .insert(100..200)
            .unwrap();
        a.union(&b);
        assert_eq!(a.backing("allocation").reads.bytes(), 150);
        assert_eq!(a.backing("different allocation").reads.bytes(), 100);
    }

    #[test]
    fn repeated_small_region_never_becomes_allocation_size() {
        let mut regions = Regions::default();
        for _ in 0..1000 {
            regions.insert(100..108).unwrap();
        }
        assert_eq!(regions.bytes(), 8);
    }

    #[test]
    fn bridge_merges_all_neighbors_and_preserves_holes() {
        let mut regions = Regions::default();
        for r in [10..20, 30..40, 50..60, 100..110, 20..50] {
            regions.insert(r).unwrap();
        }
        assert_eq!(regions.ranges().collect::<Vec<_>>(), vec![10..60, 100..110]);
        assert_eq!(regions.bytes(), 60);
        assert!(regions.insert(Range { start: 2, end: 1 }).is_err());
    }

    #[test]
    fn union_agrees_with_independent_byte_bitmap() {
        let mut regions = Regions::default();
        let mut bitmap = [false; 256];
        let mut state = 27u64;
        for _ in 0..200 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = ((state >> 32) % 257) as usize;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = ((state >> 32) % 257) as usize;
            let (start, end) = (a.min(b), a.max(b));
            regions.insert(start as u64..end as u64).unwrap();
            bitmap[start..end].fill(true);
            assert_eq!(
                regions.bytes(),
                bitmap.iter().filter(|b| **b).count() as u64
            );
        }
    }
}
