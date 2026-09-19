//! Address geometry of one typed, active-lane memory operation. This records
//! requested byte intervals; a hardware service separately names its boundary
//! and transaction granularity. There is no cache or cross-instruction reuse
//! assumption in this representation.

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AccessPattern {
    alignment: u64,
    /// Union of requested intervals relative to an allocation-aligned origin.
    /// Removing aligned whole prefixes bounds retained geometry independently
    /// of the absolute position of this access within a large tensor.
    intervals: Vec<(u64, u64)>,
}
impl AccessPattern {
    pub fn alignment(&self) -> u64 { self.alignment }
    pub fn intervals(&self) -> &[(u64, u64)] { &self.intervals }

    /// Distinct aligned blocks intersecting this one operation. The count must
    /// hold for every base residue admitted by the allocation or symbolic
    /// translation alignment; no particular unknown address is selected.
    pub fn transactions(&self, bytes: u64) -> Option<u64> {
        if !bytes.is_power_of_two() { return None; }
        let count = self.transactions_at(bytes, 0)?;
        if bytes <= self.alignment { return Some(count); }

        // There can be arbitrarily many aligned residues, but the touched
        // block union only changes when an interval endpoint crosses a block
        // boundary. Inspect the first admitted residue at each such change.
        // This depends on lane geometry, never tensor or parameter domain size.
        let mut changes = Vec::with_capacity(self.intervals.len() * 2);
        for &(start, end) in &self.intervals {
            let first_changes = bytes - start % bytes;
            let last_changes = if end % bytes == 0 { 1 } else { bytes - end % bytes + 1 };
            for change in [first_changes, last_changes] {
                let residue = change.div_ceil(self.alignment) * self.alignment;
                if residue < bytes { changes.push(residue); }
            }
        }
        changes.sort_unstable(); changes.dedup();
        for residue in changes {
            if self.transactions_at(bytes, residue)? != count { return None; }
        }
        Some(count)
    }

    fn transactions_at(&self, bytes: u64, residue: u64) -> Option<u64> {
        // Relative intervals can already reach u64::MAX. A residue is an
        // abstract translation, so count blocks in a wider integer domain.
        let (bytes, residue) = (u128::from(bytes), u128::from(residue));
        let mut count = 0u128;
        let mut previous_end = 0;
        for &(start, end) in &self.intervals {
            let first = ((u128::from(start) + residue) / bytes).max(previous_end);
            let end = (u128::from(end) + residue).div_ceil(bytes);
            count += end.saturating_sub(first);
            previous_end = previous_end.max(end);
        }
        u64::try_from(count).ok()
    }

    pub(super) fn new(alignment: u64, mut intervals: Vec<(u64, u64)>) -> Option<Self> {
        if !alignment.is_power_of_two() || intervals.is_empty() { return None; }
        intervals.sort_unstable();
        let origin = intervals[0].0 / alignment * alignment;
        let mut union: Vec<(u64, u64)> = Vec::new();
        for (start, end) in intervals {
            if start >= end { return None; }
            let range = (start - origin, end - origin);
            if let Some(last) = union.last_mut() {
                if range.0 <= last.1 { last.1 = last.1.max(range.1); continue; }
            }
            union.push(range);
        }
        Some(Self { alignment, intervals: union })
    }
}
