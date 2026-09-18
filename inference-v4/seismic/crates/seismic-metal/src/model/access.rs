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

    /// Distinct aligned blocks intersecting this one operation. An allocation
    /// with weaker alignment does not establish the unknown base-address residue.
    pub fn transactions(&self, bytes: u64) -> Option<u64> {
        if !bytes.is_power_of_two() || bytes > self.alignment { return None; }
        let mut count = 0u64;
        let mut previous_end = 0;
        for &(start, end) in &self.intervals {
            let first = (start / bytes).max(previous_end);
            let end = end.div_ceil(bytes);
            count = count.checked_add(end.saturating_sub(first))?;
            previous_end = previous_end.max(end);
        }
        Some(count)
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
