//! Exact finite integer domains, retained as disjoint arithmetic runs.
//!
//! No operation expands an interval merely to split it or remove a value.
use super::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Run {
    first: i64,
    last: i64,
    step: u64,
}
impl Run {
    fn count(&self) -> u128 {
        ((self.last as i128 - self.first as i128) as u128) / self.step as u128 + 1
    }
    fn at(&self, index: u128) -> i64 {
        (self.first as i128 + (index * self.step as u128) as i128) as i64
    }
    fn restrict(&self, lo: i64, hi: i64) -> Option<Self> {
        let lo = lo.max(self.first);
        let hi = hi.min(self.last);
        if lo > hi {
            return None;
        }
        let distance = (lo as i128 - self.first as i128) as u128;
        let first_index = distance.div_ceil(self.step as u128);
        let last_index = ((hi as i128 - self.first as i128) as u128) / self.step as u128;
        (first_index <= last_index).then(|| Self {
            first: self.at(first_index),
            last: self.at(last_index),
            step: self.step,
        })
    }
}

/// A finite set of i64 values. Empty domains represent a contradiction.
/// Cardinality is u128 because the complete i64 interval has 2^64 members.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Domain {
    runs: Vec<Run>,
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Domain {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Representation {
            runs: Vec<Run>,
        }
        let value = Representation::deserialize(deserializer)?;
        let domain = Domain { runs: value.runs };
        domain.validate().map_err(serde::de::Error::custom)?;
        Ok(Domain::from_runs(domain.runs))
    }
}
impl Domain {
    pub fn empty() -> Self {
        Self { runs: Vec::new() }
    }
    pub fn singleton(value: i64) -> Self {
        Self {
            runs: vec![Run {
                first: value,
                last: value,
                step: 1,
            }],
        }
    }
    pub fn boolean() -> Self {
        Self::interval(0, 1).expect("boolean interval")
    }
    pub fn interval(min: i64, max: i64) -> Result<Self> {
        Self::progression(min, max, 1)
    }
    /// `last` is an inclusive bound; it need not itself lie on the progression.
    pub fn progression(first: i64, last: i64, step: u64) -> Result<Self> {
        if first > last || step == 0 {
            return Err(Error::InvalidModel(
                "domain requires first <= last and positive step".into(),
            ));
        }
        let n = ((last as i128 - first as i128) as u128) / step as u128;
        Ok(Self::from_runs(vec![Run {
            first,
            last: (first as i128 + (n * step as u128) as i128) as i64,
            step,
        }]))
    }
    pub fn set(values: impl IntoIterator<Item = i64>) -> Self {
        let mut values: Vec<_> = values.into_iter().collect();
        values.sort_unstable();
        values.dedup();
        let mut runs = Vec::new();
        let mut i = 0;
        while i < values.len() {
            let first = values[i];
            let mut last = first;
            let mut step = 1;
            if i + 1 < values.len() {
                step = (values[i + 1] as i128 - first as i128) as u64;
                i += 1;
                last = values[i];
                while i + 1 < values.len() && values[i + 1] as i128 - last as i128 == step as i128 {
                    i += 1;
                    last = values[i];
                }
            }
            runs.push(Run { first, last, step });
            i += 1;
        }
        Self::from_runs(runs)
    }
    fn from_runs(runs: Vec<Run>) -> Self {
        let mut normalized: Vec<Run> = Vec::with_capacity(runs.len());
        for mut run in runs {
            if run.first == run.last {
                run.step = 1;
            }
            if let Some(previous) = normalized.last_mut() {
                if previous.step == run.step
                    && previous.last as i128 + run.step as i128 == run.first as i128
                {
                    previous.last = run.last;
                    continue;
                }
            }
            normalized.push(run);
        }
        Self { runs: normalized }
    }
    pub fn validate(&self) -> Result<()> {
        let mut last = None;
        for run in &self.runs {
            if run.step == 0
                || run.first > run.last
                || (run.last as i128 - run.first as i128) % run.step as i128 != 0
                || last.is_some_and(|previous| previous >= run.first)
            {
                return Err(Error::InvalidModel(
                    "invalid or overlapping domain runs".into(),
                ));
            }
            last = Some(run.last);
        }
        Ok(())
    }
    pub fn min(&self) -> Option<i64> {
        self.runs.first().map(|r| r.first)
    }
    pub fn max(&self) -> Option<i64> {
        self.runs.last().map(|r| r.last)
    }
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
    pub fn is_singleton(&self) -> bool {
        self.runs.len() == 1 && self.runs[0].first == self.runs[0].last
    }
    pub fn singleton_value(&self) -> Option<i64> {
        self.is_singleton().then(|| self.runs[0].first)
    }
    pub fn cardinality(&self) -> u128 {
        self.runs.iter().map(Run::count).sum()
    }
    pub fn len(&self) -> u128 {
        self.cardinality()
    }
    pub fn contains(&self, value: i64) -> bool {
        let index = self.runs.partition_point(|run| run.last < value);
        self.runs.get(index).is_some_and(|run| {
            value >= run.first && (value as i128 - run.first as i128) % run.step as i128 == 0
        })
    }
    pub fn nth(&self, mut index: u128) -> Option<i64> {
        for run in &self.runs {
            if index < run.count() {
                return Some(run.at(index));
            }
            index -= run.count();
        }
        None
    }
    pub fn values(&self) -> DomainValues<'_> {
        DomainValues {
            domain: self,
            run_index: 0,
            value_index: 0,
        }
    }
    pub fn restrict(&self, min: i64, max: i64) -> Result<Self> {
        if min > max {
            return Ok(Self::empty());
        }
        Ok(Self::from_runs(
            self.runs
                .iter()
                .filter_map(|run| run.restrict(min, max))
                .collect(),
        ))
    }
    pub fn intersect(&self, other: &Self) -> Result<Self> {
        let mut runs = Vec::new();
        let (mut a, mut b) = (0, 0);
        while a < self.runs.len() && b < other.runs.len() {
            if let Some(run) = intersect_runs(&self.runs[a], &other.runs[b])? {
                runs.push(run);
            }
            if self.runs[a].last <= other.runs[b].last {
                a += 1;
            } else {
                b += 1;
            }
        }
        Ok(Self::from_runs(runs))
    }
    /// Removes a value while retaining every other value, including interval holes.
    pub fn without(&self, value: i64) -> Self {
        let mut runs = Vec::with_capacity(self.runs.len() + 1);
        for run in &self.runs {
            if value < run.first
                || value > run.last
                || (value as i128 - run.first as i128) % run.step as i128 != 0
            {
                runs.push(run.clone());
                continue;
            }
            if value > run.first {
                runs.push(Run {
                    first: run.first,
                    last: (value as i128 - run.step as i128) as i64,
                    step: run.step,
                });
            }
            if value < run.last {
                runs.push(Run {
                    first: (value as i128 + run.step as i128) as i64,
                    last: run.last,
                    step: run.step,
                });
            }
        }
        Self::from_runs(runs)
    }
    /// A disjoint exhaustive partition, approximately balanced by cardinality.
    pub fn split(&self) -> Option<(Self, Self)> {
        let len = self.cardinality();
        if len < 2 {
            return None;
        }
        let mut remaining = len / 2;
        let mut left = Vec::new();
        let mut right = Vec::new();
        for run in &self.runs {
            if remaining == 0 {
                right.push(run.clone());
            } else if remaining >= run.count() {
                remaining -= run.count();
                left.push(run.clone());
            } else {
                left.push(Run {
                    first: run.first,
                    last: run.at(remaining - 1),
                    step: run.step,
                });
                right.push(Run {
                    first: run.at(remaining),
                    last: run.last,
                    step: run.step,
                });
                remaining = 0;
            }
        }
        Some((Self::from_runs(left), Self::from_runs(right)))
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.runs.capacity() * std::mem::size_of::<Run>()
    }
}

pub struct DomainValues<'a> {
    domain: &'a Domain,
    run_index: usize,
    value_index: u128,
}
impl Iterator for DomainValues<'_> {
    type Item = i64;
    fn next(&mut self) -> Option<Self::Item> {
        let run = self.domain.runs.get(self.run_index)?;
        let value = run.at(self.value_index);
        self.value_index += 1;
        if self.value_index == run.count() {
            self.run_index += 1;
            self.value_index = 0;
        }
        Some(value)
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
fn inverse_mod(a: u64, modulus: u64) -> u64 {
    let (mut old_r, mut r) = (a as i128, modulus as i128);
    let (mut old_s, mut s) = (1_i128, 0_i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    old_s.rem_euclid(modulus as i128) as u64
}
fn intersect_runs(a: &Run, b: &Run) -> Result<Option<Run>> {
    let lo = a.first.max(b.first);
    let hi = a.last.min(b.last);
    if lo > hi {
        return Ok(None);
    }
    let divisor = gcd(a.step, b.step);
    let delta = b.first as i128 - a.first as i128;
    if delta % divisor as i128 != 0 {
        return Ok(None);
    }
    let modulus = b.step / divisor;
    let t = if modulus == 1 {
        0_u128
    } else {
        let residue = (delta / divisor as i128).rem_euclid(modulus as i128) as u128;
        (residue * inverse_mod(a.step / divisor, modulus) as u128) % modulus as u128
    };
    let period = a.step as u128 * modulus as u128;
    let mut offset = a.step as u128 * t;
    let lo_offset = (lo as i128 - a.first as i128) as u128;
    let hi_offset = (hi as i128 - a.first as i128) as u128;
    if offset < lo_offset {
        let multiples = (lo_offset - offset).div_ceil(period);
        offset = offset
            .checked_add(
                multiples
                    .checked_mul(period)
                    .ok_or_else(|| Error::Overflow("domain intersection".into()))?,
            )
            .ok_or_else(|| Error::Overflow("domain intersection".into()))?;
    }
    if offset > hi_offset {
        return Ok(None);
    }
    let first = (a.first as i128 + offset as i128) as i64;
    let steps = (hi_offset - offset) / period;
    if steps == 0 {
        return Ok(Some(Run {
            first,
            last: first,
            step: 1,
        }));
    }
    let last = (first as i128 + (steps * period) as i128) as i64;
    Ok(Some(Run {
        first,
        last,
        step: u64::try_from(period)
            .map_err(|_| Error::Overflow("domain intersection step".into()))?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_range_splits_without_overflow_or_allocation() {
        let d = Domain::interval(i64::MIN, i64::MAX).unwrap();
        assert_eq!(d.cardinality(), 1_u128 << 64);
        let (a, b) = d.split().unwrap();
        assert_eq!(a.max(), Some(-1));
        assert_eq!(b.min(), Some(0));
        assert_eq!(a.cardinality() + b.cardinality(), d.cardinality());
    }
    #[test]
    fn all_small_intersections_and_splits_match_sets() {
        for start in -6..=6 {
            for end in start..=6 {
                for step in 1..=5 {
                    let a = Domain::progression(start, end, step).unwrap();
                    for other_start in -6..=6 {
                        for other_step in 1..=5 {
                            let b = Domain::progression(other_start, 7, other_step).unwrap();
                            assert_eq!(
                                a.intersect(&b).unwrap().values().collect::<Vec<_>>(),
                                a.values().filter(|v| b.contains(*v)).collect::<Vec<_>>()
                            );
                        }
                    }
                    if let Some((left, right)) = a.split() {
                        assert!(left.intersect(&right).unwrap().is_empty());
                        assert_eq!(
                            left.values().chain(right.values()).collect::<Vec<_>>(),
                            a.values().collect::<Vec<_>>()
                        );
                    }
                    for value in -7..=7 {
                        assert_eq!(
                            a.without(value).values().collect::<Vec<_>>(),
                            a.values().filter(|v| *v != value).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn large_coprime_steps_and_negative_origins() {
        let a = Domain::progression(i64::MIN, i64::MAX, u64::MAX - 2).unwrap();
        let b = Domain::progression(i64::MIN, i64::MAX, u64::MAX - 4).unwrap();
        assert_eq!(a.intersect(&b).unwrap(), Domain::singleton(i64::MIN));
        let a = Domain::progression(-20, 40, 7).unwrap();
        let b = Domain::progression(-11, 40, 4).unwrap();
        assert_eq!(
            a.intersect(&b).unwrap().values().collect::<Vec<_>>(),
            vec![1, 29]
        );
    }
    #[test]
    fn explicit_set_preserves_exclusions() {
        let d = Domain::set([7, -3, 1, 2, 2, 4, 9, 11]);
        assert_eq!(d.values().collect::<Vec<_>>(), vec![-3, 1, 2, 4, 7, 9, 11]);
        assert_eq!(
            d.restrict(2, 8).unwrap().values().collect::<Vec<_>>(),
            vec![2, 4, 7]
        );
        assert_eq!(
            d.intersect(&Domain::progression(-3, 11, 2).unwrap())
                .unwrap()
                .values()
                .collect::<Vec<_>>(),
            vec![-3, 1, 7, 9, 11]
        );
    }
}
