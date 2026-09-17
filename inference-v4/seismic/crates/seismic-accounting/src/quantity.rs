//! Counts and their evidence. Unknown is not zero, and an upper bound is not demand.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Count {
    Exact(u64),
    Interval(Bounds),
    Unknown { reason: String },
}

/// Constructed through `Count::interval`, so lower <= upper always holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bounds {
    lower: u64,
    upper: u64,
}

impl Count {
    pub fn interval(lower: u64, upper: u64) -> Result<Self, String> {
        if lower > upper {
            return Err("count lower bound exceeds upper bound".into());
        }
        Ok(if lower == upper {
            Self::Exact(lower)
        } else {
            Self::Interval(Bounds { lower, upper })
        })
    }

    pub fn bounds(&self) -> Option<(u64, u64)> {
        match self {
            Self::Exact(n) => Some((*n, *n)),
            Self::Interval(bounds) => Some((bounds.lower, bounds.upper)),
            Self::Unknown { .. } => None,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        match (self.bounds(), other.bounds()) {
            (Some((a, b)), Some((c, d))) => match (a.checked_add(c), b.checked_add(d)) {
                (Some(lower), Some(upper)) => Self::interval(lower, upper).unwrap(),
                _ => Self::unknown("count addition exceeds u64 range"),
            },
            _ => Self::unknown(format!("sum has unavailable term: {self:?} + {other:?}")),
        }
    }

    pub fn scale(&self, factor: u64) -> Self {
        if factor == 0 {
            return Self::Exact(0);
        }
        match self.bounds() {
            Some((a, b)) => match (a.checked_mul(factor), b.checked_mul(factor)) {
                (Some(lower), Some(upper)) => Self::interval(lower, upper).unwrap(),
                _ => Self::unknown("count multiplication exceeds u64 range"),
            },
            None => self.clone(),
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        if self == &Self::Exact(0) || other == &Self::Exact(0) {
            return Self::Exact(0);
        }
        match (self.bounds(), other.bounds()) {
            (Some((a, b)), Some((c, d))) => match (a.checked_mul(c), b.checked_mul(d)) {
                (Some(lower), Some(upper)) => Self::interval(lower, upper).unwrap(),
                _ => Self::unknown("count multiplication exceeds u64 range"),
            },
            _ => Self::unknown(format!(
                "product has unavailable term: {self:?} * {other:?}"
            )),
        }
    }
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_and_overflow_are_not_zero() {
        assert!(Count::unknown("runtime visibility")
            .add(&Count::Exact(5))
            .bounds()
            .is_none());
        assert!(Count::Exact(u64::MAX)
            .add(&Count::Exact(1))
            .bounds()
            .is_none());
        assert!(Count::Exact(u64::MAX).scale(2).bounds().is_none());
        assert_eq!(
            Count::unknown("unexecuted branch").scale(0),
            Count::Exact(0)
        );
    }

    #[test]
    fn interval_arithmetic_preserves_both_limits() {
        let a = Count::interval(2, 8).unwrap();
        assert_eq!(
            a.add(&Count::Exact(2)).scale(3),
            Count::interval(12, 30).unwrap()
        );
        assert!(Count::interval(9, 8).is_err());
    }
}
