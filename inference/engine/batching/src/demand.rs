use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

/// Readouts requested for one model-execution row.
///
/// Selection consumes logits on device but does not imply that a logits row is
/// transferred to the host. Keeping the bits distinct preserves that contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Demand(u32);

impl Demand {
    pub const NONE: Self = Self(0);
    pub const LOGITS: Self = Self(1 << 0);
    pub const FEATURES: Self = Self(1 << 1);
    pub const SELECT: Self = Self(1 << 2);
    /// Reserved for future layer-boundary feature readout.
    pub const TAPS: Self = Self(1 << 3);
    pub const ALL: Self = Self(Self::LOGITS.0 | Self::FEATURES.0 | Self::SELECT.0 | Self::TAPS.0);

    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Whether execution must compute logits, including device-only selection.
    pub const fn computes_logits(self) -> bool {
        self.intersects(Self(Self::LOGITS.0 | Self::SELECT.0))
    }
}

impl BitOr for Demand {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Demand {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for Demand {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Demand {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_computes_but_does_not_publish_logits() {
        assert!(Demand::SELECT.computes_logits());
        assert!(!Demand::SELECT.contains(Demand::LOGITS));
        assert!((Demand::SELECT | Demand::FEATURES).contains(Demand::FEATURES));
    }

    #[test]
    fn unknown_bits_are_rejected() {
        assert_eq!(Demand::from_bits(Demand::ALL.bits()), Some(Demand::ALL));
        assert_eq!(Demand::from_bits(1 << 4), None);
    }
}
