use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

/// Readouts requested for one model-execution row.
///
/// Selection consumes logits on device without publishing a host logits row;
/// likewise [`Self::DEVICE_FEATURES`] computes features on device without
/// publishing a host feature lease. Keeping the bits distinct preserves both
/// contracts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Demand(u32);

impl Demand {
    pub const NONE: Self = Self(0);
    pub const LOGITS: Self = Self(1 << 0);
    /// Compute features and publish them as a host-readable feature lease.
    pub const FEATURES: Self = Self(1 << 1);
    pub const SELECT: Self = Self(1 << 2);
    /// Reserved for future layer-boundary feature readout.
    pub const TAPS: Self = Self(1 << 3);
    /// Compute features on device without publishing a host feature lease.
    ///
    /// Mutually exclusive with [`Self::FEATURES`]: publishing features already
    /// computes them on device, so combining both absorbs into [`Self::FEATURES`]
    /// and [`Self::from_bits`] rejects the dual-bit encoding.
    pub const DEVICE_FEATURES: Self = Self(1 << 4);
    /// Every representable bit combined (`FEATURES` subsumes `DEVICE_FEATURES`).
    pub const ALL: Self = Self(Self::LOGITS.0 | Self::FEATURES.0 | Self::SELECT.0 | Self::TAPS.0);

    const VALID_BITS: u32 =
        Self::LOGITS.0 | Self::FEATURES.0 | Self::SELECT.0 | Self::TAPS.0 | Self::DEVICE_FEATURES.0;
    const BOTH_FEATURES: u32 = Self::FEATURES.0 | Self::DEVICE_FEATURES.0;

    const fn normalize(bits: u32) -> u32 {
        if bits & Self::FEATURES.0 != 0 {
            bits & !Self::DEVICE_FEATURES.0
        } else {
            bits
        }
    }

    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::VALID_BITS == 0 && (bits & Self::BOTH_FEATURES) != Self::BOTH_FEATURES {
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

    /// Whether execution must compute features on device, whether or not a
    /// host feature lease is published.
    pub const fn computes_features(self) -> bool {
        self.intersects(Self(Self::FEATURES.0 | Self::DEVICE_FEATURES.0))
    }

    /// Replace host-published feature demand with device-only feature demand.
    pub const fn device_only_features(self) -> Self {
        if self.contains(Self::FEATURES) {
            Self((self.0 & !Self::FEATURES.0) | Self::DEVICE_FEATURES.0)
        } else {
            self
        }
    }

    /// Drop logits and selection bits, keeping any feature and tap demand.
    pub const fn without_logits(self) -> Self {
        Self(self.0 & !(Self::LOGITS.0 | Self::SELECT.0))
    }
}

impl BitOr for Demand {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(Self::normalize(self.0 | rhs.0))
    }
}

impl BitOrAssign for Demand {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = Self::normalize(self.0 | rhs.0);
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
    fn device_features_computes_but_does_not_publish_features() {
        assert!(Demand::DEVICE_FEATURES.computes_features());
        assert!(!Demand::DEVICE_FEATURES.contains(Demand::FEATURES));
        assert!(Demand::FEATURES.computes_features());
        assert!(Demand::FEATURES.contains(Demand::FEATURES));
        // Publishing subsumes device-only features: both bits can never coexist.
        assert_eq!(Demand::FEATURES | Demand::DEVICE_FEATURES, Demand::FEATURES);
        assert_eq!(Demand::DEVICE_FEATURES | Demand::FEATURES, Demand::FEATURES);
        let mut combined = Demand::DEVICE_FEATURES;
        combined |= Demand::FEATURES;
        assert_eq!(combined, Demand::FEATURES);
        assert_eq!(Demand::from_bits(Demand::FEATURES.bits() | Demand::DEVICE_FEATURES.bits()), None);
        assert_eq!(
            (Demand::SELECT | Demand::DEVICE_FEATURES | Demand::TAPS).without_logits(),
            Demand::DEVICE_FEATURES | Demand::TAPS,
        );
    }

    #[test]
    fn unknown_bits_are_rejected() {
        assert_eq!(Demand::from_bits(Demand::ALL.bits()), Some(Demand::ALL));
        assert_eq!(Demand::from_bits(1 << 5), None);
    }
}
