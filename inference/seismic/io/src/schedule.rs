//! Orthogonal, total-ordered, NaN-free, clock-syscall-free scheduling keys,
//! content identity for strict join-in-flight, and batch-coalescing
//! compatibility tags.

use crate::fabric::{DomainId, RepresentationId};

/// Orthogonal priority class governing preemption and admission bounds.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityClass {
    /// Active compute is stalled or will stall synchronously on this transfer
    /// (decode demand miss, prefill critical fill, write-back freeing a blocked slot).
    Blocking = 0,
    /// Rate-matched continuous stream (prompt conveyor) filling to due-depth
    /// without starving `Blocking`.
    Streaming = 1,
    /// Speculative / predicted work targeting a future deadline (hints, front rows,
    /// draft-window carryover). Strictly bounded by knee and residual service time.
    Prefetch = 2,
    /// Opportunistic background residency improvement (permanent VRAM promotion,
    /// warmup, model load). Runs only in unused headroom.
    Background = 3,
}

impl PriorityClass {
    /// Number of priority classes.
    pub const COUNT: usize = 4;

    /// Array index (`0..4`).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// True for `Prefetch` and `Background` (subject to `Blocking` preemption,
    /// knee cap, and residual speculative service-time bounding).
    #[inline]
    pub const fn is_speculative(self) -> bool {
        matches!(self, Self::Prefetch | Self::Background)
    }
}

/// Cross-request urgency assigned by the executor's request scheduler (e.g.,
/// virtual finish time / fair-share epoch), ensuring Request A's local step 5
/// never blindly starves Request B's local step 900.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalUrgency(pub u64);

/// Request-local deadline ordering within a single request/batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalDeadline {
    pub step: u64,
    pub layer: u16,
    /// Fixed-point inverted probability: `((1.0 - p.clamp(0.0, 1.0)) * 65535.0) as u16`.
    /// `0` = highest probability (`p = 1.0`); total `Ord`, branchless, zero NaN.
    pub inv_prob_q16: u16,
}

impl LocalDeadline {
    /// Construct a deadline from `(step, layer, probability)`.
    /// Non-finite probabilities clamp safely: `NaN` maps to `0.0` (lowest priority).
    #[inline]
    pub fn from_prob(step: u64, layer: u16, prob: f64) -> Self {
        let p = if prob.is_nan() { 0.0 } else { prob.clamp(0.0, 1.0) };
        let inv_prob_q16 = ((1.0 - p) * 65535.0).round() as u16;
        Self {
            step,
            layer,
            inv_prob_q16,
        }
    }

    /// Immediate deadline (`step = 0, layer = 0, p = 1.0`).
    #[inline]
    pub const fn immediate() -> Self {
        Self {
            step: 0,
            layer: 0,
            inv_prob_q16: 0,
        }
    }
}

/// Total-ordered scheduling key replacing v1's `Want` and `Copies::TransferPriority`.
/// Lower key = dispatched earlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScheduleKey {
    pub class: PriorityClass,
    pub urgency: GlobalUrgency,
    pub local: LocalDeadline,
    /// Monotonic submission tie-breaker.
    pub seq: u64,
}

impl ScheduleKey {
    /// Convenience constructor for a `Blocking` demand transfer.
    #[inline]
    pub const fn blocking(urgency: GlobalUrgency, step: u64, layer: u16, seq: u64) -> Self {
        Self {
            class: PriorityClass::Blocking,
            urgency,
            local: LocalDeadline {
                step,
                layer,
                inv_prob_q16: 0,
            },
            seq,
        }
    }

    /// Convenience constructor for a `Streaming` transfer.
    #[inline]
    pub const fn streaming(urgency: GlobalUrgency, step: u64, layer: u16, seq: u64) -> Self {
        Self {
            class: PriorityClass::Streaming,
            urgency,
            local: LocalDeadline {
                step,
                layer,
                inv_prob_q16: 0,
            },
            seq,
        }
    }

    /// Convenience constructor for a `Prefetch` speculative transfer.
    #[inline]
    pub fn prefetch(urgency: GlobalUrgency, step: u64, layer: u16, prob: f64, seq: u64) -> Self {
        Self {
            class: PriorityClass::Prefetch,
            urgency,
            local: LocalDeadline::from_prob(step, layer, prob),
            seq,
        }
    }

    /// Convenience constructor for a `Background` transfer (e.g. permanent promotion).
    #[inline]
    pub const fn background(urgency: GlobalUrgency, seq: u64) -> Self {
        Self {
            class: PriorityClass::Background,
            urgency,
            local: LocalDeadline {
                step: u64::MAX,
                layer: u16::MAX,
                inv_prob_q16: u16::MAX,
            },
            seq,
        }
    }
}

/// Immutable content identity for strict join-in-flight deduplication.
///
/// A new request joins an existing `Queued`/`Submitting`/`Issued` physical op
/// **if and only if**:
/// 1. Content is immutable and `source_gen` matches,
/// 2. `representation_id` (quant format / layout) matches identically, and
/// 3. The joiner's [`PlacementRequirement`] is [`PlacementRequirement::AliasInDomain`]
///    matching the existing op's destination domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContentKey {
    pub asset_id: u64,
    pub slice_id: u64,
    pub representation_id: RepresentationId,
    pub source_gen: u64,
}

/// Destination placement constraint governing whether a request may alias-join
/// an in-flight transfer for the same [`ContentKey`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlacementRequirement {
    /// Joiner accepts an alias lease to any destination slot in `DomainId`
    /// (e.g. via GPU map indirection).
    AliasInDomain(DomainId),
    /// Joiner requires bytes at its own fixed destination address and CANNOT
    /// alias-join an existing op landing elsewhere.
    FixedDestination,
}

/// Compatibility tag governing work-conserving batch coalescing on a Link.
///
/// Items are coalesced into one hardware batch only when compatible so no
/// early consumer ever waits on a late-retiring batchmate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BatchCompatibility {
    /// Shares a retirement consumer (e.g., same `(step, layer)` `DecodeBack`).
    SharedConsumer { step: u64, layer: u16 },
    /// Carries an inline `DeviceSignal` (e.g., `MAP_STAGE` write) so each item
    /// signals readiness independently without waiting for batchmates.
    InlineSignal,
    /// Must be issued in its own batch (no coalescing with other ops).
    Isolated,
}

impl BatchCompatibility {
    /// Whether `other` can be coalesced into the same hardware batch as `self`.
    #[inline]
    pub const fn compatible_with(self, other: Self) -> bool {
        match (self, other) {
            (
                Self::SharedConsumer { step: s1, layer: l1 },
                Self::SharedConsumer { step: s2, layer: l2 },
            ) => s1 == s2 && l1 == l2,
            (Self::InlineSignal, Self::InlineSignal) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_key_total_order_and_nan_safety() {
        let b = ScheduleKey::blocking(GlobalUrgency(10), 5, 1, 100);
        let s = ScheduleKey::streaming(GlobalUrgency(1), 0, 0, 1);
        let p_hi = ScheduleKey::prefetch(GlobalUrgency(1), 5, 1, 0.9, 2);
        let p_lo = ScheduleKey::prefetch(GlobalUrgency(1), 5, 1, 0.2, 1);
        let p_nan = ScheduleKey::prefetch(GlobalUrgency(1), 5, 1, f64::NAN, 0);
        let bg = ScheduleKey::background(GlobalUrgency(0), 0);

        assert!(b < s);
        assert!(s < p_hi);
        assert!(p_hi < p_lo);
        assert!(p_lo < p_nan);
        assert!(p_nan < bg);

        // GlobalUrgency prevents Request A's step 5 from starving Request B's step 900.
        let req_b = ScheduleKey::prefetch(GlobalUrgency(1), 900, 0, 0.5, 10);
        let req_a = ScheduleKey::prefetch(GlobalUrgency(2), 5, 0, 0.99, 1);
        assert!(req_b < req_a);
    }

    #[test]
    fn batch_compatibility_rules() {
        let c1 = BatchCompatibility::SharedConsumer { step: 7, layer: 3 };
        let c2 = BatchCompatibility::SharedConsumer { step: 7, layer: 3 };
        let c3 = BatchCompatibility::SharedConsumer { step: 7, layer: 4 };
        assert!(c1.compatible_with(c2));
        assert!(!c1.compatible_with(c3));
        assert!(BatchCompatibility::InlineSignal.compatible_with(BatchCompatibility::InlineSignal));
        assert!(!BatchCompatibility::InlineSignal.compatible_with(c1));
        assert!(!BatchCompatibility::Isolated.compatible_with(BatchCompatibility::Isolated));
    }
}
