//! Whether a load's expert bank is at rest across a timed span: the one
//! owner of what a steady prefill sample or decode step is, which every
//! cell's timing takes ([`BankRest`]).

use crate::engine::Engine;
use serde_json::{json, Value};

/// The most spans a cell runs for each steady one it wants: past them the
/// bank is taken never to rest, and the cell is refused with what moved.
pub const STEADY_ATTEMPTS: usize = 4;

/// What a timed span's regime follows: the expert bank's resizes begun and
/// whether one runs, its slots, and its host tier's slots and fill. A span
/// runs steady when none of these moved across it, no resize ran at either
/// end and the tier's fill had finished before it (`TierFill::Filled`):
/// its misses are served as every later span's are, by a bank and tier of
/// the capacity they will keep. Which experts they hold churns at that
/// capacity, each miss displacing another, which is the steady state
/// itself. A load without a bank is always at rest, as a unified load's
/// bank is beside no tier to fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankRest {
    resizes: magnitude_executor::ExpertResizes,
    slots: u64,
    host: Option<(u64, magnitude_executor::TierFill)>,
}

impl BankRest {
    /// Its host tier's slots; none without a tier.
    pub fn host_slots(&self) -> Option<u64> {
        self.host.map(|(slots, _)| slots)
    }

    /// The bank as `engine` holds it now; none without a bank.
    pub fn of(engine: &Engine) -> Result<Option<Self>, String> {
        let residency = engine.domain.expert_residency().map_err(|error| format!("the expert residency: {error}"))?;
        Ok(residency.map(|residency| Self {
            resizes: engine.domain.expert_resizes(),
            slots: residency.slots,
            host: residency.host.map(|host| (host.slots, host.fill)),
        }))
    }

    /// The bank as `engine` holds it once its host tier is filled, waited
    /// for: a span begins only there. The tier grows and fills on the
    /// bank's own threads, for minutes where it backs much, so no span need
    /// run to drive it; the wait is woken by the fill's publication
    /// (`ExecutorDomain::wait_expert_tier_filled`) and bounded only by the
    /// run's own.
    pub fn filled(engine: &Engine) -> Result<Option<Self>, String> {
        engine.domain.wait_expert_tier_filled().map_err(|error| format!("the host tier's fill: {error}"))?;
        Self::of(engine)
    }

    /// Whether a span may begin or end at this bank: no resize runs and
    /// the tier, where there is one, is filled.
    fn settled(self) -> bool {
        !self.resizes.running && self.host.is_none_or(|(_, fill)| fill == magnitude_executor::TierFill::Filled)
    }

    /// What moved between `before`, read before a span, and `after`; none
    /// where the span ran steady.
    pub fn moved(before: Option<Self>, after: Option<Self>) -> Option<Value> {
        let (before, after) = before.zip(after)?;
        (before != after || !before.settled()).then(|| json!({
            "resizes_begun": after.resizes.begun - before.resizes.begun,
            "resizing": [before.resizes.running, after.resizes.running],
            "slots": [before.slots, after.slots],
            "host_slots": [before.host.map(|(slots, _)| slots), after.host.map(|(slots, _)| slots)],
            "tier_fill": [before.host.map(|(_, fill)| format!("{fill:?}")), after.host.map(|(_, fill)| format!("{fill:?}"))],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::TierFill::{Filled, Filling};

    fn rest(begun: u64, running: bool, slots: u64, host: Option<(u64, magnitude_executor::TierFill)>) -> Option<BankRest> {
        Some(BankRest { resizes: magnitude_executor::ExpertResizes { begun, running }, slots, host })
    }

    /// Experts churning through a bank and tier at fixed slots (the bytes
    /// only the file holds moving with each stored size swapped) leave
    /// nothing of the rest to move: steady, beside a filled tier or none.
    #[test]
    fn churn_at_fixed_slots_is_steady() {
        for at_rest in [rest(3, false, 49, None), rest(3, false, 48, Some((256, Filled)))] {
            assert_eq!(BankRest::moved(at_rest, at_rest), None, "{at_rest:?}");
        }
    }

    /// A tier still filling at either end makes a span unsteady; a filled
    /// one, steady.
    #[test]
    fn a_filling_tier_is_unsteady() {
        let filled = rest(3, false, 48, Some((256, Filled)));
        let filling = rest(3, false, 48, Some((256, Filling)));
        assert!(BankRest::moved(filling, filling).is_some());
        assert!(BankRest::moved(filling, filled).is_some());
        assert!(BankRest::moved(filled, filling).is_some());
        assert_eq!(BankRest::moved(filled, filled), None);
    }

    /// A resize begun across a span or running at either end, or slots
    /// moved, make it unsteady.
    #[test]
    fn a_resize_is_unsteady() {
        let at_rest = rest(3, false, 48, Some((256, Filled)));
        for after in [
            rest(4, false, 48, Some((256, Filled))),
            rest(3, true, 48, Some((256, Filled))),
            rest(3, false, 49, Some((256, Filled))),
            rest(3, false, 48, Some((240, Filled))),
        ] {
            assert!(BankRest::moved(at_rest, after).is_some(), "{after:?}");
        }
        let running = rest(3, true, 48, Some((225, Filled)));
        assert!(BankRest::moved(running, running).is_some(), "a resize running at both ends");
    }

    /// A load without a bank is always steady.
    #[test]
    fn no_bank_is_steady() {
        assert_eq!(BankRest::moved(None, None), None);
    }
}
