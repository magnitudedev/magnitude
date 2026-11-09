//! Whether a load's expert bank is at rest across a timed span: the one
//! owner of what a steady prefill sample or decode step is, which every
//! cell's timing takes ([`BankRest`]).

use crate::engine::Engine;
use serde_json::{json, Value};

/// The most spans a cell runs for each steady one it wants: past them the
/// bank is taken never to rest, and the cell is refused with what moved.
pub const STEADY_ATTEMPTS: usize = 4;

/// What a timed span's regime follows: the expert bank's resizes begun and
/// whether one runs, its slots, its host tier's slots, and the fill of the
/// store its misses land into (the host tier where the device places, the
/// one store where the host does). A span runs steady when none of these
/// moved across it, no resize ran at either end and the fill had finished
/// before it (`TierFill::Filled`): its misses are served as every later
/// span's are, by a store of the capacity and fill it will keep, and no
/// fill's reads share the drive with the span's own. Which experts it
/// holds churns at that capacity, each miss displacing another, which is
/// the steady state itself. A load without a bank is always at rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankRest {
    resizes: magnitude_executor::ExpertResizes,
    slots: u64,
    host: Option<u64>,
    fill: magnitude_executor::TierFill,
}

impl BankRest {
    /// Its host tier's slots; none without a tier.
    pub fn host_slots(&self) -> Option<u64> {
        self.host
    }

    /// The bank as `engine` holds it now; none without a bank.
    pub fn of(engine: &Engine) -> Result<Option<Self>, String> {
        let residency = engine.domain.expert_residency().map_err(|error| format!("the expert residency: {error}"))?;
        Ok(residency.map(|residency| Self {
            resizes: engine.domain.expert_resizes(),
            slots: residency.slots,
            host: residency.host.map(|host| host.slots),
            fill: residency.fill,
        }))
    }

    /// The bank as `engine` holds it once the store its misses land into is
    /// filled, waited for: a span begins only there. The store grows and
    /// fills on the bank's own threads, for minutes where it backs much, so
    /// no span need run to drive it; the wait is woken by the fill's
    /// publication (`ExecutorDomain::wait_expert_tier_filled`) and bounded
    /// only by the run's own.
    pub fn filled(engine: &Engine) -> Result<Option<Self>, String> {
        engine.domain.wait_expert_tier_filled().map_err(|error| format!("the expert store's fill: {error}"))?;
        Self::of(engine)
    }

    /// Whether a span may begin or end at this bank: no resize runs and
    /// the store its misses land into is filled.
    fn settled(self) -> bool {
        !self.resizes.running && self.fill == magnitude_executor::TierFill::Filled
    }

    /// What moved between `before`, read before a span, and `after`; none
    /// where the span ran steady.
    pub fn moved(before: Option<Self>, after: Option<Self>) -> Option<Value> {
        let (before, after) = before.zip(after)?;
        (before != after || !before.settled()).then(|| json!({
            "resizes_begun": after.resizes.begun - before.resizes.begun,
            "resizing": [before.resizes.running, after.resizes.running],
            "slots": [before.slots, after.slots],
            "host_slots": [before.host, after.host],
            "fill": [format!("{:?}", before.fill), format!("{:?}", after.fill)],
        }))
    }
}

/// Warm a decode to rest before its window: `warm` steps, then more until
/// one runs at rest. `unrested` is what the bank moved across the window
/// just discarded, if one was: the decode is warm then, and steps on only
/// until a step runs at rest, so no window begins inside a resize. Past
/// `STEADY_ATTEMPTS` times the cell's warm steps the bank is taken never to
/// rest. `step` runs one step and says what moved across it
/// ([`BankRest::moved`]).
pub fn warm_to_rest(
    warm: usize,
    unrested: Option<Value>,
    mut step: impl FnMut() -> Result<Option<Value>, String>,
) -> Result<(), String> {
    let wanted = if unrested.is_none() { warm } else { 0 };
    let mut moved = unrested;
    let mut warmed = 0;
    while warmed < wanted || moved.is_some() {
        if warmed == STEADY_ATTEMPTS * warm.max(1) {
            return Err(format!(
                "the expert bank was not at rest after {warmed} warm decode steps: {}",
                moved.unwrap_or(Value::Null)
            ));
        }
        moved = step()?;
        warmed += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::TierFill::{Filled, Filling};

    fn rest(begun: u64, running: bool, slots: u64, host: Option<(u64, magnitude_executor::TierFill)>) -> Option<BankRest> {
        on(begun, running, slots, host.map(|(slots, _)| slots), host.map_or(Filled, |(_, fill)| fill))
    }

    fn on(begun: u64, running: bool, slots: u64, host: Option<u64>, fill: magnitude_executor::TierFill) -> Option<BankRest> {
        Some(BankRest { resizes: magnitude_executor::ExpertResizes { begun, running }, slots, host, fill })
    }

    /// The one store (no host tier) still filling at either end makes a
    /// span unsteady, as a host tier's fill does: its fill's reads share the
    /// drive with the span's, and its records not yet filled take the
    /// span's seeds, so a span begun mid-fill is served unlike any later.
    #[test]
    fn a_filling_one_store_is_unsteady() {
        let (filled, filling) = (on(3, false, 180, None, Filled), on(3, false, 180, None, Filling));
        assert!([filled, filling].iter().all(|rest| rest.is_some_and(|rest| rest.host_slots().is_none())), "setup: a bank beside no host tier");
        assert!(BankRest::moved(filling, filling).is_some());
        assert!(BankRest::moved(filling, filled).is_some());
        assert!(BankRest::moved(filled, filling).is_some());
        assert_eq!(BankRest::moved(filled, filled), None);
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

    /// After a window the bank moved across (a resize still running at its
    /// end), the decode steps on until a step runs at rest, as long as the
    /// first warm-up may take, before the next window begins.
    #[test]
    fn a_discarded_window_warms_on_to_rest() {
        let moved = json!({ "resizing": [true, true] });
        // The resize runs on for 10 steps; the cell warms 8.
        let mut steps = 0;
        let warmed = warm_to_rest(8, Some(moved.clone()), || {
            steps += 1;
            Ok((steps <= 10).then(|| moved.clone()))
        });
        assert_eq!((warmed, steps), (Ok(()), 11));
        // A bank that never rests is refused past the cell's bound.
        let mut steps = 0;
        let refused = warm_to_rest(8, Some(moved.clone()), || {
            steps += 1;
            Ok(Some(moved.clone()))
        });
        assert!(refused.is_err());
        assert_eq!(steps, STEADY_ATTEMPTS * 8);
    }
}
