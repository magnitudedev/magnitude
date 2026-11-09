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

/// The rows past its history a timed decode of `warm` warm steps, windows
/// of `steps` and `attribution` attributed steps after its window advances
/// at most, its discarded attempts too ([`steady_window`]): what its load's
/// context holds beyond the history.
pub fn decode_context_rows(warm: usize, steps: usize, attribution: usize) -> usize {
    warm + steps + attribution + 1
}

/// A decode's one window the bank rests across, the decode never deeper
/// than its histories by more than `warm` steps and its window: each
/// attempt warms the decode to rest, then runs its window (`window`: what
/// it gives, or what the bank moved across it, discarding it), and a
/// discarded attempt's decode begins again from its histories (`rewind`),
/// so the next runs at the depth the first did, as the bank stands.
///
/// The warm-up is `warm` steps (`step` one step, saying what moved across
/// it: [`BankRest::moved`]), then more until one runs at rest, so no window
/// begins inside a resize; each `warm` steps it begins again from the
/// histories. A warm-up past `warm` steps is run again from them, `warm`
/// steps whose last ran at rest, so the window follows a warm-up as the
/// first ran. Past `STEADY_ATTEMPTS` times the cell's warm steps in one
/// warm-up, or `STEADY_ATTEMPTS` attempts, the bank is taken never to rest.
/// Gives the window and the discarded attempts' records, in order.
pub fn steady_window<S, W>(
    decode: &mut S,
    warm: usize,
    mut step: impl FnMut(&mut S) -> Result<Option<Value>, String>,
    mut window: impl FnMut(&mut S) -> Result<Result<W, Value>, String>,
    mut rewind: impl FnMut(&mut S) -> Result<(), String>,
) -> Result<(W, Vec<Value>), String> {
    let mut discarded = Vec::new();
    loop {
        let mut moved = None;
        let mut warmed = 0;
        while warmed < warm || moved.is_some() {
            if warmed == STEADY_ATTEMPTS * warm {
                return Err(format!(
                    "the expert bank was not at rest after {warmed} warm decode steps: {}",
                    moved.unwrap_or(Value::Null)
                ));
            }
            if warmed > 0 && warmed % warm == 0 {
                rewind(decode)?;
            }
            moved = step(decode)?;
            warmed += 1;
        }
        if warmed > warm {
            rewind(decode)?;
            for _ in 0..warm {
                moved = step(decode)?;
            }
        }
        let outcome = match moved {
            Some(moved) => Err(json!({ "warm": moved })),
            None => window(decode)?,
        };
        match outcome {
            Ok(window) => return Ok((window, discarded)),
            Err(moved) => discarded.push(moved),
        }
        if discarded.len() == STEADY_ATTEMPTS {
            return Err(format!(
                "no decode window ran with the expert bank at rest in {STEADY_ATTEMPTS} attempts: {}",
                Value::from(discarded)
            ));
        }
        rewind(decode)?;
    }
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

    /// A decode of a history of `history` tokens, `warm` warm steps and
    /// windows of `steps`, whose load holds `context` rows; its bank moves
    /// across its first `unrested` steps and every window (`never`), or
    /// none after them.
    struct Fake {
        history: usize,
        steps: usize,
        context: usize,
        position: usize,
        deepest: usize,
        ran: usize,
        unrested: usize,
        never: bool,
    }

    impl Fake {
        fn advance(&mut self, rows: usize) -> Result<(), String> {
            self.position += rows;
            self.deepest = self.deepest.max(self.position);
            match self.position <= self.context {
                true => Ok(()),
                false => Err(format!("advance exceeds context capacity: {} of {}", self.position, self.context)),
            }
        }
    }

    fn steady(fake: &mut Fake, warm: usize) -> Result<(usize, Vec<Value>), String> {
        let moved = json!({ "slots": [180, 181] });
        steady_window(
            fake,
            warm,
            |fake| {
                fake.advance(1)?;
                fake.ran += 1;
                Ok((fake.ran <= fake.unrested).then(|| moved.clone()))
            },
            |fake| {
                fake.advance(fake.steps)?;
                fake.ran += fake.steps;
                Ok(match fake.never {
                    true => Err(json!({ "step": 0, "moved": moved.clone() })),
                    false => Ok(fake.position - fake.history),
                })
            },
            |fake| {
                fake.position = fake.history;
                Ok(())
            },
        )
    }

    /// A decode whose bank moves across every window, or rests only after
    /// a long warm-up, runs no deeper than its histories, its warm steps
    /// and its window, the context `time` sizes its load at: every
    /// discarded attempt, and every warm-up past its warm steps, begins
    /// again from the histories.
    #[test]
    fn a_decode_whose_bank_does_not_rest_stays_within_its_context() {
        for (warm, steps) in [(8, 48), (0, 2), (120, 200)] {
            let context = 4096 + decode_context_rows(warm, steps, 0);
            let fake = |unrested, never| Fake { history: 4096, steps, context, position: 4096, deepest: 4096, ran: 0, unrested, never };
            // Every window moved across: refused once its attempts are spent.
            let mut never = fake(0, true);
            let refused = steady(&mut never, warm);
            assert!(
                refused.as_ref().is_err_and(|error| error.starts_with("no decode window ran")),
                "{warm} warm + {steps} steps: {refused:?}"
            );
            assert!(never.deepest <= context, "{warm} warm + {steps} steps reached {} of {context}", never.deepest);
            // A warm-up three times the cell's: its window follows `warm`
            // steps from the histories, at rest.
            let mut slow = fake(3 * warm, false);
            let (depth, discarded) = steady(&mut slow, warm).unwrap();
            assert_eq!((depth, discarded.len()), (warm + steps, 0), "{warm} warm + {steps} steps");
            assert!(slow.deepest <= context);
        }
        // A bank that never rests in a warm-up is refused within it.
        let mut moving = Fake { history: 4096, steps: 48, context: 4096 + decode_context_rows(8, 48, 0), position: 4096, deepest: 4096, ran: 0, unrested: usize::MAX, never: false };
        let refused = steady(&mut moving, 8);
        assert!(refused.is_err_and(|error| error.starts_with("the expert bank was not at rest after 32")));
        assert!(moving.deepest <= 4096 + 8);
    }
}
