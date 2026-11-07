//! The curves of the holdings beside the bank that the heap prices
//! ([`super::value`]): the optional components, and the cached prefixes
//! the scheduler prices (`PrefixCache::held_steps`). Each takes what it is
//! valued by as figures its owner measures; none reads a model's recorded
//! data.

use super::value::{Curve, Holding, Rate, Step};
use super::OptionalComponent;

/// What drafting with the head does to decode, as the load measured it
/// (or, before it drafted, as [`prior_accepted`] and the step model
/// expect).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Drafting {
    /// Tokens decoded a second of serving.
    pub(crate) tokens_per_s: f64,
    /// A decode step's seconds without the head.
    pub(crate) plain_step_seconds: f64,
    /// A decode step's seconds drafting with it, the draft included.
    pub(crate) drafted_step_seconds: f64,
    /// Draft tokens accepted a step, beyond the step's own.
    pub(crate) accepted_per_step: f64,
}

impl Drafting {
    /// What the head saves a second: each token costs a plain step without
    /// it, and a drafted step over the step's `1 + accepted` tokens with
    /// it. A head that costs more than it saves is worth nothing, and is
    /// released first.
    fn rate(&self) -> Rate {
        let with = self.drafted_step_seconds / (1.0 + self.accepted_per_step);
        Rate::new((self.tokens_per_s * (self.plain_step_seconds - with)).max(0.0))
    }
}

/// The draft tokens a step is expected to accept before the load measured
/// any, from the head's declared draft depth alone: with no knowledge of
/// how often a draft token is accepted, its chance `p` is uniform on
/// `[0, 1]`, and a chain accepts its `k`-th token with chance `p^k`, whose
/// mean is `1 / (k + 1)`.
pub(crate) fn prior_accepted(depth: u32) -> f64 {
    (1..=depth).map(|k| 1.0 / f64::from(k + 1)).sum()
}

/// A priced optional component of one step: held where it is bound, else
/// offered.
fn optional(component: OptionalComponent, bound: bool, step: Step) -> OptionalHolding {
    let (held, offered) = if bound { (vec![step], vec![]) } else { (vec![], vec![step]) };
    let curve = Curve { holding: Holding::Optional(component), borrows: false, held, offered };
    OptionalHolding { component, curve: Some(curve) }
}

/// The head (drafter): its `bytes` bound or not, worth what drafting saves
/// ([`Drafting`]), and at least the load each of `uses_per_s` requests
/// would pay again without it; binding or releasing it costs
/// `move_seconds`. A head whose drafting is not priced yet has no curve
/// ([`OptionalHolding::unpriced`]): what it saves is unknown, not nothing.
pub(crate) fn drafter(bytes: u64, bound: bool, drafting: Drafting, uses_per_s: f64, move_seconds: f64) -> OptionalHolding {
    let worth = drafting.rate().get().max(uses_per_s * move_seconds);
    optional(OptionalComponent::Drafter, bound, Step { bytes, worth: Rate::new(worth), move_seconds })
}

/// The vision projector: its `bytes`, worth the load time each use would
/// pay again without it (`load_seconds`), `uses_per_s` times a second.
pub(crate) fn vision(bytes: u64, bound: bool, uses_per_s: f64, load_seconds: f64) -> OptionalHolding {
    let step = Step { bytes, worth: Rate::new(uses_per_s * load_seconds), move_seconds: load_seconds };
    optional(OptionalComponent::Vision, bound, step)
}

/// A bound optional component as the heap prices it: its curve where what
/// it is worth is known, none where a figure its worth is made of is not
/// priced yet (the head's drafting before the load's decode is priced, the
/// serving rates, the model store's read line). An unknown worth is no low
/// worth: a settle moves only the priced components ([`Self::curve`]), and
/// the release ladder takes an unpriced one only once no priced one is left
/// ([`cheapest`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OptionalHolding {
    component: OptionalComponent,
    curve: Option<Curve<Holding>>,
}

impl OptionalHolding {
    pub(crate) fn unpriced(component: OptionalComponent) -> Self {
        Self { component, curve: None }
    }

    /// Its curve, where its worth is known: what a settle moves.
    pub(crate) fn curve(self) -> Option<Curve<Holding>> {
        self.curve
    }

    /// Its held rate per byte; none where its worth is unknown.
    fn per_byte(&self) -> Option<f64> {
        self.curve.as_ref().and_then(|curve| curve.held.first()).map(Step::per_byte)
    }
}

/// The optional component the release ladder releases from `holdings`, and
/// its rate per byte where known: the priced one worth least per byte, an
/// unpriced one only where none is priced.
pub(crate) fn cheapest(holdings: &[OptionalHolding]) -> Option<(OptionalComponent, Option<f64>)> {
    holdings
        .iter()
        .map(|holding| (holding.component, holding.per_byte()))
        .min_by(|a, b| match (a.1, b.1) {
            (Some(a), Some(b)) => a.total_cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        })
}

/// Cached prefixes as the scheduler prices them, `(bytes, seconds a
/// second)` each, the least valuable first: held, released in that order.
pub(crate) fn prefixes(entries: impl IntoIterator<Item = (u64, f64)>) -> Curve<Holding> {
    let held = entries
        .into_iter()
        .map(|(bytes, seconds_per_second)| Step {
            bytes,
            worth: Rate::new(seconds_per_second),
            move_seconds: 0.0,
        })
        .collect();
    Curve { holding: Holding::Prefixes, borrows: false, held, offered: vec![] }
}

#[cfg(test)]
mod tests {
    use super::super::value::{settle, Supply};
    use super::*;
    use std::time::Duration;

    fn rate_of(holding: &OptionalHolding) -> f64 {
        let curve = holding.clone().curve().expect("a priced component");
        curve.held.first().or(curve.offered.first()).unwrap().worth.get()
    }

    fn curve(holding: OptionalHolding) -> Curve<Holding> {
        holding.curve().expect("a priced component")
    }

    /// 20 tokens a second, a 50 ms plain step, a 60 ms drafted step that
    /// accepts one draft token: each token costs 30 ms instead of 50, so
    /// the head saves 0.4 s a second.
    #[test]
    fn a_drafter_is_worth_the_step_time_its_accepted_tokens_save() {
        let drafting = Drafting {
            tokens_per_s: 20.0,
            plain_step_seconds: 0.05,
            drafted_step_seconds: 0.06,
            accepted_per_step: 1.0,
        };
        let bound = drafter(1 << 30, true, drafting, 0.0, 0.5);
        assert!((rate_of(&bound) - 0.4).abs() < 1e-12);
        let bound = curve(bound);
        assert_eq!((bound.held.len(), bound.offered.len()), (1, 0));
        // Accepting nothing, drafting only costs: it is worth only the
        // loads its uses would pay again.
        let useless = drafter(1 << 30, false, Drafting { accepted_per_step: 0.0, ..drafting }, 0.1, 0.5);
        assert!((rate_of(&useless) - 0.05).abs() < 1e-12);
        let useless = curve(useless);
        assert_eq!((useless.held.len(), useless.offered.len()), (0, 1));
    }

    /// A head whose drafting is not priced yet is worth an unknown amount,
    /// not its import seconds: a settle owed more than every priced holding
    /// gives never releases it, and the release ladder releases it only
    /// once no priced component is left, whatever the priced ones are worth.
    #[test]
    fn an_unpriced_component_is_never_the_cheapest_to_release() {
        let head = OptionalHolding::unpriced(OptionalComponent::Drafter);
        // Vision worth far more per byte than any import of the head.
        let vision = vision(10, true, 100.0, 2.0);
        let curves = [prefixes([(10, 0.001)])]
            .into_iter()
            .chain([head.clone(), vision.clone()].into_iter().filter_map(OptionalHolding::curve))
            .collect::<Vec<_>>();
        let settled = settle(&curves, owed(1 << 20));
        assert!(
            settled.released.iter().all(|(holding, _)| *holding != Holding::Optional(OptionalComponent::Drafter)),
            "an unpriced head is no step a settle moves: {:?}",
            settled.released
        );
        assert_eq!(
            cheapest(&[head.clone(), vision]),
            Some((OptionalComponent::Vision, Some(20.0)))
        );
        assert_eq!(cheapest(&[head]), Some((OptionalComponent::Drafter, None)));
        assert_eq!(cheapest(&[]), None);
    }

    /// A chain of depth 1 accepts half a token; of depth 3, 1/2 + 1/3 + 1/4.
    #[test]
    fn the_acceptance_prior_follows_the_declared_depth_alone() {
        assert!((prior_accepted(1) - 0.5).abs() < 1e-12);
        assert!((prior_accepted(3) - (0.5 + 1.0 / 3.0 + 0.25)).abs() < 1e-12);
        assert_eq!(prior_accepted(0), 0.0);
    }

    /// Vision used once every 100 s that takes 2 s to load is worth 0.02.
    #[test]
    fn vision_is_worth_the_loads_its_uses_would_pay() {
        assert!((rate_of(&vision(1 << 28, true, 0.01, 2.0)) - 0.02).abs() < 1e-12);
    }

    fn owed(bytes: u64) -> Supply {
        Supply { free: 0, owed: bytes, borrowable: 0, borrow_price: 0.0, ceiling: u64::MAX, horizon: Duration::from_secs(60) }
    }

    /// What is owed comes from the holdings worth least per byte, of any
    /// kind: the prefix that saves least, then vision used once every 100 s
    /// (0.002 s/s a byte), before the prefix worth more.
    #[test]
    fn releases_take_the_least_valuable_per_byte_of_any_kind() {
        let curves = [prefixes([(10, 0.001), (10, 0.1)]), curve(vision(10, true, 0.01, 2.0))];
        let settled = settle(&curves, owed(20));
        assert_eq!(
            settled.released,
            vec![(Holding::Prefixes, 1), (Holding::Optional(OptionalComponent::Vision), 1)]
        );
    }

    /// A drafter worth more per byte than the bank's tail slots is kept
    /// over them: an ask takes the slots, and the head only once they are
    /// gone.
    #[test]
    fn a_drafter_worth_more_than_bank_slots_is_kept_over_them() {
        let drafting = Drafting { tokens_per_s: 20.0, plain_step_seconds: 0.05, drafted_step_seconds: 0.06, accepted_per_step: 1.0 };
        // 0.4 s/s over 100 bytes; two slots of 50 bytes at 0.01 s/s each.
        let head = curve(drafter(100, true, drafting, 0.0, 0.5));
        let slot = Step { bytes: 50, worth: Rate::new(0.01), move_seconds: 0.0 };
        let experts = Holding::Bank(crate::BankHolding::Experts);
        let bank = Curve { holding: experts, borrows: true, held: vec![slot, slot], offered: vec![] };
        let settled = settle(&[head.clone(), bank.clone()], owed(100));
        assert_eq!(settled.released, vec![(experts, 2)]);
        let settled = settle(&[head, bank], owed(150));
        assert_eq!(settled.released, vec![(Holding::Optional(OptionalComponent::Drafter), 1), (experts, 2)]);
    }
}
