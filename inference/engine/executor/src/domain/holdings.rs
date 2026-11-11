//! What the release ladder weighs the bound optional components by: each
//! one's worth per byte, from figures its owner measures; none reads a
//! model's recorded data.

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
    fn rate(&self) -> f64 {
        let with = self.drafted_step_seconds / (1.0 + self.accepted_per_step);
        (self.tokens_per_s * (self.plain_step_seconds - with)).max(0.0)
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

/// The bound head (drafter) of `bytes`: worth what drafting saves
/// ([`Drafting`]), and at least the load each of `uses_per_s` requests
/// would pay again without it, `move_seconds` each. A head whose drafting
/// is not priced yet is unpriced ([`OptionalHolding::unpriced`]): what it
/// saves is unknown, not nothing.
pub(crate) fn drafter(bytes: u64, drafting: Drafting, uses_per_s: f64, move_seconds: f64) -> OptionalHolding {
    let worth = drafting.rate().max(uses_per_s * move_seconds);
    OptionalHolding::priced(OptionalComponent::Drafter, worth, bytes)
}

/// The bound vision projector of `bytes`: worth the load time each use
/// would pay again without it (`load_seconds`), `uses_per_s` times a
/// second.
pub(crate) fn vision(bytes: u64, uses_per_s: f64, load_seconds: f64) -> OptionalHolding {
    OptionalHolding::priced(OptionalComponent::Vision, uses_per_s * load_seconds, bytes)
}

/// A bound optional component as the release ladder weighs it: its worth
/// per byte where it is known, none where a figure its worth is made of is
/// not priced yet (the head's drafting before the load's decode is priced,
/// the serving rates, the model store's read line). An unknown worth is no
/// low worth: the ladder takes an unpriced one only once no priced one is
/// left ([`cheapest`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OptionalHolding {
    component: OptionalComponent,
    per_byte: Option<f64>,
}

impl OptionalHolding {
    pub(crate) fn unpriced(component: OptionalComponent) -> Self {
        Self { component, per_byte: None }
    }

    /// Worth `seconds_per_second` over its `bytes`.
    fn priced(component: OptionalComponent, seconds_per_second: f64, bytes: u64) -> Self {
        Self { component, per_byte: Some(seconds_per_second / bytes.max(1) as f64) }
    }
}

/// The optional component the release ladder releases from `holdings`, and
/// its rate per byte where known: the priced one worth least per byte, an
/// unpriced one only where none is priced.
pub(crate) fn cheapest(holdings: &[OptionalHolding]) -> Option<(OptionalComponent, Option<f64>)> {
    holdings
        .iter()
        .map(|holding| (holding.component, holding.per_byte))
        .min_by(|a, b| match (a.1, b.1) {
            (Some(a), Some(b)) => a.total_cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn per_byte(holding: OptionalHolding) -> f64 {
        holding.per_byte.expect("a priced component")
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
        assert!((per_byte(drafter(100, drafting, 0.0, 0.5)) - 0.004).abs() < 1e-12);
        // Accepting nothing, drafting only costs: it is worth only the
        // loads its uses would pay again.
        let useless = drafter(100, Drafting { accepted_per_step: 0.0, ..drafting }, 0.1, 0.5);
        assert!((per_byte(useless) - 0.0005).abs() < 1e-12);
    }

    /// A head whose drafting is not priced yet is worth an unknown amount,
    /// not its import seconds: the release ladder releases it only once no
    /// priced component is left, whatever the priced ones are worth.
    #[test]
    fn an_unpriced_component_is_never_the_cheapest_to_release() {
        let head = OptionalHolding::unpriced(OptionalComponent::Drafter);
        // Vision worth far more per byte than any import of the head.
        let vision = vision(10, 100.0, 2.0);
        assert_eq!(cheapest(&[head, vision]), Some((OptionalComponent::Vision, Some(20.0))));
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

    /// Vision used once every 100 s that takes 2 s to load is worth 0.02 s
    /// a second over its bytes.
    #[test]
    fn vision_is_worth_the_loads_its_uses_would_pay() {
        assert!((per_byte(vision(100, 0.01, 2.0)) - 0.0002).abs() < 1e-12);
    }
}
