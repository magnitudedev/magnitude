//! Operational wall-time accounting. Estimates admit work, never prove a
//! deadline or predict an unmeasured candidate's performance.
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lane {
    Discovery,
    Refinement,
}

#[derive(Clone, Debug, Default)]
pub struct FeedbackCosts {
    pub domain: Duration,
    pub native_and_executable: Duration,
    pub observation: Duration,
    pub input_and_admission: Duration,
    pub reference_and_checks: Duration,
    pub publication: Duration,
    pub discovery: Duration,
    pub refinement: Duration,
}

pub(super) struct Ledger {
    started: Option<Instant>,
    allowance: Duration,
    previous_elapsed: Duration,
    publication_reserve: Duration,
    pub costs: FeedbackCosts,
}
impl Ledger {
    pub fn new(started: Instant, allowance: Duration) -> Self {
        Self {
            started: Some(started),
            allowance,
            previous_elapsed: Duration::ZERO,
            publication_reserve: Duration::ZERO,
            costs: FeedbackCosts::default(),
        }
    }
    pub fn elapsed(&self) -> Duration {
        self.previous_elapsed
            + self
                .started
                .map(|start| start.elapsed())
                .unwrap_or_default()
    }
    pub fn remaining(&self) -> Duration {
        self.allowance
            .saturating_sub(self.elapsed())
            .saturating_sub(self.publication_reserve)
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.started.map(|start| {
            start
                .checked_add(
                    self.allowance
                        .saturating_sub(self.previous_elapsed)
                        .saturating_sub(self.publication_reserve),
                )
                .expect("campaign deadline exceeds monotonic clock range")
        })
    }
    pub fn allows(&self, estimate: Option<Duration>) -> bool {
        let remaining = self.remaining();
        !remaining.is_zero() && estimate.is_none_or(|cost| cost < remaining)
    }
    pub fn next_lane(&self, discovery: bool, refinement: bool) -> Option<Lane> {
        match (discovery, refinement) {
            (false, false) => None,
            (true, false) => Some(Lane::Discovery),
            (false, true) => Some(Lane::Refinement),
            (true, true) if self.costs.discovery <= self.costs.refinement => Some(Lane::Discovery),
            (true, true) => Some(Lane::Refinement),
        }
    }
    pub fn charge(&mut self, lane: Lane, elapsed: Duration) {
        match lane {
            Lane::Discovery => self.costs.discovery += elapsed,
            Lane::Refinement => self.costs.refinement += elapsed,
        }
    }
    pub fn published(&mut self, elapsed: Duration) {
        self.costs.publication += elapsed;
        self.publication_reserve = self.publication_reserve.max(elapsed.saturating_mul(2));
    }
    /// Time between explicit preparation calls is not charged. The next call
    /// starts its clock before compatibility checks, preserving prior spending.
    pub fn resume(&mut self, additional: Duration) {
        self.allowance += additional;
        assert!(
            self.started.is_none(),
            "only suspended preparation can resume"
        );
        self.started = Some(Instant::now());
    }
    pub fn suspend(&mut self) {
        self.previous_elapsed += self
            .started
            .take()
            .expect("preparation already suspended")
            .elapsed();
    }
}

#[derive(Default)]
pub(super) struct ObservedCost {
    samples: Vec<Duration>,
}
impl ObservedCost {
    pub fn record(&mut self, elapsed: Duration) {
        if self.samples.len() == 8 {
            self.samples.remove(0);
        }
        self.samples.push(elapsed);
    }
    pub fn admission(&self) -> Option<Duration> {
        self.samples.iter().max().map(|cost| cost.saturating_mul(2))
    }
    pub fn ordering(&self) -> Option<Duration> {
        let mut samples = self.samples.clone();
        samples.sort();
        samples.get(samples.len() / 2).copied()
    }
}

/// Intents keep their age until consumed. A round is filled only after its
/// previous window drains; cheap newly generated work cannot displace it.
pub(super) struct IntentWindow<I> {
    pending: Vec<(u64, I)>,
    next_age: u64,
    deferred: Vec<(u64, I)>,
    resumed: std::collections::VecDeque<(u64, I)>,
}
impl<I> Default for IntentWindow<I> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            next_age: 0,
            deferred: Vec::new(),
            resumed: std::collections::VecDeque::new(),
        }
    }
}
impl<I> IntentWindow<I> {
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
    pub fn len(&self) -> usize {
        self.pending.len()
    }
    pub fn push(&mut self, intent: I) {
        assert!(self.pending.len() < 8);
        self.pending.push((self.next_age, intent));
        self.next_age += 1;
    }
    pub fn iter(&self) -> impl Iterator<Item = (usize, u64, &I)> {
        self.pending
            .iter()
            .enumerate()
            .map(|(i, (age, intent))| (i, *age, intent))
    }
    pub fn remove(&mut self, index: usize) -> I {
        self.pending.remove(index).1
    }
    pub fn defer(&mut self, index: usize) {
        self.deferred.push(self.pending.remove(index));
    }
    pub fn resume(&mut self) {
        self.deferred.append(&mut self.pending);
        self.deferred.extend(self.resumed.drain(..));
        self.deferred.sort_by_key(|(age, _)| *age);
        self.resumed.extend(self.deferred.drain(..));
        self.refill();
    }
    pub fn refill(&mut self) {
        if self.pending.is_empty() {
            for _ in 0..8 {
                let Some(intent) = self.resumed.pop_front() else {
                    break;
                };
                self.pending.push(intent);
            }
        }
    }
}

/// Combined native formation observations are keyed by the exact miss set.
/// We never apportion one combined duration among artifacts or extrapolate an
/// unmeasured unit's compile cost. Reuse can nevertheless reduce a union exactly.
#[derive(Default)]
pub(super) struct FormationCosts {
    records: Vec<(
        std::collections::HashSet<crate::realization::NativeArtifactRequestKey>,
        ObservedCost,
    )>,
}
impl FormationCosts {
    pub fn record(
        &mut self,
        missing: &[crate::realization::NativeArtifactRequestKey],
        elapsed: Duration,
    ) {
        let keys = missing
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        if let Some((_, cost)) = self.records.iter_mut().find(|(set, _)| *set == keys) {
            cost.record(elapsed);
        } else {
            let mut cost = ObservedCost::default();
            cost.record(elapsed);
            self.records.push((keys, cost));
        }
    }
    pub fn estimate(
        &self,
        missing: &[crate::realization::NativeArtifactRequestKey],
        admission: bool,
    ) -> Option<Duration> {
        let keys = missing
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        self.records
            .iter()
            .find(|(set, _)| *set == keys)
            .and_then(|(_, cost)| {
                if admission {
                    cost.admission()
                } else {
                    cost.ordering()
                }
            })
    }
}

/// Unknown work receives one exploratory turn, never an invented zero estimate.
/// Age breaks ties, and the bounded immutable round prevents starvation.
pub(super) fn order_cost(
    a: Option<Duration>,
    age_a: u64,
    b: Option<Duration>,
    age_b: u64,
) -> std::cmp::Ordering {
    a.is_some()
        .cmp(&b.is_some())
        .then(a.cmp(&b))
        .then(age_a.cmp(&age_b))
}

/// A finite sweep is affordable only with observed costs for every remaining
/// experiment and native union. Reserve a complete first confirmation stage
/// on both cases (2 arms * 8 observations * 2 cases) for every challenger.
pub(super) fn affordable_sweep(
    remaining: Duration,
    formation: Option<Duration>,
    screens: impl Iterator<Item = Option<Duration>>,
    confirmation: Option<Duration>,
) -> bool {
    let Some(mut total) = formation else {
        return false;
    };
    let Some(confirmation) = confirmation else {
        return false;
    };
    for screen in screens {
        let Some(screen) = screen else {
            return false;
        };
        total = total
            .saturating_add(screen)
            .saturating_add(confirmation.saturating_mul(32));
        if total >= remaining {
            return false;
        }
    }
    total < remaining
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intent_round_preserves_age_and_deferred_work_across_continuation() {
        let mut window = IntentWindow::default();
        for intent in 0..8 {
            window.push(intent);
        }
        window.defer(0);
        assert_eq!(window.remove(6), 7);
        window.resume();
        assert_eq!(
            window
                .iter()
                .map(|(_, age, intent)| (age, *intent))
                .collect::<Vec<_>>(),
            (0..7).map(|n| (n, n)).collect::<Vec<_>>()
        );
        assert!(order_cost(None, 4, Some(Duration::from_nanos(1)), 0).is_lt());
        assert!(order_cost(
            Some(Duration::from_secs(3)),
            0,
            Some(Duration::from_secs(1)),
            1
        )
        .is_gt());
    }

    #[test]
    fn enumeration_requires_measured_setup_and_confirmation_reserve() {
        let millis = Duration::from_millis(1);
        assert!(!affordable_sweep(
            Duration::from_secs(10),
            None,
            [Some(millis)].into_iter(),
            Some(millis)
        ));
        assert!(!affordable_sweep(
            Duration::from_secs(10),
            Some(millis),
            [None].into_iter(),
            Some(millis)
        ));
        assert!(!affordable_sweep(
            Duration::from_millis(33),
            Some(millis),
            [Some(millis)].into_iter(),
            Some(millis)
        ));
        assert!(affordable_sweep(
            Duration::from_millis(35),
            Some(millis),
            [Some(millis)].into_iter(),
            Some(millis)
        ));
        assert!(!affordable_sweep(
            Duration::from_millis(35),
            Some(millis),
            [Some(millis), Some(millis)].into_iter(),
            Some(millis)
        ));
    }

    #[test]
    fn lane_share_tracks_time_instead_of_experiment_count() {
        let mut ledger = Ledger::new(Instant::now(), Duration::from_secs(60));
        assert_eq!(ledger.next_lane(true, true), Some(Lane::Discovery));
        ledger.charge(Lane::Discovery, Duration::from_secs(5));
        for _ in 0..5 {
            assert_eq!(ledger.next_lane(true, true), Some(Lane::Refinement));
            ledger.charge(Lane::Refinement, Duration::from_secs(1));
        }
        assert_eq!(ledger.next_lane(true, true), Some(Lane::Discovery));
        assert_eq!(ledger.next_lane(false, true), Some(Lane::Refinement));
        assert_eq!(ledger.next_lane(false, false), None);
    }
    #[test]
    fn unknown_cost_is_distinct_and_expensive_measurements_affect_admission() {
        let mut cost = ObservedCost::default();
        assert_eq!(cost.admission(), None);
        cost.record(Duration::from_secs(2));
        cost.record(Duration::from_secs(10));
        cost.record(Duration::from_secs(1));
        assert_eq!(cost.admission(), Some(Duration::from_secs(20)));
        assert_eq!(cost.ordering(), Some(Duration::from_secs(2)));
        let ledger = Ledger::new(Instant::now(), Duration::from_secs(5));
        assert!(ledger.allows(None));
        assert!(!ledger.allows(cost.admission()));
    }
}
