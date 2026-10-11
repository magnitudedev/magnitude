//! Load-owned invocation forecasts, independent of weight ownership.
//!
//! A forecast's base counts are the served workload's launches per turn. A
//! speculative method's counts describe a bounded envelope over the widths a
//! load prepares, not how often each runs. Numerical support is supplied
//! independently by each tuning case.

use crate::PlannedMethod;
use std::collections::BTreeMap;

/// Launches per turn, by the launch's rows and by how many sequences those
/// rows belong to. Eight rows of one sequence (a drafter's verification) and
/// eight sequences of one row (a decode step of eight requests) are one row
/// count and two different launches: each row of the second reads a history
/// of its own.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct InvocationCounts(BTreeMap<(u64, u64), f64>);

impl InvocationCounts {
    /// Launches whose rows are all one sequence's.
    pub fn new(rows: impl IntoIterator<Item = (u64, f64)>) -> Self {
        Self::across(rows.into_iter().map(|(rows, count)| (rows, 1, count)))
    }
    /// Launches of `rows` rows in `sequences` sequences.
    pub fn across(launches: impl IntoIterator<Item = (u64, u64, f64)>) -> Self {
        let mut result = Self::default();
        for (rows, sequences, count) in launches {
            assert!(rows > 0 && count.is_finite() && count >= 0.);
            assert!((1..=rows).contains(&sequences));
            *result.0.entry((rows, sequences)).or_default() += count;
        }
        result
    }
    /// The launches of `rows` rows, in however many sequences.
    pub fn count(&self, rows: u64) -> f64 {
        self.sequences(rows).map(|(_, count)| count).sum()
    }
    /// The launches of `rows` rows by the sequences their rows belong to.
    pub fn sequences(&self, rows: u64) -> impl Iterator<Item = (u64, f64)> + '_ {
        self.0
            .range((rows, 0)..=(rows, u64::MAX))
            .map(|((_, sequences), count)| (*sequences, *count))
    }
    /// The launches by rows, in however many sequences.
    pub fn iter(&self) -> impl Iterator<Item = (u64, f64)> + '_ {
        let mut rows = self.0.keys().map(|(rows, _)| *rows).collect::<Vec<_>>();
        rows.dedup();
        rows.into_iter().map(|rows| (rows, self.count(rows)))
    }
    pub fn add(&mut self, other: &Self) {
        for (launch, count) in &other.0 {
            *self.0.entry(*launch).or_default() += count;
        }
    }
    pub fn scaled(&self, repetitions: usize) -> Self {
        Self(
            self.0
                .iter()
                .map(|(launch, count)| (*launch, count * repetitions as f64))
                .collect(),
        )
    }
}

/// Additive calls and alternative-width envelopes retain different meanings
/// until a tuning case projects them onto its owned representatives.
#[derive(Clone, Debug, Default)]
pub(crate) struct InvocationForecast {
    pub base: InvocationCounts,
    alternatives: Vec<InvocationCounts>,
}

impl InvocationForecast {
    pub fn new(base: InvocationCounts) -> Self {
        Self {
            base,
            alternatives: Vec::new(),
        }
    }
    pub fn add(&mut self, other: &Self) {
        self.base.add(&other.base);
        self.alternatives.extend(other.alternatives.iter().cloned());
    }
    pub fn represented(
        &self,
        mut representative: impl FnMut(u64) -> Option<u64>,
    ) -> InvocationCounts {
        // A launch represented at fewer rows than it has is in at most that
        // many sequences.
        let mut result = InvocationCounts::default();
        for ((rows, sequences), count) in &self.base.0 {
            if let Some(at) = representative(*rows) {
                result.add(&InvocationCounts::across([(at, (*sequences).min(at), *count)]));
            }
        }
        for alternative in &self.alternatives {
            // Of the alternative widths a representative stands for, the
            // most frequent one is its envelope, with that width's sequences.
            let mut envelope = BTreeMap::<u64, (u64, f64)>::new();
            for ((rows, sequences), count) in &alternative.0 {
                if let Some(at) = representative(*rows) {
                    let most = envelope.entry(at).or_insert((1, 0.));
                    if *count > most.1 {
                        *most = ((*sequences).min(at), *count);
                    }
                }
            }
            result.add(&InvocationCounts::across(
                envelope
                    .into_iter()
                    .map(|(at, (sequences, count))| (at, sequences, count)),
            ));
        }
        result
    }
}

/// Forecasts at actual graph registrations, independent of their weight scopes.
#[derive(Clone, Debug)]
pub(crate) struct GraphInvocationDemand {
    pub target: InvocationForecast,
    pub head: InvocationForecast,
    pub head_prefix: InvocationCounts,
    pub draft: InvocationForecast,
}

impl GraphInvocationDemand {
    /// `base` is the workload's target launches per turn at every width,
    /// which every method keeps; `rounds` bounds how often a turn takes one
    /// speculative action, whatever width the workload decodes at.
    pub fn new(
        method: PlannedMethod,
        base: InvocationCounts,
        prefill: InvocationCounts,
        rounds: f64,
        slots: &[u64],
        draft_definition: Option<&magnitude_family_contracts::DraftDefinition>,
        max_rows: u64,
    ) -> Self {
        let mut target = InvocationForecast::new(base);
        let mut head = InvocationForecast::default();
        let mut draft = InvocationForecast::new(prefill.clone());
        let widths: Vec<u64> = match method {
            PlannedMethod::Plain => Vec::new(),
            PlannedMethod::Mtp {
                greedy_proposals,
                sampled_proposals,
            } => (1..=u64::from(greedy_proposals.max(sampled_proposals))).collect(),
            PlannedMethod::DFlash { .. } => {
                method.block_widths().into_iter().map(u64::from).collect()
            }
        };
        // Each slot is a sequence: a launch over `slots` of them has that
        // many sequences' rows, whatever its rows per slot.
        let mut verify = Vec::new();
        let mut generated_calls = Vec::new();
        for generated in widths {
            for &slots in slots {
                let block_rows = match method {
                    PlannedMethod::DFlash { .. } => draft_definition
                        .expect("a planned block drafter has its admitted definition")
                        .block_rows(generated),
                    _ => generated,
                };
                if matches!(method, PlannedMethod::DFlash { .. }) && block_rows * slots > max_rows {
                    continue;
                }
                let verified = match method {
                    PlannedMethod::DFlash { proposals, .. } => generated.min(u64::from(proposals)),
                    _ => generated,
                };
                verify.push(((verified + 1) * slots, slots, rounds));
                generated_calls.push(match method {
                    PlannedMethod::Mtp { .. } => (slots, slots, rounds * generated as f64),
                    PlannedMethod::DFlash { .. } => (block_rows * slots, slots, rounds),
                    PlannedMethod::Plain => unreachable!(),
                });
            }
        }
        // The widths of one method are alternatives: a turn takes one of
        // them, so each launch shape is counted at the most any width gives.
        let envelope = |launches: Vec<(u64, u64, f64)>| {
            let mut most = BTreeMap::<(u64, u64), f64>::new();
            for (rows, sequences, count) in launches {
                let entry = most.entry((rows, sequences)).or_default();
                *entry = entry.max(count);
            }
            InvocationCounts(most)
        };
        target.alternatives.push(envelope(verify));
        match method {
            PlannedMethod::Mtp { .. } => head.alternatives.push(envelope(generated_calls)),
            PlannedMethod::DFlash { .. } => draft.alternatives.push(envelope(generated_calls)),
            PlannedMethod::Plain => {}
        }
        Self {
            target,
            head,
            head_prefix: prefill,
            draft,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chained_action_counts_calls_without_weight_provenance() {
        let method = PlannedMethod::Mtp {
            greedy_proposals: 3,
            sampled_proposals: 1,
        };
        let demand = GraphInvocationDemand::new(
            method,
            InvocationCounts::new([(1, 1.), (512, 2.)]),
            InvocationCounts::new([(512, 2.)]),
            1.,
            &[1],
            None,
            512,
        );
        let head = demand.head.represented(Some);
        let target = demand.target.represented(Some);
        assert_eq!(head.count(1), 3.);
        assert_eq!(target.count(4), 1.);
        assert_eq!(target.count(1), 1.);
        assert_eq!(target.count(512), 2.);
        let mut same_case = target;
        same_case.add(&head);
        assert_eq!(same_case.count(1), 4.);
        assert_eq!(same_case.count(4), 1.);
    }
    #[test]
    fn alternatives_project_before_independent_calls_aggregate() {
        let mut a = InvocationForecast::default();
        a.alternatives
            .push(InvocationCounts::new([(8, 2.), (9, 1.)]));
        let mut b = InvocationForecast::default();
        b.alternatives.push(InvocationCounts::new([(9, 3.)]));
        a.add(&b);
        assert_eq!(a.represented(|_| Some(8)).count(8), 5.);
        a.base = InvocationCounts::new([(8, 7.)]);
        assert_eq!(a.represented(|_| Some(8)).count(8), 12.);
    }
}
