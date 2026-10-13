//! Load-owned invocation forecasts, independent of weight ownership: the
//! launches each graph of the load makes for every term of the tuning
//! objective ([`crate::native::tuning::workload`]), each launch with its rows,
//! the sequences they belong to and the history they see. Numerical support
//! is supplied independently by each tuning case.

use crate::PlannedMethod;
use crate::native::tuning_workload::{Phase, Term};
use std::collections::BTreeMap;

/// One kind of launch of one objective term: the term's index, the launch's
/// rows, how many sequences those rows belong to, and the history each of
/// them sees. Eight rows of one sequence (a drafter's verification) and eight
/// sequences of one row (a decode step of eight requests) are one row count
/// and two different launches: each row of the second reads a history of its
/// own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Launch {
    pub term: usize,
    pub rows: u64,
    pub sequences: u64,
    pub history: u64,
}

/// Launches by kind: how many of each one term's measurement makes.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct InvocationCounts(BTreeMap<Launch, f64>);

impl InvocationCounts {
    /// `count` launches of `launch`.
    pub fn launch(&mut self, launch: Launch, count: f64) {
        assert!(launch.rows > 0 && count.is_finite() && count >= 0.);
        assert!((1..=launch.rows).contains(&launch.sequences));
        *self.0.entry(launch).or_default() += count;
    }
    /// The launches of `rows` rows, of any term, sequences or history.
    #[cfg(test)]
    pub fn count(&self, rows: u64) -> f64 {
        self.0
            .iter()
            .filter(|(launch, _)| launch.rows == rows)
            .map(|(_, count)| count)
            .sum()
    }
    /// Every kind of launch and how many of it.
    pub fn launches(&self) -> impl Iterator<Item = (Launch, f64)> + '_ {
        self.0.iter().map(|(launch, count)| (*launch, *count))
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
    /// The launches as a unit makes them: each at the rows of the
    /// representative that serves it, in at most that many sequences, and
    /// none that `representative` does not serve.
    pub fn represented(&self, mut representative: impl FnMut(&Launch) -> Option<u64>) -> Self {
        let mut result = Self::default();
        for (launch, count) in &self.0 {
            if let Some(rows) = representative(launch) {
                result.launch(
                    Launch {
                        rows,
                        sequences: launch.sequences.min(rows),
                        ..*launch
                    },
                    *count,
                );
            }
        }
        result
    }
}

/// The launches each of the load's graphs makes for every objective term.
#[derive(Clone, Debug)]
pub(crate) struct GraphInvocationDemand {
    /// The objective's terms, by index.
    pub terms: Vec<Term>,
    pub target: InvocationCounts,
    /// A draft head's decoder steps: one per proposal it drafts.
    pub head: InvocationCounts,
    /// A draft head priming its history with every prompt row the target
    /// prefills.
    pub head_prefix: InvocationCounts,
    /// A separate draft model's launches: its prefill and its blocks.
    pub draft: InvocationCounts,
}

impl GraphInvocationDemand {
    /// The launches of every term of `terms` for a load that decodes by
    /// `method` and launches at most `max_rows` rows. A prompt is
    /// prefilled in launches of the largest row class, the remainder in
    /// one launch of its own; a decode step is one target row per sequence
    /// without drafting or at a width of none, and otherwise the drafter's
    /// launches for the term's width and the target's verification of the
    /// drafts and their anchor.
    pub fn new(
        method: PlannedMethod,
        terms: Vec<Term>,
        draft_definition: Option<&magnitude_family_contracts::DraftDefinition>,
        max_rows: u64,
    ) -> Self {
        let mut demand = Self {
            terms: Vec::new(),
            target: InvocationCounts::default(),
            head: InvocationCounts::default(),
            head_prefix: InvocationCounts::default(),
            draft: InvocationCounts::default(),
        };
        let chunk = crate::TUNING_ROWS
            .into_iter()
            .filter(|rows| *rows <= max_rows)
            .max()
            .unwrap_or(1);
        for (term, spec) in terms.iter().enumerate() {
            let metric = spec.metric;
            let at = |rows: u64, sequences: u64, history: u64| Launch {
                term,
                rows,
                sequences,
                history,
            };
            match metric.phase {
                Phase::Prefill => {
                    // Each sequence's prompt is prefilled on its own.
                    let mut history = 0;
                    while history < metric.context {
                        let rows = chunk.min(metric.context - history);
                        let launch = at(rows, 1, history);
                        let count = metric.sequences as f64;
                        demand.target.launch(launch, count);
                        demand.head_prefix.launch(launch, count);
                        demand.draft.launch(launch, count);
                        history += rows;
                    }
                }
                Phase::Decode => {
                    let (sequences, history) = (metric.sequences, metric.context);
                    match (method, spec.width) {
                        (PlannedMethod::Plain, _) | (_, None | Some(0)) => {
                            demand.target.launch(at(sequences, sequences, history), 1.);
                        }
                        (PlannedMethod::Mtp { .. }, Some(width)) => {
                            demand
                                .target
                                .launch(at((width + 1) * sequences, sequences, history), 1.);
                            demand
                                .head
                                .launch(at(sequences, sequences, history), width as f64);
                        }
                        (PlannedMethod::DFlash { proposals, .. }, Some(width)) => {
                            let block = draft_definition
                                .expect("a planned block drafter has its admitted definition")
                                .block_rows(width);
                            let verified = width.min(u64::from(proposals));
                            demand
                                .target
                                .launch(at((verified + 1) * sequences, sequences, history), 1.);
                            demand
                                .draft
                                .launch(at(block * sequences, sequences, history), 1.);
                        }
                    }
                }
            }
        }
        demand.terms = terms;
        demand
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::tuning_workload::Metric;

    fn term(phase: Phase, context: u64, width: Option<u64>) -> Term {
        Term {
            metric: Metric {
                phase,
                context,
                sequences: 1,
            },
            width,
            weight: 1.,
        }
    }

    /// A prompt is prefilled in launches of the largest class, each at the
    /// history the ones before it wrote, and its remainder in one launch.
    #[test]
    fn a_prompt_is_prefilled_in_chunks_at_their_histories() {
        let demand = GraphInvocationDemand::new(
            PlannedMethod::Plain,
            vec![term(Phase::Prefill, 1300, None)],
            None,
            512,
        );
        let launches = demand.target.launches().collect::<Vec<_>>();
        assert_eq!(
            launches
                .iter()
                .map(|(launch, count)| (launch.rows, launch.history, *count))
                .collect::<Vec<_>>(),
            [(276, 1024, 1.), (512, 0, 1.), (512, 512, 1.)]
        );
        assert_eq!(demand.head_prefix, demand.target);
    }

    /// A drafted decode step at width w is the head's w proposals and the
    /// target's verification of w + 1 rows, both at the step's history.
    #[test]
    fn a_drafted_step_is_its_proposals_and_their_verification() {
        let method = PlannedMethod::Mtp {
            greedy_proposals: 3,
            sampled_proposals: 1,
        };
        let demand =
            GraphInvocationDemand::new(method, vec![term(Phase::Decode, 4096, Some(3))], None, 512);
        assert_eq!(demand.target.count(4), 1.);
        assert_eq!(demand.head.count(1), 3.);
        assert!(
            demand
                .target
                .launches()
                .chain(demand.head.launches())
                .all(|(launch, _)| launch.history == 4096)
        );
        // A unit represents launches at its own rows.
        let represented = demand
            .target
            .represented(|launch| (launch.rows <= 8).then_some(8));
        assert_eq!(represented.count(8), 1.);
    }
}
