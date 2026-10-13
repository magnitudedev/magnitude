//! What tuning optimizes: the measurements the engine is held to, run the
//! way the engine runs them for this load. Nothing here estimates how the
//! engine is used; every launch follows from the load's own configuration
//! (its context limit, launch rows, drafting method and its widths, and
//! concurrency). Metric shares are explicit policy; normalization comes
//! from measured baseline times.
//!
//! **Metrics.** One per phase, context point and concurrency
//! ([`metrics`]):
//!
//! - `prefill(C)`: prefilling a prompt of C tokens as the engine schedules
//!   it, in launches of the largest row class at histories 0, `max_rows`,
//!   … and its remainder in one launch;
//! - `decode(C)`: one decode step with C tokens of history, as the load
//!   decodes: one target row per sequence without drafting; with drafting,
//!   the drafter's launches for a width and the target's verification of
//!   its drafts and their anchor.
//!
//! **Score.** Every metric counts by its relative change: the tuner
//! minimizes the weighted mean over the objective's terms of
//! `time / time at the defaults`. With drafting, the adaptive
//! controller chooses a verification width at run time among those the
//! load prepares, from their measured times, so each prepared width is a
//! term of its own within its decode metric, the metric's weight divided
//! among them: no width's share depends on a guess of the controller's
//! choice, and no acceptance rate enters (a term's tokens per step divide
//! its time and its time at the defaults alike).
//!
//! **Shares.** The baseline contribution: a unit's share is the weighted mean
//! over terms of the unit's fraction of the term's time at the defaults,
//! and a point's weight within a unit is the same quantity for the point
//! ([`super::Objective`]).
//!
//! Three choices are the user's, each in one place: the context points
//! ([`contexts`]), the concurrency ([`concurrency`]) and the metrics'
//! weights ([`metric_weight`]).

use super::TuningLimits;

/// What a metric measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Phase {
    /// Prefilling a prompt of the metric's context.
    Prefill,
    /// One decode step at the metric's context of history.
    Decode,
}

/// One measurement the engine is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Metric {
    pub phase: Phase,
    /// The prompt's tokens (prefill) or the history decoded at (decode).
    pub context: u64,
    /// The sequences served together, each with a context of its own.
    pub sequences: u64,
}

/// One term of the objective: a metric, or one prepared verification width
/// of a drafted decode metric, with its weight in the score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Term {
    pub metric: Metric,
    /// The drafter's width of a drafted decode term.
    pub width: Option<u64>,
    pub weight: f64,
}

/// Explicit short- and long-context workload policy, capped at the load's limit.
pub(crate) fn contexts(limits: TuningLimits) -> Vec<u64> {
    let mut contexts = [4096, 65536]
        .map(|context: u64| context.min(limits.context_tokens))
        .to_vec();
    contexts.dedup();
    contexts
}

/// The sequences the load's metrics serve together. A load declares no
/// concurrency it serves, only the most sequences one launch admits, which
/// bounds what it can serve rather than stating what it does: its metrics
/// are of one sequence.
pub(crate) fn concurrency(_limits: TuningLimits) -> Vec<u64> {
    vec![1]
}

/// A metric's weight in the score: every metric counts alike.
pub(crate) fn metric_weight(_metric: &Metric) -> f64 {
    1.
}

/// The load's metrics.
pub(crate) fn metrics(limits: TuningLimits) -> Vec<Metric> {
    let mut metrics = Vec::new();
    for phase in [Phase::Prefill, Phase::Decode] {
        for &sequences in &concurrency(limits) {
            for &context in &contexts(limits) {
                metrics.push(Metric {
                    phase,
                    context,
                    sequences,
                });
            }
        }
    }
    metrics
}

/// The objective's terms for a load whose drafter prepares `widths` (none
/// without drafting): every metric, a drafted decode metric once for each
/// width, the weights summing to one.
pub(crate) fn terms(limits: TuningLimits, widths: &[u64]) -> Vec<Term> {
    let metrics = metrics(limits);
    let total = metrics.iter().map(metric_weight).sum::<f64>();
    metrics
        .into_iter()
        .flat_map(|metric| {
            let weight = metric_weight(&metric) / total;
            let widths = match metric.phase {
                Phase::Decode if !widths.is_empty() => {
                    widths.iter().map(|width| Some(*width)).collect::<Vec<_>>()
                }
                _ => vec![None],
            };
            let each = weight / widths.len() as f64;
            widths.into_iter().map(move |width| Term {
                metric,
                width,
                weight: each,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: TuningLimits = TuningLimits {
        max_rows: 512,
        max_projected_rows: 8,
        context_tokens: 64256,
    };

    /// Prefill and decode at every context point to the limit; a drafted
    /// decode metric divides its weight among the prepared widths.
    #[test]
    fn every_metric_counts_alike_and_widths_share_theirs() {
        assert_eq!(contexts(LIMITS), [4096, 64256]);
        let short = TuningLimits {
            context_tokens: 2048,
            ..LIMITS
        };
        assert_eq!(contexts(short), [2048]);
        let plain = terms(LIMITS, &[]);
        assert_eq!(plain.len(), 4);
        assert!(
            plain
                .iter()
                .all(|term| (term.weight - 1. / 4.).abs() < 1e-12)
        );
        let drafted = terms(LIMITS, &[1, 2, 3]);
        assert_eq!(drafted.len(), 2 + 2 * 3);
        assert!((drafted.iter().map(|term| term.weight).sum::<f64>() - 1.).abs() < 1e-12);
        let decode_64k = drafted
            .iter()
            .filter(|term| term.metric.phase == Phase::Decode && term.metric.context == 64256)
            .map(|term| term.weight)
            .sum::<f64>();
        assert!((decode_64k - 1. / 4.).abs() < 1e-12);
    }
}
