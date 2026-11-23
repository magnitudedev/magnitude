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
//!   its drafts and their anchor;
//! - `vision(C)`, of a load that encodes images: encoding one image of the
//!   largest class the load encodes, of C merged cells, each tower launch
//!   at the image's rows. Only the vision units launch in it, so it enters
//!   their objectives alone: it weighs as a text metric does, beside the
//!   text metrics rather than taking from their weights.
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
    /// Encoding one image of the metric's context of merged cells.
    Vision,
}

/// One measurement the engine is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Metric {
    pub phase: Phase,
    /// The prompt's tokens (prefill), the history decoded at (decode) or
    /// the image's merged cells (vision).
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

/// A metric's weight in the score: decode (`M <= VERIFY_ROWS`) outweighs
/// prefill (`3:1`) so decode-critical units win discovery and search budget
/// ahead of heavy prefill units.
pub(crate) fn metric_weight(metric: &Metric) -> f64 {
    match metric.phase {
        Phase::Decode => 3.,
        Phase::Prefill | Phase::Vision => 1.,
    }
}

/// The load's metrics: an image's (`images`, a load that prepares a vision
/// program) beside its text's.
pub(crate) fn metrics(limits: TuningLimits, images: bool) -> Vec<Metric> {
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
    if images {
        assert!(
            limits.image_cells > 0,
            "a load that prepares a vision program encodes images"
        );
        metrics.push(Metric {
            phase: Phase::Vision,
            context: limits.image_cells,
            sequences: 1,
        });
    }
    metrics
}

/// The objective's terms for a load whose drafter prepares `widths` (none
/// without drafting) and that encodes `images` ([`metrics`]): every metric,
/// a drafted decode metric once for each width, the text metrics' weights
/// summing to one and the image's beside them.
pub(crate) fn terms(limits: TuningLimits, widths: &[u64], images: bool) -> Vec<Term> {
    let metrics = metrics(limits, images);
    let total = metrics
        .iter()
        .filter(|metric| metric.phase != Phase::Vision)
        .map(metric_weight)
        .sum::<f64>();
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
        max_pass_rows: 512,
        max_projected_rows: 8,
        context_tokens: 64256,
        image_cells: 0,
    };

    /// Prefill and decode at every context point to the limit, with decode
    /// weighted 3x prefill; a drafted decode metric divides its weight among
    /// the prepared widths.
    #[test]
    fn every_metric_counts_alike_and_widths_share_theirs() {
        assert_eq!(contexts(LIMITS), [4096, 64256]);
        let short = TuningLimits {
            context_tokens: 2048,
            ..LIMITS
        };
        assert_eq!(contexts(short), [2048]);
        let plain = terms(LIMITS, &[], false);
        assert_eq!(plain.len(), 4);
        assert!((plain.iter().map(|term| term.weight).sum::<f64>() - 1.).abs() < 1e-12);
        let prefill_total: f64 = plain
            .iter()
            .filter(|term| term.metric.phase == Phase::Prefill)
            .map(|term| term.weight)
            .sum();
        let decode_total: f64 = plain
            .iter()
            .filter(|term| term.metric.phase == Phase::Decode)
            .map(|term| term.weight)
            .sum();
        assert!((prefill_total - 0.25).abs() < 1e-12);
        assert!((decode_total - 0.75).abs() < 1e-12);
        let drafted = terms(LIMITS, &[1, 2, 3], false);
        assert_eq!(drafted.len(), 2 + 2 * 3);
        assert!((drafted.iter().map(|term| term.weight).sum::<f64>() - 1.).abs() < 1e-12);
        let decode_64k = drafted
            .iter()
            .filter(|term| term.metric.phase == Phase::Decode && term.metric.context == 64256)
            .map(|term| term.weight)
            .sum::<f64>();
        assert!((decode_64k - 3. / 8.).abs() < 1e-12);
    }

    /// A load that encodes images holds one more metric, an image of its
    /// largest class, weighing as a prefill metric does, after the text terms,
    /// which it leaves exactly as a load without images has them.
    #[test]
    fn an_image_is_a_metric_beside_the_text_metrics() {
        let images = TuningLimits {
            image_cells: 256,
            ..LIMITS
        };
        let text = terms(images, &[1, 2], false);
        let terms = terms(images, &[1, 2], true);
        assert_eq!(terms[..text.len()], text[..]);
        let [vision] = &terms[text.len()..] else {
            panic!("one image term follows the text terms: {terms:?}");
        };
        assert_eq!(vision.metric.phase, Phase::Vision);
        assert_eq!(vision.metric.context, 256);
        assert!((vision.weight - 1. / 8.).abs() < 1e-12);
    }
}
