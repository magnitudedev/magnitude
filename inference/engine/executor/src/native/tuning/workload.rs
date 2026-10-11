//! The served workload tuning weighs its units and points by: what one
//! agent turn asks of the engine, per tuning row class.
//!
//! Derived from 137,170 turns of 1,631 Magnitude coding-agent sessions on
//! the hosted provider (`~/.magnitude/sessions`, 2026-10-07; recorded with
//! the derivation in `design/inference/engine/execution-plan.md`). Per turn
//! the history at its start is the provider's cached prefix (median 46.6K
//! tokens, 80% of turns hit), the new prompt tokens are the rest of its
//! input (median 931, mean 13,579: a miss prefills the whole conversation)
//! and its output is decoded (median 151, mean 426). New prompt tokens
//! outnumber decoded ones 32 to 1. Histories are fractions of the context:
//! the turns' inputs, against the 256K largest context their models serve,
//! fall within an octave of it as 14% (128–256K), 31%, 30%, 15%, 7%, 2% and
//! under 1% below.
//!
//! A prompt is prefilled in launches of the largest row class, the last
//! launch of a turn in the smallest class that holds its remainder: per turn
//! 26.5 launches of 512 rows at a mean history of 0.18 of the context, and
//! the remainders' classes as listed. Decoding takes 426 tokens per turn at
//! a mean history of 0.28: two thirds in single-row steps, a third in
//! verification and concurrent steps of 2, 4 and 8 rows (a third of the
//! steps each). The session data does not tell those apart; the split is
//! the engine's earlier decode mix, kept until serving reports it.

use super::{TUNING_ROWS, TuningLimits};

/// One row class's load per turn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClassLoad {
    pub rows: u64,
    /// Launches of this many rows per turn.
    pub launches: f64,
    /// Their mean visible history, as a fraction of the context.
    pub history: f64,
}

/// Decoded tokens per turn.
const DECODE_TOKENS: f64 = 426.1;
pub(crate) fn decoded_tokens() -> f64 {
    DECODE_TOKENS
}
/// The mean history a decoded token sees, as a fraction of the context.
const DECODE_HISTORY: f64 = 0.278;

/// Prefill launches per turn by row class (rows, launches, history).
const PREFILL: [(u64, f64, f64); 7] = [
    (16, 0.012, 0.250),
    (32, 0.023, 0.253),
    (64, 0.058, 0.252),
    (128, 0.178, 0.315),
    (256, 0.289, 0.271),
    (512, 26.488, 0.184),
    // Remainders of fewer rows than 16 are a few thousandths of a launch.
    (8, 0.006, 0.265),
];

/// A row class of at most this many rows streams its weights: a launch
/// costs about as much as one of this many rows of a compute-bound class
/// (27B dense expansion on an M4 Pro: 0.50 ms at one row, 24.8 ms at 512).
pub(crate) const STREAMING_EQUIVALENT_ROWS: f64 = 10.;

/// The served workload's launches of row class `rows` per turn and their
/// mean history, for an engine whose largest launch has `limits.max_rows`
/// rows: a prompt longer than that is prefilled in launches of the largest
/// class it has, which takes the launches of every larger class.
pub(crate) fn prefill_load(rows: u64, limits: TuningLimits) -> ClassLoad {
    let largest = TUNING_ROWS
        .into_iter()
        .filter(|rows| *rows <= limits.max_rows)
        .max()
        .unwrap_or(1);
    let prefill = PREFILL
        .iter()
        .filter(|(class, ..)| *class == rows || (rows == largest && *class > largest))
        .map(|(class, launches, history)| {
            // A launch of the larger class is `class / largest` of these.
            let scale = (*class).max(rows) as f64 / rows as f64;
            (launches * scale, *history)
        })
        .collect::<Vec<_>>();
    let prefill_launches = prefill.iter().map(|(launches, _)| launches).sum::<f64>();
    let prefill_history = if prefill_launches > 0. {
        prefill
            .iter()
            .map(|(launches, history)| launches * history)
            .sum::<f64>()
            / prefill_launches
    } else {
        0.
    };
    ClassLoad {
        rows,
        launches: prefill_launches,
        history: prefill_history,
    }
}

pub(crate) fn class_load(rows: u64, limits: TuningLimits) -> ClassLoad {
    let prefill = prefill_load(rows, limits);
    let decode = match rows {
        1 => DECODE_TOKENS * 2. / 3.,
        2 | 4 | 8 => DECODE_TOKENS / 9.,
        _ => 0.,
    };
    let launches = prefill.launches + decode;
    ClassLoad {
        rows,
        launches,
        history: if launches > 0. {
            (prefill.launches * prefill.history + decode * DECODE_HISTORY) / launches
        } else {
            0.
        },
    }
}

/// A row class's invocation demand among a unit's points. Device latency,
/// rather than a rows-based cost surrogate, determines its elapsed-service vote.
pub(crate) fn class_weight(rows: u64, limits: TuningLimits) -> f64 {
    class_load(rows, limits).launches
}
