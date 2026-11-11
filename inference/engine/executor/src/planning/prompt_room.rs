//! A prompt's merged-expert passes: the pass sizes a load's arena holds
//! one of ([`pass_ladder`], `ByPass`), and the routed seconds a prompt takes
//! in passes of a size ([`routed_prompt_seconds`]), over the executor's
//! generic costs. Nothing here reads recorded per-model data, nor anything
//! the load measured of its own prompts.

use crate::assessment::ExpertPass;
use crate::planning::PassLaunches;
use std::num::NonZeroUsize;

/// The pass sizes up to `most`: one launch, then doubling, to `most`,
/// which it always ends at.
pub(crate) fn pass_ladder(most: PassLaunches) -> Vec<PassLaunches> {
    let mut ladder = vec![PassLaunches::FLOOR];
    let mut launches = 1usize;
    while launches < most.get() {
        launches = (launches * 2).min(most.get());
        ladder.push(PassLaunches::new(NonZeroUsize::new(launches).expect("a ladder rung is at least one launch")));
    }
    ladder
}

/// The routed seconds a prompt of `prompt_rows` rows takes in passes over
/// `pass` launches of `region_rows` rows each, on the executor's costs
/// (`pass`, the step model's [`ExpertPass`]): each pass reads, at every
/// routed layer, the union its rows choose of the layer's `(experts,
/// selected)` (`estimate::expert_union`), and computes their choices; and
/// it reads from storage the misses of that union a partial bank does not
/// hold (`reads`, a pass's seconds by its rows; none where the bank holds
/// every expert). Fewer, longer passes read each expert fewer times, from
/// the device's memory and from storage alike; generic machine figures and
/// header shapes only.
pub(crate) fn routed_prompt_seconds(
    passes: &ExpertPass,
    layers: &[(u64, u64)],
    region_rows: u64,
    prompt_rows: u64,
    pass: PassLaunches,
    reads: &dyn Fn(u64) -> f64,
) -> f64 {
    let pass_rows = region_rows * pass.get() as u64;
    let one = |rows: u64| -> f64 {
        layers
            .iter()
            .enumerate()
            .map(|(layer, &(experts, selected))| {
                let union = crate::assessment::estimate::expert_union(experts, selected, rows);
                passes.seconds(layer, union, (rows * selected) as f64)
            })
            .sum::<f64>()
            + reads(rows)
    };
    let (whole, rest) = (prompt_rows / pass_rows, prompt_rows % pass_rows);
    whole as f64 * one(pass_rows) + if rest > 0 { one(rest) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ladder doubles from one launch and ends at its most.
    #[test]
    fn a_pass_ladder_doubles_to_its_most() {
        let launches = |count: usize| PassLaunches::new(NonZeroUsize::new(count).unwrap());
        assert_eq!(pass_ladder(launches(6)), vec![launches(1), launches(2), launches(4), launches(6)]);
        assert_eq!(pass_ladder(PassLaunches::FLOOR), vec![PassLaunches::FLOOR]);
    }
}
