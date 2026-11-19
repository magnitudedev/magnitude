//! A prompt's merged-expert passes: the pass sizes a load's arena holds
//! one of ([`pass_ladder`], `ByPass`); the routed seconds a prompt takes in
//! passes of a size are the cost model's
//! (`magnitude_cost_model::routed_prompt_seconds`).

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
