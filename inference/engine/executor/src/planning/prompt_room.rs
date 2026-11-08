//! A prompt group's room as priced holdings of the one heap
//! (`domain::value`): its places beyond the first, its merged-expert pass
//! size beyond one launch ([`PassLaunches`]).
//! Each is worth the prefill (or launch) seconds it saves times how often
//! that happens, in seconds of serving saved per second: the same unit the
//! bank's slots are valued in, so a pass buffer that saves little never
//! holds bytes a resident bank's last slot is worth far more for.
//!
//! The seconds come from the caller: at load, the step model over the
//! executor's generic costs ([`routed_prompt_seconds`]); while serving, the
//! load's measured group reads (`GroupReads`), or before they are measured
//! the same step model by group ([`routed_group_seconds`]), beside each
//! group's submission. The rates of prompts and replays come from the
//! caller too. Nothing here reads recorded per-model data.

use crate::assessment::ExpertPass;
use crate::domain::value::{Curve, Holding, Rate, Step};
use crate::planning::PassLaunches;
use std::num::NonZeroUsize;

/// The pass sizes a [`pass_rows_curve`] steps through: one launch, then
/// doubling, to `most`, which it always ends at.
pub(crate) fn pass_ladder(most: PassLaunches) -> Vec<PassLaunches> {
    let mut ladder = vec![PassLaunches::FLOOR];
    let mut launches = 1usize;
    while launches < most.get() {
        launches = (launches * 2).min(most.get());
        ladder.push(PassLaunches::new(NonZeroUsize::new(launches).expect("a ladder rung is at least one launch")));
    }
    ladder
}

/// The curve of a priced holding over the `ladder` of sizes it may hold,
/// holding `held` of them (an index into it, zero the floor): each step from
/// one rung to the next holds the bytes `bytes` adds and saves what
/// `seconds` falls by, `per_second` times a second.
fn ladder_curve<T: Copy>(
    holding: Holding,
    ladder: &[T],
    held: usize,
    bytes: &dyn Fn(T) -> u64,
    seconds: &dyn Fn(T) -> f64,
    per_second: f64,
    move_seconds: f64,
) -> Curve<Holding> {
    assert!(held < ladder.len(), "a holding holds a rung of its ladder");
    let step = |pair: &[T]| Step {
        bytes: bytes(pair[1]).checked_sub(bytes(pair[0])).expect("invariant violated: a holding's bytes fell as it grew"),
        worth: Rate::new((seconds(pair[0]) - seconds(pair[1])).max(0.0) * per_second),
        move_seconds,
    };
    let steps = ladder.windows(2).map(step).collect::<Vec<_>>();
    let (held_steps, offered) = steps.split_at(held);
    Curve { holding, borrows: false, held: held_steps.iter().rev().copied().collect(), offered: offered.to_vec() }
}

/// The pass size as a priced holding ([`Holding::PassRows`]): the steps of
/// `ladder` ([`pass_ladder`]), holding `held` of them, each the pass
/// buffers it adds (`pass_bytes`, `native_target_graph::pass_buffer_bytes`)
/// against the prompt seconds its fewer passes save (`prompt_seconds`)
/// `prompts_per_second` times a second.
pub(crate) fn pass_rows_curve(
    ladder: &[PassLaunches],
    held: usize,
    pass_bytes: &dyn Fn(PassLaunches) -> u64,
    prompt_seconds: &dyn Fn(PassLaunches) -> f64,
    prompts_per_second: f64,
    move_seconds: f64,
) -> Curve<Holding> {
    ladder_curve(Holding::PassRows, ladder, held, pass_bytes, prompt_seconds, prompts_per_second, move_seconds)
}

/// A prompt group's places beyond the `held` it holds, up to `most`, as a
/// priced holding ([`Holding::Places`]): each place more the class bytes it
/// adds (`place_bytes`, as `grant_group` grants them) against the prompt
/// seconds a group of more places saves (`prompt_seconds`),
/// `prompts_per_second` times a second. A group holds its places while it
/// runs, so none is offered for release.
pub(crate) fn places_curve(
    held: usize,
    most: usize,
    place_bytes: &dyn Fn(usize) -> u64,
    prompt_seconds: &dyn Fn(usize) -> f64,
    prompts_per_second: f64,
) -> Curve<Holding> {
    let ladder = (held..=most.max(held)).collect::<Vec<_>>();
    ladder_curve(Holding::Places, &ladder, 0, place_bytes, prompt_seconds, prompts_per_second, 0.0)
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

/// The seconds `chunks` prompt chunks take in groups of `places`, a group
/// of `steps` chunks taking `group(steps)`: the whole groups and the one
/// left over; the first error `group` meets.
pub(crate) fn grouped<E>(chunks: usize, places: usize, mut group: impl FnMut(usize) -> Result<f64, E>) -> Result<f64, E> {
    let rest = match chunks % places {
        0 => 0.0,
        rest => group(rest)?,
    };
    Ok((chunks / places) as f64 * group(places)? + rest)
}

/// The routed seconds one group of `steps` prompt chunks of a launch region
/// of `region_rows` rows each takes, in passes of up to `pass` launches
/// ([`routed_prompt_seconds`]): a pass merges only the launches of one
/// group, so a group of more places runs fewer passes where the plan's pass
/// covers more than one launch.
pub(crate) fn routed_group_seconds(
    passes: &ExpertPass,
    layers: &[(u64, u64)],
    region_rows: u64,
    steps: usize,
    pass: PassLaunches,
    reads: &dyn Fn(u64) -> f64,
) -> f64 {
    routed_prompt_seconds(passes, layers, region_rows, steps as u64 * region_rows, pass, reads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::value::{settle, Supply};
    use std::time::Duration;

    const MB: u64 = 1_000_000;

    fn launches(count: usize) -> PassLaunches {
        PassLaunches::new(NonZeroUsize::new(count).unwrap())
    }

    /// Gemma q4 on a 24 GB card: 512-row launches, a group of 150 of them
    /// (76,800 rows) whose one pass's merged experts take 3.48 GB, about
    /// 23.2 MB a launch, against a bank whose last slot (about 130 MB)
    /// makes the model resident. A pass costs every layer's experts' read
    /// once (30 layers × 128 experts of 2.9 MB at 900 GB/s, about 12.4 ms):
    /// a 16k prompt runs 32 passes at 512 rows, one at the group's size.
    fn pass_bytes(pass: PassLaunches) -> u64 {
        pass.get() as u64 * 23_200_000
    }

    fn prompt_seconds(pass: PassLaunches) -> f64 {
        (16_384u64.div_ceil(512 * pass.get() as u64)) as f64 * 12.4e-3
    }

    /// One prompt a minute; the resident step saves 20 ms of a 30-step
    /// second of decode over the streamed one.
    const PROMPTS_PER_SECOND: f64 = 1.0 / 60.0;
    const RESIDENT_SAVING: f64 = 0.020 * 30.0;

    fn room(free: u64) -> Supply {
        Supply { free, owed: 0, borrowable: 0, borrow_price: 0.0, ceiling: u64::MAX, horizon: Duration::from_secs(60) }
    }

    fn bank_last_slot() -> Curve<Holding> {
        Curve {
            holding: Holding::Bank(crate::domain::BankHolding::Experts),
            borrows: false,
            held: vec![],
            offered: vec![Step { bytes: 130 * MB, worth: Rate::new(RESIDENT_SAVING), move_seconds: 0.0 }],
        }
    }

    /// The pass size a settle leaves of `ladder`.
    fn settled_pass(ladder: &[PassLaunches], granted: &[(Holding, usize)]) -> PassLaunches {
        let steps = granted.iter().find(|(holding, _)| *holding == Holding::PassRows).map_or(0, |(_, steps)| *steps);
        ladder[steps]
    }

    /// Where the bank's last slot and a whole-group pass do not both fit,
    /// the bank is made resident and the pass takes what is left, at least
    /// one launch (512 rows, main's pass), so the prompt is no slower than
    /// main's; with room for both, the bank stays resident and the pass
    /// grows only while it saves prompt time: to the fewest launches that
    /// cover the prompt in one pass (16,384 rows over 512 a launch), the
    /// sizes beyond saving nothing and taking no room.
    #[test]
    fn a_resident_bank_outranks_a_pass_buffer_that_buys_little() {
        let ladder = pass_ladder(launches(150));
        assert_eq!((ladder[0], *ladder.last().unwrap()), (PassLaunches::FLOOR, launches(150)));
        let curves = |held| {
            [bank_last_slot(), pass_rows_curve(&ladder, held, &pass_bytes, &prompt_seconds, PROMPTS_PER_SECOND, 0.0)]
        };
        // Room for the slot and 1 GB of pass buffers beyond one launch's.
        let tight = settle(&curves(0), room(130 * MB + 1_000 * MB));
        assert!(tight.granted.contains(&(Holding::Bank(crate::domain::BankHolding::Experts), 1)), "{tight:?}");
        let pass = settled_pass(&ladder, &tight.granted);
        assert!(pass >= PassLaunches::FLOOR && pass < launches(150), "{pass:?}");
        assert!(prompt_seconds(pass) <= prompt_seconds(PassLaunches::FLOOR));
        // Room for the slot alone: the pass stays one launch, main's.
        let slot = settle(&curves(0), room(130 * MB));
        assert_eq!(settled_pass(&ladder, &slot.granted), PassLaunches::FLOOR);
        assert!(slot.granted.contains(&(Holding::Bank(crate::domain::BankHolding::Experts), 1)));
        // Room for everything: the pass covers the prompt, no further.
        let ample = settle(&curves(0), room(130 * MB + pass_bytes(launches(150))));
        assert!(ample.granted.contains(&(Holding::Bank(crate::domain::BankHolding::Experts), 1)), "{ample:?}");
        let one_pass = launches(16_384usize.div_ceil(512));
        assert_eq!(settled_pass(&ladder, &ample.granted), one_pass);
        for &beyond in ladder.iter().filter(|&&pass| pass > one_pass) {
            assert_eq!(prompt_seconds(beyond), prompt_seconds(one_pass), "{beyond:?} saves nothing");
        }
    }

    /// A pass size held beyond what is worth its bytes is given up first
    /// when a resident bank's last slot asks for room.
    #[test]
    fn a_held_pass_buffer_gives_way_to_the_resident_slot() {
        let ladder = pass_ladder(launches(150));
        let top = ladder.len() - 1;
        let curves = [bank_last_slot(), pass_rows_curve(&ladder, top, &pass_bytes, &prompt_seconds, PROMPTS_PER_SECOND, 0.0)];
        let settled = settle(&curves, room(0));
        assert!(settled.granted.contains(&(Holding::Bank(crate::domain::BankHolding::Experts), 1)), "{settled:?}");
        assert!(settled.released.iter().any(|(holding, _)| *holding == Holding::PassRows));
    }

    /// A group's places step from those it holds as their class bytes do,
    /// each worth the prompt seconds it saves; none is offered for release.
    #[test]
    fn places_step_from_those_held_by_what_they_save() {
        let places = places_curve(2, 4, &|places| places as u64 * 10 * MB, &|places| 1.0 / places as f64, 1.0);
        assert_eq!((places.held.len(), places.offered.len()), (0, 2));
        assert_eq!(places.offered[0].bytes, 10 * MB);
        assert_eq!(places.offered[0].worth, Rate::new(0.5 - 1.0 / 3.0));
        assert!(places_curve(4, 3, &|_| 0, &|_| 0.0, 1.0).offered.is_empty());
    }
}
