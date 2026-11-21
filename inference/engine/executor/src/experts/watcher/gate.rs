//! The hint gate (POLICY_HINTS, [`HintGate`]): whether each streamed
//! layer's hint launch runs, which residency publishes into the layer's
//! policy words (`Residency::publish_policy`).
//!
//! One launch computes every lead a layer hints, so the gate is the
//! launch's: it runs where what its leads save a step outweighs its price
//! (`LaunchPrices::hint`, the cost model's one price source). Each lead ℓ's
//! per-step saving at its target layer `l` is priced from the counterfactual
//! demand cohort (`interest::Cohort`, `interest::CohortRates`) and the cost
//! model's single owner of layer demand stall (`magnitude_cost_model::ReadAdmission::layer_stall`):
//!
//!   `stall_l(m) − stall_l(m − exclusive_ℓ · cohort_l)`
//!
//! where `cohort_l` is layer `l`'s measured cohort size per step (misses plus
//! hint-landed saved reads, counted on every resolved step whether hints run
//! or not) and `exclusive_ℓ` is the share of cohort reads only lead ℓ would
//! have caught (counted only while ℓ runs, so no saved stall is credited
//! twice; unmeasured reads its bound 1 until ℓ's first cohort). While ℓ is
//! off, `exclusive_ℓ` freezes with its age in tokens (A10) while `cohort_l`
//! stays live, so cohort growth restarts the lead and logs the frozen rate
//! and its age. A target no route can find an expert absent at
//! (POLICY_RESOLVED) saves nothing, and a layer that hints none launches
//! none.
//!
//! Every rate rolls on the bank-wide turnover's epochs (`horizon::Window` /
//! `horizon::Aged`).
//!
//! Each change of the word is staged and costs one break of the pipeline
//! of queued steps (`Shared::stage_policy`), at most a pass, as measured
//! (`passes::Passes::latest`); so the gate moves only where what the layers
//! that change gain per token (per-pass gain divided by `ā`), over the
//! tokens the word is expected to hold (`horizon::Hold`), outweighs that
//! break ([`HintGate::decide`]), and none moves before a pass is timed.
//! Before the first change the hold is `Held::Unbounded`, so any positive
//! gain moves the gate once and measures its first hold. Either value of the
//! word is valid for any step: the gate moves what a pass costs, never what
//! it computes.

use crate::experts::horizon::{Hold, Time, Tokens};
use crate::experts::interest::{Caught, Cohort, CohortRates, HINT_LEADS};
use magnitude_cost_model::{LaunchPrices, ReadAdmission};

/// One-sided sub-Gaussian confidence multiplier derived from the live cost
/// ratio of a wrong flip (pipeline break `pass`) to the per-step launch cost
/// (`price`): `e^{-z^2 / 2} = price / (price + pass)` =>
/// `z = sqrt(2 * ln(1 + pass / price))`. Zero when `pass == 0` (free flip),
/// growing logarithmically with the break-to-launch ratio, with no tuned
/// constants.
fn confidence_z(pass: f64, price: f64) -> f64 {
    assert!(pass >= 0.0 && price >= 0.0, "pass ({pass}) and price ({price}) must be non-negative");
    if pass == 0.0 || price == 0.0 {
        return 0.0;
    }
    (2.0 * (1.0 + pass / price).ln()).sqrt()
}

/// Whether each streamed layer's hint launch runs, and the evidence it is
/// decided on: per layer, the layers its hint ring rows hint by lead, the
/// gate decided and the gate in effect, the counterfactual cohort rates, and
/// the hold on the token clock.
pub(super) struct HintGate {
    targets: Vec<Vec<usize>>,
    /// For each target layer, the source layer hinting it at lead
    /// `1..=HINT_LEADS` (index `lead - 1`), if any.
    hinters: Vec<[Option<usize>; HINT_LEADS]>,
    decided: Vec<bool>,
    published: Vec<bool>,
    time: Time,
    rates: CohortRates,
    hold: Hold,
}

impl HintGate {
    /// None seen, of streamed layers whose hint ring rows hint `targets`,
    /// each running its hints where it hints a layer, rolling on `time`'s
    /// bank-wide epochs and clock.
    pub(super) fn new(targets: Vec<Vec<usize>>, time: Time) -> Self {
        let layers = targets.len();
        let mut hinters = vec![[None; HINT_LEADS]; layers];
        for (src, targets) in targets.iter().enumerate() {
            assert!(
                targets.len() <= HINT_LEADS,
                "layer {src} hints {} leads, past HINT_LEADS ({HINT_LEADS})",
                targets.len()
            );
            for (index, &target) in targets.iter().enumerate() {
                hinters[target][index] = Some(src);
            }
        }
        let running = targets.iter().map(|targets| !targets.is_empty()).collect::<Vec<_>>();
        let (epoch, now) = {
            let times = time.lock();
            (times.bank_epoch(), times.clock.through_latest())
        };
        Self {
            targets,
            hinters,
            decided: running.clone(),
            published: running,
            time,
            rates: CohortRates::new(layers, epoch),
            hold: Hold::started(now),
        }
    }

    /// Completed decode step resolved its interests into counterfactual
    /// demand cohorts (`interest::Learned::cohorts`): every layer's cohort
    /// size counts always, and each lead's exclusive-catch share counts only
    /// while the layer hinting that target at that lead ran its hints.
    pub(super) fn resolved(&mut self, cohorts: &[Cohort]) {
        let (epoch, now) = {
            let times = self.time.lock();
            (times.bank_epoch(), times.clock.through_latest())
        };
        for cohort in cohorts {
            let running = self.hinters[cohort.layer].map(|src| src.is_some_and(|src| self.published[src]));
            self.rates.count(cohort, &running, epoch, now);
        }
    }

    /// Residency's write of `hints` took effect: in effect from now.
    pub(super) fn publish(&mut self, hints: &[bool]) {
        self.published.copy_from_slice(hints);
    }

    /// Whether `layer`'s hints run, as in effect.
    pub(super) fn published(&self, layer: usize) -> bool {
        self.published[layer]
    }

    /// Whether each streamed layer's hint launch runs, moved from those
    /// decided where that pays a break, a pass of `rows` rows taking `pass`
    /// seconds (none before one is timed: nothing moves) and a miss
    /// costing `miss`: each runs where what it saves a step
    /// ([`Self::saving`]) exceeds its price in a pass of those rows
    /// (`LaunchPrices::hint`, none before the load's prices are had:
    /// nothing moves), and the layers whose value changes move together
    /// where what they gain per token (`gained / ā`) repays `pass` over the
    /// expected hold (`Hold::pays`). `resolved` says which layers no route
    /// can find an expert absent at.
    ///
    /// # Panics
    /// Where the load's prices refuse a hint launch, or price none for a
    /// layer that hints: its census makes every hinting layer's hint
    /// launch (`LayerCall::Hint`).
    pub(super) fn decide(&mut self, resolved: &[bool], prices: Option<&LaunchPrices>, miss: ReadAdmission, pass: Option<(u32, f64)>) -> Vec<bool> {
        let (Some(prices), Some((rows, pass))) = (prices, pass) else {
            return self.decided.clone();
        };
        let (epoch, now, tokens_per_step) = {
            let times = self.time.lock();
            (
                times.bank_epoch(),
                times.clock.through_latest(),
                times.tokens_per_step(0).expect("ā exists once a decode pass is timed"),
            )
        };
        let mut gained = 0.0;
        let hints = (0..self.targets.len())
            .map(|layer| {
                let running = self.decided[layer];
                if self.targets[layer].is_empty() {
                    return running;
                }
                let price = prices
                    .hint(layer, rows)
                    .unwrap_or_else(|error| panic!("invariant violated: the load's prices refuse streamed layer {layer}'s hint launch at {rows} rows: {error}"))
                    .unwrap_or_else(|| panic!("invariant violated: streamed layer {layer} hints {:?} and its load's census makes no hint launch of it", self.targets[layer]));
                let z = confidence_z(pass, price);
                let Some((saving, saving_on, saving_off)) = self.saving_bounds(layer, resolved, miss, epoch, now, z) else {
                    return running;
                };
                let run = if running {
                    saving_off > price
                } else {
                    saving_on > price
                };
                if run != running {
                    gained += (saving - price).abs();
                }
                run
            })
            .collect::<Vec<_>>();
        if self.hold.pays(now, gained / tokens_per_step, pass) {
            // Assumption A10: log each restarting lead's frozen exclusive-catch
            // rate and its age in tokens at restart.
            for (layer, (&was, &now_on)) in self.decided.iter().zip(&hints).enumerate() {
                if !was && now_on {
                    for (index, &target) in self.targets[layer].iter().enumerate() {
                        let lead = (index + 1) as u8;
                        if let Caught::Measured { rate, age } = self.rates.exclusive(lead, epoch, now) {
                            tracing::info!(
                                layer,
                                target,
                                lead,
                                frozen_exclusive = rate,
                                age_tokens = age.get(),
                                "hint gate restart (A10)"
                            );
                        }
                    }
                }
            }
            self.decided.clone_from(&hints);
            self.hold.changed(now);
        }
        self.decided.clone()
    }

    /// Point estimate and confidence bounds `(point, lower_for_on, upper_for_off)`
    /// of the step time `layer`'s hints save a step: summed over each unresolved
    /// target `target` it hints at lead ℓ,
    /// `stall_l(m) − stall_l(m − exclusive_ℓ · cohort_l)`.
    ///
    /// Returns `None` (undecided: hold current state) if any unresolved target
    /// has an unmeasured cohort or unmeasured exclusive-catch rate (never
    /// flipping on or off with zero evidence). If all targets are resolved,
    /// returns `Some((0.0, 0.0, 0.0))`.
    ///
    /// - `upper_for_off` uses `(exclusive_hi, cohort_hi)` at `z`: a running
    ///   layer flips OFF only when even its upper confidence bound no longer
    ///   repays its launch price.
    /// - `lower_for_on` uses `(exclusive_point, cohort_lo)` at `z`: a stopped
    ///   layer flips ON only when its measured exclusive rate and lower cohort
    ///   bound repay its launch price (never turning on with `exclusive == 0`).
    fn saving_bounds(&mut self, layer: usize, resolved: &[bool], miss: ReadAdmission, epoch: u64, now: Tokens, z: f64) -> Option<(f64, f64, f64)> {
        let (mut point, mut lower, mut upper) = (0.0, 0.0, 0.0);
        for (index, &target) in self.targets[layer].iter().enumerate() {
            if resolved[target] {
                continue;
            }
            let cohort_share = self.rates.cohort_share(target, epoch)?;
            let cohort = cohort_share.rate();
            let (cohort_lo, cohort_hi) = cohort_share.poisson(z);
            let lead = (index + 1) as u8;
            let (excl_share, _) = self.rates.exclusive_share(lead, epoch, now)?;
            let excl = excl_share.rate();
            let excl_hi = excl_share.wilson(z).1;
            let delta = |m: f64, e: f64| miss.layer_stall(m) - miss.layer_stall((m - e * m).max(0.0));
            point += delta(cohort, excl);
            lower += delta(cohort_lo, excl);
            upper += delta(cohort_hi, excl_hi);
        }
        Some((point, lower, upper))
    }

    #[cfg(test)]
    fn saving(&mut self, layer: usize, resolved: &[bool], miss: ReadAdmission, epoch: u64, now: Tokens) -> f64 {
        let mut total = 0.0;
        for (index, &target) in self.targets[layer].iter().enumerate() {
            if resolved[target] {
                continue;
            }
            let Some(cohort) = self.rates.cohort(target, epoch) else {
                continue;
            };
            let lead = (index + 1) as u8;
            let exclusive = self.rates.exclusive(lead, epoch, now).rate();
            let remaining = (cohort - exclusive * cohort).max(0.0);
            total += miss.layer_stall(cohort) - miss.layer_stall(remaining);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::super::passes::{Pass, Passes};
    use super::*;
    use crate::experts::horizon::{Advance, Held, Tier};
    use crate::experts::interest::Catchers;
    use magnitude_cost_model::RoutedKind;

    /// Launch prices whose hint launch at each hinting layer costs 0.1 ms.
    fn table_for(hinting_layers: &[usize]) -> LaunchPrices {
        use magnitude_cost_model::{Census, Kind, LaunchCounts, LayerCall, RoutedCall, TargetCensus};
        use seismic::measure::pricing::{Experts, Figures, Flops, Format, Rates, Storage, Work};
        const BF16: Format = Format("bf16");
        let decode_rows = magnitude_kernels::constants::VERIFY_ROWS;
        let routed = |flops| Kind {
            work: Work {
                experts: Some(Experts { experts: 8, selected: 2, slab_bytes: Vec::new(), slab_flops: vec![Flops { format: BF16, flops }] }),
                ..Work::default()
            },
            launches: LaunchCounts::new(vec![Ok(1); decode_rows as usize]),
        };
        let census = |tagged: bool| {
            let calls = [(1e5 as u64, 0), (1e10 as u64, 1)].map(|(flops, layer)| {
                (routed(flops), tagged.then_some(LayerCall::Routed(RoutedCall { kind: RoutedKind(layer as u32), layer })))
            });
            let hints = hinting_layers.iter().map(|&layer| {
                (
                    Kind { work: Work::default(), launches: LaunchCounts::new(vec![Ok(1); decode_rows as usize]) },
                    tagged.then_some(LayerCall::Hint(layer)),
                )
            });
            Census::new(calls.into_iter().chain(hints), 0, decode_rows)
        };
        let rates = Rates {
            activation: BF16,
            weights: Storage::ALL.iter().map(|&storage| (storage, BF16, 1e12)).collect(),
            flops: vec![(BF16, 1e12)],
            history: Vec::new(),
            launch: 1e-4,
            decode: None,
        };
        let figures = Figures { rates, composition: Ok(1.), step_pipelined: 0., step_round_trip: 0. };
        LaunchPrices::new(TargetCensus::Banked { resident: census(false), streamed: census(true) }, figures, true)
    }

    fn table() -> LaunchPrices {
        table_for(&[0])
    }

    fn miss_cost(time: &Time, stall_per_miss: f64) -> ReadAdmission {
        let mut passes = Passes::new(time.clone());
        passes.timed(Pass { rows: 1, depth: 0, seconds: 0.01, parked: stall_per_miss, misses: 1.0, ..Pass::default() });
        passes.read_admission(None)
    }

    fn catchers(leads: &[u8]) -> Catchers {
        let mut c = Catchers::default();
        for &lead in leads {
            c.caught_by(lead);
        }
        c
    }

    /// §6: the cohort counts a hint-landed hit as a saved miss; a read two
    /// leads would both have caught is neither lead's exclusive saving; a
    /// lead's gain is `stall(m) − stall(m − exclusive_ℓ · cohort_l)` per
    /// layer; off, its rate holds with age and the cohort stays live.
    #[test]
    fn the_gate_credits_a_saved_stall_once() {
        // 3 layers: layer 0 hints [1, 2] (lead 1 -> layer 1, lead 2 -> layer 2),
        // layer 1 hints [2] (lead 1 -> layer 2), layer 2 hints none.
        let time = Time::new(&[vec![128]], 32);
        let mut gate = HintGate::new(vec![vec![1, 2], vec![2], Vec::new()], time.clone());
        let cost = miss_cost(&time, 1e-3); // 1 ms per miss

        // Step 1: layer 2 has 1 miss + 1 read caught by BOTH leads 1 and 2 + 1 read caught ONLY by lead 1.
        // Cohort size = 3. Exclusive(1) = 1/3, Exclusive(2) = 0/3.
        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        let c1 = Cohort { step: 1, layer: 2, misses: 1, saved: vec![catchers(&[1, 2]), catchers(&[1])] };
        gate.resolved(&[c1]);

        let (epoch, now) = {
            let times = time.lock();
            (times.bank_epoch(), times.clock.through_latest())
        };
        assert_eq!(gate.rates.cohort(2, epoch), Some(3.0));
        assert_eq!(gate.rates.exclusive(1, epoch, now), Caught::Measured { rate: 1.0 / 3.0, age: Tokens::ZERO });
        assert_eq!(gate.rates.exclusive(2, epoch, now), Caught::Measured { rate: 0.0, age: Tokens::ZERO });

        // Layer 1 hints layer 2 at lead 1: saving = stall(3) - stall(3 - (1/3)*3) = stall(3) - stall(2) = 1 ms.
        let save_l1 = gate.saving(1, &[false, false, false], cost, epoch, now);
        assert!((save_l1 - 1e-3).abs() < 1e-12, "only lead 1's exclusive saved read is credited: {save_l1}");
        // Layer 0 hints layer 1 (no cohort yet -> 0) and layer 2 at lead 2 (exclusive_2 = 0 -> 0).
        let save_l0 = gate.saving(0, &[false, false, false], cost, epoch, now);
        assert_eq!(save_l0, 0.0, "a read caught by both leads is credited to neither lead 2 nor twice");

        // Accumulate sufficient evidence for lead 2's Wilson upper bound to drop
        // below launch price (0.1 ms) while layer 1 is resolved: layer 0 turns OFF while layer 1 stays ON.
        let prices = table_for(&[0, 1]);
        let mut step = 2u64;
        let mut decided = gate.decide(&[false, true, false], Some(&prices), cost, Some((1, 0.01)));
        gate.publish(&decided);
        for _ in 0..140 {
            time.lock().register(step, Advance::Known(Tokens::new(1.0)));
            gate.resolved(&[Cohort { step, layer: 2, misses: 1, saved: vec![catchers(&[1, 2]), catchers(&[1])] }]);
            decided = gate.decide(&[false, true, false], Some(&prices), cost, Some((1, 0.01)));
            gate.publish(&decided);
            step += 1;
            if decided == [false, true, false] {
                break;
            }
        }
        assert_eq!(decided, [false, true, false]);

        // Now turn layer 1 OFF by letting layer 2 stop missing across two epochs,
        // so its Poisson upper bound drops below launch price and hold repays break.
        for _ in 0..300 {
            {
                let mut times = time.lock();
                times.register(step, Advance::Known(Tokens::new(1.0)));
                times.moved(0, Tier::Compute);
            }
            gate.resolved(&[Cohort { step, layer: 2, misses: 0, saved: Vec::new() }]);
            let d = gate.decide(&[false, true, false], Some(&prices), cost, Some((1, 0.01)));
            gate.publish(&d);
            step += 1;
        }
        assert!(!gate.published(1), "layer 1 turned off once cohort dropped to 0");

        // While layer 1 is off, layer 2's misses return (cohort stays live),
        // while exclusive_1 stays frozen at 1/3 with growing age.
        let off_step = step;
        {
            let mut times = time.lock();
            times.register(off_step, Advance::Known(Tokens::new(1.0)));
            times.moved(0, Tier::Compute);
        }
        gate.resolved(&[Cohort { step: off_step, layer: 2, misses: 3, saved: Vec::new() }]);
        let (epoch, now) = {
            let times = time.lock();
            (times.bank_epoch(), times.clock.through_latest())
        };
        match gate.rates.exclusive(1, epoch, now) {
            Caught::Measured { rate, age } => {
                assert!((rate - 1.0 / 3.0).abs() < 1e-12, "frozen exclusive_1 holds its last rate");
                assert!(age.get() >= 100.0, "age grows while off: {age:?}");
            }
            Caught::Unmeasured => panic!("exclusive_1 must never return to unmeasured"),
        }
        assert!(gate.rates.cohort(2, epoch).unwrap() > 0.0, "cohort stays live while lead is off");
    }

    /// A layer's hints run while what their names save outweighs their
    /// launch. Once their target stops missing they stop, and stopped their
    /// exclusive-catch rate freezes with its age; the target's cohort is
    /// still counted, so once misses return the hints start again.
    #[test]
    fn stopped_hints_start_again_as_their_targets_misses_return() {
        let time = Time::new(&[vec![128]], 32);
        let mut gate = HintGate::new(vec![vec![1], Vec::new()], time.clone());
        let prices = table();
        let cost = miss_cost(&time, 1e-3);

        // Step 1..=2: target layer 1 has 2 cohort reads, 1 caught exclusively by lead 1 -> exclusive_1 = 0.5, saving = 1 ms > 0.1 ms.
        for step in 1..=2 {
            {
                let mut times = time.lock();
                times.register(step, Advance::Known(Tokens::new(1.0)));
                times.moved(0, Tier::Compute);
            }
            gate.resolved(&[Cohort { step, layer: 1, misses: 1, saved: vec![catchers(&[1])] }]);
            let decided = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
            gate.publish(&decided);
            assert!(decided[0], "stays on at step {step}");
        }

        // Target stops missing across two epochs: cohort drops to 0 and its Poisson
        // upper bound falls below launch price -> stops (Hold::Unbounded).
        let mut step = 3;
        let stopped = (0..300).any(|_| {
            {
                let mut times = time.lock();
                times.register(step, Advance::Known(Tokens::new(1.0)));
                times.moved(0, Tier::Compute);
            }
            gate.resolved(&[Cohort { step, layer: 1, misses: 0, saved: Vec::new() }]);
            let decided = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
            gate.publish(&decided);
            step += 1;
            !decided[0]
        });
        assert!(stopped, "hints that save nothing stop");

        // Target misses return (cohort = 2 per step -> saving = 0.5 * 2 * 1 ms = 1 ms, net gain = 0.9 ms/token).
        // Once the lower cohort bound exceeds price and the growing hold repays the 10 ms break, hints restart.
        let started = (0..200).any(|_| {
            {
                let mut times = time.lock();
                times.register(step, Advance::Known(Tokens::new(1.0)));
                times.moved(0, Tier::Compute);
            }
            gate.resolved(&[Cohort { step, layer: 1, misses: 2, saved: Vec::new() }]);
            let decided = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
            gate.publish(&decided);
            step += 1;
            decided[0]
        });
        assert!(started, "stopped hints restart as their target's misses return");
    }

    /// The first move is at `Held::Unbounded` (any positive gain pays),
    /// finishing the first hold. Thereafter a move must repay the break
    /// (`pass`) over `Hold::expected(now)` tokens (converted by `ā`).
    #[test]
    fn the_gate_moves_where_its_gain_over_the_expected_hold_pays_a_break() {
        let time = Time::new(&[vec![128]], 32);
        let mut gate = HintGate::new(vec![vec![1], Vec::new()], time.clone());
        let prices = table();
        let cost = miss_cost(&time, 1e-3);

        // Step 1: record exclusive_1 = 0.5, then resolved target saves 0 deterministically ->
        // first move happens immediately (Unbounded).
        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        gate.resolved(&[Cohort { step: 1, layer: 1, misses: 1, saved: vec![catchers(&[1])] }]);
        let decided = gate.decide(&[false, true], Some(&prices), cost, Some((1, 0.01)));
        assert!(!decided[0], "first positive gain moves the gate while unbounded");
        gate.publish(&decided);
        assert_eq!(gate.hold.expected(Tokens::new(1.0)), Held::Tokens(Tokens::new(1.0)));

        // Step 2: target becomes unresolved with 2 misses -> lower cohort bound > 0.1 ms,
        // point saving = 1.0 ms (gain 0.9 ms/token), hold = 2 tokens -> 1.8 ms < 10 ms break -> does NOT move yet.
        time.lock().register(2, Advance::Known(Tokens::new(1.0)));
        gate.resolved(&[Cohort { step: 2, layer: 1, misses: 2, saved: Vec::new() }]);
        let decided = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
        assert!(!decided[0], "gain below break over expected hold does not move the gate");
    }

    /// Confidence bounds derived from `pass / price` prevent 1 noisy sample
    /// (`n = 1, x = 0`) from disabling hints, while sufficient negative
    /// evidence flips the gate OFF cleanly.
    #[test]
    fn one_bad_sample_cannot_disable_hints_while_sufficient_negative_evidence_does() {
        let time = Time::new(&[vec![256]], 32);
        let mut gate = HintGate::new(vec![vec![1], Vec::new()], time.clone());
        let prices = table();
        let cost = miss_cost(&time, 1e-3); // 1 ms per miss, launch price = 0.1 ms

        // Single bad sample: 1 miss, 0 catches (n = 1, x = 0) -> stays ON.
        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        gate.resolved(&[Cohort { step: 1, layer: 1, misses: 1, saved: Vec::new() }]);
        let decided = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
        assert!(decided[0], "one bad sample must not disable hints");
        gate.publish(&decided);

        // Accumulate sufficient negative evidence (n > 1200, x = 0) -> Wilson upper
        // bound at z = sqrt(2 ln(1 + 100)) drops below 0.1 ms -> turns OFF.
        let mut turned_off = false;
        for step in 2..=140u64 {
            time.lock().register(step, Advance::Known(Tokens::new(1.0)));
            gate.resolved(&[Cohort { step, layer: 1, misses: 10, saved: Vec::new() }]);
            let d = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
            gate.publish(&d);
            if !d[0] {
                turned_off = true;
                break;
            }
        }
        assert!(turned_off, "sufficient negative evidence must disable hints");
    }

    /// A target no route can find an expert absent at saves nothing,
    /// however large its cohort: its hints stop.
    #[test]
    fn hints_of_a_resolved_target_stop() {
        let time = Time::new(&[vec![1]], 32);
        let mut gate = HintGate::new(vec![vec![1], Vec::new()], time.clone());
        let prices = table();
        let cost = miss_cost(&time, 1e-3);

        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        gate.resolved(&[Cohort { step: 1, layer: 1, misses: 1, saved: vec![catchers(&[1])] }]);
        assert!(gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)))[0]);
        assert!(!gate.decide(&[false, true], Some(&prices), cost, Some((1, 0.01)))[0], "a resolved target's hints stop");
    }

    /// Before a pass is timed or the load's prices are had nothing moves:
    /// the gate holds its hints running.
    #[test]
    fn nothing_moves_unpriced() {
        let time = Time::new(&[vec![1]], 32);
        let mut gate = HintGate::new(vec![vec![1], Vec::new()], time.clone());
        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        gate.resolved(&[Cohort { step: 1, layer: 1, misses: 0, saved: Vec::new() }]);
        assert_eq!(gate.decide(&[false, true], None, ReadAdmission::default(), Some((1, 0.01))), [true, false]);
        assert_eq!(gate.decide(&[false, true], Some(&table()), ReadAdmission::default(), None), [true, false]);
    }

    /// A bank shrink to small capacity followed by `>= 2 * records` moves
    /// within a single decode step must never clear `ā` or panic in `decide`.
    #[test]
    fn decide_survives_many_moves_in_one_step_after_shrink() {
        let time = Time::new(&[vec![64]], 32);
        let mut gate = HintGate::new(vec![vec![1], Vec::new()], time.clone());
        let prices = table();
        let cost = miss_cost(&time, 1e-3);

        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        gate.resolved(&[Cohort { step: 1, layer: 1, misses: 1, saved: vec![catchers(&[1])] }]);
        let decided = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
        gate.publish(&decided);

        {
            let mut times = time.lock();
            times.resized(0, Tier::Compute, 2);
            times.register(2, Advance::Known(Tokens::new(1.0)));
            for _ in 0..6 {
                times.moved(0, Tier::Compute);
            }
        }
        let decided = gate.decide(&[false, false], Some(&prices), cost, Some((1, 0.01)));
        assert!(decided[0], "decide succeeds with ā intact after >= 2K moves in one step");
    }
}
