//! The load's decode passes as measured ([`Passes`]): the one record of
//! what a pass takes, of which every decision priced by a pass's time
//! reads its own figure, and none keeps another.
//!
//! - **Read-ahead admission & miss stall** ([`Passes::read_admission`],
//!   `magnitude_cost_model::ReadAdmission`): the seconds a pass parked per
//!   file miss (falling back to plain-pass stall), the drive line, and
//!   whether the drive is the binding term, by which every read ahead is
//!   admitted (`ReadAdmission::break_even`) and the hint gate values what
//!   its reads save (`gate::HintGate`).
//! - **The terms** ([`Measured`], `magnitude_cost_model::Terms`): a pass's
//!   busy seconds on each resource, by how many front rows it ran a
//!   sequence (its depth): its resident compute (its time less what it
//!   parked on the drive and what its misses served on the CPU or over the
//!   link held it, [`Pass::served`]); the drive's (the reads its exact rows
//!   needed, read ahead or on demand, along the drive's line,
//!   `seismic_io::Line`); beside a host tier the link's (its reads in place
//!   and the moves beside it), the CPU site's and host memory's; and its
//!   whole time; the moves' share of each layer the route's lines carry
//!   ([`Passes::moves`]), and the shares of the time a promotion's copy is
//!   priced beside ([`Passes::beside`]).
//! - **The regime** ([`Measured`]): what the pass is decided on
//!   (`magnitude_cost_model::Regime`, the domain's `TwoRowPricing`): the
//!   resident compute at each depth, of front and verify passes alike (the
//!   same rows compute alike); each term's seconds a miss of the memory the
//!   GPU computes from holds it ([`Pass::missed`]); the reads ahead nothing
//!   used, by the depth of the row whose name led them, at the drive's
//!   seconds a read, over the front passes that ran that depth; the stall a
//!   miss held a plain pass; whether the latest pass verified; and the
//!   guesses' stand rates, read from their one owner
//!   (`interest::Stands`).
//! - **The routes** ([`Routed`], [`Measured::routes`]): how a sequence's
//!   consecutive tokens share experts (the experts each lag's token adds to
//!   those of the tokens between) and the share of its picks the bank
//!   held, which price a depth the window holds no pass of
//!   (`magnitude_cost_model::FrontRoutes`).
//!
//! Every figure is a mean over the decode passes of the bank-wide turnover's
//! window (`horizon::Window` on `horizon::Times::bank_epoch`): a pass's
//! terms price every group's reads, so they roll on the bank-wide turnover
//! rather than any single group's.

use super::telemetry::FRONT_DEPTHS;
use crate::experts::horizon::{self, Time, Window};
use crate::experts::interest::{Form, Stands};
use magnitude_cost_model::{ReadAdmission, ReadLine, Terms};
use std::ops::AddAssign;

/// One decode pass as timed: its launch's rows, its depth (the guessed rows
/// it ran a sequence, front or acceptable, 0 of a pass of none), whether
/// they were verify rows (`verifies`), its seconds, the seconds it
/// parked on its misses, its file misses, its exact rows' misses of the
/// memory the GPU computes from (`missed`: its file and host tier misses,
/// and on unified memory the reads ahead they used, which landed there),
/// the drive's reads they needed (`reads`: its file misses and the reads
/// ahead they used), the reads ahead its step spent that nothing used, by
/// the depth of the row whose name led them, those front and exact rows
/// led at index 0 and those verify rows led at 1 (`unused`, issued less
/// cancelled and used, so a step may show some below zero), its busy
/// seconds on every path and in-place tier, what its misses served on the
/// CPU or over the link held it past its hits (`served`), and what its
/// exact rows' routes showed. The paths and tiers: the drive's seconds of the reads its exact
/// rows needed; the link's of its reads in place and the moves beside them
/// (`link`), of which the moves' (`moved`, of `moved_records` records); the
/// CPU site's requests' (`cpu`); and host memory's reads and landings
/// (`memory`). None but the drive's on unified memory.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Pass {
    pub rows: u32,
    pub depth: usize,
    pub verifies: bool,
    pub seconds: f64,
    pub parked: f64,
    pub misses: f64,
    pub missed: f64,
    pub reads: f64,
    pub unused: [[f64; DEPTHS]; 2],
    pub drive: f64,
    pub link: f64,
    pub moved: f64,
    pub moved_records: f64,
    pub cpu: f64,
    pub memory: f64,
    pub served: f64,
    pub routed: Routed,
}

/// The lags a sequence's routes are measured at: one to the most front
/// rows a pass runs it.
pub(super) const LAGS: usize = DEPTHS - 1;

/// What a pass's exact rows' routes showed, summed over its streamed
/// layers: at each lag `j` (index `j − 1`), the experts its token chose
/// that none of its sequence's `j` tokens before chose (`fresh`), over the
/// layers whose tokens that far back were seen (`layers`); and its picks,
/// and those the bank held (resident or tier-held, every choice but a file
/// miss). A pass of several sequences' exact rows, which are not one
/// sequence's tokens, shows none.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Routed {
    pub fresh: [f64; LAGS],
    pub layers: [f64; LAGS],
    pub picks: f64,
    pub held: f64,
}

impl Routed {
    /// `self + routed`.
    fn add(&mut self, routed: &Routed) {
        for lag in 0..LAGS {
            self.fresh[lag] += routed.fresh[lag];
            self.layers[lag] += routed.layers[lag];
        }
        self.picks += routed.picks;
        self.held += routed.held;
    }
}

/// Sums of passes, field by field: a window's, and the verify passes among
/// them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Sums {
    passes: f64,
    verifying: f64,
    by_depth: [[f64; DEPTHS]; 2],
    pass: Pass,
}

impl Pass {
    /// `self + pass`, of every summed field (a sum's rows, depth and form
    /// are none of a sum).
    fn add(&mut self, pass: &Pass) {
        self.seconds += pass.seconds;
        self.parked += pass.parked;
        self.misses += pass.misses;
        self.missed += pass.missed;
        self.reads += pass.reads;
        for (sum, unused) in self.unused.iter_mut().flatten().zip(pass.unused.iter().flatten()) {
            *sum += unused;
        }
        self.drive += pass.drive;
        self.link += pass.link;
        self.moved += pass.moved;
        self.moved_records += pass.moved_records;
        self.cpu += pass.cpu;
        self.memory += pass.memory;
        self.served += pass.served;
        self.routed.add(&pass.routed);
    }
}

impl Sums {
    /// One pass's.
    fn of(pass: &Pass) -> Self {
        let mut by_depth = [[0.; DEPTHS]; 2];
        by_depth[usize::from(pass.verifies)][pass.depth] = 1.;
        Self { passes: 1., verifying: f64::from(u8::from(pass.verifies)), by_depth, pass: Pass { rows: 0, depth: 0, verifies: false, ..*pass } }
    }

    /// `self + sums`.
    fn add(&mut self, sums: &Sums) {
        self.passes += sums.passes;
        self.verifying += sums.verifying;
        for (sum, other) in self.by_depth.iter_mut().flatten().zip(sums.by_depth.iter().flatten()) {
            *sum += other;
        }
        self.pass.add(&sums.pass);
    }

    /// The sum of a window's two tallies.
    fn over((previous, current): (&Sums, &Sums)) -> Self {
        let mut sum = *previous;
        sum += *current;
        sum
    }

    /// A pass's mean terms ([`Terms`]): its resident compute its time less
    /// what it parked and what its misses' service held it, every path's
    /// and tier's its busy seconds; none of no pass.
    fn terms(&self) -> Option<Terms> {
        let sum = &self.pass;
        assert!(
            sum.seconds + 1e-9 >= sum.parked,
            "parked time ({}) cannot exceed wall step time ({})",
            sum.parked,
            sum.seconds
        );
        let compute = ((sum.seconds - sum.parked).max(0.) - sum.served).max(0.);
        let terms = Terms { compute, drive: sum.drive, link: sum.link, cpu: sum.cpu, memory: sum.memory };
        (self.passes > 0.).then(|| terms.over(self.passes))
    }
}

impl horizon::Tally for Sums {
    fn is_empty(&self) -> bool {
        self.passes == 0.
    }
}

/// The depths a pass runs at: none to the most front rows the decode form
/// holds past an exact row.
pub(crate) const DEPTHS: usize = magnitude_kernels::constants::VERIFY_ROWS as usize;

impl AddAssign for Sums {
    fn add_assign(&mut self, other: Self) {
        self.add(&other);
    }
}

/// The decode passes timed over the bank-wide turnover's window, their
/// sums overall and by `(form, depth)`, and the latest pass.
pub(super) struct Passes {
    time: Time,
    latest: Option<Pass>,
    all: Window<Sums>,
    sums: [Box<[Window<Sums>]>; 2],
}

impl std::fmt::Debug for Passes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Passes").field("latest", &self.latest).finish_non_exhaustive()
    }
}

/// What the regime's pass is decided on (`magnitude_cost_model::Regime`),
/// as the decode passes of the latest turnover measured it: the streamed
/// layer count (`layers`), the latest's depth (`depth`,
/// the depth in effect) and whether it verified (`verifying`, after which
/// the next pass's chain is drafted to its own depth); a pass's mean terms
/// among the passes of each depth, front and verify alike
/// (`terms[k]`, none where the window holds no such pass), of which the
/// regime reads the resident compute; every term besides the GPU's over the
/// window (`paths`, none before a miss is counted): its seconds a miss
/// ([`Pass::missed`]), and on the drive's, the first, the seconds of a
/// pass's reads ahead nothing used that rows of each depth led, at its
/// seconds a read, of each form apart: a front pass's, its front and exact
/// rows' over the front passes that ran that depth, and a verify pass's,
/// its verify rows' lookahead hints' over the verify passes that ran it
/// (none where none did; their routes are the pass's own reads, counted in
/// `needed`); the seconds a miss held a plain pass past its compute, parked or
/// served (`stall`, none where no plain pass missed); the ranking's misses
/// a token emitted at each depth of foresight (`needed`, `Reuse::shadows`);
/// a front pass's first row's stand rate (`stand`, none unjudged, which
/// the regime prices at its bound 1); a verify pass's rows' stand rates, at
/// index `i − 1` row `i`'s given the rows before it stood (`accepts`, none
/// at a depth unjudged), each read from their one owner
/// (`interest::Stands`); and its routes: each lag's fresh experts, a layer's
/// mean (`fresh`, none where no pass saw a sequence's tokens that far
/// back), and the share of picks the bank held (`held`, none of no pick).
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    pub layers: usize,
    pub depth: usize,
    pub verifying: bool,
    pub terms: [Option<Terms>; DEPTHS],
    pub verify_terms: [Option<Terms>; DEPTHS],
    pub paths: Option<Vec<magnitude_cost_model::PathTerm>>,
    pub stall: Option<f64>,
    pub needed: [[Option<magnitude_cost_model::Needed>; DEPTHS]; 2],
    pub stand: Option<f64>,
    pub accepts: [Option<f64>; FRONT_DEPTHS],
    pub fresh: [Option<f64>; LAGS],
    pub held: Option<f64>,
}

impl Measured {
    /// Mean terms for `(verify, depth)`: verify passes use their own form's
    /// measurements and extrapolate from front passes when verify has no
    /// samples yet at `depth`.
    pub fn terms_for(&self, verify: bool, depth: usize) -> Option<Terms> {
        match verify {
            false => self.terms[depth],
            true => self.verify_terms[depth].or(self.terms[depth]),
        }
    }

    /// The routes a front pass reads its experts by
    /// (`magnitude_cost_model::FrontRoutes`): the fresh experts of each lag
    /// up to the first unmeasured one, beyond which routing is taken as
    /// independent, and the held share.
    pub fn routes(&self) -> magnitude_cost_model::FrontRoutes {
        magnitude_cost_model::FrontRoutes { fresh: self.fresh.iter().map_while(|fresh| *fresh).collect(), held: self.held }
    }
}

impl Passes {
    /// None timed, rolling on `time`'s bank-wide turnover epochs.
    pub fn new(time: Time) -> Self {
        let epoch = time.lock().bank_epoch();
        Self {
            time,
            latest: None,
            all: Window::new(epoch),
            sums: std::array::from_fn(|_| (0..DEPTHS).map(|_| Window::new(epoch)).collect()),
        }
    }

    fn epoch(&self) -> u64 {
        self.time.lock().bank_epoch()
    }

    /// A decode pass was timed: counted into the bank-wide window overall
    /// and for its `(form, depth)`.
    pub fn timed(&mut self, pass: Pass) {
        assert!(pass.depth < DEPTHS, "a pass of depth {} past the decode form's rows", pass.depth);
        let epoch = self.epoch();
        let one = Sums::of(&pass);
        *self.all.count(epoch) += one;
        *self.sums[usize::from(pass.verifies)][pass.depth].count(epoch) += one;
        self.latest = Some(pass);
    }

    /// One `(form, depth)`'s sums over its window (keeping its last measured
    /// epoch while unrun).
    fn depth(&mut self, verify: bool, depth: usize, epoch: u64) -> Sums {
        self.sums[usize::from(verify)][depth].read(epoch).map(Sums::over).unwrap_or_default()
    }

    /// Every pass of the bank-wide window, together.
    fn all(&mut self) -> Sums {
        let epoch = self.epoch();
        self.all.read(epoch).map(Sums::over).unwrap_or_default()
    }

    /// The mean terms a pass over the window ([`Terms`]) and the drive's
    /// seconds a read its exact rows needed: what a write-back into the
    /// host tier is priced on (`directory::WriteBack`). None before a pass
    /// that read is timed.
    pub fn terms(&mut self) -> Option<(Terms, f64)> {
        let all = self.all();
        let terms = all.terms()?;
        (all.pass.reads > 0.).then(|| (terms, all.pass.drive / all.pass.reads))
    }

    /// What a read ahead is admitted by over the window ([`ReadAdmission`]),
    /// given the drive's measured read line (`drive_line`).
    pub fn read_admission(&mut self, drive_line: Option<ReadLine>) -> ReadAdmission {
        let epoch = self.epoch();
        let all = self.all.read(epoch).map(Sums::over).unwrap_or_default();
        let plain = self.depth(false, 0, epoch).pass;
        let terms = all.terms().unwrap_or_default();
        let demand_stall = (all.pass.misses > 0.).then(|| all.pass.parked / all.pass.misses);
        let plain_stall = (plain.missed > 0.).then(|| plain.parked / plain.missed);
        ReadAdmission { terms, drive_line, demand_stall, plain_stall }
    }

    /// The latest pass's rows, and a pass's mean time over the window: what
    /// a break of the pipeline of queued steps costs at most. None before a
    /// pass is timed.
    pub fn latest(&mut self) -> Option<(u32, f64)> {
        let latest = self.latest?;
        let all = self.all();
        (all.passes > 0.).then(|| (latest.rows, all.pass.seconds / all.passes))
    }

    /// The moves beside a decode step over the window: the seconds they
    /// held the link and the records they read; none before a pass is
    /// timed.
    pub fn moves(&mut self) -> Option<(f64, f64)> {
        let all = self.all();
        (all.passes > 0.).then(|| (all.pass.moved / all.passes, all.pass.moved_records / all.passes))
    }

    /// The shares of the decode's time over the window a promotion's copy
    /// is priced beside (`assessment::Beside`): the CPU site's reading host
    /// memory, and the link's reads in place; none of either before a pass
    /// is timed.
    pub fn beside(&mut self) -> crate::assessment::Beside {
        let Pass { seconds, link, moved, cpu, .. } = self.all().pass;
        match seconds > 0. {
            true => crate::assessment::Beside { cpu: (cpu / seconds).clamp(0., 1.), link: ((link - moved) / seconds).clamp(0., 1.) },
            false => crate::assessment::Beside::default(),
        }
    }

    /// What the regime is decided on ([`Measured`]), beside the streamed layer
    /// count (`layers`), the ranking's misses at each depth of foresight
    /// (`needed`), and the guesses' stand rates as their one owner (`stands`)
    /// holds them: the first front row's and a verify pass's rows'; none before
    /// a pass is timed.
    pub fn measured(&mut self, layers: usize, needed: [[Option<magnitude_cost_model::Needed>; DEPTHS]; 2], stands: &Stands) -> Option<Measured> {
        let latest = self.latest?;
        let epoch = self.epoch();
        let all = self.all.read(epoch).map(Sums::over).unwrap_or_default();
        let routed = all.pass.routed;
        // Each depth's unused reads ahead per pass of each form that ran that
        // depth: exact-row lookahead hints (`depth == 0`, `Basis::Hint`,
        // always stored under `unused[0][0]`) run on ALL passes (both front
        // and verify), so their waste is divided by `all.passes` and shared by
        // both forms; deeper rows (`depth >= 1`) are divided by the passes of
        // their own form that ran that depth (`AheadUse`). Each of the 4
        // resources (`drive`, `link`, `cpu`, `memory`) attributes wasted reads
        // proportionally via its own per-read cost (`seconds / all.pass.reads`
        // when `all.pass.reads > 0.`).
        let unused_per_pass = |verify: bool| {
            let form = usize::from(verify);
            (0..DEPTHS)
                .map(|depth| {
                    if depth == 0 {
                        (all.passes > 0.).then(|| all.pass.unused[0][0] / all.passes)
                    } else {
                        let passes: f64 = all.by_depth[form][depth..DEPTHS].iter().sum();
                        (passes > 0.).then(|| all.pass.unused[form][depth] / passes)
                    }
                })
                .collect::<Vec<_>>()
        };
        let unused_front = unused_per_pass(false);
        let unused_verify = unused_per_pass(true);
        let paths = (all.pass.missed > 0.).then(|| {
            let terms = Terms { compute: 0., drive: all.pass.drive, link: all.pass.link, cpu: all.pass.cpu, memory: all.pass.memory };
            let scale = |unused: &[Option<f64>], per_read: f64| unused.iter().map(|count| count.map(|n| n * per_read)).collect();
            let path = |seconds: f64| {
                let per_read = if all.pass.reads > 0. { seconds / all.pass.reads } else { 0. };
                magnitude_cost_model::PathTerm {
                    seconds: seconds / all.pass.missed,
                    wasted: magnitude_cost_model::Waste {
                        front: scale(&unused_front, per_read),
                        verify: scale(&unused_verify, per_read),
                    },
                }
            };
            terms.each_path().into_iter().map(path).collect()
        });
        let plain = self.depth(false, 0, epoch).pass;
        let front_terms: [Option<Terms>; DEPTHS] =
            std::array::from_fn(|depth| self.depth(false, depth, epoch).terms());
        let verify_terms: [Option<Terms>; DEPTHS] =
            std::array::from_fn(|depth| self.depth(true, depth, epoch).terms());
        let terms = std::array::from_fn(|depth| front_terms[depth].or(verify_terms[depth]));
        Some(Measured {
            layers,
            depth: latest.depth,
            verifying: latest.verifies,
            terms,
            verify_terms,
            paths,
            stall: (plain.missed > 0.).then(|| plain.parked / plain.missed),
            needed,
            stand: stands.rate(1, Form::Front),
            accepts: std::array::from_fn(|depth| stands.rate(depth + 1, Form::Verify)),
            fresh: std::array::from_fn(|lag| (routed.layers[lag] > 0.).then(|| routed.fresh[lag] / routed.layers[lag])),
            held: (routed.picks > 0.).then(|| (routed.held / routed.picks).clamp(0., 1.)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::horizon::{Advance, Tier, Tokens};

    /// A bank-wide time with `records` tier-0 records (each `records` moves
    /// closes one bank-wide epoch).
    fn time_of(records: u64) -> Time {
        Time::new(&[vec![records]], 16)
    }

    /// Close one bank-wide epoch on `time` (for a single-group bank of `records`).
    fn turn_epoch(time: &Time, step: u64, records: u64) {
        let mut times = time.lock();
        times.register(step, Advance::Known(Tokens::new(1.0)));
        for _ in 0..records {
            times.moved(0, Tier::Compute);
        }
    }

    /// No pass timed, rolling on a bank of 2 tier-0 records.
    fn passes() -> (Passes, Time) {
        let time = time_of(2);
        (Passes::new(time.clone()), time)
    }

    /// Stand rates no verdict has judged.
    fn unjudged(time: Time) -> Stands {
        Stands::new(time)
    }

    /// A plain pass's time.
    fn plain(seconds: f64, parked: f64, misses: f64) -> Pass {
        Pass { rows: 1, depth: 0, seconds, parked, misses, drive: misses * 1e-3, ..Pass::default() }
    }

    /// A read ahead breaks even where its pipelined marginal bandwidth cost
    /// `w_net = (t_bw - min(t_bw, idle)).max(0)` on a binding drive pays
    /// against the demand stall `s_demand` it avoids: `p* = (w_net / s_demand).min(1)`.
    /// When prefetch achieves 100% hit rate (`misses == 0`), `demand_stall` is
    /// `None` and `ReadAdmission` falls back to `plain_stall` (`plain.parked / plain.missed`)
    /// so `break_even` never collapses to `1.0`.
    #[test]
    fn the_break_even_prices_waste_on_the_binding_term() {
        let bytes = 3_090_000_u64;
        // M4 Pro drive line: lambda = 0.14 ms, t_bw = 0.67 ms for 3.09 MB.
        let m4_line = ReadLine::new(0.14e-3, 0.67e-3 / bytes as f64).unwrap();
        let admission = |seconds: f64, parked: f64, misses: f64, drive: f64, line: Option<ReadLine>| {
            let (mut passes, _) = passes();
            passes.timed(Pass { rows: 1, depth: 0, seconds, parked, misses, missed: misses, drive, ..Pass::default() });
            passes.read_admission(line)
        };

        // 1. M4 Pro (bandwidth-saturated): t_bw = 0.67 ms, s_demand = 0.81 ms -> p* = 0.67 / 0.81 ≈ 0.83.
        let m4 = admission(0.20, 0.081, 100., 0.160, Some(m4_line));
        assert!((m4.break_even(bytes, 0.) - 0.67 / 0.81).abs() < 1e-9, "{}", m4.break_even(bytes, 0.));
        // Partial idle overlap: idle = 0.27 ms -> w_net = 0.40 ms -> p* = 0.40 / 0.81.
        assert!((m4.break_even(bytes, 0.27e-3) - 0.40 / 0.81).abs() < 1e-9);
        // Full idle overlap covers t_bw -> free (0.0).
        assert_eq!(m4.break_even(bytes, 0.67e-3), 0.);
        assert_eq!(m4.break_even(bytes, 1.0e-3), 0.);
        assert!((m4.terms.compute - 0.119).abs() < 1e-9, "{}", m4.terms.compute);

        // 2. GB10 (latency-bound): t_bw = 0.08 ms, lambda = 0.32 ms, s_demand = 0.40 ms -> p* = 0.20.
        let gb10_line = ReadLine::new(0.32e-3, 0.08e-3 / bytes as f64).unwrap();
        let gb10 = admission(0.07, 0.040, 100., 0.100, Some(gb10_line));
        assert!((gb10.break_even(bytes, 0.) - 0.20).abs() < 1e-9, "{}", gb10.break_even(bytes, 0.));

        // 3. GPU-bound: drive does not bind -> p* = 0.0.
        let resident_bound = admission(0.05, 0.018, 21., 0.021, Some(m4_line));
        assert_eq!(resident_bound.break_even(bytes, 0.), 0., "non-binding drive");
        assert_eq!(resident_bound.break_even(bytes, 0.4e-3), 0., "non-binding drive with idle");

        // 4. Nothing timed -> p* = 0.0.
        assert_eq!(passes().0.read_admission(Some(m4_line)).break_even(bytes, 0.), 0., "nothing timed");

        // 5. 100% prefetch hit rate in later epochs: plain pass had stall = 0.81 ms,
        //    then two epochs of front passes have misses == 0 (demand_stall == None)
        //    while drive still binds. `plain_stall` keeps p* = 0.67 / 0.81 (never 1.0!).
        let (mut p, time) = passes();
        p.timed(Pass { rows: 1, depth: 0, seconds: 0.20, parked: 0.081, misses: 100., missed: 100., drive: 0.160, ..Pass::default() });
        turn_epoch(&time, 1, 2);
        p.timed(Pass { rows: 2, depth: 1, seconds: 0.16, parked: 0.0, misses: 0., missed: 100., drive: 0.160, ..Pass::default() });
        turn_epoch(&time, 2, 2);
        p.timed(Pass { rows: 2, depth: 1, seconds: 0.16, parked: 0.0, misses: 0., missed: 100., drive: 0.160, ..Pass::default() });
        let full_hit = p.read_admission(Some(m4_line));
        assert_eq!(full_hit.demand_stall, None);
        assert!((full_hit.plain_stall.unwrap() - 0.81e-3).abs() < 1e-12);
        assert!((full_hit.break_even(bytes, 0.) - 0.67 / 0.81).abs() < 1e-9, "100% prefetch hit must not collapse to 1.0");

        // 6. layer_stall prices queued demand reads as reads * demand_stall.or(plain_stall).
        let (mut p, _) = passes();
        p.timed(plain(0.1, 0.05, 50.));
        let cost = p.read_admission(None);
        assert!((cost.layer_stall(3.0) - 3.0 * 1e-3).abs() < 1e-12);
        assert_eq!(cost.layer_stall(0.0), 0.0);
        assert_eq!(ReadAdmission::default().layer_stall(4.0), 0.0);
    }

    /// The regime's figures are the window's: each kind's terms apart; each
    /// term's seconds a miss over every pass; the reads ahead nothing used
    /// each depth's rows led, over the passes that ran the depth, at a
    /// read's seconds; the stall of the plain passes' misses; the latest
    /// pass says which runs; the stand rates their owner's, unjudged none.
    /// Rolling two epochs with new passes drops old global sums, while an
    /// unrun depth keeps its last measured compute.
    #[test]
    fn the_regime_is_measured_over_the_window_by_kind() {
        let (mut passes, time) = passes();
        let stands = unjudged(time.clone());
        assert_eq!(passes.measured(48, [[None; DEPTHS]; 2], &stands), None);
        passes.timed(Pass { missed: 100., reads: 100., unused: [[2., 0., 0., 0., 0., 0., 0., 0.], [0.; DEPTHS]], ..plain(0.13, 0.1, 100.) });
        let deep = Pass { rows: 3, depth: 2, seconds: 0.1, parked: 0.06, misses: 60., missed: 80., reads: 80., drive: 0.08, ..Pass::default() };
        passes.timed(Pass { unused: [[2., 6., 4., 0., 0., 0., 0., 0.], [0.; DEPTHS]], ..deep });
        let measured = passes.measured(48, [[Some(magnitude_cost_model::Needed::Executed(90.)); DEPTHS]; 2], &stands).expect("passes timed");
        let compute = |depth: usize| measured.terms[depth].map(|terms| terms.compute);
        assert_eq!((measured.layers, measured.depth, measured.needed[0][0], measured.stand, measured.accepts), (48, 2, Some(magnitude_cost_model::Needed::Executed(90.)), None, [None; FRONT_DEPTHS]));
        assert!((compute(0).unwrap() - 0.03).abs() < 1e-12 && (compute(2).unwrap() - 0.04).abs() < 1e-12, "{measured:?}");
        assert_eq!(measured.terms[1], None, "no pass of depth 1: {measured:?}");
        let paths = measured.paths.as_ref().expect("a miss counted");
        assert_eq!(paths.len(), 4, "the drive, the link, the CPU, host memory");
        assert!((paths[0].seconds - 0.18 / 180.).abs() < 1e-15 && paths[1..].iter().all(|path| path.seconds == 0.), "{paths:?}");
        let wasted = paths[0].wasted.front.iter().map(|wasted| wasted.map(|seconds| (seconds * 1e6).round())).collect::<Vec<_>>();
        assert_eq!(wasted[..4], [Some(2000.), Some(6000.), Some(4000.), None], "depth 0 over both passes, 1 and 2 over the deep one, 3 none");
        assert!(paths[1].wasted.front.iter().all(|w| w.unwrap_or(0.) == 0.), "link has zero busy seconds so zero waste");
        assert!((measured.stall.unwrap() - 0.1 / 100.).abs() < 1e-15, "the plain pass's parking a miss: {measured:?}");
        // Two bank-wide epochs of plain passes replace `all` and `by_depth[0]`,
        // while unrun depth 2 keeps its last measured compute.
        turn_epoch(&time, 1, 2);
        passes.timed(plain(0.05, 0.01, 1.));
        turn_epoch(&time, 2, 2);
        passes.timed(plain(0.05, 0.01, 1.));
        let measured = passes.measured(48, [[None; DEPTHS]; 2], &stands).expect("passes timed");
        assert_eq!(measured.depth, 0);
        assert!((measured.terms[2].unwrap().compute - 0.04).abs() < 1e-12, "unrun depth keeps its last measured compute");
        assert!(measured.paths.is_none() && measured.stall.is_none(), "no miss of the memory the GPU computes from counted: {measured:?}");
        assert_eq!(passes.latest().map(|(rows, _)| rows), Some(1));
        assert!((measured.terms[0].unwrap().compute - 0.04).abs() < 1e-9, "{measured:?}");
    }

    /// Front and verify passes keep separate compute sums by `(form, depth)`
    /// rather than conflating them; when verify has no samples yet at a depth,
    /// `terms_for(true, depth)` extrapolates from the front pass at that depth.
    /// The reads ahead nothing used are tracked per form apart, and the latest
    /// pass says whether it verified.
    #[test]
    fn front_and_verify_compute_are_keyed_by_form_and_depth() {
        let (mut passes, time) = passes();
        let front = Pass { rows: 2, depth: 1, seconds: 0.05, parked: 0.01, misses: 10., missed: 10., reads: 10., drive: 0.01, unused: [[0., 4., 0., 0., 0., 0., 0., 0.], [0.; DEPTHS]], ..Pass::default() };
        passes.timed(front);
        // Depth 2 only has a front pass: verify extrapolates from front.
        passes.timed(Pass { rows: 3, depth: 2, seconds: 0.06, parked: 0.01, ..front });
        // Depth 1 has both front (compute = 0.04) and verify (compute = 0.06).
        passes.timed(Pass { verifies: true, seconds: 0.07, unused: [[0.; DEPTHS], [0., 2., 0., 0., 0., 0., 0., 0.]], ..front });
        let measured = passes.measured(48, [[Some(magnitude_cost_model::Needed::Executed(10.)); DEPTHS]; 2], &unjudged(time)).expect("passes timed");
        assert!(measured.verifying, "{measured:?}");
        assert!((measured.terms[1].unwrap().compute - 0.04).abs() < 1e-12, "front depth 1 compute alone: {measured:?}");
        assert!((measured.verify_terms[1].unwrap().compute - 0.06).abs() < 1e-12, "verify depth 1 compute alone: {measured:?}");
        assert_eq!(measured.verify_terms[2], None, "no verify pass at depth 2");
        assert!((measured.terms_for(true, 2).unwrap().compute - 0.05).abs() < 1e-12, "verify extrapolates from front at unmeasured depth 2");
        let wasted = &measured.paths.as_ref().expect("a miss counted")[0].wasted;
        let (front_w, verify_w) = (wasted.front[1].expect("a front pass ran depth 1"), wasted.verify[1].expect("a verify pass ran depth 1"));
        assert!((front_w - 4. * 1e-3).abs() < 1e-15, "the front pass's four unused reads over its two passes (depth 1 and 2): {front_w}");
        assert!((verify_w - 2. * 1e-3).abs() < 1e-15, "the verify pass's two unused reads, over it alone: {verify_w}");
    }

    /// Wasted reads are attributed proportionally across all 4 resources
    /// (`drive`, `link`, `cpu`, `memory`) via each resource's per-read cost
    /// (`seconds / all.pass.reads`) when `reads > 0`.
    #[test]
    fn wasted_reads_are_attributed_proportionally_across_all_four_resources() {
        let (mut passes, time) = passes();
        passes.timed(Pass {
            rows: 2,
            depth: 1,
            seconds: 0.10,
            parked: 0.02,
            misses: 10.,
            missed: 20.,
            reads: 10.,
            drive: 0.020,  // 2.0 ms / read
            link: 0.015,   // 1.5 ms / read
            cpu: 0.005,    // 0.5 ms / read
            memory: 0.010, // 1.0 ms / read
            unused: [[2., 4., 0., 0., 0., 0., 0., 0.], [0.; DEPTHS]],
            ..Pass::default()
        });
        let measured = passes
            .measured(48, [[Some(magnitude_cost_model::Needed::Executed(20.)); DEPTHS]; 2], &unjudged(time))
            .expect("passes timed");
        let paths = measured.paths.as_ref().expect("misses counted");
        assert_eq!(paths.len(), 4);
        let expected_per_read = [2.0e-3, 1.5e-3, 0.5e-3, 1.0e-3];
        for (idx, per_read) in expected_per_read.into_iter().enumerate() {
            let w0 = paths[idx].wasted.front[0].expect("depth 0 waste");
            let w1 = paths[idx].wasted.front[1].expect("depth 1 waste");
            assert!((w0 - 2. * per_read).abs() < 1e-15, "resource {idx} depth 0: {w0}");
            assert!((w1 - 4. * per_read).abs() < 1e-15, "resource {idx} depth 1: {w1}");
            assert_eq!(paths[idx].wasted.front[2], None, "unmeasured depth 2 stays None");
        }
    }

    /// Exact-row lookahead hints (`depth == 0`, `unused[0][0]`) run on both
    /// front and verify passes, so their waste is divided by all passes (not
    /// front passes alone) and shared by `wasted.front[0]` and `wasted.verify[0]`.
    #[test]
    fn exact_row_hint_waste_divides_by_all_passes_not_front_passes_alone() {
        let (mut passes, time) = passes();
        let base = Pass { rows: 2, depth: 1, seconds: 0.05, parked: 0.01, misses: 10., missed: 10., reads: 10., drive: 0.01, ..Pass::default() };
        // 1 front pass + 9 verify passes, each with 2 unused exact-row hints in `unused[0][0]`.
        passes.timed(Pass { verifies: false, unused: [[2., 0., 0., 0., 0., 0., 0., 0.], [0.; DEPTHS]], ..base });
        for _ in 0..9 {
            passes.timed(Pass { verifies: true, unused: [[2., 0., 0., 0., 0., 0., 0., 0.], [0.; DEPTHS]], ..base });
        }
        let measured = passes.measured(48, [[Some(magnitude_cost_model::Needed::Executed(10.)); DEPTHS]; 2], &unjudged(time)).expect("passes timed");
        let wasted = &measured.paths.as_ref().expect("a miss counted")[0].wasted;
        let front_0 = wasted.front[0].expect("depth 0 waste");
        let verify_0 = wasted.verify[0].expect("depth 0 waste");
        // 20 unused reads / 10 total passes * 1 ms = 2 ms (NOT 20 / 1 front pass = 20 ms!).
        assert!((front_0 - 2. * 1e-3).abs() < 1e-15, "front[0] must divide by all passes, got {front_0}");
        assert!((verify_0 - 2. * 1e-3).abs() < 1e-15, "verify[0] must share exact-row hint waste, got {verify_0}");
    }

    /// The three regimes the terms tell apart, each a pass's busy seconds:
    /// on unified memory the drive (120 ms of reads under 30 ms of pass)
    /// or the GPU (a 33 ms pass over 2 ms of reads) binds. On a discrete
    /// host whose RAM holds the model nothing reads the drive: a 64 ms pass
    /// whose misses' service held it 30 ms (its reads in place 26 ms and
    /// its moves 18 ms on the link, the CPU 12 ms) computes 34 ms, and the
    /// link binds at 44 ms. With RAM short of the model the drive's 90 ms
    /// binds again. The resident compute excludes both what the pass parked
    /// and what its service held it, so a discrete host's C is its GPU's
    /// alone; and the copy's shares are the CPU's and the reads in place's.
    #[test]
    fn the_terms_bind_by_topology() {
        let binding = |pass: Pass| {
            let (mut passes, _) = passes();
            passes.timed(pass);
            let all = passes.all().terms().expect("a timed pass");
            (all.compute, all.paths(), passes.beside(), passes.moves())
        };
        let unified = Pass { rows: 1, seconds: 0.150, parked: 0.120, misses: 60., drive: 0.120, ..Pass::default() };
        let (compute, paths, _, _) = binding(unified);
        assert!((compute - 0.030).abs() < 1e-12 && (paths - 0.120).abs() < 1e-12, "the drive binds");
        let (compute, paths, ..) = binding(Pass { seconds: 0.035, parked: 0.002, drive: 0.002, ..unified });
        assert!(compute > paths, "the GPU binds: {compute} against {paths}");
        let fits = Pass { rows: 1, seconds: 0.064, served: 0.030, link: 0.044, moved: 0.018, moved_records: 9., cpu: 0.012, memory: 0.010, ..Pass::default() };
        let (compute, paths, beside, moves) = binding(fits);
        assert!((compute - 0.034).abs() < 1e-12 && (paths - 0.044).abs() < 1e-12, "the link binds: {compute} {paths}");
        assert!((beside.cpu - 0.012 / 0.064).abs() < 1e-12 && (beside.link - 0.026 / 0.064).abs() < 1e-12, "{beside:?}");
        assert_eq!(moves, Some((0.018, 9.)));
        let (_, paths, ..) = binding(Pass { seconds: 0.150, parked: 0.086, drive: 0.090, misses: 45., ..fits });
        assert!((paths - 0.090).abs() < 1e-12, "the drive binds again: {paths}");
    }

    /// On a discrete host whose RAM holds every expert (`parked == 0`,
    /// `served > 0`), `stall` excludes `served` (host-tier link/CPU service
    /// is already accounted in `paths` and is not hidden by front-row file
    /// read-ahead), so `Regime` does not double-count `served` or falsely
    /// select front rows.
    #[test]
    fn discrete_host_ram_complete_does_not_double_count_served_in_stall() {
        let (mut passes, time) = passes();
        let stands = unjudged(time);
        let fits = Pass {
            rows: 1,
            depth: 0,
            seconds: 0.064,
            parked: 0.,
            misses: 0.,
            missed: 40.,
            served: 0.030,
            link: 0.044,
            moved: 0.018,
            moved_records: 9.,
            cpu: 0.012,
            memory: 0.010,
            ..Pass::default()
        };
        passes.timed(fits);
        let measured = passes
            .measured(48, [[Some(magnitude_cost_model::Needed::Executed(40.)); DEPTHS]; 2], &stands)
            .expect("passes timed");
        assert_eq!(measured.stall, Some(0.), "no file-miss park stall when Host RAM holds every expert");
        let regime = magnitude_cost_model::Regime {
            resident: (0..DEPTHS).map(|k| 0.034 + 0.002 * k as f64).collect(),
            needed: magnitude_cost_model::NeededByForm::both(vec![magnitude_cost_model::Needed::Executed(40.); DEPTHS]),
            stand: Some(0.9),
            accepts: vec![None; FRONT_DEPTHS],
            stall: measured.stall,
            overhead: None,
            paths: measured.paths.expect("VRAM misses accounted in paths"),
            routes: magnitude_cost_model::VerifyRoutes::default(),
        };
        assert_eq!(regime.depth(), 0, "100% Host RAM with flat U(k) must stay plain (depth 0), not falsely front-row");
    }

    /// The routes are the window's: each lag's fresh experts over the
    /// layers that saw it, the held share over every pick; a lag no pass
    /// saw ends the measured routes, beyond which they are independent.
    #[test]
    fn the_routes_are_measured_over_the_window() {
        let (mut passes, time) = passes();
        let stands = unjudged(time.clone());
        let routed = |fresh: f64, picks: f64, held: f64| {
            let mut routed = Routed { picks, held, ..Routed::default() };
            routed.fresh[..2].fill(fresh * 48.);
            routed.layers[..2].fill(48.);
            routed
        };
        passes.timed(Pass { routed: routed(8., 480., 120.), ..plain(0.1, 0.05, 50.) });
        passes.timed(Pass { routed: routed(6., 480., 200.), ..plain(0.1, 0.05, 50.) });
        let measured = passes.measured(48, [[None; DEPTHS]; 2], &stands).expect("passes timed");
        assert_eq!(measured.fresh[..3], [Some(7.), Some(7.), None]);
        assert!((measured.held.unwrap() - 320. / 960.).abs() < 1e-12, "{measured:?}");
        assert_eq!(measured.routes(), magnitude_cost_model::FrontRoutes { fresh: vec![7., 7.], held: measured.held });
        turn_epoch(&time, 1, 2);
        passes.timed(plain(0.05, 0.01, 1.));
        turn_epoch(&time, 2, 2);
        passes.timed(plain(0.05, 0.01, 1.));
        let measured = passes.measured(48, [[None; DEPTHS]; 2], &stands).expect("passes timed");
        assert_eq!((measured.fresh, measured.held), ([None; LAGS], None), "the routed passes left the window");
    }
}
