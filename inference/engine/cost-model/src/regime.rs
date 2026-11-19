//! The form and depth of a streamed decode pass: the regime's pass.
//!
//! A pass at depth `k` runs its exact row and `k` guessed rows on the
//! sequence's chain of guesses, in one of two forms (§7.1). Each reads the
//! same experts per token it emits; they differ in what the guessed rows
//! compute and what the pass emits.
//!
//! - **Front rows** only name the next `k` tokens' experts ahead of need:
//!   each skips what its step does not hold, and is recomputed exact by a
//!   later pass, so the pass emits one token and takes
//!
//!   `T(k) = max(C(1 + k), max_i (U_i(k) + W_i(k))·t_i) + S(k)`
//!
//!   the GPU's pass beside every other term ([`PathTerm`], one each of
//!   [`crate::Terms::each_path`]: the drive's, and beside a host tier the
//!   link's, the CPU's and host memory's), and the stall of the misses
//!   nothing named in time ([`Regime::time`]).
//! - **Verify rows** wait for every expert their tokens route to, and are
//!   accepted wherever their guesses stand: the pass emits `1 + a(k)`
//!   tokens, `a(k) = Σ_{j≤k} Π_{i≤j} σ_i` of its rows' guesses standing,
//!   `σ_i` the stand rate of row `i` given the rows before it stood. It
//!   reads every expert its rows route to, `U_i(k)·t_i` a token emitted,
//!   none missed unnamed (no `S`: every row waits); `U` counts the reads of
//!   its rows past the first fall among them, so its own waste `W'_i(k)` is
//!   only the moves no row of it reads, its rows' lookahead hints' that
//!   nothing used. Its verdict decides the next pass's tokens, so
//!   no pass is queued behind it: the host's turn between it and the next
//!   pass, which drafts the next guesses, is time the GPU idles, `O`. A
//!   token takes ([`Regime::verify_time`])
//!
//!   `V(k) = (max(C(1 + k), max_i ((1 + a(k))·U_i(k) + W'_i(k))·t_i) + O) / (1 + a(k))`
//!
//! The pass is the form and depth that take least a token emitted
//! ([`Regime::pass`]), chosen afresh at every pass the host builds.
//!
//! - **U is the ranking's, replayed at every depth.** `U(k)` is the misses
//!   of the memory the GPU computes from a token emitted, where its ranking
//!   foresees `k` tokens ahead: the live placement's own shadows at each
//!   depth, fed the step's routes as they come, counting every measure in
//!   the tokens the steps emitted (`Reuse::shadows`). A front pass emits
//!   one token, so its `U` is its pass's; a verify pass's misses are
//!   `(1 + a(k))·U`. Naming
//!   ahead lowers U only by what it keeps placed; a move named early is a
//!   move still. Each such miss holds each term its measured seconds
//!   (`U_i·t_i = U·seconds_i`): the tiers below are placed by no foresight,
//!   so the share of the misses each path carries is the one measured.
//! - **W is the moves nothing used,** by the depth of the row whose name led
//!   them (an exact row's hints at 0): a fallen guess's routes and the
//!   lookahead hints of every row, measured, of each form apart ([`Waste`]):
//!   a front pass's front and exact rows', `W`, and a verify pass's verify
//!   rows' lookahead hints, `W'` (their routes are its own reads, in `U`).
//!   `W_i(k)` and `W'_i(k)` each sum the depths a depth-`k` pass runs,
//!   `0..=k`.
//! - **S is the stall of the misses nothing named in time.** At depth 0
//!   every miss is one; at depth `k ≥ 1` the previous pass's first front
//!   row named the step's experts, and its misses go unnamed where that
//!   row's guess fell, `1 − σ_1`. Each costs the stall a miss held a plain
//!   pass. One pass of lead suffices: a move takes far less than a pass,
//!   and where a path is saturated its own term binds, not S.
//! - **Depth 0 is one pass in either form,** the plain pass, which is
//!   queued behind its predecessor as a front pass is: it is priced as
//!   front. A verify pass runs as deep as the load verifies
//!   ([`Regime::accepts`]).
//! - **One prior, the bound of what a depth saves.** A depth no pass has
//!   run has no verdict and no waste measured: its stands are priced as
//!   standing (`σ = 1`) and its waste as none, the most such rows could
//!   save, so the regime tries the depth where it pays on that bound, and
//!   the first passes there measure it. The shadows price its U on the true
//!   routes, a lower bound on its `U` and so the same bound on what it
//!   saves: the one-time exploration bound, which the first pass that runs
//!   the depth replaces for good with the routes the run measures
//!   (`Reuse::shadows`). A stall no plain pass measured is
//!   priced as none, which only a plain pass, the one that measures it,
//!   gains by; and a host turn no verify pass measured as none, which only
//!   a verify pass, the one that measures it, gains by.
//!
//! No change of depth is priced, since none costs a pass: a queued pass
//! only gets shallower (`ExecutorDomain::queue_successor`), and a pass gets
//! deeper only where the host builds it for its own reasons. No change of
//! form is priced either: the pass that leaves front rows for verify rows
//! is one the host builds, whose turn is the verify form's `O`.
//! Where no path term or stall exceeds what the rows add to the GPU's, and
//! no stand pays a verify pass's turn, the depth is 0 and the pass is plain.

/// One term besides the GPU's pass ([`crate::Terms::each_path`]): `seconds`,
/// what a miss of the memory the GPU computes from holds it (`t_i` times
/// the moves over it a miss makes); and `wasted`, a pass's seconds on it
/// of moves nothing used, by its form and the depth of the rows that led
/// them ([`Waste`]).
#[derive(Clone, Debug, PartialEq)]
pub struct PathTerm {
    pub seconds: f64,
    pub wasted: Waste,
}

/// A pass's seconds on one term of moves nothing used, of each form: at
/// index `j`, `front`, a front pass's that its rows of depth `j` led (its
/// exact rows' hints at 0), and `verify`, a verify pass's that its verify
/// rows of depth `j` led (none at 0, which a verify pass shares with the
/// plain pass's exact row); none where no pass of that form ran that
/// depth, for every depth the decode form holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Waste {
    pub front: Vec<Option<f64>>,
    pub verify: Vec<Option<f64>>,
}

/// A depth's misses per token (`Regime::needed`):
/// - `Shadow(u)`: true-route shadow before the depth has executed (misses per
///   true row, excluding discarded verify rows). A verify pass of `1 + k` rows
///   reads `(1 + k) * u` experts (`((1 + k) / emitted) * u` per emitted token).
/// - `Executed(u)`: measured once the depth has executed (`Shadows::executed`),
///   which already divides all `1 + k` rows' misses by `emitted`, so a verify
///   pass reads `emitted * u` experts (`u` per emitted token).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Needed {
    Shadow(f64),
    Executed(f64),
}

impl Needed {
    /// The numeric misses per token.
    pub fn misses(self) -> f64 {
        match self {
            Self::Shadow(u) | Self::Executed(u) => u,
        }
    }
}

/// Each form's misses per token across depths (`Regime::needed`), mirroring
/// [`Waste`]: `front` (`Form::Front`) and `verify` (`Form::Verify`), each
/// holding either the shared unexecuted true-route shadow (`Needed::Shadow`)
/// or that form's own measured misses per emitted token (`Needed::Executed`).
#[derive(Clone, Debug, PartialEq)]
pub struct NeededByForm {
    pub front: Vec<Needed>,
    pub verify: Vec<Needed>,
}

impl NeededByForm {
    /// Both forms initialized with the same `needed` vector.
    pub fn both(needed: Vec<Needed>) -> Self {
        Self { front: needed.clone(), verify: needed }
    }
}

/// What the pass is decided on, each a pass's mean over the latest
/// measured decode passes: at index `k`, `resident`, the resident pass's
/// compute at depth `k`, `C(1 + k)` (measured where the window holds such
/// passes, else priced), for every depth the launch may run; at index `k`,
/// `needed`, each form's misses a token emitted makes where its ranking
/// foresees `k` tokens ahead (`U(k)`, [`NeededByForm`]), for every depth the
/// decode form holds; `stand`, a front
/// pass's first row's stand rate (`σ_1`, none before a verdict); at index
/// `i − 1`, `accepts`, a verify pass's row `i`'s stand rate given the rows
/// before it stood (`σ_i`, none before a verdict), for every depth the load
/// verifies; `stall`, the seconds a miss held a plain pass past its compute
/// (none before one is measured); `overhead`, the seconds from a verify
/// pass's completion to the next pass's start (`O`, none before one is
/// measured); and `paths`, every term besides the GPU's.
#[derive(Clone, Debug, PartialEq)]
pub struct Regime {
    pub resident: Vec<f64>,
    pub needed: NeededByForm,
    pub stand: Option<f64>,
    pub accepts: Vec<Option<f64>>,
    pub stall: Option<f64>,
    pub overhead: Option<f64>,
    pub paths: Vec<PathTerm>,
}

/// The form and depth of a pass the host builds ([`Regime::pass`]): `k`
/// front rows, or `k` verify rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Front(usize),
    Verify(usize),
}

impl Form {
    /// The guessed rows it runs beside its exact row.
    pub fn depth(self) -> usize {
        match self {
            Self::Front(depth) | Self::Verify(depth) => depth,
        }
    }
}

impl Regime {
    /// The front rows a front pass wants: the depth whose pass takes least
    /// ([`Self::time`]), the shallowest of equals. A pass queued behind
    /// another is front, and runs it.
    pub fn depth(&self) -> usize {
        (0..self.resident.len()).fold(0, |best, k| if self.time(k) < self.time(best) { k } else { best })
    }

    /// The pass the host builds next: the form and depth whose pass takes
    /// least a token emitted ([`Self::token_time`]), the shallowest of
    /// equals, and front of equals at a depth. Verify passes range over the
    /// depths from 1 the launch may run and the load verifies.
    pub fn pass(&self) -> Form {
        let verifies = self.accepts.len().min(self.resident.len().saturating_sub(1));
        let forms = (0..self.resident.len()).flat_map(|k| std::iter::once(Form::Front(k)).chain((1..=verifies).contains(&k).then_some(Form::Verify(k))));
        forms.fold(Form::Front(0), |best, form| if self.token_time(form) < self.token_time(best) { form } else { best })
    }

    /// The time a token `form`'s pass emits takes: a front pass's whole
    /// time ([`Self::time`]), as it emits one; a verify pass's
    /// ([`Self::verify_time`]).
    pub fn token_time(&self, form: Form) -> f64 {
        match form {
            Form::Front(k) => self.time(k),
            Form::Verify(k) => self.verify_time(k),
        }
    }

    /// `T(k)`: the time of a front pass at depth `k`.
    pub fn time(&self, k: usize) -> f64 {
        let depths = self.priced(k);
        // The share of a step's misses nothing named in time.
        let unnamed = match k {
            0 => 1.,
            _ => 1. - rate(self.stand),
        };
        let needed = self.needed.front[k].misses();
        let paths = self.paths.iter().map(|path| path.time(needed, k, depths));
        paths.fold(self.resident[k], f64::max) + needed * unnamed * self.stall.unwrap_or(0.)
    }

    /// `V(k)`: the time a token of a verify pass at depth `k ≥ 1` takes,
    /// its pass's over the tokens it emits.
    pub fn verify_time(&self, k: usize) -> f64 {
        let depths = self.priced(k);
        assert!(
            (1..=self.accepts.len()).contains(&k) && self.overhead.is_none_or(|overhead| overhead >= 0.),
            "a regime prices verify rows at a depth its load verifies: {k} of {self:?}"
        );
        // The tokens it emits: its exact row's, and each row's whose guess
        // stands with every guess before it. An unmeasured deeper row carries
        // forward the last measured shallower conditional stand rate (1 before
        // any depth is measured), since conditional acceptance decays with
        // depth rather than jumping back to 1.
        let emitted = 1. + self.accepts[..k]
            .iter()
            .scan((1., 1.), |(standing, bound), &accepts| {
                let row = match accepts {
                    Some(accepts) => {
                        let measured = rate(Some(accepts));
                        *bound = measured;
                        measured
                    }
                    None => *bound,
                };
                *standing *= row;
                Some(*standing)
            })
            .sum::<f64>();
        // An unexecuted shadow measures misses per true row, whereas a verify
        // pass runs and reads all `1 + k` rows (including discarded rows past
        // the first fall). Once executed, `needed.verify[k]` already divides
        // all `1 + k` rows' misses by `emitted`.
        let (rows, needed) = match self.needed.verify[k] {
            Needed::Shadow(u) => ((1 + k) as f64, u),
            Needed::Executed(u) => (emitted, u),
        };
        let read = self.paths.iter().map(|path| rows * path.read(needed, depths) + sum(&path.wasted.verify[..=k])).fold(0., f64::max);
        // Before the host turn is measured, 0 is its lower bound: verify
        // is tried once at that bound, after which the measured host turn
        // replaces it.
        (self.resident[k].max(read) + self.overhead.unwrap_or(0.)) / emitted
    }

    /// The decode form's depths, once `k` is checked among those the
    /// launch may run and every figure is checked a rate or a time.
    fn priced(&self, k: usize) -> usize {
        let depths = self.needed.front.len();
        assert!(
            self.needed.verify.len() == depths
                && k < self.resident.len()
                && self.resident.len() <= depths
                && self.accepts.len() < depths
                && self.stall.is_none_or(|stall| stall >= 0.),
            "a regime prices depth {k} among those its launch may run: {self:?}"
        );
        depths
    }
}

/// A stand rate as priced: as measured, standing before a verdict (the
/// prior, [`Regime`]).
fn rate(stand: Option<f64>) -> f64 {
    let stand = stand.unwrap_or(1.);
    assert!((0. ..=1.).contains(&stand), "a stand rate: {stand}");
    stand
}

impl PathTerm {
    /// `(U(k) + W(k))·t` on this term, at `needed` misses a token at depth
    /// `k`, of a decode form `depths` deep: a front pass's, which emits one.
    fn time(&self, needed: f64, k: usize, depths: usize) -> f64 {
        self.read(needed, depths) + sum(&self.wasted.front[..=k])
    }

    /// `U·t` on this term, at `needed` misses a token emitted, of a decode
    /// form `depths` deep.
    fn read(&self, needed: f64, depths: usize) -> f64 {
        assert!(self.wasted.front.len() == depths && self.wasted.verify.len() == depths && self.seconds >= 0., "a term priced at every depth: {self:?}");
        needed * self.seconds
    }
}

/// Waste's seconds over depths, an unmeasured depth's as none (its bound,
/// [`Regime`]).
fn sum(wasted: &[Option<f64>]) -> f64 {
    wasted.iter().map(|wasted| wasted.unwrap_or(0.)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Code's measured pass (multi-front-cost.md §3): C(1..8).
    const CODE: [f64; 8] = [0.0334, 0.0356, 0.0375, 0.0396, 0.0419, 0.0445, 0.0475, 0.0508];

    /// A term of `seconds` a miss, wasting `unused(j)` moves of `seconds` a
    /// pass led by depth `j`.
    fn path(seconds: f64, unused: impl Fn(usize) -> Option<f64>) -> PathTerm {
        PathTerm { seconds, wasted: Waste { front: (0..8).map(|depth| unused(depth).map(|moves| moves * seconds)).collect(), verify: vec![None; 8] } }
    }

    /// A regime of code's pass over `paths`, needing `needed(k)` misses a
    /// step at depth `k`, the first row standing `stand`, no stall, and no
    /// verify pass judged or timed.
    fn regime(needed: impl Fn(usize) -> f64, stand: Option<f64>, paths: Vec<PathTerm>) -> Regime {
        Regime { resident: CODE.to_vec(), needed: NeededByForm::both((0..8).map(|k| Needed::Executed(needed(k))).collect()), stand, accepts: vec![None; 7], stall: None, overhead: None, paths }
    }

    /// A drive-bound decode (245 misses of 1 ms a step plain, the ranking
    /// keeping 4 more placed at each depth it foresees) whose rows waste
    /// little runs every front row the decode form holds.
    #[test]
    fn a_drive_bound_decode_whose_names_keep_experts_placed_runs_every_row() {
        let drive = path(1e-3, |_| Some(0.5));
        assert_eq!(regime(|k| 245. - 4. * k as f64, Some(0.997), vec![drive]).depth(), 7);
    }

    /// A GPU-bound decode (1.5 ms of moves against a 33 ms pass) runs no
    /// row, however its guesses stand and before any is judged.
    #[test]
    fn a_gpu_bound_decode_runs_no_row() {
        for stand in [Some(0.95), Some(0.3), None] {
            assert_eq!(regime(|_| 1.5, stand, vec![path(1e-3, |_| None)]).depth(), 0, "at {stand:?}");
        }
    }

    /// Rows whose names waste more moves than they keep placed stop the
    /// depth: on code's measured run (245.3 moves plain, 214.6 at seven,
    /// 115.5 unused a token, about 16 a depth, the hints' 15.5% precision),
    /// each row saves about 4.4 moves and wastes 16, so the pass runs none;
    /// where the falls' and hints' waste starts only past depth two, it
    /// runs two.
    #[test]
    fn waste_past_what_a_row_keeps_placed_stops_the_depth() {
        let needed = |k: usize| 245.3 - (245.3 - 214.6) * k as f64 / 7.;
        let code = path(1e-3, |depth| Some(if depth == 0 { 3.5 } else { 16. }));
        assert_eq!(regime(needed, Some(0.997), vec![code]).depth(), 0);
        let late = regime(needed, Some(0.997), vec![path(1e-3, |depth| Some(if depth <= 2 { 0. } else { 16. }))]);
        assert_eq!(late.depth(), 2, "{:?}", (0..8).map(|k| late.time(k)).collect::<Vec<_>>());
    }

    /// A discrete host's terms, the drive's to host memory and the link's
    /// to the device, each a miss's share of its seconds: the busiest
    /// decides. Bound by the link (1 ms a miss, against the drive's 0.2 ms,
    /// a tenth of the misses reaching it at 2 ms), the depth is priced as a
    /// unified drive's would be; bound by the drive (1.5 ms a miss, its
    /// rows past two wasting 8 reads each), the drive's waste stops the
    /// depth at two whatever the link's.
    #[test]
    fn a_discrete_host_prices_the_depth_on_its_busiest_term() {
        let needed = |k: usize| 150. - 4. * k as f64;
        let link = path(1e-3, |_| Some(0.5));
        let linked = regime(needed, Some(0.997), vec![path(0.2e-3, |_| Some(1.)), link.clone()]);
        let alone = regime(needed, Some(0.997), vec![link]);
        assert_eq!(linked.depth(), 7);
        assert_eq!((0..8).map(|k| linked.time(k)).collect::<Vec<_>>(), (0..8).map(|k| alone.time(k)).collect::<Vec<_>>(), "the drive never binds");
        let drive = path(1.5e-3, |depth| Some(if depth <= 2 { 0. } else { 8. }));
        let driven = regime(|k| 100. - 2. * k as f64, Some(0.997), vec![drive, path(0.3e-3, |_| Some(1.))]);
        assert_eq!(driven.depth(), 2, "{:?}", (0..8).map(|k| driven.time(k)).collect::<Vec<_>>());
    }

    /// Naming in time pays where the GPU binds: 30 misses of 1 ms a step
    /// under a 33.4 ms pass, each unnamed miss stalling 0.5 ms, cost the
    /// plain pass 15 ms of stall; one row whose guess stands 0.9 leaves a
    /// tenth of them unnamed (1.5 ms) for 2.2 ms more of pass, so the pass
    /// runs one row; a guess standing 0.1 leaves 13.5 ms of stall, and it
    /// runs none. No plain pass measured, no stall is priced, and it runs
    /// none, the one pass that measures it.
    #[test]
    fn misses_named_in_time_pay_against_the_rows_they_cost() {
        let stalled = |stand: f64, stall: Option<f64>| Regime { stall, ..regime(|_| 30., Some(stand), vec![path(1e-3, |_| Some(0.))]) }.depth();
        assert_eq!((stalled(0.9, Some(0.5e-3)), stalled(0.1, Some(0.5e-3))), (1, 0));
        assert_eq!(stalled(0.9, None), 0);
    }

    /// A depth no pass ran is priced at its bound: its waste as none and
    /// its stands as standing, so a drive-bound decode whose ranking keeps
    /// more placed deeper tries it; its measured waste then prices it.
    #[test]
    fn an_unmeasured_depth_is_priced_at_its_bound() {
        let needed = |k: usize| 245. - 4. * k as f64;
        assert_eq!(regime(needed, None, vec![path(1e-3, |depth| (depth <= 1).then_some(0.5))]).depth(), 7);
        assert_eq!(regime(needed, Some(0.997), vec![path(1e-3, |depth| Some(if depth <= 1 { 0.5 } else { 16. }))]).depth(), 1);
    }

    /// A regime of code's pass verifying seven rows, each standing
    /// `accepts` given the rows before it stood, its host turn `overhead`.
    fn verifying(regime: Regime, accepts: Option<f64>, overhead: Option<f64>) -> Regime {
        Regime { accepts: vec![accepts; 7], overhead, ..regime }
    }

    /// Prose is compute-bound (1.3 misses of 1 ms a step under a 33.4 ms
    /// pass): a front row saves nothing, so its front pass is plain, but
    /// verify rows standing 0.8 each emit 1 + 0.8 + 0.64 + … tokens a pass
    /// against a 5 ms host turn: six rows emit 3.95 for 52.5 ms, 13.3 ms a
    /// token, the least of any depth (seven take 13.4). A load verifying
    /// three rows runs three, the deepest it verifies.
    #[test]
    fn a_compute_bound_decode_whose_guesses_stand_verifies_at_depth() {
        let prose = verifying(regime(|_| 1.3, Some(0.8), vec![path(1e-3, |_| Some(0.))]), Some(0.8), Some(5e-3));
        assert_eq!(prose.depth(), 0);
        assert_eq!(prose.pass(), Form::Verify(6), "{:?}", (1..8).map(|k| prose.verify_time(k)).collect::<Vec<_>>());
        assert!((prose.verify_time(6) - (0.0475 + 5e-3) / (1. + (1..=6).map(|j| 0.8f64.powi(j)).sum::<f64>())).abs() < 1e-12);
        let three = Regime { accepts: vec![Some(0.8); 3], ..prose.clone() };
        assert_eq!(three.pass(), Form::Verify(3));
    }

    /// A drive-bound decode (245 misses of 1 ms a step plain, the ranking
    /// keeping 4 more placed at each depth it foresees, its names wasting
    /// nothing) reads as much a token in either form, however its verify
    /// rows stand: its verify pass pays the host turn beside, so it runs
    /// every front row.
    #[test]
    fn a_drive_bound_decode_fronts() {
        for accepts in [Some(0.95), Some(0.5)] {
            let drive = verifying(regime(|k| 245. - 4. * k as f64, Some(0.997), vec![path(1e-3, |_| Some(0.))]), accepts, Some(2e-3));
            assert_eq!(drive.pass(), Form::Front(7), "at {accepts:?}");
            assert!((1..8).all(|k| drive.verify_time(k) > drive.time(7)), "at {accepts:?}");
        }
    }

    /// A drive-bound decode (245 misses of 1 ms a step plain, the ranking
    /// keeping 4 more placed at each depth it foresees) whose front rows
    /// waste 16 moves a depth fronts nothing, 245 ms a token. Its verify
    /// rows, standing 0.5 each, read as much a token less what the ranking
    /// keeps placed: before their waste is measured seven emit 1.99 tokens
    /// at 217 ms of reads each and a 2 ms turn, 218 ms a token, and it
    /// verifies seven. Measured, its rows' lookahead hints that nothing
    /// used, 16 moves a depth, cost more than that saves (seven: 274 ms a
    /// token; one: 253), and it runs the plain pass.
    #[test]
    fn a_drive_bound_decode_whose_verify_rows_fall_early_fronts_once_their_waste_is_measured() {
        let wasting = |verify: Option<f64>| {
            let mut drive = path(1e-3, |depth| Some(if depth == 0 { 0. } else { 16. }));
            drive.wasted.verify = (0..8).map(|depth| verify.filter(|_| depth > 0).map(|moves| moves * 1e-3)).collect();
            verifying(regime(|k| 245. - 4. * k as f64, Some(0.997), vec![drive]), Some(0.5), Some(2e-3))
        };
        let unmeasured = wasting(None);
        assert_eq!(unmeasured.pass(), Form::Verify(7), "{:?}", (1..8).map(|k| unmeasured.verify_time(k)).collect::<Vec<_>>());
        assert!((unmeasured.verify_time(7) - (0.217 + 2e-3 / (2. - 0.5f64.powi(7)))).abs() < 1e-12);
        let measured = wasting(Some(16.));
        assert_eq!(measured.pass(), Form::Front(0), "{:?}", (1..8).map(|k| measured.verify_time(k)).collect::<Vec<_>>());
        assert!((measured.verify_time(7) - (0.217 + (0.112 + 2e-3) / (2. - 0.5f64.powi(7)))).abs() < 1e-12);
    }

    /// A host turn longer than what the verify rows' tokens save leaves the
    /// compute-bound decode plain: 100 ms beside the pass, against 33.4 ms
    /// for a plain pass's token, is 37 ms a token at best.
    #[test]
    fn a_long_host_turn_fronts() {
        let prose = verifying(regime(|_| 1.3, Some(0.8), vec![path(1e-3, |_| Some(0.))]), Some(0.8), Some(0.1));
        assert_eq!(prose.pass(), Form::Front(0));
        assert!((1..8).all(|k| prose.verify_time(k) > prose.time(0)));
    }

    /// Before any verify row is judged or turn timed, a compute-bound
    /// decode prices its verify rows at their bound, every guess standing
    /// and no turn: it tries the deepest verify pass, whose passes measure
    /// them. Equal times take the shallower and the front form: a regime
    /// pricing every pass alike runs a plain pass.
    #[test]
    fn unjudged_verify_rows_are_priced_at_their_bound_and_ties_front() {
        let prose = verifying(regime(|_| 1.3, None, vec![path(1e-3, |_| None)]), None, None);
        assert_eq!(prose.pass(), Form::Verify(7));
        assert!((prose.verify_time(7) - 0.0508 / 8.).abs() < 1e-15);
        let flat = Regime { resident: vec![0.04; 8], ..verifying(regime(|_| 40., Some(1.), vec![path(1e-3, |_| Some(0.))]), Some(0.), Some(0.)) };
        assert_eq!(flat.pass(), Form::Front(0), "{:?}", (1..8).map(|k| flat.verify_time(k)).collect::<Vec<_>>());
    }

    /// Unmeasured deeper verify rows carry forward the last measured shallower
    /// conditional stand rate rather than jumping back up to 1.
    #[test]
    fn unmeasured_deeper_verify_rows_carry_forward_last_measured_acceptance() {
        let partial = Regime {
            accepts: vec![Some(0.8), Some(0.5), None, None, None, None, None],
            ..verifying(regime(|_| 1.3, Some(0.8), vec![path(1e-3, |_| Some(0.))]), None, Some(5e-3))
        };
        // Depth 4 emits: 1 + 0.8 + 0.8*0.5 + 0.8*0.5*0.5 + 0.8*0.5*0.5*0.5 = 2.5
        let emitted = 1. + 0.8 + 0.4 + 0.2 + 0.1;
        assert!((partial.verify_time(4) - (CODE[4] + 5e-3) / emitted).abs() < 1e-12);
    }

    /// An unexecuted shadow (`Needed::Shadow`) measures misses per true row,
    /// so `verify_time(k)` prices all `(1 + k)` rows' reads (including
    /// discarded rows past the first fall); once executed (`Needed::Executed`),
    /// `needed[k]` already divides all `(1 + k)` rows' misses by `emitted` and
    /// is not double-counted.
    #[test]
    fn unexecuted_shadow_prices_all_verify_rows_while_executed_does_not_double_count() {
        let mut needed = vec![Needed::Shadow(108.); 8];
        needed[0] = Needed::Executed(120.);
        needed[1] = Needed::Executed(108.);
        let mut accepts = vec![None; 7];
        accepts[0] = Some(0.875);
        let reg = Regime {
            resident: vec![0.05; 8],
            needed: NeededByForm::both(needed),
            stand: Some(0.875),
            accepts,
            stall: Some(1e-3),
            overhead: Some(0.022),
            paths: vec![PathTerm {
                seconds: 1e-3,
                wasted: Waste {
                    front: vec![Some(0.03); 8],
                    verify: vec![Some(0.), Some(0.003), None, None, None, None, None, None],
                },
            }],
        };
        // Depth 7 carries forward 0.875 -> emitted = 1 + sum_{j=1..=7} 0.875^j.
        let emitted_7: f64 = 1. + (1..=7).map(|j| 0.875f64.powi(j)).sum::<f64>();
        let expected_v7 = (8. * 108. * 1e-3 + 0.003 + 0.022) / emitted_7;
        assert!((reg.verify_time(7) - expected_v7).abs() < 1e-12);
        // Depth 1 is already Executed: uses `emitted_1 * 108 * 1e-3`, not `2 * 108 * 1e-3`.
        let expected_v1 = (1.875 * 108. * 1e-3 + 0.003 + 0.022) / 1.875;
        assert!((reg.verify_time(1) - expected_v1).abs() < 1e-12);
        // Unexecuted shadows at k=2..=7 must not beat executed Verify(1).
        assert!(reg.verify_time(7) > reg.verify_time(1));
        assert_eq!(reg.pass(), Form::Verify(1));
    }

    /// `Verify(1)` measuring higher misses/tok (e.g. 94 misses/tok at 65%
    /// acceptance) does not overwrite `Front(1)`'s 64 misses/tok, so `pass()`
    /// immediately returns `Form::Front(1)` after 1 `Verify(1)` probe even if
    /// `Front(1)` has a transient 15 ms waste spike.
    #[test]
    fn verify_misses_do_not_overwrite_front_misses() {
        let mut front = vec![Needed::Shadow(75.); 8];
        front[0] = Needed::Executed(94.);
        front[1] = Needed::Executed(64.);
        let mut verify = vec![Needed::Shadow(75.); 8];
        verify[0] = Needed::Executed(94.);
        verify[1] = Needed::Executed(94.);
        let mut accepts = vec![None; 7];
        accepts[0] = Some(0.65);
        let reg = Regime {
            resident: vec![0.05; 8],
            needed: NeededByForm { front, verify },
            stand: Some(0.97),
            accepts,
            stall: Some(1e-3),
            overhead: Some(0.0173),
            paths: vec![PathTerm {
                seconds: 1.02e-3,
                wasted: Waste {
                    front: vec![Some(0.0024), Some(0.015), None, None, None, None, None, None],
                    verify: vec![Some(0.0024), Some(0.0035), None, None, None, None, None, None],
                },
            }],
        };
        assert!(reg.time(1) < reg.verify_time(1), "front(1) {} must beat verify(1) {}", reg.time(1), reg.verify_time(1));
        assert_eq!(reg.pass(), Form::Front(1));
    }
}
