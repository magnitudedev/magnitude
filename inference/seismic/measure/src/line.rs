//! The line: what a decision
//! reads of a cost made of a few terms, `t = Σ_k x_k · c_k`, each term `k`
//! a cost `c_k` paid `x_k` times by a piece of work (a fixed cost is a term
//! every piece pays once). It is fitted through measured points and refined
//! by observations of real work, with its state ([`LineState`]).
//!
//! - **The fit** ([`Sums::fit`]): the weighted least-squares terms of no
//!   negative cost (nothing takes less than no time, nor less time for
//!   more), the non-negative least-squares solution (Lawson–Hanson) over the
//!   points' normal sums. Points place a line only once they tell every
//!   term apart: their normal matrix has full rank, to rounding.
//! - **Seeds and observations** ([`Line`]): seeds are isolated measurements
//!   (a measurement job's points), observations the timings real work
//!   already takes. The observations place every direction of the costs
//!   they tell apart; the seeds settle only those the observations leave
//!   undetermined (their normal matrix's null space), and the whole line
//!   before any observation.
//!
//! What the terms and the time are, which lines exist and what feeds them,
//! is the caller's: this crate knows nothing of why it is asked.

use std::fmt;

/// A line's costs, one per term: each finite and not negative, by
/// construction ([`Self::new`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Terms<const N: usize>([f64; N]);

/// Every term is finite ([`Terms::new`]), so equality is total.
impl<const N: usize> Eq for Terms<N> {}

/// Costs that make no line: one negative or not finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvalidLine {
    pub term: usize,
    pub cost: f64,
}

impl fmt::Display for InvalidLine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "term {} of cost {} makes no line: each must be finite and not negative", self.term, self.cost)
    }
}

impl std::error::Error for InvalidLine {}

impl<const N: usize> Terms<N> {
    /// `costs` as a line's, or why they make none.
    pub const fn new(costs: [f64; N]) -> Result<Self, InvalidLine> {
        let mut term = 0;
        while term < N {
            let cost = costs[term];
            if !(cost.is_finite() && cost >= 0.) {
                return Err(InvalidLine { term, cost });
            }
            term += 1;
        }
        Ok(Self(costs))
    }

    /// Each term's cost.
    pub fn costs(&self) -> [f64; N] {
        self.0
    }

    /// The time of a piece of work paying each term `x[k]` times.
    pub fn at(&self, x: [f64; N]) -> f64 {
        x.iter().zip(&self.0).map(|(x, cost)| x * cost).sum()
    }
}

/// The line of a fixed cost and one per unit, `fixed + x · slope`: the
/// two-term case most lines are.
impl Terms<2> {
    pub const fn fixed(&self) -> f64 {
        self.0[0]
    }

    pub const fn slope(&self) -> f64 {
        self.0[1]
    }
}

/// The weighted normal sums of points `(x, t)` over `N` terms: their
/// weight, `Σ w·x·xᵀ` and `Σ w·x·t`, all a least-squares fit needs. Sums of
/// a window are the difference of the sums at its ends ([`Self::since`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sums<const N: usize> {
    pub weight: f64,
    pub xx: [[f64; N]; N],
    pub xt: [f64; N],
}

impl<const N: usize> Default for Sums<N> {
    fn default() -> Self {
        Self { weight: 0., xx: [[0.; N]; N], xt: [0.; N] }
    }
}

impl<const N: usize> Sums<N> {
    /// Count a point `(x, t)` of `weight`.
    pub fn add(&mut self, x: [f64; N], t: f64, weight: f64) {
        self.weight += weight;
        for row in 0..N {
            for column in 0..N {
                self.xx[row][column] += weight * x[row] * x[column];
            }
            self.xt[row] += weight * x[row] * t;
        }
    }

    /// Every point's weight scaled by `by`.
    pub fn scaled(self, by: f64) -> Self {
        Self {
            weight: self.weight * by,
            xx: self.xx.map(|row| row.map(|value| value * by)),
            xt: self.xt.map(|value| value * by),
        }
    }

    /// The points of both.
    pub fn joined(self, other: Self) -> Self {
        Self {
            weight: self.weight + other.weight,
            xx: std::array::from_fn(|row| std::array::from_fn(|column| self.xx[row][column] + other.xx[row][column])),
            xt: std::array::from_fn(|row| self.xt[row] + other.xt[row]),
        }
    }

    /// The points counted after `earlier`, of the same sums.
    pub fn since(self, earlier: Self) -> Self {
        self.joined(earlier.scaled(-1.))
    }

    /// The weighted least-squares costs of no negative term through the
    /// points (module doc). None of points that do not tell every term
    /// apart; costs that are not a line's where a time is not finite.
    pub fn fit(&self) -> Option<Result<Terms<N>, InvalidLine>> {
        self.settled_by(&Self::default())
    }

    /// The costs of no negative term that fit these points best and, of
    /// those, fit `settling`'s best: `settling` decides only the directions
    /// these points leave undetermined (their normal matrix's null space),
    /// never one they place. None where both together do not tell every
    /// term apart; costs that are not a line's where a time is not finite.
    fn settled_by(&self, settling: &Self) -> Option<Result<Terms<N>, InvalidLine>> {
        // A time no clock gives (not finite) makes no line, nor is solved.
        for sums in [self, settling] {
            if let Some(term) = (0..N).find(|&term| !sums.xt[term].is_finite() || sums.xx[term].iter().any(|value| !value.is_finite())) {
                return Some(Err(InvalidLine { term, cost: sums.xt[term] }));
            }
        }
        if !(self.weight > 0.) || independent(&self.joined(*settling).xx, &[true; N]).len() < N {
            return None;
        }
        Some(Terms::new(non_negative(self, settling)))
    }
}

/// The terms of `of` whose columns of the symmetric positive semi-definite
/// `matrix` are independent, to rounding: symmetric elimination over them,
/// each pivot the term whose remaining diagonal is the largest share of its
/// own. A term whose largest share is within rounding is dependent on those
/// chosen before it, as is every term left: each of the at most `N` updates
/// to a remaining diagonal is bounded by it (Cauchy–Schwarz, in a
/// semi-definite Schur complement) and rounds to about `2ε` of it.
fn independent<const N: usize>(matrix: &[[f64; N]; N], of: &[bool; N]) -> Vec<usize> {
    let share = |reduced: &[[f64; N]; N], term: usize| match matrix[term][term] > 0. {
        true => reduced[term][term] / matrix[term][term],
        false => 0.,
    };
    let mut reduced = *matrix;
    let mut left = (0..N).filter(|&term| of[term]).collect::<Vec<_>>();
    let mut chosen = Vec::with_capacity(left.len());
    while let Some(at) = (0..left.len()).max_by(|&a, &b| share(&reduced, left[a]).total_cmp(&share(&reduced, left[b]))) {
        let pivot = left.swap_remove(at);
        if !(share(&reduced, pivot) > 4. * N as f64 * f64::EPSILON) {
            break;
        }
        let value = reduced[pivot][pivot];
        for &row in &left {
            let factor = reduced[row][pivot] / value;
            for &column in &left {
                reduced[row][column] -= factor * reduced[pivot][column];
            }
        }
        chosen.push(pivot);
    }
    chosen
}

/// The costs over the terms `free`, every other term zero, that solve the
/// primary's normal equations over them and, of those, best fit
/// `settling`: by Lagrange, `[[S, Bᵀ], [B, 0]] · [c, λ] = [s, b]` over the
/// free terms, `B` and `b` the primary's rows independent there
/// ([`independent`]; the rest follow from them), `S` and `s` the settling
/// sums'. Regular where both sums together tell the free terms apart: `B`
/// has full row rank, and `S` is definite on its null space.
fn solve<const N: usize>(primary: &Sums<N>, settling: &Sums<N>, free: &[bool; N]) -> [f64; N] {
    let terms = (0..N).filter(|&term| free[term]).collect::<Vec<_>>();
    let rows = independent(&primary.xx, free);
    let mut system = Vec::with_capacity(terms.len() + rows.len());
    for &row in &terms {
        let settled = terms.iter().map(|&column| settling.xx[row][column]);
        let multipliers = rows.iter().map(|&constraint| primary.xx[constraint][row]);
        system.push(settled.chain(multipliers).chain([settling.xt[row]]).collect::<Vec<_>>());
    }
    for &constraint in &rows {
        let placed = terms.iter().map(|&column| primary.xx[constraint][column]);
        system.push(placed.chain(rows.iter().map(|_| 0.)).chain([primary.xt[constraint]]).collect());
    }
    let solved = eliminate(system);
    let mut costs = [0.; N];
    for (index, &term) in terms.iter().enumerate() {
        costs[term] = solved[index];
    }
    costs
}

/// The solution of the regular system of `system`'s rows, each its
/// coefficients then its right side: Gaussian elimination with partial
/// pivoting.
fn eliminate(mut system: Vec<Vec<f64>>) -> Vec<f64> {
    let size = system.len();
    for pivot in 0..size {
        let best = (pivot..size).max_by(|&a, &b| system[a][pivot].abs().total_cmp(&system[b][pivot].abs())).expect("a pivot row");
        system.swap(pivot, best);
        for row in pivot + 1..size {
            let factor = system[row][pivot] / system[pivot][pivot];
            for column in pivot..=size {
                system[row][column] -= factor * system[pivot][column];
            }
        }
    }
    let mut solved = vec![0.; size];
    for row in (0..size).rev() {
        let known = (row + 1..size).map(|column| system[row][column] * solved[column]).sum::<f64>();
        solved[row] = (system[row][size] - known) / system[row][row];
    }
    solved
}

/// Whether freeing `term` beside the terms `free`, whose costs are the
/// primary's best over them, is a direction only `settling` decides (it
/// adds no column independent of theirs to the primary's) and one its
/// trial prices above none.
fn settles<const N: usize>(primary: &Sums<N>, settling: &Sums<N>, free: &[bool; N], term: usize) -> bool {
    let mut with = *free;
    with[term] = true;
    !adds_column(primary, free, term) && solve(primary, settling, &with)[term] > 0.
}

/// Whether freeing `term` beside the terms `free` adds a column of the
/// primary independent of theirs ([`independent`]).
fn adds_column<const N: usize>(primary: &Sums<N>, free: &[bool; N], term: usize) -> bool {
    let mut with = *free;
    with[term] = true;
    independent(&primary.xx, &with).len() > independent(&primary.xx, free).len()
}

/// The costs of no negative term that fit the primary sums best and, of
/// those, `settling`'s best (Lawson–Hanson's active set, over that order):
/// terms join the free set while the primary's gradient favours them, or,
/// where the primary is indifferent to all, while one is a direction only
/// `settling` decides and favours ([`settles`]); a step that would turn a
/// free term negative stops at the boundary and returns it to zero. The
/// gradient admits only a term that adds a column independent of the free
/// terms': a dependent one's gradient is rounding (the primary is already
/// best over its direction), and admitting it by that rounding could enter
/// and leave it every pass; it is `settles`'s to decide.
fn non_negative<const N: usize>(primary: &Sums<N>, settling: &Sums<N>) -> [f64; N] {
    let mut costs = [0.; N];
    let mut free = [false; N];
    let scale = primary.xt.iter().fold(0f64, |most, value| most.max(value.abs())).max(f64::MIN_POSITIVE);
    let tolerance = scale * f64::EPSILON * (N * N) as f64;
    // Each pass frees one term; each inner step fixes one: at most 3N steps
    // in exact arithmetic, bounded for rounding.
    for _ in 0..4 * N + 4 {
        let gradient: [f64; N] = std::array::from_fn(|row| primary.xt[row] - (0..N).map(|column| primary.xx[row][column] * costs[column]).sum::<f64>());
        let Some(entering) = (0..N)
            .filter(|&term| !free[term] && gradient[term] > tolerance && adds_column(primary, &free, term))
            .max_by(|&a, &b| gradient[a].total_cmp(&gradient[b]))
            .or_else(|| (0..N).find(|&term| !free[term] && settles(primary, settling, &free, term)))
        else {
            break;
        };
        free[entering] = true;
        loop {
            let trial = solve(primary, settling, &free);
            if (0..N).all(|term| !free[term] || trial[term] > 0.) {
                costs = trial;
                break;
            }
            // The step toward the trial that first brings a free term to
            // zero (none where it is zero already), which leaves the free
            // set; so does any other it brings there.
            let (blocking, step) = (0..N)
                .filter(|&term| free[term] && trial[term] <= 0.)
                .map(|term| {
                    let fall = costs[term] - trial[term];
                    (term, if fall > 0. { costs[term] / fall } else { 0. })
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("a free term the trial prices at none or less");
            for term in 0..N {
                costs[term] += step * (trial[term] - costs[term]);
            }
            free[blocking] = false;
            for term in 0..N {
                if !free[term] || costs[term] <= 0. {
                    free[term] = false;
                    costs[term] = 0.;
                }
            }
        }
    }
    costs
}

/// A line's state, what a decision taking it handles: no line yet, a line
/// of seeds alone (isolated measurements), or one observations of real work
/// refine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineState<L> {
    Pending,
    Seeded(L),
    Observed(L),
}

impl<L> LineState<L> {
    /// The line, where there is one.
    pub fn line(self) -> Option<L> {
        match self {
            Self::Pending => None,
            Self::Seeded(line) | Self::Observed(line) => Some(line),
        }
    }

    /// The same state of `map` of its line.
    pub fn map<M>(self, map: impl FnOnce(L) -> M) -> LineState<M> {
        match self {
            Self::Pending => LineState::Pending,
            Self::Seeded(line) => LineState::Seeded(map(line)),
            Self::Observed(line) => LineState::Observed(map(line)),
        }
    }
}

/// A line of `N` terms through seeds and observations (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line<const N: usize> {
    seeds: Sums<N>,
    observed: Sums<N>,
}

impl<const N: usize> Line<N> {
    /// A line of no points.
    pub fn new() -> Self {
        Self { seeds: Sums::default(), observed: Sums::default() }
    }

    /// A seed: a piece of work paying each term `x[k]` times took `t`,
    /// measured in isolation.
    pub fn seed(&mut self, x: [f64; N], t: f64) {
        self.seeds.add(x, t, 1.);
    }

    /// An observation of real work paying each term `x[k]` times that took
    /// `t`, of weight one.
    pub fn observe(&mut self, x: [f64; N], t: f64) {
        self.observed.add(x, t, 1.);
    }

    /// The line now: pending while its points do not tell every term apart;
    /// seeded, the seeds' fit, before any observation; observed after, the
    /// observations' fit with the seeds settling only the directions they
    /// leave undetermined (module doc); why its points' times make no line.
    pub fn state(&self) -> Result<LineState<Terms<N>>, InvalidLine> {
        if self.observed.weight <= 0. {
            return Ok(self.seeds.fit().transpose()?.map_or(LineState::Pending, LineState::Seeded));
        }
        Ok(self.observed.settled_by(&self.seeds).transpose()?.map_or(LineState::Pending, LineState::Observed))
    }
}

impl<const N: usize> Default for Line<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line of a fixed cost and a slope through `points`.
    fn sums(points: &[(f64, f64)]) -> Sums<2> {
        let mut sums = Sums::default();
        for &(x, t) in points {
            sums.add([1., x], t, 1.);
        }
        sums
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.), "{a} against {b}");
    }

    /// Exact points give their terms; the fit minimizes squared residuals;
    /// a falling fit is flat at the mean; one that begins below zero runs
    /// through the origin; one quantity places none.
    #[test]
    fn the_fit_is_the_best_line_of_no_negative_term() {
        let exact = sums(&[(1., 3.), (2., 5.), (4., 9.)]).fit().unwrap().unwrap();
        close(exact.fixed(), 1.);
        close(exact.slope(), 2.);
        let scattered = sums(&[(1., 1.0), (2., 2.5), (3., 2.9)]).fit().unwrap().unwrap();
        close(scattered.slope(), 0.95);
        close(scattered.fixed(), 0.233_333_333_333_333_3);
        let falling = sums(&[(1., 3.), (2., 1.)]).fit().unwrap().unwrap();
        close(falling.fixed(), 2.);
        assert_eq!(falling.slope(), 0.);
        let steep = sums(&[(1., 1.), (2., 5.)]).fit().unwrap().unwrap();
        assert_eq!(steep.fixed(), 0.);
        close(steep.slope(), 11. / 5.);
        assert!(sums(&[(4., 1.), (4., 2.)]).fit().is_none());
        assert!(Sums::<2>::default().fit().is_none());
        assert!(matches!(sums(&[(1., f64::NAN), (2., 1.)]).fit(), Some(Err(_))));
    }

    /// Several terms are told apart by work that pays them in varying
    /// proportions: a base, hits and links, each link's cost found though
    /// every pass pays the base; a term the data would price negative is
    /// held at none, the rest refitted without it.
    #[test]
    fn several_terms_are_fitted_jointly_and_none_negative() {
        let cost = |hits: f64, links: f64| 4. + 2. * hits + 7. * links;
        let mut sums = Sums::<3>::default();
        for (hits, links) in [(8., 0.), (6., 2.), (4., 4.), (7., 1.), (5., 3.), (8., 2.)] {
            sums.add([1., hits, links], cost(hits, links), 1.);
        }
        let terms = sums.fit().unwrap().unwrap().costs();
        for (fitted, truth) in terms.iter().zip([4., 2., 7.]) {
            close(*fitted, truth);
        }
        // Hits that seem to save time: no term below none.
        let mut negative = Sums::<3>::default();
        for (hits, links) in [(1., 0.), (2., 1.), (3., 0.), (4., 2.)] {
            negative.add([1., hits, links], 10. - hits + 5. * links, 1.);
        }
        let terms = negative.fit().unwrap().unwrap().costs();
        assert_eq!(terms[1], 0., "{terms:?}");
        assert!(terms.iter().all(|cost| *cost >= 0.), "{terms:?}");
        // Links that never vary from hits are no separate term.
        let mut tied = Sums::<3>::default();
        for hits in [1., 2., 3.] {
            tied.add([1., hits, hits], 3. * hits, 1.);
        }
        assert!(tied.fit().is_none());
    }

    /// A window's sums are its own points'.
    #[test]
    fn a_windows_sums_are_its_points() {
        let first = sums(&[(1., 3.)]);
        let mut all = first;
        all.add([1., 2.], 5., 1.);
        all.add([1., 4.], 9., 1.);
        let window = all.since(first).fit().unwrap().unwrap();
        close(window.fixed(), 1.);
        close(window.slope(), 2.);
    }

    /// A line is pending until its points place one, seeded by seeds alone,
    /// observed once real work is among them, and the seeds place only
    /// what the observations leave unplaced.
    #[test]
    fn a_lines_state_follows_its_points() {
        let mut line = Line::<2>::new();
        assert_eq!(line.state(), Ok(LineState::Pending));
        line.seed([1., 1.], 3.);
        assert_eq!(line.state(), Ok(LineState::Pending));
        line.seed([1., 2.], 5.);
        let LineState::Seeded(seeded) = line.state().unwrap() else { panic!("seeded") };
        close(seeded.slope(), 2.);
        // Observations of one count, slower than seeded: they place the line
        // at their count exactly, whatever the seeds there say, and the
        // seed at the other count, the one direction they leave, its spread.
        for _ in 0..64 {
            line.observe([1., 1.], 4.);
        }
        let LineState::Observed(observed) = line.state().unwrap() else { panic!("observed") };
        close(observed.at([1., 1.]), 4.);
        close(observed.at([1., 2.]), 5.);
        // A line weighs every observation alike.
        let mut even = Line::<2>::new();
        for (x, t) in [(1., 2.), (3., 6.), (1., 4.)] {
            even.observe([1., x], t);
        }
        let LineState::Observed(fit) = even.state().unwrap() else { panic!("observed") };
        let direct = sums(&[(1., 2.), (3., 6.), (1., 4.)]).fit().unwrap().unwrap();
        close(fit.fixed(), direct.fixed());
        close(fit.slope(), direct.slope());
    }
}
