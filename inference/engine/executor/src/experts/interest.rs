//! The interest ledger: the one owner of which experts are wanted, by when,
//! and how surely. Every producer publishes interests (an exact row's
//! request, an ahead row's names, a lookahead router's hints); reads, their
//! priority classes and cancellation, the foresight the ranking keeps
//! experts by, and the calibration of every predicted source are derived
//! from it, by one function each, so no producer has a path of its own.
//!
//! The ledger is owned by `Prefetch`, on the request service's thread under
//! residency, and takes no lock: what other threads learn (a decode launch
//! begun, a step completed or ended, a pass decided, a request closed)
//! reaches it as a [`Told`] through a queue the service drains before it
//! reads the rings ([`Ledger::tell`]). So the decode path's own work, a
//! request's exact interests, never waits on a completion's resolution.
//!
//! An [`Interest`] is `(layer, expert, step, source, rank, lineage)`: the
//! step whose exact rows will route the layer, and the name that wants the
//! expert there. A name carries no probability of its own. The probability
//! a step uses an expert is the one rate the ledger has: measured for the
//! key of every name live for the step when a read's admission is decided
//! ([`Combination`], [`Calibration`], [`Derived`]), so no relation between
//! names is assumed, and an exact row's own request is certain. The ledger
//! holds no read: it is told each expert's read state as it derives it and
//! answers what to do with that read ([`Ledger::act`]).
//!
//! A front row's name is wanted by its request's next pass, not by the next
//! global step: a prompt chunk's launch may run between, and the request
//! may close. Until that pass begins the name is bound to it alone
//! ([`When::Pass`], dated at an estimate); its launch's begin binds it to
//! the launch's step ([`Ledger::tell`], [`Told::Began`]). A request's closing
//! drops every name its rows published ([`Told::Closed`]).
//!
//! An interest lives from its publication until its step's exact rows have
//! routed its layer. As the host sees the route (its request served, or a
//! later ring of the step), the interest is passed ([`Ledger::pass`]): it no
//! longer wants a read, and an unbegun read nothing else wants cancels. Its
//! outcome is resolved once the step completes, from the step's routing
//! record ([`Told::Completed`]): whether the exact rows chose the expert,
//! the one count every predicted source's precision is measured by. A step
//! that ends without completing drops its interests unresolved, and the
//! names its rows published with them: a pass that never completed guessed
//! nothing ([`Told::Ended`]). A pass's guesses are judged once its round is
//! decided ([`Told::Decided`]): each guessed row's standing counts into
//! [`Stands`], and the interests its request's rows beyond the first wrong
//! guess named fall: they want no read, so their reads cancel where
//! unbegun; a fallen name stays with its step's other names until the step
//! resolves, so the combination they are calibrated by counts the guesses
//! that fell as it counts those that stood.
//!
//! A step's resolution at each layer also forms its counterfactual demand
//! cohort, once ([`Cohort`]): every expert the step read there that missed,
//! or hit a record a hint's read ahead landed for it, each such saved read
//! with the hint leads whose names would have landed it in time
//! ([`Catchers`]). The hint gate prices each lead by the stall only it
//! saves (`watcher::gate`).
//!
//! Every read short of due is ranked by the step that wants it, the
//! soonest first, then by its `p`, the likeliest first
//! ([`Ledger::schedule`]): the order its link reads them in
//! (`seismic_io::Want`). The drive starts a read
//! ahead whenever no parked read is queued and the bytes in flight leave
//! it room under the knee (`seismic_io`, "Order and depth"), so the drive
//! time the parked reads leave goes to the named reads in that order: a
//! right read saves its step's park only if it lands before that step
//! reaches its layer, which the soonest wanted is the nearest to missing.
//! Which names are read at all is the price's alone (`Admit`). A front
//! row's names are ranked so too: the exact row that uses one is its
//! request's pass of its depth, the step it is dated by.
//!
//! Every rate here rolls on one turnover's epochs (`horizon::Window`): a
//! combination's on its group's tier-0 turnover, a stand rate on the
//! bank-wide one. Each reads its bound, 1, before its first count, and
//! never returns to it.
//!
//! A read cancels only if no piece of it has begun: cancelling the unstarted
//! pieces of a begun read fails the whole expert after its begun bytes were
//! read, wasting them.
//!
//! Each read ahead the executor issues is attributed to the interest that
//! led its expert's aggregate as it was issued ([`Ledger::issued`]), until
//! its bytes' fate is known ([`Spent`]): an exact row of a later step uses
//! the expert (it chooses it, served by the landed read, or requests it
//! while the read is in flight); the read is cancelled before any piece
//! began; or an exact row requests the expert again with no read in flight,
//! the landed bytes evicted unused. A read whose leading interest drops
//! unresolved with an ahead row's lineage (a wrong guess's verdict, its
//! request's closing, its pass never completing) is a fallen guess's. The
//! resolution, verdict and expiry that end interests are the one producer
//! of these fates, so what the reads ahead were worth is measured by the
//! same path that calibrates the precision they were priced by.
//!
//! Every pass with rows ahead awaits its verdict from its launch's begin
//! ([`Told::Began`], a grouped launch's as a decode launch's), by its
//! step: a verdict names the step of the pass it judges
//! ([`Told::Decided`]), and nothing is matched by order. A pass ends
//! awaiting once: by its verdict; by its discard, its outputs unused (a
//! queued pass orphaned, an aborted outcome, [`Told::Discarded`]), which
//! drops the names it published as a pass that never completed does; by
//! its launch ending uncompleted; or by its request's closing. A verdict or
//! discard of a pass that awaits none is a broken invariant. The verdict
//! judges the pass's front row's guess ([`Judged`]), which the front rows'
//! record counts by kind (`watcher::telemetry::FrontSkips`); a discard or
//! closing leaves it unjudged. So every front row of a completed pass is
//! judged once. A verify pass's acceptable rows are judged by its own
//! verdict, each up to the first that fell ([`Accepted`]), which the same
//! record counts by depth: the stand rates a verify pass is priced by.
//!
//! A front row runs on a guess one of two predictors made behind its
//! request's pass before ([`GuessKind`]): the front row's selection of a
//! pass whose chain held up to it, or the MTP head's selection one past the
//! chained ones. The ring that names a front row's experts is the one
//! authority of its guess's kind ([`Ledger::row`]), and its step's routing
//! records of its pass's ([`Told::Completed`]); the kind is judged for the
//! front rows' record (`watcher::telemetry::FrontSkips`), and prices
//! nothing: a stand rate is keyed by depth and form alone ([`Stands`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroU8;
use std::ops::AddAssign;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use super::horizon::{self, Aged, Time, Tokens, Window};
use super::watcher::{At, DecodeRow, Step};
use crate::operation::{GuessKind, RequestId, TokenId};

/// The kind of the guess step row `row` ran on, of its step's `guesses`
/// (STEP_GUESSES, as its rings and routing records carry it): none of a row
/// that is no front row, or of an idle front row, which the chain did not
/// reach and which runs on its pass's pending token (`crate::Rows::Front`):
/// its pass's front rows of a guess lead its idle ones.
fn ran_on(guesses: u32, row: usize) -> Option<GuessKind> {
    GuessKind::of_word(magnitude_kernels::guess_kind(guesses as i32, row))
}

/// The kinds of a pass's front rows' guesses by depth ([`GuessKind`]):
/// its step's STEP_GUESSES fields re-based to its first front row, so field
/// `d` is the guess of depth `d + 1`'s (0 none: a pass of no front rows has
/// none): what its rows' judgments report ([`Judged::kind`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::experts) struct Kinds(u32);

impl Kinds {
    /// The kinds of the front rows from batch row `first` on, of a step's
    /// `guesses` word (STEP_GUESSES): none past the first that ran on no
    /// guess ([`ran_on`]).
    fn of(guesses: u32, first: usize) -> Self {
        let rows = magnitude_kernels::constants::VERIFY_ROWS as usize;
        let kinds = (first..rows).map_while(|row| ran_on(guesses, row));
        Self(kinds.enumerate().fold(0, |word, (depth, kind)| word | (kind.word() as u32) << (STEP_GUESS_BITS * depth)))
    }

    /// The kind of the guess of 0-based depth `depth`, none past the pass's
    /// front rows.
    pub fn at(self, depth: usize) -> Option<GuessKind> {
        let field = (self.0 >> (STEP_GUESS_BITS * depth)) & ((1 << STEP_GUESS_BITS) - 1);
        GuessKind::of_word(field as i32)
    }
}

/// A kind's field width (streaming_words.seismic's STEP_GUESS_BITS).
const STEP_GUESS_BITS: usize = magnitude_kernels::constants::STEP_GUESS_BITS as usize;

/// An ahead row of a pass: its index among the pass's ahead rows (0 the
/// first after its exact rows), and, of a row that only runs ahead (a front
/// row), the kind of the guess it runs on (its ring's RING_GUESSES field);
/// none of a row whose acceptance may extend the commit (a verification's
/// proposal).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) struct Row {
    pub index: u8,
    pub front: Option<GuessKind>,
}

impl Row {
    /// Whether its acceptance may extend the commit.
    pub fn acceptable(self) -> bool {
        self.front.is_none()
    }
}

/// Who wants an expert. Sources are the pass's rows and routers, never its
/// kind: an exact row's request, an ahead row's route, a lookahead router's
/// hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) enum Source {
    /// An exact row's own request (or a step that waits on a read in
    /// flight): certain, and wanted now.
    Exact,
    /// An ahead row's route, which the exact row of the pass its token is
    /// exact in routes again where every guess before it stands.
    Ahead(Row),
    /// A lookahead router's hint `distance` layers ahead of its route, of
    /// an exact row's choice (no row), or of an ahead row's (whose step is
    /// that row's, as an ahead row's route is).
    Lead { distance: u8, row: Option<Row> },
}

impl Source {
    /// The ahead row it is of, if any.
    pub fn row(self) -> Option<Row> {
        match self {
            Self::Exact => None,
            Self::Ahead(row) => Some(row),
            Self::Lead { row, .. } => row,
        }
    }

    /// The depth of the row it is of: 0 of an exact row's, `j` of a pass's
    /// `j`th ahead row's.
    pub fn depth(self) -> u8 {
        self.row().map_or(0, |row| row.index + 1)
    }

    /// Whether it is of a verify row: an acceptable row's route or hint.
    pub fn verifies(self) -> bool {
        self.row().is_some_and(Row::acceptable)
    }
}

/// A predicted name apart from the pass that published it: its source and
/// the rank of the choice that named the expert. What a combination key is
/// formed of ([`Combination::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) struct Kind {
    pub source: Source,
    pub rank: u8,
}

/// The pass row a name derives from, where it is an ahead row: its pass's
/// request; its batch row in its pass's launch; and the guessed rows up to
/// it (`steps::AheadOf::guessed`), by which a verdict on the pass drops the
/// names of rows beyond its first wrong guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) struct Lineage {
    pub request: RequestId,
    pub row: usize,
    pub guessed: NonZeroU8,
}

/// The step a name is for: a step begun, or its request's `pass`-th decode
/// pass, not yet begun, dated meanwhile at `estimate` (the step it would be
/// were every launch until then the request's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum When {
    Step(Step),
    Pass { request: RequestId, pass: u64, estimate: Step },
}

impl When {
    /// The step it is bound to, or its estimate while awaiting a pass.
    pub(in crate::experts) fn step(self) -> Step {
        match self {
            Self::Step(step) | Self::Pass { estimate: step, .. } => step,
        }
    }

    /// Whether it is bound to a step begun, which resolves and expires it.
    fn bound(self) -> bool {
        matches!(self, Self::Step(_))
    }
}

/// How a ring's entry of one batch row names its experts ([`Ledger::row`]):
/// as an exact row of its own step, or as an ahead row, for the step its
/// token is exact in, of its lineage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum Origin {
    Exact,
    Ahead { row: Row, when: When, lineage: Lineage },
}

/// A producer's name of one expert, as it publishes it ([`Ledger::publish`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) struct Name {
    pub source: Source,
    /// The step whose exact rows route the layer and would use the expert.
    pub when: When,
    /// The rank of the choice that named it (0 a row's best).
    pub rank: u8,
    /// The route that published it, in the device's order.
    pub from: At,
    /// Whether the expert was absent as it was named (nothing held, kept or
    /// landed it): only such a name's outcome measures what reading ahead
    /// on its source's names is worth.
    pub candidate: bool,
    /// The ahead row it derives from, none of an exact row's name.
    pub lineage: Option<Lineage>,
}

/// One producer's want of one expert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) struct Interest {
    pub source: Source,
    /// The step that wants it: bound, or an estimated future pass.
    pub when: When,
    pub rank: u8,
    pub from: At,
    pub candidate: bool,
    pub lineage: Option<Lineage>,
    /// When its producer first named it (a republication keeps it): the
    /// earliest a read on its name could have been issued.
    named: Instant,
    /// When its step's route was seen past its layer: it wants no read any
    /// more, and waits for its outcome. The time its step read the expert
    /// there, as the host saw it.
    passed: Option<Instant>,
    /// Its guess fell: it wants no read, and waits for its step's outcome
    /// beside the names that stood ([`Calibration`]'s combinations).
    fell: bool,
    /// When it was published, among every interest: equals order as their
    /// producers named them (a ring's best first).
    order: u64,
}

impl Interest {
    /// The step that wants it: bound, or its estimate while awaiting a pass.
    pub(in crate::experts) fn step(&self) -> Step {
        self.when.step()
    }

    /// The predicted name it is; none for an exact request.
    fn kind(&self) -> Option<Kind> {
        match self.source {
            Source::Exact => None,
            source => Some(Kind { source, rank: self.rank }),
        }
    }

    /// Whether it is for a step begun, which resolves and expires it.
    fn bound(&self) -> bool {
        self.when.bound()
    }

    /// The nearest of names first ([`Look`]'s order, then the best rank):
    /// the name computed closest to the use leads an expert's read.
    fn nearness(&self) -> (Look, u8) {
        (Look::of(self.source), self.rank)
    }

    /// The lead of an exact row's hint, none of any other name: the names
    /// the hint gate runs ([`Catchers`]).
    fn hint(&self) -> Option<u8> {
        match self.source {
            Source::Lead { distance, row: None } => Some(distance),
            _ => None,
        }
    }

    /// Whether `other` is the same producer's want: one source of one
    /// request (an ahead row's names of two requests are two wants), for one
    /// step or pass.
    fn same(&self, other: &Interest) -> bool {
        self.source == other.source
            && self.lineage.map(|lineage| lineage.request) == other.lineage.map(|lineage| lineage.request)
            && match (self.when, other.when) {
                (When::Step(left), When::Step(right)) => left == right,
                (When::Pass { request: lr, pass: lp, .. }, When::Pass { request: rr, pass: rp, .. }) => (lr, lp) == (rr, rp),
                _ => false,
            }
    }
}

/// Where an expert's read stands, as the executor tells the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum Read {
    /// Nothing holds, reads or lands it.
    Absent,
    /// Its read ahead is queued or running; `begun` once any piece has been
    /// taken by a worker.
    Reading { begun: bool },
    /// Read, or read by its own request, landing into its record.
    Landing,
    /// In the bank or the tier.
    Held,
}

/// An expert's live interests together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) struct Derived {
    /// The earliest step any live interest wants it at: its read's
    /// deadline.
    pub step: Step,
    /// The probability it is used then: its names' combination's measured
    /// rate ([`Calibration`]).
    pub p: f64,
}

/// What a read of an expert is ordered as, derived from its [`Derived`]
/// (`Ledger::act`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) enum Class {
    /// Certain and wanted now: a step parks on it.
    Due,
    /// Wanted from the estimated start of `step` at its layer, used with
    /// probability `p`: ranked by `step`, the soonest first, then by `p`
    /// ([`Ledger::schedule`]).
    Expected { step: Step, p: f64 },
}

/// What to do with an expert's read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) enum Act {
    /// Nothing to change.
    Keep,
    /// Start a read ordered as `Class`.
    Issue(Class),
    /// Order its queued read as `Class`.
    Reprice(Class),
    /// Cancel its queued read: nothing wants it any more and no piece has
    /// begun.
    Cancel,
}

/// An expert's act in [`Ledger::schedule`]'s order: of a read short of
/// due, the likeliest live interest wanting it by its class's step, which
/// leads the read it issues ([`Ledger::issued`]), and the first publication
/// among them, which orders reads otherwise alike.
#[derive(Clone, Copy, Debug)]
pub(in crate::experts) struct Scheduled {
    pub layer: usize,
    pub expert: usize,
    pub act: Act,
    pub leading: Option<Interest>,
    first: u64,
}

/// How an executor admits a read ahead: whether an aggregate probability
/// `p` of use reaches the break-even of a read of its expert's bytes,
/// whatever source named it (a calibrated `p` is all that tells sources
/// apart; the break-even is the cost model's, `magnitude_cost_model::ReadAdmission`).
/// Asked only of a read short of due, so a due read is never priced.
pub(in crate::experts) trait Admit {
    fn admits(&self, p: f64) -> bool;
}

/// An expert a completed step's exact rows chose at a layer: whether a read
/// ahead of it saved a miss (`saves`, the telemetry's), and whether it was
/// served from the tier (where a read ahead lands).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) struct Chosen {
    pub expert: usize,
    pub saves: bool,
    pub tier: bool,
}

/// One predicted interest's outcome once its step completed: whether its
/// exact rows chose the expert there (`routed`), and so whether a read
/// ahead of it saved one (`used`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) struct Resolved {
    pub source: Source,
    pub layer: usize,
    pub expert: usize,
    pub candidate: bool,
    pub routed: bool,
    pub used: bool,
    /// Used, and served from the tier: its read landed before the route.
    pub early: bool,
}

/// What a step's completion resolved ([`Ledger::tell`]): its predicted
/// interests' outcomes, and at each layer its demand cohort ([`Cohort`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::experts) struct Learned {
    pub resolved: Vec<Resolved>,
    pub cohorts: Vec<Cohort>,
}

/// The kind of name a read ahead was issued for: the source of the
/// interest that led its expert's aggregate ([`Ledger::issued`]). A front
/// row's route (`Source::Ahead`), an ahead row's lookahead hint, or an
/// exact row's lookahead hint; an exact request issues no read ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum Basis {
    Front,
    AheadHint,
    Hint,
}

impl Basis {
    /// Every basis, in [`Self::index`] order.
    pub const ALL: [Basis; 3] = [Basis::Front, Basis::AheadHint, Basis::Hint];

    pub fn of(source: Source) -> Self {
        match source {
            Source::Ahead(_) => Self::Front,
            Source::Lead { row: Some(_), .. } => Self::AheadHint,
            Source::Lead { row: None, .. } => Self::Hint,
            Source::Exact => panic!("no read ahead is an exact request's"),
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// What became of a read ahead's bytes ([`Ledger::issued`]), by the
/// [`Basis`] it was issued for and whether its name's guess fell (its
/// leading interest dropped unresolved with an ahead row's lineage); of a
/// read cancelled or used, the depth of the row whose name led it and
/// whether that row was a verify row (`verify`, [`Source::verifies`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum Spent {
    /// Its leading name's guess fell: wanted by no step now, though an
    /// exact row may still use it.
    Fell { basis: Basis, bytes: u64 },
    /// Cancelled before any piece began: none of its bytes were read.
    Cancelled { basis: Basis, depth: u8, verify: bool, fell: bool, bytes: u64 },
    /// An exact row used its expert: chose it, served by the landed read,
    /// or requested it while the read was in flight.
    Used { basis: Basis, depth: u8, verify: bool, fell: bool, bytes: u64 },
    /// Settled without being used: its target step resolved without choosing
    /// it, it was replaced by a newer read of the same expert, or it was
    /// evicted/failed before demand arrived. Never emitted while the read is
    /// still pending for a future step.
    Unused { basis: Basis, depth: u8, verify: bool, fell: bool, bytes: u64 },
}

/// The target of an issued read ahead: bound to a step begun, or awaiting
/// a request's future decode pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IssuedTarget {
    Bound(Step),
    Pass(RequestId, u64),
}

/// A read ahead issued and not yet spent ([`Ledger::issued`]): what it was
/// issued for, the depth of the row whose name led it ([`Source::depth`]),
/// the publication order of the interest that led it (unique, kept across a
/// republication), its bytes, and whether that interest's guess fell; the
/// step it was admitted for (its leading interest's), its target step or
/// future pass ([`IssuedTarget`]), when it was issued,
/// and when it landed ([`Ledger::landed`]): its landing's measured
/// duration, by which a resolution finds who would have caught it.
#[derive(Clone, Copy, Debug)]
struct Issued {
    basis: Basis,
    depth: u8,
    verify: bool,
    led: u64,
    bytes: u64,
    fell: bool,
    step: Step,
    target: IssuedTarget,
    issued: Instant,
    landed: Option<Instant>,
}

/// What a decision learned of a pass's guesses (`crate::AheadVerdict`):
/// of each acceptable row, in row order, whether it stood (its token the
/// selection of the row before it); the tokens its front rows ran on, by
/// depth; and the sequence's true tokens the decision learned, at the
/// positions after its pass's exact row in order (its exact row's
/// selection, then each accepted proposal's). A front row of depth `d` is
/// judged by the decision that learns its position's truth: its own pass's
/// at depth 1, a decision `d − 1` passes later past it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::experts) struct Verdict {
    pub acceptable: Vec<bool>,
    pub front: Vec<TokenId>,
    pub truths: Vec<TokenId>,
}

/// A front row's guess as a verdict judged it ([`Ledger::take_judged`]):
/// the step whose launch ran the row, its batch row, its depth (1 its
/// pass's first front row), its guess's kind (its step's routing records'
/// RING_GUESSES), and whether the guess stood; none where no verdict judged
/// it (its pass was discarded or its request closed first, a guess before
/// it fell, or no decision learned its position's truth).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) struct Judged {
    pub step: Step,
    pub row: usize,
    pub depth: u8,
    pub kind: GuessKind,
    pub stood: Option<bool>,
}

/// An acceptable row's guess as its own pass's verdict judged it
/// ([`Ledger::take_judged`]): its depth (1 its pass's first acceptable row)
/// and whether it stood, every acceptable row before it having stood.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) struct Accepted {
    pub depth: u8,
    pub stood: bool,
}

/// A completed pass's guessed front rows awaiting their judgments
/// ([`Fronted::Rows`]): the batch row of its first (depth 1), their kinds
/// by depth, how many it ran on a guess (its idle rows past them name
/// nothing and are never judged), the depths judged and found standing so
/// far, and, once its own verdict came, the tokens they ran on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FrontPass {
    first: usize,
    kinds: Kinds,
    rows: u8,
    stood: u8,
    tokens: Option<Vec<TokenId>>,
}

impl FrontPass {
    /// Its front row of depth `depth`'s judgment as no verdict decided it.
    fn unjudged(&self, step: Step, depth: u8) -> Judged {
        let kind = self.kinds.at(usize::from(depth) - 1).expect("a front pass's row has its kind");
        Judged { step, row: self.first + usize::from(depth) - 1, depth, kind, stood: None }
    }

    /// Its rows not yet judged, as no verdict decides them.
    fn pending(&self, step: Step) -> impl Iterator<Item = Judged> + '_ {
        (self.stood + 1..=self.rows).map(move |depth| self.unjudged(step, depth))
    }
}

/// A pass with rows ahead awaiting its verdict, by its front rows: none (a
/// pass of acceptable rows); a launch yet to complete (its first front
/// row's batch row and how many it runs, idle ones included); its guessed
/// front rows awaiting their judgments ([`FrontPass`]); or front rows of no
/// guess's kind, as their kinds gave none (`ran_on`) or no routing record
/// carried them (its every streamed layer whole, so none was written): its
/// verdict counts into no stand rate and judges nothing, as nothing is
/// measured of a kind, but its first front token's fall drops the names
/// its rows published all the same.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Fronted {
    Without,
    Running { first: usize, rows: u8 },
    Rows(FrontPass),
    Unread,
}

/// A pass awaiting its verdict and, past it, its deeper front rows'
/// judgments: its exact row's position (the truths its decision learns are
/// of the positions after it), whether its own verdict came, and its rows
/// ahead ([`Fronted`]).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Awaiting {
    exact: u64,
    decided: bool,
    fronted: Fronted,
}

/// A launch begun ([`Told::Began`]).
#[derive(Debug)]
pub(in crate::experts) enum Launched {
    /// A decode launch's batch rows (`ExpertStep::decoding`), by which its
    /// rings' rows name, and the `padding` rows its class runs past them.
    Decode { rows: Box<[DecodeRow]>, padding: usize },
    /// A grouped launch's requests whose passes run rows ahead
    /// (`ExpertStep::guessing`): its rows name nothing ahead (its rings'
    /// rows are exact), and its passes await their verdicts.
    Grouped(Box<[RequestId]>),
}

/// What another thread tells the ledger, drained in order on its own
/// ([`Ledger::tell`]).
#[derive(Debug)]
pub(in crate::experts) enum Told {
    /// A launch of step `step` begins, a decode launch or a grouped one
    /// with rows ahead ([`Launched`]): sent before it is queued, so before
    /// any ring of it is written.
    Began { step: Step, launch: Launched },
    /// Step `step` completed, its exact rows choosing at each layer the
    /// experts `chose` lists (ascending), its front rows running on guesses
    /// of the kinds `guesses` holds (STEP_GUESSES, as its routing records
    /// carry it; none where every streamed layer was whole and wrote none).
    Completed { step: Step, chose: Vec<Vec<Chosen>>, guesses: Option<u32> },
    /// Step `step` ended, `completed` (it ran on the device, whatever its
    /// outputs) or not: sent once the steps know it ended, so no ring of it
    /// is read as live after it is drained.
    Ended { step: Step, completed: bool },
    /// Request `request`'s pass of step `step` was decided: its guesses
    /// and the truths it learned are `verdict`'s (none where the decision
    /// judged none and learned no truth past its exact row).
    Decided { step: Step, request: RequestId, verdict: Verdict },
    /// Request `request`'s pass of step `step` was discarded: it completed,
    /// but its outputs are no sequence's, so no verdict will judge it.
    Discarded { step: Step, request: RequestId },
    /// Request `request` closed: none of its passes will come.
    Closed(RequestId),
    /// An MTP head draft launched after a verify step completed: the drive is
    /// idle for `estimated_seconds` until the next verify step begins
    /// (`Told::Began`).
    DraftWindow { estimated_seconds: f64 },
}

/// Outcomes counted, and how many of them hit (a name's expert chosen, a
/// guess standing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::experts) struct Tally {
    counted: u64,
    hit: u64,
}

impl Tally {
    fn of(hit: bool) -> Self {
        Self { counted: 1, hit: u64::from(hit) }
    }

    /// The share that hit; none of no outcome.
    pub fn rate(self) -> Option<f64> {
        (self.counted > 0).then(|| self.hit as f64 / self.counted as f64)
    }

    /// The rate over a window's two tallies, as [`horizon::Window::read`]
    /// gives them: a window read holds an outcome.
    fn over((previous, current): (&Tally, &Tally)) -> f64 {
        let mut tally = *previous;
        tally += *current;
        tally.rate().expect("a window read holds an outcome")
    }
}

impl horizon::Tally for Tally {
    fn is_empty(&self) -> bool {
        self.counted == 0
    }
}

impl AddAssign for Tally {
    fn add_assign(&mut self, other: Self) {
        self.counted += other.counted;
        self.hit += other.hit;
    }
}

/// The form of a pass's guessed rows (§3.3): front rows, or verify rows.
/// A plain pass guesses none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) enum Form {
    Front,
    Verify,
}

impl Form {
    /// The form of rows that verify where `verifies`, else front.
    pub fn of(verifies: bool) -> Self {
        if verifies { Self::Verify } else { Self::Front }
    }
}

/// What a look saw (§3.5): its row's route, or its lookahead hint at lead
/// ℓ ∈ 1..=HINT_LEADS layers. A route is nearer than any hint of its row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) enum Seen {
    Route,
    Hint(u8),
}

/// One source of names (§3.5): what it saw, at the depth of the row it came
/// from (0 an exact row), in the row's form. Ordered nearest first: by
/// (depth, what it saw), the look computed closest to the use first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) struct Look {
    pub depth: u8,
    pub seen: Seen,
    pub form: Form,
}

/// The hint leads a look may see: 1..=HINT_LEADS.
pub(in crate::experts) const HINT_LEADS: usize = crate::planning::HINT_LEADS as usize;

/// The looks' domain: (1 + HINT_LEADS) × DEPTHS × 2.
pub(in crate::experts) const LOOKS: usize = (1 + HINT_LEADS) * super::DEPTHS * 2;

impl Look {
    /// The look `source` names by.
    pub fn of(source: Source) -> Self {
        let seen = match source {
            Source::Lead { distance, .. } => Seen::Hint(distance),
            Source::Ahead(_) | Source::Exact => Seen::Route,
        };
        Self { depth: source.depth(), seen, form: Form::of(source.verifies()) }
    }

    /// Its index in 0..LOOKS. Panics where its depth or lead is outside the
    /// domain: no row or router names past them.
    fn index(self) -> usize {
        let depth = usize::from(self.depth);
        assert!(depth < super::DEPTHS, "a look of depth {depth} past the decode form's {} rows", super::DEPTHS);
        let seen = match self.seen {
            Seen::Route => 0,
            Seen::Hint(lead) => {
                assert!((1..=HINT_LEADS).contains(&usize::from(lead)), "a hint at lead {lead} outside 1..={HINT_LEADS}");
                usize::from(lead)
            }
        };
        (depth * (1 + HINT_LEADS) + seen) * 2 + self.form as usize
    }
}

/// The ledger's combination key (§3.5), bounded by construction: the
/// nearest look that named the expert for the step, the best rank that look
/// named it at, and the number of distinct looks that named it (their
/// agreement, with no relation between them assumed). Two requests' names
/// from one look are one look. Domain LOOKS × k × LOOKS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) struct Combination {
    pub nearest: Look,
    pub rank: u8,
    pub looks: u8,
}

impl Combination {
    /// The combination of the predicted names `kinds`; none of none.
    fn of(kinds: impl IntoIterator<Item = Kind>) -> Option<Self> {
        let kinds = kinds.into_iter().map(|kind| (Look::of(kind.source), kind.rank)).collect::<Vec<_>>();
        let nearest = kinds.iter().map(|&(look, _)| look).min()?;
        let rank = kinds.iter().filter(|&&(look, _)| look == nearest).map(|&(_, rank)| rank).min()?;
        let mut looks = kinds.iter().map(|&(look, _)| look).collect::<Vec<_>>();
        looks.sort_unstable();
        looks.dedup();
        let looks = u8::try_from(looks.len()).expect("at most LOOKS looks");
        Some(Self { nearest, rank, looks })
    }

    /// Its cell in a table of `ranks` ranks. Panics where its rank is not
    /// below `ranks` or its look count is outside `1..=LOOKS`.
    fn index(self, ranks: usize) -> usize {
        let rank = usize::from(self.rank);
        assert!(rank < ranks, "a name at rank {rank} of a ring naming {ranks} a row");
        let looks = usize::from(self.looks);
        assert!((1..=LOOKS).contains(&looks), "a combination of {looks} looks outside 1..={LOOKS}");
        (self.nearest.index() * ranks + rank) * LOOKS + looks - 1
    }
}

/// The names ledger's combination rates (§3.5), one dense table per bank
/// group: for each key ([`Combination`]), the share of the experts a read's
/// admission was decided on with that key for one step that the step used,
/// over a window on the group's tier-0 epochs. It is the one probability of
/// use the ledger has: no relation between names is assumed, and nothing is
/// composed from a name's own rate. An unmeasured key reads its bound, 1: a
/// key never seen is admitted, so its outcome measures it, and a window
/// once counted never reads unmeasured again, so each key is tried at most
/// once. Memory LOOKS × k × LOOKS windows a group, whatever the run.
#[derive(Debug)]
pub(in crate::experts) struct Calibration {
    ranks: usize,
    groups: Vec<Vec<Window<Tally>>>,
}

impl Calibration {
    /// None counted, for groups at `epochs` (each group's tier-0 epoch now),
    /// of rings naming `ranks` experts a row.
    fn new(epochs: impl ExactSizeIterator<Item = u64>, ranks: usize) -> Self {
        let cells = LOOKS * ranks * LOOKS;
        Self { ranks, groups: epochs.map(|epoch| vec![Window::new(epoch); cells]).collect() }
    }

    /// Bytes held by these dense combination tables (constant in run length
    /// and τ; charged against the load).
    pub fn bytes(&self) -> u64 {
        Self::bytes_for(self.groups.len(), self.ranks)
    }

    /// Bytes a dense combination table of `groups` groups and `ranks` ranks
    /// holds.
    pub fn bytes_for(groups: usize, ranks: usize) -> u64 {
        (groups * LOOKS * ranks * LOOKS * std::mem::size_of::<Window<Tally>>()) as u64
    }

    /// The measured probability an expert `combination` named for a step of
    /// `group` is used there, at the group's `epoch`: its bound, 1, before
    /// any outcome.
    fn combined(&mut self, group: usize, combination: Combination, epoch: u64) -> f64 {
        self.groups[group][combination.index(self.ranks)].read(epoch).map_or(1., Tally::over)
    }

    /// An expert `combination` named for a step of `group` was `used` there.
    fn count(&mut self, group: usize, combination: Combination, epoch: u64, used: bool) {
        *self.groups[group][combination.index(self.ranks)].count(epoch) += Tally::of(used);
    }
}

/// Base token age after which a silent stand-rate row expires from
/// [`Stands::live_rate`] so the regime can re-probe deeper widths (matching
/// `DraftController`'s ~32-round decay timescale).
pub(in crate::experts) const STAND_STALE_BASE: f64 = 32.0;

/// Maximum exponential-backoff token age for a silent stand-rate row whose
/// re-entry probes keep falling (`32 → 64 → 128 → 256`, bounding worst-case
/// probe overhead below 0.5%).
const STAND_STALE_MAX: f64 = 256.0;

/// A stand rate's state (§3.3): unjudged (its bound 1), or judged over a
/// window on the bank-wide epochs, which holds its last measurement, with
/// its age, while its row is not run. Unjudged → judged is the only
/// transition, so raw history ([`Stands::judged`]) never returns to its bound,
/// while [`Stands::live_rate`] expires silent rows past `stale_after` tokens
/// so a transient hard passage cannot permanently trap decode at a narrow width.
pub(in crate::experts) enum Judgement {
    Unjudged,
    Judged {
        window: Aged<Tally>,
        stale_after: f64,
    },
}

/// The guessed rows a pass runs: 1..DEPTHS − 1, past its exact row.
const ROWS: usize = super::DEPTHS - 1;

/// The one owner of how often a pass's guessed rows stand (§3.3), by depth
/// and form: of each, row `j` standing, every row before it having stood,
/// counted as the verdicts judge them: a verify row's by its own pass's,
/// a front row of depth `j` by the decision that learns its token's truth,
/// `j − 1` passes after its own ([`Ledger::tell`], [`Told::Decided`]). A
/// front row's guess's kind is not part of the key: no reader prices names
/// by lineage.
///
/// The interest ledger counts it; the ranking and its shadows
/// ([`Self::through_rows`]) and the regime ([`Self::rate`]) read it, through
/// one shared handle. Each rate rolls on the bank-wide turnover's epochs
/// (`horizon::Times::bank_epoch`): a verdict prices every group's reads.
/// A row unjudged reads its bound, 1, until its first verdict; a row the
/// run stops computing keeps its last rate, with its age, and never
/// returns to its bound. Memory 2·(DEPTHS − 1) windows.
#[derive(Clone)]
pub(in crate::experts) struct Stands {
    time: Time,
    rows: Arc<Mutex<[[Judgement; 2]; ROWS]>>,
}

impl std::fmt::Debug for Stands {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Stands").finish_non_exhaustive()
    }
}

impl Stands {
    /// None judged, rolling on `time`'s bank-wide epochs.
    pub(crate) fn new(time: Time) -> Self {
        let rows = std::array::from_fn(|_| [Judgement::Unjudged, Judgement::Unjudged]);
        Self { time, rows: Arc::new(Mutex::new(rows)) }
    }

    /// Locks the stand rates, then reads the bank-wide epoch and clock under
    /// that lock so every count and read observes monotonically non-decreasing
    /// `(epoch, now)`. `horizon::Time` is a leaf lock, so `Stands::rows ->
    /// horizon::Time` cannot deadlock.
    fn lock_with_now(&self) -> (MutexGuard<'_, [[Judgement; 2]; ROWS]>, u64, Tokens) {
        let rows = self.rows.lock().expect("the stand rates");
        let times = self.time.lock();
        let (epoch, now) = (times.bank_epoch(), times.clock.through_latest());
        drop(times);
        (rows, epoch, now)
    }

    /// Row `row`'s cell. Panics where `row` is not a guessed row,
    /// 1..DEPTHS − 1: row 0 is the exact row, which always stands.
    fn cell(rows: &mut [[Judgement; 2]; ROWS], row: usize, form: Form) -> &mut Judgement {
        assert!((1..=ROWS).contains(&row), "row {row} is no guessed row of 1..={ROWS}");
        &mut rows[row - 1][form as usize]
    }

    /// Row `row` of `form`, every row before it having stood, `stood` or
    /// fell, judged now.
    pub(super) fn count(&self, row: usize, form: Form, stood: bool) {
        let (mut rows, epoch, now) = self.lock_with_now();
        let cell = Self::cell(&mut rows, row, form);
        match cell {
            Judgement::Unjudged => {
                *cell = Judgement::Judged {
                    window: Aged::new(epoch),
                    stale_after: STAND_STALE_BASE,
                };
            }
            Judgement::Judged { window, stale_after } => {
                if let Some((_, age)) = window.read(epoch, now) {
                    if age.get() >= *stale_after {
                        // Re-entering after staleness expiry: drop the frozen
                        // pre-silence epoch tally so the fresh probe stands on
                        // its own verdict, and back off exponentially if it
                        // still falls (or reset to base if it stands).
                        *window = Aged::new(epoch);
                        *stale_after = if stood {
                            STAND_STALE_BASE
                        } else {
                            (*stale_after * 2.0).min(STAND_STALE_MAX)
                        };
                    } else if stood {
                        *stale_after = STAND_STALE_BASE;
                    }
                }
            }
        }
        let Judgement::Judged { window, .. } = cell else { unreachable!("judged above") };
        *window.count(epoch, now) += Tally::of(stood);
    }

    /// r_row of `form` (§3.4) and its age, the tokens since its row was last
    /// judged: row `row` ≥ 1 standing, every row before it having stood;
    /// none unjudged (whose bound, 1, the caller applies).
    pub fn judged(&self, row: usize, form: Form) -> Option<(f64, Tokens)> {
        let (mut rows, epoch, now) = self.lock_with_now();
        Self::read(&mut rows, row, form, epoch, now)
    }

    /// r_row of `form`: [`Self::judged`]'s rate.
    pub fn rate(&self, row: usize, form: Form) -> Option<f64> {
        self.judged(row, form).map(|(rate, _)| rate)
    }

    /// The token-age staleness threshold currently governing `(row, form)`
    /// (`STAND_STALE_BASE`..=`STAND_STALE_MAX`).
    pub fn stale_after(&self, row: usize, form: Form) -> f64 {
        let mut rows = self.rows.lock().expect("the stand rates");
        match Self::cell(&mut rows, row, form) {
            Judgement::Unjudged => STAND_STALE_BASE,
            Judgement::Judged { stale_after, .. } => *stale_after,
        }
    }

    /// r_row for `preferred`, falling back to the other form's live rate when
    /// `preferred` is stale relative to it (strictly older on the token clock
    /// and the other form is still fresh), and expiring to `None` once silent
    /// for at least `stale_after` tokens so `Regime::verify_emitted` carries
    /// forward the shallower live stand rate instead of staying permanently
    /// trapped by a frozen warmup or hard-passage sample. An unjudged
    /// `preferred` row stays `None` (its optimistic bound 1).
    pub fn live_rate(&self, row: usize, preferred: Form) -> Option<f64> {
        let (mut rows, epoch, now) = self.lock_with_now();
        let (rate, age, stale_after) = Self::read_with_stale(&mut rows, row, preferred, epoch, now)?;
        let other = match preferred {
            Form::Front => Form::Verify,
            Form::Verify => Form::Front,
        };
        if let Some((other_rate, other_age, other_stale_after)) =
            Self::read_with_stale(&mut rows, row, other, epoch, now)
        {
            if other_age.get() < other_stale_after && other_age < age {
                return Some(other_rate);
            }
        }
        if age.get() >= stale_after {
            return None;
        }
        Some(rate)
    }

    /// Conditional front-row stand rates `r_1..r_{DEPTHS-1}`, unjudged rows at
    /// their bound 1.0 (design §3.3), read under a single lock of the clock
    /// and stand table.
    pub fn front_rates(&self) -> [f64; ROWS] {
        let (mut rows, epoch, now) = self.lock_with_now();
        std::array::from_fn(|index| Self::read(&mut rows, index + 1, Form::Front, epoch, now).map_or(1.0, |(rate, _)| rate))
    }

    /// σ_depth of `form` (§3.3): the probability a pass's first `depth`
    /// guessed rows all stand, Π_{row ≤ depth} r_row, an unjudged row's at
    /// its bound 1.
    pub fn through_rows(&self, depth: usize, form: Form) -> f64 {
        let (mut rows, epoch, now) = self.lock_with_now();
        (1..=depth).map(|row| Self::read(&mut rows, row, form, epoch, now).map_or(1., |(rate, _)| rate)).product()
    }

    fn read_with_stale(rows: &mut [[Judgement; 2]; ROWS], row: usize, form: Form, epoch: u64, now: Tokens) -> Option<(f64, Tokens, f64)> {
        match Self::cell(rows, row, form) {
            Judgement::Unjudged => None,
            Judgement::Judged { window, stale_after } => {
                let (tallies, age) = window.read(epoch, now).expect("a judged row holds a verdict");
                Some((Tally::over(tallies), age, *stale_after))
            }
        }
    }

    fn read(rows: &mut [[Judgement; 2]; ROWS], row: usize, form: Form, epoch: u64, now: Tokens) -> Option<(f64, Tokens)> {
        Self::read_with_stale(rows, row, form, epoch, now).map(|(rate, age, _)| (rate, age))
    }
}

/// The hint leads (1..=HINT_LEADS) whose names for a saved read would have
/// landed it in time (§3.5): name time + the landing's measured duration ≤
/// the read's time, both measured on this landing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::experts) struct Catchers(u8);

impl Catchers {
    /// Lead `lead`'s name would have landed it in time. Panics where `lead`
    /// is outside 1..=HINT_LEADS.
    pub fn caught_by(&mut self, lead: u8) {
        assert!((1..=HINT_LEADS).contains(&usize::from(lead)), "a hint at lead {lead} outside 1..={HINT_LEADS}");
        self.0 |= 1 << (lead - 1);
    }

    /// The lead that alone would have landed it; none where none or several.
    pub fn only(self) -> Option<u8> {
        (self.0.count_ones() == 1).then(|| self.0.trailing_zeros() as u8 + 1)
    }
}

/// The counterfactual demand cohort C_{s,l} (§3.5), formed once by step
/// `step`'s resolution of layer `layer`: every expert the step read there
/// that missed (`misses`), or hit a record whose landing was admitted for
/// the step by a hint name (`saved`, each with its catchers). Membership is
/// a fact of the step, whatever the gate ran: a hint that landed its expert
/// in time turns a miss into a hit, and the cohort counts it still.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::experts) struct Cohort {
    pub step: Step,
    pub layer: usize,
    pub misses: u32,
    pub saved: Vec<Catchers>,
}

impl Cohort {
    /// Its reads: misses and saved reads.
    pub fn size(&self) -> u32 {
        self.misses + u32::try_from(self.saved.len()).expect("a layer's reads fit u32")
    }

    /// The saved reads only `lead` would have caught: ℓ's exclusive saving,
    /// so no saved stall is earned twice, and the leads' savings never sum
    /// past the cohort's.
    pub fn exclusive(&self, lead: u8) -> u32 {
        u32::try_from(self.saved.iter().filter(|catchers| catchers.only() == Some(lead)).count()).expect("a layer's reads fit u32")
    }
}

/// A ratio counted over a window (§3.5): `part` over `whole` (exclusive
/// catches over cohort size, or cohort reads over steps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::experts) struct Share {
    pub part: u64,
    pub whole: u64,
}

impl Share {
    fn add(&mut self, part: u64, whole: u64) {
        self.part += part;
        self.whole += whole;
    }

    /// The summed tallies over a window's two halves.
    pub fn sum((previous, current): (&Share, &Share)) -> Self {
        Self { part: previous.part + current.part, whole: previous.whole + current.whole }
    }

    /// The ratio `part / whole`. Panics where `whole == 0`.
    pub fn rate(self) -> f64 {
        assert!(self.whole > 0, "a non-empty Share window holds a positive denominator");
        self.part as f64 / self.whole as f64
    }

    /// Wilson score binomial confidence interval `(lower, upper)` at `z`
    /// standard errors for `part` successes in `whole` trials. Returns
    /// `(0.0, 1.0)` when `whole == 0`.
    pub fn wilson(self, z: f64) -> (f64, f64) {
        if self.whole == 0 {
            return (0.0, 1.0);
        }
        let n = self.whole as f64;
        let p = (self.part as f64 / n).clamp(0.0, 1.0);
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = p + z2 / (2.0 * n);
        let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
        (((center - margin) / denom).clamp(0.0, 1.0), ((center + margin) / denom).clamp(0.0, 1.0))
    }

    /// Poisson rate confidence interval `(lower, upper)` at `z` standard
    /// errors for `part` events over `whole` steps. Panics where `whole == 0`.
    pub fn poisson(self, z: f64) -> (f64, f64) {
        assert!(self.whole > 0, "a non-empty Share holds a positive denominator");
        let m = self.part as f64;
        let s = self.whole as f64;
        let z2 = z * z;
        let half = z2 / 2.0;
        let margin = z * (m + z2 / 4.0).sqrt();
        (((m + half - margin).max(0.0)) / s, (m + half + margin) / s)
    }
}

impl horizon::Tally for Share {
    fn is_empty(&self) -> bool {
        self.whole == 0
    }
}

/// A hint lead's exclusive-catch rate (§3.5, A10): unmeasured (reads its
/// bound 1 until its first cohort while running), or measured with its age
/// in tokens since it last counted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) enum Caught {
    Unmeasured,
    Measured { rate: f64, age: Tokens },
}

impl Caught {
    /// The rate used for pricing: its bound 1 while unmeasured.
    pub fn rate(self) -> f64 {
        match self {
            Self::Unmeasured => 1.0,
            Self::Measured { rate, .. } => rate,
        }
    }
}

/// The counterfactual demand cohort rates (§3.5, Revision 1 row 2):
/// - `exclusive_ℓ` (`Aged<Share>` on bank-wide epochs, counted only while
///   lead ℓ runs on a non-empty cohort; unmeasured reads 1, and freezes
///   with its age while ℓ is off, A10);
/// - `cohort_l` (`Window<Share>` on bank-wide epochs, counted on every
///   resolved step whether hints run or not).
#[derive(Clone, Debug)]
pub(in crate::experts) struct CohortRates {
    exclusive: [Aged<Share>; HINT_LEADS],
    cohorts: Vec<Window<Share>>,
}

impl CohortRates {
    pub fn new(layers: usize, epoch: u64) -> Self {
        Self {
            exclusive: std::array::from_fn(|_| Aged::new(epoch)),
            cohorts: vec![Window::new(epoch); layers],
        }
    }

    /// Count one layer's resolved `cohort` at bank-wide `epoch` and clock
    /// `now`. `running[lead - 1]` is whether the layer hinting
    /// `cohort.layer` at `lead` ran its hints on this step.
    pub fn count(&mut self, cohort: &Cohort, running: &[bool], epoch: u64, now: Tokens) {
        assert!(running.len() <= HINT_LEADS, "running has {} entries, past HINT_LEADS ({HINT_LEADS})", running.len());
        let size = u64::from(cohort.size());
        self.cohorts[cohort.layer].count(epoch).add(size, 1);
        if size > 0 {
            for (index, &ran) in running.iter().enumerate() {
                if ran {
                    let lead = (index + 1) as u8;
                    let exclusive = u64::from(cohort.exclusive(lead));
                    self.exclusive[index].count(epoch, now).add(exclusive, size);
                }
            }
        }
    }

    /// Lead `lead`'s (1..=HINT_LEADS) exclusive-catch tallies and age at
    /// bank-wide `epoch` and clock `now`; none before any cohort counted.
    pub fn exclusive_share(&mut self, lead: u8, epoch: u64, now: Tokens) -> Option<(Share, Tokens)> {
        let index = usize::from(lead);
        assert!((1..=HINT_LEADS).contains(&index), "a hint at lead {lead} outside 1..={HINT_LEADS}");
        self.exclusive[index - 1].read(epoch, now).map(|(tallies, age)| (Share::sum(tallies), age))
    }

    /// Lead `lead`'s (1..=HINT_LEADS) exclusive-catch rate and its age at
    /// bank-wide `epoch` and clock `now`.
    pub fn exclusive(&mut self, lead: u8, epoch: u64, now: Tokens) -> Caught {
        match self.exclusive_share(lead, epoch, now) {
            None => Caught::Unmeasured,
            Some((share, age)) => Caught::Measured { rate: share.rate(), age },
        }
    }

    /// Layer `layer`'s cohort tallies `(reads, steps)` at bank-wide `epoch`;
    /// none before any step resolves there.
    pub fn cohort_share(&mut self, layer: usize, epoch: u64) -> Option<Share> {
        self.cohorts[layer].read(epoch).map(Share::sum)
    }

    /// Layer `layer`'s mean cohort size a step at bank-wide `epoch`; none
    /// before any step resolves there.
    pub fn cohort(&mut self, layer: usize, epoch: u64) -> Option<f64> {
        self.cohort_share(layer, epoch).map(Share::rate)
    }
}

#[derive(Debug, Default)]
struct Entry {
    interests: Vec<Interest>,
    /// Each step a read's admission was decided for, and the combination
    /// it was decided on, once each: what the step's outcome counts against
    /// ([`Calibration`]).
    decided: Vec<(Step, Combination)>,
}

impl Entry {
    /// Its interests that want a read: neither passed nor fallen.
    fn live(&self) -> impl Iterator<Item = &Interest> + Clone {
        self.interests.iter().filter(|interest| interest.passed.is_none() && !interest.fell)
    }

    /// Its earliest live interest's step, and the probability it is used
    /// there ([`Self::chance`]), at its group's `epoch`.
    fn derived(&self, calibration: &mut Calibration, group: usize, epoch: u64) -> Option<Derived> {
        let step = self.live().map(Interest::step).min()?;
        Some(Derived { step, p: self.chance(step, calibration, group, epoch) })
    }

    /// Whether an exact request makes this expert due at `now`: a step
    /// parks on it, unpriced.
    fn is_due(&self, now: Step) -> bool {
        self.live().any(|interest| interest.source == Source::Exact && interest.step() <= now)
    }

    /// The combination of the predicted interests for `step`, live,
    /// passed or fallen ([`Combination`]): what its use there is
    /// calibrated by; none where an exact row requests it there, which is
    /// certain, or none predicts it.
    fn named(&self, step: Step) -> Option<Combination> {
        if self.at(step).any(|interest| interest.source == Source::Exact) {
            return None;
        }
        Combination::of(self.interests.iter().filter(|interest| interest.step() == step).filter_map(Interest::kind))
    }

    /// The probability it is used at `step`: certain where an exact row
    /// requests it there, else its predicted names' combination's measured
    /// rate in `group` at `epoch`.
    fn chance(&self, step: Step, calibration: &mut Calibration, group: usize, epoch: u64) -> f64 {
        self.named(step).map_or(1., |combination| calibration.combined(group, combination, epoch))
    }

    /// Note that a read's admission for `step` was decided on its names
    /// there, once each combination.
    fn decide(&mut self, step: Step) {
        if let Some(combination) = self.named(step) {
            if !self.decided.iter().any(|(at, decided)| *at == step && *decided == combination) {
                self.decided.push((step, combination));
            }
        }
    }

    /// The earliest step a live interest wants it at whose use there is
    /// admitted by `admit`, and that probability ([`Self::chance`]): an
    /// early lookahead hint below the break-even never shadows an admitted
    /// ahead-row interest for the next step, nor pulls its deadline ahead
    /// of genuine earlier-step reads. Each step is visited in ascending
    /// order: an expert holds a few interests, so each visit is a pass over
    /// them. Each step visited is a decision on its names ([`Self::decide`]).
    fn admitted_class(&mut self, admit: &impl Admit, calibration: &mut Calibration, group: usize, epoch: u64) -> Option<Class> {
        let mut step = self.live().map(Interest::step).min()?;
        loop {
            self.decide(step);
            let p = self.chance(step, calibration, group, epoch);
            if admit.admits(p) {
                return Some(Class::Expected { step, p });
            }
            step = self.live().map(Interest::step).filter(|&later| later > step).min()?;
        }
    }

    /// The live interests wanting it at `step`.
    fn at(&self, step: Step) -> impl Iterator<Item = &Interest> + Clone {
        self.live().filter(move |interest| interest.step() == step)
    }

    /// The live interest that leads its read at `step`: the nearest name
    /// ([`Interest::nearness`]).
    fn leading(&self, step: Step) -> Option<&Interest> {
        self.at(step).min_by_key(|interest| interest.nearness())
    }

    /// Its read for `step`, served from a record that landed for that step
    /// (`issued`..=`landed`), as a saved read of the step's cohort ([`Cohort`]):
    /// none where no exact row's hint named it for the step (a hint's admission
    /// is what the cohort counts). Its catchers are the hint leads whose name,
    /// plus the landing's measured duration, came no later than the step's
    /// route there (the earliest its interests were seen passed); a route the
    /// host never saw pass before the step completed times no read, and is
    /// caught by none.
    fn saved(&self, step: Step, issued: Instant, landed: Instant) -> Option<Catchers> {
        let named = self.interests.iter().filter(|interest| interest.when == When::Step(step));
        let hints = named.clone().filter_map(|interest| Some((interest.hint()?, interest.named))).collect::<Vec<_>>();
        if hints.is_empty() {
            return None;
        }
        assert!(landed >= issued, "a read ahead landed before it was issued");
        let took = landed - issued;
        let mut catchers = Catchers::default();
        if let Some(read_at) = named.filter_map(|interest| interest.passed).min() {
            for (lead, at) in hints {
                if at + took <= read_at {
                    catchers.caught_by(lead);
                }
            }
        }
        Some(catchers)
    }

    /// A hint read ahead that was still in flight when demand arrived
    /// (hastened to demand): credit the earliest exact-row hint lead that
    /// named it for `step` (the lead that started the read and provided its
    /// overlap), so `Catchers::only()` identifies that lead for the hint gate.
    fn hastened(&self, step: Step) -> Option<Catchers> {
        let (lead, _) = self
            .interests
            .iter()
            .filter(|interest| interest.when == When::Step(step))
            .filter_map(|interest| Some((interest.hint()?, interest.named)))
            .min_by_key(|&(_, at)| at)?;
        let mut catchers = Catchers::default();
        catchers.caught_by(lead);
        Some(catchers)
    }
}

/// A decode launch begun: each batch row, the padding rows its class runs
/// past them, and each request's pass index in it.
#[derive(Debug)]
struct Launch {
    rows: Box<[DecodeRow]>,
    padding: usize,
    passes: Vec<(RequestId, u64)>,
}

impl Launch {
    fn pass(&self, request: RequestId) -> u64 {
        self.passes.iter().find(|&&(of, _)| of == request).map(|&(_, pass)| pass).expect("a launch's row's request has its pass")
    }
}

/// Every layer's experts' interests, every predicted source's calibration,
/// and the decode launches and passes the names are bound by.
pub(in crate::experts) struct Ledger {
    layers: Vec<BTreeMap<usize, Entry>>,
    /// Each layer's bank group, whose tier-0 epochs its combination rates
    /// roll on.
    groups: Vec<usize>,
    calibration: Calibration,
    stands: Stands,
    time: Time,
    /// The latest route the host has seen (step and layer): where a step
    /// is due from and an expected read's time is counted from.
    reached: (Step, usize),
    /// Per layer, the latest step whose route there the host has seen done
    /// (its request out): a ring its route published after it, or one of
    /// the next layer's routes ([`Self::route_seen`]).
    routed: Vec<Step>,
    /// The interests published so far ([`Interest::order`]).
    published: u64,
    /// The decode launches begun and not ended, by step.
    launches: BTreeMap<Step, Launch>,
    /// Each open request's decode passes begun.
    passes: HashMap<RequestId, u64>,
    /// The experts whose aggregate changed off their read's derivation (a
    /// told event), owed one ([`Self::take_owed`]).
    owed: BTreeSet<(usize, usize)>,
    /// The reads ahead issued whose bytes' fate is not yet known, by
    /// `(layer, expert)`: at most one landing per expert, so a later read of
    /// one replaces its earlier, whose landed bytes went unused.
    reads: HashMap<(usize, usize), Issued>,
    /// Hint reads ahead that were hastened to demand (still in flight when
    /// demand arrived, `Self::requested(..., true)`), keyed by
    /// `(step, layer, expert)` until `Self::resolve` forms the step's cohort.
    hastened: BTreeSet<(Step, usize, usize)>,
    /// Hint reads ahead that landed (`Self::landed`), keyed by
    /// `(step, layer, expert)` -> `(issued, landed)` until `Self::resolve`
    /// forms the step's cohort: preserves the landing duration even if the
    /// expert is evicted and re-issued before `Completed(step)` runs.
    landed_hints: HashMap<(Step, usize, usize), (Instant, Instant)>,
    /// The fates learned since last taken ([`Self::take_spent`]).
    spent: Vec<Spent>,
    /// Each open request's passes with rows ahead awaiting their verdicts,
    /// and past them their deeper front rows' judgments, by step: an open
    /// request that ran one keeps its map until it closes, so a discard of
    /// a closed request's pass is told apart.
    awaiting: HashMap<RequestId, BTreeMap<Step, Awaiting>>,
    /// The front rows' guesses judged since last taken, and the acceptable
    /// rows' ([`Self::take_judged`]).
    judged: Vec<Judged>,
    accepted: Vec<Accepted>,
    /// Active MTP head draft window (`(started, estimated_seconds)`): set on
    /// [`Told::DraftWindow`] and cleared on [`Told::Began`].
    draft_window: Option<(Instant, f64)>,
}

impl std::fmt::Debug for Ledger {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Ledger").finish_non_exhaustive()
    }
}

impl Ledger {
    /// None published, of layers in the bank groups `groups` lists (one a
    /// layer), whose rings name `ranks` experts a row, its rates rolling on
    /// `time`'s epochs, counting the verdicts into `stands` (the ranking's
    /// rates).
    pub fn new(groups: Vec<usize>, ranks: usize, time: Time, stands: Stands) -> Self {
        let layers = groups.len();
        let calibration = {
            let times = time.lock();
            let count = groups.iter().max().map_or(0, |&group| group + 1);
            Calibration::new((0..count).map(|group| times.group_epoch(group)), ranks)
        };
        Self {
            layers: (0..layers).map(|_| BTreeMap::new()).collect(),
            groups,
            calibration,
            stands,
            time,
            reached: (0, 0),
            routed: vec![0; layers],
            published: 0,
            launches: BTreeMap::new(),
            passes: HashMap::new(),
            owed: BTreeSet::new(),
            reads: HashMap::new(),
            hastened: BTreeSet::new(),
            landed_hints: HashMap::new(),
            spent: Vec::new(),
            awaiting: HashMap::new(),
            judged: Vec::new(),
            accepted: Vec::new(),
            draft_window: None,
        }
    }

    /// Bytes held by the ledger's combination calibration tables (charged
    /// against the load, §3.5).
    pub fn bytes(&self) -> u64 {
        self.calibration.bytes()
    }

    /// The calibrated probability of `combination` at `layer` (test helper).
    #[cfg(test)]
    pub(in crate::experts) fn rate(&mut self, layer: usize, combination: Combination) -> f64 {
        let group = self.groups[layer];
        let epoch = self.time.lock().group_epoch(group);
        self.calibration.combined(group, combination, epoch)
    }

    /// Whether any calibrated rank `r ∈ 0..K` of an exact row's lookahead hint
    /// at `lead` targeting `target_layer` reaches `break_even`. Unmeasured
    /// ranks read their bound `1.0` and are admitted until calibrated.
    pub fn hint_admitted(&mut self, target_layer: usize, lead: u8, break_even: f64) -> bool {
        let group = self.groups[target_layer];
        let epoch = self.epoch(target_layer);
        let look = Look::of(Source::Lead { distance: lead, row: None });
        (0..self.calibration.ranks).any(|rank| {
            let combination = Combination { nearest: look, rank: rank as u8, looks: 1 };
            self.calibration.combined(group, combination, epoch) >= break_even
        })
    }

    /// Remaining idle drive seconds in the active MTP head draft window at
    /// `now` (`0.0` outside a draft window or once its estimate has elapsed).
    pub fn draft_window_remaining(&self, now: Instant) -> f64 {
        let Some((started, estimated)) = self.draft_window else {
            return 0.0;
        };
        (estimated - now.saturating_duration_since(started).as_secs_f64()).max(0.0)
    }

    /// Whether an MTP head draft window is currently active.
    pub fn in_draft_window(&self) -> bool {
        self.draft_window.is_some()
    }

    /// The latest decode step currently begun and not yet ended.
    pub fn latest_active_step(&self) -> Option<Step> {
        self.launches.keys().next_back().copied()
    }

    /// The host has seen step `step`'s route at `layer`.
    pub fn reach(&mut self, step: Step, layer: usize) {
        self.reached = self.reached.max((step, layer));
    }

    /// The latest route the host has seen.
    pub fn reached(&self) -> (Step, usize) {
        self.reached
    }

    /// A ring showed step `step`'s route at `layer` done, its request
    /// there out.
    pub fn route_seen(&mut self, step: Step, layer: usize) {
        self.routed[layer] = self.routed[layer].max(step);
    }

    /// The latest step whose route at `layer` a ring showed done.
    pub fn routed(&self, layer: usize) -> Step {
        self.routed[layer]
    }

    /// `layer`'s group's epoch now, which its combination rates roll on.
    fn epoch(&self, layer: usize) -> u64 {
        self.time.lock().group_epoch(self.groups[layer])
    }

    /// Learn what another thread told ([`Told`]), in the order told. What a
    /// completion resolved is returned ([`Learned`]).
    pub fn tell(&mut self, told: Told) -> Learned {
        match told {
            Told::Began { step, launch: Launched::Decode { rows, padding } } => self.begin(step, rows, padding),
            Told::Began { step, launch: Launched::Grouped(requests) } => self.begin_grouped(step, &requests),
            Told::Completed { step, chose, guesses } => return self.complete(step, &chose, guesses),
            Told::Ended { step, completed } => self.expire(step, completed),
            Told::Decided { step, request, verdict } => self.verdict(step, request, &verdict),
            Told::Discarded { step, request } => self.discard(step, request),
            Told::Closed(request) => self.close(request),
            Told::DraftWindow { estimated_seconds } => {
                assert!(
                    estimated_seconds.is_finite() && estimated_seconds >= 0.0,
                    "draft window duration must be finite and non-negative, got {estimated_seconds}"
                );
                self.draft_window = (estimated_seconds > 0.0).then(|| (Instant::now(), estimated_seconds));
            }
        }
        Learned::default()
    }

    /// How step `step`'s batch row `row` names its experts, its ring
    /// carrying the step's guess kinds (RING_GUESSES): exact where the step
    /// is no decode launch begun (its rows are all exact, of its own step: a
    /// launch without a decode record). A padding row (past the launch's
    /// rows, within its class) is exact: the device routes it, and resolves
    /// and reads its choices, as a row of the step. None of a front row that
    /// ran on no guess ([`ran_on`]): it names nothing. A row past the class,
    /// or a guess kind of a row that is no front row, is the device's broken
    /// invariant.
    pub fn row(&self, step: Step, row: usize, guesses: u32) -> Option<Origin> {
        let Some(launch) = self.launches.get(&step) else {
            return Some(Origin::Exact);
        };
        let kind = ran_on(guesses, row);
        let Some(decoded) = launch.rows.get(row) else {
            let rows = launch.rows.len();
            assert!(row < rows + launch.padding, "step {step}'s ring names batch row {row} of its {rows} and {} padding", launch.padding);
            assert!(kind.is_none(), "step {step}'s ring gives its padding row {row} a guess");
            return Some(Origin::Exact);
        };
        let Some(ahead) = decoded.ahead else {
            assert!(kind.is_none(), "step {step}'s ring gives its row {row} a guess, which is no front row");
            return Some(Origin::Exact);
        };
        let front = match ahead.front() {
            true => Some(kind?),
            false => {
                assert!(kind.is_none(), "step {step}'s ring gives its row {row} a guess, which is no front row");
                None
            }
        };
        let lineage = Lineage { request: decoded.request, row, guessed: ahead.guessed() };
        let row = Row { index: ahead.row(), front };
        let when = match ahead.exact_in() {
            0 => When::Step(step),
            exact_in => {
                let pass = launch.pass(decoded.request) + u64::from(exact_in);
                match self.bound(decoded.request, pass) {
                    Some(bound) => When::Step(bound),
                    None => When::Pass { request: decoded.request, pass, estimate: step + Step::from(exact_in) },
                }
            }
        };
        Some(Origin::Ahead { row, when, lineage })
    }

    /// The step of `request`'s `pass`-th pass, if it has begun.
    fn bound(&self, request: RequestId, pass: u64) -> Option<Step> {
        self.launches.iter().find(|(_, launch)| launch.passes.contains(&(request, pass))).map(|(&step, _)| step)
    }

    /// A decode launch of step `step` begins with `rows`: each of its
    /// requests' next pass, which binds the names waiting for it, and which
    /// awaits its verdict where it runs rows ahead: acceptable rows, or front
    /// rows (a pass's front rows are its ahead rows, in depth order,
    /// [`crate::Rows`]), from its exact row's position.
    fn begin(&mut self, step: Step, rows: Box<[DecodeRow]>, padding: usize) {
        self.draft_window = None;
        let mut passes: Vec<(RequestId, u64)> = Vec::new();
        let mut ahead: BTreeMap<RequestId, Awaiting> = BTreeMap::new();
        for (index, row) in rows.iter().enumerate() {
            if let Some(of) = row.ahead {
                let exact = rows.iter().filter(|other| other.request == row.request && other.ahead.is_none()).map(|other| other.position).max();
                let awaiting = ahead.entry(row.request).or_insert_with(|| Awaiting {
                    exact: exact.expect("a pass's rows ahead follow its exact row"),
                    decided: false,
                    fronted: Fronted::Without,
                });
                if of.front() {
                    awaiting.fronted = match awaiting.fronted {
                        Fronted::Without => Fronted::Running { first: index, rows: 1 },
                        Fronted::Running { first, rows } => Fronted::Running { first, rows: rows + 1 },
                        Fronted::Rows(_) | Fronted::Unread => unreachable!("a launch's pass runs"),
                    };
                }
            }
            if passes.iter().any(|&(request, _)| request == row.request) {
                continue;
            }
            let count = self.passes.entry(row.request).or_insert(0);
            *count += 1;
            passes.push((row.request, *count));
        }
        for (request, awaiting) in ahead {
            self.awaiting.entry(request).or_default().insert(step, awaiting);
        }
        self.each(|interest| match interest.when {
            When::Pass { request, pass, .. } if passes.contains(&(request, pass)) => {
                interest.when = When::Step(step);
                true
            }
            _ => false,
        });
        for read in self.reads.values_mut() {
            if let IssuedTarget::Pass(request, pass) = read.target {
                if passes.contains(&(request, pass)) {
                    read.target = IssuedTarget::Bound(step);
                }
            }
        }
        self.launches.insert(step, Launch { rows, padding, passes });
    }

    /// A grouped launch of step `step` begins, `requests` running rows
    /// ahead in it: each such pass awaits its verdict. Its rows name
    /// nothing ahead, so it binds no name and is bound by none.
    fn begin_grouped(&mut self, step: Step, requests: &[RequestId]) {
        self.draft_window = None;
        for &request in requests {
            // A grouped pass's decision judges its proposals alone: the
            // truths it learns judge no front row (a request decoding in
            // front rows runs no grouped verification).
            self.awaiting.entry(request).or_default().insert(step, Awaiting { exact: 0, decided: false, fronted: Fronted::Without });
        }
    }

    /// `name` wants `expert` of `layer`, named at `at`. The same
    /// producer's later publication ([`Interest::same`]) replaces its
    /// earlier (a ring read again names the same experts), keeping its
    /// place in the order and when it was first named. Of a new interest,
    /// the probability its expert is used at its step, as the names live
    /// there now say ([`Entry::chance`]), is returned; none where it
    /// replaced its producer's earlier publication.
    pub fn publish(&mut self, layer: usize, expert: usize, name: Name, at: Instant) -> Option<f64> {
        let Name { source, when, rank, from, candidate, lineage } = name;
        let step = when.step();
        self.published += 1;
        let interest = Interest { source, when, rank, from, candidate, lineage, named: at, passed: None, fell: false, order: self.published };
        let (group, epoch) = (self.groups[layer], self.epoch(layer));
        let entry = self.layers[layer].entry(expert).or_default();
        let new = match entry.interests.iter_mut().find(|held| held.same(&interest)) {
            Some(held) => {
                *held = Interest { order: held.order, named: held.named, ..interest };
                false
            }
            None => {
                entry.interests.push(interest);
                true
            }
        };
        new.then(|| entry.chance(step, &mut self.calibration, group, epoch))
    }

    /// The route of each bound interest of `layer` that `passed` says no
    /// read can serve any more has passed it, seen at `at`: it wants no
    /// read, and waits for its step's outcome. An exact request passes as
    /// any source does, so its certainty no longer hides a later step's
    /// want of the expert (a front row's name for the next step). The
    /// experts whose aggregate it changed, for the executor to act on.
    pub fn pass(&mut self, layer: usize, at: Instant, passed: impl Fn(&Interest) -> bool) -> Vec<usize> {
        let mut changed = Vec::new();
        for (&expert, entry) in &mut self.layers[layer] {
            let mut any = false;
            for interest in entry.interests.iter_mut().filter(|interest| interest.passed.is_none() && interest.bound()) {
                if passed(interest) {
                    interest.passed = Some(at);
                    any = true;
                }
            }
            if any {
                changed.push(expert);
            }
        }
        changed
    }

    /// Step `step` completed, its exact rows having chosen at each layer
    /// the experts `chose` lists (ascending), its front rows running on
    /// guesses of the kinds `guesses` holds (none where no record carried
    /// them): every interest of the step ends, each combination a read's
    /// admission for it was decided on counts its outcome, and each layer
    /// forms its demand cohort ([`Cohort`]). The one resolution of every
    /// source, linear in the layer's entries. Each front row of its passes
    /// awaiting their verdicts takes its guess's kind; one of a request
    /// closed since its launch began is judged by none.
    fn complete(&mut self, step: Step, chose: &[Vec<Chosen>], guesses: Option<u32>) -> Learned {
        let mut learned = Learned::default();
        for (layer, chose) in chose.iter().enumerate() {
            self.resolve(step, layer, chose, &mut learned);
        }
        let Some(launch) = self.launches.get(&step) else {
            return learned;
        };
        let mut requests = launch.rows.iter().filter(|row| row.ahead.is_some_and(|ahead| ahead.front())).map(|row| row.request).collect::<Vec<_>>();
        requests.dedup();
        for request in requests {
            let Some(passes) = self.awaiting.get_mut(&request) else {
                // Closed since its launch began: its front rows are judged
                // by none (`close` judged none, as it awaited a launch).
                continue;
            };
            let awaiting = passes.get_mut(&step).unwrap_or_else(|| panic!("{request:?}'s pass of step {step} completed, awaiting no verdict"));
            let Fronted::Running { first, rows } = awaiting.fronted else {
                panic!("step {step}'s front rows completed twice");
            };
            // Its guessed rows lead its idle ones, which ran on no guess.
            awaiting.fronted = match guesses.map(|guesses| Kinds::of(guesses, first)) {
                Some(kinds) => match (0..rows).take_while(|&depth| kinds.at(usize::from(depth)).is_some()).count() as u8 {
                    0 => Fronted::Unread,
                    guessed => Fronted::Rows(FrontPass { first, kinds, rows: guessed, stood: 0, tokens: None }),
                },
                None => Fronted::Unread,
            };
        }
        learned
    }

    /// Step `step`'s resolution at `layer`, its exact rows having chosen
    /// `chose` there. Its cohort is formed here, once: each chosen expert
    /// a read ahead saves ([`Chosen::saves`]) that the tier did not serve
    /// missed; one it served from a record a read ahead landed for this
    /// step, which an exact row's hint named for it, is a saved read
    /// ([`Entry::saved`]); any other is a hit.
    fn resolve(&mut self, step: Step, layer: usize, chose: &[Chosen], learned: &mut Learned) {
        debug_assert!(chose.windows(2).all(|pair| pair[0].expert < pair[1].expert), "a layer's choices ascend, once each");
        let (group, epoch) = (self.groups[layer], self.epoch(layer));
        let Self { layers, calibration, owed, reads, hastened, landed_hints, spent, .. } = self;
        let mut cohort = Cohort { step, layer, misses: 0, saved: Vec::new() };
        for chosen in chose {
            // The chosen experts a read ahead was issued for, unspent until
            // now: used.
            let read = reads.remove(&(layer, chosen.expert));
            if let Some(Issued { basis, depth, verify, bytes, fell, .. }) = read {
                spent.push(Spent::Used { basis, depth, verify, fell, bytes });
            }
            match (chosen.saves, chosen.tier) {
                (false, _) => {}
                (true, false) => {
                    let saved = hastened
                        .remove(&(step, layer, chosen.expert))
                        .then(|| layers[layer].get(&chosen.expert)?.hastened(step))
                        .flatten();
                    match saved {
                        Some(catchers) => cohort.saved.push(catchers),
                        None => cohort.misses += 1,
                    }
                }
                (true, true) => {
                    let timing = landed_hints.remove(&(step, layer, chosen.expert)).or_else(|| {
                        read.filter(|read| read.step == step && read.basis == Basis::Hint)
                            .and_then(|read| read.landed.map(|landed| (read.issued, landed)))
                    });
                    let saved = timing.and_then(|(issued, landed)| layers[layer].get(&chosen.expert)?.saved(step, issued, landed));
                    cohort.saved.extend(saved);
                }
            }
        }
        hastened.retain(|&(s, l, _)| !(s <= step && l == layer));
        landed_hints.retain(|&(s, l, _), _| !(s <= step && l == layer));
        // Any read ahead bound to this step (or earlier) at `layer` that was
        // not chosen here is now settled unused. Reads still pending for a
        // future step (`Pass` or `Bound(target)` with `target > step`) stay live.
        reads.retain(|&(l, _), read| {
            if l == layer && matches!(read.target, IssuedTarget::Bound(target) if target <= step) {
                spent.push(Spent::Unused {
                    basis: read.basis,
                    depth: read.depth,
                    verify: read.verify,
                    fell: read.fell,
                    bytes: read.bytes,
                });
                false
            } else {
                true
            }
        });
        learned.cohorts.push(cohort);
        let mut emptied = Vec::new();
        for (&expert, entry) in &mut layers[layer] {
            let chosen = chose.binary_search_by_key(&expert, |chosen| chosen.expert).ok().map(|at| chose[at]);
            let used = chosen.is_some_and(|chosen| chosen.saves);
            let early = used && chosen.is_some_and(|chosen| chosen.tier);
            // Each combination a read's admission for the step was decided
            // on: its outcome, once each.
            entry.decided.retain(|&(at, combination)| {
                if at != step {
                    return true;
                }
                calibration.count(group, combination, epoch, used);
                false
            });
            let before = entry.interests.len();
            entry.interests.retain(|interest| {
                if interest.when != When::Step(step) {
                    return true;
                }
                if let (Some(_), false) = (interest.kind(), interest.fell) {
                    let Interest { source, candidate, .. } = *interest;
                    learned.resolved.push(Resolved { source, layer, expert, candidate, routed: chosen.is_some(), used, early });
                }
                false
            });
            if entry.interests.len() != before {
                owed.insert((layer, expert));
            }
            if entry.interests.is_empty() {
                emptied.push(expert);
            }
        }
        for expert in emptied {
            self.layers[layer].remove(&expert);
        }
    }

    /// The entry of `request`'s pass of step `step`, which awaits its
    /// verdict: ended now. One that awaits none, or whose verdict came, is
    /// a broken invariant.
    fn awaited(&mut self, step: Step, request: RequestId) -> Awaiting {
        let awaiting = self
            .awaiting
            .get_mut(&request)
            .and_then(|passes| passes.remove(&step))
            .filter(|awaiting| !awaiting.decided)
            .unwrap_or_else(|| panic!("{request:?}'s pass of step {step} awaits no verdict"));
        assert!(!matches!(awaiting.fronted, Fronted::Running { .. }), "{request:?}'s pass of step {step} ended before its launch completed");
        awaiting
    }

    /// Request `request`'s pass of step `step` was decided ([`Verdict`]):
    /// its acceptable rows' guesses count into the stands up to the first
    /// wrong one, and the names the pass's rows past it named drop; its
    /// front rows await their positions' truths, which this decision and
    /// the request's later ones learn ([`Self::judge`]). Only the judged
    /// pass's names drop: a later pass of the request runs on the chain made
    /// behind its predecessor from that pass's consistent rows alone
    /// (`guess_chain`), which no earlier guess's fall touches. A dropped name is unresolved, wanted
    /// by no step, so an unbegun read it alone wanted cancels and a begun
    /// one lands and falls to the ranking.
    fn verdict(&mut self, step: Step, request: RequestId, verdict: &Verdict) {
        let mut awaiting = self.awaited(step, request);
        let exact = awaiting.exact;
        let stood = verdict.acceptable.iter().take_while(|&&stood| stood).count();
        match &mut awaiting.fronted {
            Fronted::Without => {
                for (depth, &guess) in verdict.acceptable.iter().enumerate().take(stood + 1) {
                    self.stands.count(depth + 1, Form::Verify, guess);
                    let depth = u8::try_from(depth + 1).expect("a pass's acceptable rows fit a byte");
                    self.accepted.push(Accepted { depth, stood: guess });
                }
                if stood < verdict.acceptable.len() {
                    self.fall(request, step, stood);
                }
            }
            // Of no kind: it counts into no stand, but its first guess's
            // fall drops its names all the same.
            Fronted::Unread => {
                if verdict.front.first().zip(verdict.truths.first()).is_some_and(|(guess, truth)| guess != truth) {
                    self.fall(request, step, 0);
                }
            }
            Fronted::Running { .. } => unreachable!("an awaited pass's launch completed"),
            Fronted::Rows(pass) => {
                assert_eq!(verdict.front.len(), usize::from(pass.rows), "{request:?}'s verdict of step {step} judges its front rows");
                pass.tokens = Some(verdict.front.clone());
                awaiting.decided = true;
                self.awaiting.get_mut(&request).expect("an awaiting request").insert(step, awaiting);
            }
        }
        self.judge(request, step, exact, &verdict.truths);
    }

    /// The `truths` a decision of `request`'s pass of step `step` learned,
    /// of the positions after its exact row's (`exact`), judge each front
    /// row of the request's decided passes up to it whose token is at one
    /// of them, its shallower rows having stood: each counts into the
    /// stands at its depth and kind; a fall drops the names its pass's rows
    /// from it on named, and leaves the deeper rows unjudged. A front row
    /// whose position the truths passed with no decision learning it (a
    /// pass between ran no rows ahead) is left unjudged with every row
    /// deeper. A pass none of whose rows awaits a judgment any more leaves.
    fn judge(&mut self, request: RequestId, step: Step, exact: u64, truths: &[TokenId]) {
        let Some(passes) = self.awaiting.get_mut(&request) else {
            return;
        };
        let learned = exact + truths.len() as u64;
        let mut fell = Vec::new();
        let mut done = Vec::new();
        for (&of, awaiting) in passes.range_mut(..=step) {
            let Awaiting { exact: base, decided: true, fronted: Fronted::Rows(pass) } = awaiting else {
                continue;
            };
            let tokens = pass.tokens.as_ref().expect("a decided pass holds its front tokens");
            while pass.stood < pass.rows {
                let depth = pass.stood + 1;
                let position = *base + u64::from(depth);
                if position > learned {
                    break;
                }
                let kind = pass.kinds.at(usize::from(depth) - 1).expect("a front pass's row has its kind");
                let truth = position.checked_sub(exact + 1).and_then(|at| truths.get(at as usize));
                let Some(&truth) = truth else {
                    // Passed with no decision learning its truth.
                    self.judged.extend(pass.pending(of));
                    pass.stood = pass.rows;
                    break;
                };
                let stood = tokens[usize::from(depth) - 1] == truth;
                self.stands.count(usize::from(depth), Form::Front, stood);
                self.judged.push(Judged { step: of, row: pass.first + usize::from(depth) - 1, depth, kind, stood: Some(stood) });
                if !stood {
                    fell.push((of, depth - 1));
                    pass.stood = depth;
                    self.judged.extend(pass.pending(of));
                    pass.stood = pass.rows;
                    break;
                }
                pass.stood = depth;
            }
            if pass.stood == pass.rows {
                done.push(of);
            }
        }
        for of in done {
            passes.remove(&of);
        }
        for (of, stood) in fell {
            self.fall(request, of, usize::from(stood));
        }
    }

    /// The names `request`'s pass of step `step` published from its rows
    /// past its first `stood` guesses fall: they want no read, and a read
    /// one led falls with it (its name's guess is no step's), but each
    /// stays for its step's outcome ([`Interest::fell`]).
    fn fall(&mut self, request: RequestId, step: Step, stood: usize) {
        let fallen = |interest: &Interest| {
            interest.from.point.sequence == step && interest.lineage.is_some_and(|lineage| lineage.request == request && usize::from(lineage.guessed.get()) > stood)
        };
        let Self { layers, owed, reads, spent, .. } = self;
        for (layer, entries) in layers.iter_mut().enumerate() {
            for (&expert, entry) in entries.iter_mut() {
                let mut any = false;
                for interest in entry.interests.iter_mut().filter(|interest| !interest.fell && fallen(interest)) {
                    interest.fell = true;
                    any = true;
                    if let Some(read) = reads.get_mut(&(layer, expert)).filter(|read| read.led == interest.order && !read.fell) {
                        read.fell = true;
                        spent.push(Spent::Fell { basis: read.basis, bytes: read.bytes });
                    }
                }
                if any {
                    owed.insert((layer, expert));
                }
            }
        }
    }

    /// Request `request`'s pass of step `step` was discarded: it is judged
    /// by none, and the names its rows published drop, as a pass's that
    /// never completed. A request closed since awaits nothing: its passes
    /// and names went with it.
    fn discard(&mut self, step: Step, request: RequestId) {
        if !self.awaiting.contains_key(&request) {
            return;
        }
        if let Fronted::Rows(pass) = self.awaited(step, request).fronted {
            self.judged.extend(pass.pending(step));
        }
        self.drop_where(|interest| interest.lineage.is_some_and(|lineage| lineage.request == request) && interest.from.point.sequence == step);
    }

    /// Request `request` closed: every name its rows published drops, its
    /// passes are forgotten, and its front rows awaiting judgments are
    /// judged by none.
    fn close(&mut self, request: RequestId) {
        self.passes.remove(&request);
        for (step, awaiting) in self.awaiting.remove(&request).unwrap_or_default() {
            if let Fronted::Rows(pass) = awaiting.fronted {
                self.judged.extend(pass.pending(step));
            }
        }
        self.drop_where(|interest| interest.lineage.is_some_and(|lineage| lineage.request == request));
        let Self { reads, spent, .. } = self;
        reads.retain(|_, read| {
            if matches!(read.target, IssuedTarget::Pass(r, _) if r == request) {
                spent.push(Spent::Unused {
                    basis: read.basis,
                    depth: read.depth,
                    verify: read.verify,
                    fell: read.fell,
                    bytes: read.bytes,
                });
                false
            } else {
                true
            }
        });
    }

    /// A front ring of `request`'s pass of step `step` at `layer` is read:
    /// every live name an earlier pass of the request's front rows
    /// published at the layer drops. The later pass's rows guess every
    /// position the earlier's still wanted, on more verified tokens, so a
    /// position is named once (a combination would hold one token's route
    /// once per pass that named it); an earlier name of
    /// this step's own token has passed, its exact route there run before
    /// its front ring. A read one led keeps its fate, which the later
    /// pass's name of the same expert, or none, decides; an unbegun one
    /// nothing wants any more cancels.
    pub fn supersede(&mut self, layer: usize, request: RequestId, step: Step) {
        let Self { layers, owed, .. } = self;
        layers[layer].retain(|&expert, entry| {
            let before = entry.interests.len();
            entry.interests.retain(|interest| {
                let superseded = matches!(interest.source, Source::Ahead(row) if !row.acceptable())
                    && interest.from.point.sequence < step
                    && interest.lineage.is_some_and(|lineage| lineage.request == request);
                !superseded
            });
            if entry.interests.len() != before {
                owed.insert((layer, expert));
            }
            !entry.interests.is_empty()
        });
    }

    /// Step `step` ended, `completed` or not: its bound interests no
    /// completion resolved end, at every layer; a launch that never
    /// completed awaits no verdict, and takes the names its ahead rows
    /// published with it; and names still waiting for a later pass are
    /// dated past it.
    fn expire(&mut self, step: Step, completed: bool) {
        self.launches.remove(&step);
        if !completed {
            for passes in self.awaiting.values_mut() {
                passes.remove(&step);
            }
        }
        self.drop_where(|interest| interest.when == When::Step(step) || (!completed && interest.lineage.is_some() && interest.from.point.sequence == step));
        self.each(|interest| {
            if let When::Pass { estimate, .. } = &mut interest.when {
                if *estimate <= step {
                    *estimate = step + 1;
                    return true;
                }
            }
            false
        });
    }

    /// Drop every interest `drop` names, owing a derivation to each expert
    /// it changed. A read ahead an ahead row's name led falls with it: that
    /// name's guess is no step's.
    fn drop_where(&mut self, drop: impl Fn(&Interest) -> bool) {
        let Self { layers, owed, reads, spent, .. } = self;
        for (layer, entries) in layers.iter_mut().enumerate() {
            entries.retain(|&expert, entry| {
                let before = entry.interests.len();
                entry.interests.retain(|interest| {
                    if !drop(interest) {
                        return true;
                    }
                    if let Some(read) = reads.get_mut(&(layer, expert)).filter(|read| read.led == interest.order && !read.fell && interest.lineage.is_some()) {
                        read.fell = true;
                        spent.push(Spent::Fell { basis: read.basis, bytes: read.bytes });
                    }
                    false
                });
                if entry.interests.len() != before {
                    owed.insert((layer, expert));
                }
                !entry.interests.is_empty()
            });
        }
    }

    /// Change every interest `change` changes (true where it did), owing a
    /// derivation to each expert it changed.
    fn each(&mut self, mut change: impl FnMut(&mut Interest) -> bool) {
        let Self { layers, owed, .. } = self;
        for (layer, entries) in layers.iter_mut().enumerate() {
            for (&expert, entry) in entries.iter_mut() {
                let mut any = false;
                for interest in &mut entry.interests {
                    any |= change(interest);
                }
                if any {
                    owed.insert((layer, expert));
                }
            }
        }
    }

    /// A read ahead of `bytes` of `expert` of `layer` was issued at `at`,
    /// `led` the live interest that led its aggregate ([`Scheduled::leading`]):
    /// its bytes are this interest's until their fate is known ([`Spent`]),
    /// and its issue time and landing time ([`Self::landed`]) measure its
    /// landing duration for the step's cohort ([`Entry::saved`]).
    pub fn issued(&mut self, layer: usize, expert: usize, led: &Interest, bytes: u64, at: Instant) {
        let target = match led.when {
            When::Step(step) => IssuedTarget::Bound(step),
            When::Pass { request, pass, .. } => IssuedTarget::Pass(request, pass),
        };
        let read = Issued {
            basis: Basis::of(led.source),
            depth: led.source.depth(),
            verify: led.source.verifies(),
            led: led.order,
            bytes,
            fell: false,
            step: led.step(),
            target,
            issued: at,
            landed: None,
        };
        if let Some(Issued { basis, depth, verify, bytes, fell, .. }) = self.reads.insert((layer, expert), read) {
            self.spent.push(Spent::Unused { basis, depth, verify, fell, bytes });
        }
    }

    /// The read ahead of `expert` of `layer` landed at `at`: records its
    /// landing time so its step's resolution measures its duration
    /// ([`Entry::saved`]).
    pub fn landed(&mut self, layer: usize, expert: usize, at: Instant) {
        if let Some(read) = self.reads.get_mut(&(layer, expert)) {
            read.landed = Some(at);
            if read.basis == Basis::Hint {
                self.landed_hints.entry((read.step, layer, expert)).or_insert((read.issued, at));
            }
        }
    }

    /// Record a completed Path 0 (`LinkMover`) opportunistic DMA staging copy
    /// for `(step, layer, expert)` so `HintGate` credits link-staged hint hits
    /// without polluting Path 1 (`Drive`) `Spent` telemetry.
    pub fn link_landed(&mut self, step: Step, layer: usize, expert: usize, issued: Instant, landed: Instant) {
        self.landed_hints.entry((step, layer, expert)).or_insert((issued, landed));
    }

    /// Whether a Path 0 (`LinkMover`) opportunistic DMA staging copy landed for
    /// `(step, layer, expert)`.
    pub fn has_link_landed(&self, step: Step, layer: usize, expert: usize) -> bool {
        self.landed_hints.contains_key(&(step, layer, expert))
    }

    /// The read ahead of `expert` of `layer` was cancelled before any piece
    /// began.
    pub fn cancelled(&mut self, layer: usize, expert: usize) {
        if let Some(Issued { basis, depth, verify, bytes, fell, .. }) = self.reads.remove(&(layer, expert)) {
            self.spent.push(Spent::Cancelled { basis, depth, verify, fell, bytes });
        }
    }

    /// An exact row requested `expert` of `layer` absent, a read ahead of
    /// it in flight where `reading`: that read is used; with none in flight
    /// a read ahead of it landed and was evicted unused (or failed), and
    /// the request reads it anew.
    pub fn requested(&mut self, layer: usize, expert: usize, reading: bool) {
        if let Some(Issued { basis, depth, verify, bytes, fell, step, .. }) = self.reads.remove(&(layer, expert)) {
            if reading {
                self.spent.push(Spent::Used { basis, depth, verify, fell, bytes });
                if basis == Basis::Hint {
                    self.hastened.insert((step, layer, expert));
                }
            } else {
                self.spent.push(Spent::Unused { basis, depth, verify, fell, bytes });
            }
        }
    }

    /// The reads ahead's fates learned since last taken.
    pub fn take_spent(&mut self) -> Vec<Spent> {
        std::mem::take(&mut self.spent)
    }

    /// The guesses judged since last taken: the front rows', one for each
    /// front row of a completed pass, once its request's verdict on the
    /// pass came or the request closed; and the acceptable rows', each of
    /// a decided pass up to its first that fell.
    pub fn take_judged(&mut self) -> (Vec<Judged>, Vec<Accepted>) {
        (std::mem::take(&mut self.judged), std::mem::take(&mut self.accepted))
    }

    /// The experts told events changed since last taken, whose reads are
    /// owed a derivation: a dropped interest's read, wanted by nothing now,
    /// is wanted by none ([`Self::wanted`]).
    pub fn take_owed(&mut self) -> Vec<(usize, usize)> {
        std::mem::take(&mut self.owed).into_iter().collect()
    }

    /// `expert`'s aggregate, if any live interest wants it.
    #[cfg(test)]
    pub fn derived(&mut self, layer: usize, expert: usize) -> Option<Derived> {
        let (group, epoch) = (self.groups[layer], self.epoch(layer));
        self.layers[layer].get(&expert)?.derived(&mut self.calibration, group, epoch)
    }

    /// What to do with `expert`'s read, standing as `read`, admitted by
    /// `admit`: the one function that orders every read. An expert no live
    /// interest is admitted to read is not issued, and an unbegun read of it
    /// cancels.
    pub fn act(&mut self, layer: usize, expert: usize, read: Read, admit: &impl Admit) -> Act {
        if matches!(read, Read::Landing | Read::Held) {
            return Act::Keep;
        }
        let (group, epoch) = (self.groups[layer], self.epoch(layer));
        let Self { layers, calibration, reached, .. } = self;
        let class = layers[layer].get_mut(&expert).and_then(|entry| {
            if entry.is_due(reached.0) {
                Some(Class::Due)
            } else {
                entry.admitted_class(admit, calibration, group, epoch)
            }
        });
        match (read, class) {
            (Read::Landing | Read::Held, _) => Act::Keep,
            (Read::Absent, Some(class)) => Act::Issue(class),
            (Read::Absent, None) => Act::Keep,
            (Read::Reading { .. }, Some(class)) => Act::Reprice(class),
            (Read::Reading { begun: false }, None) => Act::Cancel,
            (Read::Reading { begun: true }, None) => Act::Keep,
        }
    }

    /// Every expert with a live interest, `(layer, expert)`.
    pub fn wanted(&self) -> Vec<(usize, usize)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(layer, entries)| entries.iter().filter(|(_, entry)| entry.live().next().is_some()).map(move |(&expert, _)| (layer, expert)))
            .collect()
    }

    /// Whether `(layer, expert)` currently has any live interest in the ledger
    /// (used for Path 1 -> Path 0 pipeline chain-through when an NVMe landing
    /// finishes into Tier 1).
    pub fn is_wanted(&self, layer: usize, expert: usize) -> bool {
        self.layers
            .get(layer)
            .and_then(|entries| entries.get(&expert))
            .is_some_and(|entry| entry.live().next().is_some())
    }

    /// What to do with each of `experts`' moves on `path` (`PathKind::Drive`
    /// for NVMe -> RAM or `PathKind::Link` for RAM / peer -> VRAM), each
    /// standing as `read` says and admitted by its layer's `admit`, in the
    /// one order they are done in: cancellations first, then hastenings to
    /// due, then reprices before issues. On `PathKind::Drive`, expected reads
    /// order by `(step, -p, first)`; on `PathKind::Link`, expected staging
    /// moves order by `(step, layer, -p, first)` so early layers race ahead of
    /// layer-by-layer compute.
    pub fn schedule_for_path<A: Admit>(
        &mut self,
        path: magnitude_cost_model::PathKind,
        experts: impl IntoIterator<Item = (usize, usize)>,
        mut read: impl FnMut(usize, usize) -> Read,
        admit: impl Fn(usize) -> A,
    ) -> Vec<Scheduled> {
        let mut candidates: Vec<(usize, usize, Read, u8, Step, f64, u64)> = experts
            .into_iter()
            .map(|(layer, expert)| {
                let state = read(layer, expert);
                let (group, epoch) = (self.groups[layer], self.epoch(layer));
                let (step, p, first) = match self.layers[layer].get(&expert) {
                    Some(entry) => match entry.derived(&mut self.calibration, group, epoch) {
                        Some(Derived { step, p }) => {
                            let first = entry.at(step).map(|interest| interest.order).min().unwrap_or(u64::MAX);
                            (step, p, first)
                        }
                        None => (Step::MAX, 0., u64::MAX),
                    },
                    None => (Step::MAX, 0., u64::MAX),
                };
                // Evaluate unbegun queued moves ahead before new candidates so
                // already-queued moves keep their idle-compute slot without
                // oscillating, each group ordered by `(step, -p, first)`.
                let tier = match state {
                    Read::Reading { begun: false } => 0u8,
                    Read::Absent => 1,
                    _ => 2,
                };
                (layer, expert, state, tier, step, p, first)
            })
            .collect();
        candidates.sort_by(|left, right| {
            left.3
                .cmp(&right.3)
                .then(left.4.cmp(&right.4))
                .then(right.5.total_cmp(&left.5))
                .then(left.6.cmp(&right.6))
        });
        let mut scheduled = candidates
            .into_iter()
            .map(|(layer, expert, state, ..)| {
                let act = self.act(layer, expert, state, &admit(layer));
                let entry = self.layers[layer].get(&expert);
                let (leading, first) = match act {
                    Act::Issue(Class::Expected { step, .. }) | Act::Reprice(Class::Expected { step, .. }) => {
                        let entry = entry.expect("an expected read has a live interest");
                        let leading = entry.leading(step).copied();
                        let first = entry.at(step).map(|interest| interest.order).min().unwrap_or(u64::MAX);
                        (leading, first)
                    }
                    _ => (None, u64::MAX),
                };
                Scheduled { layer, expert, act, leading, first }
            })
            .collect::<Vec<_>>();
        scheduled.sort_by(|left, right| {
            let rank = |scheduled: &Scheduled| match scheduled.act {
                Act::Cancel => (0u8, 0, 0, 0.),
                Act::Reprice(Class::Due) => (1, 0, 0, 0.),
                Act::Reprice(Class::Expected { step, p }) => (2, step, if path == magnitude_cost_model::PathKind::Link { scheduled.layer } else { 0 }, -p),
                Act::Issue(Class::Expected { step, p }) => (3, step, if path == magnitude_cost_model::PathKind::Link { scheduled.layer } else { 0 }, -p),
                Act::Issue(Class::Due) | Act::Keep => (4, 0, 0, 0.),
            };
            let (left_rank, right_rank) = (rank(left), rank(right));
            left_rank
                .0
                .cmp(&right_rank.0)
                .then(left_rank.1.cmp(&right_rank.1))
                .then(left_rank.2.cmp(&right_rank.2))
                .then(left_rank.3.total_cmp(&right_rank.3))
                .then(left.first.cmp(&right.first))
        });
        scheduled
    }

    /// Drive-path scheduling (`PathKind::Drive`), delegating to
    /// [`Self::schedule_for_path`].
    pub fn schedule<A: Admit>(&mut self, experts: impl IntoIterator<Item = (usize, usize)>, read: impl FnMut(usize, usize) -> Read, admit: impl Fn(usize) -> A) -> Vec<Scheduled> {
        self.schedule_for_path(magnitude_cost_model::PathKind::Drive, experts, read, admit)
    }

    #[cfg(test)]
    pub fn leading(&self, layer: usize, expert: usize) -> Option<Interest> {
        let entry = self.layers[layer].get(&expert)?;
        let step = entry.live().map(Interest::step).min()?;
        entry.leading(step).copied()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::horizon::{Advance, Tier};
    use super::super::watcher::AheadOf;
    use super::*;

    /// Admission at a break-even of `even`.
    struct Flat {
        even: f64,
    }

    impl Admit for Flat {
        fn admits(&self, p: f64) -> bool {
            p >= self.even
        }
    }

    /// An admission no due read may ask.
    struct Unpriced;

    impl Admit for Unpriced {
        fn admits(&self, _: f64) -> bool {
            panic!("a due read was priced")
        }
    }

    const PRICE: Flat = Flat { even: 0.5 };
    const RANKS: usize = 10;
    const FRONT: Row = Row { index: 0, front: Some(GuessKind::Chained) };
    const AHEAD: Source = Source::Ahead(FRONT);
    const LEAD: Source = Source::Lead { distance: 1, row: None };

    fn now() -> Instant {
        Instant::now()
    }

    fn name(source: Source, step: Step) -> Name {
        Name { source, when: When::Step(step), rank: 0, from: At::whole(step.saturating_sub(1), 0), candidate: true, lineage: None }
    }

    /// Each expert chosen at a layer, a read ahead saving it.
    fn chosen(experts: &[usize]) -> Vec<Chosen> {
        experts.iter().map(|&expert| Chosen { expert, saves: true, tier: false }).collect()
    }

    fn resolve(ledger: &mut Ledger, step: Step, layer: usize, experts: &[usize]) -> Vec<Resolved> {
        let mut chose = vec![Vec::new(); layer + 1];
        chose[layer] = chosen(experts);
        ledger.tell(Told::Completed { step, chose, guesses: Some(0) }).resolved
    }

    /// A step's guess kinds word of batch row `row`'s front row's guess of
    /// `kind` (RING_GUESSES).
    fn kinds(row: usize, kind: GuessKind) -> u32 {
        magnitude_kernels::step_guesses([(row, kind.word())]) as u32
    }

    /// A ledger of `layers` layers, one group, its rates its own.
    fn ledger(layers: usize) -> Ledger {
        let time = Time::new(&[vec![1000]], 16);
        Ledger::new(vec![0; layers], RANKS, time.clone(), Stands::new(time))
    }

    /// A ledger whose single-source combinations at rank 0 are measured from
    /// 100 outcomes at `p`.
    fn calibrated(layers: usize, precisions: &[(Source, f64)]) -> Ledger {
        let mut ledger = ledger(layers);
        for &(source, p) in precisions {
            let used = (p * 100.).round() as usize;
            let combination = Combination::of([Kind { source, rank: 0 }]).expect("a name");
            for index in 0..100 {
                ledger.calibration.count(0, combination, 0, index < used);
            }
        }
        ledger.reach(5, 0);
        ledger
    }

    /// Stand rates are keyed by `(depth, form)`, never by guess kind; `through_rows`
    /// is their product through `depth`; and once a row is judged it holds its last
    /// rate with its age across arbitrarily many unrun epochs and never returns to
    /// `Unjudged` / 1.
    #[test]
    fn a_stand_rate_is_judged_once_and_never_unjudged() {
        let time = Time::new(&[vec![2]], 16);
        let stands = Stands::new(time.clone());
        assert_eq!(stands.rate(1, Form::Front), None);
        assert_eq!(stands.rate(1, Form::Verify), None);
        assert_eq!(stands.through_rows(3, Form::Front), 1.0, "unjudged rows read their bound 1");

        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        // Front row 1: 2 of 3 stood; front row 2: 1 of 2 stood; verify row 1: 0 of 1 stood.
        for stood in [true, true, false] {
            stands.count(1, Form::Front, stood);
        }
        for stood in [true, false] {
            stands.count(2, Form::Front, stood);
        }
        stands.count(1, Form::Verify, false);

        assert!((stands.rate(1, Form::Front).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((stands.rate(2, Form::Front).unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(stands.rate(3, Form::Front), None, "depth 3 still unjudged");
        assert!((stands.through_rows(2, Form::Front) - (2.0 / 3.0) * 0.5).abs() < 1e-12);
        assert!((stands.through_rows(3, Form::Front) - (2.0 / 3.0) * 0.5).abs() < 1e-12, "unjudged depth 3 multiplies by 1");
        assert_eq!(stands.rate(1, Form::Verify), Some(0.0), "verify form is independent of front");
        assert_eq!(stands.through_rows(2, Form::Verify), 0.0);

        // Both forms judged at the same clock position have equal age, so
        // `live_rate` keeps each form's own rate; unjudged stays `None`.
        assert!((stands.live_rate(1, Form::Front).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(stands.live_rate(1, Form::Verify), Some(0.0));
        assert_eq!(stands.live_rate(2, Form::Verify), None, "unjudged verify stays None");

        // Advance many bank-wide epochs without judging front row 2 again:
        // it never returns to Unjudged, keeps its last rate, and its age grows.
        for step in 2..=11 {
            let mut times = time.lock();
            times.register(step, Advance::Known(Tokens::new(1.0)));
            times.moved(0, Tier::COMPUTE);
            times.moved(0, Tier::COMPUTE);
        }
        assert!(time.lock().bank_epoch() >= 10);
        let (rate, age) = stands.judged(2, Form::Front).expect("never returns to unjudged");
        assert!((rate - 0.5).abs() < 1e-12, "keeps its last measured rate across empty epochs");
        assert_eq!(age, Tokens::new(10.0), "age tracks tokens since last judgement");
        assert!((stands.through_rows(2, Form::Front) - (2.0 / 3.0) * 0.5).abs() < 1e-12);
    }

    /// Silent deeper `Form::Verify` rows expire from `live_rate` after
    /// `STAND_STALE_BASE` tokens so a transient warmup or hard-passage dip
    /// cannot trap decode at width 1 forever; re-entering drops the frozen
    /// pre-silence tally, restores `1.0` immediately when the probe stands,
    /// and backs off exponentially (`32 → 64 → 128 → 256`) when probes fall.
    #[test]
    fn silent_verify_row_expires_with_backoff_and_recovers_cleanly() {
        let time = Time::new(&[vec![4]], 16);
        let stands = Stands::new(time.clone());

        // Warmup at step 1: row 1 stands, row 2 falls 0/5 times.
        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        stands.count(1, Form::Verify, true);
        for _ in 0..5 {
            stands.count(2, Form::Verify, false);
        }
        assert_eq!(stands.live_rate(2, Form::Verify), Some(0.0));
        assert_eq!(stands.stale_after(2, Form::Verify), STAND_STALE_BASE);

        // Advance 31 tokens (still below STAND_STALE_BASE = 32) while only
        // row 1 is judged (width-1 rounds): row 2 stays at its measured 0.0.
        let mut step = 2u64;
        for _ in 0..31 {
            let mut times = time.lock();
            times.register(step, Advance::Known(Tokens::new(1.0)));
            times.moved(0, Tier::COMPUTE);
            drop(times);
            stands.count(1, Form::Verify, true);
            step += 1;
        }
        assert_eq!(stands.live_rate(1, Form::Verify), Some(1.0));
        assert_eq!(stands.live_rate(2, Form::Verify), Some(0.0), "fresh before 32 tokens");

        // One more token (age == 32): row 2 expires from `live_rate` so the
        // regime carries forward row 1's 1.0 and probes width 2, while raw
        // `judged` preserves its history.
        time.lock().register(step, Advance::Known(Tokens::new(1.0)));
        step += 1;
        assert_eq!(stands.live_rate(2, Form::Verify), None, "expired at 32 tokens");
        assert_eq!(stands.rate(2, Form::Verify), Some(0.0), "raw history preserved");

        // Re-entry probe fails: stale 0/5 is dropped, fresh 0/1 recorded,
        // and backoff doubles to 64 tokens.
        stands.count(2, Form::Verify, false);
        assert_eq!(stands.live_rate(2, Form::Verify), Some(0.0));
        assert_eq!(stands.stale_after(2, Form::Verify), STAND_STALE_BASE * 2.0);

        // Advance 32 tokens: still not expired because backoff is now 64.
        for _ in 0..32 {
            time.lock().register(step, Advance::Known(Tokens::new(1.0)));
            step += 1;
        }
        assert_eq!(stands.live_rate(2, Form::Verify), Some(0.0), "backed off to 64 tokens");

        // Advance another 32 tokens (total 64): expires again.
        for _ in 0..32 {
            time.lock().register(step, Advance::Known(Tokens::new(1.0)));
            step += 1;
        }
        assert_eq!(stands.live_rate(2, Form::Verify), None, "expired at 64 tokens");

        // Re-entry probe stands: pre-silence failure tally is dropped so the
        // single standing probe immediately reads 1.0 (not 1/2 or 1/6), and
        // `stale_after` resets to `STAND_STALE_BASE`.
        stands.count(2, Form::Verify, true);
        assert_eq!(stands.live_rate(2, Form::Verify), Some(1.0));
        assert_eq!(stands.stale_after(2, Form::Verify), STAND_STALE_BASE);
    }

    /// Concurrent `Stands` reads and counts while `Time` advances `bank_epoch`
    /// and the token clock never regress a window's epoch or `Aged`'s clock:
    /// every operation reads `(epoch, now)` while holding `Stands::rows`.
    #[test]
    fn stands_concurrent_reads_and_counts_never_regress_epoch() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let time = Time::new(&[vec![2]], 16);
        let stands = Stands::new(time.clone());
        time.lock().register(1, Advance::Known(Tokens::new(1.0)));
        for row in 1..=ROWS {
            stands.count(row, Form::Front, true);
            stands.count(row, Form::Verify, true);
        }

        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for reader in 0..3usize {
                let stands = stands.clone();
                let done = &done;
                scope.spawn(move || {
                    let mut i = 0u64;
                    while !done.load(Ordering::Relaxed) {
                        let _ = stands.front_rates();
                        let _ = stands.judged((reader % ROWS) + 1, Form::Front);
                        let _ = stands.through_rows(ROWS, Form::Verify);
                        if i.is_multiple_of(8) {
                            stands.count((reader % ROWS) + 1, Form::Front, i.is_multiple_of(16));
                        }
                        i += 1;
                    }
                });
            }
            for step in 2..=2001u64 {
                {
                    let mut times = time.lock();
                    times.register(step, Advance::Known(Tokens::new(1.0)));
                    times.resized(0, Tier::COMPUTE, 2);
                    times.moved(0, Tier::COMPUTE);
                }
                stands.count(1, Form::Front, step.is_multiple_of(2));
                stands.count(1, Form::Verify, step.is_multiple_of(3));
            }
            done.store(true, Ordering::Relaxed);
        });
        assert!(time.lock().bank_epoch() >= 2000);
        assert!(stands.judged(1, Form::Front).is_some());
        assert!(stands.judged(1, Form::Verify).is_some());
    }

    /// A combination key is `(nearest look, best rank at nearest, distinct look count)`:
    /// two requests naming the same look count once; an unmeasured key reads 1, once,
    /// and never returns to unmeasured after its first outcome; table size is constant.
    #[test]
    fn a_combination_key_is_its_nearest_look_rank_and_count() {
        let front_d1 = Source::Ahead(Row { index: 0, front: Some(GuessKind::Chained) });
        let front_d2 = Source::Ahead(Row { index: 1, front: Some(GuessKind::Head) });
        let hint_d0_l1 = Source::Lead { distance: 1, row: None };
        let hint_d0_l2 = Source::Lead { distance: 2, row: None };
        let verify_d1 = Source::Ahead(Row { index: 0, front: None });

        // Exact-row hint (depth 0, Hint(1)) is nearer than depth-0 Hint(2) and depth-1 Route.
        // Two requests naming hint_d0_l1 at ranks 4 and 2 share one look, keeping best rank 2.
        let comb = Combination::of([
            Kind { source: front_d2, rank: 0 },
            Kind { source: front_d1, rank: 1 },
            Kind { source: hint_d0_l2, rank: 3 },
            Kind { source: hint_d0_l1, rank: 4 },
            Kind { source: hint_d0_l1, rank: 2 },
            Kind { source: verify_d1, rank: 0 },
        ])
        .expect("non-empty");
        assert_eq!(comb.nearest, Look { depth: 0, seen: Seen::Hint(1), form: Form::Front });
        assert_eq!(comb.rank, 2, "best rank among names from the nearest look");
        assert_eq!(comb.looks, 5, "two names from hint_d0_l1 count as one look");

        // Unmeasured key reads 1 once; after counting an outcome, rolling epochs
        // never returns it to 1 (unmeasured).
        let mut cal = Calibration::new([0u64].into_iter(), RANKS);
        let initial_bytes = cal.bytes();
        assert_eq!(initial_bytes, (LOOKS * RANKS * LOOKS * std::mem::size_of::<Window<Tally>>()) as u64);
        assert_eq!(cal.combined(0, comb, 0), 1.0, "unmeasured reads 1");
        cal.count(0, comb, 0, false);
        assert_eq!(cal.combined(0, comb, 0), 0.0);
        assert_eq!(cal.combined(0, comb, 10), 0.0, "never returns to unmeasured after empty epochs");
        assert_eq!(cal.bytes(), initial_bytes, "table size is constant");
    }

    /// A step's resolution forms one counterfactual demand cohort per layer:
    /// misses plus tier hits landed by a hint for that step, each saved read
    /// recording only the leads whose `name_time + landing_duration <= read_time`.
    /// `CohortRates` credits exclusive catches only while a lead runs, freezes
    /// with age while off, and counts cohort size on every step.
    #[test]
    fn a_cohort_forms_from_misses_and_hint_landings_with_exclusive_catchers() {
        let mut ledger = ledger(1);
        let t0 = Instant::now();
        let lead1 = Source::Lead { distance: 1, row: None };
        let lead2 = Source::Lead { distance: 2, row: None };

        // Expert 1: named by lead 2 at t0 and lead 1 at t0 + 1ms; issued at t0, landed at t0 + 2ms (dur = 2ms);
        // route passed at t0 + 5ms -> both lead 1 (1+2=3<=5) and lead 2 (0+2=2<=5) catch it -> neither exclusive.
        ledger.publish(0, 1, name(lead2, 6), t0);
        ledger.publish(0, 1, name(lead1, 6), t0 + Duration::from_millis(1));
        let led1 = ledger.leading(0, 1).unwrap();
        ledger.issued(0, 1, &led1, 10, t0);
        ledger.landed(0, 1, t0 + Duration::from_millis(2));

        // Expert 2: named by lead 1 at t0, lead 2 at t0 + 4ms; dur = 2ms; route passed at t0 + 5ms ->
        // lead 1 (0+2=2<=5) catches it, lead 2 (4+2=6>5) does not -> exclusive to lead 1.
        ledger.publish(0, 2, name(lead1, 6), t0);
        ledger.publish(0, 2, name(lead2, 6), t0 + Duration::from_millis(4));
        let led2 = ledger.leading(0, 2).unwrap();
        ledger.issued(0, 2, &led2, 10, t0);
        ledger.landed(0, 2, t0 + Duration::from_millis(2));

        ledger.pass(0, t0 + Duration::from_millis(5), |interest| interest.step() == 6);
        let chose = vec![vec![
            Chosen { expert: 1, saves: true, tier: true },
            Chosen { expert: 2, saves: true, tier: true },
            Chosen { expert: 3, saves: true, tier: false }, // demand miss
            Chosen { expert: 4, saves: false, tier: false }, // resident hit (not in cohort)
        ]];
        let learned = ledger.tell(Told::Completed { step: 6, chose, guesses: Some(0) });
        assert_eq!(learned.cohorts.len(), 1);
        let cohort = &learned.cohorts[0];
        assert_eq!(cohort.misses, 1);
        assert_eq!(cohort.saved.len(), 2);
        assert_eq!(cohort.size(), 3);
        assert_eq!(cohort.exclusive(1), 1, "only expert 2 is exclusively caught by lead 1");
        assert_eq!(cohort.exclusive(2), 0, "expert 1 was also caught by lead 1");

        let mut rates = CohortRates::new(1, 0);
        assert_eq!(rates.exclusive(1, 0, Tokens::new(1.0)), Caught::Unmeasured);
        rates.count(cohort, &[true, true], 0, Tokens::new(1.0));
        assert_eq!(rates.cohort(0, 0), Some(3.0));
        assert_eq!(rates.exclusive(1, 0, Tokens::new(1.0)), Caught::Measured { rate: 1.0 / 3.0, age: Tokens::ZERO });
        assert_eq!(rates.exclusive(2, 0, Tokens::new(1.0)), Caught::Measured { rate: 0.0, age: Tokens::ZERO });

        // Lead 1 turns off: cohort keeps counting, lead 1 freezes with growing age.
        let miss_only = Cohort { step: 7, layer: 0, misses: 5, saved: Vec::new() };
        rates.count(&miss_only, &[false, true], 1, Tokens::new(6.0));
        assert_eq!(rates.cohort(0, 1), Some(4.0), "(3 + 5) / 2");
        assert_eq!(rates.exclusive(1, 1, Tokens::new(6.0)), Caught::Measured { rate: 1.0 / 3.0, age: Tokens::new(5.0) });
    }

    /// What a read is admitted at is the measured rate of the combination
    /// of names its admission was decided on for its step, not a
    /// composition of their precisions: the experts two sources named
    /// together, decided on as such, are used at their own rate.
    /// Unmeasured, the bound 1.
    #[test]
    fn a_combination_of_names_is_admitted_at_its_measured_rate() {
        let mut ledger = calibrated(1, &[(AHEAD, 0.5), (LEAD, 0.5)]);
        let t = now();
        let both = |ledger: &mut Ledger, expert, step| {
            ledger.publish(0, expert, name(AHEAD, step), t);
            ledger.publish(0, expert, name(LEAD, step), t);
            ledger.act(0, expert, Read::Absent, &PRICE);
        };
        both(&mut ledger, 1, 6);
        assert_eq!(ledger.derived(0, 1), Some(Derived { step: 6, p: 1. }), "a combination never counted");
        // Every expert both named at step 6 was chosen: three of four.
        for expert in 2..=4 {
            both(&mut ledger, expert, 6);
        }
        resolve(&mut ledger, 6, 0, &[1, 2, 3]);
        both(&mut ledger, 5, 7);
        assert_eq!(ledger.derived(0, 5), Some(Derived { step: 7, p: 3. / 4. }), "the pair's own rate, not 1 − ½·½");
        ledger.publish(0, 6, name(AHEAD, 7), t);
        assert_eq!(ledger.derived(0, 6).map(|derived| derived.p), Some(0.5), "a name alone at its own");
        ledger.publish(0, 6, name(Source::Exact, 7), t);
        assert_eq!(ledger.derived(0, 6), Some(Derived { step: 7, p: 1. }), "an exact request certain");
    }

    #[test]
    fn a_source_republishing_replaces_its_interest_and_keeps_its_order() {
        let mut ledger = ledger(1);
        let t = now();
        ledger.publish(0, 1, Name { rank: 3, ..name(AHEAD, 5) }, t);
        ledger.publish(0, 2, name(AHEAD, 5), t);
        ledger.publish(0, 1, name(AHEAD, 5), t);
        assert_eq!(ledger.layers[0][&1].interests.len(), 1);
        assert_eq!(ledger.leading(0, 1).unwrap().rank, 0);
        let scheduled = ledger.schedule(ledger.wanted(), |_, _| Read::Absent, |_| PRICE);
        assert_eq!(scheduled.iter().map(|scheduled| (scheduled.layer, scheduled.expert)).collect::<Vec<_>>(), [(0, 1), (0, 2)], "first named, first read");
    }

    /// A completed step resolves each name by whether its exact rows routed
    /// the expert. A read ahead issued for a routed expert is used.
    #[test]
    fn a_resolved_name_says_whether_its_expert_was_routed() {
        let mut ledger = calibrated(1, &[(AHEAD, 0.8)]);
        let t = now();
        for expert in 1..=3 {
            ledger.publish(0, expert, front(AHEAD, 6, 1), t);
        }
        let led = ledger.leading(0, 1).expect("a live interest");
        ledger.issued(0, 1, &led, 10, t);
        let resolved = resolve(&mut ledger, 6, 0, &[1, 2]);
        let fates = resolved.iter().map(|resolved| (resolved.expert, resolved.routed)).collect::<Vec<_>>();
        assert_eq!(fates, [(1, true), (2, true), (3, false)]);
        assert!(matches!(ledger.take_spent()[..], [Spent::Used { bytes: 10, .. }]), "the read ahead of a routed expert is used");
    }

    /// Two requests' first front rows are one source, each its own want:
    /// neither replaces the other, so their union prices the read and one's
    /// wrong guess leaves the other's want.
    #[test]
    fn two_requests_names_of_one_row_are_two_wants() {
        let mut ledger = ledger(1);
        let t = now();
        let of = |request| Name { lineage: Some(Lineage { request: RequestId(request), row: 1, guessed: NonZeroU8::MIN }), ..name(AHEAD, 6) };
        ledger.publish(0, 1, of(1), t);
        ledger.publish(0, 1, of(2), t);
        assert_eq!(ledger.layers[0][&1].interests.len(), 2);
        awaiting(&mut ledger, 5, &[2]);
        ledger.tell(Told::Decided { step: 5, request: RequestId(2), verdict: proposed(&[false]) });
        assert!(ledger.derived(0, 1).is_some(), "request 1's want stands");
        assert_eq!(ledger.layers[0][&1].interests[0].lineage.unwrap().request, RequestId(1));
    }

    #[test]
    fn issue_is_due_for_a_certain_interest_now_expected_when_admitted_and_kept_below_the_break_even() {
        let mut ledger = calibrated(1, &[(AHEAD, 0.8), (LEAD, 0.2)]);
        let t = now();
        ledger.publish(0, 1, name(Source::Exact, 5), t);
        ledger.publish(0, 2, name(AHEAD, 6), t);
        ledger.publish(0, 3, name(LEAD, 6), t);
        assert_eq!(ledger.act(0, 1, Read::Absent, &Unpriced), Act::Issue(Class::Due), "a due read is never priced");
        assert!(matches!(ledger.act(0, 2, Read::Absent, &PRICE), Act::Issue(Class::Expected { step: 6, .. })));
        assert_eq!(ledger.act(0, 3, Read::Absent, &PRICE), Act::Keep, "a hint below the break-even claims no staging or tier slot");
        // An early Lead hint below the break-even never shadows an admitted later-step Ahead route.
        ledger.publish(0, 4, name(LEAD, 5), t);
        ledger.publish(0, 4, name(AHEAD, 6), t);
        assert!(matches!(ledger.act(0, 4, Read::Absent, &PRICE), Act::Issue(Class::Expected { step: 6, .. })));
        // A certain interest for a step not yet reached is not yet due.
        ledger.publish(0, 5, name(Source::Exact, 6), t);
        assert_eq!(ledger.act(0, 5, Read::Absent, &PRICE), Act::Issue(Class::Expected { step: 6, p: 1. }));
        ledger.reach(6, 0);
        assert_eq!(ledger.act(0, 5, Read::Absent, &PRICE), Act::Issue(Class::Due));
    }

    #[test]
    fn a_held_or_landing_expert_needs_no_read() {
        let mut ledger = ledger(1);
        let t = now();
        ledger.reach(5, 0);
        ledger.publish(0, 1, name(Source::Exact, 5), t);
        assert_eq!(ledger.act(0, 1, Read::Landing, &PRICE), Act::Keep);
        assert_eq!(ledger.act(0, 1, Read::Held, &PRICE), Act::Keep);
    }

    /// An exact request of an expert a read ahead is reading orders that
    /// read due: the request's wait is the ledger's, not a path of its own.
    #[test]
    fn an_exact_request_reprices_the_read_ahead_it_finds_due() {
        let mut ledger = calibrated(1, &[(LEAD, 0.2)]);
        let t = now();
        ledger.publish(0, 1, name(LEAD, 6), t);
        assert_eq!(ledger.act(0, 1, Read::Reading { begun: false }, &PRICE), Act::Cancel);
        assert_eq!(ledger.act(0, 1, Read::Reading { begun: true }, &PRICE), Act::Keep);
        ledger.reach(6, 0);
        ledger.publish(0, 1, name(Source::Exact, 6), t);
        assert_eq!(ledger.act(0, 1, Read::Reading { begun: true }, &Unpriced), Act::Reprice(Class::Due));
    }

    #[test]
    fn a_passed_read_cancels_only_while_no_piece_has_begun() {
        let mut ledger = ledger(1);
        let t = now();
        ledger.publish(0, 1, name(AHEAD, 6), t);
        ledger.publish(0, 2, name(AHEAD, 6), t);
        assert_eq!(ledger.pass(0, t, |interest| interest.step() <= 6), vec![1, 2]);
        assert_eq!(ledger.act(0, 1, Read::Reading { begun: false }, &PRICE), Act::Cancel);
        assert_eq!(ledger.act(0, 2, Read::Reading { begun: true }, &PRICE), Act::Keep);
    }

    /// A completed step resolves every bound interest of its step at each
    /// layer, passed or not, used or not; each decided combination counts
    /// its outcome, and later steps' interests stand.
    #[test]
    fn resolve_counts_every_decided_combination() {
        let mut ledger = ledger(2);
        let t = now();
        ledger.publish(0, 1, name(AHEAD, 6), t);
        ledger.publish(0, 2, name(LEAD, 6), t);
        ledger.publish(0, 3, Name { candidate: false, ..name(LEAD, 6) }, t);
        ledger.publish(0, 1, name(AHEAD, 7), t);
        ledger.publish(1, 4, name(AHEAD, 6), t);
        ledger.publish(0, 1, name(Source::Exact, 6), t);
        // The candidates' reads decided before the hints pass.
        for expert in [1, 2] {
            ledger.act(0, expert, Read::Absent, &PRICE);
        }
        ledger.pass(0, t, |interest| interest.source == LEAD);
        let mut resolved = resolve(&mut ledger, 6, 0, &[1]);
        resolved.sort_by_key(|resolved| resolved.expert);
        assert_eq!(resolved.iter().map(|resolved| (resolved.expert, resolved.used)).collect::<Vec<_>>(), [(1, true), (2, false), (3, false)]);
        // Each decided read's names for the step counted together: expert
        // 1's exact request is certain, no combination decided; expert 2's
        // hint, decided; expert 3's, never decided.
        let ahead_key = Combination::of([Kind { source: AHEAD, rank: 0 }]).unwrap();
        let lead_key = Combination::of([Kind { source: LEAD, rank: 0 }]).unwrap();
        assert_eq!((ledger.calibration.combined(0, ahead_key, 0), ledger.calibration.combined(0, lead_key, 0)), (1., 0.));
        assert_eq!(ledger.derived(0, 1).map(|derived| derived.step), Some(7));
        assert_eq!(ledger.derived(0, 2), None);
        assert_eq!(ledger.derived(1, 4).map(|derived| derived.step), Some(6), "another layer stands");
        assert!(ledger.take_owed().contains(&(0, 2)), "a resolved read is owed a derivation");
    }

    #[test]
    fn expire_ends_what_no_completion_resolved() {
        let mut ledger = ledger(1);
        let t = now();
        ledger.publish(0, 1, name(AHEAD, 6), t);
        ledger.publish(0, 1, name(AHEAD, 7), t);
        ledger.tell(Told::Ended { step: 6, completed: true });
        assert_eq!(ledger.derived(0, 1).map(|derived| derived.step), Some(7));
        ledger.tell(Told::Ended { step: 7, completed: true });
        assert_eq!(ledger.derived(0, 1), None);
    }

    /// Reads are scheduled cancellations first, then hastenings to due,
    /// then the reads short of due, repriced before issued, each the
    /// soonest wanted step first, the order the ranking's foresight keeps
    /// them by, then the likeliest, then the first named: a sooner step's
    /// read goes before a later one's however likelier that is. A front
    /// row's name is dated and ranked by the step whose exact row uses it,
    /// at its combination's measured rate, never by its own pass.
    #[test]
    fn reads_are_scheduled_soonest_step_first_then_likeliest() {
        const FAR: Source = Source::Lead { distance: 2, row: None };
        let mut ledger = calibrated(2, &[(AHEAD, 0.9), (LEAD, 0.6)]);
        let t = now();
        ledger.publish(0, 1, name(LEAD, 6), t);
        ledger.publish(1, 2, name(AHEAD, 8), t);
        ledger.publish(0, 3, name(LEAD, 7), t);
        ledger.publish(1, 4, name(LEAD, 6), t);
        ledger.publish(0, 5, name(Source::Exact, 5), t);
        ledger.publish(1, 6, front(AHEAD, 6, 1), t);
        ledger.publish(0, 7, name(FAR, 6), t);
        ledger.publish(0, 8, name(AHEAD, 9), t);
        assert_eq!(ledger.pass(0, t, |interest| interest.source == FAR), vec![7]);
        let reading = |layer, expert| match (layer, expert) {
            (0, 5) | (0, 7) | (1, 2) => Read::Reading { begun: false },
            _ => Read::Absent,
        };
        let mut experts = ledger.wanted();
        experts.push((0, 7));
        let scheduled = ledger.schedule(experts, reading, |_| Flat { even: 0.3 });
        let order = scheduled.iter().map(|scheduled| (scheduled.layer, scheduled.expert)).collect::<Vec<_>>();
        assert_eq!(order, [(0, 7), (0, 5), (1, 2), (1, 6), (0, 1), (1, 4), (0, 3), (0, 8)]);
        assert_eq!((scheduled[0].act, scheduled[1].act), (Act::Cancel, Act::Reprice(Class::Due)));
        assert!(matches!(scheduled[2].act, Act::Reprice(Class::Expected { step: 8, .. })));
        assert!(matches!(scheduled[7].act, Act::Issue(Class::Expected { step: 9, p }) if p > 0.85), "the likeliest, wanted last, is read last");
        let front = scheduled[3];
        assert!(matches!(front.act, Act::Issue(Class::Expected { step: 6, p }) if (p - 0.9).abs() < 1e-12), "{:?}", front.act);
        assert_eq!(front.leading.map(|interest| interest.source), Some(AHEAD), "the front row's name leads its read");
    }

    /// A front row's name, of request 1's guessed row `guessed`, for the
    /// step after its pass.
    fn front(source: Source, step: Step, guessed: u8) -> Name {
        let guessed = NonZeroU8::new(guessed).expect("a front row is among its pass's guessed rows");
        Name { lineage: Some(Lineage { request: RequestId(1), row: 1, guessed }), ..name(source, step) }
    }

    /// An acceptable row's name, of request `request`'s guessed row
    /// `guessed` of a verification of two, for the step after its pass.
    fn proposal(request: u64, guessed: u8) -> Name {
        let guessed = NonZeroU8::new(guessed).expect("a row ahead is among its pass's guessed rows");
        let row = Row { index: guessed.get() - 1, front: None };
        Name { lineage: Some(Lineage { request: RequestId(request), row: 1, guessed }), ..name(Source::Ahead(row), 6) }
    }

    /// A front pass's verdict: the tokens its front rows ran on, and the
    /// truth its exact row selected.
    fn fronted(front: &[u32], truth: u32) -> Verdict {
        Verdict { acceptable: Vec::new(), front: front.iter().map(|&token| TokenId(token)).collect(), truths: vec![TokenId(truth)] }
    }

    /// A verification's verdict, each proposal whether it stood.
    fn proposed(stood: &[bool]) -> Verdict {
        Verdict { acceptable: stood.to_vec(), front: Vec::new(), truths: Vec::new() }
    }

    /// `requests`' passes of step `step` await their verdicts: a grouped
    /// launch's (a verification's), whose rows name nothing ahead.
    fn awaiting(ledger: &mut Ledger, step: Step, requests: &[u64]) {
        let requests = requests.iter().map(|&request| RequestId(request)).collect();
        ledger.tell(Told::Began { step, launch: Launched::Grouped(requests) });
    }

    /// A verdict counts each guess up to the first wrong one into the
    /// stands; a wrong guess drops every name of its request's rows from
    /// it on, and only those.
    #[test]
    fn a_wrong_guess_drops_its_rows_names_and_counts_into_the_stands() {
        let mut ledger = ledger(1);
        let t = now();
        ledger.publish(0, 1, proposal(1, 1), t);
        ledger.publish(0, 2, proposal(1, 2), t);
        ledger.publish(0, 3, proposal(2, 1), t);
        ledger.publish(0, 4, name(LEAD, 6), t);
        awaiting(&mut ledger, 5, &[1]);
        ledger.tell(Told::Decided { step: 5, request: RequestId(1), verdict: proposed(&[true, false]) });
        assert_eq!(ledger.take_owed(), vec![(0, 2)]);
        assert!(ledger.take_owed().is_empty(), "owed experts are taken once");
        assert!(ledger.derived(0, 1).is_some(), "the guess that stood keeps its names");
        assert_eq!(ledger.derived(0, 2), None, "the wrong guess's names drop");
        assert!(ledger.derived(0, 3).is_some() && ledger.derived(0, 4).is_some(), "another request's and an exact row's stand");
        assert_eq!(ledger.act(0, 2, Read::Reading { begun: false }, &PRICE), Act::Cancel);
        // Depth 1 stood once, depth 2 fell once.
        assert_eq!(ledger.stands.through_rows(2, Form::Verify), 0.);
        assert_eq!((ledger.stands.rate(1, Form::Verify), ledger.stands.rate(2, Form::Verify), ledger.stands.rate(3, Form::Verify)), (Some(1.), Some(0.), None));
    }

    /// A launch of five rows runs in a class of eight: the rings name its
    /// three padding rows as exact rows of the step, and no row past them.
    #[test]
    fn a_padding_row_names_as_an_exact_row() {
        let mut ledger = ledger(1);
        let rows = [DecodeRow { position: 0, request: RequestId(1), ahead: None }; 5];
        ledger.tell(Told::Began { step: 4, launch: Launched::Decode { rows: Box::new(rows), padding: 3 } });
        assert!((5..8).all(|row| ledger.row(4, row, 0) == Some(Origin::Exact)));
        let past = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ledger.row(4, 8, 0)));
        assert!(past.is_err(), "a row past the class is the device's broken invariant");
    }

    /// A front row's name is for its request's next pass: a launch of other
    /// requests between (a prompt chunk's) binds nothing, its request's next
    /// launch binds it, whose completion resolves it.
    #[test]
    fn a_front_rows_name_binds_to_its_requests_next_pass() {
        let mut ledger = ledger(1);
        let t = now();
        let row = |request, ahead| DecodeRow { position: 0, request: RequestId(request), ahead };
        let front = AheadOf::new(1, 0, 0);
        ledger.tell(Told::Began { step: 4, launch: Launched::Decode { rows: Box::new([row(1, None), row(1, Some(front))]), padding: 0 } });
        let chained = kinds(1, GuessKind::Chained);
        let Some(Origin::Ahead { row: named, when, lineage }) = ledger.row(4, 1, chained) else { panic!("an ahead row") };
        assert_eq!((named, lineage.guessed.get()), (FRONT, 1));
        assert_eq!(when, When::Pass { request: RequestId(1), pass: 2, estimate: 5 });
        assert_eq!(ledger.row(4, 0, chained), Some(Origin::Exact));
        // Its ring's guess kind makes its names that kind's.
        let Some(Origin::Ahead { row: head, .. }) = ledger.row(4, 1, kinds(1, GuessKind::Head)) else { panic!("an ahead row") };
        assert!(head.front == Some(GuessKind::Head) && head != FRONT);
        assert_eq!(ledger.row(4, 1, 0), None, "a front row that ran on no guess names nothing");
        ledger.publish(0, 3, Name { when, lineage: Some(lineage), from: At::whole(4, 0), ..name(AHEAD, 0) }, t);
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(chained) });
        ledger.tell(Told::Ended { step: 4, completed: true });
        ledger.tell(Told::Began { step: 5, launch: Launched::Decode { rows: Box::new([row(2, None)]), padding: 0 } });
        assert!(resolve(&mut ledger, 5, 0, &[]).is_empty(), "another request's launch resolves nothing of it");
        ledger.tell(Told::Ended { step: 5, completed: true });
        assert_eq!(ledger.derived(0, 3).map(|derived| derived.step), Some(6), "dated past the launches that were not its pass");
        ledger.tell(Told::Began { step: 7, launch: Launched::Decode { rows: Box::new([row(1, None)]), padding: 0 } });
        assert_eq!(ledger.derived(0, 3).map(|derived| derived.step), Some(7), "bound to its request's next launch");
        let resolved = resolve(&mut ledger, 7, 0, &[3]);
        assert_eq!(resolved.iter().map(|resolved| (resolved.expert, resolved.used)).collect::<Vec<_>>(), [(3, true)]);
    }

    /// Each pass with rows ahead awaits its verdict by its step: a front
    /// row's guess is judged by the decision that learns its position's
    /// truth, its own pass's at depth 1, in whatever order the verdicts
    /// come, with its guess's kind as its routing records carried it; a
    /// pass of acceptable rows alone judges no front row, and a request's
    /// closing leaves a pass it never decided unjudged.
    #[test]
    fn a_verdict_judges_the_front_row_of_its_own_pass() {
        let mut ledger = ledger(1);
        let row = |request, position, ahead| DecodeRow { position, request: RequestId(request), ahead };
        let (front, acceptable) = (AheadOf::new(1, 0, 0), AheadOf::new(1, 1, 0));
        let decode = |rows: &[DecodeRow]| Launched::Decode { rows: rows.into(), padding: 0 };
        ledger.tell(Told::Began { step: 4, launch: decode(&[row(1, 10, None), row(1, 11, Some(front)), row(2, 3, None)]) });
        ledger.tell(Told::Began { step: 5, launch: decode(&[row(1, 11, None), row(1, 12, Some(front))]) });
        ledger.tell(Told::Began { step: 6, launch: decode(&[row(1, 12, None), row(1, 13, Some(acceptable))]) });
        for step in 4..=6 {
            // Step 5's front row runs on the head's guess, step 4's having
            // fallen; step 6 runs no front row.
            let guesses = match step {
                4 => kinds(1, GuessKind::Chained),
                5 => kinds(1, GuessKind::Head),
                _ => 0,
            };
            ledger.tell(Told::Completed { step, chose: vec![Vec::new()], guesses: Some(guesses) });
        }
        ledger.tell(Told::Decided { step: 5, request: RequestId(1), verdict: fronted(&[30], 30) });
        ledger.tell(Told::Decided { step: 4, request: RequestId(1), verdict: fronted(&[20], 21) });
        assert_eq!(
            ledger.take_judged().0,
            [
                Judged { step: 5, row: 1, depth: 1, kind: GuessKind::Head, stood: Some(true) },
                Judged { step: 4, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(false) }
            ]
        );
        assert_eq!(ledger.stands.rate(1, Form::Front), Some(0.5), "front depth 1 pools both verdicts (1 of 2 stood)");
        ledger.tell(Told::Decided { step: 6, request: RequestId(1), verdict: proposed(&[true]) });
        assert_eq!(
            ledger.take_judged(),
            (Vec::new(), vec![Accepted { depth: 1, stood: true }]),
            "a pass of acceptable rows alone judges no front row, and its own rows by depth"
        );
        assert_eq!(ledger.stands.rate(1, Form::Verify), Some(1.0));
        ledger.tell(Told::Began { step: 7, launch: decode(&[row(1, 13, None), row(1, 14, Some(front))]) });
        ledger.tell(Told::Completed { step: 7, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Chained)) });
        ledger.tell(Told::Began { step: 9, launch: decode(&[row(1, 14, None), row(1, 15, Some(front))]) });
        ledger.tell(Told::Completed { step: 9, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Head)) });
        ledger.tell(Told::Decided { step: 7, request: RequestId(1), verdict: fronted(&[40], 40) });
        assert_eq!(ledger.take_judged().0, [Judged { step: 7, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(true) }]);
        ledger.tell(Told::Closed(RequestId(1)));
        assert_eq!(ledger.take_judged().0, [Judged { step: 9, row: 1, depth: 1, kind: GuessKind::Head, stood: None }]);
    }

    /// A discarded pass (a queued pass orphaned) is judged by none, and the
    /// names its rows published drop; the verdicts after it judge their own
    /// passes, not the one before each. A discard of a closed request's
    /// pass finds nothing to end.
    #[test]
    fn a_discarded_pass_leaves_later_verdicts_on_their_own_passes() {
        let mut ledger = ledger(1);
        let t = now();
        let row = |position, ahead| DecodeRow { position, request: RequestId(1), ahead };
        let front = AheadOf::new(1, 0, 0);
        for step in 4..=6 {
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows: Box::new([row(step + 6, None), row(step + 7, Some(front))]), padding: 0 } });
            let kind = if step == 6 { GuessKind::Head } else { GuessKind::Chained };
            ledger.tell(Told::Completed { step, chose: vec![Vec::new()], guesses: Some(kinds(1, kind)) });
        }
        // Step 5's front row names expert 2 for the pass after it.
        let lineage = Some(Lineage { request: RequestId(1), row: 1, guessed: NonZeroU8::MIN });
        ledger.publish(0, 2, Name { when: When::Step(6), lineage, from: At::whole(5, 0), ..name(AHEAD, 6) }, t);
        ledger.publish(0, 3, Name { when: When::Step(7), lineage, from: At::whole(6, 0), ..name(AHEAD, 7) }, t);
        ledger.tell(Told::Decided { step: 4, request: RequestId(1), verdict: fronted(&[11], 11) });
        ledger.tell(Told::Discarded { step: 5, request: RequestId(1) });
        assert_eq!(ledger.derived(0, 2), None, "the discarded pass's names drop");
        assert!(ledger.derived(0, 3).is_some(), "the next pass's names stand");
        ledger.tell(Told::Decided { step: 6, request: RequestId(1), verdict: fronted(&[13], 14) });
        assert_eq!(
            ledger.take_judged().0,
            [
                Judged { step: 4, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(true) },
                Judged { step: 5, row: 1, depth: 1, kind: GuessKind::Chained, stood: None },
                Judged { step: 6, row: 1, depth: 1, kind: GuessKind::Head, stood: Some(false) },
            ]
        );
        ledger.tell(Told::Closed(RequestId(1)));
        ledger.tell(Told::Discarded { step: 7, request: RequestId(1) });
        assert!(ledger.take_judged().0.is_empty());
    }

    /// A verdict of a pass that awaits none is a broken invariant: no
    /// verdict is ever matched to another pass.
    #[test]
    #[should_panic(expected = "awaits no verdict")]
    fn a_verdict_of_a_pass_that_awaits_none_panics() {
        let mut ledger = ledger(1);
        awaiting(&mut ledger, 4, &[1]);
        ledger.tell(Told::Decided { step: 5, request: RequestId(1), verdict: proposed(&[true]) });
    }

    /// A front row that ran on no guess (its step's guess kinds give it
    /// none: a queued pass whose predecessor made it none) names nothing:
    /// its verdict counts into no stand rate and judges nothing, but a
    /// guess of it that fell drops its pass's names all the same.
    #[test]
    fn a_front_row_on_no_guess_counts_into_no_stand() {
        let mut ledger = ledger(1);
        let t = now();
        let row = |position, ahead| DecodeRow { position, request: RequestId(1), ahead };
        ledger.tell(Told::Began { step: 4, launch: Launched::Decode { rows: Box::new([row(10, None), row(11, Some(AheadOf::new(1, 0, 0)))]), padding: 0 } });
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(0) });
        let lineage = Lineage { request: RequestId(1), row: 1, guessed: NonZeroU8::MIN };
        ledger.publish(0, 2, Name { when: When::Step(5), lineage: Some(lineage), from: At::whole(4, 0), ..name(AHEAD, 5) }, t);
        ledger.tell(Told::Decided { step: 4, request: RequestId(1), verdict: fronted(&[11], 12) });
        assert!(ledger.take_judged().0.is_empty(), "no guess's kind to judge");
        assert_eq!(ledger.stands.through_rows(1, Form::Front), 1., "no stand counted: its bound");
        assert_eq!(ledger.derived(0, 2), None, "its fallen guess's names drop");
    }

    /// A request's closing drops every name its rows published; a launch
    /// that ends uncompleted drops the names its ahead rows published.
    #[test]
    fn a_closed_request_or_a_void_pass_leaves_no_name() {
        let mut ledger = ledger(1);
        let t = now();
        let lineage = Some(Lineage { request: RequestId(1), row: 1, guessed: NonZeroU8::MIN });
        let when = When::Pass { request: RequestId(1), pass: 2, estimate: 5 };
        ledger.publish(0, 1, Name { when, lineage, from: At::whole(4, 0), ..name(AHEAD, 0) }, t);
        ledger.tell(Told::Closed(RequestId(1)));
        assert_eq!(ledger.derived(0, 1), None);
        ledger.tell(Told::Began { step: 8, launch: Launched::Decode { rows: Box::new([DecodeRow { position: 0, request: RequestId(3), ahead: None }]), padding: 0 } });
        let lineage = Some(Lineage { request: RequestId(3), row: 1, guessed: NonZeroU8::MIN });
        ledger.publish(0, 2, Name { when: When::Pass { request: RequestId(3), pass: 2, estimate: 9 }, lineage, from: At::whole(8, 0), ..name(AHEAD, 0) }, t);
        ledger.tell(Told::Ended { step: 8, completed: false });
        assert_eq!(ledger.derived(0, 2), None, "a pass that never completed guessed nothing");
    }

    /// An exact request passes once its route is served, as every source
    /// does: its certainty then no longer hides a later step's want of the
    /// expert, whose read the ledger issues as that want's.
    #[test]
    fn a_passed_exact_request_leaves_a_later_steps_want() {
        let mut ledger = calibrated(1, &[(AHEAD, 0.8)]);
        let t = now();
        ledger.publish(0, 1, name(Source::Exact, 5), t);
        ledger.publish(0, 1, name(AHEAD, 6), t);
        assert_eq!(ledger.act(0, 1, Read::Absent, &Unpriced), Act::Issue(Class::Due), "the request's own");
        assert_eq!(ledger.pass(0, t, |interest| interest.step() <= 5), vec![1]);
        assert!(matches!(ledger.act(0, 1, Read::Absent, &PRICE), Act::Issue(Class::Expected { step: 6, .. })));
        assert_eq!(ledger.derived(0, 1).map(|derived| derived.step), Some(6));
    }

    /// Every read ahead's bytes are spent once, by the path that ends the
    /// interests: a wrong guess's verdict fells the reads its names led
    /// (and only those: a read another request's name led stands), a later
    /// exact row uses one by choosing it at completion or by requesting it
    /// in flight, a read cancelled before it began read nothing, and one
    /// requested again with no read in flight was evicted unused.
    #[test]
    fn a_read_aheads_bytes_are_used_cancelled_or_lost_and_fall_with_its_guess() {
        let mut ledger = calibrated(1, &[(AHEAD, 0.8), (LEAD, 0.8)]);
        let t = now();
        let of = |request| Name { lineage: Some(Lineage { request: RequestId(request), row: 1, guessed: NonZeroU8::MIN }), ..name(AHEAD, 6) };
        ledger.publish(0, 1, of(1), t);
        ledger.publish(0, 2, of(1), t);
        ledger.publish(0, 3, name(LEAD, 6), t);
        ledger.publish(0, 4, of(2), t);
        ledger.publish(0, 5, name(LEAD, 6), t);
        ledger.publish(0, 6, of(1), t);
        for (expert, bytes) in [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)] {
            let led = ledger.leading(0, expert).expect("a live interest");
            ledger.issued(0, expert, &led, bytes, t);
        }
        // Another request's name for expert 6 does not hold its read up.
        ledger.publish(0, 6, of(2), t);
        awaiting(&mut ledger, 5, &[1, 2]);
        ledger.tell(Told::Decided { step: 5, request: RequestId(1), verdict: proposed(&[false]) });
        ledger.tell(Told::Decided { step: 5, request: RequestId(2), verdict: proposed(&[true]) });
        let fell = |basis, bytes| Spent::Fell { basis, bytes };
        assert_eq!(ledger.take_spent(), [fell(Basis::Front, 10), fell(Basis::Front, 20), fell(Basis::Front, 60)]);
        ledger.cancelled(0, 2);
        ledger.requested(0, 3, true);
        ledger.requested(0, 5, false);
        let spent = ledger.take_spent();
        assert_eq!(
            spent,
            [
                Spent::Cancelled { basis: Basis::Front, depth: 1, verify: false, fell: true, bytes: 20 },
                Spent::Used { basis: Basis::Hint, depth: 0, verify: false, fell: false, bytes: 30 },
                Spent::Unused { basis: Basis::Hint, depth: 0, verify: false, fell: false, bytes: 50 },
            ]
        );
        // A fallen guess's read is still used where an exact row chooses it;
        // expert 6 (bound to step 6, not chosen) settles unused here.
        resolve(&mut ledger, 6, 0, &[1, 3, 4, 5]);
        let spent = ledger.take_spent();
        assert_eq!(
            spent,
            [
                Spent::Used { basis: Basis::Front, depth: 1, verify: false, fell: true, bytes: 10 },
                Spent::Used { basis: Basis::Front, depth: 1, verify: false, fell: false, bytes: 40 },
                Spent::Unused { basis: Basis::Front, depth: 1, verify: false, fell: true, bytes: 60 },
            ]
        );
        // Spent once: a later choice of a spent read's expert spends nothing.
        resolve(&mut ledger, 7, 0, &[1, 4]);
        assert!(ledger.take_spent().is_empty());
        assert_eq!(Basis::of(Source::Lead { distance: 1, row: Some(FRONT) }), Basis::AheadHint);
    }

    /// A launch of an exact row and three front rows: front row `d` (batch
    /// row `d`) names for its request's pass `d` passes on, its lineage its
    /// `d` guessed rows.
    fn three_deep(ledger: &mut Ledger, step: Step, exact: u64) {
        let rows = (0..4).map(|row| DecodeRow { position: exact + row as u64, request: RequestId(1), ahead: (row > 0).then(|| AheadOf::new(1, 0, row - 1)) });
        ledger.tell(Told::Began { step, launch: Launched::Decode { rows: rows.collect(), padding: 0 } });
    }

    /// A pass claimed deeper than the chain its predecessor's head launch
    /// made runs idle front rows past its guessed ones, of no kind: an idle
    /// row names nothing, and its pass's verdict, which holds its guessed
    /// rows' tokens alone, judges them alone, an idle row never.
    #[test]
    fn an_idle_front_row_names_nothing_and_is_never_judged() {
        let mut ledger = ledger(1);
        three_deep(&mut ledger, 4, 10);
        let guesses = magnitude_kernels::step_guesses([(1, GuessKind::Chained.word()), (2, GuessKind::Head.word())]) as u32;
        assert_eq!(ledger.row(4, 3, guesses), None, "an idle row names nothing");
        assert!(ledger.row(4, 2, guesses).is_some(), "a guessed row names its experts");
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(guesses) });
        ledger.tell(Told::Decided { step: 4, request: RequestId(1), verdict: fronted(&[11, 12], 11) });
        assert_eq!(ledger.take_judged().0, [Judged { step: 4, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(true) }]);
        ledger.tell(Told::Closed(RequestId(1)));
        assert_eq!(ledger.take_judged().0, [Judged { step: 4, row: 2, depth: 2, kind: GuessKind::Head, stood: None }], "its idle row is never judged");
    }

    /// The step guess word of three front rows on two chained guesses and
    /// the head's one past them.
    fn three_kinds() -> u32 {
        magnitude_kernels::step_guesses([(1, GuessKind::Chained.word()), (2, GuessKind::Chained.word()), (3, GuessKind::Head.word())]) as u32
    }

    /// Each front row of a pass names for the pass its token is exact in:
    /// depth `d`, `d` passes on.
    #[test]
    fn a_front_row_of_depth_d_names_for_its_requests_pass_d_on() {
        let mut ledger = ledger(1);
        three_deep(&mut ledger, 4, 10);
        for depth in 1..=3u8 {
            let Some(Origin::Ahead { row, when, lineage }) = ledger.row(4, usize::from(depth), three_kinds()) else { panic!("an ahead row") };
            assert_eq!(row.index, depth - 1);
            assert_eq!(when, When::Pass { request: RequestId(1), pass: 1 + u64::from(depth), estimate: 4 + u64::from(depth) });
            assert_eq!(lineage.guessed.get(), depth);
        }
    }

    /// A front ring of a request's later pass supersedes the names its
    /// earlier passes' front rows published at that layer, and only those:
    /// another layer's, another request's and an acceptable row's stay.
    #[test]
    fn a_later_passs_front_ring_supersedes_the_earlier_passes_names_at_its_layer() {
        let mut ledger = ledger(2);
        let t = now();
        let at = |from: Step, guessed| Name { from: At::whole(from, 0), ..front(AHEAD, from + u64::from(guessed), guessed) };
        ledger.publish(0, 1, at(4, 1), t);
        ledger.publish(0, 2, at(4, 2), t);
        ledger.publish(1, 1, at(4, 1), t);
        ledger.publish(0, 3, Name { lineage: Some(Lineage { request: RequestId(2), ..at(4, 1).lineage.unwrap() }), ..at(4, 1) }, t);
        ledger.publish(0, 4, proposal(1, 1), t);
        ledger.publish(0, 5, at(5, 1), t);
        ledger.take_owed();
        ledger.supersede(0, RequestId(1), 5);
        assert_eq!(ledger.take_owed(), vec![(0, 1), (0, 2)]);
        assert!(ledger.derived(0, 1).is_none() && ledger.derived(0, 2).is_none(), "the earlier pass's names at the layer give way");
        assert!(ledger.derived(1, 1).is_some(), "another layer's stay until its ring");
        assert!(ledger.derived(0, 3).is_some() && ledger.derived(0, 4).is_some(), "another request's and an acceptable row's stay");
        assert!(ledger.derived(0, 5).is_some(), "the pass's own names stay");
        assert_eq!(ledger.act(0, 2, Read::Reading { begun: false }, &PRICE), Act::Cancel, "an unbegun read nothing wants cancels");
    }

    /// A wrong guess drops only its own pass's names: the queued pass after
    /// it runs on guesses made from verified tokens alone, whose names
    /// stand.
    #[test]
    fn a_fall_drops_only_its_own_passs_names() {
        let mut ledger = calibrated(1, &[(AHEAD, 0.8)]);
        let t = now();
        let row = |position, ahead| DecodeRow { position, request: RequestId(1), ahead };
        for step in [4, 5] {
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows: Box::new([row(step + 6, None), row(step + 7, Some(AheadOf::new(1, 0, 0)))]), padding: 0 } });
        }
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Chained)) });
        let lineage = Some(Lineage { request: RequestId(1), row: 1, guessed: NonZeroU8::MIN });
        ledger.publish(0, 1, Name { when: When::Step(5), lineage, from: At::whole(4, 0), ..name(AHEAD, 5) }, t);
        ledger.publish(0, 2, Name { when: When::Step(6), lineage, from: At::whole(5, 0), ..name(AHEAD, 6) }, t);
        ledger.tell(Told::Decided { step: 4, request: RequestId(1), verdict: fronted(&[11], 12) });
        assert_eq!(ledger.derived(0, 1), None, "the fallen pass's names drop");
        assert!(ledger.derived(0, 2).is_some(), "the next pass's recovered names stand");
    }

    /// A pass's front rows are judged as the decisions learn their
    /// positions' truths: depth 1 by its own, depth `d` by the decision
    /// `d − 1` passes on, each counted only behind shallower rows that
    /// stood; a fall leaves the deeper rows unjudged and drops the names
    /// of its row on.
    #[test]
    fn deeper_front_rows_are_judged_by_later_decisions() {
        let mut ledger = ledger(1);
        let t = now();
        three_deep(&mut ledger, 4, 10);
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(three_kinds()) });
        let lineage = |guessed| Some(Lineage { request: RequestId(1), row: guessed, guessed: NonZeroU8::new(guessed as u8).unwrap() });
        for depth in 1..=3 {
            ledger.publish(0, depth, Name { when: When::Step(4 + depth as u64), lineage: lineage(depth), from: At::whole(4, 0), ..name(Source::Ahead(Row { index: depth as u8 - 1, front: Some(GuessKind::Chained) }), 5) }, t);
        }
        // Pass 4 ran on tokens 11, 12, 13 at positions 11, 12, 13.
        ledger.tell(Told::Decided { step: 4, request: RequestId(1), verdict: fronted(&[11, 12, 13], 11) });
        assert_eq!(ledger.take_judged().0, [Judged { step: 4, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(true) }]);
        assert!((1..=3).all(|expert| ledger.derived(0, expert).is_some()), "the rows that stood, and those unjudged, keep theirs");
        // The passes after it ran one row ahead each; each completed step
        // resolves the names for it.
        for (step, exact) in [(5, 11), (6, 12)] {
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows: Box::new([DecodeRow { position: exact, request: RequestId(1), ahead: None }, DecodeRow { position: exact + 1, request: RequestId(1), ahead: Some(AheadOf::new(1, 0, 0)) }]), padding: 0 } });
            ledger.tell(Told::Completed { step, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Chained)) });
        }
        ledger.tell(Told::Decided { step: 5, request: RequestId(1), verdict: fronted(&[12], 12) });
        assert_eq!(
            ledger.take_judged().0,
            [
                Judged { step: 4, row: 2, depth: 2, kind: GuessKind::Chained, stood: Some(true) },
                Judged { step: 5, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(true) },
            ]
        );
        assert!(ledger.derived(0, 3).is_some(), "depth 3 awaits its truth");
        ledger.tell(Told::Decided { step: 6, request: RequestId(1), verdict: fronted(&[14], 14) });
        assert_eq!(
            ledger.take_judged().0,
            [
                Judged { step: 4, row: 3, depth: 3, kind: GuessKind::Head, stood: Some(false) },
                Judged { step: 6, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(true) },
            ]
        );
        assert_eq!(ledger.derived(0, 3), None, "the fallen row's names drop");
        // Depth 2 stood once, depth 3 fell once.
        assert_eq!((ledger.stands.rate(2, Form::Front), ledger.stands.rate(3, Form::Front)), (Some(1.), Some(0.)));
        assert!(ledger.awaiting[&RequestId(1)].is_empty(), "every row judged, every pass left");
        // A row whose truth no decision learned (a pass between ran no row
        // ahead) is left unjudged as the truths pass it.
        three_deep(&mut ledger, 7, 13);
        ledger.tell(Told::Completed { step: 7, chose: vec![Vec::new()], guesses: Some(three_kinds()) });
        ledger.tell(Told::Decided { step: 7, request: RequestId(1), verdict: fronted(&[14, 15, 16], 14) });
        ledger.take_judged();
        three_deep(&mut ledger, 9, 15);
        ledger.tell(Told::Completed { step: 9, chose: vec![Vec::new()], guesses: Some(three_kinds()) });
        ledger.tell(Told::Decided { step: 9, request: RequestId(1), verdict: fronted(&[16, 17, 18], 16) });
        assert_eq!(
            ledger.take_judged().0,
            [
                Judged { step: 7, row: 2, depth: 2, kind: GuessKind::Chained, stood: None },
                Judged { step: 7, row: 3, depth: 3, kind: GuessKind::Head, stood: None },
                Judged { step: 9, row: 1, depth: 1, kind: GuessKind::Chained, stood: Some(true) },
            ]
        );
    }

    /// Demand strictly first: the reads a fallen guess's names alone
    /// wanted, at every depth, leave the queue on its verdict (an unbegun
    /// read cancels; a begun one runs out, never preempted), and a demand
    /// read is never ordered behind a read ahead, before or after.
    #[test]
    fn a_fallen_guesss_queued_reads_leave_and_demand_is_never_behind_them() {
        let mut ledger = calibrated(1, &[(Source::Ahead(Row { index: 0, front: Some(GuessKind::Chained) }), 0.9), (Source::Ahead(Row { index: 1, front: Some(GuessKind::Chained) }), 0.9)]);
        let t = now();
        three_deep(&mut ledger, 4, 10);
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(three_kinds()) });
        for depth in 1..=2u8 {
            let Some(Origin::Ahead { row, when, lineage }) = ledger.row(4, usize::from(depth), three_kinds()) else { panic!("an ahead row") };
            ledger.publish(0, usize::from(depth), Name { source: Source::Ahead(row), when, lineage: Some(lineage), from: At::whole(4, 0), ..name(AHEAD, 0) }, t);
        }
        ledger.publish(0, 3, name(Source::Exact, 5), t);
        let queued = |ledger: &mut Ledger| {
            let read = |_, _| Read::Reading { begun: false };
            ledger.schedule([(0, 1), (0, 2), (0, 3)], read, |_| Flat { even: 0. }).into_iter().map(|scheduled| (scheduled.expert, scheduled.act)).collect::<Vec<_>>()
        };
        let before = queued(&mut ledger);
        assert_eq!(before[0], (3, Act::Reprice(Class::Due)), "demand first: {before:?}");
        assert!(before[1..].iter().all(|(_, act)| matches!(act, Act::Reprice(Class::Expected { .. }))), "{before:?}");
        ledger.tell(Told::Decided { step: 4, request: RequestId(1), verdict: fronted(&[11, 12, 13], 99) });
        let mut owed = ledger.take_owed();
        owed.sort_unstable();
        assert_eq!(owed, vec![(0, 1), (0, 2)], "the fall owes every depth's reads a derivation");
        let after = queued(&mut ledger);
        assert_eq!(after, [(1, Act::Cancel), (2, Act::Cancel), (3, Act::Reprice(Class::Due))], "the fallen reads leave, demand stays first among reads");
        assert_eq!(ledger.act(0, 1, Read::Reading { begun: true }, &PRICE), Act::Keep, "a begun read is never preempted");
    }

    /// Re-issuing a hint read for `(layer, expert)` on the same step after an
    /// earlier hint read already landed (e.g., evicted and re-issued before
    /// `Completed` runs) must not panic in `Entry::saved` and preserves the
    /// earlier landing's catcher credit.
    #[test]
    fn reissuing_an_already_landed_hint_read_before_completion_does_not_panic() {
        let mut ledger = calibrated(1, &[(LEAD, 0.8)]);
        let t0 = now();
        let t1 = t0 + Duration::from_millis(1);
        let t2 = t0 + Duration::from_millis(2);
        ledger.publish(0, 7, name(LEAD, 6), t0);
        let led = ledger.leading(0, 7).expect("live hint interest");
        ledger.issued(0, 7, &led, 100, t0);
        ledger.landed(0, 7, t1);
        ledger.pass(0, t2, |interest| interest.step() == 6);
        // Expert 7 is evicted and re-issued for the same step (landed = None in `reads`).
        ledger.issued(0, 7, &led, 100, t2);
        let learned = ledger.tell(Told::Completed {
            step: 6,
            chose: vec![vec![Chosen { expert: 7, saves: true, tier: true }]],
            guesses: None,
        });
        assert_eq!(learned.cohorts.len(), 1);
        assert_eq!(learned.cohorts[0].saved.len(), 1, "preserved landing duration credits lead 1");
        assert_eq!(learned.cohorts[0].exclusive(1), 1);
    }

    /// B-1: `Ledger::hint_admitted` checks whether any rank `0..K` for
    /// `(target_layer, lead)` has calibrated probability `>= break_even`.
    #[test]
    fn hint_admitted_reflects_calibrated_ranks_against_break_even() {
        let time = Time::new(&[vec![64]], 8);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0], 2, time, stands);
        // Unmeasured ranks read their bound 1.0 -> admitted even at break_even = 0.85.
        assert!(ledger.hint_admitted(0, 1, 0.85));

        // Calibrate both ranks 0 and 1 at lead 1 to p = 0.25 (1 hit in 4).
        let t0 = now();
        for step in 1..=4u64 {
            for rank in 0..2u8 {
                let mut n = name(LEAD, step);
                n.rank = rank;
                ledger.publish(0, rank as usize, n, t0);
                ledger.act(0, rank as usize, Read::Absent, &PRICE);
            }
            let chose = if step == 1 {
                vec![vec![Chosen { expert: 0, saves: true, tier: false }, Chosen { expert: 1, saves: true, tier: false }]]
            } else {
                vec![Vec::new()]
            };
            ledger.tell(Told::Completed { step, chose, guesses: None });
        }
        assert!(!ledger.hint_admitted(0, 1, 0.80), "calibrated p = 0.25 < 0.80 must not be admitted");
        assert!(ledger.hint_admitted(0, 1, 0.25), "calibrated p = 0.25 >= 0.25 is admitted");
        assert!(ledger.hint_admitted(0, 1, 0.0), "p* = 0.0 admits all calibrated ranks");
    }
}
