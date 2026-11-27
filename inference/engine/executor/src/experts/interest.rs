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

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::num::NonZeroU8;
use std::ops::AddAssign;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use super::horizon::{self, RegimeAged, RegimeEpoch, Time, Tokens, Window};
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

/// Who wants an expert. Sources are the pass's rows and token routing
/// history, never its kind: an exact row's request, an ahead row's route, or
/// token-conditioned routing foresight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) enum Source {
    /// An exact row's own request (or a step that waits on a read in
    /// flight): certain, and wanted now.
    Exact,
    /// An ahead row's route, which the exact row of the pass its token is
    /// exact in routes again where every guess before it stands.
    Ahead(Row),
    /// Token-conditioned cross-burst routing foresight derived from the
    /// upcoming burst's exact/drafted tokens (`Told::DraftTokens` / `Told::Began`)
    /// via live per-token and bigram routing history (`TokenRoutes`).
    Token,
    /// Cross-pass intra-token foresight: an earlier visit of the same physical
    /// expert stack within the same step routed this expert, predicting reuse
    /// at a subsequent visit of that physical stack in the same step.
    CrossPass,
}

impl Source {
    /// The ahead row it is of, if any.
    pub fn row(self) -> Option<Row> {
        match self {
            Self::Exact | Self::Token | Self::CrossPass => None,
            Self::Ahead(row) => Some(row),
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
/// interests' outcomes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::experts) struct Learned {
    pub resolved: Vec<Resolved>,
}

/// The kind of name a read ahead was issued for: the source of the
/// interest that led its expert's aggregate ([`Ledger::issued`]). A front
/// row's route (`Source::Ahead`) or token-conditioned routing foresight
/// (`Source::Token`); an exact request issues no read ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum Basis {
    Front,
    Token,
}

impl Basis {
    /// Every basis, in [`Self::index`] order.
    pub const ALL: [Basis; 2] = [Basis::Front, Basis::Token];

    pub fn of(source: Source) -> Self {
        match source {
            Source::Ahead(_) => Self::Front,
            Source::Token | Source::CrossPass => Self::Token,
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
    acceptable: usize,
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
    /// carry it; none where every streamed layer was whole and wrote none),
    /// per-layer `(row, expert)` pairs (`row_experts`) for live token routing
    /// learning (`TokenRoutes`), actual physical `physical_rows` dispatched
    /// (including front rows), and per-layer post-site `(gpu_host_misses,
    /// cpu_sited)` counts (`layer_misses`).
    Completed {
        step: Step,
        chose: Vec<Vec<Chosen>>,
        guesses: Option<u32>,
        row_experts: Vec<Vec<(u8, u16)>>,
        physical_rows: u64,
        layer_misses: Vec<(u64, u64)>,
    },
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
    /// Exact/drafted tokens for `request`'s upcoming burst (anchor `y_0` on
    /// head launch, then drafted proposals `[y_1..y_k]` on head completion):
    /// publishes token-conditioned `Source::Token` interests for the next pass.
    DraftTokens { request: RequestId, tokens: Vec<TokenId> },
    /// Per-row prompt routes exported from a completed grouped prefill chain
    /// (`SplitValues::crossing.routes`), seeding `TokenRoutes` unigrams and
    /// bigrams before decode step 1.
    PromptRoutes {
        tokens: Box<[TokenId]>,
        layer_routes: Vec<Vec<u16>>,
        selected: usize,
    },
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

/// What a look saw (§3.5): its row's route, token-conditioned routing
/// history, or an earlier cross-pass visit of the same physical stack in the
/// same step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::experts) enum Seen {
    Route,
    Token,
    CrossPass,
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

/// Distinct `Seen` values: `Route`, `Token`, and `CrossPass`.
const SEEN_KINDS: usize = 3;

/// The looks' domain: 3 × DEPTHS × 2.
pub(in crate::experts) const LOOKS: usize = SEEN_KINDS * super::DEPTHS * 2;

impl Look {
    /// The look `source` names by.
    pub fn of(source: Source) -> Self {
        let seen = match source {
            Source::Ahead(_) | Source::Exact => Seen::Route,
            Source::Token => Seen::Token,
            Source::CrossPass => Seen::CrossPass,
        };
        Self { depth: source.depth(), seen, form: Form::of(source.verifies()) }
    }

    /// Its index in 0..LOOKS. Panics where its depth is outside the domain:
    /// no row names past it.
    fn index(self) -> usize {
        let depth = usize::from(self.depth);
        assert!(depth < super::DEPTHS, "a look of depth {depth} past the decode form's {} rows", super::DEPTHS);
        let seen = match self.seen {
            Seen::Route => 0,
            Seen::Token => 1,
            Seen::CrossPass => 2,
        };
        (depth * SEEN_KINDS + seen) * 2 + self.form as usize
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
/// regime-scoped aged window on the bank-wide epochs, which holds its last
/// measurement, with its age, while its row is not run within the same
/// physical [`RegimeEpoch`]. Any physical tier/residency regime transition
/// makes prior-regime judgements inapplicable by construction, while within a
/// regime [`Stands::live_rate`] expires silent rows past `stale_after` tokens
/// so a transient hard passage cannot permanently trap decode at a narrow width.
pub(in crate::experts) enum Judgement {
    Unjudged,
    Judged {
        window: RegimeAged<Tally>,
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

    /// Locks the stand rates, then reads the physical regime epoch, bank-wide
    /// epoch, and clock under that lock so every count and read observes
    /// monotonically non-decreasing `(regime, epoch, now)`. `horizon::Time` is
    /// a leaf lock, so `Stands::rows -> horizon::Time` cannot deadlock.
    fn lock_with_now(&self) -> (MutexGuard<'_, [[Judgement; 2]; ROWS]>, RegimeEpoch, u64, Tokens) {
        let rows = self.rows.lock().expect("the stand rates");
        let times = self.time.lock();
        let (regime, epoch, now) = (times.regime_epoch(), times.bank_epoch(), times.clock.through_latest());
        drop(times);
        (rows, regime, epoch, now)
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
        let (mut rows, regime, epoch, now) = self.lock_with_now();
        let cell = Self::cell(&mut rows, row, form);
        match cell {
            Judgement::Unjudged => {
                *cell = Judgement::Judged {
                    window: RegimeAged::new(regime, epoch),
                    stale_after: STAND_STALE_BASE,
                };
            }
            Judgement::Judged { window, stale_after } => {
                match window.read(regime, epoch, now) {
                    None => {
                        // Regime transitioned: reset backoff to base; `window.count`
                        // below starts fresh at `regime`.
                        *stale_after = STAND_STALE_BASE;
                    }
                    Some((_, age)) if age.get() >= *stale_after => {
                        // Re-entering after staleness expiry within the same regime:
                        // drop the frozen pre-silence epoch tally so the fresh probe
                        // stands on its own verdict, and back off exponentially if it
                        // still falls (or reset to base if it stands).
                        window.reset(regime, epoch);
                        *stale_after = if stood {
                            STAND_STALE_BASE
                        } else {
                            (*stale_after * 2.0).min(STAND_STALE_MAX)
                        };
                    }
                    Some(_) if stood => {
                        *stale_after = STAND_STALE_BASE;
                    }
                    Some(_) => {}
                }
            }
        }
        let Judgement::Judged { window, .. } = cell else { unreachable!("judged above") };
        *window.count(regime, epoch, now) += Tally::of(stood);
    }

    /// r_row of `form` (§3.4) and its age, the tokens since its row was last
    /// judged in the current physical regime: row `row` ≥ 1 standing, every
    /// row before it having stood; none unjudged in this regime (whose bound,
    /// 1, the caller applies).
    pub fn judged(&self, row: usize, form: Form) -> Option<(f64, Tokens)> {
        let (mut rows, regime, epoch, now) = self.lock_with_now();
        Self::read(&mut rows, row, form, regime, epoch, now)
    }

    /// r_row of `form`: [`Self::judged`]'s rate.
    pub fn rate(&self, row: usize, form: Form) -> Option<f64> {
        self.judged(row, form).map(|(rate, _)| rate)
    }

    /// The token-age staleness threshold currently governing `(row, form)`
    /// (`STAND_STALE_BASE`..=`STAND_STALE_MAX`) in the current physical regime.
    pub fn stale_after(&self, row: usize, form: Form) -> f64 {
        let (mut rows, regime, _, _) = self.lock_with_now();
        match Self::cell(&mut rows, row, form) {
            Judgement::Unjudged => STAND_STALE_BASE,
            Judgement::Judged { window, stale_after } if window.regime() == regime => *stale_after,
            Judgement::Judged { .. } => STAND_STALE_BASE,
        }
    }

    /// r_row for `preferred`, falling back to the other form's live rate when
    /// `preferred` is stale relative to it (strictly older on the token clock
    /// and the other form is still fresh), and expiring to `None` once silent
    /// for at least `stale_after` tokens so `Regime::verify_emitted` carries
    /// forward the shallower live stand rate instead of staying permanently
    /// trapped by a frozen warmup or hard-passage sample. An unjudged
    /// `preferred` row (or one from a prior physical regime) stays `None` (its
    /// optimistic bound 1).
    pub fn live_rate(&self, row: usize, preferred: Form) -> Option<f64> {
        let (mut rows, regime, epoch, now) = self.lock_with_now();
        let (rate, age, stale_after) = Self::read_with_stale(&mut rows, row, preferred, regime, epoch, now)?;
        let other = match preferred {
            Form::Front => Form::Verify,
            Form::Verify => Form::Front,
        };
        if let Some((other_rate, other_age, other_stale_after)) =
            Self::read_with_stale(&mut rows, row, other, regime, epoch, now)
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
        let (mut rows, regime, epoch, now) = self.lock_with_now();
        std::array::from_fn(|index| Self::read(&mut rows, index + 1, Form::Front, regime, epoch, now).map_or(1.0, |(rate, _)| rate))
    }

    /// σ_depth of `form` (§3.3): the probability a pass's first `depth`
    /// guessed rows all stand, Π_{row ≤ depth} r_row, an unjudged row's at
    /// its bound 1.
    pub fn through_rows(&self, depth: usize, form: Form) -> f64 {
        let (mut rows, regime, epoch, now) = self.lock_with_now();
        (1..=depth).map(|row| Self::read(&mut rows, row, form, regime, epoch, now).map_or(1., |(rate, _)| rate)).product()
    }

    fn read_with_stale(
        rows: &mut [[Judgement; 2]; ROWS],
        row: usize,
        form: Form,
        regime: RegimeEpoch,
        epoch: u64,
        now: Tokens,
    ) -> Option<(f64, Tokens, f64)> {
        match Self::cell(rows, row, form) {
            Judgement::Unjudged => None,
            Judgement::Judged { window, stale_after } => {
                let (tallies, age) = window.read(regime, epoch, now)?;
                Some((Tally::over(tallies), age, *stale_after))
            }
        }
    }

    fn read(
        rows: &mut [[Judgement; 2]; ROWS],
        row: usize,
        form: Form,
        regime: RegimeEpoch,
        epoch: u64,
        now: Tokens,
    ) -> Option<(f64, Tokens)> {
        Self::read_with_stale(rows, row, form, regime, epoch, now).map(|(rate, age, _)| (rate, age))
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
    /// Path 0 (`LinkMover`) opportunistic DMA staging copies that landed
    /// (`Self::link_landed`), keyed by `(step, layer, expert)` until
    /// `Self::resolve` clears the step.
    landed_links: BTreeSet<(Step, usize, usize)>,
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
    /// Number of calibrated ranks per look (`RANKS`).
    ranks: usize,
    /// Bounded live per-token and bigram routing table (`Source::Token`).
    token_routes: TokenRoutes,
    /// Completed verify steps waiting for their per-request verdict before
    /// feeding committed rows into `token_routes` (rejected/discarded rows
    /// never train `TokenRoutes`).
    pending_token_routes: BTreeMap<Step, PendingTokenRoutes>,
    /// Drafted/anchor tokens buffered for each request's upcoming pass
    /// (`Told::DraftTokens`), cleared once that pass begins or closes.
    pending_draft_tokens: HashMap<RequestId, Vec<TokenId>>,
    /// High-confidence `Source::Token` predictions (`p >= 0.70`) from the most
    /// recent `Told::DraftTokens` or `Told::Began` publication, forwarded to
    /// `Directory` to protect upcoming drafted tokens' residents from eviction.
    token_protected: Vec<((usize, usize), Step, f64)>,
}

struct PendingTokenRoutes {
    rows: Box<[DecodeRow]>,
    row_experts: Vec<Vec<(u8, u16)>>,
    remaining: Vec<RequestId>,
}

/// Bounded per-layer expert hit counts for one unigram token or bigram pair.
#[derive(Clone, Debug, Default)]
struct TokenLayerStats {
    seen: u16,
    hits: Vec<(u16, u16)>,
}

impl TokenLayerStats {
    const MAX_EXPERTS: usize = 24;
    const DECAY_AT: u16 = 512;

    fn observe(&mut self, experts: &[u16]) {
        if self.seen >= Self::DECAY_AT {
            self.seen /= 2;
            for &mut (_, ref mut count) in &mut self.hits {
                *count /= 2;
            }
            self.hits.retain(|&(_, count)| count > 0);
        }
        self.seen = self.seen.saturating_add(1);
        for &expert in experts {
            if let Some(entry) = self.hits.iter_mut().find(|&&mut (e, _)| e == expert) {
                entry.1 = entry.1.saturating_add(1);
            } else if self.hits.len() < Self::MAX_EXPERTS {
                self.hits.push((expert, 1));
            } else if let Some((idx, _)) = self.hits.iter().enumerate().min_by_key(|&(_, &(_, c))| c) {
                if self.hits[idx].1 <= 1 {
                    self.hits[idx] = (expert, 1);
                }
            }
        }
    }

    /// Laplace-smoothed probability `hits / (seen + 1)` for each tracked expert
    /// once the token/bigram has been observed at least once.
    fn probs(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        let ready = self.seen >= 1;
        let denom = f64::from(self.seen) + 1.0;
        self.hits
            .iter()
            .filter(move |_| ready)
            .map(move |&(expert, count)| (usize::from(expert), f64::from(count) / denom))
    }
}

/// Bounded per-request committed token + per-layer expert route history used
/// for exact span matching (`expert-streaming.md` §6: when the output re-emits
/// a span of earlier tokens, the recorded routes predict the upcoming layers'
/// experts with high precision, calibrated like any other `Source::Token`).
#[derive(Default)]
struct RequestRouteHistory {
    tokens: VecDeque<TokenId>,
    /// `experts[pos][layer]` holds the expert IDs routed for `tokens[pos]` at `layer`.
    experts: VecDeque<Vec<Vec<u16>>>,
}

impl RequestRouteHistory {
    const MAX_TOKENS: usize = 4096;

    fn push(&mut self, token: TokenId, per_layer: Vec<Vec<u16>>) {
        if self.tokens.len() >= Self::MAX_TOKENS {
            self.tokens.pop_front();
            self.experts.pop_front();
        }
        self.tokens.push_back(token);
        self.experts.push_back(per_layer);
    }

    fn last_token(&self) -> Option<TokenId> {
        self.tokens.back().copied()
    }

    /// Find the most recent position `p` in `self.tokens` where the trailing
    /// 2-token suffix `(prev, cur)` (preferring a 3-token suffix `(prev2, prev, cur)`
    /// when `prev2` is present) matches. Returns `(pos, matched_len)`.
    fn match_suffix(&self, prev2: Option<TokenId>, prev: TokenId, cur: TokenId) -> Option<(usize, usize)> {
        let len = self.tokens.len();
        if len < 2 {
            return None;
        }
        if let Some(p2) = prev2 {
            if len >= 3 {
                for p in (2..len).rev() {
                    if self.tokens[p] == cur && self.tokens[p - 1] == prev && self.tokens[p - 2] == p2 {
                        return Some((p, 3));
                    }
                }
            }
        }
        for p in (1..len).rev() {
            if self.tokens[p] == cur && self.tokens[p - 1] == prev {
                return Some((p, 2));
            }
        }
        None
    }
}

/// Bounded live per-token, bigram, and per-request span routing table owned by
/// `Ledger` (`Source::Token`): learns `(token, layer) -> experts`,
/// `((prev, token), layer) -> experts`, and per-request committed route spans
/// from completed decode/verify passes, and predicts burst-level union
/// probabilities `1 - Π_j (1 - p(e | y_j))` plus exact span matches for
/// upcoming drafted tokens (`Told::DraftTokens` / `Told::Began`).
struct TokenRoutes {
    layers: usize,
    unigrams: HashMap<TokenId, Vec<TokenLayerStats>>,
    unigram_order: VecDeque<TokenId>,
    bigrams: HashMap<(TokenId, TokenId), Vec<TokenLayerStats>>,
    bigram_order: VecDeque<(TokenId, TokenId)>,
    per_request: HashMap<RequestId, RequestRouteHistory>,
}

impl TokenRoutes {
    const MAX_UNIGRAMS: usize = 4096;
    const MAX_BIGRAMS: usize = 4096;

    fn new(layers: usize) -> Self {
        Self {
            layers,
            unigrams: HashMap::new(),
            unigram_order: VecDeque::new(),
            bigrams: HashMap::new(),
            bigram_order: VecDeque::new(),
            per_request: HashMap::new(),
        }
    }

    fn remove_request(&mut self, request: RequestId) {
        self.per_request.remove(&request);
    }

    fn unigram_mut(&mut self, token: TokenId) -> &mut Vec<TokenLayerStats> {
        if !self.unigrams.contains_key(&token) {
            if self.unigrams.len() >= Self::MAX_UNIGRAMS {
                if let Some(oldest) = self.unigram_order.pop_front() {
                    self.unigrams.remove(&oldest);
                }
            }
            self.unigram_order.push_back(token);
            self.unigrams.insert(token, vec![TokenLayerStats::default(); self.layers]);
        }
        self.unigrams.get_mut(&token).expect("inserted above")
    }

    fn bigram_mut(&mut self, pair: (TokenId, TokenId)) -> &mut Vec<TokenLayerStats> {
        if !self.bigrams.contains_key(&pair) {
            if self.bigrams.len() >= Self::MAX_BIGRAMS {
                if let Some(oldest) = self.bigram_order.pop_front() {
                    self.bigrams.remove(&oldest);
                }
            }
            self.bigram_order.push_back(pair);
            self.bigrams.insert(pair, vec![TokenLayerStats::default(); self.layers]);
        }
        self.bigrams.get_mut(&pair).expect("inserted above")
    }

    /// Learn per-row routing from a completed decode/verify step, including
    /// only rows for which `include(row)` holds (so rejected draft rows and
    /// unverified front rows never train `TokenRoutes`).
    fn observe_where(&mut self, rows: &[DecodeRow], row_experts: &[Vec<(u8, u16)>], include: impl Fn(&DecodeRow) -> bool) {
        if self.layers == 0 || row_experts.is_empty() {
            return;
        }
        let mut per_row: Vec<Vec<Vec<u16>>> = vec![vec![Vec::new(); self.layers]; rows.len()];
        for (layer, pairs) in row_experts.iter().enumerate().take(self.layers) {
            for &(row, expert) in pairs {
                if let Some(layers_for_row) = per_row.get_mut(usize::from(row)) {
                    layers_for_row[layer].push(expert);
                }
            }
        }
        for (row_idx, row) in rows.iter().enumerate() {
            if !include(row) {
                continue;
            }
            let Some(token) = row.token else {
                continue;
            };
            let row_layers = std::mem::take(&mut per_row[row_idx]);
            if !row_layers.iter().any(|e| !e.is_empty()) {
                continue;
            }
            let prev = self.per_request.get(&row.request).and_then(RequestRouteHistory::last_token);
            for (layer, experts) in row_layers.iter().enumerate() {
                if experts.is_empty() {
                    continue;
                }
                self.unigram_mut(token)[layer].observe(experts);
                if let Some(prev_tok) = prev {
                    self.bigram_mut((prev_tok, token))[layer].observe(experts);
                }
            }
            self.per_request.entry(row.request).or_default().push(token, row_layers);
        }
    }

    /// Seed unigram and bigram routing tables from a completed grouped prefill
    /// prompt (`tokens` of length `M`, `layer_routes[layer]` of length
    /// `M * selected` in row-major order) before decode step 1.
    fn observe_prompt(&mut self, tokens: &[TokenId], layer_routes: &[Vec<u16>], selected: usize) {
        if self.layers == 0 || tokens.is_empty() || selected == 0 || layer_routes.is_empty() {
            return;
        }
        let expected_len = tokens.len().saturating_mul(selected);
        for (layer, routes) in layer_routes.iter().enumerate().take(self.layers) {
            if routes.len() < expected_len {
                continue;
            }
            for (row_idx, &token) in tokens.iter().enumerate() {
                let start = row_idx * selected;
                let experts = &routes[start..start + selected];
                self.unigram_mut(token)[layer].observe(experts);
                if row_idx > 0 {
                    self.bigram_mut((tokens[row_idx - 1], token))[layer].observe(experts);
                }
            }
        }
    }

    /// Predict `(layer, expert, confidence_bucket, p_union)` for an upcoming
    /// burst of `tokens` for `request`, combining:
    /// 1. Exact per-request span matching against committed route history
    ///    (rank 0 for matched positions, rank 1 for 3-gram continuation), and
    /// 2. Noisy-OR union probability across bigram (using `request`'s trailing
    ///    committed token for `tokens[0]`) and unigram stats (`seen >= 1`,
    ///    preferring `seen >= 2` before falling back to `seen >= 1`).
    /// Buckets: `0` for `p >= 0.85` / exact span match, `1` for `0.70..0.85`,
    /// `2` for `0.55..0.70`, `3` for `0.25..0.55` (admitted only when
    /// `PathAdmission::break_even` allows free-slack reads).
    fn predict(&self, request: Option<RequestId>, tokens: &[TokenId]) -> Vec<(usize, usize, u8, f64)> {
        if self.layers == 0 || tokens.is_empty() {
            return Vec::new();
        }
        let hist = request.and_then(|r| self.per_request.get(&r));
        let mut span_matches: Vec<(usize, u8)> = Vec::new();
        if let Some(hist) = hist {
            let hlen = hist.tokens.len();
            if hlen >= 1 {
                let mut ctx: Vec<TokenId> = Vec::with_capacity(2 + tokens.len());
                if hlen >= 2 {
                    ctx.push(hist.tokens[hlen - 2]);
                }
                ctx.push(hist.tokens[hlen - 1]);
                let prefix_len = ctx.len();
                ctx.extend_from_slice(tokens);
                for q_idx in 0..tokens.len() {
                    let c_idx = prefix_len + q_idx;
                    let prev = ctx[c_idx - 1];
                    let prev2 = (c_idx >= 2).then(|| ctx[c_idx - 2]);
                    if let Some((pos, matched_len)) = hist.match_suffix(prev2, prev, ctx[c_idx]) {
                        span_matches.push((pos, 0));
                        // When a 3-token span matches at the end of the current
                        // drafted prefix (e.g. anchor arrival at DraftWindow start),
                        // predict the next token in the matched span at bucket 1.
                        if q_idx + 1 == tokens.len() && matched_len >= 3 && pos + 1 < hlen {
                            span_matches.push((pos + 1, 1));
                        }
                    }
                }
            }
        }
        let trailing_prev = hist.and_then(RequestRouteHistory::last_token);
        let mut out = Vec::new();
        let mut not_used: Vec<(usize, f64)> = Vec::new();
        let mut span_best: Vec<(usize, u8)> = Vec::new();
        for layer in 0..self.layers {
            span_best.clear();
            if let Some(hist) = hist {
                for &(pos, bucket) in &span_matches {
                    for &expert in &hist.experts[pos][layer] {
                        let e = usize::from(expert);
                        if let Some(existing) = span_best.iter_mut().find(|&&mut (id, _)| id == e) {
                            existing.1 = existing.1.min(bucket);
                        } else {
                            span_best.push((e, bucket));
                        }
                    }
                }
            }
            not_used.clear();
            for (idx, &token) in tokens.iter().enumerate() {
                let prev = if idx > 0 { Some(tokens[idx - 1]) } else { trailing_prev };
                let stats = prev
                    .and_then(|p| self.bigrams.get(&(p, token)).map(|v| &v[layer]).filter(|s| s.seen >= 2))
                    .or_else(|| self.unigrams.get(&token).map(|v| &v[layer]).filter(|s| s.seen >= 2))
                    .or_else(|| prev.and_then(|p| self.bigrams.get(&(p, token)).map(|v| &v[layer]).filter(|s| s.seen >= 1)))
                    .or_else(|| self.unigrams.get(&token).map(|v| &v[layer]).filter(|s| s.seen >= 1));
                let Some(stats) = stats else {
                    continue;
                };
                for (expert, p) in stats.probs() {
                    let factor = (1.0 - p).max(0.0);
                    if let Some(entry) = not_used.iter_mut().find(|&&mut (e, _)| e == expert) {
                        entry.1 *= factor;
                    } else {
                        not_used.push((expert, factor));
                    }
                }
            }
            for &(expert, bucket) in &span_best {
                let p_span = if bucket == 0 { 1.0 } else { 0.75 };
                out.push((layer, expert, bucket, p_span));
            }
            for &(expert, miss_p) in &not_used {
                if span_best.iter().any(|&(e, _)| e == expert) {
                    continue;
                }
                let p_union = 1.0 - miss_p;
                let bucket = if p_union >= 0.85 {
                    0
                } else if p_union >= 0.70 {
                    1
                } else if p_union >= 0.55 {
                    2
                } else if p_union >= 0.25 {
                    3
                } else {
                    continue;
                };
                out.push((layer, expert, bucket, p_union));
            }
        }
        out
    }
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
            landed_links: BTreeSet::new(),
            spent: Vec::new(),
            awaiting: HashMap::new(),
            judged: Vec::new(),
            accepted: Vec::new(),
            draft_window: None,
            ranks,
            token_routes: TokenRoutes::new(layers),
            pending_token_routes: BTreeMap::new(),
            pending_draft_tokens: HashMap::new(),
            token_protected: Vec::new(),
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
            Told::Completed { step, chose, guesses, row_experts, .. } => return self.complete(step, &chose, guesses, &row_experts),
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
            Told::DraftTokens { request, tokens } => self.draft_tokens(request, &tokens),
            Told::PromptRoutes { tokens, layer_routes, selected } => {
                self.token_routes.observe_prompt(&tokens, &layer_routes, selected);
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
                    acceptable: usize::from(of.acceptable()),
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
        for &(request, _) in &passes {
            self.pending_draft_tokens.remove(&request);
        }
        let mut req_tokens: BTreeMap<RequestId, Vec<TokenId>> = BTreeMap::new();
        for row in rows.iter() {
            if let Some(tok) = row.token {
                req_tokens.entry(row.request).or_default().push(tok);
            }
        }
        self.launches.insert(step, Launch { rows, padding, passes });
        self.token_protected.clear();
        for (&request, tokens) in &req_tokens {
            self.publish_token_routes(Some(request), When::Step(step), step, tokens);
        }
    }

    /// Buffer drafted/anchor tokens for `request`'s upcoming burst and publish
    /// token-conditioned `Source::Token` interests bound to `request`'s next pass.
    fn draft_tokens(&mut self, request: RequestId, tokens: &[TokenId]) {
        if tokens.is_empty() || self.layers.is_empty() {
            return;
        }
        let buffered = self.pending_draft_tokens.entry(request).or_default();
        for &tok in tokens {
            if buffered.len() < super::DEPTHS {
                buffered.push(tok);
            }
        }
        let snapshot = buffered.clone();
        let next_pass = self.passes.get(&request).copied().unwrap_or(0) + 1;
        let estimate = self.latest_active_step().unwrap_or(self.reached.0).saturating_add(1);
        self.token_protected.clear();
        self.publish_token_routes(Some(request), When::Pass { request, pass: next_pass, estimate }, estimate, &snapshot);
    }

    /// High-confidence `Source::Token` predictions (`p >= 0.70`) from the most
    /// recent `Told::DraftTokens` or `Told::Began` publication.
    pub fn token_protected(&self) -> &[((usize, usize), Step, f64)] {
        &self.token_protected
    }

    /// Publish `Source::Token` predictions for `tokens` into `NameLedger`,
    /// recording high-confidence (`p >= 0.70`) entries for `Directory` eviction
    /// protection.
    fn publish_token_routes(&mut self, request: Option<RequestId>, when: When, step: Step, tokens: &[TokenId]) {
        if self.ranks == 0 || tokens.is_empty() {
            return;
        }
        let predictions = self.token_routes.predict(request, tokens);
        if predictions.is_empty() {
            return;
        }
        let max_rank = (self.ranks - 1).min(u8::MAX as usize) as u8;
        let now = Instant::now();
        for (layer, expert, bucket, p_union) in predictions {
            if p_union >= 0.70 {
                self.token_protected.push(((layer, expert), step, p_union));
            }
            let name = Name {
                source: Source::Token,
                when,
                rank: bucket.min(max_rank),
                from: At::whole(step, 0),
                candidate: true,
                lineage: None,
            };
            self.publish(layer, expert, name, now);
        }
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
            self.awaiting.entry(request).or_default().insert(step, Awaiting { exact: 0, acceptable: 0, decided: false, fronted: Fronted::Without });
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
    fn complete(&mut self, step: Step, chose: &[Vec<Chosen>], guesses: Option<u32>, row_experts: &[Vec<(u8, u16)>]) -> Learned {
        let mut learned = Learned::default();
        for (layer, chose) in chose.iter().enumerate() {
            self.resolve(step, layer, chose, &mut learned);
        }
        if let Some(launch) = self.launches.get(&step) {
            let mut verifiers: Vec<RequestId> = Vec::new();
            for row in launch.rows.iter() {
                if row.ahead.is_some_and(|a| !a.front()) && !verifiers.contains(&row.request) {
                    verifiers.push(row.request);
                }
            }
            // Observe exact rows of non-verifying requests immediately; defer
            // verifying requests until `verdict()` so rejected/discarded rows
            // never pollute `TokenRoutes`.
            self.token_routes.observe_where(&launch.rows, row_experts, |row| {
                row.ahead.is_none() && !verifiers.contains(&row.request)
            });
            if !verifiers.is_empty() {
                self.pending_token_routes.insert(
                    step,
                    PendingTokenRoutes {
                        rows: launch.rows.clone(),
                        row_experts: row_experts.to_vec(),
                        remaining: verifiers,
                    },
                );
            }
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
        let Self { layers, calibration, owed, reads, landed_links, spent, .. } = self;
        for chosen in chose {
            // The chosen experts a read ahead was issued for, unspent until
            // now: used.
            let read = reads.remove(&(layer, chosen.expert));
            if let Some(Issued { basis, depth, verify, bytes, fell, .. }) = read {
                spent.push(Spent::Used { basis, depth, verify, fell, bytes });
            }
        }
        landed_links.retain(|&(s, l, _)| !(s <= step && l == layer));
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
        let acceptable_count = verdict.acceptable.len();
        let stood = verdict.acceptable.iter().take_while(|&&stood| stood).count();
        if let Some(pending) = self.pending_token_routes.get_mut(&step) {
            if let Some(pos) = pending.remaining.iter().position(|&id| id == request) {
                pending.remaining.swap_remove(pos);
                self.token_routes.observe_where(&pending.rows, &pending.row_experts, |row| {
                    row.request == request
                        && match row.ahead {
                            None => true,
                            Some(a) => !a.front() && usize::from(a.row()) < stood,
                        }
                });
                if pending.remaining.is_empty() {
                    self.pending_token_routes.remove(&step);
                }
            }
        }
        for (depth, &guess) in verdict.acceptable.iter().enumerate().take(stood + 1) {
            self.stands.count(depth + 1, Form::Verify, guess);
            let depth = u8::try_from(depth + 1).expect("a pass's acceptable rows fit a byte");
            self.accepted.push(Accepted { depth, stood: guess });
        }
        if stood < acceptable_count {
            // Early rejection: cancel rejected acceptable rows AND all trailing
            // front rows (`lineage.guessed > stood`) and drop `Awaiting`.
            self.fall(request, step, stood);
            if let Fronted::Rows(pass) = awaiting.fronted {
                self.judged.extend(pass.pending(step));
            }
        } else {
            match &mut awaiting.fronted {
                Fronted::Without => {}
                // Of no kind: it counts into no stand, but its first guess's
                // fall against `truths[K]` drops all trailing front rows.
                Fronted::Unread => {
                    if verdict.front.first().zip(verdict.truths.get(acceptable_count)).is_some_and(|(guess, truth)| guess != truth) {
                        self.fall(request, step, acceptable_count);
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
            let Awaiting { exact: base, acceptable, decided: true, fronted: Fronted::Rows(pass) } = awaiting else {
                continue;
            };
            let front_base = *base + *acceptable as u64;
            let tokens = pass.tokens.as_ref().expect("a decided pass holds its front tokens");
            while pass.stood < pass.rows {
                let depth = pass.stood + 1;
                let position = front_base + u64::from(depth);
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
                    fell.push((of, *acceptable + usize::from(depth - 1)));
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
            self.fall(request, of, stood);
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
        if let Some(pending) = self.pending_token_routes.get_mut(&step) {
            pending.remaining.retain(|&id| id != request);
            if pending.remaining.is_empty() {
                self.pending_token_routes.remove(&step);
            }
        }
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
        self.pending_draft_tokens.remove(&request);
        self.token_routes.remove_request(request);
        self.pending_token_routes.retain(|_, pending| {
            pending.remaining.retain(|&id| id != request);
            !pending.remaining.is_empty()
        });
        for (step, awaiting) in self.awaiting.remove(&request).unwrap_or_default() {
            if let Fronted::Rows(pass) = awaiting.fronted {
                self.judged.extend(pass.pending(step));
            }
        }
        self.drop_where(|interest| {
            interest.lineage.is_some_and(|lineage| lineage.request == request)
                || matches!(interest.when, When::Pass { request: of, .. } if of == request)
        });
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

    /// The read ahead of `expert` of `layer` landed at `at`.
    pub fn landed(&mut self, layer: usize, expert: usize, at: Instant) {
        if let Some(read) = self.reads.get_mut(&(layer, expert)) {
            read.landed = Some(at);
        }
    }

    /// Record a completed Path 0 (`LinkMover`) opportunistic DMA staging copy
    /// for `(step, layer, expert)` so `residency` classifies link-staged hits
    /// without polluting Path 1 (`Drive`) `Spent` telemetry.
    pub fn link_landed(&mut self, step: Step, layer: usize, expert: usize, _issued: Instant, _landed: Instant) {
        self.landed_links.insert((step, layer, expert));
    }

    /// Whether a Path 0 (`LinkMover`) opportunistic DMA staging copy landed for
    /// `(step, layer, expert)`.
    pub fn has_link_landed(&self, step: Step, layer: usize, expert: usize) -> bool {
        self.landed_links.contains(&(step, layer, expert))
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
        if let Some(Issued { basis, depth, verify, bytes, fell, .. }) = self.reads.remove(&(layer, expert)) {
            if reading {
                self.spent.push(Spent::Used { basis, depth, verify, fell, bytes });
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
        // Layer-EDF ordering on both `PathKind::Drive` and `PathKind::Link`:
        // evaluate unbegun queued moves first, then order by `(step, layer, -p, first)`
        // so early layers (0..5) always race ahead of `routed_land` parks instead
        // of late layers starving Layer 0.
        let _ = path;
        candidates.sort_by(|left, right| {
            left.3
                .cmp(&right.3)
                .then(left.4.cmp(&right.4))
                .then(left.0.cmp(&right.0))
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
                Act::Reprice(Class::Expected { step, p }) => (2, step, scheduled.layer, -p),
                Act::Issue(Class::Expected { step, p }) => (3, step, scheduled.layer, -p),
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
    const TOKEN: Source = Source::Token;

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
        ledger.tell(Told::Completed { step, chose, guesses: Some(0), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() }).resolved
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
        let token_d0 = Source::Token;
        let verify_d1 = Source::Ahead(Row { index: 0, front: None });

        // Depth-0 Token is nearer than depth-1 Route.
        // Two requests naming token_d0 at ranks 4 and 2 share one look, keeping best rank 2.
        let comb = Combination::of([
            Kind { source: front_d2, rank: 0 },
            Kind { source: front_d1, rank: 1 },
            Kind { source: token_d0, rank: 4 },
            Kind { source: token_d0, rank: 2 },
            Kind { source: verify_d1, rank: 0 },
        ])
        .expect("non-empty");
        assert_eq!(comb.nearest, Look { depth: 0, seen: Seen::Token, form: Form::Front });
        assert_eq!(comb.rank, 2, "best rank among names from the nearest look");
        assert_eq!(comb.looks, 4, "two names from token_d0 count as one look");

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

    /// What a read is admitted at is the measured rate of the combination
    /// of names its admission was decided on for its step, not a
    /// composition of their precisions: the experts two sources named
    /// together, decided on as such, are used at their own rate.
    /// Unmeasured, the bound 1.
    #[test]
    fn a_combination_of_names_is_admitted_at_its_measured_rate() {
        let mut ledger = calibrated(1, &[(AHEAD, 0.5), (TOKEN, 0.5)]);
        let t = now();
        let both = |ledger: &mut Ledger, expert, step| {
            ledger.publish(0, expert, name(AHEAD, step), t);
            ledger.publish(0, expert, name(TOKEN, step), t);
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
        let mut ledger = calibrated(1, &[(AHEAD, 0.8), (TOKEN, 0.2)]);
        let t = now();
        ledger.publish(0, 1, name(Source::Exact, 5), t);
        ledger.publish(0, 2, name(AHEAD, 6), t);
        ledger.publish(0, 3, name(TOKEN, 6), t);
        assert_eq!(ledger.act(0, 1, Read::Absent, &Unpriced), Act::Issue(Class::Due), "a due read is never priced");
        assert!(matches!(ledger.act(0, 2, Read::Absent, &PRICE), Act::Issue(Class::Expected { step: 6, .. })));
        assert_eq!(ledger.act(0, 3, Read::Absent, &PRICE), Act::Keep, "a name below the break-even claims no staging or tier slot");
        // An early Token name below the break-even never shadows an admitted later-step Ahead route.
        ledger.publish(0, 4, name(TOKEN, 5), t);
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
        let mut ledger = calibrated(1, &[(TOKEN, 0.2)]);
        let t = now();
        ledger.publish(0, 1, name(TOKEN, 6), t);
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
        ledger.publish(0, 2, name(TOKEN, 6), t);
        ledger.publish(0, 3, Name { candidate: false, ..name(TOKEN, 6) }, t);
        ledger.publish(0, 1, name(AHEAD, 7), t);
        ledger.publish(1, 4, name(AHEAD, 6), t);
        ledger.publish(0, 1, name(Source::Exact, 6), t);
        // The candidates' reads decided before the names pass.
        for expert in [1, 2] {
            ledger.act(0, expert, Read::Absent, &PRICE);
        }
        ledger.pass(0, t, |interest| interest.source == TOKEN);
        let mut resolved = resolve(&mut ledger, 6, 0, &[1]);
        resolved.sort_by_key(|resolved| resolved.expert);
        assert_eq!(resolved.iter().map(|resolved| (resolved.expert, resolved.used)).collect::<Vec<_>>(), [(1, true), (2, false), (3, false)]);
        // Each decided read's names for the step counted together: expert
        // 1's exact request is certain, no combination decided; expert 2's
        // token name, decided; expert 3's, never decided.
        let ahead_key = Combination::of([Kind { source: AHEAD, rank: 0 }]).unwrap();
        let token_key = Combination::of([Kind { source: TOKEN, rank: 0 }]).unwrap();
        assert_eq!((ledger.calibration.combined(0, ahead_key, 0), ledger.calibration.combined(0, token_key, 0)), (1., 0.));
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
        const VERIFY: Source = Source::Ahead(Row { index: 0, front: None });
        let mut ledger = calibrated(2, &[(AHEAD, 0.9), (TOKEN, 0.6)]);
        let t = now();
        ledger.publish(0, 1, name(TOKEN, 6), t);
        ledger.publish(1, 2, name(AHEAD, 8), t);
        ledger.publish(0, 3, name(TOKEN, 7), t);
        ledger.publish(1, 4, name(TOKEN, 6), t);
        ledger.publish(0, 5, name(Source::Exact, 5), t);
        ledger.publish(1, 6, front(AHEAD, 6, 1), t);
        ledger.publish(0, 7, name(VERIFY, 6), t);
        ledger.publish(0, 8, name(AHEAD, 9), t);
        assert_eq!(ledger.pass(0, t, |interest| interest.source == VERIFY), vec![7]);
        let reading = |layer, expert| match (layer, expert) {
            (0, 5) | (0, 7) | (1, 2) => Read::Reading { begun: false },
            _ => Read::Absent,
        };
        let mut experts = ledger.wanted();
        experts.push((0, 7));
        let scheduled = ledger.schedule(experts, reading, |_| Flat { even: 0.3 });
        let order = scheduled.iter().map(|scheduled| (scheduled.layer, scheduled.expert)).collect::<Vec<_>>();
        // Layer-EDF ordering within step 6 puts Layer 0 `(0, 1)` ahead of Layer 1 `(1, 6)` and `(1, 4)`.
        assert_eq!(order, [(0, 7), (0, 5), (1, 2), (0, 1), (1, 6), (1, 4), (0, 3), (0, 8)]);
        assert_eq!((scheduled[0].act, scheduled[1].act), (Act::Cancel, Act::Reprice(Class::Due)));
        assert!(matches!(scheduled[2].act, Act::Reprice(Class::Expected { step: 8, .. })));
        assert!(matches!(scheduled[7].act, Act::Issue(Class::Expected { step: 9, p }) if p > 0.85), "the likeliest, wanted last, is read last");
        let front = scheduled[4];
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
        ledger.publish(0, 4, name(TOKEN, 6), t);
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
        let rows = [DecodeRow { position: 0, request: RequestId(1), ahead: None, token: None }; 5];
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
        let row = |request, ahead| DecodeRow { position: 0, request: RequestId(request), ahead, token: None };
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
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(chained), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        let row = |request, position, ahead| DecodeRow { position, request: RequestId(request), ahead, token: None };
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
            ledger.tell(Told::Completed { step, chose: vec![Vec::new()], guesses: Some(guesses), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        ledger.tell(Told::Completed { step: 7, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Chained)), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
        ledger.tell(Told::Began { step: 9, launch: decode(&[row(1, 14, None), row(1, 15, Some(front))]) });
        ledger.tell(Told::Completed { step: 9, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Head)), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        let row = |position, ahead| DecodeRow { position, request: RequestId(1), ahead, token: None };
        let front = AheadOf::new(1, 0, 0);
        for step in 4..=6 {
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows: Box::new([row(step + 6, None), row(step + 7, Some(front))]), padding: 0 } });
            let kind = if step == 6 { GuessKind::Head } else { GuessKind::Chained };
            ledger.tell(Told::Completed { step, chose: vec![Vec::new()], guesses: Some(kinds(1, kind)), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        let row = |position, ahead| DecodeRow { position, request: RequestId(1), ahead, token: None };
        ledger.tell(Told::Began { step: 4, launch: Launched::Decode { rows: Box::new([row(10, None), row(11, Some(AheadOf::new(1, 0, 0)))]), padding: 0 } });
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(0), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        ledger.tell(Told::Began { step: 8, launch: Launched::Decode { rows: Box::new([DecodeRow { position: 0, request: RequestId(3), ahead: None, token: None }]), padding: 0 } });
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
        let mut ledger = calibrated(1, &[(AHEAD, 0.8), (TOKEN, 0.8)]);
        let t = now();
        let of = |request| Name { lineage: Some(Lineage { request: RequestId(request), row: 1, guessed: NonZeroU8::MIN }), ..name(AHEAD, 6) };
        ledger.publish(0, 1, of(1), t);
        ledger.publish(0, 2, of(1), t);
        ledger.publish(0, 3, name(TOKEN, 6), t);
        ledger.publish(0, 4, of(2), t);
        ledger.publish(0, 5, name(TOKEN, 6), t);
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
                Spent::Used { basis: Basis::Token, depth: 0, verify: false, fell: false, bytes: 30 },
                Spent::Unused { basis: Basis::Token, depth: 0, verify: false, fell: false, bytes: 50 },
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
        assert_eq!(Basis::of(Source::Token), Basis::Token);
    }

    /// A launch of an exact row and three front rows: front row `d` (batch
    /// row `d`) names for its request's pass `d` passes on, its lineage its
    /// `d` guessed rows.
    fn three_deep(ledger: &mut Ledger, step: Step, exact: u64) {
        let rows = (0..4).map(|row| DecodeRow { position: exact + row as u64, request: RequestId(1), ahead: (row > 0).then(|| AheadOf::new(1, 0, row - 1)), token: None });
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
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(guesses), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        let row = |position, ahead| DecodeRow { position, request: RequestId(1), ahead, token: None };
        for step in [4, 5] {
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows: Box::new([row(step + 6, None), row(step + 7, Some(AheadOf::new(1, 0, 0)))]), padding: 0 } });
        }
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Chained)), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(three_kinds()), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows: Box::new([DecodeRow { position: exact, request: RequestId(1), ahead: None, token: None }, DecodeRow { position: exact + 1, request: RequestId(1), ahead: Some(AheadOf::new(1, 0, 0)), token: None }]), padding: 0 } });
            ledger.tell(Told::Completed { step, chose: vec![Vec::new()], guesses: Some(kinds(1, GuessKind::Chained)), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        ledger.tell(Told::Completed { step: 7, chose: vec![Vec::new()], guesses: Some(three_kinds()), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
        ledger.tell(Told::Decided { step: 7, request: RequestId(1), verdict: fronted(&[14, 15, 16], 14) });
        ledger.take_judged();
        three_deep(&mut ledger, 9, 15);
        ledger.tell(Told::Completed { step: 9, chose: vec![Vec::new()], guesses: Some(three_kinds()), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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
        ledger.tell(Told::Completed { step: 4, chose: vec![Vec::new()], guesses: Some(three_kinds()), row_experts: Vec::new(), physical_rows: 1, layer_misses: Vec::new() });
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

    /// Token-conditioned cross-burst foresight (`Source::Token` / `TokenRoutes`):
    /// learns per-token and bigram routing on `Told::Completed`, publishes
    /// high-precision (`p_union >= 0.85`, rank 0) interests into `NameLedger`
    /// when `Told::DraftTokens` arrives during `DraftWindow`, binds them to
    /// the next step on `Told::Began`, and calibrates them live on completion.
    #[test]
    fn token_routes_learns_and_publishes_high_precision_draft_tokens() {
        let time = Time::new(&[vec![64]], 8);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0, 0], 4, time, stands);
        let req = RequestId(7);
        // Observe 6 completed bursts where bigram (TokenId(10), TokenId(20))
        // consistently routes expert 3 at layer 0 and expert 5 at layer 1.
        for step in 1..=6u64 {
            let rows = Box::new([
                DecodeRow { position: step * 2, request: req, ahead: None, token: Some(TokenId(10)) },
                DecodeRow { position: step * 2 + 1, request: req, ahead: Some(AheadOf::new(1, 1, 0)), token: Some(TokenId(20)) },
            ]);
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows, padding: 0 } });
            ledger.tell(Told::Completed {
                step,
                chose: vec![
                    vec![Chosen { expert: 3, saves: true, tier: false }],
                    vec![Chosen { expert: 5, saves: true, tier: false }],
                ],
                guesses: None,
                row_experts: vec![vec![(1, 3)], vec![(1, 5)]],
                physical_rows: 2,
                layer_misses: Vec::new(),
            });
            ledger.tell(Told::Decided {
                step,
                request: req,
                verdict: Verdict { acceptable: vec![true], front: vec![], truths: vec![] },
            });
            ledger.tell(Told::Ended { step, completed: true });
        }
        // During DraftWindow for burst 7: anchor TokenId(10) arrives, then
        // drafted TokenId(20) arrives -> p_union = 6 / (6 + 1) = 0.857 >= 0.85 (rank 0!).
        ledger.tell(Told::DraftWindow { estimated_seconds: 0.012 });
        ledger.tell(Told::DraftTokens { request: req, tokens: vec![TokenId(10)] });
        ledger.tell(Told::DraftTokens { request: req, tokens: vec![TokenId(20)] });
        let wanted = ledger.wanted();
        assert!(wanted.contains(&(0, 3)), "must publish layer-0 expert 3 for drafted bigram: {wanted:?}");
        assert!(wanted.contains(&(1, 5)), "must publish layer-1 expert 5 for drafted bigram: {wanted:?}");
        let lead0 = ledger.leading(0, 3).expect("layer-0 Source::Token interest");
        assert_eq!(lead0.source, Source::Token);
        assert_eq!(lead0.rank, 0, "6/7 = 0.857 must map to highest-precision rank 0");

        // Next burst begins -> When::Pass binds to When::Step(7) and schedules via PathAdmission!
        let rows = Box::new([
            DecodeRow { position: 20, request: req, ahead: None, token: Some(TokenId(10)) },
            DecodeRow { position: 21, request: req, ahead: Some(AheadOf::new(1, 1, 0)), token: Some(TokenId(20)) },
        ]);
        ledger.tell(Told::Began { step: 7, launch: Launched::Decode { rows, padding: 0 } });
        assert_eq!(ledger.leading(0, 3).expect("bound interest").when, When::Step(7));
        let act = ledger.act(0, 3, Read::Absent, &PRICE);
        assert!(matches!(act, Act::Issue(Class::Expected { step: 7, .. })), "must schedule bound Source::Token read: {act:?}");
    }

    /// Exact span matching in `TokenRoutes`: even a single prior occurrence of a
    /// 3-token span `[100, 101, 102] -> 103` lets `DraftTokens` predict both the
    /// matched token's experts (rank 0) and the 1-step span continuation's
    /// experts (rank 1) during `DraftWindow`, and forgets the history on `Closed`.
    #[test]
    fn token_routes_span_matches_single_occurrence_and_cleans_up_on_close() {
        let time = Time::new(&[vec![64]], 8);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0, 0], 4, time, stands);
        let req = RequestId(9);
        // Commit a single 4-token span [100, 101, 102, 103] with distinct experts.
        let rows = Box::new([
            DecodeRow { position: 0, request: req, ahead: None, token: Some(TokenId(100)) },
            DecodeRow { position: 1, request: req, ahead: Some(AheadOf::new(1, 1, 0)), token: Some(TokenId(101)) },
            DecodeRow { position: 2, request: req, ahead: Some(AheadOf::new(1, 2, 0)), token: Some(TokenId(102)) },
            DecodeRow { position: 3, request: req, ahead: Some(AheadOf::new(1, 3, 0)), token: Some(TokenId(103)) },
        ]);
        ledger.tell(Told::Began { step: 1, launch: Launched::Decode { rows, padding: 0 } });
        ledger.tell(Told::Completed {
            step: 1,
            chose: vec![vec![], vec![]],
            guesses: None,
            row_experts: vec![
                vec![(0, 10), (1, 11), (2, 12), (3, 13)],
                vec![(0, 20), (1, 21), (2, 22), (3, 23)],
            ],
            physical_rows: 4,
            layer_misses: Vec::new(),
        });
        ledger.tell(Told::Decided {
            step: 1,
            request: req,
            verdict: Verdict { acceptable: vec![true, true, true], front: vec![], truths: vec![] },
        });
        ledger.tell(Told::Ended { step: 1, completed: true });

        // Commit [100, 101] again so the request's trailing context is [100, 101].
        let rows2 = Box::new([
            DecodeRow { position: 4, request: req, ahead: None, token: Some(TokenId(100)) },
            DecodeRow { position: 5, request: req, ahead: Some(AheadOf::new(1, 1, 0)), token: Some(TokenId(101)) },
        ]);
        ledger.tell(Told::Began { step: 2, launch: Launched::Decode { rows: rows2, padding: 0 } });
        ledger.tell(Told::Completed {
            step: 2,
            chose: vec![vec![], vec![]],
            guesses: None,
            row_experts: vec![vec![(0, 10), (1, 11)], vec![(0, 20), (1, 21)]],
            physical_rows: 2,
            layer_misses: Vec::new(),
        });
        ledger.tell(Told::Decided {
            step: 2,
            request: req,
            verdict: Verdict { acceptable: vec![true], front: vec![], truths: vec![] },
        });
        ledger.tell(Told::Ended { step: 2, completed: true });

        // DraftWindow starts with anchor TokenId(102) -> matches 3-token span
        // [100, 101, 102] (rank 0: experts 12, 22) AND predicts continuation
        // TokenId(103) (rank 1: experts 13, 23)!
        ledger.tell(Told::DraftWindow { estimated_seconds: 0.015 });
        ledger.tell(Told::DraftTokens { request: req, tokens: vec![TokenId(102)] });
        assert_eq!(ledger.leading(0, 12).expect("matched L0").rank, 0);
        assert_eq!(ledger.leading(1, 22).expect("matched L1").rank, 0);
        assert_eq!(ledger.leading(0, 13).expect("continuation L0").rank, 1);
        assert_eq!(ledger.leading(1, 23).expect("continuation L1").rank, 1);

        // Closing the request drops its route history cleanly.
        ledger.tell(Told::Closed(req));
        assert!(!ledger.token_routes.per_request.contains_key(&req));
    }

    /// Gate #3: `NameLedger` hybrid verify+front lifecycle:
    /// - Early rejection at `0` or `K - 1` cancels all trailing `F` front rows.
    /// - Full acceptance with mismatching bonus (`truths[K] != front[0]`) cancels
    ///   all trailing `F` front rows immediately.
    /// - Full acceptance with matching bonus (`truths[K] == front[0]`) keeps
    ///   front row 0 alive for `pass + 1` and binds to `When::Step` on next verify begin.
    /// - Padding rows and other-request launches never advance request pass horizon.
    #[test]
    fn hybrid_verify_with_front_ledger_lifecycle() {
        let make_ledger = || {
            let time = Time::new(&[vec![64]], 8);
            let stands = Stands::new(time.clone());
            Ledger::new(vec![0], 4, time, stands)
        };
        let req = RequestId(1);
        let other = RequestId(2);
        let t = Instant::now();
        let front_guesses_word = (GuessKind::Head.word() as u32) << (3 * 2) | (GuessKind::Head.word() as u32) << (4 * 2);

        let begin_hybrid = |ledger: &mut Ledger, step: Step, exact: u64, padding: usize| {
            let rows = Box::new([
                DecodeRow { position: exact, request: req, ahead: None, token: Some(TokenId(1)) },
                DecodeRow { position: exact + 1, request: req, ahead: Some(AheadOf::new(1, 2, 0)), token: Some(TokenId(10)) },
                DecodeRow { position: exact + 2, request: req, ahead: Some(AheadOf::new(1, 2, 1)), token: Some(TokenId(11)) },
                DecodeRow { position: exact + 3, request: req, ahead: Some(AheadOf::new(1, 2, 2)), token: Some(TokenId(20)) },
                DecodeRow { position: exact + 4, request: req, ahead: Some(AheadOf::new(1, 2, 3)), token: Some(TokenId(21)) },
            ]);
            ledger.tell(Told::Began { step, launch: Launched::Decode { rows, padding } });
            // Publish front row 0 (row 3 -> expert 30) and front row 1 (row 4 -> expert 31).
            for (batch_row, expert) in [(3usize, 30usize), (4usize, 31usize)] {
                let Origin::Ahead { row, when, lineage } = ledger.row(step, batch_row, front_guesses_word).expect("front row origin") else {
                    panic!("row {batch_row} must be Origin::Ahead");
                };
                ledger.publish(
                    0,
                    expert,
                    Name { source: Source::Ahead(row), when, lineage: Some(lineage), from: At::whole(step, 0), ..name(AHEAD, 0) },
                    t,
                );
            }
            ledger.tell(Told::Completed { step, chose: vec![vec![]], guesses: Some(front_guesses_word), row_experts: vec![vec![]], physical_rows: 1, layer_misses: Vec::new() });
        };

        // 1. Early rejection at 0 cancels both trailing front rows.
        {
            let mut ledger = make_ledger();
            begin_hybrid(&mut ledger, 1, 10, 0);
            let verdict = Verdict {
                acceptable: vec![false, false],
                front: vec![TokenId(20), TokenId(21)],
                truths: vec![TokenId(99)],
            };
            ledger.tell(Told::Decided { step: 1, request: req, verdict });
            assert_eq!(ledger.act(0, 30, Read::Reading { begun: false }, &PRICE), Act::Cancel);
            assert_eq!(ledger.act(0, 31, Read::Reading { begun: false }, &PRICE), Act::Cancel);
        }

        // 2. Early rejection at K - 1 cancels both trailing front rows.
        {
            let mut ledger = make_ledger();
            begin_hybrid(&mut ledger, 1, 10, 0);
            let verdict = Verdict {
                acceptable: vec![true, false],
                front: vec![TokenId(20), TokenId(21)],
                truths: vec![TokenId(10), TokenId(99)],
            };
            ledger.tell(Told::Decided { step: 1, request: req, verdict });
            assert_eq!(ledger.act(0, 30, Read::Reading { begun: false }, &PRICE), Act::Cancel);
            assert_eq!(ledger.act(0, 31, Read::Reading { begun: false }, &PRICE), Act::Cancel);
        }

        // 3. Full acceptance with mismatching bonus (truths[K] = 88 != front[0] = 20) cancels all front rows immediately.
        {
            let mut ledger = make_ledger();
            begin_hybrid(&mut ledger, 1, 10, 0);
            let verdict = Verdict {
                acceptable: vec![true, true],
                front: vec![TokenId(20), TokenId(21)],
                truths: vec![TokenId(10), TokenId(11), TokenId(88)],
            };
            ledger.tell(Told::Decided { step: 1, request: req, verdict });
            assert_eq!(ledger.act(0, 30, Read::Reading { begun: false }, &PRICE), Act::Cancel);
            assert_eq!(ledger.act(0, 31, Read::Reading { begun: false }, &PRICE), Act::Cancel);
        }

        // 4. Full acceptance with matching bonus (truths[K] == front[0] == 20):
        //    - front row 0 stays alive for pass + 1;
        //    - padding + other-request launches do NOT advance `req`'s pass horizon;
        //    - next verify begin for `req` binds When::Pass to When::Step(next_step).
        {
            let mut ledger = make_ledger();
            begin_hybrid(&mut ledger, 1, 10, 3);
            let verdict = Verdict {
                acceptable: vec![true, true],
                front: vec![TokenId(20), TokenId(21)],
                truths: vec![TokenId(10), TokenId(11), TokenId(20)],
            };
            ledger.tell(Told::Decided { step: 1, request: req, verdict });
            assert!(matches!(ledger.leading(0, 30).expect("front 0 alive").when, When::Pass { request: r, pass: 2, .. } if r == req));
            assert!(
                matches!(ledger.act(0, 30, Read::Reading { begun: false }, &PRICE), Act::Reprice(Class::Expected { .. })),
                "front row 0 must stay alive for pass + 1"
            );

            // Launch for another request + padding must NOT bind or advance `req`'s pass horizon.
            ledger.tell(Told::Began {
                step: 2,
                launch: Launched::Decode {
                    rows: Box::new([DecodeRow { position: 0, request: other, ahead: None, token: None }]),
                    padding: 3,
                },
            });
            assert!(matches!(ledger.leading(0, 30).expect("still unbound").when, When::Pass { request: r, pass: 2, .. } if r == req));

            // Next verify launch for `req` (step 3) binds front row 0 to When::Step(3).
            ledger.tell(Told::Began {
                step: 3,
                launch: Launched::Decode {
                    rows: Box::new([
                        DecodeRow { position: 13, request: req, ahead: None, token: Some(TokenId(20)) },
                        DecodeRow { position: 14, request: req, ahead: Some(AheadOf::new(1, 1, 0)), token: Some(TokenId(21)) },
                    ]),
                    padding: 0,
                },
            });
            assert_eq!(ledger.leading(0, 30).expect("bound to step 3").when, When::Step(3));
        }
    }

    /// Cross-pass intra-token foresight (`Source::CrossPass`) registers a
    /// same-step future `visit_layer` deadline for a shared physical stack,
    /// schedules a read-ahead at that deadline, and calibrates its live
    /// precision independently from token-conditioned foresight.
    #[test]
    fn cross_pass_foresight_registers_same_step_deadline_and_calibrates() {
        let time = Time::new(&[vec![4]], 32);
        let stands = Stands::new(time.clone());
        // 4 visits (0, 1, 2, 3) all in group 0; visit 0 and visit 2 share stack 0.
        let mut ledger = Ledger::new(vec![0, 0, 0, 0], 4, time, stands);
        let t0 = now();
        let cross = Name {
            source: Source::CrossPass,
            when: When::Step(5),
            rank: 0,
            from: At::whole(5, 0),
            candidate: true,
            lineage: None,
        };
        let p = ledger.publish(2, 7, cross, t0).expect("cross-pass interest publishes");
        assert_eq!(p, 1.0, "uncalibrated cross-pass rate starts at its bound 1.0");
        assert_eq!(ledger.wanted(), vec![(2, 7)]);

        let scheduled = ledger.schedule(vec![(2, 7)], |_, _| Read::Absent, |_| PRICE);
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].layer, 2);
        assert_eq!(scheduled[0].expert, 7);
        assert_eq!(scheduled[0].act, Act::Issue(Class::Expected { step: 5, p: 1.0 }));

        // Step 5 completes: visit 2 did NOT choose expert 7, so calibration drops to 0.0,
        // while Source::Token remains untouched at 1.0.
        ledger.pass(2, t0 + std::time::Duration::from_millis(1), |interest| interest.step() == 5);
        ledger.tell(Told::Completed {
            step: 5,
            chose: vec![vec![], vec![], vec![], vec![]],
            guesses: None,
            row_experts: Vec::new(),
            physical_rows: 1,
            layer_misses: Vec::new(),
        });
        let cross_key = Combination::of([Kind { source: Source::CrossPass, rank: 0 }]).unwrap();
        let token_key = Combination::of([Kind { source: TOKEN, rank: 0 }]).unwrap();
        assert_eq!(ledger.rate(2, cross_key), 0.0, "cross-pass calibrated from visit 2 outcome");
        assert_eq!(ledger.rate(2, token_key), 1.0, "token-conditioned calibration is independent");
    }

    #[test]
    fn prompt_routes_seed_unigrams_and_bigrams_before_decode_step_1() {
        assert_eq!(TokenRoutes::MAX_BIGRAMS, 4096);
        let time = Time::new(&[vec![4, 4]], 32);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0, 0], 4, time, stands);
        let req = RequestId(7);

        // Seed from grouped prefill BEFORE any decode step runs:
        // prompt tokens [10, 20, 10, 20, 10, 20], K = 2.
        // At layer 0: token 10 -> [1, 2], bigram (10, 20) -> [3, 4].
        // At layer 1: token 10 -> [5, 6], bigram (10, 20) -> [7, 8].
        let tokens: Box<[TokenId]> = [10, 20, 10, 20, 10, 20].into_iter().map(TokenId).collect();
        let layer0: Vec<u16> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let layer1: Vec<u16> = vec![5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8];
        ledger.tell(Told::PromptRoutes {
            tokens,
            layer_routes: vec![layer0, layer1],
            selected: 2,
        });

        // On decode step 1 (before any decode completion!), drafting [10, 20]
        // must immediately emit `Source::Token` predictions and high-confidence
        // eviction protection (`p >= 0.70`) for both unigram and bigram experts.
        ledger.tell(Told::DraftTokens { request: req, tokens: vec![TokenId(10), TokenId(20)] });
        assert!(
            matches!(ledger.leading(0, 1), Some(i) if i.source == Source::Token),
            "layer 0 unigram expert 1 must be predicted on token 0"
        );
        assert!(
            matches!(ledger.leading(0, 3), Some(i) if i.source == Source::Token),
            "layer 0 bigram expert 3 must be predicted on token 0"
        );
        assert!(
            matches!(ledger.leading(1, 7), Some(i) if i.source == Source::Token),
            "layer 1 bigram expert 7 must be predicted on token 0"
        );
        assert!(
            ledger.token_protected().iter().any(|&((layer, expert), _, p)| layer == 0 && expert == 3 && p >= 0.70),
            "high-confidence prompt-seeded prediction must be exported to Directory protection"
        );
    }

    #[test]
    fn layer_edf_drive_sort_orders_early_layers_ahead_of_late_layers() {
        let layers = 40usize;
        let time = Time::new(&[vec![4, 4]], 32);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0; layers], 4, time, stands);
        let now = Instant::now();

        // Publish Layer 39 FIRST with rank 0 (best rank), then Layer 0 SECOND
        // with rank 1 (slightly lower rank), both for step 2.
        ledger.publish(
            39,
            10,
            Name {
                source: Source::Token,
                when: When::Step(2),
                rank: 0,
                from: At::whole(1, 0),
                candidate: true,
                lineage: None,
            },
            now,
        );
        ledger.publish(
            0,
            5,
            Name {
                source: Source::Token,
                when: When::Step(2),
                rank: 1,
                from: At::whole(1, 0),
                candidate: true,
                lineage: None,
            },
            now,
        );

        let scheduled = ledger.schedule_for_path(
            magnitude_cost_model::PathKind::Drive,
            [(39, 10), (0, 5)],
            |_, _| Read::Absent,
            |_| Flat { even: 0.0 },
        );
        assert_eq!(scheduled.len(), 2);
        assert_eq!(scheduled[0].layer, 0, "Layer-EDF Drive sort must schedule Layer 0 ahead of Layer 39");
        assert_eq!(scheduled[0].expert, 5);
        assert_eq!(scheduled[1].layer, 39);
        assert_eq!(scheduled[1].expert, 10);
    }

    #[test]
    fn token_routes_emits_seen_once_bucket_3_and_prefers_seen_at_least_two() {
        assert_eq!(TokenRoutes::MAX_UNIGRAMS, 4096);
        let mut routes = TokenRoutes::new(1);
        // Single observation of token 42 -> expert 7 gives Laplace p = 1/(1+1) = 0.50 -> bucket 3.
        routes.observe_prompt(&[TokenId(42)], &[vec![7]], 1);
        let preds = routes.predict(None, &[TokenId(42)]);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0, 0);
        assert_eq!(preds[0].1, 7);
        assert_eq!(preds[0].2, 3, "p = 0.50 must emit bucket 3");
        assert!((preds[0].3 - 0.50).abs() < 1e-9);

        // Three observations of unigram 99 -> expert 3, plus one observation of
        // bigram (42, 99) -> expert 8 (seen == 1): unigram 99 has seen = 4,
        // hits[3] = 3 -> p = 3/5 = 0.60 (bucket 2), so predicting [42, 99] must
        // prefer the seen >= 2 unigram stats for token 99 over the seen == 1 bigram.
        routes.observe_prompt(&[TokenId(99), TokenId(99), TokenId(99)], &[vec![3, 3, 3]], 1);
        routes.observe_prompt(&[TokenId(42), TokenId(99)], &[vec![7, 8]], 1);
        let preds = routes.predict(None, &[TokenId(42), TokenId(99)]);
        assert!(
            preds.iter().any(|&(layer, expert, bucket, _)| layer == 0 && expert == 3 && bucket <= 2),
            "seen >= 2 unigram must take precedence over seen == 1 bigram: {preds:?}"
        );
    }
}
