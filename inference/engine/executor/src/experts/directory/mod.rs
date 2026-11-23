//! Which bank record holds which expert. The
//! host places every expert into the bank and is the one writer of its map
//! (`experts::map`). Every choice of what the bank keeps ranks by the one
//! ranking ([`Reuse`], measured from the completed steps' reads, which its
//! stack `Stack` feeds): a promotion's ([`Directory::promotions`]), which
//! brings the ranking's pool of each group into the bank in place of what
//! lies outside it, a prompt group's seeds' ([`Directory::seed`]) and a
//! shrink's. No layer has a share of its group's records: the ranking alone
//! splits them across the group's layers, a prompt's seeds' among them.
//!
//! It builds the load's one [`Time`] (the token clock and each tier's
//! turnover, `experts::horizon`) and feeds it every move of a record: a
//! landing into a host record ([`Directory::landed`], a write-back's
//! among them) and a promotion's completion ([`Directory::promoted`]);
//! seeds, shrinks' evictions and lends are not moves.
//!
//! The vocabulary is `SlotDirectory`'s (slot, held, free); a record's life
//! spans the device's steps, so it is a directory of its own.
//!
//! It is the one owner of every expert's location: a bank record, a record
//! of its group's host tier ([`HostRecords`]: copies in the bank's own
//! form), or only the model file. Every miss reaches the tier first: the
//! host lands a file miss into a ready tier record ([`Directory::claim_landing`]),
//! the device's steps serve the tier's misses from it, and the host promotes
//! the ones the ranking keeps into the bank in the background, on the copy
//! engine (`watcher::promotions`). The tier and the bank hold disjoint sets:
//! a promotion copies an expert into the bank and the tier surrenders its copy
//! (`HostRecords::surrender`), ready for another expert once no step may still
//! read it; an expert leaving the bank exits by `Directory::exit`, written
//! back into the tier where its expected uses pay for the copy. A
//! prompt group's seeds are the one placement the device copies: they hold
//! their records once the device committed them, which the host learns
//! once the device is past them ([`Directory::retire`]).

#[cfg(any(test, feature = "expert-routing"))]
mod replay;
mod stack;

use super::host_records::{admitted, Admit, Claim, Demand, Grant, HostRecords, InboundTable, Kept, Located, Resolution, Taking};
use super::map::{LandingMap, StampedMap};
use super::records::{Back, BankRecords, Record};
use super::horizon::{Horizon, Reader, Tier, Time};
use super::reuse::{Rank, Ranker, Ranking, Reuse, Unlogged};
use super::staging::Staged;
use super::watcher::{ParkFrom, ScratchGroup, Step, Steps, Unbegun, UnmapStep};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
pub use stack::{CountedLaunch, LayerCurve, MissCurve, RoutedRead, Stack};
pub(crate) use stack::Ranks;
#[cfg(any(test, feature = "expert-routing"))]
pub use replay::{Recency, replay};
#[cfg(any(test, feature = "expert-routing"))]
use super::exchange;

/// A piece of the device's order: a run of one group's streamed layers
/// that the device runs whole, between the pieces submitted before and
/// after it. A group begins as one piece, its own. A group cut in before
/// a begun group's layer (`Steps::begin_cut_in`) runs between that
/// group's layers before it and those from it on, which become a piece of
/// their own, ordered after the group cut in: the piece that group's begin
/// opened `behind` it. Pieces order by the group that opened them, then
/// behind it, which is the order they are submitted in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Piece {
    pub opened: Step,
    pub behind: bool,
}

impl Piece {
    /// The piece group `group`'s begin opens: its own.
    pub fn of(group: Step) -> Self {
        Self { opened: group, behind: false }
    }
}

/// A point in the device's order of streamed layers. Steps begin in groups
/// (a chain's launches, a lone step a group of one), named by their first
/// sequence; the device runs pieces ([`Piece`]) in submission order, each
/// layer-major, each layer over its steps in sequence order. So a request
/// at layer `l` of step `q` in a piece (a scratch step's round 0 there)
/// proves the device past layer `l` of the piece's steps before `q`, past
/// every earlier layer of the piece, and past every earlier piece: the
/// order is (piece, layer, sequence). A group run whole is one piece,
/// whose order is (group, layer, sequence). Its group names it for
/// lookups; no two pieces share a (layer, sequence), so the group decides
/// no order. A point the device has not reached may move later as its
/// group is cut (`Steps::begin_cut_in`): one about a resolution yet to
/// come is read from the steps when compared (`Steps::after`,
/// `Steps::point`), never kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Point {
    pub piece: Piece,
    pub layer: usize,
    pub sequence: Step,
    pub group: Step,
}

impl Point {
    /// Before every point: what the device is known past before any
    /// request, and where a group the steps forgot lies (every step of it
    /// ended, so the device runs none of it).
    pub const START: Self = Self::whole(0, 0, 0);

    /// Layer `layer` of step `sequence` of group `group`, run whole: one
    /// piece, its own.
    pub const fn whole(group: Step, layer: usize, sequence: Step) -> Self {
        Self { piece: Piece { opened: group, behind: false }, layer, sequence, group }
    }
}

/// A park read from the steps where it is served (`Steps::park`): the
/// device parked at `point` (a decode request's landing launch), and per layer
/// what its order says of every step's resolution there. Read once a
/// serve, at use: a group cut in after it only moves resolutions later,
/// so what it says stays true.
#[derive(Clone, Debug)]
pub(super) struct Park {
    pub point: Point,
    /// Per layer, the first step of the parked step's group or a later
    /// one whose resolution of the layer lies after the park (the parked
    /// step's own at its layer is done): every step of those groups below
    /// it has resolved the layer.
    pub unresolved: Vec<ParkFrom>,
    /// Per layer, the steps of groups begun before the parked step's that
    /// have yet to resolve the layer, `(first, last)`: a group the parked
    /// one was cut in before, at that group's layers from the cut on.
    /// None while nothing cuts in.
    pub earlier: Vec<Option<(Step, Step)>>,
}

impl Park {
    /// The first step whose resolution of `layer` is not before the park,
    /// of the parked step's group or a later one ([`Self::unresolved`]):
    /// what an unmapping or a landing at the park is stored from.
    pub fn unresolved(&self, layer: usize) -> ParkFrom {
        self.unresolved[layer]
    }

    /// Whether an entry of `layer` mapped for steps `[from, until)` leaves
    /// no step of an earlier group able to resolve it once the park
    /// unmaps it: no step of a group the parked one was cut in before that
    /// has yet to resolve the layer lies in it. Only such an entry may be
    /// unmapped or its record reused at the park: that group's request at
    /// the layer may have been served already, reading in place what the
    /// map resolves for it.
    pub fn frees(&self, layer: usize, from: Step, until: Step) -> bool {
        self.earlier[layer].is_none_or(|(first, last)| until <= first || from > last)
    }

    /// Whether the parked step was cut in before a group it runs inside:
    /// it may land into the records set aside for cut-ins.
    pub fn cut_in(&self) -> bool {
        self.earlier.iter().any(Option::is_some)
    }
}

/// How a seed's step and the earlier steps of its group, whose scratch
/// its seed copy reads, ended, as the seed retires by
/// ([`Directory::retire`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// One runs still: whether all end whole is unknown.
    Running,
    /// All ended, no wait of theirs ending on its still pulse.
    Ended,
    /// All ended, a wait of one ending on its still pulse before its
    /// answer.
    Tainted,
}

/// The retirement of the seeds a pass settles, whichever store holds them
/// ([`Directory::retire`]): each seed placed before `progress` ends with
/// its step, read once a step (`ending`), and is committed where its step
/// ended whole with the device's commit holding the step's low word.
pub(super) struct Settling<E: FnMut(Point) -> Ending> {
    progress: Point,
    endings: HashMap<u64, Ending>,
    ending: E,
}

impl<E: FnMut(Point) -> Ending> Settling<E> {
    pub(super) fn new(progress: Point, ending: E) -> Self {
        Self { progress, endings: HashMap::new(), ending }
    }

    /// Retire the seeds of `pending` that have ended, in order: those
    /// still running stay in it; each retired with whether it committed.
    /// `point` is a seed's placing point, `commit` the device's commit of
    /// its record.
    pub(super) fn retire<T>(&mut self, pending: &mut Vec<T>, point: impl Fn(&T) -> Point, commit: impl Fn(&T) -> i32) -> Vec<(T, bool)> {
        let mut retired = Vec::new();
        for seed in std::mem::take(pending) {
            match self.outcome(point(&seed), || commit(&seed)) {
                None => pending.push(seed),
                Some(committed) => retired.push((seed, committed)),
            }
        }
        retired
    }

    /// Whether the seed placed at `point` has not ended (None), else
    /// whether it committed: `commit` is the device's commit of its record.
    fn outcome(&mut self, point: Point, commit: impl FnOnce() -> i32) -> Option<bool> {
        let ending = (point < self.progress).then(|| *self.endings.entry(point.sequence).or_insert_with(|| (self.ending)(point)));
        match ending? {
            Ending::Running => None,
            ending => Some(ending == Ending::Ended && commit() == crate::experts::watcher::low_word(point.sequence)),
        }
    }
}

/// The one seed policy ([`Directory::seed`]), over either store's records
/// (design §3.7): `seeds`, the routed experts a layer lacks, each at its
/// place in the store's taking order, take `records` (those a seed may
/// take, each at its place, `None` for a free one): free records first,
/// then the group's kept records of any layer, least ranked first, the
/// most ranked seed against the least ranked record while `strictly_better`
/// holds (strictly higher expected future uses, never evicting on a recency
/// tie-breaker alone). Capacity is pooled across the group's layers, so no
/// layer has a room of its own. Each seed taken, with its record and the
/// record's place.
fn seeding<P: Ord + Copy, R: Ord + Copy>(
    mut seeds: Vec<P>,
    mut records: Vec<(Option<P>, R)>,
    strictly_better: impl Fn(P, P) -> bool,
) -> Vec<(P, (Option<P>, R))> {
    seeds.sort_unstable_by(|a, b| b.cmp(a));
    records.sort_unstable();
    seeds
        .into_iter()
        .zip(records)
        .take_while(|&(seed, (held, _))| held.is_none_or(|held| strictly_better(held, seed)))
        .collect()
}

/// What a placement's experts took: per expert its record; and the experts
/// evicted from those records, (layer, expert, the step that last read
/// it), whose map entries the host unmaps before the device copies into
/// them.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Placed {
    pub records: Vec<BankRecord>,
    pub evicted: Vec<(usize, usize, Step)>,
}

/// A promotion the ranking asks ([`Directory::promotions`]): expert
/// `expert` of `layer`, which its group's host tier keeps, into free
/// record `record` of the group, which no step reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Promotion {
    pub group: usize,
    pub record: BankRecord,
    pub layer: usize,
    pub expert: usize,
}

/// A record of group `group` released for the reserve
/// ([`Directory::promotions`]): `record`, whose expert `expert` of `layer`
/// left the bank, to be unmapped from the first step not yet begun and
/// free once every step below it has ended ([`Directory::drain`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Release {
    pub group: usize,
    pub record: BankRecord,
    pub layer: usize,
    pub expert: usize,
}

/// What the ranking wants at once ([`Directory::promotions`]): the
/// promotions into free records, and the records released for the reserve.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Wants {
    pub promotions: Vec<Promotion>,
    pub releases: Vec<Release>,
    /// Host-tier `(group, tier_record)` pairs surrendered during this pass,
    /// whose source generations must be bumped in `ExpertTransport`.
    pub surrendered: Vec<(usize, usize)>,
}

/// Outcome of claiming a unified VRAM slot for an on-demand decode DMA landing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum DemandSlotClaim {
    /// Expert is already permanently held in VRAM at `slot` (zero DMA needed).
    AlreadyHeld { staged: Staged, slot: usize },
    /// Claimed a `Record::Free` usable slot and transitioned it to
    /// `Record::DemandLanding(staged)`.
    Allocated { staged: Option<Staged>, slot: usize },
    /// No free usable unpinned slot available right now; caller falls back to
    /// bounded overflow (`-2 - tier_record` SM zero-copy) without blocking.
    Overflow { staged: Option<Staged> },
}

pub(super) struct Directory {
    /// The bank's records: what each holds, and the maps naming them.
    bank: BankRecords,
    /// Per group, each member layer's map word of its expert 0 and the
    /// layer (`ExpertBankPlan::home`), in map order.
    homes: Vec<Vec<(usize, usize)>>,
    /// Each group's host tier records, where the bank has a tier.
    host: Vec<HostRecords>,
    /// Their read path's mirror of where every expert is.
    inbound: Arc<InboundTable>,
    /// Live per-layer `POLICY_WHOLE` flags (`Telemetry::whole_flags`): while
    /// a layer's flag is set, active steps read its experts at `home + expert`
    /// without consulting the map, so no mid-flight promotion, seed, refill,
    /// demand claim, or shrink may evict its records.
    whole_layers: Arc<[AtomicBool]>,
    /// The running prompt group's reads the ranking has not logged yet:
    /// every expert its steps routed, each at its last step
    /// ([`Self::read_pending`]), which every taking ranks beside, a use at
    /// that step, so the store's landings and seeds follow the prompt's use
    /// of each expert as its passes resolve.
    prompt_reads: HashMap<Staged, Unlogged>,
    /// Who places into the bank: where the host does (the one store), a
    /// prompt group's scratch and seeds are its host records'.
    placer: crate::planning::Placer,
    /// The one ranking, which every choice of what to keep ranks by; its
    /// feed is the stack's (`Stack`). Its pools are each group's decode
    /// capacity ([`Self::capacity`]) less the promotions' reserve.
    ranking: Ranking,
    /// Per layer, what the bank is short of and holds beyond the ranking's
    /// pool (`Reuse::in_top`), kept as either side changes
    /// ([`Self::resettle`]): the experts of the pool the bank lacks and no
    /// placement takes (`wanted`), and those the bank holds outside it
    /// (`unwanted`), the records promotions take.
    wanted: Vec<BTreeSet<usize>>,
    unwanted: Vec<BTreeSet<usize>>,
    /// Whether a promotion may have become possible since promotions were
    /// last taken ([`Self::promotions`]): a want or a victim gained or lost,
    /// a landing into the tier, or a copy's price changed (a record freed or
    /// taken is the bank records' to mark, [`BankRecords::take_changed`]).
    stirred: bool,
    /// The ranking's reads ([`Reuse::reads`]) when promotions last withheld
    /// a want for its copy (`Self::earning`), none where they withheld
    /// none: a read since may earn it, and asks again.
    withheld: Option<u64>,
    /// Per group, the records it keeps free or releasing beyond its wants,
    /// so a promotion takes a record no step reads and copies at once
    /// ([`Self::reserve`]); the ranking's pools are the capacity less it.
    reserve: Vec<usize>,
    /// The reads a promotion must expect beyond what it displaces to earn
    /// its copy, each layer's (`ExpertLines::copy_hits`): none where a hit
    /// saves nothing, so no promotion of the layer earns; zero, a copy
    /// costing nothing, until the lines are priced.
    copy_hits: Vec<Option<f64>>,
    /// The promotions asked since the load ([`Self::asked`]).
    asked: u64,
    /// The load's time: the token clock, and the turnover of each group's
    /// tiers (tier 0 the bank beside a host tier, or the one store; tier 1
    /// the host tier beside a bank), each of its records present
    /// ([`Self::tier_records`]). Built here, shared with the ranking, the
    /// ledger and the steps' clock writers ([`Self::time`]); fed every move
    /// here and only here.
    time: Time,
    /// Each layer's price of writing a released expert back into its host
    /// tier ([`WriteBack`]), none until the passes and the copy line are
    /// measured: no write-back is asked before.
    write_backs: Vec<Option<WriteBack>>,
    /// Each release's write-back in flight, by its bank record (group,
    /// record): the grant of its tier record's landing, which maps it once
    /// written, and the step that last read its expert, which ranks it in
    /// the tier.
    backs: HashMap<(usize, usize), (Grant, Option<Step>)>,
    /// Where the host places (the one store), each prompt group's seed not
    /// yet retired: its group, the host record it took, its expert and the
    /// point of its seeding step ([`Self::seed`], [`Self::retire`]).
    seeding: Vec<(usize, u32, Staged, Point)>,
}

/// What writing a released expert back into the host tier costs and saves,
/// in the decode's terms (`magnitude_cost_model::Terms`, the passes' mean
/// busy seconds a step on each resource, `watcher::passes::Passes::terms`):
/// its copy holds the link `link` seconds once, and each use of it the tier
/// then serves saves the drive one read, `drive` seconds; a use served from
/// the tier costs what one served after its drive landing would, so only
/// the read differs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WriteBack {
    pub terms: magnitude_cost_model::Terms,
    pub drive: f64,
    pub link: f64,
}

impl WriteBack {
    /// Whether a write-back of an expert expected `uses` times over a stay
    /// of `stay` steps in the tier shortens the step: with its copy's link
    /// time and the drive reads it saves spread over the stay, the
    /// slowest term falls, `T(terms + Δ) < T(terms)` (`Terms::time`). It
    /// pays only where the drive binds and the link stays below it: where
    /// the link or another term binds, the copy shortens nothing.
    pub fn pays(&self, uses: f64, stay: f64) -> bool {
        assert!(stay > 0. && uses >= 0., "a stay of {stay} steps, {uses} uses");
        let mut after = self.terms;
        after.link += self.link / stay;
        after.drive = (after.drive - uses * self.drive / stay).max(0.);
        after.time() < self.terms.time()
    }
}

/// Where a banked expert goes as it leaves the bank, as its one exit
/// decides it ([`Exits::leaving`]): back into its host tier, ranked
/// beside its last read (`Some`), where its write-back pays; else
/// (`None`) to the copy its tier has yet to give up, or to the file.
/// Only the decision makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Leaving {
    staged: Staged,
    back: Option<Option<Step>>,
}

impl Leaving {
    /// Whether its write-back pays.
    fn backs(&self) -> bool {
        self.back.is_some()
    }
}

/// A banked expert's exit as its record carries it out: the host tier
/// record its write-back's landing took, none where it writes nothing back
/// (its write-back does not pay, or the tier has no record ready for it).
/// Only [`Directory::exit`] makes one, so every transition of a record
/// that holds an expert (`BankRecords`, which takes one) is the one exit's.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Exit {
    staged: Staged,
    back: Option<HostRecord>,
}

impl Exit {
    pub(super) fn staged(&self) -> Staged {
        self.staged
    }

    pub(super) fn back(&self) -> Option<HostRecord> {
        self.back
    }

    /// An exit that writes nothing back, where a test plays the bank.
    #[cfg(test)]
    pub(super) fn to_file(staged: Staged) -> Self {
        Self { staged, back: None }
    }

    /// An exit written back into tier record `tier`, where a test plays
    /// the bank.
    #[cfg(test)]
    pub(super) fn written_back(staged: Staged, tier: HostRecord) -> Self {
        Self { staged, back: Some(tier) }
    }
}

/// The one exit of a bank group's experts, by one hold of the ranking
/// ([`Directory::exits`]): what every expert leaving it shares, the host
/// tier's stay and what a write-back's landing would take, taken once.
struct Exits<'a> {
    directory: &'a Directory,
    reuse: &'a Reuse,
    group: usize,
    /// The host tier's move τ (`Reader::Move`: a write-back is a move) in
    /// decode steps, τ_1 / ā: what a write-back's copy is spread over. None
    /// before the first decode step, or without a host tier of records.
    stay: Option<f64>,
    /// What a write-back's landing would take with the reserve's keep, at
    /// τ_1 ([`HostRecords::displaced`]).
    displaced: std::cell::OnceCell<Option<Option<(Kept, Rank)>>>,
}

impl Exits<'_> {
    /// Whether `staged`, leaving the bank by any path (a release, a seed,
    /// a lease or a shrink), is written back into its host tier. It is
    /// where the placer keeps a host tier beside the bank, the tier lacks
    /// it (no copy it has yet to give up, no landing of it in flight), its
    /// write-back is priced ([`WriteBack`]) and its expected uses within
    /// τ_1 (`Reuse::expected`, beside its reads) make the copy pay over the
    /// stay ([`WriteBack::pays`]), and its landing's claim, with the
    /// reserve as its keep, admits it: the same victim test at τ_1 as every
    /// landing's ([`outranking`], [`admitted`]), which [`Directory::exit`]
    /// then claims by. Before the passes and the copy line are measured,
    /// and before the first decode step, no write-back is priced, so none
    /// pays.
    fn leaving(&self, staged @ (layer, expert): Staged) -> Leaving {
        let Self { directory, reuse, group, .. } = *self;
        let back = (|| {
            let price = directory.write_backs[layer]?;
            if directory.placer != crate::planning::Placer::Tier || directory.host[group].located(staged) != Located::Lacked {
                return None;
            }
            let stay = self.stay?;
            let ranker = reuse.ranker(group, Tier::HOST, Reader::Move);
            let read = beside(&directory.prompt_reads, staged, reuse.last(layer, expert));
            if !price.pays(ranker.expected(layer, expert, read).uses(), stay) {
                return None;
            }
            let admit = outranking(ranker, &directory.prompt_reads, ranker.rank(layer, expert, read));
            let host = &directory.host[group];
            let displaced = *self.displaced.get_or_init(|| host.displaced(|staged, read| admit.rank(staged, read), host.reserve()));
            admitted(displaced, |kept, victim| admit.admits(kept, victim)).then(|| read.map(|read| read.step))
        })();
        Leaving { staged, back }
    }
}

/// A record of a bank group's record tensors: what its bank map names, and
/// what a prompt group's pass reads its hits in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BankRecord(pub u32);

/// Bank record `record` of a group's records.
pub(super) fn bank_record(record: usize) -> BankRecord {
    BankRecord(u32::try_from(record).expect("bank records fit u32"))
}

impl BankRecord {
    /// Its index among its group's records.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A record of a group's host records: the one store's, or a host tier's
/// beside a discrete device. A bank record only where the host places
/// ([`Directory::bank_record`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostRecord(pub u32);

/// The later of `read`, a read of `staged` at a step the log may not have
/// brought yet (a landing's), and the running prompt group's read of it
/// ([`Directory::read_pending`]): what every taking ranks it beside.
fn beside(prompt_reads: &HashMap<Staged, Unlogged>, staged: Staged, read: Option<Step>) -> Option<Unlogged> {
    read.map(Unlogged::at).max(prompt_reads.get(&staged).copied())
}

/// How the host records rank what they keep, by a lock-free `ranker`
/// snapshotted once for the group at its tier's τ for `reader`: an expert's
/// whole rank beside its reads ([`beside`]), so experts the reads tie rank
/// by their last read, never by their indices.
fn store_rank<'a>(ranker: Ranker<'a>, prompt_reads: &'a HashMap<Staged, Unlogged>) -> impl Fn(Staged, Option<Step>) -> Rank + Copy + 'a {
    move |staged @ (layer, expert): Staged, read: Option<Step>| ranker.rank(layer, expert, beside(prompt_reads, staged, read))
}

/// How kept records are ranked during prompt seeding: uses the same
/// prompt-aware `Expected` scale as [`Reuse::rank_union`] (no future-lead
/// `p = 1.0` override from an in-flight prompt's `prompt_reads`), while
/// preserving `last` from `prompt_reads` and `read` so never-routed records
/// order before routed ones.
fn seed_rank<'a>(
    reuse: &'a Reuse,
    ranker: Ranker<'a>,
    prompt_reads: &'a HashMap<Staged, Unlogged>,
    seed_layer: usize,
    routed: &'a [(usize, u32, Step)],
    routed_total: f64,
    group_first: Step,
) -> impl Fn(Staged, Option<Step>) -> Rank + Copy + 'a {
    move |staged @ (owner, expert): Staged, read: Option<Step>| {
        let last = reuse.last(owner, expert).max(beside(prompt_reads, staged, read).map(|u| u.step));
        reuse.seed_held_rank(ranker, seed_layer, routed, routed_total, group_first, owner, expert, last)
    }
}

/// The victim test of a landing into `tier` whose claimant ranks
/// `candidate` there at its move τ (design §3.7): the taking order's rank
/// ([`store_rank`], `Reader::Move`), and the admission of the victim it
/// names: a banked copy, whose unmapping costs no move (the bank serves its
/// uses), or a kept expert the candidate outranks.
fn outranking<'a>(ranker: Ranker<'a>, prompt_reads: &'a HashMap<Staged, Unlogged>, candidate: Rank) -> impl Admit<Key = Rank> + 'a {
    (store_rank(ranker, prompt_reads), move |kept: Kept, victim: Rank| kept == Kept::Banked || victim < candidate)
}

/// A group's wanted experts as their copies price them
/// ([`Directory::promotions`]).
struct Earning {
    earning: Vec<Staged>,
    /// Whether any waits for its reads to earn its copy.
    withheld: bool,
}

/// What a group's promotions may displace ([`Directory::victims`]).
struct Victims {
    /// The records releases take, the least ranked first.
    released: Vec<(Ranked, usize)>,
    /// What a copy displaces over its stay, the least weighed first.
    floor: Vec<f64>,
}

/// An expert's rank, least first, which every choice of what the bank
/// keeps compares: the ranking's own ([`Rank`], beside its reads,
/// [`beside`]), then the (layer, expert) (which decides only between
/// experts the log never read).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Ranked {
    rank: Rank,
    layer: usize,
    expert: usize,
}

/// The most records a group of `records` records and `members` member
/// layers reserves: all but one a member layer, where it has one each, so
/// its pool keeps a record for each layer's read.
pub(super) fn most_reserved(records: usize, members: usize) -> usize {
    records.saturating_sub(members)
}

/// What a bank's host tier starts from: each group's records, every one
/// ready, and each group's bottom records the conveyor's lending reserves
/// (`planning::Lender`: its lender's, none of another group), beside who
/// places the bank's experts and the maps their placements write
/// (`placement`).
pub(in crate::experts) struct HostTier {
    pub records: Vec<u32>,
    pub lent: Vec<u32>,
    /// The records each group's reserve keeps holding no expert mapped
    /// (`ExpertBankPlan::tier_reserve`).
    pub reserve: usize,
    pub placement: Placement,
}

/// Who places the bank's experts, which decides how the steps resolving
/// their maps see an unmapping, and the maps their placements write.
pub(in crate::experts) enum Placement {
    /// The device places from a host tier: the bank's records are each
    /// layer's stamped map's one writer (`bank_maps`), each layer's host
    /// records its plain tier map's (`tier_maps`), the lender's bank
    /// records `lent`, `(group, records)`, are lent to prompt groups, and
    /// `stage_slots` reserves a dedicated even/odd double-buffered DMA staging
    /// ring per group (`lent..lent + stage_slots`), independent of `lent`.
    Tier {
        bank_maps: Vec<StampedMap>,
        tier_maps: Vec<LandingMap>,
        lent: (usize, usize),
        stage_slots: Vec<usize>,
    },
    /// The host places in the bank's own records, the one store: each
    /// layer's host records write its stamped landing map.
    Bank { landing_maps: Vec<LandingMap> },
    /// No device: an offline replay of the bank's ranking, or a test of its
    /// own placements, placing as beside a host tier and writing no map.
    Replay,
}

impl Placement {
    fn placer(&self) -> crate::planning::Placer {
        match self {
            Self::Tier { .. } | Self::Replay => crate::planning::Placer::Tier,
            Self::Bank { .. } => crate::planning::Placer::Bank,
        }
    }
}

impl HostTier {
    /// `records` host records for each group, lending none, of a directory
    /// of no device (`Placement::Replay`), whose reads come one step at a
    /// time (`FlightDepth::SERIAL`).
    pub fn replay(records: Vec<u32>) -> Self {
        Self { lent: vec![0; records.len()], records, reserve: 0, placement: Placement::Replay }
    }

    /// It, each group's reserve `reserve` records.
    #[cfg(test)]
    pub fn reserving(self, reserve: usize) -> Self {
        Self { reserve, ..self }
    }
}

impl Directory {
    /// An empty directory: groups of `records` each, and `layers` in their
    /// groups, of `experts` experts each, beside each group's empty host
    /// tier, ranking by reuse as its reads measure it, placing as `host`'s
    /// placement says.
    pub fn new(records: &[usize], layers: &[usize], experts: &[usize], host: HostTier) -> Self {
        assert_eq!(host.records.len(), records.len(), "each group has its host tier");
        assert_eq!(host.lent.len(), records.len(), "each group's host tier lends");
        // The plan's homes (`layer_homes`), which the device reads too.
        let mut homes = vec![Vec::<(usize, usize)>::new(); records.len()];
        let placed = crate::planning::layer_homes(layers.iter().zip(experts).map(|(&group, &experts)| (group, experts as u64)));
        for (layer, (&group, home)) in layers.iter().zip(placed).enumerate() {
            homes[group].push((home as usize, layer));
        }
        let placer = host.placement.placer();
        let HostTier { records: host_records, lent: host_lent, reserve: host_reserve, placement } = host;
        // The maps the host records write, one per layer, where the
        // placement writes any.
        let (bank_maps, maps) = match placement {
            Placement::Tier { bank_maps, tier_maps, lent, stage_slots } => {
                (Some((bank_maps, lent, stage_slots)), Some(tier_maps))
            }
            Placement::Bank { landing_maps } => (None, Some(landing_maps)),
            Placement::Replay => (None, None),
        };
        if let Some(maps) = &maps {
            assert_eq!(maps.len(), layers.len(), "each layer has its map");
        }
        // Each group's members, each layer with its map.
        let mut members = (0..records.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut maps = maps.map(Vec::into_iter);
        for (layer, (&group, &experts)) in layers.iter().zip(experts).enumerate() {
            members[group].push((layer, experts, maps.as_mut().map(|maps| maps.next().expect("each layer has its map"))));
        }
        let inbound = Arc::new(InboundTable::new(experts));
        let resolution = match placer {
            crate::planning::Placer::Tier => Resolution::AfterStore,
            crate::planning::Placer::Bank => Resolution::BeforeStamp,
        };
        let host = host_records
            .iter()
            .zip(members)
            .zip(host_lent)
            .map(|((&count, members), lent)| HostRecords::new(count, layers.len(), members, resolution, (lent, host_reserve), inbound.clone()))
            .collect::<Vec<_>>();
        let bank = BankRecords::new(records, &layers.iter().copied().zip(experts.iter().copied()).collect::<Vec<_>>(), bank_maps);
        // The ranking's pools: each group's decode capacity
        // ([`Self::capacity`]).
        let kept: Vec<usize> = match placer {
            crate::planning::Placer::Bank => host.iter().map(HostRecords::capacity).collect(),
            crate::planning::Placer::Tier => (0..records.len()).map(|group| bank.capacity(group)).collect(),
        };
        let time = {
            let directory = (&bank, &host[..], placer);
            let tiers = (0..records.len()).map(|group| Self::tiers_of(placer).iter().map(|&tier| Self::records_of(directory, group, tier)).collect::<Vec<_>>()).collect::<Vec<_>>();
            let behind = crate::planning::FlightDepth::PIPELINED.steps(super::DEPTHS as u64) as usize + super::DEPTHS;
            Time::new(&tiers, behind)
        };
        Self {
            bank,
            homes,
            placer,
            host,
            inbound,
            whole_layers: Arc::from([]),
            prompt_reads: HashMap::new(),
            ranking: Reuse::ranking(experts, layers, &kept, time.clone()),
            wanted: vec![BTreeSet::new(); layers.len()],
            unwanted: vec![BTreeSet::new(); layers.len()],
            stirred: false,
            withheld: None,
            reserve: vec![0; records.len()],
            copy_hits: vec![None; layers.len()],
            asked: 0,
            time,
            write_backs: vec![None; layers.len()],
            backs: HashMap::new(),
            seeding: Vec::new(),
        }
    }

    /// The tiers each group's records form under `placer`, in [`Time`]'s
    /// order: the one store alone (tier 0), or the bank (tier 0) beside its
    /// host tier (tier 1).
    fn tiers_of(placer: crate::planning::Placer) -> &'static [Tier] {
        match placer {
            crate::planning::Placer::Bank => &[Tier::COMPUTE],
            crate::planning::Placer::Tier => &[Tier::COMPUTE, Tier::HOST],
        }
    }

    /// The tier the host records are under `placer`: the one store, tier 0,
    /// or the host tier beside the bank.
    fn host_tier(&self) -> Tier {
        match self.placer {
            crate::planning::Placer::Bank => Tier::COMPUTE,
            crate::planning::Placer::Tier => Tier::HOST,
        }
    }

    /// The records group `group`'s `tier` holds storage for, its turnover's
    /// K (design §3.2): a resize changes it, and nothing else does (a lend
    /// or the promotions' reserve moves no record).
    fn records_of((bank, host, placer): (&BankRecords, &[HostRecords], crate::planning::Placer), group: usize, tier: Tier) -> u64 {
        match (placer, tier) {
            (crate::planning::Placer::Tier, Tier::COMPUTE) => bank.counts()[group] as u64,
            (crate::planning::Placer::Bank, Tier::COMPUTE) | (crate::planning::Placer::Tier, Tier::HOST) => u64::from(host[group].records()),
            (crate::planning::Placer::Bank, Tier::HOST) => unreachable!("the one store has no host tier"),
            (_, other) => unreachable!("directory has no storage for tier {other:?}"),
        }
    }

    /// Group `group`'s `tier` was resized: its turnover closes its epoch and
    /// restarts at the records it holds now (design §3.2).
    fn resized(&mut self, group: usize, tier: Tier) {
        let records = Self::records_of((&self.bank, &self.host, self.placer), group, tier);
        self.time.lock().resized(group, tier, records);
    }

    /// The load's time ([`Time`]), for the ranking's, the ledger's and the
    /// clock writers' share of it.
    pub(crate) fn time(&self) -> &Time {
        &self.time
    }

    /// Place `layer`'s `expert` in the wanted and unwanted sets by whether
    /// the ranking's pool holds it (`top`), the bank holds it, and a
    /// placement takes a record for it.
    fn settle(&mut self, layer: usize, expert: usize, top: bool) {
        let held = self.bank.held(layer, expert).is_some();
        let placing = self.bank.placing((layer, expert));
        // Whether `set` holds `expert` exactly when `member`: whether it
        // gained it.
        let keep = |set: &mut BTreeSet<usize>, member: bool| match member {
            true => set.insert(expert),
            false => {
                set.remove(&expert);
                false
            }
        };
        let wanted = keep(&mut self.wanted[layer], top && !held && !placing);
        let unwanted = keep(&mut self.unwanted[layer], held && !top);
        self.stirred |= wanted | unwanted;
    }

    /// [`Self::settle`] `layer`'s `expert` after the bank's side of it
    /// changed, by the ranking's pool as it is now.
    fn resettle(&mut self, layer: usize, expert: usize) {
        let top = Reuse::of(&self.ranking).in_top(layer, expert);
        self.settle(layer, expert, top);
    }

    /// The one ranking, for its feed (`Stack`).
    pub fn ranking(&self) -> Ranking {
        self.ranking.clone()
    }

    /// The ranking as it is now.
    pub(in crate::experts) fn reuse_now(&self) -> std::sync::MutexGuard<'_, Reuse> {
        Reuse::of(&self.ranking)
    }

    /// Decode's capacity in group `group`, the one count of the records
    /// that keep the bank's experts: where the host places (the one
    /// store), its host records' ([`HostRecords::capacity`]: present less
    /// the lease and the reserve); beside a host tier, the bank's less its
    /// lent records, the scratch's ([`BankRecords::capacity`]).
    pub fn capacity(&self, group: usize) -> usize {
        match self.placer {
            crate::planning::Placer::Bank => self.host[group].capacity(),
            crate::planning::Placer::Tier => self.bank.capacity(group),
        }
    }

    /// Each group's decode capacity ([`Self::capacity`]), in group order:
    /// what the load publishes as the records its experts are held in.
    pub fn capacities(&self) -> Vec<usize> {
        (0..self.homes.len()).map(|group| self.capacity(group)).collect()
    }

    /// Group `group`'s pool is its capacity as it is now ([`Self::capacity`])
    /// less the promotions' reserve, that within the most a group reserves
    /// ([`most_reserved`]): taken again as a store grows or shrinks, a
    /// prompt group's lease is granted and returned, or the reserve
    /// changes, the only events that change it.
    fn resize_pool(&mut self, group: usize) {
        let capacity = self.capacity(group);
        let members = (0..self.bank.layers()).filter(|&layer| self.bank.group(layer) == group).count();
        self.reserve[group] = self.reserve[group].min(most_reserved(capacity, members));
        Reuse::of(&self.ranking).resize(group, capacity - self.reserve[group]);
        // The group's victims were prepared for the pool before.
        self.host[group].unprepare();
    }

    /// Group `group` keeps `records` of its records free or releasing from
    /// now, beyond what its wants take: the ranking's pool is its capacity
    /// less them. Only where promotions place from a host tier.
    pub fn reserve(&mut self, group: usize, records: usize) {
        assert!(self.placer == crate::planning::Placer::Tier, "the one store promotes nothing");
        let members = self.layer_groups().iter().filter(|&&of| of == group).count();
        let capacity = self.capacity(group);
        let most = most_reserved(capacity, members);
        assert!(records <= most, "a reserve of {records} of group {group}'s capacity of {capacity}, beyond {most}");
        if std::mem::replace(&mut self.reserve[group], records) != records {
            self.resize_pool(group);
            self.stirred = true;
        }
    }

    /// Group `group`'s reserve ([`Self::reserve`]).
    pub fn reserved(&self, group: usize) -> usize {
        self.reserve[group]
    }

    /// Each layer's experts.
    pub fn layer_experts(&self) -> Vec<usize> {
        (0..self.bank.layers()).map(|layer| self.bank.experts(layer)).collect()
    }

    /// Each layer's group.
    pub fn layer_groups(&self) -> Vec<usize> {
        (0..self.bank.layers()).map(|layer| self.bank.group(layer)).collect()
    }

    /// The read path's mirror of where every expert is
    /// ([`InboundTable`]), which a posted read claims beside residency.
    pub fn inbound(&self) -> Arc<InboundTable> {
        self.inbound.clone()
    }

    /// Group `group`'s host records.
    pub fn host(&self, group: usize) -> &HostRecords {
        &self.host[group]
    }

    /// Every group's host records, in group order.
    pub fn hosts(&self) -> &[HostRecords] {
        &self.host
    }

    /// The host records of `layer`'s group.
    fn tier(&mut self, layer: usize) -> &mut HostRecords {
        let group = self.bank.group(layer);
        &mut self.host[group]
    }

    /// Claim a landing of `layer`'s `expert` in its group's host tier,
    /// leaving `keep` of the reserve's records or the experts still lacked
    /// where fewer, the record taken as `taking` says, where `admit`'s
    /// victim test admits it ([`HostRecords::claim`]). Every landing of an
    /// expert, whoever lands it, claims it here, so no place test differs
    /// from its claim. Short without a tier.
    pub fn claim_landing(&mut self, layer: usize, expert: usize, keep: usize, taking: Taking, admit: impl Admit) -> Claim {
        self.tier(layer).claim((layer, expert), keep, taking, admit)
    }

    /// Claim a read-ahead landing of `layer`'s `expert` named for `step`
    /// with the ledger's probability `p` (design §3.7): [`Self::claim_landing`]
    /// with the candidate's rank from `Reuse::expected_with` (replacing
    /// `step`'s lead term by `p`, never mixing) tested against the victim
    /// at the host records' move τ ([`outranking`]).
    pub fn claim_ahead(&mut self, layer: usize, expert: usize, keep: usize, taking: Taking, step: Step, p: f64) -> Claim {
        let (group, tier) = (self.bank.group(layer), self.host_tier());
        let reuse = Reuse::of(&self.ranking);
        let candidate = reuse.rank_of(layer, expert, reuse.expected_with(layer, expert, tier, step, p));
        let ranker = reuse.ranker(group, tier, Reader::Move);
        self.host[group].claim((layer, expert), keep, taking, outranking(ranker, &self.prompt_reads, candidate))
    }

    /// Claim a carryover prefetch landing of `layer`'s `expert` in its group's
    /// host tier, leaving `keep` reserve records, admitted iff its priced net
    /// stall gain `net_benefit_uses` (`(p * s_demand - (1 - p) * w_net) / s_demand`)
    /// is positive and strictly exceeds any victim eviction loss
    /// `max(0, E[victim; τ] - E[candidate; τ])`.
    pub fn claim_carryover(&mut self, layer: usize, expert: usize, keep: usize, taking: Taking, net_benefit_uses: f64) -> Claim {
        if !(net_benefit_uses.is_finite() && net_benefit_uses > 0.0) {
            return Claim::Short;
        }
        let (group, tier) = (self.bank.group(layer), self.host_tier());
        let reuse = Reuse::of(&self.ranking);
        let ranker = reuse.ranker(group, tier, Reader::Move);
        let candidate = store_rank(ranker, &self.prompt_reads)((layer, expert), None);
        let cand_uses = candidate.expected().uses();
        let admit = (store_rank(ranker, &self.prompt_reads), move |kept: Kept, victim: Rank| {
            kept == Kept::Banked || (cand_uses + net_benefit_uses) > victim.expected().uses()
        });
        self.host[group].claim((layer, expert), keep, taking, admit)
    }

    /// Claim a background host-tier landing for a cold expert read into the
    /// conveyor (`Scratch::land_conveyed`): admitted whenever the tier covers
    /// every expert (`HostRecords::covers`); otherwise takes a free ready
    /// record or displaces a kept record only when the expert outranks its
    /// victim in the reuse ranking (beside `read` and pending prompt reads).
    pub fn claim_conveyed(&mut self, layer: usize, expert: usize, read: Option<Step>) -> Claim {
        let group = self.bank.group(layer);
        if self.host[group].covers() {
            return self.host[group].claim((layer, expert), 0, Taking::Out, (|_, _| (), |_, _| true));
        }
        let tier = self.host_tier();
        let reuse = Reuse::of(&self.ranking);
        let ranker = reuse.ranker(group, tier, Reader::Move);
        let candidate = store_rank(ranker, &self.prompt_reads)((layer, expert), read);
        self.host[group].claim((layer, expert), 0, Taking::Out, outranking(ranker, &self.prompt_reads, candidate))
    }

    /// Where `layer`'s `expert` is in its group's host tier: lacked
    /// without a tier.
    pub fn located(&self, layer: usize, expert: usize) -> Located {
        self.host(self.bank.group(layer)).located((layer, expert))
    }

    /// Whether a landing of `layer`'s `expert` into its group's host tier is
    /// in flight: whoever lands it, nothing else reads it.
    pub fn lands(&self, layer: usize, expert: usize) -> bool {
        self.host(self.bank.group(layer)).lands((layer, expert))
    }

    /// The experts the tier's cover reads next, at most `count`: of each
    /// group whose host tier covers its experts ([`HostRecords::covers`]),
    /// those it lacks that `skip` does not pass over (another read carries
    /// them), by group, layer, then expert: the order the steps need them.
    /// None where no tier covers its experts, or where the host places.
    pub fn cover_order(&self, count: usize, skip: impl Fn(Staged) -> bool) -> Vec<Staged> {
        // Where the host places (the one store) its records are the bank's:
        // the store is the bank, and nothing is read ahead of use.
        if self.placer != crate::planning::Placer::Tier {
            return Vec::new();
        }
        self.host.iter().filter(|tier| tier.covers()).flat_map(|tier| tier.lacking_unskipped(&skip)).take(count).collect()
    }

    /// Whether a group's host tier that covers its experts lacks one, which
    /// the cover reads ([`Self::cover_order`]); never where the host
    /// places.
    pub fn cover_pending(&self) -> bool {
        self.placer == crate::planning::Placer::Tier && self.host.iter().any(|tier| tier.covers() && tier.lacked() > 0)
    }

    /// The lender's bank group and its lent records `[0, count)`, beside a host
    /// tier (`None` on unified memory / offline replay).
    pub fn lender(&self) -> Option<(usize, usize)> {
        self.bank.lent()
    }

    /// Whether bank record `record` of group `group` is currently `Record::Free`.
    pub fn bank_record_free(&self, group: usize, record: usize) -> bool {
        self.bank.is_free(group, record)
    }

    /// Inspect bank record `record` of group `group`.
    pub(in crate::experts) fn bank_record(&self, group: usize, record: usize) -> Option<crate::experts::records::Record> {
        self.bank.records(group).get(record).copied()
    }

    /// Whether group `group` currently has any bank record leased to a prompt group.
    pub fn group_has_lent(&self, group: usize) -> bool {
        self.bank.has_lent(group)
    }

    /// Whether every group's host tier keeps every expert of its layers,
    /// none without a tier: no decode miss reads the file.
    pub fn tier_complete(&self) -> bool {
        self.host.iter().all(HostRecords::complete)
    }

    /// Bind the live per-layer `POLICY_WHOLE` flags (`Telemetry::whole_flags`)
    /// so mid-flight victim selection never evicts a published whole layer.
    pub fn bind_whole_layers(&mut self, whole: Arc<[AtomicBool]>) {
        if self.placer == crate::planning::Placer::Bank {
            for host in &mut self.host {
                host.bind_protected_layers(Arc::clone(&whole));
            }
        }
        self.whole_layers = whole;
    }

    /// Whether `layer` currently has `POLICY_WHOLE` published on the device.
    fn is_layer_whole_published(&self, layer: usize) -> bool {
        self.whole_layers
            .get(layer)
            .is_some_and(|flag| flag.load(Ordering::Acquire))
    }

    /// Whether group `group`'s host tier keeps every expert of its layers
    /// where no flight takes it ([`HostRecords::settled`]); false where the
    /// bank has no tier.
    pub fn group_tier_settled(&self, group: usize) -> bool {
        self.host[group].settled()
    }

    /// Whether every group's layers are whole in Tier 0 (no expert of any
    /// group can miss Tier 0). On unified memory this equals `tier_complete()`;
    /// on discrete GPUs it is true only when every expert is held in VRAM.
    pub fn all_groups_whole(&self) -> bool {
        (0..self.host.len()).all(|group| self.group_whole(group))
    }

    /// Whether no expert of `layer` can miss Tier 0: every expert `0..E` is
    /// held in the bank (`Placer::Tier`) or kept in the one store (`Placer::Bank`).
    pub fn layer_whole(&self, layer: usize) -> bool {
        let group = self.bank.group(layer);
        let experts = self.bank.experts(layer);
        match self.placer {
            crate::planning::Placer::Tier => (0..experts).all(|expert| self.bank.held(layer, expert).is_some()),
            crate::planning::Placer::Bank => self.host[group].layer_complete(layer, experts),
        }
    }

    /// Whether no expert of group `group`'s layers can miss, every one
    /// mapped where a pass reads it in place: beside a host tier, held in
    /// the bank; where the host places, kept in the one store, whose
    /// records are the bank's, its lent records among them.
    pub fn group_whole(&self, group: usize) -> bool {
        match self.placer {
            crate::planning::Placer::Tier => (0..self.bank.layers())
                .filter(|&layer| self.bank.group(layer) == group)
                .all(|layer| self.layer_whole(layer)),
            crate::planning::Placer::Bank => self.host[group].complete(),
        }
    }

    /// Whether every expert `0..E` of `layer` is currently kept at its home
    /// record (`home(layer) + expert`), so every slab in `[home .. home + E)`
    /// is backed and holds its expert weight.
    pub fn layer_at_home(&self, layer: usize) -> bool {
        let group = self.bank.group(layer);
        let home = self.home(layer);
        (0..self.bank.experts(layer)).all(|expert| match self.placer {
            crate::planning::Placer::Tier => self.bank.held(layer, expert) == Some(home + expert),
            crate::planning::Placer::Bank => {
                self.host[group].kept_record((layer, expert)).is_some_and(|record| record as usize == home + expert)
            }
        })
    }

    /// Whether every expert of group `group`'s layers is at its home record
    /// ([`Self::home`]), which a route then sources without the map: a bank
    /// imported whole is, one regrown to every expert need not be.
    pub fn group_at_home(&self, group: usize) -> bool {
        (0..self.bank.layers())
            .filter(|&layer| self.bank.group(layer) == group)
            .all(|layer| self.layer_at_home(layer))
    }

    /// Claim a demand landing of `layer`'s `expert` while the device is
    /// parked at `park`, leaving no reserve but the records set aside for
    /// cut-ins, which a cut-in's park takes too ([`Self::set_pool`]): where
    /// none is ready, one a kept record's unmapping frees at once, any the
    /// parked step's route did not resolve there (`read`) and no earlier
    /// group's step may still resolve ([`HostRecords::claim_demand`]), its
    /// unmapping stored before the claim returns, so before the landing
    /// writes the record. Short without a tier.
    pub fn claim_demand(&mut self, (layer, expert): Staged, demand: Demand, read: &[(usize, u32)], park: &Park) -> Claim {
        let group = self.bank.group(layer);
        let read = |record| read.contains(&(group, record));
        if let Some(claim) = self.host[group].claim_prepared((layer, expert), demand, read, park) {
            return claim;
        }
        let reuse = Reuse::of(&self.ranking);
        let rank = store_rank(reuse.ranker(group, self.host_tier(), Reader::Move), &self.prompt_reads);
        self.host[group].claim_demand((layer, expert), demand, rank, read, park)
    }

    /// The posted read of `layer`'s `expert`, which no record took, ended
    /// unlanded or was never made ([`HostRecords::unposted`]).
    pub fn unposted(&mut self, layer: usize, expert: usize) {
        self.tier(layer).unposted((layer, expert));
    }

    /// Prepare `layer`'s group's demand victims, `depth` of them
    /// ([`HostRecords::prepare_demand`]), at the ranking as it is now, at
    /// step `sequence`: where a claim took any, or they were prepared at an
    /// earlier step (the ranking moves by steps).
    pub fn prepare_demand(&mut self, layer: usize, needed: usize, sequence: Step) {
        let group = self.bank.group(layer);
        if !self.host[group].unprepared(sequence, needed) {
            return;
        }
        // Prepare enough victims for every layer of this group in the step
        // (capped at the group's extent), so a multi-layer decode step scans
        // the group's records once rather than once per layer.
        let layers = self.layer_groups().iter().filter(|&&of| of == group).count().max(1);
        let depth = (needed * layers).min(self.host[group].records() as usize);
        let reuse = Reuse::of(&self.ranking);
        let rank = store_rank(reuse.ranker(group, self.host_tier(), Reader::Move), &self.prompt_reads);
        self.host[group].prepare_demand(rank, depth, sequence);
    }

    /// `layer`'s `expert` landed whole into tier record `record`, read at
    /// step `read`, and its landing mapped it ([`HostRecords::landed`]),
    /// ranked by its reuse beside that read (none where no step read it). A
    /// copy of an expert the bank holds (a landing a promotion or seed
    /// committed during) is banked: the tier holds alone only what the bank
    /// lacks. The one move of the host records' tier (design §3.2: a landing
    /// fills the record its claim took), a written write-back's among them
    /// ([`Self::written_back`]).
    pub fn landed(&mut self, layer: usize, HostRecord(record): HostRecord, expert: usize, read: Option<Step>) {
        let kept = match self.bank.held(layer, expert) {
            Some(_) => Kept::Banked,
            None => Kept::Alone,
        };
        self.tier(layer).landed(record, (layer, expert), read, kept);
        let (group, tier) = (self.bank.group(layer), self.host_tier());
        self.time.lock().moved(group, tier);
        // A want its tier lacked is promotable from now.
        self.stirred |= self.wanted[layer].contains(&expert);
    }

    /// The landing into tier record `record` of `layer`'s group failed: the
    /// record is ready again ([`HostRecords::unlanded`]).
    pub fn unlanded(&mut self, layer: usize, HostRecord(record): HostRecord) {
        self.tier(layer).unlanded(record);
    }

    /// Whether any group's host tier falls short of its reserve
    /// ([`HostRecords::short_of`]): whether a refill may unmap anything.
    pub fn refill_short(&self) -> bool {
        self.host.iter().any(|records| records.short_of(0))
    }

    /// Each group's records its reserve unmaps now
    /// ([`HostRecords::refill_victims`]), none of the lent records while
    /// `lending`: read from the ranking, changing nothing.
    pub fn refill_victims(&self, lending: bool) -> Vec<Vec<(u32, Staged)>> {
        let reuse = self.reuse_now();
        let tier = self.host_tier();
        self.host
            .iter()
            .enumerate()
            .map(|(group, records)| {
                let rank = store_rank(reuse.ranker(group, tier, Reader::Move), &self.prompt_reads);
                // Protect unconsumed read-aheads (landed for a step Reuse has not
                // yet logged) from immediate eviction before their step runs.
                let protected = |staged @ (layer, _): Staged, read: Option<Step>| {
                    let unconsumed = read.is_some_and(|step| reuse.lead_of(layer, step).is_some());
                    (unconsumed, rank(staged, read))
                };
                records.refill_victims(protected, lending)
            })
            .collect()
    }

    /// Unmap each group's `victims` ([`HostRecords::refill`]), under the
    /// lock every step begins under, `at` the first step not yet begun, the
    /// lent records only while no scratch-served group's lease is out.
    pub fn refill(&mut self, victims: Vec<Vec<(u32, Staged)>>, at: &Unbegun) -> bool {
        self.refill_with(victims, at, false)
    }

    pub fn refill_with(&mut self, victims: Vec<Vec<(u32, Staged)>>, at: &Unbegun, immediate: bool) -> bool {
        self.host
            .iter_mut()
            .zip(victims)
            .fold(false, |unmapped, (records, victims)| unmapped | records.refill_with(victims, at, immediate))
    }

    /// Group `group`'s host tier sets `records` ready records aside for
    /// the decode steps cut in during a prompt pass from now, none once
    /// the pass ends ([`HostRecords::set_pool`]): one decode step's
    /// expected misses of the group's layers (`misses`), and never fewer
    /// than a cut-in step's choices at one layer (`choices`), which it may
    /// hold answering at one park: so a cut-in never waits on its own
    /// pool.
    pub fn set_pool(&mut self, group: usize, misses: u32, choices: u32) {
        self.host[group].set_pool(if misses == 0 { 0 } else { misses.max(choices) });
    }

    /// The tier record keeping `layer`'s `expert` whole, if one does.
    pub fn tier_record(&self, layer: usize, expert: usize) -> Option<HostRecord> {
        self.host(self.bank.group(layer)).kept_record((layer, expert)).map(HostRecord)
    }

    /// Whether `layer`'s `expert` lives in its group's host tier: a miss of
    /// it the device serves in place, with no read.
    pub fn in_tier(&self, layer: usize, expert: usize) -> bool {
        self.host(self.bank.group(layer)).keeps((layer, expert))
    }

    /// Whether the prompt group whose last step is `last` reads `layer`'s
    /// `expert` from its group's host tier, kept or landing there
    /// ([`HostRecords::serves`]): none a cut-in landed, which the cut-ins
    /// recycle.
    pub fn tier_serves(&self, layer: usize, expert: usize, last: Step) -> bool {
        self.host(self.bank.group(layer)).serves((layer, expert), last)
    }

    /// Hold the tier record keeping `layer`'s `expert` whole for the copy
    /// of the prompt group whose last step is `last`
    /// ([`Self::tier_serves`]), which holds it until the group is past:
    /// the record, none where the tier keeps no copy that group reads.
    pub fn hold_for_pass(&mut self, layer: usize, expert: usize, last: Step) -> Option<HostRecord> {
        let tier = self.tier(layer);
        tier.serves((layer, expert), last).then(|| tier.hold((layer, expert))).flatten().map(HostRecord)
    }

    /// Hold the tier record keeping `layer`'s `expert` whole for a pull
    /// reading it ([`HostRecords::hold`]): the record, none where the tier
    /// keeps no copy of it.
    pub fn hold_tier(&mut self, layer: usize, expert: usize) -> Option<HostRecord> {
        self.tier(layer).hold((layer, expert)).map(HostRecord)
    }

    /// `layer`'s tier record `record`, which keeps its expert whole, answers
    /// the request at `past` from now: nothing takes it until the device is
    /// past there ([`HostRecords::answer`]).
    pub fn answer(&mut self, layer: usize, HostRecord(record): HostRecord, past: Point) {
        self.tier(layer).answer(record, past);
    }

    /// The pull from `layer`'s tier record `record` ended
    /// ([`HostRecords::unhold`]).
    pub fn unhold_tier(&mut self, layer: usize, HostRecord(record): HostRecord) {
        self.tier(layer).unhold(record);
    }

    /// The rank of `layer`'s `expert` in the bank (tier 0) by a lock-free
    /// `ranker`, beside a read at step `read` the log may not have brought
    /// yet ([`beside`]).
    fn ranked(&self, ranker: Ranker<'_>, layer: usize, expert: usize, read: Option<Step>) -> Ranked {
        Ranked { rank: ranker.rank(layer, expert, beside(&self.prompt_reads, (layer, expert), read)), layer, expert }
    }

    /// Each record of group `group`, the step that last read the expert it
    /// holds as `reuse` logged it (`Reuse::last`), 0 for none: what its
    /// held records rank by ([`Self::least_of`]).
    fn record_reads(&self, reuse: &Reuse, group: usize) -> Vec<Step> {
        self.bank
            .records(group)
            .iter()
            .map(|content| match *content {
                Record::Held((layer, expert)) => reuse.last(layer, expert).unwrap_or(0),
                _ => 0,
            })
            .collect()
    }

    /// Settle the ranking's changes of its pools (`Reuse::take_changes`):
    /// whether a promotion or a release may have become possible since they
    /// were last asked (a want or a victim gained or lost, [`Self::stirred`],
    /// a record freed or taken, [`BankRecords::take_changed`], or a read
    /// logged since a want was withheld, [`Self::withheld`]), so
    /// [`Self::promotions`] may ask any.
    pub fn settle_changes(&mut self) -> bool {
        let changes = {
            let mut reuse = Reuse::of(&self.ranking);
            let changes = reuse.take_changes();
            changes.into_iter().map(|(layer, expert)| (layer, expert, reuse.in_top(layer, expert))).collect::<Vec<_>>()
        };
        for (layer, expert, top) in changes {
            self.settle(layer, expert, top);
        }
        self.stirred || self.bank.changed() || self.read_since_withheld() || self.host.iter().any(HostRecords::surrenders)
    }

    /// Whether any host tier has copies to surrender (`HostRecords::surrenders`).
    pub fn has_surrenders(&self) -> bool {
        self.host.iter().any(HostRecords::surrenders)
    }

    /// Whether the ranking logged a read since promotions last withheld a
    /// want ([`Self::withheld`]).
    fn read_since_withheld(&self) -> bool {
        self.withheld.is_some_and(|reads| self.reuse_now().reads() != reads)
    }

    /// What the ranking asks now, in time proportional to what changed,
    /// capped to `budget` copy-engine slots:
    /// the ranking's changes of its pools (`Reuse::take_changes`) are
    /// settled first, and only where a want, a victim or a free record was
    /// gained or lost since ([`Self::stirred`]) is anything asked. In each
    /// group, each expert the ranking's pool of it holds, the bank lacks and
    /// its host tier keeps (`wanted`) takes a free record (up to the
    /// remaining `budget`), which no step
    /// reads, so its copy begins at once: the bank then holds the group's
    /// pool, as the ranking does (`Reuse`). Each record is
    /// its promotion's from now until its copy has completed
    /// ([`Self::promoted`]) or failed ([`Self::unpromoted`]). Then, while the
    /// group's free and releasing records fall short of its reserve plus the
    /// wants left without a record that fit within the remaining `budget`,
    /// the record of its least reused expert outside the pool (`unwanted`) is
    /// released ([`Self::left_bank`]): free once every step that may read it
    /// has ended ([`Self::drain`]), and written back into the host tier first
    /// where that pays (its exit, [`Self::exit`]). Held experts beyond the
    /// copy-engine budget stay mapped and serving hits until the copy engine
    /// is ready to copy their replacements. Before anything, each host
    /// tier gives up the copies of what the bank holds now
    /// ([`HostRecords::surrender`]): the tiers are disjoint. So a promotion never waits for a record to
    /// drain while the reserve covers what the ranking asks meanwhile; only
    /// the records a promotion may take cover a want
    /// ([`BankRecords::covering`]). Neither ever takes a lent record, the
    /// scratch's ([`BankRecords::usable`]).
    pub fn promotions(&mut self, at: &Unbegun, budget: usize) -> Wants {
        // The tier gives up what the bank holds now: its records ready for
        // other experts once no step may still read them.
        let mut asked = Wants::default();
        for (group, host) in self.host.iter_mut().enumerate() {
            for (record, _) in host.surrender_records(at) {
                asked.surrendered.push((group, record as usize));
            }
        }
        self.settle_changes();
        // Promotion runs only for decode steps (design §3.2): before the
        // first, no τ exists. What stirred stays stirred until then.
        if self.time.lock().clock.through_latest() == super::horizon::Tokens::ZERO {
            return asked;
        }
        let stirred = std::mem::take(&mut self.stirred) | self.bank.take_changed() | self.read_since_withheld();
        if !stirred {
            return asked;
        }
        let lending = at.lending();
        let mut budget_left = budget;
        let mut capped = false;
        let mut withheld = false;
        for group in 0..self.homes.len() {
            let members = self.homes[group].iter().map(|&(_, layer)| layer).collect::<Vec<_>>();
            // Earning weighs a holding displaced for each want, and releases
            // take at most a want's each and the reserve's: all ranked and
            // weighed by one hold of the ranking, at τ_0. When no prompt reads
            // override the ranking, `Reuse::top_order(group)` is already sorted
            // ascending by `Rank`, so selecting the top wants and bottom
            // victims in O(budget) avoids recomputing `expected_for` over tens
            // of thousands of queued experts on every turn.
            let (victims, priced, truncated) = {
                let reuse = Reuse::of(&self.ranking);
                let order = reuse.top_order(group);
                let need = budget_left.saturating_add(self.reserve[group]);
                let want_limit = need.max(64).saturating_mul(4).min(order.len());
                let (wanted, truncated) = if self.prompt_reads.is_empty() {
                    let mut wanted = Vec::with_capacity(want_limit);
                    let mut more = false;
                    for &(_, layer, expert) in order.iter().rev() {
                        if self.wanted[layer].contains(&expert) && self.in_tier(layer, expert) {
                            if wanted.len() < want_limit {
                                wanted.push((layer, expert));
                            } else {
                                more = true;
                                break;
                            }
                        }
                    }
                    (wanted, more)
                } else {
                    let wanted = members
                        .iter()
                        .flat_map(|&layer| self.wanted[layer].iter().map(move |&expert| (layer, expert)))
                        .filter(|&(layer, expert)| self.in_tier(layer, expert))
                        .collect::<Vec<_>>();
                    (wanted, false)
                };
                let victim_most = wanted.len().min(budget_left).saturating_add(self.reserve[group]);
                let victims = self.victims(&reuse, group, &members, victim_most, lending);
                let priced = self.earning(&reuse, group, wanted, &victims.floor, lending);
                (victims.released, priced, truncated)
            };
            withheld |= priced.withheld || (truncated && priced.earning.len() <= budget_left);
            let wanted = priced.earning;
            let take = wanted.len().min(budget_left);
            capped |= wanted.len() > budget_left || truncated;
            // Prefer permanent records past `0..lent` first so the hottest
            // experts survive across prompt groups untouched, then fill
            // `0..lent` during decode when `!lending`.
            let non_lent = self
                .bank
                .free(group)
                .filter(|&record| self.bank.usable_when(group, record, true));
            let lent_free = self
                .bank
                .free(group)
                .filter(|&record| !lending && self.bank.lends(group, record) && self.bank.usable_when(group, record, false));
            let all_free: Vec<usize> = non_lent.chain(lent_free).collect();
            let demand_reserve = if self.bank.capacity(group) > 16 {
                self.demand_free_floor(group)
            } else {
                0
            };
            let available = all_free.len().saturating_sub(demand_reserve);
            let free = all_free.into_iter().take(take.min(available)).collect::<Vec<_>>();
            let promoted = free.len();
            budget_left -= promoted;
            let short = wanted.len() - promoted;
            for ((layer, expert), record) in wanted.into_iter().zip(free) {
                let promotion = Promotion { group, record: bank_record(record), layer, expert };
                self.bank.promote(&promotion);
                self.asked += 1;
                self.resettle(layer, expert);
                asked.promotions.push(promotion);
            }
            // A want left without a record releases one now, capped to the
            // remaining copy-engine budget so held experts stay mapped and
            // serving hits until the copy engine is ready for their
            // replacements.
            let covering = self.bank.covering_when(group, lending);
            let want_releases = short.min(budget_left);
            capped |= short > budget_left;
            let releases = (self.reserve[group] + want_releases).saturating_sub(covering);
            if releases == 0 {
                continue;
            }
            for (Ranked { layer, expert, .. }, record) in victims.into_iter().take(releases) {
                let exit = self.exit_now(group, record, (layer, expert));
                if exit.back.is_some() {
                    budget_left = budget_left.saturating_sub(1);
                }
                let record = bank_record(record);
                self.bank.release(group, record, at.into(), exit);
                self.left_bank(layer, expert, None);
                asked.releases.push(Release { group, record, layer, expert });
            }
        }
        // What it took is asked for: only what changes after asks again,
        // unless copy-engine budget capped remaining wants AND this pass
        // actually made progress (otherwise nothing can change until a
        // release drains or the ranking updates).
        self.bank.take_changed();
        let progressed = !asked.promotions.is_empty() || !asked.releases.is_empty();
        self.stirred |= capped && (progressed || budget == 0);
        self.withheld = withheld.then(|| self.reuse_now().reads());
        asked
    }

    /// What group `group`'s promotions may displace, of each held expert
    /// outside the pool (`unwanted`) whose record is usable, by `reuse` at
    /// τ_0 (`Reader::Move`: a promotion is a move): the `most` least ranked,
    /// in the order releases take them (`released`), and the `most` least
    /// expected uses within τ_0, least first (`floor`, `Reuse::expected`
    /// beside their reads, [`beside`]), each selected before it is sorted.
    /// A release chooses which record leaves now, by the order the ranking
    /// keeps; a copy takes a record for its whole stay, and what the bank
    /// gives up for it is the least any unwanted holding is worth within
    /// τ_0, whichever the ranking releases first.
    fn victims(&self, reuse: &Reuse, group: usize, members: &[usize], most: usize, lending: bool) -> Victims {
        let held: Vec<(Staged, usize)> = if self.prompt_reads.is_empty() && most < usize::MAX {
            let order = reuse.top_order(group);
            let mut held = Vec::with_capacity(most.min(order.len()));
            for &(_, layer, expert) in order {
                if held.len() >= most {
                    break;
                }
                if !self.is_layer_whole_published(layer) && self.unwanted[layer].contains(&expert) {
                    let record = self.bank.held(layer, expert).expect("an unwanted expert is held");
                    if self.bank.usable_when(group, record, lending) {
                        held.push(((layer, expert), record));
                    }
                }
            }
            held
        } else {
            members
                .iter()
                .filter(|&&layer| !self.is_layer_whole_published(layer))
                .flat_map(|&layer| self.unwanted[layer].iter().map(move |&expert| (layer, expert)))
                .map(|staged @ (layer, expert)| (staged, self.bank.held(layer, expert).expect("an unwanted expert is held")))
                .filter(|&(_, record)| self.bank.usable_when(group, record, lending))
                .collect::<Vec<_>>()
        };
        let ranker = reuse.ranker(group, Tier::COMPUTE, Reader::Move);
        let mut released = held.iter().map(|&((layer, expert), record)| (self.ranked(ranker, layer, expert, None), record)).collect::<Vec<_>>();
        let mut floor = held.iter().map(|&(staged, _)| self.uses(ranker, staged)).collect::<Vec<_>>();
        if held.len() > most {
            released.select_nth_unstable(most);
            released.truncate(most);
            floor.select_nth_unstable_by(most, f64::total_cmp);
            floor.truncate(most);
        }
        released.sort_unstable();
        floor.sort_unstable_by(f64::total_cmp);
        Victims { released, floor }
    }

    /// `staged`'s expected uses within τ_0 by a lock-free `ranker`, beside
    /// its reads ([`beside`]): N at the bank's move τ (`Expected::uses`),
    /// finite.
    fn uses(&self, ranker: Ranker<'_>, staged @ (layer, expert): Staged) -> f64 {
        ranker.expected(layer, expert, beside(&self.prompt_reads, staged, None)).uses()
    }

    /// Of group `group`'s `wanted`, those whose copy pays within τ_0
    /// (`Self::price_copies`): the most used beyond its layer's price
    /// first. Usable free records past the reserve displace nothing and
    /// replace a 0-hit empty slot, so any priced `in_top` want with positive
    /// expected uses enters immediately without being starved by cold-start
    /// `τ_0` collapse; once free records past the reserve are exhausted, each
    /// further want is weighed against the unwanted holdings' uses least first
    /// (`floor`, `Self::victims`), all by one measure ([`Self::uses`]), and a
    /// want left with nothing to displace earns nothing. What a copy costs and
    /// a hit saves are the cost function's (`ExpertLines::copy_hits`).
    fn earning(&self, reuse: &Reuse, group: usize, wanted: Vec<Staged>, floor: &[f64], lending: bool) -> Earning {
        // A layer whose hit saves nothing earns no promotion at any read:
        // its wants wait for no read, so none is withheld; its lines'
        // repricing asks them again (`Self::price_copies`).
        let ranker = reuse.ranker(group, Tier::COMPUTE, Reader::Move);
        let mut ranked = wanted
            .into_iter()
            .filter_map(|staged| {
                self.copy_hits[staged.0].map(|hits| {
                    let uses = self.uses(ranker, staged);
                    (uses - hits, uses, staged)
                })
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.total_cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));
        // The records past the reserve displace nothing; a promotion into
        // the reserve is paid for by the release that refills it.
        let free = self.bank.free(group).filter(|&record| self.bank.usable_when(group, record, lending)).count();
        let mut free_left = free.saturating_sub(self.reserve[group]);
        let count = ranked.len();
        let mut earning = Vec::new();
        let mut floor_iter = floor.iter().copied();
        for (surplus, uses, staged) in ranked {
            if free_left > 0 {
                if uses > 0.0 {
                    free_left -= 1;
                    earning.push(staged);
                }
            } else {
                let Some(displaced) = floor_iter.next() else {
                    break;
                };
                if surplus - displaced >= 0.0 {
                    earning.push(staged);
                } else {
                    break;
                }
            }
        }
        Earning { withheld: earning.len() < count, earning }
    }

    /// The one exit of group `group`'s banked experts, by `reuse`
    /// ([`Exits::leaving`]): the host tier's move τ, read once, in decode
    /// steps (τ_1 / ā).
    fn exits<'a>(&'a self, reuse: &'a Reuse, group: usize) -> Exits<'a> {
        let stay = (self.placer == crate::planning::Placer::Tier && self.host[group].records() > 0)
            .then(|| {
                let times = self.time.lock();
                let tau = match times.horizon(group, Tier::HOST, Reader::Move)? {
                    Horizon::Tokens(tau) => tau.get(),
                    Horizon::Unbounded => unreachable!("a move's τ is finite once a decode step has begun"),
                };
                Some(tau / times.tokens_per_step(group).expect("ā exists wherever τ does"))
            })
            .flatten();
        Exits { directory: self, reuse, group, stay, displaced: std::cell::OnceCell::new() }
    }

    /// Carry `leaving` out of bank record `record` of group `group`: where
    /// its write-back pays, it reserves a tier record in [`Host::WritingBack`]
    /// until the copy ends ([`Self::written_back`]), leaving the expert lacked
    /// so no parked decode step ever waits on the transfer queue. Short of a
    /// ready record, or where another landing already holds it, it writes
    /// nothing back.
    fn exit(&mut self, group: usize, record: usize, leaving: Leaving) -> Exit {
        let Leaving { staged, back } = leaving;
        let Some(read) = back else {
            return Exit { staged, back: None };
        };
        let back = {
            let reuse = Reuse::of(&self.ranking);
            let ranker = reuse.ranker(group, Tier::HOST, Reader::Move);
            let candidate = ranker.rank(staged.0, staged.1, beside(&self.prompt_reads, staged, read));
            let keep = self.host[group].reserve();
            match self.host[group].claim_back(staged, keep, outranking(ranker, &self.prompt_reads, candidate)) {
                Some(grant) => {
                    let tier = HostRecord(grant.record());
                    self.backs.insert((group, record), (grant, read));
                    Some(tier)
                }
                None => None,
            }
        };
        Exit { staged, back }
    }

    /// `staged` leaves bank record `record` of group `group` by its exit,
    /// decided by the ranking now ([`Exits::leaving`]) and carried out
    /// ([`Self::exit`]).
    fn exit_now(&mut self, group: usize, record: usize, staged: Staged) -> Exit {
        let leaving = self.exits(&self.reuse_now(), group).leaving(staged);
        self.exit(group, record, leaving)
    }

    /// The releases' write-backs queued ([`BankRecords::backs_queued`]).
    pub fn backs_queued(&self) -> Vec<(usize, BankRecord, Staged, HostRecord)> {
        self.bank.backs_queued()
    }

    /// The releases' write-backs copying, each (group, record, batch).
    pub fn backing(&self) -> Vec<(usize, BankRecord, u64)> {
        self.bank.backing()
    }

    /// Queued write-back of releasing `record` of group `group` copies by
    /// batch `batch`.
    pub fn back_copy(&mut self, group: usize, record: BankRecord, batch: u64) {
        self.bank.back_copy(group, record, batch);
    }

    /// The write-back of releasing `record` of group `group` ended,
    /// `written` or not: written, its tier record is mapped and keeps the
    /// expert alone ([`Self::landed`], ranked beside its last read, which
    /// feeds the host tier's tier-1 move); else the landing ends unlanded
    /// and only the file holds it. The release drains as any from now.
    pub fn written_back(&mut self, group: usize, record: BankRecord, written: bool) {
        let Back { staged: (layer, expert), tier: HostRecord(tier), .. } = self.bank.backed(group, record);
        let (grant, read) = self.backs.remove(&(group, record.index())).expect("a write-back holds its landing's grant");
        let kept = match self.bank.held(layer, expert) {
            Some(_) => Kept::Banked,
            None => Kept::Alone,
        };
        if self.tier(layer).written_back(tier, (layer, expert), read, kept, written) {
            grant.map();
            let (group, tier) = (self.bank.group(layer), self.host_tier());
            self.time.lock().moved(group, tier);
            // A want its tier lacked is promotable from now.
            self.stirred |= self.wanted[layer].contains(&expert);
        }
    }

    /// Layer `layer`'s write-backs are priced at `price` from now, none
    /// before the passes and the copy line are measured.
    pub fn price_write_backs(&mut self, layer: usize, price: Option<WriteBack>) {
        self.write_backs[layer] = price;
    }

    /// The promotions every group has asked since the load.
    pub fn asked(&self) -> u64 {
        self.asked
    }

    /// Layer `layer`'s promotions earn their copies at `hits` reads from
    /// now, none where a hit saves nothing; what they withheld are asked
    /// again.
    pub fn price_copies(&mut self, layer: usize, hits: Option<f64>) {
        if std::mem::replace(&mut self.copy_hits[layer], hits) != hits {
            self.stirred = true;
        }
    }

    /// Every release the device is past at `progress` is free
    /// ([`BankRecords::drain`]): whether any was.
    pub fn drain(&mut self, progress: Point, steps: &Steps) -> bool {
        self.bank.drain(progress, steps)
    }

    /// The promotions queued, which copy now ([`BankRecords::queued`]).
    pub fn queued(&self) -> Vec<(usize, BankRecord, Staged)> {
        self.bank.queued()
    }

    /// The promotions copying, each (group, record, batch).
    pub fn copying(&self) -> Vec<(usize, BankRecord, u64)> {
        self.bank.copying()
    }

    /// Whether any promotion is in flight, any release draining, or any
    /// opportunistic staging slot active, and what each is doing.
    pub fn moving(&self) -> bool {
        self.bank.moving()
    }

    /// Whether any link transfer (promotion, staging DMA, or write-back
    /// release) is queued or in flight ([`BankRecords::link_moving`]).
    pub fn link_moving(&self) -> bool {
        self.bank.link_moving()
    }

    /// Find a free parity-half staging slot in `group`'s dedicated staging ring
    /// for `target_step` ([`BankRecords::free_stage_slot`]).
    pub fn free_stage_slot(&self, group: usize, target_step: Step) -> Option<usize> {
        self.bank.free_stage_slot(group, target_step)
    }

    /// Number of slots in each parity half (`even` or `odd`) of `group`'s
    /// dedicated DMA staging ring (`0` when disabled).
    pub fn stage_ring_half(&self, group: usize) -> usize {
        self.bank.stage_ring_half(group)
    }

    /// Whether `record` of `group` lies in its dedicated DMA staging ring.
    pub fn is_stage_slot(&self, group: usize, record: usize) -> bool {
        self.bank.is_stage_slot(group, record)
    }

    /// Record an opportunistic DMA staging copy into `slot` of `group`
    /// ([`BankRecords::stage`]).
    #[allow(clippy::too_many_arguments)]
    pub fn stage(&mut self, group: usize, slot: usize, staged: Staged, target_step: Step, tier: HostRecord, batch: u64, parity: u8) {
        self.bank.stage(group, slot, staged, target_step, tier, batch, parity);
    }

    /// Mark an in-flight opportunistic staging copy as landed, releasing its
    /// host-tier hold while keeping the VRAM slot reserved until `frontier > target_step`.
    pub fn staged_landed(&mut self, group: usize, slot: usize) -> Staged {
        let (staged @ (layer, _), tier) = self.bank.staged_landed(group, slot);
        self.unhold_tier(layer, tier);
        staged
    }

    /// Cancel or fail an in-flight opportunistic staging copy, releasing its
    /// host-tier hold and returning the VRAM slot to `Record::Free`.
    pub fn unstage(&mut self, group: usize, slot: usize) -> Staged {
        let (staged @ (layer, _), tier) = self.bank.unstage(group, slot);
        if let Some(tier) = tier {
            self.unhold_tier(layer, tier);
        }
        staged
    }

    /// Drain landed staging slots whose `(target_step, layer)` has been served
    /// ([`BankRecords::drain_staging`]).
    #[allow(dead_code)]
    pub fn drain_staging(&mut self, progress: Point, steps: &super::watcher::Steps) -> bool {
        self.bank.drain_staging(progress, steps)
    }

    /// Drain landed staging slots whose `(target_step, layer)` has been served
    /// and that `can_release(group, slot)` confirms are unpinned in `ExpertTransport`.
    pub fn drain_staging_with(
        &mut self,
        progress: Point,
        steps: &super::watcher::Steps,
        can_release: impl FnMut(usize, usize) -> bool,
    ) -> bool {
        self.bank.drain_staging_with(progress, steps, can_release)
    }

    /// Number of active staging slots currently held in `group` for `target_step`.
    pub fn staged_for_step(&self, group: usize, target_step: Step) -> usize {
        self.bank.staged_for_step(group, target_step)
    }

    /// In-flight opportunistic staging DMA batches: `(group, slot, batch)`.
    pub fn staging_copying(&self) -> Vec<(usize, usize, u64)> {
        self.bank.staging_copying()
    }

    /// Active staging entry for `staged` in `group`, if any:
    /// `(slot, target_step, batch)`.
    pub fn staged_entry(&self, group: usize, staged: Staged) -> Option<(usize, Step, Option<u64>)> {
        self.bank.staged_entry(group, staged)
    }

    /// Whether `staged` is currently being placed into the permanent bank by a
    /// promotion or prompt seed (`BankRecords::placing`).
    pub fn placing(&self, staged: Staged) -> bool {
        self.bank.placing(staged)
    }

    pub fn describe_promotions(&self) -> String {
        self.bank.describe()
    }

    /// Queued promotion `record` of group `group` copies from tier record
    /// `tier`, held, by batch `batch` ([`BankRecords::copy`]).
    pub fn copy(&mut self, group: usize, record: BankRecord, tier: HostRecord, batch: u64) {
        self.bank.copy(group, record, tier, batch);
    }

    /// The copies of promotions `completed`, each (group, record), have
    /// completed: each holds its expert from the map rule's step for its
    /// layer with the device known at `progress` (`Unresolved::from`, read
    /// under the steps' lock, [`BankRecords::promoted`]), banking the copy
    /// its host tier keeps, whose tier record the copy no longer holds. Each
    /// one's expert and the record it holds it in.
    pub fn promoted(&mut self, completed: Vec<(usize, BankRecord)>, progress: Point, steps: &Steps) -> Vec<(Staged, BankRecord)> {
        // Only the records' change and its mapping under the steps' lock;
        // what the tier and the ranking learn of it after, outside it.
        let bank = &mut self.bank;
        let promoted = steps.unresolved(progress, |unresolved| bank.promoted(completed, |layer| unresolved.from(layer)));
        promoted
            .into_iter()
            .map(|((layer, expert), record, tier)| {
                let group = self.bank.group(layer);
                self.time.lock().moved(group, Tier::COMPUTE);
                // Keep the host-tier copy of experts placed in the lent scratch
                // slice `0..lent` (do not surrender them as `Kept::Banked`):
                // `0..lent` is evicted on every prefill entry (`lease_records`),
                // so retaining the host copy avoids D2H write-back stalls and
                // prevents dropping ~480 hot experts to NVMe on 3-tier banks.
                if !self.bank.lends(group, record.index()) {
                    self.tier(layer).banked((layer, expert));
                }
                self.tier(layer).unhold(tier.0);
                self.resettle(layer, expert);
                ((layer, expert), record)
            })
            .collect()
    }

    /// Promotion `record` of group `group` ended without its expert (its
    /// tier record was taken before its copy began, or its copy failed):
    /// the record is free, as nothing maps it.
    pub fn unpromoted(&mut self, group: usize, record: BankRecord) {
        let ((layer, expert), tier) = self.bank.unpromoted(group, record);
        if let Some(tier) = tier {
            self.tier(layer).unhold(tier.0);
        }
        self.resettle(layer, expert);
    }

    /// Claim a unified VRAM slot in `group` for an on-demand decode DMA
    /// landing of host-tier record `tier_record`: resolves `(layer, expert)`
    /// via `HostRecords::kept_at`, returns `AlreadyHeld` if already in VRAM,
    /// allocates a `Record::Free` usable unpinned slot (`Record::DemandLanding`),
    /// or returns `Overflow` without blocking.
    pub fn claim_demand_landing(
        &mut self,
        group: usize,
        tier_record: u32,
        lending: bool,
        can_reuse: impl FnMut(usize) -> bool,
    ) -> DemandSlotClaim {
        let staged = self.host[group].kept_at(tier_record);
        if let Some((layer, expert)) = staged {
            if let Some(slot) = self.bank.held(layer, expert) {
                return DemandSlotClaim::AlreadyHeld { staged: (layer, expert), slot };
            }
        }
        match self.bank.claim_demand_landing(group, staged, lending, can_reuse) {
            Some(slot) => DemandSlotClaim::Allocated { staged, slot },
            None => DemandSlotClaim::Overflow { staged },
        }
    }

    /// Minimum number of `Record::Free` slots kept available in `group` for
    /// immediate `claim_demand_landing` without stamped-unmap delay: up to 16
    /// slots (2 × 8 top-K choices), bounded by a quarter of group capacity.
    fn demand_free_floor(&self, group: usize) -> usize {
        let cap = self.bank.capacity(group);
        if cap <= 2 {
            0
        } else {
            16usize.min(cap / 4).max(1)
        }
    }

    /// Target `Free + Releasing` headroom maintained by
    /// [`Self::replenish_demand_free`]: `demand_free_floor` for zero-overflow
    /// immediate claims plus an equal per-step promotion budget so
    /// [`Self::finish_demand_landing`] can promote ranked landings in place
    /// without ever dipping below `demand_free_floor`.
    fn demand_replenish_target(&self, group: usize) -> usize {
        let floor = self.demand_free_floor(group);
        let cap = self.bank.capacity(group);
        (floor * 2).min(cap / 3).max(floor)
    }

    /// Asynchronously release up to `want_free` lowest-ranked unpinned
    /// `Record::Held` victims in `group` (with `!is_layer_whole_published` and
    /// `can_reuse(slot)`) into `Record::Releasing` (stamped unmap via
    /// `BankRecords::release`) when `Free + Releasing` usable headroom falls
    /// below `want_free`. Non-blocking: subsequent steps find `Record::Free`
    /// slots after `drain()`.
    pub fn replenish_demand_free(
        &mut self,
        group: usize,
        want_free: usize,
        at: &Unbegun,
        mut can_reuse: impl FnMut(usize) -> bool,
    ) -> usize {
        let lending = at.lending();
        let want_free = want_free.max(self.demand_replenish_target(group));
        let short = want_free.saturating_sub(self.bank.covering_when(group, lending));
        if short == 0 {
            return 0;
        }
        let reuse = self.reuse_now();
        let mut released = 0usize;
        if self.prompt_reads.is_empty() {
            // Fast O(short) path: `top_order(group)` is ascending by rank
            // (lowest-ranked first). Collect up to `want_cands` unwanted
            // usable+reusable held experts (and up to `want_cands` lowest-ranked
            // wanted held experts as fallback if fewer than `short` unwanted
            // exist), stopping as soon as enough unwanted victims are found.
            let want_cands = short.saturating_add(8);
            let mut unwanted_cands: Vec<(Staged, usize)> = Vec::with_capacity(want_cands.min(64));
            let mut wanted_cands: Vec<(Staged, usize)> = Vec::new();
            for &(_, layer, expert) in reuse.top_order(group) {
                if self.is_layer_whole_published(layer) {
                    continue;
                }
                let Some(record) = self.bank.held(layer, expert) else {
                    continue;
                };
                if !self.bank.usable_when(group, record, lending) || !can_reuse(record) {
                    continue;
                }
                if self.unwanted[layer].contains(&expert) {
                    unwanted_cands.push(((layer, expert), record));
                    if unwanted_cands.len() >= want_cands {
                        break;
                    }
                } else if wanted_cands.len() < want_cands {
                    wanted_cands.push(((layer, expert), record));
                }
            }
            drop(reuse);
            for (staged @ (layer, expert), record) in unwanted_cands.into_iter().chain(wanted_cands) {
                if released >= short {
                    break;
                }
                let exit = self.exit_now(group, record, staged);
                self.bank.release(group, BankRecord(record as u32), at.into(), exit);
                self.left_bank(layer, expert, None);
                released += 1;
            }
            if released >= short {
                return released;
            }
            // Rare fallback (e.g. >8 write-back skips on a 3-tier bank): scan
            // remaining eligible experts for the remaining shortfall.
        } else {
            drop(reuse);
        }
        let members = self.homes[group].iter().map(|&(_, layer)| layer).collect::<Vec<_>>();
        let reuse = self.reuse_now();
        let ranker = reuse.ranker(group, Tier::COMPUTE, Reader::Move);
        // Prefer unwanted held experts first; if fewer than `short`, consider
        // lowest-ranked held experts in non-whole layers of this group.
        let mut candidates: Vec<(bool, _, Staged, usize)> = Vec::new();
        for &layer in &members {
            if self.is_layer_whole_published(layer) {
                continue;
            }
            for expert in 0..self.bank.experts(layer) {
                let Some(record) = self.bank.held(layer, expert) else {
                    continue;
                };
                if !self.bank.usable_when(group, record, lending) || !can_reuse(record) {
                    continue;
                }
                let is_wanted = !self.unwanted[layer].contains(&expert);
                candidates.push((is_wanted, self.ranked(ranker, layer, expert, None), (layer, expert), record));
            }
        }
        drop(reuse);
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (_, _, staged @ (layer, expert), record) in candidates {
            if released >= short {
                break;
            }
            let exit = self.exit_now(group, record, staged);
            // Skip victims that require a write-back copy into HostTier;
            // demand replenishment only needs fast stamped-unmap releases.
            if exit.back().is_some() {
                continue;
            }
            self.bank.release(group, BankRecord(record as u32), at.into(), exit);
            self.left_bank(layer, expert, None);
            released += 1;
        }
        released
    }

    /// Zero-recopy O(1) completion of a `Record::DemandLanding` slot:
    /// since the expert's bytes are ALREADY in VRAM (`copy_hits = 0.0`), if
    /// `Reuse` ranks `(layer, expert)` in `in_top` or at/above the pool's
    /// minimum held `uses`, transition `Record::DemandLanding -> Record::Held`
    /// in place and publish via `StampedMap::map_covering` under
    /// `steps.unresolved(progress, ...)`, then `tier.banked` + `resettle`.
    /// Otherwise return to `Record::Free` (or defer via `defer_demand_release`
    /// if a joined consumer still pins the slot). Promotion only proceeds while
    /// `free_now >= free_floor` so remaining layers in the current step never
    /// starve of immediate `Record::Free` slots; [`Self::replenish_demand_free`]
    /// maintains `demand_replenish_target` (`2 * free_floor`) so ranked landings
    /// promote in place each step while preserving the floor.
    pub fn finish_demand_landing(
        &mut self,
        group: usize,
        record: usize,
        staged: Option<Staged>,
        progress: Point,
        steps: &Steps,
        can_reuse: bool,
    ) -> bool {
        let free_floor = self.demand_free_floor(group);
        let replenish_target = self.demand_replenish_target(group);
        let free_now = self.bank.free(group).count();
        let should_promote = if let Some(staged @ (layer, expert)) = staged {
            if self.bank.held(layer, expert).is_some()
                || self.bank.placing_contains(staged)
                || free_now < free_floor
            {
                false
            } else {
                let reuse = self.reuse_now();
                let in_top = reuse.in_top(layer, expert);
                if in_top {
                    true
                } else {
                    // With `copy_hits = 0.0` (bytes already in VRAM!), keep
                    // the expert if its expected uses meet or exceed the
                    // lowest-valued usable held expert in this group (or if
                    // the group has free capacity past `replenish_target`).
                    let ranker = reuse.ranker(group, Tier::COMPUTE, Reader::Move);
                    let uses = self.uses(ranker, staged);
                    let members = self.homes[group].iter().map(|&(_, l)| l).collect::<Vec<_>>();
                    let held_total: usize = members.iter().map(|&l| self.bank.held_count(l)).sum();
                    if held_total + replenish_target < self.bank.capacity(group) {
                        uses >= 0.0
                    } else {
                        let min_floor = if self.prompt_reads.is_empty() {
                            reuse
                                .top_order(group)
                                .iter()
                                .find(|&&(_, l, e)| {
                                    !self.is_layer_whole_published(l) && self.unwanted[l].contains(&e)
                                })
                                .map(|&(_, l, e)| self.uses(ranker, (l, e)))
                        } else {
                            members
                                .iter()
                                .filter(|&&l| !self.is_layer_whole_published(l))
                                .flat_map(|&l| self.unwanted[l].iter().map(move |&e| (l, e)))
                                .map(|s| self.uses(ranker, s))
                                .reduce(f64::min)
                        };
                        min_floor.is_some_and(|floor| uses >= floor)
                    }
                }
            }
        } else {
            false
        };

        if should_promote {
            let staged @ (layer, expert) = staged.expect("should_promote implies Some(staged)");
            let bank = &mut self.bank;
            let promoted = steps.unresolved(progress, |unresolved| {
                bank.promote_demand_landing(group, record, staged, |l| unresolved.from(l))
            });
            if promoted {
                self.time.lock().moved(group, Tier::COMPUTE);
                if !self.bank.lends(group, record) {
                    self.tier(layer).banked(staged);
                }
                self.resettle(layer, expert);
                return true;
            }
        }

        if can_reuse {
            self.bank.release_demand_landing(group, record);
        } else {
            self.bank.defer_demand_release(group, record);
        }
        false
    }

    /// Release an aborted/faulted `Record::DemandLanding` slot back to
    /// `Record::Free` (fault quarantine in `ExpertTransport` prevents reuse
    /// until explicitly cleared).
    pub fn release_demand_landing(&mut self, group: usize, record: usize) {
        self.bank.release_demand_landing(group, record);
    }

    /// Drain deferred unpromoted `Record::DemandLanding` slots once
    /// `can_reuse(group, slot)` confirms no consumer lease pins them.
    pub fn drain_demand_landings(&mut self, can_reuse: impl FnMut(usize, usize) -> bool) -> bool {
        self.bank.drain_demand_landings(can_reuse)
    }

    /// The map word of `layer`'s expert 0 in its group's map: the word the
    /// device reads it at (`policy[POLICY_HOME]`, `Residency::publish_policy`).
    pub fn home(&self, layer: usize) -> usize {
        let group = self.bank.group(layer);
        self.homes[group].iter().find(|&&(_, member)| member == layer).expect("a layer is its group's").0
    }

    /// Per layer, the experts its group's host tier holds alone; none
    /// without a tier. Excludes dual-resident experts held in the bank's
    /// lent scratch slice (`0..lent`), whose host copies are intentionally
    /// retained as `Kept::Alone` so prefill leases do not drop them to NVMe.
    pub fn alone(&self, layer: usize) -> usize {
        let group = self.bank.group(layer);
        let host = self.host(group);
        let raw = host.alone()[layer];
        let lent = self.bank.seeded_from(group);
        if lent == 0 || raw == 0 {
            return raw;
        }
        let records = self.bank.records(group);
        let dual_resident = records[..lent.min(records.len())]
            .iter()
            .filter(|record| matches!(**record, Record::Held(staged) if staged.0 == layer && host.holds_alone(staged)))
            .count();
        raw - dual_resident
    }

    /// Back group `group`'s host `records`, holes or past every record:
    /// ready from now, the ranking's pool too where the tier is the one
    /// store.
    pub fn grow_host(&mut self, group: usize, records: Vec<u32>, ready_at: Step) {
        let before = self.host[group].records();
        self.host[group].grow(records, ready_at);
        self.resize_pool(group);
        if self.host[group].records() != before {
            self.resized(group, self.host_tier());
        }
    }

    /// The host records group `group` evicts to hold `target` records
    /// ([`HostRecords::evictable`]), changing nothing yet: the most valuable
    /// kept where their bytes are, by the store's observer rank (design
    /// §3.2, §4: `Reader::Observe`). None where it holds no more than
    /// `target`, and fewer than it lacks where too few are evictable.
    pub fn host_evictions(&self, group: usize, target: u32) -> Vec<u32> {
        self.host_evictions_filtered(group, target, false)
    }

    pub fn host_evictions_filtered(&self, group: usize, target: u32, skip_protected: bool) -> Vec<u32> {
        let excess = self.host[group].records().saturating_sub(target) as usize;
        let reuse = Reuse::of(&self.ranking);
        let rank = store_rank(reuse.ranker(group, self.host_tier(), Reader::Observe), &self.prompt_reads);
        let mut evicted = self.host[group].evictable_filtered(rank, skip_protected);
        evicted.truncate(excess);
        evicted
    }

    /// Evict group `group`'s host `records` ([`Self::host_evictions`],
    /// nothing changed since), beside running steps or not: holes from now,
    /// nothing moved, each kept expert unmapped from `until(expert)` on
    /// where given, else from the first step not yet begun
    /// ([`HostRecords::evict`]). The ranking's pool shrinks too where the
    /// tier is the one store.
    pub fn evict_host(&mut self, group: usize, records: &[u32], until: impl Fn(Staged) -> Option<UnmapStep>) {
        let before = self.host[group].records();
        self.host[group].evict(records, until);
        self.resize_pool(group);
        if self.host[group].records() != before {
            self.resized(group, self.host_tier());
        }
    }

    /// The bank no longer holds `layer`'s `expert`, last read at `read`,
    /// gone by its exit ([`Self::exit`]): its host tier holds it alone where
    /// a record keeps its copy, ranked beside that read; where its exit
    /// writes it back, the tier holds it once the copy ends
    /// ([`Self::written_back`]); otherwise only the file holds it.
    fn left_bank(&mut self, layer: usize, expert: usize, read: Option<Step>) {
        self.tier(layer).evicted((layer, expert), read);
        self.resettle(layer, expert);
    }

    /// Hold every expert of every layer at its home record
    /// ([`Self::home`]): a bank imported whole, every record committed.
    pub fn hold_homes(&mut self) {
        let homes = (0..self.bank.layers()).map(|layer| self.home(layer)).collect::<Vec<_>>();
        self.bank.hold_homes(|layer| homes[layer]);
        for layer in 0..self.bank.layers() {
            for expert in 0..self.bank.experts(layer) {
                self.resettle(layer, expert);
            }
        }
    }

    /// Where the host places (the one store), a bank imported whole: every
    /// expert kept in its group's records at its home ([`Self::home`]), as
    /// its slot maps name them, ranked as unread.
    pub fn keep_homes(&mut self) {
        for layer in 0..self.bank.layers() {
            let home = self.home(layer);
            for expert in 0..self.bank.experts(layer) {
                let record = u32::try_from(home + expert).expect("bank records fit u32");
                self.tier(layer).keep(record, (layer, expert));
            }
        }
    }

    /// The record holding expert `expert` of `layer`.
    pub fn held(&self, layer: usize, expert: usize) -> Option<usize> {
        self.bank.held(layer, expert)
    }

    /// Whether bank record `record` of group `group` is lent to a prompt
    /// group.
    #[cfg(test)]
    pub fn lent(&self, group: usize, record: usize) -> bool {
        matches!(self.bank.records(group)[record], Record::Lent { .. })
    }

    /// The bank record step `step`'s pass reads `layer`'s `expert` in, as
    /// the device resolves its map ([`BankRecords::resolve`]): beside a host
    /// tier, a prompt group's hits.
    pub fn resolve(&self, layer: usize, expert: usize, step: Step) -> Option<BankRecord> {
        self.bank.resolve(layer, expert, step)
    }

    /// The count of changes to what any layer holds, in the bank or its
    /// host tier alone ([`Self::held_count`], [`Self::alone`]): equal
    /// counts, equal holdings.
    pub fn holdings(&self) -> u64 {
        self.bank.holdings() + self.host.iter().map(HostRecords::holdings).sum::<u64>()
    }

    /// The experts `layer` holds.
    pub fn held_count(&self, layer: usize) -> usize {
        (0..self.bank.experts(layer)).filter(|&expert| self.bank.held(layer, expert).is_some()).count()
    }

    /// Grant scratch group `scratch_group` its lease of group `lender`'s
    /// lent records: its host records' (the conveyor's, and on the one
    /// store the scratch's, [`HostRecords::lease`]), record by record, each
    /// kept one's expert unmapped from the group's first step on before
    /// any read lands in it, a record waiting while a copy reads it or a
    /// landing or grace holds it, so each call claims what it can and the
    /// transitions that free them grant the rest; and beside a host tier
    /// the bank's, the scratch's, at once ([`BankRecords::lease`]), each
    /// held one's expert leaving the bank. The pool shrinks by the host
    /// records claimed now ([`Self::resize_pool`]). Whether the group holds
    /// the records its scratch lies in ([`Self::leased`]).
    pub fn lease(&mut self, lender: usize, scratch_group: ScratchGroup, may_evict: bool, copy: Option<&mut dyn FnMut(u32, u32)>) -> bool {
        self.lease_records(lender, scratch_group, None, may_evict, copy)
    }

    /// Grant scratch group `scratch_group` its lease of `target_records` (or
    /// all `0..lent` when `None`) of group `lender`.
    pub fn lease_records(
        &mut self,
        lender: usize,
        scratch_group: ScratchGroup,
        target_records: Option<&[u32]>,
        may_evict: bool,
        copy: Option<&mut dyn FnMut(u32, u32)>,
    ) -> bool {
        let before = self.host[lender].claimed();
        if !self.host[lender].leased_records(target_records) {
            let tier = self.host_tier();
            if copy.is_some() && self.host[lender].capacity() > 0 {
                let reuse = Reuse::of(&self.ranking);
                let rank = store_rank(reuse.ranker(lender, tier, Reader::Move), &self.prompt_reads);
                self.host[lender].lease_records(scratch_group, target_records, rank, may_evict, copy);
            } else {
                self.host[lender].lease_records(scratch_group, target_records, |_, _| 0u64, false, None);
            }
        }
        if self.host[lender].claimed() != before {
            self.resize_pool(lender);
        }
        let bank_before = self.bank.capacity(lender);
        for (group, record) in self.bank.lent_queued() {
            self.unpromoted(group, record);
        }
        // Each decided after the exits before it took their tier records.
        let exits = self.bank.lent_held().into_iter().map(|(group, record, staged)| self.exit_now(group, record, staged)).collect();
        for (layer, expert) in self.bank.lease(scratch_group, exits) {
            self.left_bank(layer, expert, None);
        }
        if self.bank.capacity(lender) != bank_before {
            self.resize_pool(lender);
        }
        self.leased_records(lender, target_records)
    }

    /// Whether bank record `record` of `group` lies in the prompt scratch
    /// lent slice `0..lent`.
    pub fn bank_lends(&self, group: usize, record: usize) -> bool {
        self.bank.lends(group, record)
    }

    /// Whether the prompt group holds the lent records of group `lender`
    /// its scratch lies in: on the one store its host records'
    /// ([`HostRecords::leased`]), beside a host tier the bank's
    /// ([`BankRecords::leased`]).
    pub fn leased(&self, lender: usize) -> bool {
        self.leased_records(lender, None)
    }

    /// Whether the prompt group holds `target_records` (or all `0..lent` when
    /// `None`) claimed for `lender`.
    pub fn leased_records(&self, lender: usize, target_records: Option<&[u32]>) -> bool {
        match self.placer {
            crate::planning::Placer::Bank => self.host[lender].leased_records(target_records),
            crate::planning::Placer::Tier => self.bank.leased(),
        }
    }

    /// The kept expert currently held in host record `record` of `group`, if any.
    pub fn host_kept_at(&self, group: usize, record: u32) -> Option<Staged> {
        self.host[group].kept_at(record)
    }

    /// Number of lent host records `[0, lent)` in group `group`.
    pub fn host_lent(&self, group: usize) -> u32 {
        self.host[group].lent()
    }

    /// Whether the prompt group's lease holds conveyor record `record` of
    /// group `group`'s host records claimed.
    pub fn lent_claimed(&self, group: usize, HostRecord(record): HostRecord) -> bool {
        self.host(group).is_claimed(record)
    }

    /// The prompt group past (the device past it, and its conveyor quiet)
    /// returns its lease empty: every host record of group `lender` it
    /// claimed ([`HostRecords::return_lease`]), its pool back to the whole
    /// decode capacity, and every bank record lent to it free
    /// ([`BankRecords::unlend`]). Its pending reads
    /// ([`Self::read_pending`]) are forgotten: the ranking logs the group's
    /// reads once its steps end, which may come after this, so until then
    /// they rank as the ranking last logged them.
    pub fn end_prompt_group(&mut self, lender: usize) {
        self.host[lender].return_lease();
        self.bank.unlend();
        self.resize_pool(lender);
        if self.bank.lent().is_some() {
            self.stirred = true;
        }
        self.prompt_reads.clear();
    }

    /// Member layers of `group` from `layer` onwards (including `layer`), at
    /// least 1: free records during a prompt pass are divided across these so
    /// an earlier layer's unretired seeds do not starve later layers of the
    /// same group before they run.
    fn remaining_layers(&self, group: usize, layer: usize) -> usize {
        self.homes[group].iter().filter(|&&(_, member)| member >= layer).count().max(1)
    }

    /// Seed `layer` at `point`, a point of scratch group `scratch_group`,
    /// with what the group routed there: `routed`, each expert with its
    /// count in the group's union and the step that last routed it. Every
    /// placer seeds so, by the device's seed copy out of the expert's
    /// scratch slot, which its rounds converted into whether it was read
    /// into a conveyor record or a slot, and by the one seed policy
    /// ([`seeding`], design §3.7): each routed expert the layer lacks whose
    /// scratch slot the group filled (`filled`), ranked by the prompt table
    /// with prior = its count share (`Reuse::rank_union`), takes free
    /// records first (divided across the group's remaining layers so an
    /// earlier layer's unretired seeds do not starve later layers), then the
    /// group's kept records of any layer, least ranked first, while it
    /// outranks them. Capacity is pooled across the group's layers. The
    /// layer's `pinned` experts (those its merge set's pass reads in place
    /// after the seeding) are never taken. Where the host places (the one
    /// store), into its host records past those it keeps ready for its
    /// `reserve` ([`Self::seed_store`]); else into the bank's records,
    /// taking no record whose expert lies inside the ranking's pool
    /// (`Reuse::in_top`, as a release's), placed as fills its seed copy and
    /// commit commit. The seeds with their records, and the experts evicted,
    /// (layer, expert).
    #[allow(clippy::too_many_arguments)]
    pub fn seed(&mut self, point: Point, scratch_group: ScratchGroup, layer: usize, routed: &[(usize, u32, Step)], pinned: &[usize], filled: impl Fn(usize) -> bool) -> (Vec<(usize, BankRecord)>, Vec<(usize, usize)>) {
        if self.placer == crate::planning::Placer::Bank {
            return self.seed_store(point, scratch_group, layer, routed, pinned, filled);
        }
        let group = self.bank.group(layer);
        let remaining = self.remaining_layers(group, layer);
        let taken = {
            let reuse = self.reuse_now();
            let ranks = reuse.rank_union(layer, routed);
            let mut reads = self.record_reads(&reuse, group);
            for &(expert, _, recency) in routed {
                if let Some(record) = self.bank.held(layer, expert) {
                    reads[record] = reads[record].max(recency);
                }
            }
            let absent = routed
                .iter()
                .zip(ranks)
                .filter(|(&(expert, _, _), _)| self.bank.held(layer, expert).is_none() && !self.bank.placing((layer, expert)) && filled(expert))
                .map(|(&(expert, _, _), rank)| Ranked { rank, layer, expert })
                .collect::<Vec<_>>();
            let pinned = pinned.iter().filter_map(|&expert| self.bank.held(layer, expert)).collect::<Vec<_>>();
            let contents = self.bank.records(group);
            // Protect `in_top` records that belong to the *current* request's
            // prefill (`!has_stale_decode_pool(owner)`) so multi-chunk prefills
            // never thrash VRAM-only experts from earlier chunks back to SSD;
            // allow evicting `in_top` records that are stale from a previous
            // request's decode (`has_stale_decode_pool(owner)`).
            let pooled = contents
                .iter()
                .map(|content| {
                    matches!(
                        *content,
                        Record::Held((owner, expert)) if !reuse.has_stale_decode_pool(owner) && reuse.in_top(owner, expert)
                    )
                })
                .collect::<Vec<_>>();
            let backs = self.kept_by_exit(&reuse, group);
            let kept = |record: usize| {
                pinned.contains(&record)
                    || pooled[record]
                    || backs[record]
                    || matches!(contents[record], Record::Held((layer, _)) if self.is_layer_whole_published(layer))
            };
            let free_limit = self.bank.free(group).filter(|&record| !kept(record)).count().div_ceil(remaining);
            let mut order = self
                .bank
                .free(group)
                .filter(|&record| !kept(record))
                .take(absent.len().min(free_limit))
                .map(|record| (None, bank_record(record)))
                .collect::<Vec<_>>();
            if order.len() < absent.len() {
                let routed_total: f64 = routed.iter().map(|&(_, count, _)| f64::from(count)).sum();
                let rank = seed_rank(
                    &reuse,
                    reuse.ranker(group, Tier::COMPUTE, Reader::Move),
                    &self.prompt_reads,
                    layer,
                    routed,
                    routed_total,
                    scratch_group.first(),
                );
                let mut held = self
                    .bank
                    .records(group)
                    .iter()
                    .enumerate()
                    .filter_map(|(record, content)| match *content {
                        Record::Held((owner, expert)) if !kept(record) => {
                            let step = (reads[record] != 0).then_some(reads[record]);
                            Some((Some(Ranked { rank: rank((owner, expert), step), layer: owner, expert }), bank_record(record)))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                // Cap per-layer evictions across `remaining` layers just like
                // `free_limit` so early layers' unretired `Record::Placing`
                // seeds never starve later layers during a full-bank prefill.
                let evict_limit = held.len().div_ceil(remaining);
                let short = (absent.len() - order.len()).min(evict_limit);
                if short < held.len() {
                    held.select_nth_unstable(short);
                    held.truncate(short);
                }
                held.sort_unstable();
                order.extend(held);
            }
            let (seeded, records): (Vec<_>, Vec<_>) = seeding(absent, order, |held, seed| {
                (held.rank.expected(), held.rank.last().is_some()) < (seed.rank.expected(), seed.rank.last().is_some())
            })
            .into_iter()
            .map(|(seed, record)| (seed.expert, record))
            .unzip();
            (seeded, self.leaving_of(&reuse, layer, records))
        };
        let (seeded, records) = taken;
        let placed = self.place_into(point, scratch_group, layer, &seeded, records);
        let evicted = placed.evicted.iter().map(|&(layer, expert, _)| (layer, expert)).collect();
        (seeded.into_iter().zip(placed.records).collect(), evicted)
    }

    /// [`Self::seed`] where the host places, into the one store: seeds rank
    /// by the prompt table (`Reuse::rank_union`), and the store's taking
    /// order ([`HostRecords::seed_order`]) offers free records first (divided
    /// across the group's remaining layers), then kept records of any layer
    /// in the group. However full the store, a seed that outranks a kept
    /// expert evicts it: seeding never waits for a ready record. The ready
    /// records the host keeps for its `reserve` are never taken. Each seed's
    /// record is the device's seed copy's to write ([`HostRecords::seed`])
    /// until it retires with its step ([`Self::retire`]); its expert stays
    /// lacked meanwhile, so no step waits on it. The seeds with their
    /// records, and the experts evicted, (layer, expert).
    #[allow(clippy::too_many_arguments)]
    fn seed_store(&mut self, point: Point, scratch_group: ScratchGroup, layer: usize, routed: &[(usize, u32, Step)], pinned: &[usize], filled: impl Fn(usize) -> bool) -> (Vec<(usize, BankRecord)>, Vec<(usize, usize)>) {
        assert_eq!(point.group, scratch_group.first(), "a seed at {point:?} of the scratch group from step {}", scratch_group.first());
        let group = self.bank.group(layer);
        let remaining = self.remaining_layers(group, layer);
        let taken = {
            let reuse = self.reuse_now();
            let ranks = reuse.rank_union(layer, routed);
            let routed_total: f64 = routed.iter().map(|&(_, count, _)| f64::from(count)).sum();
            let rank = seed_rank(
                &reuse,
                reuse.ranker(group, Tier::COMPUTE, Reader::Move),
                &self.prompt_reads,
                layer,
                routed,
                routed_total,
                scratch_group.first(),
            );
            let host = &self.host[group];
            let absent = routed
                .iter()
                .zip(ranks)
                .filter(|(&(expert, _, _), _)| host.located((layer, expert)) == Located::Lacked && filled(expert))
                .map(|(&(expert, _, _), rank)| (Kept::Alone, rank, (layer, expert)))
                .collect::<Vec<_>>();
            let pinned = |(of, expert): Staged| of == layer && pinned.contains(&expert);
            let mut order = host.seed_order(rank, pinned);
            let free = order.iter().take_while(|(held, _)| held.is_none()).count();
            let free_limit = free.div_ceil(remaining);
            if free_limit < free {
                order.drain(free_limit..free);
            }
            seeding(absent, order, |(held_kept, held_rank, _), (seed_kept, seed_rank, _)| {
                (held_kept, held_rank.expected(), held_rank.last().is_some()) < (seed_kept, seed_rank.expected(), seed_rank.last().is_some())
            })
        };
        let mut seeds = Vec::with_capacity(taken.len());
        let mut evicted = Vec::new();
        for ((.., staged), (_, record)) in taken {
            evicted.extend(self.host[group].seed(staged, record, scratch_group));
            self.seeding.push((group, record, staged, point));
            seeds.push((staged.1, BankRecord(record)));
        }
        (seeds, evicted)
    }

    /// A prompt group in flight routes `layer`'s `expert` at step `read`,
    /// which the ranking logs when the group ends: until then every taking,
    /// on either placer, ranks it beside that read, a use at its step.
    pub fn read_pending(&mut self, layer: usize, expert: usize, read: Step) {
        let pending = self.prompt_reads.entry((layer, expert)).or_insert(Unlogged::at(read));
        *pending = (*pending).max(Unlogged::at(read));
    }

    /// The device is at `progress`: every seed placed before it whose step
    /// and group have ended (`ending`, of its point) is retired, each placed
    /// by a step that ended whole holding its expert where the device
    /// committed it (`commit(group, record)` holds the step's low word),
    /// banking any copy its host tier keeps; any other leaves its record
    /// free: a tainted one's (a wait of its step or an earlier step of its
    /// group ended on its still pulse, so its seed copy may have read a
    /// slot whose round was never answered) whatever the device committed.
    /// A seed of a group still running stays pending, though the device is
    /// past its layer: its taint is known only once it ends. Every tier
    /// record graced for steps the device is past at the record's layer
    /// is ready. The seeds committed, (layer, expert, record), which the
    /// host maps.
    pub fn retire(&mut self, progress: Point, commit: impl Fn(usize, BankRecord) -> i32, ending: impl Fn(Point) -> Ending, steps: &Steps) {
        // Only the records' change and its mapping under the steps' lock;
        // what the tier and the ranking learn of it after.
        let retired = match self.placer {
            crate::planning::Placer::Bank => self.retire_seeds(progress, commit, ending, steps),
            crate::planning::Placer::Tier => self.bank.retire_fills(progress, commit, ending, steps),
        };
        for (group, (layer, expert), committed) in retired {
            if committed && self.placer == crate::planning::Placer::Tier {
                self.host[group].banked((layer, expert));
            }
            self.resettle(layer, expert);
        }
        self.retire_graces(progress, steps);
    }

    /// [`Self::retire`]'s seeds where the host places, into the one store
    /// ([`Self::seed_store`]): each retired from its host record
    /// ([`HostRecords::seeded`]), ranked beside its step. Each retired,
    /// (group, expert, whether the store keeps it).
    fn retire_seeds(&mut self, progress: Point, commit: impl Fn(usize, BankRecord) -> i32, ending: impl Fn(Point) -> Ending, steps: &Steps) -> Vec<(usize, Staged, bool)> {
        let mut retiring = Settling::new(progress, ending).retire(&mut self.seeding, |&(.., point)| point, |&(group, record, ..)| commit(group, BankRecord(record)));
        if !retiring.is_empty() {
            let host = &mut self.host;
            steps.unbegun(|at| {
                for ((group, record, staged, point), committed) in &mut retiring {
                    *committed = host[*group].seeded(*record, *staged, point.sequence, *committed, at);
                }
            });
        }
        retiring.into_iter().map(|((group, _, staged, _), kept)| (group, staged, kept)).collect()
    }

    /// The first group a pending seed was placed in: the taints of its steps
    /// and every later one are still to be read ([`Self::retire`]).
    pub fn pending_from(&self) -> Option<Step> {
        let store = self.seeding.iter().map(|&(.., point)| point.group).min();
        store.into_iter().chain(self.bank.pending_from()).min()
    }

    /// The device is parked at `park`, a decode request (its landing launch),
    /// so every resolution before it in the device's order is done: every
    /// graced record ready but those the parked step resolved there
    /// (`read`, (group, record), `decode::parked_reads`) and those a step
    /// of a group the parked one was cut in before may still resolve, each
    /// one a stamped map may still name first restamped
    /// ([`HostRecords::release`]).
    pub fn release(&mut self, park: &Park, read: &[(usize, u32)]) {
        for (group, records) in self.host.iter_mut().enumerate() {
            records.release(park, |record| read.contains(&(group, record)));
        }
    }

    /// The device is at `progress`: every host tier record graced for
    /// steps each past the record's layer there, or ended, is ready
    /// ([`HostRecords::retire`]).
    pub fn retire_graces(&mut self, progress: Point, steps: &Steps) -> bool {
        self.host
            .iter_mut()
            .fold(false, |freed, records| freed | records.retire(progress, steps))
    }

    /// Place `absent` experts of `layer` at `point`, a point of scratch group
    /// `scratch_group`, keeping the records of its `held` experts, in the eviction
    /// order of [`Self::least`], each held record ranked beside
    /// `read(record)`: a prompt group's seeds, which its seed commit
    /// commits (and the miss replay's model of a bank whose every miss is
    /// placed). The tier learns each evicted expert ([`Self::left_bank`]).
    /// `None` when the group has fewer such records than `absent` (beside
    /// those promotions take).
    pub fn place(
        &mut self,
        point: Point,
        scratch_group: ScratchGroup,
        layer: usize,
        held: &[usize],
        absent: &[usize],
        read: impl Fn(usize) -> Step,
    ) -> Option<Placed> {
        let candidates = {
            let reuse = self.reuse_now();
            let candidates = self.least(&reuse, layer, held, absent.len(), &read);
            self.leaving_of(&reuse, layer, candidates)
        };
        (candidates.len() == absent.len()).then(|| self.place_into(point, scratch_group, layer, absent, candidates))
    }

    /// Place `absent` experts of `layer` at `point`, keeping the records of
    /// its `held` experts, evicting free records first and then held records
    /// ordered by `key(record)` alone (least first, never reading `Reuse` or
    /// the clock): used by offline replay's `Recency::NextRead` bound.
    #[cfg(any(test, feature = "expert-routing"))]
    pub(super) fn place_by<K: Ord + Copy>(
        &mut self,
        point: Point,
        scratch_group: ScratchGroup,
        layer: usize,
        held: &[usize],
        absent: &[usize],
        key: impl Fn(usize) -> K,
    ) -> Option<Placed> {
        let group = self.bank.group(layer);
        let kept: Vec<usize> = held.iter().filter_map(|&expert| self.bank.held(layer, expert)).collect();
        let count = absent.len();
        let mut chosen: Vec<(Option<(usize, usize)>, usize)> = self
            .bank
            .free(group)
            .filter(|record| !kept.contains(record))
            .take(count)
            .map(|record| (None, record))
            .collect();
        let short = count - chosen.len();
        if short > 0 {
            let mut order: Vec<((K, usize, usize), usize)> = self
                .bank
                .records(group)
                .iter()
                .enumerate()
                .filter_map(|(record, content)| match *content {
                    Record::Held((owner, expert)) if !kept.contains(&record) => Some(((key(record), owner, expert), record)),
                    _ => None,
                })
                .collect();
            if short < order.len() {
                order.select_nth_unstable(short);
                order.truncate(short);
            }
            order.sort_unstable();
            chosen.extend(order.into_iter().map(|((_, owner, expert), record)| (Some((owner, expert)), record)));
        }
        if chosen.len() < count {
            return None;
        }
        let mut evicted = Vec::new();
        let mut records = Vec::with_capacity(count);
        for (&expert, (victim, record)) in absent.iter().zip(chosen) {
            let record = bank_record(record);
            let exit = victim.map(|staged| Exit { staged, back: None });
            let got = self.bank.fill(point, scratch_group, (layer, expert), record, exit);
            assert_eq!(got, victim, "a placement evicts what its record holds");
            self.record_eviction(&mut evicted, victim.map(|staged| (staged, None)));
            self.resettle(layer, expert);
            records.push(record);
        }
        Some(Placed { records, evicted })
    }

    /// Place `absent` experts of `layer` at `point`, a point of scratch group
    /// `scratch_group`, into `candidates`, one each, as [`Self::least`] ranks them:
    /// each held one's expert evicted.
    fn place_into(&mut self, point: Point, scratch_group: ScratchGroup, layer: usize, absent: &[usize], candidates: Vec<(Option<(Ranked, Leaving)>, BankRecord)>) -> Placed {
        let mut evicted = Vec::new();
        let mut records = Vec::with_capacity(absent.len());
        let group = self.bank.group(layer);
        for (&expert, (held, record)) in absent.iter().zip(candidates) {
            // A seed overwrites its victim's bytes: no record whose victim's
            // write-back pays is a candidate ([`Self::kept_by_exit`]), so
            // its exit writes nothing back.
            let exit = held.map(|(_, leaving)| self.exit(group, record.index(), leaving));
            let held = held.map(|(held, _)| held);
            let victim = self.bank.fill(point, scratch_group, (layer, expert), record, exit);
            assert_eq!(victim, held.map(|held| (held.layer, held.expert)), "a seed evicts what its record holds");
            self.record_eviction(&mut evicted, held.map(|Ranked { rank, layer, expert }| ((layer, expert), rank.last())));
            self.resettle(layer, expert);
            records.push(record);
        }
        Placed { records, evicted }
    }

    /// Record an expert a placement evicted from the bank (if any), with its
    /// last logged read step (`0` in `evicted` where unlogged), and notify the
    /// host tier ([`Self::left_bank`]).
    fn record_eviction(&mut self, evicted: &mut Vec<(usize, usize, Step)>, victim: Option<(Staged, Option<Step>)>) {
        if let Some(((owner, expert), read)) = victim {
            evicted.push((owner, expert, read.unwrap_or(0)));
            self.left_bank(owner, expert, read);
        }
    }

    /// The `count` records of `layer`'s group a placement takes first (fewer
    /// when the group has fewer), keeping those of its `kept` experts, each
    /// with its eviction rank, least first: free records first (`None`), in
    /// record order (they hold nothing, so which one a fill takes decides
    /// nothing later), then held ones by their expert's rank beside
    /// `read(record)`, the step that last read it ([`Ranked`]), all by
    /// `reuse`. The order is a function of
    /// what the records hold and when they were read, never of where they
    /// lie. None the conveyor claimed ([`Self::claim`]). The held records
    /// are ranked only when the free ones fall short.
    fn least(
        &self,
        reuse: &Reuse,
        layer: usize,
        kept: &[usize],
        count: usize,
        read: impl Fn(usize) -> Step,
    ) -> Vec<(Option<Ranked>, BankRecord)> {
        // A kept expert's record holds it, so no free record is kept.
        let kept = kept
            .iter()
            .filter_map(|&expert| self.bank.held(layer, expert))
            .collect::<Vec<_>>();
        let group = self.bank.group(layer);
        let backs = self.kept_by_exit(reuse, group);
        self.least_of(reuse, group, count, read, |record| kept.contains(&record) || backs[record])
    }

    /// Which of group `group`'s records hold an expert whose exit writes it
    /// back ([`Exits::leaving`]): what a seed never takes, as it overwrites
    /// the record's bytes at once. Such an expert leaves by a release
    /// instead, which keeps them until its write-back ends.
    fn kept_by_exit(&self, reuse: &Reuse, group: usize) -> Vec<bool> {
        let exits = self.exits(reuse, group);
        self.bank.records(group).iter().map(|content| matches!(*content, Record::Held(staged) if exits.leaving(staged).backs())).collect()
    }

    /// Each of `layer`'s group's `candidates` with its held expert's exit,
    /// decided by `reuse`, the ranking that chose them.
    fn leaving_of(&self, reuse: &Reuse, layer: usize, candidates: Vec<(Option<Ranked>, BankRecord)>) -> Vec<(Option<(Ranked, Leaving)>, BankRecord)> {
        let exits = self.exits(reuse, self.bank.group(layer));
        candidates.into_iter().map(|(held, record)| (held.map(|held| (held, exits.leaving((held.layer, held.expert)))), record)).collect()
    }

    /// The one ranking of group `group`'s records, least first, that every
    /// choice of a record to take ranks by (a placement's, [`Self::least`],
    /// and a promotion's, [`Self::promotions`]): its `count` first (fewer
    /// when the group has fewer), free records first (`None`) in record
    /// order, then held ones by [`Ranked`] (their expert's rank by `reuse`
    /// beside `read(record)`); none
    /// claimed, filling or promoting, and none `excluded` names. The held
    /// records are ranked only when the free ones fall short.
    fn least_of(
        &self,
        reuse: &Reuse,
        group: usize,
        count: usize,
        read: impl Fn(usize) -> Step,
        excluded: impl Fn(usize) -> bool,
    ) -> Vec<(Option<Ranked>, BankRecord)> {
        let typed = |least: Vec<(Option<Ranked>, usize)>| least.into_iter().map(|(rank, record)| (rank, BankRecord(u32::try_from(record).expect("bank records fit u32")))).collect();
        let mut least = self.bank.free(group).filter(|&record| !excluded(record)).take(count).map(|record| (None, record)).collect::<Vec<_>>();
        let short = count - least.len();
        if short == 0 {
            return typed(least);
        }
        let ranker = reuse.ranker(group, Tier::COMPUTE, Reader::Move);
        let mut held = self
            .bank
            .records(group)
            .iter()
            .enumerate()
            .filter_map(|(record, content)| match *content {
                Record::Held((layer, expert)) if !excluded(record) => {
                    let step = match read(record) {
                        0 => None,
                        step => Some(step),
                    };
                    Some((Some(self.ranked(ranker, layer, expert, step)), record))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if short < held.len() {
            held.select_nth_unstable(short);
            held.truncate(short);
        }
        held.sort_unstable();
        least.extend(held);
        typed(least)
    }

    /// Each group's records the bank holds storage for: every one but the
    /// holes.
    pub fn records(&self) -> Vec<usize> {
        self.bank.counts()
    }

    /// The records group `group` evicts to hold `target` records
    /// ([`BankRecords::evictable`]), changing nothing yet: its free ones
    /// first, then those releasing, then the held ones least ranked first
    /// ([`Ranked`]) beside `last_read(layer, expert)` (the step that last
    /// read it, none before any: least recent), so it keeps its most reused
    /// experts where their bytes are. None where it holds no more than
    /// `target`, and fewer than it lacks where too few are evictable.
    pub fn evictions(&self, group: usize, target: usize, last_read: impl Fn(usize, usize) -> Option<Step>) -> Vec<usize> {
        self.evictions_filtered(group, target, last_read, false)
    }

    pub fn evictions_filtered(
        &self,
        group: usize,
        target: usize,
        last_read: impl Fn(usize, usize) -> Option<Step>,
        skip_protected: bool,
    ) -> Vec<usize> {
        self.evictions_filtered_with(group, target, last_read, skip_protected, |_| true)
    }

    /// [`Self::evictions_filtered`], additionally excluding any slot where
    /// `can_reuse(record)` is `false` (e.g. pinned by an active `ConsumerLease`
    /// or in-flight/quarantined transfer in `ExpertTransport`).
    pub fn evictions_filtered_with(
        &self,
        group: usize,
        target: usize,
        last_read: impl Fn(usize, usize) -> Option<Step>,
        skip_protected: bool,
        mut can_reuse: impl FnMut(usize) -> bool,
    ) -> Vec<usize> {
        let excess = self.bank.counts()[group].saturating_sub(target);
        let (mut idle, mut held) = (Vec::new(), Vec::new());
        for (record, content) in self.bank.evictable(group) {
            if !can_reuse(record) {
                continue;
            }
            match content {
                Record::Pending { .. } | Record::Free => idle.push((0, record)),
                Record::Releasing { .. } => idle.push((1, record)),
                Record::Held((layer, expert)) => {
                    if !(skip_protected && self.is_layer_whole_published(layer)) {
                        held.push((layer, expert, last_read(layer, expert), record));
                    }
                }
                _ => unreachable!("an evictable record is pending, free, releasing or held"),
            }
        }
        idle.sort_unstable();
        let mut evicted = idle.into_iter().map(|(_, record)| record).take(excess).collect::<Vec<_>>();
        if evicted.len() < excess {
            // Ranked by one hold of the ranking, after `last_read` (which
            // may hold it) has answered.
            let mut held = {
                let reuse = self.reuse_now();
                let ranker = reuse.ranker(group, Tier::COMPUTE, Reader::Observe);
                held.into_iter().map(|(layer, expert, read, record)| (self.ranked(ranker, layer, expert, read), record)).collect::<Vec<_>>()
            };
            held.sort_unstable();
            evicted.extend(held.into_iter().map(|(_, record)| record).take(excess - evicted.len()));
        }
        evicted
    }

    /// Evict each group's `records` (`records[group]`, [`Self::evictions`],
    /// nothing changed since), under the steps' lock (`at`): holes from
    /// now, nothing moved, each held expert leaving by its exit
    /// ([`Exits::leaving`] at the first step not yet begun), unmapped from
    /// `until(expert)` on ([`BankRecords::evict`]), and the host tier
    /// learning each, ranked by `last_read`. A shrink takes the storage of
    /// what it evicts, so no exit of one may write back: where any held
    /// expert's would, the shrink evicts nothing and waits a boundary,
    /// each such expert released into its write-back now
    /// ([`BankRecords::release`]), whose record it then evicts first,
    /// idle. While any write-back of a group it evicts is in flight, it
    /// waits on it, releasing nothing more. Whether it evicted.
    pub fn evict(&mut self, records: &[Vec<usize>], last_read: impl Fn(usize, usize) -> Option<Step>, until: impl Fn(Staged) -> UnmapStep, at: &Unbegun) -> bool {
        let backing = self.bank.backs_queued().into_iter().map(|(group, ..)| group).chain(self.bank.backing().into_iter().map(|(group, ..)| group)).collect::<BTreeSet<_>>();
        if records.iter().enumerate().any(|(group, records)| !records.is_empty() && backing.contains(&group)) {
            return false;
        }
        // Each decided after the exits before it took their tier records.
        let mut exits = records
            .iter()
            .enumerate()
            .map(|(group, records)| {
                records
                    .iter()
                    .map(|&record| match self.bank.records(group)[record] {
                        Record::Held(staged) => Some(self.exit_now(group, record, staged)),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let backs = exits.iter().flatten().flatten().any(|exit| exit.back().is_some());
        if backs {
            for (group, (exits, records)) in exits.iter_mut().zip(records).enumerate() {
                for (exit, &record) in exits.iter_mut().zip(records) {
                    if exit.as_ref().is_some_and(|exit| exit.back().is_some()) {
                        let (layer, expert) = self.bank.release(group, bank_record(record), at.into(), exit.take().expect("an exit that writes back"));
                        self.left_bank(layer, expert, None);
                    }
                }
            }
            return false;
        }
        for (group, (exits, records)) in exits.into_iter().zip(records).enumerate() {
            let before = self.bank.counts()[group];
            for (layer, expert) in self.bank.evict(group, records, &until, exits) {
                self.left_bank(layer, expert, last_read(layer, expert));
            }
            self.resize_pool(group);
            if self.bank.counts()[group] != before {
                self.resized(group, Tier::COMPUTE);
            }
        }
        true
    }

    /// The first record of group `group` a seed may take: what a seed list
    /// names its records from, past the lent ones (the conveyor and the
    /// scratch, [`HostRecords::lent`]) where the host places, else
    /// [`BankRecords::seeded_from`].
    pub fn seeded_from(&self, group: usize) -> usize {
        match self.placer {
            crate::planning::Placer::Bank => self.host[group].lent() as usize,
            crate::planning::Placer::Tier => self.bank.seeded_from(group),
        }
    }

    /// Back group `group`'s `records`, holes or past every record: free from
    /// now when `ready_at == 0`, else `Record::Pending { ready_at }` until
    /// `drain_pending(frontier)` observes `frontier >= ready_at`.
    pub fn grow(&mut self, group: usize, records: Vec<usize>, ready_at: Step) {
        let before = self.bank.counts()[group];
        self.bank.grow(group, records, ready_at);
        self.resize_pool(group);
        if self.bank.counts()[group] != before {
            self.resized(group, Tier::COMPUTE);
        }
    }

    /// Release any bank or host records whose device-ordered slab table repoint
    /// is ahead of every remaining running step (`ready_at <= frontier`).
    pub fn drain_pending(&mut self, frontier: Step) -> bool {
        let mut drained = self.bank.drain_pending(frontier);
        for host in &mut self.host {
            drained |= host.drain_pending(frontier);
        }
        drained
    }
}

#[cfg(test)]
mod tests;
