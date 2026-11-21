//! Staging, the landing pool, and the reads into it: each request's demand
//! reads, and reads ahead of demand, each landing as it ends
//! (`experts::landing`). A read ahead is one of two kinds: an interest's,
//! an expert whose file is on the path (nothing holds, keeps or lands it)
//! that a lookahead router's hint or an ahead row's route named
//! ([`Prefetch::rings`]) and whose read the interest ledger derives, orders
//! and cancels (`experts::interest`, [`Prefetch::drive`]); or a covering
//! tier's, an expert the tier lacks read at the load's priority below every
//! step's ([`Prefetch::cover`]) where the tier has a record for every
//! expert. A request publishes its own misses to the ledger as certain
//! interests ([`Prefetch::want`]), which hastens the read ahead it finds to
//! demand; a request that finds a cover's read of its miss waits for a read
//! no prediction made. A tier that cannot hold every expert keeps what the
//! steps' misses and the reads ahead land.
//!
//! Every read short of due is issued, repriced and cancelled in the
//! ledger's one order (`Ledger::schedule`): the soonest wanted step first,
//! then the likeliest used, as its link reads them (`seismic_io::Want`), so
//! where staging is short the reads issued are those the drive would read
//! first. A read ahead is issued only where its landing takes a free place
//! or displaces an expert of fewer expected uses within the turnover
//! (`Directory::claim_landing`, `Reuse::expected_with`): otherwise it adds a
//! move to the path.
//!
//! The interest ledger is Prefetch's own, on the request service's thread:
//! what other threads learn reaches it through its queue
//! (`interest::Told`), drained once the rings are read ([`Prefetch::rings`],
//! so every launch a ring names is told), so a request's exact interests
//! take no lock, and a due read is never priced (telemetry's lock is taken
//! only for a read short of due, [`Priced`]).
//!
//! Every read lands one expert, its own read engine request, so each
//! landing begins as its own read ends, however many its request reads.
//! A read's end lands it once the read's end is set and its record is
//! claimed ([`Join`]: a request's read is submitted before residency
//! claims it, and where the claim comes last the request service lands
//! it; each landing is tracked by its own `Ended` from its claim). Where the
//! host places into a tier, and for every read ahead, the host compute
//! executor converts the stored record into the tier record taken for it,
//! maps it, and answers the choices of the request that named it; where
//! the bank places, a request's read appends its entry to its layer's
//! landing list and answers, and the device's landing launch converts it
//! (`landing::DeviceLanding`), its DONE freeing the staging record as the
//! watcher reads it ([`Prefetch::settle`]), so a request of more misses than
//! free staging reads its later misses into the records its earlier ones
//! free while its launch runs. A read that failed fails exactly the rows of
//! the request whose choices named its expert (`Steps::fail_request_rows`),
//! before it answers them with no record, so before its step can complete.
//! A read ahead names no request and fails nothing: a request whose expert
//! it reads waits for its landing, hastened to demand, and reads it again
//! should it fail (`Decode::resume`). The watcher learns each landing once
//! it ended ([`Prefetch::settle`]): the tier record keeps its expert, or is
//! ready again, and the staging record frees, where the device's DONE has
//! not freed it already.

use super::Shared;
use super::core::{At, Core, ReadTable};
use crate::experts::Words;
use crate::experts::directory::{Directory, HostRecord, Point};
use crate::experts::host_records::{Claim, Grant, Located, Taking};
use crate::experts::compute::Urgency;
use crate::experts::interest::{Act, Admit, Class, Ledger, Name, Origin, Read as ReadState, Scheduled, Source, Told, When};
use super::Step;
use crate::experts::landing::{Answer, Converting, DeviceLanding, Ended};
use crate::experts::source::{Outcome, Read, Ticket};
use crate::experts::staging::{Staged, StagingPool};
use seismic_io::Want;
use magnitude_kernels::constants;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Which party of a read's join ran its landing ([`Join`]): the read's
/// end, on the read worker that ended it (which works its landing's pieces
/// beside the executor's), or the claim of its records, on the request
/// service (which leaves them to the executor).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Party {
    Read,
    Claim,
}

/// What lands a read once it ended and its records are claimed.
type Lands = Box<dyn FnOnce(Outcome, &[usize], Party) + Send>;

/// A read's join ([`Join`]): what the claims registered while it ran, or
/// how it ended.
enum Parties {
    Reading(Vec<Lands>),
    Ended(Outcome, Vec<usize>),
}

/// The join of a read that lands experts into the host tier: its end, and
/// each claim of the records some of its experts land into
/// (`Prefetch::demand`, `Prefetch::read_ahead`, and a request's accounting,
/// [`account`]). A request's read is submitted by the request service's
/// look before residency decides its tier records (`decode::Look`), so
/// either may come first: whichever is second runs each landing, so
/// neither waits for the other, and an expert whose record is short never
/// holds back its request's others.
pub(super) struct Join(Mutex<Parties>);

impl Join {
    fn parties(&self) -> std::sync::MutexGuard<'_, Parties> {
        self.0.lock().expect("a read's join is never left by a panic")
    }

    /// Its read ended as `outcome`, the reads `failed` failing: every
    /// claim made lands.
    fn read_ended(&self, outcome: Outcome, failed: &[usize]) {
        let ended = Parties::Ended(outcome.clone(), failed.to_vec());
        let Parties::Reading(claims) = std::mem::replace(&mut *self.parties(), ended) else {
            unreachable!("a read ends once");
        };
        for land in claims {
            land(outcome.clone(), failed, Party::Read);
        }
    }

    /// Records are claimed for some of its experts: `land` lands them once
    /// its read ended, now where it has. It must not wait for a read.
    pub(super) fn claimed(&self, land: impl FnOnce(Outcome, &[usize], Party) + Send + 'static) {
        let mut parties = self.parties();
        match &mut *parties {
            Parties::Reading(claims) => claims.push(Box::new(land)),
            Parties::Ended(outcome, failed) => {
                let (outcome, failed) = (outcome.clone(), failed.clone());
                drop(parties);
                land(outcome, &failed, Party::Claim);
            }
        }
    }
}

/// A read submitted, and its join.
pub(super) struct Posting {
    pub(super) ticket: Ticket,
    pub(super) join: Arc<Join>,
}

/// Request `reads` from `table`, wanted as `want`, joined to the claim of
/// what they land into ([`Join`]): the one way the watcher reads experts
/// into staging. Fails only where the read engine takes no request.
pub(super) fn post(table: &ReadTable, reads: Vec<Read>, want: Want) -> Result<Posting, String> {
    let ticket = table.source().request(reads, want)?;
    let join = Arc::new(Join(Mutex::new(Parties::Reading(Vec::new()))));
    let ended = join.clone();
    ticket.on_end(move |outcome, failed| ended.read_ended(outcome, failed));
    Ok(Posting { ticket, join })
}

/// Request `naming`'s demand reads, accounted once the last of them ends
/// ([`account`]): those left, and whether every one so far read whole.
struct Accounted {
    left: Mutex<(usize, bool)>,
    started: Vec<Arc<std::sync::OnceLock<Instant>>>,
    submitted: Instant,
    timed: Option<(u64, Instant)>,
    naming: Naming,
}

/// Account request `naming`'s demand reads `postings` as the last of them
/// ends, once: a point of the read line over `timed` bytes where they read
/// all the request's misses, submitted together at its instant.
pub(super) fn account(postings: &[&Posting], naming: Naming, timed: Option<(u64, Instant)>, shared: &Arc<Shared>) {
    // When the volume began reading the jobs: from their submission, which
    // the request times where they read all its misses, else now, just
    // after.
    let accounted = Arc::new(Accounted {
        left: Mutex::new((postings.len(), true)),
        started: postings.iter().flat_map(|posting| posting.ticket.started()).collect(),
        submitted: timed.map_or_else(Instant::now, |(_, submitted)| submitted),
        timed,
        naming,
    });
    for posting in postings {
        let (shared, accounted) = (shared.clone(), accounted.clone());
        posting.join.claimed(move |outcome, _, _| {
            unpanicked(&outcome);
            let mut left = accounted.left.lock().expect("a request's reads' account");
            left.0 -= 1;
            left.1 &= matches!(outcome, Outcome::Read);
            if left.0 > 0 || !left.1 {
                return;
            }
            drop(left);
            let Accounted { started, submitted, timed, naming, .. } = &*accounted;
            let timed = timed.map(|(bytes, submitted)| (bytes, submitted.elapsed()));
            let first = started.iter().filter_map(|started| started.get().copied()).min();
            let queued = first.map_or(std::time::Duration::ZERO, |first| first.saturating_duration_since(*submitted));
            shared.landed(naming.layer, naming.sequence, naming.named.elapsed(), timed, queued);
        });
    }
}

/// What [`Ahead::begin_rings`] read, which [`Ahead::finish_rings`]
/// admits: each layer's names, the latest point a ring stood at, whether
/// any ring was read, and whether the ledger learned anything.
pub(super) struct RingsRead {
    named: Vec<(usize, Vec<RingName>)>,
    pub(super) latest: Option<Point>,
    pub(super) any: bool,
    told: bool,
}

/// A request that names reads: its step's layer and sequence, and when it
/// named them.
#[derive(Clone, Copy)]
pub(super) struct Naming {
    pub(super) layer: usize,
    pub(super) sequence: Step,
    pub(super) named: Instant,
}

/// A read a request named that lands one expert, its record claimed: the
/// expert, its posting, the staging record it reads into, the record its
/// claim took, the rows of the request whose choices named it, and the
/// choices it answers.
pub(super) struct Named {
    pub(super) expert: usize,
    pub(super) posting: Posting,
    pub(super) staged: u32,
    pub(super) grant: Grant,
    pub(super) rows: BTreeSet<usize>,
    pub(super) answer: Answer,
}

/// The host's landings of one request's demand reads (`FileLine`): from
/// the last read's end to the last landing's, a point of its group's
/// landing line, once every one landed.
pub(super) struct Landings {
    layer: usize,
    count: usize,
    /// The landings not yet ended, and the last read's end so far.
    left: Mutex<(usize, Option<Instant>)>,
}

impl Landings {
    pub(super) fn new(layer: usize, count: usize) -> Arc<Self> {
        Arc::new(Self { layer, count, left: Mutex::new((count, None)) })
    }

    /// One of its reads ended whole.
    fn read(&self) {
        let mut left = self.left.lock().expect("a request's landings");
        left.1 = Some(Instant::now());
    }

    /// One of them ended.
    fn ended(&self, shared: &Shared) {
        let mut left = self.left.lock().expect("a request's landings");
        left.0 -= 1;
        if let (0, Some(read)) = *left {
            shared.telemetry.landings_ended(self.layer, self.count, read.elapsed().as_secs_f64());
        }
    }
}

/// Whether request `naming`'s read of one expert landed whole, its
/// ticket's `failed` reads none; failed, its end is recorded, the `rows`
/// that named it fail, then `answer` answers none.
fn read_named(shared: &Shared, naming: Naming, (rows, answer): (&BTreeSet<usize>, &Answer), outcome: &Outcome, failed: &[usize], ended: &Ended) -> bool {
    if failed.is_empty() {
        shared.landings.read(naming.named.elapsed());
        return true;
    }
    if let Outcome::Failed(reason) = outcome {
        shared.steps.fail_request_rows(naming.sequence, naming.layer, rows, reason.clone());
    }
    ended.set(false);
    answer.publish(None);
    false
}

/// No read of experts ends in a panic: the watcher's threads post them,
/// so a read's panic ends the process (`seismic::threads`) before
/// any outcome reports it.
fn unpanicked(outcome: &Outcome) {
    if let Outcome::Panicked(panic) = outcome {
        unreachable!("invariant violated: a read of experts the watcher posted ended in a panic: {panic:?}");
    }
}

/// An expert on its way into its record: read into staging record
/// `record` (none once the device's DONE freed it), landing into tier
/// record `tier`, for what [`Need`] says; its landing's end, once it
/// landed (or its read failed), and its urgency on the executor, the
/// read's own. Directory places one landing of an expert at a time
/// (`Located::Landing`), so an expert has one at most.
struct Inbound {
    record: Option<u32>,
    tier: u32,
    need: Need,
    ended: Ended,
    urgency: Arc<Urgency>,
}

/// Why an expert is inbound, with the read that brings it.
enum Need {
    /// A request named it at step `read`, read by its own ticket;
    /// `device` where the device converts it (the bank places).
    Due { ticket: Ticket, read: Step, device: Option<Arc<DeviceLanding>> },
    /// Read ahead of demand, by its own ticket, for what `ahead` says.
    Ahead { ticket: Ticket, ahead: Ahead },
}

/// What a read ahead is for.
#[derive(Clone, Copy)]
enum Ahead {
    /// The interest that led its expert's aggregate as it was issued
    /// (`experts::interest`): its source, and the route at `issued` that
    /// published it. The ledger orders, reprices and cancels it.
    Interest { source: Source, issued: At },
    /// The covering tier's ([`Prefetch::cover`]), undated, which no
    /// interest names.
    Cover,
}

impl Ahead {
    /// How its landing takes its tier record.
    fn taking(self) -> Taking {
        match self {
            Self::Interest { .. } => Taking::Ahead,
            Self::Cover => Taking::Out,
        }
    }
}

impl Inbound {
    /// The read that brings it.
    fn ticket(&self) -> &Ticket {
        match &self.need {
            Need::Due { ticket, .. } => ticket,
            Need::Ahead { ticket, .. } => ticket,
        }
    }

    /// The step that read it: a request's, or the route's that published
    /// its interest; none for the cover's.
    fn read(&self) -> Option<Step> {
        match &self.need {
            Need::Due { read, .. } => Some(*read),
            Need::Ahead { ahead: Ahead::Interest { issued, .. }, .. } => Some(issued.point.sequence),
            Need::Ahead { ahead: Ahead::Cover, .. } => None,
        }
    }
}

/// A landing in flight a request waits for ([`Prefetch::reading`]): read
/// ahead for an interest the route at its `At` published, or one no
/// prediction made (the cover's, or another request's).
pub(super) enum Reading {
    Predicted(At),
    Unpredicted,
}

/// What became of one name of a hint or front ring, as the read timeline
/// records it (`candidate`: the issuing step as its job; the named layer,
/// the expert, and `fate << 16 | lead << 8 | rank`, with the staging
/// records free as it was read from bit 32; `lead` a hint ring's row, or
/// `HINT_LEADS` plus an ahead row's index for a front ring's): read; not a
/// candidate (held, kept, or landing already, a longer lead's read among
/// them); named twice in its ring; or its ring read after its step's
/// request at the layer was noticed. Each fate keeps its number in every
/// recording.
#[derive(Clone, Copy)]
enum Fate {
    Read = 0,
    Held = 1,
    Repeated = 2,
    Passed = 6,
}

impl Fate {
    /// Note `source`'s name `(expert, rank)` of `target`, published by step
    /// `sequence`'s route, with `free` staging records free.
    fn note(self, sequence: Step, target: usize, source: Source, (expert, rank): (usize, u8), free: usize) {
        let lead = match source {
            Source::Lead { distance, .. } => usize::from(distance) - 1,
            Source::Ahead(row) => crate::planning::HINT_LEADS as usize + usize::from(row.index),
            Source::Exact => unreachable!("no ring names an exact request"),
        };
        let packed = (free as i64) << 32 | (self as i64) << 16 | (lead as i64) << 8 | i64::from(rank);
        seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("candidate"), sequence, (target as i64, expert as i64, packed));
    }
}

/// A ring's names as read whole ([`ring_entries`]): its step's low word,
/// its step's front rows' guess kinds (RING_GUESSES, STEP_GUESSES's
/// fields), and each entry's
/// expert, the rank of the choice that named it, and the batch row of that
/// choice (HINT_EXPERT, HINT_RANK, HINT_ROW).
type RingEntries = (i32, u32, Vec<(i32, i32, i32)>);

/// The ring at `base` of `ring` if it changed since its step's low word
/// `seen` (zero, none yet) and was read whole: the ring is a seqlock
/// (lib/streaming/hint), the device storing sequence 0 before it rewrites
/// the list and the step's after, so a list read under one unchanged
/// nonzero sequence is that step's whole; one read while the device rewrote
/// it is read again on the next pass. At most `choices` entries.
fn ring_entries(ring: &Words, base: usize, seen: i32, choices: u64) -> Option<RingEntries> {
    let low = ring.load(base + constants::RING_SEQUENCE as usize);
    if low == 0 || low == seen {
        return None;
    }
    let count = ring.load(base + constants::RING_COUNT as usize);
    let guesses = ring.load(base + constants::RING_GUESSES as usize) as u32;
    let count = usize::try_from(count).ok().filter(|&count| count as u64 <= choices).unwrap_or_else(|| panic!("a ring of {count} entries"));
    let entry = |index: usize, word: u64| ring.load(base + (constants::HINT_ENTRY(index as u64) + word) as usize);
    let entries = (0..count)
        .map(|index| {
            let word = |word: u64| entry(index, word);
            (word(constants::HINT_EXPERT), word(constants::HINT_RANK), word(constants::HINT_ROW))
        })
        .collect();
    (ring.load(base + constants::RING_SEQUENCE as usize) == low).then_some((low, guesses, entries))
}

/// The staging records and the reads into them.
pub(super) struct Prefetch {
    /// Staging record indices, valid in every pool, under their own lock:
    /// the request service's look takes a request's beside residency.
    pub(super) staging: Arc<StagingPool>,
    /// What is inbound, by expert, until the watcher learns its landing's
    /// end ([`Self::settle`]).
    inbound: HashMap<Staged, Inbound>,
    /// How far the device is known to be (`Residency::advance`), which ends
    /// the landings it converted behind it.
    past: Point,
    /// The interest ledger (`experts::interest`): the one owner of what is
    /// wanted, by when and how surely, of which this derives every read.
    /// Other threads reach it through `inbox` ([`Self::learn`]).
    ledger: Ledger,
    inbox: Receiver<Told>,
    /// Per layer, the low word of the last step each of its rings was read
    /// for: its hint rings' rows by lead, then its front ring
    /// ([`FRONT_RING`]).
    seen: Vec<[i32; RINGS]>,
}

/// A layer's rings: its hint rings' rows by lead, then its front ring.
const RINGS: usize = crate::planning::HINT_LEADS as usize + 1;

/// The index of a layer's front ring among its rings.
const FRONT_RING: usize = crate::planning::HINT_LEADS as usize;

/// What a read ahead is priced by, read from telemetry (under its lock) once
/// and only when a read short of due asks, so deriving the reads a request
/// finds due takes no lock.
struct Priced<'a> {
    telemetry: &'a super::telemetry::Telemetry,
    price: std::cell::OnceCell<super::telemetry::ReadPrice>,
}

impl<'a> Priced<'a> {
    fn new(telemetry: &'a super::telemetry::Telemetry) -> Self {
        Self { telemetry, price: std::cell::OnceCell::new() }
    }

    fn get(&self) -> &super::telemetry::ReadPrice {
        self.price.get_or_init(|| self.telemetry.price())
    }
}

/// Remaining shared-drive idle compute window (`c_layer - queued_drive`)
/// during one `Prefetch::derive` pass (`designs/streamed-decode.md` §5):
/// the drive is a single shared resource across all layers, so one layer's
/// compute `c_layer = compute / layers` is compared against all reads
/// currently occupying the drive (every `Need::Due` plus every already-begun
/// or out-of-pass `Need::Ahead` across all layers), lazily initialized on
/// first query and decremented as reads ahead are admitted in priority order
/// inside `Ledger::schedule`.
struct IdleBudget {
    layers: usize,
    queued_bytes: Vec<u64>,
    remaining: std::cell::Cell<Option<f64>>,
}

impl IdleBudget {
    fn new(layers: usize, queued_bytes: Vec<u64>) -> Self {
        Self { layers, queued_bytes, remaining: std::cell::Cell::new(None) }
    }

    fn get(&self, price: &super::telemetry::ReadPrice) -> f64 {
        if let Some(rem) = self.remaining.get() {
            return rem;
        }
        let c_layer = if self.layers == 0 { 0.0 } else { price.compute() / self.layers as f64 };
        let queued: f64 = self.queued_bytes.iter().map(|&bytes| price.transfer_seconds(bytes)).sum();
        let rem = (c_layer - queued).max(0.0);
        self.remaining.set(Some(rem));
        rem
    }

    fn consume(&self, price: &super::telemetry::ReadPrice, transfer_seconds: f64) {
        let rem = self.get(price);
        if rem > 0.0 {
            self.remaining.set(Some((rem - transfer_seconds).max(0.0)));
        }
    }
}

/// Admission of a read ahead of an expert's `bytes`, as the ledger asks it
/// ([`Admit`]): at the break-even of a read of them given the shared drive's
/// remaining idle compute window (`ReadPrice::admits_with_idle`).
struct Admitting<'a> {
    priced: &'a Priced<'a>,
    bytes: u64,
    idle: &'a IdleBudget,
}

impl Admit for Admitting<'_> {
    fn admits(&self, p: f64) -> bool {
        let price = self.priced.get();
        let admitted = price.admits_with_idle(p, self.bytes, self.idle.get(price));
        if admitted {
            self.idle.consume(price, price.transfer_seconds(self.bytes));
        }
        admitted
    }
}

impl Prefetch {
    /// `records` staging records, every one free, for layers in `groups`
    /// whose rings name `ranks` experts a row, its ledger rolling on
    /// `time`'s epochs and counting verdicts into `ranking`'s `Stands`.
    pub(super) fn new(
        records: u32,
        groups: Vec<usize>,
        ranks: usize,
        time: crate::experts::horizon::Time,
        inbox: Receiver<Told>,
        ranking: &crate::experts::reuse::Ranking,
    ) -> Self {
        let layers = groups.len();
        let (_, stands) = crate::experts::reuse::Reuse::of(ranking).rates();
        Self {
            staging: Arc::new(StagingPool::new(records)),
            inbound: HashMap::new(),
            past: Point::START,
            ledger: Ledger::new(groups, ranks, time, stands),
            inbox,
            seen: vec![[0; RINGS]; layers],
        }
    }

    /// The low word of the last step `layer`'s front ring was read for.
    #[cfg(test)]
    pub(super) fn front_read(&self, layer: usize) -> i32 {
        self.seen[layer][FRONT_RING]
    }

    /// Track request `naming`'s demand read `named` at `at`, its record
    /// claimed: its read's end lands it where its layer's bank places, on
    /// the device ([`DeviceLanding`]: appended to the layer's landing list,
    /// under the step's ownership, then answered) or else on the host
    /// (submitted to the executor, which converts and maps its record,
    /// records its end, then answers; one of the request's host `landings`,
    /// whose line it measures), or fails it. Where the read's end ran the
    /// host's landing (`party`), its read worker works its pieces beside the
    /// executor's, so no wake stands between the read and its landing. A
    /// request's reads are accounted apart, once ([`account`]).
    pub(super) fn demand(&mut self, core: &Core, named: Named, naming: Naming, at: Point, landings: &Arc<Landings>, shared: &Arc<Shared>) {
        let Named { expert, posting: Posting { ticket, join }, staged, grant, rows, answer } = named;
        let (ended, urgency, tier) = (Ended::new(shared.events()), Urgency::waited(), grant.record());
        let device = core.layers[naming.layer]
            .landing
            .as_ref()
            .map(|list| Arc::new(DeviceLanding::new(list.clone(), at, ended.clone(), shared.landings.clone())));
        let (serving, end) = (shared.clone(), ended.clone());
        match &device {
            Some(device) => {
                let device = device.clone();
                join.claimed(move |outcome, failed, _| {
                    if !read_named(&serving, naming, (&rows, &answer), &outcome, failed, &end) {
                        return;
                    }
                    // Appended while the step owns its slot: a read ending
                    // after its step did appends nothing, its end not
                    // landed; one the list has moved past (the device past
                    // this step's launch) ends not landed and answers none.
                    match serving.steps.while_owned(naming.sequence, |_| device.append(grant, staged)) {
                        Some(true) => answer.publish(Some(tier)),
                        Some(false) => answer.publish(None),
                        None => end.set(false),
                    }
                });
            }
            None => {
                let landing = core.landing(grant, staged);
                let (landings, urgent) = (landings.clone(), urgency.clone());
                join.claimed(move |outcome, failed, party| {
                    if !read_named(&serving, naming, (&rows, &answer), &outcome, failed, &end) {
                        return;
                    }
                    landings.read();
                    let completed = serving.clone();
                    // SAFETY: the read landed whole and nothing reads into
                    // its staging record until the watcher frees it (once
                    // its end is recorded), and its tier record was ready
                    // (no step reads it) when taken.
                    let converting = unsafe {
                        Converting::new(landing, urgent, serving.landings.clone(), end, move || {
                            answer.publish(Some(tier));
                            landings.ended(&completed);
                        })
                    };
                    match party {
                        Party::Read => serving.compute.run(Arc::new(converting)),
                        Party::Claim => serving.compute.submit(Arc::new(converting)),
                    }
                });
            }
        }
        let need = Need::Due { ticket, read: naming.sequence, device };
        self.arrive((naming.layer, expert), Inbound { record: Some(staged), tier, need, ended, urgency });
    }

    /// The landing of `staged` in flight a request finds, by what read it:
    /// none where nothing lands it or its read failed (the request reads
    /// the expert anew once its end is learned).
    pub(super) fn reading(&self, staged: Staged) -> Option<Reading> {
        let inbound = self.inbound.get(&staged)?;
        if inbound.ended.get() == Some(false) {
            return None;
        }
        Some(match inbound.need {
            Need::Ahead { ahead: Ahead::Interest { issued, .. }, .. } => Reading::Predicted(issued),
            Need::Due { .. } | Need::Ahead { ahead: Ahead::Cover, .. } => Reading::Unpredicted,
        })
    }

    /// Order the landing of `staged` in flight due: a step waits for it (a
    /// decode request's miss, the ledger's due class, [`Self::want`]; or a
    /// prompt group's pull, `Scratch::awaited`), so its read and its landing
    /// are at demand from now, its landing mapping the tier record the step
    /// is then served from. Nothing where staging lands none or its read
    /// failed. Parked from the step that first waits on it, repriced then
    /// alone: a hastened read orders before every read ahead, and never
    /// later than it was; a later wait finds it due already.
    pub(super) fn hasten(&self, staged: Staged) {
        let noted = |code: i64| seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("hasten"), 0, (staged.0 as i64, staged.1 as i64, code));
        let Some(inbound) = self.inbound.get(&staged) else {
            noted(-1);
            return;
        };
        if inbound.ended.get() == Some(false) {
            noted(-2);
            return;
        }
        noted(match inbound.need {
            Need::Due { .. } => 0,
            Need::Ahead { ahead: Ahead::Interest { source: Source::Ahead(_), .. }, .. } => 3,
            Need::Ahead { ahead: Ahead::Interest { .. }, .. } => 1,
            Need::Ahead { ahead: Ahead::Cover, .. } => 2,
        });
        if !inbound.urgency.is_waited() {
            inbound.ticket().reprice(inbound.urgency.get().min(Want::now()));
            inbound.urgency.hasten();
        }
    }

    /// Step `at`'s exact rows requested `experts` of its layer absent: each
    /// is wanted now, certainly (`Source::Exact`), and any read ahead in
    /// flight for it is hastened to demand (`Self::hasten`), as is the
    /// cover's. The request reads an absent one itself (`Decode`). Certain
    /// and due, no read here is priced or scheduled through `derive`.
    pub(super) fn want(&mut self, core: &Core, shared: &Arc<Shared>, at: At, experts: &[usize]) {
        let (sequence, layer) = (at.point.sequence, at.point.layer);
        self.ledger.reach(sequence, layer);
        let now = Instant::now();
        for &expert in experts {
            let staged = (layer, expert);
            let reading = matches!(self.read_state(core, staged), ReadState::Reading { .. });
            self.ledger.requested(layer, expert, reading);
            let name = Name { source: Source::Exact, when: When::Step(sequence), rank: 0, from: at, candidate: false, lineage: None };
            self.ledger.publish(layer, expert, name, now);
            if reading {
                self.hasten(staged);
            }
        }
        self.spend(shared);
    }

    /// Pass the predicted interests whose step's route is done at their
    /// layer, so no read can serve them any more (`Ledger::pass`), and
    /// derive their reads again: one nothing else wants that no worker has
    /// begun and no step waits on is cancelled. A route is done once the
    /// request the service just served (`served`, its step and layer) is at
    /// or past it, its step has ended, or a ring showed its route done with
    /// no request there (`Ledger::routed`: a request is published before
    /// those rings, so one absent then was never made). A read begun is not
    /// cancelled (its bytes would be wasted and the expert read again): it
    /// lands and falls to the ranking. A cancelled read reads nothing more,
    /// so it takes no source time from demand; its read ends failed, which
    /// frees its staging and tier records ([`Self::settle`]), and a later
    /// request for its expert reads it anew. A store seen out of that order
    /// only costs that read.
    pub(super) fn pass(&mut self, core: &mut Core, shared: &Arc<Shared>, served: Option<(Step, usize)>) {
        let steps = &shared.steps;
        let mut changed = Vec::new();
        for layer in 0..core.layers.len() {
            let (routed, file) = (self.ledger.routed(layer), &core.layers[layer].file);
            let passed = self.ledger.pass(layer, Instant::now(), |interest| {
                let step = interest.step();
                served.is_some_and(|(sequence, at)| (step, layer) <= (sequence, at))
                    || steps.ended(step)
                    || (routed >= step && !file.requested(step, 0))
            });
            changed.extend(passed.into_iter().map(|expert| (layer, expert)));
        }
        if !changed.is_empty() {
            self.derive(core, shared, Some(changed.as_slice()));
        }
    }

    /// Track `inbound`, the one landing of `staged`.
    fn arrive(&mut self, staged: Staged, inbound: Inbound) {
        let previous = self.inbound.insert(staged, inbound);
        assert!(previous.is_none(), "expert {staged:?} lands twice at once");
    }

    /// Cancel the cover's reads no step waits for and that no read engine
    /// worker has begun: a resize or traffic hold waits only for the reads
    /// that run, never for the cover's queue. A cancelled read ends failed,
    /// freeing its staging record and its tier record, and the cover reads
    /// the expert again once nothing holds.
    pub(super) fn cancel_cover(&mut self) {
        for inbound in self.inbound.values_mut() {
            let Need::Ahead { ticket, ahead: Ahead::Cover } = &mut inbound.need else {
                continue;
            };
            if !inbound.urgency.is_waited() && !ticket.cancelled() {
                ticket.cancel();
            }
        }
    }

    /// Staging and what is in flight into it, for the exchange's watchdog.
    pub(super) fn describe(&self) -> String {
        let [mut hints, mut fronts, mut covers, mut due, mut running] = [0; 5];
        for inbound in self.inbound.values() {
            match inbound.need {
                Need::Due { .. } => due += 1,
                Need::Ahead { ahead: Ahead::Interest { source: Source::Ahead(_), .. }, .. } => fronts += 1,
                Need::Ahead { ahead: Ahead::Interest { .. }, .. } => hints += 1,
                Need::Ahead { ahead: Ahead::Cover, .. } => covers += 1,
            }
            running += usize::from(inbound.ticket().running());
        }
        format!(
            "staging {} free of {}; inbound: hints {hints}, front {fronts}, covers {covers}, due {due}; reads running for {running}",
            self.staging.free(),
            self.staging.records(),
        )
    }

    /// Whether a read or its landing is outstanding: any staging record
    /// taken. Every read into staging holds its record from before it is
    /// submitted until its landing's end is learned, or, a read given up,
    /// until it ends: an expert inbound, a request's posted read before
    /// residency serves it, one waiting for tier records, or one discarded.
    /// Each record's freeing is an event of the load.
    pub(super) fn outstanding(&self) -> bool {
        self.staging.free() < self.staging.records()
    }

    /// The device is at `progress`: the landings it converted before it have
    /// ended, as the next [`Self::settle`] learns.
    pub(super) fn past(&mut self, progress: Point) {
        self.past = self.past.max(progress);
    }

    /// How far the device is known to be.
    pub(super) fn reached(&self) -> Point {
        self.past
    }

    /// Settle the landings that ended (their ends mapped or failed their
    /// experts): each tier record keeps its expert in `directory`, or is
    /// ready again, and each staging record frees; and each device landing's
    /// staging record whose entry the device stored DONE for, its landing
    /// ending only once the device is past its launch
    /// (`DeviceLanding::end`). Where the device parks in its kernels the
    /// request service's pass reads those words while a step runs, so a
    /// request's later misses read into the records its earlier ones free.
    /// True when any ended or freed.
    pub(super) fn settle(&mut self, directory: &mut Directory) -> bool {
        let mut freed = false;
        for inbound in self.inbound.values_mut() {
            if let Need::Due { device: Some(device), .. } = &inbound.need {
                if inbound.record.is_some() && device.converted() {
                    self.staging.release(inbound.record.take().expect("checked above"));
                    freed = true;
                }
                device.end(self.past);
            }
        }
        let mut ended = Vec::new();
        self.inbound.retain(|&staged, inbound| match inbound.ended.get() {
            Some(landed) => {
                ended.push((inbound.record, staged, inbound.tier, inbound.read(), landed));
                false
            }
            None => true,
        });
        let settled = freed || !ended.is_empty();
        for (record, staged, tier, read, landed) in ended {
            self.ended(directory, record, staged, tier, read, landed);
        }
        settled
    }

    /// A landing of `staged` from staging record `record` (none, freed
    /// already) into tier record `tier`, read at step `read` (none for the
    /// cover's), ended: landed or not.
    fn ended(&mut self, directory: &mut Directory, record: Option<u32>, staged: Staged, tier: u32, read: Option<Step>, landed: bool) {
        let (layer, expert) = staged;
        match landed {
            true => {
                directory.landed(layer, HostRecord(tier), expert, read);
                self.ledger.landed(layer, expert, Instant::now());
            }
            false => directory.unlanded(layer, HostRecord(tier)),
        }
        if let Some(record) = record {
            self.staging.release(record);
        }
    }

    /// The calibrated probability of `combination` at `layer` (test helper).
    #[cfg(test)]
    pub(super) fn rate(&mut self, layer: usize, combination: crate::experts::interest::Combination) -> f64 {
        self.ledger.rate(layer, combination)
    }

    /// Learn what other threads told the ledger (`Told`), in the order told:
    /// a completion's resolutions count into the hints' recall and
    /// precision, and a verdict's judged front rows into the front gate's
    /// wrong rate. True when any was told.
    pub(super) fn learn(&mut self, shared: &Shared) -> bool {
        let mut told = false;
        while let Ok(event) = self.inbox.try_recv() {
            told = true;
            let completed = matches!(&event, Told::Completed { .. });
            let learned = self.ledger.tell(event);
            if completed {
                shared.telemetry.resolved(&learned.resolved, &learned.cohorts);
            }
        }
        let (judged, accepted) = self.ledger.take_judged();
        if !judged.is_empty() || !accepted.is_empty() {
            shared.telemetry.judged(&judged, &accepted);
        }
        self.spend(shared);
        told
    }

    /// Read each layer's rings that changed, its hint rings and its front
    /// ring, then learn what was told ([`Self::learn`]) and publish the names
    /// of the rings of a step whose request at the named layer is not yet
    /// noticed (`served`, each layer's last request noticed; a later one's
    /// names are decided, its route there having chosen) to the ledger
    /// ([`Self::admit`]); then derive every read from it ([`Self::drive`]).
    /// The rings are read before the ledger learns, so every launch a ring
    /// names has been told it (a launch is told before it is queued). Each
    /// entry names the batch row whose choice named it (`Ledger::row`), and
    /// so the step its name is for: an exact row's, the ring's own step; an
    /// ahead row's, the step its token is exact in, its lineage the ahead
    /// row's. A hint ring row of lead `d` names the layer `d` ahead
    /// (`Source::Lead`); a front ring names its own layer (`Source::Ahead`).
    /// Reads ahead only ever read the file: where the host tier keeps every
    /// expert (`Directory::tier_complete`) nothing is a candidate, and no
    /// ring is read. True when it read any ring, learned anything or read
    /// any expert.
    /// Read changed rings, drain what other threads told the ledger, and
    /// resolve each live ring's step and device progress point so residency
    /// can retire graced records behind that point before admitting names.
    pub(super) fn begin_rings(&mut self, core: &Core, shared: &Shared) -> RingsRead {
        let read = self.read_rings(core);
        let told = self.learn(shared);
        let any = !read.is_empty();
        let (named, latest) = self.name(core, shared, read);
        RingsRead { named, latest, any, told }
    }

    /// Admit `named` from `begin_rings`, and derive reads whenever rings
    /// arrived, events were told, or capacity freed.
    pub(super) fn finish_rings(
        &mut self,
        core: &mut Core,
        shared: &Arc<Shared>,
        RingsRead { named, any, told, .. }: RingsRead,
        freed: bool,
        served: &[Step],
    ) -> bool {
        for (target, names) in named {
            self.admit(core, shared, target, names, served);
        }
        if any || told || freed {
            self.drive(core, shared);
            true
        } else {
            false
        }
    }

    /// Each layer's rings that changed since they were read: each lead's row
    /// (`routed_route_streamed`'s rings: the layer one ahead, then two,
    /// `Layer::hints`; the last streamed layer has none), then the front
    /// ring.
    fn read_rings(&mut self, core: &Core) -> Vec<RingRead> {
        let words = core.extents.words;
        let ring_words = words.ring_words() as usize;
        let mut read = Vec::new();
        for (layer, streamed) in core.layers.iter().enumerate() {
            let rings = (0..streamed.hints.len()).map(|lead| (lead, &streamed.rings, lead * ring_words)).chain([(FRONT_RING, &streamed.front, 0)]);
            for (ring, words_of, base) in rings {
                if let Some((low, guesses, entries)) = ring_entries(words_of, base, self.seen[layer][ring], words.choices) {
                    self.seen[layer][ring] = low;
                    read.push(RingRead { layer, ring, low, guesses, entries });
                }
            }
        }
        read
    }

    /// The names the rings `read` make, by the layer each names, of the steps
    /// not ended: a ring of a step that ended is skipped, and the layers its
    /// route showed done are the ledger's to know ([`Ledger::route_seen`]).
    fn name(&mut self, core: &Core, shared: &Shared, read: Vec<RingRead>) -> (Vec<(usize, Vec<RingName>)>, Option<Point>) {
        let (experts, selected) = (|layer: usize| core.layers[layer].experts, core.extents.selected);
        // Each entry's expert, rank and row checked: a ring of another shape
        // is the device's broken invariant.
        let checked = |target: usize, (expert, rank, row): (i32, i32, i32)| {
            let expert = usize::try_from(expert).ok().filter(|&expert| expert < experts(target)).unwrap_or_else(|| panic!("a ring names expert {expert}"));
            let rank = u8::try_from(rank).ok().filter(|&rank| usize::from(rank) < selected).unwrap_or_else(|| panic!("a ring names choice rank {rank}"));
            let row = usize::try_from(row).unwrap_or_else(|_| panic!("a ring names row {row}"));
            (expert, rank, row)
        };
        let mut named = Vec::new();
        let mut latest: Option<Point> = None;
        let layers = core.layers.len();
        for RingRead { layer, ring, low, guesses, entries } in read {
            let Some(from) = shared.steps.at_low_word(low, layer, layers) else {
                continue;
            };
            latest = Some(latest.map_or(from.point, |prev| prev.max(from.point)));
            let sequence = from.point.sequence;
            match ring {
                FRONT_RING => {
                    // Published once the route's request was.
                    self.ledger.route_seen(sequence, layer);
                    // A front row that ran on no guess names nothing.
                    let names = entries
                        .into_iter()
                        .filter_map(|entry| {
                            let (expert, rank, row) = checked(layer, entry);
                            let Origin::Ahead { row: ahead, when, lineage } = self.ledger.row(sequence, row, guesses)? else {
                                panic!("step {sequence}'s front ring at layer {layer} names row {row}, which is no ahead row");
                            };
                            assert!(!ahead.acceptable(), "step {sequence}'s front ring at layer {layer} names row {row}, an acceptable row");
                            let name = Name { source: Source::Ahead(ahead), when, rank, from, candidate: false, lineage: Some(lineage) };
                            Some(RingName { expert, name })
                        })
                        .collect::<Vec<_>>();
                    // This pass's front rows guess every position its
                    // request's earlier passes named here, anew: theirs at
                    // this layer give way ([`Ledger::supersede`]).
                    let requests = names.iter().filter_map(|name| name.name.lineage.map(|lineage| lineage.request)).collect::<BTreeSet<_>>();
                    for request in requests {
                        self.ledger.supersede(layer, request, sequence);
                    }
                    named.push((layer, names));
                }
                lead => {
                    // The route before published its request before this ring.
                    if let Some(before) = layer.checked_sub(1) {
                        self.ledger.route_seen(sequence, before);
                    }
                    let target = core.layers[layer].hints[lead];
                    let distance = u8::try_from(lead + 1).expect("a ring's leads fit a byte");
                    let names = entries
                        .into_iter()
                        .filter_map(|entry| {
                            let (expert, rank, row) = checked(target, entry);
                            let (ahead, when, lineage) = match self.ledger.row(sequence, row, guesses)? {
                                Origin::Exact => (None, When::Step(sequence), None),
                                Origin::Ahead { row, when, lineage } => (Some(row), when, Some(lineage)),
                            };
                            let name = Name { source: Source::Lead { distance, row: ahead }, when, rank, from, candidate: false, lineage };
                            Some(RingName { expert, name })
                        })
                        .collect();
                    named.push((target, names));
                }
            }
        }
        (named, latest)
    }

    /// Whether `staged` is a read ahead's candidate: an expert that lives
    /// only in the file: the bank lacks it, and its host tier keeps, graces
    /// and lands none of it (`Directory::located`).
    fn candidate(core: &Core, staged: Staged) -> bool {
        let (layer, expert) = staged;
        core.directory.held(layer, expert).is_none() && core.directory.located(layer, expert) == Located::Lacked
    }

    /// Publish a ring's `names` of `target` to the ledger, each a candidate
    /// where its expert lives only in the file ([`Self::candidate`]), the
    /// held ones too, each with the probability the ranking carries its
    /// expert to its step should its guess fall (`Reuse::carried`), which
    /// prices a name of a later pass than its own past its stand rate. A
    /// name for a step that ended, or whose request
    /// at `target` was noticed (`served`: it names a layer the device has
    /// passed), is dropped, and one a ring repeats is published once. Each
    /// candidate an exact row's hint newly published counts with its step,
    /// with whether its probability reached the read-ahead break-even
    /// (`Telemetry::hints_named`).
    fn admit(&mut self, core: &Core, shared: &Arc<Shared>, target: usize, names: Vec<RingName>, served: &[Step]) {
        let recording = seismic_io::timeline::recording();
        let (price, bytes) = (Priced::new(&shared.telemetry), core.layers[target].expert_bytes());
        let mut hinted = Vec::new();
        let note = |fate: Fate, expert: usize, name: &Name| {
            if recording {
                fate.note(name.from.point.sequence, target, name.source, (expert, name.rank), 0);
            }
        };
        let mut seen = HashSet::new();
        let now = Instant::now();
        for RingName { expert, name } in names {
            if let When::Step(step) = name.when {
                if served[target] >= step || shared.steps.ended(step) {
                    note(Fate::Passed, expert, &name);
                    continue;
                }
            }
            let request = name.lineage.map(|lineage| lineage.request);
            let wanted = match name.when {
                When::Step(step) => (step, None),
                When::Pass { request, pass, .. } => (0, Some((request, pass))),
            };
            if !seen.insert((expert, name.source, request, wanted)) {
                note(Fate::Repeated, expert, &name);
                continue;
            }
            let candidate = Self::candidate(core, (target, expert));
            if !candidate {
                note(Fate::Held, expert, &name);
            }
            self.ledger.reach(name.from.point.sequence, name.from.point.layer);
            let published = self.ledger.publish(target, expert, Name { candidate, ..name }, now);
            if let (Source::Lead { distance, row: None }, When::Step(step), Some(p), true) = (name.source, name.when, published, candidate) {
                hinted.push((step, name.from.point.layer, distance, price.get().admits(p, bytes)));
            }
        }
        if !hinted.is_empty() {
            shared.telemetry.hints_named(&hinted);
        }
    }

    /// What a read ahead leaves free for demand, the one rule of staging
    /// room: staging records beyond those a request takes for its own, and
    /// the ready tier records beyond what its group's reserve holds past
    /// the reads ahead's staging, which this names (none where the reserve
    /// is the reads ahead's alone, a load's: a request reclaims a kept
    /// record at its park where none is ready); a read
    /// ahead's landing then displaces the tier's least reused as the tier
    /// refills its reserve. The cover leaves the requests' staging and no
    /// tier records (none named), whatever the reads ahead hold.
    fn reserve(&self, core: &Core, cover: bool) -> (usize, Option<usize>) {
        let request = core.extents.demand;
        match cover {
            true => (request, None),
            false => (request, Some(self.staging.records().saturating_sub(request))),
        }
    }

    /// Mark routes and ended steps past, then derive every read from the
    /// ledger, the soonest wanted first ([`Self::derive`]). True when any
    /// was read.
    pub(super) fn drive(&mut self, core: &mut Core, shared: &Arc<Shared>) -> bool {
        self.pass(core, shared, None);
        self.derive(core, shared, None)
    }

    /// Where `staged`'s read stands, as the ledger is told it: read ahead
    /// (the cover's among them), queued or begun; read by a request, or
    /// landing (a posted read the look made, or a landing of the
    /// conveyor's); absent, living only in the file; or held.
    fn read_state(&self, core: &Core, staged: Staged) -> ReadState {
        match self.inbound.get(&staged) {
            Some(Inbound { need: Need::Ahead { ticket, .. }, .. }) => ReadState::Reading { begun: ticket.begun() },
            Some(Inbound { need: Need::Due { .. }, .. }) => ReadState::Landing,
            None if Self::candidate(core, staged) => ReadState::Absent,
            None if core.directory.located(staged.0, staged.1) == Located::Landing => ReadState::Landing,
            None => ReadState::Held,
        }
    }

    /// Expert bytes of every inbound read currently occupying the shared drive:
    /// every demand read (`Need::Due`), plus every already-begun ahead read or
    /// unbegun, uncancelled ahead read outside `seen` (partial `derive` from
    /// `pass`), across all layers.
    fn drive_queued_bytes(&self, expert_bytes: impl Fn(usize) -> u64, seen: &HashSet<Staged>) -> Vec<u64> {
        self.inbound
            .iter()
            .filter(|(staged, inbound)| match &inbound.need {
                Need::Due { .. } => true,
                Need::Ahead { ticket, .. } => ticket.begun() || (!ticket.cancelled() && !seen.contains(staged)),
            })
            .map(|(staged, _)| expert_bytes(staged.0))
            .collect()
    }

    /// Derive the reads of `only`'s experts, or of every expert the ledger
    /// wants (`Ledger::wanted`), and of those a told event changed
    /// (`Ledger::take_owed`): each one's read state is told the ledger,
    /// which acts on its aggregate ([`Ledger::act`]) and orders the acts
    /// (`Ledger::schedule`, the soonest wanted first); then each read is
    /// cancelled, hastened to demand, repriced or issued in that order,
    /// each read leaving what demand needs ([`Self::reserve`]). Where the
    /// staging falls short the names left stay in the ledger, read as it
    /// frees. A certain interest's read of an absent expert is its
    /// request's own (`Decode`), which answers its choices and fails its
    /// rows should it fail; the cover's is the ledger's only to hasten. A
    /// due read is never priced, so deriving the reads requests find due
    /// reads nothing under telemetry's lock ([`Priced`]). True when any was
    /// read.
    fn derive(&mut self, core: &mut Core, shared: &Arc<Shared>, only: Option<&[Staged]>) -> bool {
        let reserve = self.reserve(core, false);
        let price = Priced::new(&shared.telemetry);
        let owed = if only.is_none() { self.ledger.take_owed() } else { Vec::new() };
        let mut experts = match only {
            Some(only) => only.to_vec(),
            None => self.ledger.wanted(),
        };
        experts.extend(owed);
        let mut seen = HashSet::new();
        experts.retain(|&staged| seen.insert(staged));
        let scheduled = {
            let core: &Core = core;
            let queued_bytes = self.drive_queued_bytes(|layer| core.layers[layer].expert_bytes(), &seen);
            let idle = IdleBudget::new(core.layers.len(), queued_bytes);
            // Each read's standing, taken before the ledger decides on it.
            let states = experts.iter().map(|&staged| (staged, self.read_state(core, staged))).collect::<HashMap<_, _>>();
            let read = |layer, expert| states[&(layer, expert)];
            self.ledger.schedule(experts, read, |layer| Admitting { priced: &price, bytes: core.layers[layer].expert_bytes(), idle: &idle })
        };
        let wanted = |class: Class| match class {
            Class::Due => Want::now(),
            // Ordered on its link as the ledger orders it: by the step
            // that wants it, the soonest first, then its probability of
            // use ("Order and depth").
            Class::Expected { step, p } => Want::expected(step, p),
        };
        let (mut read, mut staging_full) = (false, false);
        for Scheduled { layer, expert, act, leading, .. } in scheduled {
            let staged = (layer, expert);
            match act {
                Act::Keep | Act::Issue(Class::Due) => {}
                Act::Reprice(Class::Due) => self.hasten(staged),
                // A read ahead takes its want's price; the cover's, undated,
                // is only hastened by a want of its expert.
                Act::Reprice(class) => {
                    if let Some(Inbound { need: Need::Ahead { ticket, ahead }, urgency, .. }) = self.inbound.get(&staged) {
                        if !urgency.is_waited() && !ticket.cancelled() {
                            let want = match ahead {
                                Ahead::Interest { .. } => wanted(class),
                                Ahead::Cover => urgency.get().min(wanted(class)),
                            };
                            ticket.reprice(want);
                            urgency.set(want);
                        }
                    }
                }
                Act::Cancel => {
                    if let Some(Inbound { need: Need::Ahead { ticket, ahead: Ahead::Interest { .. } }, urgency, .. }) = self.inbound.get_mut(&staged) {
                        if !urgency.is_waited() && !ticket.begun() && !ticket.cancelled() {
                            ticket.cancel();
                            self.ledger.cancelled(layer, expert);
                        }
                    }
                }
                Act::Issue(class) => {
                    if staging_full {
                        continue;
                    }
                    let expected = match class {
                        Class::Expected { step, p } => Some((step, p)),
                        Class::Due => None,
                    };
                    let interest = leading.expect("an expected read has a leading interest");
                    let ahead = Ahead::Interest { source: interest.source, issued: interest.from };
                    match self.read_ahead(core, shared, staged, wanted(class), ahead, expected, reserve) {
                        Ok(bytes) => {
                            self.ledger.issued(layer, expert, &interest, bytes, Instant::now());
                            shared.telemetry.read_ahead(interest.source, interest.step(), staged, interest.from.point.layer, bytes);
                            read = true;
                            if seismic_io::timeline::recording() {
                                Fate::Read.note(interest.from.point.sequence, layer, interest.source, (expert, interest.rank), self.staging.free());
                            }
                        }
                        Err(Short::Refused | Short::Tier) => {}
                        Err(Short::Staging) => staging_full = true,
                    }
                }
            }
        }
        self.spend(shared);
        read
    }

    /// Hand telemetry the reads ahead's fates the ledger learned
    /// (`Ledger::take_spent`).
    fn spend(&mut self, shared: &Shared) {
        let spent = self.ledger.take_spent();
        if !spent.is_empty() {
            shared.telemetry.spent(&spent);
        }
    }

    /// Cover the host tier: where a group's tier has a record for every
    /// expert of its layers (`HostRecords::covers`), each expert it lacks
    /// that nothing carries (`Directory::cover_order`) is landed into a
    /// tier record and read into a staging record undated, behind every
    /// step's reads, as a hint's read lands, so a request that waits for
    /// one hastens its read. `carried` names the experts a prompt group's
    /// conveyor carries, which land through it (`Scratch::carries`): no
    /// expert is read twice. Where the tier cannot hold every expert there
    /// is no choice to make ahead of use, and nothing is read: it fills by
    /// its misses and the reads ahead.
    ///
    /// It reads into every staging record beyond the demand reserve, the
    /// hints' among them it shares with hints, which each pass admits
    /// before it covers, so a hint takes each record as it frees. By the
    /// read engine's law (`seismic_io`, "Order and depth") a due read
    /// waits for at most the bytes read ahead already in flight, however
    /// much less urgent work is queued, so the depth of its queue costs no
    /// step anything. It gates nothing in serving: no growth, bank move or
    /// step waits for it, and a miss that arrives first takes its own read.
    /// Only a measurement waits for its end ([`Self::covering`]), so no
    /// timed span shares the drive with it.
    pub(super) fn cover(&mut self, core: &mut Core, shared: &Arc<Shared>, carried: impl Fn(Staged) -> bool) {
        let reserve = self.reserve(core, true);
        let keep = reserve.0;
        let room = self.staging.free().saturating_sub(keep);
        if room == 0 {
            return;
        }
        for staged in core.directory.cover_order(room, carried) {
            if self.staging.free() <= keep {
                break;
            }
            match self.read_ahead(core, shared, staged, Want::undated(0.), Ahead::Cover, None, reserve) {
                Ok(_) => {}
                // Refused, or its group's tier holds no record for it now:
                // the cover goes on to the next.
                Err(Short::Refused | Short::Tier) => {}
                Err(Short::Staging) => break,
            }
        }
    }

    /// Whether the cover still reads ([`Self::cover`]): a cover's read is
    /// inbound, or a tier that covers its experts lacks one
    /// (`Directory::cover_pending`).
    pub(super) fn covering(&self, core: &Core) -> bool {
        self.inbound.values().any(|inbound| matches!(inbound.need, Need::Ahead { ahead: Ahead::Cover, .. })) || core.directory.cover_pending()
    }

    /// Read `staged` ahead, `ahead` saying for what: into a staging record
    /// wanted as `want`, of those free beyond `staging`, landing into a
    /// tier record of its group the tier takes for it leaving `tier` ready
    /// (`Directory::claim_landing`), tracked inbound, so a step that waits
    /// for it hastens its read and its landing to demand
    /// ([`Self::hasten`]), and the watcher learns its end
    /// ([`Self::settle`]). Its bytes, or what it fell short of.
    fn read_ahead(
        &mut self,
        core: &mut Core,
        shared: &Arc<Shared>,
        staged: Staged,
        want: Want,
        ahead: Ahead,
        expected: Option<(Step, f64)>,
        (staging, ahead_staging): (usize, Option<usize>),
    ) -> Result<u64, Short> {
        let (target, expert) = staged;
        let Some(record) = self.staging.take_beyond(staging) else {
            return Err(Short::Staging);
        };
        // Beyond what its group's reserve (`HostRecords::reserve`) holds
        // past the reads ahead's staging, whose landings it counts already;
        // none for the cover.
        let keep = ahead_staging.map_or(0, |ahead| core.directory.host(core.layers[target].group).reserve().saturating_sub(ahead));
        let claim = match expected {
            Some((step, p)) => core.directory.claim_ahead(target, expert, keep, ahead.taking(), step, p),
            None => core.directory.claim_landing(target, expert, keep, ahead.taking(), (|_, _| (), |_, _| false)),
        };
        let Claim::Taken(grant) = claim else {
            self.staging.release(record);
            return Err(Short::Tier);
        };
        let tier = grant.record();
        let reads = core.table.reads(target, expert, record);
        let bytes = reads.iter().map(crate::experts::source::Read::bytes).sum::<u64>();
        let landing = core.landing(grant, record);
        match post(&core.table, reads, want) {
            Ok(Posting { ticket, join }) => {
                for request in ticket.requests() {
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("ahead"), request, (target as i64, expert as i64, i64::from(want.deadline() == seismic_io::Deadline::None)));
                }
                let ended = Ended::new(shared.events());
                let urgency = Urgency::ahead(want);
                let (end, serving, urgent) = (ended.clone(), shared.clone(), urgency.clone());
                // A read ahead's end: read, its landing is submitted, whose
                // end maps its tier record; failed, it fails nothing (a step
                // waiting for it reads its expert anew).
                join.claimed(move |outcome, failed, _| {
                    unpanicked(&outcome);
                    let read = failed.is_empty() && matches!(outcome, Outcome::Read);
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("ahead_read"), 0, (target as i64, expert as i64, read as i64));
                    if read {
                        // SAFETY: as a demand landing's on the host
                        // (`Self::demand`).
                        let converting = unsafe {
                            Converting::new(landing, urgent, serving.landings.clone(), end, move || {
                                seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("ahead_landed"), 0, (target as i64, expert as i64, 0));
                            })
                        };
                        serving.compute.submit(Arc::new(converting));
                        return;
                    }
                    end.set(false);
                });
                self.arrive(staged, Inbound { record: Some(record), tier, need: Need::Ahead { ticket, ahead }, ended, urgency });
                Ok(bytes)
            }
            // The engine took no request: the read stages and lands nothing and
            // fails nothing (a demand for the expert reads it again, and
            // fails its own step if that read fails).
            Err(_) => {
                core.directory.unlanded(target, HostRecord(tier));
                self.staging.release(record);
                Err(Short::Refused)
            }
        }
    }
}

/// A ring read whole ([`Prefetch::read_rings`]): the layer whose ring it is,
/// which of its rings (a hint ring's lead, or [`FRONT_RING`]), and the step's
/// low word, front rows' guess kinds and entries it holds.
struct RingRead {
    layer: usize,
    ring: usize,
    low: i32,
    guesses: u32,
    entries: Vec<(i32, i32, i32)>,
}

/// A ring's name of one expert ([`Prefetch::admit`]).
pub(super) struct RingName {
    expert: usize,
    name: Name,
}

/// What reading one expert ahead of its request fell short of
/// ([`Prefetch::read_ahead`]): refused by the read engine; no staging
/// record beyond what it leaves; or no record of its group's tier it may
/// take.
enum Short {
    Refused,
    Staging,
    Tier,
}

#[cfg(test)]
mod tests {
    use super::{Ahead, IdleBudget, Inbound, Need, Prefetch};
    use crate::experts::compute::Urgency;
    use crate::experts::events::Events;
    use crate::experts::horizon::Time;
    use crate::experts::landing::Ended;
    use crate::experts::reuse::Reuse;
    use crate::experts::source::Ticket;
    use crate::experts::watcher::passes::{Pass, Passes};
    use crate::experts::watcher::telemetry::ReadPrice;
    use magnitude_cost_model::ReadLine;
    use seismic_io::Want;
    use std::collections::HashSet;
    use std::sync::Arc;

    /// A request's read posted before residency serves it, or left waiting
    /// for tier records, or given up while it runs, holds its staging record
    /// and is in no landing residency tracks: it is outstanding until its
    /// record frees, so no resize or traffic hold finds the load quiet while
    /// it may still write into staging.
    #[test]
    fn a_read_holding_staging_outside_the_landings_is_outstanding() {
        let time = Time::new(&[vec![2]], 32);
        let ranking = Reuse::ranking(&[4], &[0], &[2], time.clone());
        let prefetch = Prefetch::new(2, vec![0], 4, time, std::sync::mpsc::channel().1, &ranking);
        assert!(!prefetch.outstanding(), "nothing read");
        let posted = prefetch.staging.take_up_to(1);
        assert!(prefetch.outstanding(), "a posted read's staging record is taken");
        posted.into_iter().for_each(|record| prefetch.staging.release(record));
        assert!(!prefetch.outstanding(), "its record freed as it ended");
    }

    /// The drive is a single shared resource across layers: when a `Need::Due`
    /// demand read is in flight on layer `l - 1` occupying `>= c_layer`, layer
    /// `l` sees `idle == 0.0` and requires the full drive-bound break-even
    /// `break_even(read, 0.0)` instead of admitting low-probability hints;
    /// and when the drive is idle, admitting one hint consumes the shared
    /// budget across all layers.
    #[test]
    fn shared_drive_idle_budget_counts_demand_and_cross_layer_reads() {
        let layers = 35;
        let bytes = 1_000_000_u64; // 1 MB -> t_bw = 1.0 ms on 1 GB/s line (plus 0.3 ms latency, ignored by pipelined IdleBudget)
        let line = ReadLine::new(0.3e-3, 1e-9).expect("a line");
        let mut passes = Passes::new(Time::new(&[vec![1]], 8));
        // compute = 0.1 - 0.065 = 0.035 s -> c_layer = 0.035 / 35 = 0.001 s (1.0 ms).
        // Drive-bound (0.068 > 0.035), stall = 0.065 / 68 ≈ 0.956 ms <= t_bw = 1.0 ms ->
        // break_even(1.0 ms, 0.0) = (1.0 / 0.956).min(1.0) = 1.0 (only a certain read pays with no idle window).
        passes.timed(Pass { rows: 1, depth: 0, seconds: 0.1, parked: 0.065, misses: 68., drive: 0.068, ..Pass::default() });
        let price = ReadPrice::priced(passes.read_admission(Some(line)));
        assert!((price.transfer_seconds(bytes) - 1e-3).abs() < 1e-12, "IdleBudget uses pipelined t_bw, not lambda + t_bw");
        assert!(!price.admits_with_idle(0.30, bytes, 0.0) && !price.admits_with_idle(0.52, bytes, 0.0) && price.admits_with_idle(1.0, bytes, 0.0));

        let time = Time::new(&[vec![2, 2]], 32);
        let ranking = Reuse::ranking(&[4, 4], &[0, 0], &[2], time.clone());
        let mut prefetch = Prefetch::new(4, vec![0, 0], 4, time, std::sync::mpsc::channel().1, &ranking);
        let events = Arc::new(Events::default());
        let seen = HashSet::from([(1, 0)]);

        // Empty drive: `idle == c_layer == 1.0 ms`, so one 30% hint is admitted
        // and consumes the shared budget (`t_bw = 1.0 ms`), leaving 0.0 for the next candidate.
        let idle = IdleBudget::new(layers, prefetch.drive_queued_bytes(|_| bytes, &seen));
        assert!((idle.get(&price) - 1e-3).abs() < 1e-12);
        assert!(price.admits_with_idle(0.30, bytes, idle.get(&price)));
        idle.consume(&price, price.transfer_seconds(bytes));
        assert_eq!(idle.get(&price), 0.0);
        assert!(!price.admits_with_idle(0.30, bytes, idle.get(&price)));

        // Demand read (`Need::Due`) in flight on layer 0 (l - 1) occupying
        // 1.0 ms >= c_layer: layer 1 (l) sees `idle == 0.0` and must meet the
        // full drive-bound break-even (~0.511).
        prefetch.inbound.insert(
            (0, 0),
            Inbound {
                record: Some(0),
                tier: 0,
                need: Need::Due { ticket: Ticket::empty(), read: 0, device: None },
                ended: Ended::new(&events),
                urgency: Urgency::waited(),
            },
        );
        let queued = prefetch.drive_queued_bytes(|_| bytes, &seen);
        assert_eq!(queued, vec![bytes]);
        let idle = IdleBudget::new(layers, queued);
        assert_eq!(idle.get(&price), 0.0, "demand read on layer l-1 saturates shared drive");
        assert!(!price.admits_with_idle(0.30, bytes, idle.get(&price)), "low-p hint rejected when drive is busy");
        assert!(!price.admits_with_idle(0.52, bytes, idle.get(&price)), "50% hint rejected on saturated drive with no park overhead");
        assert!(price.admits_with_idle(1.0, bytes, idle.get(&price)), "certain read still admitted");

        // An unbegun ahead read in `seen` (being re-evaluated in this pass) is
        // not double-counted, while an out-of-pass ahead read on another layer
        // IS counted on the shared drive.
        prefetch.inbound.clear();
        prefetch.inbound.insert(
            (1, 0),
            Inbound {
                record: Some(0),
                tier: 0,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Cover },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.)),
            },
        );
        assert!(prefetch.drive_queued_bytes(|_| bytes, &seen).is_empty(), "in-pass unbegun ahead read is not pre-counted");
        prefetch.inbound.insert(
            (0, 1),
            Inbound {
                record: Some(1),
                tier: 1,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Cover },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.)),
            },
        );
        let queued = prefetch.drive_queued_bytes(|_| bytes, &seen);
        assert_eq!(queued, vec![bytes], "out-of-pass ahead read on layer 0 counts against shared drive");
        assert_eq!(IdleBudget::new(layers, queued).get(&price), 0.0);
    }
}
