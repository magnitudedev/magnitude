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
use crate::experts::interest::{Act, Admit, Class, Ledger, Name, Origin, Read as ReadState, Scheduled, Source, When};
use super::Step;
use crate::experts::landing::{Answer, DeviceLanding, Ended, PipelinedLanding};
use crate::experts::source::{Outcome, Read, Ticket};
use crate::experts::staging::{Staged, StagingPool};
use magnitude_cost_model::LinkIdleBudget;
use seismic_io::Want;
use magnitude_kernels::constants;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What lands a read once it ended and its records are claimed.
type Lands = Box<dyn FnOnce(Outcome, &[usize]) + Send>;

/// Per-tensor callback invoked once each routed tensor's NVMe read ends.
type TensorLand = Arc<dyn Fn(usize, Outcome) + Send + Sync>;

/// A read's join ([`Join`]): what the claims registered while it ran, or
/// how it ended, plus per-tensor read completion for pipelined conversion.
struct JoinState {
    parties: Parties,
    tensor_ended: Vec<Option<Outcome>>,
    tensor_hook: Option<TensorLand>,
}

enum Parties {
    Reading(Vec<Lands>),
    Ended(Outcome, Vec<usize>),
}

/// The join of a read that lands experts into the host tier: its end,
/// per-tensor ends for pipelined conversion, and each claim of the records
/// some of its experts land into (`Prefetch::demand`, `Prefetch::read_ahead`,
/// and a request's accounting, [`account`]). A request's read is submitted by
/// the request service's look before residency decides its tier records
/// (`decode::Look`), so either may come first: whichever is second runs each
/// landing, so neither waits for the other, and an expert whose record is
/// short never holds back its request's others.
pub(super) struct Join(Mutex<JoinState>);

impl Join {
    fn state(&self) -> std::sync::MutexGuard<'_, JoinState> {
        self.0.lock().expect("a read's join is never left by a panic")
    }

    /// Tensor `tensor`'s NVMe read ended as `outcome`: dispatch to any
    /// registered per-tensor hook, or record it for replay when claimed.
    fn tensor_read_ended(&self, tensor: usize, outcome: Outcome) {
        let hook = {
            let mut state = self.state();
            if tensor < state.tensor_ended.len() {
                state.tensor_ended[tensor] = Some(outcome.clone());
            }
            state.tensor_hook.clone()
        };
        if let Some(hook) = hook {
            hook(tensor, outcome);
        }
    }

    /// Register a per-tensor callback for pipelined conversion: replays any
    /// tensors whose NVMe reads have already ended, and fires immediately for
    /// each remaining tensor as its read finishes.
    pub(super) fn claimed_per_tensor(&self, hook: impl Fn(usize, Outcome) + Send + Sync + 'static) {
        let hook: TensorLand = Arc::new(hook);
        let ready: Vec<(usize, Outcome)> = {
            let mut state = self.state();
            state.tensor_hook = Some(hook.clone());
            state
                .tensor_ended
                .iter()
                .enumerate()
                .filter_map(|(t, out)| out.clone().map(|o| (t, o)))
                .collect()
        };
        for (tensor, outcome) in ready {
            hook(tensor, outcome);
        }
    }

    /// Its read ended as `outcome`, the reads `failed` failing: every
    /// claim made lands.
    fn read_ended(&self, outcome: Outcome, failed: &[usize]) {
        let ended = Parties::Ended(outcome.clone(), failed.to_vec());
        let Parties::Reading(claims) = std::mem::replace(&mut self.state().parties, ended) else {
            unreachable!("a read ends once");
        };
        for land in claims {
            land(outcome.clone(), failed);
        }
    }

    /// Records are claimed for some of its experts: `land` lands them once
    /// its read ended, now where it has. It must not wait for a read.
    pub(super) fn claimed(&self, land: impl FnOnce(Outcome, &[usize]) + Send + 'static) {
        let mut state = self.state();
        match &mut state.parties {
            Parties::Reading(claims) => claims.push(Box::new(land)),
            Parties::Ended(outcome, failed) => {
                let (outcome, failed) = (outcome.clone(), failed.clone());
                drop(state);
                land(outcome, &failed);
            }
        }
    }
}

/// A read submitted, and its join.
pub(super) struct Posting {
    pub(super) ticket: Ticket,
    pub(super) join: Arc<Join>,
}

/// Request `reads` across `tensor_count` routed tensors from `table`, wanted
/// as `want`, joined to the claim of what they land into ([`Join`]): the one
/// way the watcher reads experts into staging. Fails only where the read
/// engine takes no request.
pub(super) fn post(
    table: &ReadTable,
    reads: Vec<Read>,
    tensor_count: usize,
    want: Want,
    alignment: crate::experts::source::ReadAlignment,
) -> Result<Posting, String> {
    let ticket = table.source().request_per_tensor(reads, tensor_count, want, alignment)?;
    let join = Arc::new(Join(Mutex::new(JoinState {
        parties: Parties::Reading(Vec::new()),
        tensor_ended: vec![None; ticket.tensor_count()],
        tensor_hook: None,
    })));
    let tensor_join = join.clone();
    let ended = join.clone();
    ticket.on_end_with_tensors(
        move |tensor, outcome| tensor_join.tensor_read_ended(tensor, outcome),
        move |outcome, failed| ended.read_ended(outcome, failed),
    );
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
        posting.join.claimed(move |outcome, _| {
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
/// landing line, once every read and every landing have ended.
pub(super) struct Landings {
    layer: usize,
    count: usize,
    /// Remaining reads/landings and the latest read/landing timestamps.
    state: Mutex<LandingsState>,
}

struct LandingsState {
    reads_left: usize,
    landings_left: usize,
    last_read: Option<Instant>,
    last_landed: Option<Instant>,
}

impl LandingsState {
    fn finish(&self) -> Option<f64> {
        if self.reads_left == 0 && self.landings_left == 0 {
            let read = self.last_read?;
            let landed = self.last_landed?;
            Some(landed.saturating_duration_since(read).as_secs_f64())
        } else {
            None
        }
    }
}

impl Landings {
    pub(super) fn new(layer: usize, count: usize) -> Arc<Self> {
        Arc::new(Self {
            layer,
            count,
            state: Mutex::new(LandingsState {
                reads_left: count,
                landings_left: count,
                last_read: None,
                last_landed: None,
            }),
        })
    }

    fn record_read(&self, at: Instant) -> Option<f64> {
        let mut state = self.state.lock().expect("a request's landings");
        state.reads_left -= 1;
        state.last_read = Some(state.last_read.map_or(at, |prev| prev.max(at)));
        state.finish()
    }

    fn record_ended(&self, at: Instant) -> Option<f64> {
        let mut state = self.state.lock().expect("a request's landings");
        state.landings_left -= 1;
        state.last_landed = Some(state.last_landed.map_or(at, |prev| prev.max(at)));
        state.finish()
    }

    /// One of its reads ended whole.
    fn read(&self, shared: &Shared) {
        if let Some(seconds) = self.record_read(Instant::now()) {
            shared.telemetry.landings_ended(self.layer, self.count, seconds);
        }
    }

    /// One of them ended.
    fn ended(&self, shared: &Shared) {
        if let Some(seconds) = self.record_ended(Instant::now()) {
            shared.telemetry.landings_ended(self.layer, self.count, seconds);
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
    bytes: u64,
    need: Need,
    ended: Ended,
    urgency: Arc<Urgency>,
    /// True once this inbound's drive read has finished (even if its
    /// `DeviceLanding` or host `Converting` has not yet settled).
    read_done: Arc<AtomicBool>,
    /// True once a parked request already counted this in-flight prefetch as a
    /// `staged_hits` hit in `Decode::serve` (prevents double-counting on land).
    counted_hit: bool,
}

/// Metadata for a prefetched expert that landed into the tier ahead of need,
/// held until routed (`landed_hits` / `carryover_hits`) or evicted unused
/// (`carryover_wasted`).
#[derive(Clone, Copy)]
pub(super) struct LandedPrefetch {
    pub(super) bytes: u64,
    pub(super) carryover: bool,
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
    /// (`experts::interest`): its source, the route at `issued` that
    /// published it, and the target step it is read ahead for. The ledger
    /// orders, reprices and cancels it.
    Interest { source: Source, issued: At, target_step: Step },
    /// The covering tier's ([`Prefetch::cover`]), undated, which no
    /// interest names.
    Cover,
    /// Carryover prefetch from the reuse ranking (`Directory::reuse_now`),
    /// issued during an MTP `DraftWindow` (`target_step == 0`, bound to the
    /// next step on `Told::Began`) or continuously across the remaining
    /// non-drive pass window of an active verify/decode burst (`target_step`
    /// = current burst `B` for layers `l+1..L-1` or `B+1` for layers `0..=l`).
    /// Unbegun carryover reads are cancelled by `Prefetch::pass` once their
    /// `(target_step, layer)` has passed, while `seismic_io` strictly serves
    /// `Want::now()` demand reads ahead of `Want::undated` carryover.
    Carryover { target_step: Step },
}

impl Ahead {
    /// How its landing takes its tier record.
    fn taking(self) -> Taking {
        match self {
            Self::Interest { .. } | Self::Carryover { .. } => Taking::Ahead,
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

    /// The step that will read it: a request's step for demand reads, or the
    /// target step for read-aheads; none for the cover's or carryover's.
    fn read(&self) -> Option<Step> {
        match &self.need {
            Need::Due { read, .. } => Some(*read),
            Need::Ahead { ahead: Ahead::Interest { target_step, .. }, .. } => Some(*target_step),
            Need::Ahead { ahead: Ahead::Cover | Ahead::Carryover { .. }, .. } => None,
        }
    }
}

/// A landing in flight a request waits for ([`Prefetch::reading`]): read
/// ahead for an interest the route at its `At` published, a carryover
/// prefetch from live reuse history, or one no prediction made (the cover's,
/// or another request's).
pub(super) enum Reading {
    Predicted(At),
    Carryover,
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
            Source::Ahead(row) => usize::from(row.index),
            Source::Token => crate::experts::DEPTHS,
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
    let Some(count) = usize::try_from(count).ok().filter(|&count| count as u64 <= choices) else {
        // Re-check the seqlock sequence before treating an out-of-range count
        // as an invariant violation: a concurrent writer sets sequence to 0
        // before rewriting and a new nonzero sequence after.
        if ring.load(base + constants::RING_SEQUENCE as usize) != low {
            return None;
        }
        panic!(
            "a ring of {count} entries (0x{:08x}, low={low}, seen={seen}, guesses=0x{guesses:08x}, max={choices})",
            count as u32
        );
    };
    let entry = |index: usize, word: u64| ring.load(base + (constants::HINT_ENTRY(index as u64) + word) as usize);
    let entries = (0..count)
        .map(|index| {
            let word = |word: u64| entry(index, word);
            (word(constants::HINT_EXPERT), word(constants::HINT_RANK), word(constants::HINT_ROW))
        })
        .collect();
    (ring.load(base + constants::RING_SEQUENCE as usize) == low).then_some((low, guesses, entries))
}

/// The staging records and the reads into them (Path 1: NVMe -> RAM / bank).
/// The shared `NameLedger` (`experts::interest::Ledger`) is owned by
/// `Residency` and passed in so both Path 1 (`Prefetch`) and Path 0
/// (`LinkMover`) schedule from one ledger.
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
    /// Per layer, the low word of the last step its front ring was read for.
    seen: Vec<i32>,
    /// Speculative prefetch bytes issued during the current active decode/verify
    /// step (reset on `Told::Began` and `Told::DraftWindow`), capped by
    /// `PathAdmission::prefetch_byte_budget` in slack-bearing regimes.
    step_prefetch_bytes: u64,
    /// Set whenever a drive read completes (including `DeviceLanding` demand
    /// reads on unified memory that stay in `inbound` until the next layer
    /// parks), so `Residency::rings` / `Residency::hasten` immediately drives
    /// lookahead reads into the newly freed drive window.
    reads_finished: Arc<AtomicBool>,
    /// Prefetched experts that landed into the tier ahead of need, tracked
    /// until routed (`landed_hits` / `carryover_hits`) or evicted unused
    /// (`carryover_wasted`).
    landed_prefetches: HashMap<Staged, LandedPrefetch>,
    /// Currently active decode/verify steps for `DriveEngagement`'s
    /// active-burst wall clock (excludes prompt steps and inter-request idle).
    active_decode_steps: HashSet<Step>,
    /// Row count `M` of the latest decode/verify burst (`1` for plain decode,
    /// `4..6` for MTP verify), used to lift single-token reuse probability `p_1`
    /// to burst hit probability `1 - (1 - p_1)^M` when pricing carryover.
    active_decode_rows: usize,
}

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

/// Lazy Drive-path wrapper around [`magnitude_cost_model::LinkIdleBudget`] so
/// due-only derivations never lock telemetry. Spans the remaining compute
/// window of the active verify/decode burst (`remaining_layers * C / L`) plus
/// any active `DraftWindow` so the drive stays saturated across the full burst.
struct DriveIdleBudget {
    layers: usize,
    remaining_layers: usize,
    queued_bytes: Vec<u64>,
    draft_idle: f64,
    budget: std::cell::Cell<Option<LinkIdleBudget>>,
}

impl DriveIdleBudget {
    fn new(layers: usize, remaining_layers: usize, queued_bytes: Vec<u64>, draft_idle: f64) -> Self {
        Self { layers, remaining_layers, queued_bytes, draft_idle, budget: std::cell::Cell::new(None) }
    }

    fn get(&self, price: &super::telemetry::ReadPrice) -> f64 {
        if let Some(budget) = self.budget.get() {
            return budget.remaining();
        }
        let c_window = if self.layers == 0 {
            0.0
        } else {
            (self.remaining_layers.clamp(1, self.layers) as f64 / self.layers as f64) * price.non_drive_pass_seconds()
        };
        let queued: f64 = self.queued_bytes.iter().map(|&bytes| price.transfer_seconds(bytes)).sum();
        let budget = LinkIdleBudget::new(c_window, self.draft_idle, queued);
        let rem = budget.remaining();
        self.budget.set(Some(budget));
        rem
    }

    fn consume(&self, price: &super::telemetry::ReadPrice, transfer_seconds: f64) {
        let _ = self.get(price);
        if let Some(mut budget) = self.budget.get() {
            budget.consume(transfer_seconds);
            self.budget.set(Some(budget));
        }
    }
}

/// Admission of a read ahead of an expert's `bytes`, as the ledger asks it
/// ([`Admit`]): at the break-even of a read of them given the shared drive's
/// remaining idle compute window (`ReadPrice::admits_with_idle`).
struct Admitting<'a> {
    priced: &'a Priced<'a>,
    bytes: u64,
    idle: &'a DriveIdleBudget,
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
    /// `records` staging records, every one free, for `layers` streamed layers.
    pub(super) fn new(records: u32, layers: usize) -> Self {
        Self {
            staging: Arc::new(StagingPool::new(records)),
            inbound: HashMap::new(),
            past: Point::START,
            seen: vec![0; layers],
            step_prefetch_bytes: 0,
            reads_finished: Arc::new(AtomicBool::new(false)),
            landed_prefetches: HashMap::new(),
            active_decode_steps: HashSet::new(),
            active_decode_rows: 1,
        }
    }

    /// Track a decode/verify step beginning on `DriveEngagement`'s burst clock
    /// and record its active burst row count `m_active` (`1` for plain decode,
    /// `M > 1` for multi-row verify).
    pub(super) fn decode_step_began(&mut self, step: Step, m_active: usize, shared: &Shared) {
        self.active_decode_rows = m_active.max(1);
        if self.active_decode_steps.insert(step) {
            shared.telemetry.engagement().on_decode_began(Instant::now());
        }
    }

    /// Track a step ending: if it was an active decode/verify step, close its
    /// burst clock span.
    pub(super) fn step_ended(&mut self, step: Step, shared: &Shared) {
        if self.active_decode_steps.remove(&step) {
            shared.telemetry.engagement().on_decode_ended(Instant::now());
        }
    }

    /// Consume whether any drive read finished since last checked.
    pub(super) fn take_reads_finished(&self) -> bool {
        self.reads_finished.swap(false, Ordering::AcqRel)
    }

    /// Called by `Residency::learn` when `Told::Began` arrives: reset per-step
    /// speculative byte accounting and bind any untargeted draft-window
    /// carryover reads (`target_step == 0`) to `step` without cancelling them.
    /// `seismic_io` already serves `Want::now()` demand reads strictly ahead of
    /// `Want::undated` carryover, and `Prefetch::pass` cancels unbegun reads
    /// once `(step, layer)` has passed.
    pub(super) fn on_began(&mut self, step: Step) {
        self.step_prefetch_bytes = 0;
        for inbound in self.inbound.values_mut() {
            if let Need::Ahead { ahead: Ahead::Carryover { target_step }, .. } = &mut inbound.need {
                if *target_step == 0 {
                    *target_step = step;
                }
            }
        }
    }

    /// Called by `Residency::learn` when `Told::DraftWindow` arrives.
    pub(super) fn on_draft_window(&mut self) {
        self.step_prefetch_bytes = 0;
    }

    /// Consume a previously landed prefetch record when a step routes `staged`.
    pub(super) fn take_landed_prefetch(&mut self, staged: Staged) -> Option<LandedPrefetch> {
        self.landed_prefetches.remove(&staged)
    }

    /// Mark an in-flight prefetch as already counted as a `staged_hits` hit so
    /// it is not double-counted when it subsequently lands.
    pub(super) fn mark_inbound_hit(&mut self, staged: Staged) {
        if let Some(inbound) = self.inbound.get_mut(&staged) {
            inbound.counted_hit = true;
        }
    }

    /// Drop any landed prefetch entries that were evicted from the tier before
    /// being used, counting evicted carryover entries as `carryover_wasted`.
    pub(super) fn prune_evicted_prefetches(&mut self, core: &Core, shared: &Shared) {
        self.landed_prefetches.retain(|&(layer, expert), lp| {
            let still_present =
                core.directory.held(layer, expert).is_some() || core.directory.located(layer, expert) != Located::Lacked;
            if !still_present && lp.carryover {
                shared.telemetry.engagement().record_carryover_wasted();
            }
            still_present
        });
    }

    /// The low word of the last step `layer`'s front ring was read for.
    #[cfg(test)]
    pub(super) fn front_read(&self, layer: usize) -> i32 {
        self.seen[layer]
    }

    /// Track request `naming`'s demand read `named` at `at`, its record
    /// claimed: its read's end lands it where its layer's bank places, on
    /// the device ([`DeviceLanding`]: appended to the layer's landing list,
    /// under the step's ownership, then answered) or else on the host
    /// (submitted to the executor, which converts and maps its record,
    /// records its end, then answers; one of the request's host `landings`,
    /// whose line it measures), or fails it. A request's reads are
    /// accounted apart, once ([`account`]).
    pub(super) fn demand(&mut self, core: &Core, named: Named, naming: Naming, at: Point, landings: &Arc<Landings>, shared: &Arc<Shared>) {
        let Named { expert, posting: Posting { ticket, join }, staged, grant, rows, answer } = named;
        let (ended, urgency, tier) = (Ended::new(shared.events()), Urgency::waited(), grant.record());
        let read_done = Arc::new(AtomicBool::new(false));
        let device = core.layers[naming.layer]
            .landing
            .as_ref()
            .map(|list| Arc::new(DeviceLanding::new(list.clone(), at, ended.clone(), shared.landings.clone())));
        let (serving, end, done, finished) = (shared.clone(), ended.clone(), read_done.clone(), self.reads_finished.clone());
        let bytes = core.layers[naming.layer].expert_bytes();
        shared.telemetry.engagement().read_started();
        match &device {
            Some(device) => {
                let device = device.clone();
                let tensor_leads = ticket.tensor_leads().to_vec();
                join.claimed(move |outcome, failed| {
                    done.store(true, Ordering::Release);
                    finished.store(true, Ordering::Release);
                    serving.telemetry.engagement().read_finished();
                    if !read_named(&serving, naming, (&rows, &answer), &outcome, failed, &end) {
                        serving.events().notify();
                        return;
                    }
                    // Appended while the step owns its slot: a read ending
                    // after its step did appends nothing, its end not
                    // landed; one the list has moved past (the device past
                    // this step's launch) ends not landed and answers none.
                    match serving.steps.while_owned(naming.sequence, |_| device.append(grant, staged, &tensor_leads)) {
                        Some(true) => answer.publish(Some(tier)),
                        Some(false) => answer.publish(None),
                        None => end.set(false),
                    }
                    // Wake residency/requests now that the drive slot is free:
                    // `DeviceLanding::end` waits until the next layer parks, so
                    // without this notification lookahead reads cannot enter the
                    // idle compute window between layers on unified memory.
                    serving.events().notify();
                });
            }
            None => {
                let landing = core.landing_with_leads(grant, staged, ticket.tensor_leads());
                let tensor_count = landing.tensors.len();
                let (landings, urgent) = (landings.clone(), urgency.clone());
                let completed = serving.clone();
                let answer_ok = answer.clone();
                let landings_ok = landings.clone();
                let pipelined = PipelinedLanding::new(
                    landing.grant,
                    tensor_count,
                    urgent,
                    serving.landings.clone(),
                    end,
                    move || {
                        answer_ok.publish(Some(tier));
                        landings_ok.ended(&completed);
                    },
                );
                let tensors: Arc<Mutex<Vec<Option<crate::experts::landing::LandingTensor>>>> =
                    Arc::new(Mutex::new(landing.tensors.into_iter().map(Some).collect()));
                let serving_t = serving.clone();
                let pipelined_t = pipelined.clone();
                join.claimed_per_tensor(move |tensor_idx, outcome| {
                    unpanicked(&outcome);
                    let tensor = tensors
                        .lock()
                        .expect("pipelined landing tensors")
                        .get_mut(tensor_idx)
                        .and_then(Option::take);
                    let Some(tensor) = tensor else {
                        return;
                    };
                    if matches!(outcome, Outcome::Read) {
                        // SAFETY: `tensor_idx`'s NVMe read has ended, its
                        // staging record is not reused until `PipelinedLanding`
                        // sets `ended`, and its tier record is ready.
                        unsafe { pipelined_t.submit_tensor(tensor, &serving_t.compute) };
                    } else {
                        pipelined_t.fail_tensor();
                    }
                });
                join.claimed(move |outcome, failed| {
                    done.store(true, Ordering::Release);
                    finished.store(true, Ordering::Release);
                    serving.telemetry.engagement().read_finished();
                    if failed.is_empty() {
                        serving.landings.read(naming.named.elapsed());
                        landings.read(&serving);
                    } else {
                        if let Outcome::Failed(reason) = &outcome {
                            serving.steps.fail_request_rows(naming.sequence, naming.layer, &rows, reason.clone());
                        }
                        answer.publish(None);
                    }
                    serving.events().notify();
                });
            }
        }
        let need = Need::Due { ticket, read: naming.sequence, device };
        self.arrive((naming.layer, expert), Inbound { record: Some(staged), tier, bytes, need, ended, urgency, read_done, counted_hit: false });
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
            Need::Ahead { ahead: Ahead::Carryover { .. }, .. } => Reading::Carryover,
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
            Need::Ahead { ahead: Ahead::Interest { source: Source::Token, .. }, .. } => 5,
            Need::Ahead { ahead: Ahead::Interest { .. }, .. } => 1,
            Need::Ahead { ahead: Ahead::Cover, .. } => 2,
            Need::Ahead { ahead: Ahead::Carryover { .. }, .. } => 4,
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
    pub(super) fn want(&mut self, core: &Core, ledger: &mut Ledger, shared: &Arc<Shared>, at: At, experts: &[usize]) {
        let (sequence, layer) = (at.point.sequence, at.point.layer);
        ledger.reach(sequence, layer);
        let now = Instant::now();
        for &expert in experts {
            let staged = (layer, expert);
            let reading = matches!(self.read_state(core, staged), ReadState::Reading { .. });
            ledger.requested(layer, expert, reading);
            let name = Name { source: Source::Exact, when: When::Step(sequence), rank: 0, from: at, candidate: false, lineage: None };
            ledger.publish(layer, expert, name, now);
            if reading {
                self.hasten(staged);
            }
        }
        Self::spend(ledger, shared);
    }

    pub(super) fn pass(&mut self, core: &mut Core, ledger: &mut Ledger, shared: &Arc<Shared>, served: Option<(Step, usize)>) {
        let steps = &shared.steps;
        let mut changed = Vec::new();
        for layer in 0..core.layers.len() {
            let (routed, file) = (ledger.routed(layer), &core.layers[layer].file);
            let passed = ledger.pass(layer, Instant::now(), |interest| {
                let step = interest.step();
                served.is_some_and(|(sequence, at)| (step, layer) <= (sequence, at))
                    || steps.ended(step)
                    || (routed >= step && !file.requested(step, 0))
            });
            changed.extend(passed.into_iter().map(|expert| (layer, expert)));
        }
        if !changed.is_empty() {
            self.derive(core, ledger, shared, Some(changed.as_slice()));
        }
        // Cancel any unbegun carryover reads whose `(target_step, layer)`
        // the device has already passed, freeing staging for upcoming layers.
        for (&(layer, _), inbound) in &mut self.inbound {
            let Need::Ahead { ticket, ahead: Ahead::Carryover { target_step } } = &mut inbound.need else {
                continue;
            };
            if *target_step == 0 || inbound.urgency.is_waited() || ticket.begun() || ticket.cancelled() {
                continue;
            }
            let passed = served.is_some_and(|(sequence, at)| (*target_step, layer) <= (sequence, at))
                || steps.ended(*target_step);
            if passed {
                ticket.cancel();
            }
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
            let Need::Ahead { ticket, ahead: Ahead::Cover | Ahead::Carryover { .. } } = &mut inbound.need else {
                continue;
            };
            if !inbound.urgency.is_waited() && !ticket.cancelled() {
                ticket.cancel();
            }
        }
    }

    /// Staging and what is in flight into it, for the exchange's watchdog.
    pub(super) fn describe(&self) -> String {
        let [mut hints, mut fronts, mut covers, mut carryover, mut due, mut running] = [0; 6];
        for inbound in self.inbound.values() {
            match inbound.need {
                Need::Due { .. } => due += 1,
                Need::Ahead { ahead: Ahead::Interest { source: Source::Ahead(_), .. }, .. } => fronts += 1,
                Need::Ahead { ahead: Ahead::Interest { .. }, .. } => hints += 1,
                Need::Ahead { ahead: Ahead::Cover, .. } => covers += 1,
                Need::Ahead { ahead: Ahead::Carryover { .. }, .. } => carryover += 1,
            }
            running += usize::from(inbound.ticket().running());
        }
        format!(
            "staging {} free of {}; inbound: hints {hints}, front {fronts}, covers {covers}, carryover {carryover}, due {due}; reads running for {running}",
            self.staging.free(),
            self.staging.records(),
        )
    }

    /// Number of drive reads or landings currently in flight (staging records held).
    pub(super) fn in_flight(&self) -> usize {
        self.staging.records().saturating_sub(self.staging.free())
    }

    /// Whether a read or its landing is outstanding: any staging record
    /// taken. Every read into staging holds its record from before it is
    /// submitted until its landing's end is learned, or, a read given up,
    /// until it ends: an expert inbound, a request's posted read before
    /// residency serves it, one waiting for tier records, or one discarded.
    /// Each record's freeing is an event of the load.
    pub(super) fn outstanding(&self) -> bool {
        self.in_flight() > 0
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

    /// Settle the landings that ended: returns `(settled, newly_landed)` where
    /// `newly_landed` lists every `(layer, expert)` that just landed into
    /// Tier 1 (used for Path 1 -> Path 0 pipeline chain-through to `LinkMover`).
    pub(super) fn settle(&mut self, directory: &mut Directory, ledger: &mut Ledger, shared: &Shared) -> (bool, Vec<Staged>) {
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
                let prefetch_kind = match &inbound.need {
                    Need::Ahead { ahead: Ahead::Interest { .. }, .. } => Some((false, false)),
                    Need::Ahead { ticket, ahead: Ahead::Carryover { .. } } => Some((true, ticket.begun())),
                    Need::Due { .. } | Need::Ahead { ahead: Ahead::Cover, .. } => None,
                };
                ended.push((inbound.record, staged, inbound.tier, inbound.read(), landed, inbound.bytes, inbound.counted_hit, prefetch_kind));
                false
            }
            None => true,
        });
        let settled = freed || !ended.is_empty();
        let mut newly_landed = Vec::new();
        for (record, staged, tier, read, landed, bytes, counted_hit, prefetch_kind) in ended {
            if landed {
                newly_landed.push(staged);
            }
            if let Some((carryover, begun)) = prefetch_kind {
                if landed && !counted_hit {
                    self.landed_prefetches.insert(staged, LandedPrefetch { bytes, carryover });
                } else if !landed && carryover && begun && !counted_hit {
                    shared.telemetry.engagement().record_carryover_wasted();
                }
            }
            self.ended(directory, ledger, record, staged, tier, read, landed);
        }
        (settled, newly_landed)
    }

    /// A landing of `staged` from staging record `record` (none, freed
    /// already) into tier record `tier`, read at step `read` (none for the
    /// cover's), ended: landed or not.
    fn ended(&mut self, directory: &mut Directory, ledger: &mut Ledger, record: Option<u32>, staged: Staged, tier: u32, read: Option<Step>, landed: bool) {
        let (layer, expert) = staged;
        match landed {
            true => {
                directory.landed(layer, HostRecord(tier), expert, read);
                ledger.landed(layer, expert, Instant::now());
            }
            false => directory.unlanded(layer, HostRecord(tier)),
        }
        if let Some(record) = record {
            self.staging.release(record);
        }
    }

    /// Read changed rings (before `Residency::learn` drains `Told`).
    pub(super) fn read_rings_raw(&mut self, core: &Core) -> Vec<RingRead> {
        self.read_rings(core)
    }

    /// Resolve each live ring's step and device progress point after
    /// `Residency::learn` has drained `Told`.
    pub(super) fn resolve_rings(&mut self, core: &Core, ledger: &mut Ledger, shared: &Shared, read: Vec<RingRead>, told: bool) -> RingsRead {
        let any = !read.is_empty();
        let (named, latest) = self.name(core, ledger, shared, read);
        RingsRead { named, latest, any, told }
    }

    /// Admit `named` from `resolve_rings` into `ledger`, and derive Drive reads
    /// whenever rings arrived, events were told, or capacity freed.
    pub(super) fn finish_rings(
        &mut self,
        core: &mut Core,
        ledger: &mut Ledger,
        shared: &Arc<Shared>,
        RingsRead { named, any, told, .. }: RingsRead,
        freed: bool,
        served: &[Step],
    ) -> bool {
        for (target, names) in named {
            self.admit(core, ledger, shared, target, names, served);
        }
        if any || told || freed {
            if !core.directory.tier_complete() {
                self.drive(core, ledger, shared);
            } else {
                self.pass(core, ledger, shared, None);
            }
            true
        } else {
            false
        }
    }

    /// Each layer's front ring that changed since it was read.
    fn read_rings(&mut self, core: &Core) -> Vec<RingRead> {
        let words = core.extents.words;
        let mut read = Vec::new();
        for (layer, streamed) in core.layers.iter().enumerate() {
            if let Some((low, guesses, entries)) = ring_entries(&streamed.front, 0, self.seen[layer], words.choices) {
                self.seen[layer] = low;
                read.push(RingRead { layer, low, guesses, entries });
            }
        }
        read
    }

    /// The names the rings `read` make, by the layer each names, of the steps
    /// not ended: a ring of a step that ended is skipped, and the layers its
    /// route showed done are the ledger's to know ([`Ledger::route_seen`]).
    fn name(&mut self, core: &Core, ledger: &mut Ledger, shared: &Shared, read: Vec<RingRead>) -> (Vec<(usize, Vec<RingName>)>, Option<Point>) {
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
        for RingRead { layer, low, guesses, entries } in read {
            let Some(from) = shared.steps.at_low_word(low, layer, layers) else {
                continue;
            };
            latest = Some(latest.map_or(from.point, |prev| prev.max(from.point)));
            let sequence = from.point.sequence;
            // Published once the route's request was.
            ledger.route_seen(sequence, layer);
            // A front row that ran on no guess names nothing.
            let names = entries
                .into_iter()
                .filter_map(|entry| {
                    let (expert, rank, row) = checked(layer, entry);
                    let Origin::Ahead { row: ahead, when, lineage } = ledger.row(sequence, row, guesses)? else {
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
                ledger.supersede(layer, request, sequence);
            }
            named.push((layer, names));
        }
        (named, latest)
    }

    /// Whether `staged` is a read ahead's candidate: an expert absent from
    /// VRAM that either lives only in the file (`Located::Lacked`) or, on a
    /// 100%-RAM link-only bank (`tier_complete() && !all_groups_whole()`),
    /// lives in `HostTier` and is eligible for Path 0 `LinkMover` staging.
    fn candidate(core: &Core, staged: Staged) -> bool {
        let (layer, expert) = staged;
        if core.directory.held(layer, expert).is_some() {
            return false;
        }
        if core.directory.tier_complete() && !core.directory.all_groups_whole() {
            return true;
        }
        core.directory.located(layer, expert) == Located::Lacked
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
    fn admit(&mut self, core: &Core, ledger: &mut Ledger, shared: &Arc<Shared>, target: usize, names: Vec<RingName>, served: &[Step]) {
        let recording = seismic_io::timeline::recording();
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
            ledger.reach(name.from.point.sequence, name.from.point.layer);
            ledger.publish(target, expert, Name { candidate, ..name }, now);
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
    /// ledger, the soonest wanted first ([`Self::derive`]), and fill any idle
    /// drive window (MTP `DraftWindow` or in-step slack between `routed_land`
    /// parks) with victim-admitted carryover prefetches
    /// ([`Self::prefetch_carryover`]). True when any was read.
    pub(super) fn drive(&mut self, core: &mut Core, ledger: &mut Ledger, shared: &Arc<Shared>) -> bool {
        self.pass(core, ledger, shared, None);
        let mut read = self.derive(core, ledger, shared, None);
        read |= self.prefetch_carryover(core, ledger, shared);
        read
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
    /// every active demand read (`Need::Due` whose drive read has not yet
    /// finished), plus every already-begun active ahead read or unbegun,
    /// uncancelled ahead read outside `seen` (partial `derive` from `pass`),
    /// across all layers.
    fn drive_queued_bytes(&self, expert_bytes: impl Fn(usize) -> u64, seen: &HashSet<Staged>) -> Vec<u64> {
        self.inbound
            .iter()
            .filter(|(staged, inbound)| {
                if inbound.read_done.load(Ordering::Acquire) {
                    return false;
                }
                match &inbound.need {
                    Need::Due { .. } => true,
                    Need::Ahead { ticket, .. } => ticket.begun() || (!ticket.cancelled() && !seen.contains(staged)),
                }
            })
            .map(|(staged, _)| expert_bytes(staged.0))
            .collect()
    }

    /// Current active `(step, layer)` in the running decode/verify pass,
    /// combining `self.past`, `ledger.reached()`, and `ledger.latest_active_step()`.
    fn active_point(&self, ledger: &Ledger) -> (Step, usize) {
        let (reached_step, reached_layer) = ledger.reached();
        let (step, current_layer) = if (self.past.sequence, self.past.layer) > (reached_step, reached_layer) {
            (self.past.sequence, self.past.layer)
        } else {
            (reached_step, reached_layer)
        };
        match ledger.latest_active_step() {
            Some(active) if active > step => (active, 0),
            _ => (step, current_layer),
        }
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
    fn derive(&mut self, core: &mut Core, ledger: &mut Ledger, shared: &Arc<Shared>, only: Option<&[Staged]>) -> bool {
        let reserve = self.reserve(core, false);
        let price = Priced::new(&shared.telemetry);
        let owed = if only.is_none() { ledger.take_owed() } else { Vec::new() };
        let mut experts = match only {
            Some(only) => only.to_vec(),
            None => ledger.wanted(),
        };
        experts.extend(owed);
        let mut seen = HashSet::new();
        experts.retain(|&staged| seen.insert(staged));
        let draft_idle = ledger.draft_window_remaining(Instant::now());
        let (_, current_layer) = self.active_point(ledger);
        let remaining_layers = core.layers.len().saturating_sub(current_layer.saturating_add(1)).max(1);
        let scheduled = {
            let core: &Core = core;
            let queued_bytes = self.drive_queued_bytes(|layer| core.layers[layer].expert_bytes(), &seen);
            let idle = DriveIdleBudget::new(core.layers.len(), remaining_layers, queued_bytes, draft_idle);
            // Each read's standing, taken before the ledger decides on it.
            let states = experts.iter().map(|&staged| (staged, self.read_state(core, staged))).collect::<HashMap<_, _>>();
            let read = |layer, expert| states[&(layer, expert)];
            ledger.schedule_for_path(
                magnitude_cost_model::PathKind::Drive,
                experts,
                read,
                |layer| Admitting { priced: &price, bytes: core.layers[layer].expert_bytes(), idle: &idle },
            )
        };
        let wanted = |class: Class| match class {
            Class::Due => Want::now(),
            // Ordered on its link as the ledger orders it: by the step
            // that wants it, the soonest first, then its probability of
            // use ("Order and depth").
            Class::Expected { step, p } => Want::expected(step, p),
        };
        let in_draft = ledger.in_draft_window();
        let (mut read, mut staging_full) = (false, false);
        for Scheduled { layer, expert, act, leading, .. } in scheduled {
            let staged = (layer, expert);
            match act {
                Act::Keep | Act::Issue(Class::Due) => {}
                Act::Reprice(Class::Due) => self.hasten(staged),
                // A read ahead takes its want's price; the cover's and
                // carryover's, undated, are only hastened by a want of their
                // expert.
                Act::Reprice(class) => {
                    if let Some(Inbound { need: Need::Ahead { ticket, ahead }, urgency, .. }) = self.inbound.get(&staged) {
                        if !urgency.is_waited() && !ticket.cancelled() {
                            let want = match ahead {
                                Ahead::Interest { .. } => wanted(class),
                                Ahead::Cover | Ahead::Carryover { .. } => urgency.get().min(wanted(class)),
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
                            ledger.cancelled(layer, expert);
                        }
                    }
                }
                Act::Issue(class) => {
                    if staging_full {
                        continue;
                    }
                    let p = price.get();
                    if !in_draft && !p.binds() {
                        let budget = p.prefetch_byte_budget();
                        if budget > 0 && self.step_prefetch_bytes.saturating_add(core.layers[layer].expert_bytes()) > budget {
                            continue;
                        }
                    }
                    let expected = match class {
                        Class::Expected { step, p } => Some((step, p)),
                        Class::Due => None,
                    };
                    let interest = leading.expect("an expected read has a leading interest");
                    let target_step = expected.map_or(interest.from.point.sequence, |(step, _)| step);
                    let ahead = Ahead::Interest { source: interest.source, issued: interest.from, target_step };
                    match self.read_ahead(core, shared, staged, wanted(class), ahead, expected, reserve) {
                        Ok(bytes) => {
                            if !in_draft {
                                self.step_prefetch_bytes = self.step_prefetch_bytes.saturating_add(bytes);
                            }
                            ledger.issued(layer, expert, &interest, bytes, Instant::now());
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
        Self::spend(ledger, shared);
        read
    }

    fn prefetch_carryover(&mut self, core: &mut Core, ledger: &mut Ledger, shared: &Arc<Shared>) -> bool {
        if core.steps.prompt_running() {
            return false;
        }
        let reserve = self.reserve(core, false);
        let room = self.staging.free().saturating_sub(reserve.0);
        if room == 0 || core.layers.is_empty() {
            return false;
        }
        let price = Priced::new(&shared.telemetry);
        let now = Instant::now();
        let draft_idle = ledger.draft_window_remaining(now);
        if draft_idle > 0.0 {
            let p = price.get();
            let queued_secs: f64 = self
                .inbound
                .iter()
                .filter(|(_, inbound)| !inbound.read_done.load(Ordering::Acquire))
                .map(|(&(layer, _), _)| p.transfer_seconds(core.layers[layer].expert_bytes()))
                .sum();
            let mut budget = LinkIdleBudget::new(0.0, draft_idle, queued_secs);
            if budget.remaining() <= 0.0 {
                return false;
            }
            let candidates = {
                let reuse = core.directory.reuse_now();
                reuse.next_step_candidates(room, true, |layer, expert| Self::candidate(core, (layer, expert)))
            };
            let mut read = false;
            for cand in candidates {
                if self.staging.free() <= reserve.0 {
                    break;
                }
                let bytes = core.layers[cand.layer].expert_bytes();
                let t_bw = p.transfer_seconds(bytes);
                let p_eff = burst_hit_probability(cand.p_next, self.active_decode_rows);
                let net_benefit = p.carryover_net_benefit(p_eff, bytes, budget.remaining());
                if net_benefit <= 0.0 {
                    continue;
                }
                let want = Want::undated(p_eff);
                match self.read_ahead(core, shared, (cand.layer, cand.expert), want, Ahead::Carryover { target_step: 0 }, Some((0, net_benefit)), reserve) {
                    Ok(_) => {
                        read = true;
                        budget.consume(t_bw);
                        if p.binds() && t_bw > 0.0 && budget.remaining() <= 0.0 {
                            break;
                        }
                    }
                    Err(Short::Staging) => break,
                    Err(Short::Tier | Short::Refused) => {}
                }
            }
            return read;
        }

        // Continuous full-burst & cross-burst in-verify carryover: keep the
        // drive busy across the remaining non-drive pass window (compute
        // + RAM -> VRAM DMA) of burst `B` (`l+1..L-1`, priority 1) and into
        // early layers of burst `B+1` (`0..=l`, priority 2), paced to at most
        // `active_carryover_max_inflight` concurrent carryover reads on the
        // drive (refilled immediately on `reads_finished`) so Phase B demand
        // misses never suffer multi-expert head-of-line blocking behind
        // already-begun, un-cancellable carryover reads or bursty DDR5 landing
        // conversions.
        let (step, current_layer) = self.active_point(ledger);
        if step == 0 || shared.steps.ended(step) {
            return false;
        }
        let p = price.get();
        let layers = core.layers.len();
        let ref_bytes = core.layers[current_layer.min(layers - 1)].expert_bytes();
        let max_inflight = active_carryover_max_inflight(p, layers, ref_bytes);
        let mut inflight = self.inflight_carryover_reads();
        if inflight >= max_inflight {
            return false;
        }
        let binds = p.binds();
        let mut remaining_budget = if binds {
            u64::MAX
        } else {
            let b = p.prefetch_byte_budget().saturating_sub(self.step_prefetch_bytes);
            if b == 0 {
                return false;
            }
            b
        };
        let remaining_layers = layers.saturating_sub(current_layer.saturating_add(1)).max(1);
        let pass_remaining = (remaining_layers as f64 / layers as f64) * p.non_drive_pass_seconds();
        let queued_secs: f64 = self
            .inbound
            .iter()
            .filter(|(_, inbound)| !inbound.read_done.load(Ordering::Acquire))
            .map(|(&(layer, _), _)| p.transfer_seconds(core.layers[layer].expert_bytes()))
            .sum();
        let idle_window = if binds { pass_remaining } else { p.slack_seconds().max(pass_remaining) };
        let mut budget = LinkIdleBudget::new(idle_window, 0.0, queued_secs);
        if binds && budget.remaining() <= 0.0 {
            return false;
        }
        let mut candidates = {
            let reuse = core.directory.reuse_now();
            reuse.next_step_candidates(room, false, |layer, expert| Self::candidate(core, (layer, expert)))
        };
        // Earliest-deadline-first across the burst boundary: remaining layers
        // of burst B (`layer > current_layer`) first, then early layers of
        // burst B+1 (`layer <= current_layer`), each ascending by layer,
        // tie-broken by calibrated `p_next` descending.
        candidates.sort_unstable_by(|a, b| {
            let a_next_burst = a.layer <= current_layer;
            let b_next_burst = b.layer <= current_layer;
            a_next_burst
                .cmp(&b_next_burst)
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| b.p_next.total_cmp(&a.p_next))
        });
        let mut read = false;
        for cand in candidates {
            if self.staging.free() <= reserve.0 {
                break;
            }
            let bytes = core.layers[cand.layer].expert_bytes();
            if !binds && bytes > remaining_budget {
                continue;
            }
            let t_bw = p.transfer_seconds(bytes);
            let free_slack = active_carryover_free_slack(p, budget.remaining());
            let p_eff = active_carryover_hit_probability(p, cand.p_next, self.active_decode_rows);
            let net_benefit = p.carryover_net_benefit(p_eff, bytes, free_slack);
            if net_benefit <= 0.0 {
                continue;
            }
            let target_step = if cand.layer > current_layer { step } else { step.saturating_add(1) };
            let want = Want::undated(p_eff);
            match self.read_ahead(core, shared, (cand.layer, cand.expert), want, Ahead::Carryover { target_step }, Some((target_step, net_benefit)), reserve) {
                Ok(bytes) => {
                    self.step_prefetch_bytes = self.step_prefetch_bytes.saturating_add(bytes);
                    remaining_budget = remaining_budget.saturating_sub(bytes);
                    budget.consume(t_bw);
                    inflight += 1;
                    read = true;
                    if inflight >= max_inflight || (!binds && remaining_budget == 0) || (binds && t_bw > 0.0 && budget.remaining() <= 0.0) {
                        break;
                    }
                }
                Err(Short::Staging) => break,
                Err(Short::Tier | Short::Refused) => {}
            }
        }
        read
    }

    /// Number of active (uncompleted, uncancelled, non-waited) carryover reads
    /// currently in flight on the drive.
    fn inflight_carryover_reads(&self) -> usize {
        self.inbound
            .values()
            .filter(|inbound| {
                !inbound.read_done.load(Ordering::Acquire)
                    && !inbound.urgency.is_waited()
                    && matches!(
                        inbound.need,
                        Need::Ahead { ref ticket, ahead: Ahead::Carryover { .. } }
                        if !ticket.cancelled()
                    )
            })
            .count()
    }

    /// Hand telemetry the reads ahead's fates the ledger learned
    /// (`Ledger::take_spent`).
    pub(super) fn spend(ledger: &mut Ledger, shared: &Shared) {
        let spent = ledger.take_spent();
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
        let Some(record) = self.staging.take_beyond_within(staging, core.table.staging_limit(target)) else {
            return Err(Short::Staging);
        };
        // Beyond what its group's reserve (`HostRecords::reserve`) holds
        // past the reads ahead's staging, whose landings it counts already;
        // none for the cover.
        let keep = ahead_staging.map_or(0, |ahead| core.directory.host(core.layers[target].group).reserve().saturating_sub(ahead));
        let claim = match (ahead, expected) {
            (Ahead::Interest { .. }, Some((step, p))) => core.directory.claim_ahead(target, expert, keep, ahead.taking(), step, p),
            (Ahead::Carryover { .. }, Some((_, net_benefit))) => core.directory.claim_carryover(target, expert, keep, ahead.taking(), net_benefit),
            _ => core.directory.claim_landing(target, expert, keep, ahead.taking(), (|_, _| (), |_, _| false)),
        };
        let Claim::Taken(grant) = claim else {
            self.staging.release(record);
            return Err(Short::Tier);
        };
        let tier = grant.record();
        let reads = core.table.reads(target, expert, record);
        let bytes = reads.iter().map(crate::experts::source::Read::bytes).sum::<u64>();
        match post(
            &core.table,
            reads,
            core.table.tensor_count(target),
            want,
            crate::experts::source::ReadAlignment::AllowAlignedLead,
        ) {
            Ok(Posting { ticket, join }) => {
                for request in ticket.requests() {
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("ahead"), request, (target as i64, expert as i64, i64::from(want.deadline() == seismic_io::Deadline::None)));
                }
                let landing = core.landing_with_leads(grant, record, ticket.tensor_leads());
                let tensor_count = landing.tensors.len();
                let ended = Ended::new(shared.events());
                let urgency = Urgency::ahead(want);
                let read_done = Arc::new(AtomicBool::new(false));
                let (end, serving, urgent, done, finished) = (ended.clone(), shared.clone(), urgency.clone(), read_done.clone(), self.reads_finished.clone());
                shared.telemetry.engagement().read_started();
                let pipelined = PipelinedLanding::new(
                    landing.grant,
                    tensor_count,
                    urgent,
                    serving.landings.clone(),
                    end,
                    move || {
                        seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("ahead_landed"), 0, (target as i64, expert as i64, 0));
                    },
                );
                let tensors: Arc<Mutex<Vec<Option<crate::experts::landing::LandingTensor>>>> =
                    Arc::new(Mutex::new(landing.tensors.into_iter().map(Some).collect()));
                let serving_t = serving.clone();
                join.claimed_per_tensor(move |tensor_idx, outcome| {
                    unpanicked(&outcome);
                    let tensor = tensors
                        .lock()
                        .expect("pipelined ahead landing tensors")
                        .get_mut(tensor_idx)
                        .and_then(Option::take);
                    let Some(tensor) = tensor else {
                        return;
                    };
                    if matches!(outcome, Outcome::Read) {
                        // SAFETY: as a demand landing's on the host (`Self::demand`).
                        unsafe { pipelined.submit_tensor(tensor, &serving_t.compute) };
                    } else {
                        pipelined.fail_tensor();
                    }
                });
                // A read ahead's end: notify events and update engagement.
                join.claimed(move |outcome, failed| {
                    unpanicked(&outcome);
                    done.store(true, Ordering::Release);
                    finished.store(true, Ordering::Release);
                    serving.telemetry.engagement().read_finished();
                    let read = failed.is_empty() && matches!(outcome, Outcome::Read);
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("ahead_read"), 0, (target as i64, expert as i64, read as i64));
                    serving.events().notify();
                });
                self.arrive(staged, Inbound { record: Some(record), tier, bytes, need: Need::Ahead { ticket, ahead }, ended, urgency, read_done, counted_hit: false });
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

/// A front ring read whole ([`Prefetch::read_rings`]): the layer whose ring it
/// is, and the step's low word, front rows' guess kinds and entries it holds.
pub(super) struct RingRead {
    layer: usize,
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

/// Free drive slack available to an active-step carryover read (`draft_idle == 0.0`).
///
/// When the NVMe drive itself is the binding resource (`price.drive_binds()`),
/// demand misses already saturate the drive across the pass, so there is zero
/// free drive slack: every extra speculative byte extends the critical path and
/// must pay for its own drive transfer cost against `s_demand`. When the drive
/// is not binding (e.g. DDR-bound or compute-bound), the remaining non-drive
/// pass overlap window `budget_remaining` is free drive slack (while
/// `carryover_net_benefit` still prices host DDR contention when
/// `host_memory_binds()`).
fn active_carryover_free_slack(price: &super::telemetry::ReadPrice, budget_remaining: f64) -> f64 {
    if price.drive_binds() {
        0.0
    } else {
        budget_remaining
    }
}

/// Maximum concurrent in-flight `Ahead::Carryover` NVMe reads during an active
/// decode/verify step (`draft_idle == 0.0`).
///
/// Derived from the mean layer arrival interval `(non_drive_pass_seconds / layers)`
/// divided by one expert's drive transfer time `t_bw = transfer_seconds(bytes)`,
/// floored at `1`: keeps the NVMe drive continuously pipelined (refilled on
/// `reads_finished`) while bounding worst-case Phase B demand head-of-line
/// blocking behind already-begun, un-cancellable carryover reads to at most
/// one layer interval (or one expert when `t_bw >= layer_interval`).
fn active_carryover_max_inflight(price: &super::telemetry::ReadPrice, layers: usize, bytes: u64) -> usize {
    if layers == 0 || bytes == 0 {
        return 1;
    }
    let t_bw = price.transfer_seconds(bytes);
    let pass_secs = price.non_drive_pass_seconds();
    if !t_bw.is_finite() || t_bw <= 0.0 || !pass_secs.is_finite() || pass_secs <= 0.0 {
        return 1;
    }
    let layer_interval = pass_secs / layers as f64;
    let count = (layer_interval / t_bw).floor();
    if count.is_finite() && count >= 1.0 {
        (count.min(usize::MAX as f64) as usize).max(1)
    } else {
        1
    }
}

/// Probability that an expert with single-token reuse probability `p_single`
/// is routed by at least one row in an `M`-row decode/verify burst:
/// `1 - (1 - p_single)^M`.
fn burst_hit_probability(p_single: f64, rows: usize) -> f64 {
    let p = p_single.clamp(0.0, 1.0);
    if rows <= 1 || p <= 0.0 || p >= 1.0 {
        return p;
    }
    (1.0 - (1.0 - p).powi(rows.min(i32::MAX as usize) as i32)).clamp(0.0, 1.0)
}

/// Effective hit probability used to price active-step carryover (`draft_idle == 0.0`).
///
/// When `!price.drive_binds()`, the drive has idle headroom and multi-row verify
/// bursts (`M > 1`) hit an expert if any row routes it (`1 - (1 - p_1)^M`).
/// When `price.drive_binds()`, keep the conservative single-token `p_1` so
/// correlated multi-row routing never over-admits speculative SSD reads on a
/// saturated drive.
fn active_carryover_hit_probability(price: &super::telemetry::ReadPrice, p_single: f64, rows: usize) -> f64 {
    if price.drive_binds() {
        p_single.clamp(0.0, 1.0)
    } else {
        burst_hit_probability(p_single, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::{Ahead, DriveIdleBudget, Inbound, Need, Prefetch};
    use crate::experts::compute::Urgency;
    use crate::experts::events::Events;
    use crate::experts::horizon::Time;
    use crate::experts::interest::{Ledger, Stands};
    use crate::experts::landing::Ended;
    use crate::experts::source::Ticket;
    use crate::experts::watcher::passes::{Pass, Passes};
    use crate::experts::watcher::telemetry::ReadPrice;
    use magnitude_cost_model::ReadLine;
    use seismic_io::Want;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn a_read_holding_staging_outside_the_landings_is_outstanding() {
        let prefetch = Prefetch::new(2, 1);
        assert!(!prefetch.outstanding(), "nothing read");
        let posted = prefetch.staging.take_up_to(1);
        assert!(prefetch.outstanding(), "a posted read's staging record is taken");
        posted.into_iter().for_each(|record| prefetch.staging.release(record));
        assert!(!prefetch.outstanding(), "its record freed as it ended");
    }

    #[test]
    fn shared_drive_idle_budget_counts_demand_and_cross_layer_reads() {
        let layers = 35;
        let bytes = 1_000_000_u64;
        let line = ReadLine::new(0.3e-3, 1e-9).expect("a line");
        let mut passes = Passes::new(Time::new(&[vec![1]], 8));
        passes.timed(Pass { rows: 1, depth: 0, seconds: 0.1, parked: 0.065, misses: 68., drive: 0.068, ..Pass::default() });
        let price = ReadPrice::priced(passes.read_admission(Some(line)));
        assert!((price.transfer_seconds(bytes) - 1e-3).abs() < 1e-12, "DriveIdleBudget uses pipelined t_bw, not lambda + t_bw");
        assert!(!price.admits_with_idle(0.30, bytes, 0.0) && !price.admits_with_idle(0.52, bytes, 0.0) && price.admits_with_idle(1.0, bytes, 0.0));

        let mut prefetch = Prefetch::new(4, 2);
        let events = Arc::new(Events::default());
        let seen = HashSet::from([(1, 0)]);

        let idle = DriveIdleBudget::new(layers, 1, prefetch.drive_queued_bytes(|_| bytes, &seen), 0.0);
        assert!((idle.get(&price) - 1e-3).abs() < 1e-12);
        assert!(price.admits_with_idle(0.30, bytes, idle.get(&price)));
        idle.consume(&price, price.transfer_seconds(bytes));
        assert_eq!(idle.get(&price), 0.0);
        assert!(!price.admits_with_idle(0.30, bytes, idle.get(&price)));

        // Across a multi-layer remaining verify window (e.g. 20 layers left),
        // DriveIdleBudget opens `remaining_layers * c_layer`, so a single
        // demand read on layer l-1 does not stall lookahead for later layers.
        let due_done = Arc::new(AtomicBool::new(false));
        prefetch.inbound.insert(
            (0, 0),
            Inbound {
                record: Some(0),
                tier: 0,
                bytes,
                need: Need::Due { ticket: Ticket::empty(), read: 0, device: None },
                ended: Ended::new(&events),
                urgency: Urgency::waited(),
                read_done: due_done.clone(),
                counted_hit: false,
            },
        );
        let queued = prefetch.drive_queued_bytes(|_| bytes, &seen);
        assert_eq!(queued, vec![bytes]);
        let idle = DriveIdleBudget::new(layers, 1, queued.clone(), 0.0);
        assert_eq!(idle.get(&price), 0.0, "single-layer window saturated by demand read on layer l-1");
        assert!(!price.admits_with_idle(0.30, bytes, idle.get(&price)), "low-p hint rejected when single-layer window is full");
        assert!(!price.admits_with_idle(0.52, bytes, idle.get(&price)), "50% hint rejected on saturated drive with no park overhead");
        assert!(price.admits_with_idle(1.0, bytes, idle.get(&price)), "certain read still admitted");

        let full_burst = DriveIdleBudget::new(layers, 20, queued, 0.0);
        assert!((full_burst.get(&price) - 19e-3).abs() < 1e-12, "remaining verify compute window stays open behind 1 demand read");
        assert!(price.admits_with_idle(0.30, bytes, full_burst.get(&price)));

        // Once the demand read completes on the drive (even while its
        // DeviceLanding stays in `inbound` awaiting the next layer's park),
        // the shared drive is free and `DriveIdleBudget` opens `c_layer`.
        due_done.store(true, Ordering::Release);
        assert!(prefetch.drive_queued_bytes(|_| bytes, &seen).is_empty(), "finished demand read no longer occupies the drive");
        let idle = DriveIdleBudget::new(layers, 1, prefetch.drive_queued_bytes(|_| bytes, &seen), 0.0);
        assert!((idle.get(&price) - 1e-3).abs() < 1e-12, "full c_layer idle budget opens once demand read finishes");
        assert!(price.admits_with_idle(0.30, bytes, idle.get(&price)));

        prefetch.inbound.clear();
        prefetch.inbound.insert(
            (1, 0),
            Inbound {
                record: Some(0),
                tier: 0,
                bytes,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Cover },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.)),
                read_done: Arc::new(AtomicBool::new(false)),
                counted_hit: false,
            },
        );
        assert!(prefetch.drive_queued_bytes(|_| bytes, &seen).is_empty(), "in-pass unbegun ahead read is not pre-counted");
        prefetch.inbound.insert(
            (0, 1),
            Inbound {
                record: Some(1),
                tier: 1,
                bytes,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Cover },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.)),
                read_done: Arc::new(AtomicBool::new(false)),
                counted_hit: false,
            },
        );
        let queued = prefetch.drive_queued_bytes(|_| bytes, &seen);
        assert_eq!(queued, vec![bytes], "out-of-pass ahead read on layer 0 counts against shared drive");
        assert_eq!(DriveIdleBudget::new(layers, 1, queued, 0.0).get(&price), 0.0);
    }

    #[test]
    fn draft_window_opens_idle_budget_and_binds_unbegun_reads_on_began() {
        use crate::experts::interest::{Launched, Told};
        use std::time::Instant;

        let bytes = 3_090_000_u64;
        let line = ReadLine::new(0.14e-3, 1.0 / 4.6e9).expect("a line");
        let mut passes = Passes::new(Time::new(&[vec![1]], 8));
        passes.timed(Pass { rows: 1, depth: 0, seconds: 0.135, parked: 0.100, misses: 120., drive: 0.100, ..Pass::default() });
        let price = ReadPrice::priced(passes.read_admission(Some(line)));
        assert!(price.binds());
        assert!(!price.admits_with_idle(0.0, bytes, 0.0));

        let time = Time::new(&[vec![4, 4]], 32);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0, 0], 4, time, stands);
        let mut prefetch = Prefetch::new(8, 2);
        let events = Arc::new(Events::default());

        ledger.tell(Told::DraftWindow { estimated_seconds: 0.019 });
        assert!(ledger.in_draft_window());
        let rem = ledger.draft_window_remaining(Instant::now());
        assert!(rem > 0.010 && rem <= 0.019, "remaining draft window {rem}");

        let idle = DriveIdleBudget::new(48, 1, Vec::new(), rem);
        assert!(price.admits_with_idle(0.0, bytes, idle.get(&price)));

        // Insert an unbegun draft-window carryover read (`target_step == 0`).
        // `on_began(2)` must NOT cancel it (seismic_io serves `Want::now()`
        // strictly ahead of `Want::undated`), and must bind `target_step = 2`
        // so `Prefetch::pass` can cancel it once `(2, layer)` passes.
        prefetch.inbound.insert(
            (1, 0),
            Inbound {
                record: Some(0),
                tier: 0,
                bytes,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Carryover { target_step: 0 } },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.25)),
                read_done: Arc::new(AtomicBool::new(false)),
                counted_hit: false,
            },
        );
        prefetch.step_prefetch_bytes = 10_000_000;
        ledger.tell(Told::Began { step: 2, launch: Launched::Decode { rows: Box::new([]), padding: 0 } });
        prefetch.on_began(2);
        assert!(!ledger.in_draft_window());
        assert_eq!(ledger.draft_window_remaining(Instant::now()), 0.0);
        assert_eq!(prefetch.step_prefetch_bytes, 0);
        let inbound = prefetch.inbound.get(&(1, 0)).expect("draft carryover stays queued across on_began");
        assert!(!inbound.ticket().cancelled(), "on_began must not cancel unbegun draft carryover reads");
        assert!(
            matches!(inbound.need, Need::Ahead { ahead: Ahead::Carryover { target_step: 2 }, .. }),
            "on_began binds untargeted draft carryover to the new step"
        );
        let post_idle = DriveIdleBudget::new(48, 1, vec![bytes], 0.0);
        assert!(!price.admits_with_idle(0.0, bytes, post_idle.get(&price)));
    }

    /// B-3: Slack-aware `ReadPrice` exposes per-layer idle slack and
    /// `B_budget = (C - U * t_bw) * P_peak`, admitting reads within slack and
    /// capping speculative bytes when compute exceeds pipelined drive bandwidth.
    #[test]
    fn slack_aware_read_price_computes_layer_idle_and_byte_budget() {
        let bytes = 3_090_000_u64;
        let line = ReadLine::new(0.14e-3, 1.0 / 4.6e9).expect("a line");
        let mut passes = Passes::new(Time::new(&[vec![1]], 8));
        // C = 0.075 s, U * t_bw = 0.035 s -> slack = 0.040 s, B_budget = 184 MB.
        // Note: parked = 0.060 s (exposed stall), seconds = 0.135 s -> C = 0.135 - 0.060 = 0.075 s,
        // while terms.drive = 0.035 s (pipelined U * t_bw, NOT parked).
        passes.timed(Pass {
            rows: 4,
            depth: 3,
            verifies: true,
            seconds: 0.135,
            parked: 0.060,
            misses: 52.,
            drive: 0.035,
            ..Pass::default()
        });
        let price = ReadPrice::priced(passes.read_admission(Some(line)));
        assert!(!price.binds(), "pipelined drive (35 ms) < C (75 ms), even though exposed parked was 60 ms");
        assert!((price.layer_idle(48) - 0.040 / 48.0).abs() < 1e-12);
        let budget = price.prefetch_byte_budget();
        assert!((budget as f64 - 184_000_000.0).abs() <= 1.0, "expected ~184 MB budget, got {budget}");
        assert!(price.admits_for_layers(0.10, bytes, 48));
    }

    /// When `ContentionDomain::HostMemory` binds on a discrete host (e.g.
    /// 2-ch DDR5 + PCIe Gen5 + NVMe), `DriveIdleBudget` keeps the non-drive
    /// pass overlap window (compute + RAM -> VRAM DMA) open instead of
    /// zeroing it, while `PathAdmission` prices actual DDR5 contention
    /// (`bytes / host_bw`) against exposed NVMe demand stall (`s_demand`)
    /// for both hints and active-step carryover.
    #[test]
    fn drive_prefetch_overlaps_non_drive_pass_and_prices_ddr_contention_when_host_memory_binds() {
        use crate::experts::compute::Urgency;
        use magnitude_cost_model::{PathAdmission, Terms};

        let bytes = 9_000_000u64;
        let drive_bw = 6.0e9;
        let host_bw = 80.0e9;
        let line = ReadLine::new(0.15e-3, 1.0 / drive_bw).expect("a line");

        // HostMemory binds (memory 60 ms >= compute 50 ms, link 35 ms, drive 25 ms).
        let mem_bound = ReadPrice::priced(
            PathAdmission::drive(
                Terms { compute: 0.050, drive: 0.025, link: 0.035, cpu: 0.010, memory: 0.060 },
                Some(line),
                Some(1.5e-3),
                None,
            )
            .with_host_memory_secs_per_byte(Some(1.0 / host_bw)),
        );
        assert!(mem_bound.host_memory_binds());
        assert!((mem_bound.non_drive_pass_seconds() - 0.060).abs() < 1e-12);

        // DriveIdleBudget spans the remaining non-drive pass window (24/48 * 60 ms = 30 ms).
        let idle = DriveIdleBudget::new(48, 24, vec![], 0.0);
        assert!((idle.get(&mem_bound) - 0.030).abs() < 1e-12);

        // Break-even during NVMe overlap equals measured `t_mem / s_demand = drive_bw / host_bw = 0.075`.
        assert!(mem_bound.admits_with_idle(0.10, bytes, idle.get(&mem_bound)));
        assert!(!mem_bound.admits_with_idle(0.05, bytes, idle.get(&mem_bound)));

        // Active-step carryover with `budget.remaining()` admits SSD-only candidates
        // whose `p_next` clears DDR contention (`> ~0.070`), while rejecting noise.
        assert!(mem_bound.carryover_net_benefit(0.20, bytes, idle.get(&mem_bound)) > 0.0);
        assert!(mem_bound.carryover_net_benefit(0.05, bytes, idle.get(&mem_bound)) < 0.0);

        // Urgency::hasten() records a hasten timestamp so prior hint wait is not
        // mis-billed to demand.queued, and Urgency::streamed() is non-waited.
        let streamed = Urgency::streamed();
        assert!(!streamed.is_waited());
        assert_eq!(streamed.waited_since(), (false, None));

        let hint = Urgency::ahead(Want::expected(1, 0.5));
        assert_eq!(hint.waited_since(), (false, None));
        let before = std::time::Instant::now();
        hint.hasten();
        let (waited, hastened_at) = hint.waited_since();
        assert!(waited);
        assert!(hastened_at.is_some_and(|t| t >= before));
    }

    /// `Landings` emits a single calibration sample once every read AND every
    /// landing have ended, regardless of whether `ended` arrives before `read`
    /// (single-expert or multi-expert pipelined per-tensor landing).
    #[test]
    fn landings_emits_once_from_last_terminal_observation() {
        use super::Landings;
        use std::time::{Duration, Instant};

        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);

        // Single-expert: `ended` fires before aggregate `read` callback.
        let single = Landings::new(0, 1);
        assert_eq!(single.record_ended(ms(10)), None, "must wait for read callback");
        let secs = single.record_read(ms(12)).expect("must emit once when read arrives last");
        assert_eq!(secs, 0.0, "ended before read saturates to 0.0s");

        // Multi-expert: expert 0 finishes both, then expert 1's `ended` fires
        // before expert 1's `read` callback. Must NOT emit prematurely with
        // expert 0's read timestamp.
        let multi = Landings::new(0, 2);
        assert_eq!(multi.record_read(ms(10)), None);
        assert_eq!(multi.record_ended(ms(15)), None);
        assert_eq!(multi.record_ended(ms(25)), None, "must wait for expert 1's read");
        let secs = multi.record_read(ms(20)).expect("must emit once when expert 1's read arrives");
        assert!((secs - 0.005).abs() < 1e-9, "expected last_landed (25ms) - last_read (20ms) = 5ms, got {secs}");

        // Normal order: both reads finish before last landing ends.
        let normal = Landings::new(0, 2);
        assert_eq!(normal.record_read(ms(10)), None);
        assert_eq!(normal.record_read(ms(14)), None);
        assert_eq!(normal.record_ended(ms(18)), None);
        let secs = normal.record_ended(ms(22)).expect("must emit on last landing");
        assert!((secs - 0.008).abs() < 1e-9, "expected 22ms - 14ms = 8ms, got {secs}");
    }

    /// Active-step carryover free slack (`draft_idle == 0.0`) depends on
    /// whether the NVMe drive itself is the binding resource:
    /// - DDR-bound (`!drive_binds() && host_memory_binds()`): the drive has
    ///   idle overlap headroom `budget.remaining()` during non-drive pass time,
    ///   while `carryover_net_benefit` still prices DDR5 contention (`bytes / host_bw`).
    /// - SSD-bound (`drive_binds()`, unified or discrete): the drive itself is
    ///   saturated by demand misses over the pass, so active-step carryover gets
    ///   zero free drive slack (`0.0`) even when `budget.remaining() > 0`, and
    ///   unprofitable speculative reads are rejected instead of competing with
    ///   demand reads on the saturated SSD.
    #[test]
    fn active_carryover_uses_slack_when_ddr_bound_and_zero_slack_when_drive_binds() {
        use super::{active_carryover_free_slack, active_carryover_hit_probability, burst_hit_probability};
        use magnitude_cost_model::{LinkIdleBudget, PathAdmission, Terms};

        let bytes = 9_000_000u64;
        let drive_bw = 6.0e9;
        let host_bw = 80.0e9;
        let line = ReadLine::new(0.15e-3, 1.0 / drive_bw).expect("a line");
        let s_demand = 1.5e-3; // matches t_bw = 9 MB / 6 GB/s = 1.5 ms

        // Multi-row burst hit probability: 1 - (1 - p_1)^M.
        assert!((burst_hit_probability(0.20, 1) - 0.20).abs() < 1e-12);
        assert!((burst_hit_probability(0.20, 4) - (1.0 - 0.8_f64.powi(4))).abs() < 1e-12);

        // (a) DDR-bound discrete host: memory (60 ms) binds, drive (25 ms) does not.
        let ddr_bound = ReadPrice::priced(
            PathAdmission::drive(
                Terms { compute: 0.050, drive: 0.025, link: 0.035, cpu: 0.010, memory: 0.060 },
                Some(line),
                Some(s_demand),
                None,
            )
            .with_host_memory_secs_per_byte(Some(1.0 / host_bw)),
        );
        assert!(ddr_bound.binds());
        assert!(ddr_bound.host_memory_binds());
        assert!(!ddr_bound.drive_binds());

        let ddr_budget = LinkIdleBudget::new(0.030, 0.0, 0.003);
        let ddr_slack = active_carryover_free_slack(&ddr_bound, ddr_budget.remaining());
        assert!((ddr_slack - 0.027).abs() < 1e-12, "DDR-bound host uses remaining non-drive pass slack");
        // Single-token p = 0.20 clears DDR contention (drive_bw / host_bw = 0.075); noise p = 0.05 does not.
        assert!(ddr_bound.carryover_net_benefit(active_carryover_hit_probability(&ddr_bound, 0.20, 1), bytes, ddr_slack) > 0.0);
        assert!(ddr_bound.carryover_net_benefit(active_carryover_hit_probability(&ddr_bound, 0.05, 1), bytes, ddr_slack) < 0.0);
        // Multi-row verify (M = 4): p_1 = 0.15 lifts to 1 - 0.85^4 = 0.478 for burst carryover valuation.
        let p_ddr_m4 = active_carryover_hit_probability(&ddr_bound, 0.15, 4);
        assert!((p_ddr_m4 - (1.0 - 0.85_f64.powi(4))).abs() < 1e-12);
        assert!(ddr_bound.carryover_net_benefit(p_ddr_m4, bytes, ddr_slack) > ddr_bound.carryover_net_benefit(0.15, bytes, ddr_slack));

        // (b) PCIe Link-bound host with Link > Drive > Compute (e.g. 5090-128g DS4):
        // drive does NOT bind, slack_seconds = (160 - 70) = 90 ms > 0, prefetch_byte_budget > 0,
        // and multi-row burst probability lifts carryover valuation by ~3-4x.
        let pcie_bound = ReadPrice::priced(PathAdmission::drive(
            Terms { compute: 0.030, drive: 0.070, link: 0.160, cpu: 0.010, memory: 0.050 },
            Some(line),
            Some(s_demand),
            None,
        ));
        assert!(!pcie_bound.binds());
        assert!(!pcie_bound.drive_binds());
        assert!((pcie_bound.slack_seconds() - 0.090).abs() < 1e-12);
        assert!(pcie_bound.prefetch_byte_budget() > bytes);
        let pcie_slack = active_carryover_free_slack(&pcie_bound, ddr_budget.remaining());
        assert!((pcie_slack - 0.027).abs() < 1e-12);
        let p_pcie_m5 = active_carryover_hit_probability(&pcie_bound, 0.15, 5);
        assert!((p_pcie_m5 - (1.0 - 0.85_f64.powi(5))).abs() < 1e-12);
        assert!((pcie_bound.carryover_net_benefit(p_pcie_m5, bytes, pcie_slack) - p_pcie_m5).abs() < 1e-12);

        // (c1) SSD-bound unified memory host (e.g. Apple M5 Pro): drive (150 ms) > compute (50 ms).
        let ssd_unified = ReadPrice::priced(PathAdmission::drive(
            Terms { compute: 0.050, drive: 0.150, link: 0.0, cpu: 0.0, memory: 0.0 },
            Some(line),
            Some(s_demand),
            None,
        ));
        assert!(ssd_unified.binds());
        assert!(ssd_unified.drive_binds());
        assert!(!ssd_unified.host_memory_binds());
        assert_eq!(ssd_unified.slack_seconds(), 0.0);

        let ssd_budget = LinkIdleBudget::new(0.030, 0.0, 0.003);
        assert!(ssd_budget.remaining() > 0.0);
        let ssd_slack = active_carryover_free_slack(&ssd_unified, ssd_budget.remaining());
        assert_eq!(ssd_slack, 0.0, "SSD-bound host gets zero free drive slack during active pass");
        // Even with M = 5 verify rows, SSD-bound keeps conservative p_1 and rejects unprofitable reads.
        assert_eq!(active_carryover_hit_probability(&ssd_unified, 0.20, 5), 0.20);
        assert!(ssd_unified.carryover_net_benefit(active_carryover_hit_probability(&ssd_unified, 0.20, 5), bytes, ssd_slack) < 0.0);
        assert!(ssd_unified.carryover_net_benefit(active_carryover_hit_probability(&ssd_unified, 0.45, 5), bytes, ssd_slack) < 0.0);
        // Only genuinely profitable carryover (p > t_bw / (t_bw + s_demand) = 0.50) pays.
        assert!(ssd_unified.carryover_net_benefit(active_carryover_hit_probability(&ssd_unified, 0.60, 5), bytes, ssd_slack) > 0.0);

        // (c2) SSD-bound discrete host (e.g. RTX 5090-64g): drive (150 ms) > memory (50 ms).
        let ssd_discrete = ReadPrice::priced(
            PathAdmission::drive(
                Terms { compute: 0.030, drive: 0.150, link: 0.040, cpu: 0.010, memory: 0.050 },
                Some(line),
                Some(s_demand),
                None,
            )
            .with_host_memory_secs_per_byte(Some(1.0 / host_bw)),
        );
        assert!(ssd_discrete.binds());
        assert!(ssd_discrete.drive_binds());
        assert!(!ssd_discrete.host_memory_binds());
        assert_eq!(ssd_discrete.slack_seconds(), 0.0);
        let ssd_discrete_slack = active_carryover_free_slack(&ssd_discrete, ssd_budget.remaining());
        assert_eq!(ssd_discrete_slack, 0.0, "SSD-bound discrete host also gets zero free drive slack");
        assert!(ssd_discrete.carryover_net_benefit(active_carryover_hit_probability(&ssd_discrete, 0.20, 5), bytes, ssd_discrete_slack) < 0.0);
        assert!(ssd_discrete.carryover_net_benefit(active_carryover_hit_probability(&ssd_discrete, 0.60, 5), bytes, ssd_discrete_slack) > 0.0);

        // (d) Compute-bound host: neither drive nor host memory binds -> uses budget slack.
        let compute_bound = ReadPrice::priced(PathAdmission::drive(
            Terms { compute: 0.075, drive: 0.035, link: 0.0, cpu: 0.0, memory: 0.0 },
            Some(line),
            Some(s_demand),
            None,
        ));
        assert!(!compute_bound.binds());
        assert!(!compute_bound.drive_binds());
        let compute_slack = active_carryover_free_slack(&compute_bound, ddr_budget.remaining());
        assert!((compute_slack - 0.027).abs() < 1e-12);
        assert!(compute_bound.carryover_net_benefit(active_carryover_hit_probability(&compute_bound, 0.20, 4), bytes, compute_slack) > 0.0);
    }

    /// Active-step carryover paces concurrent in-flight NVMe reads to
    /// `max(1, floor((non_drive_pass_seconds / layers) / t_bw))` so Phase B
    /// demand misses never queue behind a deep backlog of already-begun,
    /// un-cancellable carryover reads, while `inflight_carryover_reads` counts
    /// only active, uncompleted, uncancelled, non-waited `Ahead::Carryover`
    /// entries.
    #[test]
    fn active_carryover_paces_inflight_reads_to_layer_interval() {
        use super::active_carryover_max_inflight;
        use magnitude_cost_model::{PathAdmission, Terms};

        let bytes = 9_000_000u64;
        let line = ReadLine::new(0.15e-3, 1.0 / 6.0e9).expect("a line"); // t_bw = 1.5 ms

        // 5090-128g DS4 (link = 160 ms, 58 layers -> layer_interval = 2.76 ms, t_bw = 1.5 ms):
        // floor(2.76 / 1.5) = 1 -> at most 1 carryover read in flight at a time.
        let pcie_bound = ReadPrice::priced(PathAdmission::drive(
            Terms { compute: 0.030, drive: 0.070, link: 0.160, cpu: 0.010, memory: 0.050 },
            Some(line),
            Some(1.5e-3),
            None,
        ));
        assert_eq!(active_carryover_max_inflight(&pcie_bound, 58, bytes), 1);

        // Slower pass (300 ms across 48 layers -> layer_interval = 6.25 ms, t_bw = 1.5 ms):
        // floor(6.25 / 1.5) = 4 -> 4 * 1.5 ms = 6.0 ms <= 1 layer interval.
        let slow_pass = ReadPrice::priced(PathAdmission::drive(
            Terms { compute: 0.030, drive: 0.070, link: 0.300, cpu: 0.010, memory: 0.050 },
            Some(line),
            Some(1.5e-3),
            None,
        ));
        assert_eq!(active_carryover_max_inflight(&slow_pass, 48, bytes), 4);

        // SSD-bound (pass = 50 ms, 58 layers -> layer_interval = 0.86 ms < t_bw): floors at 1.
        let ssd_bound = ReadPrice::priced(PathAdmission::drive(
            Terms { compute: 0.050, drive: 0.150, link: 0.0, cpu: 0.0, memory: 0.0 },
            Some(line),
            Some(1.5e-3),
            None,
        ));
        assert_eq!(active_carryover_max_inflight(&ssd_bound, 58, bytes), 1);

        // Unmeasured / degenerate inputs safely floor at 1.
        assert_eq!(active_carryover_max_inflight(&ReadPrice::unpriced(), 58, bytes), 1);
        assert_eq!(active_carryover_max_inflight(&pcie_bound, 0, bytes), 1);
        assert_eq!(active_carryover_max_inflight(&pcie_bound, 58, 0), 1);

        // `inflight_carryover_reads` counts only active, uncompleted, uncancelled,
        // non-waited `Ahead::Carryover` reads.
        let mut prefetch = Prefetch::new(8, 4);
        let events = Arc::new(Events::default());
        assert_eq!(prefetch.inflight_carryover_reads(), 0);

        let active_done = Arc::new(AtomicBool::new(false));
        prefetch.inbound.insert(
            (0, 0),
            Inbound {
                record: Some(0),
                tier: 0,
                bytes,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Carryover { target_step: 1 } },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.4)),
                read_done: active_done.clone(),
                counted_hit: false,
            },
        );
        assert_eq!(prefetch.inflight_carryover_reads(), 1);

        // Demand read, Cover read, waited carryover, cancelled carryover do NOT count.
        prefetch.inbound.insert(
            (1, 0),
            Inbound {
                record: Some(1),
                tier: 1,
                bytes,
                need: Need::Due { ticket: Ticket::empty(), read: 1, device: None },
                ended: Ended::new(&events),
                urgency: Urgency::waited(),
                read_done: Arc::new(AtomicBool::new(false)),
                counted_hit: false,
            },
        );
        prefetch.inbound.insert(
            (2, 0),
            Inbound {
                record: Some(2),
                tier: 2,
                bytes,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Cover },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.0)),
                read_done: Arc::new(AtomicBool::new(false)),
                counted_hit: false,
            },
        );
        let waited_urgency = Urgency::ahead(Want::undated(0.4));
        waited_urgency.hasten();
        prefetch.inbound.insert(
            (3, 0),
            Inbound {
                record: Some(3),
                tier: 3,
                bytes,
                need: Need::Ahead { ticket: Ticket::empty(), ahead: Ahead::Carryover { target_step: 1 } },
                ended: Ended::new(&events),
                urgency: waited_urgency,
                read_done: Arc::new(AtomicBool::new(false)),
                counted_hit: false,
            },
        );
        let mut cancelled_ticket = Ticket::empty();
        cancelled_ticket.cancel();
        prefetch.inbound.insert(
            (3, 1),
            Inbound {
                record: Some(4),
                tier: 4,
                bytes,
                need: Need::Ahead { ticket: cancelled_ticket, ahead: Ahead::Carryover { target_step: 1 } },
                ended: Ended::new(&events),
                urgency: Urgency::ahead(Want::undated(0.4)),
                read_done: Arc::new(AtomicBool::new(false)),
                counted_hit: false,
            },
        );
        assert_eq!(prefetch.inflight_carryover_reads(), 1);

        // Once the active carryover read finishes on the drive (`read_done = true`),
        // slot frees immediately for the next carryover read.
        active_done.store(true, Ordering::Release);
        assert_eq!(prefetch.inflight_carryover_reads(), 0);
    }
}
