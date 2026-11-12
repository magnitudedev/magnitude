//! Staging, the landing pool, and the reads into it: each request's demand
//! reads, and reads ahead of demand, each landing into the host tier as it
//! ends (`experts::landing`). A read ahead is one of two kinds: a hint's,
//! the next layer's hinted expert whose file is on the path (nothing holds,
//! keeps or lands it), or a covering tier's, an expert the tier lacks read
//! at the load's priority below every step's ([`Prefetch::cover`]) where
//! the tier has a record for every expert. Only a hint is a prediction: its
//! precision counts hints alone, and a request that finds a cover's read of
//! its miss waits for a read no prediction made. A tier that cannot hold
//! every expert keeps what the steps' misses and the hints land.
//!
//! A read's end submits its landing to the host compute executor on the
//! read engine's worker that ends it, once the read's end is set and its
//! tier records are claimed ([`Join`]: a request's read is submitted before
//! residency claims them, and where the claim comes last the request
//! service submits the landing; each landing is tracked by its own `Ended`
//! from its claim): the executor converts the stored record into
//! the tier record taken for it, maps it, and answers the choices of the
//! request that named it; a read that failed fails exactly the rows of that
//! request whose choices named its expert (`Steps::fail_request_rows`),
//! before it answers them with no record, so before its step can complete.
//! A read ahead names no request and fails nothing: a request whose expert
//! it reads waits for its landing, hastened to demand, and reads it again
//! should it fail (`Decode::resume`). The watcher learns each landing once
//! it ended
//! ([`Prefetch::settle`]): the tier record keeps its expert, or is ready
//! again, and the staging record frees.

use super::Shared;
use super::telemetry::Admitted;
use super::core::{At, Core, ReadTable};
use crate::experts::directory::{Directory, HostRecord, Point};
use crate::experts::host_records::{Claim, Taking};
use crate::experts::compute::Urgency;
use super::Step;
use crate::experts::landing::{Answer, Converting, DeviceLanding, DeviceWords, Ended, Landing};
use crate::experts::source::{Outcome, Read, Ticket};
use crate::experts::staging::{Staged, StagingPool};
use seismic_io::Want;
use magnitude_kernels::constants;
use std::collections::{BTreeSet, HashMap};
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

/// Account request `naming`'s demand read `ticket` (joined by `join`) as
/// it ends, once: a point of the read line over `timed` bytes where it
/// reads all the request's misses; and a read worker's panic, which ends
/// serving.
pub(super) fn account(ticket: &Ticket, join: &Join, naming: Naming, timed: Option<(u64, Instant)>, shared: &Arc<Shared>) {
    // When the volume began reading the job: from its submission, which
    // the request times where it reads all its misses, else now, just
    // after.
    let (started, submitted) = (ticket.started(), timed.map_or_else(Instant::now, |(_, submitted)| submitted));
    let shared = shared.clone();
    join.claimed(move |outcome, _, _| {
        panicked(&shared, &outcome);
        if let Outcome::Read = outcome {
            let timed = timed.map(|(bytes, submitted)| (bytes, submitted.elapsed()));
            let first = started.iter().filter_map(|started| started.get().copied()).min();
            let queued = first.map_or(std::time::Duration::ZERO, |first| first.saturating_duration_since(submitted));
            shared.landed(naming.layer, naming.sequence, naming.named.elapsed(), timed, queued);
        }
    });
}

/// A request that names reads: its step's layer and sequence, and when it
/// named them.
#[derive(Clone, Copy)]
pub(super) struct Naming {
    pub(super) layer: usize,
    pub(super) sequence: Step,
    pub(super) named: Instant,
}

/// A read a request named that lands one expert: its landing, the staging
/// record it reads into, the reads of the ticket it takes, the rows of the
/// request whose choices named it, and the choices it answers.
pub(super) struct Named {
    pub(super) landing: Landing,
    pub(super) staged: u32,
    pub(super) reads: std::ops::Range<usize>,
    pub(super) rows: BTreeSet<usize>,
    pub(super) answer: Answer,
}

/// The landings of one request's demand read: from the read's end to the
/// last one's, a point of its group's landing line (`FileLine`).
pub(super) struct Landings {
    layer: usize,
    count: usize,
    left: std::sync::atomic::AtomicUsize,
    read: Instant,
}

impl Landings {
    fn new(layer: usize, count: usize) -> Arc<Self> {
        Arc::new(Self { layer, count, left: std::sync::atomic::AtomicUsize::new(count), read: Instant::now() })
    }

    /// One of them ended.
    fn ended(&self, shared: &Shared) {
        if self.left.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
            shared.telemetry.landings_ended(self.layer, self.count, self.read.elapsed().as_secs_f64());
        }
    }
}

/// Whether named read `named` of request `naming` landed, as its ticket's
/// `failed` reads say; failed, its end is recorded, its rows fail, then it
/// answers none.
fn read_named(shared: &Shared, naming: Naming, named: &Named, outcome: &Outcome, failed: &[usize], ended: &Ended) -> bool {
    if !failed.iter().any(|read| named.reads.contains(read)) {
        shared.landings.read(naming.named.elapsed());
        return true;
    }
    if let Outcome::Failed(reason) = outcome {
        shared.steps.fail_request_rows(naming.sequence, naming.layer, &named.rows, reason.clone());
    }
    ended.set(false);
    named.answer.publish(None);
    false
}

/// End named read `named` of request `naming`, its landing on the host:
/// submitted to the executor, which converts and maps its record, records
/// its end in `ended`, then answers it. Where the read's end ran it
/// (`party`), its read worker works its pieces beside the executor's, so no
/// wake stands between the read and its landing.
fn end_named(
    shared: &Arc<Shared>,
    naming: Naming,
    named: Named,
    landings: &Arc<Landings>,
    outcome: &Outcome,
    failed: &[usize],
    (ended, urgency): (Ended, Arc<Urgency>),
    party: Party,
) {
    if !read_named(shared, naming, &named, outcome, failed, &ended) {
        return;
    }
    let (record, answer) = (named.landing.record(), named.answer);
    let (landings, serving) = (landings.clone(), shared.clone());
    // SAFETY: the read landed whole and nothing reads into its staging
    // record until the watcher frees it (once its end is recorded), and its
    // tier record was ready (no step reads it) when taken.
    let converting = unsafe {
        Converting::new(named.landing, urgency, shared.landings.clone(), ended, move || {
            answer.publish(Some(record));
            landings.ended(&serving);
        })
    };
    match party {
        Party::Read => shared.compute.run(Arc::new(converting)),
        Party::Claim => shared.compute.submit(Arc::new(converting)),
    }
}

/// End a request's named reads on the thread that ends them, their landings
/// on the device: the list of those that landed published while the step
/// owns its slot, each held as converting, then answered, so the device
/// finds its list past the barrier the answer opens. A step that ended
/// first converts nothing, and lands none.
fn end_converted(
    shared: &Arc<Shared>,
    naming: Naming,
    named: Vec<(Named, Arc<DeviceLanding>, Ended)>,
    words: &DeviceWords,
    outcome: &Outcome,
    failed: &[usize],
) {
    let landed = named
        .into_iter()
        .filter(|(named, _, ended)| read_named(shared, naming, named, outcome, failed, ended))
        .collect::<Vec<_>>();
    let pairs = landed.iter().map(|(named, ..)| (named.staged as i32, named.landing.record() as i32)).collect::<Vec<_>>();
    #[cfg(test)]
    std::thread::sleep(std::time::Duration::from_millis(shared.landings.held_ms.load(std::sync::atomic::Ordering::Relaxed)));
    let published = shared.steps.while_owned(naming.sequence, |_| words.publish(naming.sequence, &pairs)).is_some();
    for (named, device, ended) in landed {
        let (record, answer) = (named.landing.record(), named.answer);
        match published {
            true => device.converts(named.landing),
            false => ended.set(false),
        }
        answer.publish(Some(record));
    }
}

/// A panic that ended the reads `outcome` reports: a defect of the read
/// engine's, which ends serving (the stopped watcher fails every request
/// the device waits on).
fn panicked(shared: &Shared, outcome: &Outcome) {
    if let Outcome::Panicked(panic) = outcome {
        shared.end_serving(format!("a read worker panicked reading experts: {panic:?}"));
    }
}

/// An expert on its way into the host tier: read into staging record
/// `record`, landing into tier record `tier`, for what [`Need`] says; its
/// landing's end, once it landed (or its read failed), and its urgency on
/// the executor, the read's own. Directory places one landing of an expert
/// at a time (`Located::Landing`), so an expert has one at most.
struct Inbound {
    record: u32,
    tier: u32,
    need: Need,
    ended: Ended,
    urgency: Arc<Urgency>,
}

/// Why an expert is inbound, with the read that brings it.
enum Need {
    /// A request named it at step `read`: its read is the request's
    /// ticket, shared with the request's other landings, which ends before
    /// any of them; `device` where the device converts it.
    Due { ticket: Arc<Ticket>, read: Step, device: Option<Arc<DeviceLanding>> },
    /// Read ahead of demand, by its own ticket (cancelled once it can only
    /// be wrong): what it reads ahead for is `ahead`.
    Ahead { ticket: Ticket, ahead: Ahead },
}

/// What a read ahead is for.
#[derive(Clone, Copy)]
enum Ahead {
    /// A hint's, wanted from its layer's estimated start
    /// (`HintPrice::want`): the route at this `At` hinted it.
    Hint(At),
    /// The covering tier's ([`Prefetch::cover`]), undated.
    Cover,
}

impl Ahead {
    /// How its landing takes its tier record.
    fn taking(self) -> Taking {
        match self {
            Self::Hint(_) => Taking::Ahead,
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

    /// The step that read it: a request's, or the route's that hinted it;
    /// none for the cover's.
    fn read(&self) -> Option<Step> {
        match &self.need {
            Need::Due { read, .. } => Some(*read),
            Need::Ahead { ahead: Ahead::Hint(hinted), .. } => Some(hinted.point.sequence),
            Need::Ahead { ahead: Ahead::Cover, .. } => None,
        }
    }
}

/// A landing in flight a request waits for ([`Prefetch::hasten`]): a
/// hint's, issued by the route at its `At`, or one no route issued (the
/// cover's, or another request's).
pub(super) enum Reading {
    Hint(At),
    Unpredicted,
}

/// What became of one entry of a hint ring, as the read timeline records it
/// (`candidate`: the step as its job; the target layer, the expert, and
/// `short << 24 | fate << 16 | lead << 8 | rank`, with the staging records
/// free as it was admitted from bit 32): read; not a candidate (held,
/// kept, or landing already, a longer lead's read among them); named twice
/// in the ring; refused by its rank's precision (`HintPrice::pays`); paid
/// for and not read, short of what [`ShortOf`] says; or its ring read
/// after its target's request was noticed. Each fate keeps its number in
/// every recording.
#[derive(Clone, Copy)]
enum Fate {
    Read = 0,
    Held = 1,
    Repeated = 2,
    Refused = 3,
    Short = 5,
    Passed = 6,
}

impl Fate {
    /// Note the entry `(expert, rank)` of step `sequence`'s ring of lead
    /// `lead` hinting `target`, short of `short` with `free` staging
    /// records free.
    fn note(self, sequence: Step, target: usize, lead: usize, (expert, rank): (usize, usize), (short, free): (ShortOf, usize)) {
        let packed = (free as i64) << 32 | (short as i64) << 24 | (self as i64) << 16 | (lead as i64) << 8 | rank as i64;
        seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("candidate"), sequence, (target as i64, expert as i64, packed));
    }
}

/// What a hint ring entry paid for fell short of ([`Fate::Short`]).
#[derive(Clone, Copy)]
enum ShortOf {
    Nothing = 0,
    /// Staging beyond the demand reserve.
    Staging = 1,
    /// A ready tier record beyond the reserve.
    Tier = 2,
    /// The read engine took no job (`Short::Refused`).
    Claim = 3,
    /// An entry before it in the ring fell short.
    Earlier = 4,
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
    /// Per layer and lead (its rings' rows), the low word of the last step
    /// its ring row was read for.
    rings: Vec<[i32; crate::planning::HINT_LEADS as usize]>,
    /// Per layer, the last step whose route there its ring showed
    /// ([`Self::cancel_passed`]).
    routed: Vec<Step>,
}

impl Prefetch {
    /// `records` staging records, every one free, for `layers` layers.
    pub(super) fn new(records: u32, layers: usize) -> Self {
        Self {
            staging: Arc::new(StagingPool::new(records)),
            inbound: HashMap::new(),
            past: Point::START,
            rings: vec![[0; crate::planning::HINT_LEADS as usize]; layers],
            routed: vec![0; layers],
        }
    }

    /// Track the demand job `ticket` (joined by `join`), landing `named`
    /// for the request `naming` names, its tier records claimed: its end
    /// lands or fails each, on the host ([`end_named`]) or, where `device`
    /// names the words of the request's barrier at its point, on the device
    /// ([`end_converted`]). `landings` are each named landing's staging
    /// record, expert, tier record and read step, in order. A request's
    /// read is accounted apart, once ([`account`]).
    pub(super) fn demand(
        &mut self,
        (ticket, join): (&Arc<Ticket>, &Join),
        named: Vec<Named>,
        landings: Vec<(u32, Staged, u32, Step)>,
        naming: Naming,
        device: Option<(Arc<DeviceWords>, crate::experts::ExchangeTable, Point)>,
        shared: &Arc<Shared>,
    ) {
        let ends = landings.iter().map(|_| (Ended::new(shared.events()), Urgency::waited())).collect::<Vec<_>>();
        let converted = ends
            .iter()
            .map(|(ended, _)| device.as_ref().map(|(words, file, at)| Arc::new(DeviceLanding::new(words.clone(), file.clone(), *at, ended.clone(), shared.landings.clone()))))
            .collect::<Vec<_>>();
        let (shared, ended, held) = (shared.clone(), ends.clone(), converted.clone());
        let words = device.map(|(words, ..)| words);
        join.claimed(move |outcome, failed, party| {
            match &words {
                Some(words) => {
                    let named = named.into_iter().zip(held).zip(ended).map(|((named, held), (ended, _))| (named, held.expect("a device landing's"), ended)).collect();
                    end_converted(&shared, naming, named, words, &outcome, failed);
                }
                None => {
                    let landings = Landings::new(naming.layer, named.len());
                    for (named, ended) in named.into_iter().zip(ended) {
                        end_named(&shared, naming, named, &landings, &outcome, failed, ended, party);
                    }
                }
            }
        });
        assert!(!landings.is_empty(), "a demand read lands an expert");
        for (((record, staged, tier, read), (ended, urgency)), device) in landings.into_iter().zip(ends).zip(converted) {
            let need = Need::Due { ticket: ticket.clone(), read, device };
            self.arrive(staged, Inbound { record, tier, need, ended, urgency });
        }
    }

    /// The landing of `staged` in flight a step waits for (a decode
    /// request's miss, or a prompt group's pull, `Scratch::awaited`), its
    /// read and its landing at demand from now, and which kind it is: its
    /// landing maps its tier record, which the step is then served from;
    /// none where staging lands none or its read failed (the step reads the
    /// expert anew once its end is learned).
    pub(super) fn hasten(&self, staged: Staged) -> Option<Reading> {
        let noted = |code: i64| seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("hasten"), 0, (staged.0 as i64, staged.1 as i64, code));
        let Some(inbound) = self.inbound.get(&staged) else {
            noted(-1);
            return None;
        };
        let reading = match &inbound.need {
            Need::Due { .. } => {
                noted(0);
                inbound.urgency.hasten();
                return Some(Reading::Unpredicted);
            }
            Need::Ahead { ahead: Ahead::Hint(hinted), .. } => Reading::Hint(*hinted),
            Need::Ahead { ahead: Ahead::Cover, .. } => Reading::Unpredicted,
        };
        if inbound.ended.get() == Some(false) {
            noted(-2);
            return None;
        }
        noted(if matches!(reading, Reading::Hint(_)) { 1 } else { 2 });
        // Parked from the step that first waits on it, repriced then alone:
        // a hastened read orders before every read ahead, and never later
        // than it was; a later wait finds it due already.
        if !inbound.urgency.is_waited() {
            inbound.ticket().reprice(inbound.urgency.get().min(Want::now()));
            inbound.urgency.hasten();
        }
        Some(reading)
    }

    /// Cancel what every hint no request hastened has not started reading,
    /// once it can only be wrong: its target is at or before `served` (the
    /// request the service just served), its step has ended, or the device
    /// has routed past its target with no request there (the layer after's
    /// ring shows it, and a route's request is published before the next
    /// layer's ring, so one absent then was never made: every expert of that
    /// layer resolved without it). A cancelled hint reads nothing more, so it
    /// takes no source time from demand; its read ends failed, which frees
    /// its staging and tier records ([`Self::settle`]), and a later request
    /// for its expert reads it anew. A store seen out of that order only
    /// costs that read.
    pub(super) fn cancel_passed(&mut self, core: &Core, steps: &super::steps::Steps, served: Option<(Step, usize)>) {
        for (&(target, _), inbound) in &mut self.inbound {
            let Need::Ahead { ticket, ahead: Ahead::Hint(hinted) } = &mut inbound.need else {
                continue;
            };
            if inbound.urgency.is_waited() || ticket.cancelled() {
                continue;
            }
            let issued = hinted.point.sequence;
            let passed = served.is_some_and(|served| (issued, target) <= served)
                || steps.ended(issued)
                || (self.routed.get(target + 1).is_some_and(|&routed| routed >= issued) && !core.layers[target].file.requested(issued, 0));
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
        let [mut hints, mut covers, mut due, mut running] = [0; 4];
        for inbound in self.inbound.values() {
            match inbound.need {
                Need::Due { .. } => due += 1,
                Need::Ahead { ahead: Ahead::Hint(_), .. } => hints += 1,
                Need::Ahead { ahead: Ahead::Cover, .. } => covers += 1,
            }
            running += usize::from(inbound.ticket().running());
        }
        format!(
            "staging {} free of {}; inbound: hints {hints}, covers {covers}, due {due}; reads running for {running}",
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
    /// ready again, and each staging record frees. True when any ended.
    pub(super) fn settle(&mut self, directory: &mut Directory) -> bool {
        for inbound in self.inbound.values() {
            if let Need::Due { device: Some(device), .. } = &inbound.need {
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
        let settled = !ended.is_empty();
        for (record, staged, tier, read, landed) in ended {
            self.ended(directory, record, staged, tier, read, landed);
        }
        settled
    }

    /// A landing of `staged` from staging record `record` into tier record
    /// `tier`, read at step `read` (none for the cover's), ended: landed or
    /// not.
    fn ended(&mut self, directory: &mut Directory, record: u32, staged: Staged, tier: u32, read: Option<Step>, landed: bool) {
        let (layer, expert) = staged;
        match landed {
            true => directory.landed(layer, HostRecord(tier), expert, read),
            false => directory.unlanded(layer, HostRecord(tier)),
        }
        self.staging.release(record);
    }

    /// Read each layer's ring if it changed, of a step whose request at the
    /// next layer is not yet noticed (`served`, each layer's last request
    /// noticed; a later one's ring is decided, its route there having
    /// chosen), and admit its hints ([`Self::admit`]). A hint only ever
    /// reads the file: where the host tier keeps every expert
    /// (`Directory::tier_complete`) no hint has a candidate, and no ring is
    /// read. True when it read any ring.
    pub(super) fn hint(&mut self, core: &mut Core, shared: &Arc<Shared>, served: &[Step]) -> bool {
        if core.directory.tier_complete() {
            return false;
        }
        let words = core.extents.words;
        let ring_words = words.ring_words() as usize;
        let mut read = false;
        for layer in 0..core.layers.len() {
            // Each lead's row (`routed_route_streamed`'s rings: the layer
            // one ahead, then two, `Layer::hints`); the last streamed layer
            // has none.
            for lead in 0..core.layers[layer].hints.len() {
                let target = core.layers[layer].hints[lead];
                let ring = &core.layers[layer].rings;
                let base = lead * ring_words;
                // The ring carries its step's low word.
                let low = ring.load(base + constants::RING_SEQUENCE as usize);
                if low == 0 || low == self.rings[layer][lead] {
                    continue;
                }
                let count = usize::try_from(ring.load(base + constants::RING_COUNT as usize))
                    .ok()
                    .filter(|&count| count <= words.choices as usize)
                    .unwrap_or_else(|| panic!("a hint ring of {} experts", ring.load(base + constants::RING_COUNT as usize)));
                // Each entry's expert and the rank of the choice it was
                // hinted as (HINT_EXPERT, HINT_RANK).
                let entry = |index: usize, word: u64| {
                    ring.load(base + (constants::RING_EXPERTS + constants::HINT_ENTRY_WORDS * index as u64 + word) as usize)
                };
                let entries = (0..count)
                    .map(|index| (entry(index, constants::HINT_EXPERT), entry(index, constants::HINT_RANK)))
                    .collect::<Vec<_>>();
                // The ring is a seqlock (lib/streaming/hint): the device
                // stores sequence 0 before it rewrites the list and the
                // step's after, so a list read under one unchanged nonzero
                // sequence is that step's whole; one read while the device
                // rewrote it is read again on the next pass.
                if ring.load(base + constants::RING_SEQUENCE as usize) != low {
                    continue;
                }
                self.rings[layer][lead] = low;
                read = true;
                let Some(sequence) = shared.steps.of_low_word(low) else {
                    continue;
                };
                self.routed[layer] = self.routed[layer].max(sequence);
                self.cancel_passed(core, &shared.steps, None);
                // Those of a step whose request there was noticed name a
                // layer the device has passed.
                if served[target] >= sequence {
                    if seismic_io::timeline::recording() {
                        for &(expert, rank) in &entries {
                            Fate::Passed.note(sequence, target, lead, (expert as usize, rank as usize), (ShortOf::Nothing, 0));
                        }
                    }
                    continue;
                }
                let hinted = entries
                    .into_iter()
                    .map(|(expert, rank)| {
                        let expert = usize::try_from(expert)
                            .ok()
                            .filter(|&expert| expert < core.layers[target].experts)
                            .unwrap_or_else(|| panic!("a hint names expert {expert}"));
                        let rank = usize::try_from(rank)
                            .ok()
                            .filter(|&rank| rank < core.extents.selected)
                            .unwrap_or_else(|| panic!("a hint of expert {expert} names choice rank {rank}"));
                        (expert, rank)
                    })
                    .collect::<Vec<_>>();
                let issued = shared.steps.at(sequence, layer, core.layers.len());
                self.admit(core, shared, target, lead, &hinted, issued);
            }
        }
        read
    }

    /// Whether `staged` is a hint's candidate: an expert that lives only in
    /// the file: the bank lacks it, and its host tier keeps, graces and
    /// lands none of it (`Directory::located`).
    fn candidate(core: &Core, staged: Staged) -> bool {
        let (layer, expert) = staged;
        core.directory.held(layer, expert).is_none() && core.directory.located(layer, expert) == crate::experts::host_records::Located::Lacked
    }

    /// Admit the hints `hinted` (each expert with the rank of the choice it
    /// was hinted as) of `target` at lead `lead` (its ring row: 0 one layer
    /// ahead, 1 two), issued by the route at `issued`: each
    /// candidate (an expert whose file is on the path:
    /// nothing holds or lands it; one a longer lead's hint reads lands
    /// already) whose rank's precision at its lead pays for its read of its
    /// bytes, on the volume as busy as steps keep it (`HintPrice::pays`), is
    /// read into a
    /// staging record beyond the demand reserve, wanted from its layer's
    /// estimated start and worth its expected bytes spared
    /// (`HintPrice::want`), landing into a ready tier record, in the ring's
    /// order, best ranked first (`publish_hints`): the hints of one ring
    /// share a deadline, which the link orders by value, and where the
    /// staging beyond the demand reserve or the ready tier records fall
    /// short, the hints left are the least likely. Its candidates, each with whether it was
    /// read and its rank, are counted with their step (`Telemetry::hinted`).
    fn admit(&mut self, core: &mut Core, shared: &Arc<Shared>, target: usize, lead: usize, hinted: &[(usize, usize)], issued: At) {
        let sequence = issued.point.sequence;
        let recording = seismic_io::timeline::recording();
        let mut seen = BTreeSet::new();
        let ranked = hinted
            .iter()
            .copied()
            .filter(|&(expert, rank)| {
                let fate = match (seen.insert(expert), Self::candidate(core, (target, expert))) {
                    (false, _) => Fate::Repeated,
                    (true, false) => Fate::Held,
                    (true, true) => return true,
                };
                if recording {
                    fate.note(sequence, target, lead, (expert, rank), (ShortOf::Nothing, 0));
                }
                false
            })
            .collect::<Vec<_>>();
        // A hint takes staging beyond the demand reserve, so a request
        // finds its own, and the ready tier records beyond what the
        // reserve holds past the reads ahead's staging (none where the
        // reserve is the reads ahead's alone, a load's: a request reclaims a
        // kept record at its park where none is ready). Its landing then
        // displaces the tier's least reused as the tier refills its reserve.
        let request = core.extents.demand;
        let demand = core.extents.reserve.saturating_sub(self.staging.records().saturating_sub(request));
        let price = shared.telemetry.price(lead);
        let mut short = false;
        let now = Instant::now();
        for &(expert, rank) in &ranked {
            let free = if recording { self.staging.free() } else { 0 };
            let noted = |fate: Fate, of: ShortOf| {
                if recording {
                    fate.note(sequence, target, lead, (expert, rank), (of, free));
                }
            };
            if short {
                noted(Fate::Short, ShortOf::Earlier);
                continue;
            }
            if !price.pays(rank, core.layers[target].expert_bytes()) {
                noted(Fate::Refused, ShortOf::Nothing);
                continue;
            }
            let want = |bytes| price.want(rank, target - issued.point.layer, bytes, now);
            match self.read_ahead(core, shared, (target, expert), want, Ahead::Hint(issued), (request, demand)) {
                Ok(bytes) => {
                    shared.telemetry.hint(bytes);
                    noted(Fate::Read, ShortOf::Nothing);
                }
                Err(Short::Refused) => noted(Fate::Short, ShortOf::Claim),
                Err(fell @ (Short::Staging | Short::Tier)) => {
                    short = true;
                    noted(Fate::Short, if matches!(fell, Short::Staging) { ShortOf::Staging } else { ShortOf::Tier });
                }
            }
        }
        // No candidate was a hint's before: those that are now were read.
        let candidates = ranked
            .into_iter()
            .map(|(expert, rank)| Admitted { expert, rank, read: self.inbound.contains_key(&(target, expert)) })
            .collect();
        shared.telemetry.hinted(sequence, target, lead, candidates);
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
    /// its misses and hints.
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
        let keep = core.extents.demand;
        let room = self.staging.free().saturating_sub(keep);
        if room == 0 {
            return;
        }
        for staged in core.directory.cover_order(room, carried) {
            if self.staging.free() <= keep {
                break;
            }
            match self.read_ahead(core, shared, staged, |_| Want::undated(0.), Ahead::Cover, (keep, 0)) {
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

    /// Read `staged` ahead, `ahead` saying for what: into a staging record wanted as `want`
    /// gives for its bytes, of those free beyond `staging`, landing into a
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
        want: impl FnOnce(u64) -> Want,
        ahead: Ahead,
        (staging, tier): (usize, usize),
    ) -> Result<u64, Short> {
        let (target, expert) = staged;
        let Some(record) = self.staging.take_beyond(staging) else {
            return Err(Short::Staging);
        };
        let Claim::Taken(grant) = core.directory.claim_landing(target, expert, tier, ahead.taking()) else {
            self.staging.release(record);
            return Err(Short::Tier);
        };
        let tier = grant.record();
        let reads = core.table.reads(target, expert, record);
        let bytes = reads.iter().map(crate::experts::source::Read::bytes).sum::<u64>();
        let landing = core.landing(grant, record);
        let want = want(bytes);
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
                    panicked(&serving, &outcome);
                    let read = failed.is_empty() && matches!(outcome, Outcome::Read);
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("ahead_read"), 0, (target as i64, expert as i64, read as i64));
                    if read {
                        // SAFETY: as `end_named`'s.
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
                self.arrive(staged, Inbound { record, tier, need: Need::Ahead { ticket, ahead }, ended, urgency });
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
    use super::Prefetch;

    /// A request's read posted before residency serves it, or left waiting
    /// for tier records, or given up while it runs, holds its staging record
    /// and is in no landing residency tracks: it is outstanding until its
    /// record frees, so no resize or traffic hold finds the load quiet while
    /// it may still write into staging.
    #[test]
    fn a_read_holding_staging_outside_the_landings_is_outstanding() {
        let prefetch = Prefetch::new(2, 1);
        assert!(!prefetch.outstanding(), "nothing read");
        let posted = prefetch.staging.take_up_to(1);
        assert!(prefetch.outstanding(), "a posted read's staging record is taken");
        posted.into_iter().for_each(|record| prefetch.staging.release(record));
        assert!(!prefetch.outstanding(), "its record freed as it ended");
    }
}
