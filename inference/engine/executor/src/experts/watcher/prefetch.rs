//! Staging, the landing pool, and the reads into it: each request's demand
//! reads, and reads ahead of demand, each landing into the host tier as it
//! ends (`experts::landing`). A read ahead is one of two kinds: a hint's,
//! the next layer's hinted expert whose file is on the path (nothing holds,
//! keeps or lands it), or the tier's fill's, an expert the tier lacks read
//! at the load's priority below every step's ([`Prefetch::fill`]). Only a
//! hint is a prediction: its precision counts hints alone, and a request
//! that finds a fill's read of its miss waits for a read no prediction
//! made.
//!
//! A read's end submits its landing to the host compute executor on the
//! read engine's worker that ends it, once the read's end is set
//! (`Ticket::on_end`; each landing is tracked by its own `Ended` from
//! before its read is submitted): the executor converts the stored record into
//! the tier record taken for it, maps it, and answers the choices of the
//! request that named it; a read that failed fails exactly the rows of that
//! request whose choices named its expert (`Steps::fail_request_rows`),
//! before it answers them with no record, so before its step can complete.
//! A read ahead names no request and fails nothing: a request whose expert
//! it reads waits for its landing, hastened to demand, and reads it again
//! should it fail (`Decode::resume`). The watcher learns each landing once
//! it ended
//! ([`Prefetch::settle`]): the tier record keeps its expert, or is ready
//! again, and the staging record frees, keeping the stored expert where its
//! read landed.

use super::Shared;
use super::telemetry::Admitted;
use super::core::{At, Core};
use crate::experts::directory::{Directory, HostRecord, Point};
use crate::experts::host_records::Claim;
use crate::experts::compute::Urgency;
use super::Step;
use crate::experts::landing::{Answer, Converting, DeviceLanding, DeviceWords, Ended, Landing};
use crate::experts::source::{Outcome, Ticket};
use crate::experts::staging::{Staged, Staging};
use magnitude_artifacts::Priority;
use magnitude_kernels::constants;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

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

/// End named read `named` of request `naming` on the thread that ends it,
/// its landing on the host: submitted to the executor, which converts and
/// maps its record, records its end in `ended`, then answers it. The read
/// worker works its pieces beside the executor's, so no wake stands between
/// the read and its landing.
fn end_named(
    shared: &Arc<Shared>,
    naming: Naming,
    named: Named,
    landings: &Arc<Landings>,
    outcome: &Outcome,
    failed: &[usize],
    (ended, urgency): (Ended, Arc<Urgency>),
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
    shared.compute.run(Arc::new(converting));
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
    let mut published = false;
    shared.steps.while_owned(naming.sequence, || {
        words.publish(naming.sequence, &pairs);
        published = true;
    });
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

/// A read ahead of demand in flight: an expert being read into staging
/// record `record` and landed into tier record `tier` before any request
/// names it, by its ticket; its landing's end, once it landed (or its read
/// failed); its landing's urgency on the executor, the read's own.
struct Ahead {
    record: u32,
    tier: u32,
    ticket: Ticket,
    urgency: Arc<Urgency>,
    ended: Ended,
}

/// A hint's read ahead, submitted at [`Priority::Prefetch`] when the route
/// at `issued` hinted it.
struct Hint {
    ahead: Ahead,
    issued: At,
}

/// The tier's fill's read ahead, submitted at [`Priority::Load`]: no route
/// named it.
struct Fill {
    ahead: Ahead,
}

/// A landing in flight a request waits for ([`Prefetch::hasten`]): a
/// hint's, issued by the route at its `At`, or one no route issued (the
/// fill's, or another request's).
pub(super) enum Reading {
    Hint(At),
    Unpredicted,
}

/// A landing a request named, or the fill landed from what staging kept,
/// in flight: its staging record, the expert, its tier record, the step
/// that read it, its end once recorded, and its urgency on the executor.
struct Tracked {
    record: u32,
    staged: Staged,
    tier: u32,
    read: Option<Step>,
    ended: Ended,
    urgency: Arc<Urgency>,
    /// Where the device converts it.
    device: Option<Arc<DeviceLanding>>,
}

/// The staging records and the reads into them.
pub(super) struct Prefetch {
    /// Staging record indices, valid in every pool.
    pub(super) staging: Staging,
    /// Where each staging record's last read was issued: none where no
    /// route issued it (never read, or read by the tier's fill).
    pub(super) read_at: Vec<Option<At>>,
    /// Hints' reads in flight, by the expert they stage.
    hints: HashMap<Staged, Hint>,
    /// The fill's reads in flight, by the expert they stage.
    fills: HashMap<Staged, Fill>,
    /// Demand reads in flight.
    demands: Vec<Ticket>,
    /// Landings a request named, or the fill landed from staging, until the
    /// watcher learns their ends.
    landings: Vec<Tracked>,
    /// How far the device is known to be (`Residency::advance`), which ends
    /// the landings it converted behind it.
    past: Point,
    /// Per layer and lead (its rings' rows), the low word of the last step
    /// its ring row was read for.
    rings: Vec<[i32; crate::planning::HINT_LEADS as usize]>,
    /// Per layer, the last step whose route there its ring showed
    /// ([`Self::withdraw_passed`]).
    routed: Vec<Step>,
}

impl Prefetch {
    /// `records` staging records, every one free, for `layers` layers.
    pub(super) fn new(records: u32, layers: usize) -> Self {
        Self {
            staging: Staging::new(records),
            read_at: vec![None; records as usize],
            hints: HashMap::new(),
            fills: HashMap::new(),
            demands: Vec::new(),
            landings: Vec::new(),
            past: Point { group: 0, layer: 0, sequence: 0 },
            rings: vec![[0; crate::planning::HINT_LEADS as usize]; layers],
            routed: vec![0; layers],
        }
    }

    /// Track the demand job `ticket`, landing `named` for the request
    /// `naming` names, a point of the read line over `timed` bytes where it
    /// reads all the request's: its end lands or fails each, on the host
    /// ([`end_named`]) or, where `device` names the words of the request's
    /// barrier at its point, on the device ([`end_converted`]).
    /// `landings` are each named landing's staging record, expert, tier
    /// record and read step, in order.
    pub(super) fn demand(
        &mut self,
        ticket: Ticket,
        named: Vec<Named>,
        landings: Vec<(u32, Staged, u32, Step)>,
        naming: Naming,
        timed: Option<(u64, Instant)>,
        device: Option<(Arc<DeviceWords>, crate::experts::ExchangeTable, Point)>,
        shared: &Arc<Shared>,
    ) {
        let ends = landings.iter().map(|_| (Ended::new(shared.events()), Urgency::new(Priority::Demand))).collect::<Vec<_>>();
        let converted = ends
            .iter()
            .map(|(ended, _)| device.as_ref().map(|(words, file, at)| Arc::new(DeviceLanding::new(words.clone(), file.clone(), *at, ended.clone(), shared.landings.clone()))))
            .collect::<Vec<_>>();
        let (shared, ended, held) = (shared.clone(), ends.clone(), converted.clone());
        let words = device.map(|(words, ..)| words);
        // When the volume began reading the job: from its submission, which
        // the request times where it reads all its misses, else now, just
        // after.
        let (started, submitted) = (ticket.started(), timed.map_or_else(Instant::now, |(_, submitted)| submitted));
        ticket.on_end(move |outcome, failed| {
            panicked(&shared, &outcome);
            if let Outcome::Read = outcome {
                let timed = timed.map(|(bytes, submitted)| (bytes, submitted.elapsed()));
                let first = started.iter().filter_map(|started| started.get().copied()).min();
                let queued = first.map_or(std::time::Duration::ZERO, |first| first.saturating_duration_since(submitted));
                shared.landed(naming.layer, naming.sequence, naming.named.elapsed(), timed, queued);
            }
            match &words {
                Some(words) => {
                    let named = named.into_iter().zip(held).zip(ended).map(|((named, held), (ended, _))| (named, held.expect("a device landing's"), ended)).collect();
                    end_converted(&shared, naming, named, words, &outcome, failed);
                }
                None => {
                    let landings = Landings::new(naming.layer, named.len());
                    for (named, ended) in named.into_iter().zip(ended) {
                        end_named(&shared, naming, named, &landings, &outcome, failed, ended);
                    }
                }
            }
        });
        self.demands.push(ticket);
        self.landings.extend(
            landings
                .into_iter()
                .zip(ends)
                .zip(converted)
                .map(|(((record, staged, tier, read), (ended, urgency)), device)| Tracked { record, staged, tier, read: Some(read), ended, urgency, device }),
        );
    }

    /// Land `landing` of `staged`, revived whole in staging record `record`
    /// for step `read` (none for the fill's), on the executor at `urgency`, then `answer` it (a
    /// request's choices with its tier record once mapped, or nothing): the
    /// watcher learns its end as a demand's.
    pub(super) fn revive(
        &mut self,
        landing: Landing,
        (staged, record, read): (Staged, u32, Option<Step>),
        urgency: Priority,
        answer: Option<Answer>,
        shared: &Shared,
    ) {
        let ended = Ended::new(shared.events());
        let urgency = Urgency::new(urgency);
        let tier = landing.record();
        // SAFETY: staging keeps the expert whole and no read lands in a
        // revived record until the watcher frees it; the tier record is
        // ready.
        let converting = unsafe {
            Converting::new(landing, urgency.clone(), shared.landings.clone(), ended.clone(), move || {
                if let Some(answer) = answer {
                    answer.publish(Some(tier));
                }
            })
        };
        shared.compute.submit(Arc::new(converting));
        self.landings.push(Tracked { record, staged, tier, read, ended, urgency, device: None });
    }

    /// The landing of `staged` in flight a step waits for (a decode
    /// request's miss, or a prompt group's pull, `Scratch::awaited`), its
    /// read and its landing at demand from now, and which kind it is: its
    /// landing maps its tier record, which the step is then served from;
    /// none where staging lands none or its read failed (the step reads the
    /// expert anew once its end is learned).
    pub(super) fn hasten(&self, staged: Staged) -> Option<Reading> {
        if let Some(tracked) = self.landings.iter().find(|tracked| tracked.staged == staged) {
            tracked.urgency.promote(Priority::Demand);
            return Some(Reading::Unpredicted);
        }
        let reading = match (self.hints.get(&staged), self.fills.contains_key(&staged)) {
            (Some(hint), _) => Reading::Hint(hint.issued),
            (None, true) => Reading::Unpredicted,
            (None, false) => return None,
        };
        let ahead = self.ahead(staged).expect("a read ahead in flight");
        if ahead.ended.get() == Some(false) {
            return None;
        }
        ahead.ticket.promote(Priority::Demand);
        ahead.urgency.promote(Priority::Demand);
        Some(reading)
    }

    /// Withdraw what every hint no request hastened has not started reading,
    /// once it can only be wrong: its target is at or before `served` (the
    /// request the service just served), its step has ended, or the device
    /// has routed past its target with no request there (the layer after's
    /// ring shows it, and a route's request is published before the next
    /// layer's ring, so one absent then was never made: every expert of that
    /// layer resolved without it). A withdrawn hint reads nothing more, so it
    /// takes no source time from demand; its read ends failed, which frees
    /// its staging and tier records ([`Self::settle`]), and a later request
    /// for its expert reads it anew. A store seen out of that order only
    /// costs that read.
    pub(super) fn withdraw_passed(&self, core: &Core, steps: &super::steps::Steps, served: Option<(Step, usize)>) {
        for (&(target, _), hint) in &self.hints {
            if hint.ahead.urgency.get() != Priority::Prefetch {
                continue;
            }
            let issued = hint.issued.point.sequence;
            let passed = served.is_some_and(|served| (issued, target) <= served)
                || steps.ended(issued)
                || (self.routed.get(target + 1).is_some_and(|&routed| routed >= issued) && !core.layers[target].file.requested(issued, 0));
            if passed {
                hint.ahead.ticket.withdraw();
            }
        }
    }

    /// The read ahead of `staged` in flight, of either kind.
    fn ahead(&self, staged: Staged) -> Option<&Ahead> {
        self.hints.get(&staged).map(|hint| &hint.ahead).or_else(|| self.fills.get(&staged).map(|fill| &fill.ahead))
    }

    /// The reads ahead in flight, hints' and the fill's.
    fn aheads(&self) -> impl Iterator<Item = &Ahead> {
        self.hints.values().map(|hint| &hint.ahead).chain(self.fills.values().map(|fill| &fill.ahead))
    }

    /// Whether a read into staging still runs: one ahead of demand, or one
    /// a request named, by its own end, whether or not [`Self::settle`] has
    /// seen it.
    fn reading(&self) -> bool {
        self.aheads().any(|ahead| ahead.ticket.running()) || self.demands.iter().any(Ticket::running)
    }

    /// Staging and what is in flight into it, for the exchange's watchdog.
    pub(super) fn describe(&self) -> String {
        format!(
            "staging {} free of {}; reads in flight: hints {}, fills {}, demands {} ({} running); landings in flight {}",
            self.staging.free(),
            self.staging.records(),
            self.hints.len(),
            self.fills.len(),
            self.demands.len(),
            self.demands.iter().filter(|ticket| ticket.running()).count(),
            self.landings.len(),
        )
    }

    /// Whether a read or its landing is outstanding: a read running, or a
    /// landing whose end residency has not learned. Each one's end is an
    /// event of the load (its landing's `Ended`, every read landing or
    /// ending so).
    pub(super) fn outstanding(&self) -> bool {
        !self.hints.is_empty() || !self.fills.is_empty() || !self.landings.is_empty() || self.reading()
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
    /// ready again, and each staging record frees, keeping its expert where
    /// it landed. True when any ended.
    pub(super) fn settle(&mut self, directory: &mut Directory) -> bool {
        let now = Instant::now();
        for device in self.landings.iter().filter_map(|tracked| tracked.device.as_ref()) {
            device.end(self.past);
        }
        self.demands.retain(|ticket| ticket.wait_until(now).is_none());
        let mut ended = Vec::new();
        self.hints.retain(|&staged, hint| match hint.ahead.ended.get() {
            Some(landed) => {
                ended.push((hint.ahead.record, staged, hint.ahead.tier, Some(hint.issued.point.sequence), landed));
                false
            }
            None => true,
        });
        self.fills.retain(|&staged, fill| match fill.ahead.ended.get() {
            Some(landed) => {
                ended.push((fill.ahead.record, staged, fill.ahead.tier, None, landed));
                false
            }
            None => true,
        });
        let mut index = 0;
        while index < self.landings.len() {
            let Some(landed) = self.landings[index].ended.get() else {
                index += 1;
                continue;
            };
            let Tracked { record, staged, tier, read, .. } = self.landings.swap_remove(index);
            ended.push((record, staged, tier, read, landed));
        }
        let settled = !ended.is_empty();
        for (record, staged, tier, read, landed) in ended {
            self.ended(directory, record, staged, tier, read, landed);
        }
        settled
    }

    /// A landing of `staged` from staging record `record` into tier record
    /// `tier`, read at step `read` (none for the fill's), ended: landed or
    /// not.
    fn ended(&mut self, directory: &mut Directory, record: u32, staged: Staged, tier: u32, read: Option<Step>, landed: bool) {
        let (layer, expert) = staged;
        match landed {
            true => directory.landed(layer, HostRecord(tier), expert, read),
            false => directory.unlanded(layer, HostRecord(tier)),
        }
        self.staging.release(record, landed.then_some(staged));
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
                self.withdraw_passed(core, &shared.steps, None);
                // Those of a step whose request there was noticed name a
                // layer the device has passed.
                if served[target] >= sequence {
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
    /// the file (the bank lacks it, its host tier holds none of it) and
    /// nothing stages or lands (whatever lands it took its tier record,
    /// `Directory::lands`).
    fn candidate(core: &Core, staging: &Staging, staged: Staged) -> bool {
        let (layer, expert) = staged;
        !(core.directory.lands(layer, expert)
            || core.directory.held(layer, expert).is_some()
            || core.directory.in_tier(layer, expert)
            || staging.keeps(staged))
    }

    /// Admit the hints `hinted` (each expert with the rank of the choice it
    /// was hinted as) of `target` at lead `lead` (its ring row: 0 one layer
    /// ahead, 1 two), issued by the route at `issued`: each
    /// candidate (an expert whose file is on the path:
    /// nothing holds, keeps or lands it; one a longer lead's hint reads lands
    /// already) whose rank's precision at its lead pays for its read
    /// (`Telemetry::pays`) is read into a
    /// staging record beyond the demand reserve at [`Priority::Prefetch`],
    /// landing into a ready tier record, in the ring's
    /// order, best ranked first (`publish_hints`): the read engine serves a
    /// priority's jobs in submission order, so the likeliest hints are read
    /// first, and where the staging beyond the demand reserve or the ready
    /// tier records fall short, the hints
    /// left are the least likely. Its candidates, each with whether it was
    /// read and its rank, are counted with their step (`Telemetry::hinted`).
    fn admit(&mut self, core: &mut Core, shared: &Arc<Shared>, target: usize, lead: usize, hinted: &[(usize, usize)], issued: At) {
        let mut seen = BTreeSet::new();
        let ranked = hinted
            .iter()
            .copied()
            .filter(|&(expert, _)| seen.insert(expert) && Self::candidate(core, &self.staging, (target, expert)))
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
        for &(expert, rank) in &ranked {
            if !price.pays(rank) {
                continue;
            }
            match self.read_ahead(core, shared, (target, expert), Priority::Prefetch, (request, demand)) {
                Ok((ahead, bytes)) => {
                    self.read_at[ahead.record as usize] = Some(issued);
                    self.hints.insert((target, expert), Hint { ahead, issued });
                    shared.telemetry.hint(bytes);
                }
                Err(Short::Refused) => {}
                Err(Short::Staging | Short::Tier) => break,
            }
        }
        // No candidate was a hint's before: those that are now were read.
        let candidates = ranked
            .into_iter()
            .map(|(expert, rank)| Admitted { expert, rank, read: self.hints.contains_key(&(target, expert)) })
            .collect();
        shared.telemetry.hinted(issued.point.sequence, target, lead, candidates);
    }

    /// Fill the host tier: each expert its tier lacks that nothing carries,
    /// in the order the tier lands them (`Directory::fill_order`: the most
    /// reused first, a tie, never used yet, each layer's alike), is landed
    /// into a tier record beyond the tier's reserve (`Extents::reserve`,
    /// or every expert still lacked, where fewer: `HostRecords::claim`), so
    /// it displaces nothing the tier keeps: from staging where staging
    /// keeps it whole, no read; else read into a staging record at
    /// [`Priority::Load`], below every step's reads, as a hint's read
    /// lands, so a request that waits for one hastens its read. `carried`
    /// names the experts a prompt group's conveyor carries, which land
    /// through it (`Scratch::carries`): no expert is read twice.
    ///
    /// The tier's size is the plan's, decided at load, and it fills from the
    /// bank's open on: a tier with a record for every expert of its group is
    /// filled to every expert, and a decode miss of it then never reads the
    /// file; a smaller one's records beyond the reserve with the experts
    /// most reused when they are filled, which its misses' landings then
    /// displace by reuse as they rank. One rule fills both.
    ///
    /// How many it reads at once: by the read engine's law
    /// (`magnitude_artifacts::reader`) a demand read waits for at most the
    /// volume's window of bytes its workers already hold, however much less
    /// urgent work is queued, so the depth of the fill's queue costs no step
    /// anything; and its reads keep the engine busy at its rate only while
    /// its queue holds the reads that cover the latency `λ` of each next
    /// one and the watcher's turn to submit it, which no bound below the
    /// staging they land into guarantees. So the fill reads into every
    /// staging record beyond the demand reserve; the
    /// hints' records among them it shares with hints, which each pass
    /// admits before it fills, so a hint takes each record as it frees.
    pub(super) fn fill(&mut self, core: &mut Core, shared: &Arc<Shared>, carried: impl Fn(Staged) -> bool) {
        let (keep, reserve) = (core.extents.demand, core.extents.reserve);
        let room = self.staging.free().saturating_sub(keep);
        if room == 0 {
            return;
        }
        // Among equals, what staging keeps first: it lands without a read,
        // and no read of its equal takes its record.
        let staging = &self.staging;
        let order = core.directory.fill_order(reserve, room, carried, |staged| staging.keeps(staged));
        for staged in order {
            if self.staging.free() <= keep {
                break;
            }
            let (layer, expert) = staged;
            if let Some(record) = self.staging.revive(staged) {
                match core.directory.claim_landing(layer, expert, reserve) {
                    Claim::Taken(grant) => {
                        self.read_at[record as usize] = None;
                        let landing = core.landing(grant, record);
                        self.revive(landing, (staged, record, None), Priority::Load, None, shared);
                    }
                    Claim::Kept(_) | Claim::Landing | Claim::Short => self.staging.release(record, Some(staged)),
                }
                continue;
            }
            match self.read_ahead(core, shared, staged, Priority::Load, (keep, reserve)) {
                Ok((ahead, _)) => {
                    self.read_at[ahead.record as usize] = None;
                    self.fills.insert(staged, Fill { ahead });
                }
                // Refused, or its group's tier holds no record for it now:
                // the fill goes on to the next.
                Err(Short::Refused | Short::Tier) => {}
                Err(Short::Staging) => break,
            }
        }
    }

    /// Whether the fill still lands the tier ([`Self::fill`]): a fill's read
    /// or landing is in flight, a growth of the tier runs, or a tier has a
    /// record beyond its reserve that the fill takes
    /// (`Directory::fill_pending`). Not once the tier holds every record it
    /// is planned to and each beyond the reserve keeps an expert, or the
    /// tier keeps every expert: from then only misses' landings change what
    /// the tier keeps, displacing it at the same records.
    pub(super) fn filling(&self, core: &Core) -> bool {
        !self.fills.is_empty() || core.directory.fill_pending(core.extents.reserve)
    }

    /// Read `staged` into a staging record at `priority`, of those free
    /// beyond `staging`, landing into a tier record of its group the tier
    /// takes for it leaving `tier` ready (`Directory::claim_landing`): the read ahead
    /// and its bytes, which its kind tracks, so a step that waits for it
    /// hastens its read and its landing to demand ([`Self::hasten`]), and
    /// the watcher learns its end ([`Self::settle`]); or what it fell short
    /// of.
    fn read_ahead(
        &mut self,
        core: &mut Core,
        shared: &Arc<Shared>,
        staged: Staged,
        priority: Priority,
        (staging, tier): (usize, usize),
    ) -> Result<(Ahead, u64), Short> {
        let (target, expert) = staged;
        if self.staging.free() <= staging {
            return Err(Short::Staging);
        }
        let Claim::Taken(grant) = core.directory.claim_landing(target, expert, tier) else {
            return Err(Short::Tier);
        };
        let tier = grant.record();
        let directory = &core.directory;
        let record = self
            .staging
            .take(|staged| directory.staged_worth(staged))
            .expect("a record beyond the reserve");
        let reads = core.reads(target, expert, record);
        let bytes = reads.iter().map(crate::experts::source::Read::bytes).sum::<u64>();
        let landing = core.landing(grant, record);
        match core.source.submit(reads, priority) {
            Ok(ticket) => {
                let ended = Ended::new(shared.events());
                let urgency = Urgency::new(priority);
                let (end, serving, urgent) = (ended.clone(), shared.clone(), urgency.clone());
                // A read ahead's end: read, its landing is submitted, whose
                // end maps its tier record; failed, it fails nothing (a step
                // waiting for it reads its expert anew).
                ticket.on_end(move |outcome, failed| {
                    panicked(&serving, &outcome);
                    if failed.is_empty() && matches!(outcome, Outcome::Read) {
                        // SAFETY: as `end_named`'s.
                        let converting = unsafe { Converting::new(landing, urgent, serving.landings.clone(), end, || {}) };
                        serving.compute.submit(Arc::new(converting));
                        return;
                    }
                    end.set(false);
                });
                Ok((Ahead { record, tier, ticket, urgency, ended }, bytes))
            }
            // The engine took no job: the read stages and lands nothing and
            // fails nothing (a demand for the expert reads it again, and
            // fails its own step if that read fails).
            Err(_) => {
                core.directory.unlanded(target, HostRecord(tier));
                self.staging.release(record, None);
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
