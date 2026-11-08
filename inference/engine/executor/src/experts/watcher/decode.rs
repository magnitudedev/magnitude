//! Decode requests (`routed_route_streamed`): each expert neither the bank
//! nor its host tier holds (the device serves the tier's itself) lands
//! into a ready tier record of its layer's group (`experts::landing`):
//! landed by its hint's read, or read now into staging. Each choice's
//! record is written as its expert lands, so no landing waits for
//! another's read, and the request's slot is answered as its last lands; a
//! landing is no promotion into the bank, which the ranking alone decides
//! (`promotions`).
//!
//! A request is served in two parts. The request service's look ([`Look`])
//! takes the device's request with no lock of residency's and decides
//! nothing: each absent expert lacked by the read path's mirror
//! (`host_records::InboundTable`), whose word it claims, is read at once
//! into a staging record taken from the pool (`staging::StagingPool`), its
//! ranges from the load's read table (`core::ReadTable`). Residency's pass
//! then serves the request ([`Decode::serve`]): it claims each read's tier
//! record, builds its landing and does the request's accounting, and
//! serves every other absent expert (kept, landing already, or one the look
//! could not read) as before. A read's landing runs once both its read has
//! ended and its records are claimed (`prefetch::Join`), so neither waits
//! for the other.

use super::{Shared, Step};
use super::core::{At, Core, ReadTable, RoutingRecord};
use super::prefetch::{self, Join, Named, Naming, Posting, Prefetch, Reading};
use super::telemetry::{Anticipation, RequestCost};
use crate::experts::directory::{HostRecord, Park};
use crate::experts::host_records::{Claim, Demand, Grant, InboundTable, Located};
use crate::experts::landing::{Answer, Source};
use crate::experts::source::Ticket;
use crate::experts::staging::{Staged, StagingPool};
use crate::experts::{ExchangeTable, Words};
use crate::planning::RegionWords;
use magnitude_kernels::{SlotTable, constants};
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

/// What serving a decode request reads and writes: the core, the staging,
/// and what the watcher shares with the steps.
pub(super) struct Serving<'a> {
    pub(super) core: &'a mut Core,
    pub(super) prefetch: &'a mut Prefetch,
    pub(super) shared: &'a Arc<Shared>,
}

/// Fail step `sequence`'s file request of `layer`: the failure is recorded,
/// then published (the slot's failure), so its wait returns, sourcing
/// nothing.
pub(super) fn fail(core: &Core, shared: &Shared, layer: usize, sequence: Step, reason: String) {
    shared.steps.fail(sequence, reason);
    core.layers[layer].file.answer(sequence, 0, true);
    shared.parks.settle();
}

/// What the look reads of a streamed layer: its file table, its policy
/// words and its experts.
struct Watched {
    file: ExchangeTable,
    policy: Words,
    experts: usize,
}

/// The request service's look at the device's decode requests, taken
/// before residency's lock: what it reads, fixed at load, and per layer the
/// last request it took.
pub(super) struct Look {
    layers: Vec<Watched>,
    words: RegionWords,
    table: Arc<ReadTable>,
    staging: Arc<StagingPool>,
    inbound: Arc<InboundTable>,
    /// Per layer, the last request taken.
    served: Vec<Step>,
}

/// A request the look took ([`Look::look`]): its step and layer, when it
/// was noticed, each choice's expert (none for a held one), its absent
/// experts, and the read it posted of those it claimed, if any.
pub(super) struct Posted {
    pub(super) sequence: Step,
    pub(super) layer: usize,
    noticed: Instant,
    choices: Vec<Option<usize>>,
    absent: Vec<usize>,
    read: Option<PostedRead>,
}

/// The read a look posted of a request's experts: each one's staging
/// record and its reads of the posting, when it was submitted, and the
/// posting, or why the read engine took none.
struct PostedRead {
    landings: Vec<(usize, u32, Range<usize>)>,
    submitted: Instant,
    posting: Result<Posting, String>,
}

impl PostedRead {
    /// Whether it reads `expert`.
    fn reads(&self, expert: usize) -> bool {
        self.landings.iter().any(|&(read, ..)| read == expert)
    }
}

/// When the look began, for the read timeline: the previous pass's end and
/// the look's start.
#[derive(Clone, Copy)]
pub(super) struct Began {
    pub(super) ended: Instant,
    pub(super) looked: Instant,
}

impl Look {
    /// The look at `core`'s layers, its reads into `staging`.
    pub(super) fn new(core: &Core, staging: Arc<StagingPool>) -> Self {
        Self {
            layers: core
                .layers
                .iter()
                .map(|layer| Watched { file: layer.file.clone(), policy: layer.policy.clone(), experts: layer.experts })
                .collect(),
            words: core.extents.words,
            table: core.table.clone(),
            staging,
            inbound: core.directory.inbound(),
            served: vec![0; core.layers.len()],
        }
    }

    /// Each layer's last request taken.
    pub(super) fn served(&self) -> &[Step] {
        &self.served
    }

    /// Layer `layer`'s earliest file request not yet taken, taken now, by
    /// its step: none while every slot's request is taken, or of a step
    /// ended (read now: steps end beside the request service; one a device
    /// error ended, which no launch waits for). A layer's requests come in
    /// step order, the device's.
    fn request(&mut self, shared: &Shared, layer: usize) -> Option<Step> {
        let file = &self.layers[layer].file;
        let live = shared.steps.live();
        let unserved = live.start.max(self.served[layer] + 1)..live.end;
        let sequence = file.earliest_requested(unserved, 0, |step| !shared.steps.ended(step))?;
        self.served[layer] = sequence;
        Some(sequence)
    }

    /// Take each layer's next request where the device's requests are the
    /// host's to read (`Parks::parked`), each read posted at once
    /// ([`Self::post`]): a request's reads are the step's critical path,
    /// and nothing of residency's stands before them.
    pub(super) fn look(&mut self, shared: &Shared, began: Began) -> Vec<Posted> {
        if !shared.parks.parked() {
            return Vec::new();
        }
        (0..self.layers.len())
            .filter_map(|layer| {
                let sequence = self.request(shared, layer)?;
                Some(self.post(sequence, layer, began))
            })
            .collect()
    }

    /// Take step `sequence`'s request at `layer` and post the read of each
    /// absent expert whose word it claims while a staging record is free;
    /// every other absent expert is left to residency's pass.
    fn post(&mut self, sequence: Step, layer: usize, began: Began) -> Posted {
        let noticed = Instant::now();
        if let (Some(ended), Some(looked)) = (seismic_io::timeline::stamp(began.ended), seismic_io::timeline::stamp(began.looked)) {
            seismic_io::timeline::note_at(noticed, seismic_io::timeline::Kind::Noted("pass"), sequence, (layer as i64, looked as i64, looked as i64));
            seismic_io::timeline::note_at(noticed, seismic_io::timeline::Kind::Noted("pass_ended"), sequence, (layer as i64, ended as i64, 0));
        }
        let watched = &self.layers[layer];
        // No wait is encoded where the word is set, so a request there would
        // leave its choices unsourced: an invariant violation, which ends
        // serving (`Requests::serve`).
        assert_eq!(
            watched.policy.load(constants::POLICY_RESOLVED as usize),
            0,
            "step {sequence} requested the file at layer {layer}, whose experts cannot be absent"
        );
        let (file, experts, width) = (&watched.file, watched.experts, self.words.choices);
        let count = file.load(sequence, 0, constants::FILE_COUNT);
        // The route requests at most the decode form's choices, each an
        // absent expert or -1 - a held one (routed.seismic).
        let count = usize::try_from(count)
            .ok()
            .filter(|&count| count as u64 <= width)
            .unwrap_or_else(|| panic!("a file request of {count} choices"));
        let mut absent: Vec<usize> = Vec::new();
        let mut choices = Vec::with_capacity(count);
        for choice in 0..count {
            let entry = SlotTable::file_entry(choice as u64);
            let word = file.load(sequence, 0, entry + constants::FILE_CHOICE);
            // No record is the step's until its landing writes it
            // (`parked_reads`).
            file.store(sequence, 0, entry + constants::FILE_RECORD, -1);
            match usize::try_from(word) {
                Ok(expert) => {
                    assert!(expert < experts, "a request names expert {word} of {experts}");
                    if !absent.contains(&expert) {
                        absent.push(expert);
                    }
                    choices.push(Some(expert));
                }
                Err(_) => {
                    let held = -1 - i64::from(word);
                    assert!((0..experts as i64).contains(&held), "a request names held choice {word} of {experts} experts");
                    choices.push(None);
                }
            }
        }
        seismic_io::timeline::note_at(noticed, seismic_io::timeline::Kind::Noted("notice"), sequence, (layer as i64, absent.len() as i64, count as i64));
        // Each absent expert lacked, its word claimed, into a staging
        // record: those taken beyond the claims go back.
        let mut free = self.staging.take_up_to(absent.len());
        let mut landings = Vec::new();
        let mut reads = Vec::new();
        for &expert in &absent {
            if free.is_empty() {
                break;
            }
            if !self.inbound.claim((layer, expert)) {
                continue;
            }
            let record = free.pop().expect("a free record, looked at above");
            let begun = reads.len();
            reads.extend(self.table.reads(layer, expert, record));
            landings.push((expert, record, begun..reads.len()));
        }
        for record in free {
            self.staging.release(record);
        }
        let read = (!landings.is_empty()).then(|| {
            let reading = landings.len() as i64;
            seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("decided"), sequence, (layer as i64, reading, 0));
            let submitted = Instant::now();
            let posting = prefetch::post(&self.table, reads, seismic_io::Want::now());
            if let Ok(posting) = &posting {
                for request in posting.ticket.requests() {
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("demand"), request, (layer as i64, sequence as i64, reading));
                }
            }
            PostedRead { landings, submitted, posting }
        });
        Posted { sequence, layer, noticed, choices, absent, read }
    }
}

/// How a request's absent expert lands: read now, the look's posted read
/// of it, its tier record
/// claimed as residency serves it; already landed in a tier record; or once
/// its read ahead (a hint's or the tier's fill's, hastened to demand) has
/// landed, or, the look having read nothing of it (no staging record free,
/// or it was not lacked then and is now), once a staging and a tier record
/// are (`Waiting`): a shortage of records is backpressure, never a failure,
/// and never a wait of the request service.
enum Plan {
    Posted,
    Unread(Unread),
}

/// How an absent expert read by nothing now lands ([`Plan`]): kept already
/// in tier record `tier`; waiting for the landing in flight that reads it
/// (`Reading`); or waiting for a staging or tier record before it is read
/// (`Waits`).
enum Unread {
    Kept(u32),
    Reading,
    Waits,
}

/// An absent expert of a served request waiting for its read ahead's
/// landing, or to be read (`record`: once it found no staging or tier
/// record free): its choices' answer, its rows, and the request's point.
struct Waiting {
    at: At,
    expert: usize,
    answer: Answer,
    rows: BTreeSet<usize>,
    named: Instant,
    record: bool,
}

/// A served request's posted read (`ticket`, joined by `join`), some of
/// whose tier records were short as it was served: its `landings` wait for
/// them, in order, its read running or ended ([`prefetch::Join`]); the
/// others land as they would.
struct Unclaimed {
    at: At,
    naming: Naming,
    ticket: Arc<Ticket>,
    join: Arc<Join>,
    landings: Vec<Landing>,
}

/// A posted read's landing of one expert: its staging record and its reads
/// of the posting, and its request's rows and answer.
struct Landing {
    expert: usize,
    record: u32,
    reads: Range<usize>,
    rows: BTreeSet<usize>,
    answer: Answer,
}

/// The decode requests served: the landings waiting for a record, and the
/// posted reads waiting for theirs.
pub(super) struct Decode {
    waiting: Vec<Waiting>,
    unclaimed: Vec<Unclaimed>,
}

impl Decode {
    /// The misses waiting, for the exchange's watchdog: each one's step,
    /// layer and expert, and since when its request named it.
    pub(super) fn describe(&self) -> String {
        let waiting = self
            .waiting
            .iter()
            .map(|waiting| format!("step {} layer {} expert {} for {:?}", waiting.at.point.sequence, waiting.at.point.layer, waiting.expert, waiting.named.elapsed()))
            .chain(self.unclaimed.iter().map(|unclaimed| {
                format!("step {} layer {} read awaiting tier records for {:?}", unclaimed.at.point.sequence, unclaimed.at.point.layer, unclaimed.naming.named.elapsed())
            }))
            .collect::<Vec<_>>();
        format!("misses waiting [{}]", waiting.join("; "))
    }
}

/// The records the device may still read at or after its park at `layer`
/// of step `sequence` (its tier wait) that its route resolved there (its
/// routing record's sources), each as (group, record) of the records the
/// host lands into (`landing::Source`, under every placer); those
/// answered for its request are outstanding answers, which nothing takes
/// (`HostRecords::answer`). Every other resolution a record is freed past
/// lies before the park
/// ([`super::super::directory::Directory::release`]). Its routing record
/// is the parked step's: the route publishes it before its request
/// (`streaming/resolve.hpp`), so a record of another step is the device's
/// broken invariant, never a set of reads to free past.
pub(super) fn parked_reads(core: &Core, layer: usize, sequence: Step) -> Vec<(usize, u32)> {
    let placer = core.extents.placer;
    let group = core.layers[layer].group;
    let record = |word: i32| match Source::of(placer, word) {
        Source::Host(record) => Some((group, record as u32)),
        Source::Absent | Source::Device => None,
    };
    let routing = RoutingRecord::read(&core.layers[layer].routing, core.extents.words, sequence, layer);
    routing.choices().filter_map(|(_, source)| record(source)).collect()
}

/// Claim a demand landing of `staged` for `demand`, the device parked at
/// `park` (`Directory::claim_demand`): kept
/// or landing already, that (a lacked expert's alone); else a ready tier
/// record of its group, one the reserve holds ready (its graces released at
/// the park, `Directory::release`), or one a kept record's unmapping frees
/// at once, the unmapping stored (by the host records, its one writer)
/// before any landing writes the record; never one a resolution at or after
/// the park may name ([`parked_reads`]), nor a request's outstanding answer
/// (`HostRecords::answer`). The record taken answers the parked request
/// once it lands.
fn demand_claim(core: &mut Core, staged: Staged, demand: Demand, park: &Park) -> Claim {
    let read = parked_reads(core, staged.0, park.point.sequence);
    core.directory.release(park, &read);
    core.directory.claim_demand(staged, demand, &read, park)
}

/// A posted read's tier record for `staged`: taken, or none while the tier
/// has none to take.
fn posted_claim(core: &mut Core, staged: Staged, park: &Park) -> Option<Grant> {
    match demand_claim(core, staged, Demand::Posted, park) {
        Claim::Taken(grant) => Some(grant),
        Claim::Short => None,
        Claim::Kept(_) | Claim::Landing => unreachable!("a posted read's claim finds its expert lacked"),
    }
}

/// Give up a posted read's `landings` at `layer`, each (expert, staging
/// record), whose tier records were never claimed: their experts are
/// lacked again, the read's pieces not yet taken are cancelled where it
/// lands nothing else, and their staging frees as the read ends
/// (`Join::claimed`), at once where the read engine took none. Until then
/// the records stay taken, so the load's traffic is not quiet
/// (`Prefetch::outstanding`).
fn give_up(core: &mut Core, prefetch: &Prefetch, shared: &Shared, layer: usize, read: Option<(Arc<Ticket>, Arc<Join>)>, landings: impl IntoIterator<Item = (usize, u32)>) {
    let mut records = Vec::new();
    for (expert, record) in landings {
        core.directory.unposted(layer, expert);
        records.push(record);
    }
    let (staging, events) = (prefetch.staging.clone(), shared.events().clone());
    let free = move || {
        for record in records {
            staging.release(record);
        }
        events.notify();
    };
    match read {
        None => free(),
        Some((mut ticket, join)) => {
            if let Some(ticket) = Arc::get_mut(&mut ticket) {
                ticket.cancel();
            }
            join.claimed(move |_, _, _| {
                // Every request of the ticket has ended: dropping it waits
                // for none.
                drop(ticket);
                free();
            });
        }
    }
}

/// Tier records for a posted read's landings of `experts` at `layer`, the
/// device parked at `parked`, in order: each claimed while the tier has
/// one, and none for it and every later one once it is short. The one rule
/// a posted read's landings claim by, as residency serves its request and
/// as they resume (`Decode::resume`), so they claim in the order they were
/// read.
fn posted_claims(core: &mut Core, layer: usize, experts: impl IntoIterator<Item = usize>, park: &Park) -> Vec<Option<Grant>> {
    let mut short = false;
    experts
        .into_iter()
        .map(|expert| {
            let grant = if short { None } else { posted_claim(core, (layer, expert), park) };
            short |= grant.is_none();
            grant
        })
        .collect()
}

/// Whether step `sequence`'s request at `layer` ended: its step retired or
/// failed, or its request failed. Nothing waits on its answers.
fn request_ended(core: &Core, shared: &Shared, layer: usize, sequence: Step) -> bool {
    shared.steps.ended(sequence) || core.layers[layer].file.answer_of(sequence, 0) == Some(crate::experts::exchange::Answer::Failed)
}

impl Posted {
    /// Give up this request's posted read, serving having ended before
    /// residency served it ([`give_up`]); its choices are refused with the
    /// rest (`liveness::refuse`). Whether it posted any.
    pub(super) fn give_up(self, core: &mut Core, prefetch: &Prefetch, shared: &Shared) -> bool {
        let Some(PostedRead { landings, posting, .. }) = self.read else {
            return false;
        };
        let read = posting.ok().map(|Posting { ticket, join }| (Arc::new(ticket), join));
        give_up(core, prefetch, shared, self.layer, read, landings.into_iter().map(|(expert, record, _)| (expert, record)));
        true
    }
}

impl Decode {
    pub(super) fn new() -> Self {
        Self { waiting: Vec::new(), unclaimed: Vec::new() }
    }

    /// Carry what waits: the posted reads still short of tier records
    /// claim them, and each waiting expert its tier keeps now (its read
    /// ahead landed) is answered, each whose read ahead still runs passed
    /// over, and the rest a staging and a tier record are free for read
    /// now and answered as it lands, in the order they waited; one of a step
    /// that ended (retired, or failed) is dropped, as its launches returned.
    /// The landings that ended are learned first (`Prefetch::settle`), so a
    /// miss waits for its read ahead's landing alone, never for residency's
    /// next turn to learn it. True when any landing was learned, or any
    /// miss claimed, answered, landed or dropped.
    pub(super) fn resume(&mut self, serving: Serving<'_>) -> bool {
        let Serving { core, prefetch, shared } = serving;
        let mut progressed = !(self.waiting.is_empty() && self.unclaimed.is_empty()) && prefetch.settle(&mut core.directory);
        let ended = |core: &Core, at: At| request_ended(core, shared, at.point.layer, at.point.sequence);
        for mut unclaimed in std::mem::take(&mut self.unclaimed) {
            let (at, layer) = (unclaimed.at, unclaimed.at.point.layer);
            if ended(core, at) {
                let Unclaimed { ticket, join, landings, .. } = unclaimed;
                give_up(core, prefetch, shared, layer, Some((ticket, join)), landings.into_iter().map(|landing| (landing.expert, landing.record)));
                progressed = true;
                continue;
            }
            let park = shared.steps.park(at.point, core.layers.len());
            let landings = std::mem::take(&mut unclaimed.landings);
            let grants = posted_claims(core, layer, landings.iter().map(|landing| landing.expert), &park);
            let mut claimed = Vec::new();
            for (landing, grant) in landings.into_iter().zip(grants) {
                match grant {
                    Some(grant) => claimed.push((landing, grant)),
                    None => unclaimed.landings.push(landing),
                }
            }
            if !claimed.is_empty() {
                land(core, prefetch, shared, &unclaimed, claimed, None);
                progressed = true;
            }
            if !unclaimed.landings.is_empty() {
                self.unclaimed.push(unclaimed);
            }
        }
        let mut waiting = std::mem::take(&mut self.waiting);
        waiting.retain(|waits| {
            if ended(core, waits.at) {
                progressed = true;
                return false;
            }
            true
        });
        let mut index = 0;
        while let Some(waits) = waiting.get(index) {
            let (layer, expert) = (waits.at.point.layer, waits.expert);
            let staged = (layer, expert);
            match core.directory.located(layer, expert) {
                // Kept already (landed meanwhile by its read ahead):
                // answered with its record.
                Located::Kept(tier) => {
                    core.directory.answer(layer, HostRecord(tier), waits.at.point);
                    waiting.remove(index).answer.publish(Some(tier));
                    progressed = true;
                    continue;
                }
                // A landing of it is in flight, hastened to demand: it
                // lands it, or fails and lacks it again.
                Located::Landing => {
                    prefetch.hasten(staged);
                    index += 1;
                    continue;
                }
                Located::Lacked => {}
            }
            // A record to read it into: none free (a failed step's read
            // still holds one) waits for one, its landing's end an event.
            // Short of a record: waits for one, counted once.
            let short = |waiting: &mut Vec<Waiting>| {
                let waits = &mut waiting[index];
                if !std::mem::replace(&mut waits.record, true) {
                    shared.landings.waits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            };
            let Some(record) = prefetch.staging.take_beyond(0) else {
                short(&mut waiting);
                break;
            };
            let park = shared.steps.park(waits.at.point, core.layers.len());
            // Lacked, as located above: a claim takes a record, or finds
            // none ready (or a posted read claimed it since).
            let Claim::Taken(grant) = demand_claim(core, staged, Demand::Lacked, &park) else {
                prefetch.staging.release(record);
                short(&mut waiting);
                break;
            };
            let tier = grant.record();
            let waits = waiting.remove(index);
            progressed = true;
            if waits.record {
                crate::experts::landing::LandingAccount::add(&shared.landings.waited, waits.named.elapsed());
            }
            let reads = core.table.reads(layer, expert, record);
            let count = reads.len();
            let landing = Named {
                landing: core.landing(grant, record),
                staged: record,
                reads: 0..count,
                rows: waits.rows,
                answer: waits.answer,
            };
            let naming = Naming { layer, sequence: waits.at.point.sequence, named: waits.named };
            match prefetch::post(&core.table, reads, seismic_io::Want::now()) {
                Ok(Posting { ticket, join }) => {
                    for request in ticket.requests() {
                        seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("demand"), request, (layer as i64, naming.sequence as i64, -1));
                    }
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("plan"), naming.sequence, (layer as i64, expert as i64, 5));
                    prefetch::account(&ticket, &join, naming, None, shared);
                    prefetch.demand((&Arc::new(ticket), &join), vec![landing], vec![(record, staged, tier, naming.sequence)], naming, None, shared)
                }
                // The read was never made: as a failed read, only the rows that named the
                // expert fail, and its wait is answered with none, so the rest of the step serves.
                Err(reason) => {
                    core.directory.unlanded(layer, HostRecord(tier));
                    prefetch.staging.release(record);
                    shared.steps.fail_request_rows(naming.sequence, layer, &landing.rows, reason);
                    landing.answer.publish(None);
                }
            }
        }
        self.waiting = waiting;
        progressed
    }

    /// Serve the request the look posted, `at` its point, the device past
    /// every point before it: its posted read's tier records claimed and
    /// its landings joined to the read (each runs once both have come); its
    /// other absent experts each answered (kept), waiting for its landing in
    /// flight (hastened to demand), or waiting for records; and the
    /// request's accounting.
    pub(super) fn serve(&mut self, posted: Posted, at: At, serving: Serving<'_>) {
        let Serving { core, prefetch, shared } = serving;
        // A request whose step ended before residency served it is given
        // up: its posted read's staging frees as it ends.
        if request_ended(core, shared, posted.layer, posted.sequence) {
            posted.give_up(core, prefetch, shared);
            return;
        }
        let Posted { noticed: started, choices, absent, read, .. } = posted;
        let point = at.point;
        let (layer, sequence) = (point.layer, point.sequence);
        let width = core.extents.words.choices;
        let file = core.layers[layer].file.clone();
        // Choice c = m K + k is row m's (routed.seismic): the rows that
        // named an expert, which its read's failure fails alone, and the
        // choices its landing answers.
        let per_row = core.extents.selected;
        let named = |expert: usize| choices.iter().enumerate().filter(move |(_, named)| **named == Some(expert)).map(|(choice, _)| choice);
        let rows_of = |expert: usize| named(expert).map(|choice| choice / per_row).collect::<BTreeSet<_>>();
        let placer = core.extents.placer;
        // Each absent expert answers once, the last the slot.
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(absent.len()));
        let answer_of = |expert: usize| Answer {
            layer,
            file: file.clone(),
            sequence,
            choices: named(expert).collect(),
            pending: pending.clone(),
            placer,
            steps: shared.steps.clone(),
            parks: shared.parks.clone(),
        };
        let naming = Naming { layer, sequence, named: started };
        // Each absent expert's landing: the look's read, a read ahead's,
        // hastened to demand (reading), or one read once records are free.
        // The request's lead is the layers since its last read a route
        // issued was submitted: none where every miss had landed already, a
        // read it never saw. A fill's read no route issued is a demand's to
        // it: no prediction made it.
        let mut served = RequestCost { missed: absent.clone(), ..RequestCost::default() };
        let mut plans = Vec::with_capacity(absent.len());
        let mut lead: Option<u64> = None;
        let record_bytes = core.layers[layer].expert_bytes();
        for &expert in &absent {
            let staged = (layer, expert);
            let mut taken_from = |from: At| {
                let layers = from.runs_until(at);
                lead = Some(lead.map_or(layers, |lead| lead.min(layers)));
            };
            let (plan, anticipation) = match read.as_ref().is_some_and(|read| read.reads(expert)) {
                true => (Plan::Posted, Some(Anticipation::Demand)),
                false => match core.directory.located(layer, expert) {
                    // Kept already: landed after the route resolved it (a
                    // read ahead's that ended and settled since), answered
                    // with its record, an outstanding answer from now,
                    // which no claim of the request's other misses takes.
                    Located::Kept(tier) => {
                        core.directory.answer(layer, HostRecord(tier), point);
                        (Plan::Unread(Unread::Kept(tier)), None)
                    }
                    // A landing of it is in flight, whoever's (a read
                    // ahead's, another request's, the fill's): it waits for
                    // it, hastened to demand.
                    Located::Landing => match prefetch.hasten(staged) {
                        Some(Reading::Hint(issued)) => {
                            taken_from(issued);
                            (Plan::Unread(Unread::Reading), Some(Anticipation::Reading))
                        }
                        Some(Reading::Unpredicted) | None => (Plan::Unread(Unread::Reading), Some(Anticipation::Demand)),
                    },
                    // The look read nothing of it: it waits for records.
                    Located::Lacked => (Plan::Unread(Unread::Waits), Some(Anticipation::Demand)),
                },
            };
            let code = match (&plan, anticipation) {
                (Plan::Unread(Unread::Kept(_)), _) => 0,
                (_, Some(Anticipation::Reading)) => 1,
                (Plan::Unread(Unread::Reading), _) => 2,
                (Plan::Unread(Unread::Waits), _) => 3,
                _ => 4,
            };
            seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("plan"), sequence, (layer as i64, expert as i64, code));
            // A miss kept already reads nothing: no read miss, nor its bytes.
            match anticipation {
                Some(anticipation) => served.anticipated[anticipation as usize] += 1,
                None => served.missed.retain(|&missed| missed != expert),
            }
            plans.push((expert, plan));
        }
        // The posted read's tier records, each claimed now: a short tier
        // leaves its landing waiting for one, its read in flight
        // (`Self::resume`).
        let read = read.map(|read| {
            let grants = match read.posting {
                Ok(_) => posted_claims(core, layer, read.landings.iter().map(|&(expert, ..)| expert), &shared.steps.park(point, core.layers.len())),
                Err(_) => Vec::new(),
            };
            (read, grants)
        });
        let short = read.as_ref().map_or(0, |(_, grants): &(PostedRead, Vec<Option<Grant>>)| grants.iter().filter(|grant| grant.is_none()).count());
        seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("claimed"), sequence, (layer as i64, read.as_ref().map_or(0, |(read, _)| read.landings.len()) as i64, short as i64));
        // Its own hints hastened, the device is at this layer: the hints of
        // layers it has begun or passed read nothing more. Cancelling them
        // frees no record now (a cancelled read frees its own as it ends).
        prefetch.cancel_passed(core, &shared.steps, Some((sequence, layer)));
        // The next request's victims, prepared now its reads are in flight
        // where this one took any or they are of an earlier step: a claim
        // then takes the first still free, scanning nothing while its step
        // waits. Twice a request's choices, as some are read by the step a
        // claim parks or gone by then.
        if read.is_some() {
            lead = Some(0);
            core.directory.prepare_demand(layer, 2 * width as usize, sequence);
        }
        served.lead = lead;
        served.misses = served.missed.len() as u64;
        served.bytes = served.misses * record_bytes;
        served.parked = started.elapsed();
        // Counted before any answer, which releases the device to complete
        // the step: no landing is registered yet, nor answer published.
        shared.telemetry.serve(layer, sequence, served);
        // The device's convert holds its staging until the device is past
        // this barrier, which opens only once every miss is answered, so a
        // miss waiting for a record rules it out: the host's landings free
        // their records without the device.
        let convertible = short == 0 && !plans.iter().any(|(_, plan)| matches!(plan, Plan::Unread(Unread::Waits)));
        if let Some((PostedRead { landings, submitted, posting }, grants)) = read {
            let reading = landings.len();
            match posting {
                // The tier map's unmappings, a ready record's reclaim among
                // them, were stored as each claim took its record, before
                // any landing starts: no step after the parked one resolves
                // an unmapped expert to a record a landing writes, however
                // soon the landing answers.
                Ok(Posting { ticket, join }) => {
                    let timed = (reading == absent.len()).then_some((reading as u64 * record_bytes, submitted));
                    prefetch::account(&ticket, &join, naming, timed, shared);
                    let mut unclaimed = Unclaimed { at, naming, ticket: Arc::new(ticket), join, landings: Vec::new() };
                    let mut claimed = Vec::new();
                    for ((expert, record, reads), grant) in landings.into_iter().zip(grants) {
                        let landing = Landing { expert, record, reads, rows: rows_of(expert), answer: answer_of(expert) };
                        match grant {
                            Some(grant) => claimed.push((landing, grant)),
                            None => unclaimed.landings.push(landing),
                        }
                    }
                    // Where a request's misses land is priced per request:
                    // the device's convert past its barrier, or the host's
                    // landing.
                    let device = core.layers[layer]
                        .landing
                        .clone()
                        .filter(|_| convertible && shared.telemetry.device_lands(layer, reading as u64))
                        .map(|words| (words, core.layers[layer].file.clone(), point));
                    if !claimed.is_empty() {
                        land(core, prefetch, shared, &unclaimed, claimed, device);
                    }
                    if !unclaimed.landings.is_empty() {
                        self.unclaimed.push(unclaimed);
                    }
                }
                // The reads were never made: as failed reads, only the rows
                // that named their experts fail, and their waits are
                // answered with none, so the rest of the step serves.
                Err(reason) => {
                    let experts = landings.iter().map(|&(expert, ..)| expert).collect::<Vec<_>>();
                    give_up(core, prefetch, shared, layer, None, landings.into_iter().map(|(expert, record, _)| (expert, record)));
                    for expert in experts {
                        shared.steps.fail_request_rows(sequence, layer, &rows_of(expert), reason.clone());
                        answer_of(expert).publish(None);
                    }
                }
            }
        }
        let mut waits = false;
        for (expert, plan) in plans {
            match plan {
                Plan::Posted => continue,
                Plan::Unread(Unread::Kept(tier)) => {
                    answer_of(expert).publish(Some(tier));
                    continue;
                }
                // Its landing in flight, or its read, answers it as the
                // watcher resumes it (`Self::resume`).
                Plan::Unread(Unread::Reading) => {}
                Plan::Unread(Unread::Waits) => waits = true,
            }
            self.waiting.push(Waiting { at, expert, answer: answer_of(expert), rows: rows_of(expert), named: started, record: false });
        }
        // What the look could not read (staging an earlier request's
        // landings held until residency's advance learned their end) is
        // read now, its records free since.
        if waits {
            self.resume(Serving { core, prefetch, shared });
        }
    }
}

/// Join landings `claimed` of `unclaimed`'s read, each with its tier
/// record, to the read (`Prefetch::demand`): landed on the device where
/// `device` names the words of the request's barrier, else on the host.
fn land(
    core: &Core,
    prefetch: &mut Prefetch,
    shared: &Arc<Shared>,
    unclaimed: &Unclaimed,
    claimed: Vec<(Landing, Grant)>,
    device: Option<(Arc<crate::experts::landing::DeviceWords>, ExchangeTable, crate::experts::directory::Point)>,
) {
    let Unclaimed { at, naming, ticket, join, .. } = unclaimed;
    let (layer, sequence) = (at.point.layer, at.point.sequence);
    let (mut named, mut tracked) = (Vec::with_capacity(claimed.len()), Vec::with_capacity(claimed.len()));
    for (Landing { expert, record, reads, rows, answer }, grant) in claimed {
        tracked.push((record, (layer, expert), grant.record(), sequence));
        named.push(Named { landing: core.landing(grant, record), staged: record, reads, rows, answer });
    }
    prefetch.demand((ticket, join), named, tracked, *naming, device, shared);
}
