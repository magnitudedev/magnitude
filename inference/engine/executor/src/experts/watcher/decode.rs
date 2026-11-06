//! Decode requests (`routed_route_streamed`): each expert neither the bank
//! nor its host tier holds (the device serves the tier's itself) lands
//! into a ready tier record of its layer's group (`experts::landing`):
//! revived from staging at once, landed by its hint's read, or read now
//! into staging. Each choice's record is written as its expert lands, so no
//! landing waits for another's read, and the request's slot is answered as
//! its last lands; a landing is no promotion into the bank, which the
//! ranking alone decides (`promotions`).

use super::{Shared, Step};
use super::core::{At, Core, RoutingRecord};
use super::prefetch::{Named, Naming, Prefetch, Reading};
use super::telemetry::{Anticipation, RequestCost};
use crate::experts::directory::HostRecord;
use crate::experts::host_records::{Claim, Grant, Located};
use crate::experts::landing::{Answer, Source};
use crate::experts::staging::Staged;
use seismic_io::Priority;
use magnitude_kernels::{SlotTable, constants};
use std::collections::BTreeSet;
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

/// How a request's absent expert lands: from the staging record keeping it
/// (revived, no read) into a tier record; already landed in a tier record;
/// read now into a staging record and landed as it ends; or once its read
/// ahead (a hint's or the tier's fill's, hastened to demand) has landed,
/// or, the tier holding no record past its grace or staging no free record
/// (one a failed step's read still holds), once one is (`Waiting`): a
/// shortage of records is backpressure, never a failure, and never a wait
/// of the request service. Only a read is on the request's critical
/// section.
enum Plan {
    Read { grant: Grant },
    Unread(Unread),
}

/// How an absent expert read by nothing now lands ([`Plan`]): kept by
/// staging record `record`, revived into a ready tier record once the read
/// is submitted, or waiting for one; kept already in tier record `tier`;
/// waiting for the landing in flight that reads it (`Reading`); or waiting
/// for a staging or tier record before it is read (`Waits`).
enum Unread {
    Staged { record: u32 },
    Kept(u32),
    Reading,
    Waits,
}

/// How an absent expert read by nothing now lands, its tier record decided
/// ([`Unread`]): revived from staging record `record` into the tier record
/// its claim's `grant` took, kept in `tier`, or waiting (`Waiting`) for its landing in flight
/// or, where `record`, for a record.
enum Landed {
    Revived { record: u32, grant: Grant },
    Kept(u32),
    Waits { record: bool },
}

/// An absent expert of a served request waiting for its read ahead's
/// landing, or for a record (`record`): its choices' answer, its rows, and
/// the request's point.
struct Waiting {
    at: At,
    expert: usize,
    answer: Answer,
    rows: BTreeSet<usize>,
    named: Instant,
    record: bool,
}

/// The decode requests served, and the landings waiting for a record.
pub(super) struct Decode {
    /// Per layer, the last request served.
    served: Vec<Step>,
    waiting: Vec<Waiting>,
}

impl Decode {
    /// The misses waiting, for the exchange's watchdog: each one's step,
    /// layer and expert, and since when its request named it.
    pub(super) fn describe(&self) -> String {
        let waiting = self
            .waiting
            .iter()
            .map(|waiting| format!("step {} layer {} expert {} for {:?}", waiting.at.point.sequence, waiting.at.point.layer, waiting.expert, waiting.named.elapsed()))
            .collect::<Vec<_>>();
        format!("served through {:?}, misses waiting [{}]", self.served, waiting.join("; "))
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

/// Claim a demand landing of `staged`, the device parked at `parked` in a
/// group of `steps` steps (`Directory::claim_demand`): kept or landing
/// already, that; else a ready tier record of its group, one the reserve
/// holds ready (its graces released at the park, `Directory::release`), or
/// one a kept record's unmapping frees at once, the unmapping stored (by the
/// host records, its one writer) before any landing writes the record;
/// never one a resolution at or after the park may name ([`parked_reads`]),
/// nor a request's outstanding answer (`HostRecords::answer`). The record
/// taken answers the parked request once it lands.
fn demand_claim(core: &mut Core, staged: Staged, (parked, steps): (crate::experts::directory::Point, u64)) -> Claim {
    let read = parked_reads(core, staged.0, parked.sequence);
    core.directory.release((parked, steps), &read);
    core.directory.claim_demand(staged, &read, (parked, steps))
}

/// The landing `grant` allows of a parked step's absent expert from staging
/// record `record`, the device parked at `parked` in a group of `steps`
/// steps: mapped from the first step that has not resolved its layer
/// (`Point::first_unresolved`), which the device cannot pass before the
/// landing answers, so a step begun behind the parked one that has yet to
/// resolve the layer hits it, as the host serving it decides.
fn parked_landing(
    core: &Core,
    grant: crate::experts::host_records::Grant,
    record: u32,
    parked: crate::experts::directory::Point,
    steps: u64,
) -> crate::experts::landing::Landing {
    let from = Some(parked.first_unresolved(grant.staged().0, steps));
    crate::experts::landing::Landing { from, ..core.landing(grant, record) }
}

impl Decode {
    pub(super) fn new(layers: usize) -> Self {
        Self { served: vec![0; layers], waiting: Vec::new() }
    }

    /// Each layer's last request noticed (`Self::request`).
    pub(super) fn served(&self) -> &[Step] {
        &self.served
    }

    /// Answer each waiting expert its tier keeps now (its read ahead
    /// landed), pass over each whose read ahead still runs, and land the
    /// rest a tier record is now past its grace for, in the order they
    /// waited: each read now and answered as it lands (or revived from
    /// staging); one of a step that ended (retired, or failed) is dropped,
    /// as its launches returned. True when any was answered, landed or
    /// dropped.
    pub(super) fn resume(&mut self, serving: Serving<'_>) -> bool {
        let Serving { core, prefetch, shared } = serving;
        let mut progressed = false;
        let mut waiting = std::mem::take(&mut self.waiting);
        waiting.retain(|waits| {
            let (layer, sequence) = (waits.at.point.layer, waits.at.point.sequence);
            let failed = core.layers[layer].file.answer_of(sequence, 0) == Some(crate::experts::exchange::Answer::Failed);
            if shared.steps.ended(sequence) || failed {
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
            // A record to read it into, unless staging keeps it: none free
            // (a failed step's read still holds one) waits for one, its
            // landing's end an event.
            if !prefetch.staging.keeps(staged) && prefetch.staging.free() == 0 {
                break;
            }
            let steps = shared.steps.group_of(waits.at.point.sequence).1;
            // Lacked, as located above: a claim takes a record, or finds
            // none ready.
            let Claim::Taken(grant) = demand_claim(core, staged, (waits.at.point, steps)) else {
                break;
            };
            let tier = grant.record();
            let waits = waiting.remove(index);
            progressed = true;
            if waits.record {
                crate::experts::landing::LandingAccount::add(&shared.landings.waited, waits.named.elapsed());
            }
            if let Some(record) = prefetch.staging.revive(staged) {
                let landing = parked_landing(core, grant, record, waits.at.point, steps);
                let revived = (staged, record, Some(waits.at.point.sequence));
                prefetch.revive(landing, revived, Priority::Demand, Some(waits.answer), shared);
                continue;
            }
            let directory = &core.directory;
            let record = prefetch
                .staging
                .take(|staged| directory.staged_worth(staged))
                .expect("a free staging record, looked at above");
            prefetch.read_at[record as usize] = Some(waits.at);
            let reads = core.reads(layer, expert, record);
            let count = reads.len();
            let landing = Named {
                landing: parked_landing(core, grant, record, waits.at.point, steps),
                staged: record,
                reads: 0..count,
                rows: waits.rows,
                answer: waits.answer,
            };
            let naming = Naming { layer, sequence: waits.at.point.sequence, named: waits.named };
            match core.source.submit(reads, Priority::Demand) {
                Ok(ticket) => {
                    for job in ticket.jobs() {
                        seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("demand"), job, (layer as i64, naming.sequence as i64, -1));
                    }
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("plan"), naming.sequence as u64, (layer as i64, expert as i64, 5));
                    prefetch.demand(ticket, vec![landing], vec![(record, staged, tier, naming.sequence)], naming, None, None, shared)
                }
                // The read was never made: as a failed read, only the rows that named the
                // expert fail, and its wait is answered with none, so the rest of the step serves.
                Err(reason) => {
                    core.directory.unlanded(layer, HostRecord(tier));
                    prefetch.staging.release(record, None);
                    shared.steps.fail_request_rows(naming.sequence, layer, &landing.rows, reason);
                    landing.answer.publish(None);
                }
            }
        }
        self.waiting = waiting;
        progressed
    }

    /// Layer `layer`'s earliest file request not yet served, taken as
    /// served, by its step: none while every slot's request is served, or
    /// of a step ended (read now: steps end beside the request service; one
    /// a device error ended, which no launch waits for). A layer's requests
    /// come in step order, the device's.
    pub(super) fn request(&mut self, core: &Core, shared: &Shared, layer: usize) -> Option<Step> {
        let file = &core.layers[layer].file;
        let live = shared.steps.live();
        let unserved = live.start.max(self.served[layer] + 1)..live.end;
        let sequence = file.earliest_requested(unserved, 0, |step| !shared.steps.ended(step))?;
        self.served[layer] = sequence;
        Some(sequence)
    }

    /// Serve the request `at` its point, noticed at `started`, the device
    /// past every point before it. Its critical section is what its read
    /// needs: which absent experts read now, a staging record and a ready
    /// tier record each, and the read's submission. What the read does not
    /// need follows it: the request's accounting, the tier map's
    /// publication, its landings' registration and the answers of the
    /// experts landed or landing already.
    pub(super) fn serve(&mut self, at: At, started: Instant, serving: Serving<'_>) {
        let Serving { core, prefetch, shared } = serving;
        let point = at.point;
        let (layer, sequence) = (point.layer, point.sequence);
        // No wait is encoded where the word is set, so a request there would
        // leave its choices unsourced: an invariant violation, which ends
        // serving (`Requests::serve`).
        assert_eq!(
            core.layers[layer].policy.load(constants::POLICY_RESOLVED as usize),
            0,
            "step {sequence} requested the file at layer {layer}, whose experts cannot be absent"
        );
        let steps = shared.steps.group_of(sequence).1;
        let experts = core.layers[layer].experts;
        let width = core.extents.words.choices;
        let file = core.layers[layer].file.clone();
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
        seismic_io::timeline::note_at(started, seismic_io::timeline::Kind::Noted("notice"), sequence as u64, (layer as i64, absent.len() as i64, count as i64));
        // Each absent expert's landing: one staging keeps (revived), a read
        // ahead's, hastened to demand (reading), or one read now. The
        // request's lead is the layers since its last read a route issued
        // was submitted: none where every miss had landed already, a read it
        // never saw. A fill's read no route issued is a demand's to it: no
        // prediction made it.
        let mut served = RequestCost { missed: absent.clone(), ..RequestCost::default() };
        // Each absent expert's plan, none yet for one read now.
        let mut plans = Vec::with_capacity(absent.len());
        let mut fresh = 0;
        let mut lead: Option<u64> = None;
        let record_bytes = core.layers[layer].expert_bytes();
        for &expert in &absent {
            let staged = (layer, expert);
            // The request's lead is its latest read's.
            let mut taken_from = |from: At| {
                let layers = from.runs_until(at);
                lead = Some(lead.map_or(layers, |lead| lead.min(layers)));
            };
            let (plan, anticipation) = match core.directory.located(layer, expert) {
                // Kept already: landed after the route resolved it (a read
                // ahead's that ended and settled since), answered with its
                // record, an outstanding answer from now, which no claim of
                // the request's other misses takes.
                Located::Kept(tier) => {
                    core.directory.answer(layer, HostRecord(tier), point);
                    (Some(Plan::Unread(Unread::Kept(tier))), None)
                }
                // A landing of it is in flight, whoever's (a read ahead's,
                // another request's, the fill's from staging): it waits for
                // it, hastened to demand.
                Located::Landing => {
                    match prefetch.hasten(staged) {
                        Some(Reading::Hint(issued)) => {
                            taken_from(issued);
                            (Some(Plan::Unread(Unread::Reading)), Some(Anticipation::Reading))
                        }
                        Some(Reading::Unpredicted) | None => (Some(Plan::Unread(Unread::Reading)), Some(Anticipation::Demand)),
                    }
                }
                Located::Lacked => match prefetch.staging.revive(staged) {
                    Some(record) => {
                        if let Some(issued) = prefetch.read_at[record as usize] {
                            taken_from(issued);
                        }
                        (Some(Plan::Unread(Unread::Staged { record })), Some(Anticipation::Staged))
                    }
                    None => {
                        fresh += 1;
                        (None, Some(Anticipation::Demand))
                    }
                },
            };
            let code = match (&plan, anticipation) {
                (Some(Plan::Unread(Unread::Kept(_))), _) => 0,
                (_, Some(Anticipation::Reading)) => 1,
                (Some(Plan::Unread(Unread::Reading)), _) => 2,
                (_, Some(Anticipation::Staged)) => 3,
                _ => 4,
            };
            seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("plan"), sequence as u64, (layer as i64, expert as i64, code));
            // A miss kept already reads nothing: no read miss, nor its bytes.
            match anticipation {
                Some(anticipation) => served.anticipated[anticipation as usize] += 1,
                None => served.missed.retain(|&missed| missed != expert),
            }
            plans.push(plan);
        }
        // Its own hints hastened, the device is at this layer: the hints of
        // layers it has begun or passed read nothing more.
        prefetch.withdraw_passed(core, &shared.steps, Some((sequence, layer)));
        // Those read now, each into a staging record free once every revival
        // took its own: an earlier request's are free once its landings end,
        // which the device past its answers proves (`Residency::advance`
        // learned them), but one a failed step left reading frees only as
        // its read and landing end. An expert beyond them waits for one
        // (`Self::resume`), as one the tier has no ready record for.
        let mut free = prefetch.staging.free();
        let plans = absent
            .iter()
            .zip(plans)
            .map(|(&expert, plan)| {
                // Lacked, as located above: a claim takes a record, or
                // finds none ready.
                plan.unwrap_or_else(|| match (free > 0).then(|| demand_claim(core, (layer, expert), (point, steps))) {
                    Some(Claim::Taken(grant)) => {
                        free -= 1;
                        Plan::Read { grant }
                    }
                    _ => Plan::Unread(Unread::Waits),
                })
            })
            .collect::<Vec<_>>();
        // The device's convert holds its staging until the device is past
        // this barrier, which opens only once every miss is answered, so a
        // miss waiting for a record rules it out: the host's landings free
        // their records without the device.
        let convertible = !plans.iter().any(|plan| matches!(plan, Plan::Unread(Unread::Waits)));
        // The request's one demand job, its reads landing in staging
        // records taken now: a point of the served file's read line where
        // it reads all the request's misses. It is submitted before
        // anything it does not need.
        let reading = plans.iter().filter(|plan| matches!(plan, Plan::Read { .. })).count();
        let directory = &core.directory;
        let mut taken = prefetch
            .staging
            .take_many(reading, |staged| directory.staged_worth(staged))
            .expect("a free staging record for each read, counted above")
            .into_iter();
        let (mut reads, mut landings, mut tracked, mut others) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (&expert, plan) in absent.iter().zip(plans) {
            match plan {
                Plan::Read { grant } => {
                    let tier = grant.record();
                    let record = taken.next().expect("a staging record a read");
                    prefetch.read_at[record as usize] = Some(at);
                    let begun = reads.len();
                    reads.extend(core.reads(layer, expert, record));
                    landings.push(Named {
                        landing: parked_landing(core, grant, record, point, steps),
                        staged: record,
                        reads: begun..reads.len(),
                        rows: rows_of(expert),
                        answer: answer_of(expert),
                    });
                    tracked.push((record, (layer, expert), tier, sequence));
                }
                Plan::Unread(unread) => others.push((expert, unread)),
            }
        }
        let submitted = (!reads.is_empty()).then(|| (Instant::now(), core.source.submit(reads, Priority::Demand)));
        if let Some((_, Ok(ticket))) = &submitted {
            for job in ticket.jobs() {
                seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("demand"), job, (layer as i64, sequence as i64, reading as i64));
            }
        }
        if fresh > 0 {
            lead = Some(0);
        }
        served.lead = lead;
        served.misses = served.missed.len() as u64;
        served.bytes = served.misses * record_bytes;
        served.parked = started.elapsed();
        // Counted before any answer, which releases the device to complete
        // the step: no landing is registered yet, nor answer published.
        shared.telemetry.serve(layer, sequence, served);
        // A staged expert's tier record, after the reads': it reads nothing.
        let others = others
            .into_iter()
            .map(|(expert, unread)| match unread {
                // Lacked, as located above: a claim takes a record, or
                // finds none ready.
                Unread::Staged { record } => match demand_claim(core, (layer, expert), (point, steps)) {
                    Claim::Taken(grant) => (expert, Landed::Revived { record, grant }),
                    _ => {
                        prefetch.staging.release(record, Some((layer, expert)));
                        (expert, Landed::Waits { record: true })
                    }
                },
                Unread::Kept(tier) => (expert, Landed::Kept(tier)),
                Unread::Reading => (expert, Landed::Waits { record: false }),
                Unread::Waits => (expert, Landed::Waits { record: true }),
            })
            .collect::<Vec<_>>();
        // The tier map's unmappings, a ready record's reclaim among them,
        // were stored as each claim took its record, before any landing
        // starts: no step after the parked one resolves an unmapped expert
        // to a record a landing writes, however soon the landing answers.
        match submitted {
            Some((at, Ok(ticket))) => {
                let timed = (reading == absent.len()).then_some((reading as u64 * record_bytes, at));
                // Where a request's misses land is priced per request: the
                // device's convert past its barrier, or the host's landing.
                let device = core.layers[layer]
                    .landing
                    .clone()
                    .filter(|_| convertible && shared.telemetry.device_lands(layer, reading as u64))
                    .map(|words| (words, core.layers[layer].file.clone(), point));
                prefetch.demand(ticket, landings, tracked, naming, timed, device, shared);
            }
            // The reads were never made: as failed reads, only the rows that named their
            // experts fail, and their waits are answered with none, so the rest of the step
            // serves.
            Some((_, Err(reason))) => {
                for (record, (layer, _), tier, _) in tracked {
                    core.directory.unlanded(layer, HostRecord(tier));
                    prefetch.staging.release(record, None);
                }
                for landing in landings {
                    shared.steps.fail_request_rows(sequence, layer, &landing.rows, reason.clone());
                    landing.answer.publish(None);
                }
            }
            None => {}
        }
        for (expert, landed) in others {
            let record = match landed {
                Landed::Revived { record, grant } => {
                    let landing = parked_landing(core, grant, record, point, steps);
                    let revived = ((layer, expert), record, Some(sequence));
                    prefetch.revive(landing, revived, Priority::Demand, Some(answer_of(expert)), shared);
                    continue;
                }
                Landed::Kept(tier) => {
                    answer_of(expert).publish(Some(tier));
                    continue;
                }
                // Its landing in flight, or a record, answers it as the
                // watcher resumes it (`Self::resume`).
                Landed::Waits { record } => record,
            };
            if record {
                shared.landings.waits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            self.waiting.push(Waiting { at, expert, answer: answer_of(expert), rows: rows_of(expert), named: started, record });
        }
    }
}
