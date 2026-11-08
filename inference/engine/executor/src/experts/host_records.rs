//! One bank group's host tier records: copies of the group's experts in the
//! bank's own resident form, which the directory owns beside the bank's.
//! Every miss reaches the tier first: the host lands each file miss into a
//! record (`experts::landing`), and a step serves a miss from a record while
//! the layer's tier map names it (`routed.seismic`'s `resolve`); the host
//! promotes the records the ranking keeps into the bank
//! (`watcher::promotions`). A record is mapped only while its bytes are its
//! expert's whole.
//!
//! Each record is in one state ([`Host`]), changed only by
//! [`HostRecords::set`], which keeps every index of them (the ready and
//! graced records, the record keeping each expert, the landings, the
//! lacked experts, the experts held alone) by construction; and every map
//! change of the group's layers is one of its transitions, the records
//! owning the layers' landing maps (`experts::map::LandingMap`). A landing
//! maps its record where it ends, off residency's lock, through the
//! [`Grant`] its claim took: the one map write the claim allows. The states:
//! - kept: it holds an expert whole, mapped, alone (the bank lacks it) or
//!   banked (the bank holds a copy too);
//! - graced: unmapped while step `q` of group `g` was the latest begun, so
//!   layer `l`'s entries of a step at or before it may still read it (only
//!   layer `l`'s route resolves its tier map, and only layer `l`'s entries
//!   and its CPU site read what it resolved): it holds nothing once the
//!   device is past layer `l` of `q`, the point `(g, l, q)`
//!   ([`HostRecords::retire`]), or at a park that no resolution after it
//!   names it ([`HostRecords::release`]). That is the one rule a record
//!   frees by: once the device is past the last resolution that may name
//!   it, which the map's [`Resolution`] bounds;
//! - ready: holding nothing any step reads, so a landing takes it at once;
//! - landing: taken for a landing of one expert in flight, until it lands
//!   ([`HostRecords::landed`]) or its read fails ([`HostRecords::unlanded`]);
//! - claimed: a prompt group's for its life (the lender's lent records: its
//!   conveyor's, and on unified memory its scratch's too,
//!   `planning::ScratchPlan`), holding nothing a decode step reads, until
//!   it returns them empty ([`HostRecords::restore_lent`]). A claim takes a
//!   record no step that may still resolve it can name: the prompt group
//!   runs after every step begun before it, a claim unmaps a kept record
//!   from the group's first step on, and the group's own steps read the map
//!   only at a pass; every lent record is claimed before the group takes a
//!   hit, so none holds one. The reserve unmaps a lent record only while
//!   no prompt group holds or awaits them, decided where no group begins,
//!   so none is graced past the group it serves; and counts those a group
//!   holds toward itself ([`HostRecords::refill`]).
//!
//! Every expert of the group's layers is kept, landing, or lacked; the
//! lacked ones landed in order of their reuse ([`HostRecords::lacking`]),
//! the order the tier's fill lands them in.
//!
//! The tier is inclusive of the bank: a promotion copies a record into the
//! bank and the record keeps its copy, so an expert the bank then evicts is
//! held in the tier alone again with no copy, and nothing is ever written
//! back. The host keeps a reserve of ready records ([`HostRecords::refill`]),
//! so a demand landing never waits for a grace: it unmaps the kept records
//! first in taking order, the experts held alone, then the banked copies,
//! whose experts the bank serves, each the least reused first. Every order
//! is the one ranking's (`Reuse`), which each call that orders is handed
//! (`rank`): the tier keeps no rank of its own, so none goes stale. The tier
//! grows and shrinks by whole records (`ExpertBankPlan::host_records`).
//!
//! A kept record a copy reads (a prompt group's pull, or a promotion) is
//! held until the copy ends ([`HostRecords::hold`]): it keeps its expert,
//! mapped, and its place in the taking order, and the reserve never unmaps
//! it while held, so its bytes stay whole for the copy.

use super::directory::Point;
use super::map::LandingMap;
use super::staging::Staged;
use super::watcher::{Step, Unbegun};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The read path's lock-free mirror of where each expert of the bank's
/// layers is, one word an expert: set while a record of its group keeps or
/// lands it, or a read of it is in flight that no record has taken yet (a
/// request's read the request service's look posted, `watcher::decode`).
/// Whoever starts a read of a lacked expert sets it, by one compare and
/// swap ([`Self::claim`]): the look, or a claim of the host records
/// ([`HostRecords::claim`] and its kin), so no two reads of an expert are
/// ever in flight at once. Only the host records clear it, as the expert
/// becomes lacked again ([`HostRecords::set`], [`HostRecords::unposted`]).
pub(in crate::experts) struct InboundTable {
    /// Each layer's first word.
    layers: Vec<usize>,
    words: Box<[AtomicBool]>,
}

impl InboundTable {
    /// Every expert lacked, of layers of `experts` experts each.
    pub fn new(experts: &[usize]) -> Self {
        let layers = experts.iter().scan(0, |next, &count| {
            let first = *next;
            *next += count;
            Some(first)
        });
        let layers = layers.collect::<Vec<_>>();
        let words = (0..experts.iter().sum()).map(|_| AtomicBool::new(false)).collect();
        Self { layers, words }
    }

    fn word(&self, (layer, expert): Staged) -> &AtomicBool {
        &self.words[self.layers[layer] + expert]
    }

    /// Claim lacked `staged` for a read about to start: whether this call
    /// claimed it (none has, nor does a record keep or land it).
    pub fn claim(&self, staged: Staged) -> bool {
        self.word(staged).compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    /// Whether `staged` is kept, landing, or claimed by a read in flight.
    pub fn claimed(&self, staged: Staged) -> bool {
        self.word(staged).load(Ordering::Acquire)
    }

    /// `staged` is lacked again: by its host records alone.
    fn release(&self, staged: Staged) {
        assert!(self.word(staged).swap(false, Ordering::AcqRel), "{staged:?} released, which nothing claimed");
    }
}

/// Whose claim of an expert a demand's claim takes a record for
/// ([`HostRecords::claim_demand`]): a lacked expert's, which the claim
/// makes, or the read the request service's look posted, which made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Demand {
    Lacked,
    Posted,
}

/// What a kept record holds, which orders it for taking: an expert the
/// tier holds alone, or a copy of one the bank holds too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kept {
    Alone,
    Banked,
}

/// How the steps that resolve the records' map see an unmapping, which
/// bounds the last resolution that may name an unmapped record: a plain
/// word (a host tier's map beside a discrete device), which every
/// resolution after the store reads, so none after now names it; or a
/// step-stamped entry (the bank's own map on unified memory,
/// `experts::map`), held for every step before its stamp, so the steps
/// before the stamp may name it, at its layer, however late they resolve,
/// until a park restamps it ([`HostRecords::release`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Resolution {
    AfterStore,
    BeforeStamp,
}

impl Resolution {
    /// The last point that may name a record unmapped through `grace` (its
    /// layer of the last step that may resolve its stamp), the device now
    /// at `now`.
    fn bound(self, grace: Point, now: Point) -> Point {
        match self {
            Self::AfterStore => grace.min(now),
            Self::BeforeStamp => grace,
        }
    }
}

/// Where an expert of the group is: kept whole in a record, landing (a
/// landing of it is in flight, whoever's), or lacked. Every expert is
/// exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Located {
    Kept(u32),
    Landing,
    Lacked,
}

/// The one map write a landing's claim allows: once its bytes are whole,
/// expert `staged` mapped to the record taken for it ([`Grant::map`]).
/// Only a claim makes one ([`HostRecords::claim`]).
pub(in crate::experts) struct Grant {
    map: Option<LandingMap>,
    staged: Staged,
    record: u32,
    mapped: std::sync::atomic::AtomicBool,
}

impl Grant {
    /// The expert it lands, (layer, expert).
    pub fn staged(&self) -> Staged {
        self.staged
    }

    /// The tier record it lands into.
    pub fn record(&self) -> u32 {
        self.record
    }

    /// Map its expert to its record, whole now, from step `from` on where
    /// given, else from the first step not yet begun
    /// (`LandingMap::map_from`): once.
    pub fn map(&self, from: Option<Step>) {
        assert!(!self.mapped.swap(true, std::sync::atomic::Ordering::AcqRel), "a landing of {:?} maps its record twice", self.staged);
        if let Some(map) = &self.map {
            map.map_from(self.staged.1, self.record, from);
        }
    }

}

/// What a claim of an expert's landing found ([`HostRecords::claim`]),
/// the one way any landing of an expert takes a record: a record keeps it
/// whole, so nothing lands it; a landing of it is in flight, which lands
/// it; or it was lacked and the claim took a ready record for this
/// landing, with the grant of its map write, or found none it may take.
pub(super) enum Claim {
    Kept(u32),
    Landing,
    Taken(Grant),
    Short,
}

/// How the tier is ranked: an expert's rank beside a read the ranking may
/// not have brought yet (`Reuse::rank`), in an order `K` that breaks the
/// reads' ties as the reads before broke them, never by the expert's index.
pub(super) trait Rank<K: Ord + Copy>: Fn(Staged, Option<Step>) -> K {}

impl<K: Ord + Copy, F: Fn(Staged, Option<Step>) -> K> Rank<K> for F {}

/// An expert a record keeps whole, mapped: as `kept`, ranked beside
/// `read`, held by `holds` copies reading it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Whole {
    staged: Staged,
    kept: Kept,
    read: Option<Step>,
    holds: u32,
}

/// What a host record holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Host {
    /// Nothing any step reads: a landing takes it at once.
    Ready,
    /// Taken for a landing of the expert in flight; where a request's
    /// demand took it, the request at `answer` (its tier wait), which the
    /// landing answers with it.
    Landing { staged: Staged, answer: Option<Point> },
    /// Its expert whole, mapped.
    Kept(Whole),
    /// Its expert whole, mapped, and the outstanding answer of a request:
    /// the device reads it at the request's point or its latest one,
    /// `past`, until it is past there ([`HostRecords::retire`]). No claim
    /// takes it until then; the reserve may unmap it, graced with its
    /// answer.
    Answering(Whole, Point),
    /// Unmapped, the expert it held still named by the steps that may
    /// resolve it until the device is past `grace`, and read as the
    /// outstanding answer of the request at `answer`, if it was one, until
    /// the device is past there: no park before frees it.
    Graced { staged: Staged, grace: Point, answer: Option<Point> },
    /// A prompt group's, for its life.
    Claimed,
}

impl Host {
    /// The expert it keeps or lands: one no other record keeps or lands.
    fn present(self) -> Option<Staged> {
        match self {
            Self::Landing { staged, .. } | Self::Kept(Whole { staged, .. }) | Self::Answering(Whole { staged, .. }, _) => Some(staged),
            _ => None,
        }
    }

    /// What it keeps whole, kept or answering.
    fn whole(self) -> Option<Whole> {
        match self {
            Self::Kept(whole) | Self::Answering(whole, _) => Some(whole),
            _ => None,
        }
    }

    /// It, keeping `whole` instead, kept or answering as it is.
    fn with(self, whole: Whole) -> Self {
        match self {
            Self::Kept(_) => Self::Kept(whole),
            Self::Answering(_, past) => Self::Answering(whole, past),
            other => panic!("a host record holding {other:?} keeps nothing whole"),
        }
    }
}

/// A kept record's place in the taking order at `rank`: what it keeps, its
/// expert's rank beside its landing's `read`, and the expert (where `rank`
/// ties, never by where it lies).
fn taking<K: Ord + Copy>(kept: Kept, staged: Staged, read: Option<Step>, rank: &impl Rank<K>) -> (Kept, K, Staged) {
    (kept, rank(staged, read), staged)
}

pub(super) struct HostRecords {
    /// What each record holds, changed only by [`Self::set`].
    records: Vec<Host>,
    /// The indexes of `records`, each kept by [`Self::set`] alone: the
    /// ready records, in order (a landing takes the lowest past the lent
    /// ones, [`Self::take_lacked`]), the graced ones, the record keeping
    /// each expert, the record each landing in flight takes, the
    /// outstanding answers, the experts of the group's layers no record
    /// keeps or lands, per layer the experts held alone, and the records
    /// claimed.
    ready: BTreeSet<u32>,
    graced: Vec<u32>,
    answering: Vec<u32>,
    kept: HashMap<Staged, u32>,
    landing: HashMap<Staged, u32>,
    lacking: BTreeSet<Staged>,
    alone: Vec<usize>,
    claimed: usize,
    /// How the steps resolving the records' map see an unmapping.
    resolution: Resolution,
    /// The device is past every point before this ([`Self::passed`]), none
    /// before the first retirement: an answer of a request before it is
    /// consumed, so no record is its outstanding answer.
    progress: Option<Point>,
    /// The bottom records the conveyor's lending reserves for it (none but
    /// of its lender, `planning::Lender`), which no other claim takes.
    lent: u32,
    /// Each layer's landing map, of the group's own layers, whose one
    /// writer these records are; none without a device.
    maps: Vec<Option<LandingMap>>,
    /// Bumped as any layer's count of experts kept alone changes
    /// ([`Self::holdings`]).
    holdings: u64,
    /// A demand's victims, first in taking order as last prepared
    /// ([`Self::prepare_demand`]), each with the expert it kept then: a
    /// claim takes the first that still keeps it, unheld and not read
    /// ([`Self::claim_prepared`]), so no claim scans the records while a
    /// step waits.
    victims: VecDeque<(u32, Staged)>,
    /// The step the victims were prepared at, and how many.
    prepared: Option<(Step, usize)>,
    /// The kept records a prompt group's seeds displace, each with the
    /// expert it kept as the seed chose it ([`Self::displace`]): the
    /// reserve unmaps them before any other, so each seed takes a record of
    /// its own layer's.
    displaced: Vec<(u32, Staged)>,
    /// The read path's mirror of where every expert is, whose words of the
    /// group's layers these records alone clear.
    inbound: Arc<InboundTable>,
}

impl HostRecords {
    /// `records` records, every one ready, of a group whose layers are
    /// `members`, each (layer, its experts, its landing map), every expert
    /// lacked, of a bank of `layers` streamed layers, whose map steps
    /// resolve by `resolution`, mirrored in `inbound`.
    pub fn new(
        records: u32,
        layers: usize,
        members: Vec<(usize, usize, Option<LandingMap>)>,
        resolution: Resolution,
        lent: u32,
        inbound: Arc<InboundTable>,
    ) -> Self {
        let lacking = members.iter().flat_map(|&(layer, experts, _)| (0..experts).map(move |expert| (layer, expert))).collect();
        let mut maps = vec![None; layers];
        for (layer, _, map) in members {
            maps[layer] = map;
        }
        let mut host = Self {
            records: Vec::new(),
            ready: BTreeSet::new(),
            graced: Vec::new(),
            answering: Vec::new(),
            kept: HashMap::new(),
            landing: HashMap::new(),
            lacking,
            alone: vec![0; layers],
            claimed: 0,
            resolution,
            progress: None,
            lent,
            maps,
            holdings: 0,
            victims: VecDeque::new(),
            prepared: None,
            displaced: Vec::new(),
            inbound,
        };
        assert!(lent <= records, "a tier lends {lent} of its {records} records");
        host.grow(records);
        host
    }

    /// Record `record` holds `content` from now: the one change of a
    /// record, which keeps every index by construction. An expert entering
    /// a record leaves the lacked ones, which it must be one of.
    fn set(&mut self, record: u32, content: Host) {
        let old = std::mem::replace(&mut self.records[record as usize], content);
        // A copy held or let go, a read ranked: no index moves.
        if let (Host::Kept(was), Host::Kept(is)) | (Host::Answering(was, _), Host::Answering(is, _)) = (old, content) {
            if (was.staged, was.kept) == (is.staged, is.kept) {
                return;
            }
        }
        match old {
            Host::Ready => {
                assert!(self.ready.remove(&record), "a ready record is listed ready");
            }
            Host::Landing { staged, .. } => {
                self.landing.remove(&staged);
            }
            Host::Kept(Whole { staged, kept, .. }) | Host::Answering(Whole { staged, kept, .. }, _) => {
                self.kept.remove(&staged);
                if kept == Kept::Alone {
                    self.alone[staged.0] -= 1;
                    self.holdings += 1;
                }
                if matches!(old, Host::Answering(..)) {
                    self.answering.retain(|&answering| answering != record);
                }
            }
            Host::Graced { .. } => self.graced.retain(|&graced| graced != record),
            Host::Claimed => self.claimed -= 1,
        }
        match content {
            Host::Ready => assert!(self.ready.insert(record), "record {record} is listed ready once"),
            Host::Landing { staged, .. } => assert!(self.landing.insert(staged, record).is_none(), "{staged:?} lands twice"),
            Host::Kept(Whole { staged, kept, .. }) | Host::Answering(Whole { staged, kept, .. }, _) => {
                assert!(self.kept.insert(staged, record).is_none(), "{staged:?} kept twice");
                if kept == Kept::Alone {
                    self.alone[staged.0] += 1;
                    self.holdings += 1;
                }
                if matches!(content, Host::Answering(..)) {
                    self.answering.push(record);
                }
            }
            Host::Graced { .. } => self.graced.push(record),
            Host::Claimed => self.claimed += 1,
        }
        match (old.present(), content.present()) {
            (left, entered) if left == entered => {}
            (left, entered) => {
                if let Some(left) = left {
                    self.lacking.insert(left);
                    self.inbound.release(left);
                }
                // Whoever started its read claimed its word first.
                if let Some(entered) = entered {
                    assert!(self.lacking.remove(&entered), "{entered:?} is a lacked expert of the group's layers");
                    assert!(self.inbound.claimed(entered), "{entered:?} enters a record unclaimed");
                }
            }
        }
    }

    /// Move kept record `from`'s expert to ready record `to`, `from` ready
    /// then: the expert never leaves the tier, so its word stays claimed and
    /// no read, whoever's, can claim it while it moves. Its expert.
    ///
    /// The one change of two records that [`Self::set`] does not make, as
    /// `set` changes one record and releases the word of an expert leaving
    /// it. Of the indices `set` keeps, a move changes only those keyed by
    /// record: `ready` (`to` leaves it, `from` joins), `kept` (the expert's
    /// record) and `answering`. Those keyed by expert or counted by group
    /// stay as they are, the same expert kept alone or as a copy: `alone`,
    /// `holdings`, `lacking` and `landing`; neither record is graced or
    /// claimed (`graced`, `claimed`).
    fn relocate(&mut self, from: u32, to: u32) -> Staged {
        let moved = self.records[from as usize];
        let staged = moved.present().expect("a kept record keeps its expert");
        assert!(matches!(moved, Host::Kept(_) | Host::Answering(..)), "record {from} moves only kept");
        assert!(self.ready.remove(&to) && self.records[to as usize] == Host::Ready, "record {from} moves to a ready record, not {to}");
        self.records[to as usize] = moved;
        self.records[from as usize] = Host::Ready;
        self.ready.insert(from);
        self.kept.insert(staged, to);
        for answering in &mut self.answering {
            if *answering == from {
                *answering = to;
            }
        }
        staged
    }

    /// Map `staged`'s entry to `record`, or unmap it (none), from the first
    /// step not yet begun (`LandingMap::store`).
    fn store(&self, (layer, expert): Staged, record: Option<u32>) {
        if let Some(map) = &self.maps[layer] {
            map.store(expert, record);
        }
    }

    /// Unmap `staged` from its record `record` from step `from` on
    /// (`LandingMap::unmap_from`).
    fn unmap_from(&self, (layer, expert): Staged, record: u32, from: Step) {
        if let Some(map) = &self.maps[layer] {
            map.unmap_from(expert, record, from);
        }
    }

    /// The grant of mapping `staged` to `record` once landed.
    fn grant(&self, staged: Staged, record: u32) -> Grant {
        Grant { map: self.maps[staged.0].clone(), staged, record, mapped: std::sync::atomic::AtomicBool::new(false) }
    }

    /// Its records.
    pub fn records(&self) -> u32 {
        self.records.len() as u32
    }

    /// Its records ready for a landing.
    #[cfg(test)]
    pub fn ready(&self) -> usize {
        self.ready.len()
    }

    /// Its records unmapped and not yet past every resolution that may name
    /// them.
    #[cfg(test)]
    pub fn graced(&self) -> usize {
        self.graced.len()
    }

    /// Whether a graced record keeps `staged`: a step begun before it was
    /// unmapped may still resolve it there (a running prompt group's steps
    /// plan their hits on it as each opens), so no landing off a park
    /// takes the expert until the grace retires; one would map it anew and
    /// leave those resolutions unmapped.
    pub fn graces(&self, staged: Staged) -> bool {
        self.graced_experts().any(|graced| graced == staged)
    }

    /// The experts graced records keep ([`Self::graces`]).
    fn graced_experts(&self) -> impl Iterator<Item = Staged> + '_ {
        self.graced.iter().map(|&record| match self.records[record as usize] {
            Host::Graced { staged, .. } => staged,
            other => panic!("graced record {record} is {other:?}"),
        })
    }

    /// The record keeping `staged` whole, if one does.
    pub fn kept_record(&self, staged: Staged) -> Option<u32> {
        self.kept.get(&staged).copied()
    }

    /// Whether a record keeps `staged` whole, alone or banked.
    pub fn keeps(&self, staged: Staged) -> bool {
        self.kept.contains_key(&staged)
    }

    /// Whether a landing of `staged` is in flight: a record taken for it,
    /// or a posted read no record has taken yet.
    pub fn lands(&self, staged: Staged) -> bool {
        self.located(staged) == Located::Landing
    }

    /// Where `staged` is: kept, landing (a record taken for its landing, or
    /// its word claimed by a posted read, [`InboundTable`]) or lacked.
    pub fn located(&self, staged: Staged) -> Located {
        match self.kept.get(&staged) {
            Some(&record) => Located::Kept(record),
            None if self.inbound.claimed(staged) => Located::Landing,
            None => Located::Lacked,
        }
    }

    /// The posted read of `staged` that no record took ended unlanded, or
    /// was never made: lacked again.
    pub fn unposted(&mut self, staged: Staged) {
        assert!(!self.landing.contains_key(&staged) && !self.kept.contains_key(&staged), "{staged:?} unposted, which a record took");
        self.inbound.release(staged);
    }

    /// Whether the tier holds `staged` alone.
    #[cfg(test)]
    pub fn holds_alone(&self, staged: Staged) -> bool {
        self.kept.get(&staged).is_some_and(|&record| self.records[record as usize].whole().is_some_and(|whole| whole.kept == Kept::Alone))
    }

    /// Per layer, the experts the tier holds alone.
    pub fn alone(&self) -> &[usize] {
        &self.alone
    }

    /// The count of changes to what its layers keep alone: equal counts,
    /// equal counts kept alone.
    pub fn holdings(&self) -> u64 {
        self.holdings
    }

    /// Commit records up to `records`, the new ones ready.
    pub fn grow(&mut self, records: u32) {
        let from = self.records();
        assert!(records >= from, "a host tier of {from} records grows to {records}");
        self.records.resize(records as usize, Host::Ready);
        self.ready.extend(from..records);
    }

    /// Kept record `record`'s place in the taking order at `rank`; none for
    /// a record that keeps nothing, or is an outstanding answer: only a
    /// kept record is ever taken.
    fn place<K: Ord + Copy>(&self, record: u32, rank: &impl Rank<K>) -> Option<(Kept, K, Staged)> {
        match self.records[record as usize] {
            Host::Kept(Whole { staged, kept, read, .. }) => Some(taking(kept, staged, read, rank)),
            _ => None,
        }
    }

    /// Whether record `record` still keeps `staged` whole, no copy holding
    /// it and no answer outstanding: as a seed found it when it chose to
    /// displace it.
    fn still_keeps(&self, record: u32, staged: Staged) -> bool {
        matches!(self.records[record as usize], Host::Kept(Whole { staged: kept, holds: 0, .. }) if kept == staged)
    }

    /// The records seeds displaced that still keep what they kept: owed
    /// back to the reserve, each in place of the seed that took a ready
    /// record ([`Self::refill_victims`]).
    pub fn owed(&self) -> Vec<(u32, Staged)> {
        self.displaced.iter().copied().filter(|&(record, staged)| self.still_keeps(record, staged)).collect()
    }

    /// The kept records of `layer`'s experts a prompt group's seed of the
    /// layer may displace, the least ranked first at `rank`, each with its
    /// place in the taking order: past the lent records, no copy holding
    /// it, no answer outstanding, none already displaced.
    pub fn displaceable<K: Ord + Copy>(&self, layer: usize, rank: impl Rank<K>) -> Vec<((Kept, K, Staged), u32)> {
        let mut order = (self.lent..self.records())
            .filter(|&record| {
                matches!(self.records[record as usize], Host::Kept(Whole { staged: (of, _), holds: 0, .. }) if of == layer)
                    && !self.displaced.iter().any(|&(displaced, _)| displaced == record)
            })
            .filter_map(|record| Some((self.place(record, &rank)?, record)))
            .collect::<Vec<_>>();
        order.sort_unstable();
        order
    }

    /// A prompt group's seed displaces kept record `record`, which keeps
    /// `staged` ([`Self::displaceable`]): the reserve unmaps it before any
    /// other ([`Self::refill_victims`]), unless it changed meanwhile.
    pub fn displace(&mut self, record: u32, staged: Staged) {
        assert!(self.still_keeps(record, staged), "a seed displaces record {record}, which no longer keeps {staged:?} alone");
        self.displaced.push((record, staged));
    }

    /// Whether no copy holds record `record`.
    fn unheld(&self, record: u32) -> bool {
        self.records[record as usize].whole().is_none_or(|whole| whole.holds == 0)
    }

    /// Kept record `record` gives up its expert, whose map entry its caller
    /// changes: the expert. Never an outstanding answer, which the device
    /// still reads, nor one a copy holds.
    fn unkeep(&self, record: u32) -> Staged {
        match self.records[record as usize] {
            Host::Kept(Whole { staged, holds: 0, .. }) => staged,
            other => panic!("host record {record} gives up its expert from {other:?}"),
        }
    }

    /// The kept records `admits`, first in taking order at `rank`, the
    /// outstanding answers among them where `answers`.
    fn taking_order<K: Ord + Copy>(&self, rank: &impl Rank<K>, answers: bool, admits: impl Fn(u32) -> bool) -> Vec<u32> {
        let place = |record: u32| match self.records[record as usize] {
            Host::Answering(Whole { staged, kept, read, .. }, _) if answers => Some(taking(kept, staged, read, rank)),
            _ => self.place(record, rank),
        };
        let mut order = (0..self.records())
            .filter(|&record| admits(record))
            .filter_map(|record| Some((place(record)?, record)))
            .collect::<Vec<_>>();
        order.sort_unstable();
        order.into_iter().map(|(_, record)| record).collect()
    }

    /// Hold the kept record keeping `staged` whole for a copy reading it (a
    /// pull's or a promotion's): the record, none where no record keeps it.
    /// Until every copy holding it ends ([`Self::unhold`]) it keeps what it
    /// keeps, mapped, ranks as it does, and the reserve never unmaps it.
    pub fn hold(&mut self, staged: Staged) -> Option<u32> {
        let &record = self.kept.get(&staged)?;
        let content = self.records[record as usize];
        let whole = content.whole().expect("a record keeping an expert keeps it whole");
        self.set(record, content.with(Whole { holds: whole.holds + 1, ..whole }));
        Some(record)
    }

    /// A copy that held `record` ended: once no copy holds it, the reserve
    /// may unmap it again.
    pub fn unhold(&mut self, record: u32) {
        let content = self.records[record as usize];
        let whole = content.whole().unwrap_or_else(|| panic!("host record {record} released, which keeps nothing"));
        let holds = whole.holds.checked_sub(1).expect("a host record released that no copy holds");
        self.set(record, content.with(Whole { holds, ..whole }));
    }

    /// Kept record `record` is the answer of the request at `past` (its
    /// tier wait), published now: an outstanding answer until the device is
    /// past there ([`Self::retire`]), the latest of its requests'.
    pub fn answer(&mut self, record: u32, past: Point) {
        let content = self.records[record as usize];
        let (whole, past) = match content {
            Host::Kept(whole) => (whole, past),
            Host::Answering(whole, answering) => (whole, answering.max(past)),
            other => panic!("host record {record} answers a request, holding {other:?}"),
        };
        self.set(record, self.answering(whole, Some(past)));
    }

    /// Record content `whole`, the answer of the request at `past` if it
    /// is one: its outstanding answer while the device may still read it,
    /// else kept. The one place a record becomes an outstanding answer, so
    /// none is one of a request the device is known past, whichever is
    /// learned first: the device passing it ([`Self::passed`]) or its
    /// landing ending ([`Self::landed`]). A step whose wait expired ends
    /// past its request while the request's landing is in flight; on an
    /// idle load that landing's end is the last event, so it must not
    /// leave its record in flight, where a shrink would wait on it.
    fn answering(&self, whole: Whole, past: Option<Point>) -> Host {
        match past.filter(|&past| self.progress.is_none_or(|progress| past >= progress)) {
            Some(past) => Host::Answering(whole, past),
            None => Host::Kept(whole),
        }
    }

    /// The group's experts no record keeps and no landing is taken for:
    /// the most a demand could still land at once.
    pub fn lacked(&self) -> usize {
        self.lacking.len()
    }

    /// The first `count` experts the tier lacks that `skip` does not pass
    /// over, in the order its fill lands them: by `order` of each one's
    /// reuse at `rank` (the directory's one fill order,
    /// `Directory::fill_order`), with its key.
    pub fn lacking<R: Ord + Copy, K: Ord>(&self, rank: impl Rank<R>, count: usize, skip: impl Fn(Staged) -> bool, order: impl Fn(R, Staged) -> K) -> Vec<(K, Staged)> {
        let graced = self.graced_experts().collect::<std::collections::HashSet<_>>();
        let mut lacked = self
            .lacking
            .iter()
            .filter(|&&staged| !skip(staged) && !graced.contains(&staged))
            .map(|&staged| (order(rank(staged, None), staged), staged))
            .collect::<Vec<_>>();
        if count < lacked.len() {
            lacked.select_nth_unstable_by(count, |a, b| a.0.cmp(&b.0));
            lacked.truncate(count);
        }
        lacked.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        lacked
    }

    /// Whether every expert of the group is kept: no miss of it reads the
    /// file until a record gives one up.
    pub fn complete(&self) -> bool {
        self.lacking.is_empty() && self.landing.is_empty()
    }

    /// Whether every expert of the group is kept where no flight takes it,
    /// so no route mid-flight finds one absent: complete, with no expert
    /// kept in a lent record, which a prompt group's claim unmaps mid-flight
    /// ([`Self::claim_record`]).
    pub fn settled(&self) -> bool {
        self.complete() && self.records[..self.lent as usize].iter().all(|record| record.present().is_none())
    }

    /// Claim a landing of `staged`: where a record keeps it or a landing of
    /// it is in flight, that; else, the tier lacking it, a ready record
    /// taken for this landing, leaving ready at least `keep` (the demand
    /// reserve, which a prefetch or a fill leaves), or the experts still
    /// lacked after it where those are fewer (a landing beyond them could
    /// never be demanded); else short. Every landing of an expert, whoever
    /// lands it, claims it here, so no two land it at once.
    pub fn claim(&mut self, staged: Staged, keep: usize) -> Claim {
        match self.located(staged) {
            Located::Kept(record) => Claim::Kept(record),
            Located::Landing => Claim::Landing,
            Located::Lacked if self.takes(keep) => match self.inbound.claim(staged) {
                true => Claim::Taken(self.take_lacked(staged, None)),
                // A posted read claimed it since it was located.
                false => Claim::Landing,
            },
            Located::Lacked => Claim::Short,
        }
    }

    /// Take a ready record for a landing of `staged`, which the tier
    /// lacks and whose word its caller claimed, answering the request at `answer` where a demand takes it:
    /// its grant. Every landing takes its record here, the lowest ready one
    /// past the lent records (a lent one only where none past them is
    /// ready, as a prompt group's claim waits for a landing into it). A
    /// shrink keeps a landing in place and cuts only above it
    /// ([`Self::shrink_floor`]), and from the top down; so a landing lies
    /// past a cut only where every record past the lent ones below it was
    /// taken as it began: right after a growth a landing fills the records
    /// below the growth before it touches the new ones a shrink cuts first.
    fn take_lacked(&mut self, staged: Staged, answer: Option<Point>) -> Grant {
        let record = self.ready.range(self.lent..).next().or_else(|| self.ready.first()).copied().expect("a record taken is ready");
        self.set(record, Host::Landing { staged, answer });
        self.grant(staged, record)
    }

    /// Whether [`Self::claim`] of `keep` takes a record now.
    pub fn takes(&self, keep: usize) -> bool {
        self.lacked().checked_sub(1).is_some_and(|after| self.ready.len() > keep.min(after))
    }

    /// Landing `record` holds `staged` whole, and its landing mapped it
    /// ([`Grant::map`]): kept as `kept` (a copy of an expert the bank
    /// holds is banked, so the tier never holds alone what the bank holds),
    /// ranked beside `read`, the step whose miss it landed for (none for a
    /// fill's); the outstanding answer of the request whose demand took it,
    /// if one did.
    pub fn landed(&mut self, record: u32, staged: Staged, read: Option<Step>, kept: Kept) {
        let Host::Landing { staged: landing, answer } = self.records[record as usize] else {
            panic!("{staged:?} lands into record {record}, which no landing took");
        };
        assert_eq!(landing, staged, "{staged:?} lands into the record taken for it");
        let whole = Whole { staged, kept, read, holds: 0 };
        self.set(record, self.answering(whole, answer));
    }

    /// Ready record `record` already holds `staged` whole (a bank imported
    /// whole into the one store, each expert at its home, before any step):
    /// kept, as a landing of it, mapped from the first step not yet begun.
    pub fn keep(&mut self, record: u32, staged: Staged) {
        assert_eq!(self.records[record as usize], Host::Ready, "a kept home is ready");
        assert!(self.inbound.claim(staged), "a home of {staged:?} is kept once, before any read");
        self.set(record, Host::Kept(Whole { staged, kept: Kept::Alone, read: None, holds: 0 }));
        self.store(staged, Some(record));
    }

    /// Landing `record`'s read failed: it mapped nothing, and is ready
    /// again; its expert is lacked again.
    pub fn unlanded(&mut self, record: u32) {
        assert!(matches!(self.records[record as usize], Host::Landing { .. }), "a landing in flight into record {record}");
        self.set(record, Host::Ready);
    }

    /// The kept records the reserve unmaps now ([`Self::refill`]): every
    /// record a seed displaced ([`Self::displace`]) that still keeps what it
    /// kept, however full the reserve, since the seed's landing took its
    /// place ([`Self::owed`]); then, of those not held, first in taking
    /// order at `rank`,
    /// until the ready, the graced
    /// and the lent records a prompt group holds claimed (which it returns
    /// ready, so no decode expert elsewhere is unmapped for them) make
    /// `reserve`, or the experts lacked where fewer (a tier that keeps every
    /// expert keeps no reserve: no landing can be demanded of it), or none
    /// is left; the lent records among them where `lent` (no prompt group
    /// held or awaited them as this read), each with the expert it keeps.
    /// It reads the ranking, and changes nothing.
    pub fn refill_victims<K: Ord + Copy>(&self, reserve: usize, rank: impl Rank<K>, lent: bool) -> Vec<(u32, Staged)> {
        let short = |unmapped: usize| self.short_of(reserve, unmapped);
        let mut victims = self.owed();
        if !short(victims.len()) {
            return victims;
        }
        let order = self
            .taking_order(&rank, true, |record| (lent || record >= self.lent) && self.unheld(record))
            .into_iter()
            .map(|record| (record, self.records[record as usize].whole().expect("a record in taking order keeps an expert").staged));
        for victim in order {
            if !short(victims.len()) {
                break;
            }
            if !victims.contains(&victim) {
                victims.push(victim);
            }
        }
        victims
    }

    /// Whether the ready, the graced and the claimed records (every claimed
    /// one a lent one, [`Self::claim_record`]), with `unmapped` more, fall
    /// short of `reserve`, or of the experts lacked where fewer.
    pub fn short_of(&self, reserve: usize, unmapped: usize) -> bool {
        self.ready.len() + self.graced.len() + self.claimed + unmapped < reserve.min(self.lacked() + unmapped)
    }

    /// Unmap the reserve's `victims` ([`Self::refill_victims`]), under the
    /// lock every step begins under, `at` the first step not yet begun: each
    /// unchanged since it was chosen in this turn of residency (a changed
    /// one is an invariant violation), a lent record only where `lent` (no
    /// prompt group holds or awaits them, read under this lock, else passed
    /// over), is unmapped from `at` and graced through its expert's layer of
    /// `latest`, the last step begun `(group, sequence)`, and an outstanding
    /// answer through its request too. A lent record is never graced
    /// through a group's own step, which would hold its claim behind it.
    /// The work here is a check, a map store and a state change a victim.
    pub fn refill(&mut self, victims: Vec<(u32, Staged)>, lent: bool, at: &Unbegun, (group, sequence): (Step, Step)) {
        for (record, chosen) in victims {
            // Chosen in the same turn of residency, nothing between: only
            // the reservation may have changed, which a group takes as it
            // begins.
            let whole = self.records[record as usize].whole();
            assert!(whole.is_some_and(|whole| whole.staged == chosen && whole.holds == 0), "victim record {record} changed since it was chosen: {whole:?}");
            if !lent && record < self.lent {
                continue;
            }
            let (staged, answer) = match self.records[record as usize] {
                Host::Answering(Whole { staged, .. }, past) => (staged, Some(past)),
                _ => (self.unkeep(record), None),
            };
            if let Some(map) = &self.maps[staged.0] {
                map.store_unbegun(staged.1, None, Some(at));
            }
            self.set(record, Host::Graced { staged, grace: Point { group, layer: staged.0, sequence }, answer });
        }
        // A displaced record unmapped, or changed since its seed chose it,
        // is displaced no more.
        let displaced = std::mem::take(&mut self.displaced);
        self.displaced = displaced.into_iter().filter(|&(record, staged)| self.still_keeps(record, staged)).collect();
    }

    /// Prepare a demand's victims ([`Self::victims`]): the first `depth`
    /// kept records no copy holds, in taking order at `rank`. Run after a
    /// request's reads are submitted, so the scan is off every park's path;
    /// the order it leaves is a few passes old at most when a claim reads
    /// it, and each entry is checked again as it is taken.
    pub fn prepare_demand<K: Ord + Copy>(&mut self, rank: impl Rank<K>, depth: usize, sequence: Step) {
        let mut order = (0..self.records())
            .filter(|&record| self.unheld(record))
            .filter_map(|record| Some((self.place(record, &rank)?, record)))
            .collect::<Vec<_>>();
        if order.len() > depth {
            order.select_nth_unstable(depth);
            order.truncate(depth);
        }
        order.sort_unstable();
        self.victims = order.into_iter().map(|((.., staged), record)| (record, staged)).collect();
        self.prepared = Some((sequence, self.victims.len()));
    }

    /// Whether the victims want preparing again at step `sequence`: never
    /// prepared, some taken or passed over since, or prepared at an
    /// earlier step.
    pub fn unprepared(&self, sequence: Step) -> bool {
        self.prepared.is_none_or(|(at, count)| at < sequence || self.victims.len() < count)
    }

    /// [`Self::claim_demand`] from the prepared victims alone: none where it
    /// needs a scan, every prepared victim gone (since unmapped, landed
    /// into, held or read by the parked step).
    pub fn claim_prepared(&mut self, staged: Staged, demand: Demand, read: impl Fn(u32) -> bool, (parked, steps): (Point, u64)) -> Option<Claim> {
        if let Some(claim) = self.demanded(staged, demand) {
            return Some(claim);
        }
        if self.takes(0) {
            return Some(self.take_demanded(staged, demand, parked));
        }
        let record = loop {
            let (record, kept) = self.victims.pop_front()?;
            let keeps = matches!(self.records[record as usize], Host::Kept(Whole { staged, holds: 0, .. }) if staged == kept);
            if keeps && !read(record) {
                break record;
            }
        };
        if !self.claims(staged, demand) {
            self.victims.push_front((record, self.records[record as usize].present().expect("a victim keeps its expert")));
            return Some(Claim::Landing);
        }
        Some(self.take_victim(staged, record, (parked, steps)))
    }

    /// What a demand's claim of `staged` finds before it takes a record
    /// ([`Self::claim_demand`]): kept or landing already, that, for a lacked
    /// expert; none where it takes one: the expert lacked, or the posted
    /// read's, which no record has taken (any other state of it is the
    /// read path's broken invariant).
    fn demanded(&self, staged: Staged, demand: Demand) -> Option<Claim> {
        match (demand, self.located(staged)) {
            (Demand::Lacked, Located::Kept(record)) => Some(Claim::Kept(record)),
            (Demand::Lacked, Located::Landing) => Some(Claim::Landing),
            (Demand::Lacked, Located::Lacked) => None,
            (Demand::Posted, Located::Landing) if !self.landing.contains_key(&staged) => None,
            (Demand::Posted, located) => panic!("a posted read of {staged:?} claims a record, located {located:?}"),
        }
    }

    /// Whether `demand` holds `staged`'s word: a posted read's claimed it,
    /// a lacked expert's claims it now (a posted read may have since it was
    /// located).
    fn claims(&self, staged: Staged, demand: Demand) -> bool {
        demand == Demand::Posted || self.inbound.claim(staged)
    }

    /// A ready record for `demand`'s landing of `staged`, answering the
    /// request at `parked`; landing, where a posted read claimed it first.
    fn take_demanded(&mut self, staged: Staged, demand: Demand, parked: Point) -> Claim {
        match self.claims(staged, demand) {
            true => Claim::Taken(self.take_lacked(staged, Some(parked))),
            false => Claim::Landing,
        }
    }

    /// Unmap kept record `record` from the first step whose resolution of
    /// its expert's layer is not before `parked`, and take it for `staged`,
    /// whose word its caller claimed.
    fn take_victim(&mut self, staged: Staged, record: u32, (parked, steps): (Point, u64)) -> Claim {
        let unmapped = self.unkeep(record);
        self.unmap_from(unmapped, record, parked.first_unresolved(unmapped.0, steps));
        self.set(record, Host::Ready);
        Claim::Taken(self.take_lacked(staged, Some(parked)))
    }

    /// Claim a demand landing of `staged` for `demand` ([`Self::claim`]
    /// leaving no reserve, or a record for the read the request service's
    /// look posted, [`Demand`]), the device parked at `parked` (a group of `steps` steps):
    /// where it is lacked and no record is ready, unmap the first kept
    /// record, in taking order at `rank`, not held and not `read` (what the
    /// parked step resolved at its layer, `decode::parked_reads`), from the
    /// first step whose resolution of its expert's layer is not before
    /// `parked` ([`Point::first_unresolved`]): every resolution that may
    /// name it is then before `parked`, so it is ready at once, and taken,
    /// its unmapping stored before any landing writes it. Short where every
    /// kept record is held, read, or another request's outstanding answer.
    /// The record taken is the parked request's outstanding answer once it
    /// lands.
    pub fn claim_demand<K: Ord + Copy>(&mut self, staged: Staged, demand: Demand, rank: impl Rank<K>, read: impl Fn(u32) -> bool, (parked, steps): (Point, u64)) -> Claim {
        if let Some(claim) = self.demanded(staged, demand) {
            return claim;
        }
        if self.takes(0) {
            return self.take_demanded(staged, demand, parked);
        }
        let first = (0..self.records())
            .filter(|&record| self.unheld(record) && !read(record))
            .filter_map(|record| Some((self.place(record, &rank)?, record)))
            .min();
        let Some((_, record)) = first else {
            return Claim::Short;
        };
        if !self.claims(staged, demand) {
            return Claim::Landing;
        }
        self.take_victim(staged, record, (parked, steps))
    }

    /// The device parked at `parked` (a group of `steps` steps), steps
    /// running in (group, layer, sequence) order, so every resolution
    /// before it is done: every graced record the parked step did not
    /// resolve (`read`, what it resolved at its layer,
    /// `decode::parked_reads`), nor answers a request at or after the park
    /// (its own, which its entries read after it), is ready. A plain map's unmapping is read by
    /// every resolution after its store, so none after the park names it.
    /// A stamped map's names it for the steps below its stamp that have yet
    /// to resolve its layer: its entry is restamped, where it still names
    /// the record, from the first step whose resolution of that layer is
    /// not before the park ([`Point::first_unresolved`]), as a demand's
    /// reclaim unmaps, before the record is ready. So a reserve unmapped
    /// between parks is ready at the next park, however many steps have
    /// begun behind it, and the read ahead it is kept for finds it. True
    /// when any became ready.
    pub fn release(&mut self, (parked, steps): (Point, u64), read: impl Fn(u32) -> bool) -> bool {
        self.passed(parked);
        let resolution = self.resolution;
        self.free(|tier, record, staged, _, answer| {
            if read(record) || answer.is_some_and(|answer| answer >= parked) {
                return false;
            }
            if resolution == Resolution::BeforeStamp {
                tier.unmap_from(staged, record, parked.first_unresolved(staged.0, steps));
            }
            true
        })
    }

    /// The device is past every point before `progress`: every graced
    /// record ready whose last resolution that may name it
    /// ([`Resolution::bound`]) is before it, and every outstanding answer of
    /// a request before it consumed, kept. True when any became ready.
    pub fn retire(&mut self, progress: Point) -> bool {
        self.passed(progress);
        let resolution = self.resolution;
        self.free(|_, _, _, grace, answer| resolution.bound(grace, progress) < progress && answer.is_none_or(|answer| answer < progress))
    }

    /// The device is past every point before `progress`: every outstanding
    /// answer of a request before it is consumed, kept, and so is every
    /// answer a landing ends with later ([`Self::answering`]).
    fn passed(&mut self, progress: Point) {
        self.progress = Some(self.progress.map_or(progress, |known| known.max(progress)));
        for record in self.answering.clone() {
            let Host::Answering(whole, past) = self.records[record as usize] else {
                unreachable!("an answering record is listed answering");
            };
            self.set(record, self.answering(whole, Some(past)));
        }
    }

    /// Every graced record `past` its grace (the records, the record, the
    /// expert it held, the grace, the request it answers) ready, in the
    /// order graced. True when any became ready.
    fn free(&mut self, mut past: impl FnMut(&Self, u32, Staged, Point, Option<Point>) -> bool) -> bool {
        let freed = self
            .graced
            .clone()
            .into_iter()
            .filter(|&record| {
                let Host::Graced { staged, grace, answer } = self.records[record as usize] else {
                    unreachable!("a graced record is listed graced");
                };
                past(self, record, staged, grace, answer)
            })
            .collect::<Vec<_>>();
        for &record in &freed {
            self.set(record, Host::Ready);
        }
        !freed.is_empty()
    }

    /// The bank evicted `staged`, last read at `read`: a record keeping its
    /// copy holds it alone from now, ranked beside that read; without one
    /// only the file holds it.
    pub fn evicted(&mut self, staged: Staged, read: Option<Step>) {
        let Some(&record) = self.kept.get(&staged) else {
            return;
        };
        let content = self.records[record as usize];
        let whole = content.whole().expect("a record keeping an expert keeps it whole");
        self.set(record, content.with(Whole { kept: Kept::Alone, read: whole.read.max(read), ..whole }));
    }

    /// The bank holds `staged` now (a promotion or a seed committed it): a
    /// record keeping it holds a banked copy, so the tier never holds alone
    /// what the bank holds.
    pub fn banked(&mut self, staged: Staged) {
        let Some(&record) = self.kept.get(&staged) else {
            return;
        };
        let content = self.records[record as usize];
        let whole = content.whole().expect("a record keeping an expert keeps it whole");
        self.set(record, content.with(Whole { kept: Kept::Banked, ..whole }));
    }

    /// Claim lent record `record` for the prompt group whose first step is
    /// `from`: claimed already, or ready, it is claimed; kept and not held
    /// (no copy reads it), its expert is unmapped from `from` on (the group
    /// runs after every step that read it, before any read lands in it) and
    /// it is claimed; otherwise (graced, landing, held, or an outstanding
    /// answer) it waits. Whether the group holds it. A record past the lent
    /// ones is an invariant violation.
    pub fn claim_record(&mut self, record: u32, from: Step) -> bool {
        assert!(record < self.lent, "record {record} lies past the {} lent", self.lent);
        match self.records[record as usize] {
            Host::Claimed => true,
            Host::Ready => {
                self.set(record, Host::Claimed);
                true
            }
            Host::Kept(Whole { holds: 0, .. }) => {
                let staged = self.unkeep(record);
                self.unmap_from(staged, record, from);
                self.set(record, Host::Claimed);
                true
            }
            Host::Kept(_) | Host::Answering(..) | Host::Graced { .. } | Host::Landing { .. } => false,
        }
    }

    /// Whether a prompt group holds `record` claimed.
    pub fn is_claimed(&self, record: u32) -> bool {
        self.records[record as usize] == Host::Claimed
    }

    /// Every record a prompt group claimed, each a lent one, returns empty,
    /// ready.
    pub fn restore_lent(&mut self) {
        for record in 0..self.lent {
            if self.records[record as usize] == Host::Claimed {
                self.set(record, Host::Ready);
            }
        }
    }

    /// Whether record `record` is in flight, which a shrink keeps where it
    /// is, never dropping it, moving it or moving another into it
    /// ([`Self::shrink`]): a landing writes it, a copy reads it, a request's
    /// answer is outstanding at it, or a prompt group holds it. A shrink
    /// may change every other record: ready, graced, or kept and unheld.
    fn in_flight(&self, record: u32) -> bool {
        match self.records[record as usize] {
            Host::Ready | Host::Graced { .. } | Host::Kept(Whole { holds: 0, .. }) => false,
            Host::Landing { .. } | Host::Kept(_) | Host::Answering(..) | Host::Claimed => true,
        }
    }

    /// The fewest records a shrink may leave ([`Self::shrink`]): every
    /// record in flight lies below them, and the conveyor's lent records.
    pub fn shrink_floor(&self) -> u32 {
        let flying = (0..self.records()).rev().find(|&record| self.in_flight(record));
        flying.map_or(self.lent, |record| self.lent.max(record + 1))
    }

    /// Shrink to `records` records, at least [`Self::shrink_floor`], between
    /// flights (no step runs, so no graced record is read): every record in
    /// flight stays where it is, landing, held or claimed as it was; of the
    /// rest it drops the empty records first, then the kept first in taking
    /// order at `rank`, keeping the most valuable, and each kept one at or
    /// past `records` moves to a record below it that the rest leave,
    /// mapped there from the first step not yet begun, each dropped one
    /// unmapped. The moves, (from, to), whose bytes the caller copies before
    /// any step reads them.
    pub fn shrink<K: Ord + Copy>(&mut self, records: u32, rank: impl Rank<K>) -> Vec<(u32, u32)> {
        let count = self.records();
        assert!(records <= count, "a host tier of {count} records shrinks to {records}");
        let floor = self.shrink_floor();
        assert!(records >= floor, "a host tier shrinks to {records} records, below its floor of {floor}");
        let order = self.taking_order(&rank, false, |record| !self.in_flight(record));
        // Between flights no step reads a graced record: every one is empty.
        for record in self.graced.clone() {
            self.set(record, Host::Ready);
        }
        let empty = self.ready.clone();
        let dropped_kept = ((count - records) as usize).saturating_sub(empty.len());
        let (dropped, kept) = order.split_at(dropped_kept);
        // The records below the shrink that the kept leave: the empty ones,
        // then those whose experts are dropped.
        let mut targets = empty.iter().copied().chain(dropped.iter().copied()).filter(|&record| record < records);
        let moves = kept
            .iter()
            .filter(|&&record| record >= records)
            .map(|&from| (from, targets.next().expect("a kept record past the shrink has one below it")))
            .collect::<Vec<_>>();
        for &record in dropped {
            let staged = self.unkeep(record);
            self.store(staged, None);
            self.set(record, Host::Ready);
        }
        for &(from, to) in &moves {
            let staged = self.relocate(from, to);
            self.store(staged, Some(to));
        }
        assert!(self.records[records as usize..].iter().all(|&content| content == Host::Ready), "a shrink keeps nothing past it");
        self.records.truncate(records as usize);
        self.ready.retain(|&record| record < records);
        // A demand's prepared victims name records the shrink moved or
        // dropped: none stands, and the next claim prepares again.
        self.victims.clear();
        self.prepared = None;
        moves
    }

    /// The fewest records a cut beside running steps may leave
    /// ([`Self::cut_running`]): [`Self::shrink_floor`], and every graced
    /// record still read as a request's outstanding answer, which the host
    /// reads as well as the device.
    pub fn running_floor(&self) -> u32 {
        let answered = (0..self.records())
            .rev()
            .find(|&record| matches!(self.records[record as usize], Host::Graced { answer: Some(_), .. }));
        answered.map_or(self.shrink_floor(), |record| self.shrink_floor().max(record + 1))
    }

    /// Cut to `records` records, at least [`Self::running_floor`], beside
    /// running steps: every record at or past the cut is dropped, nothing
    /// moved, and the expert each keeps or was graced from is unmapped from
    /// step `until(staged)` on where its entry still names that record
    /// ([`LandingMap::unmap_from`]): the first step that may no longer read
    /// it, which no later than the first step not yet begun. Each dropped
    /// expert is lacked from now, landed again as any miss is.
    pub fn cut_running(&mut self, records: u32, until: impl Fn(Staged) -> Step) {
        let count = self.records();
        assert!(records <= count, "a host tier of {count} records is cut to {records}");
        let floor = self.running_floor();
        assert!(records >= floor, "a host tier is cut to {records} records beside running steps, below its floor of {floor}");
        for record in records..count {
            match self.records[record as usize] {
                Host::Ready => {}
                Host::Kept(Whole { staged, holds: 0, .. }) | Host::Graced { staged, answer: None, .. } => {
                    self.unmap_from(staged, record, until(staged));
                    self.set(record, Host::Ready);
                }
                other => unreachable!("record {record} past the running floor {floor} holds {other:?}"),
            }
        }
        self.records.truncate(records as usize);
        self.ready.retain(|&record| record < records);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::reuse::Score;

    /// The reserve's turn on `tier` ([`HostRecords::refill_victims`] then
    /// [`HostRecords::refill`]), a prompt group holding the lent records:
    /// `latest` read only where it unmaps any.
    fn refill(tier: &mut HostRecords, reserve: usize, rank: impl Rank<Score>, at: &Unbegun, latest: impl FnOnce() -> (Step, Step)) {
        let victims = tier.refill_victims(reserve, rank, false);
        if !victims.is_empty() {
            tier.refill(victims, false, at, latest());
        }
    }
    use crate::experts::Words;
    use crate::experts::map::{MapView, StampedMap};
    use crate::experts::watcher::Steps;
    use magnitude_kernels::constants::MAP_WORDS;
    use std::cmp::Reverse;

    /// The fill's order where nothing is shared or kept: the most reused,
    /// then by expert, then by layer.
    fn by_reuse(score: Score, (layer, expert): Staged) -> (Reverse<Score>, usize, usize) {
        (Reverse(score), expert, layer)
    }

    /// The experts of a tier's group where more than its records: every
    /// landing below may be demanded.
    const EXPERTS: usize = 64;

    /// A group of one layer of [`EXPERTS`] experts.
    const ONE: &[(usize, usize)] = &[(0, EXPERTS)];

    /// A point of decode step `sequence` at layer 0.
    fn at(sequence: Step) -> Point {
        Point { group: sequence, layer: 0, sequence }
    }

    /// `records` records of a group of `members` (layer, experts) of
    /// `layers` layers, of no device map.
    fn tier(records: u32, layers: usize, members: &[(usize, usize)], resolution: Resolution) -> HostRecords {
        HostRecords::new(records, layers, members.iter().map(|&(layer, experts)| (layer, experts, None)).collect(), resolution, 0, inbound(layers))
    }

    /// The words of `layers` layers of [`EXPERTS`] experts, every one lacked.
    fn inbound(layers: usize) -> Arc<InboundTable> {
        Arc::new(InboundTable::new(&vec![EXPERTS; layers]))
    }

    /// The maps a tier's records write, as its readers read them: a host
    /// tier's plain words, or the one store's stamped map, its steps begun
    /// through 1,000 so an entry stored keeps the step it is stored from.
    enum Maps {
        Plain(Vec<Words>),
        Stamped(Vec<MapView>),
    }

    impl Maps {
        /// The record `staged`'s entry names for every step from now, none
        /// unmapped.
        fn held(&self, (layer, expert): Staged) -> Option<u32> {
            match self {
                Self::Plain(words) => u32::try_from(words[layer].load(expert)).ok(),
                Self::Stamped(views) => views[layer].held(expert).map(|record| record as u32),
            }
        }

        /// The record step `step` resolves `staged` to, a stamped map's.
        fn resolve(&self, (layer, expert): Staged, step: Step) -> Option<u32> {
            match self {
                Self::Plain(_) => unreachable!("a plain map holds no steps"),
                Self::Stamped(views) => views[layer].resolve(expert, step).map(|record| record as u32),
            }
        }
    }

    /// [`tier`], writing real maps: plain words where `resolution` reads
    /// after the store, else stamped.
    fn mapped(records: u32, layers: usize, members: &[(usize, usize)], resolution: Resolution) -> (HostRecords, Maps) {
        let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let steps = std::sync::Arc::new(Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new()));
        steps.begin_through(1000);
        let mut maps = vec![None; layers];
        let mut plain = Vec::new();
        let mut stamped = Vec::new();
        for layer in 0..layers {
            match resolution {
                Resolution::AfterStore => {
                    let tensor = seismic::Tensor::shared(&cpu, seismic::Element::i32(), &[EXPERTS as u64]).unwrap();
                    let words = Words::new(tensor, EXPERTS as u64, -1).unwrap();
                    plain.push(words.clone());
                    maps[layer] = Some(LandingMap::Tier(words));
                }
                Resolution::BeforeStamp => {
                    let tensor = seismic::Tensor::shared(&cpu, seismic::Element::i32(), &[EXPERTS as u64, MAP_WORDS]).unwrap();
                    let map = StampedMap::new(Words::new(tensor, EXPERTS as u64 * MAP_WORDS, 0).unwrap());
                    stamped.push(map.view().clone());
                    maps[layer] = Some(LandingMap::Bank { map: std::sync::Arc::new(map), steps: steps.clone() });
                }
            }
        }
        let members = members.iter().map(|&(layer, experts)| (layer, experts, maps[layer].take())).collect();
        let tier = HostRecords::new(records, layers, members, resolution, 0, inbound(layers));
        let maps = match resolution {
            Resolution::AfterStore => Maps::Plain(plain),
            Resolution::BeforeStamp => Maps::Stamped(stamped),
        };
        (tier, maps)
    }

    impl HostRecords {
        /// A claim of lacked `staged`: its grant, none where short.
        fn grant_of(&mut self, staged: Staged, keep: usize) -> Option<Grant> {
            match self.claim(staged, keep) {
                Claim::Taken(grant) => Some(grant),
                Claim::Short => None,
                Claim::Kept(_) | Claim::Landing => panic!("{staged:?} is not lacked"),
            }
        }

        /// A claim of lacked `staged`: the record taken, none where short.
        fn take(&mut self, staged: Staged, keep: usize) -> Option<u32> {
            self.grant_of(staged, keep).map(|grant| grant.record())
        }

        /// Land `staged`: claimed, its grant's map written, learned landed.
        fn land(&mut self, staged: Staged) -> u32 {
            let grant = self.grant_of(staged, 0).expect("a record for the landing");
            grant.map(Some(0));
            self.landed(grant.record(), staged, None, Kept::Alone);
            grant.record()
        }

        /// A demand claim of lacked `staged` parked at `parked` in a group
        /// of `steps` steps, nothing read: the record taken and the expert
        /// whose record it unmapped, if one, none where short.
        fn reclaim(&mut self, staged: Staged, rank: impl Rank<Score>, parked: Point, steps: u64) -> Option<(u32, Option<Staged>)> {
            let kept = self.kept.clone();
            match self.claim_demand(staged, Demand::Lacked, rank, |_| false, (parked, steps)) {
                Claim::Taken(grant) => {
                    let unmapped = kept.into_iter().find(|&(_, record)| record == grant.record()).map(|(staged, _)| staged);
                    Some((grant.record(), unmapped))
                }
                Claim::Short => None,
                Claim::Kept(_) | Claim::Landing => panic!("{staged:?} is not lacked"),
            }
        }
    }

    /// A claim finds an expert where it is: kept, its record; landing, in
    /// flight; only a lacked one takes a record, so no two landings of one
    /// expert are ever in flight, and a failed one lacks it again.
    /// A posted read's word makes its expert landing to every other claim
    /// until a demand's claim of it takes a record (none waits while the
    /// tier has none); a posted read given up lacks it again.
    #[test]
    fn a_posted_read_holds_its_expert_until_its_record_is_claimed() {
        let inbound = inbound(1);
        let mut tier = HostRecords::new(1, 1, vec![(0, EXPERTS, None)], Resolution::AfterStore, 0, inbound.clone());
        assert!(inbound.claim((0, 1)), "a lacked expert's word is claimed once");
        assert!(!inbound.claim((0, 1)), "and never twice");
        assert_eq!(tier.located((0, 1)), Located::Landing, "a posted read lands it");
        assert!(matches!(tier.claim((0, 1), 0), Claim::Landing), "a read ahead finds it landing");
        let Claim::Taken(grant) = tier.claim_demand((0, 1), Demand::Posted, reads(&[]), |_| false, (at(5), 1)) else { panic!("the one record is ready") };
        tier.landed(grant.record(), (0, 1), Some(5), Kept::Alone);
        assert_eq!(tier.located((0, 1)), Located::Kept(grant.record()));
        assert!(inbound.claim((0, 2)));
        assert!(matches!(tier.claim_demand((0, 2), Demand::Posted, reads(&[]), |_| false, (at(5), 1)), Claim::Short), "the kept record is the request's answer");
        tier.unposted((0, 2));
        assert_eq!(tier.located((0, 2)), Located::Lacked, "a posted read given up lacks it again");
        assert!(inbound.claim((0, 2)), "and the next read claims it");
    }

    #[test]
    fn a_claim_takes_a_record_only_for_a_lacked_expert() {
        let mut tier = tier(4, 1, ONE, Resolution::BeforeStamp);
        let record = tier.take((0, 1), 0).unwrap();
        assert!(matches!(tier.claim((0, 1), 0), Claim::Landing), "a second landing of it claims the first");
        assert!(matches!(tier.claim_demand((0, 1), Demand::Lacked, reads(&[]), |_| false, (at(1), 1)), Claim::Landing));
        tier.landed(record, (0, 1), None, Kept::Alone);
        assert!(matches!(tier.claim((0, 1), 0), Claim::Kept(kept) if kept == record));
        let other = tier.take((0, 2), 0).unwrap();
        tier.unlanded(other);
        assert_eq!(tier.located((0, 2)), Located::Lacked, "a failed landing lacks it again");
    }

    /// A demand's prepared victims are taken in taking order, each only
    /// while it still keeps the expert it was prepared with, unread by the
    /// parked step: one landed into since is passed over, and once none is
    /// left the claim falls back to the scan.
    #[test]
    fn a_demand_takes_its_prepared_victims_in_order_while_they_hold() {
        let mut records = tier(3, 1, ONE, Resolution::BeforeStamp);
        let [one, two, three] = [1, 2, 3].map(|expert| records.land((0, expert)));
        let ranked = [((0, 1), 5), ((0, 2), 1), ((0, 3), 3)];
        records.prepare_demand(reads(&ranked), 3, 5);
        let taken = |claim: Option<Claim>| match claim {
            Some(Claim::Taken(grant)) => Some(grant.record()),
            Some(_) => panic!("a lacked expert's claim takes a record or none"),
            None => None,
        };
        assert_eq!(taken(records.claim_prepared((0, 4), Demand::Lacked, |_| false, (at(6), 1))), Some(two), "the least recently read first");
        assert_eq!(taken(records.claim_prepared((0, 5), Demand::Lacked, |record| record == three, (at(6), 1))), Some(one), "the parked step's read passed over");
        assert_eq!(taken(records.claim_prepared((0, 6), Demand::Lacked, |_| false, (at(6), 1))), None, "none left: the claim scans");
        let mut stale = tier(3, 1, ONE, Resolution::BeforeStamp);
        for expert in 1..=3 {
            stale.land((0, expert));
        }
        stale.prepare_demand(reads(&ranked), 1, 5);
        let Claim::Taken(grant) = stale.claim_demand((0, 4), Demand::Lacked, reads(&ranked), |_| false, (at(6), 1)) else { panic!("a kept record frees") };
        assert_eq!(grant.record(), two, "the scan takes the prepared victim first");
        assert_eq!(taken(stale.claim_prepared((0, 5), Demand::Lacked, |_| false, (at(6), 1))), None, "a victim taken since is passed over");
    }

    /// A shrink leaves no prepared victim behind: a claim in the same step
    /// prepares again and takes a record the tier still has, never one the
    /// shrink moved or dropped.
    #[test]
    fn a_claim_after_a_shrink_takes_no_victim_prepared_before_it() {
        let mut records = tier(3, 1, ONE, Resolution::BeforeStamp);
        for expert in 1..=3 {
            records.land((0, expert));
        }
        let ranked = [((0, 1), 5), ((0, 2), 1), ((0, 3), 3)];
        records.prepare_demand(reads(&ranked), 3, 5);
        records.shrink(1, reads(&ranked));
        assert!(records.unprepared(5), "the shrink's records want preparing again");
        assert_eq!(records.claim_prepared((0, 4), Demand::Lacked, |_| false, (at(5), 1)).map(|_| ()), None, "no victim prepared before the shrink is taken");
        let Claim::Taken(grant) = records.claim_demand((0, 4), Demand::Lacked, reads(&ranked), |_| false, (at(5), 1)) else { panic!("the kept record frees") };
        assert_eq!(grant.record(), 0, "the tier's one record");
    }

    /// The ranking of experts last read at `reads`, each (expert, step);
    /// none read never.
    fn reads(reads: &[(Staged, Step)]) -> impl Fn(Staged, Option<Step>) -> Score + '_ {
        move |staged, _| reads.iter().find(|&&(read, _)| read == staged).map_or(Score::NEVER, |&(_, step)| Score::at(step))
    }

    /// A tier with a record for every expert of its group fills to every
    /// expert and keeps no reserve: a fill takes a record while one more
    /// lacked expert than the reserve would remain, the last ones too, and
    /// the reserve unmaps nothing once it lacks none; lacking one again,
    /// one record ready covers it, and a fill may land it there.
    #[test]
    fn a_tier_of_every_expert_fills_and_keeps_no_reserve() {
        let rank = reads(&[]);
        let mut tier = tier(3, 1, &[(0, 3)], Resolution::AfterStore);
        for expert in 0..3 {
            let record = tier.take((0, expert), 2).expect("a fill beyond a reserve of two takes the lacked ones' records");
            tier.landed(record, (0, expert), None, Kept::Alone);
        }
        assert!(tier.complete() && tier.lacked() == 0);
        assert!(!tier.takes(0), "no expert lacked");
        refill(&mut tier, 2, &rank, &Unbegun::at(1), || unreachable!("a complete tier keeps no reserve"));
        assert!(tier.complete());
        // One dropped and its record grown back: lacking one, the reserve
        // is one record.
        tier.shrink(2, &rank);
        tier.grow(3);
        assert_eq!((tier.lacked(), tier.ready()), (1, 1));
        refill(&mut tier, 2, &rank, &Unbegun::at(1), || unreachable!("one record ready covers the one lacked"));
        let (_, lacked) = tier.lacking(&rank, 1, |_| false, by_reuse)[0];
        assert!(tier.take(lacked, 2).is_some(), "a fill of the lacked one itself leaves nothing a demand could need");
        assert!(tier.lands(lacked) && !tier.takes(0), "nothing lacked beyond the landing");
    }

    /// A tier of four records of one group of layers 0 and 2, each of three
    /// experts, two of each landed, read by `resolution`, through its maps.
    fn landed_tier(resolution: Resolution) -> (HostRecords, Maps) {
        let (mut tier, maps) = mapped(4, 3, &[(0, 3), (2, 3)], resolution);
        for staged in [(0, 0), (0, 1), (2, 0), (2, 1)] {
            tier.land(staged);
        }
        (tier, maps)
    }

    /// One rule frees a record: once the device is past the last resolution
    /// that may name it. Parked at layer 1 of step 10, steps 11 and 12 begun
    /// behind it, a record the reserve unmapped while 12 was the latest
    /// begun is graced through its layer of 12. A stamped map holds it for
    /// every step before its stamp, so steps 10..=12 may still name it:
    /// unparked, it frees only past that point; at the park its entry is
    /// restamped from the first step that has not resolved its layer, so no
    /// resolution after the park names it and it frees there, unless the
    /// parked step resolved to it. A plain map's store is read by every
    /// later resolution, so it frees at the park unless the parked step
    /// resolved to it, with no restamp.
    #[test]
    fn a_record_frees_once_past_every_resolution_that_may_name_it() {
        let rank = reads(&[]);
        let parked = (Point { group: 10, layer: 1, sequence: 10 }, 1);
        let graced = |resolution| {
            let (mut tier, maps) = landed_tier(resolution);
            refill(&mut tier, 1, &rank, &Unbegun::at(13), || (12, 12));
            let [record] = tier.graced[..] else { panic!("one record graced") };
            let Host::Graced { staged, grace, .. } = tier.records[record as usize] else { panic!("record {record} is graced") };
            assert_eq!((grace.group, grace.sequence), (12, 12), "graced through the latest step begun");
            (tier, maps, record, staged, grace)
        };
        let (mut stamped, _, _, _, grace) = graced(Resolution::BeforeStamp);
        assert!(!stamped.retire(grace), "step 12 may still name it");
        assert!(stamped.retire(Point { group: 13, layer: 0, sequence: 13 }), "past step 12");
        let (mut stamped, maps, record, staged, _) = graced(Resolution::BeforeStamp);
        assert!(!stamped.release(parked, |of| of == record), "the parked step resolved to it");
        let from = parked.0.first_unresolved(staged.0, 1);
        assert_eq!(maps.resolve(staged, from - 1), Some(record), "still named below the restamp");
        assert!(stamped.release(parked, |_| false));
        assert_eq!((maps.resolve(staged, from - 1), maps.resolve(staged, from)), (Some(record), None), "restamped from the first unresolved step");
        let (mut plain, maps, record, staged, _) = graced(Resolution::AfterStore);
        assert_eq!(maps.held(staged), None, "a plain unmapping is read by every later resolution");
        assert!(!plain.release(parked, |of| of == record), "the parked step resolved to it");
        assert!(plain.release(parked, |_| false), "no later resolution names it");
    }

    /// A reclaim, parked, unmaps from the first step whose resolution of the
    /// record's layer the device has not reached: a later layer of the
    /// parked step's group from the group's first step, its own layer from
    /// the step after it, an earlier layer from the group after; so every
    /// resolution that may name the record is before the park and it is
    /// ready at once.
    #[test]
    fn a_reclaim_unmaps_from_the_first_step_that_has_not_resolved_its_layer() {
        let rank = reads(&[]);
        let parked = Point { group: 10, layer: 1, sequence: 10 };
        let (mut tier, maps) = landed_tier(Resolution::BeforeStamp);
        let mut froms = std::collections::BTreeSet::new();
        // Each demand of a lacked expert reclaims a kept one, which is then
        // lacked, until every record lands one.
        let mut lacked = vec![(0, 2), (2, 2)];
        while let Some(staged) = lacked.pop() {
            if let Some((record, unmapped)) = tier.reclaim(staged, &rank, parked, 1) {
                assert_eq!(tier.landing.get(&staged), Some(&record), "ready at once, and taken");
                let unmapped = unmapped.expect("no record was ready: one unmapped");
                let from = (0..).find(|&step| maps.resolve(unmapped, step).is_none() && step > 0).expect("an unmapping");
                froms.insert((unmapped.0, from));
                lacked.push(unmapped);
            }
        }
        assert_eq!(froms.into_iter().collect::<Vec<_>>(), [(0, 11), (2, 10)]);
        // A group of steps 20..=23, parked at layer 1 of step 22.
        let wave = Point { group: 20, layer: 1, sequence: 22 };
        assert_eq!([2, 1, 0].map(|layer| wave.first_unresolved(layer, 4)), [20, 23, 24]);
    }

    /// The lacked experts are landed most reused first, a tie every layer's
    /// alike (by the expert's index, then its layer); a landing takes its
    /// expert out of them, a failed one puts it back, and an unmapping
    /// lacks its expert again at its reuse.
    #[test]
    fn the_lacked_experts_are_ranked_by_reuse() {
        let mut tier = tier(1, 2, &[(0, 2), (1, 2)], Resolution::AfterStore);
        let order = |tier: &HostRecords, rank: &dyn Fn(Staged, Option<Step>) -> Score| {
            tier.lacking(rank, 4, |_| false, by_reuse).into_iter().map(|(_, staged)| staged).collect::<Vec<_>>()
        };
        assert_eq!(order(&tier, &reads(&[])), [(0, 0), (1, 0), (0, 1), (1, 1)], "unused, each layer's alike");
        let rank = reads(&[((1, 1), 5), ((0, 1), 3)]);
        assert_eq!(order(&tier, &rank), [(1, 1), (0, 1), (0, 0), (1, 0)]);
        assert_eq!(tier.lacking(&rank, 2, |staged| staged == (1, 1), by_reuse).len(), 2, "the first two not passed over");
        let record = tier.take((1, 1), 0).unwrap();
        assert_eq!(order(&tier, &rank), [(0, 1), (0, 0), (1, 0)]);
        tier.unlanded(record);
        assert_eq!(order(&tier, &rank)[0], (1, 1), "a failed landing lacks it again");
        let record = tier.take((1, 1), 0).unwrap();
        tier.landed(record, (1, 1), None, Kept::Alone);
        assert!(tier.reclaim((0, 0), &rank, at(1), 1).is_some(), "a demand unmaps the one record");
        assert_eq!(order(&tier, &rank)[0], (1, 1), "unmapped, it is lacked at its reuse");
    }

    /// A record held for a pull keeps its expert, mapped, and the reserve
    /// never unmaps it while held: the next in taking order is; released,
    /// it is unmapped in its place.
    #[test]
    fn a_held_record_is_never_unmapped() {
        let rank = reads(&[((0, 1), 3), ((0, 2), 5)]);
        let mut tier = tier(2, 1, ONE, Resolution::AfterStore);
        let first = tier.take((0, 1), 0).unwrap();
        let second = tier.take((0, 2), 0).unwrap();
        tier.landed(first, (0, 1), None, Kept::Alone);
        tier.landed(second, (0, 2), None, Kept::Alone);
        assert_eq!(tier.hold((0, 9)), None, "no record keeps (0, 9)");
        assert_eq!(tier.hold((0, 1)), Some(first));
        refill(&mut tier, 1, &rank, &Unbegun::at(8), || (7, 7));
        assert!(tier.keeps((0, 1)) && !tier.keeps((0, 2)), "the held least recent is passed over");
        tier.retire(at(7));
        refill(&mut tier, 2, &rank, &Unbegun::at(9), || (8, 8));
        assert!(tier.keeps((0, 1)), "only the held record is kept, and stays");
        tier.unhold(first);
        refill(&mut tier, 2, &rank, &Unbegun::at(10), || (9, 9));
        assert!(!tier.keeps((0, 1)), "released, it is unmapped");
    }

    /// A landing takes a ready record, leaving the demand reserve to a
    /// demand, and keeps its expert alone once landed; its record is mapped
    /// by its grant alone; a failed landing's record is ready again.
    #[test]
    fn a_landing_takes_a_ready_record_beyond_what_it_keeps() {
        let (mut tier, maps) = mapped(3, 1, ONE, Resolution::AfterStore);
        let grant = tier.grant_of((0, 5), 2).expect("one ready beyond a reserve of two");
        assert_eq!(grant.record(), 0, "the lowest ready");
        assert_eq!(tier.take((0, 6), 2), None, "the reserve is the demand's");
        assert_eq!(maps.held((0, 5)), None, "unmapped until its bytes are whole");
        grant.map(None);
        tier.landed(0, (0, 5), None, Kept::Alone);
        assert_eq!(maps.held((0, 5)), Some(0), "its grant mapped it");
        assert!(tier.keeps((0, 5)) && tier.holds_alone((0, 5)));
        assert_eq!(tier.alone(), [1]);
        let record = tier.take((0, 6), 0).unwrap();
        assert!(tier.lands((0, 6)));
        tier.unlanded(record);
        assert!(!tier.lands((0, 6)));
        assert_eq!(tier.ready(), 2, "a failed landing's record is ready again");
        assert_eq!(maps.held((0, 6)), None, "a failed landing maps nothing");
    }

    /// A grant maps its record once.
    #[test]
    #[should_panic(expected = "maps its record twice")]
    fn a_grant_maps_once() {
        let mut tier = tier(2, 1, ONE, Resolution::AfterStore);
        let grant = tier.grant_of((0, 5), 0).unwrap();
        grant.map(None);
        grant.map(None);
    }

    /// The reserve: kept records are unmapped in taking order (alone least
    /// recently read, then banked copies), each graced through its layer of
    /// the latest step begun, read where no step begins, and ready once the
    /// device is past that layer of that step, never before.
    #[test]
    fn the_reserve_unmaps_in_taking_order_and_grace_ends_past_the_step() {
        let read = [((0, 1), 9), ((0, 2), 4), ((0, 3), 7), ((0, 4), 2)];
        let rank = reads(&read);
        let (mut tier, maps) = mapped(4, 1, ONE, Resolution::AfterStore);
        for (staged, _) in read {
            tier.land(staged);
        }
        tier.banked((0, 4));
        // Two records: the alone read least recently, (0, 2) then (0, 3);
        // the banked (0, 4) is last.
        refill(&mut tier, 2, &rank, &Unbegun::at(7), || (6, 6));
        assert!(maps.held((0, 2)).is_none() && maps.held((0, 3)).is_none());
        assert!(maps.held((0, 1)).is_some() && maps.held((0, 4)).is_some());
        assert!(!tier.keeps((0, 2)) && !tier.keeps((0, 3)) && tier.keeps((0, 4)));
        assert_eq!(tier.ready(), 0);
        refill(&mut tier, 2, &rank, &Unbegun::at(1), || unreachable!("the graced records make the reserve"));
        assert!(!tier.retire(at(6)), "step 6 could still read them");
        assert!(tier.retire(at(7)), "the device past layer 0 of step 6");
        assert_eq!(tier.ready(), 2);
        // Then the next alone, (0, 1), and the banked copy.
        refill(&mut tier, 4, &rank, &Unbegun::at(8), || (7, 7));
        assert!(maps.held((0, 1)).is_none() && maps.held((0, 4)).is_none());
    }

    /// The tier map follows what each record keeps whole; an eviction of a
    /// banked copy costs nothing, the tier holding it alone; the reserve
    /// takes what the ranking ranks least at the time it takes.
    #[test]
    fn the_tier_map_follows_what_records_keep() {
        let (mut tier, maps) = mapped(2, 2, &[(0, EXPERTS), (1, EXPERTS)], Resolution::AfterStore);
        let first = tier.land((1, 4));
        let second = tier.land((1, 5));
        tier.banked((1, 4));
        assert!(tier.keeps((1, 4)) && !tier.holds_alone((1, 4)));
        tier.evicted((1, 4), None);
        assert!(tier.holds_alone((1, 4)), "evicting a banked copy costs nothing: the tier holds it alone");
        // (1, 5), read in place at step 9, outranks (1, 4), read at 8.
        refill(&mut tier, 1, reads(&[((1, 4), 8), ((1, 5), 9)]), &Unbegun::at(10), || (9, 9));
        assert_eq!((maps.held((1, 4)), maps.held((1, 5))), (None, Some(second)));
        assert_ne!(first, second);
    }

    /// A shrink keeps the most valuable records: it drops the empty first,
    /// then the kept first in taking order, and moves a kept record past the
    /// shrink into one below it that a dropped one leaves, each mapped
    /// there and each dropped one unmapped; growth adds ready records.
    #[test]
    fn a_host_tier_shrinks_to_its_most_valuable_records_and_grows_ready() {
        let rank = reads(&[((0, 10), 8), ((0, 11), 2), ((0, 13), 2)]);
        let (mut tier, maps) = mapped(4, 1, ONE, Resolution::AfterStore);
        let grants = [13, 12, 11, 10].map(|expert| tier.grant_of((0, expert), 0).unwrap());
        assert_eq!(grants.each_ref().map(Grant::record), [0, 1, 2, 3], "each the lowest ready");
        for (grant, expert) in grants.iter().zip([13, 12, 11, 10]) {
            if expert != 12 {
                grant.map(None);
                tier.landed(grant.record(), (0, expert), None, Kept::Alone);
            }
        }
        tier.banked((0, 11));
        tier.unlanded(1);
        // Record 1 (empty), then record 0 (alone, step 2) are dropped:
        // record 3 (alone, step 8) moves into record 1, record 2 (banked)
        // into record 0.
        assert_eq!(tier.shrink(2, &rank), [(3, 1), (2, 0)]);
        assert_eq!(tier.records(), 2);
        assert!(tier.keeps((0, 10)) && tier.keeps((0, 11)) && !tier.keeps((0, 13)));
        assert_eq!([10, 11, 13].map(|expert| maps.held((0, expert))), [Some(1), Some(0), None]);
        assert_eq!(tier.alone(), [1]);
        tier.grow(3);
        assert_eq!(tier.take((0, 13), 0), Some(2), "the grown record is ready");
    }

    /// A shrink keeps every record in flight where it is, below its floor:
    /// a landing's record and one a copy holds are never dropped, moved or
    /// moved into. The rest shrink around them, and the landing lands where
    /// its claim took it.
    #[test]
    fn a_host_tier_shrinks_around_its_records_in_flight() {
        let rank = reads(&[((0, 9), 8), ((0, 10), 6)]);
        let (mut tier, maps) = mapped(6, 1, ONE, Resolution::AfterStore);
        let grants = [14, 13, 12, 11, 10, 9].map(|expert| tier.grant_of((0, expert), 0).unwrap());
        assert_eq!(grants.each_ref().map(Grant::record), [0, 1, 2, 3, 4, 5], "each the lowest ready");
        // Expert 14's read into record 0 fails, leaving it empty.
        tier.unlanded(0);
        for (grant, expert) in grants.iter().zip([14, 13, 12, 11, 10, 9]) {
            if expert != 12 && expert != 14 {
                grant.map(None);
                tier.landed(grant.record(), (0, expert), None, Kept::Alone);
            }
        }
        // Expert 12 lands into record 2, in flight; a copy holds expert
        // 11's record 3.
        assert_eq!(tier.hold((0, 11)), Some(3));
        assert_eq!(tier.shrink_floor(), 4, "every record in flight lies below the floor");
        // Record 0 (empty), then record 1 (expert 13, never read) are
        // dropped: record 4 (step 6) moves into record 0, record 5 (step 8)
        // into record 1.
        assert_eq!(tier.shrink(4, &rank), [(4, 0), (5, 1)]);
        assert_eq!([9, 10, 11, 13].map(|expert| maps.held((0, expert))), [Some(1), Some(0), Some(3), None]);
        assert!(tier.lands((0, 12)), "the landing is in flight where it was");
        grants[2].map(None);
        tier.landed(2, (0, 12), None, Kept::Alone);
        tier.unhold(3);
        assert_eq!(maps.held((0, 12)), Some(2));
        assert_eq!(tier.shrink_floor(), 0, "nothing in flight");
    }

    /// A cut beside running steps keeps every record in flight below it and
    /// drops every record past it, moving nothing: each expert they kept is
    /// lacked from then on and unmapped where its entry still names that
    /// record, from the step `until` gives it, the running chain's first
    /// step (900, steps begun through 1,000) or, for one the chain routed,
    /// the first step not yet begun. Steps before then still read it; none
    /// from then on does, and no landing takes a record past the cut.
    #[test]
    fn a_cut_beside_running_steps_drops_what_lies_past_it_from_the_step_each_expert_needs() {
        let (mut tier, maps) = mapped(6, 1, ONE, Resolution::BeforeStamp);
        let grants = [9, 10, 11, 12, 13, 14].map(|expert| tier.grant_of((0, expert), 0).unwrap());
        assert_eq!(grants.each_ref().map(Grant::record), [0, 1, 2, 3, 4, 5], "each the lowest ready");
        for (grant, expert) in grants.iter().zip([9, 10, 11, 12, 13, 14]) {
            if expert != 11 {
                grant.map(Some(0));
                tier.landed(grant.record(), (0, expert), None, Kept::Alone);
            }
        }
        // Expert 11 lands into record 2, in flight: no cut lies below it.
        assert_eq!(tier.running_floor(), 3, "every record in flight lies below the floor");
        grants[2].map(Some(0));
        tier.landed(2, (0, 11), None, Kept::Alone);
        assert_eq!(tier.running_floor(), 0, "nothing in flight");
        // Expert 13 the chain routed; 12 and 14 it did not.
        let (first, unbegun) = (900, 1001);
        tier.cut_running(3, |(_, expert)| if expert == 13 { Step::MAX } else { first });
        assert_eq!(tier.records(), 3);
        assert_eq!([9, 10, 11].map(|expert| maps.held((0, expert))), [Some(0), Some(1), Some(2)], "below the cut, kept");
        for (expert, until) in [(12, first), (13, unbegun), (14, first)] {
            assert!(tier.kept_record((0, expert)).is_none(), "{expert} is lacked");
            assert!(maps.resolve((0, expert), until - 1).is_some(), "{expert} is read before step {until}");
            assert_eq!(maps.resolve((0, expert), until), None, "{expert} is read from no record from step {until}");
        }
        assert_eq!(tier.take((0, 12), 0), None, "no record past the cut is ready");
    }

    /// The interleaving of `a_parked_requests_answered_record_is_never_taken_by_its_other_miss`
    /// (one record), played step by step: a request at step 5 misses two
    /// experts; the first's demand takes the one record and lands, its
    /// answer outstanding (the device has not read it yet); the reserve
    /// then unmaps it, and at the park the second miss demands a record.
    /// The answered record is the request's until the device is past it:
    /// unmapped, it is graced with its answer, so the park frees it not;
    /// no claim takes it, so the second miss finds none and waits; once the
    /// device is past the request it is free, which a later demand takes.
    #[test]
    fn a_requests_outstanding_answer_is_never_taken_by_its_other_miss() {
        let rank = reads(&[]);
        let request = at(5);
        let (mut tier, maps) = mapped(1, 1, ONE, Resolution::AfterStore);
        let Claim::Taken(first) = tier.claim_demand((0, 1), Demand::Lacked, &rank, |_| false, (request, 1)) else { panic!("the one record is ready") };
        first.map(None);
        tier.landed(first.record(), (0, 1), Some(5), Kept::Alone);
        assert!(matches!(tier.records[0], Host::Answering(_, past) if past == request), "an outstanding answer");
        assert!(!tier.claim_record_of_lent_free(), "no prompt group's claim takes it");
        refill(&mut tier, 1, &rank, &Unbegun::at(6), || (5, 5));
        assert_eq!(maps.held((0, 1)), None, "the reserve unmaps it");
        assert!(matches!(tier.records[0], Host::Graced { answer: Some(answer), .. } if answer == request), "graced with its answer");
        assert!(!tier.release((request, 1), |_| false), "its request's park frees it not");
        assert!(matches!(tier.claim_demand((0, 2), Demand::Lacked, &rank, |_| false, (request, 1)), Claim::Short), "the second miss waits");
        assert!(!tier.retire(request), "the device at the request still reads it");
        assert!(tier.retire(at(6)), "past the request it is free");
        assert!(matches!(tier.claim_demand((0, 2), Demand::Lacked, &rank, |_| false, (at(6), 1)), Claim::Taken(_)), "then a later demand takes it");
    }

    /// A demand landing that ends once the device is past its request (its
    /// step ended with its wait expired) answers nothing: kept, not an
    /// outstanding answer, so nothing holds it in flight for a shrink.
    #[test]
    fn a_landing_past_its_request_is_kept_not_answering() {
        let mut tier = tier(2, 1, ONE, Resolution::AfterStore);
        let Claim::Taken(grant) = tier.claim_demand((0, 1), Demand::Lacked, reads(&[]), |_| false, (at(5), 1)) else { panic!("ready") };
        assert_eq!(tier.shrink_floor(), 1, "the landing is in flight");
        tier.retire(at(6));
        tier.landed(grant.record(), (0, 1), Some(5), Kept::Alone);
        assert!(matches!(tier.records[0], Host::Kept(_)), "kept: the device is past its request");
        assert_eq!(tier.shrink_floor(), 0, "nothing in flight");
    }

    /// A tier keeping an expert in a lent record is complete but not
    /// settled: a prompt group's claim of that record unmaps the expert
    /// mid-flight, so a step's route there may find it absent.
    #[test]
    fn a_tier_keeping_an_expert_in_a_lent_record_is_not_settled() {
        let mut tier = HostRecords::new(2, 1, vec![(0, 2, None)], Resolution::AfterStore, 1, inbound(1));
        for expert in 0..2 {
            let Claim::Taken(grant) = tier.claim((0, expert), 0) else { panic!("a record is ready") };
            grant.map(None);
            tier.landed(grant.record(), (0, expert), None, Kept::Alone);
        }
        assert_eq!(tier.lacked(), 0, "every expert is kept");
        assert!(tier.complete() && !tier.settled(), "one is kept in the lent record");
        assert!(tier.claim_record(0, 10), "a prompt group's claim takes it");
        assert_eq!(tier.lacked(), 1, "and its expert is absent");
    }

    impl HostRecords {
        /// Whether a claim of record 0, made a lent one for the test, takes
        /// it.
        fn claim_record_of_lent_free(&mut self) -> bool {
            self.lent = 1;
            let claimed = self.claim_record(0, 5);
            self.lent = 0;
            claimed
        }
    }

    /// Giving up an outstanding answer's expert is an invariant violation.
    #[test]
    #[should_panic(expected = "gives up its expert from Answering")]
    fn an_outstanding_answer_never_gives_up_its_expert() {
        let mut tier = tier(1, 1, ONE, Resolution::AfterStore);
        let Claim::Taken(grant) = tier.claim_demand((0, 1), Demand::Lacked, reads(&[]), |_| false, (at(5), 1)) else { panic!("ready") };
        tier.landed(grant.record(), (0, 1), None, Kept::Alone);
        tier.unkeep(0);
    }
}
