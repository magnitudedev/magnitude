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
use super::reuse::Score;
use super::staging::Staged;
use super::watcher::{Step, Unbegun};
use std::collections::{BTreeSet, HashMap};

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

/// How the tier is ranked: an expert's reuse beside a read the ranking may
/// not have brought yet (`Reuse::rank`).
pub(super) trait Rank: Fn(Staged, Option<Step>) -> Score {}

impl<F: Fn(Staged, Option<Step>) -> Score> Rank for F {}

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
/// expert's reuse beside its landing's `read`, and the expert (a tie by the
/// expert, never by where it lies).
fn taking(kept: Kept, staged: Staged, read: Option<Step>, rank: &impl Rank) -> (Kept, Score, Staged) {
    (kept, rank(staged, read), staged)
}

pub(super) struct HostRecords {
    /// What each record holds, changed only by [`Self::set`].
    records: Vec<Host>,
    /// The indexes of `records`, each kept by [`Self::set`] alone: the
    /// ready records (a landing takes the last), the graced ones, the
    /// record keeping each expert, the record each landing in flight
    /// takes, the outstanding answers, the experts of the group's layers no
    /// record keeps or lands, per layer the experts held alone, and the
    /// records claimed.
    ready: Vec<u32>,
    graced: Vec<u32>,
    answering: Vec<u32>,
    kept: HashMap<Staged, u32>,
    landing: HashMap<Staged, u32>,
    lacking: BTreeSet<Staged>,
    alone: Vec<usize>,
    claimed: usize,
    /// How the steps resolving the records' map see an unmapping.
    resolution: Resolution,
    /// The bottom records the conveyor's lending reserves for it (none but
    /// of its lender, `planning::Lender`), which no other claim takes.
    lent: u32,
    /// Each layer's landing map, of the group's own layers, whose one
    /// writer these records are; none without a device.
    maps: Vec<Option<LandingMap>>,
}

impl HostRecords {
    /// `records` records, every one ready, of a group whose layers are
    /// `members`, each (layer, its experts, its landing map), every expert
    /// lacked, of a bank of `layers` streamed layers, whose map steps
    /// resolve by `resolution`.
    pub fn new(records: u32, layers: usize, members: Vec<(usize, usize, Option<LandingMap>)>, resolution: Resolution, lent: u32) -> Self {
        let lacking = members.iter().flat_map(|&(layer, experts, _)| (0..experts).map(move |expert| (layer, expert))).collect();
        let mut maps = vec![None; layers];
        for (layer, _, map) in members {
            maps[layer] = map;
        }
        let mut host = Self {
            records: Vec::new(),
            ready: Vec::new(),
            graced: Vec::new(),
            answering: Vec::new(),
            kept: HashMap::new(),
            landing: HashMap::new(),
            lacking,
            alone: vec![0; layers],
            claimed: 0,
            resolution,
            lent,
            maps,
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
                let at = self.ready.iter().rposition(|&ready| ready == record).expect("a ready record is listed ready");
                self.ready.swap_remove(at);
            }
            Host::Landing { staged, .. } => {
                self.landing.remove(&staged);
            }
            Host::Kept(Whole { staged, kept, .. }) | Host::Answering(Whole { staged, kept, .. }, _) => {
                self.kept.remove(&staged);
                if kept == Kept::Alone {
                    self.alone[staged.0] -= 1;
                }
                if matches!(old, Host::Answering(..)) {
                    self.answering.retain(|&answering| answering != record);
                }
            }
            Host::Graced { .. } => self.graced.retain(|&graced| graced != record),
            Host::Claimed => self.claimed -= 1,
        }
        match content {
            Host::Ready => self.ready.push(record),
            Host::Landing { staged, .. } => assert!(self.landing.insert(staged, record).is_none(), "{staged:?} lands twice"),
            Host::Kept(Whole { staged, kept, .. }) | Host::Answering(Whole { staged, kept, .. }, _) => {
                assert!(self.kept.insert(staged, record).is_none(), "{staged:?} kept twice");
                if kept == Kept::Alone {
                    self.alone[staged.0] += 1;
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
                }
                if let Some(entered) = entered {
                    assert!(self.lacking.remove(&entered), "{entered:?} is a lacked expert of the group's layers");
                }
            }
        }
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

    /// The record keeping `staged` whole, if one does.
    pub fn kept_record(&self, staged: Staged) -> Option<u32> {
        self.kept.get(&staged).copied()
    }

    /// Whether a record keeps `staged` whole, alone or banked.
    pub fn keeps(&self, staged: Staged) -> bool {
        self.kept.contains_key(&staged)
    }

    /// Whether a landing of `staged` is in flight.
    pub fn lands(&self, staged: Staged) -> bool {
        self.landing.contains_key(&staged)
    }

    /// Where `staged` is: kept, landing or lacked.
    pub fn located(&self, staged: Staged) -> Located {
        match (self.kept.get(&staged), self.lands(staged)) {
            (Some(&record), _) => Located::Kept(record),
            (None, true) => Located::Landing,
            (None, false) => Located::Lacked,
        }
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
    fn place(&self, record: u32, rank: &impl Rank) -> Option<(Kept, Score, Staged)> {
        match self.records[record as usize] {
            Host::Kept(Whole { staged, kept, read, .. }) => Some(taking(kept, staged, read, rank)),
            _ => None,
        }
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
    fn taking_order(&self, rank: &impl Rank, answers: bool, admits: impl Fn(u32) -> bool) -> Vec<u32> {
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
        self.set(record, Host::Answering(whole, past));
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
    pub fn lacking<K: Ord>(&self, rank: impl Rank, count: usize, skip: impl Fn(Staged) -> bool, order: impl Fn(Score, Staged) -> K) -> Vec<(K, Staged)> {
        let mut lacked = self
            .lacking
            .iter()
            .filter(|&&staged| !skip(staged))
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
    /// file.
    pub fn complete(&self) -> bool {
        self.lacking.is_empty() && self.landing.is_empty()
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
            Located::Lacked if self.takes(keep) => Claim::Taken(self.take_lacked(staged, None)),
            Located::Lacked => Claim::Short,
        }
    }

    /// Take the last ready record for a landing of `staged`, which the
    /// tier lacks, answering the request at `answer` where a demand takes
    /// it: its grant.
    fn take_lacked(&mut self, staged: Staged, answer: Option<Point>) -> Grant {
        let &record = self.ready.last().expect("a record taken is ready");
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
        self.set(record, answer.map_or(Host::Kept(whole), |past| Host::Answering(whole, past)));
    }

    /// Ready record `record` already holds `staged` whole (a bank imported
    /// whole into the one store, each expert at its home, before any step):
    /// kept, as a landing of it, mapped from the first step not yet begun.
    pub fn keep(&mut self, record: u32, staged: Staged) {
        assert_eq!(self.records[record as usize], Host::Ready, "a kept home is ready");
        self.set(record, Host::Kept(Whole { staged, kept: Kept::Alone, read: None, holds: 0 }));
        self.store(staged, Some(record));
    }

    /// Landing `record`'s read failed: it mapped nothing, and is ready
    /// again; its expert is lacked again.
    pub fn unlanded(&mut self, record: u32) {
        assert!(matches!(self.records[record as usize], Host::Landing { .. }), "a landing in flight into record {record}");
        self.set(record, Host::Ready);
    }

    /// The kept records the reserve unmaps now ([`Self::refill`]): those not
    /// held, first in taking order at `rank`, until the ready, the graced
    /// and the lent records a prompt group holds claimed (which it returns
    /// ready, so no decode expert elsewhere is unmapped for them) make
    /// `reserve`, or the experts lacked where fewer (a tier that keeps every
    /// expert keeps no reserve: no landing can be demanded of it), or none
    /// is left; the lent records among them where `lent` (no prompt group
    /// held or awaited them as this read), each with the expert it keeps.
    /// It reads the ranking, and changes nothing.
    pub fn refill_victims(&self, reserve: usize, rank: impl Rank, lent: bool) -> Vec<(u32, Staged)> {
        let short = |unmapped: usize| self.short_of(reserve, unmapped);
        if !short(0) {
            return Vec::new();
        }
        let mut victims = Vec::new();
        for record in self.taking_order(&rank, true, |record| (lent || record >= self.lent) && self.unheld(record)) {
            if !short(victims.len()) {
                break;
            }
            let staged = self.records[record as usize].whole().expect("a record in taking order keeps an expert").staged;
            victims.push((record, staged));
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
    }

    /// Claim a demand landing of `staged` ([`Self::claim`] leaving no
    /// reserve), the device parked at `parked` (a group of `steps` steps):
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
    pub fn claim_demand(&mut self, staged: Staged, rank: impl Rank, read: impl Fn(u32) -> bool, (parked, steps): (Point, u64)) -> Claim {
        match self.located(staged) {
            Located::Kept(record) => return Claim::Kept(record),
            Located::Landing => return Claim::Landing,
            Located::Lacked if self.takes(0) => return Claim::Taken(self.take_lacked(staged, Some(parked))),
            Located::Lacked => {}
        }
        let first = (0..self.records())
            .filter(|&record| self.unheld(record) && !read(record))
            .filter_map(|record| Some((self.place(record, &rank)?, record)))
            .min();
        let Some((_, record)) = first else {
            return Claim::Short;
        };
        let unmapped = self.unkeep(record);
        self.unmap_from(unmapped, record, parked.first_unresolved(unmapped.0, steps));
        self.set(record, Host::Ready);
        Claim::Taken(self.take_lacked(staged, Some(parked)))
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
        for record in self.answering.clone() {
            let Host::Answering(whole, past) = self.records[record as usize] else {
                unreachable!("an answering record is listed answering");
            };
            if past < progress {
                self.set(record, Host::Kept(whole));
            }
        }
        let resolution = self.resolution;
        self.free(|_, _, _, grace, answer| resolution.bound(grace, progress) < progress && answer.is_none_or(|answer| answer < progress))
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

    /// Shrink to `records` records between flights, no landing in flight
    /// and no step running (so no graced record is read): it drops the empty
    /// records first, then the kept first in taking order at `rank`, keeping
    /// the most valuable, and each kept one at or past `records` moves to a
    /// record below it that the rest leave, mapped there from the first
    /// step not yet begun, each dropped one unmapped. The moves, (from, to),
    /// whose bytes the caller copies before any step reads them.
    pub fn shrink(&mut self, records: u32, rank: impl Rank) -> Vec<(u32, u32)> {
        let count = self.records();
        assert!(records <= count, "a host tier of {count} records shrinks to {records}");
        assert!(self.landing.is_empty(), "a host tier shrinks with no landing in flight");
        assert!((0..count).all(|record| self.unheld(record)), "a host tier shrinks with no copy holding a record");
        assert!(self.claimed == 0, "a host tier shrinks with no prompt group holding a record");
        assert!(records >= self.lent, "a host tier shrinks past the conveyor's lent records");
        assert!(self.answering.is_empty(), "a host tier shrinks with no request's answer outstanding");
        let order = self.taking_order(&rank, false, |_| true);
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
            let moved = self.records[from as usize];
            let staged = moved.present().expect("a kept record keeps its expert");
            self.set(from, Host::Ready);
            self.set(to, moved);
            self.store(staged, Some(to));
        }
        assert!(self.records[records as usize..].iter().all(|&content| content == Host::Ready), "a shrink keeps nothing past it");
        self.records.truncate(records as usize);
        self.ready.retain(|&record| record < records);
        self.ready.sort_unstable();
        moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reserve's turn on `tier` ([`HostRecords::refill_victims`] then
    /// [`HostRecords::refill`]), a prompt group holding the lent records:
    /// `latest` read only where it unmaps any.
    fn refill(tier: &mut HostRecords, reserve: usize, rank: impl Rank, at: &Unbegun, latest: impl FnOnce() -> (Step, Step)) {
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
        HostRecords::new(records, layers, members.iter().map(|&(layer, experts)| (layer, experts, None)).collect(), resolution, 0)
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
        let cpu = seismic::Devices::discover(&[]).unwrap().open_backend(seismic::BackendName::Cpu).unwrap();
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
        let tier = HostRecords::new(records, layers, members, resolution, 0);
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
        fn reclaim(&mut self, staged: Staged, rank: impl Rank, parked: Point, steps: u64) -> Option<(u32, Option<Staged>)> {
            let kept = self.kept.clone();
            match self.claim_demand(staged, rank, |_| false, (parked, steps)) {
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
    #[test]
    fn a_claim_takes_a_record_only_for_a_lacked_expert() {
        let mut tier = tier(4, 1, ONE, Resolution::BeforeStamp);
        let record = tier.take((0, 1), 0).unwrap();
        assert!(matches!(tier.claim((0, 1), 0), Claim::Landing), "a second landing of it claims the first");
        assert!(matches!(tier.claim_demand((0, 1), reads(&[]), |_| false, (at(1), 1)), Claim::Landing));
        tier.landed(record, (0, 1), None, Kept::Alone);
        assert!(matches!(tier.claim((0, 1), 0), Claim::Kept(kept) if kept == record));
        let other = tier.take((0, 2), 0).unwrap();
        tier.unlanded(other);
        assert_eq!(tier.located((0, 2)), Located::Lacked, "a failed landing lacks it again");
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
        assert_eq!(grant.record(), 2);
        assert_eq!(tier.take((0, 6), 2), None, "the reserve is the demand's");
        assert_eq!(maps.held((0, 5)), None, "unmapped until its bytes are whole");
        grant.map(None);
        tier.landed(2, (0, 5), None, Kept::Alone);
        assert_eq!(maps.held((0, 5)), Some(2), "its grant mapped it");
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
        let grants = [10, 11, 12, 13].map(|expert| tier.grant_of((0, expert), 0).unwrap());
        assert_eq!(grants.each_ref().map(Grant::record), [3, 2, 1, 0]);
        for (grant, expert) in grants.iter().zip([10, 11, 12, 13]) {
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
        let Claim::Taken(first) = tier.claim_demand((0, 1), &rank, |_| false, (request, 1)) else { panic!("the one record is ready") };
        first.map(None);
        tier.landed(first.record(), (0, 1), Some(5), Kept::Alone);
        assert!(matches!(tier.records[0], Host::Answering(_, past) if past == request), "an outstanding answer");
        assert!(!tier.claim_record_of_lent_free(), "no prompt group's claim takes it");
        refill(&mut tier, 1, &rank, &Unbegun::at(6), || (5, 5));
        assert_eq!(maps.held((0, 1)), None, "the reserve unmaps it");
        assert!(matches!(tier.records[0], Host::Graced { answer: Some(answer), .. } if answer == request), "graced with its answer");
        assert!(!tier.release((request, 1), |_| false), "its request's park frees it not");
        assert!(matches!(tier.claim_demand((0, 2), &rank, |_| false, (request, 1)), Claim::Short), "the second miss waits");
        assert!(!tier.retire(request), "the device at the request still reads it");
        assert!(tier.retire(at(6)), "past the request it is free");
        assert!(matches!(tier.claim_demand((0, 2), &rank, |_| false, (at(6), 1)), Claim::Taken(_)), "then a later demand takes it");
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
        let Claim::Taken(grant) = tier.claim_demand((0, 1), reads(&[]), |_| false, (at(5), 1)) else { panic!("ready") };
        tier.landed(grant.record(), (0, 1), None, Kept::Alone);
        tier.unkeep(0);
    }
}
