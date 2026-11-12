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
//! - graced: unmapped from the first step not yet begun, `until`, so
//!   layer `l`'s entries of a step from its mapping's on (`from`) and
//!   before `until` may still read it (only layer `l`'s route resolves its
//!   tier map, and only layer `l`'s entries and its CPU site read what it
//!   resolved): it holds nothing once each of those steps is past layer
//!   `l` or ended (`Steps::served`, [`HostRecords::retire`]), or at a park
//!   that no resolution after it names it ([`HostRecords::release`]). That
//!   is the one rule a record frees by: once the device is past the last
//!   resolution that may name it, which the map's [`Resolution`] bounds;
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
//!   holds toward itself ([`HostRecords::refill`]);
//! - seeding: on the one store, a prompt group's seed of an expert, whose
//!   bytes the device's seed copy writes from the group's scratch slot,
//!   until the seed retires ([`HostRecords::seeded`]); the expert stays
//!   lacked meanwhile, so no step waits on the seed.
//!
//! Every expert of the group's layers is kept, landing, or lacked. The tier
//! starts with every expert lacked, and keeps only what lands by use: a
//! step's miss, a hint, a prompt group's read, which on the one store its
//! last step seeds.
//!
//! The tier is inclusive of the bank: a promotion copies a record into the
//! bank and the record keeps its copy, so an expert the bank then evicts is
//! held in the tier alone again with no copy, and nothing is ever written
//! back. The host keeps a reserve of ready records ([`HostRecords::refill`]),
//! so a demand landing never waits for a grace: it unmaps the kept records
//! first in taking order: those that hold their layers past the layers'
//! shares of the records, then the experts held alone, then the banked
//! copies, whose experts the bank serves, each the least reused first.
//! Every order and share is the one ranking's (`Reuse`), which each call
//! that orders is handed (`rank`, [`Shares`]): the tier keeps no rank or
//! share of its own, so none goes stale. The tier
//! grows and shrinks by whole records (`ExpertBankPlan::host_records`).
//!
//! A kept record a copy reads (a prompt group's pull, or a promotion) is
//! held until the copy ends ([`HostRecords::hold`]): it keeps its expert,
//! mapped, and its place in the taking order, and the reserve never unmaps
//! it while held, so its bytes stay whole for the copy.
//!
//! A decode step cut in before a prompt group's layer (`Park::cut_in`)
//! runs while that group may still resolve its layers from the cut on.
//! Its park unmaps, restamps or reuses no record a step of that group may
//! still resolve (`Park::frees`, by the step each record's entry is mapped
//! from, `from`): that group's request at the layer may have been served,
//! reading in place what its map resolves. Its landings are mapped from
//! its own step on, so no earlier group reads them, and the prompt group
//! copies none ([`HostRecords::serves`]): the cut-ins recycle them among
//! themselves. They land into the ready records the pool sets aside for
//! them ([`HostRecords::set_pool`]), which no other landing takes, so a
//! cut-in never waits on the prompt group it runs inside.

use super::directory::{Park, Point};
use super::map::LandingMap;
use super::staging::Staged;
use super::watcher::{Step, Steps, Unbegun};
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

/// Where an expert of the group is: kept whole in a record, landing (a
/// landing of it is in flight, whoever's), or lacked. Every expert is
/// exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Located {
    Kept(u32),
    /// Its record graced ([`Host::Graced`]): whole, unmapped from a step
    /// on, still named for the steps before it that may resolve it. No
    /// landing takes the expert anew, which would map it past them; a
    /// request's demand takes the record back ([`HostRecords::answer`]).
    Graced(u32),
    Landing,
    Lacked,
}

/// The one map write a landing's claim allows: once its bytes are whole,
/// expert `staged` mapped to the record taken for it ([`Grant::map`]),
/// from step `from` on where given (a parked step's demand's, the first
/// step that has not resolved its layer, `Park::unresolved`), else from
/// the first step not yet begun. Only a claim makes one
/// ([`HostRecords::claim`]).
pub(in crate::experts) struct Grant {
    map: Option<LandingMap>,
    staged: Staged,
    record: u32,
    from: Option<Step>,
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

    /// Map its expert to its record, whole now, from the step its claim
    /// names (`LandingMap::map_from`): once.
    pub fn map(&self) {
        self.map_at(self.from);
    }

    /// Map its expert to its record from step `from` on where given, else
    /// from the first step not yet begun: once.
    fn map_at(&self, from: Option<Step>) {
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

/// Whose record a landing takes ([`HostRecords::claim`]): a hint's read
/// ahead's is the reserve's own (the reserve is a record for each read
/// ahead's staging record, `ExpertBankPlan::tier_reserve`), and stays among
/// the reserve's records until the landing ends, so the reserve replaces it
/// only once it lands and never for one that ends unlanded; every other
/// landing (a seed's, a prompt group's, a demand's, the covering tier's,
/// which need none kept for them) takes its record out of the reserve at
/// once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Taking {
    Ahead,
    Out,
}

/// How the tier is ranked: an expert's rank beside a read the ranking may
/// not have brought yet (`Reuse::rank`), in an order `K` that breaks the
/// reads' ties as the reads before broke them, never by the expert's index.
pub(super) trait Rank<K: Ord + Copy>: Fn(Staged, Option<Step>) -> K {}

impl<K: Ord + Copy, F: Fn(Staged, Option<Step>) -> K> Rank<K> for F {}

/// Each layer's share of the records, the ranking's (`Reuse::shares`),
/// which the taking order holds each layer to: it takes first the least
/// ranked of every layer holding experts past its share (kept, answering
/// or landing), as many as it holds past it, then the rest by rank. So a
/// layer's misses take its own records back, as the ranking measures each
/// layer's share, and never another layer's less ranked experts, which a
/// pool of the group's layers by rank would take but no measure counted.
/// None where the records are no layer's share (a host tier beside the
/// bank's own records, whose shares the ranking measures): by rank alone.
/// `landing`: the layer of the expert a demand takes a record for, which
/// holds one more once it lands.
#[derive(Clone, Copy)]
struct Shares<'a> {
    of: Option<&'a [usize]>,
    landing: Option<usize>,
}

/// An expert a record keeps whole, mapped: as `kept`, ranked beside
/// `read`, held by `holds` copies reading it, its entry mapped for no step
/// before `from` (a parked step's landing's: none before the first step
/// that had not resolved its layer; else none known, 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Whole {
    staged: Staged,
    kept: Kept,
    read: Option<Step>,
    holds: u32,
    from: Step,
}

/// What a host record holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Host {
    /// Nothing any step reads: a landing takes it at once.
    Ready,
    /// Taken for a landing of the expert in flight, mapped for no step
    /// before `from` once landed ([`Whole`]); where a request's demand took
    /// it, the request at `answer` (its tier wait), which the landing
    /// answers with it; the reserve's while it lands where `taking` is a
    /// read ahead's.
    Landing { staged: Staged, answer: Option<Point>, from: Step, taking: Taking },
    /// Its expert whole, mapped.
    Kept(Whole),
    /// Its expert whole, mapped, and the outstanding answer of a request:
    /// the device reads it at the request's point or its latest one,
    /// `past`, until it is past there ([`HostRecords::retire`]). No claim
    /// takes it until then; the reserve may unmap it, graced with its
    /// answer.
    Answering(Whole, Point),
    /// Unmapped from step `until` on, the expert it held whole (`whole`)
    /// still named by the steps from `whole.from` before it that may
    /// resolve it until each is past its layer or ended, and read as the
    /// outstanding answer of the request at `answer`, if it was one, until
    /// the device is past there: no park before frees it. It still holds
    /// its expert ([`Host::present`]): no other record lands it meanwhile,
    /// and a request's demand takes it back as it was.
    Graced { whole: Whole, until: Step, answer: Option<Point> },
    /// A prompt group's, for its life.
    Claimed,
    /// A prompt group's seed of `staged`, whose bytes the device's seed
    /// copy writes, until the seed retires ([`HostRecords::seeded`]). It
    /// holds nothing present: the expert stays lacked meanwhile, so a step
    /// that misses it (a cut-in the group's end waits behind among them)
    /// lands it apart, never waiting on the seed.
    Seeding { staged: Staged },
}

impl Host {
    /// The expert it keeps, lands or graces: one no other record keeps,
    /// lands or graces.
    fn present(self) -> Option<Staged> {
        match self {
            Self::Landing { staged, .. } | Self::Kept(Whole { staged, .. }) | Self::Answering(Whole { staged, .. }, _) | Self::Graced { whole: Whole { staged, .. }, .. } => Some(staged),
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
    /// keeps or lands, per layer the experts held alone and the experts
    /// kept, answering or landing, and the records claimed.
    ready: BTreeSet<u32>,
    graced: Vec<u32>,
    answering: Vec<u32>,
    kept: HashMap<Staged, u32>,
    landing: HashMap<Staged, u32>,
    ahead: usize,
    lacking: BTreeSet<Staged>,
    /// How many experts the group's layers have: what its records past the
    /// lent ones must hold for it to cover them ([`Self::covers`]).
    experts: usize,
    alone: Vec<usize>,
    holding: Vec<usize>,
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
    /// The ready records set aside for decode steps cut in during a prompt
    /// pass ([`Self::set_pool`]): every other landing leaves them ready.
    pool: usize,
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
    /// The step the victims were prepared at, and how many, and the shares
    /// they were prepared at, which a claim takes them by
    /// ([`Self::claim_prepared`]): the ranking's until it resizes them,
    /// which prepares again ([`Self::unprepare`]).
    prepared: Option<(Step, usize)>,
    shares: Option<Vec<usize>>,
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
        let experts = members.iter().map(|&(_, experts, _)| experts).sum();
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
            ahead: 0,
            lacking,
            experts,
            alone: vec![0; layers],
            holding: vec![0; layers],
            claimed: 0,
            pool: 0,
            resolution,
            progress: None,
            lent,
            maps,
            holdings: 0,
            victims: VecDeque::new(),
            prepared: None,
            shares: None,
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
            Host::Landing { staged, taking, .. } => {
                self.landing.remove(&staged);
                self.ahead -= usize::from(taking == Taking::Ahead);
                self.holding[staged.0] -= 1;
            }
            Host::Kept(Whole { staged, kept, .. }) | Host::Answering(Whole { staged, kept, .. }, _) => {
                self.kept.remove(&staged);
                self.holding[staged.0] -= 1;
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
            Host::Seeding { staged } => self.holding[staged.0] -= 1,
        }
        match content {
            Host::Ready => assert!(self.ready.insert(record), "record {record} is listed ready once"),
            Host::Landing { staged, taking, .. } => {
                assert!(self.landing.insert(staged, record).is_none(), "{staged:?} lands twice");
                self.ahead += usize::from(taking == Taking::Ahead);
                self.holding[staged.0] += 1;
            }
            Host::Kept(Whole { staged, kept, .. }) | Host::Answering(Whole { staged, kept, .. }, _) => {
                assert!(self.kept.insert(staged, record).is_none(), "{staged:?} kept twice");
                self.holding[staged.0] += 1;
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
            Host::Seeding { staged } => self.holding[staged.0] += 1,
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

    /// The grant of mapping `staged` to `record` once landed, from step
    /// `from` on where given.
    fn grant(&self, staged: Staged, record: u32, from: Option<Step>) -> Grant {
        Grant { map: self.maps[staged.0].clone(), staged, record, from, mapped: std::sync::atomic::AtomicBool::new(false) }
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

    /// Whether a landing of `staged` is in flight: a record taken for it,
    /// or a posted read no record has taken yet.
    pub fn lands(&self, staged: Staged) -> bool {
        self.located(staged) == Located::Landing
    }

    /// Whether the prompt group whose last step is `last` reads `staged`
    /// from these records (its tier copy, `Directory::hold_for_pass`): kept
    /// or landing in a record whose entry may be mapped for a step through
    /// `last`, or read by a posted read no record has taken yet. A cut-in's
    /// landing is mapped from its own step on (`Park::unresolved`), past
    /// every step of the group it runs inside, which a copy would hold until
    /// that group is past: the group reads it from the file instead, so the
    /// cut-ins recycle it among themselves.
    pub fn serves(&self, staged: Staged, last: Step) -> bool {
        match (self.kept.get(&staged), self.landing.get(&staged)) {
            (Some(&record), _) => self.records[record as usize].whole().is_some_and(|whole| whole.from <= last),
            (None, Some(&record)) => matches!(self.records[record as usize], Host::Landing { from, .. } if from <= last),
            (None, None) => self.inbound.claimed(staged) && self.graced_record(staged).is_none(),
        }
    }

    /// Where `staged` is: kept, graced, landing (a record taken for its
    /// landing, or its word claimed by a posted read, [`InboundTable`]) or
    /// lacked.
    pub fn located(&self, staged: Staged) -> Located {
        match self.kept.get(&staged) {
            Some(&record) => Located::Kept(record),
            // A graced record holds its word too ([`Host::present`]).
            None if self.inbound.claimed(staged) => self.graced_record(staged).map_or(Located::Landing, Located::Graced),
            None => Located::Lacked,
        }
    }

    /// The graced record that holds `staged`, if one does: at most one
    /// ([`Host::present`]).
    fn graced_record(&self, staged: Staged) -> Option<u32> {
        self.graced.iter().copied().find(|&record| self.records[record as usize].present() == Some(staged))
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

    /// The records lent to the prompt groups, the bottom ones: the conveyor
    /// and the scratch where the tier is the lender's one store.
    pub fn lent(&self) -> u32 {
        self.lent
    }

    /// The experts `layer` holds: kept, answering, landing or seeding.
    pub fn holding(&self, layer: usize) -> usize {
        self.holding[layer]
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

    /// The records a prompt group's seed of `layer` may take
    /// ([`Self::seed`]), each with its place in the taking order at `rank`,
    /// least first: the ready records past the lent ones beyond the
    /// reserve's `reserve` and the pool set aside for cut-ins
    /// ([`Self::short_of`]: a seed never takes what the reads ahead and the
    /// cut-ins are kept ready for, so the reserve never refills by unmapping
    /// what outranks a seed) (`None`, in record order: they hold nothing,
    /// so which one a seed takes decides nothing later), then the kept
    /// records past the lent ones that no copy holds, of experts `pinned`
    /// does not name, in taking order at `shares` with the seed's layer
    /// holding one more ([`Shares`]): those of layers past their shares
    /// first, then the rest of the seed's layer by rank. Another layer's
    /// record within its share is never taken, so a seed takes no layer's
    /// share. An outstanding answer, a graced record, a landing, a seed and
    /// a claimed record are never taken.
    pub fn seed_order<K: Ord + Copy>(
        &self,
        layer: usize,
        reserve: usize,
        rank: impl Rank<K>,
        shares: &[usize],
        pinned: impl Fn(Staged) -> bool,
    ) -> Vec<(Option<(Kept, K, Staged)>, u32)> {
        let ready = self.ready.range(self.lent..).count().min(self.reserved().saturating_sub(reserve + self.pool));
        let free = self.ready.range(self.lent..).take(ready).map(|&record| (None, record));
        let kept = (self.lent..self.records())
            .filter(|&record| self.unheld(record))
            .filter_map(|record| Some((self.place(record, &rank)?, record)))
            .filter(|&((.., staged), _)| !pinned(staged))
            .collect::<Vec<_>>();
        let (past, within) = self.split_by_shares(kept, Shares { of: Some(shares), landing: Some(layer) });
        let own = within.into_iter().filter(|&((.., (of, _)), _)| of == layer);
        free.chain(past.into_iter().chain(own).map(|(place, record)| (Some(place), record))).collect()
    }

    /// A prompt group's seed of lacked `staged` takes record `record`
    /// ([`Self::seed_order`]) until the device's seed copy is retired
    /// ([`Self::seeded`]), seeding it ([`Host::Seeding`]). A kept record
    /// gives up its expert, unmapped from `until`, the group's first step:
    /// every step before it runs before the device writes the record (the
    /// device runs a group after every step begun before it, and a group
    /// cut in after the group's first step runs after its unmapping), and
    /// none of the group's steps reads it (a seed takes no record of an
    /// expert its layer's merge set routes, and a later layer's hit it
    /// takes is read instead, `Scratch::demote`). The expert it evicts, if
    /// it held one.
    pub fn seed(&mut self, staged: Staged, record: u32, until: Step) -> Option<Staged> {
        assert_eq!(self.located(staged), Located::Lacked, "a seed of {staged:?}, which the records keep or land");
        assert!(record >= self.lent, "a seed of {staged:?} takes lent record {record}");
        let evicted = match self.records[record as usize] {
            Host::Ready => None,
            Host::Kept(_) => {
                let evicted = self.unkeep(record);
                self.unmap_from(evicted, record, until);
                self.set(record, Host::Ready);
                Some(evicted)
            }
            other => panic!("a seed of {staged:?} takes record {record}, which holds {other:?}"),
        };
        self.set(record, Host::Seeding { staged });
        evicted
    }

    /// The seed of `staged` into `record` ([`Self::seed`]) retired: where
    /// `committed` (its step ended whole with the device's commit) and the
    /// expert is still lacked, and no posted read claimed it, kept, ranked
    /// beside `read`, the step that seeded it, and mapped from `at`'s first
    /// step not yet begun, under the lock every step begins under; else
    /// (uncommitted, or landed apart meanwhile) the record is ready again.
    /// Whether it keeps the seed.
    pub fn seeded(&mut self, record: u32, staged: Staged, read: Step, committed: bool, at: &Unbegun) -> bool {
        assert_eq!(self.records[record as usize], Host::Seeding { staged }, "the seed of {staged:?} retires from the record it took");
        let keeps = committed && self.located(staged) == Located::Lacked && self.inbound.claim(staged);
        if !keeps {
            self.set(record, Host::Ready);
            return false;
        }
        let from = at.next();
        self.set(record, Host::Kept(Whole { staged, kept: Kept::Alone, read: Some(read), holds: 0, from }));
        if let Some(map) = &self.maps[staged.0] {
            map.store_unbegun(staged.1, Some(record), Some(at));
        }
        true
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

    /// The experts `layer` holds past its share at `shares`.
    fn past(&self, layer: usize, shares: Shares) -> usize {
        shares.of.map_or(0, |of| (self.holding[layer] + usize::from(shares.landing == Some(layer))).saturating_sub(of[layer]))
    }

    /// Whether a demand may take a record of `layer`'s first at `shares`
    /// (its landing's counted): every layer's where no share binds, else
    /// only one past its share's.
    fn yields(&self, layer: usize, shares: Shares) -> bool {
        shares.of.is_none() || self.past(layer, shares) > 0
    }

    /// `order`, kept records each with its place in the taking order at a
    /// rank, in taking order at `shares` ([`Shares`]): those past their
    /// layers' shares, then the rest, each in place order.
    fn by_shares<P: Ord + Copy>(&self, order: Vec<((Kept, P, Staged), u32)>, shares: Shares) -> Vec<((Kept, P, Staged), u32)> {
        let (mut first, rest) = self.split_by_shares(order, shares);
        first.extend(rest);
        first
    }

    /// [`Self::by_shares`]'s two parts: those past their layers' shares,
    /// and the rest, each in place order.
    #[allow(clippy::type_complexity)]
    fn split_by_shares<P: Ord + Copy>(&self, mut order: Vec<((Kept, P, Staged), u32)>, shares: Shares) -> (Vec<((Kept, P, Staged), u32)>, Vec<((Kept, P, Staged), u32)>) {
        order.sort_unstable();
        let mut past = (0..self.holding.len()).map(|layer| self.past(layer, shares)).collect::<Vec<_>>();
        order.into_iter().partition(|&((.., (layer, _)), _)| {
            let taken = past[layer] > 0;
            past[layer] -= usize::from(taken);
            taken
        })
    }

    /// The kept records `admits`, first in taking order at `rank` and
    /// `shares`, the outstanding answers among them where `answers`.
    fn taking_order<K: Ord + Copy>(&self, rank: &impl Rank<K>, shares: Shares, answers: bool, admits: impl Fn(u32) -> bool) -> Vec<u32> {
        let place = |record: u32| match self.records[record as usize] {
            Host::Answering(Whole { staged, kept, read, .. }, _) if answers => Some(taking(kept, staged, read, rank)),
            _ => self.place(record, rank),
        };
        let order = (0..self.records())
            .filter(|&record| admits(record))
            .filter_map(|record| Some((place(record)?, record)))
            .collect::<Vec<_>>();
        self.by_shares(order, shares).into_iter().map(|(_, record)| record).collect()
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

    /// Kept or graced record `record` is the answer of the request at
    /// `past` (its tier wait), published now: an outstanding answer until
    /// the device is past there ([`Self::retire`]), the latest of its
    /// requests'. A graced one is taken back as it was: its entry mapped
    /// again from its `from`, so every step that named it still does and
    /// those after it resolve it too, and no landing reads it anew.
    pub fn answer(&mut self, record: u32, past: Point) {
        let content = self.records[record as usize];
        let (whole, past) = match content {
            Host::Kept(whole) => (whole, past),
            Host::Answering(whole, answering) => (whole, answering.max(past)),
            Host::Graced { whole, answer, .. } => {
                if let Some(map) = &self.maps[whole.staged.0] {
                    map.map_from(whole.staged.1, record, Some(whole.from));
                }
                (whole, answer.map_or(past, |answer| answer.max(past)))
            }
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

    /// Whether the records past the lent ones are as many as the group's
    /// experts: the tier can hold every one, so there is no choice of which
    /// to keep.
    pub fn covers(&self) -> bool {
        self.records() as usize - self.lent as usize >= self.experts
    }

    /// The experts the tier lacks that `skip` does not pass over, in layer
    /// and expert order.
    pub fn lacking_unskipped<'a>(&'a self, skip: impl Fn(Staged) -> bool + 'a) -> impl Iterator<Item = Staged> + 'a {
        self.lacking.iter().copied().filter(move |&staged| !skip(staged))
    }

    /// Whether every expert of the group is kept, mapped for every step
    /// from now: none lacked, landing, or graced (unmapped from a step on,
    /// though its record holds it). No miss of it reads the file until a
    /// record gives one up.
    pub fn complete(&self) -> bool {
        self.lacking.is_empty() && self.landing.is_empty() && self.graced.is_empty()
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
    /// taken for this landing, leaving the reserve's records at least `keep`
    /// (none for a read ahead), the pool
    /// beside them ([`Self::takes`]), or the experts still lacked after it
    /// where those are fewer (a landing beyond them could never be
    /// demanded); else short. The record is taken as `taking` says. Every
    /// landing of an expert, whoever lands it, claims it here, so no two
    /// land it at once.
    pub fn claim(&mut self, staged: Staged, keep: usize, taking: Taking) -> Claim {
        match self.located(staged) {
            // Whole in its record already: nothing to land.
            Located::Kept(record) | Located::Graced(record) => Claim::Kept(record),
            Located::Landing => Claim::Landing,
            Located::Lacked if self.takes(keep) => match self.inbound.claim(staged) {
                true => Claim::Taken(self.take_lacked(staged, None, taking)),
                // A posted read claimed it since it was located.
                false => Claim::Landing,
            },
            Located::Lacked => Claim::Short,
        }
    }

    /// Take a ready record for a landing of `staged`, which the tier lacks
    /// and whose word its caller claimed, answering the request parked at
    /// `park` where a demand takes it, mapped from the first step whose
    /// resolution of its layer is not before the park: its grant. Every landing takes its record here, the lowest ready one
    /// past the lent records (a lent one only where none past them is
    /// ready, as a prompt group's claim waits for a landing into it). A
    /// shrink keeps a landing in place and cuts only above it
    /// ([`Self::shrink_floor`]), and from the top down; so a landing lies
    /// past a cut only where every record past the lent ones below it was
    /// taken as it began: right after a growth a landing fills the records
    /// below the growth before it touches the new ones a shrink cuts first.
    fn take_lacked(&mut self, staged: Staged, park: Option<&Park>, taking: Taking) -> Grant {
        let record = self.ready.range(self.lent..).next().or_else(|| self.ready.first()).copied().expect("a record taken is ready");
        let from = park.map(|park| park.unresolved(staged.0));
        self.set(record, Host::Landing { staged, answer: park.map(|park| park.point), from: from.unwrap_or(0), taking });
        self.grant(staged, record, from)
    }

    /// Whether [`Self::claim`] of `keep` takes a record now: a ready one
    /// beyond the pool set aside, of the reserve's records
    /// ([`Self::reserved`]) beyond `keep` and the pool. A read ahead's
    /// landing keeps its record among the reserve's until it ends, so one
    /// that ends unlanded returns its record to a reserve that never
    /// replaced it: no record lies beyond the reserve for it to take.
    fn takes(&self, keep: usize) -> bool {
        self.ready_beyond(self.pool) && self.beyond(self.reserved(), keep + self.pool)
    }

    /// Whether a record is ready beyond `keep` ([`Self::beyond`]).
    fn ready_beyond(&self, keep: usize) -> bool {
        self.beyond(self.ready.len(), keep)
    }

    /// Whether `count` records exceed `keep`, or the experts still lacked
    /// after a landing where those are fewer (a landing beyond them could
    /// never be demanded).
    fn beyond(&self, count: usize, keep: usize) -> bool {
        self.lacked().checked_sub(1).is_some_and(|after| count > keep.min(after))
    }

    /// The reserve's records, which it keeps ([`Self::short_of`]) and a
    /// claim takes beyond ([`Self::takes`]): the ready, the graced, the
    /// claimed (every one a lent one, [`Self::claim_record`]) and those a
    /// read ahead's landing in flight took ([`Taking::Ahead`]), which its
    /// end either maps, leaving the reserve, or returns ready.
    fn reserved(&self) -> usize {
        self.ready.len() + self.graced.len() + self.claimed + self.ahead
    }

    /// The ready records a demand at `park` leaves: the pool, but where the
    /// parked step is a cut-in, whose pool it is.
    fn left_by(&self, park: &Park) -> usize {
        if park.cut_in() { 0 } else { self.pool }
    }

    /// Set `records` ready records aside for the decode steps cut in during
    /// a prompt pass from now (none once it ends): every landing but a
    /// cut-in's demand leaves them ready, and the reserve keeps them
    /// ([`Self::short_of`]).
    pub fn set_pool(&mut self, records: u32) {
        self.pool = records as usize;
    }

    /// Landing `record` holds `staged` whole, and its landing mapped it
    /// ([`Grant::map`]): kept as `kept` (a copy of an expert the bank
    /// holds is banked, so the tier never holds alone what the bank holds),
    /// ranked beside `read`, the step whose miss it landed for (none where no
    /// step read it); the outstanding answer of the request whose demand
    /// took it, if one did.
    pub fn landed(&mut self, record: u32, staged: Staged, read: Option<Step>, kept: Kept) {
        let Host::Landing { staged: landing, answer, from, .. } = self.records[record as usize] else {
            panic!("{staged:?} lands into record {record}, which no landing took");
        };
        assert_eq!(landing, staged, "{staged:?} lands into the record taken for it");
        let whole = Whole { staged, kept, read, holds: 0, from };
        self.set(record, self.answering(whole, answer));
    }

    /// Ready record `record` already holds `staged` whole (a bank imported
    /// whole into the one store, each expert at its home, before any step):
    /// kept, as a landing of it, mapped from the first step not yet begun.
    pub fn keep(&mut self, record: u32, staged: Staged) {
        assert_eq!(self.records[record as usize], Host::Ready, "a kept home is ready");
        assert!(self.inbound.claim(staged), "a home of {staged:?} is kept once, before any read");
        self.set(record, Host::Kept(Whole { staged, kept: Kept::Alone, read: None, holds: 0, from: 0 }));
        self.store(staged, Some(record));
    }

    /// Landing `record`'s read failed: it mapped nothing, and is ready
    /// again; its expert is lacked again.
    pub fn unlanded(&mut self, record: u32) {
        assert!(matches!(self.records[record as usize], Host::Landing { .. }), "a landing in flight into record {record}");
        self.set(record, Host::Ready);
    }

    /// The kept records the reserve unmaps now ([`Self::refill`]): of those
    /// not held, first in taking order at `rank`, until the reserve's
    /// records ([`Self::reserved`]: a prompt group's claimed lent records among them, which it returns
    /// ready, so no decode expert elsewhere is unmapped for them) make
    /// `reserve`, or the experts lacked where fewer (a tier that keeps every
    /// expert keeps no reserve: no landing can be demanded of it), or none
    /// is left; the lent records among them where `lent` (no prompt group
    /// held or awaited them as this read), each with the expert it keeps.
    /// It reads the ranking and its `shares`, and changes nothing.
    pub fn refill_victims<K: Ord + Copy>(&self, reserve: usize, rank: impl Rank<K>, shares: Option<&[usize]>, lent: bool) -> Vec<(u32, Staged)> {
        let short = |unmapped: usize| self.short_of(reserve, unmapped);
        let mut victims = Vec::new();
        if !short(0) {
            return victims;
        }
        let order = self
            .taking_order(&rank, Shares { of: shares, landing: None }, true, |record| (lent || record >= self.lent) && self.unheld(record))
            .into_iter()
            .map(|record| (record, self.records[record as usize].whole().expect("a record in taking order keeps an expert").staged));
        for victim in order {
            if !short(victims.len()) {
                break;
            }
            victims.push(victim);
        }
        victims
    }

    /// Whether the reserve's records ([`Self::reserved`]), with `unmapped`
    /// more, fall short of `reserve` and the pool set aside beside it, or of
    /// the experts lacked where fewer.
    pub fn short_of(&self, reserve: usize, unmapped: usize) -> bool {
        self.reserved() + unmapped < (reserve + self.pool).min(self.lacked() + unmapped)
    }

    /// Unmap the reserve's `victims` ([`Self::refill_victims`]), under the
    /// lock every step begins under, `at` the first step not yet begun: each
    /// unchanged since it was chosen in this turn of residency (a changed
    /// one is an invariant violation), a lent record only where `lent` (no
    /// prompt group holds or awaits them, read under this lock, else passed
    /// over), is unmapped from `at` and graced for the steps before it its
    /// entry may be mapped for, and an outstanding answer through its
    /// request too. A lent record is never graced for a group's own step,
    /// which would hold its claim behind it. The work here is a check, a map
    /// store and a state change a victim.
    pub fn refill(&mut self, victims: Vec<(u32, Staged)>, lent: bool, at: &Unbegun) {
        for (record, chosen) in victims {
            // Chosen in the same turn of residency, nothing between: only
            // the reservation may have changed, which a group takes as it
            // begins.
            let whole = self.records[record as usize].whole();
            assert!(whole.is_some_and(|whole| whole.staged == chosen && whole.holds == 0), "victim record {record} changed since it was chosen: {whole:?}");
            if !lent && record < self.lent {
                continue;
            }
            let (whole, answer) = match self.records[record as usize] {
                Host::Answering(whole, past) => (whole, Some(past)),
                Host::Kept(whole) => {
                    self.unkeep(record);
                    (whole, None)
                }
                other => unreachable!("victim record {record} holds {other:?}"),
            };
            if let Some(map) = &self.maps[whole.staged.0] {
                map.store_unbegun(whole.staged.1, None, Some(at));
            }
            self.set(record, Host::Graced { whole, until: at.next(), answer });
        }
    }

    /// Prepare a demand's victims ([`Self::victims`]): the first `depth`
    /// kept records no copy holds, in taking order at `rank` and `shares`,
    /// as no demand's landing counts yet: a claim takes them by the
    /// demand's ([`Self::claim_prepared`]). Run after a request's reads are
    /// submitted, so the scan is off every park's path; the order it leaves
    /// is a few passes old at most when a claim reads it, and each entry is
    /// checked again as it is taken.
    pub fn prepare_demand<K: Ord + Copy>(&mut self, rank: impl Rank<K>, shares: Option<&[usize]>, depth: usize, sequence: Step) {
        let order = (0..self.records())
            .filter(|&record| self.unheld(record))
            .filter_map(|record| Some((self.place(record, &rank)?, record)))
            .collect::<Vec<_>>();
        let order = self.by_shares(order, Shares { of: shares, landing: None });
        self.victims = order.into_iter().take(depth).map(|((.., staged), record)| (record, staged)).collect();
        self.prepared = Some((sequence, self.victims.len()));
        self.shares = shares.map(<[usize]>::to_vec);
    }

    /// The shares the victims were prepared at changed: prepare them again.
    pub fn unprepare(&mut self) {
        self.victims.clear();
        self.prepared = None;
    }

    /// Whether the victims want preparing again at step `sequence`: never
    /// prepared, some taken or passed over since, or prepared at an
    /// earlier step.
    pub fn unprepared(&self, sequence: Step) -> bool {
        self.prepared.is_none_or(|(at, count)| at < sequence || self.victims.len() < count)
    }

    /// [`Self::claim_demand`] from the prepared victims alone: the first
    /// whose layer holds past its share as the demand's landing counts
    /// (the shares they were prepared at), so the scan's own choice; none
    /// where it needs a scan, every prepared victim gone (since unmapped,
    /// landed into, held, read by the parked step, or one an earlier
    /// group's step may still resolve) or within its share.
    pub fn claim_prepared(&mut self, staged: Staged, demand: Demand, read: impl Fn(u32) -> bool, park: &Park) -> Option<Claim> {
        if let Some(claim) = self.demanded(staged, demand) {
            return Some(claim);
        }
        if self.ready_beyond(self.left_by(park)) {
            return Some(self.take_demanded(staged, demand, park));
        }
        let shares = Shares { of: self.shares.as_deref(), landing: Some(staged.0) };
        let record = loop {
            // A victim within its share stays for another layer's demand;
            // the first past its share is the scan's choice.
            let at = self.victims.iter().position(|&(_, (layer, _))| self.yields(layer, shares))?;
            let (record, kept) = self.victims.remove(at).expect("a victim found");
            let keeps = self.records[record as usize].present() == Some(kept);
            if keeps && self.reclaims(record, &read, park) {
                break record;
            }
        };
        if !self.claims(staged, demand) {
            self.victims.push_front((record, self.records[record as usize].present().expect("a victim keeps its expert")));
            return Some(Claim::Landing);
        }
        Some(self.take_victim(staged, record, park))
    }

    /// Whether a demand at `park` may take kept record `record` now: no
    /// copy holds it, the parked step did not resolve it (`read`), and no
    /// step of a group the parked one was cut in before may still resolve
    /// it ([`Park::frees`]). Never an outstanding answer.
    fn reclaims(&self, record: u32, read: impl Fn(u32) -> bool, park: &Park) -> bool {
        match self.records[record as usize] {
            Host::Kept(Whole { staged, holds: 0, from, .. }) => !read(record) && park.frees(staged.0, from, Step::MAX),
            _ => false,
        }
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
            (Demand::Lacked, Located::Graced(record)) => panic!("a demand of {staged:?} finds it graced in record {record}: its request is answered from the record first (`Directory::answer`)"),
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
    /// request parked at `park`; landing, where a posted read claimed it
    /// first.
    fn take_demanded(&mut self, staged: Staged, demand: Demand, park: &Park) -> Claim {
        match self.claims(staged, demand) {
            true => Claim::Taken(self.take_lacked(staged, Some(park), Taking::Out)),
            false => Claim::Landing,
        }
    }

    /// Unmap kept record `record` from the first step whose resolution of
    /// its expert's layer is not before `park`, and take it for `staged`,
    /// whose word its caller claimed.
    fn take_victim(&mut self, staged: Staged, record: u32, park: &Park) -> Claim {
        let unmapped = self.unkeep(record);
        self.unmap_from(unmapped, record, park.unresolved(unmapped.0));
        self.set(record, Host::Ready);
        Claim::Taken(self.take_lacked(staged, Some(park), Taking::Out))
    }

    /// Claim a demand landing of `staged` for `demand` ([`Self::claim`]
    /// leaving no reserve but the pool, which a cut-in's park takes too, or
    /// a record for the read the request service's look posted,
    /// [`Demand`]), the device parked at `park`: where it is lacked and no
    /// record is ready, unmap the first kept record, in taking order at
    /// `rank` and `shares` as this landing counts, that the park may take
    /// ([`Self::reclaims`]: not held, not
    /// `read`, what the parked step resolved at its layer,
    /// `decode::parked_reads`, and none an earlier group's step may still
    /// resolve), from the first step whose resolution of its expert's
    /// layer is not before the park ([`Park::unresolved`]): every
    /// resolution that may name it is then before the park, so it is ready
    /// at once, and taken, its unmapping stored before any landing writes
    /// it. Short where every kept record is held, read, another request's
    /// outstanding answer, or an earlier group's to resolve. The record
    /// taken is the parked request's outstanding answer once it lands.
    pub fn claim_demand<K: Ord + Copy>(&mut self, staged: Staged, demand: Demand, rank: impl Rank<K>, shares: Option<&[usize]>, read: impl Fn(u32) -> bool, park: &Park) -> Claim {
        if let Some(claim) = self.demanded(staged, demand) {
            return claim;
        }
        if self.ready_beyond(self.left_by(park)) {
            return self.take_demanded(staged, demand, park);
        }
        // The first in taking order at the shares: the least ranked of the
        // layers past theirs, else the least ranked ([`Self::by_shares`]).
        let shares = Shares { of: shares, landing: Some(staged.0) };
        let first = (0..self.records())
            .filter(|&record| self.reclaims(record, &read, park))
            .filter_map(|record| Some((self.place(record, &rank)?, record)))
            .min_by_key(|&(place @ (.., (layer, _)), record)| (self.past(layer, shares) == 0, place, record));
        let Some((_, record)) = first else {
            return Claim::Short;
        };
        if !self.claims(staged, demand) {
            return Claim::Landing;
        }
        self.take_victim(staged, record, park)
    }

    /// The device parked at `park`, so every resolution before it in the
    /// device's order is done: every graced record the parked step did not
    /// resolve (`read`, what it resolved at its layer,
    /// `decode::parked_reads`), that answers no request at or after the
    /// park (its own, which its entries read after it), and that no step of
    /// a group the parked one was cut in before may still resolve
    /// ([`Park::frees`]: that group's resolutions lie after the park), is
    /// ready. A plain map's unmapping is read by every resolution after its
    /// store, so none after the park names it. A stamped map's names it for
    /// the steps below its stamp that have yet to resolve its layer: its
    /// entry is restamped, where it still names the record, from the first
    /// step whose resolution of that layer is not before the park
    /// ([`Park::unresolved`]), as a demand's reclaim unmaps, before the
    /// record is ready. So a reserve unmapped between parks is ready at the
    /// next park, however many steps have begun behind it, and the read
    /// ahead it is kept for finds it. True when any became ready.
    pub fn release(&mut self, park: &Park, read: impl Fn(u32) -> bool) -> bool {
        self.passed(park.point);
        let resolution = self.resolution;
        self.free(|tier, record, staged, (from, until), answer| {
            if read(record) || answer.is_some_and(|answer| answer >= park.point) || !park.frees(staged.0, from, until) {
                return false;
            }
            if resolution == Resolution::BeforeStamp {
                tier.unmap_from(staged, record, park.unresolved(staged.0));
            }
            true
        })
    }

    /// The device is past every point before `progress`: every graced
    /// record ready whose steps that may name it are each past its layer
    /// there or ended (`Steps::served`, read now), and every outstanding
    /// answer of a request before it consumed, kept. True when any became
    /// ready.
    pub fn retire(&mut self, progress: Point, steps: &Steps) -> bool {
        self.passed(progress);
        self.free(|_, _, staged, (from, until), answer| {
            answer.is_none_or(|answer| answer < progress) && steps.served(from, until, staged.0, progress)
        })
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
    /// expert it held, the steps `[from, until)` that may name it, the
    /// request it answers) ready, in the order graced. True when any became
    /// ready.
    fn free(&mut self, mut past: impl FnMut(&Self, u32, Staged, (Step, Step), Option<Point>) -> bool) -> bool {
        let freed = self
            .graced
            .clone()
            .into_iter()
            .filter(|&record| {
                let Host::Graced { whole: Whole { staged, from, .. }, until, answer } = self.records[record as usize] else {
                    unreachable!("a graced record is listed graced");
                };
                past(self, record, staged, (from, until), answer)
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
            Host::Kept(_) | Host::Answering(..) | Host::Graced { .. } | Host::Landing { .. } | Host::Seeding { .. } => false,
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
            Host::Landing { .. } | Host::Kept(_) | Host::Answering(..) | Host::Claimed | Host::Seeding { .. } => true,
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
    /// order at `rank` and `shares` (the shrunk store's), keeping the most
    /// valuable, and each kept one at or
    /// past `records` moves to a record below it that the rest leave,
    /// mapped there from the first step not yet begun, each dropped one
    /// unmapped. The moves, (from, to), whose bytes the caller copies before
    /// any step reads them.
    pub fn shrink<K: Ord + Copy>(&mut self, records: u32, rank: impl Rank<K>, shares: Option<&[usize]>) -> Vec<(u32, u32)> {
        let count = self.records();
        assert!(records <= count, "a host tier of {count} records shrinks to {records}");
        let floor = self.shrink_floor();
        assert!(records >= floor, "a host tier shrinks to {records} records, below its floor of {floor}");
        let order = self.taking_order(&rank, Shares { of: shares, landing: None }, false, |record| !self.in_flight(record));
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
        self.unprepare();
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
                Host::Kept(Whole { staged, holds: 0, .. }) | Host::Graced { whole: Whole { staged, .. }, answer: None, .. } => {
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
    /// whether it unmapped any.
    fn refill(tier: &mut HostRecords, reserve: usize, rank: impl Rank<Score>, at: &Unbegun) -> bool {
        let victims = tier.refill_victims(reserve, rank, None, false);
        let unmapped = !victims.is_empty();
        tier.refill(victims, false, at);
        unmapped
    }

    /// The streamed layers a test's park reads its order at.
    const LAYERS: usize = 4;

    /// Steps 1 to 1,000 begun, each a group of one run whole, none ended:
    /// what a test's device runs.
    fn played() -> Steps {
        let steps = Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new());
        steps.begin_through(1000);
        steps
    }

    /// The device parked at `point`, a step of a group of `steps` steps
    /// from `point.group` run whole, every step before a group of one:
    /// its park.
    fn park(point: Point, steps: u64) -> Park {
        let played = Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new());
        played.begin_through(point.group - 1);
        played.begin_group(steps, None);
        played.park(point, LAYERS)
    }
    use crate::experts::Words;
    use crate::experts::map::{MapView, StampedMap};
    use crate::experts::watcher::Steps;
    use magnitude_kernels::constants::MAP_WORDS;

    /// The experts of a tier's group where more than its records: every
    /// landing below may be demanded.
    const EXPERTS: usize = 64;

    /// A group of one layer of [`EXPERTS`] experts.
    const ONE: &[(usize, usize)] = &[(0, EXPERTS)];

    /// A point of decode step `sequence` at layer 0.
    fn at(sequence: Step) -> Point {
        Point::whole(sequence, 0, sequence)
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
        let steps = std::sync::Arc::new(Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new()));
        steps.begin_through(1000);
        mapped_on(steps, records, layers, members, resolution)
    }

    /// [`mapped`], a stamped map's entries stored from the first of
    /// `steps` not yet begun.
    fn mapped_on(steps: Arc<Steps>, records: u32, layers: usize, members: &[(usize, usize)], resolution: Resolution) -> (HostRecords, Maps) {
        let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
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
            match self.claim(staged, keep, Taking::Out) {
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
            grant.map_at(Some(0));
            self.landed(grant.record(), staged, None, Kept::Alone);
            grant.record()
        }

        /// A demand claim of lacked `staged` parked at `parked` in a group
        /// of `steps` steps, nothing read: the record taken and the expert
        /// whose record it unmapped, if one, none where short.
        fn reclaim(&mut self, staged: Staged, rank: impl Rank<Score>, parked: Point, steps: u64) -> Option<(u32, Option<Staged>)> {
            let kept = self.kept.clone();
            match self.claim_demand(staged, Demand::Lacked, rank, None, |_| false, &park(parked, steps)) {
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
        assert!(matches!(tier.claim((0, 1), 0, Taking::Out), Claim::Landing), "a read ahead finds it landing");
        let Claim::Taken(grant) = tier.claim_demand((0, 1), Demand::Posted, reads(&[]), None, |_| false, &park(at(5), 1)) else { panic!("the one record is ready") };
        tier.landed(grant.record(), (0, 1), Some(5), Kept::Alone);
        assert_eq!(tier.located((0, 1)), Located::Kept(grant.record()));
        assert!(inbound.claim((0, 2)));
        assert!(matches!(tier.claim_demand((0, 2), Demand::Posted, reads(&[]), None, |_| false, &park(at(5), 1)), Claim::Short), "the kept record is the request's answer");
        tier.unposted((0, 2));
        assert_eq!(tier.located((0, 2)), Located::Lacked, "a posted read given up lacks it again");
        assert!(inbound.claim((0, 2)), "and the next read claims it");
    }

    #[test]
    fn a_claim_takes_a_record_only_for_a_lacked_expert() {
        let mut tier = tier(4, 1, ONE, Resolution::BeforeStamp);
        let record = tier.take((0, 1), 0).unwrap();
        assert!(matches!(tier.claim((0, 1), 0, Taking::Out), Claim::Landing), "a second landing of it claims the first");
        assert!(matches!(tier.claim_demand((0, 1), Demand::Lacked, reads(&[]), None, |_| false, &park(at(1), 1)), Claim::Landing));
        tier.landed(record, (0, 1), None, Kept::Alone);
        assert!(matches!(tier.claim((0, 1), 0, Taking::Out), Claim::Kept(kept) if kept == record));
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
        records.prepare_demand(reads(&ranked), None, 3, 5);
        let taken = |claim: Option<Claim>| match claim {
            Some(Claim::Taken(grant)) => Some(grant.record()),
            Some(_) => panic!("a lacked expert's claim takes a record or none"),
            None => None,
        };
        assert_eq!(taken(records.claim_prepared((0, 4), Demand::Lacked, |_| false, &park(at(6), 1))), Some(two), "the least recently read first");
        assert_eq!(taken(records.claim_prepared((0, 5), Demand::Lacked, |record| record == three, &park(at(6), 1))), Some(one), "the parked step's read passed over");
        assert_eq!(taken(records.claim_prepared((0, 6), Demand::Lacked, |_| false, &park(at(6), 1))), None, "none left: the claim scans");
        let mut stale = tier(3, 1, ONE, Resolution::BeforeStamp);
        for expert in 1..=3 {
            stale.land((0, expert));
        }
        stale.prepare_demand(reads(&ranked), None, 1, 5);
        let Claim::Taken(grant) = stale.claim_demand((0, 4), Demand::Lacked, reads(&ranked), None, |_| false, &park(at(6), 1)) else { panic!("a kept record frees") };
        assert_eq!(grant.record(), two, "the scan takes the prepared victim first");
        assert_eq!(taken(stale.claim_prepared((0, 5), Demand::Lacked, |_| false, &park(at(6), 1))), None, "a victim taken since is passed over");
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
        records.prepare_demand(reads(&ranked), None, 3, 5);
        records.shrink(1, reads(&ranked), None);
        assert!(records.unprepared(5), "the shrink's records want preparing again");
        assert_eq!(records.claim_prepared((0, 4), Demand::Lacked, |_| false, &park(at(5), 1)).map(|_| ()), None, "no victim prepared before the shrink is taken");
        let Claim::Taken(grant) = records.claim_demand((0, 4), Demand::Lacked, reads(&ranked), None, |_| false, &park(at(5), 1)) else { panic!("the kept record frees") };
        assert_eq!(grant.record(), 0, "the tier's one record");
    }

    /// The ranking of experts last read at `reads`, each (expert, step);
    /// none read never.
    fn reads(reads: &[(Staged, Step)]) -> impl Fn(Staged, Option<Step>) -> Score + '_ {
        move |staged, _| reads.iter().find(|&&(read, _)| read == staged).map_or(Score::NEVER, |&(_, step)| Score::at(step))
    }

    /// A read ahead's landing keeps its record the reserve's until it ends:
    /// the reserve unmaps nothing for it in flight, so one that ends
    /// unlanded (a hint withdrawn unread) leaves the tier at its reserve,
    /// with no record beyond it for a claim to take, and one that lands
    /// takes its record from the reserve, which then unmaps one more.
    #[test]
    fn a_read_aheads_landing_holds_its_record_in_the_reserve_until_it_ends() {
        let read = [((0, 1), 3), ((0, 2), 2), ((0, 3), 1)];
        let rank = reads(&read);
        let (mut tier, _maps) = mapped(4, 1, ONE, Resolution::AfterStore);
        for (staged, _) in read {
            tier.land(staged);
        }
        assert!(!tier.takes(1), "three kept and one ready: the tier is at its reserve of one");
        let Claim::Taken(failing) = tier.claim((0, 4), 0, Taking::Ahead) else { panic!("a read ahead takes the ready record") };
        let failing = failing.record();
        assert!(!refill(&mut tier, 1, &rank, &Unbegun::at(2)), "the landing's record is the reserve's");
        tier.unlanded(failing);
        assert!(!tier.takes(1), "a landing that ends unlanded leaves nothing beyond the reserve");
        let Claim::Taken(landing) = tier.claim((0, 5), 0, Taking::Ahead) else { panic!("a read ahead takes the ready record") };
        let landing = landing.record();
        tier.landed(landing, (0, 5), Some(2), Kept::Alone);
        assert!(refill(&mut tier, 1, &rank, &Unbegun::at(2)), "the landed record leaves the reserve");
        assert!(tier.retire(at(2), &played()));
        assert_eq!(tier.ready(), 1);
        assert!(!tier.takes(1));
    }

    /// A tier with a record for every expert of its group lands every
    /// expert and keeps no reserve: a claim takes a record while one more
    /// lacked expert than the reserve would remain, the last ones too, and
    /// the reserve unmaps nothing once it lacks none; lacking one again,
    /// one record ready covers it, and a claim may land it there.
    #[test]
    fn a_tier_of_every_expert_lands_each_and_keeps_no_reserve() {
        let rank = reads(&[]);
        let mut tier = tier(3, 1, &[(0, 3)], Resolution::AfterStore);
        for expert in 0..3 {
            let record = tier.take((0, expert), 2).expect("a claim beyond a reserve of two takes the lacked ones' records");
            tier.landed(record, (0, expert), None, Kept::Alone);
        }
        assert!(tier.complete() && tier.lacked() == 0);
        assert!(!tier.takes(0), "no expert lacked");
        assert!(!refill(&mut tier, 2, &rank, &Unbegun::at(1)), "a complete tier keeps no reserve");
        assert!(tier.complete());
        // One dropped and its record grown back: lacking one, the reserve
        // is one record.
        tier.shrink(2, &rank, None);
        tier.grow(3);
        assert_eq!((tier.lacked(), tier.ready()), (1, 1));
        assert!(!refill(&mut tier, 2, &rank, &Unbegun::at(1)), "one record ready covers the one lacked");
        let lacked = (0..3).map(|expert| (0, expert)).find(|&staged| tier.located(staged) == Located::Lacked).expect("one lacked");
        assert!(tier.take(lacked, 2).is_some(), "a claim of the lacked one itself leaves nothing a demand could need");
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
    /// begun is graced for every step before 13, so through its layer of
    /// 12. A stamped map holds it for every step before its stamp, so steps
    /// 10..=12 may still name it: unparked, it frees only past that point;
    /// at the park its entry is restamped from the first step that has not
    /// resolved its layer, so no resolution after the park names it and it
    /// frees there, unless the parked step resolved to it. A plain map's
    /// store is read by every later resolution, so it frees at the park
    /// unless the parked step resolved to it, with no restamp.
    #[test]
    fn a_record_frees_once_past_every_resolution_that_may_name_it() {
        let rank = reads(&[]);
        let parked = park(Point::whole(10, 1, 10), 1);
        let graced = |resolution| {
            let (mut tier, maps) = landed_tier(resolution);
            refill(&mut tier, 1, &rank, &Unbegun::at(13));
            let [record] = tier.graced[..] else { panic!("one record graced") };
            let Host::Graced { whole: Whole { staged, from, .. }, until, .. } = tier.records[record as usize] else { panic!("record {record} is graced") };
            assert_eq!((from, until), (0, 13), "graced for every step begun");
            (tier, maps, record, staged)
        };
        let (mut stamped, _, _, staged) = graced(Resolution::BeforeStamp);
        assert!(!stamped.retire(Point::whole(12, staged.0, 12), &played()), "step 12 may still name it");
        assert!(stamped.retire(Point::whole(13, 0, 13), &played()), "past step 12");
        let (mut stamped, maps, record, staged) = graced(Resolution::BeforeStamp);
        assert!(!stamped.release(&parked, |of| of == record), "the parked step resolved to it");
        let from = parked.unresolved(staged.0);
        assert_eq!(maps.resolve(staged, from - 1), Some(record), "still named below the restamp");
        assert!(stamped.release(&parked, |_| false));
        assert_eq!((maps.resolve(staged, from - 1), maps.resolve(staged, from)), (Some(record), None), "restamped from the first unresolved step");
        let (mut plain, maps, record, staged) = graced(Resolution::AfterStore);
        assert_eq!(maps.held(staged), None, "a plain unmapping is read by every later resolution");
        assert!(!plain.release(&parked, |of| of == record), "the parked step resolved to it");
        assert!(plain.release(&parked, |_| false), "no later resolution names it");
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
        let parked = Point::whole(10, 1, 10);
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
        let wave = park(Point::whole(20, 1, 22), 4);
        assert_eq!([2, 1, 0].map(|layer| wave.unresolved(layer)), [20, 23, 24]);
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
        refill(&mut tier, 1, &rank, &Unbegun::at(8));
        assert!(tier.keeps((0, 1)) && !tier.keeps((0, 2)), "the held least recent is passed over");
        tier.retire(at(7), &played());
        refill(&mut tier, 2, &rank, &Unbegun::at(9));
        assert!(tier.keeps((0, 1)), "only the held record is kept, and stays");
        tier.unhold(first);
        refill(&mut tier, 2, &rank, &Unbegun::at(10));
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
        grant.map();
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
        grant.map();
        grant.map();
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
        refill(&mut tier, 2, &rank, &Unbegun::at(7));
        assert!(maps.held((0, 2)).is_none() && maps.held((0, 3)).is_none());
        assert!(maps.held((0, 1)).is_some() && maps.held((0, 4)).is_some());
        assert!(!tier.keeps((0, 2)) && !tier.keeps((0, 3)) && tier.keeps((0, 4)));
        assert_eq!(tier.ready(), 0);
        assert!(!refill(&mut tier, 2, &rank, &Unbegun::at(1)), "the graced records make the reserve");
        assert!(!tier.retire(at(6), &played()), "step 6 could still read them");
        assert!(tier.retire(at(7), &played()), "the device past layer 0 of step 6");
        assert_eq!(tier.ready(), 2);
        // Then the next alone, (0, 1), and the banked copy.
        refill(&mut tier, 4, &rank, &Unbegun::at(8));
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
        refill(&mut tier, 1, reads(&[((1, 4), 8), ((1, 5), 9)]), &Unbegun::at(10));
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
                grant.map();
                tier.landed(grant.record(), (0, expert), None, Kept::Alone);
            }
        }
        tier.banked((0, 11));
        tier.unlanded(1);
        // Record 1 (empty), then record 0 (alone, step 2) are dropped:
        // record 3 (alone, step 8) moves into record 1, record 2 (banked)
        // into record 0.
        assert_eq!(tier.shrink(2, &rank, None), [(3, 1), (2, 0)]);
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
                grant.map();
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
        assert_eq!(tier.shrink(4, &rank, None), [(4, 0), (5, 1)]);
        assert_eq!([9, 10, 11, 13].map(|expert| maps.held((0, expert))), [Some(1), Some(0), Some(3), None]);
        assert!(tier.lands((0, 12)), "the landing is in flight where it was");
        grants[2].map();
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
                grant.map_at(Some(0));
                tier.landed(grant.record(), (0, expert), None, Kept::Alone);
            }
        }
        // Expert 11 lands into record 2, in flight: no cut lies below it.
        assert_eq!(tier.running_floor(), 3, "every record in flight lies below the floor");
        grants[2].map_at(Some(0));
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
        let Claim::Taken(first) = tier.claim_demand((0, 1), Demand::Lacked, &rank, None, |_| false, &park(request, 1)) else { panic!("the one record is ready") };
        first.map();
        tier.landed(first.record(), (0, 1), Some(5), Kept::Alone);
        assert!(matches!(tier.records[0], Host::Answering(_, past) if past == request), "an outstanding answer");
        assert!(!tier.claim_record_of_lent_free(), "no prompt group's claim takes it");
        refill(&mut tier, 1, &rank, &Unbegun::at(6));
        assert_eq!(maps.held((0, 1)), None, "the reserve unmaps it");
        assert!(matches!(tier.records[0], Host::Graced { answer: Some(answer), .. } if answer == request), "graced with its answer");
        assert!(!tier.release(&park(request, 1), |_| false), "its request's park frees it not");
        assert!(matches!(tier.claim_demand((0, 2), Demand::Lacked, &rank, None, |_| false, &park(request, 1)), Claim::Short), "the second miss waits");
        assert!(!tier.retire(request, &played()), "the device at the request still reads it");
        assert!(tier.retire(at(6), &played()), "past the request it is free");
        assert!(matches!(tier.claim_demand((0, 2), Demand::Lacked, &rank, None, |_| false, &park(at(6), 1)), Claim::Taken(_)), "then a later demand takes it");
    }

    /// A demand landing that ends once the device is past its request (its
    /// step ended with its wait expired) answers nothing: kept, not an
    /// outstanding answer, so nothing holds it in flight for a shrink.
    #[test]
    fn a_landing_past_its_request_is_kept_not_answering() {
        let mut tier = tier(2, 1, ONE, Resolution::AfterStore);
        let Claim::Taken(grant) = tier.claim_demand((0, 1), Demand::Lacked, reads(&[]), None, |_| false, &park(at(5), 1)) else { panic!("ready") };
        assert_eq!(tier.shrink_floor(), 1, "the landing is in flight");
        tier.retire(at(6), &played());
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
            let Claim::Taken(grant) = tier.claim((0, expert), 0, Taking::Out) else { panic!("a record is ready") };
            grant.map();
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
        let Claim::Taken(grant) = tier.claim_demand((0, 1), Demand::Lacked, reads(&[]), None, |_| false, &park(at(5), 1)) else { panic!("ready") };
        tier.landed(grant.record(), (0, 1), None, Kept::Alone);
        tier.unkeep(0);
    }

    /// A chain of steps 1 and 2 begun on `steps`, stopped before its layer
    /// `cut`, and a decode step cut in there: the decode step.
    fn cut_in(steps: &Steps, cut: usize) -> Step {
        assert_eq!(steps.begin_group(2, None), 1);
        steps.begin_group(1, Some((1, cut)))
    }

    /// The park of decode step `step` at `layer` of [`LAYERS`].
    fn parked_at(steps: &Steps, step: Step, layer: usize) -> Park {
        steps.park(steps.at(step, layer, LAYERS).point, LAYERS)
    }

    /// A stamped tier of `records` records of one group of every one of
    /// [`LAYERS`] layers, on `steps`, `kept` landed before any step (mapped
    /// from step 0, so a chain's steps resolve them).
    fn chain_tier(steps: &Arc<Steps>, records: u32, kept: &[Staged]) -> (HostRecords, Maps) {
        let members = (0..LAYERS).map(|layer| (layer, EXPERTS)).collect::<Vec<_>>();
        let (mut tier, maps) = mapped_on(steps.clone(), records, LAYERS, &members, Resolution::BeforeStamp);
        for &staged in kept {
            tier.land(staged);
        }
        (tier, maps)
    }

    /// A decode step cut in before a chain's layer 2 reuses, at the layers
    /// before, the records the chain has passed: the first a demand takes
    /// is the least ranked of them, though a record of layer 2 or 3 ranks
    /// below it. It never reuses, at layers 2 and 3, a record the chain may
    /// still resolve there: once those before are taken its demand is
    /// short, and every one of them still resolves for the chain's steps.
    /// Its landings are mapped from its own step on, so no chain step
    /// resolves them.
    #[test]
    fn a_cut_in_reuses_only_what_the_chain_has_passed() {
        let steps = Arc::new(Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new()));
        let decode = cut_in(&steps, 2);
        let kept = [(0, 1), (1, 1), (2, 1), (3, 1)];
        let (mut tier, maps) = chain_tier(&steps, 4, &kept);
        // The chain's layers rank least: a park that took by rank alone
        // would take them first.
        let rank = reads(&[((0, 1), 9), ((1, 1), 8), ((2, 1), 1), ((3, 1), 2)]);
        let park = parked_at(&steps, decode, 3);
        let mut taken = Vec::new();
        for expert in 10..12 {
            let kept = tier.kept.clone();
            let Claim::Taken(grant) = tier.claim_demand((3, expert), Demand::Lacked, &rank, None, |_| false, &park) else {
                panic!("a record the chain has passed frees")
            };
            taken.push(kept.into_iter().find(|&(_, record)| record == grant.record()).map(|(staged, _)| staged));
            grant.map();
            tier.landed(grant.record(), (3, expert), Some(decode), Kept::Alone);
            assert_eq!(maps.resolve((3, expert), 2), None, "no chain step resolves the step cut in's landing");
            assert!(maps.resolve((3, expert), decode + 1).is_some(), "the decode steps after it do");
        }
        assert_eq!(taken, [Some((1, 1)), Some((0, 1))], "layer 1's, then layer 0's");
        assert!(matches!(tier.claim_demand((3, 12), Demand::Lacked, &rank, None, |_| false, &park), Claim::Short), "no record the chain may resolve is taken");
        for staged in [(2, 1), (3, 1)] {
            let record = tier.kept_record(staged).expect("kept");
            assert!((1..=2).all(|step| maps.resolve(staged, step) == Some(record)), "{staged:?} resolves for the chain's steps");
        }
        // Graced by the reserve while the chain runs, its records past the
        // cut free at no park of the step cut in, and only once the
        // chain's passes there are past.
        let records = [(2, 1), (3, 1)].map(|staged| tier.kept_record(staged).expect("kept"));
        let graced = |tier: &HostRecords| records.iter().all(|&record| matches!(tier.records[record as usize], Host::Graced { .. }));
        assert!(steps.unbegun(|at| refill(&mut tier, 4, &rank, at)));
        assert!(graced(&tier));
        tier.release(&park, |_| false);
        assert!(graced(&tier), "the chain may still resolve them");
        assert!((1..=2).all(|step| maps.resolve((2, 1), step) == Some(records[0])), "unrestamped");
        tier.retire(steps.after(decode), &steps);
        assert!(graced(&tier), "the chain's passes at layers 2 and 3 still run");
        assert!(tier.retire(steps.after(1), &steps) && !graced(&tier), "past the chain");
    }

    /// A decode step cut in that demands an expert the reserve graced
    /// while a chain may still resolve it takes the graced record back as
    /// it was, mapped again from its own `from`: the chain's planned hits
    /// on it still resolve, and the decode steps after the cut-in resolve
    /// it too. Landing it anew would map it from the cut-in's park on and
    /// cut those hits off.
    #[test]
    fn a_cut_ins_demand_of_a_graced_expert_keeps_the_chains_planned_hits() {
        let steps = Arc::new(Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new()));
        let decode = cut_in(&steps, 2);
        let graced = (3, 1);
        let (mut tier, maps) = chain_tier(&steps, 6, &[(2, 1), graced]);
        let record = tier.kept_record(graced).expect("kept");
        assert!((1..=2).all(|step| maps.resolve(graced, step) == Some(record)), "the chain plans its hits at layer 3 on it");
        let ranked = [((2, 1), 9), (graced, 1)];
        let rank = reads(&ranked);
        assert!(steps.unbegun(|at| refill(&mut tier, 5, &rank, at)));
        let park = parked_at(&steps, decode, 3);
        let answered = if tier.located(graced) == Located::Lacked {
            let Claim::Taken(grant) = tier.claim_demand(graced, Demand::Lacked, &rank, None, |_| false, &park) else { panic!("a record for the cut-in's demand") };
            grant.map();
            tier.landed(grant.record(), graced, Some(decode), Kept::Alone);
            grant.record()
        } else {
            let Claim::Kept(kept) = tier.claim(graced, 0, Taking::Out) else { panic!("its graced record holds it whole") };
            tier.answer(kept, park.point);
            kept
        };
        assert!((1..=2).all(|step| maps.resolve(graced, step) == Some(record)), "the chain's planned hits at layer 3 still resolve");
        assert_eq!(maps.resolve(graced, decode + 1), Some(answered), "the decode steps after the cut-in resolve it");
    }

    /// A decode step cut in never waits on its own pool: with every record
    /// but the pool's kept for the chain it runs inside, past the cut,
    /// each of a run of decode steps cut in at the cut misses its layer's
    /// choices at every layer, and every demand takes a record: the pool's
    /// or its own earlier landings', recycled. The chain's records stay
    /// mapped for it throughout, and no other landing takes the pool.
    #[test]
    fn a_cut_in_never_waits_on_its_own_pool() {
        const CHOICES: u32 = 2;
        let steps = Arc::new(Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new()));
        let first = cut_in(&steps, 1);
        let chain = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)];
        let (mut tier, maps) = chain_tier(&steps, chain.len() as u32 + CHOICES, &chain);
        tier.set_pool(CHOICES);
        assert_eq!(tier.take((0, 60), 0), None, "a claim leaves the pool");
        let rank = reads(&[]);
        let mut expert = 10;
        for decode in first..first + 4 {
            if decode > first {
                assert_eq!(steps.begin_group(1, Some((1, 1))), decode);
            }
            for layer in 0..LAYERS {
                let park = parked_at(&steps, decode, layer);
                assert!(park.cut_in());
                tier.release(&park, |_| false);
                for _ in 0..CHOICES {
                    let Claim::Taken(grant) = tier.claim_demand((layer, expert), Demand::Lacked, &rank, None, |_| false, &park) else {
                        panic!("decode step {decode}'s miss at layer {layer} waited on its pool")
                    };
                    grant.map();
                    tier.landed(grant.record(), (layer, expert), Some(decode), Kept::Alone);
                    expert += 1;
                }
            }
        }
        for staged in chain {
            let record = tier.kept_record(staged).expect("the chain's records are kept");
            assert!((1..=2).all(|step| maps.resolve(staged, step) == Some(record)), "{staged:?} resolves for the chain's steps");
        }
    }

    /// The prompt group's tier copy reads no landing of a decode step cut
    /// in during it: mapped past its every step, the cut-ins recycle it.
    /// It reads a landing no step read.
    #[test]
    fn a_prompt_group_copies_no_cut_ins_landing() {
        let steps = Arc::new(Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new()));
        let decode = cut_in(&steps, 1);
        let (mut tier, _) = chain_tier(&steps, 4, &[(2, 1)]);
        let park = parked_at(&steps, decode, 2);
        let Claim::Taken(grant) = tier.claim_demand((2, 5), Demand::Lacked, reads(&[]), None, |_| false, &park) else { panic!("ready") };
        assert!(!tier.serves((2, 5), 2), "landing for the step cut in");
        grant.map();
        tier.landed(grant.record(), (2, 5), Some(decode), Kept::Alone);
        assert!(!tier.serves((2, 5), 2) && tier.serves((2, 5), decode + 1), "kept for the decode steps alone");
        assert!(tier.serves((2, 1), 2), "the chain's landing serves it");
    }
}
