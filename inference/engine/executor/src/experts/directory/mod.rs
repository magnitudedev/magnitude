//! Which bank record holds which expert. The
//! host places every expert into the bank and is the one writer of its map
//! (`experts::map`). Every choice of what the bank keeps ranks by the one
//! ranking ([`Reuse`], measured from the completed steps' reads, which its
//! stack `Stack` feeds): a promotion's ([`Directory::promotions`]), which
//! brings the ranking's share of each layer into the bank in place of what
//! lies outside it, a prompt group's seeds' ([`Directory::seed`]) and a
//! shrink's.
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
//! engine (`watcher::promotions`). The tier is inclusive of the bank: a
//! promotion copies an expert into the bank and the tier keeps its copy, so
//! the bank evicting it costs nothing, and nothing is ever written back. A
//! prompt group's seeds are the one placement the device copies: they hold
//! their records once the device committed them, which the host learns
//! once the device is past them ([`Directory::retire`]).

#[cfg(any(test, feature = "expert-routing"))]
mod replay;
mod stack;

use super::host_records::{Claim, HostRecords, Kept, Located, Resolution};
use super::map::{LandingMap, StampedMap};
use super::records::{BankRecords, Lent, Record};
use super::reuse::{Rank, Ranking, Reuse, Score};
use super::staging::Staged;
use super::watcher::{Lending, LentRecords, Step, Steps, Unbegun};
use std::collections::{BTreeSet, HashMap};
pub use stack::{LayerCurve, MissCurve, RoutedRead, Stack};
pub(crate) use stack::Ranks;
#[cfg(any(test, feature = "expert-routing"))]
pub use replay::{Recency, replay};
#[cfg(any(test, feature = "expert-routing"))]
use super::exchange;

/// A point in the device's order of streamed layers. Steps begin in groups
/// (a chain's launches, a lone step a group of one), named by their first
/// sequence and run in submission order; a group runs layer-major, each
/// layer over its steps in sequence order. So a request at layer `l` of a
/// group's step `q` (a scratch step's round 0 there) proves the device past
/// layer `l` of the group's steps before `q` and past every earlier layer of
/// all its steps: the order is (group, layer, sequence).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Point {
    pub group: Step,
    pub layer: usize,
    pub sequence: Step,
}

impl Point {
    /// The point past every layer of every step of group `group`.
    pub fn after(group: Step) -> Self {
        Self {
            group,
            layer: usize::MAX,
            sequence: Step::MAX,
        }
    }

    /// The device parked here, in a group of `steps` steps: the first step
    /// whose resolution of `layer` is not before this point. A layer past
    /// the parked one no step of the group has resolved, so the group's
    /// first; the parked layer, every step of the group up to the parked
    /// one has, so the one after it; a layer before it, every step of the
    /// group has, so the first after the group. Every resolution of `layer`
    /// by a step before it lies before this point, but the parked step's
    /// own at its layer.
    pub fn first_unresolved(self, layer: usize, steps: u64) -> Step {
        match layer.cmp(&self.layer) {
            std::cmp::Ordering::Greater => self.group,
            std::cmp::Ordering::Equal => self.sequence + 1,
            std::cmp::Ordering::Less => self.group + steps,
        }
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
}

/// How a group shrinks to `records` ([`Directory::plan_shrink`]): the
/// experts it evicts, (layer, expert), and the kept ones it moves, (layer,
/// expert, from, to).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Shrink {
    pub records: usize,
    pub evicted: Vec<(usize, usize)>,
    pub moved: Vec<(usize, usize, usize, usize)>,
}

pub(super) struct Directory {
    /// The bank's records: what each holds, and the maps naming them.
    bank: BankRecords,
    /// Per group, each member layer's map word of its expert 0 and the
    /// layer (`ExpertBankPlan::home`), in map order.
    homes: Vec<Vec<(usize, usize)>>,
    /// Each group's host tier records, where the bank has a tier.
    host: Vec<HostRecords>,
    /// Whether a growth of the records the host lands into runs, from its
    /// decision until its last slot joined ([`Self::host_growth`]): the
    /// tier's planned records exceed those it holds and fills.
    host_growing: bool,
    /// The running prompt group's reads the ranking has not logged yet
    /// (its hits and seeds, each read at its last step, [`Self::read_pending`]),
    /// which every taking ranks beside.
    prompt_reads: HashMap<Staged, Step>,
    /// Who places into the bank: where the host does (the one store), a
    /// prompt group's scratch and seeds are its host records'.
    placer: crate::planning::Placer,
    /// The one ranking, which every choice of what to keep ranks by; its
    /// feed is the stack's (`Stack`). Its shares are the records that keep
    /// the bank's experts: the host tier's where it is the bank's own
    /// records (the one store, unified memory: `placer`), else the bank's.
    ranking: Ranking,
    /// Per layer, what the bank is short of and holds beyond the ranking's
    /// share (`Reuse::in_top`), kept as either side changes
    /// ([`Self::resettle`]): the experts among the share the bank lacks and
    /// no placement takes (`wanted`), and those the bank holds outside it
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
    /// ([`Self::reserve`]); the ranking's shares are the records less it.
    reserve: Vec<usize>,
    /// The reads a promotion must expect beyond what it displaces to earn
    /// its copy, each layer's (`ExpertLines::copy_hits`): none where a hit
    /// saves nothing, so no promotion of the layer earns; zero, a copy
    /// costing nothing, until the lines are priced.
    copy_hits: Vec<Option<f64>>,
    /// Per group, the promotions it asked since its first and that one's
    /// step, none before it: what a stay in its bank lasts
    /// ([`Self::stay`]). The one count of promotions asked.
    asked: Vec<Option<Asked>>,
}

/// A group's promotions asked: the step after the first was asked at, and
/// how many it has asked since, that one included.
#[derive(Clone, Copy, Debug)]
struct Asked {
    first: Step,
    count: u64,
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

/// The later of `read`, a read of `staged` the log may not have brought
/// yet, and the running prompt group's read of it
/// ([`Directory::read_pending`]): what every taking ranks it beside.
fn beside(prompt_reads: &HashMap<Staged, Step>, staged: Staged, read: Option<Step>) -> Option<Step> {
    read.max(prompt_reads.get(&staged).copied())
}

/// How the host records rank what they keep, by `reuse`, one hold of the
/// ranking for the whole choice: an expert's reuse beside its reads
/// ([`beside`]).
fn store_rank<'a>(reuse: &'a Reuse, prompt_reads: &'a HashMap<Staged, Step>) -> impl Fn(Staged, Option<Step>) -> Score + Copy + 'a {
    move |staged @ (layer, expert): Staged, read: Option<Step>| reuse.rank(layer, expert, beside(prompt_reads, staged, read)).score()
}

/// A group's wanted experts as their copies price them
/// ([`Directory::promotions`]).
struct Earning {
    earning: Vec<Staged>,
    /// Whether any waits for its reads to earn its copy.
    withheld: bool,
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
/// every layer keeps a record of its share.
pub(super) fn most_reserved(records: usize, members: usize) -> usize {
    records.saturating_sub(members)
}

/// Each member layer's share of group `group`'s `records` records, of
/// layers in the groups `layers` names: the records over its members.
fn share(records: usize, layers: &[usize], group: usize) -> usize {
    records / layers.iter().filter(|&&of| of == group).count()
}

/// What a bank's host tier starts from: each group's records, every one
/// ready, and each group's bottom records the conveyor's lending reserves
/// (`planning::Lender`: its lender's, none of another group), beside who
/// places the bank's experts and the maps their placements write
/// (`placement`).
pub(in crate::experts) struct HostTier {
    pub records: Vec<u32>,
    pub lent: Vec<u32>,
    pub placement: Placement,
}

/// Who places the bank's experts, which decides how the steps resolving
/// their maps see an unmapping, and the maps their placements write.
pub(in crate::experts) enum Placement {
    /// The device places from a host tier: the bank's records are each
    /// layer's stamped map's one writer (`bank_maps`), each layer's host
    /// records its plain tier map's (`tier_maps`), and the lender's bank
    /// records `lent`, `(group, records)`, are lent to prompt groups.
    Tier { bank_maps: Vec<StampedMap>, tier_maps: Vec<LandingMap>, lent: (usize, usize) },
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
    /// of no device (`Placement::Replay`).
    pub fn replay(records: Vec<u32>) -> Self {
        Self { lent: vec![0; records.len()], records, placement: Placement::Replay }
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
        // The records that keep the bank's experts, which the ranking's
        // shares are: the bank's own where the device places from a host
        // tier; the host tier's where the host places, the tier then the
        // bank's own records (the one store).
        let placer = host.placement.placer();
        let kept = match placer {
            crate::planning::Placer::Bank => host.records.iter().map(|&count| count as usize).collect(),
            crate::planning::Placer::Tier => records.to_vec(),
        };
        let HostTier { records: host_records, lent: host_lent, placement } = host;
        // The maps the host records write, one per layer, where the
        // placement writes any.
        let (bank_maps, maps) = match placement {
            Placement::Tier { bank_maps, tier_maps, lent } => (Some((bank_maps, lent)), Some(tier_maps)),
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
        Self {
            bank: BankRecords::new(records, &layers.iter().copied().zip(experts.iter().copied()).collect::<Vec<_>>(), bank_maps),
            homes,
            placer,
            host: {
                let (records, lent) = (host_records, host_lent);
                let resolution = match placer {
                    crate::planning::Placer::Tier => Resolution::AfterStore,
                    crate::planning::Placer::Bank => Resolution::BeforeStamp,
                };
                records
                    .iter()
                    .zip(members)
                    .zip(lent)
                    .map(|((&count, members), lent)| HostRecords::new(count, layers.len(), members, resolution, lent))
                    .collect()
            },
            host_growing: false,
            prompt_reads: HashMap::new(),
            ranking: Reuse::ranking(experts, layers, layers.iter().map(|&group| share(kept[group], layers, group)).collect()),
            wanted: vec![BTreeSet::new(); layers.len()],
            unwanted: vec![BTreeSet::new(); layers.len()],
            stirred: false,
            withheld: None,
            reserve: vec![0; records.len()],
            copy_hits: vec![Some(0.); layers.len()],
            asked: vec![None; records.len()],
        }
    }

    /// Place `layer`'s `expert` in the wanted and unwanted sets by whether
    /// the ranking's share holds it (`top`), the bank holds it, and a
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
    /// changed, by the ranking's share as it is now.
    fn resettle(&mut self, layer: usize, expert: usize) {
        let top = Reuse::of(&self.ranking).in_top(layer, expert);
        self.settle(layer, expert, top);
    }

    /// The one ranking, for its feed (`Stack`).
    pub fn ranking(&self) -> Ranking {
        self.ranking.clone()
    }

    /// The ranking as it is now.
    fn reuse_now(&self) -> std::sync::MutexGuard<'_, Reuse> {
        Reuse::of(&self.ranking)
    }

    /// Group `group` holds `records` records from now: its layers' shares
    /// ([`share`]) of them less its reserve are the ranking's, its reserve
    /// within the most a group reserves ([`most_reserved`]). A member's
    /// share of none where the group has a record for each is an invariant
    /// violation: the reserve took what the layers hold.
    fn resized(&mut self, group: usize, records: usize) {
        let groups = (0..self.bank.layers()).map(|layer| self.bank.group(layer)).collect::<Vec<_>>();
        let members = groups.iter().filter(|&&of| of == group).count();
        self.reserve[group] = self.reserve[group].min(most_reserved(records, members));
        let kept = records - self.reserve[group];
        assert!(
            records < members || share(kept, &groups, group) > 0,
            "group {group}'s reserve of {} leaves its {members} layers no record of its {records}",
            self.reserve[group]
        );
        let mut ranking = Reuse::of(&self.ranking);
        for layer in (0..groups.len()).filter(|&layer| groups[layer] == group) {
            ranking.resize(layer, share(kept, &groups, group));
        }
    }

    /// Group `group` keeps `records` of its records free or releasing from
    /// now, beyond what its wants take: the ranking's shares are its records
    /// less them. Only where promotions place from a host tier.
    pub fn reserve(&mut self, group: usize, records: usize) {
        assert!(self.placer == crate::planning::Placer::Tier, "the one store promotes nothing");
        let members = self.layer_groups().iter().filter(|&&of| of == group).count();
        let count = self.bank.records(group).len();
        let most = most_reserved(count, members);
        assert!(records <= most, "a reserve of {records} of group {group}'s {count} records, beyond {most}");
        if std::mem::replace(&mut self.reserve[group], records) != records {
            self.resized(group, count);
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

    /// Each layer's share of its group's records under the ranking: its
    /// records less its reserve, over its members.
    #[cfg(test)]
    pub fn shares(&self) -> Vec<usize> {
        let groups = self.layer_groups();
        groups.iter().map(|&group| share(self.bank.records(group).len() - self.reserve[group], &groups, group)).collect()
    }

    /// Group `group`'s host records.
    pub fn host(&self, group: usize) -> &HostRecords {
        &self.host[group]
    }

    /// The host records of `layer`'s group.
    fn tier(&mut self, layer: usize) -> &mut HostRecords {
        let group = self.bank.group(layer);
        &mut self.host[group]
    }

    /// Claim a landing of `layer`'s `expert` in its group's host tier,
    /// leaving `keep` ready (the demand reserve, which a read ahead leaves)
    /// or the experts still lacked where fewer ([`HostRecords::claim`]):
    /// every landing of an expert, whoever lands it, claims it here. Short
    /// without a tier.
    pub fn claim_landing(&mut self, layer: usize, expert: usize, keep: usize) -> Claim {
        self.tier(layer).claim((layer, expert), keep)
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

    /// The experts the tier's fill lands next, at most `count`: of each
    /// group's host tier that takes a record leaving `keep`
    /// ([`HostRecords::takes`]), the experts it lacks that `skip` does not
    /// pass over (another read carries them), in the order each tier lands
    /// them ([`HostRecords::lacking`], by this one order) across the groups:
    /// those each layer's ranked share holds first (`Reuse::in_top`), then
    /// the most reused, among equals those `kept` names (staging keeps them
    /// whole, so they land with no read), then by expert, then by layer: a
    /// tie every layer's alike, whatever batches the fill lands in. So a tier with a
    /// record for every expert is filled to every expert, and a smaller
    /// one's records beyond the reserve with each layer's ranked share.
    pub fn fill_order(&self, keep: usize, count: usize, skip: impl Fn(Staged) -> bool, kept: impl Fn(Staged) -> bool) -> Vec<Staged> {
        let reuse = self.reuse_now();
        let rank = store_rank(&reuse, &self.prompt_reads);
        let order = |score: Score, staged @ (layer, expert): Staged| (!reuse.in_top(layer, expert), std::cmp::Reverse(score), !kept(staged), expert, layer);
        let mut order = self
            .host
            .iter()
            .filter(|tier| tier.takes(keep))
            .flat_map(|tier| tier.lacking(rank, count, &skip, order))
            .collect::<Vec<_>>();
        order.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        order.into_iter().take(count).map(|(_, staged)| staged).collect()
    }

    /// Whether the fill has records still to fill, beyond the `keep` ready
    /// each tier keeps: a growth of the tier runs, whose slots join it as
    /// they are backed, or a tier has a record ready beyond `keep` it lacks
    /// an expert for, one the fill takes (`HostRecords::takes`).
    pub fn fill_pending(&self, keep: usize) -> bool {
        self.host_growing || self.host.iter().any(|tier| tier.takes(keep))
    }

    /// A growth of the records the host lands into runs from now, or ended
    /// (`Join::Growing`, `Join::Grown`).
    pub fn host_growth(&mut self, growing: bool) {
        self.host_growing = growing;
    }

    /// Whether every group's host tier keeps every expert of its layers,
    /// none without a tier: no decode miss reads the file.
    pub fn tier_complete(&self) -> bool {
        self.host.iter().all(HostRecords::complete)
    }

    /// Whether group `group`'s host tier keeps every expert of its layers;
    /// false where the bank has no tier.
    pub fn group_tier_complete(&self, group: usize) -> bool {
        self.host[group].complete()
    }

    /// Whether no expert of group `group`'s layers can miss, every one
    /// mapped where a pass reads it in place: beside a host tier, held in
    /// the bank; where the host places, kept in the one store, whose
    /// records are the bank's.
    pub fn group_whole(&self, group: usize) -> bool {
        match self.placer {
            crate::planning::Placer::Tier => {
                (0..self.bank.layers())
                    .filter(|&layer| self.bank.group(layer) == group)
                    .all(|layer| (0..self.bank.experts(layer)).all(|expert| self.bank.held(layer, expert).is_some()))
            }
            crate::planning::Placer::Bank => self.host[group].complete(),
        }
    }

    /// Whether every expert of group `group`'s layers is at its home record
    /// ([`Self::home`]), which a route then sources without the map: a bank
    /// imported whole is, one regrown to every expert need not be.
    pub fn group_at_home(&self, group: usize) -> bool {
        (0..self.bank.layers()).filter(|&layer| self.bank.group(layer) == group).all(|layer| {
            let home = self.home(layer);
            (0..self.bank.experts(layer)).all(|expert| match self.placer {
                crate::planning::Placer::Tier => self.bank.held(layer, expert) == Some(home + expert),
                crate::planning::Placer::Bank => {
                    self.host[group].kept_record((layer, expert)).is_some_and(|record| record as usize == home + expert)
                }
            })
        })
    }

    /// Claim a demand landing of `layer`'s `expert` while the device is
    /// parked at `parked` (`layer` of a step in a group of `steps` steps),
    /// leaving no reserve: where none is ready, one a kept record's
    /// unmapping frees at once, any the parked step's route did not resolve
    /// there (`read`, [`HostRecords::claim_demand`]), its unmapping
    /// stored before the claim returns, so before the landing writes the
    /// record. Short without a tier.
    pub fn claim_demand(&mut self, (layer, expert): Staged, read: &[(usize, u32)], parked: (Point, u64)) -> Claim {
        let group = self.bank.group(layer);
        let reuse = Reuse::of(&self.ranking);
        let rank = store_rank(&reuse, &self.prompt_reads);
        self.host[group].claim_demand((layer, expert), rank, |record| read.contains(&(group, record)), parked)
    }

    /// `layer`'s `expert` landed whole into tier record `record`, read at
    /// step `read`, and its landing mapped it ([`HostRecords::landed`]),
    /// ranked by its reuse beside that read (none where no step read it: the
    /// tier's fill's). A copy of an expert the bank holds (a fill beside a
    /// bank imported whole, or a landing a promotion or seed committed
    /// during) is banked: the tier holds alone only what the bank lacks.
    pub fn landed(&mut self, layer: usize, HostRecord(record): HostRecord, expert: usize, read: Option<Step>) {
        let kept = match self.bank.held(layer, expert) {
            Some(_) => Kept::Banked,
            None => Kept::Alone,
        };
        self.tier(layer).landed(record, (layer, expert), read, kept);
        // A want its tier lacked is promotable from now.
        self.stirred |= self.wanted[layer].contains(&expert);
    }

    /// The landing into tier record `record` of `layer`'s group failed: the
    /// record is ready again ([`HostRecords::unlanded`]).
    pub fn unlanded(&mut self, layer: usize, HostRecord(record): HostRecord) {
        self.tier(layer).unlanded(record);
    }

    /// Whether any group's host tier falls short of `reserve`
    /// ([`HostRecords::short_of`]): whether a refill may unmap anything.
    pub fn refill_short(&self, reserve: usize) -> bool {
        self.host.iter().any(|records| records.short_of(reserve, 0))
    }

    /// Each group's records its reserve of `reserve` tier records unmaps
    /// now ([`HostRecords::refill_victims`]), the lent records among them
    /// where `lent`: read from the ranking, changing nothing.
    pub fn refill_victims(&self, reserve: usize, lent: bool) -> Vec<Vec<(u32, Staged)>> {
        let reuse = self.reuse_now();
        let rank = store_rank(&reuse, &self.prompt_reads);
        self.host.iter().map(|records| records.refill_victims(reserve, rank, lent)).collect()
    }

    /// Unmap each group's `victims` ([`HostRecords::refill`]), under the
    /// lock every step begins under, `at` the first step not yet begun and
    /// `latest` the last step begun, the lent records only where `lent`.
    pub fn refill(&mut self, victims: Vec<Vec<(u32, Staged)>>, lent: bool, at: &Unbegun, latest: (Step, Step)) {
        for (records, victims) in self.host.iter_mut().zip(victims) {
            records.refill(victims, lent, at, latest);
        }
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

    /// `layer`'s `expert`'s reuse ([`Reuse::rank`]), beside a read at step
    /// `read` the log may not have brought yet.
    pub fn reuse(&self, layer: usize, expert: usize, read: Option<Step>) -> Score {
        self.reuse_now().rank(layer, expert, read).score()
    }

    /// The half-life the reuse ranks by, in steps ([`Reuse::half_life`]).
    #[cfg(test)]
    pub fn half_life(&self) -> f64 {
        self.reuse_now().half_life()
    }

    /// The rank of `layer`'s `expert` by `reuse`, beside a read at step
    /// `read` the log may not have brought yet ([`beside`]).
    fn ranked(&self, reuse: &Reuse, layer: usize, expert: usize, read: Option<Step>) -> Ranked {
        Ranked { rank: reuse.rank(layer, expert, beside(&self.prompt_reads, (layer, expert), read)), layer, expert }
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

    /// Settle the ranking's changes of its shares (`Reuse::take_changes`):
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
        self.stirred || self.bank.changed() || self.read_since_withheld()
    }

    /// Whether the ranking logged a read since promotions last withheld a
    /// want ([`Self::withheld`]).
    fn read_since_withheld(&self) -> bool {
        self.withheld.is_some_and(|reads| self.reuse_now().reads() != reads)
    }

    /// What the ranking asks now, in time proportional to what changed:
    /// the ranking's changes of its shares (`Reuse::take_changes`) are
    /// settled first, and only where a want, a victim or a free record was
    /// gained or lost since ([`Self::stirred`]) is anything asked. In each
    /// group, each expert the ranking's share of its layer holds, the bank
    /// lacks and its host tier keeps (`wanted`) takes a free record, which
    /// no step reads, so its copy begins at once: the bank then holds each
    /// layer's share, as the ranking's stack does (`Reuse`). Each record is
    /// its promotion's from now until its copy has completed
    /// ([`Self::promoted`]) or failed ([`Self::unpromoted`]). Then, while the
    /// group's free and releasing records fall short of its reserve and the
    /// wants left without one, the record of its least reused expert
    /// outside its own layer's share (`unwanted`) is released
    /// ([`Self::left_bank`]): free once every step that may read it has
    /// ended ([`Self::drain`]). So a promotion never waits for a record to
    /// drain while the reserve covers what the ranking asks meanwhile; only
    /// the records a promotion may take cover a want
    /// ([`BankRecords::covering`]). Neither takes a lent record while a
    /// prompt group holds them reserved (`lending`, under the steps' lock
    /// with `at`, `watcher::lent`), whose lend unmaps what it holds
    /// instead; the reservation's end asks again ([`Lending::take_freed`]).
    pub fn promotions(&mut self, at: &Unbegun, lending: &mut Lending) -> Wants {
        self.settle_changes();
        let mut asked = Wants::default();
        let stirred = std::mem::take(&mut self.stirred) | self.bank.take_changed() | lending.take_freed() | self.read_since_withheld();
        if !stirred {
            return asked;
        }
        let mut withheld = false;
        for group in 0..self.homes.len() {
            let members = self.homes[group].iter().map(|&(_, layer)| layer).collect::<Vec<_>>();
            let wanted = members
                .iter()
                .flat_map(|&layer| self.wanted[layer].iter().map(move |&expert| (layer, expert)))
                .filter(|&(layer, expert)| self.in_tier(layer, expert))
                .collect::<Vec<_>>();
            // Earning weighs a victim for each want, and releases take at most
            // a want's each and the reserve's: all ranked and weighed by one
            // hold of the ranking, so at one half-life.
            let (victims, priced) = {
                let reuse = Reuse::of(&self.ranking);
                let victims = self.victims(&reuse, group, &members, lending, wanted.len() + self.reserve[group]);
                let priced = self.earning(&reuse, group, wanted, &victims, lending, at.next());
                (victims, priced)
            };
            withheld |= priced.withheld;
            let wanted = priced.earning;
            let free = self
                .bank
                .free(group)
                .filter(|&record| self.bank.usable(group, record, lending))
                .take(wanted.len())
                .collect::<Vec<_>>();
            let short = wanted.len() - free.len();
            for ((layer, expert), record) in wanted.into_iter().zip(free) {
                let promotion = Promotion { group, record: bank_record(record), layer, expert };
                self.bank.promote(&promotion, lending);
                self.asked[group].get_or_insert(Asked { first: at.next(), count: 0 }).count += 1;
                self.resettle(layer, expert);
                asked.promotions.push(promotion);
            }
            // A want left without a record releases one now, whatever the
            // lent records a prompt group holds: none serves it meanwhile.
            // The reserve counts those too, as they return free when the
            // group ends: releasing for them would only be refilled then.
            let (covering, returning) = (self.bank.covering(group, lending), self.bank.returning(group, lending));
            let releases = short.saturating_sub(covering).max((self.reserve[group] + short).saturating_sub(covering + returning));
            if releases == 0 {
                continue;
            }
            for (Ranked { layer, expert, .. }, record) in victims.into_iter().take(releases) {
                let record = bank_record(record);
                self.bank.release(group, record, at, lending);
                self.left_bank(layer, expert, None);
                asked.releases.push(Release { group, record, layer, expert });
            }
        }
        // What it took is asked for: only what changes after asks again.
        self.bank.take_changed();
        self.withheld = withheld.then(|| self.reuse_now().reads());
        asked
    }

    /// Group `group`'s records a release may take, in the order releases
    /// take them: each held expert outside its layer's share (`unwanted`)
    /// whose record is usable, its least ranked first: the `most` least
    /// ranked by `reuse`, selected before they are sorted.
    fn victims(&self, reuse: &Reuse, group: usize, members: &[usize], lending: &Lending, most: usize) -> Vec<(Ranked, usize)> {
        let mut victims = members
            .iter()
            .flat_map(|&layer| self.unwanted[layer].iter().map(move |&expert| (layer, expert)))
            .map(|(layer, expert)| (self.ranked(reuse, layer, expert, None), self.bank.held(layer, expert).expect("an unwanted expert is held")))
            .filter(|&(_, record)| self.bank.usable(group, record, lending))
            .collect::<Vec<_>>();
        if victims.len() > most {
            victims.select_nth_unstable(most);
            victims.truncate(most);
        }
        victims.sort_unstable();
        victims
    }

    /// Of group `group`'s `wanted`, those whose copy pays by `now`
    /// (`Self::price_copies`): the most read beyond its layer's price
    /// first, each against what it would displace, a usable free record
    /// past the reserve nothing, then each of `victims` in the order
    /// releases take them (`Self::victims`), all read over the stay
    /// (`Reuse::weighed`) by `reuse`, which ranked the victims, beside their
    /// reads ([`beside`]), so a victim ranked below another never weighs
    /// more. What a copy costs and a hit saves are the cost function's
    /// (`ExpertLines::copy_hits`).
    fn earning(&self, reuse: &Reuse, group: usize, wanted: Vec<Staged>, victims: &[(Ranked, usize)], lending: &Lending, now: Step) -> Earning {
        let stay = self.stay(group, now);
        let weighed = |staged @ (layer, expert): Staged| reuse.weighed(layer, expert, beside(&self.prompt_reads, staged, None), stay);
        // A layer whose hit saves nothing earns no promotion at any read:
        // its wants wait for no read, so none is withheld; its lines'
        // repricing asks them again (`Self::price_copies`).
        let mut ranked = wanted
            .into_iter()
            .filter_map(|staged| self.copy_hits[staged.0].map(|hits| (weighed(staged) - hits, staged)))
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        // The records past the reserve displace nothing; a promotion into
        // the reserve is paid for by the release that refills it.
        let free = self.bank.free(group).filter(|&record| self.bank.usable(group, record, lending)).count();
        let displaced = std::iter::repeat_n(0., free.saturating_sub(self.reserve[group]))
            .chain(victims.iter().map(|&(Ranked { layer, expert, .. }, _)| weighed((layer, expert))))
            .chain(std::iter::repeat(f64::INFINITY));
        let count = ranked.len();
        let earning = ranked
            .into_iter()
            .zip(displaced)
            .take_while(|&((surplus, _), displaced)| surplus - displaced >= 0.)
            .map(|((_, staged), _)| staged)
            .collect::<Vec<_>>();
        Earning { withheld: earning.len() < count, earning }
    }

    /// The steps an expert stays in group `group`'s bank once promoted, as
    /// the load measures it at step `now`: each promotion takes the place
    /// of one held, so a record's stay is the group's records over its
    /// promotions a step (Little's law); none before a promotion and a step
    /// past it.
    fn stay(&self, group: usize, now: Step) -> Option<f64> {
        let Asked { first, count } = self.asked[group]?;
        let steps = now.checked_sub(first).filter(|&steps| steps > 0)? as f64;
        Some(self.bank.records(group).len() as f64 / (count as f64 / steps))
    }

    /// The promotions every group has asked since the load.
    pub fn asked(&self) -> u64 {
        self.asked.iter().flatten().map(|asked| asked.count).sum()
    }

    /// Layer `layer`'s promotions earn their copies at `hits` reads from
    /// now, none where a hit saves nothing; what they withheld are asked
    /// again.
    pub fn price_copies(&mut self, layer: usize, hits: Option<f64>) {
        if std::mem::replace(&mut self.copy_hits[layer], hits) != hits {
            self.stirred = true;
        }
    }

    /// Every release whose unmapping the `frontier` reached is free
    /// ([`BankRecords::drain`]): whether any was.
    pub fn drain(&mut self, frontier: Step) -> bool {
        self.bank.drain(frontier)
    }

    /// The promotions queued, which copy now ([`BankRecords::queued`]).
    pub fn queued(&self, lending: &Lending) -> Vec<(usize, BankRecord, Staged)> {
        self.bank.queued(lending)
    }

    /// The promotions copying, each (group, record, batch).
    pub fn copying(&self) -> Vec<(usize, BankRecord, u64)> {
        self.bank.copying()
    }

    /// Whether any promotion is in flight or any release draining, and
    /// what each is doing.
    pub fn moving(&self) -> bool {
        self.bank.moving()
    }

    pub fn describe_promotions(&self) -> String {
        self.bank.describe()
    }

    /// Queued promotion `record` of group `group` copies from tier record
    /// `tier`, held, by batch `batch` ([`BankRecords::copy`]).
    pub fn copy(&mut self, group: usize, record: BankRecord, tier: HostRecord, batch: u64, lending: &mut Lending) {
        self.bank.copy(group, record, tier, batch, lending);
    }

    /// The copy of promotion `record` of group `group` has completed: it
    /// holds its expert from the first step not yet begun, banking the copy
    /// its host tier keeps, whose tier record the copy no longer holds.
    pub fn promoted(&mut self, completed: Vec<(usize, BankRecord)>, steps: &Steps, lent: &LentRecords) {
        // Only the records' change and its mapping under the steps' lock;
        // what the tier and the ranking learn of it after.
        for ((layer, expert), tier) in self.bank.promoted(completed, steps, lent) {
            self.tier(layer).banked((layer, expert));
            self.tier(layer).unhold(tier.0);
            self.resettle(layer, expert);
        }
    }

    /// Promotion `record` of group `group` ended without its expert (its
    /// tier record was taken before its copy began, or its copy failed):
    /// the record is free, as nothing maps it.
    pub fn unpromoted(&mut self, group: usize, record: BankRecord, lent: &LentRecords) {
        let ((layer, expert), tier) = self.bank.unpromoted(group, record, lent);
        if let Some(tier) = tier {
            self.tier(layer).unhold(tier.0);
        }
        self.resettle(layer, expert);
    }

    /// The map word of `layer`'s expert 0 in its group's map: the word the
    /// device reads it at (`policy[POLICY_HOME]`, `Residency::publish_policy`).
    pub fn home(&self, layer: usize) -> usize {
        let group = self.bank.group(layer);
        self.homes[group].iter().find(|&&(_, member)| member == layer).expect("a layer is its group's").0
    }

    /// Per layer, the experts its group's host tier holds alone; none
    /// without a tier.
    pub fn alone(&self, layer: usize) -> usize {
        self.host(self.bank.group(layer)).alone()[layer]
    }

    /// Grow group `group`'s host tier to `records` records, the new ones
    /// empty: the ranking's shares too where the tier is the one store.
    pub fn grow_host(&mut self, group: usize, records: u32) {
        self.host[group].grow(records);
        if self.placer == crate::planning::Placer::Bank {
            self.resized(group, records as usize);
        }
    }

    /// The fewest records group `group`'s host tier may shrink to: every
    /// record in flight lies below them ([`HostRecords::shrink_floor`]).
    pub fn host_floor(&self, group: usize) -> u32 {
        self.host[group].shrink_floor()
    }

    /// Shrink group `group`'s host tier to `records` records, at least its
    /// floor ([`Self::host_floor`]), between flights with no placement
    /// pending, every record in flight kept in place
    /// ([`HostRecords::shrink`]): the moves whose bytes the caller copies.
    /// The ranking's shares shrink too where the tier is the one store.
    pub fn shrink_host(&mut self, group: usize, records: u32) -> Vec<(HostRecord, HostRecord)> {
        assert!(self.bank.no_fills(), "a host tier shrinks while fills are pending");
        if self.placer == crate::planning::Placer::Bank {
            self.resized(group, records as usize);
        }
        let reuse = Reuse::of(&self.ranking);
        let rank = store_rank(&reuse, &self.prompt_reads);
        self.host[group].shrink(records, rank).into_iter().map(|(from, to)| (HostRecord(from), HostRecord(to))).collect()
    }

    /// The bank no longer holds `layer`'s `expert`, last read at `read`: its
    /// host tier holds it alone where a record keeps its copy, ranked beside
    /// that read, and otherwise only the file holds it. Nothing is written
    /// back.
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

    /// Whether the store serves `layer`'s `expert` without staging: held,
    /// kept by its group's host tier, or landing there.
    pub fn stores(&self, layer: usize, expert: usize) -> bool {
        self.held(layer, expert).is_some() || self.in_tier(layer, expert) || self.lands(layer, expert)
    }

    /// What staging's copy of `layer`'s `expert` is worth keeping
    /// (`Staging::take`): none where the store serves it, else its reuse.
    pub fn staged_worth(&self, (layer, expert): Staged) -> Option<Score> {
        (!self.stores(layer, expert)).then(|| self.reuse(layer, expert, None))
    }

    /// The experts `layer` holds.
    pub fn held_count(&self, layer: usize) -> usize {
        (0..self.bank.experts(layer)).filter(|&expert| self.bank.held(layer, expert).is_some()).count()
    }

    /// Claim lent record `record` of group `group`'s host records, its
    /// lender's, for the prompt group whose first step is `from`
    /// ([`HostRecords::claim_record`]): what it kept unmapped from `from`
    /// on, before any read lands in it. Whether the group holds it; it
    /// waits while a copy reads it or a landing or grace holds it.
    pub fn claim_lent(&mut self, group: usize, record: HostRecord, from: Step) -> bool {
        self.host[group].claim_record(record.0, from)
    }

    /// Whether a prompt group holds conveyor record `record` of group
    /// `group`'s host records claimed.
    pub fn lent_claimed(&self, group: usize, HostRecord(record): HostRecord) -> bool {
        self.host(group).is_claimed(record)
    }

    /// Lend the bank's lent records to the prompt group whose first step
    /// is `from` (beside a host tier, its scratch's), as the conveyor starts
    /// serving it ([`BankRecords::lend`]): each expert one held leaves the
    /// bank, and each promotion queued into one ends without its expert.
    pub fn lend(&mut self, from: Step, lending: &Lending) {
        let Lent { evicted, dropped } = self.bank.lend(from, lending);
        for (layer, expert) in evicted {
            self.left_bank(layer, expert, None);
        }
        for (layer, expert) in dropped {
            self.resettle(layer, expert);
        }
    }

    /// The prompt group past (the device past it, and its conveyor quiet):
    /// every host record of group `lender` it claimed returns empty, and
    /// every bank record lent to it free. Its pending reads
    /// ([`Self::read_pending`]) are forgotten: the ranking logs the group's
    /// reads once its steps end, which may come after this, so until then
    /// they rank as the ranking last logged them.
    pub fn end_prompt_group(&mut self, lender: usize) {
        self.host[lender].restore_lent();
        self.bank.unlend();
        self.prompt_reads.clear();
    }

    /// Seed `layer` at `point` with what a prompt group routed there, beside
    /// a host tier (where the device places): `routed`, each expert with the
    /// step that last routed it. Each routed expert the layer lacks whose
    /// scratch slot the group filled (`filled`), the most reused first
    /// ([`Ranked`]: reuse beside that step), takes a free record, or the
    /// group's least reused record whose expert lies outside its own
    /// layer's share (`Reuse::in_top`, as a release's) while
    /// that ranks below it (beside the later of its ranking's last read and
    /// that step, where the group routed it). No layer's share is taken: a
    /// prompt group leaves every layer the experts the ranking holds for
    /// it. The layer's `pinned` experts (those its merge set's pass reads in
    /// place after the seeding) are never taken. The seeds, placed as fills
    /// its seed copy and commit commit, with their records, and the experts
    /// evicted, (layer, expert).
    pub fn seed(&mut self, point: Point, layer: usize, routed: &[(usize, Step)], pinned: &[usize], filled: impl Fn(usize) -> bool) -> (Vec<(usize, BankRecord)>, Vec<(usize, usize)>) {
        let group = self.bank.group(layer);
        let (absent, mut records) = {
            let reuse = self.reuse_now();
            let mut reads = self.record_reads(&reuse, group);
            for &(expert, recency) in routed {
                if let Some(record) = self.bank.held(layer, expert) {
                    reads[record] = reads[record].max(recency);
                }
            }
            let mut absent = routed
                .iter()
                .filter(|&&(expert, _)| self.bank.held(layer, expert).is_none() && !self.bank.placing((layer, expert)) && filled(expert))
                .map(|&(expert, recency)| self.ranked(&reuse, layer, expert, Some(recency)))
                .collect::<Vec<_>>();
            absent.sort_unstable_by(|a, b| b.cmp(a));
            let pinned = pinned.iter().filter_map(|&expert| self.bank.held(layer, expert)).collect::<Vec<_>>();
            let shared = self.bank.records(group).iter().map(|content| matches!(*content, Record::Held((layer, expert)) if reuse.in_top(layer, expert))).collect::<Vec<_>>();
            let records = self.least_of(&reuse, group, absent.len(), |record| reads[record], |record| pinned.contains(&record) || shared[record]);
            (absent, records)
        };
        // The most reused seed against the least reused record, while it
        // outranks it.
        let taken = absent
            .iter()
            .zip(&records)
            .take_while(|(seed, (held, _))| held.is_none_or(|held| held < **seed))
            .count();
        records.truncate(taken);
        let seeded = absent[..taken].iter().map(|seed| seed.expert).collect::<Vec<_>>();
        let placed = self.place_into(point, layer, &seeded, records);
        let evicted = placed.evicted.iter().map(|&(layer, expert, _)| (layer, expert)).collect();
        (seeded.into_iter().zip(placed.records).collect(), evicted)
    }

    /// A prompt group in flight reads `layer`'s `expert` at its last step
    /// `read`, which the ranking logs when the group ends: until then every
    /// taking, on either placer, ranks it beside that read.
    pub fn read_pending(&mut self, layer: usize, expert: usize, read: Step) {
        let pending = self.prompt_reads.entry((layer, expert)).or_insert(read);
        *pending = (*pending).max(read);
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
    /// past its layer: its taint is known only once it ends. Every tier record graced through a point the device is past
    /// is ready. The seeds committed, (layer, expert, record), which the
    /// host maps.
    pub fn retire(&mut self, progress: Point, commit: impl Fn(usize, BankRecord) -> i32, ending: impl Fn(Point) -> Ending, steps: &Steps) {
        // Only the records' change and its mapping under the steps' lock;
        // what the tier and the ranking learn of it after.
        for (group, (layer, expert), committed) in self.bank.retire_fills(progress, commit, ending, steps) {
            if committed {
                self.host[group].banked((layer, expert));
            }
            self.resettle(layer, expert);
        }
        self.retire_graces(progress);
    }

    /// The first group a pending seed was placed in: the taints of its steps
    /// and every later one are still to be read ([`Self::retire`]).
    pub fn pending_from(&self) -> Option<Step> {
        self.bank.pending_from()
    }

    /// The device is parked at `parked`, a decode request (its tier wait)
    /// of a group of `steps` steps, steps running in (group, layer,
    /// sequence) order, so every resolution before it is done: every graced
    /// record ready but those the parked step resolved there (`read`,
    /// (group, record), `decode::parked_reads`), each one a stamped map may
    /// still name first restamped ([`HostRecords::release`]).
    pub fn release(&mut self, parked: (Point, u64), read: &[(usize, u32)]) {
        for (group, records) in self.host.iter_mut().enumerate() {
            records.release(parked, |record| read.contains(&(group, record)));
        }
    }

    /// The device is at `progress`: every host tier record graced through a
    /// point before it is ready ([`HostRecords::retire`]).
    pub fn retire_graces(&mut self, progress: Point) {
        for records in self.host.iter_mut() {
            records.retire(progress);
        }
    }

    /// Place `absent` experts of `layer` at `point`, keeping the records of
    /// its `held` experts, in the eviction order of [`Self::least`], each
    /// held record ranked beside `read(record)`: a prompt group's seeds,
    /// which its seed commit commits (and the miss replay's model of a bank
    /// whose every miss is placed). The tier learns each evicted expert
    /// ([`Self::left_bank`]). `None` when the group has fewer such records
    /// than `absent` (beside those promotions take).
    pub fn place(
        &mut self,
        point: Point,
        layer: usize,
        held: &[usize],
        absent: &[usize],
        read: impl Fn(usize) -> Step,
    ) -> Option<Placed> {
        let candidates = self.least(&self.reuse_now(), layer, held, absent.len(), &read);
        (candidates.len() == absent.len()).then(|| self.place_into(point, layer, absent, candidates))
    }

    /// Place `absent` experts of `layer` at `point` into `candidates`, one
    /// each, as [`Self::least`] ranks them: each held one's expert evicted.
    fn place_into(&mut self, point: Point, layer: usize, absent: &[usize], candidates: Vec<(Option<Ranked>, BankRecord)>) -> Placed {
        let mut evicted = Vec::new();
        let mut records = Vec::with_capacity(absent.len());
        for (&expert, (held, record)) in absent.iter().zip(candidates) {
            let victim = self.bank.fill(point, (layer, expert), record);
            assert_eq!(victim, held.map(|held| (held.layer, held.expert)), "a seed evicts what its record holds");
            if let Some(Ranked { rank, layer: owner, expert }) = held {
                let read = rank.last().expect("a placement ranks what it evicts beside a read");
                evicted.push((owner, expert, read));
                self.left_bank(owner, expert, Some(read));
            }
            self.resettle(layer, expert);
            records.push(record);
        }
        Placed { records, evicted }
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
        self.least_of(reuse, self.bank.group(layer), count, read, |record| kept.contains(&record))
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
        let mut held = self
            .bank
            .records(group)
            .iter()
            .enumerate()
            .filter_map(|(record, content)| match *content {
                Record::Held((layer, expert)) if !excluded(record) => Some((Some(self.ranked(reuse, layer, expert, Some(read(record)))), record)),
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

    /// Each group's records, its committed ones.
    pub fn records(&self) -> Vec<usize> {
        self.bank.counts()
    }

    /// How group `group` retires its records from `records` up, on its way
    /// to `target` records (at most `records`), between flights (no fill
    /// pending), changing nothing yet. The target keeps the group's most
    /// reused experts, by their rank ([`Ranked`]) beside
    /// `last_read(layer, expert)` (the step that last read it, none before
    /// any: least recent), as many as its records below `target` hold. Of the experts from
    /// `records` up, each the target keeps moves to a record below `target`
    /// (a free one, else one whose expert the target drops, least recent
    /// first, which is evicted), and the rest are evicted: so a shrink
    /// paced over several resizes moves each kept expert once, where its
    /// last resize would. The evictions are taken first ([`Self::evict`]),
    /// then the device moves each move's bytes, then
    /// [`Self::commit_shrink`] takes the rest of the plan. Between flights
    /// the conveyor has returned what it was lent.
    pub fn plan_shrink(
        &self,
        group: usize,
        records: usize,
        target: usize,
        last_read: impl Fn(usize, usize) -> Option<Step>,
    ) -> Shrink {
        let contents = self.bank.records(group);
        assert!(target <= records && records <= contents.len(), "a shrink of {} records to {records}, then {target}", contents.len());
        let held = contents
            .iter()
            .enumerate()
            .filter_map(|(record, content)| match *content {
                Record::Held((layer, expert)) => Some((layer, expert, last_read(layer, expert), record)),
                Record::Free => None,
                Record::Filling(..) => unreachable!("between flights no fill is pending"),
                Record::Promoting(..) | Record::Releasing { .. } => unreachable!("between flights every promotion and release has ended"),
                Record::Lent { .. } => unreachable!("the conveyor returned its records"),
                Record::Shrinking | Record::Moving(_) => unreachable!("a shrink planned while another's is uncommitted"),
            })
            .collect::<Vec<_>>();
        // Ranked by one hold of the ranking, after `last_read` (which may
        // hold it) has answered.
        let mut held = {
            let reuse = self.reuse_now();
            held.into_iter().map(|(layer, expert, read, record)| (self.ranked(&reuse, layer, expert, read), record)).collect::<Vec<_>>()
        };
        held.sort_unstable_by(|a, b| b.cmp(a));
        let (kept, dropped) = held.split_at(target.min(held.len()));
        let movers = kept.iter().filter(|held| held.1 >= records).collect::<Vec<_>>();
        // Below the target, the free records, then those whose experts the
        // target drops, least recent first.
        let displaced = dropped
            .iter()
            .rev()
            .filter(|held| held.1 < target)
            .take(movers.len().saturating_sub((0..target).filter(|&record| contents[record] == Record::Free).count()))
            .collect::<Vec<_>>();
        let mut to = (0..target)
            .filter(|&record| contents[record] == Record::Free)
            .chain(displaced.iter().map(|held| held.1));
        Shrink {
            records,
            evicted: dropped
                .iter()
                .filter(|held| held.1 >= records)
                .chain(displaced.iter().copied())
                .map(|&(Ranked { layer, expert, .. }, _)| (layer, expert))
                .collect(),
            moved: movers
                .into_iter()
                .map(|&(Ranked { layer, expert, .. }, from)| {
                    let to = to.next().expect("a kept expert has a record below the target");
                    (layer, expert, from, to)
                })
                .collect(),
        }
    }

    /// Evict what `plan` ([`Self::plan_shrink`], nothing changed since)
    /// evicts, from the first step not yet begun, and hold every record it
    /// moves into or retires from every placement until it commits
    /// ([`BankRecords::shrinking`]): always safe, as the group only loses
    /// experts, and no promotion takes a record a move writes meanwhile.
    /// The host tier learns each, ranked by `last_read` (the plan's
    /// ranking): one whose copy it keeps is held alone there, and the rest
    /// are dropped, the file holding them. Nothing is written back.
    pub fn evict(&mut self, group: usize, plan: &Shrink, last_read: impl Fn(usize, usize) -> Option<Step>, at: &Unbegun) {
        for (layer, expert) in self.bank.shrinking(group, plan, at) {
            self.left_bank(layer, expert, last_read(layer, expert));
        }
    }

    /// Of `count` candidates, `candidate(i)` the records of every group of
    /// the `i`th, shallowest first, the last the shrink's target, the deepest
    /// whose moves toward the target ([`Self::plan_shrink`]'s, nothing
    /// changed since), the records each group moves, `budget` fits, or the
    /// shallowest where none does: one resize moves at most what its budget
    /// allows, and takes at least its first step. A deeper candidate moves
    /// no fewer records of any group, so the deepest that fits is found by
    /// halving: it prices the target and O(log count) candidates, not all.
    pub fn paced(
        &self,
        count: usize,
        candidate: impl Fn(usize) -> Vec<usize>,
        budget: impl Fn(&dyn Fn(usize) -> u64) -> bool,
        last_read: impl Fn(usize, usize) -> Option<Step>,
    ) -> usize {
        let target = candidate(count.checked_sub(1).expect("a shrink has a target"));
        let fits = |index: usize| -> bool {
            let records = candidate(index);
            budget(&|group| self.plan_shrink(group, records[group], target[group], &last_read).moved.len() as u64)
        };
        if !fits(0) {
            return 0;
        }
        // The deepest known to fit, and the shallowest known not to.
        let (mut fitting, mut over) = (0, count);
        while over - fitting > 1 {
            let middle = fitting + (over - fitting) / 2;
            match fits(middle) {
                true => fitting = middle,
                false => over = middle,
            }
        }
        fitting
    }

    /// The first record of group `group` a seed may take
    /// ([`BankRecords::seeded_from`]): what a seed list names its records
    /// from.
    pub fn seeded_from(&self, group: usize) -> usize {
        self.bank.seeded_from(group)
    }

    /// Whether a shrink is planned and not yet committed
    /// ([`BankRecords::shrink_pending`]): residency promotes and releases
    /// nothing until it commits or is dropped.
    pub fn shrink_pending(&self) -> bool {
        self.bank.shrink_pending()
    }

    /// A shrink planned and never committed (its owner dropped the plan)
    /// holds its records no more ([`BankRecords::unshrink`]): a new plan
    /// replaces it.
    pub fn unshrink(&mut self) {
        self.bank.unshrink();
    }

    /// Retire group `group`'s records as `plan` says, its evictions taken
    /// ([`Self::evict`]) and its moves queued on the device ahead of any
    /// later step that reads the records.
    pub fn commit_shrink(&mut self, group: usize, plan: &Shrink, at: &Unbegun) {
        self.bank.commit_shrink(group, plan, at);
        self.resized(group, plan.records);
    }

    /// Commit group `group`'s records up to `records`, the new ones free.
    pub fn grow(&mut self, group: usize, records: usize) {
        self.bank.grow(group, records);
        self.resized(group, records);
    }
}

#[cfg(test)]
mod tests;
