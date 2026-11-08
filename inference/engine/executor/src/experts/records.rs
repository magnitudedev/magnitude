//! The bank's records: the one owner of what
//! each bank record holds and of every layer's map entry naming one
//! (`experts::map`), beside a host tier the map's one writer. Every change
//! of either is one of the transitions below, each from the states it may
//! leave, panicking from any other: no caller sets a record, an expert's
//! record or a map entry itself, so none can leave them disagreeing.
//!
//! - `Free → Promoting(Queued)`: [`BankRecords::promote`], the ranking's
//!   promotion into a record no step reads; `Queued → Copying`:
//!   [`BankRecords::copy`]; `Copying → Held`: [`BankRecords::promoted`],
//!   mapped from the first step not yet begun; `Promoting → Free`:
//!   [`BankRecords::unpromoted`]. Never into a lent record a prompt group
//!   holds reserved.
//! - `Held → Releasing`: [`BankRecords::release`], for the reserve, its
//!   expert unmapped from the first step not yet begun; `Releasing → Free`:
//!   [`BankRecords::drain`], once every step its entry was mapped for, from
//!   its mapping's step to that one, has passed its layer or ended.
//! - `Free | Held → Filling`: [`BankRecords::fill`], a prompt group's seed,
//!   its victim unmapped from the group's first step; never of an expert
//!   the map holds for a step of the group, whose scratch slot no round
//!   filled. `Filling → Held | Free`: [`BankRecords::retire_fills`].
//! - `Free | Held | Promoting(Queued) | Releasing → Lent`:
//!   [`BankRecords::lend`], once a prompt group, as the conveyor serves it:
//!   what it held unmapped from the group's first step, a queued promotion
//!   dropped; never a copy's record, a fill's, or a release's whose expert
//!   the group's steps may still read. `Lent → Free`:
//!   [`BankRecords::unlend`].
//! - `Held | Free → Shrinking`, `Held → Moving`: [`BankRecords::shrinking`],
//!   a shrink's evictions unmapped, the records it moves into or retires
//!   and those it moves from held from every other transition until it
//!   commits, residency asking no promotion or release meanwhile;
//!   `Shrinking → Held`, the rest truncated: [`BankRecords::commit_shrink`];
//!   `Shrinking → Free`, `Moving → Held`, a plan dropped uncommitted:
//!   [`BankRecords::unshrink`]. [`BankRecords::grow`] adds free ones.
//!
//! A mapping from the first step not yet begun takes the steps' [`Unbegun`]
//! (`Steps::unbegun`, its one maker), so it is stored under the lock every
//! step begins under; an unmapping from a prompt group's first step, a step
//! begun already, needs no lock. Every transition that frees a record or
//! takes one a promotion could use marks the records changed
//! ([`BankRecords::take_changed`]), so the promotions are asked again.

use super::directory::{BankRecord, HostRecord, Point, Promotion, Shrink, bank_record};
use super::map::{MapView, StampedMap};
use super::staging::Staged;
use super::watcher::{Lending, LentRecords, Step, Unbegun};
use std::collections::HashSet;

/// What a bank record holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Record {
    /// Nothing any step reads.
    Free,
    /// The expert, mapped from its mapping on.
    Held(Staged),
    /// Taken by a promotion of the expert, until its copy completes.
    Promoting(Staged, Moving),
    /// Released for the reserve: its expert of `layer`, mapped from step
    /// `from`, is unmapped from step `until` on, and it is free once every
    /// step of `[from, until)` has passed the layer or ended.
    Releasing { layer: usize, from: Step, until: Step },
    /// Taken by a prompt group's seed of the expert placed at the point,
    /// until its step ends.
    Filling(Staged, Point),
    /// Lent to the prompt group whose first step is `from`: its scratch
    /// slots, which no step reads through the map.
    Lent { from: Step },
    /// A shrink's, planned and not yet committed: a move's target, or a
    /// record past it.
    Shrinking,
    /// A move's source of a shrink planned and not yet committed: it holds
    /// the expert, mapped, until the commit maps it to its target.
    Moving(Staged),
}

/// Where a promotion is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Moving {
    /// Asked, its copy not yet queued on the copy engine.
    Queued,
    /// Copying from tier record `tier`, which it holds, by the copy
    /// engine's batch `batch`.
    Copying { tier: HostRecord, batch: u64 },
}

/// A group's records, a bit per record set while it is free. Every change
/// of a record goes through [`Self::set`], so the bits are the free records
/// by construction, and a placement finds free records without ranking the
/// group's held ones (that ranking, over 8,880 records, measured 167 µs of
/// each miss's service).
struct Records {
    contents: Vec<Record>,
    free: Vec<u64>,
}

impl Records {
    fn new(count: usize) -> Self {
        let mut records = Self { contents: Vec::new(), free: Vec::new() };
        records.grow(count);
        records
    }

    fn free(&self) -> impl Iterator<Item = usize> + '_ {
        self.free.iter().enumerate().flat_map(|(word, &bits)| {
            std::iter::successors((bits != 0).then_some(bits), |&rest| {
                let rest = rest & (rest - 1);
                (rest != 0).then_some(rest)
            })
            .map(move |rest| word * 64 + rest.trailing_zeros() as usize)
        })
    }

    fn set(&mut self, record: usize, content: Record) {
        let (word, bit) = (record / 64, 1u64 << (record % 64));
        match content {
            Record::Free => self.free[word] |= bit,
            _ => self.free[word] &= !bit,
        }
        self.contents[record] = content;
    }

    fn truncate(&mut self, records: usize) {
        self.contents.truncate(records);
        self.free.truncate(records.div_ceil(64));
        if let Some(last) = self.free.last_mut().filter(|_| records % 64 != 0) {
            *last &= (1u64 << (records % 64)) - 1;
        }
    }

    fn grow(&mut self, records: usize) {
        let from = self.contents.len();
        self.contents.resize(records, Record::Free);
        self.free.resize(records.div_ceil(64), 0);
        for record in from..records {
            self.free[record / 64] |= 1u64 << (record % 64);
        }
    }
}

/// What a prompt group's lend took: the experts it unmapped, which leave
/// the bank, and those whose queued promotions it dropped, which the bank
/// lacks.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Lent {
    pub evicted: Vec<Staged>,
    pub dropped: Vec<Staged>,
}

pub(super) struct BankRecords {
    /// Per group, its records.
    groups: Vec<Records>,
    /// Per layer, its group and the record holding each expert.
    layers: Vec<(usize, Vec<Option<usize>>)>,
    /// The experts a promotion or a seed takes a record for: no second
    /// placement takes one, so an expert is in one record at most.
    placing: HashSet<Staged>,
    /// The records promotions, releases and fills take, (group, record),
    /// in the order they were taken.
    promoting: Vec<(usize, usize)>,
    releasing: Vec<(usize, usize)>,
    filling: Vec<(usize, usize)>,
    /// Whether a record was freed, or one a promotion could use was taken,
    /// since the promotions last asked ([`Self::take_changed`]).
    changed: bool,
    /// The records a shrink planned and not yet committed holds, shrinking
    /// or moving: while any, no transition but the commit or the plan's
    /// drop takes a record ([`Self::shrink_pending`]).
    shrink_held: usize,
    /// Beside a host tier, each layer's map, whose one writer this is, and
    /// the lender's bank group and its lent records, `[0, records)`: none
    /// without a device (an offline replay) or where the host places, whose
    /// landings write the one store's.
    tier: Option<(Vec<StampedMap>, (usize, usize))>,
    /// Bumped as any expert comes to be held or leaves: what each layer
    /// holds changed since a reader last saw it ([`Self::holdings`]).
    holdings: u64,
}

impl BankRecords {
    /// Groups of `records` records each, every one free, of `layers`,
    /// each (its group, its experts), none held; beside a host tier mapped
    /// through its maps, its lender lending its lent records (`tier`).
    pub fn new(records: &[usize], layers: &[(usize, usize)], tier: Option<(Vec<StampedMap>, (usize, usize))>) -> Self {
        if let Some((maps, (group, count))) = &tier {
            assert_eq!(maps.len(), layers.len(), "each layer has its map");
            assert!(*count <= records[*group], "a group lends {count} of its {} records", records[*group]);
        }
        Self {
            groups: records.iter().map(|&count| Records::new(count)).collect(),
            layers: layers.iter().map(|&(group, experts)| (group, vec![None; experts])).collect(),
            placing: HashSet::new(),
            promoting: Vec::new(),
            releasing: Vec::new(),
            filling: Vec::new(),
            changed: false,
            shrink_held: 0,
            tier,
            holdings: 0,
        }
    }

    /// Each layer's map, beside a host tier.
    fn maps(&self) -> Option<&[StampedMap]> {
        self.tier.as_ref().map(|(maps, _)| maps.as_slice())
    }

    /// The lender's bank group and its lent records, beside a host tier.
    fn lent(&self) -> Option<(usize, usize)> {
        self.tier.as_ref().map(|&(_, lent)| lent)
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    pub fn experts(&self, layer: usize) -> usize {
        self.layers[layer].1.len()
    }

    /// `layer`'s group.
    pub fn group(&self, layer: usize) -> usize {
        self.layers[layer].0
    }

    /// The count of changes to what any layer holds: equal counts, equal
    /// holdings.
    pub fn holdings(&self) -> u64 {
        self.holdings
    }

    /// The record holding `layer`'s `expert`.
    pub fn held(&self, layer: usize, expert: usize) -> Option<usize> {
        self.layers[layer].1[expert]
    }

    /// Whether a promotion or a seed takes a record for `staged`.
    pub fn placing(&self, staged: Staged) -> bool {
        self.placing.contains(&staged)
    }

    /// Group `group`'s records.
    pub fn records(&self, group: usize) -> &[Record] {
        &self.groups[group].contents
    }

    /// Group `group`'s free records, in record order.
    pub fn free(&self, group: usize) -> impl Iterator<Item = usize> + '_ {
        self.groups[group].free()
    }

    /// The groups' record counts.
    pub fn counts(&self) -> Vec<usize> {
        self.groups.iter().map(|records| records.contents.len()).collect()
    }

    /// The first record of group `group` a seed may take: past the lent
    /// ones where it is the lender (`ExpertBankPlan::seeded_from`), which
    /// a seed never takes ([`Self::fill`]).
    pub fn seeded_from(&self, group: usize) -> usize {
        self.lent().filter(|&(lender, _)| lender == group).map_or(0, |(_, records)| records)
    }

    /// Whether record `record` of group `group` is a lent one.
    pub fn lends(&self, group: usize, record: usize) -> bool {
        self.lent().is_some_and(|(lender, records)| lender == group && record < records)
    }

    /// The record step `step`'s pass reads `layer`'s `expert` in, as the
    /// device resolves the map: none where it reads it from its scratch
    /// slot. It never names a record lent to that step's group: the lend
    /// unmapped what it held from the group's first step, and nothing is
    /// mapped into a lent record.
    pub fn resolve(&self, layer: usize, expert: usize, step: Step) -> Option<BankRecord> {
        let record = self.view(layer).resolve(expert, step)?;
        let group = self.layers[layer].0;
        if let Record::Lent { from } = self.groups[group].contents[record] {
            assert!(step < from, "step {step} resolves expert {expert} of layer {layer} in record {record}, lent from step {from}");
        }
        Some(BankRecord(u32::try_from(record).expect("bank records fit u32")))
    }

    /// `layer`'s map, as its readers read it.
    fn view(&self, layer: usize) -> &MapView {
        self.maps().expect("a bank that resolves steps maps them")[layer].view()
    }

    fn map(&self, (layer, expert): Staged, record: usize, from: Step) {
        if let Some(maps) = self.maps() {
            maps[layer].map(expert, record, from);
        }
    }

    fn unmap(&self, (layer, expert): Staged, until: Step) {
        if let Some(maps) = self.maps() {
            maps[layer].unmap(expert, until);
        }
    }

    fn set(&mut self, group: usize, record: usize, content: Record) {
        // Free, releasing, or neither: what a promotion and the reserve see.
        let kind = |content: Record| match content {
            Record::Free => 0,
            Record::Releasing { .. } => 1,
            _ => 2,
        };
        let old = self.groups[group].contents[record];
        self.changed |= kind(old) != kind(content);
        let shrink = |content: Record| matches!(content, Record::Shrinking | Record::Moving(_));
        self.shrink_held = self.shrink_held + usize::from(shrink(content)) - usize::from(shrink(old));
        self.groups[group].set(record, content);
    }

    /// Whether a record was freed, or one a promotion could use taken,
    /// since this was last asked.
    pub fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    /// Whether a record was freed, or one a promotion could use taken,
    /// since [`Self::take_changed`] was last asked.
    pub fn changed(&self) -> bool {
        self.changed
    }

    /// Whether a shrink is planned and not yet committed: it holds the
    /// records it moves from, into, and past until its commit or its drop,
    /// and residency asks no promotion or release meanwhile.
    pub fn shrink_pending(&self) -> bool {
        self.shrink_held > 0
    }

    /// Whether a promotion may take record `record` of group `group`, a
    /// prompt group holding the lent records reserved or not (`lending`).
    pub fn usable(&self, group: usize, record: usize, lending: &Lending) -> bool {
        !(lending.reserved() && self.lends(group, record))
    }

    /// Group `group`'s records a promotion may take now or once released:
    /// those free or releasing, but a lent one while a prompt group holds
    /// them reserved, and none lent to it.
    pub fn covering(&self, group: usize, lending: &Lending) -> usize {
        let free = self.free(group).filter(|&record| self.usable(group, record, lending)).count();
        let releasing = self.releasing.iter().filter(|&&(of, record)| of == group && self.usable(group, record, lending)).count();
        free + releasing
    }

    /// Group `group`'s lent records a prompt group holds reserved that
    /// return free as it ends (lent to it, or free or releasing beside its
    /// reservation): none a promotion may take meanwhile, each one it may
    /// once the group ends.
    pub fn returning(&self, group: usize, lending: &Lending) -> usize {
        match self.lent() {
            Some((lender, records)) if lender == group && lending.reserved() => self.groups[group].contents[..records]
                .iter()
                .filter(|content| matches!(content, Record::Free | Record::Releasing { .. } | Record::Lent { .. }))
                .count(),
            _ => 0,
        }
    }

    /// `staged` held in `record` of its group from now.
    fn hold(&mut self, staged: Staged, record: usize) {
        let held = &mut self.layers[staged.0].1[staged.1];
        assert!(held.is_none(), "expert {staged:?} is held already, in record {held:?}");
        *held = Some(record);
        self.holdings += 1;
        self.set(self.layers[staged.0].0, record, Record::Held(staged));
    }

    /// The expert `record` of `group` holds leaves it, unmapped from
    /// `until`: the expert, or none where the record was free.
    fn vacate(&mut self, group: usize, record: usize, until: Step) -> Option<Staged> {
        match self.groups[group].contents[record] {
            Record::Free => None,
            Record::Held(staged) => {
                self.layers[staged.0].1[staged.1] = None;
                self.holdings += 1;
                self.unmap(staged, until);
                Some(staged)
            }
            other => panic!("record {record} of group {group} is taken from {other:?}"),
        }
    }

    /// Hold every expert of every layer at `home(layer) + expert` (a bank
    /// imported whole, before any step), mapped for every step.
    pub fn hold_homes(&mut self, home: impl Fn(usize) -> usize) {
        for layer in 0..self.layers.len() {
            for expert in 0..self.layers[layer].1.len() {
                let record = home(layer) + expert;
                assert_eq!(self.groups[self.layers[layer].0].contents[record], Record::Free, "a home of a whole bank is free");
                self.hold((layer, expert), record);
                self.map((layer, expert), record, 0);
            }
        }
    }

    /// Promotion `promotion` takes its free record, which no step reads,
    /// until its copy has completed. Never a lent record while a prompt
    /// group holds them reserved (`lending`): its lend would find the copy
    /// writing it.
    pub fn promote(&mut self, promotion: &Promotion, lending: &Lending) {
        let Promotion { group, record, layer, expert } = *promotion;
        let record = record.index();
        assert!(self.usable(group, record, lending), "a promotion into lent record {record} a prompt group holds reserved");
        assert_eq!(self.groups[group].contents[record], Record::Free, "a promotion takes a free record");
        assert!(self.placing.insert((layer, expert)), "expert {expert} of layer {layer} is placed twice");
        self.set(group, record, Record::Promoting((layer, expert), Moving::Queued));
        self.promoting.push((group, record));
    }

    /// The promotions queued, each (group, record, expert), but those into
    /// lent records while a prompt group holds them reserved (`lending`),
    /// which its lend drops.
    pub fn queued(&self, lending: &Lending) -> Vec<(usize, BankRecord, Staged)> {
        self.promoting
            .iter()
            .filter_map(|&(group, record)| match self.groups[group].contents[record] {
                Record::Promoting(staged, Moving::Queued) if self.usable(group, record, lending) => Some((group, bank_record(record), staged)),
                _ => None,
            })
            .collect()
    }

    /// Release record `record` of group `group` for the reserve, its expert
    /// unmapped from the first step not yet begun, under the steps' lock
    /// with `lending` (a prompt group reserves before its steps begin): it
    /// is free once every step its entry was mapped for has passed its
    /// layer or ended ([`Self::drain`]). Never a lent record while a prompt
    /// group holds them reserved, whose lend unmaps what it holds instead.
    /// Its expert.
    pub fn release(&mut self, group: usize, BankRecord(record): BankRecord, at: &Unbegun, lending: &Lending) -> Staged {
        let record = record as usize;
        assert!(self.usable(group, record, lending), "a release of lent record {record} a prompt group holds reserved");
        let Some(staged @ (layer, expert)) = self.vacate(group, record, at.next()) else {
            panic!("a release of record {record} of group {group}, which holds nothing");
        };
        // Its entry's FROM, which the unmapping leaves: no step before it
        // resolves the entry to the record. None mapped without a device.
        let from = self.maps().map_or(0, |maps| maps[layer].view().from(expert));
        self.set(group, record, Record::Releasing { layer, from, until: at.next() });
        self.releasing.push((group, record));
        staged
    }

    /// Every release the device is past at `progress` is free: each step
    /// its entry was mapped for has passed its layer or ended
    /// (`Steps::served`), so no step reads it. Whether any was.
    pub fn drain(&mut self, progress: Point, steps: &super::watcher::Steps) -> bool {
        let (drained, releasing) = std::mem::take(&mut self.releasing).into_iter().partition::<Vec<_>, _>(|&(group, record)| {
            let Record::Releasing { layer, from, until } = self.groups[group].contents[record] else {
                panic!("record {record} of group {group} is listed releasing");
            };
            steps.served(from, until, layer, progress)
        });
        self.releasing = releasing;
        for &(group, record) in &drained {
            self.set(group, record, Record::Free);
        }
        !drained.is_empty()
    }

    /// The promotions copying, each (group, record, batch).
    pub fn copying(&self) -> Vec<(usize, BankRecord, u64)> {
        self.promoting
            .iter()
            .filter_map(|&(group, record)| match self.groups[group].contents[record] {
                Record::Promoting(_, Moving::Copying { batch, .. }) => Some((group, bank_record(record), batch)),
                _ => None,
            })
            .collect()
    }

    /// Whether any promotion is in flight or any release draining.
    pub fn moving(&self) -> bool {
        !self.promoting.is_empty() || !self.releasing.is_empty()
    }

    /// The promotions in flight and the releases draining, for the
    /// exchange's watchdog.
    pub fn describe(&self) -> String {
        let records = |taken: &[(usize, usize)]| taken.iter().map(|&(group, record)| self.groups[group].contents[record]).collect::<Vec<_>>();
        format!("promotions {:?}, releases {:?}", records(&self.promoting), records(&self.releasing))
    }

    /// Queued promotion `record` of group `group` copies from tier record
    /// `tier`, which it holds, by batch `batch`: counted against the
    /// reservation (`lending`, held across the batch's queueing) where its
    /// record is a lent one.
    pub fn copy(&mut self, group: usize, BankRecord(record): BankRecord, tier: HostRecord, batch: u64, lending: &mut Lending) {
        let record = record as usize;
        let Record::Promoting(staged, Moving::Queued) = self.groups[group].contents[record] else {
            panic!("record {record} of group {group} copies no queued promotion");
        };
        if self.lends(group, record) {
            lending.copying();
        }
        self.set(group, record, Record::Promoting(staged, Moving::Copying { tier, batch }));
    }

    /// Take promotion `record` of group `group` out of what is promoting:
    /// its expert, and the tier record its copy held.
    fn end_promotion(&mut self, group: usize, record: usize, lent: &LentRecords) -> (Staged, Option<HostRecord>) {
        let Record::Promoting(staged, moving) = self.groups[group].contents[record] else {
            panic!("record {record} of group {group} is no promotion's");
        };
        self.promoting.retain(|&taken| taken != (group, record));
        self.placing.remove(&staged);
        match moving {
            Moving::Queued => (staged, None),
            Moving::Copying { tier, .. } => {
                if self.lends(group, record) {
                    lent.copied();
                }
                (staged, Some(tier))
            }
        }
    }

    /// The copies of promotions `completed`, each (group, record), have
    /// completed: each holds its expert, mapped from the first step not yet
    /// begun, the maps alone written under the steps' lock (`steps`). Each
    /// one's expert, and the tier record its copy held.
    pub fn promoted(&mut self, completed: Vec<(usize, BankRecord)>, steps: &super::watcher::Steps, lent: &LentRecords) -> Vec<(Staged, HostRecord)> {
        let promoted = completed
            .into_iter()
            .map(|(group, BankRecord(record))| {
                let record = record as usize;
                let (staged, tier) = self.end_promotion(group, record, lent);
                let tier = tier.unwrap_or_else(|| panic!("record {record} of group {group} completed a promotion it never copied"));
                self.hold(staged, record);
                (staged, record, tier)
            })
            .collect::<Vec<_>>();
        steps.unbegun(|at| {
            for &(staged, record, _) in &promoted {
                self.map(staged, record, at.next());
            }
        });
        promoted.into_iter().map(|(staged, _, tier)| (staged, tier)).collect()
    }

    /// Promotion `record` of group `group` ended without its expert (its
    /// tier record went before its copy began, or its copy failed): the
    /// record is free, as nothing maps it. Its expert, and the tier record
    /// a copy held.
    pub fn unpromoted(&mut self, group: usize, BankRecord(record): BankRecord, lent: &LentRecords) -> (Staged, Option<HostRecord>) {
        let record = record as usize;
        let ended = self.end_promotion(group, record, lent);
        self.set(group, record, Record::Free);
        ended
    }

    /// A prompt group's seed of `staged` at `point` takes `record`, until
    /// its step ends ([`Self::retire_fills`]), its victim unmapped from the
    /// group's first step: the group's steps read none of its records but
    /// their hits, which no seed takes. The seed copies the expert's
    /// scratch slot, which a round filled only where none of the group's
    /// steps resolved it held: seeding one the map holds for any of them
    /// (`point.group` to the seeding step) would bank an unfilled slot. The
    /// victim.
    pub fn fill(&mut self, point: Point, staged: Staged, BankRecord(record): BankRecord) -> Option<Staged> {
        let record = record as usize;
        let group = self.layers[staged.0].0;
        assert!(self.layers[staged.0].1[staged.1].is_none(), "seeded expert {staged:?} is held");
        if self.maps().is_some() {
            let view = self.view(staged.0);
            assert!(
                (point.group..=point.sequence).all(|step| view.resolve(staged.1, step).is_none()),
                "a seed of expert {staged:?}, which the map holds for a step of group {}: its scratch slot no round filled",
                point.group
            );
        }
        assert!(self.placing.insert(staged), "expert {staged:?} is placed twice");
        let victim = self.vacate(group, record, point.group);
        self.set(group, record, Record::Filling(staged, point));
        self.filling.push((group, record));
        victim
    }

    /// The fill of record `record` of group `group`: its expert and point.
    fn fill_of(&self, (group, record): (usize, usize)) -> (Staged, Point) {
        match self.groups[group].contents[record] {
            Record::Filling(staged, point) => (staged, point),
            other => panic!("record {record} of group {group} is listed filling, holding {other:?}"),
        }
    }

    /// Whether no fill is pending.
    pub fn no_fills(&self) -> bool {
        self.filling.is_empty()
    }

    /// The first group a pending fill was placed in.
    pub fn pending_from(&self) -> Option<Step> {
        self.filling.iter().map(|&taken| self.fill_of(taken).1.group).min()
    }

    /// Retire every fill placed before `progress` whose step and group
    /// have ended (`ending`, read once a step, before the steps' lock): it holds its
    /// expert where its step ended whole with the device's commit
    /// (`commit(group, record)` holding the step's low word), mapped from
    /// the first step not yet begun, under the steps' lock (`steps`); else
    /// it is free. Each retired, (group, expert, committed).
    pub fn retire_fills(
        &mut self,
        progress: Point,
        commit: impl Fn(usize, BankRecord) -> i32,
        mut ending: impl FnMut(Point) -> super::directory::Ending,
        steps: &super::watcher::Steps,
    ) -> Vec<(usize, Staged, bool)> {
        use super::directory::Ending;
        let mut endings = std::collections::HashMap::new();
        let mut retiring = Vec::new();
        let filling = std::mem::take(&mut self.filling);
        for (group, record) in filling {
            let (staged, point) = self.fill_of((group, record));
            let ending = (point < progress).then(|| *endings.entry(point.sequence).or_insert_with(|| ending(point)));
            match ending {
                None | Some(Ending::Running) => self.filling.push((group, record)),
                Some(ending) => {
                    let committed = ending == Ending::Ended && commit(group, bank_record(record)) == super::watcher::low_word(point.sequence);
                    retiring.push((group, record, staged, committed));
                }
            }
        }
        let mut mapped = Vec::new();
        for &(group, record, staged, committed) in &retiring {
            self.placing.remove(&staged);
            match committed {
                true => {
                    self.hold(staged, record);
                    mapped.push((staged, record));
                }
                false => self.set(group, record, Record::Free),
            }
        }
        // The maps alone under the steps' lock.
        if !mapped.is_empty() {
            steps.unbegun(|at| {
                for &(staged, record) in &mapped {
                    self.map(staged, record, at.next());
                }
            });
        }
        retiring.into_iter().map(|(group, _, staged, committed)| (group, staged, committed)).collect()
    }

    /// Lend the lent records to the prompt group whose first step is
    /// `from`, which holds them reserved (`lending`), as the conveyor
    /// starts serving it, before it takes any hit: each expert one holds
    /// unmapped from `from` on (the steps before it, which may read it, run
    /// before the group writes it), each queued promotion into one
    /// dropped, and each release of one taken as it is. A release of a lent
    /// record is asked only before the group reserved them, before its
    /// steps began ([`Self::release`]), so its expert is unmapped from a
    /// step at or before `from`; and no promotion's copy into one runs
    /// while a group holds them reserved: so the lend waits on nothing.
    /// Nothing seeds a lent record: a group seeds only once it holds them.
    /// What it took.
    pub fn lend(&mut self, from: Step, lending: &Lending) -> Lent {
        let mut lent = Lent::default();
        let Some((group, records)) = self.lent() else {
            return lent;
        };
        assert!(lending.reserved(), "lent records lent to a group that holds no reservation");
        for record in 0..records {
            match self.groups[group].contents[record] {
                Record::Free => {}
                Record::Held(_) => lent.evicted.extend(self.vacate(group, record, from)),
                Record::Promoting(staged, Moving::Queued) => {
                    self.promoting.retain(|&taken| taken != (group, record));
                    self.placing.remove(&staged);
                    lent.dropped.push(staged);
                }
                Record::Releasing { until, .. } => {
                    assert!(until <= from, "a release of lent record {record} unmapped its expert from step {until}, past step {from} its group reads from");
                    self.releasing.retain(|&taken| taken != (group, record));
                }
                other => panic!("lent record {record} lent from {other:?}"),
            }
            self.set(group, record, Record::Lent { from });
        }
        lent
    }

    /// The prompt group the records were lent to is past: every one is
    /// free.
    pub fn unlend(&mut self) {
        let Some((group, records)) = self.lent() else {
            return;
        };
        for record in 0..records {
            match self.groups[group].contents[record] {
                Record::Lent { .. } => self.set(group, record, Record::Free),
                other => panic!("lent record {record} returns from {other:?}"),
            }
        }
    }

    /// Group `group` shrinks by `plan` (`Directory::plan_shrink`), between
    /// flights: each expert it evicts leaves, unmapped from the first step
    /// not yet begun, and every record a move writes or the shrink retires
    /// is the shrink's until it commits ([`Self::commit_shrink`]), so no
    /// placement takes one meanwhile. The evicted experts.
    pub fn shrinking(&mut self, group: usize, plan: &Shrink, at: &Unbegun) -> Vec<Staged> {
        let mut evicted = Vec::with_capacity(plan.evicted.len());
        for &(layer, expert) in &plan.evicted {
            let record = self.layers[layer].1[expert].expect("an evicted expert is held");
            evicted.extend(self.vacate(group, record, at.next()));
            self.set(group, record, Record::Shrinking);
        }
        for &(layer, expert, from, to) in &plan.moved {
            match self.groups[group].contents[to] {
                Record::Free => self.set(group, to, Record::Shrinking),
                Record::Shrinking => {}
                other => panic!("a shrink moves into record {to} of group {group}, holding {other:?}"),
            }
            assert_eq!(self.groups[group].contents[from], Record::Held((layer, expert)), "a shrink moves {:?} from where it is held", (layer, expert));
            self.set(group, from, Record::Moving((layer, expert)));
        }
        for record in plan.records..self.groups[group].contents.len() {
            match self.groups[group].contents[record] {
                Record::Free => self.set(group, record, Record::Shrinking),
                Record::Shrinking | Record::Moving(_) => {}
                other => panic!("record {record} past a shrink to {} of group {group} holds {other:?}, neither moved nor evicted", plan.records),
            }
        }
        evicted
    }

    /// A shrink planned and never committed (its owner dropped the plan
    /// before queueing its moves) holds its records no more: each it moved
    /// from holds its expert as it did, and each other is free, as nothing
    /// maps it; a new plan replaces it.
    pub fn unshrink(&mut self) {
        for group in 0..self.groups.len() {
            for record in 0..self.groups[group].contents.len() {
                match self.groups[group].contents[record] {
                    Record::Shrinking => self.set(group, record, Record::Free),
                    Record::Moving(staged) => self.set(group, record, Record::Held(staged)),
                    _ => {}
                }
            }
        }
    }

    /// Group `group`'s shrink by `plan` ([`Self::shrinking`]) commits, its
    /// moves queued on the device ahead of every later step: each kept
    /// expert is held at its new record, mapped from the first step not yet
    /// begun, and the group ends at `plan.records`.
    pub fn commit_shrink(&mut self, group: usize, plan: &Shrink, at: &Unbegun) {
        for &(layer, expert, from, to) in &plan.moved {
            assert_eq!(self.groups[group].contents[from], Record::Moving((layer, expert)), "a moved expert moves from where the plan found it");
            assert_eq!(self.groups[group].contents[to], Record::Shrinking, "a move writes a record its shrink holds");
            self.layers[layer].1[expert] = None;
            self.set(group, from, Record::Shrinking);
            self.hold((layer, expert), to);
            self.map((layer, expert), to, at.next());
        }
        let contents = &self.groups[group].contents;
        assert!(contents[plan.records..].iter().all(|&content| content == Record::Shrinking), "a shrink retires a record it does not hold");
        assert!(!contents[..plan.records].iter().any(|content| matches!(content, Record::Shrinking | Record::Moving(_))), "a shrink holds a record it keeps");
        self.shrink_held -= contents.len() - plan.records;
        self.groups[group].truncate(plan.records);
    }

    /// Commit group `group`'s records up to `records`, the new ones free.
    pub fn grow(&mut self, group: usize, records: usize) {
        assert!(records >= self.groups[group].contents.len(), "a growth to fewer records");
        self.changed |= records > self.groups[group].contents.len();
        self.groups[group].grow(records);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::Words;
    use magnitude_kernels::constants::MAP_WORDS;

    const EXPERTS: usize = 8;

    /// A bank of one group of four records, of one layer of eight experts,
    /// whose first two records the lender lends, mapped through a real map;
    /// and the map's view.
    fn bank() -> (BankRecords, MapView) {
        let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let words = EXPERTS as u64 * MAP_WORDS;
        let tensor = seismic::Tensor::shared(&cpu, seismic::Element::i32(), &[EXPERTS as u64, MAP_WORDS]).unwrap();
        let map = StampedMap::new(Words::new(tensor, words, 0).unwrap());
        let view = map.view().clone();
        (BankRecords::new(&[4], &[(0, EXPERTS)], Some((vec![map], (0, 2)))), view)
    }

    fn promotion(record: u32, expert: usize) -> Promotion {
        Promotion { group: 0, record: BankRecord(record), layer: 0, expert }
    }

    /// Hold expert `expert` in free record `record`, mapped from `from`:
    /// its promotion's copy completed.
    fn promoted(bank: &mut BankRecords, record: u32, expert: usize, from: Step) {
        let lent = LentRecords::new();
        bank.promote(&promotion(record, expert), &lent.lock());
        bank.copy(0, BankRecord(record), HostRecord(0), 0, &mut lent.lock());
        let steps = super::super::watcher::Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new());
        steps.begin_through(from - 1);
        bank.promoted(vec![(0, BankRecord(record))], &steps, &lent);
    }

    /// Every transition keeps the record, the expert's index and the map
    /// together: a promotion maps its expert from the first step not yet
    /// begun, a release unmaps it from there and frees its record once the
    /// device is past its layer of every step it was mapped for, a seed's
    /// victim leaves from its group's first step, a seed committed is
    /// mapped.
    #[test]
    fn each_transition_moves_a_record_and_its_map_entry_together() {
        let (mut bank, view) = bank();
        let steps = super::super::watcher::Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new());
        steps.begin_through(13);
        promoted(&mut bank, 2, 3, 4);
        assert_eq!((bank.held(0, 3), view.resolve(3, 3), view.resolve(3, 4)), (Some(2), None, Some(2)));
        let lent = LentRecords::new();
        assert_eq!(bank.release(0, BankRecord(2), &Unbegun::at(9), &lent.lock()), (0, 3));
        assert_eq!((bank.held(0, 3), view.resolve(3, 8), view.resolve(3, 9)), (None, Some(2), None), "the released expert is read until 9");
        assert!(!bank.drain(Point::whole(8, 0, 8), &steps) && !bank.free(0).any(|record| record == 2), "steps below 9 still read it");
        assert!(bank.drain(Point::whole(9, 0, 9), &steps) && bank.free(0).any(|record| record == 2));
        let point = Point::whole(12, 0, 13);
        promoted(&mut bank, 3, 6, 10);
        assert_eq!(bank.fill(point, (0, 7), BankRecord(3)), Some((0, 6)));
        assert_eq!((view.resolve(6, 11), view.resolve(6, 12)), (Some(3), None), "a seed's victim leaves from its group's first step");
        let retired = bank.retire_fills(Point::whole(13, usize::MAX, Step::MAX), |_, _| super::super::watcher::low_word(13), |_| super::super::directory::Ending::Ended, &steps);
        assert_eq!(retired, [(0, (0, 7), true)]);
        assert_eq!((bank.held(0, 7), view.resolve(7, 14)), (Some(3), Some(3)));
    }

    /// A regression: the bank let a prompt group's hit go mid-group (then a
    /// promotion's victim, now a release), unmapping it only past the steps
    /// begun, so the group's steps still read it in the bank and no round
    /// filled its scratch slot; the group's last step seeded it, banking
    /// the unfilled slot. Seeding an expert the map holds for any step of
    /// the group is refused.
    #[test]
    #[should_panic(expected = "its scratch slot no round filled")]
    fn a_seed_of_a_hit_released_mid_group_panics() {
        let (mut bank, _) = bank();
        promoted(&mut bank, 2, 5, 1);
        // Group 3's steps 3 and 4 have begun: the release unmaps 5 from 5.
        let lent = LentRecords::new();
        bank.release(0, BankRecord(2), &Unbegun::at(5), &lent.lock());
        bank.fill(Point::whole(3, 0, 4), (0, 5), BankRecord(3));
    }

    /// A regression: a record taken from a lent one after a prompt group's
    /// steps began (then a promotion's victim, now a release) unmapped its
    /// expert only from a step past the group's first, so the group read it
    /// as a hit while its scratch overwrote the record. Lending a record
    /// whose expert the group may still read is refused.
    #[test]
    #[should_panic(expected = "past step 3 its group reads from")]
    fn lending_a_record_whose_expert_the_group_still_reads_panics() {
        let (mut bank, _) = bank();
        promoted(&mut bank, 0, 5, 1);
        let lent = LentRecords::new();
        bank.release(0, BankRecord(0), &Unbegun::at(7), &lent.lock());
        assert!(lent.reserve());
        bank.lend(3, &lent.lock());
    }

    /// The reservation that makes it so: no promotion takes and no release
    /// unmaps a lent record a prompt group holds reserved.
    #[test]
    #[should_panic(expected = "a prompt group holds reserved")]
    fn a_promotion_into_a_reserved_lent_record_panics() {
        let (mut bank, _) = bank();
        let lent = LentRecords::new();
        assert!(lent.reserve());
        bank.promote(&promotion(1, 6), &lent.lock());
    }

    #[test]
    #[should_panic(expected = "a prompt group holds reserved")]
    fn a_release_of_a_reserved_lent_record_panics() {
        let (mut bank, _) = bank();
        promoted(&mut bank, 1, 5, 1);
        let lent = LentRecords::new();
        assert!(lent.reserve());
        bank.release(0, BankRecord(1), &Unbegun::at(7), &lent.lock());
    }

    /// One asked before the reservation the lend takes as it is: a queued
    /// promotion dropped, a release taken, its expert unmapped from a step
    /// before the group's: the lend takes every lent record at once and
    /// waits on nothing (a claim that waited for such a
    /// promotion, which waited for the group's steps, never ended). What a
    /// lend took is unmapped for every step of the group, so no step of it
    /// resolves a hit in a lent record.
    #[test]
    fn a_lend_waits_on_nothing_and_leaves_no_hit_in_a_lent_record() {
        let (mut bank, view) = bank();
        promoted(&mut bank, 1, 4, 1);
        promoted(&mut bank, 2, 5, 1);
        let lent = LentRecords::new();
        bank.promote(&promotion(0, 6), &lent.lock());
        assert!(lent.reserve());
        let taken = bank.lend(3, &lent.lock());
        assert_eq!(taken, Lent { evicted: vec![(0, 4)], dropped: vec![(0, 6)] });
        assert!(!bank.moving() && !bank.placing((0, 6)));
        assert_eq!((view.resolve(4, 2), view.resolve(4, 3)), (Some(1), None));
        assert_eq!(bank.resolve(0, 4, 3), None);
        bank.unlend();
        assert_eq!(bank.free(0).collect::<Vec<_>>(), [0, 1, 3]);
    }

    /// No copy into a lent record runs while a group holds them reserved:
    /// a group begins only once every such copy completed. A lend meeting
    /// one is refused.
    #[test]
    #[should_panic(expected = "lent from Promoting")]
    fn lending_a_record_a_copy_writes_panics() {
        let (mut bank, _) = bank();
        let lent = LentRecords::new();
        bank.promote(&promotion(0, 6), &lent.lock());
        bank.copy(0, BankRecord(0), HostRecord(0), 0, &mut lent.lock());
        // The group waits for the copy, holding the records reserved.
        assert!(!lent.reserve());
        bank.lend(3, &lent.lock());
    }

    /// Nothing seeds or promotes into a lent record: the transition into
    /// one is refused.
    #[test]
    #[should_panic(expected = "is taken from Lent")]
    fn a_seed_into_a_lent_record_panics() {
        let (mut bank, _) = bank();
        let lent = LentRecords::new();
        assert!(lent.reserve());
        bank.lend(3, &lent.lock());
        bank.fill(Point::whole(3, 0, 3), (0, 2), BankRecord(0));
    }

    /// A reserve's records cover a want only where a promotion may take
    /// them: never a lent record while a prompt group holds them reserved,
    /// nor one lent to it.
    #[test]
    fn lent_records_cover_no_want_a_promotion_cannot_take() {
        let (mut bank, _) = bank();
        let lent = LentRecords::new();
        assert_eq!(bank.covering(0, &lent.lock()), 4);
        assert_eq!(bank.returning(0, &lent.lock()), 0);
        assert!(lent.reserve());
        assert_eq!(bank.covering(0, &lent.lock()), 2, "the reserved lent records cover no want now");
        assert_eq!(bank.returning(0, &lent.lock()), 2, "they return free as the group ends");
        bank.lend(3, &lent.lock());
        let lending = lent.lock();
        assert_eq!((bank.covering(0, &lending), bank.returning(0, &lending)), (2, 2), "nor do the records lent, until they return");
        drop(lending);
        lent.release();
        assert_eq!(bank.covering(0, &lent.lock()), 2);
        assert!(bank.take_changed() && lent.lock().take_freed(), "the lend took free records, the release freed the reservation");
    }

    /// A release between a shrink's plan and its commit of a record the
    /// plan moves from (its expert outside its layer's share, ranked across
    /// layers by the plan): refused, as the commit would find it gone.
    #[test]
    #[should_panic(expected = "is taken from Moving")]
    fn a_release_of_a_shrinks_source_panics() {
        let (mut bank, _) = bank();
        promoted(&mut bank, 0, 1, 1);
        promoted(&mut bank, 3, 2, 1);
        let plan = Shrink { records: 2, evicted: vec![(0, 1)], moved: vec![(0, 2, 3, 0)] };
        bank.shrinking(0, &plan, &Unbegun::at(5));
        assert!(bank.shrink_pending());
        let lent = LentRecords::new();
        bank.release(0, BankRecord(3), &Unbegun::at(5), &lent.lock());
    }

    /// A shrink's plan dropped uncommitted returns what it held: its
    /// sources held as they were, the rest free.
    #[test]
    fn a_dropped_shrink_returns_its_records() {
        let (mut bank, view) = bank();
        promoted(&mut bank, 0, 1, 1);
        promoted(&mut bank, 3, 2, 1);
        let plan = Shrink { records: 2, evicted: vec![(0, 1)], moved: vec![(0, 2, 3, 0)] };
        bank.shrinking(0, &plan, &Unbegun::at(5));
        bank.unshrink();
        assert!(!bank.shrink_pending());
        assert_eq!((bank.held(0, 2), view.held(2)), (Some(3), Some(3)), "the source keeps its expert, mapped");
        assert_eq!(bank.free(0).collect::<Vec<_>>(), [0, 1, 2]);
    }

    /// A shrink's records are its own from its plan to its commit: no
    /// promotion takes a move's target or a record past it meanwhile, which
    /// the commit would overwrite or truncate.
    #[test]
    #[should_panic(expected = "a promotion takes a free record")]
    fn a_promotion_into_a_shrinks_record_panics() {
        let (mut bank, _) = bank();
        promoted(&mut bank, 0, 1, 1);
        promoted(&mut bank, 3, 2, 1);
        let plan = Shrink { records: 2, evicted: vec![(0, 1)], moved: vec![(0, 2, 3, 0)] };
        bank.shrinking(0, &plan, &Unbegun::at(5));
        let lent = LentRecords::new();
        bank.promote(&promotion(0, 4), &lent.lock());
    }
}
