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
//!   mapped from the map rule's step for its layer, which its caller reads
//!   under the lock every step begins under, and for every step its entry
//!   resolved already ([`BankRecords::map`]); `Promoting → Free`:
//!   [`BankRecords::unpromoted`]. Never into a lent record: those are the
//!   scratch's, so no copy into one can run as a prompt group's lease
//!   takes them.
//!
//! Every expert leaves the bank by its one exit (`Directory::exit`, an
//! [`Exit`] every transition from `Held` takes): written back into the
//! host tier where that pays, its record then releasing until the copy
//! ends, else to the tier's copy or the file, its record taken at once.
//!
//! - `Held → Releasing`: [`BankRecords::release`], for the reserve or a
//!   shrink whose victim's exit writes it back, never of a lent record but
//!   by a lease ([`BankRecords::lease`]), its expert unmapped from the
//!   first step not yet begun (a lease's, from its group's first), and,
//!   where its exit writes it back, copied into the host tier from the
//!   record ([`Back`]: queued, [`BankRecords::back_copy`] posts it,
//!   [`BankRecords::backed`] ends it, written or failed);
//!   `Releasing → Free`: [`BankRecords::drain`], once every step its entry
//!   was mapped for, from its mapping's step to that one, has passed its
//!   layer or ended, and its write-back, if any, has ended: nothing
//!   overwrites the bytes a write-back reads.
//! - `Free | Held → Filling`: [`BankRecords::fill`], a prompt group's seed,
//!   its victim unmapped from the group's first step, never one whose exit
//!   writes it back; never of an expert
//!   the map holds for a step of the group, whose scratch slot no round
//!   filled. `Filling → Held | Free`: [`BankRecords::retire_fills`].
//! - `Free | Held → Lent`: [`BankRecords::lease`], a prompt group's lease
//!   as the conveyor serves it: what it held (a home of a bank imported
//!   whole) unmapped from the group's first step, where its exit writes
//!   it back once released and drained. `Lent → Free`:
//!   [`BankRecords::unlend`].
//! - `Free | Held | Releasing → Absent`: [`BankRecords::evict`], a shrink's,
//!   beside running steps or not: the record's storage leaves the bank, its
//!   expert unmapped from the step its caller names (never one whose exit
//!   writes it back, nor a release whose write-back reads it), and nothing
//!   moves (the records allow holes). `Absent → Free`: [`BankRecords::grow`], a slot
//!   whose storage the bank holds again.
//!
//! Every mapping and unmapping is stored from a step its rule makes
//! ([`MapStep`], [`UnmapStep`]): one from the first step not yet begun takes
//! the steps' `Unbegun` (`Steps::unbegun`, its one maker), so it is stored
//! under the lock every step begins under; an unmapping from a prompt
//! group's first step ([`ScratchGroup`]), a step begun already, needs no
//! lock. Every transition that frees a record or takes one a promotion
//! could use marks the records changed ([`BankRecords::take_changed`]), so
//! the promotions are asked again.

use super::directory::{BankRecord, Exit, HostRecord, Point, Promotion, bank_record};
use super::map::{MapView, StampedMap};
use super::staging::Staged;
use super::watcher::{MapFrom, MapStep, ScratchGroup, Step, UnmapStep};
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
    /// step of `[from, until)` has passed the layer or ended and its
    /// write-back into the host tier (`back`), if it has one, has ended.
    Releasing { layer: usize, from: Step, until: Step, back: Option<Back> },
    /// Taken by a prompt group's seed of the expert placed at the point,
    /// until its step ends.
    Filling(Staged, Point),
    /// Lent to the prompt group whose first step is `from`: its scratch
    /// slots, which no step reads through the map.
    Lent { from: Step },
    /// Idle slot reserved for the dedicated even/odd DMA staging ring
    /// (`stage_rings`). Never `Record::Free` or `Record::Held`, so `free()`,
    /// `evictable()`, `seed()`, `promote()`, `fill()`, and `shrink()` cannot
    /// name it by construction.
    StageIdle,
    /// Taken by an opportunistic DMA staging copy of `staged` for `target_step`
    /// in parity half `parity` (`0` = even steps, `1` = odd steps). `tier` and
    /// `batch` are `Some` while the DMA batch is in flight and `None` once
    /// landed (held read-only for `target_step` until `Steps::frontier() > target_step`).
    Staging {
        staged: Staged,
        target_step: Step,
        tier: Option<HostRecord>,
        batch: Option<u64>,
        parity: u8,
    },
    /// A hole: the bank holds no storage for it, so nothing is placed in it
    /// until a growth backs it again.
    Absent,
}

/// A release's write-back of its expert into the host tier, the reverse
/// of a promotion: tier record `tier`, which a landing of the expert took
/// (`HostRecords::claim`), written from the releasing record by the copy
/// engine's batch `batch` once posted (none while queued). The tier holds
/// the expert alone once it is written, so each expert is held in one
/// place: the bank, the tier, or only the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Back {
    pub staged: Staged,
    pub tier: HostRecord,
    pub batch: Option<u64>,
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

/// A group's records, a bit per record set while it is free, the count the
/// bank holds storage for (every one but the holes), and the count lent to
/// a prompt group's lease. Every change of a record goes through
/// [`Self::set`], so the bits and the counts are the free, present and lent
/// records by construction, and a placement finds free records without
/// ranking the group's held ones (that ranking, over 8,880 records,
/// measured 167 µs of each miss's service).
struct Records {
    contents: Vec<Record>,
    free: Vec<u64>,
    present: usize,
    lent: usize,
}

impl Records {
    fn new(count: usize) -> Self {
        let mut records = Self { contents: Vec::new(), free: Vec::new(), present: 0, lent: 0 };
        records.grow(0..count);
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
        let old = std::mem::replace(&mut self.contents[record], content);
        self.present = self.present + usize::from(old == Record::Absent) - usize::from(content == Record::Absent);
        let lent = |content: Record| usize::from(matches!(content, Record::Lent { .. }));
        self.lent = self.lent + lent(content) - lent(old);
    }

    /// Back `records`, each a hole or past every record: free from now.
    fn grow(&mut self, records: impl IntoIterator<Item = usize>) {
        for record in records {
            if record >= self.contents.len() {
                // Past every record: holes up to it, which `set` then fills.
                self.contents.resize(record + 1, Record::Absent);
                self.free.resize((record + 1).div_ceil(64), 0);
            }
            assert_eq!(self.contents[record], Record::Absent, "a growth backs record {record}, which the bank holds");
            self.set(record, Record::Free);
        }
    }
}

pub(super) struct BankRecords {
    /// Per group, its records.
    groups: Vec<Records>,
    /// Per layer, its group and the record holding each expert.
    layers: Vec<(usize, Vec<Option<usize>>)>,
    /// The experts a promotion or a seed takes a record for: no second
    /// placement takes one, so an expert is in one record at most.
    placing: HashSet<Staged>,
    /// The records promotions, releases, fills and opportunistic staging take,
    /// (group, record), in the order they were taken.
    promoting: Vec<(usize, usize)>,
    releasing: Vec<(usize, usize)>,
    filling: Vec<(usize, usize)>,
    staging: Vec<(usize, usize)>,
    /// Whether a record was freed, or one a promotion could use was taken,
    /// since the promotions last asked ([`Self::take_changed`]).
    changed: bool,
    /// Beside a host tier, each layer's map, whose one writer this is, and
    /// the lender's bank group and its lent records, `[0, records)`: none
    /// without a device (an offline replay) or where the host places, whose
    /// landings write the one store's.
    tier: Option<(Vec<StampedMap>, (usize, usize))>,
    /// Per group, the dedicated even/odd double-buffered DMA staging ring
    /// (`lent..lent + stage_slots`), completely disjoint from prompt scratch
    /// (`0..lent`) and protected from promotions, seeds, and shrinks so
    /// `LinkMover` stages Host-RAM misses even when `lent == 0` or on
    /// non-lender groups. Empty (`0..0`) on unified memory or whole banks.
    stage_rings: Vec<std::ops::Range<usize>>,
    /// Bumped as any expert comes to be held or leaves: what each layer
    /// holds changed since a reader last saw it ([`Self::holdings`]).
    holdings: u64,
}

impl BankRecords {
    /// Groups of `records` records each, every one free, of `layers`,
    /// each (its group, its experts), none held; beside a host tier mapped
    /// through its maps, its lender lending its lent records and each group
    /// reserving its dedicated even/odd staging ring (`tier`).
    pub fn new(
        records: &[usize],
        layers: &[(usize, usize)],
        tier: Option<(Vec<StampedMap>, (usize, usize), Vec<usize>)>,
    ) -> Self {
        let (tier, stage_rings) = match tier {
            Some((maps, (lender, lent), stage_slots)) => {
                assert_eq!(maps.len(), layers.len(), "each layer has its map");
                assert_eq!(stage_slots.len(), records.len(), "each group has its staging ring count");
                assert!(lent <= records[lender], "a group lends {lent} of its {} records", records[lender]);
                let rings = records
                    .iter()
                    .enumerate()
                    .map(|(group, &count)| {
                        let start = if group == lender { lent } else { 0 };
                        let slots = stage_slots[group];
                        assert!(slots % 2 == 0, "group {group} staging ring count {slots} must be even");
                        assert!(
                            start + slots <= count,
                            "group {group} lent ({start}) + staging ({slots}) exceeds its {count} records"
                        );
                        start..start + slots
                    })
                    .collect();
                (Some((maps, (lender, lent))), rings)
            }
            None => (None, vec![0..0; records.len()]),
        };
        let mut groups: Vec<Records> = records.iter().map(|&count| Records::new(count)).collect();
        for (group, range) in stage_rings.iter().enumerate() {
            for record in range.clone() {
                groups[group].set(record, Record::StageIdle);
            }
        }
        Self {
            groups,
            layers: layers.iter().map(|&(group, experts)| (group, vec![None; experts])).collect(),
            placing: HashSet::new(),
            promoting: Vec::new(),
            releasing: Vec::new(),
            filling: Vec::new(),
            staging: Vec::new(),
            changed: false,
            tier,
            stage_rings,
            holdings: 0,
        }
    }

    /// Each layer's map, beside a host tier.
    fn maps(&self) -> Option<&[StampedMap]> {
        self.tier.as_ref().map(|(maps, _)| maps.as_slice())
    }

    /// The lender's bank group and its lent records, beside a host tier.
    pub(super) fn lent(&self) -> Option<(usize, usize)> {
        self.tier.as_ref().map(|&(_, lent)| lent)
    }

    /// Whether record `record` of group `group` is currently `Record::Free`.
    pub(super) fn is_free(&self, group: usize, record: usize) -> bool {
        matches!(self.groups[group].contents.get(record), Some(Record::Free))
    }

    /// Whether group `group` currently has any record leased to a prompt group.
    pub(super) fn has_lent(&self, group: usize) -> bool {
        self.groups[group].lent > 0
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

    /// The records of each group the bank holds storage for: every one but
    /// the holes.
    pub fn counts(&self) -> Vec<usize> {
        self.groups.iter().map(|records| records.present).collect()
    }

    /// The first record of group `group` a seed may take: past the lent
    /// ones where it is the lender (`ExpertBankPlan::seeded_from`), matching
    /// the GPU `routed_seed_commit` base offset (`lent`). Staging-ring slots
    /// are excluded from `free()`, `evictable()`, `seed()`, `promote()`, and
    /// `shrink()` by construction via `Record::StageIdle` / `Record::Staging`.
    pub fn seeded_from(&self, group: usize) -> usize {
        self.lent().filter(|&(lender, _)| lender == group).map_or(0, |(_, records)| records)
    }

    /// Whether record `record` of group `group` is a lent one.
    pub fn lends(&self, group: usize, record: usize) -> bool {
        self.lent().is_some_and(|(lender, records)| lender == group && record < records)
    }

    /// Whether record `record` of group `group` lies in its dedicated DMA
    /// staging ring (`lent..lent + stage_slots`).
    pub fn is_stage_slot(&self, group: usize, record: usize) -> bool {
        self.stage_rings.get(group).is_some_and(|range| range.contains(&record))
    }

    /// The number of slots in each parity half (`even` or `odd`) of group
    /// `group`'s dedicated DMA staging ring (`0` when disabled).
    pub fn stage_ring_half(&self, group: usize) -> usize {
        self.stage_rings.get(group).map_or(0, |range| range.len() / 2)
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

    /// Map `expert` to `record`, which holds it whole, from `from` on, or
    /// from its entry's FROM where that covers an earlier step
    /// (`StampedMap::map_covering`).
    fn map(&self, (layer, expert): Staged, record: usize, from: MapStep) {
        if let Some(maps) = self.maps() {
            maps[layer].map_covering(expert, record, from);
        }
    }

    fn unmap(&self, (layer, expert): Staged, until: UnmapStep) {
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

    /// Whether a promotion or a release may take record `record` of group
    /// `group`: any but a lent scratch record or a dedicated staging-ring slot.
    pub fn usable(&self, group: usize, record: usize) -> bool {
        !self.lends(group, record) && !self.is_stage_slot(group, record)
    }

    /// Group `group`'s records a promotion may take now or once released:
    /// those free or releasing, but a lent one.
    pub fn covering(&self, group: usize) -> usize {
        let free = self.free(group).filter(|&record| self.usable(group, record)).count();
        let releasing = self.releasing.iter().filter(|&&(of, record)| of == group && self.usable(group, record)).count();
        free + releasing
    }

    /// Decode's capacity in group `group`: the records the bank holds
    /// storage for, less the lent ones (the scratch's) and the dedicated
    /// DMA staging ring.
    pub fn capacity(&self, group: usize) -> usize {
        self.groups[group].present - self.seeded_from(group) - self.stage_rings[group].len()
    }

    /// `staged` held in `record` of its group from now.
    fn hold(&mut self, staged: Staged, record: usize) {
        let held = &mut self.layers[staged.0].1[staged.1];
        assert!(held.is_none(), "expert {staged:?} is held already, in record {held:?}");
        *held = Some(record);
        self.holdings += 1;
        self.set(self.layers[staged.0].0, record, Record::Held(staged));
    }

    /// The expert held record `record` of `group` holds leaves the bank
    /// by `exit`, its one exit's (`Directory::exit`), unmapped from
    /// `until`. Its bytes stay in the record only where its exit writes it
    /// back ([`Self::release_from`]). The expert.
    fn unhold(&mut self, group: usize, record: usize, until: UnmapStep, exit: &Exit) -> Staged {
        let Record::Held(staged) = self.groups[group].contents[record] else {
            panic!("record {record} of group {group} is left from {:?}", self.groups[group].contents[record]);
        };
        assert_eq!(exit.staged(), staged, "record {record} of group {group} is left by another expert's exit");
        self.layers[staged.0].1[staged.1] = None;
        self.holdings += 1;
        self.unmap(staged, until);
        staged
    }

    /// Record `record` of group `group` is taken now for another use: the
    /// expert it holds, if any, leaves the bank by `exit`, which writes
    /// nothing back (the record's bytes are overwritten or its storage
    /// leaves), unmapped from `until`. The expert, none where the record
    /// was free.
    fn vacate(&mut self, group: usize, record: usize, until: UnmapStep, exit: Option<Exit>) -> Option<Staged> {
        match (self.groups[group].contents[record], exit) {
            (Record::Free, None) => None,
            (Record::Held(_), Some(exit)) => {
                assert_eq!(exit.back(), None, "record {record} of group {group} is taken while its expert's exit writes it back");
                Some(self.unhold(group, record, until, &exit))
            }
            (content, exit) => panic!("record {record} of group {group} is taken from {content:?} by exit {exit:?}"),
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
                self.map((layer, expert), record, MapStep::EVERY);
            }
        }
    }

    /// Promotion `promotion` takes its free record, which no step reads,
    /// until its copy has completed. Never a lent record: a lease would
    /// find the copy writing it, and the copy may wait in the driver behind
    /// a step parked on the scratch the lease holds back (`experts::copies`).
    pub fn promote(&mut self, promotion: &Promotion) {
        let Promotion { group, record, layer, expert } = *promotion;
        let record = record.index();
        assert!(self.usable(group, record), "a promotion into lent record {record}");
        assert_eq!(self.groups[group].contents[record], Record::Free, "a promotion takes a free record");
        assert!(self.placing.insert((layer, expert)), "expert {expert} of layer {layer} is placed twice");
        self.set(group, record, Record::Promoting((layer, expert), Moving::Queued));
        self.promoting.push((group, record));
    }

    /// The promotions queued, each (group, record, expert).
    pub fn queued(&self) -> Vec<(usize, BankRecord, Staged)> {
        self.promoting
            .iter()
            .filter_map(|&(group, record)| match self.groups[group].contents[record] {
                Record::Promoting(staged, Moving::Queued) => Some((group, bank_record(record), staged)),
                _ => None,
            })
            .collect()
    }

    /// Release record `record` of group `group`, its expert leaving the
    /// bank by `exit` (`Directory::exit`), unmapped from `until` (the first
    /// step not yet begun, under the steps' lock, for the reserve or a
    /// shrink): written back into the tier record its exit took, if it
    /// took one, it is free once every step its entry was mapped for has
    /// passed its layer or ended and its write-back has ended
    /// ([`Self::drain`]). Never a lent record, whose lease leaves what it
    /// holds instead ([`Self::lease`]). Its expert.
    pub fn release(&mut self, group: usize, BankRecord(record): BankRecord, until: UnmapStep, exit: Exit) -> Staged {
        let record = record as usize;
        assert!(self.usable(group, record), "a release of lent record {record}");
        self.release_from(group, record, until, exit)
    }

    /// [`Self::release`] of any record, a lent one's among them.
    fn release_from(&mut self, group: usize, record: usize, until: UnmapStep, exit: Exit) -> Staged {
        let staged @ (layer, expert) = self.unhold(group, record, until, &exit);
        // Its entry's FROM, which the unmapping leaves: no step before it
        // resolves the entry to the record. None mapped without a device.
        let from = self.maps().map_or(0, |maps| maps[layer].view().from(expert));
        let back = exit.back().map(|tier| Back { staged, tier, batch: None });
        self.set(group, record, Record::Releasing { layer, from, until: until.step(), back });
        self.releasing.push((group, record));
        staged
    }

    /// Every release the device is past at `progress` is free: each step
    /// its entry was mapped for has passed its layer or ended
    /// (`Steps::served`), so no step reads it. Whether any was.
    pub fn drain(&mut self, progress: Point, steps: &super::watcher::Steps) -> bool {
        let (drained, releasing) = std::mem::take(&mut self.releasing).into_iter().partition::<Vec<_>, _>(|&(group, record)| {
            let Record::Releasing { layer, from, until, back } = self.groups[group].contents[record] else {
                panic!("record {record} of group {group} is listed releasing");
            };
            back.is_none() && steps.served(from, until, layer, progress)
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

    /// The releases' write-backs queued, each (group, record, expert, the
    /// tier record it writes).
    pub fn backs_queued(&self) -> Vec<(usize, BankRecord, Staged, HostRecord)> {
        self.releasing
            .iter()
            .filter_map(|&(group, record)| match self.groups[group].contents[record] {
                Record::Releasing { back: Some(Back { staged, tier, batch: None }), .. } => Some((group, bank_record(record), staged, tier)),
                _ => None,
            })
            .collect()
    }

    /// The releases' write-backs copying, each (group, record, batch).
    pub fn backing(&self) -> Vec<(usize, BankRecord, u64)> {
        self.releasing
            .iter()
            .filter_map(|&(group, record)| match self.groups[group].contents[record] {
                Record::Releasing { back: Some(Back { batch: Some(batch), .. }), .. } => Some((group, bank_record(record), batch)),
                _ => None,
            })
            .collect()
    }

    /// Queued write-back of releasing `record` of group `group` copies by
    /// batch `batch`.
    pub fn back_copy(&mut self, group: usize, BankRecord(record): BankRecord, batch: u64) {
        let record = record as usize;
        let Record::Releasing { layer, from, until, back: Some(back @ Back { batch: None, .. }) } = self.groups[group].contents[record] else {
            panic!("record {record} of group {group} copies no queued write-back");
        };
        self.set(group, record, Record::Releasing { layer, from, until, back: Some(Back { batch: Some(batch), ..back }) });
    }

    /// The write-back of releasing `record` of group `group` ended, written
    /// or failed (a queued one whose copy never began among them): the
    /// record drains as any release from now. The write-back.
    pub fn backed(&mut self, group: usize, BankRecord(record): BankRecord) -> Back {
        let record = record as usize;
        let Record::Releasing { layer, from, until, back: Some(back) } = self.groups[group].contents[record] else {
            panic!("record {record} of group {group} ends no write-back");
        };
        self.set(group, record, Record::Releasing { layer, from, until, back: None });
        back
    }

    /// Whether any promotion is in flight, any release draining, or any
    /// opportunistic staging slot active (in-flight DMA or held for an
    /// unretired step).
    pub fn moving(&self) -> bool {
        !self.promoting.is_empty() || !self.releasing.is_empty() || !self.staging.is_empty()
    }

    /// The promotions in flight, releases draining, and active staging slots,
    /// for the exchange's watchdog.
    pub fn describe(&self) -> String {
        let records = |taken: &[(usize, usize)]| taken.iter().map(|&(group, record)| self.groups[group].contents[record]).collect::<Vec<_>>();
        format!("promotions {:?}, releases {:?}, staging {:?}", records(&self.promoting), records(&self.releasing), records(&self.staging))
    }

    /// Find a free slot in `group`'s dedicated parity half for `target_step`
    /// (`range.start..range.start + half` for even steps,
    /// `range.start + half..range.end` for odd steps, where `half = range.len() / 2`),
    /// completely independent of `lent == 0` and prompt scratch leases.
    /// Enforces zero aliasing by construction: returns `None` if any slot in
    /// that parity half still holds `Record::Staging` for a different step
    /// `!= target_step` (until `drain_staging(frontier)` frees it once
    /// `Steps::frontier() > prev_target`).
    pub fn free_stage_slot(&self, group: usize, target_step: Step) -> Option<usize> {
        let range = self.stage_rings.get(group)?;
        let half = range.len() / 2;
        if half == 0 {
            return None;
        }
        let parity = (target_step & 1) as usize;
        let base = range.start + parity * half;
        let slice = &self.groups[group].contents[base..base + half];
        if slice.iter().any(|content| matches!(content, Record::Staging { target_step: prev, .. } if *prev != target_step)) {
            return None;
        }
        slice.iter().position(|&content| content == Record::StageIdle).map(|offset| base + offset)
    }

    /// Transition `StageIdle -> Staging` for an opportunistic DMA staging copy
    /// of `staged` into `slot` of `group` for `target_step` in parity half
    /// `parity`.
    #[allow(clippy::too_many_arguments)]
    pub fn stage(
        &mut self,
        group: usize,
        slot: usize,
        staged: Staged,
        target_step: Step,
        tier: HostRecord,
        batch: u64,
        parity: u8,
    ) {
        assert!(self.is_stage_slot(group, slot), "staging slot {slot} of group {group} must be in its dedicated staging ring");
        assert_eq!(self.groups[group].contents[slot], Record::StageIdle, "staging takes an idle staging-ring slot");
        self.set(
            group,
            slot,
            Record::Staging { staged, target_step, tier: Some(tier), batch: Some(batch), parity },
        );
        self.staging.push((group, slot));
    }

    /// Mark an in-flight `Record::Staging` DMA copy as landed: clears `tier`
    /// and `batch` so the host-tier hold is released while the VRAM slot
    /// remains `Record::Staging` (read-only for `target_step`) until
    /// `drain_staging(frontier)` observes `frontier > target_step`.
    pub fn staged_landed(&mut self, group: usize, slot: usize) -> (Staged, HostRecord) {
        let Record::Staging { staged, target_step, tier: Some(tier), batch: Some(_), parity } = self.groups[group].contents[slot] else {
            panic!("record {slot} of group {group} has no in-flight staging copy to land");
        };
        self.set(group, slot, Record::Staging { staged, target_step, tier: None, batch: None, parity });
        (staged, tier)
    }

    /// Cancel or fail an in-flight `Record::Staging` copy, returning the slot
    /// immediately to `Record::StageIdle`.
    pub fn unstage(&mut self, group: usize, slot: usize) -> (Staged, Option<HostRecord>) {
        let Record::Staging { staged, tier, .. } = self.groups[group].contents[slot] else {
            panic!("record {slot} of group {group} is not staging");
        };
        self.staging.retain(|&taken| taken != (group, slot));
        self.set(group, slot, Record::StageIdle);
        (staged, tier)
    }

    /// Drain landed `Record::Staging` slots whose `target_step` has passed
    /// `frontier` (`frontier > target_step`), returning them to `Record::StageIdle`.
    /// Never frees a slot whose DMA batch is still in flight (`batch.is_some()`).
    pub fn drain_staging(&mut self, frontier: Step) -> bool {
        let (drained, kept) = std::mem::take(&mut self.staging).into_iter().partition::<Vec<_>, _>(|&(group, record)| {
            let Record::Staging { target_step, batch, .. } = self.groups[group].contents[record] else {
                panic!("record {record} of group {group} is listed staging");
            };
            batch.is_none() && frontier > target_step
        });
        self.staging = kept;
        for &(group, record) in &drained {
            self.set(group, record, Record::StageIdle);
        }
        !drained.is_empty()
    }

    /// In-flight staging DMA batches: each `(group, slot, batch)`.
    pub fn staging_copying(&self) -> Vec<(usize, usize, u64)> {
        self.staging
            .iter()
            .filter_map(|&(group, record)| match self.groups[group].contents[record] {
                Record::Staging { batch: Some(batch), .. } => Some((group, record, batch)),
                _ => None,
            })
            .collect()
    }

    /// Whether any opportunistic staging DMA batch is currently in flight.
    pub fn staging_in_flight(&self) -> bool {
        self.staging
            .iter()
            .any(|&(group, record)| matches!(self.groups[group].contents[record], Record::Staging { batch: Some(_), .. }))
    }

    /// Active staging entry for `staged` in `group`, if any:
    /// `(slot, target_step, batch)`.
    pub fn staged_entry(&self, group: usize, staged: Staged) -> Option<(usize, Step, Option<u64>)> {
        self.staging.iter().find_map(|&(g, record)| {
            if g != group {
                return None;
            }
            match self.groups[g].contents[record] {
                Record::Staging { staged: s, target_step, batch, .. } if s == staged => Some((record, target_step, batch)),
                _ => None,
            }
        })
    }

    /// Queued promotion `record` of group `group` copies from tier record
    /// `tier`, which it holds, by batch `batch`.
    pub fn copy(&mut self, group: usize, BankRecord(record): BankRecord, tier: HostRecord, batch: u64) {
        let record = record as usize;
        let Record::Promoting(staged, Moving::Queued) = self.groups[group].contents[record] else {
            panic!("record {record} of group {group} copies no queued promotion");
        };
        self.set(group, record, Record::Promoting(staged, Moving::Copying { tier, batch }));
    }

    /// Take promotion `record` of group `group` out of what is promoting:
    /// its expert, and the tier record its copy held.
    fn end_promotion(&mut self, group: usize, record: usize) -> (Staged, Option<HostRecord>) {
        let Record::Promoting(staged, moving) = self.groups[group].contents[record] else {
            panic!("record {record} of group {group} is no promotion's");
        };
        self.promoting.retain(|&taken| taken != (group, record));
        self.placing.remove(&staged);
        match moving {
            Moving::Queued => (staged, None),
            Moving::Copying { tier, .. } => (staged, Some(tier)),
        }
    }

    /// The copies of promotions `completed`, each (group, record), have
    /// completed: each holds its expert, mapped from `from(layer)`, the one
    /// map rule's step for its layer (the caller's, read under the lock
    /// every step begins under): the record was free and holds the expert
    /// whole, so any step from it on that resolves the layer after the
    /// stores reads it there. Each one's expert, its record, and the tier
    /// record its copy held.
    pub fn promoted(&mut self, completed: Vec<(usize, BankRecord)>, from: impl Fn(usize) -> MapFrom) -> Vec<(Staged, BankRecord, HostRecord)> {
        let promoted = completed
            .into_iter()
            .map(|(group, BankRecord(record))| {
                let record = record as usize;
                let (staged, tier) = self.end_promotion(group, record);
                let tier = tier.unwrap_or_else(|| panic!("record {record} of group {group} completed a promotion it never copied"));
                self.hold(staged, record);
                // The caller (`Directory::promoted`) feeds the tier-0 move
                // (`Times::moved(group, Tier::COMPUTE)`) outside the steps' lock.
                (staged, record, tier)
            })
            .collect::<Vec<_>>();
        for &(staged, record, _) in &promoted {
            self.map(staged, record, from(staged.0).into());
        }
        promoted.into_iter().map(|(staged, record, tier)| (staged, bank_record(record), tier)).collect()
    }

    /// Promotion `record` of group `group` ended without its expert (its
    /// tier record went before its copy began, or its copy failed): the
    /// record is free, as nothing maps it. Its expert, and the tier record
    /// a copy held.
    pub fn unpromoted(&mut self, group: usize, BankRecord(record): BankRecord) -> (Staged, Option<HostRecord>) {
        let record = record as usize;
        let ended = self.end_promotion(group, record);
        self.set(group, record, Record::Free);
        ended
    }

    /// Scratch group `scratch_group`'s seed of `staged` at `point`, a point of it,
    /// takes `record`, until its step ends ([`Self::retire_fills`]), its
    /// victim unmapped from the group's first step: the group's steps read
    /// none of its records but their hits, which no seed takes. The seed
    /// copies the expert's scratch slot, which a round filled only where
    /// none of the group's steps resolved it held: seeding one the map
    /// holds for any of them (`point.group` to the seeding step) would bank
    /// an unfilled slot. Its victim, if any, leaves the bank by `exit`,
    /// which writes nothing back: the seed overwrites its bytes. The
    /// victim.
    pub fn fill(&mut self, point: Point, scratch_group: ScratchGroup, staged: Staged, BankRecord(record): BankRecord, exit: Option<Exit>) -> Option<Staged> {
        assert_eq!(point.group, scratch_group.first(), "a seed at {point:?} of the scratch group from step {}", scratch_group.first());
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
        let victim = self.vacate(group, record, scratch_group.into(), exit);
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
        ending: impl FnMut(Point) -> super::directory::Ending,
        steps: &super::watcher::Steps,
    ) -> Vec<(usize, Staged, bool)> {
        let mut filling = std::mem::take(&mut self.filling);
        let retired = super::directory::Settling::new(progress, ending).retire(&mut filling, |&taken| self.fill_of(taken).1, |&(group, record)| commit(group, bank_record(record)));
        self.filling = filling;
        let retiring = retired.into_iter().map(|((group, record), committed)| (group, record, self.fill_of((group, record)).0, committed)).collect::<Vec<_>>();
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
                    self.map(staged, record, at.into());
                }
            });
        }
        retiring.into_iter().map(|(group, _, staged, committed)| (group, staged, committed)).collect()
    }

    /// The lent records holding an expert (a home of a bank imported
    /// whole) a lease has yet to take, each (group, record, expert): those
    /// whose exits ([`Self::lease`]) are decided.
    pub fn lent_held(&self) -> Vec<(usize, usize, Staged)> {
        let Some((group, records)) = self.lent() else {
            return Vec::new();
        };
        (0..records)
            .filter_map(|record| match self.groups[group].contents[record] {
                Record::Held(staged) => Some((group, record, staged)),
                _ => None,
            })
            .collect()
    }

    /// Grant scratch group `scratch_group` its lease of the lent records as
    /// the conveyor serves it, before it takes any hit: each a free one at
    /// once; each holding an expert (a home of a bank imported whole) as
    /// that expert leaves the bank by its exit in `exits`
    /// ([`Self::lent_held`]), unmapped from the group's first step on (the
    /// steps before it, which may read it, run before the group writes
    /// it): at once where its exit writes nothing back, else released
    /// ([`Self::release_from`]) and granted by a later call once its
    /// write-back has ended and it drained. Nothing else is ever in one: no
    /// promotion, release or seed takes a lent record. The experts that
    /// left the bank.
    pub fn lease(&mut self, scratch_group: ScratchGroup, exits: Vec<Exit>) -> Vec<Staged> {
        let mut left = Vec::new();
        let mut exits = exits.into_iter().map(|exit| (exit.staged(), exit)).collect::<std::collections::HashMap<_, _>>();
        let Some((group, records)) = self.lent() else {
            assert!(exits.is_empty(), "exits from lent records where none is lent");
            return left;
        };
        for record in 0..records {
            match self.groups[group].contents[record] {
                Record::Free => {}
                Record::Held(staged) => {
                    let exit = exits.remove(&staged).unwrap_or_else(|| panic!("lent record {record} holds {staged:?}, which has no exit"));
                    if exit.back().is_some() {
                        left.push(self.release_from(group, record, scratch_group.into(), exit));
                        continue;
                    }
                    left.extend(self.vacate(group, record, scratch_group.into(), Some(exit)));
                }
                // Granted already, its write-back still reads it, or an
                // opportunistic staging slot is draining past the frontier.
                Record::Lent { .. } | Record::Releasing { .. } | Record::Staging { .. } => continue,
                other => panic!("lent record {record} leased from {other:?}"),
            }
            self.set(group, record, Record::Lent { from: scratch_group.first() });
        }
        assert!(exits.is_empty(), "exits of experts no lent record holds: {exits:?}");
        left
    }

    /// Whether the prompt group's lease holds every lent record: true where
    /// nothing is lent.
    pub fn leased(&self) -> bool {
        self.lent().is_none_or(|(group, records)| self.groups[group].lent == records)
    }

    /// The prompt group the records were leased to is past: every one is
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

    /// The records of group `group` a shrink may evict now, beside running
    /// steps or not: past the lent ones (the floor's, which the conveyor
    /// borrows), and free, releasing with no write-back reading it, or
    /// held, so none a promotion, a write-back, a seed or a prompt group
    /// writes or reads in place. Each with what it holds.
    pub fn evictable(&self, group: usize) -> impl Iterator<Item = (usize, Record)> + '_ {
        let lent = self.seeded_from(group);
        self.groups[group].contents.iter().copied().enumerate().skip(lent).filter(|(_, content)| matches!(content, Record::Free | Record::Releasing { back: None, .. } | Record::Held(_)))
    }

    /// Evict `records` of group `group` ([`Self::evictable`] each): holes
    /// from now, nothing moved. A held one's expert leaves, unmapped from
    /// step `until(expert)` on, so the steps before it still read it where
    /// its bytes are until the bank releases its storage behind them, by
    /// its exit (`exits`, each record's, none for one holding no expert),
    /// which writes nothing back; a releasing one's expert left already.
    /// The experts that left.
    pub fn evict(&mut self, group: usize, records: &[usize], until: impl Fn(Staged) -> UnmapStep, exits: Vec<Option<Exit>>) -> Vec<Staged> {
        assert_eq!(records.len(), exits.len(), "each evicted record has its exit");
        let mut evicted = Vec::new();
        for (&record, exit) in records.iter().zip(exits) {
            assert!(!self.lends(group, record), "a shrink evicts lent record {record} of group {group}");
            assert!(!self.is_stage_slot(group, record), "a shrink evicts staging ring record {record} of group {group}");
            match self.groups[group].contents[record] {
                Record::Free => assert!(exit.is_none(), "a free record has no exit"),
                Record::Releasing { back: None, .. } => {
                    assert!(exit.is_none(), "a releasing record's expert left already");
                    self.releasing.retain(|&taken| taken != (group, record));
                }
                Record::Held(staged) => evicted.extend(self.vacate(group, record, until(staged), exit)),
                other => panic!("a shrink evicts record {record} of group {group}, holding {other:?}"),
            }
            self.set(group, record, Record::Absent);
        }
        evicted
    }

    /// Back group `group`'s `records`, holes or past every record: free
    /// from now.
    pub fn grow(&mut self, group: usize, records: impl IntoIterator<Item = usize>) {
        let before = self.groups[group].present;
        self.groups[group].grow(records);
        self.changed |= self.groups[group].present > before;
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
        let tensor = seismic::Tensor::shared(&seismic::Charges::tally(&cpu), seismic::Element::i32(), &[EXPERTS as u64, MAP_WORDS]).unwrap();
        let map = StampedMap::new(Words::new(tensor, words, 0).unwrap());
        let view = map.view().clone();
        (BankRecords::new(&[4], &[(0, EXPERTS)], Some((vec![map], (0, 2), vec![0]))), view)
    }

    fn bank_with_staging(records: usize, lent: usize, stage_slots: usize) -> (BankRecords, MapView) {
        let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let words = EXPERTS as u64 * MAP_WORDS;
        let tensor = seismic::Tensor::shared(&seismic::Charges::tally(&cpu), seismic::Element::i32(), &[EXPERTS as u64, MAP_WORDS]).unwrap();
        let map = StampedMap::new(Words::new(tensor, words, 0).unwrap());
        let view = map.view().clone();
        (BankRecords::new(&[records], &[(0, EXPERTS)], Some((vec![map], (0, lent), vec![stage_slots]))), view)
    }

    fn promotion(record: u32, expert: usize) -> Promotion {
        Promotion { group: 0, record: BankRecord(record), layer: 0, expert }
    }

    /// Hold expert `expert` in free record `record`, mapped from `from`:
    /// its promotion's copy completed.
    fn promoted(bank: &mut BankRecords, record: u32, expert: usize, from: Step) {
        bank.promote(&promotion(record, expert));
        bank.copy(0, BankRecord(record), HostRecord(0), 0);
        bank.promoted(vec![(0, BankRecord(record))], |_| MapFrom::at(from));
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
        assert_eq!(bank.release(0, BankRecord(2), UnmapStep::at(9), Exit::to_file((0, 3))), (0, 3));
        assert_eq!((bank.held(0, 3), view.resolve(3, 8), view.resolve(3, 9)), (None, Some(2), None), "the released expert is read until 9");
        assert!(!bank.drain(Point::whole(8, 0, 8), &steps) && !bank.free(0).any(|record| record == 2), "steps below 9 still read it");
        assert!(bank.drain(Point::whole(9, 0, 9), &steps) && bank.free(0).any(|record| record == 2));
        let point = Point::whole(12, 0, 13);
        promoted(&mut bank, 3, 6, 10);
        assert_eq!(bank.fill(point, ScratchGroup::at(12, 2), (0, 7), BankRecord(3), Some(Exit::to_file((0, 6)))), Some((0, 6)));
        assert_eq!((view.resolve(6, 11), view.resolve(6, 12)), (Some(3), None), "a seed's victim leaves from its group's first step");
        let retired = bank.retire_fills(Point::whole(13, usize::MAX, Step::MAX), |_, _| super::super::watcher::low_word(13), |_| super::super::directory::Ending::Ended, &steps);
        assert_eq!(retired, [(0, (0, 7), true)]);
        assert_eq!((bank.held(0, 7), view.resolve(7, 14)), (Some(3), Some(3)));
    }

    /// A release written back into the host tier keeps its bytes until
    /// the write-back ends: it never drains, nor may a shrink take it,
    /// while its copy is queued or in flight, though every step it was
    /// mapped for is past; once the write-back ends it drains as any.
    #[test]
    fn a_release_written_back_holds_its_bytes_until_the_copy_ends() {
        let (mut bank, _) = bank();
        let steps = super::super::watcher::Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new());
        steps.begin_through(13);
        promoted(&mut bank, 2, 3, 4);
        bank.release(0, BankRecord(2), UnmapStep::at(9), Exit::written_back((0, 3), HostRecord(5)));
        assert_eq!(bank.backs_queued(), [(0, BankRecord(2), (0, 3), HostRecord(5))]);
        assert!(!bank.evictable(0).any(|(record, _)| record == 2), "a shrink never takes a record a write-back reads");
        bank.back_copy(0, BankRecord(2), 7);
        assert_eq!((bank.backs_queued(), bank.backing()), (vec![], vec![(0, BankRecord(2), 7)]));
        assert!(!bank.drain(Point::whole(12, 0, 12), &steps), "past every step it was mapped for, its copy still reads it");
        assert_eq!(bank.backed(0, BankRecord(2)), Back { staged: (0, 3), tier: HostRecord(5), batch: Some(7) });
        assert!(bank.drain(Point::whole(12, 0, 12), &steps) && bank.free(0).any(|record| record == 2));
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
        bank.release(0, BankRecord(2), UnmapStep::at(5), Exit::to_file((0, 5)));
        bank.fill(Point::whole(3, 0, 4), ScratchGroup::at(3, 2), (0, 5), BankRecord(3), None);
    }

    /// A regression: a hit a prompt group's steps planned at their request
    /// was released mid-group and its expert promoted again, whose mapping
    /// from the first step not yet begun overwrote the one entry, so the
    /// group's later passes resolved it absent and read a scratch slot no
    /// round filled. A remap keeps every step the entry resolved.
    #[test]
    fn a_remap_keeps_the_steps_its_entry_resolved() {
        let (mut bank, view) = bank();
        promoted(&mut bank, 2, 5, 1);
        // Group 3's steps 3 and 4 have begun: the release unmaps 5 from 5.
        bank.release(0, BankRecord(2), UnmapStep::at(5), Exit::to_file((0, 5)));
        promoted(&mut bank, 3, 5, 6);
        assert_eq!((view.resolve(5, 0), view.resolve(5, 4), view.resolve(5, 6)), (None, Some(3), Some(3)));
    }

    /// No promotion and no release takes a lent record: those are the
    /// scratch's, so no copy into one runs as a lease takes them.
    #[test]
    #[should_panic(expected = "a promotion into lent record 1")]
    fn a_promotion_into_a_lent_record_panics() {
        let (mut bank, _) = bank();
        bank.promote(&promotion(1, 6));
    }

    #[test]
    #[should_panic(expected = "a release of lent record 1")]
    fn a_release_of_a_lent_record_panics() {
        let (mut bank, _) = bank();
        bank.hold((0, 5), 1);
        bank.release(0, BankRecord(1), UnmapStep::at(7), Exit::to_file((0, 5)));
    }

    /// The lease takes every lent record at once, a home one held (a bank
    /// imported whole) unmapped for every step of the group, so no step of
    /// it resolves a hit in a lent record; a second grant takes nothing,
    /// and the return frees them. Neither covers a want, nor counts in
    /// decode's capacity.
    #[test]
    fn a_lease_is_whole_at_once_and_leaves_no_hit_in_a_lent_record() {
        let (mut bank, view) = bank();
        bank.hold((0, 4), 1);
        bank.map((0, 4), 1, MapStep::EVERY);
        promoted(&mut bank, 2, 5, 1);
        assert_eq!((bank.covering(0), bank.capacity(0)), (1, 2), "the lent records cover no want, and hold no decode expert");
        assert_eq!(bank.lent_held(), [(0, 1, (0, 4))]);
        assert_eq!(bank.lease(ScratchGroup::at(3, 2), vec![Exit::to_file((0, 4))]), [(0, 4)]);
        assert!(bank.leased(), "it took every lent record at once");
        assert!(bank.lease(ScratchGroup::at(3, 2), Vec::new()).is_empty(), "and a second grant takes nothing");
        assert_eq!((view.resolve(4, 2), view.resolve(4, 3)), (Some(1), None));
        assert_eq!(bank.resolve(0, 4, 3), None);
        assert_eq!(bank.held(0, 5), Some(2), "a record past the lent ones keeps its expert");
        bank.unlend();
        assert!(!bank.leased());
        assert_eq!(bank.free(0).collect::<Vec<_>>(), [0, 1, 3]);
        assert_eq!(bank.covering(0), 1);
    }

    /// A lent record whose expert's exit writes it back is released into
    /// its write-back, not lent: the lease waits for it, granting the rest,
    /// and a later grant takes it once the copy ended and it drained.
    #[test]
    fn a_lease_waits_for_a_lent_records_write_back() {
        let (mut bank, _) = bank();
        let steps = super::super::watcher::Steps::new(crate::planning::FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new());
        steps.begin_through(13);
        bank.hold((0, 4), 1);
        bank.map((0, 4), 1, MapStep::EVERY);
        assert_eq!(bank.lease(ScratchGroup::at(3, 2), vec![Exit::written_back((0, 4), HostRecord(6))]), [(0, 4)]);
        assert!(!bank.leased(), "its write-back holds the record");
        assert_eq!(bank.covering(0), 2, "a lent release covers no want");
        assert!(bank.lent_held().is_empty() && bank.lease(ScratchGroup::at(3, 2), Vec::new()).is_empty());
        assert!(!bank.leased());
        bank.back_copy(0, BankRecord(1), 7);
        bank.backed(0, BankRecord(1));
        assert!(bank.drain(Point::whole(3, 0, 3), &steps));
        assert!(bank.lease(ScratchGroup::at(3, 2), Vec::new()).is_empty());
        assert!(bank.leased(), "granted once the copy ended and it drained");
    }

    /// Nothing seeds or promotes into a lent record: the transition into
    /// one is refused.
    #[test]
    #[should_panic(expected = "is taken from Lent")]
    fn a_seed_into_a_lent_record_panics() {
        let (mut bank, _) = bank();
        bank.lease(ScratchGroup::at(3, 2), Vec::new());
        bank.fill(Point::whole(3, 0, 3), ScratchGroup::at(3, 2), (0, 2), BankRecord(0), None);
    }

    /// A shrink evicts with holes and moves nothing: an evicted held
    /// expert is unmapped from the step its caller names, so the steps
    /// before it still read it where it was, a free record evicts at once,
    /// the lent ones are never evictable, and every kept expert stays in its
    /// record. A growth backs a hole again, free, and one past every record.
    #[test]
    fn a_shrink_evicts_with_holes_and_a_growth_backs_them_again() {
        let (mut bank, view) = bank();
        promoted(&mut bank, 2, 3, 4);
        promoted(&mut bank, 3, 5, 4);
        assert_eq!(bank.evictable(0).map(|(record, _)| record).collect::<Vec<_>>(), [2, 3], "the lent records are the floor's");
        assert_eq!(bank.evict(0, &[2], |_| UnmapStep::at(9), vec![Some(Exit::to_file((0, 3)))]), [(0, 3)]);
        assert_eq!((bank.held(0, 3), view.resolve(3, 8), view.resolve(3, 9)), (None, Some(2), None), "read by the steps before 9 where it was");
        assert_eq!((bank.held(0, 5), view.resolve(5, 9)), (Some(3), Some(3)), "the kept expert stays in its record");
        assert_eq!(bank.counts(), [3]);
        assert!(!bank.free(0).any(|record| record == 2), "a hole is never free");
        bank.grow(0, [2, 5]);
        assert_eq!(bank.counts(), [5]);
        assert!(bank.free(0).any(|record| record == 2) && bank.free(0).any(|record| record == 5));
        assert_eq!(bank.records(0)[4], Record::Absent, "a growth past every record leaves the holes below it");
        assert!(bank.evict(0, &[2, 5], |_| UnmapStep::at(9), vec![None, None]).is_empty(), "free records leave nothing");
        assert_eq!(bank.counts(), [3]);
    }

    /// A shrink never evicts a lent record: the floor holds them for the
    /// conveyor.
    #[test]
    #[should_panic(expected = "a shrink evicts lent record 1")]
    fn evicting_a_lent_record_panics() {
        let (mut bank, _) = bank();
        bank.evict(0, &[1], |_| UnmapStep::at(9), vec![None]);
    }

    /// A growth of a record the bank holds is a broken invariant.
    #[test]
    #[should_panic(expected = "which the bank holds")]
    fn growing_a_held_record_panics() {
        let (mut bank, _) = bank();
        bank.grow(0, [3]);
    }

    /// `Record::Staging` lives in a dedicated even/odd double-buffered ring in
    /// `BankRecords` completely independent of `lent == 0`, is covered by
    /// `moving()`, protected from shrinks and promotions, and frees slots only
    /// once `drain_staging(frontier)` observes `frontier > target_step` and the
    /// DMA batch has landed.
    #[test]
    fn record_staging_is_covered_by_moving_and_drains_past_frontier() {
        // Dedicated staging ring with `lent == 0`: slots 0 (even) and 1 (odd),
        // permanent bank records 2..4.
        let (mut bank, _) = bank_with_staging(4, 0, 2);
        assert!(!bank.moving());
        assert_eq!(bank.stage_ring_half(0), 1);
        assert_eq!(bank.capacity(0), 2, "staging ring slots are excluded from permanent decode capacity");
        assert_eq!(bank.evictable(0).map(|(r, _)| r).collect::<Vec<_>>(), [2, 3], "staging ring is never evictable");

        // Step 2 (even): takes slot 0 even though lent == 0.
        assert_eq!(bank.free_stage_slot(0, 2), Some(0));
        bank.stage(0, 0, (0, 5), 2, HostRecord(10), 42, 0);
        assert!(bank.moving(), "in-flight staging must mark BankRecords::moving()");
        assert!(bank.staging_in_flight());
        assert_eq!(bank.staging_copying(), vec![(0, 0, 42)]);
        assert_eq!(bank.staged_entry(0, (0, 5)), Some((0, 2, Some(42))));
        // Even half is full; odd half (step 3) takes slot 1 concurrently.
        assert_eq!(bank.free_stage_slot(0, 2), None);
        assert_eq!(bank.free_stage_slot(0, 3), Some(1));
        bank.stage(0, 1, (0, 6), 3, HostRecord(11), 43, 1);

        // Frontier advances past step 2 while DMA is still in flight -> must NOT drain!
        assert!(!bank.drain_staging(3));
        assert!(bank.moving());

        // DMA lands for step 2 -> tier hold released, but slot stays Staging until frontier > 2.
        assert_eq!(bank.staged_landed(0, 0), ((0, 5), HostRecord(10)));
        assert_eq!(bank.staged_entry(0, (0, 5)), Some((0, 2, None)));
        assert!(!bank.drain_staging(2), "frontier == target_step must not drain");
        assert_eq!(bank.free_stage_slot(0, 4), None, "even half blocked while step 2 is unretired");

        // Frontier > 2 drains slot 0 cleanly while step 3 (in-flight) stays active.
        assert!(bank.drain_staging(3));
        assert_eq!(bank.free_stage_slot(0, 4), Some(0));
        assert!(bank.moving(), "step 3 still staging");

        // Cancel/unstage step 3 frees slot 1 immediately -> moving() becomes false.
        assert_eq!(bank.unstage(0, 1), ((0, 6), Some(HostRecord(11))));
        assert!(!bank.moving());
        assert!(!bank.staging_in_flight());

        // Disjoint from prompt scratch `0..lent`: when `lent == 2` and
        // `stage_slots == 2`, staging uses `2..4` and prompt `lease()` takes
        // `0..2` cleanly without colliding with active staging slots.
        let (mut disjoint, _) = bank_with_staging(6, 2, 2);
        assert_eq!(disjoint.free_stage_slot(0, 2), Some(2));
        disjoint.stage(0, 2, (0, 5), 2, HostRecord(10), 99, 0);
        assert!(disjoint.lease(ScratchGroup::at(4, 1), Vec::new()).is_empty());
        assert!(disjoint.leased(), "prompt lease on 0..2 succeeds without colliding with staging ring 2..4");
    }
}
