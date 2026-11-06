//! The conveyor: a prompt group's cold experts
//! of a layer, read as each step's scratch request publishes them (every
//! step's request of a layer comes before any of its rounds there), each
//! once per group, and converted by the step's rounds from its records
//! (`routed_convert`'s source). The conveyor works only for a group:
//! decode never takes a record from it.
//!
//! One rule carries every expert: it is read into a conveyor record, which
//! the host addresses, and a round converts it from that record in place:
//! on unified memory from the device's own memory, on a discrete device
//! over the link (host-shared records), so no host-issued transfer stands
//! between a read and the round that waits for it. Reads never wait for a
//! device round: they run as far ahead as the conveyor has records, and a
//! round waits only for the experts its list names.
//!
//! Records are taken from what is free, in the device's order of the
//! experts that need them (by layer, then by the step that reads each, then
//! by the round that lists it): an expert takes a record only while the
//! free ones exceed what the experts before it still lack, and a need that
//! appears late (a hit a scratch claim demoted, a read taken
//! over) is met first by the latest expert after it that holds a record it
//! has not listed, which yields it and is read again. So the earliest
//! expert the device waits on always progresses, its listed records
//! freeing round by round. An expert is carried at most once: a step that
//! comes to read one the conveyor carries already (a failed step's) lists
//! that record.
//!
//! Records its lender lends it are the store's between groups; a group
//! claims them as its first request is taken, before any hit is
//! (`Scratch::claim_lent`), so no hit lies in one, and they return empty once
//! the group is past: the conveyor holds the stored form, which
//! `routed_convert` repacks, so a claimed record holds no bank-form expert.
//! What an in-place conveyor's records hold whole as they return, the
//! group's seeding having ranked it, is copied into staging
//! (`Scratch::stage_landed`): a decode demand for it then takes it from
//! there without a read.
//!
//! Beside a host tier, a cold expert the tier keeps whole is not read and
//! takes no record: the fill's tier copy reads its tier record, already in
//! the resident form, in place into its scratch slot after round 0, and no
//! round converts it (`Scratch::fix_rounds`). A cold expert read into the
//! conveyor lands into the tier too, from its conveyor record
//! ([`ConveyorLanding`]), which takes no read until that landing ends.
//!
//! Where the host addresses the scratch (unified memory), its slots carry
//! experts too, at no memory: an expert may be read into the base of a
//! scratch slot no step reads or converts into until the expert's round
//! has converted it ([`Self::take`], `Scratch::lendable_slot`), and
//! its round converts it from there (`routed_convert`'s negative `from`).
//! A slot's record is `records + slot`, past the conveyor's own; it frees
//! as a conveyor record does, and holds no bytes kept past the group.

use super::source::{Outcome, Records, Ticket};
use super::watcher::Step;

/// Where an expert is on its way into the conveyor: being read into
/// conveyor record `record`, held whole in it, or listed for `lists`
/// rounds' converts, each of which the device may read until the round
/// after it (a step that failed leaves its list standing, and the next step
/// to route the expert lists it again).
pub(super) enum State {
    Reading { ticket: Ticket, record: u32 },
    Held { record: u32 },
    Listed { record: u32, lists: u32 },
}

/// The group the conveyor serves: its name and last step, whether the
/// device is past it (the conveyor then clears once nothing of the group
/// runs, `Scratch::clear_past`, and takes no later group's request before),
/// and whether it holds every lent host record claimed: claims return
/// only as it clears, so once it does it holds them until then, and
/// nothing is left to claim (`Scratch::claim_lent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Serving {
    pub group: Step,
    pub last: Step,
    pub past: bool,
    pub claimed: bool,
}

/// The group lending the conveyor its bottom `records` host records
/// (`planning::Lender`; on unified memory the scratch's too), claimed as a
/// group begins (`Directory::claim_lent`), and where each staging pool's
/// conveyor lies in them (`planning::ConveyorPlan`): pool `p`'s conveyor
/// lies at `offset` bytes of its routed tensor's lent records
/// ([`LentPool`]).
pub(super) struct Lending {
    pub group: usize,
    pub records: usize,
    pub pools: Vec<LentPool>,
}

/// A pool's conveyor in the lent records: each lent record of its routed
/// tensor `record` bytes, the conveyor's records at `offset` bytes of them,
/// where `conveyor` places them.
pub(super) struct LentPool {
    pub record: u64,
    pub offset: u64,
    pub conveyor: seismic::RecordPlacement,
}

impl Lending {
    /// The lent records conveyor record `record` of pool `pool` lies across.
    pub fn records(&self, pool: usize, record: u32) -> std::ops::RangeInclusive<usize> {
        let LentPool { record: bytes, offset, conveyor } = &self.pools[pool];
        let start = offset + conveyor.offset(u64::from(record));
        let lent = |byte: u64| usize::try_from(byte / bytes).expect("a lent record");
        lent(start)..=lent(start + conveyor.record() - 1)
    }
}

/// The reads [`Conveyor::settle`] found ended, the experts whose reads
/// landed whole, (layer, expert, conveyor record), and the experts whose
/// reads failed, with why.
#[derive(Default)]
pub(super) struct Settled {
    pub ended: usize,
    pub landed: Vec<(usize, usize, u32)>,
    pub failed: Vec<(usize, usize, String)>,
}

/// An expert held whole in conveyor record `record` of its layer's form,
/// landing from it into its host tier's record `tier`, read for step `read`
/// (`Conveyor::land`): the conveyor record takes no read until its landing
/// ends, as `ended` records (landed: its bytes are whole).
pub(super) struct ConveyorLanding {
    pub layer: usize,
    pub expert: usize,
    pub record: u32,
    pub tier: u32,
    pub read: Option<Step>,
    pub ended: crate::experts::landing::Ended,
}

/// What carries the conveyor: `records` records of each form, where they
/// lie in the records its lender lends, and each staging pool's records as
/// the host addresses them, which reads land in.
pub(super) struct Carrier {
    pub records: usize,
    /// The scratch slots that may carry experts: the widest layer's where
    /// the host addresses the scratch, else none.
    pub slots: usize,
    pub lending: Lending,
    pub pools: Vec<Records>,
}

/// Whose stored bytes a conveyor record holds, from its read's
/// start until the group is past or another read takes the record (only a
/// read into the record writes it, and its lent records stay claimed until
/// then), whether or not the expert is still carried: expert `expert` of
/// `layer`, whole once `landed`, last routed by the group's step `recency`
/// (its seeding's).
#[derive(Clone, Copy)]
struct Bytes {
    layer: usize,
    expert: usize,
    landed: bool,
    recency: Option<Step>,
}

pub(super) struct Conveyor {
    /// Each streamed layer's form: the records its experts take.
    form_of: Vec<usize>,
    /// Per layer, where each expert carried is.
    carried: Vec<Vec<Option<State>>>,
    /// The experts being read, (layer, expert): what settling looks at.
    reading: Vec<(usize, usize)>,
    /// Per form, each conveyor record's expert (layer, expert).
    occupant: Vec<Vec<Option<(usize, usize)>>>,
    /// Per form, whose bytes each conveyor record holds.
    bytes: Vec<Vec<Option<Bytes>>>,
    /// Per scratch slot, the expert (layer, expert) whose stored bytes it
    /// carries, from its read's start until its round's converts are past.
    slots: Vec<Option<(usize, usize)>>,
    /// The landings into the host tier from conveyor records in flight.
    landings: Vec<ConveyorLanding>,
    /// The group served, from its first scratch request until the device is
    /// past it.
    pub serving: Option<Serving>,
    pub carrier: Carrier,
}

/// The order a carried expert lies in a round's list: whole first, then
/// being read, and not yet read.
pub(super) fn readiness(state: Option<&State>) -> u8 {
    match state {
        Some(State::Held { .. } | State::Listed { .. }) => 0,
        Some(State::Reading { .. }) => 1,
        None => 2,
    }
}

impl Conveyor {
    /// A conveyor of the forms `form_of` names (per streamed layer, of
    /// layers of `experts` experts each), carried by `carrier`; every
    /// record empty.
    pub fn new(form_of: Vec<usize>, experts: &[usize], carrier: Carrier) -> Self {
        let forms = form_of.iter().max().map_or(0, |last| last + 1);
        Self {
            carried: experts.iter().map(|&experts| (0..experts).map(|_| None).collect()).collect(),
            reading: Vec::new(),
            occupant: vec![vec![None; carrier.records]; forms],
            bytes: vec![vec![None; carrier.records]; forms],
            slots: vec![None; carrier.slots],
            landings: Vec::new(),
            form_of,
            serving: None,
            carrier,
        }
    }

    /// The group lending the records that carry it.
    pub fn lending(&self) -> &Lending {
        &self.carrier.lending
    }


    /// The scratch slot record `record` names, if it names one (past the
    /// conveyor's own records).
    pub fn slot_of(&self, record: u32) -> Option<u32> {
        (record as usize >= self.carrier.records).then(|| record - self.carrier.records as u32)
    }

    /// The `from` a round's list names record `record` by
    /// (`routed_convert`): a conveyor record, or -1 - slot.
    pub fn listed_from(&self, record: u32) -> i32 {
        match self.slot_of(record) {
            Some(slot) => -1 - slot as i32,
            None => record as i32,
        }
    }

    /// The scratch slots that may carry an expert.
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// The expert whose stored bytes scratch slot `slot` carries, if any.
    pub fn borrower(&self, slot: usize) -> Option<(usize, usize)> {
        self.slots.get(slot).copied().flatten()
    }

    /// The record that names scratch slot `slot`.
    pub fn slot_record(&self, slot: usize) -> u32 {
        assert!(slot < self.slots.len(), "scratch slot {slot} of {}", self.slots.len());
        (self.carrier.records + slot) as u32
    }

    /// Free record `record` of `layer`'s form for another read, a conveyor
    /// record or a slot; a conveyor record keeps its bytes for staging
    /// once the group is past (a slot keeps none).
    fn unoccupy(&mut self, layer: usize, record: u32) {
        match self.slot_of(record) {
            Some(slot) => self.slots[slot as usize] = None,
            None => self.occupant[self.form_of[layer]][record as usize] = None,
        }
    }

    /// Where expert `expert` of `layer` is, if carried.
    pub fn state(&self, layer: usize, expert: usize) -> Option<&State> {
        self.carried[layer][expert].as_ref()
    }

    /// Whether expert `expert` of `layer` is whole where the device reads
    /// it next: held in its conveyor record, or listed by a round, which
    /// leaves the record whole.
    pub fn whole(&self, layer: usize, expert: usize) -> bool {
        matches!(self.state(layer, expert), Some(State::Held { .. } | State::Listed { .. }))
    }

    fn free(records: &[Option<(usize, usize)>]) -> impl Iterator<Item = u32> + '_ {
        records.iter().enumerate().filter(|(_, occupant)| occupant.is_none()).map(|(record, _)| record as u32)
    }

    /// The conveyor records of `layer`'s form free that `admits`: none
    /// carries an expert and no landing reads it.
    pub fn free_conveyor(&self, layer: usize, admits: impl Fn(u32) -> bool) -> Vec<u32> {
        let form = self.form_of[layer];
        let landing = |record: u32| {
            self.landings.iter().any(|landing| self.form_of[landing.layer] == form && landing.record == record)
        };
        Self::free(&self.occupant[form]).filter(|&record| !landing(record) && admits(record)).collect()
    }

    /// Each staging pool's conveyor records, as the host addresses them.
    pub fn pools(&self) -> &[Records] {
        &self.carrier.pools
    }

    /// `landing` lands an expert from its conveyor record into the host
    /// tier from now: the record takes no read until it ends.
    pub fn land(&mut self, landing: ConveyorLanding) {
        self.landings.push(landing);
    }

    /// Take the conveyor landings that ended: each landed, as a landing
    /// from a record whose bytes are whole always does.
    pub fn landed(&mut self) -> Vec<ConveyorLanding> {
        let (ended, running): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.landings).into_iter().partition(|landing| landing.ended.get().is_some());
        self.landings = running;
        ended
    }

    /// Whether `a` and `b` take records of one form.
    pub fn shares_form(&self, a: usize, b: usize) -> bool {
        self.form_of[a] == self.form_of[b]
    }

    /// Expert `expert` of `layer` takes record `record` for its read: a
    /// conveyor record, or a scratch slot (one `Scratch::lendable_slot`
    /// found free of every step's use until the expert's round converts
    /// it).
    pub fn take(&mut self, layer: usize, expert: usize, record: u32) {
        if let Some(slot) = self.slot_of(record) {
            let borrower = &mut self.slots[slot as usize];
            assert!(borrower.is_none(), "a scratch slot carries two experts");
            *borrower = Some((layer, expert));
            return;
        }
        let form = self.form_of[layer];
        let slot = &mut self.occupant[form][record as usize];
        assert!(slot.is_none(), "a conveyor record carries two experts");
        *slot = Some((layer, expert));
        self.bytes[form][record as usize] = Some(Bytes { layer, expert, landed: false, recency: None });
    }

    /// Give back record `record`, which a layer of `layer`'s form took, its
    /// bytes forgotten.
    pub fn untake(&mut self, layer: usize, record: u32) {
        self.unoccupy(layer, record);
        if self.slot_of(record).is_none() {
            self.bytes[self.form_of[layer]][record as usize] = None;
        }
    }

    /// Expert `expert` of `layer` is being read into conveyor record
    /// `record`, which it took.
    pub fn reading(&mut self, layer: usize, expert: usize, ticket: Ticket, record: u32) {
        let slot = &mut self.carried[layer][expert];
        assert!(slot.is_none(), "an expert carried twice");
        *slot = Some(State::Reading { ticket, record });
        self.reading.push((layer, expert));
    }

    /// The group's seeding at `layer` ranked its routed experts, each with
    /// the step that last routed it there: what ranks the bytes of those
    /// the conveyor holds.
    pub fn rank(&mut self, layer: usize, routed: &[(usize, Step)]) {
        for bytes in self.bytes[self.form_of[layer]].iter_mut().flatten().filter(|bytes| bytes.layer == layer) {
            if let Some(&(_, recency)) = routed.iter().find(|&&(expert, _)| expert == bytes.expert) {
                bytes.recency = Some(recency);
            }
        }
    }

    /// The group is past and no read of it runs ([`Self::clear`]): every
    /// expert whose stored bytes a record holds whole and the
    /// group ranked, (recency, layer, expert, record), forgotten here, as
    /// the records return to the bank.
    pub fn take_landed(&mut self) -> Vec<(Step, usize, usize, u32)> {
        self.bytes
            .iter_mut()
            .flat_map(|records| records.iter_mut().enumerate())
            .filter_map(|(record, bytes)| match bytes.take()? {
                Bytes { layer, expert, landed: true, recency: Some(recency) } => Some((recency, layer, expert, record as u32)),
                _ => None,
            })
            .collect()
    }

    /// How many experts are being read now.
    pub fn reads_in_flight(&self) -> usize {
        self.reading.len()
    }

    /// Settle the reads that ended (`outcome`, none while one runs): one
    /// that landed holds its expert whole; one that failed carries nothing,
    /// its record free. The reads that ended, the experts landed whole, and
    /// the experts whose reads failed, with why.
    pub fn settle(&mut self, mut outcome: impl FnMut(&Ticket) -> Option<Outcome>) -> Settled {
        let mut settled = Settled::default();
        let mut reading = std::mem::take(&mut self.reading);
        reading.retain(|&(layer, expert)| {
            let Some(State::Reading { ticket, record }) = &self.carried[layer][expert] else {
                unreachable!("expert {expert} of layer {layer} is listed as being read");
            };
            let record = *record;
            let Some(ended) = outcome(ticket) else {
                return true;
            };
            settled.ended += 1;
            match ended {
                Outcome::Read => {
                    self.carried[layer][expert] = Some(State::Held { record });
                    let bytes = match self.slot_of(record) {
                        Some(_) => None,
                        None => self.bytes[self.form_of[layer]][record as usize].as_mut(),
                    };
                    if let Some(bytes) = bytes {
                        bytes.landed = true;
                    }
                    settled.landed.push((layer, expert, record));
                }
                Outcome::Failed(reason) => {
                    self.carried[layer][expert] = None;
                    self.untake(layer, record);
                    settled.failed.push((layer, expert, reason));
                }
                Outcome::Panicked(panic) => panic.resume(),
            }
            false
        });
        self.reading = reading;
        settled
    }

    /// List the held expert `expert` of `layer` for a round: its conveyor
    /// record.
    pub fn list(&mut self, layer: usize, expert: usize) -> u32 {
        let (record, lists) = match self.carried[layer][expert] {
            Some(State::Held { record }) => (record, 1),
            Some(State::Listed { record, lists }) => (record, lists + 1),
            _ => panic!("listed expert {expert} of layer {layer}, which the conveyor does not hold"),
        };
        self.carried[layer][expert] = Some(State::Listed { record, lists });
        record
    }

    /// Expert `expert` of `layer`, held whole in a conveyor record no round
    /// lists, yields that record to an expert the device needs before it:
    /// it is forgotten, to be read again.
    pub fn yield_record(&mut self, layer: usize, expert: usize) {
        match self.carried[layer][expert].take() {
            Some(State::Held { record }) => self.untake(layer, record),
            _ => panic!("expert {expert} of layer {layer} yields no record it holds unlisted"),
        }
    }

    /// A round that listed expert `expert` of `layer` is past (the device
    /// converted it, or is past every step that could): its record frees
    /// once no list stands.
    pub fn release(&mut self, layer: usize, expert: usize) {
        let Some(State::Listed { record, lists }) = self.carried[layer][expert] else {
            panic!("released expert {expert} of layer {layer}, which no round listed");
        };
        self.carried[layer][expert] = match lists {
            1 => {
                self.unoccupy(layer, record);
                None
            }
            _ => Some(State::Listed { record, lists: lists - 1 }),
        };
    }

    /// Free every unlisted record of a layer the served group is past
    /// (`past`): no step of the group lists it any more. A landed record
    /// keeps its bytes for staging once the group is past.
    pub fn release_past(&mut self, past: impl Fn(usize) -> bool) {
        for layer in (0..self.carried.len()).filter(|&layer| past(layer)) {
            for expert in 0..self.carried[layer].len() {
                match self.carried[layer][expert] {
                    // A listed record frees as its rounds pass.
                    Some(State::Held { record }) => {
                        self.carried[layer][expert] = None;
                        self.unoccupy(layer, record);
                    }

                    _ => {}
                }
            }
        }
    }

    /// Whether nothing of the group past runs any more: no read, no
    /// landing from a conveyor record (each learned by [`Self::landed`]
    /// first). Each one's end is an event of the load.
    pub fn quiet(&self) -> bool {
        self.landings.is_empty()
            && !self.reading.iter().any(|&(layer, expert)| {
                matches!(&self.carried[layer][expert], Some(State::Reading { ticket, .. }) if ticket.running())
            })
    }

    /// Empty every record, the group past and nothing of it running
    /// ([`Self::quiet`]).
    pub fn clear(&mut self) {
        self.reading.clear();
        for experts in &mut self.carried {
            experts.iter_mut().for_each(|state| *state = None);
        }
        for records in &mut self.occupant {
            records.iter_mut().for_each(|record| *record = None);
        }
        self.slots.iter_mut().for_each(|slot| *slot = None);
    }
}
