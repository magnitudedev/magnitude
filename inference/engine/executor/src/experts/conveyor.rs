//! The conveyor: a prompt group's cold experts
//! of a layer, read as each step's scratch request publishes them (every
//! step's request of a layer comes before any of its rounds there), each
//! once per group, and converted by the step's rounds from its records
//! (`routed_round`'s source), each as its read lands. The conveyor works only for a group:
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
//! Records its lender lends it are the store's between groups; they are a
//! group's lease, granted from its first request on, before any hit is
//! taken (`Scratch::lease`), so no hit lies in one, and returned empty once
//! the group is past: the conveyor holds the stored form, which
//! `routed_round` repacks, so a claimed record holds no bank-form expert.
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
//! has converted it ([`Self::take`], `Scratch::lendable_slot`, once the
//! group's lease holds the scratch), and
//! its round converts it from there (`routed_round`'s negative `from`).
//! A slot's record is `records + slot`, past the conveyor's own; it frees
//! as a conveyor record does, and holds no bytes kept past the group.

use super::source::{Outcome, Records, Ticket};
use super::watcher::{ScratchGroup, Step};

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

/// The group the conveyor serves, a prompt group (its first step and its
/// steps, as the steps that began it read them), and whether the device is
/// past it (the conveyor then clears once nothing of the group runs,
/// `Scratch::clear_past`, and takes no later group's request before). What
/// of its lease it holds is its lender's host records' to say
/// (`Directory::leased`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Serving {
    pub group: ScratchGroup,
    pub past: bool,
}

/// The group lending the conveyor its bottom host records
/// (`planning::Lender`; on unified memory the scratch's too), which are a
/// prompt group's lease (`Directory::lease`, whose records count them,
/// `HostRecords::lent`), and where each staging pool's conveyor lies in
/// them (`planning::ConveyorPlan`): pool `p`'s conveyor lies at `offset`
/// bytes of its routed tensor's lent records ([`LentPool`]).
pub(super) struct Lending {
    pub group: usize,
    pub pools: Vec<LentPool>,
}

/// A pool's conveyor in the lent records: each lent record of its routed
/// tensor `record` bytes, the conveyor's records of `conveyor` bytes each at
/// `offset` bytes of them, one after another.
pub(super) struct LentPool {
    pub record: u64,
    pub offset: u64,
    pub conveyor: u64,
}

impl Lending {
    /// The lent records conveyor record `record` of pool `pool` lies across.
    pub fn records(&self, pool: usize, record: u32) -> std::ops::RangeInclusive<usize> {
        let LentPool { record: bytes, offset, conveyor } = &self.pools[pool];
        let start = offset + conveyor * u64::from(record);
        let lent = |byte: u64| usize::try_from(byte / bytes).expect("a lent record");
        lent(start)..=lent(start + conveyor - 1)
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
/// ends, as `ended` records (landed: its bytes are whole; unlanded if
/// cancelled via `cancel`).
pub(super) struct ConveyorLanding {
    pub layer: usize,
    pub expert: usize,
    pub record: u32,
    pub tier: u32,
    pub read: Option<Step>,
    pub ended: crate::experts::landing::Ended,
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
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

pub(super) struct Conveyor {
    /// Each streamed layer's staging pool indices, in record order.
    layer_pools: Vec<Vec<usize>>,
    /// Per layer, where each expert carried is.
    carried: Vec<Vec<Option<State>>>,
    /// The experts being read, (layer, expert): what settling looks at.
    reading: Vec<(usize, usize)>,
    /// Each physical conveyor slot's expert (layer, expert): all staging pools
    /// share one conveyor region, so each slot has at most one owner.
    occupant: Vec<Option<(usize, usize)>>,
    /// Per scratch slot, the expert (layer, expert) whose stored bytes it
    /// carries, from its read's start until its round's converts are past.
    slots: Vec<Option<(usize, usize)>>,
    /// The landings into the host tier from conveyor records in flight.
    landings: Vec<ConveyorLanding>,
    /// The group served, from its first scratch request until the device is
    /// past it.
    pub serving: Option<Serving>,
    pub carrier: Carrier,
    /// A test's hold on the group's reads ([`Self::hold_reads`]).
    #[cfg(test)]
    held: bool,
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
    /// A conveyor for layers whose routed tensors use `layer_pools` (of
    /// layers of `experts` experts each), carried by `carrier`; every
    /// record empty.
    pub fn new(layer_pools: Vec<Vec<usize>>, experts: &[usize], carrier: Carrier) -> Self {
        Self {
            carried: experts.iter().map(|&experts| (0..experts).map(|_| None).collect()).collect(),
            reading: Vec::new(),
            occupant: vec![None; carrier.records],
            slots: vec![None; carrier.slots],
            landings: Vec::new(),
            layer_pools,
            serving: None,
            carrier,
            #[cfg(test)]
            held: false,
        }
    }

    /// While `held`, no read of the conveyor has ended as it observes them
    /// ([`Self::ended`]): a test holds a group's reads in flight for as
    /// long as it needs, whatever the drive's speed. A release is observed
    /// at the load's next event.
    #[cfg(test)]
    pub fn hold_reads(&mut self, held: bool) {
        self.held = held;
    }

    /// How `ticket`'s read ended, none while it runs: the one observation
    /// settling ([`Self::settle`]) and quiet ([`Self::quiet`]) read.
    fn ended(&self, ticket: &Ticket) -> Option<Outcome> {
        #[cfg(test)]
        if self.held {
            return None;
        }
        ticket.wait_until(std::time::Instant::now())
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

    /// The `from` a round's landing list names record `record` by
    /// (`routed_round`): a conveyor record, or -1 - slot.
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

    /// Free record `record` for another read, a conveyor record or a slot.
    fn unoccupy(&mut self, record: u32) {
        match self.slot_of(record) {
            Some(slot) => self.slots[slot as usize] = None,
            None => self.occupant[record as usize] = None,
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

    /// Whether conveyor record `rec_a` of `layer_a` and `rec_b` of `layer_b`
    /// overlap in the shared conveyor backing: always for equal slot indices,
    /// or when layers with distinct pool record sizes overlap byte ranges in
    /// any shared carrier tensor.
    fn overlaps(&self, layer_a: usize, rec_a: u32, layer_b: usize, rec_b: u32) -> bool {
        if rec_a == rec_b {
            return true;
        }
        let pa = &self.layer_pools[layer_a];
        let pb = &self.layer_pools[layer_b];
        if pa == pb {
            return false;
        }
        let pools = &self.carrier.lending.pools;
        pa.iter().zip(pb).any(|(&pa, &pb)| {
            if pa == pb {
                return false;
            }
            let a = &pools[pa];
            let b = &pools[pb];
            let a_start = a.offset + u64::from(rec_a) * a.conveyor;
            let a_end = a_start + a.conveyor;
            let b_start = b.offset + u64::from(rec_b) * b.conveyor;
            let b_end = b_start + b.conveyor;
            a_start < b_end && b_start < a_end
        })
    }

    /// Whether conveyor record `record` of `layer` overlaps any currently
    /// occupied conveyor slot or in-flight conveyor landing.
    fn conflicts(&self, layer: usize, record: u32) -> bool {
        self.occupant
            .iter()
            .enumerate()
            .any(|(other_rec, occ)| occ.is_some_and(|(other_layer, _)| self.overlaps(layer, record, other_layer, other_rec as u32)))
            || self.landings.iter().any(|landing| self.overlaps(layer, record, landing.layer, landing.record))
    }

    /// The conveyor records free for `layer` that `admits`: none carries an
    /// expert, none overlaps an occupied record of another pool size, and no
    /// landing reads it.
    pub fn free_conveyor(&self, layer: usize, admits: impl Fn(u32) -> bool) -> Vec<u32> {
        Self::free(&self.occupant).filter(|&record| !self.conflicts(layer, record) && admits(record)).collect()
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

    /// Cancel every in-flight landing from a conveyor record into the host
    /// tier so the next group never stalls behind previous-group CPU repacks.
    pub fn cancel_landings(&mut self) {
        for landing in &self.landings {
            landing.cancel.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    /// Cancel any in-flight landing holding or overlapping conveyor record
    /// `record` for `layer`.
    pub fn cancel_landing_for(&mut self, layer: usize, record: u32) {
        for landing in &self.landings {
            if landing.record == record || self.overlaps(layer, record, landing.layer, landing.record) {
                landing.cancel.store(true, std::sync::atomic::Ordering::Release);
            }
        }
    }

    /// A conveyor record for `layer` that `admits` and is occupied by no
    /// active GPU round (nor overlaps one), held only by an uncancelled
    /// [`ConveyorLanding`].
    pub fn landing_only_record(&self, layer: usize, admits: impl Fn(u32) -> bool) -> Option<u32> {
        Self::free(&self.occupant).find(|&record| {
            admits(record)
                && !self.occupant.iter().enumerate().any(|(other_rec, occ)| {
                    occ.is_some_and(|(other_layer, _)| self.overlaps(layer, record, other_layer, other_rec as u32))
                })
                && self.landings.iter().any(|landing| {
                    (landing.record == record || self.overlaps(layer, record, landing.layer, landing.record))
                        && !landing.cancel.load(std::sync::atomic::Ordering::Acquire)
                })
        })
    }

    /// Take the conveyor landings that ended: each ended with `Some(true)`
    /// (landed) or `Some(false)` (cancelled).
    pub fn landed(&mut self) -> Vec<ConveyorLanding> {
        let (ended, running): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.landings).into_iter().partition(|landing| landing.ended.get().is_some());
        self.landings = running;
        ended
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
        assert!(
            !self.conflicts(layer, record),
            "conveyor record {record} of layer {layer} overlaps an occupied record or landing"
        );
        let slot = &mut self.occupant[record as usize];
        assert!(slot.is_none(), "a conveyor record carries two experts");
        *slot = Some((layer, expert));
    }

    /// Give back record `record`.
    pub fn untake(&mut self, record: u32) {
        self.unoccupy(record);
    }

    /// Expert `expert` of `layer` is being read into conveyor record
    /// `record`, which it took.
    pub fn reading(&mut self, layer: usize, expert: usize, ticket: Ticket, record: u32) {
        let slot = &mut self.carried[layer][expert];
        assert!(slot.is_none(), "an expert carried twice");
        *slot = Some(State::Reading { ticket, record });
        self.reading.push((layer, expert));
    }

    /// How many experts are being read now.
    pub fn reads_in_flight(&self) -> usize {
        self.reading.len()
    }

    /// Settle the reads that ended ([`Self::ended`]): one that landed
    /// holds its expert whole; one that failed carries nothing, its record
    /// free. The reads that ended, the experts landed whole, and the experts
    /// whose reads failed, with why.
    pub fn settle(&mut self) -> Settled {
        let mut settled = Settled::default();
        let mut reading = std::mem::take(&mut self.reading);
        reading.retain(|&(layer, expert)| {
            let Some(State::Reading { ticket, record }) = &self.carried[layer][expert] else {
                unreachable!("expert {expert} of layer {layer} is listed as being read");
            };
            let record = *record;
            let Some(ended) = self.ended(ticket) else {
                return true;
            };
            settled.ended += 1;
            match ended {
                Outcome::Read => {
                    self.carried[layer][expert] = Some(State::Held { record });
                    settled.landed.push((layer, expert, record));
                }
                Outcome::Failed(reason) => {
                    self.carried[layer][expert] = None;
                    self.untake(record);
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
            Some(State::Held { record }) => self.untake(record),
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
                self.unoccupy(record);
                None
            }
            _ => Some(State::Listed { record, lists: lists - 1 }),
        };
    }

    /// Free every unlisted record of a layer the served group is past
    /// (`past`): no step of the group lists it any more.
    pub fn release_past(&mut self, past: impl Fn(usize) -> bool) {
        for layer in (0..self.carried.len()).filter(|&layer| past(layer)) {
            for expert in 0..self.carried[layer].len() {
                match self.carried[layer][expert] {
                    // A listed record frees as its rounds pass.
                    Some(State::Held { record }) => {
                        self.carried[layer][expert] = None;
                        self.unoccupy(record);
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
                matches!(&self.carried[layer][expert], Some(State::Reading { ticket, .. }) if self.ended(ticket).is_none())
            })
    }

    /// Empty every record, the group past and nothing of it running
    /// ([`Self::quiet`]).
    pub fn clear(&mut self) {
        self.reading.clear();
        for experts in &mut self.carried {
            experts.iter_mut().for_each(|state| *state = None);
        }
        self.occupant.iter_mut().for_each(|record| *record = None);
        self.slots.iter_mut().for_each(|slot| *slot = None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::events::Events;
    use crate::experts::landing::Ended;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn test_conveyor(layer_pools: Vec<Vec<usize>>, records: usize) -> Conveyor {
        let experts = vec![4; layer_pools.len()];
        let carrier = Carrier {
            records,
            slots: 0,
            lending: Lending { group: 0, pools: Vec::new() },
            pools: Vec::new(),
        };
        Conveyor::new(layer_pools, &experts, carrier)
    }

    fn dummy_landing(events: &Arc<Events>, layer: usize, expert: usize, record: u32, tier: u32) -> (ConveyorLanding, Arc<AtomicBool>, Ended) {
        let cancel = Arc::new(AtomicBool::new(false));
        let ended = Ended::new(events);
        let landing = ConveyorLanding {
            layer,
            expert,
            record,
            tier,
            read: Some(1),
            ended: ended.clone(),
            cancel: cancel.clone(),
        };
        (landing, cancel, ended)
    }

    /// `landing_only_record` identifies conveyor records unoccupied by any GPU
    /// round and held only by an uncancelled `ConveyorLanding`; `cancel_landing_for`
    /// cancels matching record landings, and `cancel_landings` cancels all.
    #[test]
    fn conveyor_landing_cancellation_and_lookup() {
        let events = Arc::new(Events::default());
        let mut conveyor = test_conveyor(vec![vec![], vec![], vec![]], 3);
        let (l0, c0, e0) = dummy_landing(&events, 0, 0, 0, 0);
        let (l1, c1, e1) = dummy_landing(&events, 1, 1, 1, 1);
        let (l2, c2, e2) = dummy_landing(&events, 2, 0, 2, 2);
        conveyor.land(l0);
        conveyor.land(l1);
        conveyor.land(l2);

        // Occupy record 0 with an active GPU round: records 1 and 2 are landing-only.
        conveyor.take(0, 2, 0);
        assert_eq!(conveyor.landing_only_record(1, |_| true), Some(1));

        // Cancel record 1: cancels l1 only.
        conveyor.cancel_landing_for(0, 1);
        assert!(!c0.load(Ordering::Acquire));
        assert!(c1.load(Ordering::Acquire));
        assert!(!c2.load(Ordering::Acquire));
        assert_eq!(conveyor.landing_only_record(1, |_| true), Some(2));

        // `cancel_landings` cancels every remaining in-flight landing.
        conveyor.cancel_landings();
        assert!(c0.load(Ordering::Acquire));
        assert!(c2.load(Ordering::Acquire));
        assert_eq!(conveyor.landing_only_record(1, |_| true), None, "already-cancelled landings skipped");

        // Settling ended landings drains them cleanly.
        e0.set(false);
        e1.set(false);
        e2.set(true);
        let drained = conveyor.landed();
        assert_eq!(drained.len(), 3);
        assert!(conveyor.quiet());
    }

    /// Mixed-quant layers sharing one conveyor region prevent both slot-index
    /// collisions and byte-range overlaps across distinct pool record sizes.
    #[test]
    fn shared_conveyor_prevents_byte_overlap_across_pool_sizes() {
        // Pool 0: 500 bytes/record; Pool 1: 800 bytes/record; both at offset 0.
        let carrier = Carrier {
            records: 4,
            slots: 0,
            lending: Lending {
                group: 0,
                pools: vec![
                    LentPool { record: 1000, offset: 0, conveyor: 500 },
                    LentPool { record: 1000, offset: 0, conveyor: 800 },
                ],
            },
            pools: Vec::new(),
        };
        // Layer 0 uses pool 0 (500 B), layer 1 uses pool 1 (800 B).
        let mut conveyor = Conveyor::new(vec![vec![0], vec![1]], &[8, 8], carrier);
        // Layer 0 takes slot 1 -> bytes [500, 1000).
        conveyor.take(0, 0, 1);
        // For layer 0 (same pool): slots 0, 2, 3 are free.
        assert_eq!(conveyor.free_conveyor(0, |_| true), vec![0, 2, 3]);
        // For layer 1 (800 B/slot):
        // - slot 0: [0, 800) overlaps [500, 1000) -> blocked
        // - slot 1: occupied -> blocked
        // - slot 2: [1600, 2400) disjoint -> free
        // - slot 3: [2400, 3200) disjoint -> free
        assert_eq!(conveyor.free_conveyor(1, |_| true), vec![2, 3]);
        conveyor.take(1, 0, 2);
        // Layer 0 slot 3 is [1500, 2000), which overlaps layer 1 slot 2 [1600, 2400) -> blocked!
        // Layer 0 slot 0 is [0, 500), disjoint -> free!
        assert_eq!(conveyor.free_conveyor(0, |_| true), vec![0]);
    }
}
