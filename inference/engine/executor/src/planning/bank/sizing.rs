use super::{ExpertBankPlan, Placer};
use crate::memory::ClaimDemand;
use std::num::NonZeroU64;

/// The shallowest shrink from `from` toward `floor` releasing `bytes` of
/// `charge` (one byte when none is owed), or as much as the floor releases;
/// none where the floor releases nothing. Charges fall with the holding but
/// not linearly: record counts, granules and minimum allocations make
/// plateaus, so a holding one less may release nothing.
pub(crate) fn releasing<E>(from: u64, floor: u64, bytes: u64, charge: impl Fn(u64) -> Result<u64, E>) -> Result<Option<u64>, E> {
    let current = charge(from)?;
    let available = current - charge(floor)?;
    if available == 0 {
        return Ok(None);
    }
    let limit = current - bytes.max(1).min(available);
    let (mut low, mut high) = (floor, from - 1);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if charge(middle)? <= limit {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(Some(low))
}

/// The commitment a holding at `from` grows to in one step: the largest `to`
/// at most `saturation` (where the record's misses stop falling, so a step
/// past it backs bytes that save nothing) whose bytes beyond `from`,
/// `bytes(to)`, are at most `room`, the most its store's claim admits; none
/// where the holding is saturated or one more does not fit. A commitment's
/// bytes never fall as it grows, so it is found by halving: `bytes` is
/// asked `O(log(saturation − from))` times.
///
/// Whether the step is taken is the domain's (`expert_bank_growths`): where
/// what `to` saves exceeds the holding's last resize. Two gaps of that rule:
/// the decode's priced step is not monotone in the slots (its window lays
/// misses out in whole steps, and the bank's slots change where a host-held
/// miss is placed), so at a boundary where `to` saves too little a smaller
/// step that would pay is not taken, and nothing grows; and the resize a
/// holding last held, of one size, prices a resize of another until that
/// one is measured.
pub fn growth_target<E>(
    from: u64,
    saturation: u64,
    room: u64,
    bytes: impl Fn(u64) -> Result<u64, E>,
) -> Result<Option<u64>, E> {
    if saturation <= from || bytes(from + 1)? > room {
        return Ok(None);
    }
    // The largest known to fit, and the smallest known not to.
    let (mut fitting, mut over) = (from + 1, saturation + 1);
    while over - fitting > 1 {
        let middle = fitting + (over - fitting) / 2;
        match bytes(middle)? <= room {
            true => fitting = middle,
            false => over = middle,
        }
    }
    Ok(Some(fitting))
}

impl ExpertBankPlan {
    /// Group `group`'s committed records at `slots` slots: each of its
    /// layers' slots ([`Self::layer_slots`]), where the host places beside
    /// them the tier's reserve ([`Self::tier_reserve`]), and never fewer
    /// than its least records ([`Self::least_records`]); at most its
    /// records ([`Self::records`]), where it holds every expert and needs
    /// nothing beside them. A group of few layers commits more records
    /// than its slots; the global slots are not raised for it, so no other
    /// group is charged its bound.
    pub fn records_at(&self, group: usize, slots: u64) -> u64 {
        let held = self.member_experts(group).map(|experts| Self::layer_slots(slots, experts)).sum::<u64>();
        let landing = match self.placer() {
            Placer::Tier => 0,
            Placer::Bank => self.tier_reserve(),
        };
        (held + landing).max(self.least_records(group))
    }

    /// The least records group `group` holds for every step to make
    /// progress: what the engine needs resident at once, the more of what a
    /// decode step ([`Self::decode_records`]) and a prompt group
    /// ([`Self::prompt_records`]) hold live at once, and never fewer than
    /// the one record a reserved tensor commits (Seismic's
    /// `committed_bytes`). Speed is not its concern (the assessment's
    /// estimate at the floor reports that).
    pub fn least_records(&self, group: usize) -> u64 {
        self.decode_records(group).max(self.prompt_records(group)).max(1)
    }

    /// The records of group `group`'s bank a prompt group holds live at
    /// once: the lender's lent records, one widest layer of scratch for
    /// every group and, where the conveyor lies in the bank's own records
    /// (the one store, [`ConveyorCarrier::Bank`]), the conveyor beside it
    /// ([`Self::scratch`]), and there the tier's reserve
    /// ([`Self::tier_reserve`]), which holds graces a claim cannot take; none of any other group, whose experts a
    /// prompt group never takes. The one place the prompt group's need is
    /// stated.
    pub fn prompt_records(&self, group: usize) -> u64 {
        if group != self.lender_group() {
            return 0;
        }
        let lent = self.scratch().lender.records;
        match self.placer() {
            Placer::Bank => lent + self.tier_reserve(),
            Placer::Tier => lent,
        }
    }

    /// The records group `group` holds for a decode step to make progress,
    /// from what takes a bank record and how long it holds it.
    ///
    /// - Where the device places from a host tier (`Placer::Tier`, a
    ///   discrete device), no decode step takes a bank record: every miss is
    ///   served from the tier (the CPU site computes it, or the device reads
    ///   it in place over the link), every file miss lands into a tier record
    ///   (`landing::Source`), and a promotion only replaces a record's
    ///   expert. None.
    /// - Where the host places into the bank (`Placer::Bank`, unified
    ///   memory, the one store), every file miss lands into a bank record. A
    ///   step parked at layer `l` for its misses takes a ready record, or
    ///   reclaims a kept one no resolution at or after the park may name
    ///   (`decode::ready_record`). The device runs in (group, layer,
    ///   sequence) order, so a park proves every earlier step past `l`: of
    ///   the steps in flight (`FlightDepth::steps`: the groups in flight,
    ///   a decode step's lookahead successor, a prompt group's places) only the parked step's resolution of one
    ///   layer holds records at once, its hits and its landings together at
    ///   most `R = min(DECODE_ROWS · K, E)` distinct experts, the widest
    ///   member's `E` (`decode::parked_reads`). A step begun behind it holds
    ///   records only through graces stamped past the park, which the
    ///   reserve's unmappings make and keep, ready and graced together, to at
    ///   most the tier's reserve (`HostRecords::refill`,
    ///   [`Self::tier_reserve`]). So the group
    ///   holds `R` beside the reserve: whatever the graces hold, `R` of the
    ///   rest are ready, kept and reclaimable, or landing and so soon kept.
    pub fn decode_records(&self, group: usize) -> u64 {
        match self.placer() {
            Placer::Tier => 0,
            Placer::Bank => self.request_records(group) + self.tier_reserve(),
        }
    }

    /// One decode request's distinct experts at most, at any of group
    /// `group`'s layers: `R` of [`Self::decode_records`].
    fn request_records(&self, group: usize) -> u64 {
        self.choices().min(self.widest_member(group))
    }

    /// The host slots a host tier at `from` grows to in one step
    /// ([`growth_target`]) beside a bank of `slots` slots: toward every
    /// expert, its ceiling (it is inclusive of the bank: it holds the bank's
    /// experts too), past the bank's slots, its charge beyond `from`'s
    /// ([`Self::host_charge`]) within `room`; none without a tier, whole, or
    /// with no room for one past the bank. What a live load's growth step
    /// takes where the memory's budget grows; the tier's load-time size is
    /// the plan's ([`Self::host_slots`]).
    pub fn host_growth(&self, from: u64, slots: u64, room: u64) -> Result<Option<u64>, String> {
        let Some(bound) = self.host_ceiling().map(NonZeroU64::get) else {
            return Ok(None);
        };
        let held = self.host_charge(from)?;
        // An inclusive tier's first slots, up to the bank's, hold copies of
        // the bank's experts and save nothing: a growth reaches past them in
        // one step, or none fits.
        let least = from.max(slots);
        growth_target(least, bound, room, |to| Ok(self.host_charge(to)? - held))
    }

    /// The most host slots a host tier holds: the widest layer's experts, a
    /// host slot a record of each layer of every group, as a bank slot is,
    /// so the tier's records hold every expert (the bank's among them);
    /// none without a tier.
    pub fn host_ceiling(&self) -> Option<NonZeroU64> {
        self.tier_reservation().map(|_| self.widest())
    }

    /// Records of group `group`'s host tier tensors (`TR`): every expert of
    /// its layers and, of the conveyor's lender, the conveyor's lent
    /// records ([`ConveyorCarrier::HostTier`]), reserved so a resize keeps
    /// every binding; none without a tier.
    pub fn host_extent(&self, group: usize) -> u64 {
        match self.tier_reservation() {
            Some(_) => self.member_experts(group).sum::<u64>() + self.tier_lent(group),
            None => 0,
        }
    }

    /// Group `group`'s committed host tier records at `host_slots` host
    /// slots: each of its layers' slots ([`Self::layer_slots`]), and never
    /// fewer than the landing reserve ([`Self::landing_records`]), so a
    /// demand landing finds a record
    /// past its grace whatever the tier holds, beside the conveyor's lent
    /// records ([`Self::tier_lent`]), its bottom ones, which a prompt group
    /// holds for its life; at most its extent; none without a tier.
    pub fn host_records(&self, group: usize, host_slots: u64) -> u64 {
        match self.tier_reservation() {
            Some(_) => (self
                .member_experts(group)
                .map(|experts| Self::layer_slots(host_slots, experts))
                .sum::<u64>()
                .max(self.landing_records())
                + self.tier_lent(group))
            .min(self.host_extent(group)),
            None => 0,
        }
    }

    /// What the host tier is charged at `host_slots` host slots: each
    /// group's record tensors' committed records there, resident bytes in
    /// whole granules of the tier's reservation (Seismic's rule); none
    /// without a tier.
    pub fn host_charge(&self, host_slots: u64) -> Result<u64, String> {
        let Some(reservation) = self.tier_reservation() else {
            return Ok(0);
        };
        self.record_tensors().try_fold(0u64, |total, (group, kind, tensor)| {
            let held = self.reserved_records(group, kind, tensor, crate::planning::BankTier::Host)?;
            held.record
                .get()
                .checked_mul(self.host_records(group, host_slots))
                .and_then(|bytes| reservation.charge(held, self.host_extent(group), bytes))
                .and_then(|charge| total.checked_add(charge))
                .ok_or_else(|| "host tier bytes overflow".to_owned())
        })
    }

    /// The most slots each record tensor's committed bytes keep within the
    /// device's single-allocation limit `max_allocation`, which bounds each
    /// native allocation backing them (`seismic::Reservation::most_backed`).
    pub fn most_slots(&self, max_allocation: u64) -> Result<u64, String> {
        self.record_tensors().try_fold(self.widest().get(), |most, (group, kind, tensor)| {
            let record = tensor.record_bytes()?;
            let held = self.reserved_records(group, kind, tensor, crate::planning::BankTier::Bank)?;
            let backed = self.reservation.of_records(held).most_backed(max_allocation) / record;
            // The most slots whose records of every member layer fit; zero
            // slots hold no records, so always fit.
            let fitting = (1..=most)
                .rev()
                .find(|&slots| {
                    self.member_experts(group).map(|experts| Self::layer_slots(slots, experts)).sum::<u64>() <= backed
                })
                .unwrap_or(0);
            Ok(fitting)
        })
    }

    /// What the bank demands ([`Self::storage_demand`]), by the part that
    /// owns each byte: its layout's ([`BankLayout::owned`]) and the places a
    /// group holds at its floor's one, each part's sum.
    pub fn charges(&self) -> Result<Vec<(&'static str, ClaimDemand)>, String> {
        let (layout, floor) = (self.layout()?, self.placed_allocations(NonZeroU64::MIN)?);
        let mut charges = Vec::<(&'static str, ClaimDemand)>::new();
        let owned = layout.owned().into_iter().chain(floor.iter().map(|allocation| ("places", allocation)));
        for (owner, allocation) in owned {
            let demand = allocation.storage_demand(&self.reservation_of(allocation)?)?;
            match charges.iter_mut().find(|(of, _)| *of == owner) {
                Some((_, total)) => *total = total.checked_plus(demand).ok_or("bank charges overflow")?,
                None => charges.push((owner, demand)),
            }
        }
        Ok(charges)
    }
}
