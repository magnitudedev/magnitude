use super::{decode_set, ExpertBankPlan, Placer};
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
        let held = self.member_layers(group).map(|layer| self.layer_vram_slots(layer, slots)).sum::<u64>();
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
    /// one record. Speed is not its concern (the assessment's
    /// estimate at the floor reports that).
    pub fn least_records(&self, group: usize) -> u64 {
        self.decode_records(group).max(self.prompt_records(group)).max(1)
    }

    /// The records of group `group`'s bank a prompt group holds live at
    /// once: the lender's lent records, one window of scratch
    /// ([`Self::window_slots`]) for every group and, where the conveyor lies
    /// in the bank's own records (the one store, [`ConveyorCarrier::Bank`]),
    /// the conveyor beside it ([`Self::scratch`]), and there the tier's
    /// reserve ([`Self::tier_reserve`]), which holds graces a claim cannot
    /// take; none of any other group, whose experts a prompt group never
    /// takes. The one place the prompt group's need is stated.
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
    ///   most `R = min(VERIFY_ROWS · K, E)` distinct experts, the widest
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
    /// `group`'s layers: `R` of [`Self::decode_records`], the decode set
    /// ([`decode_set`]) of the group's widest member across `VERIFY_ROWS`
    /// (every launch with `M <= VERIFY_ROWS`, including short prompts and
    /// verification batches, runs the decode form).
    pub(super) fn request_records(&self, group: usize) -> u64 {
        let selected = self
            .member_layers(group)
            .map(|layer| layer.selected)
            .max()
            .unwrap_or_else(|| self.selected());
        decode_set(selected, self.widest_member(group), magnitude_kernels::constants::VERIFY_ROWS)
    }

    /// Host-tier slots for `layer` at `(device_slots, host_slots)`:
    /// `0` when `layer` is whole in VRAM (`vram >= layer.experts`), otherwise
    /// `host_slots.min(layer.experts)` so a host tier with enough RAM can hold
    /// full cover (`HostRecords::complete()` / `tier_complete()`) with zero
    /// NVMe spills.
    pub fn layer_host_slots(&self, layer: &super::BankLayer, device_slots: u64, host_slots: u64) -> u64 {
        if !self.tiered() {
            return 0;
        }
        if self.layer_vram_slots(layer, device_slots) >= layer.experts {
            return 0;
        }
        host_slots.min(layer.experts)
    }

    /// The most host slots useful beside a device bank at `device_slots`:
    /// the widest `layer.experts` across all partial streamed layers (`0` when
    /// `!tiered()` or `device_slots` is whole).
    pub fn host_saturation_at(&self, device_slots: u64) -> u64 {
        if !self.tiered() {
            return 0;
        }
        self.layers()
            .iter()
            .filter(|layer| self.layer_vram_slots(layer, device_slots) < layer.experts)
            .map(|layer| layer.experts)
            .max()
            .unwrap_or(0)
    }

    /// The fixed device coordinate against which this plan's host-tier records
    /// and charges are sized: `self.slots` for a partial bank (disjoint sizing
    /// `H_l <= E_l - V_l(self.slots)`), or `0` for a bank planned whole (so
    /// that a whole bank that later shrinks to partial can still grow its host
    /// tier up to `host_saturation_at(current_slots)`, while `BankShape::host`
    /// holds it at `0` whenever the device bank is whole).
    pub fn host_reference_slots(&self) -> u64 {
        if self.whole() { 0 } else { self.slots }
    }

    /// The most host slots a disjoint host tier holds beside a device bank at
    /// `device_slots`: `host_saturation_at(device_slots)`, capped by
    /// `host_cap` when set; none without a tier.
    pub fn host_ceiling_at(&self, device_slots: u64) -> Option<u64> {
        self.tiered().then(|| {
            let sat = self.host_saturation_at(device_slots);
            self.host_cap.map_or(sat, |cap| cap.min(sat))
        })
    }

    /// The most host slots a host tier holds at this plan's reference device
    /// coordinate ([`Self::host_reference_slots`]); none without a tier.
    pub fn host_ceiling(&self) -> Option<u64> {
        self.host_ceiling_at(self.host_reference_slots())
    }

    /// Records of group `group`'s host tier tensors (`TR`): every expert of
    /// its layers and, of the conveyor's lender, the conveyor's lent
    /// records ([`ConveyorCarrier::HostTier`]); none without a tier.
    pub fn host_extent(&self, group: usize) -> u64 {
        match self.tiered() {
            true => self.member_experts(group).sum::<u64>() + self.host_lent(group),
            false => 0,
        }
    }

    /// Single holdings-to-records owner for group `group`'s committed host
    /// tier records at fixed `device_slots`, `host_slots`, and `paths`:
    /// each member layer's disjoint host slots ([`Self::layer_host_slots`]),
    /// floored by the group's landing reserve (`group_landing_records_with_paths`),
    /// plus `host_lent`, capped at `host_extent`; none without a tier.
    pub fn host_records_at_with_paths(
        &self,
        group: usize,
        device_slots: u64,
        host_slots: u64,
        paths: super::BankPaths,
    ) -> u64 {
        match self.tiered() {
            true => (self
                .member_layers(group)
                .map(|layer| self.layer_host_slots(layer, device_slots, host_slots))
                .sum::<u64>()
                .max(self.group_landing_records_with_paths(group, paths))
                + self.host_lent(group))
            .min(self.host_extent(group)),
            false => 0,
        }
    }

    /// Group `group`'s committed host tier records at `host_slots` host slots
    /// beside this plan's reference device coordinate.
    pub fn host_records(&self, group: usize, host_slots: u64) -> u64 {
        self.host_records_at_with_paths(group, self.host_reference_slots(), host_slots, self.paths)
    }

    /// What the host tier's pools are charged at fixed `device_slots`,
    /// `host_slots`, and `paths`: whole blocks holding each group's granule
    /// for each record committed beyond `host_lent`; none without a tier.
    pub fn host_charge_at_with_paths(
        &self,
        device_slots: u64,
        host_slots: u64,
        paths: super::BankPaths,
    ) -> Result<u64, String> {
        if !self.tiered() {
            return Ok(0);
        }
        self.pooled_charge_for(self.blocks(), |group| {
            self.host_records_at_with_paths(group, device_slots, host_slots, paths)
                .checked_sub(self.host_lent(group))
                .expect("a group's host records hold its lent ones")
        })
        .ok_or_else(|| "host tier bytes overflow".to_owned())
    }

    /// What the host tier's pools are charged at `host_slots` host slots
    /// beside this plan's reference device coordinate.
    pub fn host_charge(&self, host_slots: u64) -> Result<u64, String> {
        self.host_charge_at_with_paths(self.host_reference_slots(), host_slots, self.paths)
    }

    /// What the host tier at `host_slots` holds past its floor: cache,
    /// never required bytes. The plan grows the tier into the host RAM its
    /// charge leaves, the charge holding the tier at its floor
    /// (`ExecutionPlanDraft::with_host_tier`); the load holds exactly this
    /// as its store's bank cache (`ServedBank::new`: a holding's charge
    /// above its floor, the tier's floor none, from this plan's charges);
    /// assessment reports it as the tier's cache
    /// (`ColdSource::HostTier::cache_bytes`).
    ///
    /// The charge is monotone in slots by construction (each group's
    /// records and their granule-rounded backing only rise with them, its
    /// table fixed by its extent), so a tier charged less than its floor is
    /// an invariant violation.
    pub fn host_cache(&self, host_slots: u64) -> Result<u64, String> {
        Ok(self
            .host_charge(host_slots)?
            .checked_sub(self.host_charge(0)?)
            .expect("a host tier is charged less than its floor"))
    }

    /// The largest of the bank's lent allocations (`BankAllocation::Lent`):
    /// each is one allocation, so the device's single-allocation limit
    /// bounds it.
    pub fn largest_lent(&self) -> Result<u64, String> {
        let layout = self.layout()?;
        let largest = layout.lent.iter().flat_map(|lent| lent.iter()).try_fold(0u64, |most, (_, allocation)| {
            Ok::<_, String>(most.max(allocation.charged_bytes(self.host())?))
        })?;
        Ok(largest)
    }

    /// What the bank demands ([`Self::storage_demand`]), by the part that
    /// owns each byte: its layout's ([`BankLayout::owned`]), the places a
    /// group holds at its floor's one, and its pools' granules
    /// ([`Self::pooled_charges`]), each part's sum.
    pub fn charges(&self) -> Result<Vec<(&'static str, ClaimDemand)>, String> {
        let (layout, floor) = (self.layout()?, self.placed_allocations(NonZeroU64::MIN)?);
        let mut charges = Vec::<(&'static str, ClaimDemand)>::new();
        let owned = layout.owned().into_iter().chain(floor.iter().map(|allocation| ("places", allocation)));
        let pooled = self.pooled_charges()?;
        let owned = owned
            .map(|(owner, allocation)| Ok::<_, String>((owner, allocation.storage_demand(self.host())?)))
            .chain(pooled.into_iter().map(Ok));
        for owned in owned {
            let (owner, demand) = owned?;
            match charges.iter_mut().find(|(of, _)| *of == owner) {
                Some((_, total)) => *total = total.checked_plus(demand).ok_or("bank charges overflow")?,
                None => charges.push((owner, demand)),
            }
        }
        Ok(charges)
    }
}
