//! Pure engine-level memory authority.
//!
//! This module consumes Seismic observations and produces typed decisions. It
//! does not allocate device memory and it does not replace Seismic's byte
//! ledger. Physical callers commit a holding only after the backend allocation
//! succeeds; the holding is the engine's classification of that charged
//! allocation.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HoldingId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClaimId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldingClass {
    Surplus,
    Retained,
    Dormant,
    Live,
    InFlight,
    Model,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryObservation {
    /// The stable fit of the device's storage: what its device and upload
    /// placements hold together (`Fit::storage_beside`), each store within
    /// process limits and grants, less its planning reserve, and counted
    /// once however many placements reach it.
    pub capacity_bytes: u64,
    /// Current Seismic charge of the device's storage, every kind.
    pub charged_bytes: u64,
}

/// The contracts a claim's bytes are placed under: a claim's words are its
/// `Shared` bytes.
pub const CLAIMED: [seismic::Contract; 3] =
    [seismic::Contract::DeviceOnly, seismic::Contract::Upload, seismic::Contract::Shared];

/// What one claim demands of its device's storage: bytes of device, upload
/// and host storage, the part of its device bytes the backend stages
/// through host RAM, and device and host bytes a reservation commits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClaimDemand {
    /// Device bytes allocated as the backend allocates, which the free space
    /// of the blocks the device holds serves first.
    pub device: u64,
    /// Upload storage bytes: what the host only writes (a reservation's
    /// record table).
    pub upload: u64,
    /// Host storage bytes allocated as the backend allocates: shared
    /// tensors (`Tensor::shared`), which the host reads and writes in place
    /// and the free space of the host blocks the device holds serves first.
    pub shared: u64,
    pub staged: u64,
    /// Device bytes a reservation backs: fresh pages of the device's store,
    /// which no block's free space serves.
    pub reserved: u64,
    /// Host storage bytes a reservation backs (a host tier's): fresh host
    /// pages, which no block's free space serves.
    pub host: u64,
}

impl ClaimDemand {
    /// The storage the claim covers: its device, reserved, upload, shared
    /// and host bytes.
    pub fn bytes(self) -> u64 {
        self.device
            .saturating_add(self.reserved)
            .saturating_add(self.upload)
            .saturating_add(self.shared)
            .saturating_add(self.host)
    }

    /// Exact sum for planning, where overflow must refuse the plan.
    pub fn checked_bytes(self) -> Option<u64> {
        self.device
            .checked_add(self.reserved)?
            .checked_add(self.upload)?
            .checked_add(self.shared)?
            .checked_add(self.host)
    }

    /// Exact composition of independently owned storage demands.
    pub fn checked_plus(self, other: Self) -> Option<Self> {
        Some(Self {
            device: self.device.checked_add(other.device)?,
            upload: self.upload.checked_add(other.upload)?,
            shared: self.shared.checked_add(other.shared)?,
            staged: self.staged.checked_add(other.staged)?,
            reserved: self.reserved.checked_add(other.reserved)?,
            host: self.host.checked_add(other.host)?,
        })
    }

    /// Exact composition of every one of `demands` ([`Self::checked_plus`]).
    pub fn checked_sum(demands: impl IntoIterator<Item = Self>) -> Option<Self> {
        demands.into_iter().try_fold(Self::default(), Self::checked_plus)
    }

    /// A reservation's backing in its tier and its record table, an
    /// ordinary allocation in the table's own placement.
    pub fn reservation(tier: crate::planning::BankTier, charges: seismic::ReservationCharges) -> Self {
        let mut demand = Self::default();
        match tier {
            crate::planning::BankTier::Bank => demand.reserved = charges.backing,
            crate::planning::BankTier::Host => demand.host = charges.backing,
        }
        if let Some(table) = charges.table {
            match table.placement() {
                seismic::Contract::Upload => demand.upload = table.bytes(),
                seismic::Contract::Shared | seismic::Contract::Words => demand.shared = table.bytes(),
                seismic::Contract::DeviceOnly => demand.device = table.bytes(),
            }
        }
        demand
    }

    /// Physical placement totals; reserved backing stays distinct in the
    /// claim itself so ordinary allocator free blocks cannot serve it.
    pub fn storage_bytes(self, kind: seismic::Contract) -> Option<u64> {
        match kind {
            seismic::Contract::DeviceOnly => self.device.checked_add(self.reserved),
            seismic::Contract::Upload => Some(self.upload),
            seismic::Contract::Shared => self.shared.checked_add(self.host),
            // A claim's words are its `shared` bytes: they lie where
            // fixed `Shared` buffers do, by the same admission.
            seismic::Contract::Words => Some(0),
        }
    }

    /// This demand and `other` together.
    pub fn plus(self, other: Self) -> Self {
        Self {
            device: self.device.saturating_add(other.device),
            upload: self.upload.saturating_add(other.upload),
            shared: self.shared.saturating_add(other.shared),
            staged: self.staged.saturating_add(other.staged),
            reserved: self.reserved.saturating_add(other.reserved),
            host: self.host.saturating_add(other.host),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Holding {
    pub id: HoldingId,
    pub bytes: u64,
    pub class: HoldingClass,
}

/// One outstanding claim: what it demands until its physical operation's
/// charge is classified or it is cancelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryClaim {
    pub id: ClaimId,
    pub demand: ClaimDemand,
    pub class: HoldingClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryStanding {
    pub observation: MemoryObservation,
    pub classified_bytes: u64,
    pub unattributed_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryError {
    InvalidObservation,
    UnknownHolding(HoldingId),
    UnknownClaim(ClaimId),
    HoldingOverflow,
    ChargeMismatch { classified: u64, charged: u64 },
    NotReleasable(HoldingClass),
    /// A claim reduced to more than it demanded ([`MemoryHeap::reduce_claim`]).
    ClaimGrew { claimed: u64, reduced: u64 },
}

/// Engine-level ownership state: the classified holdings and the one ledger
/// of outstanding claims. The physical allocation and Seismic charge remain
/// outside this type; callers add a holding only after the physical
/// operation succeeds. Whether a claim fits is its owner's fit decision
/// (`DeviceHeap`, over `platform::Fit`), which reads the claims here.
#[derive(Clone, Debug)]
pub struct MemoryHeap {
    observation: Option<MemoryObservation>,
    holdings: BTreeMap<HoldingId, Holding>,
    claims: BTreeMap<ClaimId, MemoryClaim>,
    next_id: u64,
}

impl MemoryHeap {
    pub fn new() -> Self {
        Self {
            observation: None,
            holdings: BTreeMap::new(),
            claims: BTreeMap::new(),
            next_id: 1,
        }
    }

    pub fn observe(&mut self, observation: MemoryObservation) -> Result<(), MemoryError> {
        if observation.charged_bytes > observation.capacity_bytes {
            return Err(MemoryError::InvalidObservation);
        }
        self.observation = Some(observation);
        Ok(())
    }

    pub fn is_observed(&self) -> bool {
        self.observation.is_some()
    }

    pub fn standing(&self) -> Result<MemoryStanding, MemoryError> {
        let observation = self.observation.ok_or(MemoryError::InvalidObservation)?;
        let classified_bytes = self.classified_bytes()?;
        let unattributed_bytes = observation
            .charged_bytes
            .checked_sub(classified_bytes)
            .ok_or(MemoryError::ChargeMismatch {
                classified: classified_bytes,
                charged: observation.charged_bytes,
            })?;
        Ok(MemoryStanding {
            observation,
            classified_bytes,
            unattributed_bytes,
        })
    }

    pub fn holdings(&self) -> impl Iterator<Item = Holding> + '_ {
        self.holdings.values().copied()
    }

    pub fn claims(&self) -> impl Iterator<Item = MemoryClaim> + '_ {
        self.claims.values().copied()
    }

    /// What every outstanding claim demands together.
    pub fn claimed(&self) -> ClaimDemand {
        self.claims
            .values()
            .fold(ClaimDemand::default(), |total, claim| {
                total.plus(claim.demand)
            })
    }

    pub fn classified_bytes(&self) -> Result<u64, MemoryError> {
        self.holdings.values().try_fold(0u64, |total, holding| {
            total
                .checked_add(holding.bytes)
                .ok_or(MemoryError::HoldingOverflow)
        })
    }

    pub fn insert(&mut self, bytes: u64, class: HoldingClass) -> Result<HoldingId, MemoryError> {
        let observation = self.observation.ok_or(MemoryError::InvalidObservation)?;
        let classified = self.classified_bytes()?;
        let next = classified
            .checked_add(bytes)
            .ok_or(MemoryError::HoldingOverflow)?;
        if next > observation.charged_bytes {
            return Err(MemoryError::ChargeMismatch {
                classified: next,
                charged: observation.charged_bytes,
            });
        }
        let id = HoldingId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(MemoryError::HoldingOverflow)?;
        self.holdings.insert(id, Holding { id, bytes, class });
        Ok(id)
    }

    pub fn resize(&mut self, id: HoldingId, bytes: u64) -> Result<(), MemoryError> {
        let observation = self.observation.ok_or(MemoryError::InvalidObservation)?;
        let holding = self
            .holdings
            .get(&id)
            .copied()
            .ok_or(MemoryError::UnknownHolding(id))?;
        let classified = self
            .classified_bytes()?
            .saturating_sub(holding.bytes)
            .checked_add(bytes)
            .ok_or(MemoryError::HoldingOverflow)?;
        if classified > observation.charged_bytes {
            return Err(MemoryError::ChargeMismatch {
                classified,
                charged: observation.charged_bytes,
            });
        }
        self.holdings.insert(id, Holding { bytes, ..holding });
        Ok(())
    }

    /// Record a claim of `demand`, which its owner's fit decided, before a
    /// fallible physical allocation. Seismic remains the byte-charge
    /// authority; this claim is planning state only.
    pub fn claim(
        &mut self,
        demand: ClaimDemand,
        class: HoldingClass,
    ) -> Result<MemoryClaim, MemoryError> {
        let id = ClaimId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(MemoryError::HoldingOverflow)?;
        let claim = MemoryClaim { id, demand, class };
        self.claims.insert(id, claim);
        Ok(claim)
    }

    /// `id`'s claim now demands `demand`, no more than it did: what its operation has brought
    /// under Seismic's charge so far is the charge's, no longer the claim's.
    pub fn reduce_claim(&mut self, id: ClaimId, demand: ClaimDemand) -> Result<(), MemoryError> {
        let claim = self.claims.get_mut(&id).ok_or(MemoryError::UnknownClaim(id))?;
        if demand.bytes() > claim.demand.bytes() {
            return Err(MemoryError::ClaimGrew { claimed: claim.demand.bytes(), reduced: demand.bytes() });
        }
        claim.demand = demand;
        Ok(())
    }

    pub fn cancel_claim(&mut self, id: ClaimId) -> Result<MemoryClaim, MemoryError> {
        self.claims.remove(&id).ok_or(MemoryError::UnknownClaim(id))
    }

    pub fn commit_claim(
        &mut self,
        id: ClaimId,
        charged_bytes: u64,
    ) -> Result<HoldingId, MemoryError> {
        let claim = self
            .claims
            .remove(&id)
            .ok_or(MemoryError::UnknownClaim(id))?;
        if charged_bytes > claim.demand.bytes() {
            self.claims.insert(id, claim);
            return Err(MemoryError::ChargeMismatch {
                classified: charged_bytes,
                charged: claim.demand.bytes(),
            });
        }
        match self.insert(charged_bytes, claim.class) {
            Ok(holding) => Ok(holding),
            Err(error) => {
                self.claims.insert(id, claim);
                Err(error)
            }
        }
    }

    pub fn remove(&mut self, id: HoldingId) -> Result<Holding, MemoryError> {
        let holding = self
            .holdings
            .remove(&id)
            .ok_or(MemoryError::UnknownHolding(id))?;
        if matches!(
            holding.class,
            HoldingClass::Live | HoldingClass::InFlight | HoldingClass::Model
        ) {
            self.holdings.insert(id, holding);
            return Err(MemoryError::NotReleasable(holding.class));
        }
        Ok(holding)
    }
}

impl Default for MemoryHeap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(charged: u64, available: u64) -> MemoryHeap {
        let mut heap = MemoryHeap::new();
        heap.observe(MemoryObservation {
            capacity_bytes: charged + available,
            charged_bytes: charged,
        })
        .unwrap();
        heap
    }

    #[test]
    fn classifications_reconcile_against_seismic_charge() {
        let mut heap = heap(100, 20);
        heap.insert(40, HoldingClass::Model).unwrap();
        heap.insert(60, HoldingClass::Live).unwrap();
        assert_eq!(heap.standing().unwrap().unattributed_bytes, 0);
    }

    #[test]
    fn live_and_in_flight_holdings_cannot_be_released_directly() {
        let mut heap = heap(20, 0);
        let id = heap.insert(20, HoldingClass::Live).unwrap();
        assert_eq!(
            heap.remove(id),
            Err(MemoryError::NotReleasable(HoldingClass::Live))
        );
    }

    /// A storage demand of `device` device bytes alone.
    fn device(device: u64) -> ClaimDemand {
        ClaimDemand {
            device,
            ..ClaimDemand::default()
        }
    }

    #[test]
    fn claims_hold_their_demand_and_commit_only_observed_charge() {
        let mut heap = heap(80, 40);
        let claim = heap.claim(device(30), HoldingClass::Dormant).unwrap();
        let other = ClaimDemand {
            device: 4,
            upload: 2,
            shared: 6,
            staged: 1,
            reserved: 3,
            host: 5,
        };
        heap.claim(other, HoldingClass::Live).unwrap();
        assert_eq!(heap.claims().count(), 2);
        assert_eq!(heap.claimed(), device(30).plus(other));
        assert_eq!(
            heap.commit_claim(claim.id, 31),
            Err(MemoryError::ChargeMismatch {
                classified: 31,
                charged: 30,
            })
        );
        let holding = heap.commit_claim(claim.id, 24).unwrap();
        assert_eq!(heap.claimed(), other);
        assert_eq!(heap.claims().count(), 1);
        assert_eq!(
            heap.holdings()
                .find(|item| item.id == holding)
                .unwrap()
                .bytes,
            24
        );
    }

    #[test]
    fn peak_claim_releases_into_one_measured_existing_holding() {
        let mut heap = heap(80, 40);
        let model = heap.insert(80, HoldingClass::Model).unwrap();
        let peak = heap.claim(device(30), HoldingClass::Model).unwrap();
        assert_eq!(heap.claims().count(), 1);
        heap.observe(MemoryObservation {
            capacity_bytes: 120,
            charged_bytes: 92,
        })
        .unwrap();
        heap.cancel_claim(peak.id).unwrap();
        heap.resize(model, 92).unwrap();
        let standing = heap.standing().unwrap();
        assert_eq!(standing.classified_bytes, 92);
        assert_eq!(standing.unattributed_bytes, 0);
        assert_eq!(heap.holdings().count(), 1);
        assert_eq!(heap.claims().count(), 0);
    }
}
