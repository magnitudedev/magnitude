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
pub enum Pressure {
    Normal,
    Pressure,
    Emergency,
    Blind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryObservation {
    /// Stable domain capacity, after applicable process and device limits.
    pub capacity_bytes: u64,
    /// Fresh bytes available to an additional allocation.
    pub available_bytes: u64,
    /// Current Seismic charge for this engine/device scope.
    pub charged_bytes: u64,
    pub pressure: Pressure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryNeed {
    pub minimum_bytes: u64,
    pub preferred_bytes: u64,
    pub class: HoldingClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Holding {
    pub id: HoldingId,
    pub bytes: u64,
    pub class: HoldingClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryClaim {
    pub id: ClaimId,
    pub bytes: u64,
    pub class: HoldingClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryStanding {
    pub observation: MemoryObservation,
    pub classified_bytes: u64,
    pub unattributed_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAction {
    Grant { bytes: u64 },
    ReleaseRequired { bytes: u64 },
    Wait,
    Reject { required: u64, available: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryError {
    InvalidNeed,
    InvalidObservation,
    UnknownHolding(HoldingId),
    UnknownClaim(ClaimId),
    HoldingOverflow,
    ChargeMismatch { classified: u64, charged: u64 },
    NotReleasable(HoldingClass),
}

/// Engine-level ownership and decision state. The physical allocation and
/// Seismic charge remain outside this type; callers add a holding only after
/// the physical operation succeeds.
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
        if observation.available_bytes > observation.capacity_bytes
            || observation.charged_bytes > observation.capacity_bytes
        {
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

    fn claimed_bytes(&self) -> Result<u64, MemoryError> {
        self.claims.values().try_fold(0u64, |total, claim| {
            total
                .checked_add(claim.bytes)
                .ok_or(MemoryError::HoldingOverflow)
        })
    }

    pub fn classified_bytes(&self) -> Result<u64, MemoryError> {
        self.holdings.values().try_fold(0u64, |total, holding| {
            total
                .checked_add(holding.bytes)
                .ok_or(MemoryError::HoldingOverflow)
        })
    }

    pub fn decide(&self, need: MemoryNeed) -> Result<MemoryAction, MemoryError> {
        if need.minimum_bytes > need.preferred_bytes {
            return Err(MemoryError::InvalidNeed);
        }
        let standing = self.standing()?;
        match standing.observation.pressure {
            Pressure::Blind | Pressure::Emergency => return Ok(MemoryAction::Wait),
            Pressure::Pressure => {
                return Ok(MemoryAction::ReleaseRequired {
                    bytes: need.minimum_bytes,
                })
            }
            Pressure::Normal => {}
        }
        let available = standing
            .observation
            .available_bytes
            .saturating_sub(self.claimed_bytes()?);
        let bytes = need.preferred_bytes.min(available);
        if bytes >= need.minimum_bytes {
            Ok(MemoryAction::Grant { bytes })
        } else {
            Ok(MemoryAction::Reject {
                required: need.minimum_bytes,
                available,
            })
        }
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

    /// Reserve capacity before a fallible physical allocation. Seismic remains
    /// the byte-charge authority; this claim is planning state only.
    pub fn claim(&mut self, need: MemoryNeed) -> Result<MemoryClaim, MemoryError> {
        let bytes = match self.decide(need)? {
            MemoryAction::Grant { bytes } if bytes >= need.minimum_bytes => bytes,
            _ => return Err(MemoryError::InvalidObservation),
        };
        let id = ClaimId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(MemoryError::HoldingOverflow)?;
        let claim = MemoryClaim {
            id,
            bytes,
            class: need.class,
        };
        self.claims.insert(id, claim);
        Ok(claim)
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
        if charged_bytes > claim.bytes {
            self.claims.insert(id, claim);
            return Err(MemoryError::ChargeMismatch {
                classified: charged_bytes,
                charged: claim.bytes,
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

    fn heap(charged: u64, available: u64, pressure: Pressure) -> MemoryHeap {
        let mut heap = MemoryHeap::new();
        heap.observe(MemoryObservation {
            capacity_bytes: charged + available,
            available_bytes: available,
            charged_bytes: charged,
            pressure,
        })
        .unwrap();
        heap
    }

    #[test]
    fn classifications_reconcile_against_seismic_charge() {
        let mut heap = heap(100, 20, Pressure::Normal);
        heap.insert(40, HoldingClass::Model).unwrap();
        heap.insert(60, HoldingClass::Live).unwrap();
        assert_eq!(heap.standing().unwrap().unattributed_bytes, 0);
    }

    #[test]
    fn pressure_blocks_new_growth_until_release() {
        let heap = heap(100, 20, Pressure::Pressure);
        assert_eq!(
            heap.decide(MemoryNeed {
                minimum_bytes: 8,
                preferred_bytes: 16,
                class: HoldingClass::Live,
            })
            .unwrap(),
            MemoryAction::ReleaseRequired { bytes: 8 }
        );
    }

    #[test]
    fn normal_growth_is_limited_by_fresh_availability() {
        let heap = heap(100, 20, Pressure::Normal);
        assert_eq!(
            heap.decide(MemoryNeed {
                minimum_bytes: 8,
                preferred_bytes: 32,
                class: HoldingClass::Surplus,
            })
            .unwrap(),
            MemoryAction::Grant { bytes: 20 }
        );
    }

    #[test]
    fn live_and_in_flight_holdings_cannot_be_released_directly() {
        let mut heap = heap(20, 0, Pressure::Normal);
        let id = heap.insert(20, HoldingClass::Live).unwrap();
        assert_eq!(
            heap.remove(id),
            Err(MemoryError::NotReleasable(HoldingClass::Live))
        );
    }

    #[test]
    fn claims_reserve_headroom_and_commit_only_observed_charge() {
        let mut heap = heap(80, 40, Pressure::Normal);
        let claim = heap
            .claim(MemoryNeed {
                minimum_bytes: 20,
                preferred_bytes: 30,
                class: HoldingClass::Dormant,
            })
            .unwrap();
        assert_eq!(heap.claims().count(), 1);
        assert_eq!(
            heap.decide(MemoryNeed {
                minimum_bytes: 8,
                preferred_bytes: 20,
                class: HoldingClass::Dormant,
            })
            .unwrap(),
            MemoryAction::Grant { bytes: 10 }
        );
        let holding = heap.commit_claim(claim.id, 24).unwrap();
        assert_eq!(heap.claims().count(), 0);
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
        let mut heap = heap(80, 40, Pressure::Normal);
        let model = heap.insert(80, HoldingClass::Model).unwrap();
        let peak = heap
            .claim(MemoryNeed {
                minimum_bytes: 30,
                preferred_bytes: 30,
                class: HoldingClass::Model,
            })
            .unwrap();
        assert_eq!(heap.claims().count(), 1);
        heap.observe(MemoryObservation {
            capacity_bytes: 120,
            available_bytes: 28,
            charged_bytes: 92,
            pressure: Pressure::Normal,
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

    #[test]
    fn rejection_reports_headroom_after_claims() {
        let mut heap = heap(80, 40, Pressure::Normal);
        heap.claim(MemoryNeed {
            minimum_bytes: 30,
            preferred_bytes: 30,
            class: HoldingClass::Dormant,
        })
        .unwrap();
        assert_eq!(
            heap.decide(MemoryNeed {
                minimum_bytes: 20,
                preferred_bytes: 20,
                class: HoldingClass::Live,
            })
            .unwrap(),
            MemoryAction::Reject {
                required: 20,
                available: 10,
            }
        );
    }
}
