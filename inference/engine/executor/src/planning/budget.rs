//! The single memory budget owner for static catalog assessment, device-free
//! load preview, and opened-device planning / loading, together with the
//! load's once-decided [`PlanForm`].
//!
//! Every fit, room, and host-left calculation goes exclusively through
//! [`AssessmentMemoryCharge::store_demands`] against one [`MemoryBudget`]:
//! - [`MemoryBudget::stable`]: structural capacity (`available == capacity`),
//!   used only by cached static catalog assessment.
//! - [`MemoryBudget::live_status`]: live headroom from a `seismic::Status`
//!   snapshot (`FitMode::Live`), used by device-free load preview.
//! - [`MemoryBudget::live_ledger`]: live headroom from an opened device's
//!   settled `DeviceReading` and `DeviceLedgers` (`room_after` + `held`), used
//!   by opened-device Prepare and Load.
//! - [`MemoryBudget::fixed`]: synthetic budget for unit tests.

use super::{AssessmentFit, AssessmentFitVerdict, AssessmentMemoryCharge, ExecutionPlanDraft, PlannedDevice, StoreDemands};
use crate::assessment::DomainFit;
use crate::memory::ledger::{DeviceLedgers, DeviceReading};
use crate::platform::{DomainRole, Fit, FitMode, SelectedDevice};
use crate::PlanError;
use seismic::{Contract, DeviceTopology, Executor, Status, StoreId};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

/// Whether a load holds its model's experts whole-resident or streams them
/// through an expert bank.
///
/// Decided once at load preview against [`MemoryBudget::live_status`], stamped
/// onto `ExecutionManifest`, included in `prepared_key`, and honored without
/// toggling across Prepare and Load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlanForm {
    /// Experts held whole (either in a whole bank or as resident weights).
    Whole,
    /// Experts streamed through an expert bank sized to live memory.
    Streamed,
}

/// Memory budget for one physical store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreBudget {
    /// Store identity in the topology (`None` only for synthetic
    /// [`MemoryBudget::fixed`] budgets).
    pub store: Option<StoreId>,
    pub role: DomainRole,
    pub capacity: u64,
    pub available: u64,
}

/// A whole plan's fit against a [`MemoryBudget`]: the device allocation
/// store's available room, what the plan requires there, and how far the plan
/// is short and in which store role (with that limiting store's exact
/// `required` and `available`), if it does not fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WholeFit {
    /// Device allocation store available room.
    pub available: u64,
    /// Device allocation store bytes required by the plan.
    pub needed: u64,
    /// Deficit in the limiting store, or `None` when the plan fits.
    pub short: Option<NonZeroU64>,
    /// Role of the limiting store when `short` is `Some`.
    pub limiting: Option<DomainRole>,
    /// Required bytes in the limiting store (or device store when fitting).
    pub limiting_required: u64,
    /// Available bytes in the limiting store (or device store when fitting).
    pub limiting_available: u64,
}

/// The single memory budget owner shared by static assessment, device-free
/// load preview, and opened-device planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// The executor's device allocation store. On unified memory (Metal, CPU),
    /// host RAM is this store and `host` is `None`.
    pub device: StoreBudget,
    /// Separate host RAM store on split (discrete GPU) topologies; `None` on
    /// unified memory.
    pub host: Option<StoreBudget>,
}

impl MemoryBudget {
    /// Static capacity budget (`FitMode::Stable`, `available == capacity`).
    ///
    /// Used only by cached static catalog assessment.
    pub fn stable(
        topology: &DeviceTopology,
        executor: &Executor,
        status: &Status,
    ) -> Result<Self, PlanError> {
        Self::from_fit(topology, executor, status, FitMode::Stable)
    }

    /// Static capacity budget for a planned device at its selection reading.
    pub fn of_stable(device: &PlannedDevice) -> Result<Self, PlanError> {
        Self::stable(device.topology(), device.executor(), device.status())
    }

    /// Static capacity budget for a selected device at its selection reading.
    pub fn of_selected_stable(selected: &SelectedDevice) -> Result<Self, PlanError> {
        Self::stable(&selected.reading.topology, &selected.info, &selected.reading.status)
    }

    /// Live headroom budget from a device-free `Status` snapshot
    /// (`FitMode::Live`: `host.supply.total()` bounded by `grant_left`).
    ///
    /// Used by device-free Load Preview.
    pub fn live_status(
        topology: &DeviceTopology,
        executor: &Executor,
        status: &Status,
    ) -> Result<Self, PlanError> {
        Self::from_fit(topology, executor, status, FitMode::Live)
    }

    /// Live headroom budget for a selected device at its selection reading.
    pub fn of_selected_live(selected: &SelectedDevice) -> Result<Self, PlanError> {
        Self::live_status(&selected.reading.topology, &selected.info, &selected.reading.status)
    }

    /// Live headroom budget from an opened device's settled `DeviceReading`
    /// and `DeviceLedgers`: `reading.room_after(ledgers).device.room() + held`
    /// (and host room on split GPUs).
    ///
    /// `held` is device bytes the load already holds that the charges it prices
    /// count too (e.g. `AttestedPrograms::DEVICE_BYTES` during `finalize_readout`).
    pub fn live_ledger(
        topology: &DeviceTopology,
        executor: &Executor,
        reading: &DeviceReading,
        ledgers: &DeviceLedgers,
        held: u64,
    ) -> Self {
        let at = reading.room_after(ledgers);
        let host_ram = topology.host_ram().id();
        let unified_host = DomainRole::of(topology, executor, host_ram) == DomainRole::Allocation;
        let device_path = (!unified_host).then(|| {
            executor
                .paths(Contract::DeviceOnly)
                .first()
                .and_then(|id| topology.access_path(*id))
                .expect("invariant violated: executor has no device storage path")
        });
        let device_store = device_path.map_or(host_ram, |path| path.store());
        let device_available = at.device.room().saturating_add(held);
        let device_capacity = match device_path.and_then(|path| path.grant()) {
            Some(grant) => at.device.capacity().min(grant.bytes.value).max(device_available),
            None => at.device.capacity(),
        };
        let device = StoreBudget {
            store: Some(device_store),
            role: DomainRole::Allocation,
            capacity: device_capacity,
            available: device_available,
        };
        let host = if unified_host {
            None
        } else {
            let h = at
                .host
                .as_ref()
                .expect("invariant violated: split store demands require a host room reading");
            Some(StoreBudget {
                store: Some(host_ram),
                role: DomainRole::Staging,
                capacity: h.capacity(),
                available: h.pinned_room(),
            })
        };
        Self { device, host }
    }

    /// Synthetic budget with fixed device (and optional host) bytes, for unit
    /// tests.
    pub fn fixed(device: u64, host: Option<u64>) -> Self {
        Self {
            device: StoreBudget {
                store: None,
                role: DomainRole::Allocation,
                capacity: device,
                available: device,
            },
            host: host.map(|bytes| StoreBudget {
                store: None,
                role: DomainRole::Staging,
                capacity: bytes,
                available: bytes,
            }),
        }
    }

    /// Structural capacity of the device allocation store.
    pub fn capacity(&self) -> u64 {
        self.device.capacity
    }

    /// Available bytes in the device allocation store.
    pub fn available(&self) -> u64 {
        self.device.available
    }

    fn from_fit(
        topology: &DeviceTopology,
        executor: &Executor,
        status: &Status,
        mode: FitMode,
    ) -> Result<Self, PlanError> {
        let fit = Fit::new(topology, status);
        let host_ram = topology.host_ram().id();
        let unified_host = DomainRole::of(topology, executor, host_ram) == DomainRole::Allocation;
        let device_placement = fit
            .placement(executor, Contract::DeviceOnly)
            .map_err(|error| PlanError::ResourcePlanning(error.to_string()))?;
        let device_store = if unified_host {
            host_ram
        } else {
            executor
                .paths(Contract::DeviceOnly)
                .first()
                .and_then(|id| topology.access_path(*id))
                .map(|path| path.store())
                .ok_or_else(|| {
                    PlanError::ResourcePlanning("executor has no device storage path".into())
                })?
        };
        let wired_reserve = if unified_host && executor.backend() == seismic::BackendName::Metal {
            crate::memory::supply::Ceiling::execution_reserve_for(device_placement.stable_bytes)
        } else {
            0
        };
        let device_capacity = device_placement.stable_bytes.saturating_sub(wired_reserve);
        let device_available = match mode {
            FitMode::Stable => device_capacity,
            FitMode::Live => {
                let mut live = device_placement.ceiling_bytes.min(device_capacity);
                if unified_host && executor.backend() == seismic::BackendName::Metal {
                    if let Some(wired) = status.sample.host.wired {
                        let limit = match wired.limit {
                            seismic::WiredLimit::OsDefault => device_placement.stable_bytes,
                            seismic::WiredLimit::Set(bytes) => bytes.min(device_placement.stable_bytes),
                        };
                        let reserve = crate::memory::supply::Ceiling::execution_reserve_for(limit);
                        live = live.min(limit.saturating_sub(wired.wire_count).saturating_sub(reserve));
                    }
                }
                live
            }
        };
        let device = StoreBudget {
            store: Some(device_store),
            role: DomainRole::Allocation,
            capacity: device_capacity,
            available: device_available,
        };
        let host = if unified_host {
            None
        } else {
            let host_fit = fit
                .store(host_ram)
                .map_err(|error| PlanError::ResourcePlanning(error.to_string()))?;
            let host_available = match mode {
                FitMode::Stable => host_fit.capacity_bytes,
                FitMode::Live => crate::memory::supply::Reading::host(&status.sample.host).pinned_headroom(0, 0),
            };
            Some(StoreBudget {
                store: Some(host_ram),
                role: DomainRole::Staging,
                capacity: host_fit.capacity_bytes,
                available: host_available,
            })
        };
        Ok(Self { device, host })
    }

    /// Full per-store fit of `charge` against this budget via the single
    /// [`AssessmentMemoryCharge::store_demands`] decomposition.
    pub fn fit(
        &self,
        topology: &DeviceTopology,
        executor: &Executor,
        charge: &AssessmentMemoryCharge,
    ) -> Result<AssessmentFit, PlanError> {
        let demands = charge
            .store_demands(topology, executor)
            .map_err(PlanError::ResourcePlanning)?;
        let domain_fit = |budget: StoreBudget, store: StoreId, required: u64| -> Result<DomainFit, PlanError> {
            let remaining_bytes = if required <= budget.available {
                i64::try_from(budget.available - required)
                    .map_err(|_| PlanError::Arithmetic("fit remainder exceeds i64"))?
            } else {
                -i64::try_from(required - budget.available)
                    .map_err(|_| PlanError::Arithmetic("fit deficit exceeds i64"))?
            };
            Ok(DomainFit {
                role: budget.role,
                domain: store,
                capacity_bytes: budget.available,
                required_bytes: required,
                remaining_bytes,
            })
        };
        let host_ram = topology.host_ram().id();
        let domains = match demands {
            StoreDemands::Unified { total } => {
                let store = self.device.store.unwrap_or(host_ram);
                vec![domain_fit(self.device, store, total)?]
            }
            StoreDemands::Split { device, host } => {
                let device_store = match self.device.store {
                    Some(store) => store,
                    None => executor
                        .paths(Contract::DeviceOnly)
                        .first()
                        .and_then(|id| topology.access_path(*id))
                        .map(|path| path.store())
                        .ok_or_else(|| {
                            PlanError::ResourcePlanning("executor has no device storage path".into())
                        })?,
                };
                let mut domains = vec![domain_fit(self.device, device_store, device)?];
                if let Some(host_budget) = self.host {
                    let host_store = host_budget.store.unwrap_or(host_ram);
                    domains.push(domain_fit(host_budget, host_store, host)?);
                }
                domains
            }
        };
        let verdict = domains
            .iter()
            .filter_map(|domain| {
                Some((
                    domain.domain,
                    NonZeroU64::new(domain.remaining_bytes.min(0).unsigned_abs())?,
                ))
            })
            .max_by_key(|&(_, deficit_bytes)| deficit_bytes)
            .map_or(AssessmentFitVerdict::Fits, |(limiting, deficit_bytes)| {
                AssessmentFitVerdict::DoesNotFit {
                    limiting,
                    deficit_bytes,
                }
            });
        Ok(AssessmentFit { verdict, domains })
    }

    /// Bytes left in the device allocation store beside `draft`'s mandatory
    /// part (`mandatory`).
    pub fn room(
        &self,
        draft: &ExecutionPlanDraft,
        mandatory: &AssessmentMemoryCharge,
    ) -> Result<u64, PlanError> {
        let needed = mandatory
            .store_demands(draft.device().topology(), draft.device().executor())
            .map_err(PlanError::ResourcePlanning)?
            .allocation_bytes();
        Ok(self.device.available.saturating_sub(needed))
    }

    /// Whether live available headroom is strictly below structural capacity
    /// on any store (`MemoryBudget::stable` always has `available == capacity`).
    pub fn live_below_capacity(&self) -> bool {
        self.device.available < self.device.capacity
            || self.host.is_some_and(|h| h.available < h.capacity)
    }

    /// Whether `charge`'s full served-context workload fits every store's
    /// structural `capacity`, independently of momentary live headroom.
    pub fn fits_capacity(
        &self,
        draft: &ExecutionPlanDraft,
        charge: &AssessmentMemoryCharge,
    ) -> Result<bool, PlanError> {
        let demands = charge
            .store_demands(draft.device().topology(), draft.device().executor())
            .map_err(PlanError::ResourcePlanning)?;
        Ok(match demands {
            StoreDemands::Unified { total } => total <= self.device.capacity,
            StoreDemands::Split { device, host } => {
                device <= self.device.capacity && self.host.is_none_or(|h| host <= h.capacity)
            }
        })
    }

    /// Fit of `draft` charged `charge` as a whole plan against this budget.
    pub fn fit_whole(
        &self,
        draft: &ExecutionPlanDraft,
        charge: &AssessmentMemoryCharge,
    ) -> Result<WholeFit, PlanError> {
        let demands = charge
            .store_demands(draft.device().topology(), draft.device().executor())
            .map_err(PlanError::ResourcePlanning)?;
        Ok(self.fit_demands(demands))
    }

    /// Fit of `draft`'s actual startup peak (`AssessmentMemoryCharge::startup_store_demands`,
    /// excluding elastic KV history allocated on demand after load) against
    /// this budget's live available headroom.
    pub fn fit_startup(
        &self,
        draft: &ExecutionPlanDraft,
        charge: &AssessmentMemoryCharge,
    ) -> Result<WholeFit, PlanError> {
        let demands = charge
            .startup_store_demands(draft.device().topology(), draft.device().executor())
            .map_err(PlanError::ResourcePlanning)?;
        Ok(self.fit_demands(demands))
    }

    fn fit_demands(&self, demands: StoreDemands) -> WholeFit {
        let device_avail = self.device.available;
        match demands {
            StoreDemands::Unified { total } => {
                let short = NonZeroU64::new(total.saturating_sub(device_avail));
                WholeFit {
                    available: device_avail,
                    needed: total,
                    short,
                    limiting: short.map(|_| DomainRole::Allocation),
                    limiting_required: total,
                    limiting_available: device_avail,
                }
            }
            StoreDemands::Split { device, host } => {
                let device_short = device.saturating_sub(device_avail);
                let (host_avail, host_short) = match self.host {
                    Some(h) => (h.available, host.saturating_sub(h.available)),
                    None => (u64::MAX, 0),
                };
                if device_short >= host_short {
                    let short = NonZeroU64::new(device_short);
                    WholeFit {
                        available: device_avail,
                        needed: device,
                        short,
                        limiting: short.map(|_| DomainRole::Allocation),
                        limiting_required: device,
                        limiting_available: device_avail,
                    }
                } else {
                    let short = NonZeroU64::new(host_short);
                    WholeFit {
                        available: device_avail,
                        needed: device,
                        short,
                        limiting: short.map(|_| DomainRole::Staging),
                        limiting_required: host,
                        limiting_available: host_avail,
                    }
                }
            }
        }
    }

    /// Host RAM left beside `charge`'s host store demand plus `growth` bytes
    /// (for sizing a linked host tier): negative where it does not fit.
    pub fn host_left(
        &self,
        draft: &ExecutionPlanDraft,
        charge: &AssessmentMemoryCharge,
        growth: u64,
    ) -> Result<i64, PlanError> {
        let Some(host) = self.host else {
            return Ok(i64::MAX);
        };
        let host_demand = charge
            .store_demands(draft.device().topology(), draft.device().executor())
            .map_err(PlanError::ResourcePlanning)?
            .host_store_bytes()
            .expect("invariant violated: a linked host tier requires split device and host stores");
        let demand = host_demand
            .checked_add(growth)
            .ok_or(PlanError::Arithmetic("a load's host charge overflows"))?;
        i64::try_from(host.available)
            .and_then(|room| i64::try_from(demand).map(|demand| room - demand))
            .map_err(|_| PlanError::Arithmetic("a host tier's room exceeds i64"))
    }
}
