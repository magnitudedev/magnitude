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
    /// Physical host RAM budget on unified Metal where `device` is bounded by
    /// the wired GPU working-set ceiling (`HostGrant::Wired`). Enforces
    /// `device + host <= unified_host_ram` beside `device <= self.device` so
    /// unwired Rust host heap (`host_bytes`) is charged against host RAM
    /// supply rather than Metal's wired ceiling.
    pub unified_host_ram: Option<StoreBudget>,
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
        let unified_host_ram = (unified_host && at.device.wired_limit().is_some()).then(|| {
            let host_available = at.device.host_room().saturating_add(held);
            StoreBudget {
                store: Some(host_ram),
                role: DomainRole::Allocation,
                capacity: at.device.host_capacity().max(host_available),
                available: host_available,
            }
        });
        Self { device, host, unified_host_ram }
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
            unified_host_ram: None,
        }
    }

    /// Attach a physical host RAM budget for unified Metal tests where the
    /// wired GPU ceiling (`self.device`) lies below host RAM supply.
    pub fn with_unified_host_ram(mut self, bytes: u64) -> Self {
        self.unified_host_ram = Some(StoreBudget {
            store: self.device.store,
            role: DomainRole::Allocation,
            capacity: bytes,
            available: bytes,
        });
        self
    }

    /// Credit the worker process's already-resident untracked host footprint
    /// (`phys_footprint - ledger_allocated`) back onto the host-RAM budget
    /// (clamped to `capacity`). At live load (`device opened`), OS `supply`
    /// already excludes the worker's own resident host memory, while
    /// `AssessmentMemoryCharge::host_bytes()` accounts for the worker's total
    /// host memory rather than only its post-open delta.
    ///
    /// - On wired-ceiling unified Metal (`self.unified_host_ram.is_some()`),
    ///   credits `unified_host_ram.available` without altering the Metal wired
    ///   ceiling (`self.device`).
    /// - On unwired unified memory (`self.host.is_none()` and
    ///   `self.unified_host_ram.is_none()`, e.g. Linux/Windows CUDA or Vulkan
    ///   UMA), `self.device` IS host RAM, so credits `self.device.available`.
    /// - On split discrete GPUs (`self.host.is_some()`), leaves `self.device`
    ///   (VRAM) untouched.
    pub fn credit_own_host_footprint(mut self, own_untracked: u64) -> Self {
        if let Some(ram) = self.unified_host_ram.as_mut() {
            ram.available = ram
                .available
                .saturating_add(own_untracked)
                .min(ram.capacity);
        } else if self.host.is_none() {
            self.device.available = self
                .device
                .available
                .saturating_add(own_untracked)
                .min(self.device.capacity);
        }
        self
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
        let unified_host_ram = if unified_host && executor.backend() == seismic::BackendName::Metal {
            let host_fit = fit
                .store(host_ram)
                .map_err(|error| PlanError::ResourcePlanning(error.to_string()))?;
            let host_available = match mode {
                FitMode::Stable => host_fit.capacity_bytes,
                FitMode::Live => host_fit.headroom_bytes,
            };
            Some(StoreBudget {
                store: Some(host_ram),
                role: DomainRole::Allocation,
                capacity: host_fit.capacity_bytes,
                available: host_available,
            })
        } else {
            None
        };
        Ok(Self { device, host, unified_host_ram })
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
            StoreDemands::Unified { device, host } => {
                let store = self.device.store.unwrap_or(host_ram);
                let total = device.saturating_add(host);
                match self.unified_host_ram {
                    Some(ram) => {
                        let wired_margin = i128::from(self.device.available) - i128::from(device);
                        let ram_margin = i128::from(ram.available) - i128::from(total);
                        if wired_margin <= ram_margin {
                            vec![domain_fit(self.device, store, device)?]
                        } else {
                            vec![domain_fit(ram, store, total)?]
                        }
                    }
                    None => vec![domain_fit(self.device, store, total)?],
                }
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
        let demands = mandatory
            .store_demands(draft.device().topology(), draft.device().executor())
            .map_err(PlanError::ResourcePlanning)?;
        Ok(match demands {
            StoreDemands::Unified { device, host } => match self.unified_host_ram {
                Some(ram) => self
                    .device
                    .available
                    .saturating_sub(device)
                    .min(ram.available.saturating_sub(device.saturating_add(host))),
                None => self.device.available.saturating_sub(device.saturating_add(host)),
            },
            StoreDemands::Split { device, .. } => self.device.available.saturating_sub(device),
        })
    }

    /// Whether live available headroom is strictly below structural capacity
    /// on any store (`MemoryBudget::stable` always has `available == capacity`).
    pub fn live_below_capacity(&self) -> bool {
        self.device.available < self.device.capacity
            || self.host.is_some_and(|h| h.available < h.capacity)
            || self.unified_host_ram.is_some_and(|h| h.available < h.capacity)
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
            StoreDemands::Unified { device, host } => match self.unified_host_ram {
                Some(ram) => {
                    device <= self.device.capacity
                        && device.saturating_add(host) <= ram.capacity
                }
                None => device.saturating_add(host) <= self.device.capacity,
            },
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
            StoreDemands::Unified { device, host } => {
                let total = device.saturating_add(host);
                match self.unified_host_ram {
                    Some(ram) => {
                        let wired_short = device.saturating_sub(device_avail);
                        let ram_short = total.saturating_sub(ram.available);
                        if wired_short >= ram_short {
                            let short = NonZeroU64::new(wired_short);
                            WholeFit {
                                available: device_avail,
                                needed: device,
                                short,
                                limiting: short.map(|_| DomainRole::Allocation),
                                limiting_required: device,
                                limiting_available: device_avail,
                            }
                        } else {
                            let short = NonZeroU64::new(ram_short);
                            WholeFit {
                                available: ram.available,
                                needed: total,
                                short,
                                limiting: short.map(|_| DomainRole::Allocation),
                                limiting_required: total,
                                limiting_available: ram.available,
                            }
                        }
                    }
                    None => {
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

    /// `(available, required)` in host RAM beside `charge`'s host store demand
    /// plus `growth` bytes, or `None` on unified stores.
    fn host_available_and_required(
        &self,
        draft: &ExecutionPlanDraft,
        charge: &AssessmentMemoryCharge,
        growth: u64,
    ) -> Result<Option<(u64, u64)>, PlanError> {
        let Some(host) = self.host else {
            return Ok(None);
        };
        let host_demand = charge
            .store_demands(draft.device().topology(), draft.device().executor())
            .map_err(PlanError::ResourcePlanning)?
            .host_store_bytes()
            .expect("invariant violated: a linked host tier requires split device and host stores");
        let required = host_demand
            .checked_add(growth)
            .ok_or(PlanError::Arithmetic("a load's host charge overflows"))?;
        Ok(Some((host.available, required)))
    }

    /// Host RAM left beside `charge`'s host store demand plus `growth` bytes
    /// (for sizing a linked host tier): negative where it does not fit.
    pub fn host_left(
        &self,
        draft: &ExecutionPlanDraft,
        charge: &AssessmentMemoryCharge,
        growth: u64,
    ) -> Result<i64, PlanError> {
        let Some((available, required)) = self.host_available_and_required(draft, charge, growth)? else {
            return Ok(i64::MAX);
        };
        i64::try_from(available)
            .and_then(|room| i64::try_from(required).map(|demand| room - demand))
            .map_err(|_| PlanError::Arithmetic("a host tier's room exceeds i64"))
    }

    /// Host-RAM shortfall beside `charge`'s host store demand plus `growth`
    /// bytes: `Some((required, available, deficit_bytes))` when `required`
    /// exceeds `host.available`, or `None` when it fits (or on unified stores).
    pub fn host_shortfall(
        &self,
        draft: &ExecutionPlanDraft,
        charge: &AssessmentMemoryCharge,
        growth: u64,
    ) -> Result<Option<(u64, u64, NonZeroU64)>, PlanError> {
        let Some((available, required)) = self.host_available_and_required(draft, charge, growth)? else {
            return Ok(None);
        };
        Ok(NonZeroU64::new(required.saturating_sub(available))
            .map(|deficit| (required, available, deficit)))
    }
}
