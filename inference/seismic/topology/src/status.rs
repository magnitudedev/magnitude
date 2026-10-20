//! Status (spec §2.5): each store's headroom for this process and what is
//! left of each grant, in backend-neutral types, from two sources. A
//! [`Sample`] holds what needs no opened executor; a [`RuntimeStatus`] what
//! only an opened executor knows, which the runtime reports. [`Status`]
//! combines both, the input fit reads. Each driver figure's meaning is
//! decided here, beside the reading.

use crate::discovery::Natives;
use crate::host::{self, DisplacementHistory, HostStatus, ObservationError};
use crate::model::{
    AccessPathId, DeviceTopology, GrantSource, Reported, ReportedBy, Store, StoreId, StoreKind,
};
use std::collections::HashMap;
use std::time::SystemTime;

/// One timestamped reading of what needs no opened executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// When the sample was taken.
    pub taken: SystemTime,
    /// Host RAM: the OS's availability, the process limits that bound it,
    /// and distress.
    pub host_ram: Result<HostStatus, ObservationError>,
    /// Each volume's space available to this process.
    pub volumes: Vec<(StoreId, Result<Reported<u64>, ObservationError>)>,
    /// What is left of each grant readable without an opened executor:
    /// Vulkan heap budgets less usage, Metal's working set less allocated.
    pub grants: Vec<(AccessPathId, Result<Reported<u64>, ObservationError>)>,
}

/// What only opened executors know, as the runtime reports it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeStatus {
    /// Each store's ledger: the bytes this process's executors have charged
    /// to it. A store absent here has no charge.
    pub charges: Vec<(StoreId, u64)>,
    /// What is left of each grant only an opened executor reads: a discrete
    /// CUDA device's free memory (`ReportedBy::CudaFree`), and an
    /// integrated CUDA device's grant less what the executor holds.
    pub grants: Vec<(AccessPathId, Result<Reported<u64>, ObservationError>)>,
}

/// Status from both sources: the input of fit. For an unopened executor the
/// runtime reports nothing, and fit plans on stable fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Status {
    /// What needs no opened executor.
    pub sample: Sample,
    /// What only the opened executors know.
    pub runtime: RuntimeStatus,
}

impl Status {
    /// The bytes this process has charged to `store`.
    pub fn charged(&self, store: StoreId) -> u64 {
        self.runtime
            .charges
            .iter()
            .filter(|(charged, _)| *charged == store)
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// What `store` has left for this process: host RAM's availability as
    /// the OS reports it, within every process limit; a volume's space
    /// available to this process; a device memory store's capacity less the
    /// process's charges, which the OS does not report.
    pub fn headroom(&self, store: &Store) -> Result<u64, ObservationError> {
        match store.kind() {
            StoreKind::HostRam => self
                .sample
                .host_ram
                .as_ref()
                .map(HostStatus::available)
                .map_err(Clone::clone),
            StoreKind::Volume(_) => self
                .sample
                .volumes
                .iter()
                .find(|(volume, _)| *volume == store.id())
                .ok_or_else(|| {
                    ObservationError::Failed("the sample is of another topology revision".into())
                })?
                .1
                .as_ref()
                .map(|available| available.value)
                .map_err(Clone::clone),
            StoreKind::DeviceMemory(_) => Ok(store
                .capacity()
                .value
                .saturating_sub(self.charged(store.id()))),
        }
    }

    /// What is left of `path`'s grant: the runtime's reading where an
    /// opened executor gave one, else the sample's. `None` when neither
    /// source read it.
    pub fn grant_left(
        &self,
        path: AccessPathId,
    ) -> Option<Result<Reported<u64>, &ObservationError>> {
        if let Some((_, left)) = self.runtime.grants.iter().find(|(of, _)| *of == path) {
            return Some(left.as_ref().copied());
        }
        self.sample
            .grants
            .iter()
            .find(|(of, _)| *of == path)
            .map(|(_, left)| left.as_ref().copied())
    }
}

pub(crate) fn sample(
    topology: &DeviceTopology,
    natives: &Natives,
    history: &mut DisplacementHistory,
) -> Sample {
    let taken = SystemTime::now();
    let host_ram = host::status(history);
    let volumes = topology
        .stores()
        .iter()
        .filter(|store| matches!(store.kind(), StoreKind::Volume(_)))
        .map(|store| {
            let path = topology
                .volume_paths(store.id())
                .next()
                .expect("a volume store is on a served path");
            (store.id(), crate::volume::volume_available(path))
        })
        .collect();
    let mut budgets = HashMap::new();
    let grants = topology
        .grant_sources()
        .map(|(path, grant, source)| (path, grant_left(source, grant, natives, &mut budgets)))
        .collect();
    Sample {
        taken,
        host_ram,
        volumes,
        grants,
    }
}

/// Heap budgets of each Vulkan device, read once per sample.
type Budgets = HashMap<[u8; 16], Result<Vec<(u64, u64)>, ObservationError>>;

/// What is left of a grant of `grant` bytes whose status `source` reads.
fn grant_left(
    source: &GrantSource,
    grant: u64,
    natives: &Natives,
    budgets: &mut Budgets,
) -> Result<Reported<u64>, ObservationError> {
    match source {
        // The heaps' budgets less their usage, of which the grant takes the
        // usage beyond what paths ahead of it hold, up to its size.
        GrantSource::VulkanHeaps {
            uuid,
            heaps,
            before,
        } => {
            let device = budgets
                .entry(*uuid)
                .or_insert_with(|| vulkan_budgets(*uuid))
                .as_ref()
                .map_err(Clone::clone)?;
            if heaps.iter().any(|&heap| heap >= device.len()) {
                return Err(ObservationError::Failed(
                    "the Vulkan heap layout changed; refresh the inventory".into(),
                ));
            }
            Ok(Reported::new(
                attributed(device, heaps, *before, grant),
                ReportedBy::VulkanHeapBudget,
            ))
        }
        GrantSource::MetalWorkingSet {
            registry_id,
            before,
        } => metal_left(*registry_id, *before, grant, natives),
    }
}

/// What is left of a grant of `grant` bytes on `heaps` of a device whose
/// heaps report `(budget, usage)` each: the heaps' usage is attributed along
/// the placement, `before` bytes to the paths ahead of this one first, and
/// what is left is bounded by the heaps' budgets less their usage.
fn attributed(device: &[(u64, u64)], heaps: &[usize], before: u64, grant: u64) -> u64 {
    let (budget, usage) = heaps
        .iter()
        .map(|&heap| device[heap])
        .fold((0, 0), |(budget, usage), heap| {
            (budget + heap.0, usage + heap.1)
        });
    budget.min(before + grant).saturating_sub(usage.max(before))
}

#[cfg(not(target_os = "macos"))]
fn vulkan_budgets(uuid: [u8; 16]) -> Result<Vec<(u64, u64)>, ObservationError> {
    match seismic_vulkan::heap_budgets(uuid) {
        Ok(Some(budgets)) => Ok(budgets
            .into_iter()
            .map(|heap| (heap.budget, heap.usage))
            .collect()),
        Ok(None) => Err(ObservationError::Failed(
            "the Vulkan device is gone; refresh the inventory".into(),
        )),
        Err(error) => Err(ObservationError::Failed(error.to_string())),
    }
}

/// Vulkan is not built on macOS, so no topology there holds a Vulkan grant.
#[cfg(target_os = "macos")]
fn vulkan_budgets(_uuid: [u8; 16]) -> Result<Vec<(u64, u64)>, ObservationError> {
    Err(ObservationError::Unsupported(
        "this build has no Vulkan runtime".into(),
    ))
}

/// The Metal device's recommended working set less what this process has
/// allocated on it, attributed along the placement as heap usage is.
#[cfg(target_os = "macos")]
fn metal_left(
    registry_id: u64,
    before: u64,
    grant: u64,
    natives: &Natives,
) -> Result<Reported<u64>, ObservationError> {
    let handle = natives
        .metal
        .iter()
        .find(|handle| handle.registry_id() == registry_id)
        .ok_or_else(|| ObservationError::Failed(format!("Metal device {registry_id:x} is gone")))?;
    let working_set = [(
        handle.recommended_working_set_bytes(),
        handle.current_allocated_bytes(),
    )];
    Ok(Reported::new(
        attributed(&working_set, &[0], before, grant),
        ReportedBy::MetalWorkingSetUnallocated,
    ))
}

/// Metal exists on macOS only, so no topology elsewhere holds a Metal grant.
#[cfg(not(target_os = "macos"))]
fn metal_left(
    _registry_id: u64,
    _before: u64,
    _grant: u64,
    _natives: &Natives,
) -> Result<Reported<u64>, ObservationError> {
    Err(ObservationError::Unsupported(
        "this build has no Metal runtime".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::attributed;

    /// A device whose memory is the host's, with 96 of its heaps' 110 bytes
    /// its own memory: usage is its own memory's first, the host path's
    /// beyond it, and neither path has more left than the budgets.
    #[test]
    fn heap_usage_is_attributed_to_own_memory_first() {
        let heaps = [(70, 30), (35, 0)];
        let all = [0, 1];
        // 30 used: all of it own memory's.
        assert_eq!(attributed(&heaps, &all, 0, 96), 66.min(105 - 30));
        assert_eq!(attributed(&heaps, &all, 96, 14), 9);
        assert_eq!(
            attributed(&heaps, &all, 0, 96) + attributed(&heaps, &all, 96, 14),
            75
        );
        let heaps = [(70, 70), (35, 30)];
        // 100 used: own memory full, 4 of the host path's 14.
        assert_eq!(attributed(&heaps, &all, 0, 96), 0);
        assert_eq!(attributed(&heaps, &all, 96, 14), 5.min(10));
        // A discrete device's one heap.
        assert_eq!(attributed(&[(24, 20), (64, 0)], &[0], 0, 24), 4);
        // A reduced driver budget belongs to the same ordered placement;
        // each path must not independently claim the remaining total.
        let heaps = [(35, 20), (15, 0)];
        assert_eq!(attributed(&heaps, &all, 0, 96), 30);
        assert_eq!(attributed(&heaps, &all, 96, 14), 0);
    }
}
