//! The CUDA backend's description of its devices in the topology's terms
//! (`seismic_topology::CudaReport`): the memory it allocates, as kinds.

use crate::RangeMemory;
use seismic_lang::registry::BackendName;
use seismic_topology::{
    Contract, CudaMemory, CudaReport, Diagnostic, DiagnosticSource, HostReads, KindMemory, KindReport, Obtained,
    Reach, Reservation, ReservationChunks, Reserving, Unreservable,
};

/// Device memory: `cuMemAlloc` blocks, and VMM granules at the device.
pub const DEVICE_KIND: u32 = 0;
/// Pinned host memory (`cuMemHostAlloc`).
pub const PINNED_KIND: u32 = 1;
/// Memory of the host node (`cuMemCreate` at `HOST_NUMA`).
pub const HOST_NODE_KIND: u32 = 2;
/// Host pages registered for the device (`cuMemHostRegister`).
pub const REGISTERED_KIND: u32 = 3;

/// The kinds CUDA allocates on a device whose memory is the host's
/// (`in_host`) or its own: device memory, which the host cannot map, and
/// pinned host memory always; host-node memory on a device whose memory is
/// the host's, and registered host pages on one whose memory is its own,
/// where each is read to reserve. `created` is how memory CUDA creates
/// reserves, `registered` how registered host pages do.
pub fn kinds(in_host: bool, created: Reserving, registered: Reserving) -> Vec<KindReport> {
    let mapped = Reach::Mapped { reads: HostReads::Cached };
    let kind = |id, name: &str, reach, obtained, memory, reservation| KindReport { id, name: name.into(), reach, obtained, memory, reservation };
    let own = if in_host { Ok(None) } else { created.clone() };
    let mut kinds = vec![
        kind(DEVICE_KIND, "device memory", Reach::Device, Obtained::Allocated, KindMemory::DeviceMemory, own),
        kind(PINNED_KIND, "pinned host memory", mapped, Obtained::Allocated, KindMemory::HostRam, Ok(None)),
    ];
    if in_host && created != Ok(None) {
        kinds.push(kind(HOST_NODE_KIND, "host-node memory", mapped, Obtained::Allocated, KindMemory::HostRam, created));
    }
    if !in_host && registered != Ok(None) {
        kinds.push(kind(REGISTERED_KIND, "registered host pages", mapped, Obtained::ImportedHostPages, KindMemory::HostRam, registered));
    }
    kinds
}
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Every CUDA device's report, and which of the others each reads directly;
/// what cannot be read, as `diagnostics`.
pub fn describe(diagnostics: &mut Vec<Diagnostic>) -> Vec<CudaReport> {
    let diagnostic = |message: String| Diagnostic {
        source: DiagnosticSource::Backend(BackendName::Cuda),
        message,
    };
    let count = match crate::device_count() {
        Ok(0) => {
            diagnostics.push(Diagnostic::no_device(BackendName::Cuda));
            return Vec::new();
        }
        Ok(count) => count,
        Err(error) => {
            diagnostics.push(diagnostic(error.to_string()));
            return Vec::new();
        }
    };
    let mut described = Vec::new();
    for ordinal in 0..count {
        match crate::describe(ordinal) {
            Ok(descriptor) => described.push(descriptor),
            Err(error) => {
                diagnostics.push(diagnostic(format!("device ordinal {ordinal}: {error}")))
            }
        }
    }
    let mut reports = Vec::new();
    for descriptor in &described {
        let mut peers = Vec::new();
        for peer in &described {
            if peer.ordinal == descriptor.ordinal {
                continue;
            }
            match crate::can_access_peer(descriptor.ordinal, peer.ordinal) {
                Ok(true) => peers.push(peer.uuid),
                Ok(false) => {}
                Err(error) => diagnostics.push(diagnostic(format!(
                    "peer access of ordinal {} to {}: {error}",
                    descriptor.ordinal, peer.ordinal
                ))),
            }
        }
        let (ordinal, uuid) = (descriptor.ordinal, descriptor.uuid);
        // A memory whose granule cannot be read is reported unreservable,
        // with why.
        let in_host = descriptor.memory_is_host();
        let reserving = |memory: RangeMemory| {
            crate::reservation_granule(ordinal, memory)
                .map(|granule| {
                    granule.map(|granule| Reservation {
                        granule,
                        host_addressed: memory.host_addressed(),
                        chunks: crate::RESERVATION_CHUNKS.map(|unit| ReservationChunks { unit, table: Contract::DeviceOnly, whole: false }),
                    })
                })
                .map_err(|error| Unreservable::Unread(format!("its {memory:?} reservation granule: {error}")))
        };
        let created = reserving(if in_host { RangeMemory::HostNode } else { RangeMemory::Device });
        let registered = if in_host { Ok(None) } else { reserving(RangeMemory::Registered) };
        reports.push(CudaReport {
            uuid,
            luid: descriptor.luid,
            name: descriptor.name.clone(),
            driver: format!("cuda driver {}", descriptor.driver_api),
            admission: descriptor.admission().map_err(|error| error.to_string()),
            total: descriptor.total_memory_bytes,
            max_allocation: crate::max_allocation_bytes(descriptor.total_memory_bytes),
            integrated: descriptor.integrated,
            l2: (descriptor.l2_cache_bytes > 0).then_some(descriptor.l2_cache_bytes),
            memory: if descriptor.memory_is_host() {
                CudaMemory::Host
            } else {
                CudaMemory::Own {
                    usable: usable_at_first_read(uuid, || {
                        crate::usable_memory_bytes(ordinal, uuid)
                            .map_err(|error| error.to_string())
                    }),
                }
            },
            peers,
            kinds: kinds(in_host, created, registered),
        });
    }
    reports
}

/// The memory a discrete CUDA device lets this process use, as the driver
/// reports it once this process first creates the device's context
/// (`crate::usable_memory_bytes`), read once per process and
/// device: the first read precedes any allocation of this process, so a
/// later discovery (a refresh, or another inventory) keeps it rather than
/// reading the process's own holdings. A failed read is not kept.
fn usable_at_first_read(
    uuid: [u8; 16],
    read: impl FnOnce() -> Result<u64, String>,
) -> Result<u64, String> {
    static USABLE: OnceLock<Mutex<HashMap<[u8; 16], u64>>> = OnceLock::new();
    let mut usable = USABLE
        .get_or_init(Mutex::default)
        .lock()
        .expect("the usable-memory registry lock is never poisoned");
    if let Some(bytes) = usable.get(&uuid) {
        return Ok(*bytes);
    }
    let bytes = read()?;
    usable.insert(uuid, bytes);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::usable_at_first_read;
    use crate::RangeMemory;

    /// The first successful read stands for the process; after a failure it
    /// is read again.
    #[test]
    fn usable_memory_is_kept_for_the_process_and_a_failure_is_not() {
        let uuid = [0xa5; 16];
        let busy = || Err("busy".to_owned());
        assert_eq!(usable_at_first_read(uuid, busy), Err("busy".to_owned()));
        assert_eq!(usable_at_first_read(uuid, || Ok(6 << 30)), Ok(6 << 30));
        assert_eq!(usable_at_first_read(uuid, || Ok(1 << 30)), Ok(6 << 30));
    }

    /// The recorded fixtures' CUDA kinds are the ones this backend declares
    /// for their reservations.
    #[test]
    fn the_fixtures_kinds_are_the_backends() {
        let reserving = |kinds: &[seismic_topology::KindReport], id| {
            kinds.iter().find(|kind| kind.id == id).map_or(Ok(None), |kind| kind.reservation.clone())
        };
        let rtx3090 = seismic_topology::fixtures::rtx3090_cuda(seismic_topology::fixtures::rtx3090_uuid(), Vec::new());
        let (created, registered) = (reserving(&rtx3090.kinds, super::DEVICE_KIND), reserving(&rtx3090.kinds, super::REGISTERED_KIND));
        assert_eq!(super::kinds(false, created, registered), rtx3090.kinds);
        let gb10 = seismic_topology::fixtures::gb10().cuda.remove(0);
        assert_eq!(super::kinds(true, reserving(&gb10.kinds, super::HOST_NODE_KIND), Ok(None)), gb10.kinds);
    }

    /// On GB10, the recorded fixture's host-NUMA reservation granule is
    /// the one the driver reports for device storage there; elsewhere it checks
    /// nothing.
    #[test]
    fn the_gb10_fixtures_host_granule_is_the_drivers() {
        let Some((_exclusive, _opened)) = crate::test_device() else {
            return;
        };
        let descriptor = crate::describe(0).expect("an opened device describes");
        if descriptor.uuid != seismic_topology::fixtures::GB10_UUID {
            return;
        }
        let memory = RangeMemory::HostNode;
        let granule = crate::reservation_granule(0, memory).expect("GB10's granule reads");
        eprintln!("GB10 {memory:?} reservation granule: {granule:?}");
        assert_eq!(granule, Some(seismic_topology::fixtures::GB10_HOST_GRANULE));
    }
}
