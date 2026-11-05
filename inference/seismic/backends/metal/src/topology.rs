//! The Metal backend's description of its devices in the topology's terms
//! (`seismic_topology::MetalReport`).

use crate::DeviceHandle;
use seismic_lang::registry::BackendName;
use seismic_topology::{
    Contract, Diagnostic, HostReads, KindMemory, KindReport, MetalReport, Obtained, Reach, Reservation,
    ReservationChunks,
};

/// The backend's one kind: shared storage mode, which every contract is
/// allocated in, in host RAM on a device whose memory is the host's.
pub fn kind() -> KindReport {
    KindReport {
        id: 0,
        name: "shared storage".into(),
        reach: Reach::Mapped { reads: HostReads::Cached },
        obtained: Obtained::Allocated,
        memory: KindMemory::HostRam,
        reservation: Ok(Some(Reservation {
            granule: crate::device::RESERVATION_GRANULE as u64,
            host_addressed: crate::device::RESERVATIONS_HOST_ADDRESSED,
            chunks: crate::device::RESERVATION_CHUNKS.map(|unit| ReservationChunks { unit, table: Contract::DeviceOnly }),
        })),
    }
}

/// Every Metal device's report, with its handle, which status samples
/// read; that there is none, as `diagnostics`. A device's language and
/// tensor-operation probes compile.
pub fn describe(diagnostics: &mut Vec<Diagnostic>) -> Vec<(MetalReport, DeviceHandle)> {
    let handles = DeviceHandle::discover();
    if handles.is_empty() {
        diagnostics.push(Diagnostic::no_device(BackendName::Metal));
    }
    handles
        .into_iter()
        .map(|handle| {
            let admission = crate::profile::language(&handle)
                .map(|_| ())
                .map_err(|error| error.to_string());
            let report = MetalReport {
                registry_id: handle.registry_id(),
                name: handle.name(),
                unified: handle.has_unified_memory(),
                tensor_operations: admission.is_ok()
                    && crate::profile::forms_tensor_operations(&handle),
                admission,
                working_set: handle.recommended_working_set_bytes(),
                max_allocation: handle.max_allocation_bytes(),
                kind: kind(),
            };
            (report, handle)
        })
        .collect()
}
