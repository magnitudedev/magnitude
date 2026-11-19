//! The Metal backend's description of its devices in the topology's terms
//! (`seismic_topology::MetalReport`).

use crate::DeviceHandle;
use seismic_lang::registry::BackendName;
use seismic_native_target::ArtifactStore;
use seismic_topology::{Diagnostic, HostReads, KindMemory, KindReport, MetalReport, Reach};

/// The id of shared storage mode ([`kinds`]).
pub const SHARED_KIND: u32 = 0;

/// The id of private storage mode ([`kinds`]).
pub const PRIVATE_KIND: u32 = 1;

/// The backend's kinds on a device whose memory is the host's
/// (`unified`), or its own:
/// - shared storage mode, which the host maps: host RAM, and on a unified
///   device every contract's;
/// - on a device whose memory is its own (an Intel Mac's discrete AMD GPU),
///   private storage mode too: its own memory, which the host cannot map,
///   and where device work's own buffers lie.
pub fn kinds(unified: bool) -> Vec<KindReport> {
    let shared = KindReport {
        id: SHARED_KIND,
        name: "shared storage".into(),
        reach: Reach::Mapped { reads: HostReads::Cached },
        memory: KindMemory::HostRam,
    };
    match unified {
        true => vec![shared],
        false => vec![
            KindReport { id: PRIVATE_KIND, name: "private storage".into(), reach: Reach::Device, memory: KindMemory::DeviceMemory },
            shared,
        ],
    }
}

/// The kind of the device's own memory, which the store it is opened with
/// ledgers (device work's own buffers): private storage on a device whose
/// memory is its own, shared storage on a unified one.
pub fn own_kind(unified: bool) -> u32 {
    match unified {
        true => SHARED_KIND,
        false => PRIVATE_KIND,
    }
}

/// Every Metal device's report, with its handle, which status samples
/// read; that there is none, as `diagnostics`. A device's capability probes
/// answer its language and tensor-operation support, read from `store` and
/// compiled only where it lacks them (`profile::probe_answers`).
pub fn describe(
    diagnostics: &mut Vec<Diagnostic>,
    store: &dyn ArtifactStore,
) -> Vec<(MetalReport, DeviceHandle)> {
    let handles = DeviceHandle::discover();
    if handles.is_empty() {
        diagnostics.push(Diagnostic::no_device(BackendName::Metal));
    }
    handles
        .into_iter()
        .map(|handle| {
            let answers = crate::profile::probe_answers(&handle, store);
            let report = MetalReport {
                registry_id: handle.registry_id(),
                name: handle.name(),
                unified: handle.has_unified_memory(),
                tensor_operations: answers.forms_tensor_operations(&handle),
                admission: answers.language().map(|_| ()).map_err(|error| error.to_string()),
                working_set: handle.recommended_working_set_bytes(),
                max_allocation: handle.max_allocation_bytes(),
                kinds: kinds(handle.has_unified_memory()),
            };
            (report, handle)
        })
        .collect()
}
