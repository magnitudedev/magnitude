//! The CUDA backend's description of its devices in the topology's terms
//! (`seismic_topology::CudaReport`): the memory it allocates, as kinds.

use seismic_lang::registry::BackendName;
use seismic_topology::{
    CudaMemory, CudaReport, Diagnostic, DiagnosticSource, HostNode, HostReads, KindMemory,
    KindReport, PciLocation, Reach,
};

/// Device memory (`cuMemAlloc`).
pub const DEVICE_KIND: u32 = 0;
/// Pinned host memory (`cuMemHostAlloc`).
pub const PINNED_KIND: u32 = 1;

/// The kinds CUDA allocates: device memory, which the host cannot map, and
/// pinned host memory.
pub fn kinds() -> Vec<KindReport> {
    let kind = |id, name: &str, reach, memory| KindReport { id, name: name.into(), reach, memory };
    vec![
        kind(DEVICE_KIND, "device memory", Reach::Device, KindMemory::DeviceMemory),
        kind(PINNED_KIND, "pinned host memory", Reach::Mapped { reads: HostReads::Cached }, KindMemory::HostRam),
    ]
}

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Every CUDA device's report, paired with its PCI bus location; what cannot
/// be read, as `diagnostics`.
pub fn describe(diagnostics: &mut Vec<Diagnostic>) -> Vec<(CudaReport, Option<PciLocation>)> {
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
        let (ordinal, uuid) = (descriptor.ordinal, descriptor.uuid);
        let [domain, bus, device] = descriptor.pci;
        let memory = if descriptor.memory_is_host() {
            CudaMemory::Host
        } else {
            let usable = usable_at_first_read(uuid, || {
                crate::open::usable_memory_report(ordinal, uuid)
                    .map_err(|error| error.to_string())
            })
            .map(|(bytes, warning)| {
                if let Some(message) = warning {
                    diagnostics.push(diagnostic(format!("device ordinal {ordinal}: {message}")));
                }
                bytes
            });
            CudaMemory::Own { usable }
        };
        reports.push((
            CudaReport {
                uuid,
                luid: descriptor.luid,
                name: descriptor.name.clone(),
                driver: format!("cuda driver {}", descriptor.driver_api),
                admission: descriptor.admission().map_err(|error| error.to_string()),
                total: descriptor.total_memory_bytes,
                max_allocation: crate::max_allocation_bytes(descriptor.total_memory_bytes),
                allocation_granularity: descriptor.allocation_granularity,
                integrated: descriptor.integrated,
                l2: (descriptor.l2_cache_bytes > 0).then_some(descriptor.l2_cache_bytes),
                host_node: HostNode::Any,
                memory,
                kinds: kinds(),
            },
            Some(PciLocation { domain, bus, device }),
        ));
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
    read: impl FnOnce() -> Result<(u64, Option<String>), String>,
) -> Result<(u64, Option<String>), String> {
    static USABLE: OnceLock<Mutex<HashMap<[u8; 16], (u64, Option<String>)>>> = OnceLock::new();
    let mut usable = USABLE
        .get_or_init(Mutex::default)
        .lock()
        .expect("the usable-memory registry lock is never poisoned");
    if let Some(entry) = usable.get(&uuid) {
        return Ok(entry.clone());
    }
    let entry = read()?;
    usable.insert(uuid, entry.clone());
    Ok(entry)
}

#[cfg(test)]
mod tests {
    use super::usable_at_first_read;

    /// The first successful read stands for the process; after a failure it
    /// is read again.
    #[test]
    fn usable_memory_is_kept_for_the_process_and_a_failure_is_not() {
        let uuid = [0xa5; 16];
        let busy = || Err("busy".to_owned());
        assert_eq!(usable_at_first_read(uuid, busy), Err("busy".to_owned()));
        assert_eq!(
            usable_at_first_read(uuid, || Ok((6 << 30, None))),
            Ok((6 << 30, None))
        );
        assert_eq!(
            usable_at_first_read(uuid, || Ok((1 << 30, None))),
            Ok((6 << 30, None))
        );
    }

    /// The recorded fixtures' CUDA kinds are the ones this backend declares.
    #[test]
    fn the_fixtures_kinds_are_the_backends() {
        let rtx3090 = seismic_topology::fixtures::rtx3090_cuda(seismic_topology::fixtures::rtx3090_uuid());
        assert_eq!(super::kinds(), rtx3090.kinds);
        let gb10 = seismic_topology::fixtures::gb10().cuda.remove(0);
        assert_eq!(super::kinds(), gb10.kinds);
    }
}
