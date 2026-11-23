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
        let admission = descriptor.admission().map_err(|error| error.to_string());
        let memory = cuda_memory(
            descriptor.memory_is_host(),
            &admission,
            uuid,
            || {
                crate::open::usable_memory_report(ordinal, uuid)
                    .map_err(|error| error.to_string())
            },
            |message| diagnostics.push(diagnostic(format!("device ordinal {ordinal}: {message}"))),
        );
        reports.push((
            CudaReport {
                uuid,
                luid: descriptor.luid,
                name: descriptor.name.clone(),
                driver: format!("cuda driver {}", descriptor.driver_api),
                admission,
                total: descriptor.total_memory_bytes,
                max_allocation: crate::max_allocation_bytes(descriptor.total_memory_bytes),
                allocation_granularity: descriptor.allocation_granularity,
                integrated: descriptor.integrated,
                l2: (descriptor.l2_cache_bytes > 0).then_some(descriptor.l2_cache_bytes),
                host_node: HostNode::Any,
                memory,
                kinds: kinds(),
                properties: Some(seismic_lang::device::DeviceProperties::Cuda(
                    descriptor.properties,
                )),
            },
            Some(PciLocation { domain, bus, device }),
        ));
    }
    reports
}

/// How a CUDA device's memory is reported: host memory directly, or its own
/// device memory, querying usable bytes through a CUDA context only when the
/// device passes backend admission.
fn cuda_memory(
    memory_is_host: bool,
    admission: &Result<(), String>,
    uuid: [u8; 16],
    read_usable: impl FnOnce() -> Result<(u64, Option<String>), String>,
    mut on_warning: impl FnMut(String),
) -> CudaMemory {
    if memory_is_host {
        return CudaMemory::Host;
    }
    let usable = match admission {
        Ok(()) => usable_at_first_read(uuid, read_usable).map(|(bytes, warning)| {
            if let Some(message) = warning {
                on_warning(message);
            }
            bytes
        }),
        Err(reason) => Err(reason.clone()),
    };
    CudaMemory::Own { usable }
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
    use super::{cuda_memory, usable_at_first_read};
    use seismic_topology::CudaMemory;

    /// An unadmitted device (for example pre-sm_80) never creates a CUDA
    /// context during discovery.
    #[test]
    fn unadmitted_device_skips_usable_memory_read() {
        let mut probed = false;
        let reason = "compute capability 6.1 is below 8.0".to_owned();
        let memory = cuda_memory(
            false,
            &Err(reason.clone()),
            [0x5a; 16],
            || {
                probed = true;
                Ok((8 << 30, None))
            },
            |_| panic!("unexpected warning"),
        );
        assert!(!probed, "unadmitted device must not create a CUDA context");
        assert!(matches!(memory, CudaMemory::Own { usable: Err(ref err) } if err == &reason));
    }

    /// An admitted discrete device reads usable memory once and forwards
    /// warnings; a host-memory device does not read usable memory.
    #[test]
    fn admitted_discrete_device_reads_usable_memory_and_host_memory_skips_it() {
        let mut probed = false;
        assert!(matches!(
            cuda_memory(
                true,
                &Ok(()),
                [0x5b; 16],
                || {
                    probed = true;
                    Ok((8 << 30, None))
                },
                |_| {},
            ),
            CudaMemory::Host
        ));
        assert!(!probed);

        let mut warnings = Vec::new();
        let memory = cuda_memory(
            false,
            &Ok(()),
            [0x5c; 16],
            || Ok((12 << 30, Some("clamped".to_owned()))),
            |warning| warnings.push(warning),
        );
        assert!(matches!(memory, CudaMemory::Own { usable: Ok(bytes) } if bytes == 12 << 30));
        assert_eq!(warnings, ["clamped"]);
    }

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
