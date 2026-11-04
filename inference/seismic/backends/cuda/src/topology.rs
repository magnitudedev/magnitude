//! The CUDA backend's description of its devices in the topology's terms
//! (`seismic_topology::CudaReport`).

use seismic_lang::registry::BackendName;
use seismic_topology::{CudaMemory, CudaReport, Diagnostic, DiagnosticSource};
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
            memory: if descriptor.integrated || descriptor.numa_node {
                CudaMemory::Host
            } else {
                CudaMemory::Own {
                    usable: usable_at_first_read(uuid, || {
                        crate::usable_memory_bytes(ordinal, uuid).map_err(|error| error.to_string())
                    }),
                }
            },
            peers,
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
}
