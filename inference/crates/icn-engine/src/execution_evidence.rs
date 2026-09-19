//! Native target-model allocation identity, before physical-domain aggregation.

use llama_cpp_2::context::{LlamaMemoryBreakdown, LlamaMemoryLocation};
use serde::Serialize;

/// Read-only resident evidence. Host-addressable unified memory is not relabelled as CPU memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResidentModelAllocation {
    Host {
        model_bytes: u64,
    },
    Device {
        backend: String,
        physical_id: Option<String>,
        native_index: usize,
        model_bytes: u64,
    },
}

pub(crate) fn target_model_allocations(
    allocations: &[LlamaMemoryBreakdown],
) -> Vec<ResidentModelAllocation> {
    allocations
        .iter()
        .filter(|allocation| allocation.model_bytes > 0)
        .map(|allocation| match &allocation.location {
            LlamaMemoryLocation::Host => ResidentModelAllocation::Host {
                model_bytes: allocation.model_bytes,
            },
            LlamaMemoryLocation::Device {
                backend,
                physical_id,
                native_index,
            } => ResidentModelAllocation::Device {
                backend: backend.clone(),
                physical_id: physical_id.clone(),
                native_index: *native_index,
                model_bytes: allocation.model_bytes,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_native_device_identity_and_excludes_non_model_buffers() {
        let allocations = [
            LlamaMemoryBreakdown {
                location: LlamaMemoryLocation::Host,
                model_bytes: 128,
                context_bytes: 1024,
                compute_bytes: 2048,
            },
            LlamaMemoryBreakdown {
                location: LlamaMemoryLocation::Device {
                    backend: "Metal".to_owned(),
                    physical_id: None,
                    native_index: 2,
                },
                model_bytes: 4096,
                context_bytes: 8192,
                compute_bytes: 512,
            },
            LlamaMemoryBreakdown {
                location: LlamaMemoryLocation::Device {
                    backend: "CUDA".to_owned(),
                    physical_id: Some("gpu-1".to_owned()),
                    native_index: 3,
                },
                model_bytes: 0,
                context_bytes: 8192,
                compute_bytes: 512,
            },
        ];
        assert_eq!(
            target_model_allocations(&allocations),
            vec![
                ResidentModelAllocation::Host { model_bytes: 128 },
                ResidentModelAllocation::Device {
                    backend: "Metal".to_owned(),
                    physical_id: None,
                    native_index: 2,
                    model_bytes: 4096,
                },
            ]
        );
    }
}
