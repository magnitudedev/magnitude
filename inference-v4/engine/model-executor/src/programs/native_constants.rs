//! Host constants of sealed graphs. A constant is declared as an external
//! port while its graph is built. Binding uploads each distinct value once and
//! attaches it with the weights as a static binding, so no run writes it.

use super::graph::draft::GraphDraft;
use seismic::{Device, Element, NativeGraphStorageBytes, NativePort, Tensor};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct GraphConstant {
    port: NativePort,
    element: Element,
    extents: Vec<u64>,
    bytes: Arc<[u8]>,
}

impl GraphConstant {
    fn same_storage_as(&self, other: &Self) -> bool {
        self.element == other.element && self.extents == other.extents && self.bytes == other.bytes
    }

    pub(crate) fn storage_bytes(&self) -> Result<u64, String> {
        self.element
            .canonical_byte_len(&self.extents)
            .map_err(|error| error.to_string())
    }

    /// A rank-1 `i32` constant.
    pub(crate) fn i32<G: GraphDraft>(graph: &mut G, values: &[i32]) -> Result<Self, String> {
        Self::rank1(
            graph,
            Element::i32(),
            values.len(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }

    /// A rank-1 `f32` constant.
    pub(crate) fn f32<G: GraphDraft>(graph: &mut G, values: &[f32]) -> Result<Self, String> {
        Self::rank1(
            graph,
            Element::f32(),
            values.len(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }

    /// The rank-1 `i32` constant `0, 1, …, count - 1`.
    pub(crate) fn identity<G: GraphDraft>(graph: &mut G, count: u64) -> Result<Self, String> {
        let values = (0..count)
            .map(|index| i32::try_from(index).map_err(|_| "identity index exceeds i32".to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        Self::i32(graph, &values)
    }

    fn rank1<G: GraphDraft>(
        graph: &mut G,
        element: Element,
        length: usize,
        bytes: Arc<[u8]>,
    ) -> Result<Self, String> {
        let extents = vec![u64::try_from(length).map_err(|_| "constant length exceeds u64")?];
        let port = graph
            .port(element, &extents)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            port,
            element,
            extents,
            bytes,
        })
    }

    pub(crate) fn port(&self) -> &NativePort {
        &self.port
    }
}

/// The binder uploads one tensor for each distinct constant value. Count the
/// same unique set before binding so the capacity grant covers its allocation.
pub(crate) fn distinct_storage_bytes<'a>(
    constants: impl IntoIterator<Item = &'a GraphConstant>,
) -> Result<u64, String> {
    let mut unique = Vec::<&GraphConstant>::new();
    let mut total = 0u64;
    for constant in constants {
        if unique.iter().any(|other| other.same_storage_as(constant)) {
            continue;
        }
        total = total
            .checked_add(constant.storage_bytes()?)
            .ok_or("graph constant charge overflows")?;
        unique.push(constant);
    }
    Ok(total)
}

/// Metadata-only family holdings. Each graph contributes independently to
/// the three arena maxima, while constants are uploaded once per distinct
/// value across the family by the production binder.
pub(crate) struct CheckedGraphFamilyResources {
    storage: Option<NativeGraphStorageBytes>,
    constants: Vec<GraphConstant>,
}

impl CheckedGraphFamilyResources {
    pub(crate) fn new() -> Self {
        Self {
            storage: None,
            constants: Vec::new(),
        }
    }

    pub(crate) fn include(
        &mut self,
        storage: NativeGraphStorageBytes,
        constants: impl IntoIterator<Item = GraphConstant>,
    ) {
        self.storage = Some(match self.storage {
            None => storage,
            Some(previous) => NativeGraphStorageBytes {
                workspace: previous.workspace.max(storage.workspace),
                output: previous.output.max(storage.output),
                upload: previous.upload.max(storage.upload),
            },
        });
        for constant in constants {
            if !self
                .constants
                .iter()
                .any(|prior| prior.same_storage_as(&constant))
            {
                self.constants.push(constant);
            }
        }
    }

    pub(crate) fn finish(self) -> Result<CheckedGraphResources, String> {
        Ok(CheckedGraphResources {
            storage: self.storage.ok_or("checked graph family has no class")?,
            binding_constant_bytes: distinct_storage_bytes(self.constants.iter())?,
        })
    }
}

pub(crate) struct CheckedGraphResources {
    pub(crate) storage: NativeGraphStorageBytes,
    pub(crate) binding_constant_bytes: u64,
}

/// Device tensors of bound constants, one per distinct (element, extents,
/// bytes). Constants are small and few, so a linear search suffices.
pub(crate) struct ConstantTensors {
    device: Device,
    uploaded: Vec<(GraphConstant, Tensor)>,
}

impl ConstantTensors {
    pub(crate) fn new(device: Device) -> Self {
        Self {
            device,
            uploaded: Vec::new(),
        }
    }

    pub(crate) fn tensor(&mut self, constant: &GraphConstant) -> Result<Tensor, String> {
        if let Some((_, tensor)) = self
            .uploaded
            .iter()
            .find(|(uploaded, _)| uploaded.same_storage_as(constant))
        {
            return Ok(tensor.clone());
        }
        let tensor = Tensor::from_host(
            &self.device,
            constant.element,
            &constant.extents,
            &constant.bytes,
        )
        .map_err(|error| format!("graph constant upload failed: {error}"))?;
        self.uploaded.push((constant.clone(), tensor.clone()));
        Ok(tensor)
    }

    /// Keep the unique uploaded allocations available to the owner census.
    pub(crate) fn into_tensors(self) -> Vec<Tensor> {
        self.uploaded
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog};

    #[test]
    fn binding_claim_matches_distinct_physical_constant_charge() {
        let catalog = DeviceCatalog::discover().unwrap();
        let device = catalog.open_backend(BackendName::Cpu).unwrap();
        let mut graph = device.native_graph();
        let first = GraphConstant::i32(&mut graph, &[1, 2, 3]).unwrap();
        let tied = GraphConstant::i32(&mut graph, &[1, 2, 3]).unwrap();
        let other = GraphConstant::f32(&mut graph, &[1.0, 2.0]).unwrap();
        let claim = distinct_storage_bytes([&first, &tied, &other]).unwrap();
        let before = device.memory_usage().charged;
        let mut uploaded = ConstantTensors::new(device.clone());
        uploaded.tensor(&first).unwrap();
        uploaded.tensor(&tied).unwrap();
        uploaded.tensor(&other).unwrap();
        assert_eq!(device.memory_usage().charged - before, claim);
        drop(uploaded);
        assert_eq!(device.memory_usage().charged, before);
    }
}
