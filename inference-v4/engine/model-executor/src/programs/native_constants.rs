//! Host constants of sealed graphs. A constant is declared as an external
//! port while its graph is built. Binding uploads each distinct value once and
//! attaches it with the weights as a static binding, so no run writes it.

use seismic::{Device, Element, NativeGraph, NativePort, Tensor};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct GraphConstant {
    port: NativePort,
    element: Element,
    extents: Vec<u64>,
    bytes: Arc<[u8]>,
}

impl GraphConstant {
    /// A rank-1 `i32` constant.
    pub(crate) fn i32(graph: &mut NativeGraph, values: &[i32]) -> Result<Self, String> {
        Self::rank1(
            graph,
            Element::i32(),
            values.len(),
            values.iter().flat_map(|value| value.to_le_bytes()).collect(),
        )
    }

    /// A rank-1 `f32` constant.
    pub(crate) fn f32(graph: &mut NativeGraph, values: &[f32]) -> Result<Self, String> {
        Self::rank1(
            graph,
            Element::f32(),
            values.len(),
            values.iter().flat_map(|value| value.to_le_bytes()).collect(),
        )
    }

    /// The rank-1 `i32` constant `0, 1, …, count - 1`.
    pub(crate) fn identity(graph: &mut NativeGraph, count: u64) -> Result<Self, String> {
        let values = (0..count)
            .map(|index| i32::try_from(index).map_err(|_| "identity index exceeds i32".to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        Self::i32(graph, &values)
    }

    fn rank1(
        graph: &mut NativeGraph,
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
        let same = |uploaded: &GraphConstant| {
            uploaded.element == constant.element
                && uploaded.extents == constant.extents
                && uploaded.bytes == constant.bytes
        };
        if let Some((_, tensor)) = self.uploaded.iter().find(|(uploaded, _)| same(uploaded)) {
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
}
