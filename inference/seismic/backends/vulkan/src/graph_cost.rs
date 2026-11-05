//! What a device's recorded graphs hold ([`GraphCost::Chunked`]): the
//! driver's command memory per launch, measured once per device in a
//! process, and the argument chunks the backend packs their blocks into
//! ([`seismic_compiler::graph_cost::argument_bytes`]).
//!
//! The command memory is measured as the graphs' own is
//! (`DirectGraphBuilder::instantiate`): the growth of the device heap's
//! usage across recording that no block explains, here over [`LAUNCHES`]
//! launches of an empty pipeline recorded into one graph, each as a
//! sealed submission's launch is.

use crate::device::{Device, ARGUMENT_CHUNK_BYTES};
use crate::direct::{argument_block_bytes, DirectGraphBuilder, DirectLaunch};
use crate::formation::DirectModule;
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::GraphCost;
use seismic_native_target::ProgramEntry;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// The launches the command memory is measured over: as many as a chunk
/// holds of the least argument block, so their blocks fill a whole chunk.
const LAUNCHES: u64 = ARGUMENT_CHUNK_BYTES / argument_block_bytes(0, 0);

/// An empty compute shader of one invocation (SPIR-V 1.0): `main`, local
/// size 1×1×1, returning at once.
pub(crate) const EMPTY: [u32; 35] = [
    // Magic, version 1.0, generator, bound 5, schema.
    0x0723_0203, 0x0001_0000, 0, 5, 0,
    // OpCapability Shader
    0x0002_0011, 1,
    // OpMemoryModel Logical GLSL450
    0x0003_000E, 0, 1,
    // OpEntryPoint GLCompute %1 "main"
    0x0005_000F, 5, 1, 0x6E69_616D, 0,
    // OpExecutionMode %1 LocalSize 1 1 1
    0x0006_0010, 1, 17, 1, 1, 1,
    // %2 = OpTypeVoid
    0x0002_0013, 2,
    // %3 = OpTypeFunction %2
    0x0003_0021, 3, 2,
    // %1 = OpFunction %2 None %3
    0x0005_0036, 2, 1, 0, 3,
    // %4 = OpLabel; OpReturn; OpFunctionEnd
    0x0002_00F8, 4, 0x0001_00FD, 0x0001_0038,
];

/// What graphs recorded on `device` hold, measured on its first call in
/// the process and kept for the device after.
pub fn graph_cost(device: &Device) -> Result<GraphCost, ExecutionError> {
    static COSTS: Mutex<BTreeMap<[u8; 16], GraphCost>> = Mutex::new(BTreeMap::new());
    let mut costs = COSTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let uuid = device.facts().uuid;
    if let Some(cost) = costs.get(&uuid) {
        return Ok(*cost);
    }
    let cost = measure(device)?;
    costs.insert(uuid, cost);
    Ok(cost)
}

fn measure(device: &Device) -> Result<GraphCost, ExecutionError> {
    let entry = ProgramEntry {
        symbol: "main".into(),
        group_size: Some([1, 1, 1]),
        constants: Vec::new(),
    };
    let (module, _) = DirectModule::new(device, &[(&EMPTY[..], &entry)], &[]).map_err(|error| {
        ExecutionError::SubmissionFailed(format!("measuring the graphs' command memory: {error}"))
    })?;
    let launch = DirectLaunch {
        module: &module,
        function: 0,
        buffers: &[],
        words: &[],
        scalar_results: None,
        groups: [1, 1, 1],
        grid: None,
    };
    let mut builder = DirectGraphBuilder::new(device)?;
    for _ in 0..LAUNCHES {
        builder.launch(&launch)?;
    }
    let graph = builder.instantiate()?;
    Ok(GraphCost::Chunked {
        launch: graph.command_bytes().div_ceil(LAUNCHES),
        chunk: ARGUMENT_CHUNK_BYTES,
    })
}
