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
use crate::VulkanPrograms;
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::GraphCost;
use seismic_native_target::{ProgramEntry, ProgramSource};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// The launches the command memory is measured over: as many as a chunk
/// holds of the least argument block, so their blocks fill a whole chunk.
const LAUNCHES: u64 = ARGUMENT_CHUNK_BYTES / argument_block_bytes(0, 0);

/// An empty compute shader, formed with a group of one invocation
/// ([`measure`]).
const EMPTY: &str = "#version 460\nlayout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\nvoid main() {}\n";

/// What a device's recorded graphs are charged at before [`graph_cost`]
/// measured its command memory: a bound above what devices measured so far
/// took (those measured took none a launch), beside the backend's own
/// argument chunks, so what is planned at it still fits once the device's
/// own cost is read.
pub const UNMEASURED: GraphCost = GraphCost::Chunked { launch: 1 << 10, chunk: ARGUMENT_CHUNK_BYTES };

/// What graphs recorded on `device` hold, measured on its first call in
/// the process and kept for the device after. The empty pipeline it is
/// measured with is formed through the device's `programs`, so a process
/// after the first that formed it loads it.
pub fn graph_cost(device: &Device, programs: &VulkanPrograms) -> Result<GraphCost, ExecutionError> {
    static COSTS: Mutex<BTreeMap<[u8; 16], GraphCost>> = Mutex::new(BTreeMap::new());
    let mut costs = COSTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let uuid = device.facts().uuid;
    if let Some(cost) = costs.get(&uuid) {
        return Ok(*cost);
    }
    let cost = measure(device, programs)?;
    costs.insert(uuid, cost);
    Ok(cost)
}

fn measure(device: &Device, programs: &VulkanPrograms) -> Result<GraphCost, ExecutionError> {
    let entry = ProgramEntry {
        symbol: "main".into(),
        group_size: Some([1, 1, 1]),
        constants: Vec::new(),
    };
    let source = ProgramSource { text: EMPTY.into(), entries: vec![entry.clone()] };
    let formed = programs.form(std::slice::from_ref(&source)).map_err(|error| {
        ExecutionError::SubmissionFailed(format!("measuring the graphs' command memory: {error}"))
    })?;
    let (module, function) = formed[0].program(&entry);
    let launch = DirectLaunch {
        module,
        function,
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
