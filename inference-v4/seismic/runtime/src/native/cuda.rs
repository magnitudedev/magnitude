//! CUDA submission of native dispatch lists.
//!
//! A submission of sealed graph runs (one run, or a sequence of runs
//! submitted together) replays a CUDA graph: its launches, with every
//! argument fixed, instantiated once and launched with one driver call. The
//! arguments are the plans' sealed words and geometry plus the addresses of
//! the buffers the runs bind, so the device keeps graphs keyed by the plans
//! and those addresses; a submission binding the same storage as an earlier
//! one replays that submission's graph, and one binding new storage forms a
//! new graph. Per-step values never live in arguments: host-written inputs
//! are written into each run's upload region, which the graph addresses.
//!
//! Standalone calls, and every launch of a launch-detail trace (timed one by
//! one), are launched individually.

use super::{DispatchList, NativeRoute, RouteSubmission};
use crate::api::CallError;
use crate::driver::{typed_buffer, Allocation};
use seismic_compiler::errors::ExecutionError;
use seismic_cuda::direct::{DirectBatch, DirectGraph, DirectGraphBuilder, DirectLaunch};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Cuda = seismic_cuda::Cuda;
type Executor = seismic_cuda::Executor;

/// Graphs one device keeps; the least recently replayed is dropped beyond
/// it. The steady state is one graph per step shape (row class and batch)
/// and workspace slot, plus the runs submitted alone.
const REPLAY_CAPACITY: usize = 1024;

/// A device's CUDA graphs of sealed-plan submissions, keyed by the plans'
/// identities followed by every buffer address the submission binds.
#[derive(Default)]
pub(crate) struct Replays {
    entries: Mutex<ReplayEntries>,
}

#[derive(Default)]
struct ReplayEntries {
    clock: u64,
    graphs: HashMap<Vec<u64>, Replay>,
}

struct Replay {
    /// Clock of the last replay.
    used: u64,
    graph: DirectGraph,
    /// The plans the graph's kernels belong to, kept loaded while it lives.
    _retained: Arc<dyn Any + Send + Sync>,
}

/// Encode `list` on the device's stream: a replay when the list is made of
/// sealed plans and is not timed, else launch by launch.
pub(super) fn encode(
    device: &seismic_cuda::Device,
    replays: &Replays,
    list: &impl DispatchList,
    repetitions: usize,
    timed: bool,
    retained: &Arc<dyn Any + Send + Sync>,
) -> Result<RouteSubmission, CallError> {
    let mut batch = if timed {
        DirectBatch::timed(device)
    } else {
        DirectBatch::new(device)
    }
    .map_err(CallError::Execution)?;
    match list.plans().filter(|_| !timed) {
        Some(mut key) => {
            key.extend(addresses(list));
            let mut entries = replays
                .entries
                .lock()
                .expect("native graph replays lock is never poisoned");
            entries.clock += 1;
            let clock = entries.clock;
            if !entries.graphs.contains_key(&key) {
                if entries.graphs.len() == REPLAY_CAPACITY {
                    let oldest = entries
                        .graphs
                        .iter()
                        .min_by_key(|(_, replay)| replay.used)
                        .map(|(key, _)| key.clone())
                        .expect("a full replay set has an entry");
                    entries.graphs.remove(&oldest);
                }
                let mut builder =
                    DirectGraphBuilder::new(device).map_err(CallError::Execution)?;
                each_launch(list, |launch| match launch {
                    Some(launch) => builder.launch(launch),
                    None => Ok(()),
                })?;
                let graph = builder.instantiate().map_err(CallError::Execution)?;
                entries.graphs.insert(
                    key.clone(),
                    Replay {
                        used: clock,
                        graph,
                        _retained: retained.clone(),
                    },
                );
            }
            let replay = entries
                .graphs
                .get_mut(&key)
                .expect("the submission's graph was formed above");
            replay.used = clock;
            for _ in 0..repetitions {
                batch.replay(&replay.graph).map_err(CallError::Execution)?;
            }
        }
        None => {
            for _ in 0..repetitions {
                each_launch(list, |launch| match launch {
                    Some(launch) => batch.launch(launch),
                    None => {
                        batch.skip();
                        Ok(())
                    }
                })?;
            }
        }
    }
    batch
        .commit()
        .map(RouteSubmission::Cuda)
        .map_err(CallError::Execution)
}

/// Visit every launch of one pass over `list` in order; `None` for an
/// inactive launch.
fn each_launch(
    list: &impl DispatchList,
    mut visit: impl FnMut(Option<&DirectLaunch<'_>>) -> Result<(), ExecutionError>,
) -> Result<(), CallError> {
    let mut buffers = Vec::new();
    let mut typed = Vec::new();
    for index in 0..list.count() {
        buffers.clear();
        let dispatch = list.dispatch(index, &mut buffers);
        let NativeRoute::Cuda { module, .. } = &dispatch.kernel.route else {
            unreachable!("one device has one native route");
        };
        typed.clear();
        typed.extend(
            buffers
                .iter()
                .map(|(allocation, offset)| (typed_buffer::<Cuda, Executor>(allocation), *offset)),
        );
        let scalars = typed_buffer::<Cuda, Executor>(&dispatch.kernel.scalars);
        for (function, launch) in dispatch.launches.iter().enumerate() {
            let launch = launch.map(|launch| DirectLaunch {
                module,
                function,
                buffers: &typed,
                words: dispatch.word_bytes,
                scalar_results: (scalars, 0),
                grid: launch.groups,
                block: launch.threads,
                shared_bytes: launch.shared_bytes,
            });
            visit(launch.as_ref()).map_err(CallError::Execution)?;
        }
    }
    Ok(())
}

/// Every buffer address one pass over `list` binds, in dispatch order: with
/// the sealed plans, they fix every launch argument.
fn addresses(list: &impl DispatchList) -> Vec<u64> {
    let mut buffers: Vec<(&Allocation, u64)> = Vec::new();
    for index in 0..list.count() {
        list.dispatch(index, &mut buffers);
    }
    buffers
        .iter()
        .map(|(allocation, offset)| typed_buffer::<Cuda, Executor>(allocation).pointer() + offset)
        .collect()
}
