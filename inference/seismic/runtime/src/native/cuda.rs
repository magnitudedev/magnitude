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
//!
//! A timeline boundary (`timeline`) divides the calls: those between two
//! boundaries replay a graph of their own, and at the boundary the stream
//! waits on a gate's host word (`cuStreamWaitValue32`) or writes a signal's
//! (`cuStreamWriteValue32`).

use super::graph_replays::Replays;
use super::timeline::{TimelineKind, TimelineNode, TimelineOperation};
use super::{DispatchList, NativeRoute, RouteSubmission};
use std::ops::Range;
use crate::api::CallError;
use crate::driver::{Allocation, typed_buffer};
use seismic_compiler::errors::ExecutionError;
use seismic_cuda::direct::{DirectBatch, DirectGraph, DirectGraphBuilder, DirectLaunch};
use std::any::Any;
use std::sync::Arc;

type Cuda = seismic_cuda::Cuda;
type Executor = seismic_cuda::Executor;

/// A device's CUDA graphs of sealed-plan submissions.
pub(crate) type CudaReplays = Replays<DirectGraph>;

impl super::graph_replays::RecordedGraph for DirectGraph {
    fn device_bytes(&self) -> u64 {
        DirectGraph::device_bytes(self)
    }
}

/// Encode `list` on the device's stream: a replay when the list is made of
/// sealed plans and is not timed, else launch by launch.
pub(super) fn encode(
    device: &seismic_cuda::Device,
    replays: &CudaReplays,
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
    // The calls between timeline boundaries form one segment: a graph of
    // its own when replayed, the stream waiting on or writing the
    // boundary's words between segments.
    let segments = segments(list);
    match list.plans().filter(|_| !timed) {
        Some(plans) => {
            let addresses = addresses(list);
            for _ in 0..repetitions {
                for (calls, boundary) in &segments {
                    if !calls.is_empty() {
                        let mut key = plans.clone();
                        key.extend(&addresses);
                        key.extend([calls.start as u64, calls.end as u64]);
                        replays.replay(
                            key,
                            retained,
                            || {
                                let mut builder = DirectGraphBuilder::new(device)
                                    .map_err(CallError::Execution)?;
                                each_launch(list, calls.clone(), |launch| match launch {
                                    Some(launch) => builder.launch(launch),
                                    None => Ok(()),
                                })?;
                                builder.instantiate().map_err(CallError::Execution)
                            },
                            |graph| batch.replay(graph).map_err(CallError::Execution),
                        )?;
                    }
                    timelines(&mut batch, list.timelines(*boundary))?;
                }
            }
        }
        None => {
            for _ in 0..repetitions {
                for (calls, boundary) in &segments {
                    each_launch(list, calls.clone(), |launch| match launch {
                        Some(launch) => batch.launch(launch),
                        None => {
                            batch.skip();
                            Ok(())
                        }
                    })?;
                    timelines(&mut batch, list.timelines(*boundary))?;
                }
            }
        }
    }
    batch
        .commit()
        .map(RouteSubmission::Cuda)
        .map_err(CallError::Execution)
}

/// The calls of `list` between timeline boundaries, each range with the
/// boundary that ends it, from a leading empty range for the operations
/// before the first call to the range ending at the list's end.
fn segments(list: &impl DispatchList) -> Vec<(Range<usize>, usize)> {
    let mut segments = Vec::new();
    let mut first = 0;
    for boundary in 0..=list.count() {
        if boundary == list.count() || !list.timelines(boundary).is_empty() {
            segments.push((first..boundary, boundary));
            first = boundary;
        }
    }
    segments
}

/// Queue a boundary's timeline operations on the stream.
fn timelines(batch: &mut DirectBatch, operations: &[TimelineOperation]) -> Result<(), CallError> {
    for operation in operations {
        let TimelineKind::Cuda(word) = operation.timeline.kind() else {
            unreachable!("a run binds timelines of its own device");
        };
        // The word holds a value's low 32 bits (`timeline`).
        let value = operation.value as u32;
        match operation.node {
            TimelineNode::Wait => batch.wait_word(word, value),
            TimelineNode::Signal => batch.write_word(word, value),
        }
        .map_err(CallError::Execution)?;
    }
    Ok(())
}

/// Visit every launch of the calls `calls` of one pass over `list` in
/// order; `None` for an inactive launch.
fn each_launch(
    list: &impl DispatchList,
    calls: Range<usize>,
    mut visit: impl FnMut(Option<&DirectLaunch<'_>>) -> Result<(), ExecutionError>,
) -> Result<(), CallError> {
    let mut buffers = Vec::new();
    let mut typed = Vec::new();
    for index in calls {
        buffers.clear();
        let dispatch = list.dispatch(index, &mut buffers);
        let NativeRoute::Cuda { launches, .. } = &dispatch.kernel.route else {
            unreachable!("one device has one native route");
        };
        typed.clear();
        typed.extend(
            buffers
                .iter()
                .map(|(allocation, offset)| (typed_buffer::<Cuda, Executor>(allocation), *offset)),
        );
        let scalars = dispatch
            .kernel
            .scalar_slots()
            .map(|slots| (typed_buffer::<Cuda, Executor>(slots), 0));
        for (function, launch) in dispatch.launches.iter().enumerate() {
            let launch = launch.map(|launch| {
                let (module, module_function) = launches.launch(function);
                DirectLaunch {
                    module,
                    function: module_function,
                    buffers: &typed,
                    words: dispatch.word_bytes,
                    scalar_results: scalars,
                    grid: launch.groups,
                    block: launch.threads,
                    shared_bytes: launch.shared_bytes,
                }
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
