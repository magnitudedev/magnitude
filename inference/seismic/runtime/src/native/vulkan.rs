//! Vulkan submission of native dispatch lists (Vulkan backend spec §8.6),
//! the counterpart of `cuda.rs`.
//!
//! A submission of sealed graph runs (one run, or a sequence of runs
//! submitted together) replays a secondary command buffer: its launches,
//! with every argument block fixed, recorded once and executed inside the
//! submission's primary buffer. Recorded graphs are keyed by the plans and
//! every buffer address the runs bind; per-step host values live in each
//! run's upload region, which the argument blocks address, never in the
//! blocks themselves.
//!
//! Standalone calls, and every launch of a launch-detail trace (timed one by
//! one), are recorded launch by launch.

use super::graph_replays::Replays;
use super::{DispatchList, NativeRoute, RouteSubmission};
use crate::api::CallError;
use crate::backends::vulkan_buffer;
use crate::driver::Allocation;
use seismic_compiler::errors::ExecutionError;
use seismic_vulkan::direct::{DirectBatch, DirectGraph, DirectGraphBuilder, DirectLaunch};
use std::any::Any;
use std::sync::Arc;

/// A device's recorded graphs of sealed-plan submissions.
pub(crate) type VulkanReplays = Replays<DirectGraph>;

/// Encode `list` on the device's queue: a replay when the list is made of
/// sealed plans and is not timed, else launch by launch. `timed` carries
/// the launch count of a launch-detail trace.
pub(super) fn encode(
    device: &seismic_vulkan::Device,
    replays: &VulkanReplays,
    list: &impl DispatchList,
    repetitions: usize,
    timed: Option<usize>,
    retained: &Arc<dyn Any + Send + Sync>,
) -> Result<RouteSubmission, CallError> {
    let mut batch = match timed {
        Some(launches) => DirectBatch::timed(device, launches),
        None => DirectBatch::new(device),
    }
    .map_err(CallError::Execution)?;
    match list.plans().filter(|_| timed.is_none()) {
        Some(mut key) => {
            key.extend(addresses(list));
            replays.replay(
                key,
                retained,
                || {
                    let mut builder =
                        DirectGraphBuilder::new(device).map_err(CallError::Execution)?;
                    each_launch(list, |launch| match launch {
                        Some(launch) => builder.launch(launch),
                        None => Ok(()),
                    })?;
                    builder.instantiate().map_err(CallError::Execution)
                },
                |graph| {
                    for _ in 0..repetitions {
                        batch.replay(graph).map_err(CallError::Execution)?;
                    }
                    Ok(())
                },
            )?;
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
        .map(RouteSubmission::Vulkan)
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
        let NativeRoute::Vulkan { module, .. } = &dispatch.kernel.route else {
            unreachable!("one device has one native route");
        };
        typed.clear();
        typed.extend(
            buffers
                .iter()
                .map(|(allocation, offset)| (vulkan_buffer(allocation), *offset)),
        );
        let scalars = vulkan_buffer(&dispatch.kernel.scalars);
        for (function, launch) in dispatch.launches.iter().enumerate() {
            let launch = launch.map(|launch| DirectLaunch {
                module,
                function,
                buffers: &typed,
                words: dispatch.word_bytes,
                scalar_results: (scalars, 0),
                groups: launch.groups,
            });
            visit(launch.as_ref()).map_err(CallError::Execution)?;
        }
    }
    Ok(())
}

/// Every buffer address one pass over `list` binds, in dispatch order: with
/// the sealed plans, they fix every argument block.
fn addresses(list: &impl DispatchList) -> Vec<u64> {
    let mut buffers: Vec<(&Allocation, u64)> = Vec::new();
    for index in 0..list.count() {
        list.dispatch(index, &mut buffers);
    }
    buffers
        .iter()
        .map(|(allocation, offset)| vulkan_buffer(allocation).address() + offset)
        .collect()
}
