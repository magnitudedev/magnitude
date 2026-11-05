//! CUDA submission of native dispatch lists.
//!
//! A submission of sealed graph runs (one run, or a sequence of runs
//! submitted together) replays CUDA graphs: one per run's calls between
//! timeline boundaries, launched with one driver call. A graph is keyed by
//! its shape (`ReplayKeys`): the run's layout, the calls' place in it and
//! the launches its slot's graph of the layout holds (`LayoutForms`: every
//! launch any plan of the layout the slot runs makes), never the class or
//! the storage they bind. Its arguments are the plan's sealed words and
//! geometry plus the buffer addresses the run binds, and its nodes are
//! enabled where the run makes and encodes their launches; a replay binding
//! other parameters than the graph's sets the changed launches' first
//! (`DirectGraph::bind`). Per-step values never live in
//! arguments: host-written inputs are written into each run's upload region,
//! which the graph addresses. A graph's launches run in stream order, a
//! chain.
//!
//! Graphs are instantiated only as a graph slot is created
//! ([`instantiate`]), a graph of each layout for each run of it its launch
//! makes, over placeholder storage; serving takes them (`Replays::record`)
//! and never instantiates. A slot is created only between flights, with no device
//! work queued: the driver holds the context while instantiation waits on
//! the device, so behind a parked kernel it would hold every other thread's
//! driver calls, a park's answer among them (a transfer), until the park
//! gave up. A run takes a graph bound to its storage, else one whose
//! launches completed (`DirectGraph::idle`), so setting its parameters
//! waits for no device work.
//!
//! Standalone calls and every launch of a launch-detail trace (timed one by
//! one) run launch by launch, in stream order. A launched plan's run
//! (`NativeGraphDraft::seal_launched`: per-use work whose shapes no load
//! enumerates) runs launch by launch in its place among the graphs of its
//! submission, and its slots instantiate nothing.
//!
//! A span mark inside a graph (`spans`) is a node of it, in its chain, which
//! a sampled run sets to stamp its events, its identity part of what the
//! graph binds.
//!
//! Native work runs on one of two streams: compute, or fill beside it
//! (`NativeQueue::Fill`), whose moves overlap compute and whose spinning
//! waits hold only later fills. A submission's stream first waits on the
//! end events of the other stream's submissions it must follow by what it
//! touches (`Allocation::queue_order`, `cuStreamWaitEvent`).
//!
//! A timeline boundary (`timeline`) divides the calls: those between two
//! boundaries replay a graph of their own, and at the boundary the stream
//! waits on a gate's host word (`cuStreamWaitValue32`) or writes a signal's
//! (`cuStreamWriteValue32`).

use super::graph_replays::{Replays, Shaped};
use super::timeline::{TimelineAction, TimelineKind, TimelineOperation};
use super::{graph_segments, DispatchList, EncodeFailure, Encoding, NativeRoute, ReplayKeys};
use crate::completion::Committed;
use seismic_cuda::NativeStream;
use std::ops::Range;
use crate::api::CallError;
use crate::driver::{Allocation, typed_buffer};
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::GraphCensus;
use seismic_cuda::direct::{DirectBatch, DirectGraph, DirectGraphBuilder, DirectLaunch};
use std::any::Any;
use std::sync::Arc;

type Cuda = seismic_cuda::Cuda;
type Executor = seismic_cuda::Executor;

/// A device's CUDA graphs of sealed-plan submissions.
pub(crate) type CudaReplays = Replays<DirectGraph>;

impl super::graph_replays::RecordedGraph for DirectGraph {
    fn census(&self) -> GraphCensus {
        DirectGraph::census(self)
    }
    fn idle(&self) -> Result<bool, CallError> {
        DirectGraph::idle(self).map_err(CallError::Execution)
    }
    /// An update of a graph in flight waits for it on the host (20.5 µs a
    /// launch against 2.0 µs, `graph_cost::graph_update_cost_probe`).
    fn updates_in_flight() -> bool {
        seismic_lang::registry::BackendName::Cuda.graphs_update_in_flight()
    }
}

/// Encode `work` on its queue's stream (the compute stream, or the fill
/// stream beside it), after the work of the other stream it follows: a
/// replay when the list is made of sealed plans and is not timed, else
/// launch by launch, its graphs kept within `budget`. A failure after work
/// was issued abandons the batch: the issued work runs on, fenced by the
/// abandoned submission.
pub(super) fn encode(
    device: &seismic_cuda::Device,
    replays: &CudaReplays,
    budget: GraphCensus,
    work: &Encoding<'_, impl DispatchList>,
) -> Result<Committed, EncodeFailure> {
    let stream = match work.queue {
        super::NativeQueue::Compute => NativeStream::Compute,
        super::NativeQueue::Fill => NativeStream::Fill,
    };
    let mut batch = if work.timed.is_some() {
        DirectBatch::timed(device, stream)
    } else {
        DirectBatch::new(device, stream)
    }
    .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    // Nothing is issued yet: a failed wait abandons no work.
    for fence in work.after {
        let route = fence.native().expect("a native queue's fence is native work");
        let Committed::Cuda(submission) = route.committed() else {
            unreachable!("one device's native work is of its backend");
        };
        batch
            .follow(submission)
            .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    }
    match issue(&mut batch, device, replays, budget, work) {
        Ok(()) => batch
            .commit()
            .map(Committed::Cuda)
            .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error))),
        Err(error) => Err(match batch.abandon() {
            Ok(fenced) => EncodeFailure {
                error,
                committed: Some(Committed::Cuda(fenced)),
            },
            // The stream was waited for, or the context lets nothing more
            // run: nothing issued runs on.
            Err(fence) => EncodeFailure::uncommitted(CallError::Unfenced {
                issue: Box::new(error),
                fence,
            }),
        }),
    }
}

/// Issue `work` on `batch`'s stream.
fn issue(
    batch: &mut DirectBatch,
    device: &seismic_cuda::Device,
    replays: &CudaReplays,
    budget: GraphCensus,
    work: &Encoding<'_, impl DispatchList>,
) -> Result<(), CallError> {
    let Encoding { list, repetitions, retained, .. } = *work;
    let timed = work.timed.is_some();
    // Which launches the run encodes, by their conditions' words as the host
    // holds them now: a graph records every launch a call can make, and
    // runs those encoded.
    let encoded = list.encoded();
    match list.runs().filter(|_| !timed) {
        Some(runs) => {
            // Each run's calls between timeline nodes are one graph of their
            // shape, instantiated with the run's slot
            // ([`instantiate`]), except a launched plan's, launched call by
            // call in their place; the stream waits on or writes the
            // boundaries' words between graphs.
            let segments = graph_segments(list, Some(&runs));
            let keys = ReplayKeys { runs, encoded: None };
            let (graphed, shaped): (Vec<Range<usize>>, Vec<Shaped>) = segments
                .iter()
                .filter(|(calls, _)| !calls.is_empty() && !keys.launched(calls))
                .map(|(calls, _)| (calls.clone(), shaped(list, &keys, calls, &encoded)))
                .unzip();
            // Serving never instantiates: a graph its slots lack is a gap in
            // their census.
            let taken = replays.record(&shaped, budget, retained, |index| {
                panic!(
                    "a CUDA run of shape {:?} has no graph its slots instantiated: their census lacks it ({budget:?})",
                    shaped[index].key
                )
            })?;
            for _ in 0..repetitions {
                let mut graphs = graphed.iter().zip(&shaped).zip(&taken);
                for (calls, boundary) in &segments {
                    if !calls.is_empty() && keys.launched(calls) {
                        launch(batch, list, calls.clone(), &encoded)?;
                    } else if !calls.is_empty() {
                        let ((calls, shaped), &identity) = graphs.next().expect("a graph per segment of calls");
                        replays.replay(
                            shaped,
                            identity,
                            // The graph's launches, in the order its nodes
                            // were added, bound to this run's storage and
                            // run where encoded.
                            |graph| {
                                let form = keys.form(calls);
                                let mut node = 0;
                                each_launch(list, calls.clone(), &encoded, |index, ordinal, launch, runs| {
                                    if !holds(form[index - calls.start], ordinal) {
                                        assert!(launch.is_none(), "a run makes only launches its layout's graph holds");
                                        return Ok(());
                                    }
                                    match launch {
                                        Some(launch) => graph.bind(device, node, launch, runs)?,
                                        // Another plan of the layout makes it.
                                        None => graph.disable(node)?,
                                    }
                                    node += 1;
                                    Ok(())
                                })
                                .map_err(CallError::Execution)?;
                                assert_eq!(node, graph.nodes(), "a shape's launches are its graph's nodes");
                                let stamped = keys.stamps(list, calls).into_iter().map(|stamp| (cuda_stamps(stamp), stamp.mark));
                                graph.sample(&stamped.collect::<Vec<_>>()).map_err(CallError::Execution)
                            },
                            |graph| batch.replay(graph).map_err(CallError::Execution),
                        )?;
                    }
                    stamps(batch, list.stamps(*boundary))?;
                    timelines(batch, list.timelines(*boundary))?;
                }
            }
        }
        None => {
            for _ in 0..repetitions {
                for (calls, boundary) in &graph_segments(list, None) {
                    launch(batch, list, calls.clone(), &encoded)?;
                    stamps(batch, list.stamps(*boundary))?;
                    timelines(batch, list.timelines(*boundary))?;
                }
            }
        }
    }
    Ok(())
}

/// Launch the calls `calls` of `list` on `batch`'s stream one by one, each
/// launch where `encoded`.
fn launch(batch: &mut DirectBatch, list: &impl DispatchList, calls: Range<usize>, encoded: &[Vec<bool>]) -> Result<(), CallError> {
    each_launch(list, calls, encoded, |_, _, launch, runs| match launch.filter(|_| runs) {
        Some(launch) => batch.launch(launch),
        None => {
            batch.skip();
            Ok(())
        }
    })
    .map_err(CallError::Execution)
}

/// The graph of the segment `calls` of `list`'s run: keyed by its run's
/// layout, its place in the run and the launches the layout's graph holds
/// of each call (`LayoutForms`), bound to every parameter a replay sets
/// (the run's plan, its launches' functions, geometry and words; the
/// launches it encodes; the buffers' addresses; the stamps its marks record).
fn shaped(list: &impl DispatchList, keys: &ReplayKeys<'_>, calls: &Range<usize>, encoded: &[Vec<bool>]) -> Shaped {
    let mut bound = vec![keys.plan(calls)];
    bound.extend(super::encoded_words(encoded, calls.clone()));
    bound.extend(addresses(list, calls.clone()));
    bound.push(keys.stamps(list, calls).first().map_or(0, |stamp| cuda_stamps(stamp).identity()));
    keys.shaped(calls, keys.form(calls).iter().copied(), bound)
}

/// Whether a call's word of a layout's graph (`LayoutForms`) holds launch
/// `ordinal`.
fn holds(word: u64, ordinal: usize) -> bool {
    word >> ordinal & 1 == 1
}

/// Record the graph of the segment `calls` of the plan runs `lists`, which
/// are of one layout: a node for each launch `form` holds of a call (a word
/// per call of the plan), each following the one before. A launch is the
/// first run's, run where it encodes it and does device work, or where
/// that run does not make it, another's that does, disabled until a run
/// making it binds it ([`DirectGraph::bind`]). How a node follows the one
/// before is its kernel entry's, alike for every plan of a layout. A node
/// more stands at each of `marks`, the calls a mark the graph holds precedes,
/// in order.
fn record<L: DispatchList>(
    device: &seismic_cuda::Device,
    lists: &[L],
    calls: Range<usize>,
    form: &[u64],
    marks: &[usize],
) -> Result<DirectGraph, ExecutionError> {
    let encoded = lists[0].encoded();
    let mut builder = DirectGraphBuilder::new(device)?;
    let mut nodes = 0usize;
    let mut marks = marks.iter().peekable();
    for index in calls {
        while marks.next_if_eq(&&index).is_some() {
            builder.mark()?;
        }
        for ordinal in (0..u64::BITS as usize).filter(|&ordinal| holds(form[index], ordinal)) {
            let previous = nodes.checked_sub(1);
            let added = lists
                .iter()
                .enumerate()
                .find_map(|(made, list)| {
                    with_launch(list, index, ordinal, |launch| {
                        let runs = made == 0 && encoded[index][ordinal] && !launch.is_empty();
                        builder.launch(launch, previous.as_slice(), runs)
                    })
                })
                .expect("a layout's graph holds only launches a plan of it makes");
            added?;
            nodes += 1;
        }
    }
    builder.instantiate()
}

/// Instantiate `count` graphs of each segment of `lists[0]`, one sealed
/// plan's run over placeholder storage, for a slot of its family as it is
/// created ([`super::SlotGraphs`]): each a graph of the plan's layout whose
/// nodes are the launches `form` holds, which the plans `lists` of that
/// layout make ([`record`]), each pushed to `owned`, the slot's, which
/// releases it as it drops. A replay binds a graph to its run's storage
/// before it runs. A slot is created only between flights, with no device
/// work queued: instantiation holds the context while the driver waits on
/// the device, which a kernel parked on a host answer made by driver calls
/// would hang, so one with work queued is an invariant violation.
pub(super) fn instantiate<L: DispatchList>(
    device: &seismic_cuda::Device,
    replays: &CudaReplays,
    budget: GraphCensus,
    lists: &[L],
    count: usize,
    retained: &Arc<dyn Any + Send + Sync>,
    owned: &mut Vec<(Vec<u64>, u64)>,
) -> Result<(), ExecutionError> {
    assert!(
        device.idle()?,
        "a graph slot is created only between flights, with no device work queued"
    );
    let list = &lists[0];
    let encoded = list.encoded();
    let runs = list.runs().expect("a slot's graphs are a sealed plan's");
    let form = runs[0].form.clone();
    let segments = graph_segments(list, Some(&runs));
    let keys = ReplayKeys { runs, encoded: None };
    for (calls, _) in segments.iter().filter(|(calls, _)| !calls.is_empty()) {
        let key = shaped(list, &keys, calls, &encoded).key;
        let marks = keys.marks(calls).collect::<Vec<_>>();
        for _ in 0..count {
            let graph = record(device, lists, calls.clone(), &form, &marks)?;
            owned.push((key.clone(), replays.instantiated(key.clone(), graph, budget, retained)));
        }
    }
    Ok(())
}

/// Stamp a boundary's marks on the stream, once the work before them
/// completes (`spans`).
fn stamps(batch: &mut DirectBatch, stamps: &[super::spans::Stamp]) -> Result<(), CallError> {
    for stamp in stamps {
        batch.stamp(cuda_stamps(stamp), stamp.mark).map_err(CallError::Execution)?;
    }
    Ok(())
}

/// A stamp's CUDA stamps.
fn cuda_stamps(stamp: &super::spans::Stamp) -> &seismic_cuda::direct::Stamps {
    // Where CUDA is the only backend with stamps, the pattern is irrefutable.
    #[allow(irrefutable_let_patterns)]
    let super::spans::Stamps::Cuda(stamps) = &*stamp.stamps else {
        unreachable!("a run's stamps are of its own device");
    };
    stamps
}

/// Queue a boundary's timeline operations on the stream.
fn timelines(batch: &mut DirectBatch, operations: &[TimelineOperation]) -> Result<(), CallError> {
    for operation in operations {
        let TimelineKind::Cuda(timeline) = operation.timeline.kind() else {
            unreachable!("a run binds timelines of its own device");
        };
        // The word holds a value's low 32 bits (`timeline`).
        let value = operation.value as u32;
        match operation.action {
            TimelineAction::Wait => batch.wait_word(&timeline.gate, value),
            TimelineAction::Signal => batch.write_word(&timeline.reached, value),
        }
        .map_err(CallError::Execution)?;
    }
    Ok(())
}

/// Visit every launch of the calls `calls` of one pass over `list`, by call
/// and declaration ordinal, in order: `None` for one whose `when` condition
/// does not hold (`CallLaunches::geometry`), with whether it runs, encoded
/// (`encoded`) and of device work.
fn each_launch(
    list: &impl DispatchList,
    calls: Range<usize>,
    encoded: &[Vec<bool>],
    mut visit: impl FnMut(usize, usize, Option<&DirectLaunch<'_>>, bool) -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    let (mut buffers, mut typed) = (Vec::new(), Vec::new());
    for index in calls {
        call_launches(list, index, &mut buffers, &mut typed, |ordinal, launch| {
            let runs = launch.is_some_and(|launch| encoded[index][ordinal] && !launch.is_empty());
            visit(index, ordinal, launch, runs)
        })?;
    }
    Ok(())
}

/// `make` of launch `ordinal` of call `index` of `list`, where the call
/// makes it (its `when` condition holds); else none.
fn with_launch<R>(list: &impl DispatchList, index: usize, ordinal: usize, make: impl FnOnce(&DirectLaunch<'_>) -> R) -> Option<R> {
    let mut make = Some(make);
    let mut made = None;
    call_launches(list, index, &mut Vec::new(), &mut Vec::new(), |at, launch| {
        if at == ordinal {
            made = launch.map(|launch| make.take().expect("a call's launch is visited once")(launch));
        }
        Ok::<_, std::convert::Infallible>(())
    })
    .unwrap_or_else(|never| match never {});
    made
}

/// Visit each launch of call `index` of `list` by declaration ordinal, in
/// order, `None` for one it does not make, its buffers bound in `buffers`
/// and `typed` (kept by the caller across calls).
fn call_launches<'s, E>(
    list: &'s impl DispatchList,
    index: usize,
    buffers: &mut Vec<(&'s Allocation, u64)>,
    typed: &mut Vec<(&'s crate::driver::Buffer<Cuda, Executor>, u64)>,
    mut visit: impl FnMut(usize, Option<&DirectLaunch<'_>>) -> Result<(), E>,
) -> Result<(), E> {
    buffers.clear();
    let dispatch = list.dispatch(index, buffers);
    let NativeRoute::Cuda { launches, .. } = &dispatch.kernel.route else {
        unreachable!("one device has one native route");
    };
    typed.clear();
    typed.extend(buffers.iter().map(|(allocation, offset)| (typed_buffer::<Cuda, Executor>(allocation), *offset)));
    let scalars = dispatch
        .kernel
        .scalar_slots()
        .map(|slots| (typed_buffer::<Cuda, Executor>(slots), 0));
    for (ordinal, launch) in dispatch.launches.iter().enumerate() {
        let Some(launch) = launch else {
            visit(ordinal, None)?;
            continue;
        };
        let (module, function) = launches.launch(ordinal);
        visit(
            ordinal,
            Some(&DirectLaunch {
                module,
                function,
                buffers: typed,
                words: super::word_bytes(dispatch.words),
                scalar_results: scalars,
                grid: launch.groups,
                block: launch.threads,
                shared_bytes: launch.shared_bytes,
            }),
        )?;
    }
    Ok(())
}

/// Which launches a call makes (those whose `when` condition holds), as a
/// word of a CUDA graph's shape: bit `ordinal` set for each, a node of the
/// graph. What else a launch is, its function, geometry and arguments, a
/// replay binds (`DirectGraph::bind`); how a node follows the one before
/// (`DirectModule::programmatic`) is its kernel entry's, alike for every
/// class of a layout, which binding asserts.
pub(super) fn launch_structure(geometry: &[Option<super::LaunchGeometry>], into: &mut Vec<u64>) {
    assert!(geometry.len() <= 64, "a call declares at most 64 launches");
    into.push(
        geometry
            .iter()
            .enumerate()
            .filter(|(_, launch)| launch.is_some())
            .fold(0, |made, (ordinal, _)| made | 1 << ordinal),
    );
}

/// Every buffer address `calls` of `list` bind, in dispatch order: with the
/// sealed plans, they fix every launch argument of their graph.
fn addresses(list: &impl DispatchList, calls: Range<usize>) -> Vec<u64> {
    let mut buffers: Vec<(&Allocation, u64)> = Vec::new();
    for index in calls {
        list.dispatch(index, &mut buffers);
    }
    buffers
        .iter()
        .map(|(allocation, offset)| typed_buffer::<Cuda, Executor>(allocation).pointer() + offset)
        .collect()
}
