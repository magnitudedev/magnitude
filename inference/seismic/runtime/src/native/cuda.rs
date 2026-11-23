//! CUDA submission of native dispatch lists.
//!
//! A submission of sealed graph runs (one run, or a sequence of runs
//! submitted together) replays CUDA graphs: one per run, launched with one
//! driver call, which holds the timeline nodes inside the run as nodes of
//! its own. A graph is keyed by
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
//! A timeline node inside a run (`graph_segments`) is a node of its graph,
//! in its chain (`timeline_nodes`): a wait on a gate's host word, or a write
//! of a signal's, each bound to the run's word and value as its launches
//! are (`DirectGraph::bind_word`), so a block's run is one graph whatever
//! parks, publications and waits it holds. A write's waiters wake as it
//! lands (`DirectBatch::replay`).
//!
//! A graph is instantiated as a run first needs one ([`record_run`]), of
//! its layout over placeholder storage, only once the device's queued work
//! has drained (`Replays::record`'s `prepare`): the driver holds the context
//! while instantiation waits on the device, so behind a parked kernel it
//! would hold every other thread's driver calls, a park's answer among them
//! (a transfer), until the park gave up. A run takes a graph bound to its
//! storage, else one whose launches completed (`DirectGraph::idle`), so
//! setting its parameters waits for no device work.
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
//! A timeline boundary between runs, or in a list launched call by call
//! (`timeline`), divides the calls: at it the stream waits on a gate's host
//! word (`cuStreamWaitValue32`) or writes a signal's
//! (`cuStreamWriteValue32`).

use super::graph_replays::{Replays, Shaped};
use super::timeline::{TimelineAction, TimelineKind, TimelineNode, TimelineOperation};
use super::{graph_segments, DispatchList, EncodeFailure, Encoding, NativeRoute, ReplayKeys};
use crate::api::CallError;
use crate::completion::{Committed, DeviceCompletion};
use crate::driver::{Allocation, typed_buffer};
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::GraphCensus;
use seismic_cuda::direct::{DirectBatch, DirectGraph, DirectGraphBuilder, DirectLaunch, DirectSubmission, HostWord};
use seismic_cuda::NativeStream;
use seismic_lang::registry::BackendName;
use std::any::Any;
use std::ops::Range;
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
    /// Recorded over placeholder storage ([`record_run`]).
    fn recorded_bound() -> bool {
        false
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
    follow_fences(&mut batch, work.after)
        .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    for sync in work.after_transfers {
        if let Some(event) = sync.cuda_event() {
            batch
                .follow_event(event)
                .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
        }
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

/// Encode ordered moves (`super::moves`) on the compute stream, after
/// `after` (the fill stream's work they follow), each step's ends at the
/// device addresses `address` names: refused where none of them was
/// issued.
///
/// # Panics
/// Where issuing failed after a step was issued (`super::moves::submit`).
pub(super) fn encode_moves(
    device: &seismic_cuda::Device,
    after: &[Arc<dyn DeviceCompletion>],
    steps: &[super::moves::MoveStep<'_>],
    address: impl Fn(&Arc<Allocation>, u64) -> u64,
) -> Result<Committed, ExecutionError> {
    use super::moves::MoveStep;
    let mut batch = DirectBatch::new(device, NativeStream::Compute)?;
    follow_fences(&mut batch, after)?;
    for (index, step) in steps.iter().enumerate() {
        let issued = match *step {
            MoveStep::Copy { source, source_offset, destination, destination_offset, bytes } => {
                batch.copy(address(destination, destination_offset), address(source, source_offset), bytes)
            }
            MoveStep::Word { destination, offset, value } => batch.write_u64(address(destination, offset), value),
        };
        if let Err(error) = issued {
            assert!(index == 0, "invariant violated: ordered moves failed after {index} of them were issued: {error}");
            // Only waits on the fill stream were issued: the batch ends
            // behind them.
            drop(batch.abandon());
            return Err(error);
        }
    }
    let committed = batch
        .commit()
        .unwrap_or_else(|error| panic!("invariant violated: ordered moves were issued but their end was not recorded: {error}"));
    Ok(Committed::Cuda(committed))
}

/// Encode a batch of host-to-device (`cuMemcpyHtoDAsync`) and device-to-device
/// (`cuMemcpyDtoDAsync`) byte copies on `NativeStream::Fill`, after `after`
/// (cross-queue fences from `Allocation::queue_order`), ending with
/// `cuEventRecord` (`DirectBatch::commit`) so subsequent compute work reading
/// the written allocations waits via `DirectBatch::follow` (`cuStreamWaitEvent`).
pub(super) fn encode_fill_copies(
    device: &seismic_cuda::Device,
    after: &[Arc<dyn DeviceCompletion>],
    runs: &[super::FillCopyRun<'_>],
) -> Result<Committed, EncodeFailure> {
    enum CheckedCopy {
        Upload { source: *const u8, destination: u64, bytes: u64 },
        Within { source: u64, destination: u64, bytes: u64 },
    }
    let refused = |why: &str| EncodeFailure::uncommitted(CallError::Execution(ExecutionError::SubmissionFailed(why.into())));
    let mut checked = Vec::with_capacity(runs.len());
    for run in runs {
        if run.bytes == 0 {
            continue;
        }
        let src_buf = typed_buffer::<Cuda, Executor>(run.source.allocation());
        let dst_buf = typed_buffer::<Cuda, Executor>(run.destination.allocation());
        let src_off = run.source.byte_offset() + run.source_offset;
        let dst_off = run.destination.byte_offset() + run.destination_offset;
        let dst_ptr = dst_buf.pointer() + dst_off;
        let src_host = run.source.allocation().storage().host_address();
        let dst_host = run.destination.allocation().storage().host_address();
        match (src_host, dst_host) {
            (Some(mapped), None) => {
                let offset = usize::try_from(src_off).map_err(|_| refused("fill copy source offset exceeds host address space"))?;
                // SAFETY: `src_off + run.bytes <= source.byte_len()` was verified by the caller.
                let source = unsafe { mapped.as_ptr().add(offset) };
                checked.push(CheckedCopy::Upload { source, destination: dst_ptr, bytes: run.bytes });
            }
            (None, None) => {
                checked.push(CheckedCopy::Within {
                    source: src_buf.pointer() + src_off,
                    destination: dst_ptr,
                    bytes: run.bytes,
                });
            }
            (_, Some(_)) => {
                return Err(refused("fill DMA destination must be device memory"));
            }
        }
    }
    let mut batch = DirectBatch::new(device, NativeStream::Fill)
        .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    follow_fences(&mut batch, after)
        .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    for step in checked {
        let issued = match step {
            // SAFETY: caller retains every source and destination allocation until completion.
            CheckedCopy::Upload { source, destination, bytes } => unsafe { batch.upload(destination, source, bytes) },
            CheckedCopy::Within { source, destination, bytes } => batch.copy(destination, source, bytes),
        };
        if let Err(error) = issued {
            let error = CallError::Execution(error);
            return Err(match batch.abandon() {
                Ok(fenced) => EncodeFailure { error, committed: Some(Committed::Cuda(fenced)) },
                Err(fence) => EncodeFailure::uncommitted(CallError::Unfenced { issue: Box::new(error), fence }),
            });
        }
    }
    batch
        .commit()
        .map(Committed::Cuda)
        .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))
}

/// One segment of a sealed-plan submission: its work (if any) and the
/// boundary after it whose stamps and timelines follow.
struct Segment<'l> {
    work: SegmentWork<'l>,
    boundary: usize,
}

enum SegmentWork<'l> {
    Empty,
    Direct(Range<usize>),
    Graphed(GraphedSegment<'l>),
}

/// A graphed run segment resolved to its shape and recorded graph identity.
struct GraphedSegment<'l> {
    calls: Range<usize>,
    words: Vec<(&'l HostWord, u32)>,
    shaped: Shaped,
    identity: u64,
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
            // Each run's calls are one graph of their shape, its timeline
            // nodes inside it, instantiated with the run's slot
            // ([`instantiate`]), except a launched plan's, launched call by
            // call in their place; the stream waits on or writes the words
            // of the boundaries between graphs.
            let keys = ReplayKeys { runs, encoded: None };
            let segments = plan_segments(device, replays, budget, list, &keys, &encoded, retained)?;
            for _ in 0..repetitions {
                for segment in &segments {
                    match &segment.work {
                        SegmentWork::Empty => {}
                        SegmentWork::Direct(calls) => launch(batch, list, calls.clone(), &encoded)?,
                        SegmentWork::Graphed(graph) => replay_segment(batch, device, replays, list, &keys, &encoded, graph)?,
                    }
                    stamps(batch, list.stamps(segment.boundary))?;
                    timelines(batch, list.timelines(segment.boundary))?;
                }
            }
        }
        None => {
            let segments = graph_segments(list, None);
            for _ in 0..repetitions {
                for (calls, boundary) in &segments {
                    launch(batch, list, calls.clone(), &encoded)?;
                    stamps(batch, list.stamps(*boundary))?;
                    timelines(batch, list.timelines(*boundary))?;
                }
            }
        }
    }
    Ok(())
}

/// Classify each segment of a sealed-plan submission once, record any missing
/// layout graphs, and bind every graphed segment to its resolved graph
/// identity.
fn plan_segments<'l>(
    device: &seismic_cuda::Device,
    replays: &CudaReplays,
    budget: GraphCensus,
    list: &'l impl DispatchList,
    keys: &ReplayKeys<'_>,
    encoded: &[Vec<bool>],
    retained: &Arc<dyn Any + Send + Sync>,
) -> Result<Vec<Segment<'l>>, CallError> {
    enum Pending<'l> {
        Empty,
        Direct(Range<usize>),
        Graphed {
            calls: Range<usize>,
            words: Vec<(&'l HostWord, u32)>,
        },
    }

    let raw = graph_segments(list, Some(&keys.runs));
    let mut pending = Vec::with_capacity(raw.len());
    let mut graphed_layouts = Vec::new();
    let mut shaped_graphs = Vec::new();
    for (calls, boundary) in raw {
        let work = if calls.is_empty() {
            Pending::Empty
        } else if keys.launched(&calls) {
            assert!(
                !keys.run(&calls).holds_timelines,
                "a direct (launched) run must not hold timelines inside an unsplit segment"
            );
            Pending::Direct(calls)
        } else {
            let run = keys.run(&calls);
            assert_eq!(calls, run.calls, "a graphed run is one graph");
            let words = held_operations(list, keys, &calls);
            shaped_graphs.push(shaped(list, keys, &calls, encoded, &words));
            graphed_layouts.push(&run.graph);
            Pending::Graphed { calls, words }
        };
        pending.push((work, boundary));
    }
    // A run that finds no graph of its shape it may take records its
    // layout's, once the device's queued work has drained.
    let taken = replays.record(
        &shaped_graphs,
        budget,
        retained,
        || device.drain().map_err(CallError::Execution),
        |index| {
            let (key, graph) = super::graph::record_layout(graphed_layouts[index]).map_err(CallError::Execution)?;
            assert_eq!(key, shaped_graphs[index].key, "a run's graph is its layout's");
            Ok(graph)
        },
    )?;
    let mut resolved = shaped_graphs.into_iter().zip(taken);
    let segments = pending
        .into_iter()
        .map(|(work, boundary)| {
            let work = match work {
                Pending::Empty => SegmentWork::Empty,
                Pending::Direct(calls) => SegmentWork::Direct(calls),
                Pending::Graphed { calls, words } => {
                    let (shaped, identity) = resolved
                        .next()
                        .expect("one recorded graph per graphed segment");
                    SegmentWork::Graphed(GraphedSegment { calls, words, shaped, identity })
                }
            };
            Segment { work, boundary }
        })
        .collect();
    debug_assert!(resolved.next().is_none());
    Ok(segments)
}

/// Replay one graphed segment on `batch`'s stream, rebinding its launches
/// where its storage or encoded launches changed and its word nodes to this
/// run's words.
fn replay_segment(
    batch: &mut DirectBatch,
    device: &seismic_cuda::Device,
    replays: &CudaReplays,
    list: &impl DispatchList,
    keys: &ReplayKeys<'_>,
    encoded: &[Vec<bool>],
    segment: &GraphedSegment<'_>,
) -> Result<(), CallError> {
    replays.replay(
        &segment.shaped,
        segment.identity,
        // The graph's launches, in the order its nodes were added, bound to
        // this run's storage and run where encoded, unless they bind what
        // they bound before; then its words, to this run's.
        |graph, before| {
            let launches = segment.shaped.bound.len() - 2 * segment.words.len();
            if before.is_none_or(|before| before.get(..launches) != segment.shaped.bound.get(..launches)) {
                bind_launches(device, graph, list, keys, &segment.calls, encoded)?;
            }
            assert_eq!(segment.words.len(), graph.words(), "a run's timeline nodes are its graph's words");
            for (index, &(word, value)) in segment.words.iter().enumerate() {
                graph.bind_word(index, word, value).map_err(CallError::Execution)?;
            }
            Ok(())
        },
        |graph| batch.replay(graph).map_err(CallError::Execution),
    )
}

/// Bind `graph`'s launches, in the order its nodes were added, to the run
/// of `calls` of `list`: its storage, each run where encoded, and the
/// stamps of a sampled run at its marks.
fn bind_launches(
    device: &seismic_cuda::Device,
    graph: &mut DirectGraph,
    list: &impl DispatchList,
    keys: &ReplayKeys<'_>,
    calls: &Range<usize>,
    encoded: &[Vec<bool>],
) -> Result<(), CallError> {
    let form = keys.form(calls);
    let mut node = 0;
    each_launch(list, calls.clone(), encoded, |index, ordinal, launch, runs| {
        if !holds(form[index - calls.start], ordinal) {
            assert!(
                launch.is_none_or(|launch| launch.is_empty()),
                "a run makes only non-empty launches its layout's graph holds"
            );
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
}

/// The nodes a CUDA graph holds of a timeline node inside it
/// (`graph_segments`): a wait's one, a signal's write and the event its
/// word's waiters wake behind (`DirectGraphBuilder::write_word`); none of a
/// park's or a publication's, whose kernels park themselves.
pub(super) fn timeline_nodes(node: TimelineNode) -> u64 {
    const _: () = assert!(BackendName::Cuda.parks_in_kernels());
    match node {
        TimelineNode::Wait => 1,
        TimelineNode::Signal => 2,
        TimelineNode::Park | TimelineNode::Publish => 0,
    }
}

/// The calls of `list` each timeline node the graph of `calls` holds
/// precedes, in order, once each: those of a wait or a signal inside it
/// (`PlanRun::held_timelines`).
fn held_boundaries(keys: &ReplayKeys<'_>, calls: &Range<usize>) -> Vec<usize> {
    let mut boundaries = keys
        .run(calls)
        .held_timelines(calls)
        .filter(|&(_, node)| timeline_nodes(node) != 0)
        .map(|(at, _)| at)
        .collect::<Vec<_>>();
    boundaries.dedup();
    boundaries
}

/// What the run binds the words of the graph of `calls` of `list` to, in
/// the order its word nodes were added: each timeline operation at a node
/// it holds, its host word (a gate's, or a signal's reached word) and the
/// value's low 32 bits (`timeline`).
fn held_operations<'l>(list: &'l impl DispatchList, keys: &ReplayKeys<'_>, calls: &Range<usize>) -> Vec<(&'l HostWord, u32)> {
    held_boundaries(keys, calls)
        .into_iter()
        .flat_map(|boundary| list.timelines(boundary))
        .map(cuda_timeline_word)
        .collect()
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
/// launches it encodes; the buffers' addresses; the stamps its marks
/// record), and last, two words each, its word nodes' words and values
/// (`words`, [`held_operations`]).
fn shaped(list: &impl DispatchList, keys: &ReplayKeys<'_>, calls: &Range<usize>, encoded: &[Vec<bool>], words: &[(&HostWord, u32)]) -> Shaped {
    let mut bound = vec![keys.plan(calls)];
    bound.extend(super::encoded_words(encoded, calls.clone()));
    bound.extend(addresses(list, calls.clone()));
    bound.push(keys.stamps(list, calls).first().map_or(0, |stamp| cuda_stamps(stamp).identity()));
    bound.extend(words.iter().flat_map(|&(word, value)| [word.device_address(), u64::from(value)]));
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
/// before is its kernel entry's, alike for every plan of a layout. Before
/// each call stand a node for each of `marks` the graph holds there, then
/// the nodes of each of `timelines` there (`timeline_nodes`), each on
/// `placeholder` until a replay binds its run's word
/// ([`DirectGraph::bind_word`]).
fn record<L: DispatchList>(
    device: &seismic_cuda::Device,
    lists: &[L],
    calls: Range<usize>,
    form: &[u64],
    marks: &[usize],
    timelines: &[(usize, TimelineNode)],
    placeholder: &HostWord,
) -> Result<DirectGraph, ExecutionError> {
    let encoded = lists[0].encoded();
    let mut builder = DirectGraphBuilder::new(device)?;
    let mut dag = super::hazard::DagOrder::default();
    let mut buffers = Vec::new();
    let mut reaches = Vec::new();
    let mut accesses = Vec::new();
    let mut after = Vec::new();
    let mut nodes = 0usize;
    let mut marks = marks.iter().peekable();
    let mut timelines = timelines.iter().peekable();
    for index in calls {
        while marks.next_if_eq(&&index).is_some() {
            builder.mark()?;
            dag.reset(nodes);
        }
        let mut parked_call = false;
        while let Some((_, node)) = timelines.next_if(|(at, _)| *at == index) {
            match node {
                TimelineNode::Wait => {
                    builder.wait_word(placeholder, 0)?;
                    dag.reset(nodes);
                }
                TimelineNode::Signal => {
                    builder.write_word(placeholder, 0)?;
                    dag.reset(nodes);
                }
                TimelineNode::Park => {
                    // A parking kernel spins on a host word; isolate it in the
                    // DAG (`alone = true`) so it never shares the GPU with
                    // concurrent kernels.
                    parked_call = true;
                }
                TimelineNode::Publish => {}
            }
        }
        if form[index] == 0 {
            continue;
        }
        accesses.clear();
        for list in lists {
            buffers.clear();
            let dispatch = list.dispatch(index, &mut buffers);
            reaches.clear();
            list.reaches(index, &mut reaches);
            super::hazard::dispatch_accesses(&dispatch, &reaches, &mut accesses);
        }
        for ordinal in (0..u64::BITS as usize).filter(|&ordinal| holds(form[index], ordinal)) {
            let alone = parked_call
                || lists.iter().any(|list| {
                    buffers.clear();
                    let dispatch = list.dispatch(index, &mut buffers);
                    dispatch.launches[ordinal].is_some_and(|launch| launch.barrier.is_some())
                });
            dag.place_next(nodes, &accesses, alone, &mut after);
            let added = lists
                .iter()
                .enumerate()
                .find_map(|(made, list)| {
                    with_launch(list, index, ordinal, |launch| {
                        if launch.is_empty() {
                            return None;
                        }
                        let runs = made == 0 && encoded[index][ordinal];
                        Some(builder.launch(launch, &after, runs))
                    })
                    .flatten()
                })
                .expect("a layout's graph holds only non-empty launches a plan of it makes");
            added?;
            nodes += 1;
        }
    }
    builder.instantiate()
}

/// Record the graph of a run of `lists[0]`, one sealed plan's run over
/// placeholder storage: a graph of the plan's layout whose nodes are the
/// launches its form holds, which the plans `lists` of that layout make
/// ([`record`]), its word nodes on a placeholder word. A replay binds it to
/// its run's storage and words before it runs. Returns its shape's key with
/// it. Instantiation holds the context while the driver waits on the
/// device, which a kernel parked on a host answer made by driver calls
/// would hang, so a recording with work queued is an invariant violation.
/// It holds by one guard: every recording, and the drain before it, runs
/// under the device's submission order lock (`NativeState::order`: a
/// submission's encoding holds it, `submit`, and so does
/// `NativeGraphFamily::record_decode`), so no native submission is queued
/// between the drain and the recording; other work on the device's native
/// streams there would break the invariant this asserts.
pub(super) fn record_run<L: DispatchList>(device: &seismic_cuda::Device, lists: &[L]) -> Result<(Vec<u64>, DirectGraph), ExecutionError> {
    assert!(device.idle()?, "a graph is recorded only with no device work queued");
    let list = &lists[0];
    let runs = list.runs().expect("a recorded graph is a sealed plan's");
    let form = runs[0].graph.form.clone();
    let keys = ReplayKeys { runs, encoded: None };
    let calls = 0..list.count();
    let key = run_key(lists);
    let marks = keys.marks(&calls).collect::<Vec<_>>();
    let timelines = keys.run(&calls).held_timelines(&calls).collect::<Vec<_>>();
    let graph = record(device, lists, calls, &form, &marks, &timelines, device.placeholder_word())?;
    Ok((key, graph))
}

/// The key of the graph of a run of `lists[0]`, one sealed plan's run over
/// placeholder storage ([`record_run`]), which binds no timeline.
pub(super) fn run_key<L: DispatchList>(lists: &[L]) -> Vec<u64> {
    let list = &lists[0];
    let keys = ReplayKeys { runs: list.runs().expect("a recorded graph is a sealed plan's"), encoded: None };
    shaped(list, &keys, &(0..list.count()), &list.encoded(), &[]).key
}

/// Queue waits on `batch` for every prior native submission in `after`.
fn follow_fences(batch: &mut DirectBatch, after: &[Arc<dyn DeviceCompletion>]) -> Result<(), ExecutionError> {
    for fence in after {
        batch.follow(cuda_submission(fence.as_ref()))?;
    }
    Ok(())
}

/// A native queue fence's CUDA submission.
fn cuda_submission(fence: &dyn DeviceCompletion) -> &DirectSubmission {
    let route = fence.native().expect("a native queue's fence is native work");
    let Committed::Cuda(submission) = route.committed() else {
        unreachable!("one device's native work is of its backend");
    };
    submission
}

/// A timeline operation's CUDA host word (a gate's for a wait, a signal's
/// reached word for a signal) and its low 32-bit value (`timeline`).
fn cuda_timeline_word(operation: &TimelineOperation) -> (&HostWord, u32) {
    let TimelineKind::Cuda(timeline) = operation.timeline.kind() else {
        unreachable!("a run binds timelines of its own device");
    };
    let word = match operation.action {
        TimelineAction::Wait => &timeline.gate,
        TimelineAction::Signal => &timeline.reached,
    };
    (word, operation.value as u32)
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

/// Stamp a boundary's marks on the stream, once the work before them
/// completes (`spans`).
fn stamps(batch: &mut DirectBatch, stamps: &[super::spans::Stamp]) -> Result<(), CallError> {
    for stamp in stamps {
        batch.stamp(cuda_stamps(stamp), stamp.mark).map_err(CallError::Execution)?;
    }
    Ok(())
}

/// Queue a boundary's timeline operations on the stream.
fn timelines(batch: &mut DirectBatch, operations: &[TimelineOperation]) -> Result<(), CallError> {
    for operation in operations {
        let (word, value) = cuda_timeline_word(operation);
        match operation.action {
            TimelineAction::Wait => batch.wait_word(word, value),
            TimelineAction::Signal => batch.write_word(word, value),
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
            .filter(|(_, launch)| launch.is_some_and(|launch| !launch.is_empty()))
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
