//! Vulkan submission of native dispatch lists (Vulkan backend spec §8.6),
//! the counterpart of `cuda.rs`.
//!
//! A submission of sealed graph runs (one run, or a sequence of runs
//! submitted together) replays secondary command buffers: one graph per
//! run's calls between timeline boundaries, recorded once and executed
//! inside the submission's primary buffers. A launch's only push constant
//! is its argument block's address, so a graph's commands hold no buffer
//! address but an indirect dispatch's grid: a graph is keyed by its shape
//! (`ReplayKeys`: the run's plan, the calls' place in it, the launches they
//! encode, and where its grids lie), and a replay binding other storage
//! than the graph's rewrites the changed argument blocks in queue order
//! first (`DirectBatch::rebind`). Per-step host values live in each run's
//! upload region, which the argument blocks address, never in the blocks
//! themselves.
//!
//! Drivers reset a device whose queue submission runs past a watchdog (two
//! seconds by default under Windows TDR and recent amdgpu kernels), so a
//! sealed submission is divided at launch boundaries into queue submissions
//! of bounded measured device time. Every launch's device time is learned
//! per launch kind (pipeline and group counts) from the timestamps each
//! queue submission already writes: a launch of an unmeasured kind is
//! submitted alone, and measured launches are grouped up to
//! [`SUBMISSION_SECONDS`]. A division is recorded as one graph with a
//! segment (secondary command buffer) per part, so its parts share argument
//! storage, and is kept while its parts stay within
//! [`DIVISION_LIMIT_SECONDS`] and was formed from measured launches only.
//!
//! Standalone calls, and every launch of a launch-detail trace (timed one by
//! one), are recorded launch by launch into one queue submission.
//!
//! A span mark inside a graph (`spans`) ends a segment of it and no part:
//! a run not sampled executes the segments as one, and a sampled run stamps
//! between them in its primary buffer.
//!
//! A timeline boundary (`timeline`) ends a queue submission: the one before
//! it signals its signals' timeline semaphores, and the one after it waits
//! on its gates'. The calls between boundaries are recorded and divided as
//! a submission of their own.

use super::graph_replays::{Replays, Shaped};
use super::timeline::{TimelineAction, TimelineKind, TimelineOperation, VulkanTimeline};
use super::{graph_segments, DispatchList, EncodeFailure, Encoding, NativeRoute, ReplayKeys};
use crate::completion::Committed;
use crate::api::CallError;
use crate::backends::vulkan_buffer;
use crate::driver::Allocation;
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::GraphCensus;
use seismic_vulkan::direct::{
    DirectBatch, DirectGraph, DirectGraphBuilder, DirectLaunch, DirectSubmission, Gate, HostTimeline, LaunchKind, Reached,
    Measured, TimelineAnchor, Unmeasured,
};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// Measured device time a queue submission of sealed runs is formed within:
/// far below any driver watchdog, so a launch whose time grows between
/// measurements (attention over a lengthening history) stays clear of it.
const SUBMISSION_SECONDS: f64 = 0.1;
/// A recorded division is kept while each of its parts is estimated within
/// this; past it the submission is divided again.
const DIVISION_LIMIT_SECONDS: f64 = 0.4;

/// A device's recorded graphs of sealed-plan submissions, and the measured
/// device time of each launch kind they run.
#[derive(Default)]
pub(crate) struct VulkanReplays {
    graphs: Replays<Divided>,
    costs: Arc<LaunchCosts<LaunchKind>>,
}

impl VulkanReplays {
    /// What the graphs cost serving so far (`Replays::counts`), each
    /// division formed again (`Divided::refresh`) among the recordings.
    pub(crate) fn counts(&self) -> super::graph_replays::GraphReplays {
        self.graphs.counts()
    }

    /// What the graphs of plan `plan`'s runs cost so far.
    pub(crate) fn plan_counts(&self, plan: u64) -> super::graph_replays::GraphReplays {
        self.graphs.plan_counts(plan)
    }

    /// Drop every graph if none is held (`Replays::clear_unheld`).
    pub(crate) fn clear_unheld(&self, unheld: impl FnOnce() -> bool) {
        self.graphs.clear_unheld(unheld)
    }
}

/// One sealed submission's launches (the non-empty ones, in order) divided
/// into consecutive parts, recorded as one graph with a segment per part;
/// `measured` says whether every launch had a measured time when the parts
/// were formed.
impl super::graph_replays::RecordedGraph for Divided {
    /// None before its first replay records it (`Divided::refresh`).
    fn census(&self) -> GraphCensus {
        self.graph.as_ref().map_or(GraphCensus::default(), DirectGraph::census)
    }
    fn idle(&self) -> Result<bool, CallError> {
        Ok(true)
    }
    /// An argument-block rewrite runs in queue order beside the launches
    /// in flight (`DirectBatch::rebind`).
    fn updates_in_flight() -> bool {
        seismic_lang::registry::BackendName::Vulkan.graphs_update_in_flight()
    }
}

struct Divided {
    launches: Vec<LaunchKind>,
    /// The marks the graph holds, each the launch it precedes, in order.
    marks: Vec<usize>,
    parts: Vec<Part>,
    graph: Option<DirectGraph>,
    measured: bool,
    /// Whether the graph binds other storage than the replay about to run
    /// (`Replays::replay`): its argument blocks are rewritten before its
    /// first part runs, unless it is recorded again.
    stale: bool,
}

/// A range of a division's launches and its distinct launch kinds with how
/// often each occurs, the segments of the graph that hold them, and the
/// graph's marks among them, each with the segment it follows.
struct Part {
    launches: Range<usize>,
    kinds: Vec<(LaunchKind, u32)>,
    segments: Range<usize>,
    marks: Vec<(usize, usize)>,
}

/// Encode `work` on the device's queue: a replay when the list is made of
/// sealed plans and is not timed, else launch by launch, its graphs kept
/// within `budget`. Where its device time is read (a trace, a measurement),
/// every queue submission is measured.
pub(super) fn encode(
    device: &seismic_vulkan::Device,
    replays: &VulkanReplays,
    budget: GraphCensus,
    work: &Encoding<'_, impl DispatchList>,
) -> Result<Committed, EncodeFailure> {
    let Encoding { list, repetitions, observed, retained, .. } = *work;
    let timed = work.timed.map(|(_, launches)| launches);
    // The calls between timeline boundaries form one segment, and a sealed
    // run's of them a graph. A boundary ends a queue submission: the next
    // one waits on its gates' timelines, and the one it ends signals its
    // signals'.
    let encoded = list.encoded();
    let Some(runs) = list.runs().filter(|_| timed.is_none()) else {
        let segments = graph_segments(list, None);
        let new_batch = || {
            match timed {
                Some(launches) => DirectBatch::timed(device, launches),
                None => DirectBatch::new(device),
            }
            .map_err(CallError::Execution)
        };
        let mut submissions = Issuing::default();
        let issued = (|| {
            let mut batch = new_batch()?;
            for _ in 0..repetitions {
                for (calls, boundary) in &segments {
                    each_launch(list, calls.clone(), &encoded, |launch| match launch {
                        Some(launch) => batch.launch(launch),
                        None => {
                            batch.skip();
                            Ok(())
                        }
                    })?;
                    for stamp in list.stamps(*boundary) {
                        batch.stamp(vulkan_stamps(stamp), stamp.mark);
                    }
                    let operations = list.timelines(*boundary);
                    if operations.is_empty() {
                        continue;
                    }
                    for operation in operations {
                        if operation.action == TimelineAction::Signal {
                            batch.signal_timeline(&vulkan_timeline(operation).reached, operation.value);
                        }
                    }
                    submissions.push(QueueSubmission::Measured(
                        std::mem::replace(&mut batch, new_batch()?)
                            .commit()
                            .map_err(CallError::Execution)?,
                    ));
                    for operation in operations {
                        if operation.action == TimelineAction::Wait {
                            batch.wait_timeline(&vulkan_timeline(operation).gate, operation.value);
                        }
                    }
                }
            }
            submissions.push(QueueSubmission::Measured(batch.commit().map_err(CallError::Execution)?));
            Ok::<(), CallError>(())
        })();
        if let Err(error) = issued {
            return Err(submissions.failed(error));
        }
        return Ok(Committed::Vulkan(submissions.into_submission(None)));
    };
    let segments = graph_segments(list, Some(&runs));
    let keys = ReplayKeys { runs, encoded: Some(&encoded) };
    let (graphed, shaped): (Vec<Range<usize>>, Vec<Shaped>) = segments
        .iter()
        .filter(|(calls, _)| !calls.is_empty())
        .map(|(calls, _)| {
            let (bound, grids) = bindings(list, calls.clone(), &encoded);
            (calls.clone(), keys.shaped(calls, grids, bound))
        })
        .unzip();
    // Recording a division takes nothing a park's answer needs (a Vulkan
    // park is a submission boundary, and recording command buffers holds
    // no device-wide lock), so it waits for nothing.
    let taken = replays.graphs.record(&shaped, budget, retained, |index| {
        let (mut launches, mut marks) = (Vec::new(), Vec::new());
        let mut held = keys.marks(&graphed[index]).peekable();
        for call in graphed[index].clone() {
            while held.next_if_eq(&call).is_some() {
                marks.push(launches.len());
            }
            each_launch(list, call..call + 1, &encoded, |launch| {
                launches.extend(launch.map(DirectLaunch::kind));
                Ok(())
            })?;
        }
        Ok(Divided {
            launches,
            marks,
            parts: Vec::new(),
            graph: None,
            measured: false,
            stale: false,
        })
    })
    .map_err(EncodeFailure::uncommitted)?;
    let mut queued = Queued::new(device, &replays.costs, observed);
    let issued = (|| {
        for _ in 0..repetitions {
            let mut graphs = graphed.iter().zip(&shaped).zip(&taken);
            for (calls, boundary) in &segments {
                if !calls.is_empty() {
                    let ((calls, shaped), &identity) = graphs.next().expect("a graph per segment of calls");
                    replays.graphs.replay(
                        shaped,
                        identity,
                        |divided, _| {
                            divided.stale = true;
                            Ok(())
                        },
                        |divided| {
                            if divided.refresh(device, list, calls.clone(), &encoded, &replays.costs)? {
                                replays.graphs.count_recorded(&shaped.key);
                            }
                            // The launches' argument blocks as this run
                            // binds them, which its first part rewrites.
                            let mut rebind = None;
                            if std::mem::take(&mut divided.stale) {
                                let mut blocks = Vec::new();
                                each_launch(list, calls.clone(), &encoded, |launch| {
                                    blocks.extend(launch.filter(|launch| !launch.is_empty()).map(DirectLaunch::argument_block));
                                    Ok(())
                                })?;
                                rebind = Some(blocks);
                            }
                            let stamps = keys.stamps(list, calls);
                            let Divided { parts, graph, .. } = divided;
                            let graph = graph.as_mut().expect("a refreshed division is recorded");
                            for part in parts.iter() {
                                queued.add(graph, part, &stamps, replays.costs.estimate(&part.kinds), rebind.take())?;
                            }
                            Ok(())
                        },
                    )?;
                }
                queued.stamps(list.stamps(*boundary))?;
                queued.timelines(list.timelines(*boundary))?;
            }
        }
        // A division is recorded at its replay: the budget holds it there.
        replays.graphs.fit_after_replay(&taken, budget);
        queued.finish()
    })();
    let submissions = match issued {
        Ok(submissions) => submissions,
        Err(error) => {
            // The open batch, and any rewrite of argument blocks it carried,
            // is dropped uncommitted: the graphs bind what is unknown.
            replays.graphs.unbind(&shaped, |divided| {
                if let Some(graph) = &mut divided.graph {
                    graph.unbind();
                }
            });
            return Err(std::mem::take(&mut queued.submissions).failed(error));
        }
    };
    Ok(Committed::Vulkan(submissions.submissions.into_submission(Some(Feedback {
        costs: replays.costs.clone(),
        kinds: submissions.kinds,
    }))))
}

impl Divided {
    /// Form the parts again unless the current ones were formed from
    /// measured launches and are each still estimated within
    /// [`DIVISION_LIMIT_SECONDS`]; record the graph again when they change.
    /// Whether it recorded a graph that was recorded before.
    fn refresh(
        &mut self,
        device: &seismic_vulkan::Device,
        list: &impl DispatchList,
        calls: Range<usize>,
        encoded: &[Vec<bool>],
        costs: &LaunchCosts<LaunchKind>,
    ) -> Result<bool, CallError> {
        let current = self.measured
            && self.graph.is_some()
            && self.parts.iter().all(|part| {
                costs
                    .estimate(&part.kinds)
                    .is_some_and(|seconds| seconds <= DIVISION_LIMIT_SECONDS)
            });
        if current {
            return Ok(false);
        }
        let estimates = costs.estimates(&self.launches);
        let measured = estimates.iter().all(Option::is_some);
        let mut ranges = divide(&estimates, SUBMISSION_SECONDS);
        if ranges.is_empty() {
            // No launch does device work: one empty part keeps the
            // submission's ordering and completion.
            ranges.push(0..0);
        }
        if self.graph.is_some()
            && ranges.len() == self.parts.len()
            && ranges
                .iter()
                .zip(&self.parts)
                .all(|(range, part)| *range == part.launches)
        {
            self.measured = measured;
            return Ok(false);
        }
        let (ends, located) = located(&ranges, &self.marks);
        let mut builder = DirectGraphBuilder::new(device).map_err(CallError::Execution)?;
        let mut index = 0usize;
        let mut segment = 0usize;
        each_launch(list, calls, encoded, |launch| {
            let Some(launch) = launch else {
                return Ok(());
            };
            while ends[segment] <= index && segment + 1 < ends.len() {
                segment += 1;
                builder.segment()?;
            }
            index += 1;
            builder.launch(launch)
        })?;
        for _ in segment + 1..ends.len() {
            builder.segment().map_err(CallError::Execution)?;
        }
        let graph = builder.instantiate().map_err(CallError::Execution)?;
        // Recorded from this replay's launches: it binds their storage.
        self.stale = false;
        self.parts = ranges
            .into_iter()
            .zip(located)
            .map(|(launches, (segments, marks))| Part {
                kinds: counted(&self.launches[launches.clone()]),
                launches,
                segments,
                marks,
            })
            .collect();
        self.measured = measured;
        let again = self.graph.replace(graph).is_some();
        Ok(again)
    }
}

/// The segments of a graph of the parts `ranges` holding `marks`, the
/// launches its marks precede, in order: the launch each segment ends at, and
/// for each part its segments and its marks, each with the segment it
/// follows and its ordinal. A part ends a segment, and so does a mark: a mark
/// where a part ends is the part's.
fn located(ranges: &[Range<usize>], marks: &[usize]) -> (Vec<usize>, Vec<(Range<usize>, Vec<(usize, usize)>)>) {
    let mut ends = Vec::new();
    let mut parts = Vec::with_capacity(ranges.len());
    let mut ordinal = 0;
    for range in ranges {
        let first = ends.len();
        let mut held = Vec::new();
        while marks.get(ordinal).is_some_and(|&mark| mark <= range.end) {
            held.push((ends.len(), ordinal));
            ends.push(marks[ordinal]);
            ordinal += 1;
        }
        if ends.len() == first || ends.last() != Some(&range.end) {
            ends.push(range.end);
        }
        parts.push((first..ends.len(), held));
    }
    (ends, parts)
}

/// Distinct kinds of `launches` with their counts, in first-use order.
fn counted<K: Copy + Eq>(launches: &[K]) -> Vec<(K, u32)> {
    let mut kinds: Vec<(K, u32)> = Vec::new();
    for launch in launches {
        match kinds.iter_mut().find(|(kind, _)| kind == launch) {
            Some((_, count)) => *count += 1,
            None => kinds.push((*launch, 1)),
        }
    }
    kinds
}

/// Consecutive launch ranges: a launch without an estimate alone, the rest
/// grouped while their estimated time stays within `seconds` (a launch
/// estimated beyond it alone).
fn divide(estimates: &[Option<f64>], seconds: f64) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut sum = 0.0;
    for (index, estimate) in estimates.iter().enumerate() {
        match estimate {
            None => {
                if start < index {
                    ranges.push(start..index);
                }
                ranges.push(index..index + 1);
                start = index + 1;
                sum = 0.0;
            }
            Some(estimate) => {
                if start < index && sum + estimate > seconds {
                    ranges.push(start..index);
                    start = index;
                    sum = 0.0;
                }
                sum += estimate;
            }
        }
    }
    if start < estimates.len() {
        ranges.push(start..estimates.len());
    }
    ranges
}

/// Measured device time per launch kind, learned from completed queue
/// submissions.
struct LaunchCosts<K> {
    seconds: Mutex<HashMap<K, f64>>,
    /// Estimated device time queued in unmeasured batches since the last
    /// measured one ([`Self::measures`]).
    unmeasured: Mutex<f64>,
}

impl<K> Default for LaunchCosts<K> {
    fn default() -> Self {
        Self {
            seconds: Mutex::new(HashMap::new()),
            unmeasured: Mutex::new(0.0),
        }
    }
}

impl<K: Copy + Eq + Hash> LaunchCosts<K> {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<K, f64>> {
        self.seconds
            .lock()
            .expect("launch cost lock is never poisoned")
    }

    fn estimates(&self, launches: &[K]) -> Vec<Option<f64>> {
        let seconds = self.lock();
        launches
            .iter()
            .map(|launch| seconds.get(launch).copied())
            .collect()
    }

    /// Whether a batch whose first part is estimated at `estimate` writes
    /// its timestamps: where its kinds are unmeasured (`None`), and
    /// otherwise once a [`SUBMISSION_SECONDS`] of estimated work has gone
    /// unmeasured since the last, so every kind's estimate is relearned
    /// within that much of its work (a launch whose time grows, attention
    /// over a lengthening history, is caught before a submission formed by
    /// it could reach a watchdog) at a measured batch's cost (two
    /// timestamps and a query reset, about 5 µs on an RTX 3090) per
    /// [`SUBMISSION_SECONDS`] of device work.
    fn measures(&self, estimate: Option<f64>) -> bool {
        let Some(estimate) = estimate else {
            return true;
        };
        let mut unmeasured = self.unmeasured.lock().expect("launch cost lock is never poisoned");
        *unmeasured += estimate;
        if *unmeasured >= SUBMISSION_SECONDS {
            *unmeasured = 0.0;
            return true;
        }
        false
    }

    /// A part of `estimate` joined an unmeasured batch: its work goes
    /// unmeasured too ([`Self::measures`]).
    fn accrue(&self, estimate: f64) {
        *self.unmeasured.lock().expect("launch cost lock is never poisoned") += estimate;
    }

    /// The estimated time of launches counted by kind; `None` when a kind
    /// has not been measured.
    fn estimate(&self, kinds: &[(K, u32)]) -> Option<f64> {
        let seconds = self.lock();
        sum(&seconds, kinds)
    }

    /// Learn from a completed queue submission that ran `kinds` in
    /// `measured` seconds: the estimates of its kinds are scaled to sum to
    /// it, and unmeasured kinds take an equal share per launch.
    fn observe(&self, kinds: &[(K, u32)], measured: f64) {
        if kinds.is_empty() || !measured.is_finite() || measured < 0.0 {
            return;
        }
        let mut seconds = self.lock();
        match sum(&seconds, kinds) {
            Some(estimated) if estimated > 0.0 => {
                let scale = measured / estimated;
                for (kind, _) in kinds {
                    if let Some(value) = seconds.get_mut(kind) {
                        *value *= scale;
                    }
                }
            }
            _ => {
                let launches = kinds.iter().map(|(_, count)| *count as f64).sum::<f64>();
                for (kind, _) in kinds {
                    let share = measured / launches;
                    let value = seconds.entry(*kind).or_insert(share);
                    if *value <= 0.0 {
                        *value = share;
                    }
                }
            }
        }
    }
}

fn sum<K: Eq + Hash>(seconds: &HashMap<K, f64>, kinds: &[(K, u32)]) -> Option<f64> {
    kinds.iter().try_fold(0.0, |total, (kind, count)| {
        seconds
            .get(kind)
            .map(|value| total + value * f64::from(*count))
    })
}

/// Parts queued into batches: consecutive parts share a queue submission
/// while their estimated time fits [`SUBMISSION_SECONDS`]; a part of
/// unmeasured launches is submitted alone. Each batch is committed as soon as
/// the next part does not fit it, so the device starts on it at once.
struct Queued<'a> {
    device: &'a seismic_vulkan::Device,
    costs: &'a LaunchCosts<LaunchKind>,
    /// Whether every batch is measured (the run is observed).
    observed: bool,
    open: Option<(QueueBatch, Option<f64>, Vec<(LaunchKind, u32)>)>,
    /// Gates the next batch opened waits on.
    gates: Vec<TimelineOperation>,
    submissions: Issuing,
    kinds: Vec<Vec<(LaunchKind, u32)>>,
}

struct QueuedSubmissions {
    submissions: Issuing,
    kinds: Vec<Vec<(LaunchKind, u32)>>,
}

impl<'a> Queued<'a> {
    fn new(device: &'a seismic_vulkan::Device, costs: &'a LaunchCosts<LaunchKind>, observed: bool) -> Self {
        Self {
            device,
            costs,
            observed,
            open: None,
            gates: Vec::new(),
            submissions: Issuing::default(),
            kinds: Vec::new(),
        }
    }

    /// Open a batch, which waits on the gates of the boundary before it:
    /// measured where the run is observed or the costs ask
    /// ([`LaunchCosts::measures`], for its first part's `estimate`; a batch
    /// of no part has no work to learn).
    fn open(&mut self, estimate: Option<Option<f64>>) -> Result<(), CallError> {
        let measured = self.observed || estimate.is_some_and(|estimate| self.costs.measures(estimate));
        let mut batch = QueueBatch::new(self.device, measured).map_err(CallError::Execution)?;
        for gate in self.gates.drain(..) {
            batch.wait_timeline(&vulkan_timeline(&gate).gate, gate.value);
        }
        self.open = Some((batch, Some(0.0), Vec::new()));
        Ok(())
    }

    /// Stamp a boundary's marks in the open batch (an empty one when none is
    /// open), once the work queued before them completes (`spans`).
    fn stamps(&mut self, stamps: &[super::spans::Stamp]) -> Result<(), CallError> {
        if stamps.is_empty() {
            return Ok(());
        }
        if self.open.is_none() {
            self.open(None)?;
        }
        let (batch, ..) = self.open.as_mut().expect("a batch is open");
        for stamp in stamps {
            batch.stamp(vulkan_stamps(stamp), stamp.mark);
        }
        Ok(())
    }

    /// A boundary's timeline operations: its signals end the open batch
    /// (an empty one when none is open), and its gates make the next batch
    /// wait.
    fn timelines(&mut self, operations: &[TimelineOperation]) -> Result<(), CallError> {
        if operations.is_empty() {
            return Ok(());
        }
        if operations
            .iter()
            .any(|operation| operation.action == TimelineAction::Signal)
        {
            if self.open.is_none() {
                self.open(None)?;
            }
            let (batch, ..) = self.open.as_mut().expect("a batch is open");
            for operation in operations {
                if operation.action == TimelineAction::Signal {
                    batch.signal_timeline(&vulkan_timeline(operation).reached, operation.value);
                }
            }
        }
        self.commit()?;
        self.gates.extend(
            operations
                .iter()
                .filter(|operation| operation.action == TimelineAction::Wait)
                .cloned(),
        );
        Ok(())
    }

    /// Queue `part` of `graph`, having first rewritten the graph's argument
    /// blocks to `rebind` in the same batch, where the replay binds other
    /// storage than the graph; where `stamps` is a sampled run's at the
    /// graph's marks, each stamped at its mark.
    fn add(
        &mut self,
        graph: &mut DirectGraph,
        part: &Part,
        stamps: &[&super::spans::Stamp],
        estimate: Option<f64>,
        rebind: Option<Vec<Vec<u8>>>,
    ) -> Result<(), CallError> {
        let joins = matches!(
            (&self.open, estimate),
            (Some((_, Some(queued), _)), Some(estimate)) if queued + estimate <= SUBMISSION_SECONDS
        );
        if !joins {
            self.commit()?;
            self.open(Some(estimate))?;
        } else if let (Some((QueueBatch::Unmeasured(_), ..)), Some(estimate)) = (&self.open, estimate) {
            self.costs.accrue(estimate);
        }
        let (batch, queued, kinds) = self.open.as_mut().expect("a batch is open");
        if let Some(blocks) = rebind {
            batch.rebind(graph, blocks).map_err(CallError::Execution)?;
        }
        let mut next = part.segments.start;
        for &(segment, mark) in part.marks.iter().filter(|_| !stamps.is_empty()) {
            batch.replay_segments(graph, next..segment + 1).map_err(CallError::Execution)?;
            next = segment + 1;
            batch.stamp(vulkan_stamps(stamps[mark]), stamps[mark].mark);
        }
        if next < part.segments.end {
            batch.replay_segments(graph, next..part.segments.end).map_err(CallError::Execution)?;
        }
        *queued = queued
            .zip(estimate)
            .map(|(queued, estimate)| queued + estimate);
        for (kind, count) in &part.kinds {
            match kinds.iter_mut().find(|(queued, _)| queued == kind) {
                Some((_, queued)) => *queued += count,
                None => kinds.push((*kind, *count)),
            }
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<(), CallError> {
        if let Some((batch, _, kinds)) = self.open.take() {
            self.submissions
                .push(batch.commit().map_err(CallError::Execution)?);
            self.kinds.push(kinds);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<QueuedSubmissions, CallError> {
        if self.open.is_none() && (self.submissions.is_empty() || !self.gates.is_empty()) {
            // Nothing was queued (no repetition), or gates end the list: an
            // empty batch still orders the submission, waits on them and
            // reports its completion.
            self.open(None)?;
        }
        self.commit()?;
        Ok(QueuedSubmissions {
            submissions: std::mem::take(&mut self.submissions),
            kinds: std::mem::take(&mut self.kinds),
        })
    }
}

/// Queue submissions committed while issuing, not yet handed to a
/// completion. They run on storage the caller regains, so dropped unhanded
/// (issuing unwound) they are waited for.
#[derive(Default)]
struct Issuing(Vec<QueueSubmission>);

impl Issuing {
    fn push(&mut self, submission: QueueSubmission) {
        self.0.push(submission);
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Hand every submission to the work's completion.
    fn into_submission(mut self, feedback: Option<Feedback>) -> VulkanSubmission {
        VulkanSubmission::new(std::mem::take(&mut self.0), feedback)
    }

    /// Issuing failed after these were committed: the uncommitted batch is
    /// discarded, and the committed ones run on as the submission's
    /// completion, which the caller can answer even where one is parked.
    fn failed(self, error: CallError) -> EncodeFailure {
        let committed = (!self.is_empty()).then(|| Committed::Vulkan(self.into_submission(None)));
        EncodeFailure { error, committed }
    }
}

impl Drop for Issuing {
    /// Submissions never handed to a completion end before what they use
    /// is released; a fault ends one as completion does, and the issue
    /// that abandoned them is what their caller reports.
    fn drop(&mut self) {
        for submission in &self.0 {
            match submission.finish() {
                Ok(()) | Err(_) => {}
            }
        }
    }
}

/// What a completed submission teaches its device's launch costs.
struct Feedback {
    costs: Arc<LaunchCosts<LaunchKind>>,
    kinds: Vec<Vec<(LaunchKind, u32)>>,
}

/// One unit of native work as one or more queue submissions, in commit
/// order: the last completes after every earlier one.
pub(crate) struct VulkanSubmission {
    submissions: Vec<QueueSubmission>,
    feedback: Mutex<Option<Feedback>>,
}

/// One queue submission of a [`VulkanSubmission`]: measured (its device
/// interval read once it completes: its kinds' costs learned from it, a
/// trace placed by it) or not.
enum QueueSubmission {
    Measured(DirectSubmission<Measured>),
    Unmeasured(DirectSubmission<Unmeasured>),
}

impl QueueSubmission {
    fn completed(&self) -> Result<bool, ExecutionError> {
        match self {
            Self::Measured(submission) => submission.completed(),
            Self::Unmeasured(submission) => submission.completed(),
        }
    }

    fn finish(&self) -> Result<(), ExecutionError> {
        match self {
            Self::Measured(submission) => submission.finish(),
            Self::Unmeasured(submission) => submission.finish(),
        }
    }

    fn handed(&self) -> Option<(std::time::Instant, std::time::Instant)> {
        match self {
            Self::Measured(submission) => submission.handed(),
            Self::Unmeasured(submission) => submission.handed(),
        }
    }

    fn opened(&self) -> Option<(std::time::Instant, seismic_native_target::GateOpening)> {
        match self {
            Self::Measured(submission) => submission.opened(),
            Self::Unmeasured(submission) => submission.opened(),
        }
    }

    /// The measured submission, or the error a reading of an unmeasured
    /// one's device time is: an observed run measures every one.
    fn measured(&self) -> Result<&DirectSubmission<Measured>, ExecutionError> {
        match self {
            Self::Measured(submission) => Ok(submission),
            Self::Unmeasured(_) => Err(ExecutionError::SubmissionFailed(
                "an unobserved Vulkan submission's device time was read".into(),
            )),
        }
    }
}

/// The open batch of a [`Queued`], measured or not.
enum QueueBatch {
    Measured(DirectBatch<Measured>),
    Unmeasured(DirectBatch<Unmeasured>),
}

impl QueueBatch {
    fn new(device: &seismic_vulkan::Device, measured: bool) -> Result<Self, ExecutionError> {
        Ok(match measured {
            true => Self::Measured(DirectBatch::new(device)?),
            false => Self::Unmeasured(DirectBatch::unmeasured(device)?),
        })
    }

    fn wait_timeline(&mut self, timeline: &HostTimeline<Gate>, value: u64) {
        match self {
            Self::Measured(batch) => batch.wait_timeline(timeline, value),
            Self::Unmeasured(batch) => batch.wait_timeline(timeline, value),
        }
    }

    fn signal_timeline(&mut self, timeline: &HostTimeline<Reached>, value: u64) {
        match self {
            Self::Measured(batch) => batch.signal_timeline(timeline, value),
            Self::Unmeasured(batch) => batch.signal_timeline(timeline, value),
        }
    }

    fn stamp(&mut self, stamps: &seismic_vulkan::direct::Stamps, mark: usize) {
        match self {
            Self::Measured(batch) => batch.stamp(stamps, mark),
            Self::Unmeasured(batch) => batch.stamp(stamps, mark),
        }
    }

    fn replay_segments(&mut self, graph: &DirectGraph, segments: Range<usize>) -> Result<(), ExecutionError> {
        match self {
            Self::Measured(batch) => batch.replay_segments(graph, segments),
            Self::Unmeasured(batch) => batch.replay_segments(graph, segments),
        }
    }

    fn rebind(&mut self, graph: &mut DirectGraph, blocks: Vec<Vec<u8>>) -> Result<(), ExecutionError> {
        match self {
            Self::Measured(batch) => batch.rebind(graph, blocks),
            Self::Unmeasured(batch) => batch.rebind(graph, blocks),
        }
    }

    fn commit(self) -> Result<QueueSubmission, ExecutionError> {
        Ok(match self {
            Self::Measured(batch) => QueueSubmission::Measured(batch.commit()?),
            Self::Unmeasured(batch) => QueueSubmission::Unmeasured(batch.commit()?),
        })
    }
}

impl VulkanSubmission {
    fn new(submissions: Vec<QueueSubmission>, feedback: Option<Feedback>) -> Self {
        assert!(
            !submissions.is_empty(),
            "VulkanSubmission::new precondition: at least one queue submission"
        );
        Self {
            submissions,
            feedback: Mutex::new(feedback),
        }
    }

    fn last(&self) -> &QueueSubmission {
        self.submissions
            .last()
            .expect("a Vulkan submission has a queue submission")
    }

    /// Teach the launch costs once, after completion.
    fn observe(&self) {
        let Some(feedback) = self
            .feedback
            .lock()
            .expect("submission feedback lock is never poisoned")
            .take()
        else {
            return;
        };
        for (submission, kinds) in self.submissions.iter().zip(&feedback.kinds) {
            if let QueueSubmission::Measured(submission) = submission {
                if let Ok(seconds) = submission.device_seconds() {
                    feedback.costs.observe(kinds, seconds);
                }
            }
        }
    }

    /// Whether every queue submission completed, without waiting: the last
    /// completes after every earlier one, and a refusal or the device's
    /// loss reaches it too.
    pub(crate) fn completed(&self) -> Result<bool, ExecutionError> {
        let complete = self.last().completed()?;
        if complete {
            self.observe();
        }
        Ok(complete)
    }

    pub(crate) fn finish(&self) -> Result<(), ExecutionError> {
        for submission in &self.submissions {
            submission.finish()?;
        }
        self.observe();
        Ok(())
    }

    pub(crate) fn device_seconds(&self) -> Result<f64, ExecutionError> {
        self.submissions.iter().try_fold(0.0, |total, submission| {
            Ok(total + submission.measured()?.device_seconds()?)
        })
    }

    pub(crate) fn device_interval(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<(f64, f64), ExecutionError> {
        let (start, _) = self.submissions[0].measured()?.device_interval(anchor)?;
        let (_, end) = self.last().measured()?.device_interval(anchor)?;
        Ok((start, end))
    }

    /// Each queue submission's hand-off to the queue and device interval,
    /// on the anchor's host clock, in commit order. Reading its interval
    /// waits for its completion, and a completed one was handed over.
    pub(crate) fn queue_submissions(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Vec<super::trace::TracedQueueSubmission>, ExecutionError> {
        self.submissions
            .iter()
            .map(|submission| {
                let device = submission.measured()?.device_interval(anchor)?;
                let (handed, taken) = submission
                    .handed()
                    .expect("a completed queue submission was handed to the queue");
                let opened = submission
                    .opened()
                    .map(|(at, opening)| (super::trace::host_seconds_at(at), opening));
                Ok(super::trace::TracedQueueSubmission {
                    handed: super::trace::host_seconds_at(handed),
                    taken: super::trace::host_seconds_at(taken),
                    opened,
                    device,
                })
            })
            .collect()
    }

    /// Per-launch intervals of a timed submission: one queue submission,
    /// or one per timeline segment, each timing its own launches in order.
    pub(crate) fn launch_intervals(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        let mut intervals = Vec::new();
        for submission in &self.submissions {
            let Some(own) = submission.measured()?.launch_intervals(anchor)? else {
                return Ok(None);
            };
            intervals.extend(own);
        }
        Ok(Some(intervals))
    }
}

/// The semaphores of the timeline an operation bound on this device: a
/// signal raises its `reached`, a wait waits on its `gate`
/// (`VulkanTimeline`).
/// A stamp's Vulkan stamps.
fn vulkan_stamps(stamp: &super::spans::Stamp) -> &seismic_vulkan::direct::Stamps {
    let super::spans::Stamps::Vulkan(stamps) = &*stamp.stamps else {
        unreachable!("a run's stamps are of its own device");
    };
    stamps
}

fn vulkan_timeline(operation: &TimelineOperation) -> &VulkanTimeline {
    let TimelineKind::Vulkan(timeline) = operation.timeline.kind() else {
        unreachable!("a run binds timelines of its own device");
    };
    timeline
}

/// Visit every launch of the calls `calls` of one pass over `list` in
/// order; `None` for an inactive launch or one `encoded` leaves out.
fn each_launch(
    list: &impl DispatchList,
    calls: Range<usize>,
    encoded: &[Vec<bool>],
    mut visit: impl FnMut(Option<&DirectLaunch<'_>>) -> Result<(), ExecutionError>,
) -> Result<(), CallError> {
    let mut buffers = Vec::new();
    let mut typed = Vec::new();
    for index in calls {
        buffers.clear();
        let dispatch = list.dispatch(index, &mut buffers);
        let NativeRoute::Vulkan { launches, .. } = &dispatch.kernel.route else {
            unreachable!("one device has one native route");
        };
        typed.clear();
        typed.extend(
            buffers
                .iter()
                .map(|(allocation, offset)| (vulkan_buffer(allocation), *offset)),
        );
        let scalars = dispatch
            .kernel
            .scalar_slots()
            .map(|slots| (vulkan_buffer(slots), 0));
        for (ordinal, launch) in dispatch.launches.iter().enumerate() {
            let launch = launch.filter(|_| encoded[index][ordinal]).map(|launch| {
                let (module, function) = launches.launch(ordinal);
                DirectLaunch {
                    module,
                    function,
                    buffers: &typed,
                    words: super::word_bytes(dispatch.words),
                    scalar_results: scalars,
                    groups: launch.groups,
                    grid: launch.grid.map(|buffer| typed[dispatch.kernel.argument(buffer)]),
                }
            });
            visit(launch.as_ref()).map_err(CallError::Execution)?;
        }
    }
    Ok(())
}

/// Every buffer address `calls` of `list` bind, in dispatch order, which
/// with the sealed plans fix every argument block of their graph; and where
/// the indirect grid of every launch they encode that reads one lies, in
/// launch order: an indirect dispatch names its grid's buffer in the
/// graph's commands, so that is part of the graph's shape.
fn bindings(list: &impl DispatchList, calls: Range<usize>, encoded: &[Vec<bool>]) -> (Vec<u64>, Vec<u64>) {
    let mut bound = Vec::new();
    let mut grids = Vec::new();
    let mut buffers: Vec<(&Allocation, u64)> = Vec::new();
    for index in calls {
        buffers.clear();
        let dispatch = list.dispatch(index, &mut buffers);
        let address = |(allocation, offset): &(&Allocation, u64)| vulkan_buffer(allocation).address() + offset;
        bound.extend(buffers.iter().map(address));
        for (ordinal, launch) in dispatch.launches.iter().enumerate() {
            if let Some(grid) = launch.filter(|_| encoded[index][ordinal]).and_then(|launch| launch.grid) {
                grids.push(address(&buffers[dispatch.kernel.argument(grid)]));
            }
        }
    }
    (bound, grids)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parts and marks each end a segment, once where they meet; a mark is
    /// held by the part it ends or lies in, and one before any launch makes
    /// an empty segment it follows.
    #[test]
    fn marks_end_segments_of_the_parts_they_lie_in() {
        let (ends, parts) = located(&[0..4, 4..9], &[2, 4, 6]);
        assert_eq!(ends, [2, 4, 6, 9]);
        assert_eq!(parts, [(0..2, vec![(0, 0), (1, 1)]), (2..4, vec![(2, 2)])]);
        let (ends, parts) = located(&[0..3], &[0]);
        assert_eq!(ends, [0, 3]);
        assert_eq!(parts, [(0..2, vec![(0, 0)])]);
        assert_eq!(located(&[0..0], &[]), (vec![0], vec![(0..1, Vec::new())]));
    }

    /// A batch is measured where its kinds are unmeasured, and otherwise
    /// once a submission's worth of estimated work went unmeasured, joined
    /// parts' work counted with the opening part's.
    #[test]
    fn a_batch_is_measured_once_a_submissions_worth_of_work_went_unmeasured() {
        let costs = LaunchCosts::<u32>::default();
        assert!(costs.measures(None), "unmeasured kinds are measured");
        assert!(!costs.measures(Some(0.04)));
        costs.accrue(0.04);
        assert!(costs.measures(Some(0.03)), "0.11 s went unmeasured");
        assert!(!costs.measures(Some(0.05)), "the count starts again");
    }

    #[test]
    fn unmeasured_launches_are_divided_alone() {
        assert_eq!(divide(&[None, None, None], 0.1), vec![0..1, 1..2, 2..3]);
        assert_eq!(
            divide(&[Some(0.01), None, Some(0.01), Some(0.01)], 0.1),
            vec![0..1, 1..2, 2..4]
        );
    }

    #[test]
    fn measured_launches_are_grouped_within_the_budget() {
        assert_eq!(divide(&[Some(0.001); 50], 0.1), vec![0..50]);
        assert_eq!(
            divide(&[Some(0.04), Some(0.04), Some(0.04), Some(0.04)], 0.1),
            vec![0..2, 2..4]
        );
        // A launch beyond the budget stands alone.
        assert_eq!(
            divide(&[Some(0.01), Some(0.5), Some(0.01)], 0.1),
            vec![0..1, 1..2, 2..3]
        );
        assert!(divide(&[], 0.1).is_empty());
    }

    #[test]
    fn costs_learn_unmeasured_kinds_and_scale_measured_ones() {
        let costs = LaunchCosts::<u32>::default();
        assert_eq!(costs.estimate(&[(1, 1)]), None);
        costs.observe(&[(1, 1)], 0.02);
        costs.observe(&[(2, 1)], 0.01);
        assert_eq!(
            costs.estimates(&[1, 2, 3]),
            vec![Some(0.02), Some(0.01), None]
        );
        // Four launches of kind 1 and two of kind 2 estimated 0.1 s took
        // 0.2 s: both kinds double.
        costs.observe(&[(1, 4), (2, 2)], 0.2);
        let estimate = costs.estimate(&[(1, 1), (2, 1)]).expect("both measured");
        assert!((estimate - 0.06).abs() < 1e-12);
        // Repeated launches of one kind share the measured time.
        costs.observe(&[(3, 4)], 0.08);
        assert!((costs.estimate(&[(3, 1)]).expect("measured") - 0.02).abs() < 1e-12);
    }

    #[test]
    fn counted_kinds_keep_first_use_order() {
        assert_eq!(counted(&[3, 1, 3, 3, 2, 1]), vec![(3, 3), (1, 2), (2, 1)]);
    }
}
