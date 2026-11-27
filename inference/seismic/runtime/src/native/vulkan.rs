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
//! one), are recorded launch by launch and divided into queue submissions under
//! the same [`SUBMISSION_SECONDS`] budget so multi-repetition tuning and long
//! traces stay clear of driver watchdogs too.
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
use super::hazard::{self, LevelLaunch, PlacementScratch};
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
    Measured, Timed, TimelineAnchor, Timing, Unmeasured,
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
    placement: Mutex<PlacementScratch>,
}

impl VulkanReplays {
    /// What the graphs cost serving so far (`Replays::counts`), each
    /// division formed again (`Divided::refresh`) among the recordings.
    pub(crate) fn counts(&self) -> super::graph_replays::GraphReplays {
        self.graphs.counts()
    }

    /// Release graphs beyond `budget` (`Replays::fit_to`).
    pub(crate) fn fit_to(&self, budget: GraphCensus) {
        self.graphs.fit_to(budget);
    }

    /// What the graphs of plan `plan`'s runs cost so far.
    pub(crate) fn plan_counts(&self, plan: u64) -> super::graph_replays::GraphReplays {
        self.graphs.plan_counts(plan)
    }

    /// Track a live plan key in `by_plan` so it is pruned when dropped.
    pub(crate) fn register_plan(&self, plan: u64) {
        self.graphs.register_plan(plan);
    }

    /// Release a live plan key from `by_plan` when dropped.
    pub(crate) fn unregister_plan(&self, plan: u64) {
        self.graphs.unregister_plan(plan);
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
    /// Launches that begin a new watchdog-bounded repeat round and must open a
    /// fresh queue submission rather than coalescing with earlier launches.
    split_before: Vec<bool>,
    /// ASAP hazard-level schedule of the segment's launches (in recorded order).
    schedule: Vec<LevelLaunch>,
    /// The marks the graph holds, each the scheduled launch it precedes, in order.
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
    split_before: bool,
}

/// Lazily-created side-queue transfer queue used by `submit_fill_copies` on
/// Vulkan devices that expose a distinct `side_family`.
pub(crate) struct VulkanFillQueue {
    queue: Arc<seismic_vulkan::transfer::TransferQueue>,
    producer_queue_id: u64,
    next: std::sync::atomic::AtomicU64,
}

/// A GPU-side timeline wait together with an optional owner handle (such as
/// `Arc<TransferQueue>`) that keeps the waited `vk::Semaphore` alive until
/// the waiting submission retires on the GPU.
struct DeviceWait {
    semaphore: seismic_vulkan::vk::Semaphore,
    value: u64,
    owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

/// Collect GPU-side `(semaphore, value, owner)` waits from cross-queue
/// allocation fences (`after`) and explicit transfer handles
/// (`after_transfers`), deduplicating to the maximum value per semaphore. Any
/// fence that is not backed by a GPU-submitted Vulkan timeline (or is still
/// held behind a host gate) is waited on the host so we never emit a "wait
/// before signal".
fn collect_device_waits(
    device_id: u64,
    after: &[Arc<dyn crate::completion::DeviceCompletion>],
    after_transfers: &[super::transfer::TransferSync],
) -> Result<Vec<DeviceWait>, ExecutionError> {
    let mut waits: Vec<DeviceWait> = Vec::new();
    let mut push_wait = |semaphore: seismic_vulkan::vk::Semaphore,
                         value: u64,
                         owner: Option<Arc<dyn std::any::Any + Send + Sync>>| {
        if let Some(existing) = waits.iter_mut().find(|wait| wait.semaphore == semaphore) {
            existing.value = existing.value.max(value);
            if existing.owner.is_none() {
                existing.owner = owner;
            }
        } else {
            waits.push(DeviceWait { semaphore, value, owner });
        }
    };
    for fence in after {
        if fence.completed()? {
            continue;
        }
        if let Some(route) = fence.native() {
            if let Committed::Vulkan(submission) = route.committed() {
                if let Some((semaphore, value, owner)) = submission.ready_timeline_wait() {
                    push_wait(semaphore, value, owner);
                    continue;
                }
            }
        }
        fence.ended()?;
    }
    for sync in after_transfers {
        if sync.id().device_id != device_id {
            return Err(ExecutionError::SubmissionFailed(format!(
                "transfer sync {:?} belongs to device {}, not Vulkan device {device_id}",
                sync.id(),
                sync.id().device_id,
            )));
        }
        let Some((semaphore, value, owner)) = sync.vulkan_wait() else {
            return Err(ExecutionError::SubmissionFailed(format!(
                "transfer sync {:?} is not backed by a Vulkan timeline semaphore",
                sync.id(),
            )));
        };
        push_wait(semaphore, value, Some(owner));
    }
    Ok(waits)
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
    let Encoding { device_id, list, repetitions, observed, retained, after, after_transfers, .. } = *work;
    let device_waits = collect_device_waits(device_id, after, after_transfers)
        .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    let timed = work.timed.map(|(_, launches)| launches);
    // The calls between timeline boundaries form one segment, and a sealed
    // run's of them a graph. A boundary ends a queue submission: the next
    // one waits on its gates' timelines, and the one it ends signals its
    // signals'.
    let encoded = list.encoded();
    let Some(runs) = list.runs().filter(|_| timed.is_none()) else {
        let segments = graph_segments(list, None);
        let mut submissions = Issuing::default();
        let issued = match timed {
            Some(launches) => encode_ungraphed(
                list,
                repetitions,
                &segments,
                &encoded,
                &replays.costs,
                false,
                &device_waits,
                &mut submissions,
                || DirectBatch::timed(device, launches),
                QueueSubmission::Timed,
            ),
            None => encode_ungraphed(
                list,
                repetitions,
                &segments,
                &encoded,
                &replays.costs,
                true,
                &device_waits,
                &mut submissions,
                || DirectBatch::new(device),
                QueueSubmission::Measured,
            ),
        };
        let kinds = match issued {
            Ok(kinds) => kinds,
            Err(error) => return Err(submissions.failed(error)),
        };
        let feedback = Feedback {
            costs: Arc::clone(&replays.costs),
            kinds,
        };
        return Ok(Committed::Vulkan(submissions.into_submission(Some(feedback))));
    };
    let segments = graph_segments(list, Some(&runs));
    let keys = ReplayKeys { runs, encoded: Some(&encoded) };
    let (graphed, shaped): (Vec<Range<usize>>, Vec<Shaped>) = {
        let mut scratch = replays
            .placement
            .lock()
            .expect("vulkan placement scratch is never poisoned");
        segments
            .iter()
            .filter(|(calls, _)| !calls.is_empty())
            .map(|(calls, _)| {
                let (bound, grids) = bindings(list, calls.clone(), &encoded);
                let placement = hazard::relative_placement_with(&mut scratch, list, calls.clone());
                (calls.clone(), keys.shaped(calls, grids.into_iter().chain(placement), bound, Vec::new()))
            })
            .unzip()
    };
    // Recording a division takes nothing a park's answer needs (a Vulkan
    // park is a submission boundary, and recording command buffers holds
    // no device-wide lock), so it waits for nothing.
    let taken = replays.graphs.record(&shaped, budget, retained, || Ok(()), |index| {
        let marks_in: Vec<usize> = keys.marks(&graphed[index]).collect();
        let (schedule, marks) = hazard::schedule_calls(list, graphed[index].clone(), &encoded, &marks_in);
        let mut launches = Vec::with_capacity(schedule.len());
        let split_before = schedule.iter().map(|launch| launch.split_before).collect();
        each_scheduled_launch(list, graphed[index].start, &schedule, |launch, _, _| {
            launches.push(launch.kind());
            Ok(())
        })?;
        Ok(Divided {
            launches,
            split_before,
            schedule,
            marks,
            parts: Vec::new(),
            graph: None,
            measured: false,
            stale: false,
        })
    })
    .map_err(EncodeFailure::uncommitted)?;
    let mut queued = Queued::new(device, &replays.costs, observed, device_waits);
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
                            if divided.refresh(device, list, calls.start, &replays.costs)? {
                                replays.graphs.count_recorded(&shaped.key);
                            }
                            // The launches' argument blocks as this run
                            // binds them, in scheduled launch order, which its
                            // first part rewrites.
                            let mut rebind = None;
                            if std::mem::take(&mut divided.stale) {
                                let mut blocks = Vec::with_capacity(divided.schedule.len());
                                each_scheduled_launch(list, calls.start, &divided.schedule, |launch, _, _| {
                                    blocks.push(launch.argument_block());
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
        calls_start: usize,
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
        let mut ranges = divide(&estimates, &self.split_before, SUBMISSION_SECONDS);
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
        each_scheduled_launch(list, calls_start, &self.schedule, |launch, barrier, _| {
            while ends[segment] <= index && segment + 1 < ends.len() {
                segment += 1;
                builder.segment()?;
            }
            index += 1;
            builder.launch_with_barrier(launch, barrier)
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
                split_before: self.split_before.get(launches.start).copied().unwrap_or(false),
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

/// Consecutive launch ranges: a launch without an estimate alone, a launch
/// with `split_before` starting a fresh range, and the rest grouped while
/// their estimated time stays within `seconds` (a launch estimated beyond it
/// alone).
fn divide(estimates: &[Option<f64>], split_before: &[bool], seconds: f64) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut sum = 0.0;
    for (index, estimate) in estimates.iter().enumerate() {
        let split = split_before.get(index).copied().unwrap_or(false);
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
                if start < index && (split || sum + estimate > seconds) {
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

    /// The estimated time of one launch of kind `kind`, if measured.
    fn estimate_one(&self, kind: K) -> Option<f64> {
        self.lock().get(&kind).copied()
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

fn fits_submission(queued: Option<f64>, estimate: Option<f64>) -> bool {
    matches!((queued, estimate), (Some(queued), Some(estimate)) if queued + estimate <= SUBMISSION_SECONDS)
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
    /// Device-signalled timeline semaphores the first batch opened waits on
    /// (from `after` fill copies and `after_transfers`), retaining their
    /// owners until the waiting submission retires.
    device_waits: Vec<DeviceWait>,
    submissions: Issuing,
    kinds: Vec<Vec<(LaunchKind, u32)>>,
}

struct QueuedSubmissions {
    submissions: Issuing,
    kinds: Vec<Vec<(LaunchKind, u32)>>,
}

impl<'a> Queued<'a> {
    fn new(
        device: &'a seismic_vulkan::Device,
        costs: &'a LaunchCosts<LaunchKind>,
        observed: bool,
        device_waits: Vec<DeviceWait>,
    ) -> Self {
        Self {
            device,
            costs,
            observed,
            open: None,
            gates: Vec::new(),
            device_waits,
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
        for wait in self.device_waits.drain(..) {
            batch.wait_device_wait(wait);
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
        let joins = !part.split_before
            && self
                .open
                .as_ref()
                .is_some_and(|(_, queued, _)| fits_submission(*queued, estimate));
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

/// One unit of native work on Vulkan: either one or more compute queue
/// submissions in commit order, or a side-queue fill-copy batch on
/// `TransferQueue`.
pub(crate) struct VulkanSubmission {
    kind: VulkanSubmissionKind,
}

enum VulkanSubmissionKind {
    Compute {
        submissions: Vec<QueueSubmission>,
        feedback: Mutex<Option<Feedback>>,
    },
    Fill {
        queue: Arc<seismic_vulkan::transfer::TransferQueue>,
        value: u64,
        sync: super::transfer::TransferSync,
    },
}

/// Queue one ungraphed launch into `batch`, splitting into a new queue
/// submission if `split_before` is set or adding it would exceed
/// [`SUBMISSION_SECONDS`] (or either side is unmeasured).
#[allow(clippy::too_many_arguments)]
fn queue_ungraphed_launch<T: Timing>(
    launch: &DirectLaunch<'_>,
    barrier: bool,
    split_before: bool,
    costs: &LaunchCosts<LaunchKind>,
    batch: &mut DirectBatch<T>,
    queued: &mut Option<f64>,
    kinds: &mut Vec<(LaunchKind, u32)>,
    batch_kinds: &mut Vec<Vec<(LaunchKind, u32)>>,
    submissions: &mut Issuing,
    new_batch: &mut impl FnMut() -> Result<DirectBatch<T>, ExecutionError>,
    wrap: &impl Fn(DirectSubmission<T>) -> QueueSubmission,
) -> Result<(), ExecutionError> {
    let kind = launch.kind();
    let estimate = costs.estimate_one(kind);
    if !kinds.is_empty() && (split_before || !fits_submission(*queued, estimate)) {
        let next = new_batch()?;
        submissions.push(wrap(std::mem::replace(batch, next).commit()?));
        batch_kinds.push(std::mem::take(kinds));
        *queued = Some(0.0);
    }
    *queued = if kinds.is_empty() {
        estimate
    } else {
        queued.zip(estimate).map(|(queued, estimate)| queued + estimate)
    };
    match kinds.iter_mut().find(|(seen, _)| *seen == kind) {
        Some((_, count)) => *count += 1,
        None => kinds.push((kind, 1)),
    }
    batch.launch_with_barrier(launch, barrier)
}

/// Encode an ungraphed list (timed or plain measured) into one or more queue
/// submissions split at timeline boundaries and whenever adding the next launch
/// would exceed [`SUBMISSION_SECONDS`] (or either the batch or launch is
/// unmeasured, matching [`Queued`]). Returns each committed submission's launch
/// kinds so completion teaches [`LaunchCosts`].
#[allow(clippy::too_many_arguments)]
fn encode_ungraphed<T: Timing>(
    list: &impl DispatchList,
    repetitions: usize,
    segments: &[(Range<usize>, usize)],
    encoded: &[Vec<bool>],
    costs: &LaunchCosts<LaunchKind>,
    use_asap: bool,
    device_waits: &[DeviceWait],
    submissions: &mut Issuing,
    mut new_batch: impl FnMut() -> Result<DirectBatch<T>, ExecutionError>,
    wrap: impl Fn(DirectSubmission<T>) -> QueueSubmission,
) -> Result<Vec<Vec<(LaunchKind, u32)>>, CallError> {
    let mut batch = new_batch().map_err(CallError::Execution)?;
    for wait in device_waits {
        match &wait.owner {
            Some(owner) => batch.wait_device_semaphore_owned(wait.semaphore, wait.value, Arc::clone(owner)),
            None => batch.wait_device_semaphore(wait.semaphore, wait.value),
        }
    }
    let mut queued: Option<f64> = Some(0.0);
    let mut kinds: Vec<(LaunchKind, u32)> = Vec::new();
    let mut batch_kinds: Vec<Vec<(LaunchKind, u32)>> = Vec::new();
    let schedules: Vec<Vec<LevelLaunch>> = if use_asap {
        segments
            .iter()
            .map(|(calls, _)| hazard::schedule_calls(list, calls.clone(), encoded, &[]).0)
            .collect()
    } else {
        Vec::new()
    };
    for _ in 0..repetitions {
        for (seg_idx, (calls, boundary)) in segments.iter().enumerate() {
            if use_asap {
                let mut first = true;
                each_scheduled_launch(list, calls.start, &schedules[seg_idx], |launch, barrier, split_before| {
                    // The first launch of each segment/repetition must barrier
                    // against any earlier work already recorded in `batch`;
                    // `DirectBatch::launch_with_barrier` elides it when nothing
                    // precedes it in the batch.
                    let barrier = barrier || std::mem::take(&mut first);
                    queue_ungraphed_launch(
                        launch,
                        barrier,
                        split_before,
                        costs,
                        &mut batch,
                        &mut queued,
                        &mut kinds,
                        &mut batch_kinds,
                        submissions,
                        &mut new_batch,
                        &wrap,
                    )
                })?;
            } else {
                each_launch(list, calls.clone(), encoded, |launch, split_before| match launch {
                    Some(launch) => queue_ungraphed_launch(
                        launch,
                        true,
                        split_before,
                        costs,
                        &mut batch,
                        &mut queued,
                        &mut kinds,
                        &mut batch_kinds,
                        submissions,
                        &mut new_batch,
                        &wrap,
                    ),
                    None => {
                        batch.skip();
                        Ok(())
                    }
                })?;
            }
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
            let next = new_batch().map_err(CallError::Execution)?;
            submissions.push(wrap(
                std::mem::replace(&mut batch, next)
                    .commit()
                    .map_err(CallError::Execution)?,
            ));
            batch_kinds.push(std::mem::take(&mut kinds));
            queued = Some(0.0);
            for operation in operations {
                if operation.action == TimelineAction::Wait {
                    batch.wait_timeline(&vulkan_timeline(operation).gate, operation.value);
                }
            }
        }
    }
    submissions.push(wrap(batch.commit().map_err(CallError::Execution)?));
    batch_kinds.push(kinds);
    Ok(batch_kinds)
}

/// One queue submission of a [`VulkanSubmission`]: measured (its device
/// interval read once it completes: its kinds' costs learned from it, a
/// trace placed by it), timed (each launch's device interval read too), or not.
enum QueueSubmission {
    Measured(DirectSubmission<Measured>),
    Timed(DirectSubmission<Timed>),
    Unmeasured(DirectSubmission<Unmeasured>),
}

impl QueueSubmission {
    fn completed(&self) -> Result<bool, ExecutionError> {
        match self {
            Self::Measured(submission) => submission.completed(),
            Self::Timed(submission) => submission.completed(),
            Self::Unmeasured(submission) => submission.completed(),
        }
    }

    fn finish(&self) -> Result<(), ExecutionError> {
        match self {
            Self::Measured(submission) => submission.finish(),
            Self::Timed(submission) => submission.finish(),
            Self::Unmeasured(submission) => submission.finish(),
        }
    }

    fn handed(&self) -> Option<(std::time::Instant, std::time::Instant)> {
        match self {
            Self::Measured(submission) => submission.handed(),
            Self::Timed(submission) => submission.handed(),
            Self::Unmeasured(submission) => submission.handed(),
        }
    }

    fn timeline_wait(&self) -> (seismic_vulkan::vk::Semaphore, u64) {
        match self {
            Self::Measured(submission) => submission.timeline_wait(),
            Self::Timed(submission) => submission.timeline_wait(),
            Self::Unmeasured(submission) => submission.timeline_wait(),
        }
    }

    fn opened(&self) -> Option<(std::time::Instant, seismic_native_target::GateOpening)> {
        match self {
            Self::Measured(submission) => submission.opened(),
            Self::Timed(submission) => submission.opened(),
            Self::Unmeasured(submission) => submission.opened(),
        }
    }

    fn device_seconds(&self) -> Result<f64, ExecutionError> {
        match self {
            Self::Measured(submission) => submission.device_seconds(),
            Self::Timed(submission) => submission.device_seconds(),
            Self::Unmeasured(_) => Err(ExecutionError::SubmissionFailed(
                "an unobserved Vulkan submission's device time was read".into(),
            )),
        }
    }

    fn device_interval(&self, anchor: &TimelineAnchor) -> Result<(f64, f64), ExecutionError> {
        match self {
            Self::Measured(submission) => submission.device_interval(anchor),
            Self::Timed(submission) => submission.device_interval(anchor),
            Self::Unmeasured(_) => Err(ExecutionError::SubmissionFailed(
                "an unobserved Vulkan submission's device time was read".into(),
            )),
        }
    }

    fn launch_intervals(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        match self {
            Self::Timed(submission) => submission.launch_intervals(anchor).map(Some),
            Self::Measured(_) => Ok(None),
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

    fn wait_device_wait(&mut self, wait: DeviceWait) {
        match (self, wait.owner) {
            (Self::Measured(batch), Some(owner)) => {
                batch.wait_device_semaphore_owned(wait.semaphore, wait.value, owner)
            }
            (Self::Unmeasured(batch), Some(owner)) => {
                batch.wait_device_semaphore_owned(wait.semaphore, wait.value, owner)
            }
            (Self::Measured(batch), None) => batch.wait_device_semaphore(wait.semaphore, wait.value),
            (Self::Unmeasured(batch), None) => batch.wait_device_semaphore(wait.semaphore, wait.value),
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

/// Encode ordered moves (`super::moves`) as one queue submission of
/// copies and word writes (`DirectBatch::transfer`), which the queue runs
/// after every earlier submission and any fill copies in `after`: refused,
/// nothing submitted, where a step is refused or the submission fails.
pub(super) fn encode_moves(
    device: &seismic_vulkan::Device,
    device_id: u64,
    after: &[Arc<dyn crate::completion::DeviceCompletion>],
    steps: &[super::moves::MoveStep<'_>],
) -> Result<Committed, ExecutionError> {
    use super::moves::MoveStep;
    use crate::backends::vulkan_transfer_end;
    let (mut copies, mut words) = (Vec::new(), Vec::new());
    for step in steps {
        match *step {
            MoveStep::Copy { source, source_offset, destination, destination_offset, bytes } => {
                copies.push((vulkan_transfer_end(source, source_offset), vulkan_transfer_end(destination, destination_offset), bytes))
            }
            MoveStep::Word { destination, offset, value } => words.push((vulkan_transfer_end(destination, offset), value)),
        }
    }
    let device_waits = collect_device_waits(device_id, after, &[])?;
    let mut batch = DirectBatch::unmeasured(device)?;
    for wait in device_waits {
        match wait.owner {
            Some(owner) => batch.wait_device_semaphore_owned(wait.semaphore, wait.value, owner),
            None => batch.wait_device_semaphore(wait.semaphore, wait.value),
        }
    }
    batch.transfer(&copies, &words)?;
    let submission = QueueSubmission::Unmeasured(batch.commit()?);
    Ok(Committed::Vulkan(VulkanSubmission::new(vec![submission], None)))
}

/// Encode a batch of byte copies on Vulkan: on `side_family` via
/// `TransferQueue::copy_after` when the device exposes a side queue (waiting
/// on compute readers in `after` at `TRANSFER` so no slot is overwritten until
/// the final reader retires, and returning a timeline-semaphore-backed
/// `TransferSync`), or on the compute queue via `DirectBatch::transfer` on
/// single-queue Vulkan devices.
pub(super) fn encode_fill_copies(
    device: &seismic_vulkan::Device,
    fill_queue_slot: &Mutex<Option<Arc<VulkanFillQueue>>>,
    device_id: u64,
    after: &[Arc<dyn crate::completion::DeviceCompletion>],
    runs: &[super::FillCopyRun<'_>],
) -> Result<Committed, EncodeFailure> {
    use crate::backends::vulkan_transfer_end;
    let device_waits = collect_device_waits(device_id, after, &[])
        .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    if !device.has_side_queue() {
        let copies = runs
            .iter()
            .filter(|run| run.bytes > 0)
            .map(|run| {
                (
                    vulkan_transfer_end(run.source.allocation(), run.source.byte_offset() + run.source_offset),
                    vulkan_transfer_end(
                        run.destination.allocation(),
                        run.destination.byte_offset() + run.destination_offset,
                    ),
                    run.bytes,
                )
            })
            .collect::<Vec<_>>();
        let mut batch = DirectBatch::unmeasured(device)
            .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
        for wait in device_waits {
            match wait.owner {
                Some(owner) => batch.wait_device_semaphore_owned(wait.semaphore, wait.value, owner),
                None => batch.wait_device_semaphore(wait.semaphore, wait.value),
            }
        }
        batch
            .transfer(&copies, &[])
            .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
        let submission = QueueSubmission::Unmeasured(
            batch
                .commit()
                .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?,
        );
        return Ok(Committed::Vulkan(VulkanSubmission::new(vec![submission], None)));
    }
    let fill_queue = {
        let mut slot = fill_queue_slot
            .lock()
            .expect("vulkan fill queue lock is never poisoned");
        if let Some(existing) = slot.as_ref() {
            Arc::clone(existing)
        } else {
            let queue = Arc::new(
                seismic_vulkan::transfer::TransferQueue::new(device)
                    .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?,
            );
            let created = Arc::new(VulkanFillQueue {
                queue,
                producer_queue_id: super::transfer::next_queue_id(),
                next: std::sync::atomic::AtomicU64::new(1),
            });
            *slot = Some(Arc::clone(&created));
            created
        }
    };
    let items: Vec<seismic_vulkan::transfer::TransferItem<'_>> = runs
        .iter()
        .map(|run| {
            seismic_vulkan::transfer::TransferItem::Run(seismic_vulkan::transfer::TransferRun {
                source: vulkan_transfer_end(run.source.allocation(), run.source.byte_offset() + run.source_offset),
                destination: vulkan_transfer_end(
                    run.destination.allocation(),
                    run.destination.byte_offset() + run.destination_offset,
                ),
                bytes: run.bytes,
            })
        })
        .collect();
    let raw_waits: Vec<(seismic_vulkan::vk::Semaphore, u64)> = device_waits
        .iter()
        .map(|wait| (wait.semaphore, wait.value))
        .collect();
    let wait_owners: Vec<Arc<dyn std::any::Any + Send + Sync>> = device_waits
        .into_iter()
        .filter_map(|wait| wait.owner)
        .collect();
    // Caller holds `device.native.order`, serializing value assignment and
    // submission on `fill_queue` so timeline values are strictly increasing.
    let value = fill_queue.next.load(std::sync::atomic::Ordering::Relaxed);
    fill_queue
        .queue
        .copy_after(&items, value, &raw_waits, wait_owners)
        .map_err(|error| EncodeFailure::uncommitted(CallError::Execution(error)))?;
    fill_queue
        .next
        .store(value + 1, std::sync::atomic::Ordering::Relaxed);
    let sync = super::transfer::TransferSync::from_vulkan_queue(
        device_id,
        fill_queue.producer_queue_id,
        value,
        Arc::clone(&fill_queue.queue),
    );
    Ok(Committed::Vulkan(VulkanSubmission::fill(
        Arc::clone(&fill_queue.queue),
        value,
        sync,
    )))
}

impl VulkanSubmission {
    fn new(submissions: Vec<QueueSubmission>, feedback: Option<Feedback>) -> Self {
        assert!(
            !submissions.is_empty(),
            "VulkanSubmission::new precondition: at least one queue submission"
        );
        Self {
            kind: VulkanSubmissionKind::Compute {
                submissions,
                feedback: Mutex::new(feedback),
            },
        }
    }

    fn fill(
        queue: Arc<seismic_vulkan::transfer::TransferQueue>,
        value: u64,
        sync: super::transfer::TransferSync,
    ) -> Self {
        Self {
            kind: VulkanSubmissionKind::Fill { queue, value, sync },
        }
    }

    /// The `(semaphore, value, owner)` timeline wait for this submission when
    /// its signal is already submitted to a Vulkan queue (never parked behind
    /// a host gate).
    pub(crate) fn ready_timeline_wait(
        &self,
    ) -> Option<(
        seismic_vulkan::vk::Semaphore,
        u64,
        Option<Arc<dyn std::any::Any + Send + Sync>>,
    )> {
        match &self.kind {
            VulkanSubmissionKind::Compute { submissions, .. } => {
                let last = submissions.last().expect("a Vulkan submission has a queue submission");
                last.handed().is_some().then(|| {
                    let (semaphore, value) = last.timeline_wait();
                    (semaphore, value, None)
                })
            }
            VulkanSubmissionKind::Fill { queue, value, .. } => Some((
                queue.semaphore(),
                *value,
                Some(Arc::clone(queue) as Arc<dyn std::any::Any + Send + Sync>),
            )),
        }
    }

    /// The `TransferSync` handle if this submission is a side-queue fill copy.
    #[allow(dead_code)]
    pub(crate) fn transfer_sync(&self) -> Option<super::transfer::TransferSync> {
        match &self.kind {
            VulkanSubmissionKind::Compute { .. } => None,
            VulkanSubmissionKind::Fill { sync, .. } => Some(sync.clone()),
        }
    }

    /// Teach the launch costs once, after completion.
    fn observe(&self) {
        let VulkanSubmissionKind::Compute { submissions, feedback } = &self.kind else {
            return;
        };
        let Some(feedback) = feedback
            .lock()
            .expect("submission feedback lock is never poisoned")
            .take()
        else {
            return;
        };
        for (submission, kinds) in submissions.iter().zip(&feedback.kinds) {
            let seconds = match submission {
                QueueSubmission::Measured(submission) => submission.device_seconds().ok(),
                QueueSubmission::Timed(submission) => submission.device_seconds().ok(),
                QueueSubmission::Unmeasured(_) => None,
            };
            if let Some(seconds) = seconds {
                feedback.costs.observe(kinds, seconds);
            }
        }
    }

    /// Whether every queue submission completed, without waiting: the last
    /// completes after every earlier one, and a refusal or the device's
    /// loss reaches it too.
    pub(crate) fn completed(&self) -> Result<bool, ExecutionError> {
        match &self.kind {
            VulkanSubmissionKind::Compute { submissions, .. } => {
                let complete = submissions
                    .last()
                    .expect("a Vulkan submission has a queue submission")
                    .completed()?;
                if complete {
                    self.observe();
                }
                Ok(complete)
            }
            VulkanSubmissionKind::Fill { queue, value, .. } => queue.reached(*value),
        }
    }

    pub(crate) fn finish(&self) -> Result<(), ExecutionError> {
        match &self.kind {
            VulkanSubmissionKind::Compute { submissions, .. } => {
                for submission in submissions {
                    submission.finish()?;
                }
                self.observe();
                Ok(())
            }
            VulkanSubmissionKind::Fill { queue, value, .. } => {
                while !queue.wait(*value, std::time::Duration::from_secs(1))? {}
                Ok(())
            }
        }
    }

    pub(crate) fn device_seconds(&self) -> Result<f64, ExecutionError> {
        match &self.kind {
            VulkanSubmissionKind::Compute { submissions, .. } => {
                submissions.iter().try_fold(0.0, |total, submission| {
                    Ok(total + submission.device_seconds()?)
                })
            }
            VulkanSubmissionKind::Fill { .. } => Err(ExecutionError::SubmissionFailed(
                "a Vulkan fill copy submission has no compute timestamps".into(),
            )),
        }
    }

    pub(crate) fn device_interval(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<(f64, f64), ExecutionError> {
        match &self.kind {
            VulkanSubmissionKind::Compute { submissions, .. } => {
                let (start, _) = submissions[0].device_interval(anchor)?;
                let (_, end) = submissions
                    .last()
                    .expect("a Vulkan submission has a queue submission")
                    .device_interval(anchor)?;
                Ok((start, end))
            }
            VulkanSubmissionKind::Fill { .. } => Err(ExecutionError::SubmissionFailed(
                "a Vulkan fill copy submission has no compute timestamps".into(),
            )),
        }
    }

    /// Each queue submission's hand-off to the queue and device interval,
    /// on the anchor's host clock, in commit order. Reading its interval
    /// waits for its completion, and a completed one was handed over.
    pub(crate) fn queue_submissions(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Vec<super::trace::TracedQueueSubmission>, ExecutionError> {
        match &self.kind {
            VulkanSubmissionKind::Compute { submissions, .. } => submissions
                .iter()
                .map(|submission| {
                    let device = submission.device_interval(anchor)?;
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
                .collect(),
            VulkanSubmissionKind::Fill { .. } => Err(ExecutionError::SubmissionFailed(
                "a Vulkan fill copy submission has no compute timestamps".into(),
            )),
        }
    }

    /// Per-launch intervals of a timed submission: one queue submission,
    /// or one per timeline segment, each timing its own launches in order.
    pub(crate) fn launch_intervals(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        match &self.kind {
            VulkanSubmissionKind::Compute { submissions, .. } => {
                let mut intervals = Vec::new();
                for submission in submissions {
                    let Some(own) = submission.launch_intervals(anchor)? else {
                        return Ok(None);
                    };
                    intervals.extend(own);
                }
                Ok(Some(intervals))
            }
            VulkanSubmissionKind::Fill { .. } => Ok(None),
        }
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

/// A launch of call `index` as it is encoded: `None` when inactive, left
/// out by `encoded`, or empty ([`super::LaunchGeometry::is_empty`]): an
/// empty launch does no device work, so no path records or dispatches it.
fn encoded_launch(
    launch: Option<super::LaunchGeometry>,
    encoded: &[Vec<bool>],
    index: usize,
    ordinal: usize,
) -> Option<super::LaunchGeometry> {
    launch.filter(|launch| encoded[index][ordinal] && !launch.is_empty())
}

/// Visit every launch of `schedule` (with offsets relative to `base`) in
/// scheduled order, passing whether a compute barrier must precede it and
/// whether it begins a watchdog-bounded repeat round.
fn each_scheduled_launch(
    list: &impl DispatchList,
    base: usize,
    schedule: &[LevelLaunch],
    mut visit: impl FnMut(&DirectLaunch<'_>, bool, bool) -> Result<(), ExecutionError>,
) -> Result<(), CallError> {
    let mut buffers = Vec::new();
    let mut typed = Vec::new();
    for &LevelLaunch { offset, ordinal, barrier, split_before } in schedule {
        let index = base + offset;
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
        let launch = dispatch.launches[ordinal]
            .expect("scheduled launch is active and non-empty");
        let (module, function) = launches.launch(ordinal);
        let direct = DirectLaunch {
            module,
            function,
            buffers: &typed,
            words: super::word_bytes(dispatch.words),
            scalar_results: scalars,
            groups: launch.groups,
            grid: launch.grid.map(|buffer| typed[dispatch.kernel.argument(buffer)]),
        };
        visit(&direct, barrier, split_before).map_err(CallError::Execution)?;
    }
    Ok(())
}

/// Visit every launch of the calls `calls` of one pass over `list` in
/// order; `None` for a launch not encoded ([`encoded_launch`]).
fn each_launch(
    list: &impl DispatchList,
    calls: Range<usize>,
    encoded: &[Vec<bool>],
    mut visit: impl FnMut(Option<&DirectLaunch<'_>>, bool) -> Result<(), ExecutionError>,
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
        let repeat_first = dispatch
            .kernel
            .implementation
            .repeat
            .as_ref()
            .filter(|_| dispatch.rounds > 1)
            .map(|repeat| repeat.first);
        for ordinal in dispatch.kernel.implementation.dispatch_order(dispatch.rounds) {
            let launch = encoded_launch(dispatch.launches[ordinal], encoded, index, ordinal).map(|launch| {
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
            visit(launch.as_ref(), repeat_first == Some(ordinal)).map_err(CallError::Execution)?;
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
        for ordinal in dispatch.kernel.implementation.dispatch_order(dispatch.rounds) {
            if let Some(grid) =
                encoded_launch(dispatch.launches[ordinal], encoded, index, ordinal).and_then(|launch| launch.grid)
            {
                grids.push(address(&buffers[dispatch.kernel.argument(grid)]));
            }
        }
    }
    (bound, grids)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only an active, encoded launch with work is encoded: an empty grid or
    /// an empty group is left out like an inactive launch, so recording,
    /// rebinding and the graph's shape all see the same launches.
    #[test]
    fn empty_launches_are_not_encoded() {
        let launch = |groups, threads| {
            Some(super::super::LaunchGeometry { groups, grid: None, barrier: None, threads, shared_bytes: 0 })
        };
        let encoded = vec![vec![true, false]];
        let work = launch([2, 1, 1], [32, 1, 1]);
        assert_eq!(encoded_launch(work, &encoded, 0, 0), work);
        assert_eq!(encoded_launch(work, &encoded, 0, 1), None);
        assert_eq!(encoded_launch(None, &encoded, 0, 0), None);
        assert_eq!(encoded_launch(launch([0, 1, 1], [32, 1, 1]), &encoded, 0, 0), None);
        assert_eq!(encoded_launch(launch([2, 1, 1], [0, 1, 1]), &encoded, 0, 0), None);
    }

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
        assert_eq!(divide(&[None, None, None], &[], 0.1), vec![0..1, 1..2, 2..3]);
        assert_eq!(
            divide(&[Some(0.01), None, Some(0.01), Some(0.01)], &[], 0.1),
            vec![0..1, 1..2, 2..4]
        );
    }

    #[test]
    fn measured_launches_are_grouped_within_the_budget() {
        assert_eq!(divide(&[Some(0.001); 50], &[], 0.1), vec![0..50]);
        assert_eq!(
            divide(&[Some(0.04), Some(0.04), Some(0.04), Some(0.04)], &[], 0.1),
            vec![0..2, 2..4]
        );
        // A launch beyond the budget stands alone.
        assert_eq!(
            divide(&[Some(0.01), Some(0.5), Some(0.01)], &[], 0.1),
            vec![0..1, 1..2, 2..3]
        );
        // A repeat-round boundary starts a fresh submission even when estimates are tiny.
        assert_eq!(
            divide(
                &[Some(0.001), Some(0.001), Some(0.001), Some(0.001), Some(0.001)],
                &[false, true, false, true, false],
                0.1,
            ),
            vec![0..1, 1..3, 3..5]
        );
        assert!(divide(&[], &[], 0.1).is_empty());
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

    #[test]
    fn fits_submission_splits_unmeasured_and_over_budget_launches() {
        let costs = LaunchCosts::<u32>::default();
        assert_eq!(costs.estimate_one(1), None);
        assert!(!fits_submission(Some(0.01), None));
        assert!(!fits_submission(None, Some(0.01)));
        assert!(!fits_submission(None, None));
        costs.observe(&[(1, 1)], 0.04);
        assert_eq!(costs.estimate_one(1), Some(0.04));
        assert!(fits_submission(Some(0.04), costs.estimate_one(1)));
        assert!(!fits_submission(Some(0.08), costs.estimate_one(1)));
    }

    #[test]
    fn collect_device_waits_rejects_foreign_and_non_vulkan_sync() {
        let foreign = super::super::transfer::TransferSync::cpu_for_test(99, 1);
        let err = collect_device_waits(42, &[], &[foreign])
            .err()
            .expect("foreign-device sync must be rejected");
        assert!(err.to_string().contains("belongs to device 99"));

        let non_vulkan = super::super::transfer::TransferSync::cpu_for_test(42, 1);
        let err = collect_device_waits(42, &[], &[non_vulkan])
            .err()
            .expect("non-Vulkan sync must be rejected");
        assert!(err.to_string().contains("not backed by a Vulkan timeline semaphore"));
    }
}
