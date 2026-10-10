//! Vulkan submission of native dispatch lists (Vulkan backend spec §8.6),
//! the counterpart of `cuda.rs`.
//!
//! A submission of sealed graph runs (one run, or a sequence of runs
//! submitted together) replays secondary command buffers: its launches,
//! with every argument block fixed, recorded once and executed inside the
//! submission's primary buffers. Recorded graphs are keyed by the plans and
//! every buffer address the runs bind; per-step host values live in each
//! run's upload region, which the argument blocks address, never in the
//! blocks themselves.
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

use super::graph_replays::Replays;
use super::{DispatchList, NativeRoute, RouteSubmission};
use crate::api::CallError;
use crate::backends::vulkan_buffer;
use crate::driver::Allocation;
use seismic_compiler::errors::ExecutionError;
use seismic_vulkan::direct::{
    DirectBatch, DirectGraph, DirectGraphBuilder, DirectLaunch, DirectSubmission, LaunchKind,
    TimelineAnchor,
};
use std::any::Any;
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

/// One sealed submission's launches (the non-empty ones, in order) divided
/// into consecutive parts, recorded as one graph with a segment per part;
/// `measured` says whether every launch had a measured time when the parts
/// were formed.
struct Divided {
    launches: Vec<LaunchKind>,
    parts: Vec<Part>,
    graph: Option<DirectGraph>,
    measured: bool,
}

/// A range of a division's launches and its distinct launch kinds with how
/// often each occurs.
struct Part {
    launches: Range<usize>,
    kinds: Vec<(LaunchKind, u32)>,
}

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
    let Some(mut key) = list.plans().filter(|_| timed.is_none()) else {
        let mut batch = match timed {
            Some(launches) => DirectBatch::timed(device, launches),
            None => DirectBatch::new(device),
        }
        .map_err(CallError::Execution)?;
        for _ in 0..repetitions {
            each_launch(list, |launch| match launch {
                Some(launch) => batch.launch(launch),
                None => {
                    batch.skip();
                    Ok(())
                }
            })?;
        }
        let submission = batch.commit().map_err(CallError::Execution)?;
        return Ok(RouteSubmission::Vulkan(VulkanSubmission::new(
            vec![submission],
            None,
        )));
    };
    key.extend(addresses(list));
    let mut queued = Queued::new(device);
    replays.graphs.replay(
        key,
        retained,
        || {
            let mut launches = Vec::new();
            each_launch(list, |launch| {
                launches.extend(launch.map(DirectLaunch::kind));
                Ok(())
            })?;
            Ok(Divided {
                launches,
                parts: Vec::new(),
                graph: None,
                measured: false,
            })
        },
        |divided| {
            divided.refresh(device, list, &replays.costs)?;
            let graph = divided
                .graph
                .as_ref()
                .expect("a refreshed division is recorded");
            for _ in 0..repetitions {
                for (segment, part) in divided.parts.iter().enumerate() {
                    queued.add(graph, segment, part, replays.costs.estimate(&part.kinds))?;
                }
            }
            Ok(())
        },
    )?;
    let submissions = queued.finish()?;
    Ok(RouteSubmission::Vulkan(VulkanSubmission::new(
        submissions.submissions,
        Some(Feedback {
            costs: replays.costs.clone(),
            kinds: submissions.kinds,
        }),
    )))
}

impl Divided {
    /// Form the parts again unless the current ones were formed from
    /// measured launches and are each still estimated within
    /// [`DIVISION_LIMIT_SECONDS`]; record the graph again when they change.
    fn refresh(
        &mut self,
        device: &seismic_vulkan::Device,
        list: &impl DispatchList,
        costs: &LaunchCosts<LaunchKind>,
    ) -> Result<(), CallError> {
        let current = self.measured
            && self.graph.is_some()
            && self.parts.iter().all(|part| {
                costs
                    .estimate(&part.kinds)
                    .is_some_and(|seconds| seconds <= DIVISION_LIMIT_SECONDS)
            });
        if current {
            return Ok(());
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
            return Ok(());
        }
        let mut builder = DirectGraphBuilder::new(device).map_err(CallError::Execution)?;
        let mut index = 0usize;
        let mut part = 0usize;
        each_launch(list, |launch| {
            let Some(launch) = launch else {
                return Ok(());
            };
            while ranges[part].end <= index && part + 1 < ranges.len() {
                part += 1;
                builder.segment()?;
            }
            index += 1;
            builder.launch(launch)
        })?;
        for _ in part + 1..ranges.len() {
            builder.segment().map_err(CallError::Execution)?;
        }
        let graph = builder.instantiate().map_err(CallError::Execution)?;
        self.parts = ranges
            .into_iter()
            .map(|launches| Part {
                kinds: counted(&self.launches[launches.clone()]),
                launches,
            })
            .collect();
        self.measured = measured;
        self.graph = Some(graph);
        Ok(())
    }
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
}

impl<K> Default for LaunchCosts<K> {
    fn default() -> Self {
        Self {
            seconds: Mutex::new(HashMap::new()),
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
    open: Option<(DirectBatch, Option<f64>, Vec<(LaunchKind, u32)>)>,
    submissions: Vec<DirectSubmission>,
    kinds: Vec<Vec<(LaunchKind, u32)>>,
}

struct QueuedSubmissions {
    submissions: Vec<DirectSubmission>,
    kinds: Vec<Vec<(LaunchKind, u32)>>,
}

impl<'a> Queued<'a> {
    fn new(device: &'a seismic_vulkan::Device) -> Self {
        Self {
            device,
            open: None,
            submissions: Vec::new(),
            kinds: Vec::new(),
        }
    }

    fn add(
        &mut self,
        graph: &DirectGraph,
        segment: usize,
        part: &Part,
        estimate: Option<f64>,
    ) -> Result<(), CallError> {
        let joins = matches!(
            (&self.open, estimate),
            (Some((_, Some(queued), _)), Some(estimate)) if queued + estimate <= SUBMISSION_SECONDS
        );
        if !joins {
            self.commit()?;
            let batch = DirectBatch::new(self.device).map_err(CallError::Execution)?;
            self.open = Some((batch, Some(0.0), Vec::new()));
        }
        let (batch, queued, kinds) = self.open.as_mut().expect("a batch is open");
        batch
            .replay_segment(graph, segment)
            .map_err(CallError::Execution)?;
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

    fn finish(mut self) -> Result<QueuedSubmissions, CallError> {
        if self.open.is_none() && self.submissions.is_empty() {
            // Nothing was queued (no repetition): an empty batch still
            // orders the submission and reports its completion.
            let batch = DirectBatch::new(self.device).map_err(CallError::Execution)?;
            self.open = Some((batch, Some(0.0), Vec::new()));
        }
        self.commit()?;
        Ok(QueuedSubmissions {
            submissions: std::mem::take(&mut self.submissions),
            kinds: std::mem::take(&mut self.kinds),
        })
    }
}

impl Drop for Queued<'_> {
    /// A failure after some batches were committed leaves them running on
    /// storage the caller regains: observe them before returning it.
    fn drop(&mut self) {
        for submission in &self.submissions {
            submission.wait_complete();
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
    submissions: Vec<DirectSubmission>,
    feedback: Mutex<Option<Feedback>>,
}

impl VulkanSubmission {
    fn new(submissions: Vec<DirectSubmission>, feedback: Option<Feedback>) -> Self {
        assert!(
            !submissions.is_empty(),
            "VulkanSubmission::new precondition: at least one queue submission"
        );
        Self {
            submissions,
            feedback: Mutex::new(feedback),
        }
    }

    fn last(&self) -> &DirectSubmission {
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
            if let Ok(seconds) = submission.device_seconds() {
                feedback.costs.observe(kinds, seconds);
            }
        }
    }

    pub(crate) fn is_complete(&self) -> bool {
        let complete = self.last().is_complete();
        if complete {
            self.observe();
        }
        complete
    }

    pub(crate) fn wait_complete(&self) {
        self.last().wait_complete();
        if self.last().finish().is_ok() {
            self.observe();
        }
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
            Ok(total + submission.device_seconds()?)
        })
    }

    pub(crate) fn device_interval(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<(f64, f64), ExecutionError> {
        let (start, _) = self.submissions[0].device_interval(anchor)?;
        let (_, end) = self.last().device_interval(anchor)?;
        Ok((start, end))
    }

    /// Per-launch intervals of a timed submission, which is always one
    /// queue submission.
    pub(crate) fn launch_intervals(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        match self.submissions.as_slice() {
            [submission] => submission.launch_intervals(anchor),
            _ => Ok(None),
        }
    }
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
        let NativeRoute::Vulkan { launches, .. } = &dispatch.kernel.route else {
            unreachable!("one device has one native route");
        };
        typed.clear();
        typed.extend(
            buffers
                .iter()
                .map(|(allocation, offset)| (vulkan_buffer(allocation), *offset)),
        );
        let scalars = vulkan_buffer(&dispatch.kernel.scalars);
        for (ordinal, launch) in dispatch.launches.iter().enumerate() {
            let (module, function) = launches.launch(ordinal);
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

#[cfg(test)]
mod tests {
    use super::*;

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
