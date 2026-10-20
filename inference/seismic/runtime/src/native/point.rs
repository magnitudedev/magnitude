//! Work as `seismic-measure` times it: a formed kernel's calls
//! ([`KernelPoint`]), each argument set of a rotation validated and placed
//! once and every pass resubmitting them, and a sealed graph's runs
//! ([`GraphPoint`]).

use super::graph::{
    NativeGraphBindings, NativeGraphCompletion, NativeGraphOutputs, NativeGraphPlan,
    NativeGraphSequence, NativeGraphSlot,
};
use super::{CallError, NativeBoundCall, NativePrepared, NativeSubmission, StandaloneCalls};
use crate::api::device::DeviceInner;
use crate::api::kernel::{EncodedArgs, EncodedOutputs};
use crate::api::tensor::TensorInner;
use seismic_lang::registry::BackendName;
use seismic_measure::{Gauge, Pending, Work};
use seismic_topology::PhysicalIdentity;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

/// A formed kernel's calls over a rotation of argument sets, each placed
/// once with results of its own: a pass is one call of each, in order.
pub struct KernelPoint {
    kernel: Arc<NativePrepared>,
    calls: Vec<NativeBoundCall>,
    arguments: Vec<EncodedArgs>,
    rotation_bytes: u64,
}

impl KernelPoint {
    /// Validate and place every call of `rotation`. Results of each argument
    /// set are allocated once and reused by every pass, or taken from
    /// `outputs` (and added to it) where a caller shares results among the
    /// placements of one point; scratch is the standalone arena.
    pub(crate) fn place(
        kernel: &Arc<NativePrepared>,
        rotation: Vec<EncodedArgs>,
        mut outputs: Option<&mut Vec<Vec<Arc<TensorInner>>>>,
    ) -> Result<Self, CallError> {
        if rotation.is_empty() {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let arguments = rotation.clone();
        let calls = {
            let mut standalone = kernel
                .standalone
                .lock()
                .expect("native standalone-call lock poisoned");
            rotation
                .into_iter()
                .enumerate()
                .map(|(index, args)| {
                    let supplied =
                        outputs
                            .as_ref()
                            .and_then(|pool| pool.get(index))
                            .map(|tensors| {
                                let mut encoded = EncodedOutputs::new();
                                for tensor in tensors {
                                    encoded.push_tensor(tensor.clone());
                                }
                                encoded
                            });
                    let call = kernel.prepare_call(&mut standalone, args, supplied)?;
                    if let Some(pool) = outputs.as_mut() {
                        if index == pool.len() {
                            pool.push(call.results.clone());
                        }
                    }
                    Ok(call)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut distinct = HashSet::new();
        for call in &calls {
            for (allocation, _) in &call.access {
                distinct.insert((allocation.identity(), allocation.bytes()));
            }
        }
        Ok(Self {
            kernel: kernel.clone(),
            calls,
            arguments,
            rotation_bytes: distinct.into_iter().map(|(_, bytes)| bytes).sum(),
        })
    }

    /// Distinct bytes the rotation's calls read and write.
    pub fn rotation_bytes(&self) -> u64 {
        self.rotation_bytes
    }

    /// The formed kernel the point calls.
    pub(crate) fn kernel(&self) -> &Arc<NativePrepared> {
        &self.kernel
    }

    /// The placed calls, one per argument set.
    pub(super) fn calls(&self) -> &[NativeBoundCall] {
        &self.calls
    }

    /// The argument sets of the rotation.
    pub(crate) fn arguments(&self) -> &[EncodedArgs] {
        &self.arguments
    }
}

impl Pending<CallError> for NativeSubmission {
    fn seconds(&self) -> Result<f64, CallError> {
        self.device_seconds()
    }

    fn wait(&self) -> Result<(), CallError> {
        NativeSubmission::wait(self)
    }
}

impl Gauge<CallError> for KernelPoint {
    fn submit(&self, passes: usize) -> Result<Box<dyn Pending<CallError>>, CallError> {
        Ok(Box::new(Work::submit(self, passes)?))
    }
}

impl Work for KernelPoint {
    type Error = CallError;
    type Pending = NativeSubmission;
    type Clock = PhysicalIdentity;

    /// `passes` passes over the rotation as one unit of device work.
    /// Writable state is not restored between passes: every configuration
    /// is timed the same way.
    fn submit(&self, passes: usize) -> Result<NativeSubmission, CallError> {
        StandaloneCalls {
            kernel: &self.kernel,
            calls: &self.calls,
        }
        .submit(passes)
    }

    fn calls_per_pass(&self) -> usize {
        self.calls.len()
    }

    fn clock(&self) -> Option<PhysicalIdentity> {
        clock(&self.kernel.public_device)
    }

    /// The same calls placed again, with results of their own.
    fn gauge(&self) -> Result<Box<dyn Gauge<CallError>>, CallError> {
        Ok(Box::new(Self::place(
            &self.kernel,
            self.arguments.clone(),
            None,
        )?))
    }
}

/// The clock of `device`'s physical device, shared by its executors; none
/// for the CPU, whose clock is not raised.
fn clock(device: &DeviceInner) -> Option<PhysicalIdentity> {
    (device.info.backend() != BackendName::Cpu).then(|| {
        device
            .topology
            .physical_device(device.info.physical())
            .expect("an opened executor's device is of its revision")
            .identity()
            .clone()
    })
}

/// What a pass of a [`GraphPoint`] is timed as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphTiming {
    /// The device time of the run, from its submission's trace: a sample's
    /// passes are that many runs queued as one submission.
    Device,
    /// The host time from a run's submission to its wait's return, each
    /// run submitted once the one before it completed.
    Waited,
    /// The host time of runs submitted back to back, each its own
    /// submission, from the first submission to the return of the wait
    /// for the last.
    Queued,
}

/// A sealed native graph run with fixed bindings, as `seismic-measure`
/// times it: a pass is one run, timed as [`GraphTiming`] says. Host-timed
/// runs complete inside their submission ([`Work::submit`]), so the
/// session's lead before a sample has completed when the sample's first
/// run starts: at an idle device, with the run's first-use costs paid.
pub struct GraphPoint {
    plan: Arc<NativeGraphPlan>,
    bindings: NativeGraphBindings,
    slot: RefCell<NativeGraphSlot>,
    timing: GraphTiming,
}

/// A submitted sample of a [`GraphPoint`]: the device work of its runs, or
/// the host time they took.
pub struct GraphPending(Sampled);

enum Sampled {
    Device(NativeSubmission),
    Host(f64),
}

impl GraphPoint {
    /// Runs of `plan` with `bindings`, timed as `timing` says. A plan with
    /// timeline nodes is refused: its runs wait on or signal a timeline and
    /// are not the same work twice.
    pub(crate) fn new(
        plan: &Arc<NativeGraphPlan>,
        bindings: NativeGraphBindings,
        timing: GraphTiming,
    ) -> Result<Self, CallError> {
        if plan.has_timelines() {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::NativeGraphTimelineMeasured,
            ));
        }
        Ok(Self {
            plan: plan.clone(),
            bindings,
            slot: RefCell::new(plan.new_slot().map_err(super::tensor_error)?),
            timing,
        })
    }

    /// The outputs of `runs` runs, allocated before any is timed.
    fn outputs(&self, runs: usize) -> Result<Vec<NativeGraphOutputs>, CallError> {
        (0..runs)
            .map(|_| self.plan.new_outputs().map_err(super::tensor_error))
            .collect()
    }

    /// One run attached to the point's slot and queued on `sequence`.
    fn queue(
        &self,
        slot: &mut NativeGraphSlot,
        outputs: NativeGraphOutputs,
        sequence: &mut NativeGraphSequence,
    ) -> Result<(), CallError> {
        slot.attach(self.bindings.clone(), outputs)?
            .queue(sequence)
            .map(drop)
    }

    /// One run attached to the point's slot and submitted alone.
    fn submit_run(
        &self,
        slot: &mut NativeGraphSlot,
        outputs: NativeGraphOutputs,
    ) -> Result<NativeGraphCompletion, CallError> {
        slot.attach(self.bindings.clone(), outputs)?
            .submit()
            .map(|(_, completion)| completion)
    }

    /// The host seconds of `passes` runs: each waited for before the next
    /// is submitted, or, `queued`, all submitted before the first is waited
    /// for.
    fn host_seconds(&self, passes: usize, queued: bool) -> Result<f64, CallError> {
        let mut slot = self.slot.borrow_mut();
        let outputs = self.outputs(passes)?;
        let began = Instant::now();
        if queued {
            let completions = outputs
                .into_iter()
                .map(|outputs| self.submit_run(&mut slot, outputs))
                .collect::<Result<Vec<_>, _>>()?;
            for completion in completions {
                completion.wait()?;
            }
        } else {
            for outputs in outputs {
                self.submit_run(&mut slot, outputs)?.wait()?;
            }
        }
        Ok(began.elapsed().as_secs_f64())
    }

    /// `passes` runs queued as one submission, timed by its trace.
    fn device_submission(&self, passes: usize) -> Result<NativeSubmission, CallError> {
        let mut slot = self.slot.borrow_mut();
        let mut sequence = NativeGraphSequence::new(self.plan.device());
        for outputs in self.outputs(passes)? {
            self.queue(&mut slot, outputs, &mut sequence)?;
        }
        Ok(sequence.submit()?.into_submission())
    }
}

impl Pending<CallError> for GraphPending {
    fn seconds(&self) -> Result<f64, CallError> {
        match &self.0 {
            Sampled::Device(submission) => submission.device_seconds(),
            Sampled::Host(seconds) => Ok(*seconds),
        }
    }

    fn wait(&self) -> Result<(), CallError> {
        match &self.0 {
            Sampled::Device(submission) => submission.wait(),
            Sampled::Host(_) => Ok(()),
        }
    }
}

impl Gauge<CallError> for GraphPoint {
    fn submit(&self, passes: usize) -> Result<Box<dyn Pending<CallError>>, CallError> {
        Ok(Box::new(Work::submit(self, passes)?))
    }
}

impl Work for GraphPoint {
    type Error = CallError;
    type Pending = GraphPending;
    type Clock = PhysicalIdentity;

    /// `passes` runs: queued as one submission for device time; for host
    /// time, run as [`GraphTiming`] says and complete before this returns.
    fn submit(&self, passes: usize) -> Result<GraphPending, CallError> {
        Ok(GraphPending(match self.timing {
            GraphTiming::Device => Sampled::Device(self.device_submission(passes)?),
            GraphTiming::Waited => Sampled::Host(self.host_seconds(passes, false)?),
            GraphTiming::Queued => Sampled::Host(self.host_seconds(passes, true)?),
        }))
    }

    fn calls_per_pass(&self) -> usize {
        1
    }

    fn clock(&self) -> Option<PhysicalIdentity> {
        clock(self.plan.device())
    }

    /// The same runs on a slot of their own, timed on the device: what
    /// raises the clock is the device's work, whatever a sample reads.
    fn gauge(&self) -> Result<Box<dyn Gauge<CallError>>, CallError> {
        Ok(Box::new(Self::new(
            &self.plan,
            self.bindings.clone(),
            GraphTiming::Device,
        )?))
    }
}

/// A kernel's calls or a graph's runs: either point, so one request can
/// measure both.
pub enum Point {
    /// A formed kernel's calls.
    Kernel(KernelPoint),
    /// A sealed graph's runs.
    Graph(GraphPoint),
}

/// A submitted sample of a [`Point`].
pub struct PointPending(Submitted);

enum Submitted {
    Kernel(NativeSubmission),
    Graph(GraphPending),
}

impl Pending<CallError> for PointPending {
    fn seconds(&self) -> Result<f64, CallError> {
        match &self.0 {
            Submitted::Kernel(pending) => pending.seconds(),
            Submitted::Graph(pending) => pending.seconds(),
        }
    }

    fn wait(&self) -> Result<(), CallError> {
        match &self.0 {
            Submitted::Kernel(pending) => Pending::wait(pending),
            Submitted::Graph(pending) => pending.wait(),
        }
    }
}

impl Work for Point {
    type Error = CallError;
    type Pending = PointPending;
    type Clock = PhysicalIdentity;

    fn submit(&self, passes: usize) -> Result<PointPending, CallError> {
        Ok(PointPending(match self {
            Self::Kernel(point) => Submitted::Kernel(Work::submit(point, passes)?),
            Self::Graph(point) => Submitted::Graph(Work::submit(point, passes)?),
        }))
    }

    fn calls_per_pass(&self) -> usize {
        match self {
            Self::Kernel(point) => point.calls_per_pass(),
            Self::Graph(point) => point.calls_per_pass(),
        }
    }

    fn clock(&self) -> Option<PhysicalIdentity> {
        match self {
            Self::Kernel(point) => point.clock(),
            Self::Graph(point) => point.clock(),
        }
    }

    fn gauge(&self) -> Result<Box<dyn Gauge<CallError>>, CallError> {
        match self {
            Self::Kernel(point) => point.gauge(),
            Self::Graph(point) => point.gauge(),
        }
    }
}
