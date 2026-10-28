//! The CPU native ABI. CPU native implementations are Rust compiled into the
//! binary; `seismic-build` wraps each entry's source in a typed context over
//! [`CpuInvocation`] and registers, per launch, one function per
//! instruction-set tier and dense element binding. Tuning parameters are
//! runtime values; weight operands reach kernels as components resolved for
//! the device's tier and the operand's representation.

use std::sync::Arc;

use seismic_compiler::errors::ExecutionError;
use seismic_cpu::{NativeStep, NativeSteps};
use seismic_native_cpu::{Tier, WeightOperand};

use super::trace::host_seconds;
use super::timeline::{TimelineKind, TimelineNode};
use super::{Dispatch, DispatchList, NativeRoute};
use crate::backends::CpuOpened;
use crate::driver::typed_buffer;

/// One launch function: the invocation, the work item's group coordinate,
/// and the item's private `shared_bytes` buffer.
///
/// # Safety
/// The function's tier is present on the host: the runtime calls it only
/// when it was selected for a tier at or below the device's detected tier.
pub type CpuKernelFn = unsafe fn(&CpuInvocation<'_>, [u64; 3], &mut [u8]);

/// One compiled form of a launch: its tier and the dense element binding
/// (registry names, in [`CpuNativeKernels::elements`] order) it is
/// monomorphized for.
pub struct CpuVariant {
    pub tier: Tier,
    pub elements: &'static [&'static str],
    pub function: CpuKernelFn,
}

/// The compiled forms of one launch.
pub struct CpuLaunchVariants {
    pub kernel: &'static str,
    pub variants: &'static [CpuVariant],
}

/// Every launch of one CPU native implementation, in declaration order.
pub struct CpuNativeKernels {
    /// Digest of the authored asset, the CPU library files of its source
    /// root, and the Seismic CPU library version.
    pub digest: &'static str,
    /// The entry's dense element parameters, which variants are
    /// monomorphized over.
    pub elements: &'static [&'static str],
    pub launches: &'static [CpuLaunchVariants],
}

impl CpuNativeKernels {
    /// The function of launch `launch` for `tier` and the dense element
    /// binding `elements`.
    pub(crate) fn function(
        &self,
        launch: usize,
        tier: Tier,
        elements: &[&str],
    ) -> Option<CpuKernelFn> {
        self.launches
            .get(launch)?
            .variants
            .iter()
            .find(|variant| variant.tier == tier && variant.elements == elements)
            .map(|variant| variant.function)
    }
}

/// The ABI of one CPU native call, shared by all its work items. Buffers are
/// in ABI order (tensor parameters, tensor results, scratch). Words have the
/// same layout as the GPU argument words.
pub struct CpuInvocation<'a> {
    pub(crate) buffers: &'a [*mut u8],
    /// Registry storage name of each buffer (`registry::storage_of`); scratch
    /// is `"bytes"`.
    pub(crate) representations: &'a [&'static str],
    /// The components of each buffer's representation on the call's tier,
    /// with its operand recipe, for buffers of a weight representation.
    pub(crate) weights: &'a [Option<WeightOperand>],
    pub(crate) words: &'a [u64],
    /// Values of the declared tuning parameters, in declaration order.
    pub(crate) params: &'a [u64],
    pub(crate) scalar_results: *mut u64,
    pub(crate) groups: [u64; 3],
    pub(crate) threads: [u64; 3],
}

// Work items run concurrently and write disjoint regions through the raw
// pointers; the pointed-to storage outlives the synchronous launch.
unsafe impl Sync for CpuInvocation<'_> {}

impl CpuInvocation<'_> {
    /// Base address of buffer `index` (already offset to the tensor view).
    pub fn buffer(&self, index: usize) -> *mut u8 {
        self.buffers[index]
    }
    pub fn representation(&self, index: usize) -> &'static str {
        self.representations[index]
    }
    /// The components of buffer `index`, a weight operand.
    pub fn weights(&self, index: usize) -> WeightOperand {
        self.weights[index].unwrap_or_else(|| {
            panic!(
                "buffer {index} (`{}`) is not bound to a weight representation",
                self.representations[index]
            )
        })
    }
    pub fn word(&self, index: usize) -> u64 {
        self.words[index]
    }
    /// Declared tuning parameter `index`.
    pub fn param(&self, index: usize) -> u64 {
        self.params[index]
    }
    /// Slot `index` of the scalar results. Only one work item may write a
    /// given slot.
    pub fn scalar_result(&self, index: usize) -> *mut u64 {
        // SAFETY: the runtime sizes the scalar storage to every slot of the
        // checked schema, and generated contexts index only those slots.
        unsafe { self.scalar_results.add(index) }
    }
    /// Groups of the current launch on each axis.
    pub fn groups(&self) -> [u64; 3] {
        self.groups
    }
    /// Declared participants of one group on each axis.
    pub fn threads(&self) -> [u64; 3] {
        self.threads
    }
}

/// The CPU route of one prepared implementation.
#[derive(Clone)]
pub(crate) struct CpuRoute {
    pub(crate) opened: Arc<CpuOpened>,
    pub(crate) tier: Tier,
    /// One function per launch, for the route's tier and element binding.
    pub(crate) launches: Vec<CpuKernelFn>,
    /// Participants per launch; `0` is every participant.
    pub(crate) workers: Vec<usize>,
    /// Values of the declared tuning parameters.
    pub(crate) params: Vec<u64>,
}

/// One active launch of a submission as the pool runs it.
struct Step<'a> {
    function: CpuKernelFn,
    invocation: CpuInvocation<'a>,
    items: u64,
    shared_bytes: u64,
    workers: usize,
}

/// The active launches of one pass over a submission's calls, repeated.
struct Steps<'a> {
    steps: Vec<Step<'a>>,
    repetitions: usize,
}

impl NativeSteps for Steps<'_> {
    fn count(&self) -> usize {
        self.steps.len() * self.repetitions
    }

    fn step(&self, index: usize) -> NativeStep {
        let step = &self.steps[index % self.steps.len()];
        NativeStep {
            items: step.items,
            shared_bytes: step.shared_bytes,
            workers: step.workers,
        }
    }

    fn run(&self, index: usize, item: u64, shared: &mut [u8]) {
        let step = &self.steps[index % self.steps.len()];
        let [x, y, _] = step.invocation.groups;
        // SAFETY: the route selected the function for a tier the device has
        // (`CpuRoute::tier`, validated at preparation).
        unsafe {
            (step.function)(
                &step.invocation,
                [item % x, (item / x) % y, item / (x * y)],
                shared,
            )
        };
    }
}

/// Runs `repetitions` passes over `list` as one pool job: one wake of the
/// pool, a barrier between consecutive launches. Returns the outcome and the
/// interval the job ran, on the [`host_seconds`] clock.
pub(super) fn run(
    opened: &Arc<CpuOpened>,
    list: &impl DispatchList,
    repetitions: usize,
) -> (Result<(), ExecutionError>, (f64, f64)) {
    let executor = opened.executor();
    let mut pointers = Vec::new();
    let mut storages = Vec::new();
    let mut weights = Vec::new();
    let calls = calls(list, &mut pointers, &mut storages, &mut weights);
    let tables = Tables {
        pointers: &pointers,
        storages: &storages,
        weights: &weights,
    };
    let now = host_seconds();
    if !list.has_timelines() {
        return match steps(&calls, tables, executor.workers()) {
            Ok(steps) if steps.is_empty() => (Ok(()), (now, now)),
            Ok(steps) => executor.run_native(&Steps { steps, repetitions }, host_seconds),
            Err(error) => (Err(error), (now, now)),
        };
    }
    // The calls between timeline boundaries run as pool jobs of their own;
    // at a boundary this thread waits for, or signals, its timelines.
    let mut end = now;
    for _ in 0..repetitions {
        let mut first = 0;
        for boundary in 0..=calls.len() {
            let operations = list.timelines(boundary);
            if operations.is_empty() && boundary < calls.len() {
                continue;
            }
            match steps(&calls[first..boundary], tables, executor.workers()) {
                Ok(steps) if steps.is_empty() => {}
                Ok(steps) => {
                    let (outcome, (_, finished)) = executor.run_native(
                        &Steps {
                            steps,
                            repetitions: 1,
                        },
                        host_seconds,
                    );
                    end = finished;
                    if outcome.is_err() {
                        return (outcome, (now, end));
                    }
                }
                Err(error) => return (Err(error), (now, end)),
            }
            first = boundary;
            for operation in operations {
                let TimelineKind::Cpu(timeline) = operation.timeline.kind() else {
                    unreachable!("a run binds timelines of its own device");
                };
                match operation.node {
                    TimelineNode::Wait => timeline.wait_unbounded(operation.value),
                    TimelineNode::Signal => timeline.signal(operation.value),
                }
            }
        }
    }
    (Ok(()), (now, end))
}

/// One call of a submission: its dispatch, the range of its buffers in the
/// [`Tables`], and its scalar-result storage.
type Call<'l> = (Dispatch<'l>, std::ops::Range<usize>, *mut u64);

/// The per-buffer tables of a submission's calls.
#[derive(Clone, Copy)]
struct Tables<'a> {
    pointers: &'a [*mut u8],
    /// Each buffer's storage name (`registry::storage_of`): what the CPU
    /// components and conversions read, the same for every operand recipe.
    storages: &'a [&'static str],
    weights: &'a [Option<WeightOperand>],
}

/// Every call of `list`, appending the calls' buffer addresses to
/// `pointers`, their storage names to `storages` and their resolved weight
/// components to `weights`.
fn calls<'l>(
    list: &'l impl DispatchList,
    pointers: &mut Vec<*mut u8>,
    storages: &mut Vec<&'static str>,
    weights: &mut Vec<Option<WeightOperand>>,
) -> Vec<Call<'l>> {
    type Cpu = seismic_cpu::Cpu;
    type Executor = seismic_cpu::Executor;
    let mut buffers = Vec::new();
    let mut calls = Vec::with_capacity(list.count());
    for index in 0..list.count() {
        buffers.clear();
        let dispatch = list.dispatch(index, &mut buffers);
        let NativeRoute::Cpu(route) = &dispatch.kernel.route else {
            unreachable!("one device has one native route");
        };
        let first = pointers.len();
        pointers.extend(buffers.iter().map(|(allocation, offset)| {
            let base = typed_buffer::<Cpu, Executor>(allocation).data_pointer();
            // SAFETY: tensor views and scratch placements lie inside their
            // allocations (checked when the views were formed).
            unsafe { base.add(*offset as usize) }
        }));
        for representation in dispatch.representations {
            // Scratch (`"bytes"`) is no registered element.
            let element = seismic_lang::registry::representation(representation);
            let storage = element.map_or(*representation, |element| {
                seismic_lang::registry::representation_info(
                    seismic_lang::registry::storage_of(element),
                )
                .name
            });
            storages.push(storage);
            weights.push(element.and_then(|element| WeightOperand::of(route.tier, element)));
        }
        let scalars = typed_buffer::<Cpu, Executor>(&dispatch.kernel.scalars)
            .data_pointer()
            .cast::<u64>();
        calls.push((dispatch, first..pointers.len(), scalars));
    }
    calls
}

/// The active launches of `calls` on a pool of `participants`.
fn steps<'a>(
    calls: &'a [Call<'_>],
    tables: Tables<'a>,
    participants: usize,
) -> Result<Vec<Step<'a>>, ExecutionError> {
    let mut steps = Vec::new();
    for (dispatch, range, scalars) in calls {
        let NativeRoute::Cpu(route) = &dispatch.kernel.route else {
            unreachable!("one device has one native route");
        };
        for ((function, workers), launch) in route
            .launches
            .iter()
            .zip(&route.workers)
            .zip(dispatch.launches)
        {
            let Some(launch) = launch else { continue };
            let items = launch.groups[0]
                .checked_mul(launch.groups[1])
                .and_then(|items| items.checked_mul(launch.groups[2]))
                .ok_or_else(|| {
                    ExecutionError::SubmissionFailed("native CPU grid overflows".into())
                })?;
            steps.push(Step {
                function: *function,
                invocation: CpuInvocation {
                    buffers: &tables.pointers[range.clone()],
                    representations: &tables.storages[range.clone()],
                    weights: &tables.weights[range.clone()],
                    words: dispatch.words,
                    params: &route.params,
                    scalar_results: *scalars,
                    groups: launch.groups,
                    threads: launch.threads,
                },
                items,
                shared_bytes: launch.shared_bytes,
                workers: if *workers == 0 {
                    participants
                } else {
                    *workers
                },
            });
        }
    }
    Ok(steps)
}
