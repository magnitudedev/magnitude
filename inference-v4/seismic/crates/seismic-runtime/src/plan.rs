//! Reusable compiled compositions with retained, checked runtime bindings.
use crate::{Buffer, Candidate, Device, ExecutionObservation, Kernel};
use seismic_lang::{
    lower::{lower_specialized, Options},
    plan::{Plan, ScalarSource},
    program::Program,
};
use seismic_realization::storage::plane_byte_offset;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
pub trait Bindings {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer>;
    fn scalar(&self, name: &str) -> Option<f64>;
}
struct Slot {
    root: String,
    plane: String,
    offset: usize,
    bytes: usize,
}
struct Step {
    entry: String,
    kernel: usize,
    slots: Vec<Slot>,
    scalars: Vec<ScalarSource>,
}
#[derive(Clone, Debug)]
pub struct StepObservation {
    pub entry: String,
    pub execution: ExecutionObservation,
}
pub struct CompiledPlan {
    kernels: Vec<Rc<RefCell<Kernel>>>,
    steps: Vec<Step>,
}
impl CompiledPlan {
    pub fn compile(
        device: &Device,
        program: &Program,
        plan: &Plan,
        lowering: &Options,
        candidate: Candidate,
    ) -> Result<Self, String> {
        PlanCompiler::new(device, program, lowering.clone(), candidate).compile(plan)
    }
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }
    /// Resolve all named inputs and checked subviews before the first kernel.
    /// This baseline completes each kernel synchronously; batched native submission
    /// is a separate realization and is not claimed by this execution path.
    pub fn execute(&mut self, bindings: &dyn Bindings) -> Result<(), String> {
        self.invoke(bindings, false).map(|_| ())
    }
    pub fn execute_observed(
        &mut self,
        bindings: &dyn Bindings,
    ) -> Result<Vec<StepObservation>, String> {
        self.invoke(bindings, true)
    }
    pub fn prepare(&self, bindings: &dyn Bindings) -> Result<Submission, String> {
        let mut prepared = Vec::new();
        for step in &self.steps {
            let buffers = step
                .slots
                .iter()
                .map(|slot| {
                    let buffer = bindings
                        .buffer(&slot.root, &slot.plane)
                        .ok_or_else(|| format!("unbound tensor {}.{}", slot.root, slot.plane))?;
                    let end = slot
                        .offset
                        .checked_add(slot.bytes)
                        .ok_or("plan byte range overflow")?;
                    buffer.view(slot.offset..end)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let scalars = step
                .scalars
                .iter()
                .map(|source| match source {
                    ScalarSource::Literal(value) => Ok(*value),
                    ScalarSource::Param(name) => bindings
                        .scalar(name)
                        .ok_or_else(|| format!("unbound scalar {name}")),
                })
                .collect::<Result<Vec<_>, String>>()?;
            seismic_realization::encode_scalars(
                self.kernels[step.kernel].borrow().scalars(),
                &scalars,
            )?;
            prepared.push(BoundInvocation {
                entry: step.entry.clone(),
                kernel: self.kernels[step.kernel].clone(),
                buffers,
                scalars,
            });
        }
        Ok(Submission {
            invocations: prepared,
        })
    }
    fn invoke(
        &mut self,
        bindings: &dyn Bindings,
        observed: bool,
    ) -> Result<Vec<StepObservation>, String> {
        let mut submission = self.prepare(bindings)?;
        if observed {
            submission.execute_steps_observed()
        } else {
            submission.execute_sequential().map(|_| Vec::new())
        }
    }
}

struct BoundInvocation {
    entry: String,
    kernel: Rc<RefCell<Kernel>>,
    buffers: Vec<Buffer>,
    scalars: Vec<f64>,
}
/// Bound code and allocation pins. Preparation does not execute numerical work.
/// Appending preserves source order and extends resource lifetime through completion.
#[derive(Default)]
pub struct Submission {
    invocations: Vec<BoundInvocation>,
}
impl Submission {
    pub fn append(&mut self, mut other: Self) {
        self.invocations.append(&mut other.invocations);
    }
    pub fn len(&self) -> usize {
        self.invocations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.invocations.is_empty()
    }
    pub fn execute_sequential(&mut self) -> Result<(), String> {
        for invocation in &self.invocations {
            invocation
                .kernel
                .try_borrow_mut()
                .map_err(|_| "shared kernel is already executing")?
                .execute(&invocation.buffers, &invocation.scalars)
                .map_err(|e| format!("{}: {e}", invocation.entry))?;
        }
        Ok(())
    }
    pub fn execute_steps_observed(&mut self) -> Result<Vec<StepObservation>, String> {
        self.invocations
            .iter()
            .map(|invocation| {
                let execution = invocation
                    .kernel
                    .try_borrow_mut()
                    .map_err(|_| "shared kernel is already executing")?
                    .execute_observed(&invocation.buffers, &invocation.scalars)
                    .map_err(|e| format!("{}: {e}", invocation.entry))?;
                Ok(StepObservation {
                    entry: invocation.entry.clone(),
                    execution,
                })
            })
            .collect()
    }
    /// Metal encodes one command buffer. CPU and CUDA currently retain serial
    /// native invocations; CUDA reports event sums, not a continuous batch interval.
    /// No asynchronous work escapes this method, including on failure.
    pub fn execute_batched(&mut self) -> Result<ExecutionObservation, String> {
        let start = std::time::Instant::now();
        let mut backend = None;
        for invocation in &self.invocations {
            let kernel = invocation
                .kernel
                .try_borrow()
                .map_err(|_| "shared kernel is already executing")?;
            let kind = std::mem::discriminant(&kernel.executable);
            kernel.validate_tuning(&invocation.buffers, &invocation.scalars)?;
            if backend.is_some_and(|previous| previous != kind) {
                return Err("mixed backends in submission".into());
            }
            backend = Some(kind);
        }
        #[cfg(target_os = "macos")]
        {
            let kernels = self
                .invocations
                .iter()
                .map(|i| {
                    i.kernel
                        .try_borrow()
                        .map_err(|_| "shared kernel is already executing".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(first) = kernels.first() {
                if let crate::Executable::Metal { device, .. } = &first.executable {
                    let invocations = kernels
                        .iter()
                        .zip(&self.invocations)
                        .map(|(kernel, bound)| {
                            let crate::Executable::Metal { pipeline, .. } = &kernel.executable
                            else {
                                return Err("mixed backends in Metal submission".to_string());
                            };
                            let buffers = bound
                                .buffers
                                .iter()
                                .map(|b| match &b.0 {
                                    crate::Storage::Metal(b) => Ok(b),
                                    _ => Err("non-Metal buffer in Metal submission".to_string()),
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(seismic_metal::runtime::Invocation {
                                pipeline,
                                buffers,
                                scalars: pipeline.emitted.encode_scalars(&bound.scalars)?,
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let seconds = device.run_many(&invocations, 1)?;
                    return Ok(ExecutionObservation {
                        host_seconds: start.elapsed().as_secs_f64(),
                        device_seconds: Some(seconds),
                        device_scope: Some(crate::DeviceTimingScope::CommandBuffer),
                    });
                }
            }
        }
        let mut device_seconds = None;
        for invocation in &self.invocations {
            let observation = invocation
                .kernel
                .try_borrow_mut()
                .map_err(|_| "shared kernel is already executing")?
                .execute_observed(&invocation.buffers, &invocation.scalars)?;
            if let Some(seconds) = observation.device_seconds {
                *device_seconds.get_or_insert(0.) += seconds;
            }
        }
        Ok(ExecutionObservation {
            host_seconds: start.elapsed().as_secs_f64(),
            device_seconds,
            device_scope: device_seconds.map(|_| crate::DeviceTimingScope::KernelEventSum),
        })
    }
}

#[derive(Clone)]
pub struct KernelChoice {
    pub lowering: Options,
    pub candidate: Candidate,
}
#[derive(Clone)]
pub struct CompilationChoices {
    pub default: KernelChoice,
    pub kernels: BTreeMap<String, KernelChoice>,
}
type KernelKey = (String, Vec<(String, i64)>, Vec<(String, String)>);
/// Compilation reuse scoped to one immutable program, device and explicit
/// realization. Native code and scratch remain alive through retained plans.
/// Shared kernels execute serially; asynchronous parallelism needs separate
/// invocation scratch ownership before relaxing that rule.
pub struct PlanCompiler<'a> {
    device: &'a Device,
    program: &'a Program,
    choices: CompilationChoices,
    kernels: BTreeMap<KernelKey, Rc<RefCell<Kernel>>>,
}
impl<'a> PlanCompiler<'a> {
    pub fn new(
        device: &'a Device,
        program: &'a Program,
        lowering: Options,
        candidate: Candidate,
    ) -> Self {
        Self {
            device,
            program,
            choices: CompilationChoices {
                default: KernelChoice {
                    lowering,
                    candidate,
                },
                kernels: BTreeMap::new(),
            },
            kernels: BTreeMap::new(),
        }
    }
    pub fn with_choices(
        device: &'a Device,
        program: &'a Program,
        choices: CompilationChoices,
    ) -> Result<Self, String> {
        for name in choices.kernels.keys() {
            if !program.functions.iter().any(|f| &f.name == name) {
                return Err(format!("choice names unknown kernel {name}"));
            }
        }
        Ok(Self {
            device,
            program,
            choices,
            kernels: BTreeMap::new(),
        })
    }
    pub fn program(&self) -> &Program {
        self.program
    }
    pub fn device(&self) -> &Device {
        self.device
    }
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }
    pub fn compile(&mut self, plan: &Plan) -> Result<CompiledPlan, String> {
        let mut kernels: Vec<Rc<RefCell<Kernel>>> = Vec::new();
        let mut keys = BTreeMap::new();
        let mut steps = Vec::new();
        for step in &plan.steps {
            let mut shapes = step
                .shapes
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect::<Vec<_>>();
            shapes.sort();
            let mut elements = step
                .elements
                .iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect::<Vec<_>>();
            elements.sort();
            let key = (step.kernel.clone(), shapes, elements);
            let index = if let Some(index) = keys.get(&key) {
                *index
            } else {
                let kernel = if let Some(kernel) = self.kernels.get(&key) {
                    kernel.clone()
                } else {
                    let choice = self
                        .choices
                        .kernels
                        .get(&step.kernel)
                        .unwrap_or(&self.choices.default);
                    let lowered = lower_specialized(
                        self.program,
                        &step.kernel,
                        self.device.backend(),
                        &step.shapes,
                        &step.elements,
                        &choice.lowering,
                    ).map_err(|error| format!("lowering {} with shapes {:?}: {error}", step.kernel, key.1))?;
                    let kernel = Rc::new(RefCell::new(
                        self.device.compile(&lowered, choice.candidate.clone()).map_err(|error| format!("compiling {} with shapes {:?}: {error}", step.kernel, key.1))?,
                    ));
                    self.kernels.insert(key.clone(), kernel.clone());
                    kernel
                };
                let index = kernels.len();
                kernels.push(kernel);
                keys.insert(key, index);
                index
            };
            let kernel = kernels[index].borrow();
            let mut slots = Vec::new();
            for slot in kernel.buffers() {
                let binding = step
                    .tensors
                    .iter()
                    .find(|t| t.param == slot.parameter)
                    .ok_or_else(|| {
                        format!(
                            "plan kernel {} has no binding for {}",
                            step.kernel, slot.parameter
                        )
                    })?;
                let offset = plane_byte_offset(&binding.elem, &slot.plane, binding.elem_offset)?;
                slots.push(Slot {
                    root: binding.root.clone(),
                    plane: slot.plane.clone(),
                    offset,
                    bytes: slot.bytes,
                });
            }
            let scalars = kernel
                .scalars()
                .iter()
                .map(|p| {
                    step.scalars
                        .iter()
                        .find(|(name, _)| name == &p.name)
                        .map(|(_, source)| source.clone())
                        .ok_or_else(|| format!("unbound plan scalar {}", p.name))
                })
                .collect::<Result<_, _>>()?;
            steps.push(Step {
                entry: step.kernel.clone(),
                kernel: index,
                slots,
                scalars,
            });
        }
        Ok(CompiledPlan { kernels, steps })
    }
}
