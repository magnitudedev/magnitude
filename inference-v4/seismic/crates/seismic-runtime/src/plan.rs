//! Reusable compiled compositions with retained, checked runtime bindings.
use crate::{Buffer, Candidate, Device, ExecutionObservation, Kernel};
use seismic_lang::{
    lower::{Options, lower_specialized},
    plan::{Plan, ScalarSource},
    program::Program,
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    rc::Rc,
};
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
    enclosing: Option<Enclosing>,
    kernels: Vec<Rc<RefCell<Kernel>>>,
    steps: Vec<Step>,
}
impl CompiledPlan {
    pub fn compile_diagnostic(
        device: &Device,
        program: &Program,
        plan: &Plan,
        lowering: &Options,
        candidate: Candidate,
    ) -> Result<Self, String> {
        PlanCompiler::diagnostic(device, program, lowering.clone(), candidate).compile(plan)
    }
    pub fn step_count(&self) -> usize {
        self.enclosing.as_ref().map_or(self.steps.len(), |_| 1)
    }
    pub fn kernel_count(&self) -> usize {
        self.enclosing
            .as_ref()
            .map_or(self.kernels.len(), |e| e.kernels.borrow().len())
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
        if let Some(enclosing) = &self.enclosing {
            return enclosing.prepare(bindings);
        }
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

/// External hardware facts and explicit search budgets for the existing tuner.
/// The backend's form defines admissible mechanisms, not handpicked kernels.
#[derive(Clone)]
pub struct Settings {
    pub hardware: crate::tuner::Hardware,
    pub form: crate::tuner::Form,
    pub derivation_limits: seismic_accounting::workload::DerivationLimits,
    pub search: seismic_accounting::selection::Budget,
}
struct Enclosing {
    device: Device,
    program: Rc<Program>,
    entry: String,
    shapes: HashMap<String, i64>,
    elements: HashMap<String, seismic_lang::types::Elem>,
    options: Options,
    settings: Settings,
    buffers: Vec<seismic_realization::BufferSpec>,
    scalars: Vec<seismic_lang::abi::ScalarParameter>,
    pending: RefCell<
        Vec<(
            seismic_accounting::workload::ScalarWorkload,
            crate::tuner::Progress,
        )>,
    >,
    kernels: RefCell<
        Vec<(
            seismic_accounting::workload::ScalarWorkload,
            Rc<RefCell<Kernel>>,
        )>,
    >,
}
impl Enclosing {
    fn prepare(&self, bindings: &dyn Bindings) -> Result<Submission, String> {
        let buffers = self
            .buffers
            .iter()
            .map(|s| {
                let b = bindings
                    .buffer(&s.parameter, &s.plane)
                    .ok_or_else(|| format!("unbound tensor {}.{}", s.parameter, s.plane))?;
                b.view(0..s.bytes)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let scalars = self
            .scalars
            .iter()
            .map(|s| {
                bindings
                    .scalar(&s.name)
                    .ok_or_else(|| format!("unbound scalar {}", s.name))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let workload = crate::tuner::workload(&self.entry, &buffers, &self.scalars, &scalars)?;
        let existing = self
            .kernels
            .borrow()
            .iter()
            .find(|(w, _)| w == &workload)
            .map(|(_, k)| k.clone());
        let kernel = match existing {
            Some(k) => k,
            None => {
                let facts = self.device.facts();
                let request = crate::tuner::Request {
                    input: crate::tuner::Input::Portable {
                        program: &self.program,
                        entry: &self.entry,
                        shapes: &self.shapes,
                        elements: &self.elements,
                        options: &self.options,
                    },
                    device: &facts,
                    form: self.settings.form.clone(),
                    hardware: &self.settings.hardware,
                    workload: &workload,
                    derivation_limits: self.settings.derivation_limits,
                };
                let previous = {
                    let mut pending = self.pending.borrow_mut();
                    pending
                        .iter()
                        .position(|(w, _)| w == &workload)
                        .map(|at| pending.swap_remove(at).1)
                };
                let outcome = match previous {
                    Some(progress) => {
                        crate::tuner::resume(&request, progress, self.settings.search)?
                    }
                    None => crate::tuner::tune(&request, self.settings.search)?,
                };
                let selected = match outcome {
                    crate::tuner::Outcome::Optimal(selected) => selected,
                    crate::tuner::Outcome::Incomplete(progress) => {
                        let unresolved = progress.unresolved();
                        let exhausted = progress.exhausted_derivations().collect::<Vec<_>>();
                        let message = format!(
                            "composition tuning incomplete: lower bound {}, feasible upper {:?}; {} choice regions, {} deferred model derivations, {} unfinished schedules, {} models with unavailable resource mappings; exhausted limits: {exhausted:?}",
                            progress.lower_bound()?,
                            progress.feasible_upper(),
                            unresolved.choice_regions,
                            unresolved.derivations,
                            unresolved.schedules,
                            unresolved.unmapped_models
                        );
                        self.pending.borrow_mut().push((workload, progress));
                        return Err(message);
                    }
                    crate::tuner::Outcome::Infeasible => {
                        return Err(
                            "composition has no legal execution in the selected form".into()
                        );
                    }
                };
                let compiled = self.device.compile_tuned(selected)?;
                if compiled.buffers() != self.buffers || compiled.scalars() != self.scalars {
                    return Err("selected execution changed the enclosing entry ABI".into());
                }
                let k = Rc::new(RefCell::new(compiled));
                self.kernels.borrow_mut().push((workload, k.clone()));
                k
            }
        };
        Ok(Submission {
            invocations: vec![BoundInvocation {
                entry: self.entry.clone(),
                kernel,
                buffers,
                scalars,
            }],
        })
    }
}

#[derive(Clone)]
pub struct Diagnostic {
    pub lowering: Options,
    pub candidate: Candidate,
}
type KernelKey = (
    String,
    Vec<(String, i64)>,
    Vec<(String, String)>,
    Vec<String>,
);
enum Configuration {
    Tune(Settings),
    Diagnostic(Diagnostic),
}
/// Reuse is scoped to the same program/device/settings. Both explicit assignments
/// and selection consume the enclosing source and produce its invocation ABI.
pub struct PlanCompiler<'a> {
    device: &'a Device,
    program: &'a Program,
    configuration: Configuration,
    kernels: BTreeMap<KernelKey, Rc<RefCell<Kernel>>>,
}
impl<'a> PlanCompiler<'a> {
    pub fn diagnostic(
        device: &'a Device,
        program: &'a Program,
        lowering: Options,
        candidate: Candidate,
    ) -> Self {
        Self {
            device,
            program,
            configuration: Configuration::Diagnostic(Diagnostic {
                lowering,
                candidate,
            }),
            kernels: BTreeMap::new(),
        }
    }
    pub fn new(device: &'a Device, program: &'a Program, settings: Settings) -> Self {
        Self {
            device,
            program,
            configuration: Configuration::Tune(settings),
            kernels: BTreeMap::new(),
        }
    }
    pub fn compile_entry(
        &mut self,
        entry: &str,
        shapes: &HashMap<String, i64>,
        elements: &HashMap<String, seismic_lang::types::Elem>,
        ownership: &seismic_lang::composition::Ownership,
    ) -> Result<CompiledPlan, String> {
        let settings = match &self.configuration {
            Configuration::Tune(settings) => settings,
            Configuration::Diagnostic(assignment) => {
                let mut shape_key = shapes
                    .iter()
                    .map(|(n, v)| (n.clone(), *v))
                    .collect::<Vec<_>>();
                shape_key.sort();
                let mut element_key = elements
                    .iter()
                    .map(|(n, v)| (n.clone(), v.to_string()))
                    .collect::<Vec<_>>();
                element_key.sort();
                let mut options = assignment.lowering.clone();
                options
                    .ownership
                    .intermediates
                    .extend(ownership.intermediates.iter().cloned());
                let key = (
                    entry.to_string(),
                    shape_key,
                    element_key,
                    options.ownership.intermediates.iter().cloned().collect(),
                );
                let kernel = if let Some(k) = self.kernels.get(&key) {
                    k.clone()
                } else {
                    let lowered = lower_specialized(
                        self.program,
                        entry,
                        self.device.backend(),
                        shapes,
                        elements,
                        &options,
                    )?;
                    let kernel = Rc::new(RefCell::new(
                        self.device
                            .compile(&lowered, assignment.candidate.clone())?,
                    ));
                    self.kernels.insert(key, kernel.clone());
                    kernel
                };
                let k = kernel.borrow();
                let slots = k
                    .buffers()
                    .iter()
                    .map(|b| Slot {
                        root: b.parameter.clone(),
                        plane: b.plane.clone(),
                        offset: 0,
                        bytes: b.bytes,
                    })
                    .collect();
                let scalars = k
                    .scalars()
                    .iter()
                    .map(|s| ScalarSource::Param(s.name.clone()))
                    .collect();
                drop(k);
                return Ok(CompiledPlan {
                    enclosing: None,
                    kernels: vec![kernel],
                    steps: vec![Step {
                        entry: entry.into(),
                        kernel: 0,
                        slots,
                        scalars,
                    }],
                });
            }
        };
        let source = self
            .program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or("unknown composition entry")?;
        seismic_lang::program::validate_element_bindings(source, elements)?;
        for name in &source.shape_params {
            if !shapes.contains_key(name) {
                return Err(format!("unbound entry shape {name}"));
            }
        }
        let env = shapes
            .iter()
            .map(|(n, v)| (n.clone(), seismic_lang::sym::Sym::constant(*v)))
            .collect();
        let params = source
            .params
            .iter()
            .map(|(n, t)| {
                (
                    n.clone(),
                    seismic_lang::lower::subst_elem_ty(
                        &seismic_lang::lower::subst_ty(t, &env),
                        elements,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let indices = source
            .index_params
            .iter()
            .map(|(n, b)| {
                (
                    n.clone(),
                    seismic_lang::lower::subst_sym(b, &env, &HashMap::new()),
                )
            })
            .collect::<Vec<_>>();
        let (buffers, scalars) = seismic_realization::storage::parameter_types(&params, &indices)?;
        let enclosing = Enclosing {
            device: self.device.clone(),
            program: Rc::new(self.program.clone()),
            entry: entry.into(),
            shapes: shapes.clone(),
            elements: elements.clone(),
            options: Options {
                piece: None,
                ownership: ownership.clone(),
            },
            settings: settings.clone(),
            buffers,
            scalars,
            pending: RefCell::new(Vec::new()),
            kernels: RefCell::new(Vec::new()),
        };
        Ok(CompiledPlan {
            enclosing: Some(enclosing),
            kernels: Vec::new(),
            steps: Vec::new(),
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
        self.compile_entry(
            &plan.function,
            &plan.shapes,
            &plan.elements,
            &plan.ownership,
        )
    }
}
