//! Reusable compiled compositions with retained, checked runtime bindings.
use crate::{Buffer, Device, ExecutionObservation, Kernel};
use seismic_lang::{
    lower::Options,
    plan::Plan,
    program::Program,
};
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
};
pub trait Bindings {
    /// Immutable invocation input whose actual bytes affect indexing/control flow.
    fn known_buffer(&self, _root: &str, _plane: &str) -> bool { false }
    /// Admitted varying control values, modeled uniformly before selection and
    /// checked against actual bytes before every native submission.
    fn buffer_domains(&self, _root: &str, _plane: &str) -> Vec<crate::tuner::BufferIntegerDomain> { Vec::new() }
    fn scalar_domain(&self, _name: &str) -> Option<crate::tuner::IntegerRange> { None }
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer>;
    fn scalar(&self, name: &str) -> Option<f64>;
}
#[derive(Clone, Debug)]
pub struct StepObservation {
    pub entry: String,
    pub execution: ExecutionObservation,
}
pub struct CompiledPlan {
    enclosing: Rc<Enclosing>,
}
impl CompiledPlan {
    pub fn supports_integer_domains(&self) -> bool { self.enclosing.settings.form.supports_integer_domains() }
    pub fn shares_compilation(&self, other: &Self) -> bool { Rc::ptr_eq(&self.enclosing, &other.enclosing) }
    pub fn step_count(&self) -> usize { 1 }
    pub fn kernel_count(&self) -> usize { self.enclosing.kernels.borrow().len() }
    pub fn execute(&mut self, bindings: &dyn Bindings) -> Result<(), String> {
        self.prepare(bindings)?.execute_sequential()
    }
    pub fn execute_observed(&mut self, bindings: &dyn Bindings) -> Result<Vec<StepObservation>, String> {
        self.prepare(bindings)?.execute_steps_observed()
    }
    pub fn prepare(&self, bindings: &dyn Bindings) -> Result<Submission, String> {
        self.enclosing.prepare(bindings)
    }
    /// Bind a logical entry's ABI; selection still precedes native compilation.
    pub fn execute_buffers(&mut self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        if buffers.len() != self.enclosing.buffers.len() || scalars.len() != self.enclosing.scalars.len() {
            return Err("entry binding count differs from its logical ABI".into());
        }
        struct Positional<'a> {
            buffers: HashMap<(&'a str, &'a str), &'a Buffer>,
            scalars: HashMap<&'a str, f64>,
        }
        impl Bindings for Positional<'_> {
            fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> { self.buffers.get(&(root, plane)).copied() }
            fn scalar(&self, name: &str) -> Option<f64> { self.scalars.get(name).copied() }
        }
        let bindings = Positional {
            buffers: self.enclosing.buffers.iter().zip(buffers).map(|(s,b)| ((s.parameter.as_str(),s.plane.as_str()),b)).collect(),
            scalars: self.enclosing.scalars.iter().zip(scalars).map(|(s,v)| (s.name.as_str(),*v)).collect(),
        };
        self.enclosing.prepare(&bindings)?.execute_sequential()
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
        // The whole batch validates before encoding. No invocation may mutate
        // another invocation's content-conditioned allocation between checks.
        for bound in &self.invocations {
            let kernel = bound.kernel.try_borrow().map_err(|_| "shared kernel is already executing")?;
            for (slot, binding) in kernel.tuning.workload().buffers.iter().enumerate() {
                let known = kernel.tuning.workload().conditions_allocation(binding.allocation);
                if !known { continue; }
                for other in &self.invocations {
                    let other_kernel = other.kernel.try_borrow().map_err(|_| "shared kernel is already executing")?;
                    for (other_slot, buffer) in other.buffers.iter().enumerate() {
                        if bound.buffers[slot].shares_allocation(buffer) && !other_kernel.conditions.read_only_buffers().contains(&other_slot) {
                            return Err("batch may modify a content-conditioned allocation".into());
                        }
                    }
                }
            }
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
        let mut workload = crate::tuner::workload(&self.entry, &buffers, &self.scalars, &scalars)?;
        let known = self.buffers.iter().enumerate().filter_map(|(slot, b)| bindings.known_buffer(&b.parameter, &b.plane).then_some(slot)).collect::<Vec<_>>();
        crate::tuner::capture_contents(&mut workload, &buffers, &known)?;
        let mut domains = Vec::new();
        for (slot, scalar) in self.scalars.iter().enumerate() {
            if let Some(range) = bindings.scalar_domain(&scalar.name) {
                domains.push(crate::tuner::IntegerDomain {
                    input: crate::tuner::IntegerInput::Scalar { slot },
                    bytes: scalar.dtype.bytes() as u8,
                    signed: scalar.dtype == seismic_lang::types::DType::I32,
                    range,
                });
            }
        }
        for (slot, buffer) in self.buffers.iter().enumerate() {
            let binding = &workload.buffers[slot];
            for domain in bindings.buffer_domains(&buffer.parameter, &buffer.plane) {
                if domain.offset.checked_add(u64::from(domain.bytes)).is_none_or(|end| end > binding.bytes) {
                    return Err("integer input domain exceeds its parameter view".into());
                }
                domains.push(crate::tuner::IntegerDomain {
                    input: crate::tuner::IntegerInput::Allocation {
                        allocation: binding.allocation,
                        offset: binding.offset.checked_add(domain.offset).ok_or("integer input offset overflow")?,
                    },
                    bytes: domain.bytes,
                    signed: domain.signed,
                    range: domain.range,
                });
            }
        }
        crate::tuner::generalize_inputs(&mut workload, &buffers, &self.scalars, domains)?;
        let existing = self
            .kernels
            .borrow()
            .iter()
            .find(|(w, _)| w.covers(&workload))
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
                        let missing = progress.missing_mappings().flat_map(|(_, reasons)| reasons.iter()).collect::<std::collections::BTreeSet<_>>();
                        let unsupported = progress.unsupported_analyses().collect::<Vec<_>>();
                        let message = format!(
                            "composition {} tuning incomplete after {} nodes: lower bound {}, feasible upper {:?}; {} choice regions, {} deferred model derivations, {} unfinished schedules, {} models with unavailable resource mappings; missing mappings: {missing:?}; exhausted limits: {exhausted:?}; unsupported analyses: {unsupported:?}",
                            self.entry,
                            progress.nodes_visited(),
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

/// Compiles logical entries through completed automatic selection only.
pub struct PlanCompiler<'a> {
    device: &'a Device,
    program: Rc<Program>,
    settings: Settings,
    entries: Vec<Rc<Enclosing>>,
}
impl<'a> PlanCompiler<'a> {
    pub fn new(device: &'a Device, program: &'a Program, settings: Settings) -> Self {
        Self { device, program: Rc::new(program.clone()), settings, entries: Vec::new() }
    }
    pub fn compile_entry(
        &mut self,
        entry: &str,
        shapes: &HashMap<String, i64>,
        elements: &HashMap<String, seismic_lang::types::Elem>,
        ownership: &seismic_lang::composition::Ownership,
    ) -> Result<CompiledPlan, String> {
        if let Some(enclosing) = self.entries.iter().find(|e| e.entry == entry && &e.shapes == shapes && &e.elements == elements && &e.options.ownership == ownership) {
            return Ok(CompiledPlan { enclosing: enclosing.clone() });
        }
        let settings = &self.settings;
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
            program: self.program.clone(),
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
        let enclosing = Rc::new(enclosing);
        self.entries.push(enclosing.clone());
        Ok(CompiledPlan { enclosing })
    }
    pub fn program(&self) -> &Program {
        &self.program
    }
    pub fn device(&self) -> &Device {
        self.device
    }
    pub fn kernel_count(&self) -> usize {
        self.entries.iter().map(|e| e.kernels.borrow().len()).sum()
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
