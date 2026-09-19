//! Reusable compiled entries with retained, checked runtime bindings.
//!
//! One entry compiles once per `(entry, shapes, elements)`. Buffer contents and scalar
//! values are never part of that identity, so changing control inputs or decode
//! positions reuse the compiled kernel.
use crate::{Buffer, Device, ExecutionObservation, Kernel};
#[cfg(target_os = "macos")]
use crate::{DeviceTimingScope, Executable};
use seismic_compiler::selection::{Budget, Strategy};
use seismic_lang::{family::{Numerics, Workload}, sir::Program};
use seismic_lang::types::Elem;
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
};
pub trait Bindings {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer>;
    fn scalar(&self, name: &str) -> Option<f64>;
}
#[derive(Clone, Debug)]
pub struct StepObservation {
    pub entry: String,
    pub execution: ExecutionObservation,
    /// Native per-dispatch stage intervals of this step, in launch order.
    pub dispatches: Vec<crate::DispatchProfile>,
}
pub struct CompiledPlan {
    enclosing: Rc<Enclosing>,
}
impl CompiledPlan {
    pub fn shares_compilation(&self, other: &Self) -> bool { Rc::ptr_eq(&self.enclosing, &other.enclosing) }
    pub fn step_count(&self) -> usize { 1 }
    /// Kernels natively compiled so far; selection runs at the first preparation.
    pub fn kernel_count(&self) -> usize { usize::from(self.enclosing.kernel.borrow().is_some()) }
    pub fn execute(&mut self, bindings: &dyn Bindings) -> Result<(), String> {
        self.prepare(bindings)?.execute_sequential()
    }
    pub fn execute_observed(&mut self, bindings: &dyn Bindings) -> Result<Vec<StepObservation>, String> {
        self.prepare(bindings)?.execute_steps_observed()
    }
    pub fn prepare(&self, bindings: &dyn Bindings) -> Result<Submission, String> {
        self.enclosing.prepare(bindings)
    }
    /// The selected kernel, compiling it if this entry has not been prepared yet.
    pub fn kernel(&self) -> Result<Rc<RefCell<Kernel>>, String> {
        self.enclosing.kernel()
    }
    /// Bind the entry's ABI positionally, in the order of `Kernel::buffers`/`scalars`.
    pub fn execute_buffers(&mut self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        let kernel = self.enclosing.kernel()?;
        let mut submission = {
            let abi = kernel.try_borrow().map_err(|_| "shared kernel is already executing")?;
            if buffers.len() != abi.buffers().len() || scalars.len() != abi.scalars().len() {
                return Err("entry binding count differs from its logical ABI".into());
            }
            Submission {
                invocations: vec![BoundInvocation {
                    entry: self.enclosing.entry.clone(),
                    kernel: kernel.clone(),
                    buffers: abi.buffers().iter().zip(buffers).map(|(s, b)| b.view(0..s.bytes)).collect::<Result<_, _>>()?,
                    scalars: scalars.to_vec(),
                }],
            }
        };
        submission.execute_sequential()
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
                let (execution, dispatches) = invocation
                    .kernel
                    .try_borrow_mut()
                    .map_err(|_| "shared kernel is already executing")?
                    .execute_profiled(&invocation.buffers, &invocation.scalars)
                    .map_err(|e| format!("{}: {e}", invocation.entry))?;
                Ok(StepObservation {
                    entry: invocation.entry.clone(),
                    execution,
                    dispatches,
                })
            })
            .collect()
    }
    /// Metal encodes one command buffer; the CPU runs the invocations in source order. The
    /// whole batch validates before any of it executes. No asynchronous work escapes this
    /// method, including on failure.
    pub fn execute_batched(&mut self) -> Result<ExecutionObservation, String> {
        let start = std::time::Instant::now();
        let kernels = self
            .invocations
            .iter()
            .map(|i| i.kernel.try_borrow().map_err(|_| "shared kernel is already executing".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        for (kernel, bound) in kernels.iter().zip(&self.invocations) {
            kernel
                .validate_invocation(&bound.buffers, &bound.scalars)
                .map_err(|e| format!("{}: {e}", bound.entry))?;
        }
        if kernels.is_empty() {
            return Ok(ExecutionObservation {
                host_seconds: start.elapsed().as_secs_f64(),
                device_seconds: None,
                device_scope: None,
            });
        }
        // A CPU or CUDA batch runs in source order; each kernel reports its own interval.
        let sequential = |kernels: Vec<std::cell::Ref<'_, Kernel>>| -> Result<ExecutionObservation, String> {
            drop(kernels);
            let (mut seconds, mut scope) = (0.0, None);
            for bound in &self.invocations {
                let mut kernel = bound.kernel.try_borrow_mut().map_err(|_| "shared kernel is already executing")?;
                let observed = kernel.execute_observed(&bound.buffers, &bound.scalars).map_err(|e| format!("{}: {e}", bound.entry))?;
                seconds += observed.device_seconds.ok_or("an observed execution reported no device time")?;
                scope = observed.device_scope;
            }
            Ok(ExecutionObservation { host_seconds: start.elapsed().as_secs_f64(), device_seconds: Some(seconds), device_scope: scope })
        };
        #[cfg(target_os = "macos")]
        let one_command_buffer = matches!(kernels.first().map(|kernel| &kernel.executable), Some(Executable::Metal { .. }));
        #[cfg(not(target_os = "macos"))]
        let one_command_buffer = false;
        if !one_command_buffer {
            return sequential(kernels);
        }
        #[cfg(not(target_os = "macos"))]
        return Err("no batched backend exists on this host".into());
        #[cfg(target_os = "macos")]
        {
            let Some(Executable::Metal { device, .. }) = kernels.first().map(|kernel| &kernel.executable) else {
                return Err("a batched Metal submission lost its first kernel".into());
            };
            let invocations = kernels
                .iter()
                .zip(&self.invocations)
                .map(|(kernel, bound)| {
                    let Executable::Metal { pipeline, .. } = &kernel.executable else {
                        return Err(format!("{}: a batched Metal submission holds a kernel of another backend", bound.entry));
                    };
                    Ok(seismic_metal::runtime::Invocation {
                        pipeline,
                        buffers: bound.buffers.iter().map(Buffer::metal).collect::<Result<_, _>>()?,
                        scalars: pipeline.emitted.encode_scalars(&bound.scalars)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let seconds = device.run_many(&invocations, 1)?;
            Ok(ExecutionObservation {
                host_seconds: start.elapsed().as_secs_f64(),
                device_seconds: Some(seconds),
                device_scope: Some(DeviceTimingScope::CommandBuffer),
            })
        }
    }
}

/// Explicit search budget. Implementation choices are compiler-owned; the backend and its
/// capacities come from the device the plan compiles for.
#[derive(Clone, Copy, Debug, Default)]
pub struct Settings {
    pub budget: Budget,
    /// How the seed is improved; replaces `budget.strategy` for every entry of the plan.
    pub strategy: Strategy,
    /// Numerical latitude of selection; part of every compiled entry's workload identity.
    pub numerics: Numerics,
}
struct Enclosing {
    device: Device,
    program: Rc<Program>,
    entry: String,
    workload: Workload,
    settings: Settings,
    kernel: RefCell<Option<Rc<RefCell<Kernel>>>>,
}
impl Enclosing {
    /// Joint selection, then native compilation of exactly the selected witness. A failed
    /// selection is an error; nothing is retained and no other kernel substitutes.
    fn kernel(&self) -> Result<Rc<RefCell<Kernel>>, String> {
        if let Some(kernel) = self.kernel.borrow().as_ref() {
            return Ok(kernel.clone());
        }
        let selected = self.device.select(&self.program, &self.entry, &self.workload, Budget { strategy: self.settings.strategy, ..self.settings.budget })
            .map_err(|e| format!("{}: {e}", self.entry))?;
        let kernel = Rc::new(RefCell::new(
            self.device.compile_selected(selected).map_err(|e| format!("{}: {e}", self.entry))?,
        ));
        *self.kernel.borrow_mut() = Some(kernel.clone());
        Ok(kernel)
    }
    fn prepare(&self, bindings: &dyn Bindings) -> Result<Submission, String> {
        let kernel = self.kernel()?;
        let (buffers, scalars) = {
            let abi = kernel.try_borrow().map_err(|_| "shared kernel is already executing")?;
            let buffers = abi
                .buffers()
                .iter()
                .map(|s| {
                    bindings
                        .buffer(&s.parameter, &s.plane)
                        .ok_or_else(|| format!("unbound tensor {}.{}", s.parameter, s.plane))?
                        .view(0..s.bytes)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let scalars = abi
                .scalars()
                .iter()
                .map(|s| bindings.scalar(&s.name).ok_or_else(|| format!("unbound scalar {}", s.name)))
                .collect::<Result<Vec<_>, String>>()?;
            (buffers, scalars)
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

/// Compiles exported entries through joint selection only.
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
    pub fn settings(&self) -> Settings {
        self.settings
    }
    /// Compilation identity is exactly `(entry, shapes, elements, numerics)`.
    pub fn compile_entry(
        &mut self,
        entry: &str,
        shapes: &HashMap<String, i64>,
        elements: &HashMap<String, Elem>,
    ) -> Result<CompiledPlan, String> {
        let workload = Workload {
            shapes: shapes.iter().map(|(n, v)| (n.clone(), *v)).collect(),
            elems: elements.iter().map(|(n, e)| (n.clone(), e.clone())).collect(),
            numerics: self.settings.numerics,
        };
        if let Some(enclosing) = self.entries.iter().find(|e| e.entry == entry && e.workload == workload) {
            return Ok(CompiledPlan { enclosing: enclosing.clone() });
        }
        if self.program.export(entry).is_none() {
            return Err(format!("unknown exported entry {entry}"));
        }
        let enclosing = Rc::new(Enclosing {
            device: self.device.clone(),
            program: self.program.clone(),
            entry: entry.into(),
            workload,
            settings: self.settings,
            kernel: RefCell::new(None),
        });
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
        self.entries.iter().filter(|e| e.kernel.borrow().is_some()).count()
    }
}
