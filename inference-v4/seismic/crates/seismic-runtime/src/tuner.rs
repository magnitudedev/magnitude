//! Native workload bindings and backend composition for compiler-owned tuning.
use crate::Buffer;
use seismic_accounting::workload as model;
use std::collections::BTreeMap;
/// Build exact allocation/alias/scalar conditions from retained native bindings.
/// No data transfer, candidate compilation or performance observation occurs.
pub fn workload(
    identity: impl Into<String>,
    buffers: &[Buffer],
    parameters: &[seismic_lang::abi::ScalarParameter],
    scalars: &[f64],
) -> Result<ScalarWorkload, String> {
    let mut allocations = Vec::new();
    let mut bindings: Vec<model::BufferBinding> = Vec::new();
    for (index, buffer) in buffers.iter().enumerate() {
        let allocation = if let Some(previous) = buffers[..index]
            .iter()
            .position(|b| buffer.shares_allocation(b))
        {
            bindings[previous].allocation
        } else {
            let id = allocations.len() as u64;
            allocations.push(model::Allocation {
                id,
                bytes: buffer.1.bytes as u64,
                alignment: buffer.model_alignment()?,
                known_bytes: BTreeMap::new(),
            });
            id
        };
        bindings.push(model::BufferBinding {
            allocation,
            offset: buffer.allocation_offset() as u64,
            bytes: buffer.len() as u64,
        });
    }
    Ok(ScalarWorkload {
        identity: identity.into(),
        allocations,
        buffers: bindings,
        scalars: seismic_lang::abi::ScalarLayout::words(parameters)?.encode(scalars)?,
    })
}
/// Establish every workload condition used by derivation before submission.
/// Allocation IDs are local names: their equality/inequality and relative byte
/// positions must match actual retained allocations, not parameter names.
pub(crate) fn validate_bindings(
    workload: &ScalarWorkload,
    buffers: &[Buffer],
    parameters: &[seismic_lang::abi::ScalarParameter],
    scalars: &[f64],
) -> Result<(), String> {
    if buffers.len() != workload.buffers.len() {
        return Err("tuned workload buffer count changed".into());
    }
    let encoded = seismic_lang::abi::ScalarLayout::words(parameters)?.encode(scalars)?;
    if encoded != workload.scalars {
        return Err("scalar bindings differ from the tuned workload".into());
    }
    for (index, (buffer, binding)) in buffers.iter().zip(&workload.buffers).enumerate() {
        let allocation = workload
            .allocations
            .iter()
            .find(|a| a.id == binding.allocation)
            .ok_or("tuned workload has no allocation for a binding")?;
        if !allocation.known_bytes.is_empty() {
            return Err("unverified content specialization".into());
        }
        if !allocation.alignment.is_power_of_two()
            || allocation.alignment > buffer.model_alignment()?
        {
            return Err("actual allocation does not establish the tuned alignment".into());
        }
        if u64::try_from(buffer.1.bytes).map_err(|_| "allocation size overflow")?
            != allocation.bytes
            || u64::try_from(buffer.allocation_offset())
                .map_err(|_| "allocation offset overflow")?
                != binding.offset
            || u64::try_from(buffer.len()).map_err(|_| "view size overflow")? != binding.bytes
        {
            return Err(format!(
                "buffer {index} differs from the tuned allocation/view conditions"
            ));
        }
        for other in 0..index {
            if buffer.shares_allocation(&buffers[other])
                != (binding.allocation == workload.buffers[other].allocation)
            {
                return Err("actual alias relationships differ from the tuned workload".into());
            }
        }
    }
    Ok(())
}

use crate::{DeviceFacts, execution::Execution};
pub use compiler::Input;
use seismic_accounting::{
    execution_model::ScalarHardware,
    schedule,
    selection::{self, Objective},
    workload::{DerivationError, DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner::{self as compiler, Preparation};
use seismic_lang::lowered_ir::LoweredIr;

/// Backend input facts. Implementation structure and dependencies remain owned
/// by the prepared execution; these inputs never supply an alternate program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hardware {
    Cpu(ScalarHardware),
    Cuda(seismic_cuda::model::CudaHardware),
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::model::Hardware),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Form {
    CpuScalar,
    CudaScalar,
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::tuning::Form),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImplementationConditions {
    Cpu(seismic_cpu::tuning::Conditions),
    Cuda(seismic_cuda::tuning::Conditions),
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::tuning::Conditions),
}
/// Exact composition inputs retained alongside the compiler's derived objective.
/// The runtime checks the device again before compiling the selected execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conditions {
    device: DeviceFacts,
    implementation: ImplementationConditions,
}
impl Conditions {
    pub fn device(&self) -> &DeviceFacts {
        &self.device
    }
    pub fn implementation(&self) -> &ImplementationConditions {
        &self.implementation
    }
    pub(crate) fn validate_device(&self, device: &DeviceFacts) -> Result<(), String> {
        if &self.device != device {
            return Err("native device differs from the tuned device conditions".into());
        }
        Ok(())
    }
}
pub type TunedIr = compiler::TunedIr<Execution, Conditions>;
pub type Artifact = compiler::Artifact<Conditions>;
pub type Progress = compiler::Progress<Execution, Conditions>;
pub type Outcome = compiler::Outcome<Execution, Conditions>;
pub struct Request<'a> {
    pub input: Input<'a>,
    pub device: &'a DeviceFacts,
    pub form: Form,
    pub hardware: &'a Hardware,
    pub workload: &'a ScalarWorkload,
    pub derivation_limits: DerivationLimits,
}
enum Implementation {
    Cpu(seismic_cpu::tuning::Backend),
    Cuda(seismic_cuda::tuning::Backend),
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::tuning::Backend),
}
struct Composition {
    device: DeviceFacts,
    implementation: Implementation,
}
pub(crate) fn map_preparation<A>(
    prepared: Preparation<A>,
    wrap: impl FnOnce(A) -> Execution,
) -> Preparation<Execution> {
    match prepared {
        Preparation::Choice { name, alternatives } => Preparation::Choice { name, alternatives },
        Preparation::Execution(execution) => Preparation::Execution(wrap(execution)),
        Preparation::Infeasible(v) => Preparation::Infeasible(v),
    }
}
impl compiler::Backend for Composition {
    type Execution = Execution;
    type Conditions = Conditions;
    fn name(&self) -> &'static str {
        match &self.implementation {
            Implementation::Cpu(b) => b.name(),
            Implementation::Cuda(b) => b.name(),
            #[cfg(target_os = "macos")]
            Implementation::Metal(b) => b.name(),
        }
    }
    fn conditions(&self) -> Conditions {
        Conditions {
            device: self.device.clone(),
            implementation: match &self.implementation {
                Implementation::Cpu(b) => ImplementationConditions::Cpu(b.conditions()),
                Implementation::Cuda(b) => ImplementationConditions::Cuda(b.conditions()),
                #[cfg(target_os = "macos")]
                Implementation::Metal(b) => ImplementationConditions::Metal(b.conditions()),
            },
        }
    }
    fn description(&self) -> compiler::Description {
        match &self.implementation {
            Implementation::Cpu(b) => b.description(),
            Implementation::Cuda(b) => b.description(),
            #[cfg(target_os = "macos")]
            Implementation::Metal(b) => b.description(),
        }
    }
    fn prepare(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<Preparation<Execution>, String> {
        Ok(match &self.implementation {
            Implementation::Cpu(b) => map_preparation(b.prepare(function, path)?, Execution::Cpu),
            Implementation::Cuda(b) => map_preparation(b.prepare(function, path)?, Execution::Cuda),
            #[cfg(target_os = "macos")]
            Implementation::Metal(b) => {
                map_preparation(b.prepare(function, path)?, Execution::Metal)
            }
        })
    }
    fn refine(
        &self,
        alternatives: &selection::Domain,
        index: usize,
    ) -> Result<Option<Preparation<Execution>>, String> {
        match &self.implementation {
            Implementation::Cpu(b) => Ok(b
                .refine(alternatives, index)?
                .map(|p| map_preparation(p, Execution::Cpu))),
            Implementation::Cuda(b) => Ok(b
                .refine(alternatives, index)?
                .map(|p| map_preparation(p, Execution::Cuda))),
            #[cfg(target_os = "macos")]
            Implementation::Metal(b) => Ok(b
                .refine(alternatives, index)?
                .map(|p| map_preparation(p, Execution::Metal))),
        }
    }
    fn analyze(
        &self,
        execution: &Execution,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<schedule::Model, DerivationError> {
        match (&self.implementation, execution) {
            (Implementation::Cpu(b), Execution::Cpu(e)) => b.analyze(e, workload, limits),
            (Implementation::Cuda(b), Execution::Cuda(e)) => b.analyze(e, workload, limits),
            #[cfg(target_os = "macos")]
            (Implementation::Metal(b), Execution::Metal(e)) => b.analyze(e, workload, limits),
            _ => Err("execution and analysis backend differ".into()),
        }
    }
    fn relax(
        &self,
        alternatives: &selection::Domain,
        indices: std::ops::Range<usize>,
        workload: &ScalarWorkload,
    ) -> Result<Option<schedule::Demand>, String> {
        match &self.implementation {
            Implementation::Cpu(b) => b.relax(alternatives, indices, workload),
            Implementation::Cuda(b) => b.relax(alternatives, indices, workload),
            #[cfg(target_os = "macos")]
            Implementation::Metal(b) => b.relax(alternatives, indices, workload),
        }
    }
    fn materialize(
        &self,
        execution: &Execution,
        objective: &Objective,
    ) -> Result<Execution, String> {
        match (&self.implementation, execution) {
            (Implementation::Cpu(b), Execution::Cpu(e)) => {
                b.materialize(e, objective).map(Execution::Cpu)
            }
            (Implementation::Cuda(b), Execution::Cuda(e)) => {
                b.materialize(e, objective).map(Execution::Cuda)
            }
            #[cfg(target_os = "macos")]
            (Implementation::Metal(b), Execution::Metal(e)) => {
                b.materialize(e, objective).map(Execution::Metal)
            }
            _ => Err("execution and materialization backend differ".into()),
        }
    }
    fn check_materialization(
        &self,
        source: &Execution,
        selected: &Execution,
        objective: &Objective,
    ) -> Result<(), String> {
        match (&self.implementation, source, selected) {
            (Implementation::Cpu(b), Execution::Cpu(source), Execution::Cpu(selected)) => {
                b.check_materialization(source, selected, objective)
            }
            (Implementation::Cuda(b), Execution::Cuda(source), Execution::Cuda(selected)) => {
                b.check_materialization(source, selected, objective)
            }
            #[cfg(target_os = "macos")]
            (Implementation::Metal(b), Execution::Metal(source), Execution::Metal(selected)) => {
                b.check_materialization(source, selected, objective)
            }
            _ => Err("source, selected execution and materialization backend differ".into()),
        }
    }
}
impl Request<'_> {
    fn backend(&self) -> Result<Composition, String> {
        if self
            .workload
            .allocations
            .iter()
            .any(|a| !a.known_bytes.is_empty())
        {
            return Err("native tuning requires value-independent tensor bindings; known bytes need immutable content binding".into());
        }
        let implementation = match (&self.form, self.device, self.hardware) {
            (
                Form::CpuScalar,
                DeviceFacts::Cpu {
                    architecture,
                    operating_system,
                },
                Hardware::Cpu(hardware),
            ) if *architecture == std::env::consts::ARCH
                && *operating_system == std::env::consts::OS =>
            {
                Implementation::Cpu(seismic_cpu::tuning::Backend::new(hardware)?)
            }
            (Form::CudaScalar, DeviceFacts::Cuda(device), Hardware::Cuda(hardware)) => {
                Implementation::Cuda(seismic_cuda::tuning::Backend::new(device, hardware)?)
            }
            #[cfg(target_os = "macos")]
            (Form::Metal(form), DeviceFacts::Metal(device), Hardware::Metal(hardware)) => {
                Implementation::Metal(seismic_metal::tuning::Backend::new(device, hardware, form)?)
            }
            _ => return Err("execution form, device and hardware input backends differ".into()),
        };
        Ok(Composition {
            device: self.device.clone(),
            implementation,
        })
    }
    fn compiler<'a>(&'a self, backend: &'a Composition) -> compiler::Request<'a, Composition> {
        compiler::Request {
            input: self.input,
            backend,
            workload: self.workload,
            derivation_limits: self.derivation_limits,
        }
    }
}
pub fn tune(request: &Request<'_>, budget: selection::Budget) -> Result<Outcome, String> {
    let backend = request.backend()?;
    compiler::tune(&request.compiler(&backend), budget)
}
pub fn resume(
    request: &Request<'_>,
    progress: Progress,
    budget: selection::Budget,
) -> Result<Outcome, String> {
    let backend = request.backend()?;
    compiler::resume(&request.compiler(&backend), progress, budget)
}
