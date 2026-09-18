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

use crate::{choices, DeviceFacts};
pub use compiler::Input;
use seismic_accounting::{
    execution_model::ScalarHardware,
    selection,
    workload::{DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner as compiler;
pub type TunedIr =
    compiler::TunedIr<seismic_realization::ScalarProgram, seismic_cpu::tuning::Conditions>;
pub type Artifact = compiler::Artifact<seismic_cpu::tuning::Conditions>;
pub type Progress =
    compiler::Progress<seismic_realization::ScalarProgram, seismic_cpu::tuning::Conditions>;
pub type Outcome =
    compiler::Outcome<seismic_realization::ScalarProgram, seismic_cpu::tuning::Conditions>;
pub struct Request<'a> {
    pub input: Input<'a>,
    pub device: &'a DeviceFacts,
    pub form: choices::Form,
    pub hardware: &'a ScalarHardware,
    pub workload: &'a ScalarWorkload,
    pub derivation_limits: DerivationLimits,
}
impl Request<'_> {
    fn backend(&self) -> Result<seismic_cpu::tuning::Backend, String> {
        match (&self.form, self.device) {
            (
                choices::Form::CpuScalar,
                DeviceFacts::Cpu {
                    architecture,
                    operating_system,
                },
            ) if *architecture == std::env::consts::ARCH
                && *operating_system == std::env::consts::OS => {}
            _ => return Err("tuning requires the host CPU scalar form".into()),
        }
        if self
            .workload
            .allocations
            .iter()
            .any(|a| !a.known_bytes.is_empty())
        {
            return Err("native tuning requires value-independent tensor bindings; known bytes need immutable content binding".into());
        }
        seismic_cpu::tuning::Backend::new(self.hardware)
    }
    fn compiler<'a>(
        &'a self,
        backend: &'a seismic_cpu::tuning::Backend,
    ) -> compiler::Request<'a, seismic_cpu::tuning::Backend> {
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
