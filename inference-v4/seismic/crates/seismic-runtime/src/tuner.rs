//! Native workload bindings and backend composition for compiler-owned tuning.
use crate::Buffer;
use seismic_accounting::workload as model;
use std::collections::BTreeMap;
pub use model::{IntegerDomain, IntegerInput, IntegerRange};

/// One checked varying integer field relative to a bound buffer view.
#[derive(Clone, Debug)]
pub struct BufferIntegerDomain {
    pub offset: u64,
    pub bytes: u8,
    pub signed: bool,
    pub range: IntegerRange,
}
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
    Ok(ScalarWorkload { integer_domains: Vec::new(),
        identity: identity.into(),
        allocations,
        buffers: bindings,
        scalars: seismic_lang::abi::ScalarLayout::words(parameters)?.encode(scalars)?,
    })
}
/// Capture explicitly bound control data, never weights or scratch by size/name
/// heuristics. Native submission rechecks the bytes and read-only conditions.
pub fn capture_contents(workload: &mut ScalarWorkload, buffers: &[Buffer], slots: &[usize]) -> Result<(), String> {
    for &slot in slots {
        let buffer = buffers.get(slot).ok_or("unknown content binding")?;
        let binding = workload.buffers.get(slot).ok_or("unknown content workload binding")?;
        let allocation = workload.allocations.iter_mut().find(|a| a.id == binding.allocation).ok_or("missing content allocation")?;
        let mut bytes = vec![0; buffer.len()];
        buffer.read(&mut bytes)?;
        for (i, byte) in bytes.into_iter().enumerate() {
            let offset = binding.offset.checked_add(i as u64).ok_or("content offset overflow")?;
            if offset >= allocation.bytes { return Err("content outside allocation".into()); }
            if allocation.known_bytes.insert(offset, byte).is_some_and(|previous| previous != byte) {
                return Err("conflicting aliased content bindings".into());
            }
        }
    }
    Ok(())
}
/// Establish a uniform workload domain before selection, checking today's
/// invocation as well as every later submission against the same contract.
/// A backend must derive the whole domain or leave selection unresolved.
pub fn generalize_inputs(
    workload: &mut ScalarWorkload,
    buffers: &[Buffer],
    parameters: &[seismic_lang::abi::ScalarParameter],
    mut domains: Vec<IntegerDomain>,
) -> Result<(), String> {
    if !workload.integer_domains.is_empty() { return Err("workload already has integer domains".into()); }
    domains.sort_by(|a, b| a.input.cmp(&b.input));
    let mut generalized = workload.clone();
    for domain in &domains {
        let bytes = integer_input_bytes(domain, workload, buffers, &workload.scalars)?;
        if !domain.accepts(&bytes) { return Err("invocation does not establish its integer input domain".into()); }
        match domain.input {
            IntegerInput::Scalar { slot } => {
                let start = slot.checked_mul(8).ok_or("scalar domain offset overflow")?;
                let end = start.checked_add(usize::from(domain.bytes)).ok_or("scalar domain offset overflow")?;
                let destination = generalized.scalars.get_mut(start..end).ok_or("scalar domain exceeds ABI")?;
                destination.copy_from_slice(&(domain.range.min as u64).to_le_bytes()[..usize::from(domain.bytes)]);
            }
            IntegerInput::Allocation { allocation, offset } => {
                let end = offset.checked_add(u64::from(domain.bytes)).ok_or("integer input offset overflow")?;
                let a = generalized.allocations.iter_mut().find(|a| a.id == allocation).ok_or("integer input allocation missing")?;
                a.known_bytes.retain(|byte, _| *byte < offset || *byte >= end);
            }
        }
    }
    generalized.integer_domains = domains;
    generalized.validate()?;
    validate_scalar_domains(&generalized, parameters)?;
    *workload = generalized;
    Ok(())
}
fn validate_scalar_domains(workload: &ScalarWorkload, parameters: &[seismic_lang::abi::ScalarParameter]) -> Result<(), String> {
    use seismic_lang::types::DType;
    for domain in &workload.integer_domains {
        let IntegerInput::Scalar { slot } = domain.input else { continue; };
        let parameter = parameters.get(slot).ok_or("integer domain scalar missing")?;
        let signed = match parameter.dtype {
            DType::I32 => true,
            DType::U32 => false,
            _ => return Err("integer input domain requires an integer scalar".into()),
        };
        if parameter.dtype.bytes() != u32::from(domain.bytes) || signed != domain.signed {
            return Err("integer input domain differs from scalar type".into());
        }
        if parameter.index_bound.is_some_and(|bound| domain.range.min < 0 || domain.range.max >= i128::from(bound)) {
            return Err("integer input domain exceeds source index bound".into());
        }
    }
    Ok(())
}
fn integer_input_bytes(domain: &IntegerDomain, workload: &ScalarWorkload, buffers: &[Buffer], scalars: &[u8]) -> Result<Vec<u8>, String> {
    match domain.input {
        IntegerInput::Scalar { slot } => {
            let start = slot.checked_mul(8).ok_or("scalar domain offset overflow")?;
            let end = start.checked_add(usize::from(domain.bytes)).ok_or("scalar domain offset overflow")?;
            Ok(scalars.get(start..end).ok_or("scalar domain exceeds ABI")?.to_vec())
        }
        IntegerInput::Allocation { allocation, offset } => {
            let end = offset.checked_add(u64::from(domain.bytes)).ok_or("integer input offset overflow")?;
            let (slot, binding) = workload.buffers.iter().enumerate().find(|(_, b)|
                b.allocation == allocation && b.offset <= offset && b.offset.checked_add(b.bytes).is_some_and(|bound| end <= bound))
                .ok_or("integer input lies outside bound views")?;
            let start = usize::try_from(offset - binding.offset).map_err(|_| "integer input offset exceeds address range")?;
            let length = usize::from(domain.bytes);
            let buffer = buffers.get(slot).ok_or("integer input buffer missing")?;
            let mut bytes = vec![0; length];
            buffer.view(start..start.checked_add(length).ok_or("integer input read overflow")?)?.read(&mut bytes)?;
            Ok(bytes)
        }
    }
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
    workload.validate()?;
    validate_scalar_domains(workload, parameters)?;
    if buffers.len() != workload.buffers.len() {
        return Err("tuned workload buffer count changed".into());
    }
    let encoded = seismic_lang::abi::ScalarLayout::words(parameters)?.encode(scalars)?;
    if !workload.accepts_scalars(&encoded) {
        return Err("scalar bindings differ from the tuned workload".into());
    }
    for (index, (buffer, binding)) in buffers.iter().zip(&workload.buffers).enumerate() {
        let allocation = workload
            .allocations
            .iter()
            .find(|a| a.id == binding.allocation)
            .ok_or("tuned workload has no allocation for a binding")?;
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
    for allocation in &workload.allocations {
        let mut remaining = allocation.known_bytes.clone();
        if remaining.is_empty() { continue; }
        for (slot, binding) in workload.buffers.iter().enumerate().filter(|(_, b)| b.allocation == allocation.id) {
            let end = binding.offset.checked_add(binding.bytes).ok_or("content range overflow")?;
            let covered = remaining.range(binding.offset..end).map(|(&offset, &byte)| (offset, byte)).collect::<Vec<_>>();
            let (Some(first), Some(last)) = (covered.first(), covered.last()) else { continue; };
            let start = usize::try_from(first.0 - binding.offset).map_err(|_| "content offset exceeds address range")?;
            let length = usize::try_from(last.0 - first.0 + 1).map_err(|_| "content length exceeds address range")?;
            let mut actual = vec![0; length];
            buffers[slot].view(start..start + length)?.read(&mut actual)?;
            for (offset, expected) in &covered {
                if actual[(*offset - first.0) as usize] != *expected { return Err("buffer contents differ from tuned workload".into()); }
                remaining.remove(offset);
            }
        }
        if !remaining.is_empty() { return Err("content condition lies outside bound views".into()); }
    }
    for domain in &workload.integer_domains {
        if matches!(domain.input, IntegerInput::Allocation { .. })
            && !domain.accepts(&integer_input_bytes(domain, workload, buffers, &encoded)?) {
            return Err("buffer contents differ from tuned integer input domain".into());
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
    Metal,
}
impl Form {
    /// Whether this accounting implementation can derive a uniform execution
    /// over explicitly varying integer controls. Exact workloads remain valid
    /// on forms that do not yet provide this analysis.
    pub fn supports_integer_domains(&self) -> bool {
        if matches!(self, Self::CudaScalar) { return true; }
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal => true,
            _ => false,
        }
    }
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
        Preparation::Unresolved(reason) => Preparation::Unresolved(reason),
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
    ) -> Result<schedule::evaluation::Model, DerivationError> {
        match (&self.implementation, execution) {
            (Implementation::Cpu(b), Execution::Cpu(e)) => b.analyze(e, workload, limits),
            (Implementation::Cuda(b), Execution::Cuda(e)) => b.analyze(e, workload, limits),
            #[cfg(target_os = "macos")]
            (Implementation::Metal(b), Execution::Metal(e)) => b.analyze(e, workload, limits),
            _ => Err("execution and analysis backend differ".into()),
        }
    }
    fn relax_execution(
        &self,
        execution: &Execution,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        match (&self.implementation, execution) {
            (Implementation::Cpu(b), Execution::Cpu(e)) => b.relax_execution(e, workload, limits),
            (Implementation::Cuda(b), Execution::Cuda(e)) => b.relax_execution(e, workload, limits),
            #[cfg(target_os = "macos")]
            (Implementation::Metal(b), Execution::Metal(e)) => b.relax_execution(e, workload, limits),
            _ => Err("execution and relaxation backend differ".into()),
        }
    }
    fn relax(
        &self,
        alternatives: &selection::Domain,
        indices: std::ops::Range<usize>,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        match &self.implementation {
            Implementation::Cpu(b) => b.relax(alternatives, indices, workload, limits),
            Implementation::Cuda(b) => b.relax(alternatives, indices, workload, limits),
            #[cfg(target_os = "macos")]
            Implementation::Metal(b) => b.relax(alternatives, indices, workload, limits),
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
        match self.input {
            Input::Lowered(_) => return Err("native tuning requires portable source; preselected lowered IR cannot bypass frontend selection".into()),
            Input::Portable { options, .. } if options.piece.is_some() => return Err("native tuning does not accept a fixed stream piece; decomposition is compiler-owned".into()),
            Input::Portable { .. } => {}
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
            (Form::Metal, DeviceFacts::Metal(device), Hardware::Metal(hardware)) => {
                Implementation::Metal(seismic_metal::tuning::Backend::new(device, hardware, &seismic_metal::tuning::Form::Automatic)?)
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

#[cfg(test)]
mod input_domain_tests {
    use super::*;
    #[test]
    fn generalized_controls_are_checked_at_preparation_and_submission() {
        let device = crate::Device::cpu();
        let backing = device.buffer_from(&[0u8; 32]).unwrap();
        let control = backing.view(8..24).unwrap();
        control.view(4..8).unwrap().write(&6i32.to_le_bytes()).unwrap();
        let buffers = [control];
        let parameters = [seismic_lang::abi::ScalarParameter {
            name: "position".into(), dtype: seismic_lang::types::DType::I32, index_bound: Some(16),
        }];
        let mut facts = workload("varying controls", &buffers, &parameters, &[4.]).unwrap();
        capture_contents(&mut facts, &buffers, &[0]).unwrap();
        let domains = vec![
            IntegerDomain { input: IntegerInput::Scalar { slot: 0 }, bytes: 4, signed: true,
                range: IntegerRange { min: 0, max: 12, stride: 4 } },
            IntegerDomain { input: IntegerInput::Allocation { allocation: 0, offset: 12 }, bytes: 4, signed: true,
                range: IntegerRange { min: 2, max: 10, stride: 2 } },
        ];
        generalize_inputs(&mut facts, &buffers, &parameters, domains.clone()).unwrap();
        assert_eq!(&facts.scalars[..4], &0i32.to_le_bytes());
        assert!(facts.allocations[0].known_bytes.range(12..16).next().is_none());
        assert!(facts.conditions_allocation(0));
        validate_bindings(&facts, &buffers, &parameters, &[4.]).unwrap();
        buffers[0].view(4..8).unwrap().write(&10i32.to_le_bytes()).unwrap();
        validate_bindings(&facts, &buffers, &parameters, &[12.]).unwrap();
        assert!(validate_bindings(&facts, &buffers, &parameters, &[2.]).is_err());
        buffers[0].view(4..8).unwrap().write(&9i32.to_le_bytes()).unwrap();
        assert!(validate_bindings(&facts, &buffers, &parameters, &[12.]).unwrap_err().contains("integer input domain"));
        let mut new_facts = workload("varying controls", &buffers, &parameters, &[4.]).unwrap();
        let previous = new_facts.clone();
        assert!(generalize_inputs(&mut new_facts, &buffers, &parameters, domains).is_err());
        assert_eq!(new_facts, previous, "failed preparation must not publish generalized facts");
        buffers[0].view(4..8).unwrap().write(&2i32.to_le_bytes()).unwrap();
        buffers[0].view(0..4).unwrap().write(&1i32.to_le_bytes()).unwrap();
        assert!(validate_bindings(&facts, &buffers, &parameters, &[8.]).unwrap_err().contains("contents differ"));
        let mut outside_source = workload("varying controls", &buffers, &parameters, &[4.]).unwrap();
        assert!(generalize_inputs(&mut outside_source, &buffers, &parameters, vec![
            IntegerDomain { input: IntegerInput::Scalar { slot: 0 }, bytes: 4, signed: true,
                range: IntegerRange { min: 0, max: 20, stride: 4 } },
        ]).unwrap_err().contains("source index bound"));
        let signed = [seismic_lang::abi::ScalarParameter::plain("offset", seismic_lang::types::DType::I32)];
        let mut negative = workload("signed varying input", &[], &signed, &[-4.]).unwrap();
        generalize_inputs(&mut negative, &[], &signed, vec![
            IntegerDomain { input: IntegerInput::Scalar { slot: 0 }, bytes: 4, signed: true,
                range: IntegerRange { min: -8, max: 8, stride: 4 } },
        ]).unwrap();
        assert_eq!(negative.scalars, [(-8i32).to_le_bytes(), [0; 4]].concat());
        validate_bindings(&negative, &[], &signed, &[-4.]).unwrap();
        validate_bindings(&negative, &[], &signed, &[4.]).unwrap();
        assert!(validate_bindings(&negative, &[], &signed, &[-2.]).is_err());
    }
}
