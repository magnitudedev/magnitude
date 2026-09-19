//! Concrete typed realization contracts shared by compilation, accounting and
//! native backends. This crate does not emit code, predict time or select plans.
use seismic_lang::abi::ScalarParameter;
pub mod execution;
pub mod storage;
pub mod phases;
use cranelift_codegen::ir;
pub use cranelift_codegen::isa::CallConv;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub parameter: String,
    pub plane: String,
    pub bytes: usize,
    /// Required base-address alignment for this typed storage plane.
    pub alignment: usize,
}
/// Derived source invocation requirements retained through preparation/emission.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InvocationConditions {
    read_only_buffers: Vec<usize>,
    independent_buffers: Vec<usize>,
    alias_pairs: Vec<(usize, usize, bool)>,
}
impl InvocationConditions {
    pub fn from_lowered(function: &seismic_lang::exec::lowered_ir::LoweredIr) -> Result<Self, String> {
        function.ownership.validate(function)?;
        let (parameters, _) = storage::parameters(function)?;
        let mut alias_pairs = Vec::new();
        for requirement in &function.alias_requirements {
            let left = function
                .params
                .get(requirement.left)
                .ok_or("invalid source alias parameter")?;
            let right = function
                .params
                .get(requirement.right)
                .ok_or("invalid source alias parameter")?;
            if !matches!(left.1, seismic_lang::exec::types::Ty::Tensor(_))
                || !matches!(right.1, seismic_lang::exec::types::Ty::Tensor(_))
            {
                return Err("source alias requirement needs tensor parameters".into());
            }
            for (a, left_plane) in parameters
                .iter()
                .enumerate()
                .filter(|(_, p)| p.parameter == left.0)
            {
                for (b, right_plane) in parameters
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.parameter == right.0)
                {
                    alias_pairs.push((
                        a,
                        b,
                        requirement.exact_allowed && left_plane.plane == right_plane.plane,
                    ));
                }
            }
        }
        let read_only_buffers = parameters.iter().enumerate().filter_map(|(slot, buffer)| {
            let parameter = function.params.iter().position(|(name, _)| name == &buffer.parameter)?;
            let (id, _) = function.vars.iter().enumerate().find(|(_, var)| matches!(var.kind, seismic_lang::exec::ir::VarKind::Param(p) if p == parameter))?;
            seismic_lang::exec::effects::tensor_parameter_read_only(&function.body, id).then_some(slot)
        }).collect();
        Ok(Self {
            read_only_buffers,
            alias_pairs,
            independent_buffers: parameters
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    function
                        .ownership
                        .intermediates
                        .contains(&p.parameter)
                        .then_some(i)
                })
                .collect(),
        })
    }
    /// Parameters proven unmodified by the admitted computation, including aliases.
    pub fn read_only_buffers(&self) -> &[usize] { &self.read_only_buffers }
    pub fn independent_buffers(&self) -> &[usize] {
        &self.independent_buffers
    }
    pub fn alias_pairs(&self) -> &[(usize, usize, bool)] {
        &self.alias_pairs
    }
    /// Validate the used ABI byte ranges. The caller supplies backing identity
    /// and byte offset; allocation capacities are checked by its binding path.
    pub fn validate_aliases(
        &self,
        buffers: &[BufferSpec],
        locate: impl Fn(usize) -> (u64, u64),
    ) -> Result<(), String> {
        for &(a, b, exact_allowed) in &self.alias_pairs {
            let (left, right) = (
                buffers.get(a).ok_or("invalid alias ABI slot")?,
                buffers.get(b).ok_or("invalid alias ABI slot")?,
            );
            let ((left_id, left_offset), (right_id, right_offset)) = (locate(a), locate(b));
            let left_end = left_offset
                .checked_add(left.bytes as u64)
                .ok_or("alias range overflow")?;
            let right_end = right_offset
                .checked_add(right.bytes as u64)
                .ok_or("alias range overflow")?;
            if left_id == right_id
                && left.bytes != 0
                && right.bytes != 0
                && left_offset < right_end
                && right_offset < left_end
                && !(exact_allowed && left_offset == right_offset && left.bytes == right.bytes)
            {
                return Err("source parallel binding has unsafe overlapping storage".into());
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathFunction {
    Exp,
    ExpFast,
    Log,
    Sin,
    Cos,
}
impl MathFunction {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Exp | Self::ExpFast => "seismic_exp",
            Self::Log => "seismic_log",
            Self::Sin => "seismic_sin",
            Self::Cos => "seismic_cos",
        }
    }
}
/// Semantic calls expanded into visible terminal operations before accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipantOperation {
    LaneIndex,
    ShuffleIndex,
    Reduce(seismic_lang::exec::ir::ReduceOp),
}
/// Concrete storage and instruction program, not a cost estimate. Math imports
/// retain semantic identities; the target must supply an admitted implementation.
#[derive(Clone)]
pub struct ScalarProgram {
    pub conditions: InvocationConditions,
    pub function: ir::Function,
    pub buffers: Vec<BufferSpec>,
    /// Source-visible prefix of `buffers`. Remaining planes are invocation-owned
    /// phase storage and are never supplied by the caller.
    pub public_buffer_count: usize,
    pub scalars: Vec<ScalarParameter>,
    pub scratch_bytes: usize,
    pub imports: Vec<(ir::FuncRef, MathFunction)>,
    pub backend_calls: Vec<(ir::FuncRef, ParticipantOperation)>,
    pub participation: dispatch::Participation,
    pub work_items: u64,
    pub dispatch: Dispatch,
    pub loads: Vec<seismic_lang::exec::normalize::loads::Decision>,
    pub execution: execution::ExecutionEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    /// One invocation evaluates all source loops sequentially.
    Sequential,
    /// One invocation per outer parallel-domain point. Entry argument four is
    /// its linear index. Each invocation receives disjoint private scratch.
    ParallelRoot,
}

/// Explicit legal realization alternatives. Their ordering is not a performance preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStrategy {
    Materialize,
    BorrowProvenReadOnly,
}
#[derive(Clone, Copy, Debug)]
pub struct ScalarOptions {
    pub dispatch: Dispatch,
    pub loads: LoadStrategy,
}

/// Encode the shared scalar invocation ABI with checked typed values.
pub fn encode_scalars(schema: &[ScalarParameter], scalars: &[f64]) -> Result<Vec<u64>, String> {
    let bytes = seismic_lang::abi::ScalarLayout::words(schema)?.encode(scalars)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// Source-ordered phases with physical completion between them. Appended storage
/// retains values across launches for the lifetime of this invocation.
pub struct ScalarSequence {
    pub name: String,
    pub phases: Vec<ScalarPhase>,
    pub public_buffer_count: usize,
    pub retained: Vec<phases::RetainedValue>,
}
pub struct ScalarPhase {
    pub source_statement: usize,
    pub program: ScalarProgram,
}

pub mod dispatch;

pub mod graph;
pub mod integer;
pub mod scheduling;

pub mod liveness;
