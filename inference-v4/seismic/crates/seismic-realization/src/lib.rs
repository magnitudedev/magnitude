//! Concrete typed realization contracts shared by compilation, accounting and
//! native backends. This crate does not emit code, predict time or select plans.
use seismic_lang::abi::ScalarParameter;
pub mod execution;
pub mod storage;
pub mod memory;
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
/// Concrete storage and instruction program, not a cost estimate. Math imports
/// retain semantic identities; the target must supply an admitted implementation.
pub struct ScalarProgram {
    pub function: ir::Function,
    pub buffers: Vec<BufferSpec>,
    pub scalars: Vec<ScalarParameter>,
    pub scratch_bytes: usize,
    pub imports: Vec<(ir::FuncRef, MathFunction)>,
    pub work_items: u64,
    pub dispatch: Dispatch,
    pub loads: LoadStrategy,
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

/// Source-ordered phases with physical completion between them. Tensor parameters
/// retain their backing identity; phase-local values cannot escape their phase.
pub struct ScalarSequence {
    pub name: String,
    pub phases: Vec<ScalarPhase>,
}
pub struct ScalarPhase {
    pub source_statement: usize,
    pub program: ScalarProgram,
}

pub mod dispatch;

pub mod graph;
