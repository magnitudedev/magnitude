//! Derived resource accounting, with explicit units, provenance and uncertainty.

pub mod memory;
pub mod multiplicity;
pub mod quantity;
pub mod realization;
pub mod region;
pub mod resource;
pub mod work;

/// Portable algorithm operations and accessed regions. Neither component alone
/// proves a physical traffic obligation or a particular realization's duration.
#[derive(Clone, Debug)]
pub struct ComputationAccount {
    pub work: work::WorkAccount,
    pub memory: memory::MemoryAccount,
}

pub fn derive(
    program: &seismic_lang::program::Program,
    function: &str,
    shapes: &std::collections::HashMap<String, i64>,
    bindings: &memory::Bindings,
    access_analysis_budget: usize,
) -> Result<ComputationAccount, String> {
    derive_specialized(
        program,
        function,
        shapes,
        &std::collections::HashMap::new(),
        bindings,
        access_analysis_budget,
    )
}

pub fn derive_specialized(
    program: &seismic_lang::program::Program,
    function: &str,
    shapes: &std::collections::HashMap<String, i64>,
    elements: &std::collections::HashMap<String, seismic_lang::types::Elem>,
    bindings: &memory::Bindings,
    access_analysis_budget: usize,
) -> Result<ComputationAccount, String> {
    Ok(ComputationAccount {
        work: work::derive_specialized(program, function, shapes, elements)?,
        memory: memory::derive_specialized(
            program,
            function,
            shapes,
            elements,
            bindings,
            access_analysis_budget,
        )?,
    })
}

pub mod schedule;

pub mod selection;

pub mod execution_model;


pub mod workload;

pub mod authority;
