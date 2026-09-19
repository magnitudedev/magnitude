//! `realize`: the instantiated execution IR plus one deterministic rule per former backend
//! decision. Nothing here is searched or ranked.
//!
//!   phases         every root statement run is one phase, in source order: a root `parallel`
//!                  domain is one phase whose work items are its pieces; a run of serial root
//!                  statements is one single-item phase. Values that cross a phase boundary use
//!                  invocation-owned retained storage (`seismic_realization::phases`)
//!   dispatch       the pieces of a phase are claimed one at a time by the worker threads; a
//!                  single-item phase runs on the calling thread. No widening, no compiler split
//!   load           borrow where the borrow proof holds, else materialize: the greatest
//!                  fixpoint of the proof starting from all-borrow (`scalar_load_rule`)
//!   storage        every tile, region result and materialized snapshot is a new bump slot of
//!                  the executing worker's scratch, at its native element type; never reused
//!   reduction      ascending order on the owning thread, which satisfies both the ordered
//!                  and the unordered contract
//!   traversal      one element per iteration; no unrolling, no vector transfer
//!   floating point no contraction; FMA appears only where authored
use super::Limits;
use seismic_compiler::selection::SelectionError;
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_realization::{Dispatch, InvocationConditions, ScalarSequence};

/// The realized execution of one selected witness: source-ordered scalar phases.
pub struct Execution {
    pub(crate) sequence: ScalarSequence,
    pub(crate) conditions: InvocationConditions,
    pub(crate) codegen: crate::codegen::Policy,
}

impl Execution {
    pub fn name(&self) -> &str {
        &self.sequence.name
    }

    pub fn conditions(&self) -> &InvocationConditions {
        &self.conditions
    }

    pub fn sequence(&self) -> &ScalarSequence {
        &self.sequence
    }

    /// Scratch bytes one worker needs: the largest phase.
    pub fn scratch_bytes(&self) -> usize {
        self.sequence.phases.iter().map(|phase| phase.program.scratch_bytes).max().unwrap_or(0)
    }

    /// The scalar instruction listing of every phase, in execution order.
    pub fn listing(&self) -> String {
        let mut out = String::new();
        for (ordinal, phase) in self.sequence.phases.iter().enumerate() {
            out.push_str(&format!(
                "; {} phase {ordinal}: {} work item(s), {} scratch bytes per worker\n{}\n",
                self.sequence.name,
                phase.program.work_items,
                phase.program.scratch_bytes,
                phase.program.function.display()
            ));
        }
        out
    }
}

impl std::fmt::Debug for Execution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Execution").field("name", &self.sequence.name).field("phases", &self.sequence.phases.len()).finish()
    }
}

pub(super) fn realize(limits: &Limits, mut lowered: LoweredIr) -> Result<Execution, SelectionError> {
    if lowered.backend != super::TARGET {
        return Err(SelectionError::Reconstruction(format!("`{}` was instantiated for `{}`, not `{}`", lowered.name, lowered.backend, super::TARGET)));
    }
    let name = lowered.name.clone();
    let classify = |reason: String| {
        if reason.contains("no static") || reason.contains("not static") || reason.contains("no proven capacity") {
            SelectionError::AnalysisUnavailable(format!("`{name}`: {reason}"))
        } else {
            SelectionError::UnsupportedMapping(format!("`{name}`: {reason}"))
        }
    };
    let codegen = crate::codegen::Policy::host().map_err(|reason| SelectionError::UnsupportedMapping(format!("`{name}`: {reason}")))?;
    let conditions = InvocationConditions::from_lowered(&lowered).map_err(&classify)?;
    seismic_compiler::scalar_load_rule(&mut lowered).map_err(&classify)?;
    let sequence = seismic_compiler::scalar_sequence_resolved(&lowered, codegen.call_conv(), Dispatch::ParallelRoot).map_err(&classify)?;
    for (ordinal, phase) in sequence.phases.iter().enumerate() {
        if !phase.program.backend_calls.is_empty() {
            return Err(SelectionError::UnsupportedMapping(format!("`{name}` phase {ordinal}: participant operations have no CPU mapping")));
        }
        if u64::try_from(phase.program.scratch_bytes).map_or(true, |bytes| bytes > limits.max_scratch_bytes) {
            return Err(SelectionError::IncompatibleComposition(format!(
                "`{name}` phase {ordinal}: {} scratch bytes per worker exceed the {} byte limit",
                phase.program.scratch_bytes, limits.max_scratch_bytes
            )));
        }
    }
    Ok(Execution { sequence, conditions, codegen })
}
