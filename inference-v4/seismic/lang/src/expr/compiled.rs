//! Self-contained evaluators compiled from arena nodes (spec §5.3).
//!
//! A `Compiled<T>` owns everything it needs: it is the form in which
//! `ExecutableVariant` carries guards, durations, layouts and geometry after the
//! arena is gone. It evaluates against an [`InvocationValues`] built by the
//! generated call bindings.

use super::{DurationEstimate, EvalError, PartialAssignment, SymbolId, SymbolValue};
use std::fmt;

/// Symbol values supplied by one invocation: call dimensions from tensor
/// descriptors, call scalars from arguments, target constants from the
/// profile the variant was compiled for. Indexed by symbol.
#[derive(Clone, Debug, Default)]
pub struct InvocationValues {
    values: Vec<(SymbolId, SymbolValue)>,
}

impl InvocationValues {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn bind(&mut self, symbol: SymbolId, value: SymbolValue) {
        self.values.retain(|(s, _)| *s != symbol);
        self.values.push((symbol, value));
    }
    pub fn get(&self, symbol: SymbolId) -> Option<SymbolValue> {
        self.values
            .iter()
            .find(|(s, _)| *s == symbol)
            .map(|(_, v)| *v)
    }
}

/// A compiled evaluator. `Send + Sync` so a prepared kernel can be shared.
pub struct Compiled<T> {
    program: Box<dyn Fn(&InvocationValues) -> Result<T, EvalError> + Send + Sync>,
    /// The symbols the evaluator reads, in a fixed order, so a binding table
    /// can be validated against it before any evaluation.
    reads: Vec<SymbolId>,
}

impl<T> Compiled<T> {
    pub(crate) fn new(
        reads: Vec<SymbolId>,
        program: Box<dyn Fn(&InvocationValues) -> Result<T, EvalError> + Send + Sync>,
    ) -> Self {
        Self { program, reads }
    }

    pub fn evaluate(&self, values: &InvocationValues) -> Result<T, EvalError> {
        (self.program)(values)
    }

    pub fn reads(&self) -> &[SymbolId] {
        &self.reads
    }
}

impl<T> fmt::Debug for Compiled<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compiled")
            .field("reads", &self.reads)
            .finish()
    }
}

pub type CompiledPredicate = Compiled<bool>;
pub type CompiledNat = Compiled<u64>;
pub type CompiledInt = Compiled<i64>;
pub type CompiledDuration = Compiled<DurationEstimate>;

/// Constructs a symbol-free predicate for compiler-side qualification gates.
#[doc(hidden)]
pub fn constant_predicate(value: bool) -> CompiledPredicate {
    Compiled::new(Vec::new(), Box::new(move |_| Ok(value)))
}

/// Solver-only evaluator whose remaining reads may be finite decisions.
/// Its input is deliberately [`PartialAssignment`], so it cannot be supplied
/// where invocation-bound compiled expressions are accepted.
pub struct CompiledDecisionPredicate {
    program: Box<dyn Fn(&PartialAssignment) -> Result<bool, EvalError> + Send + Sync>,
}

impl CompiledDecisionPredicate {
    pub(crate) fn new(
        program: Box<dyn Fn(&PartialAssignment) -> Result<bool, EvalError> + Send + Sync>,
    ) -> Self {
        Self { program }
    }

    pub fn evaluate(&self, values: &PartialAssignment) -> Result<bool, EvalError> {
        (self.program)(values)
    }
}

impl fmt::Debug for CompiledDecisionPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledDecisionPredicate").finish()
    }
}
