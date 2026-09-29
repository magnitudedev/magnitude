//! Self-contained evaluators compiled from arena nodes (spec §5.3).
//!
//! A `Compiled<T>` owns everything it needs: it is the form in which
//! `ExecutableVariant` carries guards, durations, layouts and geometry after the
//! arena is gone. It evaluates against an [`InvocationValues`] built by the
//! generated call bindings.

use super::{DurationEstimate, EvalError, PartialAssignment, SymbolId, SymbolValue};
use num_bigint::{BigInt, BigUint};
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
            .map(|(_, v)| v.clone())
    }
}

/// A compiled evaluator. `Send + Sync` so a prepared kernel can be shared.
pub struct Compiled<T> {
    program: std::sync::Arc<dyn Fn(&InvocationValues) -> Result<T, EvalError> + Send + Sync>,
    retained_bytes: usize,
    /// The symbols the evaluator reads, in a fixed order, so a binding table
    /// can be validated against it before any evaluation.
    reads: std::sync::Arc<[SymbolId]>,
}

impl<T> Compiled<T> {
    pub(crate) fn new(
        reads: Vec<SymbolId>,
        retained_bytes: usize,
        program: Box<dyn Fn(&InvocationValues) -> Result<T, EvalError> + Send + Sync>,
    ) -> Self {
        let retained_bytes = retained_bytes
            .saturating_add(reads.len() * std::mem::size_of::<SymbolId>())
            .saturating_add(std::mem::size_of::<Self>());
        Self {
            program: program.into(),
            reads: reads.into(),
            retained_bytes,
        }
    }

    pub fn evaluate(&self, values: &InvocationValues) -> Result<T, EvalError> {
        (self.program)(values)
    }

    pub fn retained_metadata_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub fn reads(&self) -> &[SymbolId] {
        &self.reads
    }
}

impl<T> Clone for Compiled<T> {
    fn clone(&self) -> Self {
        Self {
            program: self.program.clone(),
            reads: self.reads.clone(),
            retained_bytes: self.retained_bytes,
        }
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
pub type CompiledNat = Compiled<BigUint>;
pub type CompiledInt = Compiled<BigInt>;
pub type CompiledDuration = Compiled<DurationEstimate>;

impl Compiled<BigUint> {
    /// Explicit finite projection at an address, allocation, launch, or ABI consumer.
    pub fn evaluate_u64(&self, values: &InvocationValues) -> Result<u64, EvalError> {
        self.evaluate(values)?
            .try_into()
            .map_err(|_| EvalError::Unrepresentable)
    }
}

impl Compiled<BigInt> {
    /// Explicit finite projection; mathematical evaluation itself remains exact.
    pub fn evaluate_i64(&self, values: &InvocationValues) -> Result<i64, EvalError> {
        self.evaluate(values)?
            .try_into()
            .map_err(|_| EvalError::Unrepresentable)
    }
}

/// Constructs a symbol-free predicate for compiler-side qualification gates.
#[doc(hidden)]
pub fn constant_predicate(value: bool) -> CompiledPredicate {
    Compiled::new(
        Vec::new(),
        std::mem::size_of::<bool>(),
        Box::new(move |_| Ok(value)),
    )
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
