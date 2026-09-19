//! Immutable model and original-execution reconstruction travel together.
//! This boundary contains no solver strategy or candidate-scoring callback.
use magnitude_solver::{FeasibleSolution, Model};
use seismic_accounting::objective::Objective;
use seismic_lang::lowered_ir::LoweredIr;
use std::sync::Arc;

/// A semantic assignment, rather than an ordinal in a replay traversal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub identity: String,
    pub value: i64,
}

pub struct Reconstructed<E> {
    pub execution: E,
    pub source: LoweredIr,
    pub objective: Objective,
    pub decisions: Vec<Decision>,
}

/// Implementations retain original IR and typed binding handles. They must
/// validate emitted operations, storage, dispatch, effects and schedule against
/// those handles; reconstruction cannot introduce new implementation choices.
pub trait Reconstruction<E>: 'static {
    fn reconstruct(&self, witness: &FeasibleSolution, lower_bound: u64) -> Result<Reconstructed<E>, String>;
}

pub struct Export<E> {
    model: Arc<Model>,
    reconstruction: Box<dyn Reconstruction<E>>,
}
impl<E: 'static> Export<E> {
    pub fn new(model: Arc<Model>, reconstruction: impl Reconstruction<E>) -> Result<Self, String> {
        model.validate().map_err(|e| e.to_string())?;
        Ok(Self { model, reconstruction: Box::new(reconstruction) })
    }
    pub fn model(&self) -> &Arc<Model> { &self.model }
    pub fn reconstruct(&self, witness: &FeasibleSolution, lower_bound: u64) -> Result<Reconstructed<E>, String> {
        if witness.model() != self.model.as_ref() {
            return Err("solver assignment belongs to a different execution family".into());
        }
        let checked = self.model.validate_assignment(witness.values()).map_err(|e| e.to_string())?;
        if checked.infeasible || checked.unresolved.is_some() || checked.exact_cost != Some(witness.cost()) {
            return Err("solver assignment does not establish complete family feasibility".into());
        }
        let selected = self.reconstruction.reconstruct(witness, lower_bound)?;
        // Objective construction checks and owns the immutable original witness.
        // Only the joint model's cost correspondence remains to establish here.
        if selected.objective.cost().upper() != witness.cost() {
            return Err("reconstructed objective differs from the original model assignment".into());
        }
        let mut identities = std::collections::BTreeSet::new();
        if selected.decisions.iter().any(|decision| decision.identity.is_empty() || !identities.insert(&decision.identity)) {
            return Err("reconstruction has missing or duplicate semantic decision identities".into());
        }
        Ok(selected)
    }
    /// Backend composition changes only the execution wrapper. The immutable
    /// model, assignment validation and source correspondence stay unchanged.
    pub fn map<T: 'static>(self, map: impl Fn(E) -> T + 'static) -> Export<T> {
        struct Mapped<E, T, F> {
            inner: Box<dyn Reconstruction<E>>,
            map: F,
            output: std::marker::PhantomData<fn() -> T>,
        }
        impl<E: 'static, T: 'static, F: Fn(E) -> T + 'static> Reconstruction<T> for Mapped<E, T, F> {
            fn reconstruct(&self, witness: &FeasibleSolution, lower_bound: u64) -> Result<Reconstructed<T>, String> {
                let selected = self.inner.reconstruct(witness, lower_bound)?;
                Ok(Reconstructed { execution: (self.map)(selected.execution), source: selected.source,
                    objective: selected.objective, decisions: selected.decisions })
            }
        }
        Export { model: self.model, reconstruction: Box::new(Mapped {
            inner: self.reconstruction, map, output: std::marker::PhantomData,
        }) }
    }
}
