//! Numerical acceptance consumes construction-owned applicability.
use super::{CompiledCandidate, NumericalAcceptance, RetainedCandidate};
use crate::numerics::{NumericalAssessment, NumericalRegion};
use seismic_lang::expr::{compiled::InvocationValues, ExprArena};
use seismic_lang::precision::PrecisionPolicy;
use seismic_native_target::TargetFamily;
use std::sync::Arc;

impl<T: TargetFamily> CompiledCandidate<T> {
    pub(crate) fn accept_numerically(
        &self,
        arena: &mut ExprArena,
        policy: &PrecisionPolicy,
    ) -> Option<RetainedCandidate<T>> {
        let basis = self
            .implementation
            .numerical_applicability()
            .basis(policy)?;
        let guard = arena.compile_bool_with(self.safety, &self.fixed);
        if guard.reads().is_empty()
            && !guard
                .evaluate(&InvocationValues::new())
                .expect("closed candidate guard")
        {
            return None;
        }
        Some(RetainedCandidate {
            body: self.body.clone(),
            native: self.native.clone(),
            admission: Arc::new(NumericalAcceptance {
                assessment: NumericalAssessment {
                    regions: vec![NumericalRegion {
                        scope: guard.clone(),
                        basis,
                    }],
                },
                guard,
            }),
        })
    }
}
