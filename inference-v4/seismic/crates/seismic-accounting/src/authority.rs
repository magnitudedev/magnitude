//! What a resource model is allowed to establish. Model feasibility and native
//! correspondence are different claims; neither names nor validation reports
//! convert a relaxation into an executable implementation.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRelationship {
    /// Optimistic resource/dependency constraints can establish a lower bound.
    /// A feasible schedule in this model is not a feasible execution witness.
    OptimisticRelaxation,
    /// Feasible schedules are upper witnesses only inside the stated model.
    ConditionalExecution { native: NativeRelationship },
}
impl ModelRelationship {
    pub const fn hypothetical_execution() -> Self {
        Self::ConditionalExecution {
            native: NativeRelationship::Unestablished,
        }
    }
    pub const fn allows_feasible_upper(&self) -> bool {
        matches!(self, Self::ConditionalExecution { .. })
    }
    pub fn require_feasible_upper(&self) -> Result<(), String> {
        if self.allows_feasible_upper() {
            Ok(())
        } else {
            Err("an optimistic relaxation cannot supply a feasible execution upper bound".into())
        }
    }
}

/// Native compilation has not been shown to preserve the modeled execution.
/// This explicit limitation is not a second implementation or a proof contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeRelationship {
    Unestablished,
}
