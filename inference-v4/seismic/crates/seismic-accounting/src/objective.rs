//! Interpretation of a reconstructed execution witness. Global solver proof
//! status belongs to the selection session, never to a child schedule.
use crate::schedule::{self, structured};
use std::sync::Arc;

/// Bounds on the complete request's objective, in its declared time unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    lower: u64,
    upper: u64,
}
impl Cost {
    pub fn lower(self) -> u64 {
        self.lower
    }
    pub fn upper(self) -> u64 {
        self.upper
    }
    pub fn is_exact(self) -> bool {
        self.lower == self.upper
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Witness {
    Flat {
        model: Arc<schedule::Model>,
        schedule: schedule::Schedule,
    },
    Structured(structured::Witness),
}

/// Checked original-model correspondence plus the enclosing search's bound.
/// Constructing this value establishes feasibility, not executable selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Objective {
    witness: Witness,
    cost: Cost,
}
impl Objective {
    /// `lower_bound` must be the bound returned by the enclosing shared Search.
    /// A caller without such a bound can supply zero. No local optimality is
    /// inferred from an assignment, even when it came from a global optimum.
    pub fn from_flat(
        model: Arc<schedule::Model>,
        schedule: schedule::Schedule,
        lower_bound: u64,
    ) -> Result<Self, String> {
        model.check_execution_upper(&schedule)?;
        let cost = checked_cost(lower_bound, schedule.completion)?;
        Ok(Self {
            witness: Witness::Flat { model, schedule },
            cost,
        })
    }

    pub fn from_structured(witness: structured::Witness, lower_bound: u64) -> Result<Self, String> {
        witness.check_execution_upper()?;
        let cost = checked_cost(lower_bound, witness.completion())?;
        Ok(Self {
            witness: Witness::Structured(witness),
            cost,
        })
    }

    pub fn flat(&self) -> Result<(&schedule::Model, &schedule::Schedule), String> {
        match &self.witness {
            Witness::Flat { model, schedule } => Ok((model, schedule)),
            Witness::Structured(_) => Err("objective retains a structured schedule".into()),
        }
    }
    pub fn structured(&self) -> Option<&structured::Witness> {
        match &self.witness {
            Witness::Structured(witness) => Some(witness),
            _ => None,
        }
    }
    pub fn cost(&self) -> Cost {
        self.cost
    }
    pub fn timebase(&self) -> &schedule::Timebase {
        match &self.witness {
            Witness::Flat { model, .. } => &model.timebase,
            Witness::Structured(witness) => &witness.model().timebase,
        }
    }
    pub fn check_execution_upper(&self) -> Result<(), String> {
        let completion = match &self.witness {
            Witness::Flat { model, schedule } => {
                model.check_execution_upper(schedule)?;
                schedule.completion
            }
            Witness::Structured(witness) => {
                witness.check_execution_upper()?;
                witness.completion()
            }
        };
        if completion != self.cost.upper {
            return Err("objective differs from execution completion".into());
        }
        checked_cost(self.cost.lower, completion).map(|_| ())
    }
}
fn checked_cost(lower: u64, upper: u64) -> Result<Cost, String> {
    if lower > upper {
        return Err("global objective bound exceeds reconstructed completion".into());
    }
    Ok(Cost { lower, upper })
}
