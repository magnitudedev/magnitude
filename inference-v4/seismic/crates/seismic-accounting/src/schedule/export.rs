//! Mathematical contributions to one caller-owned model. This module never
//! creates Search, fixes an instruction order, or solves a child schedule.
use super::{evaluation, independent::Fragment, structured, symbolic::Encoding};
use crate::objective::Objective;
use magnitude_solver::model::{Literal, ModelBuilder, Obligation, ObligationKind, VarId};
use std::sync::Arc;

/// Complete finite horizon for fixed source-derived execution constraints.
/// Removing idle gaps never increases a reservation or lifetime, so the sum of
/// operation latencies covers a feasible schedule whenever one exists. This is
/// not a feasibility claim or a schedule seed. Repeat counts are multiplied
/// mathematically without expanding dynamic occurrences.
pub fn horizon(model: &evaluation::Model) -> Result<i64, String> {
    fn node(value: &structured::Node) -> Result<u64, String> {
        match value {
            structured::Node::Operation(operation) => Ok(operation.latency),
            structured::Node::Compose { children, .. } => {
                children.iter().try_fold(0u64, |sum, child| {
                    sum.checked_add(node(child)?)
                        .ok_or_else(|| "finite scheduling horizon exceeds u64".into())
                })
            }
            structured::Node::Repeat { count, body, .. } => node(body)?
                .checked_mul(*count)
                .ok_or_else(|| "finite repeated scheduling horizon exceeds u64".into()),
            structured::Node::Scope { body, .. } => node(body),
        }
    }
    let horizon = match model {
        evaluation::Model::Flat(model) => {
            model.operations.iter().try_fold(0u64, |sum, operation| {
                sum.checked_add(operation.latency)
                    .ok_or_else(|| "finite scheduling horizon exceeds u64".to_string())
            })?
        }
        evaluation::Model::Structured { model, .. } => node(&model.root)?,
    };
    i64::try_from(horizon).map_err(|_| {
        "finite scheduling horizon exceeds the common model's exact i64 time domain".into()
    })
}

/// The original computation remains owned even when the shared vocabulary
/// cannot yet express its complete scheduling relation.
pub struct Binding {
    original: Arc<evaluation::Model>,
    presence: Option<VarId>,
    fragment: Option<Fragment>,
    obligations: Vec<Obligation>,
}
impl Binding {
    /// Append to the enclosing resource boundary. All overlapping fragments
    /// must share `encoding`; only its owner calls `finish` and installs a cost.
    /// `presence` is the complete enclosing activation condition.
    pub fn append(
        builder: &mut ModelBuilder,
        encoding: &mut Encoding,
        original: Arc<evaluation::Model>,
        presence: Option<VarId>,
    ) -> Result<Self, String> {
        let (resources, unmapped) = match original.as_ref() {
            evaluation::Model::Flat(model) => {
                model.validate()?;
                (&model.resources, &model.unmapped)
            }
            evaluation::Model::Structured { model, .. } => {
                model.lower_bound()?;
                (&model.resources, &model.unmapped)
            }
        };
        if resources != encoding.resources() {
            return Err(
                "schedule export resources differ from the enclosing resource scope".into(),
            );
        }
        encoding
            .bind_timebase(original.timebase())
            .map_err(|e| e.to_string())?;
        let mut obligations = Vec::new();
        for reason in unmapped {
            obligations.push(Obligation {
                kind: ObligationKind::Analysis,
                reason: reason.clone(),
            });
        }
        let flat = if !obligations.is_empty() {
            None
        } else {
            match original.as_ref() {
                evaluation::Model::Flat(model) => Some(Arc::new(model.clone())),
                evaluation::Model::Structured {
                    model,
                    expansion_limit,
                } => match model.expand(*expansion_limit) {
                    Ok(model) => Some(Arc::new(model)),
                    Err(crate::workload::DerivationError::Exhausted(_)) => {
                        obligations.push(Obligation {
                            kind: ObligationKind::Analysis,
                            reason: format!("{}: compact serial/parallel occurrence scheduling exceeds finite expansion capacity {expansion_limit}; the common Model needs a typed repetition relation preserving per-occurrence starts, offsets, shared reservations, scopes, recurrence edges and witness occurrence access", model.identity),
                        });
                        None
                    }
                    Err(error) => return Err(error.to_string()),
                },
            }
        };
        let flat = match flat {
            Some(model)
                if model
                    .operations
                    .iter()
                    .any(|op| i64::try_from(op.latency).is_err()) =>
            {
                obligations.push(Obligation {
                    kind: ObligationKind::Analysis,
                    reason: "operation latency exceeds the shared model's exact i64 time domain"
                        .into(),
                });
                None
            }
            other => other,
        };
        for obligation in &obligations {
            builder.obligation(
                presence.into_iter().map(|p| Literal::new(p, 1)).collect(),
                obligation.kind,
                obligation.reason.clone(),
            );
        }
        let fragment = flat
            .map(|model| {
                Fragment::append(builder, encoding, model, presence).map_err(|e| e.to_string())
            })
            .transpose()?;
        Ok(Self {
            original,
            presence,
            fragment,
            obligations,
        })
    }

    pub fn original(&self) -> &Arc<evaluation::Model> {
        &self.original
    }
    pub fn fragment(&self) -> Option<&Fragment> {
        self.fragment.as_ref()
    }
    pub fn obligations(&self) -> &[Obligation] {
        &self.obligations
    }

    /// The enclosing caller first validates the assignment against its complete
    /// immutable model. This second check establishes original IR correspondence.
    pub fn reconstruct(
        &self,
        values: &[i64],
        global_lower_bound: u64,
    ) -> Result<Option<Objective>, String> {
        if let Some(presence) = self.presence {
            match values.get(presence.0) {
                Some(0) => return Ok(None),
                Some(1) => {}
                _ => return Err("missing or nonboolean schedule presence".into()),
            }
        }
        let fragment = self
            .fragment
            .as_ref()
            .ok_or("schedule construction remains unresolved")?;
        let schedule = fragment
            .reconstruct(values)
            .map_err(|e| e.to_string())?
            .ok_or("active schedule fragment reconstructed as absent")?;
        let objective = match self.original.as_ref() {
            evaluation::Model::Flat(_) => {
                Objective::from_flat(fragment.model().clone(), schedule, global_lower_bound)?
            }
            evaluation::Model::Structured { model, .. } => {
                let witness =
                    structured::Witness::from_schedule(Arc::new(model.clone()), schedule)?;
                Objective::from_structured(witness, global_lower_bound)?
            }
        };
        Ok(Some(objective))
    }
}
