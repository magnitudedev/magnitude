//! Semantic oracle over a monomorphized [`LogicalEntry`]. Deterministic
//! programs return one value; unordered floating associations return an
//! explicit allowed-outcome relation plus one non-authoritative witness.
//!
//! This is not a second compilation or production path. It executes the
//! sealed reference candidate of each semantic family in source order and is
//! used only by differential validation. It consumes no SIR, checker symbols,
//! ABI layouts, schedules, or backend code.

mod oracle;
mod scalar;
mod tensor;
pub mod value;

pub use tensor::{round_to, TensorData};
pub use value::Value;

use crate::entry::{AssociationOutcome, LogicalEntry, NumericalRole, SemanticNodeView};
use crate::ids::NodeId;
use crate::types::DType;

/// One flattened invocation argument, in [`crate::entry::CallSchema`] order.
#[derive(Clone, Debug)]
pub enum Arg {
    Tensor(usize),
    Scalar(DType, f64),
    Index(i64),
    Range(i64, i64),
}

/// A reference execution of exactly one monomorphized entry.
pub struct Interpreter<'a> {
    entry: &'a LogicalEntry,
    pub tensors: Vec<TensorData>,
}

#[derive(Clone, Debug)]
pub enum OracleOutcome {
    Deterministic(Vec<Value>),
    Allowed {
        /// One witness produced by the reference evaluator. It is not the
        /// definition of unordered execution.
        representative: Vec<Value>,
        relation: AllowedOutcomeRelation,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowedOutcomeRelation {
    associations: Vec<AllowedAssociation>,
}

impl AllowedOutcomeRelation {
    pub fn associations(&self) -> &[AllowedAssociation] {
        &self.associations
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowedAssociation {
    pub node: NodeId,
    pub outcome: AssociationOutcome,
}

impl<'a> Interpreter<'a> {
    pub fn new(entry: &'a LogicalEntry) -> Self {
        Self {
            entry,
            tensors: Vec::new(),
        }
    }

    pub fn add_tensor(&mut self, tensor: TensorData) -> usize {
        self.tensors.push(tensor);
        self.tensors.len() - 1
    }

    /// Executes the root family's sealed numerical reference and returns its
    /// flattened result leaves in call-schema order.
    pub fn run(&mut self, arguments: &[Arg]) -> Result<OracleOutcome, String> {
        let representative = self.run_reference(arguments)?;
        let root = self.entry.program().root();
        let reference = self
            .entry
            .program()
            .family(root)
            .candidates()
            .iter()
            .find(|candidate| candidate.numerical == NumericalRole::Reference)
            .unwrap_or_else(|| unreachable!("checked family has no reference candidate"));
        let function = self.entry.program().function(reference.function);
        let mut associations = Vec::new();
        collect_allowed_associations(function, function.root(), &mut associations);
        if associations.is_empty() {
            Ok(OracleOutcome::Deterministic(representative))
        } else {
            Ok(OracleOutcome::Allowed {
                representative,
                relation: AllowedOutcomeRelation { associations },
            })
        }
    }
}

fn collect_allowed_associations(
    function: &crate::entry::SemanticFunction,
    region: crate::ids::RegionId,
    output: &mut Vec<AllowedAssociation>,
) {
    for (node_id, node) in function.nodes(region) {
        for event in node.events() {
            if let crate::entry::NumericalOutcome::AllowedAssociation(outcome) =
                event.numerical_outcome()
            {
                output.push(AllowedAssociation {
                    node: node_id,
                    outcome: *outcome,
                });
            }
        }
        match node.view() {
            SemanticNodeView::Reduce {
                op,
                unordered: true,
                input,
                ..
            } => {
                let input_dtype = match &function.value(input).ty {
                    crate::entry::SemanticType::Tensor(tensor) => {
                        crate::registry::representation_info(tensor.representation).decoded
                    }
                    crate::entry::SemanticType::Scalar(dtype) => *dtype,
                    crate::entry::SemanticType::Index { .. }
                    | crate::entry::SemanticType::Range { .. }
                    | crate::entry::SemanticType::Tuple(_)
                    | crate::entry::SemanticType::Opaque { .. }
                    | crate::entry::SemanticType::Void => {
                        unreachable!("checked reduction input is not numeric")
                    }
                };
                let accumulator = crate::intrinsics::accumulator_dtype(op, input_dtype);
                if accumulator.is_float() {
                    output.push(AllowedAssociation {
                        node: node_id,
                        outcome: AssociationOutcome::Reassociated { accumulator },
                    });
                }
            }
            SemanticNodeView::If {
                then, otherwise, ..
            } => {
                collect_allowed_associations(function, then, output);
                collect_allowed_associations(function, otherwise, output);
            }
            SemanticNodeView::Loop { body, .. } => {
                collect_allowed_associations(function, body, output);
            }
            SemanticNodeView::Primitive { .. }
            | SemanticNodeView::Intrinsic { .. }
            | SemanticNodeView::Elementwise { .. }
            | SemanticNodeView::Reduce {
                unordered: false, ..
            }
            | SemanticNodeView::Call { .. }
            | SemanticNodeView::Alloc { .. }
            | SemanticNodeView::Fill { .. }
            | SemanticNodeView::Copy { .. }
            | SemanticNodeView::RepresentationConvert { .. }
            | SemanticNodeView::View { .. }
            | SemanticNodeView::ElementRead { .. }
            | SemanticNodeView::ElementWrite { .. }
            | SemanticNodeView::Store { .. }
            | SemanticNodeView::Atomic { .. }
            | SemanticNodeView::Check { .. }
            | SemanticNodeView::TuplePack { .. }
            | SemanticNodeView::TupleGet { .. }
            | SemanticNodeView::Extent { .. } => {}
        }
    }
}
