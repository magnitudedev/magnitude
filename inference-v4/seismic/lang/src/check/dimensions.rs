//! The one construction of "how every family dimension is obtained from
//! observed input axes" (L16). A family's plan is built once from its
//! contract signature; every call applies it to the caller's actual axes,
//! and entry construction replays it from the evaluated arguments.

use super::resolve::{SigParam, SignatureDimension};
use crate::expr::{AnyExpr, ExprArena, IntExpr, NodeView, SymbolId};
use crate::types::ValueType;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// One tensor axis of one (possibly tuple-nested) parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedAxis {
    pub parameter: u32,
    pub path: Vec<u32>,
    pub axis: u32,
}

/// A value an inverse operation combines with.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Known {
    Observation(ObservedAxis),
    /// An earlier solved family dimension.
    Dimension(u32),
    Constant(i64),
    /// An expression over earlier solved dimensions, in the contract arena.
    Expression(IntExpr),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InverseOp {
    Add(Known),
    Subtract(Known),
    ReverseSubtract(Known),
    DivideExact(Known),
}

/// Solve `dimension` from `observation` by the inverse operations, in order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DimensionStep {
    pub dimension: u32,
    pub observation: ObservedAxis,
    pub operations: Vec<InverseOp>,
}

/// The family's dimension plan over its contract's arena.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DimensionPlan {
    /// The contract's dimension symbols, in declaration order.
    dimensions: Vec<SymbolId>,
    steps: Vec<DimensionStep>,
}

/// Dimensions no input tensor axis determines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DimensionPlanError {
    pub underdetermined: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DimensionCallError {
    /// An exact division of the plan is not provably exact at this call.
    InexactDivision { dimension: u32 },
}

impl DimensionPlan {
    pub(crate) fn steps(&self) -> &[DimensionStep] {
        &self.steps
    }
}

/// Every tensor axis of `ty` under parameter `parameter`, with its tuple path.
pub(crate) fn tensor_axes(parameter: u32, ty: &ValueType) -> Vec<(ObservedAxis, IntExpr)> {
    fn walk(
        parameter: u32,
        ty: &ValueType,
        path: &mut Vec<u32>,
        output: &mut Vec<(ObservedAxis, IntExpr)>,
    ) {
        match ty {
            ValueType::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(u32::try_from(index).expect("tuple has more than u32::MAX items"));
                    walk(parameter, item, path, output);
                    path.pop();
                }
            }
            ValueType::Tensor(tensor) => {
                for (axis, extent) in tensor.axes.iter().enumerate() {
                    output.push((
                        ObservedAxis {
                            parameter,
                            path: path.clone(),
                            axis: u32::try_from(axis).expect("tensor rank exceeds u32::MAX"),
                        },
                        *extent,
                    ));
                }
            }
            _ => {}
        }
    }
    let mut output = Vec::new();
    walk(parameter, ty, &mut Vec::new(), &mut output);
    output
}

/// The plan of a signature whose dimensions and parameter types live in
/// `arena`: `dimension_inference_order` over every input tensor axis.
pub(crate) fn plan_dimensions(
    arena: &ExprArena,
    dimensions: &[SignatureDimension],
    params: &[SigParam],
) -> Result<DimensionPlan, DimensionPlanError> {
    let observations = params
        .iter()
        .enumerate()
        .flat_map(|(ordinal, parameter)| {
            tensor_axes(
                u32::try_from(ordinal).expect("signature has more than u32::MAX parameters"),
                &parameter.ty,
            )
        })
        .collect::<Vec<_>>();
    let specs = dimensions
        .iter()
        .map(|dimension| (dimension.name.as_str(), dimension.symbol, dimension.admits_zero))
        .collect::<Vec<_>>();
    let nodes = observations
        .iter()
        .map(|(_, extent)| AnyExpr::Int(*extent))
        .collect::<Vec<_>>();
    let order = dimension_inference_order(arena, &specs, &nodes)
        .map_err(|underdetermined| DimensionPlanError { underdetermined })?;
    let ordinal = |symbol: SymbolId| {
        dimensions
            .iter()
            .position(|dimension| dimension.symbol == symbol)
            .expect("plan step names a signature dimension")
    };
    let known = |known: InferenceKnown| match known {
        InferenceKnown::Observation(index) => Known::Observation(observations[index].0.clone()),
        InferenceKnown::Expression(AnyExpr::Int(expression)) => {
            match (super::prove::constant(arena, expression), arena.view(AnyExpr::Int(expression))) {
                (Some(value), _) => Known::Constant(value),
                (None, NodeView::Symbol(symbol)) => Known::Dimension(
                    u32::try_from(ordinal(symbol)).expect("signature has more than u32::MAX dimensions"),
                ),
                (None, _) => Known::Expression(expression),
            }
        }
        InferenceKnown::Expression(_) => {
            unreachable!("signature axes are integer expressions")
        }
    };
    let steps = order
        .into_iter()
        .map(|(dimension, observation, operations)| DimensionStep {
            dimension: u32::try_from(ordinal(dimension))
                .expect("signature has more than u32::MAX dimensions"),
            observation: observations[observation].0.clone(),
            operations: operations
                .into_iter()
                .map(|operation| match operation {
                    InferenceOp::Add(k) => InverseOp::Add(known(k)),
                    InferenceOp::Subtract(k) => InverseOp::Subtract(known(k)),
                    InferenceOp::ReverseSubtract(k) => InverseOp::ReverseSubtract(known(k)),
                    InferenceOp::DivideExact(k) => InverseOp::DivideExact(known(k)),
                })
                .collect(),
        })
        .collect();
    Ok(DimensionPlan {
        dimensions: dimensions.iter().map(|dimension| dimension.symbol).collect(),
        steps,
    })
}

/// The family dimensions at one call, in the caller's arena. `actual`
/// returns the caller's axis for an observed formal axis; a seed replaces
/// its dimension's step.
pub(crate) fn apply_at_call(
    plan: &DimensionPlan,
    family: &ExprArena,
    caller: &mut ExprArena,
    actual: &dyn Fn(&ObservedAxis) -> IntExpr,
    seeds: &[(u32, IntExpr)],
) -> Result<Vec<IntExpr>, DimensionCallError> {
    let mut solved: Vec<Option<IntExpr>> = vec![None; plan.dimensions.len()];
    for (dimension, value) in seeds {
        solved[*dimension as usize] = Some(*value);
    }
    for step in &plan.steps {
        if solved[step.dimension as usize].is_some() {
            continue;
        }
        let mut value = actual(&step.observation);
        for operation in &step.operations {
            let (InverseOp::Add(known)
            | InverseOp::Subtract(known)
            | InverseOp::ReverseSubtract(known)
            | InverseOp::DivideExact(known)) = operation;
            let known = match known {
                Known::Observation(axis) => actual(axis),
                Known::Dimension(dimension) => solved[*dimension as usize]
                    .expect("plan uses a dimension before its step"),
                Known::Constant(constant) => caller.int(*constant),
                Known::Expression(expression) => {
                    let mut map = |symbol: SymbolId, _: &mut ExprArena| {
                        let ordinal = plan
                            .dimensions
                            .iter()
                            .position(|dimension| *dimension == symbol)
                            .expect("plan expression mentions a non-dimension symbol");
                        AnyExpr::Int(
                            solved[ordinal].expect("plan uses a dimension before its step"),
                        )
                    };
                    super::xfer::transfer_int(family, *expression, caller, &mut map)
                }
            };
            value = match operation {
                InverseOp::Add(_) => caller.int_add(value, known),
                InverseOp::Subtract(_) => caller.int_sub(value, known),
                InverseOp::ReverseSubtract(_) => caller.int_sub(known, value),
                InverseOp::DivideExact(_) => super::prove::divide_exact(caller, value, known)
                    .ok_or(DimensionCallError::InexactDivision {
                        dimension: step.dimension,
                    })?,
            };
        }
        solved[step.dimension as usize] = Some(value);
    }
    Ok(solved
        .into_iter()
        .map(|value| value.expect("the plan solves every family dimension"))
        .collect())
}

pub(super) fn dimension_inference_order(
    arena: &ExprArena,
    dimensions: &[(&str, SymbolId, bool)],
    observations: &[AnyExpr],
) -> Result<Vec<(SymbolId, usize, Vec<InferenceOp>)>, Vec<String>> {
    // `Err` names every dimension no remaining observation determines.
    let observation_keys = observations
        .iter()
        .copied()
        .map(|observation| CanonicalObservationKey::new(arena, observation))
        .collect::<Vec<_>>();
    let mut unresolved = dimensions
        .iter()
        .map(|(_, symbol, _)| *symbol)
        .collect::<BTreeSet<_>>();
    let guaranteed_nonzero = dimensions
        .iter()
        .filter_map(|(_, symbol, admits_zero)| (!admits_zero).then_some(*symbol))
        .collect::<BTreeSet<_>>();
    let mut steps = Vec::with_capacity(dimensions.len());
    while !unresolved.is_empty() {
        let selected = dimensions.iter().find_map(|(_, dimension, _)| {
            if !unresolved.contains(dimension) {
                return None;
            }
            observations
                .iter()
                .enumerate()
                .find_map(|(observation, axis)| {
                    inverse_operations(
                        arena,
                        *axis,
                        *dimension,
                        &unresolved,
                        &guaranteed_nonzero,
                        observations,
                        &observation_keys,
                        observation,
                    )
                    .map(|operations| (*dimension, observation, operations))
                })
        });
        let Some((dimension, observation, operations)) = selected else {
            return Err(dimensions
                .iter()
                .filter(|(_, symbol, _)| unresolved.contains(symbol))
                .map(|(name, _, _)| (*name).to_owned())
                .collect());
        };
        unresolved.remove(&dimension);
        steps.push((dimension, observation, operations));
    }
    Ok(steps)
}

#[derive(Clone, Copy, Debug)]
pub(super) enum InferenceKnown {
    Observation(usize),
    Expression(AnyExpr),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum InferenceOp {
    Add(InferenceKnown),
    Subtract(InferenceKnown),
    DivideExact(InferenceKnown),
    ReverseSubtract(InferenceKnown),
}

/// DR1's private equality domain for tensor-axis observations.  This is
/// deliberately narrower than the proof normal form: it canonicalizes only
/// the equivalences the call ABI promises (associativity/commutativity of
/// addition and multiplication, the binary/product spelling of
/// multiplication, and value-preserving integer/natural shape wrappers on
/// their defined domain). In
/// particular, it never distributes products over sums or reorders a
/// subtraction/division.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalObservationKey {
    digest: [u8; 32],
    form: CanonicalObservationForm,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CanonicalObservationForm {
    Integer(i128),
    Dimension(SymbolId),
    Add(Vec<CanonicalObservationKey>),
    Product(Vec<CanonicalObservationKey>),
    ScalarInteger {
        operation: String,
        operands: Vec<(String, CanonicalObservationKey)>,
    },
    Ordered {
        operation: u8,
        operands: Vec<CanonicalObservationKey>,
    },
}

impl CanonicalObservationKey {
    fn new(arena: &ExprArena, expression: AnyExpr) -> Self {
        fn collect_associative(
            arena: &ExprArena,
            expression: AnyExpr,
            operation: crate::expr::BinaryOp,
            output: &mut Vec<CanonicalObservationKey>,
        ) {
            match arena.view(expression) {
                NodeView::Binary { op, lhs, rhs } if op == operation => {
                    collect_associative(arena, lhs, operation, output);
                    collect_associative(arena, rhs, operation, output);
                }
                NodeView::Nary {
                    op: crate::expr::NaryOp::Product,
                    operands,
                } if operation == crate::expr::BinaryOp::Mul => {
                    for operand in operands {
                        collect_associative(arena, *operand, operation, output);
                    }
                }
                _ => output.push(CanonicalObservationKey::new(arena, expression)),
            }
        }

        fn ordered_tag(op: crate::expr::BinaryOp) -> u8 {
            match op {
                crate::expr::BinaryOp::Sub => 0,
                crate::expr::BinaryOp::Div => 1,
                crate::expr::BinaryOp::CeilDiv => 2,
                crate::expr::BinaryOp::Rem => 3,
                crate::expr::BinaryOp::Min => 4,
                crate::expr::BinaryOp::Max => 5,
                crate::expr::BinaryOp::AlignUp => 6,
                crate::expr::BinaryOp::And => 7,
                crate::expr::BinaryOp::Or => 8,
                crate::expr::BinaryOp::Implies => 9,
                crate::expr::BinaryOp::Iff => 10,
                crate::expr::BinaryOp::Add | crate::expr::BinaryOp::Mul => {
                    unreachable!("commutative shape operation has its own canonical form")
                }
            }
        }

        let form = match arena.view(expression) {
            NodeView::NatConst(value) => CanonicalObservationForm::Integer(i128::from(value)),
            NodeView::IntConst(value) => CanonicalObservationForm::Integer(i128::from(value)),
            NodeView::Symbol(symbol) => CanonicalObservationForm::Dimension(symbol),
            NodeView::ScalarInteger {
                operation,
                operands,
            } => CanonicalObservationForm::ScalarInteger {
                operation: format!("{operation:?}"),
                operands: operands
                    .iter()
                    .map(|(dtype, value)| (dtype.name().into(), Self::new(arena, (*value).into())))
                    .collect(),
            },
            NodeView::Unary {
                op: crate::expr::UnaryOp::IntFromNat | crate::expr::UnaryOp::NatFromInt,
                operand,
            } => {
                // This is equality of observed shape values on the admitted
                // domain, not an expression rewrite. Both conversions preserve
                // the integer value whenever defined. The inference plan keeps
                // and validates the original axes after solving all dimensions,
                // including every conversion's definedness requirements.
                return Self::new(arena, operand);
            }
            NodeView::Binary {
                op: crate::expr::BinaryOp::Add | crate::expr::BinaryOp::Mul,
                ..
            }
            | NodeView::Nary {
                op: crate::expr::NaryOp::Product,
                ..
            } => {
                let operation = match arena.view(expression) {
                    NodeView::Binary { op, .. } => op,
                    NodeView::Nary { .. } => crate::expr::BinaryOp::Mul,
                    _ => unreachable!(),
                };
                let mut operands = Vec::new();
                collect_associative(arena, expression, operation, &mut operands);
                operands.sort_by(|left, right| {
                    left.digest
                        .cmp(&right.digest)
                        .then_with(|| left.form.cmp(&right.form))
                });
                match operation {
                    crate::expr::BinaryOp::Add => CanonicalObservationForm::Add(operands),
                    crate::expr::BinaryOp::Mul => CanonicalObservationForm::Product(operands),
                    _ => unreachable!(),
                }
            }
            NodeView::Binary { op, lhs, rhs } => CanonicalObservationForm::Ordered {
                operation: ordered_tag(op),
                operands: vec![Self::new(arena, lhs), Self::new(arena, rhs)],
            },
            NodeView::Unary { op, operand } => CanonicalObservationForm::Ordered {
                operation: match op {
                    crate::expr::UnaryOp::Not => 32,
                    crate::expr::UnaryOp::NatFromInt => 33,
                    crate::expr::UnaryOp::IntFromNat => 34,
                    crate::expr::UnaryOp::IntFromScalar => 35,
                    crate::expr::UnaryOp::ScalarIntegerDefined => 36,
                },
                operands: vec![Self::new(arena, operand)],
            },
            // External tensor axes are checked integer/natural shape
            // expressions.  Reaching another expression category would be a
            // checker/entry-builder invariant breach rather than an
            // ambiguous call schema.
            NodeView::BoolConst(_)
            | NodeView::ScalarConst { .. }
            | NodeView::Nary { .. }
            | NodeView::Select { .. }
            | NodeView::Cmp { .. }
            | NodeView::In { .. }
            | NodeView::Fold { .. }
            | NodeView::Duration(_)
            | NodeView::DurationScale { .. } => {
                unreachable!("call-schema observation is not a checked integer shape expression")
            }
        };
        let digest = canonical_observation_digest(arena, &form);
        Self { digest, form }
    }
}

fn canonical_observation_digest(arena: &ExprArena, form: &CanonicalObservationForm) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"seismic-call-observation-v1");
    match form {
        CanonicalObservationForm::Integer(value) => {
            digest.update([0]);
            digest.update(value.to_le_bytes());
        }
        CanonicalObservationForm::ScalarInteger {
            operation,
            operands,
        } => {
            digest.update([5]);
            digest.update((operation.len() as u64).to_le_bytes());
            digest.update(operation.as_bytes());
            digest.update((operands.len() as u64).to_le_bytes());
            for (dtype, operand) in operands {
                digest.update((dtype.len() as u64).to_le_bytes());
                digest.update(dtype.as_bytes());
                digest.update(operand.digest);
            }
        }
        CanonicalObservationForm::Dimension(symbol) => {
            digest.update([1]);
            match arena.symbol_kind(*symbol) {
                crate::expr::SymbolKind::TemplateDimension(ordinal) => {
                    digest.update([0]);
                    digest.update(ordinal.to_le_bytes());
                }
                crate::expr::SymbolKind::CallDimension(id) => {
                    digest.update([1]);
                    digest.update((id.index() as u64).to_le_bytes());
                }
                _ => unreachable!("call-schema shape contains a non-dimension symbol"),
            }
        }
        CanonicalObservationForm::Add(operands) => {
            digest.update([2]);
            digest.update((operands.len() as u64).to_le_bytes());
            for operand in operands {
                digest.update(operand.digest);
            }
        }
        CanonicalObservationForm::Product(operands) => {
            digest.update([3]);
            digest.update((operands.len() as u64).to_le_bytes());
            for operand in operands {
                digest.update(operand.digest);
            }
        }
        CanonicalObservationForm::Ordered {
            operation,
            operands,
        } => {
            digest.update([4, *operation]);
            digest.update((operands.len() as u64).to_le_bytes());
            for operand in operands {
                digest.update(operand.digest);
            }
        }
    }
    digest.finalize().into()
}

fn inverse_operations(
    arena: &ExprArena,
    expression: AnyExpr,
    dimension: SymbolId,
    unresolved: &BTreeSet<SymbolId>,
    guaranteed_nonzero: &BTreeSet<SymbolId>,
    observations: &[AnyExpr],
    observation_keys: &[CanonicalObservationKey],
    root_observation: usize,
) -> Option<Vec<InferenceOp>> {
    fn contains(arena: &ExprArena, expression: AnyExpr, dimension: SymbolId) -> bool {
        arena.free_symbols(expression).contains(&dimension)
    }

    fn positive(
        arena: &ExprArena,
        expression: AnyExpr,
        guaranteed_nonzero: &BTreeSet<SymbolId>,
    ) -> bool {
        match arena.view(expression) {
            NodeView::NatConst(value) => value > 0,
            NodeView::IntConst(value) => value > 0,
            NodeView::Symbol(symbol) => guaranteed_nonzero.contains(&symbol),
            NodeView::Unary { operand, .. } => positive(arena, operand, guaranteed_nonzero),
            NodeView::Binary {
                op: crate::expr::BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                positive(arena, lhs, guaranteed_nonzero) && positive(arena, rhs, guaranteed_nonzero)
            }
            NodeView::Binary {
                op: crate::expr::BinaryOp::Add,
                lhs,
                rhs,
            } => {
                let left = positive(arena, lhs, guaranteed_nonzero);
                let right = positive(arena, rhs, guaranteed_nonzero);
                match expression {
                    // Natural expressions are nonnegative, so one positive
                    // summand proves the complete sum positive.
                    AnyExpr::Nat(_) => left || right,
                    // Checked integer shape expressions may also contain
                    // subtraction. Requiring both summands positive is the
                    // conservative structural proof that needs no range
                    // assumptions beyond the dimension contract.
                    AnyExpr::Int(_) => left && right,
                    AnyExpr::Bool(_) | AnyExpr::Duration(_) | AnyExpr::Scalar(_) => false,
                }
            }
            NodeView::Nary {
                op: crate::expr::NaryOp::Product,
                operands,
            } => operands
                .iter()
                .all(|operand| positive(arena, *operand, guaranteed_nonzero)),
            _ => false,
        }
    }

    fn known(
        arena: &ExprArena,
        expression: AnyExpr,
        unresolved: &BTreeSet<SymbolId>,
        _observations: &[AnyExpr],
        observation_keys: &[CanonicalObservationKey],
        root_observation: usize,
    ) -> Option<InferenceKnown> {
        let key = CanonicalObservationKey::new(arena, expression);
        if let Some(observation) =
            observation_keys
                .iter()
                .enumerate()
                .find_map(|(observation, candidate)| {
                    (observation != root_observation && *candidate == key).then_some(observation)
                })
        {
            return Some(InferenceKnown::Observation(observation));
        }
        let has_unresolved = arena
            .free_symbols(expression)
            .into_iter()
            .any(|symbol| unresolved.contains(&symbol));
        (!has_unresolved).then_some(InferenceKnown::Expression(expression))
    }

    fn walk(
        arena: &ExprArena,
        expression: AnyExpr,
        dimension: SymbolId,
        unresolved: &BTreeSet<SymbolId>,
        guaranteed_nonzero: &BTreeSet<SymbolId>,
        observations: &[AnyExpr],
        observation_keys: &[CanonicalObservationKey],
        root_observation: usize,
    ) -> Option<Vec<InferenceOp>> {
        match arena.view(expression) {
            NodeView::Symbol(symbol) if symbol == dimension => Some(Vec::new()),
            NodeView::Unary { operand, .. } => walk(
                arena,
                operand,
                dimension,
                unresolved,
                guaranteed_nonzero,
                observations,
                observation_keys,
                root_observation,
            ),
            NodeView::Binary { op, lhs, rhs } => {
                let left_known = known(
                    arena,
                    lhs,
                    unresolved,
                    observations,
                    observation_keys,
                    root_observation,
                );
                let right_known = known(
                    arena,
                    rhs,
                    unresolved,
                    observations,
                    observation_keys,
                    root_observation,
                );
                let left = contains(arena, lhs, dimension) && left_known.is_none();
                let right = contains(arena, rhs, dimension) && right_known.is_none();
                if left == right {
                    return None;
                }
                let (next, operation) = if left {
                    let other = right_known?;
                    let operation = match op {
                        crate::expr::BinaryOp::Add => InferenceOp::Subtract(other),
                        crate::expr::BinaryOp::Sub => InferenceOp::Add(other),
                        crate::expr::BinaryOp::Mul if positive(arena, rhs, guaranteed_nonzero) => {
                            InferenceOp::DivideExact(other)
                        }
                        _ => return None,
                    };
                    (lhs, operation)
                } else {
                    let other = left_known?;
                    let operation = match op {
                        crate::expr::BinaryOp::Add => InferenceOp::Subtract(other),
                        crate::expr::BinaryOp::Sub => InferenceOp::ReverseSubtract(other),
                        crate::expr::BinaryOp::Mul if positive(arena, lhs, guaranteed_nonzero) => {
                            InferenceOp::DivideExact(other)
                        }
                        _ => return None,
                    };
                    (rhs, operation)
                };
                let mut operations = vec![operation];
                operations.extend(walk(
                    arena,
                    next,
                    dimension,
                    unresolved,
                    guaranteed_nonzero,
                    observations,
                    observation_keys,
                    root_observation,
                )?);
                Some(operations)
            }
            NodeView::Nary {
                op: crate::expr::NaryOp::Product,
                operands,
            } => {
                let targets = operands
                    .iter()
                    .enumerate()
                    .filter(|(_, operand)| {
                        contains(arena, **operand, dimension)
                            && known(
                                arena,
                                **operand,
                                unresolved,
                                observations,
                                observation_keys,
                                root_observation,
                            )
                            .is_none()
                    })
                    .map(|(index, operand)| (index, *operand))
                    .collect::<Vec<_>>();
                if targets.len() != 1 {
                    return None;
                }
                let (target_index, target) = targets[0];
                let mut operations = Vec::new();
                for (index, operand) in operands.iter().enumerate() {
                    if index == target_index {
                        continue;
                    }
                    if !positive(arena, *operand, guaranteed_nonzero) {
                        return None;
                    }
                    operations.push(InferenceOp::DivideExact(known(
                        arena,
                        *operand,
                        unresolved,
                        observations,
                        observation_keys,
                        root_observation,
                    )?));
                }
                operations.extend(walk(
                    arena,
                    target,
                    dimension,
                    unresolved,
                    guaranteed_nonzero,
                    observations,
                    observation_keys,
                    root_observation,
                )?);
                Some(operations)
            }
            _ => None,
        }
    }

    walk(
        arena,
        expression,
        dimension,
        unresolved,
        guaranteed_nonzero,
        observations,
        observation_keys,
        root_observation,
    )
}

#[cfg(test)]
mod dimension_inference_tests {
    use super::*;
    use crate::entry::CallSchema;
    use crate::expr::NatExpr;

    fn dimensions<'a>(
        arena: &mut ExprArena,
        names: &'a [&'a str],
        admit_zero: &[bool],
    ) -> (Vec<(&'a str, SymbolId, bool)>, Vec<NatExpr>) {
        let schema = CallSchema::fresh_id();
        let mut specs = Vec::new();
        let mut values = Vec::new();
        for (ordinal, (name, admits_zero)) in names.iter().zip(admit_zero).enumerate() {
            let (symbol, value) = arena.call_dimension(CallSchema::dimension_id(schema, ordinal));
            specs.push((*name, symbol, *admits_zero));
            values.push(value);
        }
        (specs, values)
    }

    #[test]
    fn canonical_observation_key_matches_commuted_associative_products() {
        let mut arena = ExprArena::new();
        let (_, values) = dimensions(&mut arena, &["N", "G", "V"], &[false, false, false]);
        let ng = arena.nat_mul(values[0], values[1]);
        let binary = arena.nat_mul(ng, values[2]);
        let product = arena.nat_product(&[values[2], values[1], values[0]]);

        assert_eq!(
            CanonicalObservationKey::new(&arena, AnyExpr::Nat(binary)),
            CanonicalObservationKey::new(&arena, AnyExpr::Nat(product))
        );
    }

    #[test]
    fn canonical_observation_key_preserves_subtraction_order() {
        let mut arena = ExprArena::new();
        let (_, values) = dimensions(&mut arena, &["N", "G"], &[false, false]);
        let n = arena.int_from_nat(values[0]);
        let g = arena.int_from_nat(values[1]);
        let ng = arena.int_sub(n, g);
        let gn = arena.int_sub(g, n);

        assert_ne!(
            CanonicalObservationKey::new(&arena, AnyExpr::Int(ng)),
            CanonicalObservationKey::new(&arena, AnyExpr::Int(gn))
        );
    }

    #[test]
    fn recurrent_observed_product_system_is_triangular() {
        let mut arena = ExprArena::new();
        let (specs, values) = dimensions(&mut arena, &["NK", "GV"], &[false, false]);
        let p = arena.nat_mul(values[0], values[1]);
        let p_reordered = arena.nat_product(&[values[1], values[0]]);
        let two = arena.nat(2);
        let twice_nk = arena.nat_mul(two, values[0]);
        let q = arena.nat_add(twice_nk, p_reordered);

        let plan = dimension_inference_order(&arena, &specs, &[AnyExpr::Nat(p), AnyExpr::Nat(q)])
            .expect("P = NK*GV, Q = 2*NK + P must be triangular");

        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].0, specs[0].1);
        assert_eq!(plan[0].1, 1);
        assert_eq!(plan[1].0, specs[1].1);
        assert_eq!(plan[1].1, 0);
        assert!(plan[1]
            .2
            .iter()
            .any(|operation| matches!(operation, InferenceOp::DivideExact(_))));
    }

    #[test]
    fn direct_affine_and_chained_dimensions_use_schema_order() {
        let mut arena = ExprArena::new();
        let (specs, values) = dimensions(&mut arena, &["N", "P", "Q"], &[false, false, false]);
        let three = arena.nat(3);
        let two = arena.nat(2);
        let twice_p = arena.nat_mul(two, values[1]);
        let affine_p = arena.nat_add(twice_p, three);
        let chained_q = arena.nat_add(values[2], values[1]);

        let plan = dimension_inference_order(
            &arena,
            &specs,
            &[
                AnyExpr::Nat(values[0]),
                AnyExpr::Nat(affine_p),
                AnyExpr::Nat(chained_q),
            ],
        )
        .expect("direct, affine, and chained dimensions must be triangular");

        assert_eq!(
            plan.iter().map(|step| step.0).collect::<Vec<_>>(),
            specs.iter().map(|spec| spec.1).collect::<Vec<_>>()
        );
        assert_eq!(
            plan.iter().map(|step| step.1).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn ambiguous_product_is_rejected() {
        let mut arena = ExprArena::new();
        let (specs, values) = dimensions(&mut arena, &["N", "G"], &[false, false]);
        let product = arena.nat_mul(values[0], values[1]);
        assert!(dimension_inference_order(&arena, &specs, &[AnyExpr::Nat(product)]).is_err());
    }

    #[test]
    fn possibly_zero_observed_divisor_is_rejected() {
        let mut arena = ExprArena::new();
        let (specs, values) = dimensions(&mut arena, &["N", "G"], &[false, true]);
        let product = arena.nat_mul(values[0], values[1]);
        assert!(dimension_inference_order(
            &arena,
            &specs,
            &[AnyExpr::Nat(values[1]), AnyExpr::Nat(product)],
        )
        .is_err());
    }

    #[test]
    fn positive_natural_sum_can_be_an_exact_divisor() {
        let mut arena = ExprArena::new();
        let (specs, values) = dimensions(
            &mut arena,
            &["G", "KV", "P", "S"],
            &[false, false, false, true],
        );
        let two = arena.nat(2);
        let two_p = arena.nat_mul(two, values[2]);
        let width = arena.nat_add(two_p, values[3]);
        let kv_width = arena.nat_mul(values[1], width);
        let product = arena.nat_product(&[values[1], values[0], width]);

        let plan = dimension_inference_order(
            &arena,
            &specs,
            &[
                AnyExpr::Nat(values[1]),
                AnyExpr::Nat(values[2]),
                AnyExpr::Nat(width),
                AnyExpr::Nat(kv_width),
                AnyExpr::Nat(product),
            ],
        )
        .expect("a positive natural sum is a valid exact divisor");

        assert_eq!(plan.len(), 4);
        assert!(plan
            .iter()
            .find(|step| step.0 == specs[0].1)
            .expect("G step")
            .2
            .iter()
            .any(|operation| matches!(operation, InferenceOp::DivideExact(_))));
    }

    #[test]
    fn positive_checked_integer_sum_can_be_an_exact_divisor() {
        let mut arena = ExprArena::new();
        let (specs, values) = dimensions(
            &mut arena,
            &["G", "KV", "P", "S"],
            &[false, false, false, false],
        );
        let g = arena.int_from_nat(values[0]);
        let kv = arena.int_from_nat(values[1]);
        let p = arena.int_from_nat(values[2]);
        let s = arena.int_from_nat(values[3]);
        let two = arena.int(2);
        let two_p = arena.int_mul(two, p);
        let width = arena.int_add(two_p, s);
        let kv_width = arena.int_mul(kv, width);
        let kv_g = arena.int_mul(kv, g);
        let twice_kv_g = arena.int_mul(kv_g, two);
        let product = arena.int_mul(twice_kv_g, width);

        let plan = dimension_inference_order(
            &arena,
            &specs,
            &[
                AnyExpr::Int(kv),
                AnyExpr::Int(p),
                AnyExpr::Int(width),
                AnyExpr::Int(kv_width),
                AnyExpr::Int(product),
            ],
        )
        .expect("a positive checked integer sum is a valid exact divisor");

        assert_eq!(plan.len(), 4);
        assert!(plan
            .iter()
            .find(|step| step.0 == specs[0].1)
            .expect("G step")
            .2
            .iter()
            .any(|operation| matches!(operation, InferenceOp::DivideExact(_))));
    }
}
