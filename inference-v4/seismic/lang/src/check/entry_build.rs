//! Exact monomorphization of checked templates into one entry-owned semantic
//! program and expression arena.

use super::{ir, xfer};
use crate::checked::{internals::Module, SourceDiagnostic};
use crate::entry::*;
use crate::expr::{
    AnyExpr, BoolExpr, CmpOp, ExprArena, IntExpr, NatExpr, NodeView, SymbolId, SymbolSort,
    TargetPredicate,
};
use crate::ids::*;
use crate::intrinsics::{Constant, PrimitiveId};
use crate::syntax::ast::{AssignOp, BinaryOp};
use crate::types::{DType, Elem, ValueType};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};

fn view_scalar_values(transform: &ViewTransform) -> impl Iterator<Item = SemanticValueId> + '_ {
    let mut values = Vec::new();
    if let ViewTransform::Slice { axes } = transform {
        for axis in axes {
            match axis {
                SliceAxis::Point(ScalarRef::Value(value)) => values.push(*value),
                SliceAxis::Range { start, end } => {
                    if let Some(ScalarRef::Value(value)) = start {
                        values.push(*value);
                    }
                    if let Some(ScalarRef::Value(value)) = end {
                        values.push(*value);
                    }
                }
                SliceAxis::Point(ScalarRef::Static(_)) | SliceAxis::Full => {}
            }
        }
    }
    values.into_iter()
}

pub(crate) fn build_entry(
    module: &Module,
    entry: EntryId,
    bindings: &ElementBindings,
) -> Result<LogicalEntry, SourceDiagnostic> {
    let info = &module.entries[entry.index()];
    let family_ordinal = module.entry_families[entry.index()];
    let source_family = &module.families[family_ordinal];
    let contract = &module.definitions[source_family.contract.index()];
    let element_bindings = validate_elements(module, info, bindings)?;

    let program_id = ProgramId::fresh();
    let schema_id = CallSchema::fresh_id();
    let mut arena = ExprArena::new();
    let mut dimensions = Vec::new();
    let mut shape_values = Vec::new();
    for (ordinal, dimension) in contract.dimensions.iter().enumerate() {
        let id = CallSchema::dimension_id(schema_id, ordinal);
        let (symbol, value) = arena.call_dimension(id);
        dimensions.push(Dimension {
            id,
            name: dimension.name.clone(),
            symbol,
            admits_zero: dimension.admits_zero,
        });
        shape_values.push(arena.int_from_nat(value));
    }

    let mut builder = Builder::new(program_id, arena);
    let root_candidates = family_definitions(source_family)
        .into_iter()
        .map(|definition| CandidateSpec {
            definition,
            shape_args: shape_values.clone(),
            elements: element_bindings.clone(),
            // EntryDomain is exactly the root contract's precondition. Once
            // invocation validates that domain, the sealed portable contract
            // is universally applicable. Alternatives retain their own
            // structural target predicates.
            call_site_proved: definition == source_family.contract,
        })
        .collect();
    let root = instantiate_family(&mut builder, module, family_ordinal, root_candidates, true)?;

    let root_function = builder.families[root.index()]
        .as_ref()
        .and_then(|family| {
            family
                .candidates()
                .iter()
                .find(|candidate| candidate.numerical == NumericalRole::Reference)
        })
        .map(|candidate| candidate.function)
        .unwrap_or_else(|| panic!("checked entry family has no portable reference body"));
    let root_semantic = builder.functions[root_function.index()]
        .as_ref()
        .unwrap_or_else(|| panic!("root function slot was not populated"));

    let (parameters, aliases, domain_terms) = build_schema_parameters(
        &mut builder.arena,
        schema_id,
        contract,
        root_semantic,
        &shape_values,
        &element_bindings,
    );
    let results = build_schema_results(root_semantic);
    let dimension_inference =
        build_dimension_inference_plan(&builder.arena, &dimensions, &parameters)
            .map_err(|message| source_error(module, contract.file, contract.span, message))?;
    let schema = CallSchema::new(
        &builder.arena,
        schema_id,
        dimensions,
        dimension_inference,
        parameters,
        results,
        aliases,
    );
    let mut terms = domain_terms;
    for (dimension, value) in contract.dimensions.iter().zip(&shape_values) {
        if !dimension.admits_zero {
            let one = builder.arena.int(1);
            terms.push(builder.arena.int_cmp(CmpOp::Ge, *value, one));
        }
    }
    terms.extend(predicate_terms(&mut builder.arena, contract, &shape_values));
    let entry_checks = builder
        .entry_checks
        .remove(&root_function)
        .unwrap_or_default();
    terms.extend(entry_checks.into_iter().map(|condition| {
        invocation_condition(&mut builder.arena, root_semantic, &schema, condition).unwrap_or_else(
            || panic!("entry-known safety condition could not be expressed in EntryDomain"),
        )
    }));
    let predicate = builder.arena.all(&terms);
    let totality = builder.arena.side_conditions(AnyExpr::Bool(predicate));
    let predicate = builder.arena.and(predicate, totality);
    let domain = EntryDomain::new(&builder.arena, predicate);

    let families = builder
        .families
        .into_iter()
        .map(|family| family.unwrap_or_else(|| panic!("semantic family slot was not populated")))
        .collect();
    let functions = builder
        .functions
        .into_iter()
        .map(|function| {
            function.unwrap_or_else(|| panic!("semantic function slot was not populated"))
        })
        .collect();
    let program = SemanticProgram::new(crate::entry::internals::Program::new(
        program_id, root, families, functions,
    ));
    Ok(LogicalEntry::new(
        info.stable,
        module.semantic_hash,
        schema,
        domain,
        builder.arena,
        program,
    ))
}

/// Build the one deterministic dimension solve plan for an external call.
/// All input axes are observations. In source order, repeatedly choose the
/// first axis expression containing exactly one unresolved dimension and
/// isolate that dimension through exact inverse operations. Other observed
/// axes may stand in for identical subexpressions, which permits deterministic
/// elimination such as observing both `N*G` and `2*N + N*G`.
fn build_dimension_inference_plan(
    arena: &ExprArena,
    dimensions: &[Dimension],
    parameters: &[Parameter],
) -> Result<DimensionInferencePlan, String> {
    let observations = parameters
        .iter()
        .flat_map(|parameter| match &parameter.kind {
            ParameterKind::Tensor { axes, .. } => axes.iter().copied().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect::<Vec<_>>();
    let dimension_specs = dimensions
        .iter()
        .map(|dimension| {
            (
                dimension.name.as_str(),
                dimension.symbol,
                dimension.admits_zero,
            )
        })
        .collect::<Vec<_>>();
    let observation_nodes = observations
        .iter()
        .copied()
        .map(AnyExpr::Nat)
        .collect::<Vec<_>>();
    let order = dimension_inference_order(arena, &dimension_specs, &observation_nodes)?;
    let steps = order
        .into_iter()
        .map(|(dimension, observation, operations)| {
            let operations = operations
                .into_iter()
                .map(|operation| match operation {
                    InferenceOp::Add(known) => DimensionInferenceOp::Add(final_known(known)),
                    InferenceOp::Subtract(known) => {
                        DimensionInferenceOp::Subtract(final_known(known))
                    }
                    InferenceOp::DivideExact(known) => {
                        DimensionInferenceOp::DivideExact(final_known(known))
                    }
                    InferenceOp::ReverseSubtract(known) => {
                        DimensionInferenceOp::ReverseSubtract(final_known(known))
                    }
                })
                .collect();
            (
                dimension,
                observation,
                observations[observation],
                operations,
            )
        })
        .collect();

    Ok(DimensionInferencePlan::new(observations.len(), steps))
}

fn final_known(known: InferenceKnown) -> DimensionInferenceKnown {
    match known {
        InferenceKnown::Observation(observation) => {
            DimensionInferenceKnown::Observation(observation)
        }
        InferenceKnown::Expression(AnyExpr::Nat(expression)) => {
            DimensionInferenceKnown::Nat(expression)
        }
        InferenceKnown::Expression(AnyExpr::Int(expression)) => {
            DimensionInferenceKnown::Int(expression)
        }
        InferenceKnown::Expression(_) => {
            panic!("dimension inference produced a non-integer known expression")
        }
    }
}

/// Check external-call dimension closure before a `CheckedModule` is minted.
/// All portable families are entries: the language has no second visibility
/// category, so an underdetermined portable signature is a source error even
/// when another function also calls it internally.
pub(super) fn validate_external_dimension_inference(
    definition: &ir::Definition,
) -> Result<(), String> {
    fn tensor_axes(ty: &ValueType, output: &mut Vec<AnyExpr>) {
        match ty {
            ValueType::Tuple(items) => {
                for item in items.iter() {
                    tensor_axes(item, output);
                }
            }
            ValueType::Tensor(tensor) => {
                output.extend(tensor.axes.iter().copied().map(AnyExpr::Int))
            }
            ValueType::Scalar(_)
            | ValueType::Index { .. }
            | ValueType::Range { .. }
            | ValueType::Opaque { .. }
            | ValueType::Void => {}
        }
    }

    let dimensions = definition
        .dimensions
        .iter()
        .map(|dimension| {
            (
                dimension.name.as_str(),
                dimension.symbol,
                dimension.admits_zero,
            )
        })
        .collect::<Vec<_>>();
    let mut observations = Vec::new();
    for parameter in &definition.params {
        tensor_axes(&parameter.ty, &mut observations);
    }
    dimension_inference_order(&definition.arena, &dimensions, &observations).map(|_| ())
}

fn dimension_inference_order(
    arena: &ExprArena,
    dimensions: &[(&str, SymbolId, bool)],
    observations: &[AnyExpr],
) -> Result<Vec<(SymbolId, usize, Vec<InferenceOp>)>, String> {
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
            let names = dimensions
                .iter()
                .filter(|(_, symbol, _)| unresolved.contains(symbol))
                .map(|(name, _, _)| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "external entry dimensions {names} are underdetermined or require a nonlinear/invocation-dependent inversion; every dimension must be uniquely derivable from input tensor extents"
            ));
        };
        unresolved.remove(&dimension);
        steps.push((dimension, observation, operations));
    }
    Ok(steps)
}

#[derive(Clone, Copy, Debug)]
enum InferenceKnown {
    Observation(usize),
    Expression(AnyExpr),
}

#[derive(Clone, Copy, Debug)]
enum InferenceOp {
    Add(InferenceKnown),
    Subtract(InferenceKnown),
    DivideExact(InferenceKnown),
    ReverseSubtract(InferenceKnown),
}

/// DR1's private equality domain for tensor-axis observations.  This is
/// deliberately narrower than the proof normal form: it canonicalizes only
/// the equivalences the call ABI promises (associativity/commutativity of
/// addition and multiplication, the binary/product spelling of
/// multiplication, and lossless `Nat -> Int` shape wrappers).  In
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
            NodeView::Unary {
                op: crate::expr::UnaryOp::IntFromNat,
                operand,
            } if matches!(
                arena.view(operand),
                NodeView::NatConst(_) | NodeView::Symbol(_)
            ) =>
            {
                // `Nat -> Int` is lossless for a nonnegative dimension or
                // constant.  No other conversion is erased.
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

fn validate_elements(
    module: &Module,
    info: &crate::checked::EntryInfo,
    supplied: &ElementBindings,
) -> Result<BTreeMap<String, RepresentationId>, SourceDiagnostic> {
    let expected: BTreeSet<_> = info.element_parameters.iter().cloned().collect();
    let actual: BTreeSet<_> = supplied.iter().map(|(name, _)| name.to_owned()).collect();
    if expected != actual {
        let missing: Vec<_> = expected.difference(&actual).cloned().collect();
        let extra: Vec<_> = actual.difference(&expected).cloned().collect();
        return Err(SourceDiagnostic {
            path: module.sources.files()[0].path.clone(),
            span: crate::span::Span::default(),
            message: format!(
                "element bindings do not match the entry parameters; missing: {missing:?}; extra: {extra:?}"
            ),
        });
    }
    Ok(supplied
        .iter()
        .map(|(name, representation)| (name.to_owned(), representation))
        .collect())
}

fn family_definitions(family: &ir::Family) -> Vec<FunctionId> {
    family
        .bodies
        .iter()
        .chain(&family.lowerings)
        .copied()
        .collect()
}

#[derive(Clone)]
struct CandidateSpec {
    definition: FunctionId,
    shape_args: Vec<IntExpr>,
    elements: BTreeMap<String, RepresentationId>,
    /// Nested calls have already proved their contract precondition in the
    /// caller. Root candidates retain their structural applicability.
    call_site_proved: bool,
}

struct Builder {
    program: ProgramId,
    arena: ExprArena,
    families: Vec<Option<Family>>,
    functions: Vec<Option<SemanticFunction>>,
    entry_checks: BTreeMap<FunctionId, Vec<SemanticValueId>>,
}

impl Builder {
    fn new(program: ProgramId, arena: ExprArena) -> Self {
        Self {
            program,
            arena,
            families: Vec::new(),
            functions: Vec::new(),
            entry_checks: BTreeMap::new(),
        }
    }
    fn reserve_family(&mut self) -> FamilyId {
        let id = FamilyId::new(
            self.program,
            u32::try_from(self.families.len()).expect("entry has more than u32::MAX families"),
        );
        self.families.push(None);
        id
    }
    fn reserve_function(&mut self) -> FunctionId {
        let id = FunctionId::new(
            self.program,
            u32::try_from(self.functions.len()).expect("entry has more than u32::MAX functions"),
        );
        self.functions.push(None);
        id
    }
}

fn instantiate_family(
    builder: &mut Builder,
    module: &Module,
    source_family: usize,
    specs: Vec<CandidateSpec>,
    entry_family: bool,
) -> Result<FamilyId, SourceDiagnostic> {
    let id = builder.reserve_family();
    let family = &module.families[source_family];
    let mut candidates = Vec::new();
    let mut reference_assigned = false;
    for spec in specs {
        let definition = &module.definitions[spec.definition.index()];
        let Some(elements) = specialize_elements(definition, &spec.elements) else {
            continue;
        };
        let applicability = if spec.call_site_proved {
            let yes = builder.arena.bool(true);
            TargetPredicate::new(&builder.arena, yes)
                .unwrap_or_else(|_| panic!("constant applicability was rejected"))
        } else {
            let Some(applicability) = applicability(builder, definition, &spec.shape_args) else {
                // A specialized alternative whose predicate depends on a
                // loop/data value cannot participate in target planning.
                // The portable contract remains available; source may guard
                // and call a separately named helper explicitly.
                continue;
            };
            applicability
        };
        let entry_root = entry_family && spec.definition == family.contract;
        let function = instantiate_function(
            builder,
            module,
            spec.definition.index(),
            spec.shape_args,
            elements,
            entry_root,
        )?;
        let kind = match definition.kind {
            ir::DefKind::Body { target: None } => CandidateKind::Portable,
            ir::DefKind::Body {
                target: Some(backend),
            } => CandidateKind::Helper { backend },
            ir::DefKind::Lower { target } => CandidateKind::Lowering { backend: target },
        };
        let numerical = if spec.definition == family.contract {
            assert!(
                definition.kind.is_portable_body(),
                "checked family contract is not a portable body"
            );
            assert!(
                !reference_assigned,
                "family has more than one reference body"
            );
            reference_assigned = true;
            NumericalRole::Reference
        } else {
            NumericalRole::Alternative
        };
        candidates.push(Candidate {
            function,
            kind,
            requires: definition.requires.clone(),
            applicability,
            numerical,
        });
    }
    if candidates.is_empty() {
        return Err(source_error(
            module,
            module.definitions[family.contract.index()].file,
            module.definitions[family.contract.index()].span,
            "no implementation applies to these element bindings",
        ));
    }
    if !reference_assigned {
        return Err(source_error(
            module,
            module.definitions[family.contract.index()].file,
            module.definitions[family.contract.index()].span,
            "no portable reference implementation applies to these element bindings",
        ));
    }
    builder.families[id.index()] = Some(Family::new(family.name.clone(), candidates));
    Ok(id)
}

fn specialize_elements(
    definition: &ir::Definition,
    inherited: &BTreeMap<String, RepresentationId>,
) -> Option<BTreeMap<String, RepresentationId>> {
    let mut out = inherited.clone();
    for (name, element) in &definition.elem_bindings {
        let concrete = match element {
            Elem::Dtype(dtype) => crate::registry::dense(*dtype),
            Elem::Repr(representation) => *representation,
            Elem::Param(parameter) => *out.get(parameter)?,
        };
        if out.get(name).is_some_and(|existing| *existing != concrete) {
            return None;
        }
        out.insert(name.clone(), concrete);
    }
    Some(out)
}

fn applicability(
    builder: &mut Builder,
    definition: &ir::Definition,
    shapes: &[IntExpr],
) -> Option<TargetPredicate> {
    let mut terms = Vec::new();
    for predicate in &definition.predicates {
        let expression = transfer_template_int(
            &mut builder.arena,
            definition,
            shapes,
            predicate.expression(),
        );
        let zero = builder.arena.int(0);
        terms.push(match predicate {
            ir::Predicate::NonNegative(_) => builder.arena.int_cmp(CmpOp::Ge, expression, zero),
            ir::Predicate::Zero(_) => builder.arena.int_cmp(CmpOp::Eq, expression, zero),
            ir::Predicate::NonZero(_) => builder.arena.int_cmp(CmpOp::Ne, expression, zero),
        });
    }
    let predicate = builder.arena.all(&terms);
    TargetPredicate::new(&builder.arena, predicate).ok()
}

trait PredicateExpr {
    fn expression(&self) -> IntExpr;
}
impl PredicateExpr for ir::Predicate {
    fn expression(&self) -> IntExpr {
        match *self {
            Self::NonNegative(e) | Self::Zero(e) | Self::NonZero(e) => e,
        }
    }
}

fn transfer_template_int(
    arena: &mut ExprArena,
    definition: &ir::Definition,
    shapes: &[IntExpr],
    expression: IntExpr,
) -> IntExpr {
    let mut map = |symbol: SymbolId, _: &mut ExprArena| {
        let ordinal = definition
            .dimensions
            .iter()
            .position(|dimension| dimension.symbol == symbol)
            .unwrap_or_else(|| panic!("template expression mentions a non-dimension symbol"));
        AnyExpr::Int(shapes[ordinal])
    };
    xfer::transfer_int(&definition.arena, expression, arena, &mut map)
}

fn source_error(
    module: &Module,
    file: usize,
    span: crate::span::Span,
    message: impl Into<String>,
) -> SourceDiagnostic {
    SourceDiagnostic {
        path: module.sources.files()[file].path.clone(),
        span,
        message: message.into(),
    }
}

fn build_schema_parameters(
    arena: &mut ExprArena,
    schema: SchemaId,
    definition: &ir::Definition,
    function: &SemanticFunction,
    _shapes: &[IntExpr],
    _elements: &BTreeMap<String, RepresentationId>,
) -> (Vec<Parameter>, Vec<AliasRule>, Vec<BoolExpr>) {
    let mut parameters = Vec::new();
    let mut domain = Vec::new();
    for (ordinal, semantic) in function.parameters().iter().enumerate() {
        let id = CallSchema::parameter_id(schema, ordinal);
        let ty = function.value(semantic.value).ty.clone();
        let kind = match ty {
            SemanticType::Tensor(tensor) => ParameterKind::Tensor {
                access: match semantic.access {
                    ParameterAccess::Owned => TensorAccess::Owned,
                    ParameterAccess::Shared => TensorAccess::Shared,
                    ParameterAccess::Mutable => TensorAccess::Mutable,
                    ParameterAccess::Scalar => {
                        panic!("checked tensor parameter has scalar access")
                    }
                },
                representation: tensor.representation,
                axes: tensor.axes,
            },
            SemanticType::Scalar(dtype) => {
                let symbol = arena.call_scalar(id, scalar_sort(dtype));
                ParameterKind::Scalar { dtype, symbol }
            }
            SemanticType::Index { bound } => {
                let symbol = arena.call_scalar(id, SymbolSort::Int);
                let value = arena.int_symbol(symbol);
                let bound = arena.int_from_nat(bound);
                let zero = arena.int(0);
                domain.push(arena.int_cmp(CmpOp::Ge, value, zero));
                domain.push(arena.int_cmp(CmpOp::Lt, value, bound));
                ParameterKind::Index {
                    bound: arena.nat_from_int(bound),
                    symbol,
                }
            }
            SemanticType::Range { bound } => {
                let start = arena.call_scalar(id, SymbolSort::Int);
                let end = arena.call_scalar(id, SymbolSort::Int);
                let start_value = arena.int_symbol(start);
                let end_value = arena.int_symbol(end);
                let bound_int = arena.int_from_nat(bound);
                let zero = arena.int(0);
                domain.push(arena.int_cmp(CmpOp::Ge, start_value, zero));
                domain.push(arena.int_cmp(CmpOp::Ge, end_value, start_value));
                domain.push(arena.int_cmp(CmpOp::Le, end_value, bound_int));
                ParameterKind::Range { bound, start, end }
            }
            SemanticType::Tuple(_) | SemanticType::Opaque { .. } | SemanticType::Void => {
                panic!("unsupported checked entry parameter escaped signature checking")
            }
        };
        parameters.push(Parameter {
            id,
            source: semantic.source,
            path: semantic.path.clone(),
            name: semantic.name.clone(),
            kind,
            value: semantic.value,
        });
    }
    let aliases = definition
        .aliases
        .iter()
        .flat_map(|(left, right)| {
            let left = function
                .parameters()
                .iter()
                .enumerate()
                .filter_map(move |(ordinal, parameter)| {
                    (parameter.source as usize == *left).then_some(ordinal)
                })
                .collect::<Vec<_>>();
            let right = function
                .parameters()
                .iter()
                .enumerate()
                .filter_map(move |(ordinal, parameter)| {
                    (parameter.source as usize == *right).then_some(ordinal)
                })
                .collect::<Vec<_>>();
            left.into_iter()
                .flat_map(move |left| right.clone().into_iter().map(move |right| (left, right)))
        })
        .map(|(left, right)| AliasRule::MayOverlap(parameters[left].id, parameters[right].id))
        .collect();
    // All tensor pairs not explicitly admitted to alias are disjoint whenever
    // either side is mutable/owned. Shared/shared pairs may overlap.
    let mut aliases: Vec<AliasRule> = aliases;
    for left in 0..parameters.len() {
        for right in left + 1..parameters.len() {
            let left_source = function.parameters()[left].source as usize;
            let right_source = function.parameters()[right].source as usize;
            if definition.aliases.iter().any(|(a, b)| {
                (*a == left_source && *b == right_source)
                    || (*a == right_source && *b == left_source)
            }) {
                continue;
            }
            let tensor =
                |parameter: &Parameter| matches!(parameter.kind, ParameterKind::Tensor { .. });
            let shared = |parameter: &Parameter| {
                matches!(
                    parameter.kind,
                    ParameterKind::Tensor {
                        access: TensorAccess::Shared,
                        ..
                    }
                )
            };
            if tensor(&parameters[left]) && tensor(&parameters[right]) {
                aliases.push(if shared(&parameters[left]) && shared(&parameters[right]) {
                    AliasRule::MayOverlap(parameters[left].id, parameters[right].id)
                } else {
                    AliasRule::Disjoint(parameters[left].id, parameters[right].id)
                });
            }
        }
    }
    (parameters, aliases, domain)
}

fn scalar_sort(dtype: DType) -> SymbolSort {
    match dtype {
        DType::F32 | DType::BF16 | DType::F16 => SymbolSort::Scalar(dtype),
        DType::I32 => SymbolSort::Int,
        DType::U32 => SymbolSort::Nat,
        DType::Bool => SymbolSort::Scalar(DType::Bool),
    }
}

fn build_schema_results(function: &SemanticFunction) -> Vec<ResultLeaf> {
    fn flatten(
        function: &SemanticFunction,
        value: SemanticValueId,
        path: &mut Vec<u32>,
        output: &mut Vec<ResultLeaf>,
    ) {
        match &function.value(value).ty {
            SemanticType::Tuple(items) => {
                // Tuple results are represented by projection nodes in the
                // semantic function. A checked return therefore supplies one
                // result value per leaf; retaining this branch is an invariant
                // guard for malformed internal construction.
                if !items.is_empty() {
                    panic!("tuple result was not flattened by semantic lowering");
                }
            }
            SemanticType::Tensor(tensor) => output.push(ResultLeaf {
                path: path.clone(),
                kind: ResultKind::Tensor {
                    representation: tensor.representation,
                    axes: tensor.axes.clone(),
                },
                value,
            }),
            SemanticType::Scalar(dtype) => output.push(ResultLeaf {
                path: path.clone(),
                kind: ResultKind::Scalar(*dtype),
                value,
            }),
            SemanticType::Index { bound } => output.push(ResultLeaf {
                path: path.clone(),
                kind: ResultKind::Index { bound: *bound },
                value,
            }),
            SemanticType::Range { bound } => output.push(ResultLeaf {
                path: path.clone(),
                kind: ResultKind::Range { bound: *bound },
                value,
            }),
            SemanticType::Void => {}
            SemanticType::Opaque { .. } => {
                panic!("backend-opaque value escaped a portable entry result")
            }
        }
    }
    let mut output = Vec::new();
    for (ordinal, value) in function.results().iter().copied().enumerate() {
        let mut path = if function.results().len() == 1 {
            Vec::new()
        } else {
            vec![u32::try_from(ordinal).expect("result count exceeds u32::MAX")]
        };
        flatten(function, value, &mut path, &mut output);
    }
    output
}

#[derive(Clone, Copy)]
enum InvocationExpr {
    Nat(NatExpr),
    Int(IntExpr),
    Bool(BoolExpr),
}

fn invocation_condition(
    arena: &mut ExprArena,
    function: &SemanticFunction,
    schema: &CallSchema,
    condition: SemanticValueId,
) -> Option<BoolExpr> {
    let mut memo = BTreeMap::new();
    match invocation_value(arena, function, schema, condition, &mut memo)? {
        InvocationExpr::Bool(value) => Some(value),
        InvocationExpr::Nat(_) | InvocationExpr::Int(_) => None,
    }
}

fn invocation_value(
    arena: &mut ExprArena,
    function: &SemanticFunction,
    schema: &CallSchema,
    value: SemanticValueId,
    memo: &mut BTreeMap<SemanticValueId, InvocationExpr>,
) -> Option<InvocationExpr> {
    if let Some(expression) = memo.get(&value) {
        return Some(*expression);
    }
    let expression = match function.value(value).origin {
        ValueOrigin::Parameter => {
            let parameter = schema
                .parameters()
                .iter()
                .find(|parameter| parameter.value == value)?;
            match &parameter.kind {
                ParameterKind::Index { symbol, .. } => {
                    InvocationExpr::Int(arena.int_symbol(*symbol))
                }
                ParameterKind::Scalar {
                    dtype: DType::I32,
                    symbol,
                } => InvocationExpr::Int(arena.int_symbol(*symbol)),
                ParameterKind::Scalar {
                    dtype: DType::U32,
                    symbol,
                } => InvocationExpr::Nat(arena.nat_symbol(*symbol)),
                ParameterKind::Tensor { .. }
                | ParameterKind::Scalar { .. }
                | ParameterKind::Range { .. } => return None,
            }
        }
        ValueOrigin::RegionParameter(_) => return None,
        ValueOrigin::Node(node_id) => {
            let node = function.node(node_id);
            match node.view() {
                SemanticNodeView::Extent {
                    tensor: input,
                    axis,
                    ..
                } => {
                    let SemanticType::Tensor(tensor) = &function.value(input).ty else {
                        return None;
                    };
                    let extent = *tensor.axes.get(axis as usize)?;
                    InvocationExpr::Int(arena.int_from_nat(extent))
                }
                SemanticNodeView::Primitive {
                    primitive, inputs, ..
                } => {
                    let mut inputs = inputs
                        .iter()
                        .map(|input| invocation_value(arena, function, schema, *input, memo))
                        .collect::<Option<Vec<_>>>()?;
                    match primitive {
                        PrimitiveId::Constant(Constant::Int(value)) => {
                            InvocationExpr::Int(arena.int(*value))
                        }
                        PrimitiveId::Constant(Constant::Bool(value)) => {
                            InvocationExpr::Bool(arena.bool(*value))
                        }
                        PrimitiveId::Unary(crate::syntax::ast::UnaryOp::Neg) => {
                            let InvocationExpr::Int(value) = inputs[0] else {
                                return None;
                            };
                            let zero = arena.int(0);
                            InvocationExpr::Int(arena.int_sub(zero, value))
                        }
                        PrimitiveId::Unary(crate::syntax::ast::UnaryOp::Not) => {
                            let InvocationExpr::Bool(value) = inputs[0] else {
                                return None;
                            };
                            InvocationExpr::Bool(arena.not(value))
                        }
                        PrimitiveId::Binary(op) => {
                            invocation_binary(arena, *op, inputs.remove(0), inputs.remove(0))?
                        }
                        PrimitiveId::Cast(DType::I32) => match inputs[0] {
                            InvocationExpr::Nat(value) => {
                                InvocationExpr::Int(arena.int_from_nat(value))
                            }
                            InvocationExpr::Int(value) => InvocationExpr::Int(value),
                            InvocationExpr::Bool(_) => return None,
                        },
                        PrimitiveId::Cast(DType::U32) => match inputs[0] {
                            InvocationExpr::Int(value) => {
                                InvocationExpr::Nat(arena.nat_from_int(value))
                            }
                            InvocationExpr::Nat(value) => InvocationExpr::Nat(value),
                            InvocationExpr::Bool(_) => return None,
                        },
                        PrimitiveId::Select => {
                            let [InvocationExpr::Bool(condition), then, otherwise] =
                                inputs.as_slice()
                            else {
                                return None;
                            };
                            match (*then, *otherwise) {
                                (InvocationExpr::Int(then), InvocationExpr::Int(otherwise)) => {
                                    InvocationExpr::Int(
                                        arena.int_select(*condition, then, otherwise),
                                    )
                                }
                                (InvocationExpr::Nat(then), InvocationExpr::Nat(otherwise)) => {
                                    InvocationExpr::Nat(
                                        arena.nat_select(*condition, then, otherwise),
                                    )
                                }
                                _ => return None,
                            }
                        }
                        PrimitiveId::Constant(Constant::Float(_))
                        | PrimitiveId::TuplePack
                        | PrimitiveId::TupleGet(_)
                        | PrimitiveId::RangeMake
                        | PrimitiveId::RangeStart
                        | PrimitiveId::RangeEnd
                        | PrimitiveId::Unary(_)
                        | PrimitiveId::Cast(_)
                        | PrimitiveId::Math(_)
                        | PrimitiveId::TensorAlloc
                        | PrimitiveId::Fill(_)
                        | PrimitiveId::Materialize
                        | PrimitiveId::Clone
                        | PrimitiveId::Load
                        | PrimitiveId::RepresentationConvert(_)
                        | PrimitiveId::Decode
                        | PrimitiveId::Transpose
                        | PrimitiveId::Reshape
                        | PrimitiveId::SliceView { .. }
                        | PrimitiveId::ElementRead { .. }
                        | PrimitiveId::Extent { .. }
                        | PrimitiveId::Atomic { .. }
                        | PrimitiveId::Reduce { .. }
                        | PrimitiveId::Symbolic(_) => return None,
                    }
                }
                SemanticNodeView::Intrinsic { .. }
                | SemanticNodeView::Elementwise { .. }
                | SemanticNodeView::Reduce { .. }
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
                | SemanticNodeView::If { .. }
                | SemanticNodeView::Loop { .. }
                | SemanticNodeView::Check { .. }
                | SemanticNodeView::TuplePack { .. }
                | SemanticNodeView::TupleGet { .. } => return None,
            }
        }
    };
    memo.insert(value, expression);
    Some(expression)
}

fn invocation_binary(
    arena: &mut ExprArena,
    op: BinaryOp,
    left: InvocationExpr,
    right: InvocationExpr,
) -> Option<InvocationExpr> {
    use InvocationExpr::{Bool, Int, Nat};
    Some(match (left, right) {
        (Int(left), Int(right)) => match op {
            BinaryOp::Add => Int(arena.int_add(left, right)),
            BinaryOp::Sub => Int(arena.int_sub(left, right)),
            BinaryOp::Mul => Int(arena.int_mul(left, right)),
            BinaryOp::Div => Int(arena.int_div(left, right)),
            BinaryOp::Rem => Int(arena.int_rem(left, right)),
            BinaryOp::Eq => Bool(arena.int_cmp(CmpOp::Eq, left, right)),
            BinaryOp::Ne => Bool(arena.int_cmp(CmpOp::Ne, left, right)),
            BinaryOp::Lt => Bool(arena.int_cmp(CmpOp::Lt, left, right)),
            BinaryOp::Le => Bool(arena.int_cmp(CmpOp::Le, left, right)),
            BinaryOp::Gt => Bool(arena.int_cmp(CmpOp::Gt, left, right)),
            BinaryOp::Ge => Bool(arena.int_cmp(CmpOp::Ge, left, right)),
            _ => return None,
        },
        (Nat(left), Nat(right)) => match op {
            BinaryOp::Add => Nat(arena.nat_add(left, right)),
            BinaryOp::Sub => Nat(arena.nat_sub(left, right)),
            BinaryOp::Mul => Nat(arena.nat_mul(left, right)),
            BinaryOp::Div => Nat(arena.nat_div(left, right)),
            BinaryOp::Rem => Nat(arena.nat_rem(left, right)),
            BinaryOp::Eq => Bool(arena.nat_cmp(CmpOp::Eq, left, right)),
            BinaryOp::Ne => Bool(arena.nat_cmp(CmpOp::Ne, left, right)),
            BinaryOp::Lt => Bool(arena.nat_cmp(CmpOp::Lt, left, right)),
            BinaryOp::Le => Bool(arena.nat_cmp(CmpOp::Le, left, right)),
            BinaryOp::Gt => Bool(arena.nat_cmp(CmpOp::Gt, left, right)),
            BinaryOp::Ge => Bool(arena.nat_cmp(CmpOp::Ge, left, right)),
            _ => return None,
        },
        (Bool(left), Bool(right)) => match op {
            BinaryOp::And => Bool(arena.and(left, right)),
            BinaryOp::Or => Bool(arena.or(left, right)),
            BinaryOp::Eq => Bool(arena.iff(left, right)),
            BinaryOp::Ne => {
                let equal = arena.iff(left, right);
                Bool(arena.not(equal))
            }
            _ => return None,
        },
        _ => return None,
    })
}

fn predicate_terms(
    arena: &mut ExprArena,
    definition: &ir::Definition,
    shapes: &[IntExpr],
) -> Vec<BoolExpr> {
    definition
        .predicates
        .iter()
        .map(|predicate| {
            let expression =
                transfer_template_int(arena, definition, shapes, predicate.expression());
            let zero = arena.int(0);
            match predicate {
                ir::Predicate::NonNegative(_) => arena.int_cmp(CmpOp::Ge, expression, zero),
                ir::Predicate::Zero(_) => arena.int_cmp(CmpOp::Eq, expression, zero),
                ir::Predicate::NonZero(_) => arena.int_cmp(CmpOp::Ne, expression, zero),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Checked-template -> entry-owned semantic program
// ---------------------------------------------------------------------------

fn instantiate_function(
    builder: &mut Builder,
    module: &Module,
    source: usize,
    shapes: Vec<IntExpr>,
    elements: BTreeMap<String, RepresentationId>,
    entry_root: bool,
) -> Result<FunctionId, SourceDiagnostic> {
    let definition = &module.definitions[source];
    let id = builder.reserve_function();
    let mut lowering = FunctionLowering::new(
        builder, module, definition, id, shapes, elements, entry_root,
    );
    let function = lowering.lower()?;
    let entry_checks = std::mem::take(&mut lowering.entry_checks);
    drop(lowering);
    if entry_root {
        builder.entry_checks.insert(id, entry_checks);
    }
    builder.functions[id.index()] = Some(function);
    Ok(id)
}

struct RegionWork {
    id: RegionId,
    kind: RegionKind,
    parameters: Vec<SemanticValueId>,
    results: Vec<SemanticValueId>,
    nodes: Vec<SemanticNode>,
}

struct FunctionLowering<'a, 'm> {
    builder: &'a mut Builder,
    module: &'m Module,
    definition: &'m ir::Definition,
    id: FunctionId,
    shapes: Vec<IntExpr>,
    elements: BTreeMap<String, RepresentationId>,
    values: Vec<ValueInfo>,
    regions: Vec<Option<RegionWork>>,
    locals: Vec<Option<SemanticValueId>>,
    symbols: HashMap<SymbolId, IntExpr>,
    runtime_values: HashMap<SemanticValueId, IntExpr>,
    parameters: Vec<FunctionParameter>,
    entry_root: bool,
    entry_checks: Vec<SemanticValueId>,
    parallel_depth: u32,
    parallel_participants: Vec<BinderId>,
    participant_bindings: HashMap<ir::LocalId, BinderId>,
    last_event: HashMap<RegionId, Vec<SemanticEventId>>,
}

impl<'a, 'm> FunctionLowering<'a, 'm> {
    fn new(
        builder: &'a mut Builder,
        module: &'m Module,
        definition: &'m ir::Definition,
        id: FunctionId,
        shapes: Vec<IntExpr>,
        elements: BTreeMap<String, RepresentationId>,
        entry_root: bool,
    ) -> Self {
        let mut symbols = HashMap::new();
        for (dimension, value) in definition.dimensions.iter().zip(&shapes) {
            symbols.insert(dimension.symbol, *value);
        }
        Self {
            builder,
            module,
            definition,
            id,
            shapes,
            elements,
            values: Vec::new(),
            regions: Vec::new(),
            locals: vec![None; definition.body.locals.len()],
            symbols,
            runtime_values: HashMap::new(),
            parameters: Vec::new(),
            entry_root,
            entry_checks: Vec::new(),
            parallel_depth: 0,
            parallel_participants: Vec::new(),
            participant_bindings: HashMap::new(),
            last_event: HashMap::new(),
        }
    }

    fn lower(&mut self) -> Result<SemanticFunction, SourceDiagnostic> {
        let root = self.reserve_region(RegionKind::Root);
        for (source, parameter) in self.definition.params.clone().into_iter().enumerate() {
            let access = match parameter.ownership {
                ir::Ownership::Owned => ParameterAccess::Owned,
                ir::Ownership::Shared => ParameterAccess::Shared,
                ir::Ownership::Exclusive => ParameterAccess::Mutable,
                ir::Ownership::Value => ParameterAccess::Scalar,
            };
            let ty = self.semantic_type(&parameter.ty);
            let value = self.lower_parameter(
                root,
                u32::try_from(source).expect("function has more than u32::MAX parameters"),
                &parameter.name,
                &[],
                ty,
                access,
                parameter.span,
            )?;
            self.locals[parameter.local.index()] = Some(value);
            if let Some(source_symbol) = self.definition.body.locals[parameter.local.index()].symbol
            {
                let runtime = self.runtime_int(value);
                self.symbols.insert(source_symbol, runtime);
            }
        }
        let block = self.definition.body.root.clone();
        let results = self.lower_block(root, &block)?;
        let regions = std::mem::take(&mut self.regions)
            .into_iter()
            .map(|region| {
                let region =
                    region.unwrap_or_else(|| panic!("semantic region slot was not populated"));
                Region::new(region.kind, region.parameters, region.results, region.nodes)
            })
            .collect();
        let parameters = std::mem::take(&mut self.parameters);
        let values = std::mem::take(&mut self.values);
        Ok(SemanticFunction::new(
            crate::entry::internals::Function::new(
                self.id,
                self.definition.stable,
                self.definition.name.clone(),
                self.definition.span,
                parameters,
                results,
                root,
                regions,
                values,
            ),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_parameter(
        &mut self,
        root: RegionId,
        source: u32,
        name: &str,
        path: &[u32],
        ty: SemanticType,
        access: ParameterAccess,
        span: crate::span::Span,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        if let SemanticType::Tuple(items) = &ty {
            let mut inputs = Vec::with_capacity(items.len());
            for (ordinal, item) in items.iter().cloned().enumerate() {
                let mut child = path.to_vec();
                child.push(u32::try_from(ordinal).expect("tuple has more than u32::MAX items"));
                inputs.push(self.lower_parameter(root, source, name, &child, item, access, span)?);
            }
            return Ok(self.emit(root, NodeKind::TuplePack, inputs, vec![ty], span)[0]);
        }
        let leaf_access = match &ty {
            SemanticType::Tensor(_) if access == ParameterAccess::Scalar => ParameterAccess::Owned,
            _ => access,
        };
        let mut ty = ty;
        if let SemanticType::Tensor(tensor) = &mut ty {
            if leaf_access == ParameterAccess::Mutable
                && crate::registry::representation_info(tensor.representation).access
                    != crate::registry::RepresentationAccess::ReadWrite
            {
                return Err(SourceDiagnostic {
                    path: self.module.sources.files()[self.definition.file]
                        .path
                        .clone(),
                    span,
                    message: format!(
                        "representation `{}` is decode-only and cannot bind a mutable tensor parameter",
                        crate::registry::representation_info(tensor.representation).name
                    ),
                });
            }
            tensor.storage = TensorStorage::Parameter(leaf_access);
        }
        let value = self.value(ty, ValueOrigin::Parameter, span);
        let mut leaf_name = name.to_owned();
        for ordinal in path {
            leaf_name.push('_');
            leaf_name.push_str(&ordinal.to_string());
        }
        self.parameters.push(FunctionParameter {
            source,
            path: path.to_vec(),
            name: leaf_name,
            value,
            access: leaf_access,
        });
        Ok(value)
    }

    fn reserve_region(&mut self, kind: RegionKind) -> RegionId {
        let id = RegionId::new(
            self.id,
            u32::try_from(self.regions.len()).expect("function has more than u32::MAX regions"),
        );
        self.regions.push(Some(RegionWork {
            id,
            kind,
            parameters: Vec::new(),
            results: Vec::new(),
            nodes: Vec::new(),
        }));
        id
    }

    fn value(
        &mut self,
        ty: SemanticType,
        origin: ValueOrigin,
        span: crate::span::Span,
    ) -> SemanticValueId {
        let id = SemanticValueId::new(
            self.id,
            u32::try_from(self.values.len()).expect("function has more than u32::MAX values"),
        );
        self.values.push(ValueInfo { ty, origin, span });
        id
    }

    fn region_parameter(
        &mut self,
        region: RegionId,
        ty: SemanticType,
        span: crate::span::Span,
    ) -> SemanticValueId {
        let value = self.value(ty, ValueOrigin::RegionParameter(region), span);
        self.region_mut(region).parameters.push(value);
        value
    }

    fn region_mut(&mut self, id: RegionId) -> &mut RegionWork {
        assert_eq!(
            id.function(),
            self.id,
            "region belongs to another semantic function"
        );
        self.regions[id.index()]
            .as_mut()
            .unwrap_or_else(|| panic!("semantic region slot is absent"))
    }

    fn emit(
        &mut self,
        region: RegionId,
        kind: NodeKind,
        inputs: Vec<SemanticValueId>,
        output_types: Vec<SemanticType>,
        span: crate::span::Span,
    ) -> Vec<SemanticValueId> {
        let ordinal = self.region_mut(region).nodes.len();
        let node = NodeId::new(
            region,
            u32::try_from(ordinal).expect("region has more than u32::MAX nodes"),
        );
        let outputs = output_types
            .into_iter()
            .map(|ty| self.value(ty, ValueOrigin::Node(node), span))
            .collect::<Vec<_>>();
        let events = self.semantic_events(node, &kind, &inputs, &outputs);
        self.region_mut(region).nodes.push(SemanticNode::new(
            kind,
            inputs,
            outputs.clone(),
            events,
            span,
        ));
        outputs
    }

    fn participant_domain(&self) -> ParticipantDomain {
        if self.parallel_participants.is_empty() {
            ParticipantDomain::Single
        } else {
            ParticipantDomain::Parallel(self.parallel_participants.clone().into_boxed_slice())
        }
    }

    fn tensor_representation(&self, value: SemanticValueId) -> Option<RepresentationId> {
        match &self.values[value.index()].ty {
            SemanticType::Tensor(tensor) => Some(tensor.representation),
            _ => None,
        }
    }

    fn semantic_events(
        &mut self,
        node: NodeId,
        kind: &NodeKind,
        inputs: &[SemanticValueId],
        outputs: &[SemanticValueId],
    ) -> Vec<SemanticEvent> {
        let participants = self.participant_domain();
        let mut specs = Vec::new();
        match kind {
            NodeKind::Atomic { op, capability } => {
                let place = *inputs
                    .first()
                    .unwrap_or_else(|| panic!("checked atomic has no write place"));
                let representation = self
                    .tensor_representation(place)
                    .unwrap_or_else(|| panic!("checked atomic place is not a tensor"));
                let dtype = crate::registry::representation_info(representation).decoded;
                let outcome = match op {
                    crate::intrinsics::AtomicOp::Add
                        if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) =>
                    {
                        AssociationOutcome::Reassociated { accumulator: dtype }
                    }
                    crate::intrinsics::AtomicOp::Add
                    | crate::intrinsics::AtomicOp::Max
                    | crate::intrinsics::AtomicOp::Min => AssociationOutcome::Exact,
                };
                assert_eq!(capability.place(), place);
                assert_eq!(capability.outcome(), outcome);
                let id = SemanticEventId::new(node, 0);
                let event = SemanticEvent::checked(
                    id,
                    Some(place),
                    capability.indices().to_vec(),
                    AccessKind::AtomicRmw {
                        op: *op,
                        capability: capability.clone(),
                    },
                    Some(representation),
                    capability.participants().clone(),
                    self.last_event
                        .get(&node.region())
                        .cloned()
                        .unwrap_or_default(),
                    VisibilityScope::Command,
                    match outcome {
                        AssociationOutcome::Exact => NumericalOutcome::Deterministic,
                        outcome => NumericalOutcome::AllowedAssociation(outcome),
                    },
                );
                self.last_event.insert(node.region(), vec![id]);
                return vec![event];
            }
            NodeKind::If { then, otherwise } => {
                // Child operations own their events. Control nodes never
                // duplicate those events as a second semantic authority.
                let mut terminal = self.last_event.get(then).cloned().unwrap_or_default();
                for event in self.last_event.get(otherwise).into_iter().flatten() {
                    if !terminal.contains(event) {
                        terminal.push(*event);
                    }
                }
                self.last_event.insert(node.region(), terminal);
            }
            NodeKind::Loop { body, .. } => {
                // Include the pre-loop predecessor for the zero-visit path
                // and the body terminal for every nonempty execution.
                let mut terminal = self
                    .last_event
                    .get(&node.region())
                    .cloned()
                    .unwrap_or_default();
                for event in self.last_event.get(body).into_iter().flatten() {
                    if !terminal.contains(event) {
                        terminal.push(*event);
                    }
                }
                self.last_event.insert(node.region(), terminal);
            }
            NodeKind::Alloc | NodeKind::Fill { .. } => {
                for output in outputs {
                    if let Some(representation) = self.tensor_representation(*output) {
                        specs.push((
                            Some(*output),
                            AccessKind::Write(None),
                            Some(representation),
                            VisibilityScope::Participant,
                            NumericalOutcome::Deterministic,
                        ));
                    }
                }
            }
            NodeKind::Primitive(_)
            | NodeKind::View(_)
            | NodeKind::Check { .. }
            | NodeKind::TuplePack
            | NodeKind::TupleGet { .. }
            | NodeKind::Extent { .. } => {}
            NodeKind::Intrinsic(intrinsic) => {
                let signature = crate::registry::intrinsic_signature(*intrinsic);
                let reads =
                    signature
                        .arguments
                        .iter()
                        .enumerate()
                        .filter_map(|(ordinal, argument)| {
                            matches!(
                                argument.category,
                                crate::registry::OperandCategory::Readable { .. }
                                    | crate::registry::OperandCategory::Writable { .. }
                            )
                            .then_some(inputs[ordinal])
                        });
                let writes = signature.effects.writes.iter().map(|ordinal| {
                    inputs[usize::try_from(*ordinal).expect("intrinsic write index exceeds usize")]
                });
                Self::push_access_specs(self, &mut specs, reads, writes);
                Self::push_owned_output_specs(self, &mut specs, outputs);
            }
            NodeKind::Elementwise(_) => {
                let reads = inputs
                    .iter()
                    .copied()
                    .filter(|value| self.tensor_representation(*value).is_some())
                    .collect::<Vec<_>>();
                Self::push_access_specs(self, &mut specs, reads, std::iter::empty());
            }
            NodeKind::Reduce { .. } | NodeKind::ElementRead => {
                Self::push_access_specs(
                    self,
                    &mut specs,
                    inputs.first().copied(),
                    std::iter::empty(),
                );
            }
            NodeKind::Call { family } => {
                let reference = self.builder.families[family.index()]
                    .as_ref()
                    .and_then(|family| {
                        family
                            .candidates()
                            .iter()
                            .find(|candidate| candidate.numerical == NumericalRole::Reference)
                    })
                    .map(|candidate| candidate.function)
                    .unwrap_or_else(|| panic!("checked call family has no reference function"));
                let function = self.builder.functions[reference.index()]
                    .as_ref()
                    .unwrap_or_else(|| panic!("checked call reference function is absent"));
                let reads = function
                    .parameters()
                    .iter()
                    .zip(inputs)
                    .filter_map(|(parameter, input)| {
                        matches!(
                            parameter.access,
                            ParameterAccess::Shared
                                | ParameterAccess::Owned
                                | ParameterAccess::Mutable
                        )
                        .then_some(*input)
                    })
                    .collect::<Vec<_>>();
                let writes = function
                    .parameters()
                    .iter()
                    .zip(inputs)
                    .filter_map(|(parameter, input)| {
                        (parameter.access == ParameterAccess::Mutable).then_some(*input)
                    })
                    .collect::<Vec<_>>();
                Self::push_access_specs(self, &mut specs, reads, writes);
                Self::push_owned_output_specs(self, &mut specs, outputs);
            }
            NodeKind::Copy | NodeKind::RepresentationConvert { .. } => {
                Self::push_access_specs(
                    self,
                    &mut specs,
                    inputs.first().copied(),
                    std::iter::empty(),
                );
                Self::push_owned_output_specs(self, &mut specs, outputs);
            }
            NodeKind::ElementWrite { authority } => {
                let place = *inputs
                    .first()
                    .unwrap_or_else(|| panic!("checked element write has no place"));
                Self::push_access_specs(self, &mut specs, [place], std::iter::empty());
                let representation = self
                    .tensor_representation(place)
                    .unwrap_or_else(|| panic!("checked element write place is not a tensor"));
                if let Some(authority) = authority {
                    assert_eq!(
                        &participants,
                        authority.participants(),
                        "checked write authority participant domain changed during monomorphization"
                    );
                }
                specs.push((
                    Some(place),
                    AccessKind::Write(authority.clone()),
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
            NodeKind::Store { authority } => {
                let destination = *inputs
                    .first()
                    .unwrap_or_else(|| panic!("checked store has no destination"));
                let source = *inputs
                    .get(1)
                    .unwrap_or_else(|| panic!("checked store has no source"));
                Self::push_access_specs(
                    self,
                    &mut specs,
                    [destination, source],
                    std::iter::empty(),
                );
                let representation = self
                    .tensor_representation(destination)
                    .unwrap_or_else(|| panic!("checked store destination is not a tensor"));
                if let Some(authority) = authority {
                    assert_eq!(
                        &participants,
                        authority.participants(),
                        "checked store authority participant domain changed during monomorphization"
                    );
                }
                specs.push((
                    Some(destination),
                    AccessKind::Write(authority.clone()),
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
        }
        let mut prior = self
            .last_event
            .get(&node.region())
            .cloned()
            .unwrap_or_default();
        let mut events = Vec::with_capacity(specs.len());
        let location_indices = match kind {
            NodeKind::ElementRead => inputs[1..].to_vec(),
            NodeKind::ElementWrite { .. } => inputs[1..inputs.len() - 1].to_vec(),
            _ => Vec::new(),
        };
        for (ordinal, (place, access, representation, visibility, numerical)) in
            specs.into_iter().enumerate()
        {
            let id = SemanticEventId::new(
                node,
                u32::try_from(ordinal).expect("node has more than u32::MAX semantic events"),
            );
            let dependencies = prior.clone();
            events.push(SemanticEvent::checked(
                id,
                place,
                location_indices.clone(),
                access,
                representation,
                participants.clone(),
                dependencies,
                visibility,
                numerical,
            ));
            prior = vec![id];
        }
        if !events.is_empty() {
            self.last_event.insert(node.region(), prior);
        }
        events
    }

    fn push_access_specs(
        &mut self,
        specs: &mut Vec<(
            Option<SemanticValueId>,
            AccessKind,
            Option<RepresentationId>,
            VisibilityScope,
            NumericalOutcome,
        )>,
        reads: impl IntoIterator<Item = SemanticValueId>,
        writes: impl IntoIterator<Item = SemanticValueId>,
    ) {
        for place in reads {
            if let Some(representation) = self.tensor_representation(place) {
                specs.push((
                    Some(place),
                    AccessKind::Read,
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
        }
        for place in writes {
            let representation = self
                .tensor_representation(place)
                .unwrap_or_else(|| panic!("checked write place is not a tensor"));
            specs.push((
                Some(place),
                AccessKind::Write(None),
                Some(representation),
                VisibilityScope::Participant,
                NumericalOutcome::Deterministic,
            ));
        }
    }

    fn push_owned_output_specs(
        &self,
        specs: &mut Vec<(
            Option<SemanticValueId>,
            AccessKind,
            Option<RepresentationId>,
            VisibilityScope,
            NumericalOutcome,
        )>,
        outputs: &[SemanticValueId],
    ) {
        for output in outputs {
            if let Some(representation) = self.tensor_representation(*output) {
                specs.push((
                    Some(*output),
                    AccessKind::Write(None),
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
        }
    }

    fn semantic_type(&mut self, ty: &ValueType) -> SemanticType {
        match ty {
            ValueType::Scalar(dtype) => SemanticType::Scalar(*dtype),
            ValueType::Index { bound } => {
                let bound = self.transfer_int(*bound);
                SemanticType::Index {
                    bound: self.builder.arena.nat_from_int(bound),
                }
            }
            ValueType::Range { bound } => {
                let bound = self.transfer_int(*bound);
                SemanticType::Range {
                    bound: self.builder.arena.nat_from_int(bound),
                }
            }
            ValueType::Tensor(tensor) => {
                let representation = self.resolve_elem(&tensor.elem);
                let axes = tensor
                    .axes
                    .iter()
                    .map(|axis| {
                        let axis = self.transfer_int(*axis);
                        self.builder.arena.nat_from_int(axis)
                    })
                    .collect();
                SemanticType::Tensor(TensorSemantics {
                    representation,
                    axes,
                    storage: TensorStorage::Computed,
                })
            }
            ValueType::Tuple(items) => {
                SemanticType::Tuple(items.iter().map(|item| self.semantic_type(item)).collect())
            }
            ValueType::Opaque { capability, name } => SemanticType::Opaque {
                capability: *capability,
                name,
            },
            ValueType::Void => SemanticType::Void,
        }
    }

    fn resolve_elem(&self, elem: &Elem) -> RepresentationId {
        match elem {
            Elem::Dtype(dtype) => crate::registry::dense(*dtype),
            Elem::Repr(representation) => *representation,
            Elem::Param(name) => *self
                .elements
                .get(name)
                .unwrap_or_else(|| panic!("unbound element parameter in monomorphization")),
        }
    }

    fn transfer_int(&mut self, expression: IntExpr) -> IntExpr {
        let symbols = &self.symbols;
        let mut map =
            |symbol: SymbolId, _: &mut ExprArena| {
                AnyExpr::Int(*symbols.get(&symbol).unwrap_or_else(|| {
                    panic!("checked expression contains an unbound body symbol")
                }))
            };
        xfer::transfer_int(
            &self.definition.arena,
            expression,
            &mut self.builder.arena,
            &mut map,
        )
    }

    fn runtime_int(&mut self, value: SemanticValueId) -> IntExpr {
        if let Some(expression) = self.runtime_values.get(&value) {
            return *expression;
        }
        let (_, expression) = self.builder.arena.runtime_value(value);
        self.runtime_values.insert(value, expression);
        expression
    }

    fn lower_block(
        &mut self,
        region: RegionId,
        block: &ir::Block,
    ) -> Result<Vec<SemanticValueId>, SourceDiagnostic> {
        for statement in &block.statements {
            self.lower_stmt(region, statement)?;
        }
        Ok(match &block.terminator {
            ir::Terminator::Continue => Vec::new(),
            ir::Terminator::Return(values) => {
                let mut out = Vec::new();
                for value in values {
                    let value = self.lower_expr(region, value)?;
                    self.flatten_value(region, value, value, &mut out, block)?;
                }
                out
            }
        })
    }

    fn flatten_value(
        &mut self,
        region: RegionId,
        value: SemanticValueId,
        _root: SemanticValueId,
        output: &mut Vec<SemanticValueId>,
        block: &ir::Block,
    ) -> Result<(), SourceDiagnostic> {
        let ty = self.values[value.index()].ty.clone();
        match ty {
            SemanticType::Tuple(items) => {
                for (index, item) in items.into_iter().enumerate() {
                    let projected = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![item],
                        self.definition.span,
                    )[0];
                    self.flatten_value(region, projected, projected, output, block)?;
                }
            }
            SemanticType::Void => {}
            _ => output.push(value),
        }
        Ok(())
    }

    fn lower_stmt(
        &mut self,
        region: RegionId,
        statement: &ir::Stmt,
    ) -> Result<(), SourceDiagnostic> {
        match statement {
            ir::Stmt::Let { pattern, value, .. } => {
                let span = value.span;
                let value = self.lower_expr(region, value)?;
                self.bind_pattern(region, pattern, value, span)?;
            }
            ir::Stmt::Assign {
                place,
                op,
                value,
                authorities,
            } => {
                fn element_places(place: &ir::Place) -> usize {
                    match place {
                        ir::Place::Local(_) => 0,
                        ir::Place::Element { .. } => 1,
                        ir::Place::Tuple(places) => places.iter().map(element_places).sum(),
                    }
                }
                let expected_authorities = if self.parallel_depth == 0 {
                    0
                } else {
                    element_places(place)
                };
                assert_eq!(
                    authorities.len(),
                    expected_authorities,
                    "checked assignment authority count differs from its exact write regions"
                );
                let span = value.span;
                let value = self.lower_expr(region, value)?;
                self.assign(region, place, *op, value, authorities, span)?;
            }
            ir::Stmt::Evaluate(expression) => {
                self.lower_expr(region, expression)?;
            }
            ir::Stmt::If {
                condition,
                then_body,
                else_body,
            } => self.lower_if(region, condition, then_body, else_body)?,
            ir::Stmt::Loop {
                kind,
                binder,
                start,
                end,
                body,
            } => self.lower_loop(region, *kind, *binder, start, end, body)?,
        }
        Ok(())
    }

    fn bind_pattern(
        &mut self,
        region: RegionId,
        pattern: &ir::Pattern,
        value: SemanticValueId,
        span: crate::span::Span,
    ) -> Result<(), SourceDiagnostic> {
        match pattern {
            ir::Pattern::Local(local) => {
                self.locals[local.index()] = Some(value);
                if let Some(source_symbol) = self.definition.body.locals[local.index()].symbol {
                    let runtime = self.runtime_int(value);
                    self.symbols.insert(source_symbol, runtime);
                }
            }
            ir::Pattern::Tuple(patterns) => {
                let SemanticType::Tuple(items) = self.values[value.index()].ty.clone() else {
                    panic!("checked tuple pattern received a non-tuple value")
                };
                assert_eq!(
                    patterns.len(),
                    items.len(),
                    "checked tuple pattern arity changed during monomorphization"
                );
                for (index, (pattern, ty)) in patterns.iter().zip(items).enumerate() {
                    let component = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![ty],
                        span,
                    )[0];
                    self.bind_pattern(region, pattern, component, span)?;
                }
            }
        }
        Ok(())
    }

    fn assign(
        &mut self,
        region: RegionId,
        place: &ir::Place,
        op: AssignOp,
        value: SemanticValueId,
        authorities: &[ir::ExclusiveWriteCapability],
        span: crate::span::Span,
    ) -> Result<(), SourceDiagnostic> {
        match place {
            ir::Place::Local(local) => {
                let value = if op == AssignOp::Assign {
                    value
                } else {
                    let old = self.local(*local);
                    let binary = match op {
                        AssignOp::Add => BinaryOp::Add,
                        AssignOp::Sub => BinaryOp::Sub,
                        AssignOp::Mul => BinaryOp::Mul,
                        AssignOp::Assign => unreachable!(),
                    };
                    let ty = self.values[old.index()].ty.clone();
                    self.emit(
                        region,
                        self.primitive_kind(&PrimitiveId::Binary(binary), &ty),
                        vec![old, value],
                        vec![ty],
                        span,
                    )[0]
                };
                self.locals[local.index()] = Some(value);
            }
            ir::Place::Tuple(places) => {
                let SemanticType::Tuple(items) = self.values[value.index()].ty.clone() else {
                    panic!("checked tuple assignment received a non-tuple value")
                };
                assert_eq!(
                    places.len(),
                    items.len(),
                    "checked tuple assignment arity changed during monomorphization"
                );
                for (index, (place, ty)) in places.iter().zip(items).enumerate() {
                    let component = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![ty],
                        span,
                    )[0];
                    self.assign(region, place, op, component, authorities, span)?;
                }
            }
            ir::Place::Element { root, indices } => {
                let supplied =
                    authorities
                        .iter()
                        .find(|authority| authority.region() == place)
                        .map(|authority| {
                            authority
                                .participants()
                                .iter()
                                .map(|participant| {
                                    *self.participant_bindings.get(participant).unwrap_or_else(|| {
                                    panic!("checked write authority names an inactive participant")
                                })
                                })
                                .collect::<Vec<_>>()
                        });
                let base = self.local(*root);
                let semantic_participants = supplied.map(|supplied| {
                    assert!(!supplied.is_empty());
                    assert!(supplied
                        .iter()
                        .all(|participant| self.parallel_participants.contains(participant)));
                    let participants = ParticipantDomain::Parallel(supplied.into_boxed_slice());
                    participants
                });
                if let SemanticType::Tensor(tensor) = &self.values[base.index()].ty {
                    if crate::registry::representation_info(tensor.representation).access
                        != crate::registry::RepresentationAccess::ReadWrite
                    {
                        return Err(SourceDiagnostic {
                            path: self.module.sources.files()[self.definition.file]
                                .path
                                .clone(),
                            span,
                            message: format!(
                                "representation `{}` is decode-only and has no canonical write contract",
                                crate::registry::representation_info(tensor.representation).name
                            ),
                        });
                    }
                }
                let mut inputs = vec![base];
                let mut slice_axes = Vec::with_capacity(indices.len());
                for (axis, index) in indices.iter().enumerate() {
                    match index {
                        ir::Index::Point {
                            value: point,
                            runtime_check,
                        } => {
                            let point = self.lower_expr(region, point)?;
                            if *runtime_check {
                                self.emit_point_check(region, base, axis, point, span);
                            }
                            inputs.push(point);
                            slice_axes.push(SliceAxis::Point(ScalarRef::Value(point)));
                        }
                        ir::Index::Range {
                            start,
                            end,
                            check_start,
                            check_order,
                            check_end,
                        } => {
                            let start_value = match start {
                                Some(start) => Some(self.lower_expr(region, start)?),
                                None => None,
                            };
                            let end_value = match end {
                                Some(end) => Some(self.lower_expr(region, end)?),
                                None => None,
                            };
                            self.emit_range_checks(
                                region,
                                base,
                                axis,
                                start_value,
                                end_value,
                                (*check_start, *check_order, *check_end),
                                span,
                            );
                            if let Some(start) = start_value {
                                inputs.push(start);
                            }
                            if let Some(end) = end_value {
                                inputs.push(end);
                            }
                            slice_axes.push(if start_value.is_none() && end_value.is_none() {
                                SliceAxis::Full
                            } else {
                                SliceAxis::Range {
                                    start: start_value.map(ScalarRef::Value),
                                    end: end_value.map(ScalarRef::Value),
                                }
                            });
                        }
                    }
                }
                let stored_value = if op == AssignOp::Assign {
                    value
                } else {
                    let representation = match &self.values[base.index()].ty {
                        SemanticType::Tensor(tensor) => tensor.representation,
                        _ => panic!("checked element place has a non-tensor root"),
                    };
                    let dtype = crate::registry::representation_info(representation).decoded;
                    let old = self.emit(
                        region,
                        NodeKind::ElementRead,
                        inputs.clone(),
                        vec![SemanticType::Scalar(dtype)],
                        span,
                    )[0];
                    let binary = match op {
                        AssignOp::Add => BinaryOp::Add,
                        AssignOp::Sub => BinaryOp::Sub,
                        AssignOp::Mul => BinaryOp::Mul,
                        AssignOp::Assign => unreachable!(),
                    };
                    self.emit(
                        region,
                        NodeKind::Primitive(PrimitiveId::Binary(binary)),
                        vec![old, value],
                        vec![SemanticType::Scalar(dtype)],
                        span,
                    )[0]
                };
                let tensor_store = matches!(
                    self.values[stored_value.index()].ty,
                    SemanticType::Tensor(_)
                );
                let (kind, store_inputs) = if tensor_store {
                    let base_tensor = match &self.values[base.index()].ty {
                        SemanticType::Tensor(tensor) => tensor.clone(),
                        _ => panic!("checked store base is not a tensor"),
                    };
                    slice_axes.extend(
                        (slice_axes.len()..base_tensor.axes.len()).map(|_| SliceAxis::Full),
                    );
                    let transform = ViewTransform::Slice { axes: slice_axes };
                    let stored_tensor = match &self.values[stored_value.index()].ty {
                        SemanticType::Tensor(tensor) => tensor,
                        _ => unreachable!(),
                    };
                    let destination = self.emit(
                        region,
                        NodeKind::View(transform.clone()),
                        inputs,
                        vec![SemanticType::Tensor(TensorSemantics {
                            representation: base_tensor.representation,
                            axes: stored_tensor.axes.clone(),
                            storage: TensorStorage::View { base, transform },
                        })],
                        span,
                    )[0];
                    (
                        NodeKind::Store {
                            authority: semantic_participants.map(|participants| {
                                ExclusiveWriteCapability::checked(
                                    destination,
                                    Vec::new(),
                                    participants,
                                )
                            }),
                        },
                        vec![destination, stored_value],
                    )
                } else {
                    let semantic_indices = inputs[1..].to_vec();
                    inputs.push(stored_value);
                    (
                        NodeKind::ElementWrite {
                            authority: semantic_participants.map(|participants| {
                                ExclusiveWriteCapability::checked(
                                    base,
                                    semantic_indices,
                                    participants,
                                )
                            }),
                        },
                        inputs,
                    )
                };
                let ty = self.values[base.index()].ty.clone();
                let updated = self.emit(region, kind, store_inputs, vec![ty], span)[0];
                self.locals[root.index()] = Some(updated);
            }
        }
        Ok(())
    }

    fn local(&self, local: ir::LocalId) -> SemanticValueId {
        self.locals[local.index()]
            .unwrap_or_else(|| panic!("checked local used before semantic definition"))
    }

    fn primitive_kind(&self, primitive: &PrimitiveId, output: &SemanticType) -> NodeKind {
        match primitive {
            PrimitiveId::TuplePack => NodeKind::TuplePack,
            PrimitiveId::TupleGet(index) => NodeKind::TupleGet { index: *index },
            PrimitiveId::TensorAlloc => NodeKind::Alloc,
            PrimitiveId::Fill(value) => NodeKind::Fill { value: *value },
            PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load => NodeKind::Copy,
            PrimitiveId::RepresentationConvert(_) => {
                panic!("representation conversion is resolved during entry construction")
            }
            PrimitiveId::SliceView { .. } | PrimitiveId::Transpose | PrimitiveId::Reshape => {
                panic!("view primitives require their structural transform")
            }
            PrimitiveId::ElementRead { .. } => NodeKind::ElementRead,
            PrimitiveId::Atomic { .. } => {
                panic!("checked atomic primitive lacks its identity-bound authority")
            }
            PrimitiveId::Reduce {
                op,
                axis,
                unordered,
            } => NodeKind::Reduce {
                op: *op,
                axis: *axis,
                unordered: *unordered,
            },
            PrimitiveId::Extent { axis } => NodeKind::Extent { axis: *axis },
            PrimitiveId::Constant(_)
            | PrimitiveId::Symbolic(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Unary(_)
            | PrimitiveId::Binary(_)
            | PrimitiveId::Cast(_)
            | PrimitiveId::Math(_)
            | PrimitiveId::Select
            | PrimitiveId::Decode => {
                if matches!(output, SemanticType::Tensor(_)) {
                    NodeKind::Elementwise(primitive.clone())
                } else {
                    NodeKind::Primitive(primitive.clone())
                }
            }
        }
    }

    fn scalar_constant(
        &mut self,
        region: RegionId,
        value: i64,
        span: crate::span::Span,
    ) -> SemanticValueId {
        self.emit(
            region,
            NodeKind::Primitive(PrimitiveId::Constant(Constant::Int(value))),
            Vec::new(),
            vec![SemanticType::Scalar(DType::I32)],
            span,
        )[0]
    }

    fn scalar_typed_constant(
        &mut self,
        region: RegionId,
        dtype: DType,
        value: i64,
        span: crate::span::Span,
    ) -> SemanticValueId {
        self.emit(
            region,
            NodeKind::Primitive(PrimitiveId::Constant(Constant::Int(value))),
            Vec::new(),
            vec![SemanticType::Scalar(dtype)],
            span,
        )[0]
    }

    fn extent_value(
        &mut self,
        region: RegionId,
        base: SemanticValueId,
        axis: usize,
        span: crate::span::Span,
    ) -> SemanticValueId {
        self.emit(
            region,
            NodeKind::Extent {
                axis: u32::try_from(axis).expect("tensor rank exceeds u32::MAX"),
            },
            vec![base],
            vec![SemanticType::Scalar(DType::I32)],
            span,
        )[0]
    }

    fn comparison(
        &mut self,
        region: RegionId,
        op: BinaryOp,
        left: SemanticValueId,
        right: SemanticValueId,
        span: crate::span::Span,
    ) -> SemanticValueId {
        self.emit(
            region,
            NodeKind::Primitive(PrimitiveId::Binary(op)),
            vec![left, right],
            vec![SemanticType::Scalar(DType::Bool)],
            span,
        )[0]
    }

    fn check(
        &mut self,
        region: RegionId,
        condition: SemanticValueId,
        reason: CheckReason,
        span: crate::span::Span,
    ) {
        if self.entry_root
            && matches!(
                self.regions[region.index()].as_ref().map(|work| &work.kind),
                Some(RegionKind::Root)
            )
            && self.entry_known(condition, &mut BTreeSet::new())
        {
            self.entry_checks.push(condition);
            return;
        }
        self.emit(
            region,
            NodeKind::Check { reason },
            vec![condition],
            Vec::new(),
            span,
        );
    }

    /// Whether a safety condition is determined entirely by invocation
    /// dimensions/scalars. Such a condition belongs to EntryDomain; tensor
    /// contents, loop binders, and region-local state remain execution-time
    /// checks.
    fn entry_known(&self, value: SemanticValueId, seen: &mut BTreeSet<SemanticValueId>) -> bool {
        if !seen.insert(value) {
            return true;
        }
        match self.values[value.index()].origin {
            ValueOrigin::Parameter => matches!(
                self.values[value.index()].ty,
                SemanticType::Scalar(_) | SemanticType::Index { .. } | SemanticType::Range { .. }
            ),
            ValueOrigin::RegionParameter(_) => false,
            ValueOrigin::Node(node) => {
                let work = self.regions[node.region().index()]
                    .as_ref()
                    .unwrap_or_else(|| panic!("semantic region slot is absent"));
                let node = &work.nodes[node.ordinal()];
                match node.view() {
                    SemanticNodeView::Primitive {
                        primitive, inputs, ..
                    } => {
                        matches!(
                            primitive,
                            PrimitiveId::Constant(Constant::Int(_) | Constant::Bool(_))
                                | PrimitiveId::Unary(
                                    crate::syntax::ast::UnaryOp::Neg
                                        | crate::syntax::ast::UnaryOp::Not
                                )
                                | PrimitiveId::Binary(
                                    BinaryOp::Add
                                        | BinaryOp::Sub
                                        | BinaryOp::Mul
                                        | BinaryOp::Div
                                        | BinaryOp::Rem
                                        | BinaryOp::Eq
                                        | BinaryOp::Ne
                                        | BinaryOp::Lt
                                        | BinaryOp::Le
                                        | BinaryOp::Gt
                                        | BinaryOp::Ge
                                        | BinaryOp::And
                                        | BinaryOp::Or
                                )
                                | PrimitiveId::Cast(DType::I32 | DType::U32)
                                | PrimitiveId::Select
                        ) && inputs.iter().all(|input| self.entry_known(*input, seen))
                    }
                    SemanticNodeView::TuplePack { .. } | SemanticNodeView::TupleGet { .. } => false,
                    SemanticNodeView::Extent {
                        tensor: input,
                        axis,
                        ..
                    } => {
                        let SemanticType::Tensor(tensor) = &self.values[input.index()].ty else {
                            panic!("checked extent input is not a tensor")
                        };
                        tensor.axes.get(axis as usize).is_some_and(|extent| {
                            self.entry_expression_known(AnyExpr::Nat(*extent), seen)
                        })
                    }
                    SemanticNodeView::View {
                        base, transform, ..
                    } => {
                        self.entry_known(base, seen)
                            && view_scalar_values(transform)
                                .all(|input| self.entry_known(input, seen))
                    }
                    SemanticNodeView::Intrinsic { .. }
                    | SemanticNodeView::Elementwise { .. }
                    | SemanticNodeView::Reduce { .. }
                    | SemanticNodeView::Call { .. }
                    | SemanticNodeView::Alloc { .. }
                    | SemanticNodeView::Fill { .. }
                    | SemanticNodeView::Copy { .. }
                    | SemanticNodeView::RepresentationConvert { .. }
                    | SemanticNodeView::ElementRead { .. }
                    | SemanticNodeView::ElementWrite { .. }
                    | SemanticNodeView::Store { .. }
                    | SemanticNodeView::Atomic { .. }
                    | SemanticNodeView::If { .. }
                    | SemanticNodeView::Loop { .. }
                    | SemanticNodeView::Check { .. } => false,
                }
            }
        }
    }

    fn entry_expression_known(
        &self,
        expression: AnyExpr,
        seen: &mut BTreeSet<SemanticValueId>,
    ) -> bool {
        match self.builder.arena.view(expression) {
            NodeView::NatConst(_)
            | NodeView::IntConst(_)
            | NodeView::BoolConst(_)
            | NodeView::ScalarConst { .. } => true,
            NodeView::Symbol(symbol) => match self.builder.arena.symbol_kind(symbol) {
                crate::expr::SymbolKind::CallDimension(_)
                | crate::expr::SymbolKind::CallScalar(_) => true,
                crate::expr::SymbolKind::RuntimeValue(value) => self.entry_known(value, seen),
                crate::expr::SymbolKind::TemplateDimension(_)
                | crate::expr::SymbolKind::TargetConstant(_)
                | crate::expr::SymbolKind::Decision(_)
                | crate::expr::SymbolKind::LoopBinder(_)
                | crate::expr::SymbolKind::ScheduleSlot(_) => false,
            },
            NodeView::Unary { operand, .. } => self.entry_expression_known(operand, seen),
            NodeView::Binary { lhs, rhs, .. } | NodeView::Cmp { lhs, rhs, .. } => {
                self.entry_expression_known(lhs, seen) && self.entry_expression_known(rhs, seen)
            }
            NodeView::Nary { operands, .. } => operands
                .iter()
                .all(|operand| self.entry_expression_known(*operand, seen)),
            NodeView::Select {
                cond,
                then,
                otherwise,
            } => {
                self.entry_expression_known(AnyExpr::Bool(cond), seen)
                    && self.entry_expression_known(then, seen)
                    && self.entry_expression_known(otherwise, seen)
            }
            NodeView::In { operand, .. } => self.entry_expression_known(operand, seen),
            NodeView::Fold { .. } | NodeView::Duration(_) | NodeView::DurationScale { .. } => false,
        }
    }

    /// The region parameter whose storage a tensor value aliases. This
    /// follows only identity-preserving view and mutation nodes; computed
    /// tensors never inherit an alias merely because they read a parameter.
    fn region_storage_origin(
        &self,
        region: RegionId,
        value: SemanticValueId,
        seen: &mut BTreeSet<SemanticValueId>,
    ) -> Option<usize> {
        if !seen.insert(value) {
            return None;
        }
        let work = self.regions[region.index()].as_ref()?;
        if let Some(ordinal) = work
            .parameters
            .iter()
            .position(|parameter| *parameter == value)
        {
            return Some(ordinal);
        }
        let ValueOrigin::Node(node) = self.values[value.index()].origin else {
            return None;
        };
        if node.region() != region {
            return None;
        }
        let aliased = match work.nodes[node.ordinal()].view() {
            SemanticNodeView::View { base, .. } => base,
            SemanticNodeView::ElementWrite { place, .. }
            | SemanticNodeView::Atomic { place, .. } => place,
            SemanticNodeView::Store { destination, .. } => destination,
            SemanticNodeView::Loop { carries, .. } => carries
                .iter()
                .find(|carry| carry.result == value)
                .map(|carry| carry.initial)?,
            _ => return None,
        };
        self.region_storage_origin(region, aliased, seen)
    }

    fn emit_point_check(
        &mut self,
        region: RegionId,
        base: SemanticValueId,
        axis: usize,
        point: SemanticValueId,
        span: crate::span::Span,
    ) {
        let zero = self.scalar_constant(region, 0, span);
        let extent = self.extent_value(region, base, axis, span);
        let lower = self.comparison(region, BinaryOp::Ge, point, zero, span);
        self.check(region, lower, CheckReason::IndexBound, span);
        let upper = self.comparison(region, BinaryOp::Lt, point, extent, span);
        self.check(region, upper, CheckReason::IndexBound, span);
    }

    fn emit_range_checks(
        &mut self,
        region: RegionId,
        base: SemanticValueId,
        axis: usize,
        start: Option<SemanticValueId>,
        end: Option<SemanticValueId>,
        enabled: (bool, bool, bool),
        span: crate::span::Span,
    ) {
        if !enabled.0 && !enabled.1 && !enabled.2 {
            return;
        }
        let zero = self.scalar_constant(region, 0, span);
        let extent = self.extent_value(region, base, axis, span);
        let start = start.unwrap_or(zero);
        let end = end.unwrap_or(extent);
        if enabled.0 {
            let condition = self.comparison(region, BinaryOp::Ge, start, zero, span);
            self.check(region, condition, CheckReason::IndexBound, span);
        }
        if enabled.1 {
            let condition = self.comparison(region, BinaryOp::Le, start, end, span);
            self.check(region, condition, CheckReason::RangeOrder, span);
        }
        if enabled.2 {
            let condition = self.comparison(region, BinaryOp::Le, end, extent, span);
            self.check(region, condition, CheckReason::IndexBound, span);
        }
    }

    fn lower_expr(
        &mut self,
        region: RegionId,
        expression: &ir::Expr,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        match &expression.kind {
            ir::ExprKind::Local(local) => Ok(self.local(*local)),
            ir::ExprKind::Literal(literal) => {
                let constant = match literal {
                    ir::Literal::Int(value) => Constant::Int(*value),
                    ir::Literal::Float(value) => Constant::Float(*value),
                    ir::Literal::Bool(value) => Constant::Bool(*value),
                };
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Primitive(PrimitiveId::Constant(constant)),
                    Vec::new(),
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Dimension(ordinal) => {
                let symbolic = self.shapes
                    [usize::try_from(*ordinal).expect("dimension ordinal does not fit usize")];
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Primitive(PrimitiveId::Symbolic(symbolic)),
                    Vec::new(),
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::PlaneView { base, plane } => {
                let base = self.lower_expr(region, base)?;
                let mut ty = self.semantic_type(&expression.ty);
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: ViewTransform::Plane { plane: *plane },
                    };
                }
                Ok(self.emit(
                    region,
                    NodeKind::View(ViewTransform::Plane { plane: *plane }),
                    vec![base],
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Primitive { id, operands } => {
                self.lower_primitive(region, id, operands, expression)
            }
            ir::ExprKind::Atomic {
                op,
                place,
                indices,
                value,
                authority,
            } => {
                let ir::ExprKind::Local(binding) = &place.kind else {
                    panic!("checked atomic authority is not bound to a local place")
                };
                let ir::Place::Element {
                    root: expected_root,
                    indices: expected_indices,
                } = authority.region()
                else {
                    panic!("checked atomic authority is not bound to an element region")
                };
                let expected_points = expected_indices
                    .iter()
                    .map(|index| match index {
                        ir::Index::Point { value, .. } => value,
                        ir::Index::Range { .. } => {
                            panic!("checked atomic authority contains a range region")
                        }
                    })
                    .collect::<Vec<_>>();
                assert!(expected_points.into_iter().eq(indices.iter()));
                let actual = self.local(*binding);
                let expected = self.local(*expected_root);
                if self.region_storage_origin(region, actual, &mut BTreeSet::new())
                    != self.region_storage_origin(region, expected, &mut BTreeSet::new())
                    && actual != expected
                {
                    panic!("checked atomic authority is bound to different storage")
                }
                let checked_participants = authority
                    .participants()
                    .iter()
                    .map(|participant| {
                        *self
                            .participant_bindings
                            .get(participant)
                            .unwrap_or_else(|| {
                                panic!("checked atomic authority names an inactive participant")
                            })
                    })
                    .collect::<Vec<_>>();
                assert!(checked_participants
                    .iter()
                    .all(|participant| self.parallel_participants.contains(participant)));
                assert!(matches!(authority.order(), ir::CheckedAtomicOrder::Relaxed));
                match authority.scope() {
                    ir::CheckedAtomicScope::Participant => {
                        assert!(checked_participants.is_empty())
                    }
                    ir::CheckedAtomicScope::Participants(participants) => {
                        let scope =
                            participants
                                .iter()
                                .map(|participant| {
                                    *self.participant_bindings.get(participant).unwrap_or_else(|| {
                                    panic!("checked atomic scope names an inactive participant")
                                })
                                })
                                .collect::<Vec<_>>();
                        assert_eq!(scope, checked_participants);
                    }
                }
                assert!(matches!(
                    authority.publication(),
                    ir::CheckedAtomicPublication::CommandCompletion
                ));
                let representation = match &self.values[actual.index()].ty {
                    SemanticType::Tensor(tensor) => tensor.representation,
                    _ => panic!("checked atomic place is not a tensor"),
                };
                let dtype = crate::registry::representation_info(representation).decoded;
                let semantic_outcome = match authority.outcome() {
                    ir::CheckedAssociationOutcome::Exact => AssociationOutcome::Exact,
                    ir::CheckedAssociationOutcome::Reassociated { accumulator } => {
                        assert_eq!(accumulator, dtype);
                        AssociationOutcome::Reassociated { accumulator }
                    }
                };
                let expected_outcome = match op {
                    crate::intrinsics::AtomicOp::Add if dtype.is_float() => {
                        AssociationOutcome::Reassociated { accumulator: dtype }
                    }
                    crate::intrinsics::AtomicOp::Add
                    | crate::intrinsics::AtomicOp::Max
                    | crate::intrinsics::AtomicOp::Min => AssociationOutcome::Exact,
                };
                assert_eq!(semantic_outcome, expected_outcome);
                let participants = if checked_participants.is_empty() {
                    ParticipantDomain::Single
                } else {
                    ParticipantDomain::Parallel(checked_participants.into_boxed_slice())
                };
                let mut inputs = vec![actual];
                let mut semantic_indices = Vec::with_capacity(indices.len());
                for index in indices {
                    let index = self.lower_expr(region, index)?;
                    semantic_indices.push(index);
                    inputs.push(index);
                }
                let capability = AtomicCapability::checked(
                    actual,
                    semantic_indices,
                    participants,
                    semantic_outcome,
                );
                inputs.push(self.lower_expr(region, value)?);
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Atomic {
                        op: *op,
                        capability,
                    },
                    inputs,
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Intrinsic { id, args } => {
                let signature = crate::registry::intrinsic_signature(*id);
                match signature.execution {
                    crate::registry::IntrinsicExecution::WithinEnclosingParallel
                        if self.parallel_depth == 0 =>
                    {
                        return Err(SourceDiagnostic {
                            path: self.module.sources.files()[self.definition.file]
                                .path
                                .clone(),
                            span: expression.span,
                            message: format!(
                                "intrinsic `{}` requires an enclosing parallel loop",
                                signature.name
                            ),
                        });
                    }
                    crate::registry::IntrinsicExecution::WholeTensor { .. }
                        if self.parallel_depth != 0 =>
                    {
                        return Err(SourceDiagnostic {
                            path: self.module.sources.files()[self.definition.file]
                                .path
                                .clone(),
                            span: expression.span,
                            message: format!(
                                "whole-tensor intrinsic `{}` cannot be nested in a parallel loop",
                                signature.name
                            ),
                        });
                    }
                    crate::registry::IntrinsicExecution::WithinEnclosingParallel
                    | crate::registry::IntrinsicExecution::WholeTensor { .. } => {}
                }
                let mut inputs = Vec::new();
                for argument in args {
                    inputs.push(self.lower_expr(region, argument)?);
                }
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Intrinsic(*id),
                    inputs,
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Call { call, args } => self.lower_call(region, call, args, expression),
        }
    }

    fn lower_primitive(
        &mut self,
        region: RegionId,
        primitive: &PrimitiveId,
        operands: &[ir::Expr],
        expression: &ir::Expr,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        let mut inputs = Vec::new();
        for operand in operands {
            inputs.push(self.lower_expr(region, operand)?);
        }
        let mut ty = if let PrimitiveId::SliceView { indices } = primitive {
            let base = inputs[0];
            let SemanticType::Tensor(base_tensor) = self.values[base.index()].ty.clone() else {
                panic!("checked slice base is not a tensor")
            };
            let representation = match &expression.ty {
                ValueType::Tensor(tensor) => self.resolve_elem(&tensor.elem),
                _ => panic!("checked slice result is not a tensor"),
            };
            let mut operand = 1usize;
            let mut axes = Vec::new();
            for (axis, index) in indices.iter().enumerate() {
                match index {
                    crate::intrinsics::IndexSlot::Point => {
                        operand += 1;
                    }
                    crate::intrinsics::IndexSlot::Range { start, end } => {
                        let start = if *start {
                            let value = self.runtime_int(inputs[operand]);
                            operand += 1;
                            value
                        } else {
                            self.builder.arena.int(0)
                        };
                        let end = if *end {
                            let value = self.runtime_int(inputs[operand]);
                            operand += 1;
                            value
                        } else {
                            self.builder.arena.int_from_nat(base_tensor.axes[axis])
                        };
                        let width = self.builder.arena.int_sub(end, start);
                        axes.push(self.builder.arena.nat_from_int(width));
                    }
                }
            }
            axes.extend(base_tensor.axes[indices.len()..].iter().copied());
            if let ValueType::Tensor(source_tensor) = &expression.ty {
                for (source_axis, semantic_axis) in source_tensor.axes.iter().zip(&axes) {
                    if let NodeView::Symbol(symbol) =
                        self.definition.arena.view(AnyExpr::Int(*source_axis))
                    {
                        if !self.symbols.contains_key(&symbol) {
                            let semantic_axis = self.builder.arena.int_from_nat(*semantic_axis);
                            self.symbols.insert(symbol, semantic_axis);
                        }
                    }
                }
            }
            SemanticType::Tensor(TensorSemantics {
                representation,
                axes,
                storage: TensorStorage::Computed,
            })
        } else {
            self.semantic_type(&expression.ty)
        };
        if matches!(primitive, PrimitiveId::TensorAlloc) {
            if let SemanticType::Tensor(tensor) = &ty {
                if crate::registry::representation_info(tensor.representation).access
                    != crate::registry::RepresentationAccess::ReadWrite
                {
                    return Err(SourceDiagnostic {
                        path: self.module.sources.files()[self.definition.file]
                            .path
                            .clone(),
                        span: expression.span,
                        message: format!(
                            "representation `{}` is decode-only and cannot be allocated as writable logical storage",
                            crate::registry::representation_info(tensor.representation).name
                        ),
                    });
                }
            }
        }
        if matches!(primitive, PrimitiveId::RepresentationConvert(_)) {
            let [input] = inputs.as_slice() else {
                panic!("checked representation conversion has wrong arity")
            };
            let SemanticType::Tensor(source) = &self.values[input.index()].ty else {
                panic!("checked representation conversion source is not a tensor")
            };
            let SemanticType::Tensor(destination) = &mut ty else {
                panic!("checked representation conversion result is not a tensor")
            };
            let Some(conversion) = crate::registry::representation_conversion(
                source.representation,
                destination.representation,
            ) else {
                return Err(SourceDiagnostic {
                    path: self.module.sources.files()[self.definition.file]
                        .path
                        .clone(),
                    span: expression.span,
                    message: format!(
                        "no exact representation conversion is registered from `{}` to `{}`",
                        crate::registry::representation_info(source.representation).name,
                        crate::registry::representation_info(destination.representation).name,
                    ),
                });
            };
            destination.storage = TensorStorage::Owned;
            return Ok(self.emit(
                region,
                NodeKind::RepresentationConvert {
                    conversion: conversion.id,
                },
                vec![*input],
                vec![ty],
                expression.span,
            )[0]);
        }
        match primitive {
            PrimitiveId::Binary(BinaryOp::Div | BinaryOp::Rem)
                if matches!(expression.ty, ValueType::Scalar(DType::I32 | DType::U32)) =>
            {
                let ValueType::Scalar(dtype) = &expression.ty else {
                    unreachable!()
                };
                let dtype = *dtype;
                let zero = self.scalar_typed_constant(region, dtype, 0, expression.span);
                let nonzero =
                    self.comparison(region, BinaryOp::Ne, inputs[1], zero, expression.span);
                self.check(region, nonzero, CheckReason::DivideByZero, expression.span);
                if dtype == DType::I32 {
                    let minimum = self.scalar_typed_constant(
                        region,
                        dtype,
                        i64::from(i32::MIN),
                        expression.span,
                    );
                    let negative_one =
                        self.scalar_typed_constant(region, dtype, -1, expression.span);
                    let lhs_safe =
                        self.comparison(region, BinaryOp::Ne, inputs[0], minimum, expression.span);
                    let rhs_safe = self.comparison(
                        region,
                        BinaryOp::Ne,
                        inputs[1],
                        negative_one,
                        expression.span,
                    );
                    let safe = self.emit(
                        region,
                        NodeKind::Primitive(PrimitiveId::Binary(BinaryOp::Or)),
                        vec![lhs_safe, rhs_safe],
                        vec![SemanticType::Scalar(DType::Bool)],
                        expression.span,
                    )[0];
                    self.check(
                        region,
                        safe,
                        CheckReason::SignedDivisionOverflow,
                        expression.span,
                    );
                }
            }
            PrimitiveId::ElementRead { arity } => {
                let base = inputs[0];
                for axis in 0..usize::try_from(*arity).expect("index arity does not fit usize") {
                    self.emit_point_check(region, base, axis, inputs[axis + 1], expression.span);
                }
            }
            PrimitiveId::SliceView { indices } => {
                let base = inputs[0];
                let mut operand = 1;
                for (axis, slot) in indices.iter().enumerate() {
                    match slot {
                        crate::intrinsics::IndexSlot::Point => {
                            self.emit_point_check(
                                region,
                                base,
                                axis,
                                inputs[operand],
                                expression.span,
                            );
                            operand += 1;
                        }
                        crate::intrinsics::IndexSlot::Range { start, end } => {
                            let start_value = if *start {
                                let value = inputs[operand];
                                operand += 1;
                                Some(value)
                            } else {
                                None
                            };
                            let end_value = if *end {
                                let value = inputs[operand];
                                operand += 1;
                                Some(value)
                            } else {
                                None
                            };
                            self.emit_range_checks(
                                region,
                                base,
                                axis,
                                start_value,
                                end_value,
                                (*start, *start || *end, *end),
                                expression.span,
                            );
                        }
                    }
                }
            }
            PrimitiveId::Reduce {
                op:
                    crate::intrinsics::ReduceOp::Max
                    | crate::intrinsics::ReduceOp::Min
                    | crate::intrinsics::ReduceOp::Argmax,
                axis,
                ..
            } => {
                let extent = self.extent_value(region, inputs[0], *axis as usize, expression.span);
                let zero = self.scalar_constant(region, 0, expression.span);
                let nonempty = self.comparison(region, BinaryOp::Gt, extent, zero, expression.span);
                self.check(
                    region,
                    nonempty,
                    CheckReason::Custom("reduction axis must be nonempty".to_owned()),
                    expression.span,
                );
            }
            PrimitiveId::Constant(_)
            | PrimitiveId::Symbolic(_)
            | PrimitiveId::TuplePack
            | PrimitiveId::TupleGet(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Unary(_)
            | PrimitiveId::Binary(_)
            | PrimitiveId::Cast(_)
            | PrimitiveId::Math(_)
            | PrimitiveId::Select
            | PrimitiveId::TensorAlloc
            | PrimitiveId::Fill(_)
            | PrimitiveId::Materialize
            | PrimitiveId::Clone
            | PrimitiveId::Load
            | PrimitiveId::RepresentationConvert(_)
            | PrimitiveId::Decode
            | PrimitiveId::Transpose
            | PrimitiveId::Reshape
            | PrimitiveId::Extent { .. }
            | PrimitiveId::Atomic { .. }
            | PrimitiveId::Reduce {
                op: crate::intrinsics::ReduceOp::Sum,
                ..
            } => {}
        }
        let kind = match primitive {
            PrimitiveId::SliceView { indices } => {
                let base = inputs[0];
                let mut scalars = inputs[1..].iter().copied();
                let mut axes = Vec::with_capacity(indices.len());
                for index in indices {
                    match index {
                        crate::intrinsics::IndexSlot::Point => {
                            axes.push(SliceAxis::Point(ScalarRef::Value(
                                scalars.next().expect("checked slice omitted point operand"),
                            )))
                        }
                        crate::intrinsics::IndexSlot::Range { start, end } => {
                            let start = start.then(|| {
                                ScalarRef::Value(
                                    scalars.next().expect("checked slice omitted range start"),
                                )
                            });
                            let end = end.then(|| {
                                ScalarRef::Value(
                                    scalars.next().expect("checked slice omitted range end"),
                                )
                            });
                            if start.is_none() && end.is_none() {
                                axes.push(SliceAxis::Full);
                            } else {
                                axes.push(SliceAxis::Range { start, end });
                            }
                        }
                    }
                }
                let transform = ViewTransform::Slice { axes };
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: transform.clone(),
                    };
                }
                NodeKind::View(transform)
            }
            PrimitiveId::Transpose => {
                let base = inputs[0];
                let rank = match &self.values[base.index()].ty {
                    SemanticType::Tensor(t) => t.axes.len(),
                    _ => 0,
                };
                let transform = ViewTransform::Transpose {
                    permutation: (0..rank).rev().map(|i| i as u32).collect(),
                };
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: transform.clone(),
                    };
                }
                NodeKind::View(transform)
            }
            PrimitiveId::Reshape => {
                let base = inputs[0];
                let axes = match &ty {
                    SemanticType::Tensor(t) => t.axes.clone(),
                    _ => Vec::new(),
                };
                let transform = ViewTransform::Reshape { axes };
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: transform.clone(),
                    };
                }
                NodeKind::View(transform)
            }
            PrimitiveId::TensorAlloc
            | PrimitiveId::Fill(_)
            | PrimitiveId::Materialize
            | PrimitiveId::Clone
            | PrimitiveId::RepresentationConvert(_) => {
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::Owned;
                }
                self.primitive_kind(primitive, &ty)
            }
            PrimitiveId::Load
            | PrimitiveId::Decode
            | PrimitiveId::ElementRead { .. }
            | PrimitiveId::Reduce { .. } => self.primitive_kind(primitive, &ty),
            PrimitiveId::Atomic { .. } => {
                panic!("checked atomic primitive bypassed its authority-bearing node")
            }
            PrimitiveId::Constant(_)
            | PrimitiveId::Symbolic(_)
            | PrimitiveId::TuplePack
            | PrimitiveId::TupleGet(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Unary(_)
            | PrimitiveId::Binary(_)
            | PrimitiveId::Cast(_)
            | PrimitiveId::Math(_)
            | PrimitiveId::Select
            | PrimitiveId::Extent { .. } => self.primitive_kind(primitive, &ty),
        };
        // Allocation extents and the `like` operand of a fill are evaluated
        // above so their symbolic shape information is transferred into the
        // result type. They are not runtime data dependencies of the storage
        // operation itself: Alloc and Fill carry their complete operation in
        // the output semantics / node kind and therefore intentionally have
        // no semantic inputs.
        let node_inputs = if matches!(primitive, PrimitiveId::TensorAlloc | PrimitiveId::Fill(_)) {
            Vec::new()
        } else {
            inputs
        };
        Ok(self.emit(region, kind, node_inputs, vec![ty], expression.span)[0])
    }

    fn lower_call(
        &mut self,
        region: RegionId,
        call: &ir::Call,
        args: &[ir::Expr],
        expression: &ir::Expr,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        let mut evaluated = Vec::with_capacity(args.len());
        for argument in args {
            evaluated.push(self.lower_expr(region, argument)?);
        }
        let mut specs = Vec::new();
        for candidate in &call.candidates {
            if !candidate
                .requires_elems
                .iter()
                .all(|(caller_parameter, required)| {
                    self.elements.get(caller_parameter).copied()
                        == Some(self.resolve_elem(required))
                })
            {
                continue;
            }
            let shape_args = candidate
                .shape_args
                .iter()
                .map(|shape| self.transfer_int(*shape))
                .collect();
            let mut elements = self.elements.clone();
            for (name, elem) in &candidate.elem_args {
                elements.insert(name.clone(), self.resolve_elem(elem));
            }
            specs.push(CandidateSpec {
                definition: candidate.definition,
                shape_args,
                elements,
                call_site_proved: candidate.applicability_proven,
            });
            // Every candidate in a checked family has the same canonical
            // parameter order. The first candidate supplies the call's
            // argument permutation below.
        }
        let family =
            instantiate_family(self.builder, self.module, call.family.index(), specs, false)?;
        let order = &call
            .candidates
            .first()
            .unwrap_or_else(|| panic!("checked call has no candidates"))
            .arg_order;
        let mut inputs = Vec::new();
        for index in order {
            let value = evaluated[*index];
            let parameter = self.values[value.index()].ty.clone();
            self.flatten_call_argument(region, value, &parameter, &mut inputs, expression.span);
        }
        let ty = self.semantic_type(&expression.ty);
        let mut leaf_types = Vec::new();
        Self::flatten_call_type(&ty, &mut leaf_types);
        let leaves = self.emit(
            region,
            NodeKind::Call { family },
            inputs,
            leaf_types,
            expression.span,
        );
        let mut leaves = leaves.into_iter();
        let value = self.reconstruct_call_value(region, &ty, &mut leaves, expression.span);
        assert!(
            leaves.next().is_none(),
            "checked call result reconstruction left an unbound contract leaf"
        );
        Ok(value)
    }

    fn flatten_call_argument(
        &mut self,
        region: RegionId,
        value: SemanticValueId,
        ty: &SemanticType,
        leaves: &mut Vec<SemanticValueId>,
        span: crate::span::Span,
    ) {
        match ty {
            SemanticType::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    let component = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![item.clone()],
                        span,
                    )[0];
                    self.flatten_call_argument(region, component, item, leaves, span);
                }
            }
            SemanticType::Void => {}
            _ => leaves.push(value),
        }
    }

    /// Function contracts and Call nodes carry results in canonical leaf
    /// order. Tuples are source structure, reconstructed explicitly after the
    /// call so implementation splicing never has to rediscover a result tree.
    fn flatten_call_type(ty: &SemanticType, leaves: &mut Vec<SemanticType>) {
        match ty {
            SemanticType::Tuple(items) => {
                for item in items {
                    Self::flatten_call_type(item, leaves);
                }
            }
            SemanticType::Void => {}
            leaf => leaves.push(leaf.clone()),
        }
    }

    fn reconstruct_call_value(
        &mut self,
        region: RegionId,
        ty: &SemanticType,
        leaves: &mut impl Iterator<Item = SemanticValueId>,
        span: crate::span::Span,
    ) -> SemanticValueId {
        match ty {
            SemanticType::Tuple(items) => {
                let inputs = items
                    .iter()
                    .map(|item| self.reconstruct_call_value(region, item, leaves, span))
                    .collect();
                self.emit(region, NodeKind::TuplePack, inputs, vec![ty.clone()], span)[0]
            }
            SemanticType::Void => self.emit(
                region,
                NodeKind::TuplePack,
                Vec::new(),
                vec![SemanticType::Tuple(Vec::new())],
                span,
            )[0],
            _ => leaves
                .next()
                .unwrap_or_else(|| panic!("checked call contract omitted a result leaf")),
        }
    }

    fn region_parameter_tree(
        &mut self,
        region: RegionId,
        ty: &SemanticType,
        span: crate::span::Span,
        leaves: &mut Vec<SemanticValueId>,
    ) -> SemanticValueId {
        match ty {
            SemanticType::Tuple(items) => {
                let inputs = items
                    .iter()
                    .map(|item| self.region_parameter_tree(region, item, span, leaves))
                    .collect();
                self.emit(region, NodeKind::TuplePack, inputs, vec![ty.clone()], span)[0]
            }
            SemanticType::Void => self.emit(
                region,
                NodeKind::TuplePack,
                Vec::new(),
                vec![SemanticType::Tuple(Vec::new())],
                span,
            )[0],
            leaf => {
                let value = self.region_parameter(region, leaf.clone(), span);
                leaves.push(value);
                value
            }
        }
    }

    fn lower_if(
        &mut self,
        parent_region: RegionId,
        condition: &ir::Expr,
        then_body: &ir::Block,
        else_body: &ir::Block,
    ) -> Result<(), SourceDiagnostic> {
        let condition_value = self.lower_expr(parent_region, condition)?;
        let before = self.locals.clone();

        let then_region = self.reserve_region(RegionKind::Then);
        let else_region = self.reserve_region(RegionKind::Else);
        let predecessor = self
            .last_event
            .get(&parent_region)
            .cloned()
            .unwrap_or_default();
        self.last_event.insert(then_region, predecessor.clone());
        self.last_event.insert(else_region, predecessor);
        let mut captures = Vec::new();
        let mut then_locals = before.clone();
        let mut else_locals = before.clone();
        for (ordinal, value) in before.iter().copied().enumerate() {
            let Some(value) = value else { continue };
            let ty = self.values[value.index()].ty.clone();
            let span = self.definition.body.locals[ordinal].span;
            let mut parent_leaves = Vec::new();
            self.flatten_call_argument(parent_region, value, &ty, &mut parent_leaves, span);
            let mut then_leaves = Vec::new();
            let then_parameter =
                self.region_parameter_tree(then_region, &ty, span, &mut then_leaves);
            let mut else_leaves = Vec::new();
            let else_parameter =
                self.region_parameter_tree(else_region, &ty, span, &mut else_leaves);
            assert_eq!(parent_leaves.len(), then_leaves.len());
            assert_eq!(parent_leaves.len(), else_leaves.len());
            then_locals[ordinal] = Some(then_parameter);
            else_locals[ordinal] = Some(else_parameter);
            captures.push((
                ordinal,
                value,
                ty,
                parent_leaves,
                then_parameter,
                else_parameter,
            ));
        }
        self.locals = then_locals;
        let then_result = self.lower_block(then_region, then_body)?;
        assert!(
            then_result.is_empty(),
            "checked branch contains an early return"
        );
        let then_after = self.locals.clone();

        self.locals = else_locals;
        let else_result = self.lower_block(else_region, else_body)?;
        assert!(
            else_result.is_empty(),
            "checked branch contains an early return"
        );
        let else_after = self.locals.clone();

        let mut changed = Vec::new();
        for (ordinal, _, ty, _, then_parameter, else_parameter) in &captures {
            let then_value = then_after[*ordinal];
            let else_value = else_after[*ordinal];
            if then_value != Some(*then_parameter) || else_value != Some(*else_parameter) {
                let then_value =
                    then_value.unwrap_or_else(|| panic!("checked branch loses a live local"));
                let else_value =
                    else_value.unwrap_or_else(|| panic!("checked branch loses a live local"));
                if matches!(ty, SemanticType::Tensor(_)) {
                    let then_origin =
                        self.region_storage_origin(then_region, then_value, &mut BTreeSet::new());
                    let else_origin =
                        self.region_storage_origin(else_region, else_value, &mut BTreeSet::new());
                    if then_origin.is_some()
                        && then_origin
                            == self.region_storage_origin(
                                then_region,
                                *then_parameter,
                                &mut BTreeSet::new(),
                            )
                        && else_origin.is_some()
                        && else_origin
                            == self.region_storage_origin(
                                else_region,
                                *else_parameter,
                                &mut BTreeSet::new(),
                            )
                    {
                        // Both arms mutate the captured storage in place. The
                        // child semantic events carry the mutation; no tensor
                        // SSA value must cross the control-flow join.
                        continue;
                    }
                }
                changed.push((*ordinal, then_value, else_value));
            }
        }
        let mut then_results = Vec::new();
        let mut else_results = Vec::new();
        let mut output_types = Vec::new();
        for (_, then_value, else_value) in &changed {
            let ty = self.values[then_value.index()].ty.clone();
            self.flatten_call_argument(
                then_region,
                *then_value,
                &ty,
                &mut then_results,
                condition.span,
            );
            self.flatten_call_argument(
                else_region,
                *else_value,
                &ty,
                &mut else_results,
                condition.span,
            );
            Self::flatten_call_type(&ty, &mut output_types);
        }
        self.region_mut(then_region).results = then_results;
        self.region_mut(else_region).results = else_results;
        self.locals = before;
        let mut inputs = vec![condition_value];
        inputs.extend(
            captures
                .iter()
                .flat_map(|(_, _, _, leaves, _, _)| leaves.iter().copied()),
        );
        let outputs = self.emit(
            parent_region,
            NodeKind::If {
                then: then_region,
                otherwise: else_region,
            },
            inputs,
            output_types,
            condition.span,
        );
        let mut outputs = outputs.into_iter();
        for (ordinal, then_value, _) in changed {
            let ty = self.values[then_value.index()].ty.clone();
            let value =
                self.reconstruct_call_value(parent_region, &ty, &mut outputs, condition.span);
            self.locals[ordinal] = Some(value);
        }
        assert!(
            outputs.next().is_none(),
            "branch join left an unbound result leaf"
        );
        Ok(())
    }

    fn lower_loop(
        &mut self,
        parent_region: RegionId,
        kind: ir::LoopKind,
        binder: ir::LocalId,
        start: &ir::Expr,
        end: &ir::Expr,
        body: &ir::Block,
    ) -> Result<(), SourceDiagnostic> {
        let start_value = self.lower_expr(parent_region, start)?;
        let end_value = self.lower_expr(parent_region, end)?;
        let start_expression = self.transfer_int(
            start
                .sym
                .expect("checked loop start has no symbolic integer expression"),
        );
        let end_expression = self.transfer_int(
            end.sym
                .expect("checked loop end has no symbolic integer expression"),
        );
        let start_value = self.loop_index(parent_region, start_value, start_expression, start.span);
        let end_value = self.loop_index(parent_region, end_value, end_expression, end.span);
        let before = self.locals.clone();
        let body_region = self.reserve_region(RegionKind::Root);
        let predecessor = self
            .last_event
            .get(&parent_region)
            .cloned()
            .unwrap_or_default();
        self.last_event.insert(body_region, predecessor);
        let binder_source = self.definition.body.locals[binder.index()].clone();
        let binder_ty = self.semantic_type(&binder_source.ty);
        let binder_value = self.region_parameter(body_region, binder_ty, binder_source.span);
        let (expression_binder, binder_symbol, binder_int) = self.builder.arena.loop_binder();
        if let Some(source_symbol) = self.definition.body.locals[binder.index()].symbol {
            self.symbols.insert(source_symbol, binder_int);
        }
        let mut child = before.clone();
        child[binder.index()] = Some(binder_value);
        let mut captures = Vec::new();
        for (ordinal, value) in before.iter().copied().enumerate() {
            if ordinal == binder.index() {
                continue;
            }
            if let Some(value) = value {
                let ty = self.values[value.index()].ty.clone();
                let span = self.definition.body.locals[ordinal].span;
                let mut parent_leaves = Vec::new();
                self.flatten_call_argument(parent_region, value, &ty, &mut parent_leaves, span);
                let mut parameter_leaves = Vec::new();
                let parameter =
                    self.region_parameter_tree(body_region, &ty, span, &mut parameter_leaves);
                assert_eq!(parent_leaves.len(), parameter_leaves.len());
                child[ordinal] = Some(parameter);
                captures.push((
                    ordinal,
                    value,
                    ty,
                    parent_leaves,
                    parameter,
                    parameter_leaves,
                ));
            }
        }
        self.region_mut(body_region).kind = RegionKind::LoopBody {
            binder: BinderId::new(body_region, 0),
            expression_binder,
            binder_symbol,
            binder_value,
        };
        self.locals = child;
        let parallel = matches!(kind, ir::LoopKind::Independent);
        if parallel {
            self.parallel_depth = self
                .parallel_depth
                .checked_add(1)
                .unwrap_or_else(|| panic!("parallel loop nesting exceeds u32::MAX"));
            let participant = BinderId::new(body_region, 0);
            self.parallel_participants.push(participant);
            assert!(self
                .participant_bindings
                .insert(binder, participant)
                .is_none());
        }
        let returned = self.lower_block(body_region, body);
        if parallel {
            let popped = self
                .parallel_participants
                .pop()
                .unwrap_or_else(|| panic!("parallel participant stack is unbalanced"));
            assert_eq!(popped, BinderId::new(body_region, 0));
            assert_eq!(
                self.participant_bindings.remove(&binder),
                Some(BinderId::new(body_region, 0))
            );
            self.parallel_depth -= 1;
        }
        let returned = returned?;
        assert!(
            returned.is_empty(),
            "checked loop body contains an early return"
        );
        let after = self.locals.clone();
        let mut carries = Vec::new();
        let mut output_types = Vec::new();
        let mut yielded = Vec::new();
        let all_capture_values = captures
            .iter()
            .flat_map(|(_, _, _, leaves, _, _)| leaves.iter().copied())
            .collect::<Vec<_>>();
        let mut final_leaf_sources = Vec::new();
        self.locals = before;
        for (ordinal, _, ty, initial_leaves, _parameter, parameter_leaves) in &captures {
            let next = after[*ordinal].unwrap_or_else(|| panic!("checked loop loses a live local"));
            let mut next_leaves = Vec::new();
            self.flatten_call_argument(
                body_region,
                next,
                ty,
                &mut next_leaves,
                self.definition.body.locals[*ordinal].span,
            );
            assert_eq!(next_leaves.len(), parameter_leaves.len());
            let mut source_indices = Vec::with_capacity(next_leaves.len());
            for ((initial, parameter), next) in
                initial_leaves.iter().zip(parameter_leaves).zip(next_leaves)
            {
                if next == *parameter {
                    source_indices.push(None);
                    continue;
                }
                if parallel {
                    let next_storage =
                        self.region_storage_origin(body_region, next, &mut BTreeSet::new());
                    let parameter_storage =
                        self.region_storage_origin(body_region, *parameter, &mut BTreeSet::new());
                    let aliases_parameter_storage =
                        matches!(self.values[next.index()].ty, SemanticType::Tensor(_))
                            && next_storage.is_some()
                            && next_storage == parameter_storage;
                    if aliases_parameter_storage {
                        source_indices.push(None);
                        continue;
                    }
                    return Err(SourceDiagnostic {
                        path: self.module.sources.files()[self.definition.file].path.clone(),
                        span: self.definition.body.locals[*ordinal].span,
                        message: "a `parallel for` body cannot carry reassigned state; use an explicit reduction or write through a disjoint tensor view".to_owned(),
                    });
                }
                let result = SemanticValueId::new(
                    self.id,
                    u32::try_from(self.values.len() + output_types.len())
                        .expect("function has more than u32::MAX values"),
                );
                carries.push(Carry {
                    initial: *initial,
                    parameter: *parameter,
                    yielded: next,
                    result,
                });
                yielded.push(next);
                output_types.push(self.values[next.index()].ty.clone());
                source_indices.push(Some(output_types.len() - 1));
            }
            final_leaf_sources.push((*ordinal, ty.clone(), initial_leaves.clone(), source_indices));
        }
        self.region_mut(body_region).results = yielded;
        let mut inputs = vec![start_value, end_value];
        inputs.extend(all_capture_values.iter().copied());
        let semantic_kind = match kind {
            ir::LoopKind::Ordered => LoopKind::Ordered,
            ir::LoopKind::Independent => LoopKind::Parallel,
        };
        let outputs = self.emit(
            parent_region,
            NodeKind::Loop {
                kind: semantic_kind,
                body: body_region,
                carries: carries.clone(),
            },
            inputs,
            output_types,
            start.span,
        );
        for (carry, output) in carries.iter().zip(&outputs) {
            assert_eq!(
                carry.result, *output,
                "loop carry result allocation diverged from node outputs"
            );
        }
        for (ordinal, ty, initial, sources) in final_leaf_sources {
            let leaves = sources
                .into_iter()
                .enumerate()
                .map(|(leaf, source)| source.map_or(initial[leaf], |output| outputs[output]))
                .collect::<Vec<_>>();
            let mut leaves = leaves.into_iter();
            let value = self.reconstruct_call_value(parent_region, &ty, &mut leaves, start.span);
            assert!(
                leaves.next().is_none(),
                "loop join left an unbound result leaf"
            );
            self.locals[ordinal] = Some(value);
        }
        Ok(())
    }

    /// Crosses the checked scalar/index boundary for loop scheduling.
    ///
    /// Surface range endpoints are `i32` symbolic expressions because they
    /// participate in ordinary checked arithmetic. A semantic loop, however,
    /// is scheduled over the non-negative index domain. Range checking has
    /// already proved non-negativity before this point, so materialize that
    /// proof as an explicit symbolic conversion instead of allowing scalar
    /// values to enter index-only scheduling APIs.
    fn loop_index(
        &mut self,
        region: RegionId,
        scalar: SemanticValueId,
        expression: IntExpr,
        span: crate::span::Span,
    ) -> SemanticValueId {
        let value = self.builder.arena.nat_from_int(expression);
        let one = self.builder.arena.nat(1);
        let bound = self.builder.arena.nat_add(value, one);
        self.emit(
            region,
            NodeKind::Primitive(PrimitiveId::Cast(DType::U32)),
            vec![scalar],
            vec![SemanticType::Index { bound }],
            span,
        )[0]
    }
}

#[cfg(test)]
mod dimension_inference_tests {
    use super::*;

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
