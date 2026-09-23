//! `PreparedKernel<T, H>` and invocation validation/selection (spec §10.4,
//! §12).
//!
//! Fields and constructor are private; only completed candidate evaluation
//! creates one. Invocation selection applies the evaluator-produced function
//! and verifies the returned candidate remains applicable. A bad index or
//! false guard after validation is an internal bug.

use crate::candidate_domain::NonEmpty;
use crate::errors::InvocationError;
use crate::evaluation_session::{PreparedCandidateId, PreparedPortfolio};
use crate::executable::ExecutableVariant;
use seismic_lang::entry::ParameterKind;
use seismic_lang::entry::{CallSchema, CompiledDimensionInferencePlan, SemanticEventManifest};
use seismic_lang::expr::compiled::{
    CompiledDuration, CompiledNat, CompiledPredicate, InvocationValues,
};
use seismic_lang::expr::{PartialAssignment, SymbolValue};
use seismic_lang::ids::{ModuleHash, RepresentationId, StableEntryId};
use std::sync::Arc;

/// Opaque, immutable invocation-to-candidate function. Evaluators may derive
/// it by any method; runtime only applies the completed mapping.
pub struct SelectionFunction {
    candidate_operands: Box<[PreparedCandidateId]>,
    retained_bytes: u64,
    program: Box<dyn Fn(&InvocationValues) -> CandidateIndex + Send + Sync>,
}

impl std::fmt::Debug for SelectionFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectionFunction").finish_non_exhaustive()
    }
}

impl SelectionFunction {
    pub(crate) fn candidate_operands(&self) -> &[PreparedCandidateId] {
        &self.candidate_operands
    }

    pub(crate) fn retained_metadata_bytes(&self) -> u64 {
        self.retained_bytes
    }

    /// Applies the completed evaluator policy to one validated invocation.
    pub fn apply(&self, values: &InvocationValues) -> CandidateIndex {
        (self.program)(values)
    }

    pub fn analytical_minimum(candidates: PreparedPortfolio<CompiledDuration>) -> Self {
        let (candidate_operands, candidates) = candidates.into_operands();
        let candidate_operands = candidate_operands.into_vec().into_boxed_slice();
        let candidates = candidates.into_vec();
        let retained_bytes = (std::mem::size_of::<Self>()
            + candidate_operands.len() * std::mem::size_of::<PreparedCandidateId>()
            + candidates.capacity() * std::mem::size_of::<(CompiledPredicate, CompiledDuration)>()
            + candidates
                .iter()
                .map(|(guard, score)| {
                    guard.retained_metadata_bytes() + score.retained_metadata_bytes()
                })
                .sum::<usize>()) as u64;
        Self {
            candidate_operands,
            retained_bytes,
            program: Box::new(move |values| {
                let index = candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(index, (guard, score))| {
                        guard
                            .evaluate(values)
                            .unwrap_or_else(|error| {
                                panic!("validated invocation could not evaluate candidate guard: {error:?}")
                            })
                            .then(|| {
                                let score = score.evaluate(values).unwrap_or_else(|error| {
                                    panic!("validated invocation could not evaluate analytical decision: {error:?}")
                                });
                                (index, score.upper())
                            })
                    })
                    .min_by_key(|(index, score)| (*score, *index))
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                CandidateIndex(index)
            }),
        }
    }

    /// Builds a total decision from ordered invocation predicates. The first
    /// true predicate whose candidate is applicable wins; candidate zero is
    /// the constructionally universal default. Ordinals are checked against
    /// the admitted portfolio at construction.
    pub fn ordered_decision<P>(
        candidates: PreparedPortfolio<P>,
        cases: Vec<(CompiledPredicate, CandidateIndex)>,
    ) -> Self {
        let (candidate_operands, candidates) = candidates.into_operands();
        let candidate_operands = candidate_operands.into_vec().into_boxed_slice();
        let candidate_guards = candidates
            .into_vec()
            .into_iter()
            .map(|(guard, _)| guard)
            .collect::<Vec<_>>();
        let candidate_count = candidate_guards.len();
        assert!(
            cases
                .iter()
                .all(|(_, index)| index.as_usize() < candidate_count),
            "selection case is out of range"
        );
        let retained_bytes = (std::mem::size_of::<Self>()
            + candidate_operands.len() * std::mem::size_of::<PreparedCandidateId>()
            + candidate_guards.capacity() * std::mem::size_of::<CompiledPredicate>()
            + cases.capacity() * std::mem::size_of::<(CompiledPredicate, CandidateIndex)>()
            + candidate_guards
                .iter()
                .map(CompiledPredicate::retained_metadata_bytes)
                .sum::<usize>()
            + cases
                .iter()
                .map(|(predicate, _)| predicate.retained_metadata_bytes())
                .sum::<usize>()) as u64;
        Self {
            candidate_operands,
            retained_bytes,
            program: Box::new(move |values| {
                cases
                    .iter()
                    .find_map(|(predicate, candidate)| {
                        let requested = predicate
                            .evaluate(values)
                            .unwrap_or_else(|error| {
                                panic!("validated invocation could not evaluate selection decision: {error:?}")
                            });
                        let applicable = candidate_guards[candidate.as_usize()]
                            .evaluate(values)
                            .unwrap_or_else(|error| {
                                panic!("validated invocation could not evaluate candidate guard: {error:?}")
                            });
                        (requested && applicable).then_some(*candidate)
                    })
                    .unwrap_or(CandidateIndex(0))
            }),
        }
    }
}

/// An ordinal into the candidate list owned by the same selection policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CandidateIndex(usize);

impl CandidateIndex {
    pub(crate) const fn from_usize(index: usize) -> Self {
        Self(index)
    }

    pub const fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub struct PreparedKernel<T: seismic_target::TargetFamily, H> {
    semantic_events: Arc<SemanticEventManifest>,
    device: seismic_target::DeviceDescriptionIdentity,
    evaluation: crate::evaluation::EvaluationIdentity,
    invocation: Arc<InvocationContract>,
    selection: SelectionFunction,
    variants: NonEmpty<ExecutableVariant<T, H>>,
    planning_report: crate::planning::PlanningReport,
}

impl<T: seismic_target::TargetFamily, H> PreparedKernel<T, H> {
    /// Packages the preparation-owned policy and its diagnostic report.
    pub(crate) fn prepare(
        semantic_events: Arc<SemanticEventManifest>,
        device: seismic_target::DeviceDescriptionIdentity,
        evaluation: crate::evaluation::EvaluationIdentity,
        invocation: Arc<InvocationContract>,
        selection: SelectionFunction,
        variants: NonEmpty<ExecutableVariant<T, H>>,
        planning_report: crate::planning::PlanningReport,
    ) -> Self {
        Self {
            semantic_events,
            device,
            evaluation,
            invocation,
            selection,
            variants,
            planning_report,
        }
    }

    pub fn entry(&self) -> StableEntryId {
        self.invocation.entry
    }
    pub fn module(&self) -> ModuleHash {
        self.invocation.module
    }
    pub fn schema(&self) -> &CallSchema {
        self.invocation.schema()
    }
    pub fn semantic_event_manifest(&self) -> &SemanticEventManifest {
        &self.semantic_events
    }
    pub fn device_identity(&self) -> &seismic_target::DeviceDescriptionIdentity {
        &self.device
    }
    pub fn evaluation_identity(&self) -> &crate::evaluation::EvaluationIdentity {
        &self.evaluation
    }
    #[doc(hidden)]
    pub fn invocation_contract(&self) -> &InvocationContract {
        &self.invocation
    }
    pub fn variants(&self) -> &NonEmpty<ExecutableVariant<T, H>> {
        &self.variants
    }
    pub fn planning_report(&self) -> &crate::planning::PlanningReport {
        &self.planning_report
    }

    /// The preparation-owned candidate selected at this variant position.
    /// Observation uses the same identity that the selection function retains.
    pub fn candidate_for_variant(&self, variant: VariantIndex) -> PreparedCandidateId {
        *self
            .selection
            .candidate_operands()
            .get(variant.as_usize())
            .expect("selected variant is absent from its preparation")
    }

    /// Deterministic selection among applicable variants.
    #[doc(hidden)]
    pub fn select(&self, values: &InvocationValues) -> VariantIndex {
        internals::select(self, values)
    }
}

/// The position selected from one prepared kernel's immutable variant list.
///
/// Selection does not expose the executable variant or its native handles.
/// Consumers use this ordinal only with the same prepared kernel's immutable
/// variant list before entering the compiler-owned execution path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VariantIndex(usize);

impl VariantIndex {
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

/// A caller-side tensor descriptor, as generated bindings present it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub device: DeviceIdentity,
    pub representation: RepresentationId,
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
    /// Allocation identity and byte range, for alias checks.
    pub allocation: u64,
    pub byte_offset: u64,
    pub byte_len: u64,
}

/// Opaque device identity for `WrongDevice` checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceIdentity(pub u64);

/// A caller-side argument.
#[derive(Clone, Debug, PartialEq)]
pub enum ArgumentValue {
    Tensor(TensorDescriptor),
    F32(f32),
    /// IEEE-754 binary16 bits.  The public crate supplies the typed wrapper;
    /// this private ABI does not silently widen a source `f16` argument.
    F16(u16),
    /// bfloat16 bits.  The public crate supplies the typed wrapper.
    BF16(u16),
    I32(i32),
    U32(u32),
    Bool(bool),
    Index(seismic_lang::expr::BigUint),
    Range {
        start: seismic_lang::expr::BigUint,
        end: seismic_lang::expr::BigUint,
    },
}

/// The complete executable invocation contract. Symbolic schema expressions
/// are compiled exactly once while the entry arena still exists; runtime
/// never retains an arena or attempts to reconstruct shape equations.
#[derive(Debug)]
#[doc(hidden)]
pub struct InvocationContract {
    entry: StableEntryId,
    module: ModuleHash,
    schema: Arc<CallSchema>,
    dimension_inference: CompiledDimensionInferencePlan,
    target_domain: CompiledPredicate,
    parameters: Vec<ParameterContract>,
}

#[derive(Debug)]
enum ParameterContract {
    Tensor { axes: Vec<CompiledNat> },
    Scalar,
    Index { bound: CompiledNat },
    Range { bound: CompiledNat },
}

impl InvocationContract {
    pub fn schema(&self) -> &CallSchema {
        &self.schema
    }
    pub fn entry(&self) -> StableEntryId {
        self.entry
    }
    pub fn module(&self) -> ModuleHash {
        self.module
    }

    /// Compiles the public invocation contract of an entry without creating a
    /// candidate domain. Direct native entry points use this path: they still get
    /// the ordinary Seismic call-boundary validation, but perform no
    /// implementation search, solving, scheduling, or portfolio construction.
    #[doc(hidden)]
    pub fn compile_entry(entry: &seismic_lang::entry::LogicalEntry) -> Self {
        let fixed = PartialAssignment::new();
        let arena = entry.arena();
        let schema = entry.schema();
        let parameters = schema
            .parameters()
            .iter()
            .map(|parameter| match &parameter.kind {
                ParameterKind::Tensor { axes, .. } => ParameterContract::Tensor {
                    axes: axes.iter().map(|axis| arena.compile_nat(*axis)).collect(),
                },
                ParameterKind::Scalar { .. } => ParameterContract::Scalar,
                ParameterKind::Index { bound, .. } => ParameterContract::Index {
                    bound: arena.compile_nat(*bound),
                },
                ParameterKind::Range { bound, .. } => ParameterContract::Range {
                    bound: arena.compile_nat(*bound),
                },
            })
            .collect();
        Self {
            entry: entry.identity(),
            module: entry.module_hash(),
            schema: entry.shared_schema(),
            dimension_inference: schema.compile_dimension_inference(arena, &fixed),
            target_domain: arena.compile_bool(entry.domain().predicate().node()),
            parameters,
        }
    }

    pub(crate) fn compile<T: seismic_target::TargetFamily>(
        domain: &crate::candidate_domain::CandidateDomain<'_, T>,
    ) -> Self {
        let arena = domain.arena();
        let schema = domain.schema();
        let target_domain = domain.target_domain();
        let mut fixed_values = PartialAssignment::new();
        for (symbol, value) in domain.constants().bindings() {
            fixed_values.bind(*symbol, value.clone());
        }
        let fixed = &fixed_values;
        let parameters = schema
            .parameters()
            .iter()
            .map(|parameter| match &parameter.kind {
                ParameterKind::Tensor { axes, .. } => ParameterContract::Tensor {
                    axes: axes
                        .iter()
                        .map(|axis| arena.compile_nat_with(*axis, fixed))
                        .collect(),
                },
                ParameterKind::Scalar { .. } => ParameterContract::Scalar,
                ParameterKind::Index { bound, .. } => ParameterContract::Index {
                    bound: arena.compile_nat_with(*bound, fixed),
                },
                ParameterKind::Range { bound, .. } => ParameterContract::Range {
                    bound: arena.compile_nat_with(*bound, fixed),
                },
            })
            .collect();
        Self {
            entry: domain.entry(),
            module: domain.module(),
            schema: schema.clone(),
            dimension_inference: schema.compile_dimension_inference(&arena, fixed),
            target_domain: arena.compile_bool_with(target_domain.predicate().node(), fixed),
            parameters,
        }
    }
}

/// Validates one invocation against the schema and target domain, binding
/// every call dimension and scalar symbol. This is the single validator
/// generated bindings call; it runs before any allocation (§12.2).
pub fn validate_invocation(
    contract: &InvocationContract,
    device: DeviceIdentity,
    arguments: &[ArgumentValue],
) -> Result<InvocationValues, InvocationError> {
    internals::validate_invocation(contract, device, arguments)
}

mod internals {
    use super::*;

    pub(super) fn select<T: seismic_target::TargetFamily, H>(
        kernel: &PreparedKernel<T, H>,
        values: &InvocationValues,
    ) -> VariantIndex {
        VariantIndex(crate::executable::select_candidate_index(
            &kernel.selection,
            kernel.variants.as_slice(),
            values,
        ))
    }

    pub(super) fn validate_invocation(
        contract: &InvocationContract,
        device: DeviceIdentity,
        arguments: &[ArgumentValue],
    ) -> Result<InvocationValues, InvocationError> {
        let schema = contract.schema();
        if arguments.len() != schema.parameters().len() {
            panic!("generated argument arity does not match its content-addressed call schema");
        }

        let mut values = InvocationValues::new();

        // Observe every input tensor axis in the schema's canonical order,
        // then execute the sealed construction-time elimination plan. There
        // is no runtime search and no caller-supplied shape metadata.
        let mut observations = Vec::new();
        for (parameter, argument) in schema.parameters().iter().zip(arguments) {
            match (&parameter.kind, argument) {
                (ParameterKind::Tensor { axes, .. }, ArgumentValue::Tensor(tensor)) => {
                    if tensor.extents.len() != axes.len() {
                        return Err(InvocationError::ShapeMismatch {
                            parameter: parameter_label(parameter),
                            axis: u32::try_from(tensor.extents.len().min(axes.len()))
                                .unwrap_or(u32::MAX),
                        });
                    }
                    observations.extend_from_slice(&tensor.extents);
                }
                (ParameterKind::Tensor { .. }, _) => {
                    panic!("generated tensor Rust type disagrees with its checked schema")
                }
                _ => {}
            }
        }
        if observations.len() != contract.dimension_inference.observation_count() {
            panic!("sealed dimension observation count disagrees with its call schema");
        }
        if let Err(failure) = contract
            .dimension_inference
            .infer(&observations, &mut values)
        {
            let (parameter, axis) = observation_location(schema, failure.observation())
                .unwrap_or_else(|| panic!("sealed dimension plan names an absent observation"));
            return Err(InvocationError::ShapeMismatch {
                parameter: parameter_label(parameter),
                axis: u32::try_from(axis).unwrap_or(u32::MAX),
            });
        }

        // Bind source scalar symbols with their exact sorts.
        for (parameter, argument) in schema.parameters().iter().zip(arguments) {
            match (&parameter.kind, argument) {
                (ParameterKind::Tensor { .. }, ArgumentValue::Tensor(_)) => {}
                (ParameterKind::Scalar { dtype, symbol }, value) => {
                    let value = scalar_value(*dtype, value).unwrap_or_else(|| {
                        panic!("generated scalar Rust type disagrees with its checked schema")
                    });
                    values.bind(*symbol, value);
                }
                (ParameterKind::Index { symbol, .. }, ArgumentValue::Index(value)) => {
                    values.bind(*symbol, SymbolValue::Nat(value.clone()));
                }
                (
                    ParameterKind::Range { start, end, .. },
                    ArgumentValue::Range {
                        start: first,
                        end: last,
                    },
                ) => {
                    values.bind(*start, SymbolValue::Nat(first.clone()));
                    values.bind(*end, SymbolValue::Nat(last.clone()));
                }
                _ => panic!("generated argument kind disagrees with its checked schema"),
            }
        }

        for ((parameter, argument), expected) in schema
            .parameters()
            .iter()
            .zip(arguments)
            .zip(&contract.parameters)
        {
            match (&parameter.kind, argument, expected) {
                (
                    ParameterKind::Tensor {
                        representation,
                        axes,
                        ..
                    },
                    ArgumentValue::Tensor(tensor),
                    ParameterContract::Tensor {
                        axes: expected_axes,
                    },
                ) => {
                    if tensor.device != device {
                        return Err(InvocationError::WrongDevice {
                            parameter: parameter_label(parameter),
                        });
                    }
                    if tensor.representation != *representation {
                        return Err(InvocationError::WrongRepresentation {
                            parameter: parameter_label(parameter),
                        });
                    }
                    if expected_axes.len() != axes.len() {
                        panic!("compiled tensor-axis contract disagrees with its call schema");
                    }
                    if !valid_tensor_descriptor(tensor) {
                        return Err(InvocationError::InvalidTensorDescriptor {
                            parameter: parameter_label(parameter),
                        });
                    }
                    for (axis, (actual, expected)) in
                        tensor.extents.iter().zip(expected_axes).enumerate()
                    {
                        let expected = expected.evaluate(&values).map_err(|_| {
                            InvocationError::ShapeMismatch {
                                parameter: parameter_label(parameter),
                                axis: u32::try_from(axis).unwrap_or(u32::MAX),
                            }
                        })?;
                        if seismic_lang::expr::BigUint::from(*actual) != expected {
                            return Err(InvocationError::ShapeMismatch {
                                parameter: parameter_label(parameter),
                                axis: u32::try_from(axis).unwrap_or(u32::MAX),
                            });
                        }
                    }
                }
                (ParameterKind::Scalar { .. }, _, ParameterContract::Scalar) => {}
                (
                    ParameterKind::Index { .. },
                    ArgumentValue::Index(value),
                    ParameterContract::Index { bound },
                ) => {
                    let bound = bound.evaluate(&values).unwrap_or_else(|error| {
                        panic!("checked index bound is not total after argument binding: {error:?}")
                    });
                    if value >= &bound {
                        return Err(InvocationError::ScalarOutOfDomain {
                            parameter: parameter_label(parameter),
                        });
                    }
                }
                (
                    ParameterKind::Range { .. },
                    ArgumentValue::Range { start, end },
                    ParameterContract::Range { bound },
                ) => {
                    let bound = bound.evaluate(&values).unwrap_or_else(|error| {
                        panic!("checked range bound is not total after argument binding: {error:?}")
                    });
                    if start > end || end > &bound {
                        return Err(InvocationError::ScalarOutOfDomain {
                            parameter: parameter_label(parameter),
                        });
                    }
                }
                _ => panic!("compiled invocation contract disagrees with its call schema"),
            }
        }

        for alias in schema.aliases() {
            let (first, second) = match *alias {
                seismic_lang::entry::AliasRule::Disjoint(first, second) => (first, second),
                seismic_lang::entry::AliasRule::MayOverlap(_, _) => continue,
            };
            let first_tensor = tensor_argument(schema, arguments, first);
            let second_tensor = tensor_argument(schema, arguments, second);
            if byte_ranges_overlap(first_tensor, second_tensor) {
                return Err(InvocationError::IllegalAliasing {
                    first: parameter_label(schema.parameter(first)),
                    second: parameter_label(schema.parameter(second)),
                });
            }
        }

        match contract.target_domain.evaluate(&values) {
            Ok(true) => Ok(values),
            Ok(false) | Err(_) => Err(InvocationError::OutsideTargetDomain),
        }
    }

    /// The public descriptor's range is the exact affine footprint that the
    /// invocation will bind. This is checked before any reached schedule step;
    /// execution may then treat its tensor geometry as an admitted value.
    pub(super) fn valid_tensor_descriptor(tensor: &TensorDescriptor) -> bool {
        seismic_ir::storage::valid_concrete_view(
            tensor.representation,
            &tensor.extents,
            &tensor.strides,
            tensor.byte_offset,
            tensor.byte_len,
        )
    }

    fn parameter_label(parameter: &seismic_lang::entry::Parameter) -> String {
        let mut label = parameter.name.clone();
        for child in &parameter.path {
            label.push('.');
            label.push_str(&child.to_string());
        }
        label
    }

    fn observation_location(
        schema: &CallSchema,
        mut observation: usize,
    ) -> Option<(&seismic_lang::entry::Parameter, usize)> {
        for parameter in schema.parameters() {
            let ParameterKind::Tensor { axes, .. } = &parameter.kind else {
                continue;
            };
            if observation < axes.len() {
                return Some((parameter, observation));
            }
            observation -= axes.len();
        }
        None
    }

    fn scalar_value(
        dtype: seismic_lang::types::DType,
        value: &ArgumentValue,
    ) -> Option<SymbolValue> {
        match (dtype, value) {
            (seismic_lang::types::DType::F32, ArgumentValue::F32(value)) => {
                Some(SymbolValue::F32(*value))
            }
            (seismic_lang::types::DType::F16, ArgumentValue::F16(value)) => {
                Some(SymbolValue::F16(*value))
            }
            (seismic_lang::types::DType::BF16, ArgumentValue::BF16(value)) => {
                Some(SymbolValue::BF16(*value))
            }
            (seismic_lang::types::DType::I32, ArgumentValue::I32(value)) => {
                Some(SymbolValue::I32(*value))
            }
            (seismic_lang::types::DType::U32, ArgumentValue::U32(value)) => {
                Some(SymbolValue::U32(*value))
            }
            (seismic_lang::types::DType::Bool, ArgumentValue::Bool(value)) => {
                Some(SymbolValue::Bool(*value))
            }
            _ => None,
        }
    }

    fn tensor_argument<'a>(
        schema: &CallSchema,
        arguments: &'a [ArgumentValue],
        parameter: seismic_lang::ids::ParameterId,
    ) -> &'a TensorDescriptor {
        let position = schema
            .parameters()
            .iter()
            .position(|candidate| candidate.id == parameter)
            .unwrap_or_else(|| panic!("alias rule names a parameter outside its schema"));
        let ArgumentValue::Tensor(tensor) = &arguments[position] else {
            panic!("alias rule names a non-tensor parameter");
        };
        tensor
    }

    fn byte_ranges_overlap(first: &TensorDescriptor, second: &TensorDescriptor) -> bool {
        if first.allocation != second.allocation || first.byte_len == 0 || second.byte_len == 0 {
            return false;
        }
        let first_end = first
            .byte_offset
            .checked_add(first.byte_len)
            .unwrap_or(u64::MAX);
        let second_end = second
            .byte_offset
            .checked_add(second.byte_len)
            .unwrap_or(u64::MAX);
        first.byte_offset < second_end && second.byte_offset < first_end
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use seismic_lang::expr::ExprArena;

    #[test]
    fn selection_function_is_a_total_invocation_to_candidate_mapping() {
        let mut arena = ExprArena::default();
        let always = arena.bool(true);
        let never = arena.bool(false);
        let guard0 = arena.bool(true);
        let guard1 = arena.bool(true);
        let guard2 = arena.bool(true);
        let selection = SelectionFunction::ordered_decision(
            PreparedPortfolio::<()>::from_test_guards(
                NonEmpty::new(vec![
                    arena.compile_bool(guard0),
                    arena.compile_bool(guard1),
                    arena.compile_bool(guard2),
                ])
                .unwrap(),
            ),
            vec![
                (arena.compile_bool(never), CandidateIndex::from_usize(2)),
                (arena.compile_bool(always), CandidateIndex::from_usize(1)),
            ],
        );
        assert_eq!(
            selection.apply(&InvocationValues::new()),
            CandidateIndex::from_usize(1)
        );
    }

    #[test]
    fn selection_function_uses_its_structural_default() {
        let mut arena = ExprArena::default();
        let requested = arena.bool(true);
        let universal = arena.bool(true);
        let optional = arena.bool(false);
        let selection = SelectionFunction::ordered_decision(
            PreparedPortfolio::<()>::from_test_guards(
                NonEmpty::new(vec![
                    arena.compile_bool(universal),
                    arena.compile_bool(optional),
                ])
                .unwrap(),
            ),
            vec![(arena.compile_bool(requested), CandidateIndex::from_usize(1))],
        );
        assert_eq!(
            selection.apply(&InvocationValues::new()),
            CandidateIndex::from_usize(0)
        );
    }

    #[test]
    #[should_panic(expected = "selection case is out of range")]
    fn selection_function_rejects_a_foreign_candidate_index() {
        let mut arena = ExprArena::default();
        let always = arena.bool(true);
        let guard = arena.bool(true);
        let _ = SelectionFunction::ordered_decision(
            PreparedPortfolio::<()>::from_test_guards(
                NonEmpty::new(vec![arena.compile_bool(guard)]).unwrap(),
            ),
            vec![(arena.compile_bool(always), CandidateIndex::from_usize(1))],
        );
    }
}

#[cfg(test)]
mod descriptor_tests {
    use super::*;

    #[test]
    fn admitted_strided_descriptor_has_an_exact_aligned_footprint() {
        let descriptor = TensorDescriptor {
            device: DeviceIdentity(1),
            representation: seismic_lang::registry::dense(seismic_lang::types::DType::F32),
            extents: vec![2, 3],
            strides: vec![1, 2],
            allocation: 7,
            byte_offset: 4,
            byte_len: 24,
        };
        assert!(internals::valid_tensor_descriptor(&descriptor));
        assert!(!internals::valid_tensor_descriptor(&TensorDescriptor {
            byte_len: 20, ..descriptor.clone()
        }));
        assert!(!internals::valid_tensor_descriptor(&TensorDescriptor {
            byte_offset: 2, ..descriptor.clone()
        }));
        assert!(!internals::valid_tensor_descriptor(&TensorDescriptor {
            strides: vec![u64::MAX, 2], ..descriptor.clone()
        }));
        assert!(!internals::valid_tensor_descriptor(&TensorDescriptor {
            byte_offset: u64::MAX - 3, ..descriptor
        }));
    }
}
