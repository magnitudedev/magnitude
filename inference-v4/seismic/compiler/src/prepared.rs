//! `PreparedKernel<T, H>` and invocation validation/selection (spec §10.4,
//! §12).
//!
//! Fields and constructor are private; only the portfolio builder creates
//! one, after proving coverage. Selection evaluates variant guards and
//! chooses the applicable variant with minimum `(cost, identity)`. Zero
//! matches after validation is an internal panic (§13.3.6).

use crate::candidate_domain::NonEmpty;
use crate::errors::InvocationError;
use crate::executable::ExecutableVariant;
use seismic_lang::entry::ParameterKind;
use seismic_lang::entry::{CallSchema, CompiledDimensionInferencePlan, SemanticEventManifest};
use seismic_lang::expr::compiled::{CompiledNat, CompiledPredicate, InvocationValues};
use seismic_lang::expr::{ExprArena, PartialAssignment, SymbolValue};
use seismic_lang::ids::{ModuleHash, RepresentationId, StableEntryId};
use std::sync::Arc;

#[derive(Debug)]
pub struct PreparedKernel<T: seismic_target::TargetFamily, H> {
    entry: StableEntryId,
    module: ModuleHash,
    schema: Arc<CallSchema>,
    semantic_events: Arc<SemanticEventManifest>,
    device: seismic_target::DeviceDescriptionIdentity,
    evaluation: crate::evaluation::EvaluationIdentity,
    invocation: InvocationContract,
    variants: NonEmpty<ExecutableVariant<T, H>>,
    coverage: crate::planning::PlanningCoverage,
}

impl<T: seismic_target::TargetFamily, H> PreparedKernel<T, H> {
    /// Private: the coverage builder is the only caller (§2.2).
    pub(crate) fn prepare(
        entry: StableEntryId,
        module: ModuleHash,
        schema: Arc<CallSchema>,
        semantic_events: Arc<SemanticEventManifest>,
        device: seismic_target::DeviceDescriptionIdentity,
        evaluation: crate::evaluation::EvaluationIdentity,
        invocation: InvocationContract,
        variants: NonEmpty<ExecutableVariant<T, H>>,
        coverage: crate::planning::PlanningCoverage,
    ) -> Self {
        Self {
            entry,
            module,
            schema,
            semantic_events,
            device,
            evaluation,
            invocation,
            variants,
            coverage,
        }
    }

    pub fn entry(&self) -> StableEntryId {
        self.entry
    }
    pub fn module(&self) -> ModuleHash {
        self.module
    }
    pub fn schema(&self) -> &CallSchema {
        &self.schema
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
    pub fn planning_coverage(&self) -> &crate::planning::PlanningCoverage {
        &self.coverage
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
    Index(u64),
    Range {
        start: u64,
        end: u64,
    },
}

/// The complete executable invocation contract. Symbolic schema expressions
/// are compiled exactly once while the entry arena still exists; runtime
/// never retains an arena or attempts to reconstruct shape equations.
#[derive(Debug)]
#[doc(hidden)]
pub struct InvocationContract {
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
            dimension_inference: schema.compile_dimension_inference(arena, &fixed),
            target_domain: arena.compile_bool(entry.domain().predicate().node()),
            parameters,
        }
    }

    pub(crate) fn compile(
        arena: &ExprArena,
        schema: &CallSchema,
        target_domain: crate::candidate_domain::TargetDomain,
        fixed: &PartialAssignment,
    ) -> Self {
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
            dimension_inference: schema.compile_dimension_inference(arena, fixed),
            target_domain: arena.compile_bool_with(target_domain.predicate().node(), fixed),
            parameters,
        }
    }
}

/// Validates one invocation against the schema and target domain, binding
/// every call dimension and scalar symbol. This is the single validator
/// generated bindings call; it runs before any allocation (§12.2).
pub fn validate_invocation(
    schema: &CallSchema,
    contract: &InvocationContract,
    device: DeviceIdentity,
    arguments: &[ArgumentValue],
) -> Result<InvocationValues, InvocationError> {
    internals::validate_invocation(schema, contract, device, arguments)
}

mod internals {
    use super::*;

    pub(super) fn select<T: seismic_target::TargetFamily, H>(
        kernel: &PreparedKernel<T, H>,
        values: &InvocationValues,
    ) -> VariantIndex {
        VariantIndex(crate::executable::select_variant_index(
            kernel.variants.as_slice(),
            values,
        ))
    }

    /// Pure selection over already evaluated applicability/cost data. Its
    /// type has no solver, backend service, or compilation capability.
    pub(super) fn choose_best<T, D: Ord, K: Ord>(
        candidates: impl IntoIterator<Item = (T, D, K)>,
    ) -> Option<T> {
        candidates
            .into_iter()
            .min_by(|(_, left_cost, left_key), (_, right_cost, right_key)| {
                left_cost.cmp(right_cost).then(left_key.cmp(right_key))
            })
            .map(|(candidate, _, _)| candidate)
    }

    pub(super) fn validate_invocation(
        schema: &CallSchema,
        contract: &InvocationContract,
        device: DeviceIdentity,
        arguments: &[ArgumentValue],
    ) -> Result<InvocationValues, InvocationError> {
        if arguments.len() != schema.parameters().len()
            || contract.parameters.len() != schema.parameters().len()
        {
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
                    values.bind(*symbol, SymbolValue::Nat(*value));
                }
                (
                    ParameterKind::Range { start, end, .. },
                    ArgumentValue::Range {
                        start: first,
                        end: last,
                    },
                ) => {
                    values.bind(*start, SymbolValue::Nat(*first));
                    values.bind(*end, SymbolValue::Nat(*last));
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
                    if tensor.strides.len() != tensor.extents.len() {
                        return Err(InvocationError::ShapeMismatch {
                            parameter: parameter_label(parameter),
                            axis: 0,
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
                        if *actual != expected {
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
                    if *value >= bound {
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
                    if start > end || *end > bound {
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
    use super::internals::choose_best;

    #[test]
    fn invocation_selection_uses_only_evaluated_cost_and_identity() {
        let evaluated = [
            ("universal", 10_u64, "z"),
            ("fast", 9, "z"),
            ("tie", 9, "a"),
        ];
        assert_eq!(choose_best(evaluated), Some("tie"));
        assert_eq!(choose_best(Vec::<(&str, u64, &str)>::new()), None);
    }
}
