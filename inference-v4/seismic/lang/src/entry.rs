//! `LogicalEntry`, `CallSchema`, and `EntryDomain` (spec §3.4, §10.1).
//!
//! A `LogicalEntry` is the monomorphized semantics of one exported entry:
//! its operation graph, call schema, exact semantic domain, canonical
//! portable operations and authoritative semantic events, available lowerings with their
//! capability dependencies, and stable provenance. It contains no target
//! choices, implementation candidates, layouts, placements, schedules,
//! allocations, or native capability tests.
//!
//! It is built only by [`crate::checked::CheckedModule::entry`], consumed by
//! `seismic_compiler::plan_space`. Read access is through accessors; every
//! id is arena-scoped and region identity is carried where it matters.
//!
//! W1/W2 own the internals. The read surface below is frozen.

use crate::expr::{
    compiled::{CompiledInt, CompiledNat, InvocationValues},
    AnyExpr, EntryPredicate, ExprArena, IntExpr, LoopBinderId, NatExpr, NodeView,
    PartialAssignment, ScalarExpr, SymbolId, SymbolValue, TargetPredicate, F32, I32, U32,
};
use crate::ids::{
    BinderId, CapabilityId, DimensionId, FamilyId, FunctionId, IntrinsicId, ModuleHash, NodeId,
    ParameterId, ProgramId, RegionId, RepresentationConversionId, RepresentationId, SchemaId,
    SemanticValueId, StableEntryId, StableFunctionId,
};
use crate::intrinsics::{AtomicOp, PrimitiveId, ReduceOp};
use crate::registry::BackendName;
use crate::span::Span;
use crate::types::DType;
use std::collections::BTreeMap;
use num_bigint::BigUint;
use num_traits::{CheckedSub, Zero};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Element bindings
// ---------------------------------------------------------------------------

/// Compile-time element bindings for an entry's element parameters (§14.5).
/// The only compile-time bindings that exist.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ElementBindings {
    bindings: BTreeMap<String, RepresentationId>,
}

impl ElementBindings {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn bind(mut self, parameter: &str, representation: RepresentationId) -> Self {
        self.bindings.insert(parameter.to_owned(), representation);
        self
    }
    pub fn get(&self, parameter: &str) -> Option<RepresentationId> {
        self.bindings.get(parameter).copied()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, RepresentationId)> + '_ {
        self.bindings.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

// ---------------------------------------------------------------------------
// Call schema
// ---------------------------------------------------------------------------

/// The typed invocation contract of one monomorphized entry (§12.2).
/// Everything a generated binding validates comes from here.
#[derive(Debug)]
pub struct CallSchema {
    id: SchemaId,
    dimensions: Vec<Dimension>,
    dimension_inference: DimensionInferencePlan,
    parameters: Vec<Parameter>,
    results: Vec<ResultLeaf>,
    aliases: Vec<AliasRule>,
}

impl CallSchema {
    pub(crate) fn fresh_id() -> SchemaId {
        SchemaId::fresh()
    }

    pub(crate) fn new(
        arena: &ExprArena,
        id: SchemaId,
        dimensions: Vec<Dimension>,
        dimension_inference: DimensionInferencePlan,
        parameters: Vec<Parameter>,
        results: Vec<ResultLeaf>,
        aliases: Vec<AliasRule>,
    ) -> Self {
        assert!(dimensions
            .iter()
            .all(|dimension| dimension.id.schema() == id));
        assert!(parameters
            .iter()
            .all(|parameter| parameter.id.schema() == id));
        dimension_inference.assert_schema(arena, id, &dimensions, &parameters);
        assert!(aliases.iter().all(|alias| match alias {
            AliasRule::Disjoint(a, b) | AliasRule::MayOverlap(a, b) => {
                a.schema() == id && b.schema() == id
            }
        }));
        Self {
            id,
            dimensions,
            dimension_inference,
            parameters,
            results,
            aliases,
        }
    }

    pub(crate) fn dimension_id(id: SchemaId, ordinal: usize) -> DimensionId {
        DimensionId::new(
            id,
            u32::try_from(ordinal).expect("call schema has more than u32::MAX dimensions"),
        )
    }

    pub(crate) fn parameter_id(id: SchemaId, ordinal: usize) -> ParameterId {
        ParameterId::new(
            id,
            u32::try_from(ordinal).expect("call schema has more than u32::MAX parameters"),
        )
    }

    /// Symbolic dimensions, in declaration order.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }
    pub fn parameter(&self, id: ParameterId) -> &Parameter {
        assert_eq!(
            id.schema(),
            self.id,
            "CallSchema received a ParameterId owned by another schema (§13.3.2)"
        );
        self.parameters.get(id.index()).unwrap_or_else(|| {
            panic!("CallSchema received a ParameterId outside its parameter arena (§13.3.2)")
        })
    }
    /// Dense invocation-slot ordinal of a schema-owned parameter id.
    pub fn parameter_ordinal(&self, id: ParameterId) -> usize {
        assert_eq!(
            id.schema(),
            self.id,
            "CallSchema received a ParameterId owned by another schema (§13.3.2)"
        );
        assert!(
            id.index() < self.parameters.len(),
            "CallSchema received a ParameterId outside its parameter arena (§13.3.2)"
        );
        id.index()
    }
    pub fn results(&self) -> &[ResultLeaf] {
        &self.results
    }
    pub fn aliases(&self) -> &[AliasRule] {
        &self.aliases
    }

    /// Seal the schema-owned inference expressions into their arena-free
    /// runtime form. The resulting plan performs no search: its observation
    /// order and elimination steps were fixed when the entry was built.
    #[doc(hidden)]
    pub fn compile_dimension_inference(
        &self,
        arena: &ExprArena,
        fixed: &PartialAssignment,
    ) -> CompiledDimensionInferencePlan {
        self.dimension_inference.compile(arena, fixed)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dimension {
    pub id: DimensionId,
    pub name: String,
    /// The `CallDimension` symbol in the entry's arena.
    pub symbol: SymbolId,
    pub(crate) admits_zero: bool,
}

/// Entry-construction-owned elimination plan for call dimensions.
///
/// Every input tensor axis is observed in schema order. Each step chooses one
/// authored axis equation and carries the canonical inverse operations that
/// isolate one not-yet-bound dimension. Fields are private so runtime cannot
/// invent a different solve order or shape contract.
#[derive(Debug)]
pub struct DimensionInferencePlan {
    observation_count: usize,
    steps: Vec<DimensionInferenceStep>,
}

#[derive(Clone, Debug)]
struct DimensionInferenceStep {
    dimension: SymbolId,
    observation: usize,
    axis: NatExpr,
    operations: Vec<DimensionInferenceOp>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DimensionInferenceKnown {
    Observation(usize),
    Nat(NatExpr),
    Int(IntExpr),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DimensionInferenceOp {
    Add(DimensionInferenceKnown),
    Subtract(DimensionInferenceKnown),
    DivideExact(DimensionInferenceKnown),
    ReverseSubtract(DimensionInferenceKnown),
}

impl DimensionInferencePlan {
    pub(crate) fn new(
        observation_count: usize,
        steps: Vec<(SymbolId, usize, NatExpr, Vec<DimensionInferenceOp>)>,
    ) -> Self {
        Self {
            observation_count,
            steps: steps
                .into_iter()
                .map(
                    |(dimension, observation, axis, operations)| DimensionInferenceStep {
                        dimension,
                        observation,
                        axis,
                        operations,
                    },
                )
                .collect(),
        }
    }

    fn assert_schema(
        &self,
        arena: &ExprArena,
        schema: SchemaId,
        dimensions: &[Dimension],
        parameters: &[Parameter],
    ) {
        let observed = parameters
            .iter()
            .map(|parameter| match &parameter.kind {
                ParameterKind::Tensor { axes, .. } => axes.len(),
                _ => 0,
            })
            .sum::<usize>();
        assert_eq!(
            self.observation_count, observed,
            "DimensionInferencePlan observation count differs from its CallSchema"
        );
        assert_eq!(
            self.steps.len(),
            dimensions.len(),
            "DimensionInferencePlan does not bind every CallSchema dimension"
        );
        let mut seen = Vec::with_capacity(dimensions.len());
        for step in &self.steps {
            assert!(
                step.observation < self.observation_count,
                "DimensionInferencePlan step names an absent tensor-axis observation"
            );
            let _ = arena.view(AnyExpr::Nat(step.axis));
            for operation in &step.operations {
                let known = match operation {
                    DimensionInferenceOp::Add(known)
                    | DimensionInferenceOp::Subtract(known)
                    | DimensionInferenceOp::DivideExact(known)
                    | DimensionInferenceOp::ReverseSubtract(known) => known,
                };
                match known {
                    DimensionInferenceKnown::Observation(observation) => {
                        assert!(
                            *observation < self.observation_count,
                            "DimensionInferencePlan operation names an absent tensor-axis observation"
                        )
                    }
                    DimensionInferenceKnown::Nat(expression) => {
                        let _ = arena.view(AnyExpr::Nat(*expression));
                    }
                    DimensionInferenceKnown::Int(expression) => {
                        let _ = arena.view(AnyExpr::Int(*expression));
                    }
                }
            }
            let id = match arena.symbol_kind(step.dimension) {
                crate::expr::SymbolKind::CallDimension(id) => id,
                _ => panic!(
                    "DimensionInferencePlan step names a symbol that is not a call dimension"
                ),
            };
            assert_eq!(
                id.schema(),
                schema,
                "DimensionInferencePlan dimension belongs to another CallSchema (§13.3.2)"
            );
            assert!(
                dimensions
                    .iter()
                    .any(|dimension| dimension.id == id && dimension.symbol == step.dimension),
                "DimensionInferencePlan dimension is absent from its CallSchema"
            );
            assert!(
                !seen.contains(&step.dimension),
                "DimensionInferencePlan binds one CallSchema dimension more than once"
            );
            seen.push(step.dimension);
        }
    }

    fn compile(
        &self,
        arena: &ExprArena,
        fixed: &PartialAssignment,
    ) -> CompiledDimensionInferencePlan {
        CompiledDimensionInferencePlan {
            observation_count: self.observation_count,
            steps: self
                .steps
                .iter()
                .map(|step| CompiledDimensionInferenceStep {
                    dimension: step.dimension,
                    observation: step.observation,
                    axis: arena.compile_nat_with(step.axis, fixed),
                    operations: step
                        .operations
                        .iter()
                        .map(|operation| match operation {
                            DimensionInferenceOp::Add(known) => CompiledDimensionInferenceOp::Add(
                                compile_known(arena, fixed, *known),
                            ),
                            DimensionInferenceOp::Subtract(known) => {
                                CompiledDimensionInferenceOp::Subtract(compile_known(
                                    arena, fixed, *known,
                                ))
                            }
                            DimensionInferenceOp::DivideExact(known) => {
                                CompiledDimensionInferenceOp::DivideExact(compile_known(
                                    arena, fixed, *known,
                                ))
                            }
                            DimensionInferenceOp::ReverseSubtract(known) => {
                                CompiledDimensionInferenceOp::ReverseSubtract(compile_known(
                                    arena, fixed, *known,
                                ))
                            }
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

fn compile_known(
    arena: &ExprArena,
    fixed: &PartialAssignment,
    known: DimensionInferenceKnown,
) -> CompiledDimensionInferenceKnown {
    match known {
        DimensionInferenceKnown::Observation(observation) => {
            CompiledDimensionInferenceKnown::Observation(observation)
        }
        DimensionInferenceKnown::Nat(expression) => {
            CompiledDimensionInferenceKnown::Nat(arena.compile_nat_with(expression, fixed))
        }
        DimensionInferenceKnown::Int(expression) => {
            CompiledDimensionInferenceKnown::Int(arena.compile_int_with(expression, fixed))
        }
    }
}

/// Arena-free, deterministic invocation-time form of a dimension inference
/// plan. Evaluation only executes the entry-construction-selected steps.
#[derive(Debug)]
pub struct CompiledDimensionInferencePlan {
    observation_count: usize,
    steps: Vec<CompiledDimensionInferenceStep>,
}

#[derive(Debug)]
struct CompiledDimensionInferenceStep {
    dimension: SymbolId,
    observation: usize,
    axis: CompiledNat,
    operations: Vec<CompiledDimensionInferenceOp>,
}

#[derive(Debug)]
enum CompiledDimensionInferenceKnown {
    Observation(usize),
    Nat(CompiledNat),
    Int(CompiledInt),
}

#[derive(Debug)]
enum CompiledDimensionInferenceOp {
    Add(CompiledDimensionInferenceKnown),
    Subtract(CompiledDimensionInferenceKnown),
    DivideExact(CompiledDimensionInferenceKnown),
    ReverseSubtract(CompiledDimensionInferenceKnown),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimensionInferenceFailure {
    observation: usize,
}

impl DimensionInferenceFailure {
    pub fn observation(self) -> usize {
        self.observation
    }
}

impl CompiledDimensionInferencePlan {
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Bind every call dimension from the already flattened input-axis
    /// observations. Failure means the caller's extents do not admit the
    /// exact integer solution sealed into this entry.
    pub fn infer(
        &self,
        observations: &[u64],
        values: &mut InvocationValues,
    ) -> Result<(), DimensionInferenceFailure> {
        if observations.len() != self.observation_count {
            panic!(
                "CompiledDimensionInferencePlan received an observation vector that disagrees with its generated CallSchema contract"
            );
        }
        for step in &self.steps {
            let actual = observations[step.observation];
            let mut value = BigUint::from(actual);
            for operation in &step.operations {
                let failure = || DimensionInferenceFailure {
                    observation: step.observation,
                };
                match operation {
                    CompiledDimensionInferenceOp::Add(known) => {
                        let known =
                            evaluate_known(known, observations, values).ok_or_else(failure)?;
                        value += known;
                    }
                    CompiledDimensionInferenceOp::Subtract(known) => {
                        let known =
                            evaluate_known(known, observations, values).ok_or_else(failure)?;
                        value = value.checked_sub(&known).ok_or_else(failure)?;
                    }
                    CompiledDimensionInferenceOp::ReverseSubtract(known) => {
                        let known =
                            evaluate_known(known, observations, values).ok_or_else(failure)?;
                        value = known.checked_sub(&value).ok_or_else(failure)?;
                    }
                    CompiledDimensionInferenceOp::DivideExact(known) => {
                        let divisor =
                            evaluate_known(known, observations, values).ok_or_else(failure)?;
                        if divisor.is_zero() || (&value % &divisor) != BigUint::zero() {
                            return Err(failure());
                        }
                        value /= divisor;
                    }
                }
            }
            values.bind(step.dimension, SymbolValue::Nat(value));
        }
        // An observed subexpression can eliminate a dimension whose own value
        // is solved later. Validate the original equations only after the full
        // triangular solve, retaining their partial-operation checks.
        for step in &self.steps {
            if step.axis.evaluate(values).ok() != Some(BigUint::from(observations[step.observation])) {
                return Err(DimensionInferenceFailure {
                    observation: step.observation,
                });
            }
        }
        Ok(())
    }
}

fn evaluate_known(
    known: &CompiledDimensionInferenceKnown,
    observations: &[u64],
    values: &InvocationValues,
) -> Option<BigUint> {
    match known {
        CompiledDimensionInferenceKnown::Observation(observation) => {
            observations.get(*observation).copied().map(BigUint::from)
        }
        CompiledDimensionInferenceKnown::Nat(expression) => expression.evaluate(values).ok(),
        CompiledDimensionInferenceKnown::Int(expression) => expression
            .evaluate(values)
            .ok()
            .and_then(|value| value.to_biguint()),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parameter {
    pub id: ParameterId,
    /// Authored parameter ordinal and tuple path. Invocation is leaf-flat,
    /// while bindings use these coordinates to reconstruct the source ABI.
    pub source: u32,
    pub path: Vec<u32>,
    pub name: String,
    pub kind: ParameterKind,
    /// The semantic value this parameter binds in the root function.
    pub value: SemanticValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParameterKind {
    Tensor {
        access: TensorAccess,
        representation: RepresentationId,
        /// Axis extents over dimension symbols and constants.
        axes: Vec<NatExpr>,
    },
    /// A scalar argument; `symbol` is its `CallScalar` symbol.
    Scalar { dtype: DType, symbol: SymbolId },
    /// `index[bound]`: `0 <= value < bound`.
    Index { bound: NatExpr, symbol: SymbolId },
    /// `range[bound]`: `0 <= start <= end <= bound`; two symbols.
    Range {
        bound: NatExpr,
        start: SymbolId,
        end: SymbolId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorAccess {
    /// Moved into the call; the runtime consumes the caller's tensor.
    Owned,
    Shared,
    Mutable,
}

/// One result leaf, by ordinal tuple path (§3.1: names never define
/// identity).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultLeaf {
    pub path: Vec<u32>,
    pub kind: ResultKind,
    /// The semantic value the root function returns at this leaf.
    pub value: SemanticValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResultKind {
    Tensor {
        representation: RepresentationId,
        axes: Vec<NatExpr>,
    },
    Scalar(DType),
    Index {
        bound: NatExpr,
    },
    Range {
        bound: NatExpr,
    },
}

/// Alias contract between parameters, proved from the source (§8.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AliasRule {
    /// The two parameters must not share any byte.
    Disjoint(ParameterId, ParameterId),
    /// The two parameters may overlap (both shared reads).
    MayOverlap(ParameterId, ParameterId),
}

// ---------------------------------------------------------------------------
// Entry domain
// ---------------------------------------------------------------------------

/// The exact semantic domain implied by types and source constraints (§10.1).
/// Free symbols are call dimensions and call scalars only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryDomain {
    predicate: EntryPredicate,
}

impl EntryDomain {
    pub(crate) fn new(arena: &ExprArena, predicate: crate::expr::BoolExpr) -> Self {
        let predicate = EntryPredicate::new(arena, predicate)
            .unwrap_or_else(|_| panic!("entry domain contains a non-call symbol"));
        Self { predicate }
    }
    pub fn predicate(&self) -> EntryPredicate {
        self.predicate
    }
}

// ---------------------------------------------------------------------------
// The semantic program
// ---------------------------------------------------------------------------

/// The monomorphized semantics of one entry.
#[derive(Debug)]
pub struct LogicalEntry {
    identity: StableEntryId,
    module: ModuleHash,
    schema: Arc<CallSchema>,
    domain: EntryDomain,
    arena: ExprArena,
    program: SemanticProgram,
}

/// Borrowed checked semantics over the owning expression arena. Preparation
/// can retain the source program while extending that same arena with compiler
/// expressions; reference evaluation requires neither an arena copy nor a
/// second LogicalEntry owner.
#[derive(Clone, Copy)]
pub struct LogicalEntryView<'a> {
    schema: &'a CallSchema,
    domain: EntryDomain,
    arena: &'a ExprArena,
    program: &'a SemanticProgram,
}

impl<'a> LogicalEntryView<'a> {
    pub fn from_parts(
        schema: &'a CallSchema,
        domain: EntryDomain,
        arena: &'a ExprArena,
        program: &'a SemanticProgram,
    ) -> Self {
        Self {
            schema,
            domain,
            arena,
            program,
        }
    }
    pub fn schema(self) -> &'a CallSchema {
        self.schema
    }
    pub fn domain(self) -> EntryDomain {
        self.domain
    }
    pub fn arena(self) -> &'a ExprArena {
        self.arena
    }
    pub fn program(self) -> &'a SemanticProgram {
        self.program
    }
}

impl LogicalEntry {
    pub fn as_view(&self) -> LogicalEntryView<'_> {
        LogicalEntryView::from_parts(&self.schema, self.domain, &self.arena, &self.program)
    }

    pub(crate) fn new(
        identity: StableEntryId,
        module: ModuleHash,
        schema: CallSchema,
        domain: EntryDomain,
        arena: ExprArena,
        mut program: SemanticProgram,
    ) -> Self {
        // Close every expression-bearing artifact over this entry's sole
        // arena. An expression handle from another entry must fail here,
        // never later in planning or invocation.
        for dimension in schema.dimensions() {
            assert!(matches!(
                arena.symbol_kind(dimension.symbol),
                crate::expr::SymbolKind::CallDimension(id) if id == dimension.id
            ));
        }
        for parameter in schema.parameters() {
            match &parameter.kind {
                ParameterKind::Tensor { axes, .. } => {
                    axes.iter().for_each(|axis| {
                        let _ = arena.view(AnyExpr::Nat(*axis));
                    });
                }
                ParameterKind::Scalar { dtype, symbol } => {
                    assert!(matches!(
                        arena.symbol_kind(*symbol),
                        crate::expr::SymbolKind::CallScalar(id) if id == parameter.id
                    ));
                    assert_eq!(arena.symbol_sort(*symbol), crate::expr::SymbolSort::Scalar(*dtype),
                        "call scalar must retain its declared source word sort");
                }
                ParameterKind::Index { bound, symbol } => {
                    let _ = arena.view(AnyExpr::Nat(*bound));
                    assert!(matches!(
                        arena.symbol_kind(*symbol),
                        crate::expr::SymbolKind::CallScalar(id) if id == parameter.id
                    ));
                }
                ParameterKind::Range { bound, start, end } => {
                    let _ = arena.view(AnyExpr::Nat(*bound));
                    for symbol in [start, end] {
                        assert!(matches!(
                            arena.symbol_kind(*symbol),
                            crate::expr::SymbolKind::CallScalar(id) if id == parameter.id
                        ));
                    }
                }
            }
        }
        for result in schema.results() {
            match &result.kind {
                ResultKind::Tensor { axes, .. } => axes.iter().for_each(|axis| {
                    let _ = arena.view(AnyExpr::Nat(*axis));
                }),
                ResultKind::Index { bound } | ResultKind::Range { bound } => {
                    let _ = arena.view(AnyExpr::Nat(*bound));
                }
                ResultKind::Scalar(_) => {}
            }
        }
        let _ = arena.view(AnyExpr::Bool(domain.predicate.node()));
        for (_, family) in program.families() {
            let references = family
                .candidates()
                .iter()
                .filter(|candidate| candidate.numerical == NumericalRole::Reference)
                .collect::<Vec<_>>();
            assert_eq!(references.len(), 1, "semantic family lacks one reference");
            assert!(matches!(references[0].kind, CandidateKind::Portable));
            assert!(
                matches!(
                    arena.view(AnyExpr::Bool(references[0].applicability.node())),
                    NodeView::BoolConst(true)
                ),
                "reference implementation is not universal over its admitted call domain"
            );
            for candidate in family.candidates() {
                let _ = arena.view(AnyExpr::Bool(candidate.applicability.node()));
            }
        }
        program.seal_reference_candidates();
        let reference = program.family(program.root()).reference();
        let reference = program.function(reference.function());
        assert_eq!(schema.parameters().len(), reference.parameters().len());
        assert!(schema
            .parameters()
            .iter()
            .zip(reference.parameters())
            .all(|(schema, function)| schema.value == function.value));
        assert!(schema
            .results()
            .iter()
            .all(|result| reference.results().contains(&result.value)));
        for (_, function) in program.functions() {
            for (_, value) in function.values() {
                validate_type_arena(&arena, &value.ty);
            }
        }
        Self {
            identity,
            module,
            schema: Arc::new(schema),
            domain,
            arena,
            program,
        }
    }

    pub fn identity(&self) -> StableEntryId {
        self.identity
    }
    pub fn module_hash(&self) -> ModuleHash {
        self.module
    }
    pub fn schema(&self) -> &CallSchema {
        &self.schema
    }
    pub fn shared_schema(&self) -> Arc<CallSchema> {
        self.schema.clone()
    }
    pub fn domain(&self) -> &EntryDomain {
        &self.domain
    }
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }
    pub fn program(&self) -> &SemanticProgram {
        &self.program
    }

    /// Public storage projection of the reference program's authoritative
    /// semantic events. Workflow admission consumes this manifest directly;
    /// it never reconstructs hazards from operand shapes or read/write flags.
    pub fn semantic_event_manifest(&self) -> SemanticEventManifest {
        let reference = self.program.family(self.program.root()).reference();
        let function = self.program.function(reference.function());
        let parameter_storage = self
            .schema
            .parameters()
            .iter()
            .filter_map(|parameter| match &parameter.kind {
                ParameterKind::Tensor { .. } => Some((
                    canonical_storage(function, parameter.value),
                    PublicRegion::Parameter(parameter.id),
                )),
                ParameterKind::Scalar { .. }
                | ParameterKind::Index { .. }
                | ParameterKind::Range { .. } => None,
            })
            .collect::<Vec<_>>();
        let result_storage = self
            .schema
            .results()
            .iter()
            .enumerate()
            .filter_map(|(ordinal, result)| match &result.kind {
                ResultKind::Tensor { .. } => Some((
                    canonical_storage(function, result.value),
                    PublicRegion::Result {
                        ordinal: u32::try_from(ordinal)
                            .expect("entry has more than u32::MAX result leaves"),
                        path: result.path.clone().into_boxed_slice(),
                    },
                )),
                ResultKind::Scalar(_) | ResultKind::Index { .. } | ResultKind::Range { .. } => None,
            })
            .collect::<Vec<_>>();
        let mut accesses = Vec::new();
        collect_manifest_accesses(
            function,
            function.root(),
            &parameter_storage,
            &result_storage,
            &mut accesses,
        );
        let public_events = accesses
            .iter()
            .map(|access| access.event)
            .collect::<std::collections::BTreeSet<_>>();
        let mut dependency_graph = BTreeMap::new();
        collect_event_dependencies(function, function.root(), &mut dependency_graph);
        for access in &mut accesses {
            let mut projected = Vec::new();
            let mut seen = std::collections::BTreeSet::new();
            for dependency in access.ordering_dependencies.iter().copied() {
                project_public_dependencies(
                    dependency,
                    &public_events,
                    &dependency_graph,
                    &mut seen,
                    &mut projected,
                );
            }
            access.ordering_dependencies = projected.into_boxed_slice();
        }
        SemanticEventManifest { accesses }
    }

    /// The consuming transition into planning: the arena moves on, the
    /// program and schema are read by the implementation builder.
    pub fn into_parts(self) -> LogicalEntryParts {
        LogicalEntryParts {
            identity: self.identity,
            module: self.module,
            schema: self.schema,
            domain: self.domain,
            arena: self.arena,
            program: self.program,
        }
    }
}

fn project_public_dependencies(
    dependency: SemanticEventId,
    public: &std::collections::BTreeSet<SemanticEventId>,
    graph: &BTreeMap<SemanticEventId, Box<[SemanticEventId]>>,
    seen: &mut std::collections::BTreeSet<SemanticEventId>,
    output: &mut Vec<SemanticEventId>,
) {
    if !seen.insert(dependency) {
        return;
    }
    if public.contains(&dependency) {
        output.push(dependency);
        return;
    }
    if let Some(predecessors) = graph.get(&dependency) {
        for predecessor in predecessors.iter().copied() {
            project_public_dependencies(predecessor, public, graph, seen, output);
        }
    }
}

fn collect_event_dependencies(
    function: &SemanticFunction,
    region: RegionId,
    graph: &mut BTreeMap<SemanticEventId, Box<[SemanticEventId]>>,
) {
    for (_, node) in function.nodes(region) {
        for event in node.events() {
            graph.insert(
                event.id(),
                event.ordering_dependencies().to_vec().into_boxed_slice(),
            );
        }
        match node.view() {
            SemanticNodeView::If {
                then, otherwise, ..
            } => {
                collect_event_dependencies(function, then, graph);
                collect_event_dependencies(function, otherwise, graph);
            }
            SemanticNodeView::Loop { body, .. } => {
                collect_event_dependencies(function, body, graph)
            }
            SemanticNodeView::Primitive { .. }
            | SemanticNodeView::Intrinsic { .. }
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
            | SemanticNodeView::Check { .. }
            | SemanticNodeView::TuplePack { .. }
            | SemanticNodeView::TupleGet { .. }
            | SemanticNodeView::Extent { .. } => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PublicRegion {
    Parameter(ParameterId),
    Result { ordinal: u32, path: Box<[u32]> },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ManifestAccessKind {
    Read,
    Write {
        exclusive: bool,
    },
    AtomicRmw {
        op: AtomicOp,
        order: AtomicMemoryOrder,
        scope: VisibilityScope,
        publication: PublicationEdge,
        outcome: AssociationOutcome,
    },
    Barrier {
        cohort: ParticipantDomain,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ManifestAccess {
    pub event: SemanticEventId,
    pub region: PublicRegion,
    pub indices: Box<[SemanticValueId]>,
    pub representation: Option<RepresentationId>,
    pub kind: ManifestAccessKind,
    pub participants: ParticipantDomain,
    pub ordering_dependencies: Box<[SemanticEventId]>,
    pub visibility: VisibilityScope,
    pub numerical_outcome: NumericalOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticEventManifest {
    accesses: Vec<ManifestAccess>,
}

impl SemanticEventManifest {
    pub fn accesses(&self) -> &[ManifestAccess] {
        &self.accesses
    }
}

fn collect_manifest_accesses(
    function: &SemanticFunction,
    region: RegionId,
    parameters: &[(SemanticValueId, PublicRegion)],
    results: &[(SemanticValueId, PublicRegion)],
    output: &mut Vec<ManifestAccess>,
) {
    for (_, node) in function.nodes(region) {
        for event in node.events() {
            let public_region = event.place().and_then(|place| {
                let storage = canonical_storage(function, place);
                parameters
                    .iter()
                    .chain(results)
                    .find_map(|(candidate, region)| (*candidate == storage).then(|| region.clone()))
            });
            let Some(public_region) = public_region else {
                continue;
            };
            let kind = match event.kind() {
                SemanticEventKind::MayFail => continue,
                SemanticEventKind::Read => ManifestAccessKind::Read,
                SemanticEventKind::Write(authority) => ManifestAccessKind::Write {
                    exclusive: authority.is_some(),
                },
                SemanticEventKind::AtomicRmw { op, capability } => ManifestAccessKind::AtomicRmw {
                    op: *op,
                    order: capability.order(),
                    scope: capability.scope().clone(),
                    publication: capability.publication(),
                    outcome: capability.outcome(),
                },
                SemanticEventKind::Barrier(capability) => ManifestAccessKind::Barrier {
                    cohort: capability.cohort().clone(),
                },
            };
            output.push(ManifestAccess {
                event: event.id(),
                region: public_region,
                indices: event.indices().to_vec().into_boxed_slice(),
                representation: event.representation(),
                kind,
                participants: event.participants().clone(),
                ordering_dependencies: event.ordering_dependencies().to_vec().into_boxed_slice(),
                visibility: event.visibility().clone(),
                numerical_outcome: event.numerical_outcome().clone(),
            });
        }
        match node.view() {
            SemanticNodeView::If {
                then, otherwise, ..
            } => {
                collect_manifest_accesses(function, then, parameters, results, output);
                collect_manifest_accesses(function, otherwise, parameters, results, output);
            }
            SemanticNodeView::Loop { body, .. } => {
                collect_manifest_accesses(function, body, parameters, results, output);
            }
            SemanticNodeView::Primitive { .. }
            | SemanticNodeView::Intrinsic { .. }
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
            | SemanticNodeView::Check { .. }
            | SemanticNodeView::TuplePack { .. }
            | SemanticNodeView::TupleGet { .. }
            | SemanticNodeView::Extent { .. } => {}
        }
    }
}

fn canonical_storage(function: &SemanticFunction, value: SemanticValueId) -> SemanticValueId {
    fn resolve(
        function: &SemanticFunction,
        value: SemanticValueId,
        seen: &mut std::collections::BTreeSet<SemanticValueId>,
    ) -> SemanticValueId {
        if !seen.insert(value) {
            return value;
        }
        match function.value(value).origin {
            ValueOrigin::Parameter => value,
            ValueOrigin::RegionParameter(region) => {
                let Some(parent) = find_region_capture(function, region, value) else {
                    return value;
                };
                resolve(function, parent, seen)
            }
            ValueOrigin::Node(node) => match function.node(node).view() {
                SemanticNodeView::View { base, .. } => resolve(function, base, seen),
                SemanticNodeView::ElementWrite { place, .. }
                | SemanticNodeView::Atomic { place, .. } => resolve(function, place, seen),
                SemanticNodeView::Store { destination, .. } => resolve(function, destination, seen),
                SemanticNodeView::Primitive { .. }
                | SemanticNodeView::Intrinsic { .. }
                | SemanticNodeView::Elementwise { .. }
                | SemanticNodeView::Reduce { .. }
                | SemanticNodeView::Call { .. }
                | SemanticNodeView::Alloc { .. }
                | SemanticNodeView::Fill { .. }
                | SemanticNodeView::Copy { .. }
                | SemanticNodeView::RepresentationConvert { .. }
                | SemanticNodeView::ElementRead { .. }
                | SemanticNodeView::If { .. }
                | SemanticNodeView::Loop { .. }
                | SemanticNodeView::Check { .. }
                | SemanticNodeView::TuplePack { .. }
                | SemanticNodeView::TupleGet { .. }
                | SemanticNodeView::Extent { .. } => value,
            },
        }
    }
    resolve(function, value, &mut std::collections::BTreeSet::new())
}

fn find_region_capture(
    function: &SemanticFunction,
    child: RegionId,
    parameter: SemanticValueId,
) -> Option<SemanticValueId> {
    let ordinal = function
        .region(child)
        .parameters()
        .iter()
        .position(|value| *value == parameter)?;
    fn search(
        function: &SemanticFunction,
        region: RegionId,
        child: RegionId,
        ordinal: usize,
    ) -> Option<SemanticValueId> {
        for (_, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::If {
                    captures,
                    then,
                    otherwise,
                    ..
                } => {
                    if then == child || otherwise == child {
                        return captures.get(ordinal).copied();
                    }
                    if let Some(value) = search(function, then, child, ordinal) {
                        return Some(value);
                    }
                    if let Some(value) = search(function, otherwise, child, ordinal) {
                        return Some(value);
                    }
                }
                SemanticNodeView::Loop { captures, body, .. } => {
                    if body == child {
                        return ordinal
                            .checked_sub(1)
                            .and_then(|capture| captures.get(capture).copied());
                    }
                    if let Some(value) = search(function, body, child, ordinal) {
                        return Some(value);
                    }
                }
                SemanticNodeView::Primitive { .. }
                | SemanticNodeView::Intrinsic { .. }
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
                | SemanticNodeView::Check { .. }
                | SemanticNodeView::TuplePack { .. }
                | SemanticNodeView::TupleGet { .. }
                | SemanticNodeView::Extent { .. } => {}
            }
        }
        None
    }
    search(function, function.root(), child, ordinal)
}

fn validate_type_arena(arena: &ExprArena, ty: &SemanticType) {
    match ty {
        SemanticType::Scalar(_)
        | SemanticType::Integer
        | SemanticType::Opaque { .. }
        | SemanticType::Void => {}
        SemanticType::Index { bound } | SemanticType::Range { bound } => {
            let _ = arena.view(AnyExpr::Nat(*bound));
        }
        SemanticType::Tensor(tensor) => {
            tensor.axes.iter().for_each(|axis| {
                let _ = arena.view(AnyExpr::Nat(*axis));
            });
            match &tensor.storage {
                TensorStorage::View { transform, .. } => validate_view_arena(arena, transform),
                TensorStorage::Computed | TensorStorage::Owned | TensorStorage::Parameter(_) => {}
            }
        }
        SemanticType::Tuple(items) => {
            items
                .iter()
                .for_each(|item| validate_type_arena(arena, item));
        }
    }
}

fn validate_view_arena(arena: &ExprArena, transform: &ViewTransform) {
    let validate_scalar = |scalar: &ScalarRef| {
        if let ScalarRef::Static(value) = scalar {
            let _ = arena.view(AnyExpr::Nat(*value));
        }
    };
    match transform {
        ViewTransform::Slice { axes } => {
            for axis in axes {
                match axis {
                    SliceAxis::Point { value, .. } => validate_scalar(value),
                    SliceAxis::Range { start, end, .. } => {
                        start.iter().for_each(|value| validate_scalar(value));
                        end.iter().for_each(|value| validate_scalar(value));
                    }
                    SliceAxis::Full => {}
                }
            }
        }
        ViewTransform::Reshape { axes } => axes.iter().for_each(|axis| {
            let _ = arena.view(AnyExpr::Nat(*axis));
        }),
        ViewTransform::Identity | ViewTransform::Plane { .. } | ViewTransform::Transpose { .. } => {
        }
    }
}

/// The owned components of a consumed `LogicalEntry`.
#[derive(Debug)]
pub struct LogicalEntryParts {
    pub identity: StableEntryId,
    pub module: ModuleHash,
    pub schema: Arc<CallSchema>,
    pub domain: EntryDomain,
    pub arena: ExprArena,
    pub program: SemanticProgram,
}

/// The monomorphized function families and functions reachable from the
/// entry. Calls are semantically resolved; there are no recursive cycles.
#[derive(Debug)]
pub struct SemanticProgram {
    inner: internals::Program,
}

/// Exact checked subject shared by independently lowered copies of one entry.
/// The source set is canonicalized by `check_source`; element bindings are in
/// declared-key order. Compiler and registry versions are fixed by this build.
#[derive(Clone, Debug)]
pub struct CheckedProgramSubject {
    sources: Arc<crate::checked::SourceSet>,
    elements: Vec<(String, RepresentationId)>,
    digest: [u8; 32],
}

impl PartialEq for CheckedProgramSubject {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.sources, &other.sources) && self.elements == other.elements)
            || (self.digest == other.digest
                && self.sources == other.sources
                && self.elements == other.elements)
    }
}

impl Eq for CheckedProgramSubject {}

impl CheckedProgramSubject {
    pub(crate) fn new(
        sources: Arc<crate::checked::SourceSet>,
        elements: Vec<(String, RepresentationId)>,
    ) -> Self {
        let mut subject = Self {
            sources,
            elements,
            digest: [0; 32],
        };
        subject.digest = subject.compute_digest();
        subject
    }

    pub fn sources(&self) -> &crate::checked::SourceSet {
        &self.sources
    }

    pub fn elements(&self) -> &[(String, RepresentationId)] {
        &self.elements
    }

    /// Compact label only; equality must compare the exact subject.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn compute_digest(&self) -> [u8; 32] {
        use sha2::Digest as _;
        let mut hash = sha2::Sha256::new();
        hash.update(b"seismic-checked-program-subject-v1");
        for version in [
            crate::bundle::COMPILER_SEMANTIC_VERSION,
            crate::registry::REGISTRY_REVISION,
        ] {
            hash.update((version.len() as u64).to_le_bytes());
            hash.update(version.as_bytes());
        }
        hash.update((self.sources.files().len() as u64).to_le_bytes());
        for source in self.sources.files() {
            hash.update((source.path.len() as u64).to_le_bytes());
            hash.update(source.path.as_bytes());
            hash.update((source.text.len() as u64).to_le_bytes());
            hash.update(source.text.as_bytes());
        }
        hash.update((self.elements.len() as u64).to_le_bytes());
        for (name, representation) in &self.elements {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update((representation.index() as u64).to_le_bytes());
        }
        hash.finalize().into()
    }
}

impl std::hash::Hash for CheckedProgramSubject {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // A collision shares a bucket; PartialEq still compares the full subject.
        std::hash::Hash::hash(&self.digest, state);
    }
}

impl SemanticProgram {
    pub(crate) fn new(inner: internals::Program) -> Self {
        Self { inner }
    }
    /// The entry family.
    pub fn root(&self) -> FamilyId {
        self.inner.root()
    }
    pub fn family(&self, id: FamilyId) -> &Family {
        self.inner.family(id)
    }
    pub fn families(&self) -> impl Iterator<Item = (FamilyId, &Family)> + '_ {
        self.inner.families()
    }
    pub fn function(&self, id: FunctionId) -> &SemanticFunction {
        self.inner.function(id)
    }
    pub fn functions(&self) -> impl Iterator<Item = (FunctionId, &SemanticFunction)> + '_ {
        self.inner.functions()
    }
    /// Canonical checked sources shared by every function in this program.
    /// Equality of this value is exact and independent of stable-label digests.
    pub fn sources(&self) -> &crate::checked::SourceSet {
        self.inner.subject().sources()
    }
    pub fn subject(&self) -> &Arc<CheckedProgramSubject> {
        self.inner.subject()
    }

    fn seal_reference_candidates(&mut self) {
        self.inner.seal_reference_candidates();
    }
}

/// One function family: every candidate sharing one contract (§3.1).
#[derive(Debug)]
pub struct Family {
    name: String,
    candidates: Vec<Candidate>,
    reference: Option<usize>,
}

impl Family {
    pub(crate) fn new(name: String, candidates: Vec<Candidate>) -> Self {
        assert!(
            !candidates.is_empty(),
            "a semantic family has no candidates"
        );
        Self {
            name,
            candidates,
            reference: None,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Applicable candidates in declaration order. Exactly one sealed
    /// portable contract body is the universal numerical reference (§9).
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// The checker-sealed universal reference candidate.
    pub fn reference(&self) -> ReferenceCandidate<'_> {
        ReferenceCandidate(
            self.candidates
                .get(
                    self.reference
                        .expect("semantic family reference was not sealed"),
                )
                .expect("sealed semantic family reference is outside its candidate arena"),
        )
    }

    /// Every non-reference alternative, in declaration order.
    pub fn alternatives(&self) -> impl Iterator<Item = &Candidate> {
        let reference = self
            .reference
            .expect("semantic family reference was not sealed");
        self.candidates
            .iter()
            .enumerate()
            .filter_map(move |(ordinal, candidate)| (ordinal != reference).then_some(candidate))
    }

    fn seal_reference(&mut self) {
        assert!(
            self.reference.is_none(),
            "semantic family reference sealed twice"
        );
        let mut references = self
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.numerical == NumericalRole::Reference);
        let (ordinal, candidate) = references
            .next()
            .expect("semantic family lacks one reference");
        assert!(
            references.next().is_none(),
            "semantic family has multiple references"
        );
        assert!(matches!(candidate.kind, CandidateKind::Portable));
        self.reference = Some(ordinal);
    }
}

/// Opaque evidence that the checker validated the unique, portable,
/// unconditionally applicable reference candidate for this family.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceCandidate<'a>(&'a Candidate);

impl<'a> ReferenceCandidate<'a> {
    pub fn function(self) -> FunctionId {
        self.0.function
    }
    pub fn candidate(self) -> &'a Candidate {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub function: FunctionId,
    pub kind: CandidateKind,
    /// Capabilities this candidate and its transitive helper calls require.
    pub requires: Vec<CapabilityId>,
    /// Exact source applicability after all shape/element substitutions into
    /// the LogicalEntry arena.
    pub applicability: TargetPredicate,
    pub numerical: NumericalRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CandidateKind {
    Portable,
    Lowering { backend: BackendName },
    Helper { backend: BackendName },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericalRole {
    /// The first applicable portable body: defines reference semantics.
    Reference,
    /// Any other body or lowering: an alternative whose transfer is derived.
    Alternative,
}

/// One monomorphized function body.
#[derive(Debug)]
pub struct SemanticFunction {
    inner: internals::Function,
}

impl SemanticFunction {
    pub(crate) fn new(inner: internals::Function) -> Self {
        Self { inner }
    }
    pub fn id(&self) -> FunctionId {
        self.inner.id()
    }
    pub fn stable(&self) -> StableFunctionId {
        self.inner.stable()
    }
    /// Ordinal of the checked source definition instantiated by this function.
    /// Interpret it only together with the exact checked source module.
    pub fn source_definition(&self) -> u64 {
        self.inner.source_definition()
    }
    pub fn name(&self) -> &str {
        self.inner.name()
    }
    pub fn span(&self) -> Span {
        self.inner.span()
    }
    /// Parameters as values of the root region.
    pub fn parameters(&self) -> &[FunctionParameter] {
        self.inner.parameters()
    }
    pub fn initialization(&self) -> &crate::initialization::InitializationContract {
        &self.inner.initialization
    }
    /// The values the function returns, in leaf order.
    pub fn results(&self) -> &[SemanticValueId] {
        self.inner.results()
    }
    pub fn root(&self) -> RegionId {
        self.inner.root()
    }
    pub fn region(&self, id: RegionId) -> &Region {
        self.inner.region(id)
    }
    /// Nodes of one region in execution order.
    pub fn nodes(&self, region: RegionId) -> impl Iterator<Item = (NodeId, &SemanticNode)> + '_ {
        self.inner.nodes(region)
    }
    pub fn node(&self, id: NodeId) -> &SemanticNode {
        self.inner.node(id)
    }
    /// Language scalar meaning of these actual typed operands. Event sealing,
    /// construction and source scope inference consume this same selection.
    pub fn scalar_recipe(
        &self,
        primitive: &PrimitiveId,
        operands: &[SemanticValueId],
    ) -> Option<Arc<crate::reference_math::ReferenceRecipe>> {
        semantic_scalar_recipe(primitive, operands.iter().map(|id| &self.value(*id).ty))
    }
    pub(crate) fn has_failure_events(&self) -> bool {
        let mut regions = vec![self.root()];
        while let Some(region) = regions.pop() {
            for (_, node) in self.nodes(region) {
                if node
                    .events()
                    .iter()
                    .any(|event| matches!(event.kind(), SemanticEventKind::MayFail))
                {
                    return true;
                }
                match node.view() {
                    SemanticNodeView::If {
                        then, otherwise, ..
                    } => regions.extend([then, otherwise]),
                    SemanticNodeView::Loop { body, .. } => regions.push(body),
                    _ => {}
                }
            }
        }
        false
    }
    pub fn value(&self, id: SemanticValueId) -> &ValueInfo {
        self.inner.value(id)
    }
    pub fn values(&self) -> impl Iterator<Item = (SemanticValueId, &ValueInfo)> + '_ {
        self.inner.values()
    }
}

/// The checker calls the same operation selection while constructing its
/// actual typed node, before that node is sealed into a SemanticFunction.
pub(crate) fn semantic_scalar_recipe<'a>(
    primitive: &PrimitiveId,
    operands: impl IntoIterator<Item = &'a SemanticType>,
) -> Option<Arc<crate::reference_math::ReferenceRecipe>> {
    let operation = crate::reference_math::scalar_operation(primitive)?;
    let operands = operands.into_iter().collect::<Vec<_>>();
    let mixed_index = matches!(primitive, PrimitiveId::Binary(_))
        && operands
            .iter()
            .any(|ty| matches!(ty, SemanticType::Scalar(DType::I32)));
    let types = operands
        .into_iter()
        .map(|ty| match ty {
            SemanticType::Scalar(dtype) => Some(*dtype),
            SemanticType::Tensor(tensor) => {
                Some(crate::registry::representation_info(tensor.representation).decoded)
            }
            SemanticType::Index { .. } if mixed_index => Some(DType::I32),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(crate::reference_math::scalar_recipe(operation, &types))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionParameter {
    /// Source leaves and captured shape dimensions share the function's one
    /// complete parameter product.
    pub origin: FunctionParameterOrigin,
    pub name: String,
    pub value: SemanticValueId,
    pub access: ParameterAccess,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionParameterOrigin {
    Source { ordinal: u32, path: Vec<u32> },
    ShapeDimension { ordinal: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterAccess {
    Owned,
    Shared,
    Mutable,
    Scalar,
}

#[derive(Debug)]
pub struct Region {
    kind: RegionKind,
    /// Values defined by the region header (loop binder, branch captures).
    parameters: Vec<SemanticValueId>,
    /// Values the region yields to its parent node.
    results: Vec<SemanticValueId>,
    nodes: Vec<SemanticNode>,
    initialization: Option<crate::initialization::LoopInitialization>,
}

impl Region {
    pub(crate) fn new(
        kind: RegionKind,
        parameters: Vec<SemanticValueId>,
        results: Vec<SemanticValueId>,
        nodes: Vec<SemanticNode>,
        initialization: Option<crate::initialization::LoopInitialization>,
    ) -> Self {
        Self {
            kind,
            parameters,
            results,
            nodes,
            initialization,
        }
    }
    pub fn loop_initialization(&self) -> Option<&crate::initialization::LoopInitialization> {
        self.initialization.as_ref()
    }
    pub fn kind(&self) -> &RegionKind {
        &self.kind
    }
    pub fn parameters(&self) -> &[SemanticValueId] {
        &self.parameters
    }
    pub fn results(&self) -> &[SemanticValueId] {
        &self.results
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Root,
    Then,
    Else,
    LoopBody {
        binder: BinderId,
        /// Exact expression-arena identity used by symbolic expressions in
        /// this lexical loop body. Keeping this relation explicit makes the
        /// semantic body independently executable.
        expression_binder: LoopBinderId,
        binder_symbol: SymbolId,
        /// The value carrying the binder inside the body.
        binder_value: SemanticValueId,
    },
}

/// Static facts about one SSA value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfo {
    pub ty: SemanticType,
    pub origin: ValueOrigin,
    pub span: Span,
}

/// The canonical semantic type of a value, monomorphized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticType {
    Scalar(DType),
    /// Exact signed mathematical quantity.
    Integer,
    Index {
        bound: NatExpr,
    },
    Range {
        bound: NatExpr,
    },
    /// A tensor value of any storage realization; the compiler decides
    /// materialization (§7.4).
    Tensor(TensorSemantics),
    Tuple(Vec<SemanticType>),
    /// A backend-opaque intrinsic value.
    Opaque {
        capability: CapabilityId,
        name: &'static str,
    },
    Void,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSemantics {
    pub representation: RepresentationId,
    pub axes: Vec<NatExpr>,
    /// How the value relates to storage. Computed values have no storage
    /// until the compiler materializes them.
    pub storage: TensorStorage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorStorage {
    /// A pure computed expression (elementwise, cast, broadcast, ...).
    Computed,
    /// A view of an existing place; `base` is the viewed value.
    View {
        base: SemanticValueId,
        transform: ViewTransform,
    },
    /// An owned allocation created in this function.
    Owned,
    /// A parameter place (shared or mutable borrow, or owned input).
    Parameter(ParameterAccess),
}

/// A view transform over a base tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewTransform {
    Identity,
    /// One storage plane of a packed logical representation.
    Plane {
        plane: u32,
    },
    Slice {
        axes: Vec<SliceAxis>,
    },
    Transpose {
        permutation: Vec<u32>,
    },
    Reshape {
        axes: Vec<NatExpr>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SliceAxis {
    /// Point index; the axis is removed.
    Point {
        value: ScalarRef,
        /// The checked source emitted a runtime bound check. When false,
        /// validity is a checker-owned premise in this source control scope.
        runtime_check: bool,
    },
    /// `start..end` on the axis, preserving omitted endpoints.
    Range {
        start: Option<ScalarRef>,
        end: Option<ScalarRef>,
        /// Existing checked-source decisions, retained with their operands.
        /// These do not authorize a physical view without correspondence and scope.
        check_start: bool,
        check_order: bool,
        check_end: bool,
    },
    Full,
}

/// A scalar operand in an index position: a constant, an expression over
/// call symbols/binders, or a runtime SSA scalar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarRef {
    Static(NatExpr),
    Value(SemanticValueId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueOrigin {
    Parameter,
    RegionParameter(RegionId),
    Node(NodeId),
}

/// One semantic node.
#[derive(Debug)]
pub struct SemanticNode {
    data: NodeData,
    events: Vec<SemanticEvent>,
    span: Span,
}

impl SemanticNode {
    pub(crate) fn new(
        kind: NodeKind,
        inputs: Vec<SemanticValueId>,
        outputs: Vec<SemanticValueId>,
        events: Vec<SemanticEvent>,
        span: Span,
    ) -> Self {
        let one_output = |outputs: Vec<SemanticValueId>, operation: &str| {
            let [output] = outputs.as_slice() else {
                panic!("checked {operation} node must have exactly one output")
            };
            *output
        };
        let data = match kind {
            NodeKind::Primitive(primitive) => NodeData::Primitive {
                primitive,
                inputs,
                output: one_output(outputs, "primitive"),
            },
            NodeKind::Intrinsic(intrinsic) => NodeData::Intrinsic {
                intrinsic,
                inputs,
                output: one_output(outputs, "intrinsic"),
            },
            NodeKind::Elementwise(primitive) => NodeData::Elementwise {
                primitive,
                inputs,
                output: one_output(outputs, "elementwise"),
            },
            NodeKind::Reduce {
                op,
                axis,
                unordered,
            } => {
                let [input] = inputs.as_slice() else {
                    panic!("checked reduce node must have exactly one input")
                };
                NodeData::Reduce {
                    op,
                    axis,
                    unordered,
                    input: *input,
                    output: one_output(outputs, "reduce"),
                }
            }
            NodeKind::Call { family } => NodeData::Call {
                family,
                inputs,
                outputs,
            },
            NodeKind::Alloc => {
                NodeData::Alloc {
                    extents: inputs,
                    output: one_output(outputs, "alloc"),
                }
            }
            NodeKind::Fill { value } => {
                let [like] = inputs.as_slice() else {
                    panic!("checked fill node needs its evaluated shape source")
                };
                NodeData::Fill {
                    value,
                    like: *like,
                    output: one_output(outputs, "fill"),
                }
            }
            NodeKind::Copy => {
                let [input] = inputs.as_slice() else {
                    panic!("checked copy node must have exactly one input")
                };
                NodeData::Copy {
                    input: *input,
                    output: one_output(outputs, "copy"),
                }
            }
            NodeKind::RepresentationConvert { conversion } => {
                let [input] = inputs.as_slice() else {
                    panic!("checked representation conversion must have exactly one input")
                };
                NodeData::RepresentationConvert {
                    conversion,
                    input: *input,
                    output: one_output(outputs, "representation conversion"),
                }
            }
            NodeKind::View(transform) => {
                let Some((&base, extents)) = inputs.split_first() else {
                    panic!("checked view node has no base")
                };
                let extents = if matches!(transform, ViewTransform::Reshape { .. }) {
                    extents.to_vec()
                } else {
                    // Slice coordinates are held as actual ScalarRef::Value
                    // operands by the typed transform itself.
                    Vec::new()
                };
                NodeData::View {
                    base,
                    extents,
                    transform,
                    output: one_output(outputs, "view"),
                }
            }
            NodeKind::ElementRead => {
                let Some((&place, indices)) = inputs.split_first() else {
                    panic!("checked element read node has no place")
                };
                NodeData::ElementRead {
                    place,
                    indices: indices.to_vec(),
                    output: one_output(outputs, "element read"),
                }
            }
            NodeKind::ElementWrite { .. } => {
                let Some((&place, tail)) = inputs.split_first() else {
                    panic!("checked element write node has no place")
                };
                let Some((&value, indices)) = tail.split_last() else {
                    panic!("checked element write node has no value")
                };
                NodeData::ElementWrite {
                    place,
                    indices: indices.to_vec(),
                    value,
                    output: one_output(outputs, "element write"),
                }
            }
            NodeKind::Store { .. } => {
                let [destination, value] = inputs.as_slice() else {
                    panic!("checked store node must have destination and value")
                };
                NodeData::Store {
                    destination: *destination,
                    value: *value,
                    output: one_output(outputs, "store"),
                }
            }
            NodeKind::Atomic { op, .. } => {
                let Some((&place, arguments)) = inputs.split_first() else {
                    panic!("checked atomic node has no place")
                };
                NodeData::Atomic {
                    op,
                    place,
                    arguments: arguments.to_vec(),
                    output: one_output(outputs, "atomic"),
                }
            }
            NodeKind::If { then, otherwise } => {
                let Some((&condition, captures)) = inputs.split_first() else {
                    panic!("checked if node has no condition")
                };
                NodeData::If {
                    condition,
                    captures: captures.to_vec(),
                    outputs,
                    then,
                    otherwise,
                }
            }
            NodeKind::Loop {
                kind,
                body,
                carries,
            } => {
                let [start, end, captures @ ..] = inputs.as_slice() else {
                    panic!("checked loop node has no bounds")
                };
                NodeData::Loop {
                    kind,
                    start: *start,
                    end: *end,
                    captures: captures.to_vec(),
                    outputs,
                    body,
                    carries,
                }
            }
            NodeKind::Check { reason } => {
                let [condition] = inputs.as_slice() else {
                    panic!("checked check node must have exactly one condition")
                };
                assert!(outputs.is_empty(), "checked check node cannot have outputs");
                NodeData::Check {
                    condition: *condition,
                    reason,
                }
            }
            NodeKind::TuplePack => NodeData::TuplePack {
                inputs,
                output: one_output(outputs, "tuple pack"),
            },
            NodeKind::TupleGet { index } => {
                let [tuple] = inputs.as_slice() else {
                    panic!("checked tuple projection must have exactly one input")
                };
                NodeData::TupleGet {
                    tuple: *tuple,
                    index,
                    output: one_output(outputs, "tuple projection"),
                }
            }
            NodeKind::Extent { axis } => {
                let [tensor] = inputs.as_slice() else {
                    panic!("checked extent node must have exactly one input")
                };
                NodeData::Extent {
                    tensor: *tensor,
                    axis,
                    output: one_output(outputs, "extent"),
                }
            }
        };
        Self { data, events, span }
    }
    pub fn view(&self) -> SemanticNodeView<'_> {
        self.data.view()
    }
    /// Complete memory semantics for this operation. This is authoritative;
    /// access summaries are projections of these events.
    pub fn events(&self) -> &[SemanticEvent] {
        &self.events
    }
    pub fn span(&self) -> Span {
        self.span
    }
    /// Values that must dominate this node, in canonical operand order.
    ///
    /// This is the sole dependency projection of the sealed node variants;
    /// compiler consumers must not reconstruct arity or dependency rules.
    pub fn dependencies(&self) -> Vec<SemanticValueId> {
        node_dependencies(self.view())
    }
    /// Complete ordered result projection of the sealed operation. Product
    /// transport consumers use this instead of reconstructing output arities.
    pub fn results(&self) -> Vec<SemanticValueId> {
        match self.view() {
            SemanticNodeView::Primitive { output, .. }
            | SemanticNodeView::Intrinsic { output, .. }
            | SemanticNodeView::Elementwise { output, .. }
            | SemanticNodeView::Reduce { output, .. }
            | SemanticNodeView::Alloc { output, .. }
            | SemanticNodeView::Fill { output, .. }
            | SemanticNodeView::Copy { output, .. }
            | SemanticNodeView::RepresentationConvert { output, .. }
            | SemanticNodeView::View { output, .. }
            | SemanticNodeView::ElementRead { output, .. }
            | SemanticNodeView::ElementWrite { output, .. }
            | SemanticNodeView::Store { output, .. }
            | SemanticNodeView::Atomic { output, .. }
            | SemanticNodeView::TuplePack { output, .. }
            | SemanticNodeView::TupleGet { output, .. }
            | SemanticNodeView::Extent { output, .. } => vec![output],
            SemanticNodeView::Call { outputs, .. }
            | SemanticNodeView::If { outputs, .. }
            | SemanticNodeView::Loop { outputs, .. } => outputs.to_vec(),
            SemanticNodeView::Check { .. } => vec![],
        }
    }
}

#[derive(Debug)]
enum NodeData {
    Primitive {
        primitive: PrimitiveId,
        inputs: Vec<SemanticValueId>,
        output: SemanticValueId,
    },
    Intrinsic {
        intrinsic: IntrinsicId,
        inputs: Vec<SemanticValueId>,
        output: SemanticValueId,
    },
    Elementwise {
        primitive: PrimitiveId,
        inputs: Vec<SemanticValueId>,
        output: SemanticValueId,
    },
    Reduce {
        op: ReduceOp,
        axis: u32,
        unordered: bool,
        input: SemanticValueId,
        output: SemanticValueId,
    },
    Call {
        family: FamilyId,
        inputs: Vec<SemanticValueId>,
        outputs: Vec<SemanticValueId>,
    },
    Alloc {
        extents: Vec<SemanticValueId>,
        output: SemanticValueId,
    },
    Fill {
        value: crate::intrinsics::FillConstant,
        like: SemanticValueId,
        output: SemanticValueId,
    },
    Copy {
        input: SemanticValueId,
        output: SemanticValueId,
    },
    RepresentationConvert {
        conversion: RepresentationConversionId,
        input: SemanticValueId,
        output: SemanticValueId,
    },
    View {
        base: SemanticValueId,
        extents: Vec<SemanticValueId>,
        transform: ViewTransform,
        output: SemanticValueId,
    },
    ElementRead {
        place: SemanticValueId,
        indices: Vec<SemanticValueId>,
        output: SemanticValueId,
    },
    ElementWrite {
        place: SemanticValueId,
        indices: Vec<SemanticValueId>,
        value: SemanticValueId,
        output: SemanticValueId,
    },
    Store {
        destination: SemanticValueId,
        value: SemanticValueId,
        output: SemanticValueId,
    },
    Atomic {
        op: AtomicOp,
        place: SemanticValueId,
        arguments: Vec<SemanticValueId>,
        output: SemanticValueId,
    },
    If {
        condition: SemanticValueId,
        captures: Vec<SemanticValueId>,
        outputs: Vec<SemanticValueId>,
        then: RegionId,
        otherwise: RegionId,
    },
    Loop {
        kind: LoopKind,
        start: SemanticValueId,
        end: SemanticValueId,
        captures: Vec<SemanticValueId>,
        outputs: Vec<SemanticValueId>,
        body: RegionId,
        carries: Vec<Carry>,
    },
    Check {
        condition: SemanticValueId,
        reason: CheckReason,
    },
    TuplePack {
        inputs: Vec<SemanticValueId>,
        output: SemanticValueId,
    },
    TupleGet {
        tuple: SemanticValueId,
        index: u32,
        output: SemanticValueId,
    },
    Extent {
        tensor: SemanticValueId,
        axis: u32,
        output: SemanticValueId,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum SemanticNodeView<'a> {
    Primitive {
        primitive: &'a PrimitiveId,
        inputs: &'a [SemanticValueId],
        output: SemanticValueId,
    },
    Intrinsic {
        intrinsic: IntrinsicId,
        inputs: &'a [SemanticValueId],
        output: SemanticValueId,
    },
    Elementwise {
        primitive: &'a PrimitiveId,
        inputs: &'a [SemanticValueId],
        output: SemanticValueId,
    },
    Reduce {
        op: ReduceOp,
        axis: u32,
        unordered: bool,
        input: SemanticValueId,
        output: SemanticValueId,
    },
    Call {
        family: FamilyId,
        inputs: &'a [SemanticValueId],
        outputs: &'a [SemanticValueId],
    },
    Alloc {
        extents: &'a [SemanticValueId],
        output: SemanticValueId,
    },
    Fill {
        value: crate::intrinsics::FillConstant,
        like: SemanticValueId,
        output: SemanticValueId,
    },
    Copy {
        input: SemanticValueId,
        output: SemanticValueId,
    },
    RepresentationConvert {
        conversion: RepresentationConversionId,
        input: SemanticValueId,
        output: SemanticValueId,
    },
    View {
        base: SemanticValueId,
        extents: &'a [SemanticValueId],
        transform: &'a ViewTransform,
        output: SemanticValueId,
    },
    ElementRead {
        place: SemanticValueId,
        indices: &'a [SemanticValueId],
        output: SemanticValueId,
    },
    ElementWrite {
        place: SemanticValueId,
        indices: &'a [SemanticValueId],
        value: SemanticValueId,
        output: SemanticValueId,
    },
    Store {
        destination: SemanticValueId,
        value: SemanticValueId,
        output: SemanticValueId,
    },
    Atomic {
        op: AtomicOp,
        place: SemanticValueId,
        arguments: &'a [SemanticValueId],
        output: SemanticValueId,
    },
    If {
        condition: SemanticValueId,
        captures: &'a [SemanticValueId],
        outputs: &'a [SemanticValueId],
        then: RegionId,
        otherwise: RegionId,
    },
    Loop {
        kind: LoopKind,
        start: SemanticValueId,
        end: SemanticValueId,
        captures: &'a [SemanticValueId],
        outputs: &'a [SemanticValueId],
        body: RegionId,
        carries: &'a [Carry],
    },
    Check {
        condition: SemanticValueId,
        reason: &'a CheckReason,
    },
    TuplePack {
        inputs: &'a [SemanticValueId],
        output: SemanticValueId,
    },
    TupleGet {
        tuple: SemanticValueId,
        index: u32,
        output: SemanticValueId,
    },
    Extent {
        tensor: SemanticValueId,
        axis: u32,
        output: SemanticValueId,
    },
}

impl NodeData {
    fn view(&self) -> SemanticNodeView<'_> {
        match self {
            Self::Primitive {
                primitive,
                inputs,
                output,
            } => SemanticNodeView::Primitive {
                primitive,
                inputs,
                output: *output,
            },
            Self::Intrinsic {
                intrinsic,
                inputs,
                output,
            } => SemanticNodeView::Intrinsic {
                intrinsic: *intrinsic,
                inputs,
                output: *output,
            },
            Self::Elementwise {
                primitive,
                inputs,
                output,
            } => SemanticNodeView::Elementwise {
                primitive,
                inputs,
                output: *output,
            },
            Self::Reduce {
                op,
                axis,
                unordered,
                input,
                output,
            } => SemanticNodeView::Reduce {
                op: *op,
                axis: *axis,
                unordered: *unordered,
                input: *input,
                output: *output,
            },
            Self::Call {
                family,
                inputs,
                outputs,
            } => SemanticNodeView::Call {
                family: *family,
                inputs,
                outputs,
            },
            Self::Alloc { extents, output } => SemanticNodeView::Alloc {
                extents,
                output: *output,
            },
            Self::Fill { value, like, output } => SemanticNodeView::Fill {
                value: *value,
                like: *like,
                output: *output,
            },
            Self::Copy { input, output } => SemanticNodeView::Copy {
                input: *input,
                output: *output,
            },
            Self::RepresentationConvert {
                conversion,
                input,
                output,
            } => SemanticNodeView::RepresentationConvert {
                conversion: *conversion,
                input: *input,
                output: *output,
            },
            Self::View {
                base,
                extents,
                transform,
                output,
            } => SemanticNodeView::View {
                base: *base,
                extents,
                transform,
                output: *output,
            },
            Self::ElementRead {
                place,
                indices,
                output,
            } => SemanticNodeView::ElementRead {
                place: *place,
                indices,
                output: *output,
            },
            Self::ElementWrite {
                place,
                indices,
                value,
                output,
            } => SemanticNodeView::ElementWrite {
                place: *place,
                indices,
                value: *value,
                output: *output,
            },
            Self::Store {
                destination,
                value,
                output,
            } => SemanticNodeView::Store {
                destination: *destination,
                value: *value,
                output: *output,
            },
            Self::Atomic {
                op,
                place,
                arguments,
                output,
            } => SemanticNodeView::Atomic {
                op: *op,
                place: *place,
                arguments,
                output: *output,
            },
            Self::If {
                condition,
                captures,
                outputs,
                then,
                otherwise,
            } => SemanticNodeView::If {
                condition: *condition,
                captures,
                outputs,
                then: *then,
                otherwise: *otherwise,
            },
            Self::Loop {
                kind,
                start,
                end,
                captures,
                outputs,
                body,
                carries,
            } => SemanticNodeView::Loop {
                kind: *kind,
                start: *start,
                end: *end,
                captures,
                outputs,
                body: *body,
                carries,
            },
            Self::Check { condition, reason } => SemanticNodeView::Check {
                condition: *condition,
                reason,
            },
            Self::TuplePack { inputs, output } => SemanticNodeView::TuplePack {
                inputs,
                output: *output,
            },
            Self::TupleGet {
                tuple,
                index,
                output,
            } => SemanticNodeView::TupleGet {
                tuple: *tuple,
                index: *index,
                output: *output,
            },
            Self::Extent {
                tensor,
                axis,
                output,
            } => SemanticNodeView::Extent {
                tensor: *tensor,
                axis: *axis,
                output: *output,
            },
        }
    }
}

fn node_dependencies(node: SemanticNodeView<'_>) -> Vec<SemanticValueId> {
    let mut values = Vec::new();
    match node {
        SemanticNodeView::Primitive { inputs, .. }
        | SemanticNodeView::Intrinsic { inputs, .. }
        | SemanticNodeView::Elementwise { inputs, .. }
        | SemanticNodeView::Call { inputs, .. }
        | SemanticNodeView::TuplePack { inputs, .. } => values.extend_from_slice(inputs),
        SemanticNodeView::Reduce { input, .. }
        | SemanticNodeView::Copy { input, .. }
        | SemanticNodeView::RepresentationConvert { input, .. } => values.push(input),
        SemanticNodeView::View {
            base, extents, transform, ..
        } => {
            values.push(base);
            values.extend_from_slice(extents);
            if let ViewTransform::Slice { axes } = transform {
                for axis in axes {
                    match axis {
                        SliceAxis::Point {
                            value: ScalarRef::Value(value),
                            ..
                        } => values.push(*value),
                        SliceAxis::Range { start, end, .. } => {
                            if let Some(ScalarRef::Value(value)) = start {
                                values.push(*value);
                            }
                            if let Some(ScalarRef::Value(value)) = end {
                                values.push(*value);
                            }
                        }
                        SliceAxis::Point {
                            value: ScalarRef::Static(_),
                            ..
                        }
                        | SliceAxis::Full => {}
                    }
                }
            }
        }
        SemanticNodeView::ElementRead { place, indices, .. } => {
            values.push(place);
            values.extend_from_slice(indices);
        }
        SemanticNodeView::ElementWrite {
            place,
            indices,
            value,
            ..
        } => {
            values.push(place);
            values.extend_from_slice(indices);
            values.push(value);
        }
        SemanticNodeView::Store {
            destination, value, ..
        } => values.extend([destination, value]),
        SemanticNodeView::Atomic {
            place, arguments, ..
        } => {
            values.push(place);
            values.extend_from_slice(arguments);
        }
        SemanticNodeView::If {
            condition,
            captures,
            ..
        } => {
            values.push(condition);
            values.extend_from_slice(captures);
        }
        SemanticNodeView::Loop {
            start,
            end,
            captures,
            ..
        } => {
            values.extend([start, end]);
            values.extend_from_slice(captures);
        }
        SemanticNodeView::Check { condition, .. } => values.push(condition),
        SemanticNodeView::TupleGet { tuple, .. } => values.push(tuple),
        SemanticNodeView::Extent { tensor, .. } => values.push(tensor),
        SemanticNodeView::Alloc { extents, .. } => values.extend_from_slice(extents),
        SemanticNodeView::Fill { like, .. } => values.push(like),
    }
    values
}

fn node_outputs(node: SemanticNodeView<'_>) -> Vec<SemanticValueId> {
    match node {
        SemanticNodeView::Primitive { output, .. }
        | SemanticNodeView::Intrinsic { output, .. }
        | SemanticNodeView::Elementwise { output, .. }
        | SemanticNodeView::Reduce { output, .. }
        | SemanticNodeView::Alloc { output, .. }
        | SemanticNodeView::Fill { output, .. }
        | SemanticNodeView::Copy { output, .. }
        | SemanticNodeView::RepresentationConvert { output, .. }
        | SemanticNodeView::View { output, .. }
        | SemanticNodeView::ElementRead { output, .. }
        | SemanticNodeView::ElementWrite { output, .. }
        | SemanticNodeView::Store { output, .. }
        | SemanticNodeView::Atomic { output, .. }
        | SemanticNodeView::TuplePack { output, .. }
        | SemanticNodeView::TupleGet { output, .. }
        | SemanticNodeView::Extent { output, .. } => vec![output],
        SemanticNodeView::Call { outputs, .. }
        | SemanticNodeView::If { outputs, .. }
        | SemanticNodeView::Loop { outputs, .. } => outputs.to_vec(),
        SemanticNodeView::Check { .. } => Vec::new(),
    }
}

/// Identity of one event in the sealed semantic program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticEventId {
    node: NodeId,
    ordinal: u32,
}

impl SemanticEventId {
    pub(crate) const fn new(node: NodeId, ordinal: u32) -> Self {
        Self { node, ordinal }
    }
    pub const fn node(self) -> NodeId {
        self.node
    }
    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }
}

/// Logical source participants that can perform an access. Physical lanes and
/// launch geometry are deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParticipantDomain {
    Single,
    Parallel(Box<[BinderId]>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicMemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VisibilityScope {
    Participant,
    Participants(ParticipantDomain),
    Command,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PublicationEdge {
    CommandCompletion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AssociationOutcome {
    Exact,
    Reassociated { accumulator: DType },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NumericalOutcome {
    Deterministic,
    AllowedAssociation(AssociationOutcome),
}

/// Proof that all participants reaching a non-atomic write select disjoint
/// logical storage. Fields are private so only checked lowering can mint it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExclusiveWriteCapability {
    place: SemanticValueId,
    indices: Box<[SemanticValueId]>,
    participants: ParticipantDomain,
}

impl ExclusiveWriteCapability {
    pub(crate) fn checked(
        place: SemanticValueId,
        indices: Vec<SemanticValueId>,
        participants: ParticipantDomain,
    ) -> Self {
        Self {
            place,
            indices: indices.into_boxed_slice(),
            participants,
        }
    }
    pub fn place(&self) -> SemanticValueId {
        self.place
    }
    pub fn participants(&self) -> &ParticipantDomain {
        &self.participants
    }
    pub fn indices(&self) -> &[SemanticValueId] {
        &self.indices
    }
}

/// Complete authority for one source atomic RMW. The capability is bound to
/// the exact place and logical participant identity checked for the node.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AtomicCapability {
    place: SemanticValueId,
    indices: Box<[SemanticValueId]>,
    participants: ParticipantDomain,
    order: AtomicMemoryOrder,
    scope: VisibilityScope,
    publication: PublicationEdge,
    outcome: AssociationOutcome,
}

impl AtomicCapability {
    pub(crate) fn checked(
        place: SemanticValueId,
        indices: Vec<SemanticValueId>,
        participants: ParticipantDomain,
        outcome: AssociationOutcome,
    ) -> Self {
        let scope = match &participants {
            ParticipantDomain::Single => VisibilityScope::Participant,
            ParticipantDomain::Parallel(_) => VisibilityScope::Participants(participants.clone()),
        };
        Self {
            place,
            indices: indices.into_boxed_slice(),
            participants,
            order: AtomicMemoryOrder::Relaxed,
            scope,
            publication: PublicationEdge::CommandCompletion,
            outcome,
        }
    }
    pub fn place(&self) -> SemanticValueId {
        self.place
    }
    pub fn participants(&self) -> &ParticipantDomain {
        &self.participants
    }
    pub fn indices(&self) -> &[SemanticValueId] {
        &self.indices
    }
    pub fn order(&self) -> AtomicMemoryOrder {
        self.order
    }
    pub fn scope(&self) -> &VisibilityScope {
        &self.scope
    }
    pub fn publication(&self) -> PublicationEdge {
        self.publication
    }
    pub fn outcome(&self) -> AssociationOutcome {
        self.outcome
    }
}

/// Authority for a future explicit semantic barrier. No current source form
/// constructs one; keeping the constructor private prevents an intrinsic from
/// acquiring barrier semantics by convention.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BarrierCapability {
    cohort: ParticipantDomain,
    visibility: VisibilityScope,
}

impl BarrierCapability {
    pub(crate) fn checked(cohort: ParticipantDomain, visibility: VisibilityScope) -> Self {
        Self { cohort, visibility }
    }
    pub fn cohort(&self) -> &ParticipantDomain {
        &self.cohort
    }
    pub fn visibility(&self) -> &VisibilityScope {
        &self.visibility
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SemanticEventKind {
    Read,
    Write(Option<ExclusiveWriteCapability>),
    AtomicRmw {
        op: AtomicOp,
        capability: AtomicCapability,
    },
    Barrier(BarrierCapability),
    /// Evaluating this source operation can terminate its continuation.
    MayFail,
}

/// Complete source event carried by a checked operation: memory, synchronization
/// or failure. All share the same source-order predecessor relation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SemanticEvent {
    id: SemanticEventId,
    region: RegionId,
    place: Option<SemanticValueId>,
    indices: Box<[SemanticValueId]>,
    kind: SemanticEventKind,
    representation: Option<RepresentationId>,
    participants: ParticipantDomain,
    ordering_dependencies: Box<[SemanticEventId]>,
    visibility: VisibilityScope,
    numerical_outcome: NumericalOutcome,
}

impl SemanticEvent {
    pub(crate) fn checked(
        id: SemanticEventId,
        place: Option<SemanticValueId>,
        indices: Vec<SemanticValueId>,
        kind: SemanticEventKind,
        representation: Option<RepresentationId>,
        participants: ParticipantDomain,
        ordering_dependencies: Vec<SemanticEventId>,
        visibility: VisibilityScope,
        numerical_outcome: NumericalOutcome,
    ) -> Self {
        Self {
            id,
            region: id.node().region(),
            place,
            indices: indices.into_boxed_slice(),
            kind,
            representation,
            participants,
            ordering_dependencies: ordering_dependencies.into_boxed_slice(),
            visibility,
            numerical_outcome,
        }
    }
    pub fn id(&self) -> SemanticEventId {
        self.id
    }
    pub fn region(&self) -> RegionId {
        self.region
    }
    pub fn place(&self) -> Option<SemanticValueId> {
        self.place
    }
    pub fn indices(&self) -> &[SemanticValueId] {
        &self.indices
    }
    pub fn kind(&self) -> &SemanticEventKind {
        &self.kind
    }
    pub fn representation(&self) -> Option<RepresentationId> {
        self.representation
    }
    pub fn participants(&self) -> &ParticipantDomain {
        &self.participants
    }
    pub fn ordering_dependencies(&self) -> &[SemanticEventId] {
        &self.ordering_dependencies
    }
    pub fn visibility(&self) -> &VisibilityScope {
        &self.visibility
    }
    pub fn numerical_outcome(&self) -> &NumericalOutcome {
        &self.numerical_outcome
    }
}

/// The closed set of semantic operations.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NodeKind {
    /// A registry primitive (arithmetic, cast, math, select, extent, ...).
    Primitive(PrimitiveId),
    /// A capability intrinsic with a registered typed signature.
    Intrinsic(IntrinsicId),
    /// Elementwise computed tensor expression over the inputs.
    Elementwise(PrimitiveId),
    Reduce {
        op: ReduceOp,
        axis: u32,
        unordered: bool,
    },
    /// A resolved call; the callee is chosen among `family`'s candidates by
    /// the implementation builder.
    Call {
        family: FamilyId,
    },
    /// Owned allocation of a tensor value.
    Alloc,
    /// Create owned storage initialized to a registry fill constant.
    Fill {
        value: crate::intrinsics::FillConstant,
    },
    /// Copy a value into an owned place (`to_owned`, `clone`).
    Copy,
    RepresentationConvert {
        conversion: RepresentationConversionId,
    },
    /// Define a view (`inputs[0]` is the base).
    View(ViewTransform),
    ElementRead,
    ElementWrite {
        authority: Option<ExclusiveWriteCapability>,
    },
    /// Write a tensor value into a place (slice assignment).
    Store {
        authority: Option<ExclusiveWriteCapability>,
    },
    Atomic {
        op: AtomicOp,
        capability: AtomicCapability,
    },
    If {
        then: RegionId,
        otherwise: RegionId,
    },
    Loop {
        kind: LoopKind,
        body: RegionId,
        carries: Vec<Carry>,
    },
    /// A data-dependent safety obligation (§12.4): `inputs[0]` is the
    /// condition; failure is a typed execution error.
    Check {
        reason: CheckReason,
    },
    /// Tuple construction / projection.
    TuplePack,
    TupleGet {
        index: u32,
    },
    /// The compile-time extent of an axis of a tensor input.
    Extent {
        axis: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoopKind {
    Ordered,
    Parallel,
}

/// A carried value: `initial` enters the body as `region parameter`, the
/// body yields the next value, and `result` is the final value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Carry {
    pub initial: SemanticValueId,
    pub parameter: SemanticValueId,
    pub yielded: SemanticValueId,
    pub result: SemanticValueId,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CheckReason {
    IndexBound,
    RangeOrder,
    Custom(String),
}

/// Scalar expressions over call symbols in the entry arena, for numerical
/// analysis of scalar parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarParameterExpr {
    F32(ScalarExpr<F32>),
    I32(ScalarExpr<I32>),
    U32(ScalarExpr<U32>),
}

pub(crate) mod internals {
    //! W1/W2-owned. The arenas behind `SemanticProgram` and
    //! `SemanticFunction`.

    use super::*;

    #[derive(Debug)]
    pub(crate) struct Program {
        id: ProgramId,
        subject: Arc<CheckedProgramSubject>,
        root: FamilyId,
        families: Vec<Family>,
        functions: Vec<SemanticFunction>,
    }

    impl Program {
        pub(crate) fn new(
            id: ProgramId,
            subject: Arc<CheckedProgramSubject>,
            root: FamilyId,
            families: Vec<Family>,
            functions: Vec<SemanticFunction>,
        ) -> Self {
            assert_eq!(root.program(), id, "root family belongs to another program");
            assert!(root.index() < families.len());
            assert!(families.iter().enumerate().all(|(ordinal, family)| {
                family.candidates.iter().all(|candidate| {
                    candidate.function.program() == id
                        && candidate.function.index() < functions.len()
                }) && ordinal < u32::MAX as usize
            }));
            assert!(functions.iter().enumerate().all(|(ordinal, function)| {
                ordinal < u32::MAX as usize && function.id() == FunctionId::new(id, ordinal as u32)
            }));
            for function in &functions {
                for region in &function.inner.regions {
                    for node in &region.nodes {
                        if let SemanticNodeView::Call { family, .. } = node.view() {
                            assert_eq!(family.program(), id);
                            assert!(family.index() < families.len());
                        }
                    }
                }
            }
            Self {
                id,
                subject,
                root,
                families,
                functions,
            }
        }

        pub(crate) fn root(&self) -> FamilyId {
            self.root
        }
        pub(crate) fn subject(&self) -> &Arc<CheckedProgramSubject> {
            &self.subject
        }
        pub(crate) fn family(&self, id: FamilyId) -> &Family {
            assert_eq!(
                id.program(),
                self.id,
                "SemanticProgram received a FamilyId owned by another program (§13.3.2)"
            );
            self.families.get(id.index()).unwrap_or_else(|| {
                panic!("SemanticProgram received a FamilyId outside its family arena (§13.3.2)")
            })
        }
        pub(crate) fn families(&self) -> impl Iterator<Item = (FamilyId, &Family)> + '_ {
            self.families
                .iter()
                .enumerate()
                .map(|(ordinal, family)| (FamilyId::new(self.id, ordinal as u32), family))
        }
        pub(crate) fn seal_reference_candidates(&mut self) {
            for family in &mut self.families {
                family.seal_reference();
            }
        }
        pub(crate) fn function(&self, id: FunctionId) -> &SemanticFunction {
            assert_eq!(
                id.program(),
                self.id,
                "SemanticProgram received a FunctionId owned by another program (§13.3.2)"
            );
            self.functions.get(id.index()).unwrap_or_else(|| {
                panic!("SemanticProgram received a FunctionId outside its function arena (§13.3.2)")
            })
        }
        pub(crate) fn functions(
            &self,
        ) -> impl Iterator<Item = (FunctionId, &SemanticFunction)> + '_ {
            self.functions
                .iter()
                .enumerate()
                .map(|(ordinal, function)| (FunctionId::new(self.id, ordinal as u32), function))
        }
    }

    #[derive(Debug)]
    pub(crate) struct Function {
        id: FunctionId,
        stable: StableFunctionId,
        source_definition: u64,
        name: String,
        span: Span,
        parameters: Vec<FunctionParameter>,
        results: Vec<SemanticValueId>,
        root: RegionId,
        regions: Vec<Region>,
        values: Vec<ValueInfo>,
        pub(super) initialization: crate::initialization::InitializationContract,
    }

    impl Function {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn new(
            id: FunctionId,
            stable: StableFunctionId,
            source_definition: u64,
            name: String,
            span: Span,
            parameters: Vec<FunctionParameter>,
            results: Vec<SemanticValueId>,
            root: RegionId,
            regions: Vec<Region>,
            values: Vec<ValueInfo>,
            initialization: crate::initialization::InitializationContract,
        ) -> Self {
            assert_eq!(
                root.function(),
                id,
                "root region belongs to another function"
            );
            assert!(root.index() < regions.len());
            assert!(parameters
                .iter()
                .all(|p| p.value.function() == id && p.value.index() < values.len()));
            assert!(parameters
                .iter()
                .all(|parameter| values[parameter.value.index()].origin == ValueOrigin::Parameter));
            assert!(results
                .iter()
                .all(|v| v.function() == id && v.index() < values.len()));
            assert!(values.iter().all(|value| match value.origin {
                ValueOrigin::Parameter => true,
                ValueOrigin::RegionParameter(region) => region.function() == id,
                ValueOrigin::Node(node) => node.region().function() == id,
            }));
            for value in &values {
                if let SemanticType::Tensor(tensor) = &value.ty {
                    if let TensorStorage::View { base, transform } = &tensor.storage {
                        assert_eq!(base.function(), id);
                        validate_transform_values(transform, id, values.len());
                    }
                }
            }
            for (region_ordinal, region) in regions.iter().enumerate() {
                let region_id = RegionId::new(
                    id,
                    u32::try_from(region_ordinal).expect("function has more than u32::MAX regions"),
                );
                assert!(region
                    .parameters
                    .iter()
                    .chain(&region.results)
                    .all(|value| { value.function() == id && value.index() < values.len() }));
                assert!(region.parameters.iter().all(|value| {
                    values[value.index()].origin == ValueOrigin::RegionParameter(region_id)
                }));
                if let RegionKind::LoopBody {
                    binder,
                    binder_value,
                    ..
                } = &region.kind
                {
                    assert_eq!(binder.region(), region_id);
                    assert!(region.parameters.contains(binder_value));
                }
                let mut available = std::collections::BTreeSet::new();
                if region_ordinal == root.index() {
                    available.extend(parameters.iter().map(|parameter| parameter.value));
                }
                available.extend(region.parameters.iter().copied());
                for (node_ordinal, node) in region.nodes.iter().enumerate() {
                    let node_id = NodeId::new(
                        region_id,
                        u32::try_from(node_ordinal).expect("region has more than u32::MAX nodes"),
                    );
                    let dependencies = node_dependencies(node.view());
                    let outputs = node_outputs(node.view());
                    assert!(dependencies
                        .iter()
                        .chain(&outputs)
                        .all(|value| value.function() == id && value.index() < values.len()));
                    assert!(dependencies.iter().all(|value| available.contains(value)));
                    for event in node.events() {
                        assert_eq!(event.id().node(), node_id);
                        assert_eq!(event.region(), region_id);
                        if let Some(place) = event.place() {
                            assert_eq!(place.function(), id);
                            assert!(place.index() < values.len());
                            assert!(available.contains(&place) || outputs.contains(&place));
                            if let Some(representation) = event.representation() {
                                assert!(matches!(
                                    &values[place.index()].ty,
                                    SemanticType::Tensor(tensor)
                                        if tensor.representation == representation
                                ));
                            }
                        }
                        match event.kind() {
                            SemanticEventKind::MayFail => {
                                assert!(event.place().is_none());
                                assert!(event.representation().is_none());
                            }
                            SemanticEventKind::Read | SemanticEventKind::Write(None) => {}
                            SemanticEventKind::Write(Some(capability)) => {
                                assert_eq!(event.place(), Some(capability.place()));
                                assert_eq!(event.indices(), capability.indices());
                                assert_eq!(event.participants(), capability.participants());
                            }
                            SemanticEventKind::AtomicRmw { capability, .. } => {
                                assert_eq!(event.place(), Some(capability.place()));
                                assert_eq!(event.indices(), capability.indices());
                                assert_eq!(event.participants(), capability.participants());
                                assert_eq!(
                                    event.numerical_outcome(),
                                    &match capability.outcome() {
                                        AssociationOutcome::Exact =>
                                            NumericalOutcome::Deterministic,
                                        outcome => NumericalOutcome::AllowedAssociation(outcome),
                                    }
                                );
                            }
                            SemanticEventKind::Barrier(capability) => {
                                assert!(event.place().is_none());
                                assert_eq!(event.participants(), capability.cohort());
                                assert_eq!(event.visibility(), capability.visibility());
                            }
                        }
                    }
                    assert!(outputs.iter().all(|output| {
                        values[output.index()].origin == ValueOrigin::Node(node_id)
                    }));
                    match node.view() {
                        SemanticNodeView::If {
                            then, otherwise, ..
                        } => {
                            assert_eq!(then.function(), id);
                            assert_eq!(otherwise.function(), id);
                            assert!(then.index() < regions.len());
                            assert!(otherwise.index() < regions.len());
                            assert!(matches!(&regions[then.index()].kind, RegionKind::Then));
                            assert!(matches!(&regions[otherwise.index()].kind, RegionKind::Else));
                        }
                        SemanticNodeView::Loop { body, carries, .. } => {
                            assert_eq!(body.function(), id);
                            assert!(body.index() < regions.len());
                            assert!(matches!(
                                &regions[body.index()].kind,
                                RegionKind::LoopBody { .. }
                            ));
                            for carry in carries {
                                assert!(available.contains(&carry.initial));
                                assert!(regions[body.index()]
                                    .parameters
                                    .contains(&carry.parameter));
                                assert!(regions[body.index()].results.contains(&carry.yielded));
                                assert!(outputs.contains(&carry.result));
                            }
                        }
                        SemanticNodeView::Primitive { .. }
                        | SemanticNodeView::Intrinsic { .. }
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
                        | SemanticNodeView::Check { .. }
                        | SemanticNodeView::TuplePack { .. }
                        | SemanticNodeView::TupleGet { .. }
                        | SemanticNodeView::Extent { .. } => {}
                    }
                    available.extend(outputs);
                }
                assert!(region.results.iter().all(|value| available.contains(value)));
            }
            Self {
                id,
                stable,
                source_definition,
                name,
                span,
                parameters,
                results,
                root,
                regions,
                values,
                initialization,
            }
        }

        pub(crate) fn id(&self) -> FunctionId {
            self.id
        }
        pub(crate) fn stable(&self) -> StableFunctionId {
            self.stable
        }
        pub(crate) fn source_definition(&self) -> u64 {
            self.source_definition
        }
        pub(crate) fn name(&self) -> &str {
            &self.name
        }
        pub(crate) fn span(&self) -> Span {
            self.span
        }
        pub(crate) fn parameters(&self) -> &[FunctionParameter] {
            &self.parameters
        }
        pub(crate) fn results(&self) -> &[SemanticValueId] {
            &self.results
        }
        pub(crate) fn root(&self) -> RegionId {
            self.root
        }
        pub(crate) fn region(&self, id: RegionId) -> &Region {
            assert_eq!(
                id.function(),
                self.id,
                "SemanticFunction received a RegionId owned by another function (§13.3.2)"
            );
            self.regions.get(id.index()).unwrap_or_else(|| {
                panic!("SemanticFunction received a RegionId outside its region arena (§13.3.2)")
            })
        }
        pub(crate) fn nodes(
            &self,
            region: RegionId,
        ) -> impl Iterator<Item = (NodeId, &SemanticNode)> + '_ {
            assert_eq!(
                region.function(),
                self.id,
                "SemanticFunction received a RegionId owned by another function (§13.3.2)"
            );
            self.regions
                .get(region.index())
                .unwrap_or_else(|| {
                    panic!(
                        "SemanticFunction received a RegionId outside its region arena (§13.3.2)"
                    )
                })
                .nodes
                .iter()
                .enumerate()
                .map(move |(ordinal, node)| (NodeId::new(region, ordinal as u32), node))
        }
        pub(crate) fn node(&self, id: NodeId) -> &SemanticNode {
            assert_eq!(
                id.region().function(),
                self.id,
                "SemanticFunction received a NodeId owned by another function (§13.3.2)"
            );
            let region = self.regions.get(id.region().index()).unwrap_or_else(|| {
                panic!("SemanticFunction received a NodeId whose region is outside its arena (§13.3.2)")
            });
            region.nodes.get(id.ordinal()).unwrap_or_else(|| {
                panic!("SemanticFunction received a NodeId outside its node arena (§13.3.2)")
            })
        }
        pub(crate) fn value(&self, id: SemanticValueId) -> &ValueInfo {
            assert_eq!(
                id.function(),
                self.id,
                "SemanticFunction received a SemanticValueId owned by another function (§13.3.2)"
            );
            self.values.get(id.index()).unwrap_or_else(|| {
                panic!(
                    "SemanticFunction received a SemanticValueId outside its value arena (§13.3.2)"
                )
            })
        }
        pub(crate) fn values(&self) -> impl Iterator<Item = (SemanticValueId, &ValueInfo)> + '_ {
            self.values
                .iter()
                .enumerate()
                .map(move |(ordinal, value)| (SemanticValueId::new(self.id, ordinal as u32), value))
        }
    }

    fn validate_transform_values(transform: &ViewTransform, function: FunctionId, values: usize) {
        let scalar = |value: &ScalarRef| {
            if let ScalarRef::Value(value) = value {
                assert_eq!(value.function(), function);
                assert!(value.index() < values);
            }
        };
        if let ViewTransform::Slice { axes } = transform {
            for axis in axes {
                match axis {
                    SliceAxis::Point { value, .. } => scalar(value),
                    SliceAxis::Range { start, end, .. } => {
                        start.iter().for_each(|value| scalar(value));
                        end.iter().for_each(|value| scalar(value));
                    }
                    SliceAxis::Full => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod dimension_inference_tests {
    use super::*;

    #[test]
    fn sealed_plan_rejects_inexact_and_zero_division() {
        let mut arena = ExprArena::new();
        let schema = CallSchema::fresh_id();
        let (n_symbol, n) = arena.call_dimension(CallSchema::dimension_id(schema, 0));
        let (g_symbol, g) = arena.call_dimension(CallSchema::dimension_id(schema, 1));
        let product = arena.nat_mul(n, g);
        let plan = DimensionInferencePlan::new(
            2,
            vec![
                (n_symbol, 0, n, Vec::new()),
                (
                    g_symbol,
                    1,
                    product,
                    vec![DimensionInferenceOp::DivideExact(
                        DimensionInferenceKnown::Nat(n),
                    )],
                ),
            ],
        )
        .compile(&arena, &PartialAssignment::new());

        let mut values = InvocationValues::new();
        assert_eq!(
            plan.infer(&[3, 10], &mut values).unwrap_err().observation(),
            1
        );

        let mut values = InvocationValues::new();
        assert_eq!(
            plan.infer(&[0, 0], &mut values).unwrap_err().observation(),
            1
        );
    }
}

#[cfg(test)]
mod checked_program_subject_tests {
    use super::CheckedProgramSubject;
    use crate::checked::{SourceFile, SourceSet};
    use std::sync::Arc;

    #[test]
    fn equal_digest_does_not_make_different_sources_equal() {
        let subject = |text: &str| {
            CheckedProgramSubject::new(
                Arc::new(SourceSet::new(vec![SourceFile {
                    path: "subject.seismic".into(),
                    text: text.into(),
                }])),
                Vec::new(),
            )
        };
        let left = subject("fn probe() -> f32:\n    return 1.0\n");
        let mut right = subject("fn probe() -> f32:\n    return 2.0\n");
        right.digest = left.digest;
        assert_ne!(left, right);
        let mut map = std::collections::HashMap::new();
        map.insert(left, 1);
        map.insert(right, 2);
        assert_eq!(map.len(), 2);
    }
}
