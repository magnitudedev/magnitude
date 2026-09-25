//! Immutable authored-body definitions and their owned physical materializations.
//!
//! Family definitions do not depend on construction effort or cache contents.
//! A domain owns constructed candidates and the one authoritative expression
//! arena in which their choices and constraints were constructed.
//! It contains no native compiler, artifact registry, reflected description,
//! executor, or analytical profile.

mod construction_choice;
mod materialization;
pub use crate::portable::capacity::{CapacityPending, CapacityReason};
pub use construction_choice::{
    BodyChoice, BodyMapping, BodySelection, CallLocation, CallPath, ConstructionCoordinate,
};
pub use materialization::{
    ConstructionAdvance, ConstructionAllowance, ConstructionExclusion, ConstructionPending,
    Materialization, MaterializedRead,
};

use crate::errors::PreparationError;
use crate::expression::PlanningExpr;
use crate::numerics::StructuralNumericalObligation;
use crate::refinement::{ChoiceDeclaration, ConstructedCandidate};
use crate::target::{CompilerRegistry, TargetConstants};
use seismic_lang::entry::{
    CallSchema, EntryDomain, LogicalEntry, LogicalEntryView, SemanticEventManifest, SemanticProgram,
};
use seismic_lang::expr::{
    compiled::{CompiledPredicate, InvocationValues},
    AnyExpr, BoolExpr, DecisionId, EntryPredicate, ExprArena, PartialAssignment, SymbolValue,
};
use seismic_lang::ids::{FunctionId, ModuleHash, StableEntryId, StableFunctionId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_native_target::{DeviceDescription, DeviceDescriptionIdentity};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

static NEXT_DOMAIN_TOKEN: AtomicU64 = AtomicU64::new(1);

/// One immutable construction definition, rooted in an existing checked body.
///
/// The checked program owns the body's semantics and dependencies. This value
/// names that body; it contains no closed IR, search cursor or copied rule set.
#[derive(Clone, Debug)]
pub struct CandidateFamily {
    body: FunctionId,
    identity: StableFunctionId,
}

impl CandidateFamily {
    pub fn identity(&self) -> StableFunctionId {
        self.identity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetDomain {
    predicate: EntryPredicate,
    identity: [u8; 32],
}

impl TargetDomain {
    pub(crate) fn new(predicate: EntryPredicate, identity: [u8; 32]) -> Self {
        Self {
            predicate,
            identity,
        }
    }
    pub fn predicate(&self) -> EntryPredicate {
        self.predicate
    }
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

#[derive(Debug)]
pub struct NonEmpty<T> {
    items: Vec<T>,
}

impl<T> NonEmpty<T> {
    pub fn new(items: Vec<T>) -> Option<Self> {
        (!items.is_empty()).then_some(Self { items })
    }
    pub fn first(&self) -> &T {
        &self.items[0]
    }
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }
    pub(crate) fn push(&mut self, item: T) {
        self.items.push(item);
    }
    pub(crate) fn into_vec(self) -> Vec<T> {
        self.items
    }
}

/// Unchecked search input. Only a [`CandidateDomain`] can construct one, so a
/// proposal is always associated with the domain API that will check it.
#[derive(Clone, Debug)]
pub struct CandidateProposal {
    domain: u64,
    family: ConstructionCoordinate,
    choices: Vec<(crate::refinement::PhysicalChoice, i64)>,
}

/// Canonical structural candidate name. Choices are in declaration order and
/// contain exactly the active choices; there is no public constructor.
#[derive(Clone, Debug)]
pub struct CandidateCoordinate {
    domain: u64,
    family: ConstructionCoordinate,
    choices: Vec<(crate::refinement::PhysicalChoice, i64)>,
}

impl PartialEq for CandidateCoordinate {
    fn eq(&self, other: &Self) -> bool {
        self.family == other.family && self.choices == other.choices
    }
}
impl Eq for CandidateCoordinate {}
impl std::hash::Hash for CandidateCoordinate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.family, state);
        std::hash::Hash::hash(&self.choices, state);
    }
}

impl CandidateCoordinate {
    pub(crate) fn decision_choices(
        &self,
        declarations: &[ChoiceDeclaration],
    ) -> Result<Vec<(DecisionId, i64)>, CoordinateError> {
        bind_physical_choices(declarations, &self.choices)
    }

    pub fn family(&self) -> &ConstructionCoordinate {
        &self.family
    }
    pub fn choices(&self) -> &[(crate::refinement::PhysicalChoice, i64)] {
        &self.choices
    }
}

#[derive(Clone, Debug)]
pub struct CandidatePoint {
    coordinate: CandidateCoordinate,
    invocation: InvocationValues,
}

impl CandidatePoint {
    pub fn new(coordinate: CandidateCoordinate, invocation: InvocationValues) -> Self {
        Self {
            coordinate,
            invocation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintOrigin {
    Invocation,
    SemanticApplicability,
    StructuralLegality,
}

#[derive(Clone, Debug)]
pub struct DomainConstraint {
    origin: ConstraintOrigin,
    predicate: BoolExpr,
}

impl DomainConstraint {
    pub fn origin(&self) -> ConstraintOrigin {
        self.origin
    }
    pub fn predicate(&self) -> BoolExpr {
        self.predicate
    }
}

#[derive(Clone, Debug)]
pub struct ConstraintSet {
    conjuncts: Vec<DomainConstraint>,
    combined: BoolExpr,
}

impl ConstraintSet {
    pub fn conjuncts(&self) -> &[DomainConstraint] {
        &self.conjuncts
    }
    pub fn predicate(&self) -> BoolExpr {
        self.combined
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordinateError {
    ForeignDomain,
    UnknownFamily,
    UnknownPhysicalChoice(crate::refinement::PhysicalChoice),
    MissingChoice(DecisionId),
    DuplicateChoice(DecisionId),
    ForeignChoice(DecisionId),
    ValueOutsideAxis { decision: DecisionId, value: i64 },
    InactiveChoice(DecisionId),
    IndeterminateActivation(DecisionId),
    EmptyRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MembershipError {
    Coordinate(CoordinateError),
    Invocation(seismic_lang::expr::EvalError),
}

#[derive(Debug)]
pub struct InvocationRegion {
    predicate: CompiledPredicate,
}

impl InvocationRegion {
    pub fn contains(
        &self,
        invocation: &InvocationValues,
    ) -> Result<bool, seismic_lang::expr::EvalError> {
        self.predicate.evaluate(invocation)
    }
}

#[derive(Debug)]
pub struct CandidateSlice<'a, B: seismic_native_target::TargetFamily> {
    coordinate: CandidateCoordinate,
    family: &'a Arc<ConstructedCandidate<B>>,
    constraints: &'a ConstraintSet,
    applicability: InvocationRegion,
    numerical: StructuralNumericalObligation,
}

pub(crate) struct ConstructedCandidateHandle<B: seismic_native_target::TargetFamily> {
    coordinate: CandidateCoordinate,
    family: Arc<ConstructedCandidate<B>>,
    constraint: BoolExpr,
    is_universal: bool,
    // Handles pin the same append-only expression storage as their domain.
    // They cannot acquire a read guard independently of that domain.
    arena: Arc<RwLock<ExprArena>>,
}

impl<B: seismic_native_target::TargetFamily> Clone for ConstructedCandidateHandle<B> {
    fn clone(&self) -> Self {
        Self {
            coordinate: self.coordinate.clone(),
            family: self.family.clone(),
            constraint: self.constraint,
            is_universal: self.is_universal,
            arena: self.arena.clone(),
        }
    }
}

/// A bounded read of one constructed physical member. This is neither
/// numerical acceptance nor authority to execute the member.
pub(crate) struct ConstructedCandidateRead<'a, B: seismic_native_target::TargetFamily> {
    candidate: &'a ConstructedCandidateHandle<B>,
    arena: RwLockReadGuard<'a, ExprArena>,
}

impl<B: seismic_native_target::TargetFamily> ConstructedCandidateRead<'_, B> {
    pub(crate) fn candidate(&self) -> &ConstructedCandidateHandle<B> {
        self.candidate
    }

    pub(crate) fn arena(&self) -> &ExprArena {
        &self.arena
    }
}

pub(crate) struct ReferenceRead<'a> {
    schema: &'a CallSchema,
    domain: EntryDomain,
    program: &'a SemanticProgram,
    arena: RwLockReadGuard<'a, ExprArena>,
}

/// The one checked entry's construction inputs, borrowed together from its
/// candidate domain. Only that owner can form this borrow; source construction
/// cannot be started with a schema from a different checked program.
pub(crate) struct SourceEntryBorrow<'a> {
    arena: &'a mut ExprArena,
    program: &'a SemanticProgram,
    schema: &'a CallSchema,
}

impl<'a> SourceEntryBorrow<'a> {
    fn new(arena: &'a mut ExprArena, program: &'a SemanticProgram, schema: &'a CallSchema) -> Self {
        Self {
            arena,
            program,
            schema,
        }
    }

    pub(crate) fn into_parts(self) -> (&'a mut ExprArena, &'a SemanticProgram, &'a CallSchema) {
        (self.arena, self.program, self.schema)
    }
}

impl ReferenceRead<'_> {
    pub(crate) fn view(&self) -> LogicalEntryView<'_> {
        LogicalEntryView::from_parts(self.schema, self.domain, &self.arena, self.program)
    }
}

impl<B: seismic_native_target::TargetFamily> ConstructedCandidateHandle<B> {
    pub(crate) fn coordinate(&self) -> &CandidateCoordinate {
        &self.coordinate
    }
    pub(crate) fn family(&self) -> &Arc<ConstructedCandidate<B>> {
        &self.family
    }
    pub(crate) fn constraint(&self) -> BoolExpr {
        self.constraint
    }
    pub(crate) fn is_universal(&self) -> bool {
        self.is_universal
    }
}

impl<'a, B: seismic_native_target::TargetFamily> CandidateSlice<'a, B> {
    pub fn coordinate(&self) -> &CandidateCoordinate {
        &self.coordinate
    }
    pub fn applicability(&self) -> &InvocationRegion {
        &self.applicability
    }
    pub fn numerical_requirement(&self) -> StructuralNumericalObligation {
        self.numerical
    }
    pub fn executable(&self) -> crate::evaluation::TargetClosedExecutableView<'a, B> {
        crate::evaluation::TargetClosedExecutableView::new(
            self.family.as_ref(),
            self.constraints.conjuncts(),
        )
    }
}

#[derive(Debug)]
pub struct CandidateDomain<'ctx, B: seismic_native_target::TargetFamily> {
    domain_token: u64,
    entry: StableEntryId,
    module: ModuleHash,
    schema: Arc<CallSchema>,
    semantic_events: Arc<SemanticEventManifest>,
    source_domain: EntryDomain,
    source_program: SemanticProgram,
    families: Vec<CandidateFamily>,
    target_domain: TargetDomain,
    constants: TargetConstants,
    target: &'ctx DeviceDescription<B>,
    registry: &'ctx CompilerRegistry<B>,
    // Storage lifetime machinery only: the domain remains the sole mutable
    // owner. Read guards borrow this domain, so Rust prevents advancing it while
    // a view is alive. Handles retain storage but expose no lock acquisition.
    arena: Arc<RwLock<ExprArena>>,
    /// The first member is the mandatory universal construction. Optional
    /// materializations append after it and may be evicted independently.
    materialized: NonEmpty<DomainCandidate<B>>,
    precision: PrecisionPolicy,
    materializations: HashMap<ConstructionCoordinate, materialization::Progress<B>>,
}

impl<'ctx, B: seismic_native_target::TargetFamily> CandidateDomain<'ctx, B> {
    pub fn entry(&self) -> StableEntryId {
        self.entry
    }
    pub fn module(&self) -> ModuleHash {
        self.module
    }
    pub fn schema(&self) -> &Arc<CallSchema> {
        &self.schema
    }
    pub(crate) fn reference(&self) -> ReferenceRead<'_> {
        ReferenceRead {
            schema: &self.schema,
            domain: self.source_domain,
            program: &self.source_program,
            arena: self.arena.read().expect("domain expression owner poisoned"),
        }
    }

    pub fn semantic_event_manifest(&self) -> &Arc<SemanticEventManifest> {
        &self.semantic_events
    }
    pub fn target_domain(&self) -> TargetDomain {
        self.target_domain
    }
    pub fn constants(&self) -> &TargetConstants {
        &self.constants
    }
    pub fn arena(&self) -> impl std::ops::Deref<Target = ExprArena> + '_ {
        self.arena.read().expect("domain expression owner poisoned")
    }
    pub fn precision(&self) -> &PrecisionPolicy {
        &self.precision
    }
    pub fn device_identity(&self) -> &DeviceDescriptionIdentity {
        self.target.identity()
    }

    /// Definitions are present even when no optional physical member has been
    /// constructed. Construction budgets never alter this list.
    pub fn families(&self) -> &[CandidateFamily] {
        &self.families
    }

    pub(crate) fn constructed(&self) -> impl Iterator<Item = ConstructedCandidateView<'_, B>> {
        self.materialized
            .iter()
            .map(|candidate| ConstructedCandidateView { candidate })
    }

    pub fn proposal(
        &self,
        family: ConstructionCoordinate,
        choices: Vec<(crate::refinement::PhysicalChoice, i64)>,
    ) -> CandidateProposal {
        CandidateProposal {
            domain: self.domain_token,
            family,
            choices,
        }
    }

    pub(crate) fn proposal_for_decisions(
        &self,
        family: ConstructionCoordinate,
        choices: Vec<(DecisionId, i64)>,
    ) -> CandidateProposal {
        let declarations = self
            .family(&family)
            .expect("proposal names a constructed definition")
            .family
            .choices();
        self.proposal(
            family.clone(),
            stable_physical_choices(declarations, &choices),
        )
    }

    pub fn universal_proposal(&self) -> CandidateProposal {
        let universal = self.materialized.first();
        CandidateProposal {
            domain: self.domain_token,
            family: universal.construction.clone(),
            choices: stable_physical_choices(
                universal.family.choices(),
                &general_choices(&self.arena(), &universal.family),
            ),
        }
    }

    /// Validates a raw proposal and removes inactive choices and input-order
    /// differences, producing the one canonical coordinate.
    pub fn canonicalize(
        &self,
        proposal: CandidateProposal,
    ) -> Result<CandidateCoordinate, CoordinateError> {
        if proposal.domain != self.domain_token {
            return Err(CoordinateError::ForeignDomain);
        }
        let candidate = self.family(&proposal.family)?;
        let choices = bind_physical_choices(candidate.family.choices(), &proposal.choices)?;
        canonical_coordinate(self.domain_token, &self.arena(), candidate, &choices)
    }

    /// Checks the coordinate against the authoritative family and constraint
    /// relation. Since coordinates have no public constructor, failure here
    /// indicates stale data from a different domain snapshot.
    pub fn check(
        &self,
        coordinate: &CandidateCoordinate,
    ) -> Result<CandidateSlice<'_, B>, CoordinateError> {
        if coordinate.domain != self.domain_token {
            return Err(CoordinateError::ForeignDomain);
        }
        let candidate = self.family(&coordinate.family)?;
        let choices = coordinate.decision_choices(candidate.family.choices())?;
        let (mut fixed, canonical) =
            canonical_choice_binding(&self.arena(), candidate.family.choices(), &choices)?;
        if canonical != choices {
            return Err(CoordinateError::InactiveChoice(
                choices
                    .iter()
                    .find(|choice| !canonical.contains(choice))
                    .unwrap()
                    .0,
            ));
        }
        for (symbol, value) in self.constants.bindings() {
            fixed.bind(*symbol, value.clone());
        }
        let applicability = self
            .arena()
            .compile_bool_with(candidate.constraints.predicate(), &fixed);
        if applicability.reads().is_empty()
            && matches!(applicability.evaluate(&InvocationValues::new()), Ok(false))
        {
            return Err(CoordinateError::EmptyRegion);
        }
        Ok(CandidateSlice {
            coordinate: coordinate.clone(),
            family: &candidate.family,
            constraints: &candidate.constraints,
            applicability: InvocationRegion {
                predicate: applicability,
            },
            numerical: candidate.numerical,
        })
    }

    pub fn contains(&self, point: &CandidatePoint) -> Result<bool, MembershipError> {
        let slice = self
            .check(&point.coordinate)
            .map_err(MembershipError::Coordinate)?;
        slice
            .applicability
            .contains(&point.invocation)
            .map_err(MembershipError::Invocation)
    }

    pub(crate) fn checked_preparation_candidate(
        &self,
        coordinate: &CandidateCoordinate,
    ) -> Result<ConstructedCandidateHandle<B>, CoordinateError> {
        self.check(coordinate)?;
        let candidate = self.family(coordinate.family())?;
        Ok(ConstructedCandidateHandle {
            coordinate: coordinate.clone(),
            family: candidate.family.clone(),
            constraint: candidate.constraints.predicate(),
            is_universal: *coordinate
                == self
                    .canonicalize(self.universal_proposal())
                    .expect("general coordinate is closed"),
            arena: self.arena.clone(),
        })
    }

    pub(crate) fn read<'a>(
        &'a self,
        candidate: &'a ConstructedCandidateHandle<B>,
    ) -> Result<ConstructedCandidateRead<'a, B>, CoordinateError> {
        if candidate.coordinate.domain != self.domain_token
            || !Arc::ptr_eq(&candidate.arena, &self.arena)
        {
            return Err(CoordinateError::ForeignDomain);
        }
        Ok(ConstructedCandidateRead {
            candidate,
            arena: self.arena.read().expect("domain expression owner poisoned"),
        })
    }

    pub(crate) fn owns_coordinate(&self, coordinate: &CandidateCoordinate) -> bool {
        coordinate.domain == self.domain_token
    }

    pub(crate) fn domain_token(&self) -> u64 {
        self.domain_token
    }

    pub(crate) fn structural_candidates(&self) -> impl Iterator<Item = &DomainCandidate<B>> {
        self.materialized.iter()
    }

    pub(crate) fn arena_mut(&mut self) -> RwLockWriteGuard<'_, ExprArena> {
        self.arena
            .write()
            .expect("domain expression owner poisoned")
    }

    pub(crate) fn numerical_context(&self) -> (PrecisionPolicy, TargetDomain, TargetConstants) {
        (
            self.precision.clone(),
            self.target_domain,
            self.constants.clone(),
        )
    }

    fn family(
        &self,
        identity: &ConstructionCoordinate,
    ) -> Result<&DomainCandidate<B>, CoordinateError> {
        self.materialized
            .iter()
            .find(|candidate| &candidate.construction == identity)
            .ok_or(CoordinateError::UnknownFamily)
    }

    pub(crate) fn from_parts(parts: CandidateDomainParts<'ctx, B>) -> Self {
        Self {
            domain_token: parts.domain_token,
            entry: parts.entry,
            module: parts.module,
            schema: parts.schema,
            semantic_events: parts.semantic_events,
            source_domain: parts.source_domain,
            source_program: parts.source_program,
            families: parts.families,
            target_domain: parts.target_domain,
            constants: parts.constants,
            target: parts.target,
            registry: parts.registry,
            arena: Arc::new(RwLock::new(parts.arena)),
            materialized: parts.materialized,
            precision: parts.precision,
            materializations: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn into_parts(self) -> CandidateDomainParts<'ctx, B> {
        CandidateDomainParts {
            domain_token: self.domain_token,
            entry: self.entry,
            module: self.module,
            schema: self.schema,
            semantic_events: self.semantic_events,
            source_domain: self.source_domain,
            source_program: self.source_program,
            families: self.families,
            target_domain: self.target_domain,
            constants: self.constants,
            target: self.target,
            registry: self.registry,
            arena: Arc::try_unwrap(self.arena)
                .expect("cannot decompose a domain with pinned candidate handles")
                .into_inner()
                .expect("domain expression owner poisoned"),
            materialized: self.materialized,
            precision: self.precision,
        }
    }
}

pub(crate) struct ConstructedCandidateView<'a, B: seismic_native_target::TargetFamily> {
    candidate: &'a DomainCandidate<B>,
}

impl<'a, B: seismic_native_target::TargetFamily> ConstructedCandidateView<'a, B> {
    pub fn identity(&self) -> &ConstructionCoordinate {
        &self.candidate.construction
    }
    pub fn choices(&self) -> &'a [ChoiceDeclaration] {
        self.candidate.family.choices()
    }
}

pub(crate) struct CandidateDomainParts<'ctx, B: seismic_native_target::TargetFamily> {
    pub domain_token: u64,
    pub entry: StableEntryId,
    pub module: ModuleHash,
    pub schema: Arc<CallSchema>,
    pub semantic_events: Arc<SemanticEventManifest>,
    pub source_domain: EntryDomain,
    pub source_program: SemanticProgram,
    pub families: Vec<CandidateFamily>,
    pub target_domain: TargetDomain,
    pub constants: TargetConstants,
    pub target: &'ctx DeviceDescription<B>,
    pub registry: &'ctx CompilerRegistry<B>,
    pub arena: ExprArena,
    pub materialized: NonEmpty<DomainCandidate<B>>,
    pub precision: PrecisionPolicy,
}

#[derive(Debug)]
pub(crate) struct DomainCandidate<B: seismic_native_target::TargetFamily> {
    pub(crate) construction: ConstructionCoordinate,
    pub(crate) family: Arc<ConstructedCandidate<B>>,
    pub(crate) constraints: ConstraintSet,
    pub(crate) numerical: StructuralNumericalObligation,
}

impl<B: seismic_native_target::TargetFamily> Clone for DomainCandidate<B> {
    fn clone(&self) -> Self {
        Self {
            family: self.family.clone(),
            construction: self.construction.clone(),
            constraints: self.constraints.clone(),
            numerical: self.numerical,
        }
    }
}

fn bind_physical_choices(
    declarations: &[ChoiceDeclaration],
    choices: &[(crate::refinement::PhysicalChoice, i64)],
) -> Result<Vec<(DecisionId, i64)>, CoordinateError> {
    choices
        .iter()
        .map(|(axis, value)| {
            let declaration = declarations
                .get(axis.ordinal as usize)
                .filter(|declaration| declaration.kind() == axis.kind)
                .ok_or(CoordinateError::UnknownPhysicalChoice(*axis))?;
            Ok((declaration.decision(), *value))
        })
        .collect()
}
fn stable_physical_choices(
    declarations: &[ChoiceDeclaration],
    choices: &[(DecisionId, i64)],
) -> Vec<(crate::refinement::PhysicalChoice, i64)> {
    choices
        .iter()
        .map(|(decision, value)| {
            let (ordinal, declaration) = declarations
                .iter()
                .enumerate()
                .find(|(_, declaration)| declaration.decision() == *decision)
                .expect("internal choice is not declared by its physical construction");
            (
                crate::refinement::PhysicalChoice {
                    ordinal: ordinal as u32,
                    kind: declaration.kind(),
                },
                *value,
            )
        })
        .collect()
}

pub(crate) fn general_choices<B: seismic_native_target::TargetFamily>(
    arena: &ExprArena,
    family: &ConstructedCandidate<B>,
) -> Vec<(DecisionId, i64)> {
    family
        .choices()
        .iter()
        .map(|choice| {
            assert_eq!(
                choice.kind(),
                crate::refinement::ChoiceKind::AllocationSlot,
                "the sequential reference route has only allocation partition choices"
            );
            (
                choice.decision(),
                *arena
                    .decision_domain(choice.decision())
                    .values()
                    .last()
                    .expect("nonempty slot domain"),
            )
        })
        .collect()
}

pub(crate) fn canonical_choice_binding(
    arena: &ExprArena,
    declarations: &[ChoiceDeclaration],
    choices: &[(DecisionId, i64)],
) -> Result<(PartialAssignment, Vec<(DecisionId, i64)>), CoordinateError> {
    let declared = declarations
        .iter()
        .map(ChoiceDeclaration::decision)
        .collect::<HashSet<_>>();
    let mut supplied = HashMap::new();
    for (decision, value) in choices {
        if !declared.contains(decision) {
            return Err(CoordinateError::ForeignChoice(*decision));
        }
        if supplied.insert(*decision, *value).is_some() {
            return Err(CoordinateError::DuplicateChoice(*decision));
        }
        if !arena.decision_domain(*decision).values().contains(value) {
            return Err(CoordinateError::ValueOutsideAxis {
                decision: *decision,
                value: *value,
            });
        }
    }

    let mut assignment = PartialAssignment::new();
    let mut canonical = Vec::new();
    for declaration in declarations {
        let active = arena.compile_bool_with(declaration.active_when(), &assignment);
        if !active.reads().is_empty() {
            return Err(CoordinateError::IndeterminateActivation(
                declaration.decision(),
            ));
        }
        let active = active
            .evaluate(&InvocationValues::new())
            .map_err(|_| CoordinateError::IndeterminateActivation(declaration.decision()))?;
        if !active {
            continue;
        }
        let decision = declaration.decision();
        let value = supplied
            .get(&decision)
            .copied()
            .ok_or(CoordinateError::MissingChoice(decision))?;
        assignment.bind(
            arena.decision_symbol(decision),
            SymbolValue::Int((value).into()),
        );
        canonical.push((decision, value));
    }
    Ok((assignment, canonical))
}

pub(crate) fn canonical_coordinate<B: seismic_native_target::TargetFamily>(
    domain: u64,
    arena: &ExprArena,
    candidate: &DomainCandidate<B>,
    choices: &[(DecisionId, i64)],
) -> Result<CandidateCoordinate, CoordinateError> {
    let (_, choices) = canonical_choice_binding(arena, candidate.family.choices(), choices)?;
    Ok(CandidateCoordinate {
        domain,
        family: candidate.construction.clone(),
        choices: stable_physical_choices(candidate.family.choices(), &choices),
    })
}

pub fn construct_candidate_domain<'ctx, T>(
    entry: LogicalEntry,
    device: &'ctx DeviceDescription<T>,
    registry: &'ctx CompilerRegistry<T>,
    precision: &PrecisionPolicy,
) -> Result<CandidateDomain<'ctx, T>, PreparationError>
where
    T: seismic_native_target::TargetFamily,
{
    internals::candidate_domain(entry, device, registry, precision)
}

pub(crate) mod internals {
    use super::*;
    use seismic_lang::entry::{ParameterKind, ResultKind};
    use seismic_lang::expr::{CmpOp, NodeView, RootName};

    pub(super) fn candidate_domain<'ctx, T>(
        entry: LogicalEntry,
        target: &'ctx DeviceDescription<T>,
        registry: &'ctx CompilerRegistry<T>,
        precision: &PrecisionPolicy,
    ) -> Result<CandidateDomain<'ctx, T>, PreparationError>
    where
        T: seismic_native_target::TargetFamily,
    {
        if target.limits().max_grid[0] == 0 {
            return Err(PreparationError::UniversalClosure(
                "target contract has zero one-dimensional grid capacity".into(),
            ));
        }
        let semantic_events = Arc::new(entry.semantic_event_manifest());
        let seismic_lang::entry::LogicalEntryParts {
            identity,
            module,
            schema,
            domain,
            mut arena,
            program,
        } = entry.into_parts();
        let constants = crate::target::bind_target_constants(target, &mut arena);
        let target_domain = self::target_domain(&mut arena, &schema, domain, target)?;
        let families = program
            .family(program.root())
            .candidates()
            .iter()
            .filter(|candidate| {
                use seismic_lang::entry::CandidateKind;
                let target_matches = match candidate.kind {
                    CandidateKind::Portable => true,
                    CandidateKind::Lowering { backend } | CandidateKind::Helper { backend } => {
                        backend == T::NAME
                    }
                };
                target_matches
                    && candidate
                        .requires
                        .iter()
                        .all(|capability| target.supports_capability(*capability))
            })
            .map(|candidate| CandidateFamily {
                body: candidate.function,
                identity: program.function(candidate.function).stable(),
            })
            .collect();
        let reference = program.family(program.root()).reference().candidate();
        let root = BodySelection::new(
            &program,
            program.function(reference.function),
            crate::portable::SemanticMode::Portable,
        );
        let (source, context) = crate::implementation::begin_source_construction(
            SourceEntryBorrow::new(&mut arena, &program, &schema),
            root.clone(),
            target,
            registry,
            &constants,
            precision,
        );
        let (universal_family, calls) = crate::portable::construct_general(source, context)?;
        let general_construction = ConstructionCoordinate {
            root: root.clone(),
            calls,
        };

        validate_structural_universal(
            program.family(program.root()).name(),
            &mut arena,
            &universal_family,
            target_domain.predicate().node(),
            &constants,
        )?;

        let target_node = target_domain.predicate().node();
        let universal = structural_candidate(
            &mut arena,
            universal_family,
            general_construction.clone(),
            target_node,
            precision,
        );
        let domain_token = NEXT_DOMAIN_TOKEN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .expect("candidate domain identity space exhausted");
        Ok(CandidateDomain::from_parts(CandidateDomainParts {
            domain_token,
            entry: identity,
            module,
            schema,
            semantic_events,
            source_domain: domain,
            source_program: program,
            families,
            target_domain,
            constants,
            target,
            registry,
            arena,
            materialized: NonEmpty::new(vec![universal]).expect("universal member exists"),
            precision: precision.clone(),
        }))
    }

    fn validate_structural_universal<T: seismic_native_target::TargetFamily>(
        entry_name: &str,
        arena: &mut ExprArena,
        family: &ConstructedCandidate<T>,
        target_domain: BoolExpr,
        constants: &TargetConstants,
    ) -> Result<(), PreparationError> {
        let mut fixed = PartialAssignment::new();
        for (symbol, value) in constants.bindings() {
            fixed.bind(*symbol, value.clone());
        }
        for (decision, value) in general_choices(arena, family) {
            fixed.bind(
                arena.decision_symbol(decision),
                SymbolValue::Int((value).into()),
            );
        }
        let target_domain = arena.partial(target_domain, &fixed);
        let semantic = arena.partial(family.semantic_coverage().node(), &fixed);
        let structural = arena.partial(family.hard_constraints(), &fixed);
        let coverage = arena.all(&[semantic, structural]);
        if !arena.entails(target_domain, coverage) {
            return Err(PreparationError::UniversalClosure(format!(
                "entry {entry_name}: structural universal member is not total over TargetDomain (domain={}, required={})",
                crate::implementation::expression_detail(arena, target_domain.into(), 12),
                crate::implementation::expression_detail(arena, coverage.into(), 12),
            )));
        }
        Ok(())
    }

    pub(super) fn structural_candidate<T: seismic_native_target::TargetFamily>(
        arena: &mut ExprArena,
        family: ConstructedCandidate<T>,
        construction: ConstructionCoordinate,
        target: BoolExpr,
        precision: &PrecisionPolicy,
    ) -> DomainCandidate<T> {
        let semantic = family.semantic_coverage().node();
        let hard = family.hard_constraints();
        let numerical = crate::numerics::structural_obligation(
            arena,
            family.numerical_applicability(),
            precision,
        );
        let conjuncts = vec![
            DomainConstraint {
                origin: ConstraintOrigin::Invocation,
                predicate: target,
            },
            DomainConstraint {
                origin: ConstraintOrigin::SemanticApplicability,
                predicate: semantic,
            },
            DomainConstraint {
                origin: ConstraintOrigin::StructuralLegality,
                predicate: hard,
            },
        ];
        let combined = arena.all(
            &conjuncts
                .iter()
                .map(DomainConstraint::predicate)
                .collect::<Vec<_>>(),
        );
        DomainCandidate {
            construction,
            family: Arc::new(family),
            constraints: ConstraintSet {
                conjuncts,
                combined,
            },
            numerical,
        }
    }

    pub(crate) fn planning_projection(
        arena: &mut ExprArena,
        predicate: BoolExpr,
    ) -> PlanningExpr<BoolExpr> {
        let mut terms = Vec::new();
        collect_planning_terms(arena, predicate, &mut terms);
        let node = arena.all(&terms);
        PlanningExpr::new(arena, node)
            .unwrap_or_else(|| panic!("planning projection retained an invocation expression"))
    }

    fn collect_planning_terms(arena: &ExprArena, predicate: BoolExpr, output: &mut Vec<BoolExpr>) {
        match arena.view(AnyExpr::Bool(predicate)) {
            NodeView::Binary {
                op: seismic_lang::expr::BinaryOp::And,
                lhs: AnyExpr::Bool(left),
                rhs: AnyExpr::Bool(right),
            } => {
                collect_planning_terms(arena, left, output);
                collect_planning_terms(arena, right, output);
            }
            NodeView::Nary {
                op: seismic_lang::expr::NaryOp::All,
                operands,
            } => {
                for operand in operands {
                    let AnyExpr::Bool(term) = *operand else {
                        panic!("Boolean conjunction contains a non-Boolean operand")
                    };
                    collect_planning_terms(arena, term, output);
                }
            }
            _ if PlanningExpr::new(arena, predicate).is_some() => output.push(predicate),
            _ => {}
        }
    }

    fn target_domain<B: seismic_native_target::TargetFamily>(
        arena: &mut ExprArena,
        schema: &CallSchema,
        entry: seismic_lang::entry::EntryDomain,
        target: &DeviceDescription<B>,
    ) -> Result<TargetDomain, PreparationError> {
        let max_index = if target.limits().max_index_bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << target.limits().max_index_bits) - 1
        };
        let max_index = arena.nat(max_index);
        let max_allocation = arena.nat(target.limits().max_allocation_bytes);
        let mut terms = vec![entry.predicate().node()];
        let mut tensor = |representation,
                          axes: &[seismic_lang::expr::NatExpr]|
         -> Result<(), PreparationError> {
            if !target.dtypes().representations.contains(&representation) {
                return Err(PreparationError::TargetDomainUnrepresentable(format!(
                    "target {:?} does not support representation `{}`",
                    B::NAME,
                    seismic_lang::registry::representation_info(representation).name,
                )));
            }
            for axis in axes {
                terms.push(arena.nat_cmp(CmpOp::Le, *axis, max_index));
                terms.push(arena.side_conditions(AnyExpr::Nat(*axis)));
            }
            let bytes = seismic_ir::storage::tensor_bytes(arena, representation, axes);
            terms.push(arena.side_conditions(AnyExpr::Nat(bytes)));
            terms.push(arena.nat_cmp(CmpOp::Le, bytes, max_allocation));
            terms.push(arena.nat_cmp(CmpOp::Le, bytes, max_index));
            Ok(())
        };
        for parameter in schema.parameters() {
            if let ParameterKind::Tensor {
                representation,
                axes,
                ..
            } = &parameter.kind
            {
                tensor(*representation, axes)?;
            }
        }
        for result in schema.results() {
            if let ResultKind::Tensor {
                representation,
                axes,
            } = &result.kind
            {
                tensor(*representation, axes)?;
            }
        }
        let predicate = arena.all(&terms);
        let predicate = seismic_lang::expr::EntryPredicate::new(arena, predicate)
            .unwrap_or_else(|_| panic!("target-domain construction retained a compiler symbol"));
        let root = arena.root(RootName::Guard, AnyExpr::Bool(predicate.node()));
        let identity = arena.canonical_digest(&[root]).bytes();
        Ok(TargetDomain::new(predicate, identity))
    }
}

#[cfg(test)]
mod candidate_domain_tests {
    use super::*;
    use seismic_lang::expr::FiniteDomain;

    #[test]
    fn family_definitions_do_not_depend_on_optional_construction_allowance() {
        use crate::realization::demand_driven_tests::device;
        use crate::realization::demand_driven_tests::registry;
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};
        use seismic_lang::entry::ElementBindings;
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "definition-effort.seismic".into(),
            text: "fn probe(x: f32) -> f32:\n    return x\n".into(),
        }]))
        .unwrap();
        let target = device();
        let registry = registry();
        let build = || {
            let entry = module
                .entry(
                    module.entry_named("probe").unwrap(),
                    &ElementBindings::default(),
                )
                .unwrap();
            construct_candidate_domain(entry, &target, &registry, &PrecisionPolicy::Exact).unwrap()
        };
        let unvisited = build();
        let mut visited = build();
        let selected = visited
            .root_selections()
            .into_iter()
            .find(|selection| selection.mapping == BodyMapping::Independent)
            .unwrap();
        let result = visited.advance(
            &ConstructionCoordinate::root(selected),
            ConstructionAllowance {
                work_units: 100_000,
                wall_time: std::time::Duration::from_secs(30),
            },
        );
        assert!(matches!(result.state, Materialization::Ready(_)));
        assert_eq!(unvisited.constructed().count(), 1);
        assert!(visited.constructed().count() > unvisited.constructed().count());
        assert_eq!(
            unvisited
                .families()
                .iter()
                .map(CandidateFamily::identity)
                .collect::<Vec<_>>(),
            visited
                .families()
                .iter()
                .map(CandidateFamily::identity)
                .collect::<Vec<_>>()
        );
        assert_eq!(unvisited.families().len(), 1);
    }

    #[test]
    fn equal_body_labels_cannot_select_a_different_checked_source() {
        use crate::realization::demand_driven_tests::device;
        use crate::realization::demand_driven_tests::registry;
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};
        use seismic_lang::entry::ElementBindings;
        use std::time::Duration;

        let target = device();
        let registry = registry();
        let build = |literal: &str| {
            let module = check_source(SourceSet::new(vec![SourceFile {
                path: "identity-boundary.seismic".into(),
                text: format!("fn probe(x: f32) -> f32:\n    return x + {literal}\n"),
            }]))
            .unwrap();
            let entry = module
                .entry(
                    module.entry_named("probe").unwrap(),
                    &ElementBindings::default(),
                )
                .unwrap();
            construct_candidate_domain(entry, &target, &registry, &PrecisionPolicy::Exact).unwrap()
        };
        let left = build("1.0");
        let mut right = build("2.0");
        let mut foreign = left.root_selections()[0].clone();
        let resident = right.root_selections()[0].clone();
        // Force the compact label and ordinal to agree while retaining the
        // independently checked source subject from the other program.
        foreign.body = resident.body;
        foreign.source_definition = resident.source_definition;
        assert_ne!(foreign, resident);
        let result = right.advance(
            &ConstructionCoordinate::root(foreign),
            ConstructionAllowance {
                work_units: 100_000,
                wall_time: Duration::from_secs(30),
            },
        );
        assert!(matches!(
            result.state,
            Materialization::Excluded(ConstructionExclusion::UnknownBody)
        ));
    }

    #[test]
    fn suspended_construction_restart_and_general_route_have_stable_coordinates() {
        use crate::realization::demand_driven_tests::registry;
        use crate::realization::demand_driven_tests::{device, FakeTarget};
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};
        use seismic_lang::entry::ElementBindings;
        use std::time::Duration;
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "resumable-choices.seismic".into(),
            text: "fn helper(x: f32) -> f32:\n    return x + 1.0\n\nfn probe(x: f32) -> f32:\n    let a = helper(x)\n    return helper(a)\n".into(),
        }])).unwrap();
        let target = device();
        let registry = registry();
        let build = || {
            let entry = module
                .entry(
                    module.entry_named("probe").unwrap(),
                    &ElementBindings::default(),
                )
                .unwrap();
            construct_candidate_domain(entry, &target, &registry, &PrecisionPolicy::Exact).unwrap()
        };
        fn finish(
            domain: &mut CandidateDomain<'_, FakeTarget>,
            mut path: ConstructionCoordinate,
            quantum: u64,
        ) -> ConstructionCoordinate {
            for _ in 0..100_000 {
                match domain
                    .advance(
                        &path,
                        ConstructionAllowance {
                            work_units: quantum,
                            wall_time: Duration::from_secs(30),
                        },
                    )
                    .state
                {
                    Materialization::Ready(canonical) => return canonical,
                    Materialization::Choice(choice) => {
                        path = path.select(&choice, choice.alternatives[0].clone())
                    }
                    Materialization::Pending(ConstructionPending::WorkAllowance) => {}
                    other => panic!("fixture failed source construction: {other:?}"),
                }
            }
            panic!("bounded source fixture did not complete")
        }
        fn coordinate(
            domain: &CandidateDomain<'_, FakeTarget>,
            path: ConstructionCoordinate,
        ) -> CandidateCoordinate {
            let values = {
                let read = domain.read_materialized(&path).unwrap();
                read.candidate()
                    .choices()
                    .iter()
                    .enumerate()
                    .map(|(ordinal, choice)| {
                        let values = read.arena().decision_domain(choice.decision());
                        (
                            crate::refinement::PhysicalChoice {
                                ordinal: ordinal as u32,
                                kind: choice.kind(),
                            },
                            *values.values().last().unwrap(),
                        )
                    })
                    .collect()
            };
            domain.canonicalize(domain.proposal(path, values)).unwrap()
        }
        let mut sliced = build();
        let root = sliced
            .root_selections()
            .into_iter()
            .find(|selection| selection.mapping == BodyMapping::Independent)
            .unwrap();
        let path = ConstructionCoordinate::root(root);
        assert!(matches!(
            sliced
                .advance(
                    &path,
                    ConstructionAllowance {
                        work_units: 1,
                        wall_time: Duration::from_secs(30)
                    }
                )
                .state,
            Materialization::Pending(ConstructionPending::WorkAllowance)
        ));
        let finished = finish(&mut sliced, path.clone(), 1);
        let first = coordinate(&sliced, finished.clone());
        let identity = sliced
            .read_materialized(&finished)
            .unwrap()
            .candidate()
            .identity()
            .clone();
        let mut uninterrupted = build();
        let other = finish(&mut uninterrupted, path, 100_000);
        assert_eq!(first, coordinate(&uninterrupted, other.clone()));
        assert_eq!(
            identity,
            *uninterrupted
                .read_materialized(&other)
                .unwrap()
                .candidate()
                .identity()
        );
        let general = sliced.general_construction();
        sliced.materialized.items.retain(|candidate| {
            candidate.construction == general || candidate.construction != finished
        });
        let rebuilt = finish(&mut sliced, finished.clone(), 2);
        assert_eq!(first, coordinate(&sliced, rebuilt.clone()));
        assert_eq!(
            identity,
            *sliced
                .read_materialized(&rebuilt)
                .unwrap()
                .candidate()
                .identity()
        );
        assert!(
            sliced.check(&first).is_ok(),
            "membership rebinds stable axes to the reconstructed private decisions"
        );

        let general = sliced.canonicalize(sliced.universal_proposal()).unwrap();
        let general_root = sliced.general_construction().root;
        let general_path = finish(&mut sliced, ConstructionCoordinate::root(general_root), 1);
        assert_eq!(
            general,
            coordinate(&sliced, general_path),
            "required and ordinary routes name the same general coordinate"
        );
    }

    #[test]
    fn constructed_handle_pins_data_and_expressions_after_cache_eviction() {
        use crate::evaluation_session::boundary_tests::domain_with_optional;
        use crate::realization::demand_driven_tests::device;
        use crate::realization::demand_driven_tests::registry;
        let target = device();
        let registry = registry();
        let (mut domain, coordinate) = domain_with_optional(&target, &registry);
        let handle = domain.checked_preparation_candidate(&coordinate).unwrap();
        let (identity, constraint, expression) = {
            let read = domain.read(&handle).unwrap();
            let constraint = read.candidate().constraint();
            (
                read.candidate().family().identity().clone(),
                constraint,
                format!("{:?}", read.arena().view(AnyExpr::Bool(constraint))),
            )
        };
        domain.materialized.items.truncate(1);
        // Growing the shared arena and evicting a cache entry cannot change a
        // pinned physical member or the meaning of its expression handles.
        domain.arena_mut().nat(9_000_000_001);
        let read = domain.read(&handle).unwrap();
        assert_eq!(read.candidate().family().identity(), &identity);
        assert_eq!(read.candidate().constraint(), constraint);
        assert_eq!(
            format!("{:?}", read.arena().view(AnyExpr::Bool(constraint))),
            expression
        );
        assert!(read
            .candidate()
            .family()
            .kernels()
            .kernels()
            .next()
            .is_some());
        let (foreign, _) = domain_with_optional(&target, &registry);
        assert!(matches!(
            foreign.read(&handle),
            Err(CoordinateError::ForeignDomain)
        ));
    }

    fn declarations() -> (
        ExprArena,
        Vec<ChoiceDeclaration>,
        DecisionId,
        DecisionId,
        DecisionId,
    ) {
        let mut arena = ExprArena::new();
        let parent = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let child = arena.decision(FiniteDomain::new(vec![4, 8]).unwrap());
        let foreign = arena.decision(FiniteDomain::new(vec![7]).unwrap());
        let parent_active = arena.bool(true);
        let child_active = arena.decision_is(parent, 1);
        (
            arena,
            vec![
                ChoiceDeclaration {
                    kind: crate::refinement::ChoiceKind::WorkgroupSize,
                    decision: parent,
                    meaning: "algorithm",
                    active_when: parent_active,
                },
                ChoiceDeclaration {
                    kind: crate::refinement::ChoiceKind::WorkgroupSize,
                    decision: child,
                    meaning: "tile",
                    active_when: child_active,
                },
            ],
            parent,
            child,
            foreign,
        )
    }

    #[test]
    fn canonical_binding_omits_inactive_nested_choices() {
        let (arena, declarations, parent, child, _) = declarations();
        let (_, canonical) =
            canonical_choice_binding(&arena, &declarations, &[(child, 8), (parent, 0)]).unwrap();
        assert_eq!(canonical, vec![(parent, 0)]);
    }

    #[test]
    fn active_nested_choice_is_required_and_canonically_ordered() {
        let (arena, declarations, parent, child, _) = declarations();
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(parent, 1)]),
            Err(CoordinateError::MissingChoice(id)) if id == child
        ));
        let (_, canonical) =
            canonical_choice_binding(&arena, &declarations, &[(child, 4), (parent, 1)]).unwrap();
        assert_eq!(canonical, vec![(parent, 1), (child, 4)]);
    }

    #[test]
    fn malformed_choice_sets_are_rejected() {
        let (arena, declarations, parent, child, foreign) = declarations();
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(parent, 0), (parent, 0)]),
            Err(CoordinateError::DuplicateChoice(id)) if id == parent
        ));
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(foreign, 7)]),
            Err(CoordinateError::ForeignChoice(id)) if id == foreign
        ));
        assert!(matches!(
            canonical_choice_binding(&arena, &declarations, &[(parent, 1), (child, 5)]),
            Err(CoordinateError::ValueOutsideAxis { decision: id, value: 5 }) if id == child
        ));
    }
}
