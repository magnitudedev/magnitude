//! Plan-space assembly, solver export, and physical sealing (package P1).
//!
//! `form_plan_space` runs O1 -> (universal + catalog rules) -> S1 -> D1 ->
//! K1 -> M1 for every occurrence and alternative and admits only fully
//! sealed strategies; `solver_model` exports every decision and the exact
//! consequence facts; `resolve` substitutes a complete assignment, converts
//! every id to a dense index, proves every retained expression's bound under
//! the workload envelope, emits the invocation contract, and seals the
//! physical plan.
//!
//! Forbidden here: graph traversal, allocation policy, route inference,
//! resource recomputation, any default choice. Resolution is a pure
//! homomorphic substitution over the sealed strategies and the call
//! boundaries (a callee's boundary residences and scalar routes are replaced
//! by the caller's resolved routes for the same canonical leaves — the one
//! substitution D1's boundary design requires). A contradicted invariant
//! here is a compiler defect (`plan_space::defect`), never a runtime
//! category: every value was proven before substitution.

use crate::consequences::{ClosedStrategy, ConsequenceFormer};
use crate::failure::{CompilerDefect, Package};
use crate::ids::{
    BlockId, CanonicalLeafId, CanonicalValueId, DenseIndex, ExecutorScalarSlotId, InvocationValueId,
    OccurrenceId, OwnedGraphKey, OwnedValueRef, ResidenceId, StorageIx, StrategyId,
};
use crate::invocation::{
    AbiIntegerType, Bound, ExprBuilder, ExprDefect, GuardedExecutionExpr, InvocationContract,
    ScalarDomain, ShapeFieldContract,
};
use crate::kernel::{ExecutableDialect, KernelFormer, PlaneRef};
use crate::numerics;
use crate::occurrence::{InstantiatedInput, InstantiatedResult, OccurrenceFacts, OccurrenceForest};
use crate::physical::{
    AccessMode, ExecutionExpr, GuardPredicate, LaunchBinding, LaunchInput, LaunchOutput,
    LaunchResources, NativeFactDeclaration, PhysicalPlan, PhysicalSchedule, PhysicalStep,
    PlanResources, ResolutionIdentity, ResultField, ScalarSource, SealedBranch, SealedCall,
    SealedCarry, SealedFill, SealedGuard, SealedJoin, SealedLaunch, SealedRepeat, SealedValue,
    StatusField, StorageFact, StoragePlacement,
};
use crate::plan_space::{
    self, CompleteAssignment, Decision, NumericalContext, PlanSpace, SolvedValues,
    SolverModelView, StrategyFactsRef,
};
use crate::residence::{
    DataflowFormer, PullCounter, Residence, ResidenceChoice, ResidenceSource, StorageScope,
};
use crate::routes::{ScalarRoute, ValueRoute, ViewTransformTemplate};
use crate::strategy::{
    CarryEdge, ExecutorGuardPredicate, JoinEdge, MappingCatalog, NativeFactDomain,
    NativeFactKind, ParticipantPolicy, RuleQuery, ShapeSchedule, ShapeStep, StrategyFormer,
};
use crate::target::EffectiveTargetProfile;
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::logical::boundary::BoundaryLeaf;
use seismic_lang::logical::{
    self, Access, EffectiveTargetIdentity, GraphValueId, IdVec, LogicalProgram,
    RuntimeScalarExpr, SafetyObligation,
};
use seismic_lang::repr;
use seismic_lang::sir::ParamOwnership;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::types::{
    DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValuePath, ValueType,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// form_plan_space
// ---------------------------------------------------------------------------

pub(crate) fn form_plan_space<'l, D: ExecutableDialect>(
    logical: &'l LogicalProgram,
    profile: &EffectiveTargetProfile,
    catalog: &impl MappingCatalog<D>,
) -> Result<PlanSpace<'l, D>, CompilerDefect> {
    let forest = OccurrenceForest::expand(logical)?;
    let facts = forest.facts();
    let universal = crate::formation::strategy::universal_rules();
    let backend = catalog.rules();
    let mut strategies: Vec<(OccurrenceId, IdVec<StrategyId, ClosedStrategy<D>>)> = Vec::new();
    for (occurrence, record) in facts.occurrences() {
        let mut sealed = Vec::new();
        for alternative in record.alternatives.iter() {
            let query = RuleQuery {
                facts,
                profile,
                occurrence,
                logical_alternative: alternative.key.logical_alternative,
            };
            let mut proposals = Vec::new();
            for rule in universal.iter().chain(backend.iter()) {
                proposals.extend(rule.propose(&query));
            }
            let mut complete = 0usize;
            for proposal in proposals {
                let shape = StrategyFormer::form(facts, profile, proposal)?;
                let routed = DataflowFormer::form(facts, profile, shape)?;
                let mut kernels = Vec::with_capacity(routed.blocks().len());
                for (block, _) in routed.blocks().entries() {
                    kernels.push((
                        block,
                        KernelFormer::form::<D>(facts, &routed, block, catalog.intrinsics())?,
                    ));
                }
                let kernels = IdVec::from_iter(kernels);
                let consequences = ConsequenceFormer::derive::<D>(
                    &routed,
                    &kernels,
                    catalog.cost_model(),
                    &profile.limits,
                )?;
                sealed.push(ClosedStrategy::seal(routed, kernels, consequences));
                complete += 1;
            }
            if complete == 0 {
                return Err(CompilerDefect::new(
                    Package::S1,
                    format!(
                        "logical alternative {} of occurrence {} yields no complete strategy: \
                         the universal rules (universal.streaming, universal.point-serial) must \
                         propose and complete every applicable alternative",
                        alternative.key.logical_alternative, occurrence.0
                    ),
                ));
            }
        }
        strategies.push((occurrence, IdVec::new(sealed)));
    }
    Ok(PlanSpace::seal(
        logical,
        forest,
        profile.clone(),
        IdVec::from_iter(strategies),
        catalog.native_fact_domains().to_vec(),
    ))
}

// ---------------------------------------------------------------------------
// solver_model
// ---------------------------------------------------------------------------

pub(crate) fn solver_model<'s, D: ExecutableDialect>(
    space: &'s PlanSpace<'_, D>,
    numerics: &NumericalContext<'_>,
) -> SolverModelView<'s> {
    let facts = space.forest().facts();
    let entry = facts.entry();
    let table = space.strategies_table();
    let mut decisions = Vec::new();
    let mut exports = Vec::new();
    for (occurrence, strategies) in table.entries() {
        let options: Vec<StrategyId> = strategies.ids().collect();
        let inactive_allowed = occurrence != entry
            && table.iter().any(|strategies| {
                strategies
                    .iter()
                    .any(|closed| closed.routed().shape().absorbed().contains_key(&occurrence))
            });
        decisions.push(Decision::Strategy {
            occurrence,
            options,
            inactive_allowed,
        });
        for (strategy, closed) in strategies.entries() {
            exports.push(StrategyFactsRef {
                occurrence,
                strategy,
                consequences: closed.consequences(),
                residences: closed.routed().residences(),
                routed: closed.routed(),
                absorbed: closed.routed().shape().absorbed(),
            });
            for (parameter, declaration) in closed.consequences().tuning.entries() {
                decisions.push(Decision::Tuning {
                    occurrence,
                    strategy,
                    parameter,
                    lower: declaration.lower,
                    upper: declaration.upper,
                });
            }
            for (residence, record) in closed.routed().residences().iter() {
                if let ResidenceChoice::SolverChoice { options, var } = &record.choice {
                    decisions.push(Decision::Scope {
                        occurrence,
                        strategy,
                        residence,
                        var: *var,
                        options: options.as_slice().to_vec(),
                    });
                }
            }
        }
    }
    let mut runtime_capacities = BTreeMap::new();
    for extent in space.logical().runtime_extents() {
        runtime_capacities.insert(extent.id, extent.capacity);
    }
    let profile = space.profile();
    SolverModelView {
        decisions,
        facts: exports,
        arena_capacity: profile.limits.max_device_bytes,
        precision: numerics.precision.clone(),
        evidence: numerics.evidence.to_vec(),
        runtime_capacities,
        logical: space.logical().identity,
        target: EffectiveTargetIdentity {
            backend: profile.backend.clone(),
            capability_fingerprint: profile.capability_fingerprint.clone(),
        },
        toolchain: profile.toolchain_fingerprint.clone(),
        workload: numerics::workload_fingerprint(&space.logical().domain),
        entry,
    }
}

// ---------------------------------------------------------------------------
// resolve
// ---------------------------------------------------------------------------

pub(crate) fn resolve<D: ExecutableDialect>(
    space: PlanSpace<'_, D>,
    assignment: CompleteAssignment,
) -> PhysicalPlan<D> {
    let (logical, forest, profile, strategies, native_fact_domains) = space.into_parts();
    let facts = forest.facts();
    // Graph value ids are unique across the whole program; locate the one
    // owned graph that declares each value a runtime expression reads.
    let mut value_graph = BTreeMap::new();
    for (_, record) in facts.occurrences() {
        for alternative in record.alternatives.iter() {
            for value in facts.graph(alternative.key).values() {
                value_graph.entry(value.id()).or_insert(alternative.key);
            }
        }
    }
    // The descent spine's first entry is the entry occurrence and its
    // proven selection; the resolution consumes the spine positionally from
    // here.
    let resolved = assignment.resolved().clone();
    let spine = resolved.spine;
    let (entry_occurrence, entry_strategy) = match spine.first() {
        Some(entry) => *entry,
        None => unreachable!("validate produces a spine headed by the entry occurrence"),
    };
    let mut resolver = Resolver {
        logical,
        facts,
        profile,
        strategies,
        native_fact_domains,
        values: assignment.values().clone(),
        assignment,
        builder: ExprBuilder::new(),
        buffers: Vec::new(),
        buffer_of: BTreeMap::new(),
        scalars: Vec::new(),
        scalar_slot_of: BTreeMap::new(),
        abi_expr_of: BTreeMap::new(),
        aliases: Vec::new(),
        pending_ranges: BTreeMap::new(),
        result_fields: Vec::new(),
        result_field_of: BTreeMap::new(),
        extent_terms: BTreeMap::new(),
        storages: Vec::new(),
        storage_of: BTreeMap::new(),
        scalar_slots: BTreeMap::new(),
        status_fields: Vec::new(),
        value_graph,
        contexts: BTreeMap::new(),
        current: entry_occurrence,
        current_strategy: entry_strategy,
        current_subst: Substitution::default(),
        spine,
        spine_next: 1,
        launches: 0,
        calls: 0,
        branches: 0,
        repeats: 0,
        guards: 0,
        native_facts: 0,
        last_guard: None,
        entry: entry_occurrence,
    };
    resolver.build_abi(entry_strategy);
    let schedule =
        resolver.resolve_occurrence(entry_occurrence, entry_strategy, &Substitution::default());
    resolver.finish(schedule)
}

/// A scalar route fully resolved to the plan's dense domains.
#[derive(Clone)]
enum ResolvedScalar {
    Abi { slot: crate::ids::ScalarSlot },
    Executor { occurrence: OccurrenceId, slot: ExecutorScalarSlotId },
    Result { field: crate::ids::ResultFieldIx },
}

/// A tensor route fully resolved: the (occurrence, strategy, residence)
/// that physically holds it, with the value's transform composed in
/// residence coordinates and the routed value's view-side tensor type.
#[derive(Clone)]
struct ResolvedTensor {
    occurrence: OccurrenceId,
    strategy: StrategyId,
    residence: ResidenceId,
    access: Access,
    transform: ViewTransformTemplate,
    ty: TensorType,
}

#[derive(Clone)]
enum ResolvedLeaf {
    Scalar(ResolvedScalar),
    Tensor(ResolvedTensor),
}

/// The call-boundary substitution: the callee's boundary residences and
/// scalar leaves replaced by the caller's resolved routes for the same
/// canonical leaves.
#[derive(Clone, Default)]
struct Substitution {
    tensors: BTreeMap<ResidenceId, ResolvedTensor>,
    scalars: BTreeMap<CanonicalLeafId, ResolvedScalar>,
}

struct Resolver<'l, D: ExecutableDialect> {
    logical: &'l LogicalProgram,
    facts: &'l OccurrenceFacts<'l>,
    profile: EffectiveTargetProfile,
    strategies: IdVec<OccurrenceId, IdVec<StrategyId, ClosedStrategy<D>>>,
    native_fact_domains: Vec<NativeFactDomain>,
    assignment: CompleteAssignment,
    values: SolvedValues,
    builder: ExprBuilder,
    // root ABI
    buffers: Vec<crate::invocation::BufferContract>,
    buffer_of: BTreeMap<(OccurrenceId, ResidenceId, u32), crate::ids::BufferSlot>,
    scalars: Vec<crate::invocation::ScalarContract>,
    scalar_slot_of: BTreeMap<(BoundaryLeaf, Option<RangeEndpoint>), crate::ids::ScalarSlot>,
    abi_expr_of: BTreeMap<crate::ids::ScalarSlot, InvocationValueId>,
    aliases: Vec<crate::invocation::AliasContract>,
    /// Range parameters whose endpoints were contracted; the relation is
    /// recorded once both endpoints exist.
    pending_ranges:
        BTreeMap<ValuePath, (InvocationValueId, Option<InvocationValueId>, Option<InvocationValueId>)>,
    result_fields: Vec<ResultField>,
    result_field_of: BTreeMap<(ValuePath, Option<RangeEndpoint>), crate::ids::ResultFieldIx>,
    /// The actual-value term of every runtime extent built so far
    /// (invocation-derived or guarded executor arithmetic).
    extent_terms: BTreeMap<RuntimeExtentId, Term>,
    // dense global allocations
    storages: Vec<crate::physical::PhysicalStorage<D>>,
    storage_of: BTreeMap<(OccurrenceId, ResidenceId, u32), StorageIx>,
    scalar_slots: BTreeMap<(OccurrenceId, ExecutorScalarSlotId), crate::ids::ScalarSlotIx>,
    status_fields: Vec<StatusField>,
    // value identity and resolution contexts
    value_graph: BTreeMap<GraphValueId, OwnedGraphKey>,
    /// The (strategy, substitution) of every inlined occurrence
    /// (runtime-extent expressions resolve in whichever context routes
    /// their leaves).
    contexts: BTreeMap<OccurrenceId, (StrategyId, Substitution)>,
    current: OccurrenceId,
    current_strategy: StrategyId,
    current_subst: Substitution,
    /// The validated descent spine, consumed in order at every call
    /// descent (each entry proves the selection of the occurrence the
    /// resolution reaches).
    spine: Vec<(OccurrenceId, StrategyId)>,
    spine_next: usize,
    launches: usize,
    calls: usize,
    branches: usize,
    repeats: usize,
    guards: usize,
    native_facts: usize,
    /// The most recently sealed executor guard: the guard that structurally
    /// dominates the partial executor arithmetic sealed after it.
    last_guard: Option<crate::ids::GuardIx>,
    entry: OccurrenceId,
}

impl<'l, D: ExecutableDialect> Resolver<'l, D> {
    // -- selection ----------------------------------------------------------

    /// The sealed strategy of one (occurrence, strategy) pair the spine or
    /// the current context names; dense indexing, no lookup.
    fn strategy(
        &self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
    ) -> &ClosedStrategy<D> {
        &self.strategies[occurrence][strategy]
    }

    /// The next descent entry: the (occurrence, strategy) the resolution
    /// reaches at one call descent, in the spine's proven order.
    fn next_spine_entry(&mut self) -> (OccurrenceId, StrategyId) {
        let entry = self.spine[self.spine_next];
        self.spine_next += 1;
        entry
    }

    // -- routes ---------------------------------------------------------------

    fn resolve_leaf(
        &self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        leaf: CanonicalLeafId,
    ) -> ResolvedLeaf {
        let routes = self.strategy(occurrence, strategy).routed().routes();
        match routes.route(leaf) {
            // D1 never routes a void leaf (a void value has no leaf).
            ValueRoute::Void => unreachable!("D1 never routes a void leaf"),
            ValueRoute::Scalar(scalar) => {
                if let Some(resolved) = subst.scalars.get(&leaf) {
                    return ResolvedLeaf::Scalar(resolved.clone());
                }
                ResolvedLeaf::Scalar(match scalar {
                    ScalarRoute::RootAbi { leaf, endpoint } => ResolvedScalar::Abi {
                        slot: self.abi_slot(leaf, *endpoint),
                    },
                    ScalarRoute::ExecutorSlot(slot) => {
                        ResolvedScalar::Executor { occurrence, slot: *slot }
                    }
                    ScalarRoute::ResultField { leaf, endpoint } => ResolvedScalar::Result {
                        field: self.result_field(leaf, *endpoint),
                    },
                })
            }
            ValueRoute::Tensor(tensor) => {
                let ty = self.tensor_type_of_leaf(leaf);
                if let Some(resolved) = subst.tensors.get(&tensor.residence) {
                    return ResolvedLeaf::Tensor(ResolvedTensor {
                        occurrence: resolved.occurrence,
                        strategy: resolved.strategy,
                        residence: resolved.residence,
                        access: tensor.access,
                        transform: ViewTransformTemplate::compose(
                            &resolved.transform,
                            &tensor.transform,
                        ),
                        ty,
                    });
                }
                ResolvedLeaf::Tensor(ResolvedTensor {
                    occurrence,
                    strategy,
                    residence: tensor.residence,
                    access: tensor.access,
                    transform: tensor.transform.clone(),
                    ty,
                })
            }
        }
    }

    /// The ABI scalar slot of one entry-boundary leaf: build_abi allocated
    /// a contract for every scalar leaf of the entry's inputs, so the table
    /// is total over the leaves the entry's routes name.
    fn abi_slot(
        &self,
        leaf: &BoundaryLeaf,
        endpoint: Option<RangeEndpoint>,
    ) -> crate::ids::ScalarSlot {
        self.scalar_slot_of[&(leaf.clone(), endpoint)]
    }

    /// The result field of one result leaf: build_abi allocated a field for
    /// every scalar leaf of the entry's results, so the table is total over
    /// the leaves the entry's routes name.
    fn result_field(
        &self,
        boundary: &BoundaryLeaf,
        endpoint: Option<RangeEndpoint>,
    ) -> crate::ids::ResultFieldIx {
        match boundary {
            BoundaryLeaf::Result { leaf } => {
                self.result_field_of[&(leaf.clone(), endpoint)]
            }
            // A result-field route names an entry result leaf.
            BoundaryLeaf::Input { .. } => {
                unreachable!("a result-field route names a result leaf")
            }
        }
    }

    /// The global dense executor slot of one strategy-local slot.
    fn global_slot(
        &mut self,
        occurrence: OccurrenceId,
        slot: ExecutorScalarSlotId,
    ) -> crate::ids::ScalarSlotIx {
        if let Some(existing) = self.scalar_slots.get(&(occurrence, slot)) {
            return *existing;
        }
        let index = crate::ids::ScalarSlotIx::from_index(self.scalar_slots.len());
        self.scalar_slots.insert((occurrence, slot), index);
        index
    }

    // -- storages ---------------------------------------------------------------

    /// The solved scope of one residence: the fixed scope, or the cell of
    /// the validate-produced dense scope table (every choice var of every
    /// strategy is decoded, so the table is total).
    fn selected_scope(
        &self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        residence: ResidenceId,
    ) -> StorageScope {
        let record = self
            .strategy(occurrence, strategy)
            .routed()
            .residences()
            .residence(residence);
        match &record.choice {
            ResidenceChoice::Fixed(scope) => *scope,
            ResidenceChoice::SolverChoice { var, .. } => {
                self.assignment.resolved().scopes[occurrence.0 as usize]
                    [strategy.0 as usize][var.0 as usize]
            }
        }
    }

    /// The dense storages of one residence's planes, allocated once.
    fn ensure_storages(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        residence: ResidenceId,
    ) -> Vec<StorageIx> {
        let plane_count = self
            .strategy(occurrence, strategy)
            .routed()
            .residences()
            .residence(residence)
            .planes
            .len();
        let mut cached = Vec::with_capacity(plane_count);
        let mut missing = false;
        for plane in 0..plane_count {
            match self.storage_of.get(&(occurrence, residence, plane as u32)) {
                Some(ix) => cached.push(*ix),
                None => {
                    missing = true;
                    break;
                }
            }
        }
        if !missing {
            return cached;
        }
        let record = self
            .strategy(occurrence, strategy)
            .routed()
            .residences()
            .residence(residence)
            .clone();
        let scope = self.selected_scope(occurrence, strategy, residence);
        let mut out = Vec::with_capacity(plane_count);
        for (plane, plan) in record.planes.iter().enumerate() {
            let plane = plane as u32;
            let placement = match scope {
                // build_abi allocated a buffer for every plane of every
                // root-boundary residence, so the table is total over the
                // ABI-scoped residences the seal reads.
                StorageScope::Abi => StoragePlacement::Abi {
                    slot: self.buffer_of[&(occurrence, residence, plane)],
                },
                // Arena placement proves the cell was certified, so the
                // validate-produced dense offset table is total here.
                StorageScope::DeviceArena => StoragePlacement::Arena {
                    offset: self.assignment.resolved().offsets[occurrence.0 as usize]
                        [strategy.0 as usize][residence.0 as usize][plane as usize],
                },
                StorageScope::Workgroup => StoragePlacement::Workgroup,
                StorageScope::Participant => StoragePlacement::Participant,
            };
            let bytes = self.eval_strategy_sym(occurrence, strategy, &plan.bytes);
            let layout = {
                let descriptor = plane_descriptor(&record.shape, &plan.plane);
                let template = if scope == StorageScope::Abi {
                    D::public_layout(&record.shape, descriptor)
                } else {
                    D::internal_layout(&record.shape, descriptor)
                };
                D::resolve_layout(&template, &self.values)
            };
            let ix = StorageIx::from_index(self.storages.len());
            self.storages.push(crate::physical::PhysicalStorage {
                placement,
                replication: record.replication,
                bytes,
                alignment: plan.alignment.max(1),
                plane: plan.plane.clone(),
                layout,
            });
            self.storage_of.insert((occurrence, residence, plane), ix);
            out.push(ix);
        }
        out
    }

    /// Evaluate one of the strategy's retained expressions: the tuning atoms
    /// are qualified by the strategy's identity and every qualified symbol is
    /// solved. The identical qualification is applied by the solver export.
    fn eval_strategy_sym(
        &self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        expr: &Sym,
    ) -> u64 {
        let qualified = qualify(expr, occurrence, strategy);
        self.values.eval(&qualified)
    }

    // -- the schedule -----------------------------------------------------------

    fn resolve_occurrence(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
    ) -> PhysicalSchedule<D> {
        self.contexts.insert(occurrence, (strategy, subst.clone()));
        let schedule = self
            .strategy(occurrence, strategy)
            .routed()
            .shape()
            .schedule()
            .clone();
        let saved = (self.current, self.current_strategy, self.current_subst.clone());
        self.current = occurrence;
        self.current_strategy = strategy;
        self.current_subst = subst.clone();
        let resolved = self.resolve_shape_schedule(occurrence, strategy, subst, &schedule);
        self.current = saved.0;
        self.current_strategy = saved.1;
        self.current_subst = saved.2;
        resolved
    }

    fn resolve_shape_schedule(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        schedule: &ShapeSchedule,
    ) -> PhysicalSchedule<D> {
        let mut steps = Vec::with_capacity(schedule.steps.len());
        for step in &schedule.steps {
            if let Some(step) = self.resolve_step(occurrence, strategy, subst, step) {
                steps.push(step);
            }
        }
        PhysicalSchedule { steps }
    }

    fn resolve_step(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        step: &ShapeStep,
    ) -> Option<PhysicalStep<D>> {
        match step {
            ShapeStep::Launch(block) => {
                let interface = self.strategy(occurrence, strategy).routed().blocks()[*block]
                    .interface
                    .clone();
                if interface.iteration.launch_condition()
                    == crate::dispatch::LaunchCondition::AlwaysSkip
                {
                    // Zero work: no native dispatch is submitted.
                    return None;
                }
                Some(PhysicalStep::Launch(
                    self.resolve_launch(occurrence, strategy, subst, *block),
                ))
            }
            ShapeStep::Guard { obligation, predicate } => {
                Some(self.resolve_guard(obligation, predicate))
            }
            ShapeStep::PullCounterReset { block } => {
                let (residence, skip) = {
                    let routed_block =
                        &self.strategy(occurrence, strategy).routed().blocks()[*block];
                    let skip = routed_block.interface.iteration.launch_condition()
                        == crate::dispatch::LaunchCondition::AlwaysSkip;
                    (routed_block.pull_counter, skip)
                };
                if skip {
                    return None;
                }
                // M1's seal admits a reset step only for a block with a
                // pull-counter residence.
                let PullCounter::Counter(residence) = residence else {
                    unreachable!("a pull-counter reset step names a counter residence (M1 seal)")
                };
                let storage = self.ensure_storages(occurrence, strategy, residence)[0];
                let bytes = {
                    let plane = self
                        .strategy(occurrence, strategy)
                        .routed()
                        .residences()
                        .residence(residence)
                        .planes
                        .first()
                        .clone();
                    self.eval_strategy_sym(occurrence, strategy, &plane.bytes)
                };
                Some(PhysicalStep::Fill(SealedFill { storage, bytes }))
            }
            ShapeStep::Call { .. } => {
                // The spine hands over the proven selection of the child
                // this descent reaches, in the resolution's own order.
                let (child, child_strategy) = self.next_spine_entry();
                let child_subst =
                    self.child_substitution(occurrence, strategy, child, child_strategy, subst);
                let body = self.resolve_occurrence(child, child_strategy, &child_subst);
                let id = crate::ids::CallIx::from_index(self.calls);
                self.calls += 1;
                Some(PhysicalStep::Call(SealedCall { id, body }))
            }
            ShapeStep::If {
                condition,
                then_schedule,
                else_schedule,
                joins,
                ..
            } => {
                let condition =
                    self.scalar_source_of_value(occurrence, strategy, subst, *condition);
                let then_schedule =
                    self.resolve_shape_schedule(occurrence, strategy, subst, then_schedule);
                let else_schedule =
                    self.resolve_shape_schedule(occurrence, strategy, subst, else_schedule);
                let mut sealed_joins = Vec::with_capacity(joins.len());
                for join in joins {
                    if let JoinEdge::Value {
                        then_value,
                        else_value,
                        joined,
                    } = join
                    {
                        for leaf in 0..self.leaf_count(*joined) {
                            sealed_joins.push(SealedJoin {
                                then_value: self
                                    .sealed_value_leaf(occurrence, strategy, subst, *then_value, leaf),
                                else_value: self
                                    .sealed_value_leaf(occurrence, strategy, subst, *else_value, leaf),
                                joined: self
                                    .sealed_value_leaf(occurrence, strategy, subst, *joined, leaf),
                            });
                        }
                    }
                }
                let id = crate::ids::BranchIx::from_index(self.branches);
                self.branches += 1;
                Some(PhysicalStep::If(SealedBranch {
                    id,
                    condition,
                    then_schedule,
                    else_schedule,
                    joins: sealed_joins,
                }))
            }
            ShapeStep::Repeat {
                kind,
                start,
                end,
                bound,
                binder,
                body,
                carries,
                ..
            } => {
                if bound.as_static() == Some(0) {
                    // Zero visits: the loop and its whole body never execute.
                    return None;
                }
                let start = self.value_execution_expr(occurrence, strategy, subst, *start);
                let end = self.value_execution_expr(occurrence, strategy, subst, *end);
                let bound = self.extent_execution_expr(bound);
                // An invocation-known repeat range is validated by the
                // contract before submission; the relation's proof holds
                // wherever the range is orderable at all.
                if let (
                    ExecutionExpr::Invocation(start_id),
                    ExecutionExpr::Invocation(end_id),
                    ExecutionExpr::Invocation(bound_id),
                ) = (&start, &end, &bound)
                {
                    let relation = self.builder.range_ordered(*start_id, *end_id, *bound_id);
                    if relation.is_err() {
                        unreachable!("a retained repeat range is orderable (S1 static proof)");
                    }
                }
                let binder = self.binder_slot(occurrence, strategy, subst, *binder);
                let body = self.resolve_shape_schedule(occurrence, strategy, subst, body);
                let mut sealed_carries = Vec::with_capacity(carries.len());
                for carry in carries {
                    if let CarryEdge::Value {
                        initial,
                        parameter,
                        update,
                        result,
                    } = carry
                    {
                        for leaf in 0..self.leaf_count(*initial) {
                            sealed_carries.push(SealedCarry {
                                initial: self
                                    .sealed_value_leaf(occurrence, strategy, subst, *initial, leaf),
                                current: self
                                    .sealed_value_leaf(occurrence, strategy, subst, *parameter, leaf),
                                update: self
                                    .sealed_value_leaf(occurrence, strategy, subst, *update, leaf),
                                result: self
                                    .sealed_value_leaf(occurrence, strategy, subst, *result, leaf),
                            });
                        }
                    }
                }
                let id = crate::ids::RepeatIx::from_index(self.repeats);
                self.repeats += 1;
                Some(PhysicalStep::Repeat(SealedRepeat {
                    id,
                    kind: *kind,
                    start,
                    end,
                    bound,
                    binder,
                    body,
                    carries: sealed_carries,
                }))
            }
        }
    }

    // -- launches ------------------------------------------------------------

    fn resolve_launch(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        block: BlockId,
    ) -> SealedLaunch<D> {
        let selected = self.strategy(occurrence, strategy);
        let interface = selected.routed().blocks()[block].interface.clone();
        let kernel = selected.kernels()[block].clone();
        let facts = selected.consequences().blocks[block].clone();
        let policy = selected.routed().shape().blocks()[block]
            .participants
            .policy
            .clone();
        let pull_counter = selected.routed().blocks()[block].pull_counter;

        // Geometry: the exact runtime total and the solved participant count.
        // A serialized traversal owns the whole domain with one participant,
        // so its native grid is a single workgroup; every other policy covers
        // the domain by grid stride, with `ceil(work_items / participants)`
        // workgroups along the first axis.
        let participants = self
            .eval_strategy_sym(occurrence, strategy, &interface.iteration.participants);
        let participants_id = self.constant_node(participants);
        let work_term = self.linear_total_term(&interface.iteration.total);
        let workgroups = if matches!(policy, ParticipantPolicy::Serial) {
            [
                ExecutionExpr::Invocation(self.constant_node(1)),
                ExecutionExpr::Invocation(self.constant_node(1)),
                ExecutionExpr::Invocation(self.constant_node(1)),
            ]
        } else {
            let workgroups0 = match &work_term {
                Term::Inv(id) => {
                    let divided = self.builder.ceil_div(*id, participants_id);
                    Term::Inv(self.expr(divided))
                }
                Term::Result { field, bound } => {
                    // S1 seals a dominating executor guard before any launch
                    // whose total is value-derived.
                    let guard = match self.last_guard {
                        Some(guard) => guard,
                        None => unreachable!(
                            "value-derived launch geometry is dominated by a sealed guard (S1)"
                        ),
                    };
                    Term::Guarded(GuardedExecutionExpr::CeilDiv(
                        Box::new(GuardedExecutionExpr::ResultField {
                            field: *field,
                            bound: *bound,
                            guard,
                        }),
                        Box::new(GuardedExecutionExpr::Invocation(participants_id)),
                        guard,
                    ))
                }
                Term::Guarded(expr) => {
                    let guard = match self.last_guard {
                        // S1 seals a dominating executor guard before any
                        // launch whose total is value-derived.
                        Some(guard) => guard,
                        None => unreachable!(
                            "value-derived launch geometry is dominated by a sealed guard (S1)"
                        ),
                    };
                    Term::Guarded(GuardedExecutionExpr::CeilDiv(
                        Box::new(expr.clone()),
                        Box::new(GuardedExecutionExpr::Invocation(participants_id)),
                        guard,
                    ))
                }
            };
            [
                workgroups0.into_execution(),
                ExecutionExpr::Invocation(self.constant_node(1)),
                ExecutionExpr::Invocation(self.constant_node(1)),
            ]
        };
        let work_items = work_term.into_execution();

        // Kernel inputs, outputs, and locals in interface id order.
        let mut scalar_bindings = 0u32;
        let mut inputs = Vec::with_capacity(interface.inputs.len());
        for decl in interface.inputs.iter() {
            let input = match self.resolve_leaf(occurrence, strategy, subst, decl.leaf) {
                ResolvedLeaf::Scalar(resolved) => {
                    let slot = scalar_bindings;
                    scalar_bindings += 1;
                    LaunchInput::Scalar {
                        slot,
                        source: self.scalar_source(resolved),
                        dtype: self.leaf_dtype(decl.leaf),
                    }
                }
                ResolvedLeaf::Tensor(resolved) => {
                    LaunchInput::Storage(self.tensor_views(&resolved))
                }
            };
            inputs.push(input);
        }
        let mut outputs = Vec::with_capacity(interface.outputs.len());
        for decl in interface.outputs.iter() {
            let output = match self.resolve_leaf(occurrence, strategy, subst, decl.leaf) {
                ResolvedLeaf::Scalar(ResolvedScalar::Executor { occurrence, slot }) => {
                    let slot = self.global_slot(occurrence, slot);
                    LaunchOutput::ExecutorSlot {
                        slot,
                        dtype: self.leaf_dtype(decl.leaf),
                    }
                }
                ResolvedLeaf::Scalar(ResolvedScalar::Result { field }) => {
                    LaunchOutput::ResultField {
                        field,
                        dtype: self.leaf_dtype(decl.leaf),
                    }
                }
                // D1 never routes a produced value to a root ABI input.
                ResolvedLeaf::Scalar(ResolvedScalar::Abi { .. }) => unreachable!(
                    "a kernel output never publishes into a root ABI input scalar (D1)"
                ),
                ResolvedLeaf::Tensor(resolved) => {
                    LaunchOutput::Storage(self.tensor_views(&resolved))
                }
            };
            outputs.push(output);
        }
        let mut locals = Vec::with_capacity(interface.locals.len());
        let mut workgroup_storage = Vec::new();
        let mut participant_storage = Vec::new();
        for decl in interface.locals.iter() {
            // Kernel-local residences are internal to this strategy: never
            // substituted.
            let resolved = ResolvedTensor {
                occurrence,
                strategy,
                residence: decl.route.residence,
                access: decl.route.access,
                transform: decl.route.transform.clone(),
                ty: self.tensor_type_of_value(decl.value),
            };
            let views = self.tensor_views(&resolved);
            match self.selected_scope(occurrence, strategy, resolved.residence) {
                StorageScope::Workgroup => {
                    workgroup_storage.extend(views.as_slice().iter().map(|view| view.storage))
                }
                StorageScope::Participant => {
                    participant_storage.extend(views.as_slice().iter().map(|view| view.storage))
                }
                _ => {}
            }
            locals.push(views);
        }

        // Status fields: one dense field per template, in kernel order.
        let mut status_fields = Vec::with_capacity(kernel.status_fields().len());
        for (_, obligation) in kernel.status_fields() {
            status_fields.push(self.allocate_status_field(obligation.clone()));
        }

        // Direct storage bindings, in input/output/local order, then the
        // pull counter.
        let mut bindings = Vec::new();
        for input in &inputs {
            if let LaunchInput::Storage(views) = input {
                for view in views.as_slice() {
                    bindings.push(LaunchBinding {
                        slot: bindings.len() as u32,
                        storage: view.storage,
                        access: input_access(view.access),
                    });
                }
            }
        }
        for output in &outputs {
            if let LaunchOutput::Storage(views) = output {
                for view in views.as_slice() {
                    bindings.push(LaunchBinding {
                        slot: bindings.len() as u32,
                        storage: view.storage,
                        access: output_access(view.access),
                    });
                }
            }
        }
        for local in &locals {
            for view in local.as_slice() {
                bindings.push(LaunchBinding {
                    slot: bindings.len() as u32,
                    storage: view.storage,
                    access: AccessMode::ReadWrite,
                });
            }
        }
        let mut counter_storage = None;
        if let PullCounter::Counter(residence) = pull_counter {
            let storage = self.ensure_storages(occurrence, strategy, residence)[0];
            bindings.push(LaunchBinding {
                slot: bindings.len() as u32,
                storage,
                access: AccessMode::Atomic,
            });
            counter_storage = Some(storage);
        }

        // The address-space fact of every binding, in binding-slot order.
        let storage_facts = bindings
            .iter()
            .map(|binding| {
                let storage = &self.storages[binding.storage.index()];
                match storage.placement {
                    StoragePlacement::Abi { .. } | StoragePlacement::Arena { .. } => {
                        StorageFact::Global
                    }
                    StoragePlacement::Workgroup => StorageFact::Workgroup {
                        bytes: storage.bytes,
                        alignment: storage.alignment,
                    },
                    StoragePlacement::Participant => StorageFact::Participant {
                        bytes: storage.bytes,
                        alignment: storage.alignment,
                    },
                }
            })
            .collect();

        // The exact selected resource contract, evaluated from M1's
        // expressions.
        let resources = LaunchResources {
            workgroup_bytes: self
                .eval_strategy_sym(occurrence, strategy, &facts.workgroup_bytes),
            private_bytes_per_participant: self.eval_strategy_sym(
                occurrence,
                strategy,
                &facts.private_bytes_per_participant,
            ),
            direct_bindings: facts.direct_bindings,
            static_code_units: facts.static_code_units,
            required_subgroup_width: facts.required_subgroup_width,
            barriers: facts.barriers,
        };

        // Native fact declarations this launch must have reflected.
        let mut native_facts = Vec::new();
        if matches!(policy, ParticipantPolicy::GridCooperative { .. }) {
            native_facts.push(self.native_fact(NativeFactKind::MaxResidentParticipants));
        }
        if facts.required_subgroup_width.is_some() {
            native_facts.push(self.native_fact(NativeFactKind::NativeSubgroupWidth));
        }

        let id = crate::ids::LaunchIx::from_index(self.launches);
        self.launches += 1;
        SealedLaunch {
            id,
            work_items,
            participants: ExecutionExpr::Invocation(participants_id),
            workgroups,
            bindings,
            storage_facts,
            workgroup_storage,
            participant_storage,
            kernel,
            inputs,
            outputs,
            locals,
            status_fields,
            pull_counter: counter_storage,
            resources,
            native_facts,
        }
    }

    fn native_fact(&mut self, kind: NativeFactKind) -> NativeFactDeclaration {
        // form_plan_space retained the catalog's declared domains; a launch
        // asks only for the kinds its policy and resources require, which
        // the catalog contract covers.
        let domain = self
            .native_fact_domains
            .iter()
            .find(|domain| domain.kind == kind)
            .unwrap_or_else(|| {
                unreachable!("the backend catalog declares a domain for every native fact kind it can require")
            });
        let index = crate::ids::NativeFactIx::from_index(self.native_facts);
        self.native_facts += 1;
        NativeFactDeclaration {
            index,
            kind,
            min: domain.min,
            max: domain.max,
        }
    }

    fn allocate_status_field(
        &mut self,
        obligation: crate::ids::ObligationRef,
    ) -> crate::ids::StatusFieldIx {
        let node = match self.facts.node(&obligation.node) {
            Ok(node) => node,
            // An obligation names a node of the strategy's sealed shape.
            Err(_) => unreachable!("an obligation names a minted node (S1 seal)"),
        };
        let kind = safety_kind(&node.safety[obligation.index as usize]);
        let index = crate::ids::StatusFieldIx::from_index(self.status_fields.len());
        self.status_fields.push(StatusField {
            index,
            obligation: obligation.clone(),
            kind,
        });
        index
    }

    fn resolve_guard(
        &mut self,
        obligation: &crate::ids::ObligationRef,
        predicate: &ExecutorGuardPredicate,
    ) -> PhysicalStep<D> {
        let status = self.allocate_status_field(obligation.clone());
        let id = crate::ids::GuardIx::from_index(self.guards);
        self.guards += 1;
        self.last_guard = Some(id);
        let predicate = match predicate {
            ExecutorGuardPredicate::ProductFits { factors, bits } => {
                GuardPredicate::ProductFits {
                    factors: factors
                        .iter()
                        .map(|factor| self.extent_execution_expr(factor))
                        .collect(),
                    bits: *bits,
                }
            }
            ExecutorGuardPredicate::ExtentPositive { extent } => {
                GuardPredicate::ExtentPositive {
                    extent: self.extent_execution_expr(extent),
                }
            }
        };
        PhysicalStep::Guard(SealedGuard {
            id,
            obligation: obligation.clone(),
            status,
            predicate,
        })
    }

    // -- the call boundary substitution -----------------------------------

    fn child_substitution(
        &mut self,
        parent: OccurrenceId,
        parent_strategy: StrategyId,
        child: OccurrenceId,
        child_strategy: StrategyId,
        parent_subst: &Substitution,
    ) -> Substitution {
        let alternative = self
            .strategy(child, child_strategy)
            .routed()
            .shape()
            .root()
            .logical_alternative;
        let boundary = self
            .facts
            .alternative(OwnedGraphKey {
                occurrence: child,
                logical_alternative: alternative,
            })
            .boundary
            .clone();
        let mut subst = Substitution::default();
        for input in boundary.inputs.values() {
            // O1 instantiates call inputs for every non-entry boundary.
            let InstantiatedInput::Call { callee, .. } = input else {
                unreachable!("a non-entry boundary instantiates call inputs (O1)")
            };
            self.bind_boundary_leaf(
                parent,
                parent_strategy,
                child,
                child_strategy,
                parent_subst,
                *callee,
                &mut subst,
            );
        }
        for result in boundary.results.values() {
            let InstantiatedResult::Call { callee, .. } = result else {
                unreachable!("a non-entry boundary instantiates call results (O1)")
            };
            self.bind_boundary_leaf(
                parent,
                parent_strategy,
                child,
                child_strategy,
                parent_subst,
                *callee,
                &mut subst,
            );
        }
        subst
    }

    fn bind_boundary_leaf(
        &mut self,
        parent: OccurrenceId,
        parent_strategy: StrategyId,
        child: OccurrenceId,
        child_strategy: StrategyId,
        parent_subst: &Substitution,
        callee: crate::ids::OwnedValueRef,
        subst: &mut Substitution,
    ) {
        let value = match self.facts.canonical_value(callee) {
            Ok(value) => value,
            // A boundary instantiation names minted values.
            Err(_) => unreachable!("a boundary instantiation names minted values (O1)"),
        };
        for leaf in self.facts.leaves(value).iter().copied() {
            let child_route = self
                .strategy(child, child_strategy)
                .routed()
                .routes()
                .route(leaf)
                .clone();
            let resolved = self.resolve_leaf(parent, parent_strategy, parent_subst, leaf);
            match (&child_route, &resolved) {
                (ValueRoute::Scalar(_), ResolvedLeaf::Scalar(scalar)) => {
                    subst.scalars.insert(leaf, scalar.clone());
                }
                (ValueRoute::Tensor(child_tensor), ResolvedLeaf::Tensor(resolved_tensor)) => {
                    subst.tensors.insert(
                        child_tensor.residence,
                        ResolvedTensor {
                            occurrence: resolved_tensor.occurrence,
                            strategy: resolved_tensor.strategy,
                            residence: resolved_tensor.residence,
                            access: child_tensor.access,
                            transform: ViewTransformTemplate::compose(
                                &resolved_tensor.transform,
                                &child_tensor.transform,
                            ),
                            ty: self.tensor_type_of_leaf(leaf),
                        },
                    );
                }
                // One canonical leaf has one value kind; both strategies
                // route it by that kind.
                _ => unreachable!("both sides route one canonical leaf by its one kind"),
            }
        }
    }

    // -- values and scalars --------------------------------------------------

    fn leaf_count(&self, value: CanonicalValueId) -> usize {
        self.facts.leaves(value).len()
    }

    fn leaf_of(&self, value: CanonicalValueId, index: usize) -> CanonicalLeafId {
        self.facts.leaves(value)[index]
    }

    /// The leaf type of one canonical leaf: the component type at its path.
    fn leaf_type(&self, leaf: CanonicalLeafId) -> ValueType {
        let record = self.facts.leaf(leaf);
        let mut ty = self.facts.value_type(record.value);
        for component in record.path.0.iter() {
            match ty {
                // A leaf path is built from the tuple structure it indexes.
                ValueType::Tuple(items) => {
                    ty = items.as_slice()[*component as usize].clone();
                }
                _ => unreachable!("a leaf path traverses only tuple types (O1)"),
            }
        }
        ty
    }

    fn leaf_dtype(&self, leaf: CanonicalLeafId) -> DType {
        match self.leaf_type(leaf) {
            ValueType::Scalar(dtype) => dtype,
            ValueType::Index { .. } | ValueType::Range { .. } => DType::I32,
            // Every caller holds a scalar-kind leaf.
            _ => unreachable!("a kernel scalar leaf holds a scalar, index, or range"),
        }
    }

    /// The view-side tensor type of one tensor-routed leaf: the routed
    /// value's type, not the residence's storage type.
    fn tensor_type_of_leaf(&self, leaf: CanonicalLeafId) -> TensorType {
        self.tensor_type_of_value(self.facts.leaf(leaf).value)
    }

    /// The view-side tensor type of one tensor-routed canonical value.
    fn tensor_type_of_value(&self, value: CanonicalValueId) -> TensorType {
        match self.facts.value_type(value) {
            ValueType::Tensor(ty) => ty,
            // D1 routes a tensor leaf only for a tensor value.
            _ => unreachable!("a tensor-routed value is a tensor (D1)"),
        }
    }

    fn scalar_source_of_value(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        value: CanonicalValueId,
    ) -> ScalarSource {
        let leaf = self.leaf_of(value, 0);
        match self.resolve_leaf(occurrence, strategy, subst, leaf) {
            ResolvedLeaf::Scalar(resolved) => self.scalar_source(resolved),
            // A control scalar's value is scalar-kind, so its leaf routes
            // scalar.
            _ => unreachable!("a control scalar's leaf routes scalar"),
        }
    }

    fn scalar_source(&mut self, resolved: ResolvedScalar) -> ScalarSource {
        match resolved {
            ResolvedScalar::Abi { slot } => ScalarSource::Abi(slot),
            ResolvedScalar::Executor { occurrence, slot } => {
                ScalarSource::Executor(self.global_slot(occurrence, slot))
            }
            ResolvedScalar::Result { field } => ScalarSource::Result(field),
        }
    }

    /// One leaf of a joined or carried value: a kernel scalar or a tensor view.
    fn sealed_value_leaf(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        value: CanonicalValueId,
        leaf_index: usize,
    ) -> SealedValue {
        let leaf = self.leaf_of(value, leaf_index);
        match self.resolve_leaf(occurrence, strategy, subst, leaf) {
            ResolvedLeaf::Scalar(resolved) => {
                SealedValue::Scalar(self.scalar_source(resolved))
            }
            ResolvedLeaf::Tensor(resolved) => {
                SealedValue::Tensor(self.tensor_views(&resolved))
            }
        }
    }

    /// The physical views of one resolved tensor: one per plane of its
    /// residence, in the residence's plane order, each carrying the routed
    /// value's view-side tensor type.
    fn tensor_views(
        &mut self,
        resolved: &ResolvedTensor,
    ) -> NonEmpty<crate::physical::PhysicalStorageView> {
        let storages =
            self.ensure_storages(resolved.occurrence, resolved.strategy, resolved.residence);
        let views: Vec<_> = storages
            .into_iter()
            .map(|storage| crate::physical::PhysicalStorageView {
                storage,
                access: resolved.access,
                transform: resolved.transform.clone(),
                ty: resolved.ty.clone(),
            })
            .collect();
        // A residence declares at least one plane (D1).
        NonEmpty::new(views).unwrap_or_else(|| {
            unreachable!("a tensor residence declares at least one plane (D1)")
        })
    }

    fn binder_slot(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        binder: CanonicalValueId,
    ) -> crate::ids::ScalarSlotIx {
        let leaf = self.leaf_of(binder, 0);
        match self.resolve_leaf(occurrence, strategy, subst, leaf) {
            ResolvedLeaf::Scalar(ResolvedScalar::Executor { occurrence, slot }) => {
                self.global_slot(occurrence, slot)
            }
            // A loop binder is produced per visit, never a boundary value,
            // so its leaf routes to an executor slot.
            _ => unreachable!("a retained repeat's binder routes to an executor slot (D1)"),
        }
    }

    // -- expressions -----------------------------------------------------------

    fn constant_node(&mut self, value: u64) -> InvocationValueId {
        match self.builder.constant(value) {
            Ok(id) => id,
            // A plan's derived table fits the u32 id domain.
            Err(_) => unreachable!("the derived table fits the u32 id domain"),
        }
    }

    fn expr(&mut self, result: Result<InvocationValueId, ExprDefect>) -> InvocationValueId {
        match result {
            Ok(id) => id,
            // The solver's model construction materialized the identical
            // expressions over the identical domains and would have rejected
            // an unprovable one at plan time.
            Err(defect_expr) => unreachable!(
                "an invocation expression was proven at model construction: {defect_expr}"
            ),
        }
    }

    fn add_node(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> InvocationValueId {
        let sum = self.builder.add(left, right);
        self.expr(sum)
    }

    fn mul_node(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> InvocationValueId {
        let product = self.builder.mul(left, right);
        self.expr(product)
    }

    fn div_node(
        &mut self,
        left: InvocationValueId,
        right: InvocationValueId,
    ) -> InvocationValueId {
        let quotient = self.builder.div(left, right);
        self.expr(quotient)
    }

    /// The execution expression of one scalar canonical value (a repeat
    /// bound endpoint or similar single-leaf control scalar).
    fn value_execution_expr(
        &mut self,
        occurrence: OccurrenceId,
        strategy: StrategyId,
        subst: &Substitution,
        value: CanonicalValueId,
    ) -> ExecutionExpr {
        let leaf = self.leaf_of(value, 0);
        match self.resolve_leaf(occurrence, strategy, subst, leaf) {
            ResolvedLeaf::Scalar(ResolvedScalar::Abi { slot }) => {
                ExecutionExpr::Invocation(self.abi_expr(slot))
            }
            ResolvedLeaf::Scalar(ResolvedScalar::Executor { occurrence, slot }) => {
                let bound = self.value_slot_bound(value);
                ExecutionExpr::Guarded(GuardedExecutionExpr::Executor {
                    slot: self.global_slot(occurrence, slot),
                    bound,
                })
            }
            ResolvedLeaf::Scalar(ResolvedScalar::Result { field }) => {
                ExecutionExpr::ResultField(field)
            }
            // A control scalar's value is scalar-kind, so its leaf routes
            // scalar.
            _ => unreachable!("a control scalar's leaf routes scalar"),
        }
    }

    /// The proved domain of one scalar value's executor slot.
    fn value_slot_bound(&self, value: CanonicalValueId) -> Bound {
        use seismic_lang::logical::value::GraphValueKind;
        match self.facts.value_kind(value) {
            GraphValueKind::Index { bound } | GraphValueKind::Range { bound } => {
                Bound {
                    min: 0,
                    max: self.extent_capacity(bound),
                }
            }
            GraphValueKind::Scalar(DType::I32 | DType::U32) => {
                Bound { min: 0, max: u64::from(u32::MAX) }
            }
            // Executor arithmetic reads integer-domain values only.
            _ => unreachable!("executor arithmetic reads integer-domain values"),
        }
    }

    /// The checked capacity that bounds one extent (proofs and resources).
    /// Every `@runtime<N>` atom of the program is bound by `validate`.
    fn extent_capacity(&self, extent: &ExtentExpr) -> u64 {
        match extent {
            ExtentExpr::Static(n) => *n,
            ExtentExpr::Runtime(id) => self.values.symbols[&format!("@runtime{}", id.0)],
            // W1 leaves no symbolic extents after specialization.
            ExtentExpr::Sym(_) => unreachable!("no symbolic extent survives specialization (W1)"),
        }
    }

    /// The invocation-derived node of one root ABI scalar slot. Index and
    /// range-endpoint nodes were built with their contracts; plain integer
    /// scalars are built here with their representation range.
    fn abi_expr(&mut self, slot: crate::ids::ScalarSlot) -> InvocationValueId {
        if let Some(existing) = self.abi_expr_of.get(&slot) {
            return *existing;
        }
        // build_abi allocated one contract per slot, in slot order.
        let contract = self.scalars[slot.index()].clone();
        match &contract.domain {
            ScalarDomain::Index { value, .. } => *value,
            // Both range-endpoint nodes were built with their contracts.
            ScalarDomain::RangeEndpoint { .. } => unreachable!(
                "a range endpoint scalar's invocation node is built with its contract"
            ),
            ScalarDomain::Any => {
                let repr = match AbiIntegerType::of(contract.dtype) {
                    Ok(repr) => repr,
                    // Execution arithmetic reads only integer ABI scalars;
                    // every other scalar's node is never requested.
                    Err(_) => unreachable!(
                        "execution arithmetic reads only integer ABI scalars"
                    ),
                };
                let built = self
                    .builder
                    .abi_scalar(slot, repr, None, repr.representable());
                let node = self.expr(built);
                self.abi_expr_of.insert(slot, node);
                node
            }
        }
    }

    /// The execution expression of one extent: static exact, runtime as the
    /// extent's actual-value expression.
    fn extent_execution_expr(&mut self, extent: &ExtentExpr) -> ExecutionExpr {
        self.extent_term(extent).into_execution()
    }

    fn extent_term(&mut self, extent: &ExtentExpr) -> Term {
        match extent {
            ExtentExpr::Static(n) => Term::Inv(self.constant_node(*n)),
            ExtentExpr::Runtime(id) => self.runtime_extent_term(*id),
            // W1 leaves no symbolic extents after specialization.
            ExtentExpr::Sym(_) => unreachable!("no symbolic extent survives specialization (W1)"),
        }
    }

    fn runtime_extent_term(&mut self, id: RuntimeExtentId) -> Term {
        if let Some(term) = self.extent_terms.get(&id) {
            if let Term::Inv(reserved) = term {
                if reserved.index() == usize::MAX {
                    // Runtime extents compose only through earlier extents.
                    unreachable!("runtime extents compose acyclically (L1)");
                }
            }
            return term.clone();
        }
        let value = self.logical.runtime_extent(id).value.clone();
        // Reserve the cache slot first: extents compose only through
        // earlier extents, so the reservation breaks cycles.
        self.extent_terms.insert(
            id,
            Term::Inv(InvocationValueId::from_index(usize::MAX)),
        );
        let term = self.runtime_term(&value);
        self.extent_terms.insert(id, term.clone());
        term
    }

    fn linear_total_term(&mut self, total: &crate::dispatch::LinearTotal) -> Term {
        match total {
            crate::dispatch::LinearTotal::Static(n) => Term::Inv(self.constant_node(*n)),
            crate::dispatch::LinearTotal::Runtime { product, .. } => {
                self.runtime_term(product)
            }
        }
    }

    /// One retained runtime scalar expression as a term: invocation-derived
    /// where every graph value it reads is invocation-known, guarded
    /// executor arithmetic where an executor slot feeds it. Resolved in the
    /// current occurrence's context.
    fn runtime_term(&mut self, expr: &RuntimeScalarExpr) -> Term {
        match expr {
            RuntimeScalarExpr::Const(value) => {
                // A runtime extent constant is a nonnegative length.
                Term::Inv(self.constant_node(u64::try_from(*value).unwrap_or_else(|_| {
                    unreachable!("a runtime scalar constant is nonnegative (L1)")
                })))
            }
            RuntimeScalarExpr::ShapeField(field) => {
                let domain = self.logical.shape_field(*field).domain;
                let node = match self.builder.shape_field(*field, domain) {
                    Ok(node) => node,
                    // A plan's derived table fits the u32 id domain.
                    Err(_) => unreachable!("the derived table fits the u32 id domain"),
                };
                Term::Inv(node)
            }
            RuntimeScalarExpr::Extent(id) => self.runtime_extent_term(*id),
            RuntimeScalarExpr::Value(value) => {
                let leaf = self.producer_leaf(*value);
                let occurrence = self.current;
                let strategy = self.current_strategy;
                let subst = self.current_subst.clone();
                let bound = self.value_slot_bound_of_leaf(leaf);
                match self.resolve_leaf(occurrence, strategy, &subst, leaf) {
                    ResolvedLeaf::Scalar(ResolvedScalar::Abi { slot }) => {
                        Term::Inv(self.abi_expr(slot))
                    }
                    ResolvedLeaf::Scalar(ResolvedScalar::Executor { occurrence, slot }) => {
                        Term::Guarded(GuardedExecutionExpr::Executor {
                            slot: self.global_slot(occurrence, slot),
                            bound,
                        })
                    }
                    ResolvedLeaf::Scalar(ResolvedScalar::Result { field }) => {
                        Term::Result { field, bound }
                    }
                    // A runtime scalar expression reads scalar values.
                    _ => unreachable!("a runtime scalar expression reads scalar values"),
                }
            }
            RuntimeScalarExpr::Add(left, right) => {
                let (left, right) = (self.runtime_term(left), self.runtime_term(right));
                self.combine(left, right, |builder, l, r| {
                    builder.add(l, r).map(Term::Inv)
                })
            }
            RuntimeScalarExpr::Mul(left, right) => {
                let (left, right) = (self.runtime_term(left), self.runtime_term(right));
                self.combine(left, right, |builder, l, r| {
                    builder.mul(l, r).map(Term::Inv)
                })
            }
            RuntimeScalarExpr::Sub(left, right) => {
                let (left, right) = (self.runtime_term(left), self.runtime_term(right));
                self.partial(
                    PartialKind::Sub,
                    |builder, l, r| builder.sub(l, r).map(Term::Inv),
                    left,
                    right,
                )
            }
            RuntimeScalarExpr::Div(left, right) => {
                let (left, right) = (self.runtime_term(left), self.runtime_term(right));
                self.partial(
                    PartialKind::Div,
                    |builder, l, r| builder.div(l, r).map(Term::Inv),
                    left,
                    right,
                )
            }
            RuntimeScalarExpr::Rem(left, right) => {
                let (left, right) = (self.runtime_term(left), self.runtime_term(right));
                self.partial(
                    PartialKind::Rem,
                    |builder, l, r| builder.rem(l, r).map(Term::Inv),
                    left,
                    right,
                )
            }
        }
    }

    /// A total operation: builder arithmetic when both operands are
    /// invocation-derived, guarded arithmetic (retaining the proved bound)
    /// otherwise.
    fn combine(
        &mut self,
        left: Term,
        right: Term,
        invocation: impl FnOnce(
            &mut ExprBuilder,
            InvocationValueId,
            InvocationValueId,
        ) -> Result<Term, ExprDefect>,
    ) -> Term {
        match (left, right) {
            (Term::Inv(l), Term::Inv(r)) => invocation(&mut self.builder, l, r).unwrap_or_else(
                |defect_expr| {
                    unreachable!(
                        "an invocation expression was proven at model construction: {defect_expr}"
                    )
                },
            ),
            (left, right) => {
                let (left_bound, right_bound) = (self.term_bound(&left), self.term_bound(&right));
                // Every operand bound was proven to fit the solver's i64
                // domains at model construction, so their sum fits u64.
                let bound = add_bounds(left_bound, right_bound).unwrap_or_else(|| {
                    unreachable!("a guarded sum of i64-bounded operands fits u64")
                });
                let guard = self.last_guard;
                let operation =
                    match (left.into_guarded(guard), right.into_guarded(guard)) {
                        (l, r) => GuardedExecutionExpr::Add(Box::new(l), Box::new(r), bound),
                    };
                Term::Guarded(operation)
            }
        }
    }

    /// A partial operation: builder arithmetic (with its interval or
    /// relation proof) when both operands are invocation-derived, guarded
    /// arithmetic under the dominating sealed guard otherwise.
    fn partial(
        &mut self,
        kind: PartialKind,
        invocation: impl FnOnce(
            &mut ExprBuilder,
            InvocationValueId,
            InvocationValueId,
        ) -> Result<Term, ExprDefect>,
        left: Term,
        right: Term,
    ) -> Term {
        let guard = self.last_guard;
        match (left, right) {
            (Term::Inv(l), Term::Inv(r)) => invocation(&mut self.builder, l, r).unwrap_or_else(
                |defect_expr| {
                    unreachable!(
                        "an invocation expression was proven at model construction: {defect_expr}"
                    )
                },
            ),
            (left, right) => {
                // S1 seals a dominating guard before every use of partial
                // executor arithmetic.
                let guard = match guard {
                    Some(guard) => guard,
                    None => unreachable!(
                        "partial executor arithmetic is dominated by a sealed guard (S1)"
                    ),
                };
                Term::Guarded(kind.build(
                    Box::new(left.into_guarded(Some(guard))),
                    Box::new(right.into_guarded(Some(guard))),
                    guard,
                ))
            }
        }
    }

    fn term_bound(&mut self, term: &Term) -> Bound {
        match term {
            Term::Inv(id) => self.builder.bound(*id),
            Term::Guarded(expr) => {
                let mut bounds = BTreeMap::new();
                guarded_bound(expr, &mut |id| self.builder.bound(id), &mut bounds)
            }
            Term::Result { bound, .. } => *bound,
        }
    }

    fn value_slot_bound_of_leaf(&self, leaf: CanonicalLeafId) -> Bound {
        let record = self.facts.leaf(leaf);
        self.value_slot_bound(record.value)
    }

    /// The canonical leaf of a scalar graph value read by a runtime scalar
    /// expression.
    fn producer_leaf(&self, value: GraphValueId) -> CanonicalLeafId {
        // value_graph covers every value of every owned graph; a scalar
        // value carries at least one leaf.
        let graph = self.value_graph[&value];
        let owned = OwnedValueRef { graph, value };
        let canonical = match self.facts.canonical_value(owned) {
            Ok(canonical) => canonical,
            // value_graph covers every value of every owned graph, all of
            // which expansion canonicalized.
            Err(_) => unreachable!("an owned graph value has a canonical identity (O1)"),
        };
        self.facts.leaves(canonical)[0]
    }

    // -- the root ABI -------------------------------------------------------

    fn entry_boundary(&self, strategy: StrategyId) -> crate::occurrence::InstantiatedBoundary {
        // The entry's boundary schema is shared by every alternative; the
        // selected alternative's instantiation names the same leaves.
        let alternative = self
            .strategy(self.entry, strategy)
            .routed()
            .shape()
            .root()
            .logical_alternative;
        self.facts
            .alternative(OwnedGraphKey {
                occurrence: self.entry,
                logical_alternative: alternative,
            })
            .boundary
            .clone()
    }

    fn entry_interface(&self) -> logical::FunctionInterface {
        self.facts.occurrence(self.entry).interface.clone()
    }

    fn build_abi(&mut self, strategy: StrategyId) {
        let boundary = self.entry_boundary(strategy);
        let interface = self.entry_interface();
        // Inputs: scalar leaves become ABI scalar contracts; tensor leaves
        // become one buffer per plane of their route's residence.
        for (leaf_key, input) in &boundary.inputs {
            // O1 instantiates root inputs only for the entry boundary.
            let InstantiatedInput::Root { callee, ownership, .. } = input else {
                unreachable!("the entry instantiates root inputs (O1)")
            };
            let value = match self.facts.canonical_value(*callee) {
                Ok(value) => value,
                // The entry boundary names minted values.
                Err(_) => unreachable!("the entry boundary names minted values (O1)"),
            };
            for leaf in self.facts.leaves(value).iter().copied() {
                let record = self.facts.leaf(leaf);
                match self.leaf_type(leaf) {
                    ValueType::Scalar(_)
                    | ValueType::Index { .. }
                    | ValueType::Range { .. } => {
                        let param = match leaf_key {
                            BoundaryLeaf::Input { param, .. } => *param,
                            // An input-boundary key is an input leaf.
                            BoundaryLeaf::Result { .. } => {
                                unreachable!("an input-boundary key is an input leaf (O1)")
                            }
                        };
                        let name = interface.params[param as usize].name.clone();
                        self.scalar_contract(leaf, leaf_key, &name, record.endpoint);
                    }
                    ValueType::Tensor(_) => {
                        let role = crate::invocation::BufferRole::Parameter {
                            ordinal: match leaf_key {
                                BoundaryLeaf::Input { param, .. } => *param,
                                BoundaryLeaf::Result { .. } => 0,
                            },
                            ownership: *ownership,
                        };
                        self.tensor_buffer(leaf, &record.path, role);
                    }
                    ValueType::Tuple(_) | ValueType::Void => {}
                    // The checked language keeps capability values off the
                    // public ABI.
                    ValueType::CapabilityValue(_) => unreachable!(
                        "a capability value cannot cross the public ABI (checked language)"
                    ),
                }
            }
        }
        // Results: scalar leaves become result fields; tensor leaves become
        // result buffers unless their route names a root input residence
        // (pass-through aliasing).
        for (_, result) in &boundary.results {
            // O1 instantiates root results only for the entry boundary.
            let InstantiatedResult::Root { callee } = result else {
                unreachable!("the entry instantiates root results (O1)")
            };
            let value = match self.facts.canonical_value(*callee) {
                Ok(value) => value,
                // The entry boundary names minted values.
                Err(_) => unreachable!("the entry boundary names minted values (O1)"),
            };
            for leaf in self.facts.leaves(value).iter().copied() {
                let record = self.facts.leaf(leaf);
                match self.leaf_type(leaf) {
                    ValueType::Scalar(_)
                    | ValueType::Index { .. }
                    | ValueType::Range { .. } => {
                        let dtype = match self.leaf_type(leaf) {
                            ValueType::Scalar(dtype) => dtype,
                            _ => DType::I32,
                        };
                        let index =
                            crate::ids::ResultFieldIx::from_index(self.result_fields.len());
                        self.result_field_of
                            .insert((record.path.clone(), record.endpoint), index);
                        self.result_fields.push(ResultField {
                            index,
                            path: record.path.clone(),
                            endpoint: record.endpoint,
                            dtype,
                        });
                    }
                    ValueType::Tensor(_) => {
                        // A pass-through result names the input's residence:
                        // no result buffer, the parameter buffer aliases it.
                        let aliased = match self.resolve_leaf(
                            self.entry,
                            strategy,
                            &Substitution::default(),
                            leaf,
                        ) {
                            ResolvedLeaf::Tensor(resolved) => {
                                resolved.occurrence == self.entry
                                    && matches!(
                                        self.strategy(resolved.occurrence, resolved.strategy)
                                            .routed()
                                            .residences()
                                            .residence(resolved.residence)
                                            .source,
                                        ResidenceSource::RootInput(_)
                                    )
                            }
                            // A tensor leaf routes tensor.
                            _ => unreachable!("a tensor leaf routes tensor (D1)"),
                        };
                        if !aliased {
                            self.tensor_buffer(
                                leaf,
                                &record.path,
                                crate::invocation::BufferRole::Result,
                            );
                        }
                    }
                    ValueType::Tuple(_) | ValueType::Void => {}
                    // The checked language keeps capability values off the
                    // public ABI.
                    ValueType::CapabilityValue(_) => unreachable!(
                        "a capability value cannot cross the public ABI (checked language)"
                    ),
                }
            }
        }
        // Range relations: one per range parameter, once both endpoints
        // exist (a range leaf always carries both endpoints).
        for (path, (bound, start, end)) in self.pending_ranges.iter() {
            match (start, end) {
                (Some(start), Some(end)) => {
                    let relation = self.builder.range_ordered(*start, *end, *bound);
                    if relation.is_err() {
                        unreachable!("a range parameter is orderable under the envelope");
                    }
                }
                _ => unreachable!(
                    "the range parameter at {path:?} has both endpoint contracts (O1)"
                ),
            }
        }
        // Alias rules: shared-read parameter ranges may overlap; every other
        // pair of live ranges is disjoint.
        let ownership_of: BTreeMap<crate::ids::BufferSlot, ParamOwnership> = self
            .buffers
            .iter()
            .filter_map(|contract| match contract.role {
                crate::invocation::BufferRole::Parameter { ownership, .. } => {
                    Some((contract.slot, ownership))
                }
                _ => None,
            })
            .collect();
        for index in 0..self.buffers.len() {
            for other in index + 1..self.buffers.len() {
                let (left, right) = (self.buffers[index].slot, self.buffers[other].slot);
                let both_shared = matches!(
                    (ownership_of.get(&left), ownership_of.get(&right)),
                    (Some(ParamOwnership::Shared), Some(ParamOwnership::Shared))
                );
                self.aliases.push(crate::invocation::AliasContract {
                    left,
                    right,
                    rule: if both_shared {
                        crate::invocation::AliasRule::MayOverlap
                    } else {
                        crate::invocation::AliasRule::MustDisjoint
                    },
                });
            }
        }
    }

    fn scalar_contract(
        &mut self,
        leaf: CanonicalLeafId,
        leaf_key: &BoundaryLeaf,
        param_name: &str,
        endpoint: Option<RangeEndpoint>,
    ) {
        let record = self.facts.leaf(leaf);
        let name = abi_scalar_name(param_name, &record.path, endpoint);
        match self.leaf_type(leaf) {
            ValueType::Scalar(dtype) => {
                let slot = crate::ids::ScalarSlot::from_index(self.scalars.len());
                self.scalar_slot_of.insert((leaf_key.clone(), None), slot);
                self.scalars.push(crate::invocation::ScalarContract {
                    slot,
                    name,
                    dtype,
                    domain: ScalarDomain::Any,
                });
            }
            ValueType::Index { bound } => {
                let slot = crate::ids::ScalarSlot::from_index(self.scalars.len());
                let bound_id = match self.extent_term(&bound) {
                    Term::Inv(id) => id,
                    // An interface index bound is a shape expression, never
                    // a value-derived extent.
                    _ => unreachable!("an interface index bound is invocation-known"),
                };
                let bound_max = self.builder.bound(bound_id).max;
                let built = self.builder.abi_scalar(
                    slot,
                    AbiIntegerType::I32,
                    None,
                    Bound { min: 0, max: bound_max },
                );
                let value = self.expr(built);
                self.scalar_slot_of.insert((leaf_key.clone(), None), slot);
                self.scalars.push(crate::invocation::ScalarContract {
                    slot,
                    name,
                    dtype: DType::I32,
                    domain: ScalarDomain::Index { value, bound: bound_id },
                });
                self.abi_expr_of.insert(slot, value);
            }
            ValueType::Range { bound } => {
                // A range leaf always carries its endpoint.
                let endpoint = match endpoint {
                    Some(endpoint) => endpoint,
                    None => unreachable!("a range leaf carries an endpoint (O1)"),
                };
                let bound_id = match self.extent_term(&bound) {
                    Term::Inv(id) => id,
                    // An interface range bound is a shape expression, never
                    // a value-derived extent.
                    _ => unreachable!("an interface range bound is invocation-known"),
                };
                let bound_max = self.builder.bound(bound_id).max;
                let slot = crate::ids::ScalarSlot::from_index(self.scalars.len());
                let built = self.builder.abi_scalar(
                    slot,
                    AbiIntegerType::I32,
                    Some(endpoint),
                    Bound { min: 0, max: bound_max },
                );
                let node = self.expr(built);
                self.scalar_slot_of
                    .insert((leaf_key.clone(), Some(endpoint)), slot);
                self.scalars.push(crate::invocation::ScalarContract {
                    slot,
                    name,
                    dtype: DType::I32,
                    domain: ScalarDomain::RangeEndpoint { endpoint },
                });
                self.abi_expr_of.insert(slot, node);
                let entry = self
                    .pending_ranges
                    .entry(record.path.clone())
                    .or_insert((bound_id, None, None));
                match endpoint {
                    RangeEndpoint::Start => entry.1 = Some(node),
                    RangeEndpoint::End => entry.2 = Some(node),
                }
            }
            // build_abi calls this only for scalar-kind leaves.
            _ => unreachable!("an ABI scalar contract holds a scalar-kind leaf"),
        }
    }

    fn tensor_buffer(
        &mut self,
        leaf: CanonicalLeafId,
        path: &ValuePath,
        role: crate::invocation::BufferRole,
    ) {
        let entry_strategy = self.current_strategy;
        let ResolvedLeaf::Tensor(resolved) = self.resolve_leaf(
            self.entry,
            entry_strategy,
            &Substitution::default(),
            leaf,
        )
        else {
            // A tensor leaf routes tensor.
            unreachable!("a tensor boundary leaf routes tensor (D1)")
        };
        let record: Residence = self
            .strategy(resolved.occurrence, resolved.strategy)
            .routed()
            .residences()
            .residence(resolved.residence)
            .clone();
        for (plane, plan) in record.planes.iter().enumerate() {
            let plane = plane as u32;
            let slot = crate::ids::BufferSlot::from_index(self.buffers.len());
            self.buffer_of
                .insert((resolved.occurrence, resolved.residence, plane), slot);
            let plane_name = match &plan.plane {
                crate::residence::StoragePlane::Dense => "dense".to_string(),
                crate::residence::StoragePlane::Representation { name, .. } => name.clone(),
            };
            let dtype = plane_descriptor(&record.shape, &plan.plane).storage_dtype();
            // The byte requirement is an expression of the actual extents of
            // the residence's shape (the invocation contract owns this
            // formula; capacity resources remain M1's).
            let bytes = self.plane_bytes_expr(&record.shape, plan);
            self.buffers.push(crate::invocation::BufferContract {
                slot,
                path: path.clone(),
                plane: plane_name,
                role: role.clone(),
                dtype,
                bytes,
                alignment: plan.alignment.max(1),
            });
        }
    }

    /// One plane's byte requirement as an expression of actual validated
    /// extents (D1's plane-size formula over actual extent values).
    fn plane_bytes_expr(
        &mut self,
        shape: &TensorType,
        plan: &crate::residence::PlanePlan,
    ) -> InvocationValueId {
        let extent = |resolver: &mut Self, extent: &ExtentExpr| -> InvocationValueId {
            match resolver.extent_term(extent) {
                Term::Inv(id) => id,
                // An interface tensor shape is a shape expression, never a
                // value-derived extent.
                _ => unreachable!("an interface tensor shape is invocation-known"),
            }
        };
        let mut elements = self.constant_node(1);
        for axis in &shape.axes {
            let factor = extent(self, axis);
            elements = self.mul_node(elements, factor);
        }
        match &shape.elem {
            Elem::Dtype(dtype) => {
                let width = self.constant_node(u64::from(dtype.bytes()));
                self.mul_node(elements, width)
            }
            // W1 leaves no element parameters after specialization.
            Elem::Param(_) => unreachable!("no element parameter survives specialization (W1)"),
            Elem::Repr(name) => {
                // The checker admits only registered representations, which
                // resolve a packing axis within the rank.
                let Some(representation) = repr::lookup(name) else {
                    unreachable!("a checked representation name is registered")
                };
                let Some(packed_axis) = shape.packed_axis else {
                    unreachable!("a packed tensor carries its packed axis")
                };
                let Some(columns) = shape.axes.get(packed_axis) else {
                    unreachable!("a packed axis is within the rank")
                };
                let group = representation.storage_group();
                let columns = extent(self, columns);
                let group_node = self.constant_node(u64::from(group));
                // Physical width: the packed axis rounded up to a whole
                // storage group.
                let group_minus_1 = self.constant_node(u64::from(group - 1));
                let padded = self.add_node(columns, group_minus_1);
                let width = self.div_node(padded, group_node);
                let width = self.mul_node(width, group_node);
                let mut rows = self.constant_node(1);
                for (axis, axis_extent) in shape.axes.iter().enumerate() {
                    if axis != packed_axis {
                        let factor = extent(self, axis_extent);
                        rows = self.mul_node(rows, factor);
                    }
                }
                let repr_plane = match plane_descriptor(shape, &plan.plane) {
                    PlaneRef::Repr { plane } => plane,
                    // A representation tensor's planes are representation
                    // planes.
                    PlaneRef::Dense { .. } => unreachable!(
                        "a representation tensor's planes are representation planes (D1/K1)"
                    ),
                };
                let group_of_plane = self.constant_node(u64::from(repr_plane.group));
                let fields = self.constant_node(u64::from(repr_plane.fields));
                let row_entries = self.div_node(width, group_of_plane);
                let row_entries = self.mul_node(row_entries, fields);
                match &repr_plane.encoding {
                    repr::PlaneEncoding::Dense(dtype) => {
                        let entry_bytes = self.constant_node(u64::from(dtype.bytes()));
                        let total = self.mul_node(row_entries, entry_bytes);
                        self.mul_node(rows, total)
                    }
                    repr::PlaneEncoding::Packed { bits, .. } => {
                        let bits = self.constant_node(u64::from(*bits));
                        let words = self.mul_node(row_entries, bits);
                        let thirty_one = self.constant_node(31);
                        let words = self.add_node(words, thirty_one);
                        let thirty_two = self.constant_node(32);
                        let words = self.div_node(words, thirty_two);
                        let four = self.constant_node(4);
                        let total = self.mul_node(words, four);
                        self.mul_node(rows, total)
                    }
                }
            }
        }
    }

    // -- the seal ---------------------------------------------------------------

    fn finish(mut self, schedule: PhysicalSchedule<D>) -> PhysicalPlan<D> {
        // Complete the runtime extent table for extents the sealed plan
        // reads: each one's actual-value term is built in whichever inlined
        // context routes its leaves (all such contexts resolve the same
        // leaves to the same routes). An extent whose values no selected
        // strategy routes is read by nothing in the sealed plan and gets no
        // entry.
        for extent in self.logical.runtime_extents() {
            if self.extent_terms.contains_key(&extent.id) {
                continue;
            }
            let Some(context) = self.context_for_extent(&extent.value) else {
                continue;
            };
            let saved = (
                self.current,
                self.current_strategy,
                self.current_subst.clone(),
            );
            let (context_strategy, context_subst) = &self.contexts[&context];
            self.current = context;
            self.current_strategy = *context_strategy;
            self.current_subst = context_subst.clone();
            self.runtime_extent_term(extent.id);
            self.current = saved.0;
            self.current_strategy = saved.1;
            self.current_subst = saved.2;
        }
        let runtime_extents: BTreeMap<RuntimeExtentId, ExecutionExpr> = self
            .extent_terms
            .iter()
            .map(|(id, term)| (*id, term.clone().into_execution()))
            .collect();

        let mut arena_bytes = 0u64;
        for storage in &self.storages {
            if let StoragePlacement::Arena { offset } = storage.placement {
                // The packing certificate proved every extent within the
                // capacity.
                let end = offset
                    .checked_add(storage.bytes)
                    .unwrap_or_else(|| unreachable!("an arena extent is within the capacity"));
                arena_bytes = arena_bytes.max(end);
            }
        }
        let mut max_workgroup_bytes = 0u64;
        let mut max_private_bytes = 0u64;
        let mut direct_bindings = 0u64;
        fn walk<'p, D: ExecutableDialect>(
            steps: &'p [PhysicalStep<D>],
            visit: &mut dyn FnMut(&'p SealedLaunch<D>),
        ) {
            for step in steps {
                match step {
                    PhysicalStep::Launch(launch) => visit(launch),
                    PhysicalStep::Guard(_) | PhysicalStep::Fill(_) => {}
                    PhysicalStep::Call(call) => walk(&call.body.steps, visit),
                    PhysicalStep::If(branch) => {
                        walk(&branch.then_schedule.steps, visit);
                        walk(&branch.else_schedule.steps, visit);
                    }
                    PhysicalStep::Repeat(repeat) => walk(&repeat.body.steps, visit),
                }
            }
        }
        let mut launches = Vec::new();
        walk(&schedule.steps, &mut |launch| launches.push(launch));
        for launch in launches {
            max_workgroup_bytes = max_workgroup_bytes.max(launch.resources.workgroup_bytes);
            max_private_bytes = max_private_bytes
                .max(launch.resources.private_bytes_per_participant);
            direct_bindings += u64::from(launch.resources.direct_bindings);
        }
        // The dense id domains are u32, so the field counts and their word
        // totals fit u64.
        let result_bytes = (self.result_fields.len() as u64)
            .checked_mul(8)
            .unwrap_or_else(|| unreachable!("u32-bounded field counts fit the byte domain"));
        let status_bytes = (self.status_fields.len() as u64)
            .checked_mul(4)
            .unwrap_or_else(|| unreachable!("u32-bounded field counts fit the byte domain"));
        let resources = PlanResources {
            arena_bytes,
            max_workgroup_bytes,
            max_private_bytes_per_participant: max_private_bytes,
            direct_bindings: u32::try_from(direct_bindings).unwrap_or_else(|_| {
                unreachable!("the per-launch binding limits bound the total by the plan size")
            }),
            scalar_slots: u32::try_from(self.scalar_slots.len()).unwrap_or_else(|_| {
                unreachable!("u32-bounded slot ids bound the slot total")
            }),
            result_bytes,
            status_bytes,
        };

        let mut shape_fields = Vec::with_capacity(self.logical.shape_fields.len());
        for (_, field) in self.logical.shape_fields.entries() {
            shape_fields.push(ShapeFieldContract {
                name: field.name.clone(),
                domain: field.domain,
            });
        }
        let (derived, relations) = self.builder.finish();
        let contract = InvocationContract::new(
            crate::ids::DenseMap::from_vec(shape_fields),
            self.buffers,
            self.scalars,
            self.aliases,
            relations,
            derived,
        );
        let selections = self
            .assignment
            .selections()
            .iter()
            .map(|(occurrence, strategy)| (*occurrence, *strategy))
            .collect();
        let identity = ResolutionIdentity {
            logical: self.logical.identity,
            entry: self.facts.occurrence(self.entry).interface.name.clone(),
            target: self.logical.target.clone(),
            toolchain_fingerprint: self.profile.toolchain_fingerprint.clone(),
            selections,
        };
        PhysicalPlan::seal(
            identity,
            contract,
            crate::ids::DenseMap::from_vec(self.storages),
            schedule,
            crate::ids::DenseMap::from_vec(self.result_fields),
            crate::ids::DenseMap::from_vec(self.status_fields),
            runtime_extents,
            resources,
            self.assignment.numerical().clone(),
            self.assignment.estimated_cost(),
            self.assignment.optimal(),
        )
    }

    /// The inlined occurrence whose spine strategy routes the leaves a
    /// runtime extent's value expression reads, or `None` when no inlined
    /// strategy routes them (the extent is read by nothing in the sealed
    /// plan).
    fn context_for_extent(&self, value: &RuntimeScalarExpr) -> Option<OccurrenceId> {
        let mut leaves = Vec::new();
        collect_value_refs(value, &mut leaves);
        for value in leaves {
            let leaf = self.producer_leaf(value);
            for &(occurrence, strategy) in &self.spine {
                if self.strategy(occurrence, strategy)
                    .routed()
                    .routes()
                    .contains(leaf)
                {
                    return Some(occurrence);
                }
            }
        }
        None
    }
}

fn collect_value_refs(expr: &RuntimeScalarExpr, out: &mut Vec<GraphValueId>) {
    match expr {
        RuntimeScalarExpr::Value(value) => out.push(*value),
        RuntimeScalarExpr::Add(left, right)
        | RuntimeScalarExpr::Sub(left, right)
        | RuntimeScalarExpr::Mul(left, right)
        | RuntimeScalarExpr::Div(left, right)
        | RuntimeScalarExpr::Rem(left, right) => {
            collect_value_refs(left, out);
            collect_value_refs(right, out);
        }
        RuntimeScalarExpr::Const(_)
        | RuntimeScalarExpr::ShapeField(_)
        | RuntimeScalarExpr::Extent(_) => {}
    }
}

// ---------------------------------------------------------------------------
// Terms (invocation-derived or guarded executor arithmetic)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Term {
    Inv(InvocationValueId),
    Guarded(GuardedExecutionExpr),
    /// A bare read of one result field: representable as an execution
    /// expression, but not as an operand of the guarded-arithmetic algebra.
    Result {
        field: crate::ids::ResultFieldIx,
        bound: Bound,
    },
}

impl Term {
    fn into_execution(self) -> ExecutionExpr {
        match self {
            Term::Inv(id) => ExecutionExpr::Invocation(id),
            Term::Guarded(expr) => ExecutionExpr::Guarded(expr),
            Term::Result { field, .. } => ExecutionExpr::ResultField(field),
        }
    }

    fn into_guarded(self, guard: Option<crate::ids::GuardIx>) -> GuardedExecutionExpr {
        match self {
            Term::Inv(id) => GuardedExecutionExpr::Invocation(id),
            Term::Guarded(expr) => expr,
            Term::Result { field, bound } => {
                // S1 seals a dominating guard before every use of a
                // result-field read.
                let guard = match guard {
                    Some(guard) => guard,
                    None => unreachable!(
                        "a result-field read is dominated by a sealed guard (S1)"
                    ),
                };
                GuardedExecutionExpr::ResultField { field, bound, guard }
            }
        }
    }
}

/// The guarded form of a partial operation, under the sealed guard that
/// structurally dominates every use of its result.
#[derive(Clone, Copy)]
enum PartialKind {
    Sub,
    Div,
    Rem,
}

impl PartialKind {
    fn build(
        self,
        left: Box<GuardedExecutionExpr>,
        right: Box<GuardedExecutionExpr>,
        guard: crate::ids::GuardIx,
    ) -> GuardedExecutionExpr {
        match self {
            PartialKind::Sub => GuardedExecutionExpr::Sub(left, right, guard),
            PartialKind::Div => GuardedExecutionExpr::Div(left, right, guard),
            PartialKind::Rem => GuardedExecutionExpr::Rem(left, right, guard),
        }
    }
}

fn guarded_bound(
    expr: &GuardedExecutionExpr,
    invocation_bound: &mut dyn FnMut(InvocationValueId) -> Bound,
    cache: &mut BTreeMap<InvocationValueId, Bound>,
) -> Bound {
    match expr {
        GuardedExecutionExpr::Invocation(id) => {
            if let Some(bound) = cache.get(id) {
                return *bound;
            }
            let bound = invocation_bound(*id);
            cache.insert(*id, bound);
            bound
        }
        GuardedExecutionExpr::Executor { bound, .. } => *bound,
        GuardedExecutionExpr::ResultField { bound, .. } => *bound,
        GuardedExecutionExpr::Add(_, _, bound) | GuardedExecutionExpr::Mul(_, _, bound) => *bound,
        GuardedExecutionExpr::Sub(left, right, _) => difference_bound(
            guarded_bound(left, invocation_bound, cache),
            guarded_bound(right, invocation_bound, cache),
        ),
        GuardedExecutionExpr::Div(left, right, _) => quotient_bound(
            guarded_bound(left, invocation_bound, cache),
            guarded_bound(right, invocation_bound, cache),
        ),
        GuardedExecutionExpr::Rem(left, right, _) => remainder_bound(
            guarded_bound(left, invocation_bound, cache),
            guarded_bound(right, invocation_bound, cache),
        ),
        GuardedExecutionExpr::CeilDiv(left, right, _) => ceil_quotient_bound(
            guarded_bound(left, invocation_bound, cache),
            guarded_bound(right, invocation_bound, cache),
        ),
        GuardedExecutionExpr::Min(left, right) => {
            let (left, right) = (
                guarded_bound(left, invocation_bound, cache),
                guarded_bound(right, invocation_bound, cache),
            );
            Bound {
                min: left.min.min(right.min),
                max: left.max.min(right.max),
            }
        }
    }
}

fn add_bounds(left: Bound, right: Bound) -> Option<Bound> {
    Some(Bound {
        min: left.min.checked_add(right.min)?,
        max: left.max.checked_add(right.max)?,
    })
}

fn difference_bound(left: Bound, right: Bound) -> Bound {
    Bound {
        min: if left.min >= right.max {
            left.min - right.max
        } else {
            0
        },
        max: if left.max >= right.min {
            left.max - right.min
        } else {
            0
        },
    }
}

fn divisor_interval(right: Bound) -> Option<Bound> {
    if right.max == 0 {
        None
    } else {
        Some(Bound {
            min: right.min.max(1),
            max: right.max,
        })
    }
}

fn quotient_bound(left: Bound, right: Bound) -> Bound {
    match divisor_interval(right) {
        Some(divisor) => Bound {
            min: left.min / divisor.max,
            max: left.max / divisor.min,
        },
        None => Bound { min: 0, max: 0 },
    }
}

fn remainder_bound(left: Bound, right: Bound) -> Bound {
    match divisor_interval(right) {
        Some(divisor) => Bound {
            min: 0,
            max: left.max.min(divisor.max - 1),
        },
        None => Bound { min: 0, max: 0 },
    }
}

fn ceil_quotient_bound(left: Bound, right: Bound) -> Bound {
    match divisor_interval(right) {
        Some(divisor) => Bound {
            min: left.min.div_ceil(divisor.max),
            max: left.max.div_ceil(divisor.min),
        },
        None => Bound { min: 0, max: 0 },
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn safety_kind(obligation: &SafetyObligation) -> crate::failure::SafetyKind {
    match obligation {
        SafetyObligation::ExtentPositive { .. } => crate::failure::SafetyKind::ExtentPositive,
        SafetyObligation::IndexInBounds { .. } => crate::failure::SafetyKind::IndexInBounds,
        SafetyObligation::RangeInBounds { .. } => crate::failure::SafetyKind::RangeInBounds,
        SafetyObligation::DivisorNonZero { .. } => crate::failure::SafetyKind::DivisorNonZero,
        SafetyObligation::SignedDivisionNoOverflow { .. } => {
            crate::failure::SafetyKind::SignedDivisionNoOverflow
        }
        SafetyObligation::ShiftInRange { .. } => crate::failure::SafetyKind::ShiftInRange,
        SafetyObligation::ShapeProductFits { .. } => crate::failure::SafetyKind::ShapeProductFits,
    }
}

fn input_access(access: Access) -> AccessMode {
    match access {
        Access::Shared => AccessMode::Read,
        Access::Exclusive => AccessMode::ReadWrite,
    }
}

fn output_access(access: Access) -> AccessMode {
    match access {
        Access::Shared => AccessMode::Write,
        Access::Exclusive => AccessMode::ReadWrite,
    }
}

/// The typed descriptor of one declared residence plane, built through the
/// single authority; the inadmissible pairs are D1/K1 construction defects
/// that the residence plane plans were sealed against at formation.
fn plane_descriptor(
    shape: &TensorType,
    plane: &crate::residence::StoragePlane,
) -> PlaneRef {
    match PlaneRef::of(shape, plane) {
        Ok(descriptor) => descriptor,
        Err(_) => unreachable!("a residence plane matches its tensor's element (D1/K1)"),
    }
}

fn abi_scalar_name(param: &str, path: &ValuePath, endpoint: Option<RangeEndpoint>) -> String {
    let base = if path.0.is_empty() {
        param.to_string()
    } else {
        format!("{param}{}", path)
    };
    match endpoint {
        None => base,
        Some(RangeEndpoint::Start) => format!("{base}.start"),
        Some(RangeEndpoint::End) => format!("{base}.end"),
    }
}

/// Qualify one strategy's tuning atoms by its `(occurrence, strategy)`
/// identity. The identical function qualifies the atoms on the solver-export
/// side (`seismic_compiler::planning`); the reserved `@`-prefixed atom
/// families are global and never qualified.
pub(crate) fn qualify(expr: &Sym, occurrence: OccurrenceId, strategy: StrategyId) -> Sym {
    fn qualify_atom(atom: &Atom, occurrence: OccurrenceId, strategy: StrategyId) -> Atom {
        match atom {
            Atom::Param(name) => {
                if name.starts_with('@') {
                    Atom::Param(name.clone())
                } else {
                    Atom::Param(plan_space::tuning_symbol(occurrence, strategy, name))
                }
            }
            Atom::Quot(numerator, denominator) => Atom::Quot(
                Box::new(qualify(numerator, occurrence, strategy)),
                Box::new(qualify(denominator, occurrence, strategy)),
            ),
            Atom::Rem(numerator, denominator) => Atom::Rem(
                Box::new(qualify(numerator, occurrence, strategy)),
                Box::new(qualify(denominator, occurrence, strategy)),
            ),
        }
    }
    let mut out = Sym::constant(0);
    for (monomial, coefficient) in expr.monomials() {
        let mut term = Sym::constant(coefficient);
        for (factor, degree) in monomial {
            let factor = Sym::atom(qualify_atom(factor, occurrence, strategy));
            for _ in 0..*degree {
                term = term.mul(&factor);
            }
        }
        out = out.add(&term);
    }
    out
}

