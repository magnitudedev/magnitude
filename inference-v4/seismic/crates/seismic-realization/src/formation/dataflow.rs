//! Route and residence derivation (package D1).
//!
//! One pass over the closed shape's cross edges, blocks, retained steps, and
//! root boundary: every non-void value is leafed through the canonical leaf
//! registry (O1); the total `RouteTable` is built; one residence is assigned
//! per canonical logical storage chain; one spill residence is introduced
//! per computed tensor crossing a cut; view routes are derived from the
//! source residence plus the complete composed transform; exact lifetimes
//! and legal placement domains are computed (root leaves ABI-only;
//! cross-step and state arena; workgroup/participant only for kernel-local
//! lifetimes, as solver choices); every `ClosedKernelInterface` is built.
//!
//! Aliasing is never recorded: a moved parameter a callee returns, a
//! pass-through `&mut` result, and two views of one storage are two routes
//! naming one `ResidenceId`, by construction of the canonical storage chain.
//!
//! Invariants sealed here (checked in `Former::seal`):
//!   domain(RouteTable) = every nonlocal semantic leaf required by the shape
//!   every TensorRoute.residence exists in ResidenceGraph
//!   every residence source has at least one route or a kernel-local owner
//!   kernel-local SSA leaves are not routed
//!   each logical storage state chain names one ResidenceId
//!   each computed cross-cut tensor names exactly one spill ResidenceId
//!   ABI residences only for root boundary leaves
//!   workgroup/participant options only with kernel-local lifetimes

use crate::dispatch::{AxisMapping, LinearIterationMap};
use crate::failure::{CompilerDefect, Package};
use crate::ids::{
    BlockId, CanonicalLeafId, CanonicalStorageId, CanonicalValueId, ChoiceVarId,
    ExecutorScalarSlotId, KernelAxisId, KernelInputId, KernelLocalId, KernelOutputId,
    OwnedGraphKey, OwnedNodeRef, OwnedStateRef, OwnedStorageRef, OwnedValueRef, OwnedViewRef,
    PlanParamId, ResidenceId, StepId,
};
use crate::occurrence::{InstantiatedFinalState, InstantiatedInput, InstantiatedResult, OccurrenceFacts};
use crate::residence::{
    ClosedKernelInterface, KernelAxisDecl, KernelInputDecl, KernelLocalDecl, KernelOutputDecl,
    Lifetime, PlanePlan, PullCounter, Replication, Residence, ResidenceChoice, ResidenceGraph,
    ResidenceSource, RoutedBlock, RoutedStrategy, ScheduleStep, StepAnchor, StoragePlane,
    StorageScope,
};
use crate::routes::{
    RouteTable, ScalarRoute, SliceAxisTemplate, TensorRoute, ValueRoute, ViewStepKind,
    ViewStepTemplate, ViewTransformTemplate,
};
use crate::strategy::{
    CarryEdge, ClosedStrategyShape, EdgeEnd, JoinEdge, LocalResidenceRequirement, ParticipantPolicy,
    ShapeSchedule, ShapeStep,
};
use crate::target::EffectiveTargetProfile;
use seismic_lang::logical::boundary::BoundaryLeaf;
use seismic_lang::logical::value::{GraphValueKind, LogicalStorageOwner, TensorSource, ViewBase};
use seismic_lang::logical::{
    Access, CallInput, IdVec, LogicalNodeKind, RegionParameter, SliceAxis, ViewTransform,
};
use seismic_lang::repr;
use seismic_lang::sir::ParamOwnership;
use seismic_lang::sym::Sym;
use seismic_lang::types::{DType, Elem, ExtentExpr, NonEmpty, TensorType, ValueType};
use std::collections::{BTreeMap, BTreeSet};

/// The sole body of `DataflowFormer::form`. `profile` is the target the
/// shape was formed against; routes and residences depend on no target
/// limit (limits become solver constraints in M1), so it is consumed only
/// to bind the former to the same profile object S1 used.
pub(crate) fn form(
    facts: &OccurrenceFacts<'_>,
    profile: &EffectiveTargetProfile,
    shape: ClosedStrategyShape,
) -> Result<RoutedStrategy, CompilerDefect> {
    let formed = Former::new(facts, profile, &shape)?.form()?;
    Ok(RoutedStrategy::seal(
        shape,
        formed.routes,
        formed.residences,
        formed.blocks,
        formed.steps,
        formed.scalar_slots,
    ))
}

/// The sealed parts of one routed strategy, before the shape is moved in.
struct Formed {
    routes: RouteTable,
    residences: ResidenceGraph,
    blocks: IdVec<BlockId, RoutedBlock>,
    steps: IdVec<StepId, ScheduleStep>,
    scalar_slots: u32,
}

// ---------------------------------------------------------------------------
// Anchors: where a tensor value's bytes live, before residences exist
// ---------------------------------------------------------------------------

/// The owner of a tensor value's bytes, reached by composing the value's
/// view with every caller argument view up to a storage this strategy
/// holds a residence for.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Anchor {
    /// A tensor input leaf of the strategy's root occurrence boundary.
    BoundaryInput(BoundaryLeaf),
    /// A canonical logical storage chain whose owning storage is `Local`.
    Storage(CanonicalStorageId),
    /// A computed tensor: the class itself is the owner.
    Computed(CanonicalValueId),
}

/// A tensor value resolved to its anchor with the complete transform from
/// the anchor's coordinates to the value's, and the value's access.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Resolved {
    anchor: Anchor,
    transform: ViewTransformTemplate,
    access: Access,
}

struct Former<'f, 'l> {
    facts: &'f OccurrenceFacts<'l>,
    profile: &'f EffectiveTargetProfile,
    shape: &'f ClosedStrategyShape,
    root_key: OwnedGraphKey,
    /// Dense pre-order schedule steps and the step of every non-root end.
    steps: Vec<ScheduleStep>,
    end_step: BTreeMap<EdgeEnd, StepId>,
    /// The end producing every canonical value class this strategy defines.
    producer: BTreeMap<CanonicalValueId, EdgeEnd>,
    /// Root occurrence boundary: input leaf per input value class (with its
    /// ownership) and result leaf per result value class.
    boundary_inputs: BTreeMap<CanonicalValueId, (BoundaryLeaf, ParamOwnership)>,
    boundary_results: BTreeMap<CanonicalValueId, BoundaryLeaf>,
    /// The ends touching every storage chain through state (reads, writes,
    /// retained call states, control carries/joins, root entry/final states).
    storage_ends: BTreeMap<CanonicalStorageId, BTreeSet<EdgeEnd>>,
    /// The ends of every cross edge, by leaf (producer and consumers).
    edge_ends: BTreeMap<CanonicalLeafId, BTreeSet<EdgeEnd>>,
    /// Every non-local leaf the shape requires a route for.
    required: BTreeSet<CanonicalLeafId>,
    /// The resolved anchor of every required tensor leaf.
    resolved: BTreeMap<CanonicalLeafId, Resolved>,
    /// Kernel-local residence requirements by block and value.
    local_requirements: BTreeMap<(BlockId, CanonicalValueId), LocalResidenceRequirement>,
    // Residence construction.
    residences: Vec<Residence>,
    choice_vars: u32,
    storage_residence: BTreeMap<CanonicalStorageId, ResidenceId>,
    spill_residence: BTreeMap<CanonicalValueId, ResidenceId>,
    boundary_input_residence: BTreeMap<BoundaryLeaf, ResidenceId>,
    /// The pull counter residence of every `DynamicPull` block.
    pull_counters: BTreeMap<BlockId, ResidenceId>,
    // Route construction.
    routes: BTreeMap<CanonicalLeafId, ValueRoute>,
    scalar_slots: u32,
}

impl<'f, 'l> Former<'f, 'l> {
    fn new(
        facts: &'f OccurrenceFacts<'l>,
        profile: &'f EffectiveTargetProfile,
        shape: &'f ClosedStrategyShape,
    ) -> Result<Former<'f, 'l>, CompilerDefect> {
        let root = shape.root();
        let root_key = OwnedGraphKey {
            occurrence: root.occurrence,
            logical_alternative: root.logical_alternative,
        };
        let mut former = Former {
            facts,
            profile,
            shape,
            root_key,
            steps: Vec::new(),
            end_step: BTreeMap::new(),
            producer: BTreeMap::new(),
            boundary_inputs: BTreeMap::new(),
            boundary_results: BTreeMap::new(),
            storage_ends: BTreeMap::new(),
            edge_ends: BTreeMap::new(),
            required: BTreeSet::new(),
            resolved: BTreeMap::new(),
            local_requirements: BTreeMap::new(),
            residences: Vec::new(),
            choice_vars: 0,
            storage_residence: BTreeMap::new(),
            spill_residence: BTreeMap::new(),
            boundary_input_residence: BTreeMap::new(),
            pull_counters: BTreeMap::new(),
            routes: BTreeMap::new(),
            scalar_slots: 0,
        };
        former.index_schedule()?;
        former.index_boundary()?;
        former.index_producers()?;
        former.index_storage_touches()?;
        former.index_required()?;
        former.index_local_requirements()?;
        Ok(former)
    }

    fn defect(&self, invariant: impl std::fmt::Display) -> CompilerDefect {
        CompilerDefect::new(
            Package::D1,
            format!(
                "rule `{}` on `{}`: {invariant}",
                self.shape.rule(),
                self.profile.backend
            ),
        )
    }

    fn canonical(
        &self,
        graph: OwnedGraphKey,
        value: seismic_lang::logical::GraphValueId,
    ) -> Result<CanonicalValueId, CompilerDefect> {
        self.facts.canonical_value(OwnedValueRef { graph, value })
    }

    fn storage_chain(
        &self,
        graph: OwnedGraphKey,
        state: seismic_lang::logical::StateTokenId,
    ) -> Result<CanonicalStorageId, CompilerDefect> {
        self.facts.canonical_storage(self.facts.state_storage(OwnedStateRef { graph, state }))
    }

    // -- schedule steps ---------------------------------------------------

    /// Number every step in pre-order and record the span each control step
    /// encloses.
    fn index_schedule(&mut self) -> Result<(), CompilerDefect> {
        let schedule = self.shape.schedule();
        self.walk_schedule(schedule)?;
        Ok(())
    }

    fn walk_schedule(&mut self, schedule: &ShapeSchedule) -> Result<(), CompilerDefect> {
        for step in &schedule.steps {
            let id = StepId(self.steps.len() as u32);
            match step {
                ShapeStep::Launch(block) => {
                    self.steps.push(ScheduleStep {
                        anchor: StepAnchor::Launch(*block),
                        last: id,
                    });
                    self.record_end(EdgeEnd::Block(*block), id)?;
                }
                ShapeStep::Guard { obligation, .. } => {
                    self.steps.push(ScheduleStep {
                        anchor: StepAnchor::Guard(obligation.clone()),
                        last: id,
                    });
                }
                ShapeStep::PullCounterReset { block } => {
                    self.steps.push(ScheduleStep {
                        anchor: StepAnchor::PullCounterReset(*block),
                        last: id,
                    });
                }
                ShapeStep::Call { node, .. } => {
                    self.steps.push(ScheduleStep {
                        anchor: StepAnchor::Call(node.clone()),
                        last: id,
                    });
                    self.record_end(EdgeEnd::Call(node.clone()), id)?;
                }
                ShapeStep::If {
                    node,
                    then_schedule,
                    else_schedule,
                    ..
                } => {
                    self.steps.push(ScheduleStep {
                        anchor: StepAnchor::If(node.clone()),
                        last: id,
                    });
                    self.record_end(EdgeEnd::Control(node.clone()), id)?;
                    self.walk_schedule(then_schedule)?;
                    self.walk_schedule(else_schedule)?;
                    self.close_span(id);
                }
                ShapeStep::Repeat { node, body, .. } => {
                    self.steps.push(ScheduleStep {
                        anchor: StepAnchor::Repeat(node.clone()),
                        last: id,
                    });
                    self.record_end(EdgeEnd::Control(node.clone()), id)?;
                    self.walk_schedule(body)?;
                    self.close_span(id);
                }
            }
        }
        Ok(())
    }

    fn record_end(&mut self, end: EdgeEnd, id: StepId) -> Result<(), CompilerDefect> {
        if self.end_step.insert(end.clone(), id).is_some() {
            return Err(self.defect(format!("{end:?} is scheduled twice")));
        }
        Ok(())
    }

    fn close_span(&mut self, id: StepId) {
        let last = StepId(self.steps.len() as u32 - 1);
        self.steps[id.0 as usize].last = last;
    }

    /// The step interval `[first, last]` of one end; `None` is the root
    /// boundary (live for the whole strategy).
    fn span_of(&self, end: &EdgeEnd) -> Result<Option<(StepId, StepId)>, CompilerDefect> {
        match end {
            EdgeEnd::RootBoundary => Ok(None),
            EdgeEnd::Block(_) | EdgeEnd::Call(_) | EdgeEnd::Control(_) => {
                let id = *self
                    .end_step
                    .get(end)
                    .ok_or_else(|| self.defect(format!("{end:?} is not a step of the schedule")))?;
                Ok(Some((id, self.steps[id.0 as usize].last)))
            }
        }
    }

    /// The exact lifetime of a residence touched at `ends`.
    fn lifetime_of(&self, ends: &BTreeSet<EdgeEnd>) -> Result<Lifetime, CompilerDefect> {
        if ends.is_empty() {
            return Err(self.defect("a residence is touched by no schedule end"));
        }
        if ends.contains(&EdgeEnd::RootBoundary) {
            return Ok(Lifetime::Whole);
        }
        let mut first: Option<StepId> = None;
        let mut last: Option<StepId> = None;
        for end in ends {
            let Some((f, l)) = self.span_of(end)? else {
                continue;
            };
            first = Some(first.map_or(f, |current| current.min(f)));
            last = Some(last.map_or(l, |current| current.max(l)));
        }
        let (Some(first), Some(mut last)) = (first, last) else {
            return Err(self.defect("a residence is touched by no schedule end"));
        };
        // A value entering a repeat from outside must survive every visit.
        loop {
            let mut widened = false;
            for (index, step) in self.steps.iter().enumerate() {
                let id = StepId(index as u32);
                if !matches!(step.anchor, StepAnchor::Repeat(_)) {
                    continue;
                }
                if first < id && id <= last && last < step.last {
                    last = step.last;
                    widened = true;
                }
            }
            if !widened {
                break;
            }
        }
        Ok(Lifetime::Steps { first, last })
    }

    // -- root occurrence boundary ----------------------------------------

    fn index_boundary(&mut self) -> Result<(), CompilerDefect> {
        let boundary = &self.facts.alternative(self.root_key).boundary;
        for (leaf, input) in &boundary.inputs {
            let (callee, ownership) = match input {
                InstantiatedInput::Root {
                    callee, ownership, ..
                }
                | InstantiatedInput::Call {
                    callee, ownership, ..
                } => (*callee, *ownership),
            };
            let value = self.facts.canonical_value(callee)?;
            if self
                .boundary_inputs
                .insert(value, (leaf.clone(), ownership))
                .is_some()
            {
                return Err(self.defect(format!(
                    "boundary leaf {leaf:?} shares its value class with another input leaf"
                )));
            }
        }
        for (leaf, result) in &boundary.results {
            let callee = match result {
                InstantiatedResult::Root { callee } | InstantiatedResult::Call { callee, .. } => *callee,
            };
            let value = self.facts.canonical_value(callee)?;
            // A value returned through two result leaves is one class with
            // two leaves; the first leaf in order names the result field or
            // residence, the others route to the same place.
            self.boundary_results.entry(value).or_insert_with(|| leaf.clone());
        }
        Ok(())
    }

    // -- producers ----------------------------------------------------------

    fn register_producer(&mut self, node: &OwnedNodeRef, end: EdgeEnd) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        let logical = facts.node(node)?;
        for output in &logical.outputs {
            let owned = OwnedValueRef {
                graph: node.graph,
                value: output.id(),
            };
            let value = facts.canonical_value(owned)?;
            if facts.members(value)[0] == owned {
                self.producer.insert(value, end.clone());
            } else {
                // A retained call's outputs are produced by its callee: the
                // call is their producer within this strategy.
                self.producer.entry(value).or_insert_with(|| end.clone());
            }
        }
        for region in crate::strategy::child_regions(facts, node) {
            for parameter in &facts.region(&region)?.parameters {
                if let RegionParameter::Value { id, .. } = parameter {
                    let owned = OwnedValueRef {
                        graph: node.graph,
                        value: *id,
                    };
                    let value = facts.canonical_value(owned)?;
                    if facts.members(value)[0] == owned {
                        self.producer.insert(value, end.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn index_producers(&mut self) -> Result<(), CompilerDefect> {
        for value in self.boundary_inputs.keys().copied().collect::<Vec<_>>() {
            self.producer.insert(value, EdgeEnd::RootBoundary);
        }
        for (block, cut) in self.shape.blocks().entries() {
            for node in cut.nodes.iter().cloned().collect::<Vec<_>>() {
                self.register_producer(&node, EdgeEnd::Block(block))?;
            }
        }
        let ends: Vec<(OwnedNodeRef, EdgeEnd)> = self
            .end_step
            .keys()
            .filter_map(|end| match end {
                EdgeEnd::Call(node) | EdgeEnd::Control(node) => Some((node.clone(), end.clone())),
                EdgeEnd::RootBoundary | EdgeEnd::Block(_) => None,
            })
            .collect();
        for (node, end) in ends {
            self.register_producer(&node, end)?;
        }
        Ok(())
    }

    // -- storage touches ------------------------------------------------

    fn touch_storage(&mut self, chain: CanonicalStorageId, end: EdgeEnd) {
        self.storage_ends.entry(chain).or_default().insert(end);
    }

    /// Every schedule end that reads or writes a storage chain through
    /// state: block state edges, retained call boundary states, retained
    /// control state carries/joins, and the root boundary's entry and final
    /// states.
    fn index_storage_touches(&mut self) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        for (block, cut) in self.shape.blocks().entries() {
            for chain in cut.state_reads.iter().chain(&cut.state_writes) {
                self.touch_storage(*chain, EdgeEnd::Block(block));
            }
        }
        let boundary = &facts.alternative(self.root_key).boundary;
        for input in boundary.inputs.values() {
            let state = match input {
                InstantiatedInput::Root { callee_state, .. } => *callee_state,
                InstantiatedInput::Call { states, .. } => states.map(|(callee, _)| callee),
            };
            if let Some(state) = state {
                let chain = facts.canonical_storage(facts.state_storage(state))?;
                self.touch_storage(chain, EdgeEnd::RootBoundary);
            }
        }
        for final_state in boundary.final_states.values() {
            let callee = match final_state {
                InstantiatedFinalState::Root { callee } | InstantiatedFinalState::Call { callee, .. } => *callee,
            };
            let chain = facts.canonical_storage(facts.state_storage(callee))?;
            self.touch_storage(chain, EdgeEnd::RootBoundary);
        }
        let mut touches: Vec<(CanonicalStorageId, EdgeEnd)> = Vec::new();
        self.collect_step_state_touches(self.shape.schedule(), &mut touches)?;
        for (chain, end) in touches {
            self.touch_storage(chain, end);
        }
        Ok(())
    }

    fn collect_step_state_touches(
        &self,
        schedule: &ShapeSchedule,
        out: &mut Vec<(CanonicalStorageId, EdgeEnd)>,
    ) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        for step in &schedule.steps {
            match step {
                ShapeStep::Launch(_) | ShapeStep::Guard { .. } | ShapeStep::PullCounterReset { .. } => {}
                ShapeStep::Call { node, .. } => {
                    let LogicalNodeKind::Call(call) = &facts.node(node)?.kind else {
                        return Err(self.defect(format!("call step {node:?} is not a call node")));
                    };
                    let end = EdgeEnd::Call(node.clone());
                    for input in call.boundary.inputs.values() {
                        if let CallInput::Tensor { state, .. } = input {
                            out.push((self.storage_chain(node.graph, *state)?, end.clone()));
                        }
                    }
                    for state in call.boundary.final_states.values() {
                        out.push((self.storage_chain(node.graph, *state)?, end.clone()));
                    }
                }
                ShapeStep::If {
                    node,
                    then_schedule,
                    else_schedule,
                    joins,
                    ..
                } => {
                    for join in joins {
                        if let JoinEdge::State { storage } = join {
                            out.push((*storage, EdgeEnd::Control(node.clone())));
                        }
                    }
                    self.collect_step_state_touches(then_schedule, out)?;
                    self.collect_step_state_touches(else_schedule, out)?;
                }
                ShapeStep::Repeat {
                    node, body, carries, ..
                } => {
                    for carry in carries {
                        if let CarryEdge::State { storage } = carry {
                            out.push((*storage, EdgeEnd::Control(node.clone())));
                        }
                    }
                    self.collect_step_state_touches(body, out)?;
                }
            }
        }
        Ok(())
    }

    // -- the required leaf set ---------------------------------------------

    fn require_value(&mut self, value: CanonicalValueId) {
        for leaf in self.facts.leaves(value) {
            self.required.insert(*leaf);
        }
    }

    /// Every non-local leaf: cross-edge leaves, block interface leaves,
    /// retained call boundary leaves, every leaf a retained control step
    /// names, and every root boundary leaf. Computed independently of the
    /// route table so the seal can compare the two.
    fn index_required(&mut self) -> Result<(), CompilerDefect> {
        for edge in self.shape.edges() {
            self.required.insert(edge.leaf);
            let ends = self.edge_ends.entry(edge.leaf).or_default();
            ends.insert(edge.producer.clone());
            ends.extend(edge.consumers.iter().cloned());
        }
        for cut in self.shape.blocks().iter() {
            self.required.extend(cut.inputs.iter().copied());
            self.required.extend(cut.outputs.iter().copied());
        }
        for value in self.boundary_inputs.keys().copied().collect::<Vec<_>>() {
            self.require_value(value);
        }
        for value in self.boundary_results.keys().copied().collect::<Vec<_>>() {
            self.require_value(value);
        }
        let mut values = Vec::new();
        self.collect_step_values(self.shape.schedule(), &mut values)?;
        for value in values {
            self.require_value(value);
        }
        Ok(())
    }

    fn collect_step_values(
        &self,
        schedule: &ShapeSchedule,
        out: &mut Vec<CanonicalValueId>,
    ) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        for step in &schedule.steps {
            match step {
                ShapeStep::Launch(_) | ShapeStep::Guard { .. } | ShapeStep::PullCounterReset { .. } => {}
                ShapeStep::Call { node, .. } => {
                    let LogicalNodeKind::Call(call) = &facts.node(node)?.kind else {
                        return Err(self.defect(format!("call step {node:?} is not a call node")));
                    };
                    for input in call.boundary.inputs.values() {
                        let value = match input {
                            CallInput::Value(value)
                            | CallInput::Tensor { value, .. }
                            | CallInput::Computed { value, .. } => *value,
                        };
                        out.push(self.canonical(node.graph, value)?);
                    }
                    for value in call.boundary.results.values() {
                        out.push(self.canonical(node.graph, *value)?);
                    }
                }
                ShapeStep::If {
                    condition,
                    then_schedule,
                    else_schedule,
                    joins,
                    ..
                } => {
                    out.push(*condition);
                    for join in joins {
                        match join {
                            JoinEdge::Value {
                                then_value,
                                else_value,
                                joined,
                            } => out.extend([*then_value, *else_value, *joined]),
                            JoinEdge::State { .. } => {}
                        }
                    }
                    self.collect_step_values(then_schedule, out)?;
                    self.collect_step_values(else_schedule, out)?;
                }
                ShapeStep::Repeat {
                    start,
                    end,
                    binder,
                    body,
                    carries,
                    ..
                } => {
                    out.extend([*start, *end, *binder]);
                    for carry in carries {
                        match carry {
                            CarryEdge::Value {
                                initial,
                                parameter,
                                update,
                                result,
                            } => out.extend([*initial, *parameter, *update, *result]),
                            CarryEdge::State { .. } => {}
                        }
                    }
                    self.collect_step_values(body, out)?;
                }
            }
        }
        Ok(())
    }

    fn index_local_requirements(&mut self) -> Result<(), CompilerDefect> {
        for (block, cut) in self.shape.blocks().entries() {
            for requirement in &cut.local_residences {
                if requirement.scope == StorageScope::Abi {
                    return Err(self.defect(format!(
                        "block {} requires an ABI residence for value {}; ABI residences belong to root boundary leaves only",
                        block.0, requirement.value.0
                    )));
                }
                if self
                    .local_requirements
                    .insert((block, requirement.value), requirement.clone())
                    .is_some()
                {
                    return Err(self.defect(format!(
                        "block {} requires a local residence for value {} twice",
                        block.0, requirement.value.0
                    )));
                }
            }
        }
        Ok(())
    }

    // -- anchors ------------------------------------------------------------

    /// The tensor type of a canonical value.
    fn tensor_type(&self, value: CanonicalValueId) -> Result<TensorType, CompilerDefect> {
        match self.facts.value_kind(value) {
            GraphValueKind::Tensor { ty, .. } => Ok(ty.clone()),
            other => Err(self.defect(format!(
                "value {} is routed as a tensor but its kind is {other:?}",
                value.0
            ))),
        }
    }

    /// Resolve a tensor value class to its anchor and complete transform.
    fn resolve_value(&self, value: CanonicalValueId) -> Result<Resolved, CompilerDefect> {
        if let Some((leaf, ownership)) = self.boundary_inputs.get(&value) {
            let access = match ownership {
                ParamOwnership::Shared => Access::Shared,
                ParamOwnership::Owned | ParamOwnership::Exclusive => Access::Exclusive,
                ParamOwnership::Value => {
                    return Err(self.defect(format!(
                        "tensor boundary leaf {leaf:?} is declared by value"
                    )))
                }
            };
            return Ok(Resolved {
                anchor: Anchor::BoundaryInput(leaf.clone()),
                transform: ViewTransformTemplate::identity(),
                access,
            });
        }
        match self.facts.value_kind(value) {
            GraphValueKind::Tensor {
                source: TensorSource::Computed,
                ..
            } => Ok(Resolved {
                anchor: Anchor::Computed(value),
                transform: ViewTransformTemplate::identity(),
                access: Access::Exclusive,
            }),
            GraphValueKind::Tensor {
                source: TensorSource::View(view),
                ..
            } => {
                let producer = self.facts.members(value)[0];
                let view = self.facts.view(OwnedViewRef {
                    graph: producer.graph,
                    view: *view,
                });
                match view.base {
                    ViewBase::Storage(storage) => {
                        let owned = OwnedStorageRef {
                            graph: producer.graph,
                            storage,
                        };
                        let (anchor, base) = self.resolve_storage(owned)?;
                        let local = self.qualify_view_transform(
                            producer.graph,
                            &self.facts.storage(owned).shape.axes,
                            &view.transform,
                        )?;
                        Ok(Resolved {
                            anchor,
                            transform: ViewTransformTemplate::compose(&base, &local),
                            access: view.access,
                        })
                    }
                    ViewBase::Value(base) => {
                        // A view of a computed tensor reads the base's bytes;
                        // it has no storage and no spill of its own. The
                        // anchor is the base's (a computed spill holds the
                        // base's value), and this view's transform composes
                        // onto the base's resolution.
                        let base_value = self.canonical(producer.graph, base)?;
                        let resolved_base = self.resolve_value(base_value)?;
                        let base_ty = self.tensor_type(base_value)?;
                        let local = self.qualify_view_transform(
                            producer.graph,
                            &base_ty.axes,
                            &view.transform,
                        )?;
                        Ok(Resolved {
                            anchor: resolved_base.anchor,
                            transform: ViewTransformTemplate::compose(
                                &resolved_base.transform,
                                &local,
                            ),
                            access: view.access,
                        })
                    }
                }
            }
            other => Err(self.defect(format!(
                "value {} is resolved as a tensor but its kind is {other:?}",
                value.0
            ))),
        }
    }

    /// Resolve a logical storage to its anchor and the transform from the
    /// anchor's coordinates to the storage's own: identity for a local
    /// storage and for the root occurrence's parameters; the caller's
    /// argument view (recursively) for a callee parameter storage.
    fn resolve_storage(
        &self,
        storage: OwnedStorageRef,
    ) -> Result<(Anchor, ViewTransformTemplate), CompilerDefect> {
        let facts = self.facts;
        match &facts.storage(storage).owner {
            LogicalStorageOwner::Local => Ok((
                Anchor::Storage(facts.canonical_storage(storage)?),
                ViewTransformTemplate::identity(),
            )),
            LogicalStorageOwner::Parameter(leaf) => {
                if storage.graph == self.root_key {
                    return Ok((Anchor::BoundaryInput(leaf.clone()), ViewTransformTemplate::identity()));
                }
                let boundary = &facts.alternative(storage.graph).boundary;
                let Some(input) = boundary.inputs.get(leaf) else {
                    return Err(self.defect(format!(
                        "parameter storage {storage:?} names boundary leaf {leaf:?} which its graph's boundary lacks"
                    )));
                };
                match input {
                    InstantiatedInput::Root { .. } => Err(self.defect(format!(
                        "parameter storage {storage:?} belongs to a root-instantiated graph other than the strategy root"
                    ))),
                    // A computed call argument: the caller passed a computed
                    // tensor (directly or through a view of one), so this
                    // parameter storage has no caller state — the view of it
                    // reads the argument value's own anchor.
                    InstantiatedInput::Call { caller, states: None, .. } => {
                        let argument = facts.canonical_value(*caller)?;
                        self.resolve_value(argument)
                            .map(|resolved| (resolved.anchor, resolved.transform))
                    }
                    InstantiatedInput::Call {
                        caller,
                        states: Some((_, caller_state)),
                        ..
                    } => {
                        let argument = facts.canonical_value(*caller)?;
                        let expected = facts.state_storage(*caller_state);
                        if self.boundary_inputs.contains_key(&argument) {
                            return self
                                .resolve_value(argument)
                                .map(|resolved| (resolved.anchor, resolved.transform));
                        }
                        let GraphValueKind::Tensor {
                            source: TensorSource::View(view),
                            ..
                        } = facts.value_kind(argument)
                        else {
                            return Err(self.defect(format!(
                                "call argument {caller:?} for {leaf:?} carries caller state but is not a view"
                            )));
                        };
                        let producer = facts.members(argument)[0];
                        let argument_storage = match facts
                            .view(OwnedViewRef {
                                graph: producer.graph,
                                view: *view,
                            })
                            .base
                        {
                            ViewBase::Storage(argument_storage) => argument_storage,
                            // A computed base cannot be the state this
                            // argument passes: the state names a storage.
                            ViewBase::Value(_) => {
                                return Err(self.defect(format!(
                                    "call argument {caller:?} for {leaf:?} carries caller state but views a computed value"
                                )))
                            }
                        };
                        if producer.graph != expected.graph || argument_storage != expected.storage {
                            return Err(self.defect(format!(
                                "call argument {caller:?} views a storage other than the state it passes for {leaf:?}"
                            )));
                        }
                        self.resolve_value(argument)
                            .map(|resolved| (resolved.anchor, resolved.transform))
                    }
                }
            }
        }
    }

    /// One view step relative to its storage's shape, with dynamic endpoints
    /// as canonical leaves.
    fn qualify_view_transform(
        &self,
        graph: OwnedGraphKey,
        source_shape: &[ExtentExpr],
        transform: &ViewTransform,
    ) -> Result<ViewTransformTemplate, CompilerDefect> {
        let endpoint = |value: seismic_lang::logical::GraphValueId| -> Result<CanonicalLeafId, CompilerDefect> {
            let canonical = self.canonical(graph, value)?;
            match self.facts.leaves(canonical) {
                [leaf] => Ok(*leaf),
                leaves => Err(self.defect(format!(
                    "slice endpoint value {} has {} leaves; an endpoint is one scalar leaf",
                    canonical.0,
                    leaves.len()
                ))),
            }
        };
        let kind = match transform {
            ViewTransform::Identity => return Ok(ViewTransformTemplate::identity()),
            ViewTransform::Reshape { source_shape: declared } => {
                if declared.as_slice() != source_shape {
                    return Err(self.defect(
                        "reshape source shape disagrees with its backing storage",
                    ));
                }
                ViewStepKind::Reshape
            }
            ViewTransform::Transpose { permutation } => {
                let axes: BTreeSet<u32> = permutation.iter().copied().collect();
                if permutation.len() != source_shape.len()
                    || axes.len() != source_shape.len()
                    || axes
                        .iter()
                        .enumerate()
                        .any(|(axis, value)| usize::try_from(*value) != Ok(axis))
                {
                    return Err(self.defect(
                        "transpose permutation is not a bijection of its source rank",
                    ));
                }
                ViewStepKind::Transpose {
                    permutation: permutation.clone(),
                }
            }
            ViewTransform::Slice { axes } => {
                if axes.len() != source_shape.len() {
                    return Err(self.defect(format!(
                        "slice axis descriptor count {} disagrees with source rank {}",
                        axes.len(),
                        source_shape.len()
                    )));
                }
                let mut templates = Vec::with_capacity(axes.len());
                for axis in axes {
                    templates.push(match axis {
                        SliceAxis::Full => SliceAxisTemplate::Full,
                        SliceAxis::Point(value) => SliceAxisTemplate::Point(endpoint(*value)?),
                        SliceAxis::Range { start, end } => SliceAxisTemplate::Range {
                            start: start.map(&endpoint).transpose()?,
                            end: end.map(&endpoint).transpose()?,
                        },
                    });
                }
                ViewStepKind::Slice { axes: templates }
            }
        };
        Ok(ViewTransformTemplate {
            steps: vec![ViewStepTemplate {
                source_shape: source_shape.to_vec(),
                kind,
            }],
        })
    }

    // -- planes ---------------------------------------------------------------

    fn extent_symbol(&self, extent: &ExtentExpr) -> Result<Sym, CompilerDefect> {
        let bound = match extent {
            ExtentExpr::Static(n) => *n,
            ExtentExpr::Runtime(id) => self.facts.runtime_extent(*id).capacity,
            ExtentExpr::Sym(sym) => {
                return Err(self.defect(format!(
                    "symbolic extent `{sym}` survived specialization"
                )))
            }
        };
        let bound = i64::try_from(bound)
            .map_err(|_| self.defect(format!("extent {extent} exceeds the size domain")))?;
        Ok(Sym::constant(bound))
    }

    /// The representation planes of one tensor shape with their bytes at
    /// capacity and alignment (the one plane-size authority).
    fn plane_plans(&self, shape: &TensorType) -> Result<NonEmpty<PlanePlan>, CompilerDefect> {
        let elements = shape
            .axes
            .iter()
            .try_fold(Sym::constant(1), |total, extent| {
                Ok::<_, CompilerDefect>(total.mul(&self.extent_symbol(extent)?))
            })?;
        let plans = match &shape.elem {
            Elem::Dtype(dtype) => vec![PlanePlan {
                plane: StoragePlane::Dense,
                bytes: elements.scale(i64::from(dtype.bytes())),
                alignment: u64::from(dtype.bytes()).max(1),
            }],
            Elem::Param(name) => {
                return Err(self.defect(format!(
                    "element parameter `{name}` survived specialization"
                )))
            }
            Elem::Repr(name) => {
                let Some(representation) = repr::lookup(name) else {
                    return Err(self.defect(format!("unknown representation `{name}`")));
                };
                let Some(packed_axis) = shape.packed_axis else {
                    return Err(self.defect(format!(
                        "packed `{name}` tensor has no packed axis"
                    )));
                };
                let Some(columns) = shape.axes.get(packed_axis) else {
                    return Err(self.defect(format!(
                        "packed axis {packed_axis} is outside rank {}",
                        shape.axes.len()
                    )));
                };
                let columns = self.extent_symbol(columns)?;
                let storage_group = i64::from(representation.storage_group());
                let physical_width = columns
                    .add(&Sym::constant(storage_group - 1))
                    .quot(&Sym::constant(storage_group))
                    .scale(storage_group);
                let rows = shape
                    .axes
                    .iter()
                    .enumerate()
                    .filter(|(axis, _)| *axis != packed_axis)
                    .try_fold(Sym::constant(1), |total, (_, extent)| {
                        Ok::<_, CompilerDefect>(total.mul(&self.extent_symbol(extent)?))
                    })?;
                representation
                    .planes()
                    .into_iter()
                    .enumerate()
                    .map(|(ordinal, plane)| {
                        let row_entries = physical_width
                            .quot(&Sym::constant(i64::from(plane.group)))
                            .scale(i64::from(plane.fields));
                        let bytes = match &plane.encoding {
                            repr::PlaneEncoding::Dense(dtype) => {
                                rows.mul(&row_entries.scale(i64::from(dtype.bytes())))
                            }
                            repr::PlaneEncoding::Packed { bits, .. } => rows.mul(
                                &row_entries
                                    .scale(i64::from(*bits))
                                    .add(&Sym::constant(31))
                                    .quot(&Sym::constant(32))
                                    .scale(4),
                            ),
                        };
                        PlanePlan {
                            plane: StoragePlane::Representation {
                                name: plane.name.to_string(),
                                ordinal: ordinal as u32,
                            },
                            bytes,
                            alignment: u64::from(plane.dtype().bytes()).max(1),
                        }
                    })
                    .collect()
            }
        };
        NonEmpty::new(plans).ok_or_else(|| {
            self.defect(format!("representation of {shape:?} declares no plane"))
        })
    }

    // -- residences -----------------------------------------------------

    fn push_residence(&mut self, residence: Residence) -> ResidenceId {
        let id = ResidenceId(self.residences.len() as u32);
        self.residences.push(residence);
        id
    }

    /// `first` plus `rest` are the scope options: a scope choice cannot be
    /// constructed without a first scope, so the option list is nonempty by
    /// the signature itself.
    fn solver_choice(
        &mut self,
        first: StorageScope,
        rest: Vec<StorageScope>,
    ) -> Result<ResidenceChoice, CompilerDefect> {
        let var = ChoiceVarId(self.choice_vars);
        self.choice_vars += 1;
        let options = NonEmpty::new(std::iter::once(first).chain(rest).collect())
            .ok_or_else(|| self.defect("a scope choice lists at least one scope"))?;
        Ok(ResidenceChoice::SolverChoice { options, var })
    }

    /// The replication of an unnamed kernel-local storage in `block`: every
    /// storage inside a linear/cooperative block lives inside its
    /// independent point, so each participant holds its own replica; a serial
    /// block has one participant.
    fn local_replication(&self, block: BlockId) -> Replication {
        match self.shape.blocks()[block].participants.policy {
            ParticipantPolicy::Serial => Replication::Once,
            ParticipantPolicy::Linear { .. }
            | ParticipantPolicy::Cooperative { .. }
            | ParticipantPolicy::DynamicPull { .. }
            | ParticipantPolicy::GridCooperative { .. } => Replication::PerParticipant,
        }
    }

    /// The type of the leaf at `path` of a value type.
    fn leaf_type(&self, leaf: CanonicalLeafId) -> Result<ValueType, CompilerDefect> {
        let record = self.facts.leaf(leaf);
        let mut current = self.facts.value_type(record.value);
        for index in &record.path.0 {
            current = match current {
                ValueType::Tuple(items) => match items.as_slice().get(*index as usize) {
                    Some(item) => item.clone(),
                    None => {
                        return Err(self.defect(format!(
                            "leaf {} names tuple component {index} outside its value",
                            leaf.0
                        )))
                    }
                },
                other => {
                    return Err(self.defect(format!(
                        "leaf {} descends into non-tuple type {other}",
                        leaf.0
                    )))
                }
            };
        }
        Ok(current)
    }

    /// Resolve every required tensor leaf to its anchor.
    fn resolve_required(&mut self) -> Result<(), CompilerDefect> {
        for leaf in self.required.iter().copied().collect::<Vec<_>>() {
            if let ValueType::Tensor(_) = self.leaf_type(leaf)? {
                let record = self.facts.leaf(leaf);
                if !record.path.0.is_empty() {
                    return Err(self.defect(format!(
                        "tensor leaf {} is a component of tuple value {}; the canonical registry supplies no component view for it",
                        leaf.0, record.value.0
                    )));
                }
                let resolved = self.resolve_value(record.value)?;
                self.resolved.insert(leaf, resolved);
            }
        }
        Ok(())
    }

    /// Every required leaf anchored to a storage chain, with the ends of its
    /// cross edges.
    fn chain_route_ends(&self, chain: CanonicalStorageId) -> (bool, BTreeSet<EdgeEnd>) {
        let mut ends = BTreeSet::new();
        let mut crossing = false;
        for (leaf, resolved) in &self.resolved {
            if resolved.anchor != Anchor::Storage(chain) {
                continue;
            }
            crossing = true;
            if let Some(edge_ends) = self.edge_ends.get(leaf) {
                ends.extend(edge_ends.iter().cloned());
            }
        }
        (crossing, ends)
    }

    fn form_residences(&mut self) -> Result<(), CompilerDefect> {
        self.resolve_required()?;
        self.form_boundary_input_residences()?;
        self.form_boundary_result_residences()?;
        self.form_storage_residences()?;
        self.form_spill_residences()?;
        self.form_required_local_residences()?;
        self.form_pull_counters()?;
        Ok(())
    }

    /// One 4-byte device pull counter per `DynamicPull` block, live from
    /// the block's `PullCounterReset` step (immediately before its launch)
    /// to its launch. Not a semantic value: no leaf, no route.
    fn form_pull_counters(&mut self) -> Result<(), CompilerDefect> {
        for (block, cut) in self.shape.blocks().entries() {
            if !matches!(cut.participants.policy, ParticipantPolicy::DynamicPull { .. }) {
                continue;
            }
            let reset = self
                .steps
                .iter()
                .position(|step| step.anchor == StepAnchor::PullCounterReset(block))
                .ok_or_else(|| {
                    self.defect(format!(
                        "block {} pulls dynamically but the schedule has no PullCounterReset for it",
                        block.0
                    ))
                })?;
            let launch = self
                .steps
                .iter()
                .position(|step| step.anchor == StepAnchor::Launch(block))
                .ok_or_else(|| self.defect(format!("block {} has no launch step", block.0)))?;
            if launch != reset + 1 {
                return Err(self.defect(format!(
                    "block {}: PullCounterReset is step {reset} but the launch is step {launch}; the reset immediately precedes the launch",
                    block.0
                )));
            }
            let shape = TensorType::new(vec![ExtentExpr::Static(1)], Elem::Dtype(DType::U32));
            let planes = self.plane_plans(&shape)?;
            let id = self.push_residence(Residence {
                source: ResidenceSource::PullCounter { block },
                shape,
                planes,
                lifetime: Lifetime::Steps {
                    first: StepId(reset as u32),
                    last: StepId(launch as u32),
                },
                choice: ResidenceChoice::Fixed(StorageScope::DeviceArena),
                replication: Replication::Once,
            });
            self.pull_counters.insert(block, id);
        }
        Ok(())
    }

    fn form_boundary_input_residences(&mut self) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        let boundary = &facts.alternative(self.root_key).boundary;
        for (leaf, input) in &boundary.inputs {
            let (callee, state) = match input {
                InstantiatedInput::Root {
                    callee, callee_state, ..
                } => (*callee, *callee_state),
                InstantiatedInput::Call { callee, states, .. } => {
                    (*callee, states.map(|(callee_state, _)| callee_state))
                }
            };
            let value = facts.canonical_value(callee)?;
            let ty = self.facts.value_type(value);
            let (ValueType::Tensor(shape), Some(state)) = (&ty, state) else {
                if let (ValueType::Tensor(_), None) | (_, Some(_)) = (&ty, state) {
                    return Err(self.defect(format!(
                        "boundary leaf {leaf:?} pairs type {ty} with the wrong state presence"
                    )));
                }
                continue;
            };
            let chain = facts.canonical_storage(facts.state_storage(state))?;
            let id = match self.storage_residence.get(&chain) {
                Some(id) => *id,
                None => {
                    let planes = self.plane_plans(shape)?;
                    let id = self.push_residence(Residence {
                        source: ResidenceSource::RootInput(leaf.clone()),
                        shape: shape.clone(),
                        planes,
                        lifetime: Lifetime::Whole,
                        choice: ResidenceChoice::Fixed(StorageScope::Abi),
                        replication: Replication::Once,
                    });
                    self.storage_residence.insert(chain, id);
                    id
                }
            };
            self.boundary_input_residence.insert(leaf.clone(), id);
        }
        Ok(())
    }

    fn form_boundary_result_residences(&mut self) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        let boundary = &facts.alternative(self.root_key).boundary;
        for (leaf, result) in &boundary.results {
            let callee = match result {
                InstantiatedResult::Root { callee } | InstantiatedResult::Call { callee, .. } => *callee,
            };
            let value = facts.canonical_value(callee)?;
            let GraphValueKind::Tensor { .. } = facts.value_kind(value) else {
                continue;
            };
            let resolved = self.resolve_value(value)?;
            match resolved.anchor {
                // Pass-through of a parameter: the result names the input
                // residence (aliasing by route equality).
                Anchor::BoundaryInput(_) => {}
                Anchor::Storage(chain) => {
                    if self.storage_residence.contains_key(&chain) {
                        continue;
                    }
                    let owning = facts.storage_members(chain)[0];
                    let shape = facts.storage(owning).shape.clone();
                    let planes = self.plane_plans(&shape)?;
                    let id = self.push_residence(Residence {
                        source: ResidenceSource::RootResult(leaf.clone()),
                        shape,
                        planes,
                        lifetime: Lifetime::Whole,
                        choice: ResidenceChoice::Fixed(StorageScope::Abi),
                        replication: Replication::Once,
                    });
                    self.storage_residence.insert(chain, id);
                }
                Anchor::Computed(computed) => {
                    if self.spill_residence.contains_key(&computed) {
                        continue;
                    }
                    // The residence holds the anchor value's bytes in the
                    // anchor's own coordinates (the base's, for a computed
                    // base view result); the result leaf's route carries its
                    // view transform from these coordinates.
                    let shape = self.tensor_type(computed)?;
                    let planes = self.plane_plans(&shape)?;
                    let id = self.push_residence(Residence {
                        source: ResidenceSource::RootResult(leaf.clone()),
                        shape,
                        planes,
                        lifetime: Lifetime::Whole,
                        choice: ResidenceChoice::Fixed(StorageScope::Abi),
                        replication: Replication::Once,
                    });
                    self.spill_residence.insert(computed, id);
                }
            }
        }
        Ok(())
    }

    /// The local residence requirements whose value resolves to `chain`.
    fn requirements_of_chain(
        &self,
        chain: CanonicalStorageId,
    ) -> Result<Vec<(BlockId, LocalResidenceRequirement)>, CompilerDefect> {
        let mut out = Vec::new();
        for ((block, value), requirement) in &self.local_requirements {
            let resolved = self.resolve_value(*value)?;
            if resolved.anchor == Anchor::Storage(chain) {
                out.push((*block, requirement.clone()));
            }
        }
        Ok(out)
    }

    fn form_storage_residences(&mut self) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        let mut chains: BTreeSet<CanonicalStorageId> = self.storage_ends.keys().copied().collect();
        for resolved in self.resolved.values() {
            if let Anchor::Storage(chain) = resolved.anchor {
                chains.insert(chain);
            }
        }
        for (_, value) in self.local_requirements.keys() {
            if let Anchor::Storage(chain) = self.resolve_value(*value)?.anchor {
                chains.insert(chain);
            }
        }
        for chain in chains {
            if self.storage_residence.contains_key(&chain) {
                continue;
            }
            let (crossing, route_ends) = self.chain_route_ends(chain);
            let mut ends = route_ends;
            if let Some(touched) = self.storage_ends.get(&chain) {
                ends.extend(touched.iter().cloned());
            }
            if ends.contains(&EdgeEnd::RootBoundary) {
                return Err(self.defect(format!(
                    "storage chain {} is touched by the root boundary but is neither a parameter storage nor a returned local storage",
                    chain.0
                )));
            }
            let owning = facts.storage_members(chain)[0];
            let shape = facts.storage(owning).shape.clone();
            let planes = self.plane_plans(&shape)?;
            let local_block = match ends.iter().collect::<Vec<&EdgeEnd>>().as_slice() {
                [EdgeEnd::Block(block)] if !crossing => Some(*block),
                _ => None,
            };
            let requirements = self.requirements_of_chain(chain)?;
            let residence = match local_block {
                Some(block) => {
                    let (choice, replication) = match requirements.as_slice() {
                        [] => {
                            // The scope of an unrequired local residence is
                            // fixed by its replication: per-participant
                            // copies live in participant-private storage,
                            // workgroup-subgroup copies and single shared
                            // copies in workgroup storage. A solver choice
                            // across these would admit a placement whose
                            // sharing contradicts the replication.
                            let replication = self.local_replication(block);
                            let choice = match replication {
                                Replication::PerParticipant => {
                                    ResidenceChoice::Fixed(StorageScope::Participant)
                                }
                                Replication::Once
                                | Replication::PerWorkgroup
                                | Replication::PerSubgroup => {
                                    ResidenceChoice::Fixed(StorageScope::Workgroup)
                                }
                            };
                            (choice, replication)
                        }
                        [(owner, first), rest @ ..] => {
                            if *owner != block {
                                return Err(self.defect(format!(
                                    "block {} requires a local residence for storage chain {} which is local to block {}",
                                    owner.0, chain.0, block.0
                                )));
                            }
                            for (other_owner, other) in rest {
                                if *other_owner != block
                                    || other.scope != first.scope
                                    || other.replication != first.replication
                                {
                                    return Err(self.defect(format!(
                                        "storage chain {} receives conflicting local residence requirements",
                                        chain.0
                                    )));
                                }
                            }
                            (ResidenceChoice::Fixed(first.scope), first.replication)
                        }
                    };
                    Residence {
                        source: ResidenceSource::LogicalStorage(chain),
                        shape,
                        planes,
                        lifetime: Lifetime::KernelLocal(block),
                        choice,
                        replication,
                    }
                }
                None => {
                    if let Some((owner, _)) = requirements.first() {
                        return Err(self.defect(format!(
                            "block {} requires a local residence for storage chain {} which crosses a cut",
                            owner.0, chain.0
                        )));
                    }
                    Residence {
                        source: ResidenceSource::LogicalStorage(chain),
                        shape,
                        planes,
                        lifetime: self.lifetime_of(&ends)?,
                        choice: ResidenceChoice::Fixed(StorageScope::DeviceArena),
                        replication: Replication::Once,
                    }
                }
            };
            let id = self.push_residence(residence);
            self.storage_residence.insert(chain, id);
        }
        Ok(())
    }

    fn form_spill_residences(&mut self) -> Result<(), CompilerDefect> {
        // One spill per computed anchor, live over the union of the ends of
        // every leaf resolving to it: the anchor's own leaf and every
        // computed-base view's leaf share the anchor's bytes.
        let mut computed: BTreeMap<CanonicalValueId, BTreeSet<EdgeEnd>> = BTreeMap::new();
        let mut crossing: BTreeSet<CanonicalValueId> = BTreeSet::new();
        for (leaf, resolved) in &self.resolved {
            if let Anchor::Computed(value) = resolved.anchor {
                let ends = computed.entry(value).or_default();
                ends.extend(self.edge_ends.get(leaf).into_iter().flatten().cloned());
                if self.edge_ends.contains_key(leaf) {
                    crossing.insert(value);
                }
            }
        }
        for (value, ends) in computed {
            if self.spill_residence.contains_key(&value) {
                continue;
            }
            if !crossing.contains(&value) {
                return Err(self.defect(format!(
                    "computed tensor {} is required outside its block but crosses no cut",
                    value.0
                )));
            }
            if ends.len() < 2 {
                return Err(self.defect(format!(
                    "computed tensor {} crosses a cut with a single end {ends:?}",
                    value.0
                )));
            }
            let shape = self.tensor_type(value)?;
            let planes = self.plane_plans(&shape)?;
            let lifetime = self.lifetime_of(&ends)?;
            let id = self.push_residence(Residence {
                source: ResidenceSource::ComputedSpill(value),
                shape,
                planes,
                lifetime,
                choice: ResidenceChoice::Fixed(StorageScope::DeviceArena),
                replication: Replication::Once,
            });
            self.spill_residence.insert(value, id);
        }
        Ok(())
    }

    /// Kernel-local residences the proposal requires for computed values
    /// that live inside one block (storage-backed requirements were bound in
    /// `form_storage_residences`).
    fn form_required_local_residences(&mut self) -> Result<(), CompilerDefect> {
        let requirements: Vec<(BlockId, CanonicalValueId, LocalResidenceRequirement)> = self
            .local_requirements
            .iter()
            .map(|((block, value), requirement)| (*block, *value, requirement.clone()))
            .collect();
        for (block, value, requirement) in requirements {
            let resolved = self.resolve_value(value)?;
            match resolved.anchor {
                Anchor::BoundaryInput(leaf) => {
                    return Err(self.defect(format!(
                        "block {} requires a local residence for boundary leaf {leaf:?}; boundary residences are ABI-fixed",
                        block.0
                    )))
                }
                Anchor::Storage(chain) => {
                    let id = *self.storage_residence.get(&chain).ok_or_else(|| {
                        self.defect(format!(
                            "storage chain {} of a local requirement has no residence",
                            chain.0
                        ))
                    })?;
                    if self.residences[id.0 as usize].lifetime != Lifetime::KernelLocal(block) {
                        return Err(self.defect(format!(
                            "storage chain {} required locally by block {} is not local to it",
                            chain.0, block.0
                        )));
                    }
                }
                Anchor::Computed(computed) => {
                    if self.spill_residence.contains_key(&computed) {
                        return Err(self.defect(format!(
                            "block {} requires a local residence for computed value {} which crosses a cut",
                            block.0, computed.0
                        )));
                    }
                    if self.producer.get(&computed) != Some(&EdgeEnd::Block(block)) {
                        return Err(self.defect(format!(
                            "block {} requires a local residence for computed value {} which it does not produce",
                            block.0, computed.0
                        )));
                    }
                    let shape = self.tensor_type(computed)?;
                    let planes = self.plane_plans(&shape)?;
                    self.push_residence(Residence {
                        source: ResidenceSource::KernelLocal {
                            block,
                            value: computed,
                        },
                        shape,
                        planes,
                        lifetime: Lifetime::KernelLocal(block),
                        choice: ResidenceChoice::Fixed(requirement.scope),
                        replication: requirement.replication,
                    });
                }
            }
        }
        Ok(())
    }

    // -- routes ---------------------------------------------------------------

    fn residence_of(&self, anchor: &Anchor) -> Result<ResidenceId, CompilerDefect> {
        let found = match anchor {
            Anchor::BoundaryInput(leaf) => self.boundary_input_residence.get(leaf).copied(),
            Anchor::Storage(chain) => self.storage_residence.get(chain).copied(),
            Anchor::Computed(value) => self.spill_residence.get(value).copied(),
        };
        found.ok_or_else(|| self.defect(format!("{anchor:?} has no residence")))
    }

    fn scalar_route(&mut self, leaf: CanonicalLeafId) -> ScalarRoute {
        let record = self.facts.leaf(leaf);
        let endpoint = record.endpoint;
        if let Some((boundary, _)) = self.boundary_inputs.get(&record.value) {
            return ScalarRoute::RootAbi {
                leaf: boundary.clone(),
                endpoint,
            };
        }
        if let Some(boundary) = self.boundary_results.get(&record.value) {
            return ScalarRoute::ResultField {
                leaf: boundary.clone(),
                endpoint,
            };
        }
        let slot = ExecutorScalarSlotId(self.scalar_slots);
        self.scalar_slots += 1;
        ScalarRoute::ExecutorSlot(slot)
    }

    /// The sealed route of one block interface leaf; every such leaf is
    /// routed before interfaces are formed.
    fn sealed_route(&self, leaf: CanonicalLeafId) -> Result<&ValueRoute, CompilerDefect> {
        self.routes.get(&leaf).ok_or_else(|| {
            self.defect(format!("interface leaf {} is not routed", leaf.0))
        })
    }

    fn route_of(&mut self, leaf: CanonicalLeafId) -> Result<ValueRoute, CompilerDefect> {
        match self.leaf_type(leaf)? {
            ValueType::Tensor(_) => {
                let resolved = self
                    .resolved
                    .get(&leaf)
                    .cloned()
                    .ok_or_else(|| self.defect(format!("tensor leaf {} was not resolved", leaf.0)))?;
                let residence = self.residence_of(&resolved.anchor)?;
                Ok(ValueRoute::Tensor(TensorRoute {
                    residence,
                    access: resolved.access,
                    transform: resolved.transform,
                }))
            }
            ValueType::Scalar(_) | ValueType::Index { .. } | ValueType::Range { .. } => {
                Ok(ValueRoute::Scalar(self.scalar_route(leaf)))
            }
            ValueType::Tuple(_) => Err(self.defect(format!(
                "leaf {} is a tuple; the canonical registry flattens tuples into leaves",
                leaf.0
            ))),
            ValueType::Void => Err(self.defect(format!("leaf {} is void; void has no leaf", leaf.0))),
            ValueType::CapabilityValue(ty) => Err(self.defect(format!(
                "capability value leaf {} ({}.{}) crosses a strategy boundary; a capability value lives inside one launch",
                leaf.0, ty.target, ty.name
            ))),
        }
    }

    /// Route every required leaf, then every dynamic slice endpoint the
    /// tensor routes name (they are scalar leaves the addressing needs, so
    /// they are non-local by construction and join the required set).
    fn form_routes(&mut self) -> Result<(), CompilerDefect> {
        for leaf in self.required.iter().copied().collect::<Vec<_>>() {
            let route = self.route_of(leaf)?;
            self.routes.insert(leaf, route);
        }
        let mut endpoints: BTreeSet<CanonicalLeafId> = BTreeSet::new();
        for route in self.routes.values() {
            if let ValueRoute::Tensor(tensor) = route {
                endpoints.extend(tensor.transform.dynamic_endpoints());
            }
        }
        for endpoint in endpoints {
            if self.routes.contains_key(&endpoint) {
                continue;
            }
            self.required.insert(endpoint);
            let route = self.route_of(endpoint)?;
            self.routes.insert(endpoint, route);
        }
        Ok(())
    }

    // -- closed kernel interfaces -------------------------------------------

    fn producer_block(&self, leaf: CanonicalLeafId) -> Option<BlockId> {
        match self.producer.get(&self.facts.leaf(leaf).value) {
            Some(EdgeEnd::Block(block)) => Some(*block),
            Some(EdgeEnd::RootBoundary | EdgeEnd::Call(_) | EdgeEnd::Control(_)) | None => None,
        }
    }

    /// The tensor values produced inside `block` (node outputs, producer
    /// members only), in canonical order.
    fn block_tensor_values(&self, block: BlockId) -> Result<BTreeSet<CanonicalValueId>, CompilerDefect> {
        let facts = self.facts;
        let mut out = BTreeSet::new();
        for node in self.shape.blocks()[block].nodes.iter() {
            for output in &facts.node(node)?.outputs {
                let owned = OwnedValueRef {
                    graph: node.graph,
                    value: output.id(),
                };
                let value = facts.canonical_value(owned)?;
                if facts.members(value)[0] == owned
                    && matches!(facts.value_kind(value), GraphValueKind::Tensor { .. })
                {
                    out.insert(value);
                }
            }
        }
        Ok(out)
    }

    fn form_interfaces(&mut self) -> Result<IdVec<BlockId, RoutedBlock>, CompilerDefect> {
        let facts = self.facts;
        // Dynamic endpoints of tensor interface routes: inputs of the block
        // that addresses them unless it produces them; outputs of the block
        // that produces them.
        let mut extra_inputs: BTreeMap<BlockId, BTreeSet<CanonicalLeafId>> = BTreeMap::new();
        let mut extra_outputs: BTreeMap<BlockId, BTreeSet<CanonicalLeafId>> = BTreeMap::new();
        for (block, cut) in self.shape.blocks().entries() {
            for leaf in cut.inputs.iter().chain(&cut.outputs) {
                let ValueRoute::Tensor(tensor) = self.sealed_route(*leaf)? else {
                    continue;
                };
                for endpoint in tensor.transform.dynamic_endpoints() {
                    let producer = self.producer_block(endpoint);
                    if producer != Some(block) && !cut.inputs.contains(&endpoint) {
                        extra_inputs.entry(block).or_default().insert(endpoint);
                    }
                    if let Some(source) = producer {
                        if source != block && !self.shape.blocks()[source].outputs.contains(&endpoint) {
                            extra_outputs.entry(source).or_default().insert(endpoint);
                        }
                    }
                }
            }
        }
        let mut blocks = Vec::new();
        for (block, cut) in self.shape.blocks().entries() {
            let mut input_leaves: BTreeSet<CanonicalLeafId> = cut.inputs.clone();
            input_leaves.extend(extra_inputs.get(&block).into_iter().flatten().copied());
            let mut output_leaves: BTreeSet<CanonicalLeafId> = cut.outputs.clone();
            output_leaves.extend(extra_outputs.get(&block).into_iter().flatten().copied());
            let inputs = IdVec::new(
                input_leaves
                    .iter()
                    .enumerate()
                    .map(|(index, leaf)| {
                        Ok(KernelInputDecl {
                            id: KernelInputId(index as u32),
                            leaf: *leaf,
                            route: self.sealed_route(*leaf)?.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>, CompilerDefect>>()?,
            );
            let outputs = IdVec::new(
                output_leaves
                    .iter()
                    .enumerate()
                    .map(|(index, leaf)| {
                        Ok(KernelOutputDecl {
                            id: KernelOutputId(index as u32),
                            leaf: *leaf,
                            route: self.sealed_route(*leaf)?.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>, CompilerDefect>>()?,
            );

            let block_values = self.block_tensor_values(block)?;
            let mut locals = Vec::new();
            for (index, residence) in self.residences.iter().enumerate() {
                if residence.lifetime != Lifetime::KernelLocal(block) {
                    continue;
                }
                let id = ResidenceId(index as u32);
                match &residence.source {
                    ResidenceSource::LogicalStorage(chain) => {
                        for value in &block_values {
                            let resolved = self.resolve_value(*value)?;
                            if resolved.anchor != Anchor::Storage(*chain) {
                                continue;
                            }
                            locals.push(KernelLocalDecl {
                                id: KernelLocalId(locals.len() as u32),
                                value: *value,
                                route: TensorRoute {
                                    residence: id,
                                    access: resolved.access,
                                    transform: resolved.transform,
                                },
                            });
                        }
                    }
                    ResidenceSource::KernelLocal { value, .. } => {
                        locals.push(KernelLocalDecl {
                            id: KernelLocalId(locals.len() as u32),
                            value: *value,
                            route: TensorRoute {
                                residence: id,
                                access: Access::Exclusive,
                                transform: ViewTransformTemplate::identity(),
                            },
                        });
                    }
                    ResidenceSource::RootInput(_)
                    | ResidenceSource::RootResult(_)
                    | ResidenceSource::ComputedSpill(_)
                    | ResidenceSource::PullCounter { .. } => {
                        return Err(self.defect(format!(
                            "residence {} has a kernel-local lifetime but source {:?}",
                            index, residence.source
                        )))
                    }
                }
            }

            let mut axes = Vec::new();
            let mut extents = Vec::new();
            for (index, axis) in cut.participants.independent_axes.iter().enumerate() {
                if usize::try_from(axis.ordinal) != Ok(index) {
                    return Err(self.defect(format!(
                        "block {} lists independent axis ordinal {} at position {index}",
                        block.0, axis.ordinal
                    )));
                }
                axes.push(KernelAxisDecl {
                    id: KernelAxisId(index as u32),
                    binder: Some(axis.binder),
                });
                extents.push(axis.extent.clone());
            }
            let participants = |reference: crate::strategy::TuningRef| -> Result<Sym, CompilerDefect> {
                match self.shape.tuning().get(PlanParamId(reference.0)) {
                    Some(declaration) => Ok(Sym::param(&declaration.name)),
                    None => Err(self.defect(format!(
                        "block {} names tuning parameter {} which the shape does not declare",
                        block.0, reference.0
                    ))),
                }
            };
            let (mapping, pull_counter) = match &cut.participants.policy {
                ParticipantPolicy::Serial => (AxisMapping::Serialized, PullCounter::None),
                ParticipantPolicy::Linear { participants: count }
                | ParticipantPolicy::GridCooperative { participants: count } => (
                    AxisMapping::GridStride {
                        participants: participants(*count)?,
                    },
                    PullCounter::None,
                ),
                ParticipantPolicy::Cooperative { width, .. } => (
                    AxisMapping::GridStride {
                        participants: participants(*width)?,
                    },
                    PullCounter::None,
                ),
                ParticipantPolicy::DynamicPull { participants: count } => {
                    let counter = *self.pull_counters.get(&block).ok_or_else(|| {
                        self.defect(format!("block {} pulls dynamically without a counter", block.0))
                    })?;
                    (
                        AxisMapping::DynamicPull {
                            participants: participants(*count)?,
                            counter,
                        },
                        PullCounter::Counter(counter),
                    )
                }
            };
            let iteration =
                LinearIterationMap::from_axes(&extents, mapping, &|id| facts.runtime_extent(id))
                    .map_err(|error| {
                        self.defect(format!("block {} iteration map: {error}", block.0))
                    })?;

            blocks.push(RoutedBlock {
                id: block,
                interface: ClosedKernelInterface {
                    inputs,
                    outputs,
                    locals: IdVec::new(locals),
                    axes: IdVec::new(axes),
                    iteration,
                },
                pull_counter,
            });
        }
        Ok(IdVec::new(blocks))
    }

    // -- seal -----------------------------------------------------------------

    fn seal_check(&self, blocks: &IdVec<BlockId, RoutedBlock>) -> Result<(), CompilerDefect> {
        let routed: BTreeSet<CanonicalLeafId> = self.routes.keys().copied().collect();
        if routed != self.required {
            let missing: Vec<u32> = self.required.difference(&routed).map(|leaf| leaf.0).collect();
            let extra: Vec<u32> = routed.difference(&self.required).map(|leaf| leaf.0).collect();
            return Err(self.defect(format!(
                "route domain differs from the required leaf set; unrouted {missing:?}, unrequired {extra:?}"
            )));
        }
        if std::env::var_os("SEISMIC_DEBUG_SEAL").is_some() {
            for (id, cut) in self.shape.blocks().entries() {
                eprintln!(
                    "  BLOCK {} nodes={:?}\n    inputs={:?} outputs={:?}",
                    id.0,
                    cut.nodes.iter().map(|n| (n.graph.occurrence.0, n.node.node.0, format!("{:?}", self.facts.node(n).unwrap().kind).chars().take(60).collect::<String>())).collect::<Vec<_>>(),
                    cut.inputs.iter().map(|l| l.0).collect::<Vec<_>>(),
                    cut.outputs.iter().map(|l| l.0).collect::<Vec<_>>(),
                );
            }
        }
        let mut referenced: BTreeSet<ResidenceId> = BTreeSet::new();
        for (leaf, route) in &self.routes {
            if let ValueRoute::Tensor(tensor) = route {
                if tensor.residence.0 as usize >= self.residences.len() {
                    return Err(self.defect(format!(
                        "leaf {} routes to residence {} which does not exist",
                        leaf.0, tensor.residence.0
                    )));
                }
                referenced.insert(tensor.residence);
            }
            if let Some(block) = self.producer_block(*leaf) {
                // A storage-backed view's dataflow is its storage's (state
                // edges and storage routes); the block outputs come from the
                // strategy's writer classification, not the origin node (an
                // allocation declares the storage and produces no data).
                // The origin-block requirement binds computed tensors only.
                let leaf_value = self.facts.leaf(*leaf).value;
                if std::env::var_os("SEISMIC_DEBUG_SEAL").is_some() {
                    eprintln!(
                        "  PRODUCER-CHECK leaf {} value {} kind={:?} members={:?} producer={:?}",
                        leaf.0,
                        leaf_value.0,
                        self.facts.value_kind(leaf_value),
                        self.facts.members(leaf_value),
                        self.producer.get(&leaf_value),
                    );
                }
                let storage_backed = crate::strategy::storage_backed_view(
                    self.facts,
                    leaf_value,
                );
                // The origin-block requirement binds computed tensors: a
                // tensor originating in the block must leave as an output to
                // be routed. Scalars are exempt — an in-block scalar (a
                // constant, an absorbed loop's binder serving a slice
                // endpoint route) needs no output, and a cross-block
                // scalar's output comes from the strategy's edges.
                let is_tensor = matches!(
                    self.facts.value_kind(leaf_value),
                    seismic_lang::logical::value::GraphValueKind::Tensor { .. }
                );
                if is_tensor
                    && !storage_backed
                    && !blocks[block]
                        .interface
                        .outputs
                        .iter()
                        .any(|output| output.leaf == *leaf)
                {
                    return Err(self.defect(format!(
                        "leaf {} is produced by block {} and routed but is not an output of that block (outputs={:?}, cut outputs={:?}, value={})",
                        leaf.0,
                        block.0,
                        blocks[block].interface.outputs.iter().map(|o| o.leaf.0).collect::<Vec<_>>(),
                        self.shape.blocks()[block].outputs.iter().map(|l| l.0).collect::<Vec<_>>(),
                        self.facts.leaf(*leaf).value.0,
                    )));
                }
            }
        }
        let mut local_owners: BTreeSet<ResidenceId> = BTreeSet::new();
        for routed_block in blocks.iter() {
            for local in routed_block.interface.locals.iter() {
                local_owners.insert(local.route.residence);
            }
        }
        let mut storages: BTreeSet<CanonicalStorageId> = BTreeSet::new();
        let mut spills: BTreeSet<CanonicalValueId> = BTreeSet::new();
        for (index, residence) in self.residences.iter().enumerate() {
            let id = ResidenceId(index as u32);
            match &residence.source {
                ResidenceSource::LogicalStorage(chain) => {
                    if !storages.insert(*chain) {
                        return Err(self.defect(format!(
                            "storage chain {} names more than one residence",
                            chain.0
                        )));
                    }
                }
                ResidenceSource::ComputedSpill(value) => {
                    if !spills.insert(*value) {
                        return Err(self.defect(format!(
                            "computed tensor {} names more than one spill residence",
                            value.0
                        )));
                    }
                }
                ResidenceSource::PullCounter { block } => {
                    if !matches!(
                        self.shape.blocks()[*block].participants.policy,
                        ParticipantPolicy::DynamicPull { .. }
                    ) {
                        return Err(self.defect(format!(
                            "residence {index} is the pull counter of block {} whose policy is not DynamicPull",
                            block.0
                        )));
                    }
                    if self.pull_counters.get(block) != Some(&id) {
                        return Err(self.defect(format!(
                            "block {} names more than one pull counter",
                            block.0
                        )));
                    }
                    let reset = self
                        .steps
                        .iter()
                        .position(|step| step.anchor == StepAnchor::PullCounterReset(*block));
                    let launch = self
                        .steps
                        .iter()
                        .position(|step| step.anchor == StepAnchor::Launch(*block));
                    let expected = match (reset, launch) {
                        (Some(reset), Some(launch)) => Lifetime::Steps {
                            first: StepId(reset as u32),
                            last: StepId(launch as u32),
                        },
                        (None, _) | (_, None) => {
                            return Err(self.defect(format!(
                                "block {} has a pull counter but no reset/launch step pair",
                                block.0
                            )))
                        }
                    };
                    if residence.lifetime != expected {
                        return Err(self.defect(format!(
                            "pull counter of block {} is live over {:?}, not exactly reset->launch {expected:?}",
                            block.0, residence.lifetime
                        )));
                    }
                    if blocks[*block].pull_counter != PullCounter::Counter(id) {
                        return Err(self.defect(format!(
                            "block {} does not name its pull counter residence {index}",
                            block.0
                        )));
                    }
                }
                ResidenceSource::RootInput(_) | ResidenceSource::RootResult(_) | ResidenceSource::KernelLocal { .. } => {}
            }
            let is_root = matches!(
                residence.source,
                ResidenceSource::RootInput(_) | ResidenceSource::RootResult(_)
            );
            let kernel_local = matches!(residence.lifetime, Lifetime::KernelLocal(_));
            let scopes: Vec<StorageScope> = match &residence.choice {
                ResidenceChoice::Fixed(scope) => vec![*scope],
                ResidenceChoice::SolverChoice { options, .. } => options.iter().copied().collect(),
            };
            for scope in scopes {
                match scope {
                    StorageScope::Abi if !is_root => {
                        return Err(self.defect(format!(
                            "residence {index} ({:?}) is ABI-placed but is not a root boundary leaf",
                            residence.source
                        )))
                    }
                    StorageScope::Workgroup | StorageScope::Participant if !kernel_local => {
                        return Err(self.defect(format!(
                            "residence {index} ({:?}) admits {scope:?} without a kernel-local lifetime",
                            residence.source
                        )))
                    }
                    StorageScope::Abi
                    | StorageScope::DeviceArena
                    | StorageScope::Workgroup
                    | StorageScope::Participant => {}
                }
            }
            let is_pull_counter = matches!(residence.source, ResidenceSource::PullCounter { .. });
            if !referenced.contains(&id) && !(kernel_local && local_owners.contains(&id)) && !is_pull_counter {
                return Err(self.defect(format!(
                    "residence {index} ({:?}) has no route and no kernel-local owner",
                    residence.source
                )));
            }
        }
        // Every dynamically pulling block has its counter, and no other
        // block names one.
        for (block, cut) in self.shape.blocks().entries() {
            let pulls = matches!(cut.participants.policy, ParticipantPolicy::DynamicPull { .. });
            match (pulls, blocks[block].pull_counter) {
                (true, PullCounter::Counter(_)) | (false, PullCounter::None) => {}
                (true, PullCounter::None) => {
                    return Err(self.defect(format!("block {} pulls dynamically without a counter", block.0)))
                }
                (false, PullCounter::Counter(_)) => {
                    return Err(self.defect(format!(
                        "block {} names a pull counter but does not pull dynamically",
                        block.0
                    )))
                }
            }
        }
        Ok(())
    }

    fn form(mut self) -> Result<Formed, CompilerDefect> {
        self.form_residences()?;
        self.form_routes()?;
        let blocks = self.form_interfaces()?;
        self.seal_check(&blocks)?;
        Ok(Formed {
            routes: RouteTable::seal(self.routes),
            residences: ResidenceGraph::seal(IdVec::new(self.residences), self.choice_vars),
            blocks,
            steps: IdVec::new(self.steps),
            scalar_slots: self.scalar_slots,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formation::strategy::{form as form_shape, universal_rules};
    use crate::ids::OccurrenceId;
    use crate::occurrence::OccurrenceForest;
    use crate::strategy::{MappingProposal, RuleQuery};
    use crate::target::TargetLimits;
    use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
    use seismic_lang::logical::{construct, EffectiveTargetIdentity, LogicalProgram};
    use seismic_lang::program::{compile, SourceFile};
    use seismic_lang::sir::Program;

    fn check(source: &str) -> Program {
        let files = vec![SourceFile {
            path: "test.seismic".to_string(),
            text: source.to_string(),
        }];
        compile(&files).unwrap_or_else(|diagnostics| {
            panic!(
                "the source checks: {}",
                diagnostics
                    .iter()
                    .map(|d| d.render())
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })
    }

    fn supports_all(_: &seismic_lang::sir::IntrinsicUse) -> Result<(), String> {
        Ok(())
    }

    fn logical(source: &str, entry: &str, shapes: &[(&str, u64)]) -> LogicalProgram {
        let program = check(source);
        let shapes = shapes
            .iter()
            .map(|(name, value)| (name.to_string(), ShapeBinding::Exact(*value)))
            .collect();
        let domain = SpecializationDomain::new(&program, entry, shapes, BTreeMap::new())
            .expect("the exact domain binds every entry parameter");
        let target = EffectiveTargetIdentity {
            backend: "cpu".to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
        };
        construct(&program, &target, &supports_all, &domain).expect("construction succeeds")
    }

    fn profile() -> EffectiveTargetProfile {
        EffectiveTargetProfile {
            backend: "cpu".to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
            toolchain_fingerprint: "test-toolchain".to_string(),
            effective_signatures: BTreeSet::new(),
            limits: TargetLimits {
                max_participants: 1024,
                max_workgroups_axis: [65535; 3],
                max_workgroup_bytes: 32768,
                max_explicit_private_bytes: 4096,
                max_direct_bindings: 31,
                max_device_bytes: 1 << 30,
                cooperative_grid: None,
            },
        }
    }

    fn proposals(
        facts: &OccurrenceFacts<'_>,
        profile: &EffectiveTargetProfile,
        occurrence: OccurrenceId,
        logical_alternative: u32,
    ) -> Vec<MappingProposal> {
        let query = RuleQuery {
            facts,
            profile,
            occurrence,
            logical_alternative,
        };
        universal_rules()
            .iter()
            .flat_map(|rule| rule.propose(&query))
            .collect()
    }

    /// Every routed strategy of every occurrence, alternative, and universal
    /// proposal of the program.
    fn routed_strategies(facts: &OccurrenceFacts<'_>) -> Vec<(OccurrenceId, RoutedStrategy)> {
        let profile = profile();
        let mut out = Vec::new();
        for (occurrence, record) in facts.occurrences() {
            for alternative in 0..record.alternatives.len() as u32 {
                for proposal in proposals(facts, &profile, occurrence, alternative) {
                    let shape = form_shape(facts, &profile, proposal)
                        .unwrap_or_else(|defect| panic!("{defect}"));
                    let routed = form(facts, &profile, shape).unwrap_or_else(|defect| panic!("{defect}"));
                    out.push((occurrence, routed));
                }
            }
        }
        assert!(!out.is_empty());
        out
    }

    fn key(occurrence: OccurrenceId, alternative: u32) -> OwnedGraphKey {
        OwnedGraphKey {
            occurrence,
            logical_alternative: alternative,
        }
    }

    /// The leaves the shape requires, computed as the certificate states:
    /// cross-edge leaves, block interface leaves, retained call boundary
    /// leaves, retained control leaves, and root boundary leaves.
    fn required_leaves(facts: &OccurrenceFacts<'_>, routed: &RoutedStrategy) -> BTreeSet<CanonicalLeafId> {
        let shape = routed.shape();
        let mut required = BTreeSet::new();
        for edge in shape.edges() {
            required.insert(edge.leaf);
        }
        for cut in shape.blocks().iter() {
            required.extend(cut.inputs.iter().copied());
            required.extend(cut.outputs.iter().copied());
        }
        let root = shape.root();
        let boundary = &facts.alternative(key(root.occurrence, root.logical_alternative)).boundary;
        for input in boundary.inputs.values() {
            let callee = match input {
                InstantiatedInput::Root { callee, .. } | InstantiatedInput::Call { callee, .. } => *callee,
            };
            required.extend(facts.leaves(facts.canonical_value(callee).unwrap()).iter().copied());
        }
        for result in boundary.results.values() {
            let callee = match result {
                InstantiatedResult::Root { callee } | InstantiatedResult::Call { callee, .. } => *callee,
            };
            required.extend(facts.leaves(facts.canonical_value(callee).unwrap()).iter().copied());
        }
        fn walk(facts: &OccurrenceFacts<'_>, schedule: &ShapeSchedule, required: &mut BTreeSet<CanonicalLeafId>) {
            let mut values = Vec::new();
            for step in &schedule.steps {
                match step {
                    ShapeStep::Launch(_) | ShapeStep::Guard { .. } | ShapeStep::PullCounterReset { .. } => {}
                    ShapeStep::Call { node, .. } => {
                        let LogicalNodeKind::Call(call) = &facts.node(node).unwrap().kind else {
                            panic!("a call step is a call node");
                        };
                        for input in call.boundary.inputs.values() {
                            let value = match input {
                                CallInput::Value(value)
                                | CallInput::Tensor { value, .. }
                                | CallInput::Computed { value, .. } => *value,
                            };
                            values.push(
                                facts
                                    .canonical_value(OwnedValueRef { graph: node.graph, value })
                                    .unwrap(),
                            );
                        }
                        for value in call.boundary.results.values() {
                            values.push(
                                facts
                                    .canonical_value(OwnedValueRef {
                                        graph: node.graph,
                                        value: *value,
                                    })
                                    .unwrap(),
                            );
                        }
                    }
                    ShapeStep::If { condition, then_schedule, else_schedule, joins, .. } => {
                        values.push(*condition);
                        for join in joins {
                            if let JoinEdge::Value { then_value, else_value, joined } = join {
                                values.extend([*then_value, *else_value, *joined]);
                            }
                        }
                        walk(facts, then_schedule, required);
                        walk(facts, else_schedule, required);
                    }
                    ShapeStep::Repeat { start, end, binder, body, carries, .. } => {
                        values.extend([*start, *end, *binder]);
                        for carry in carries {
                            if let CarryEdge::Value { initial, parameter, update, result } = carry {
                                values.extend([*initial, *parameter, *update, *result]);
                            }
                        }
                        walk(facts, body, required);
                    }
                }
            }
            for value in values {
                required.extend(facts.leaves(value).iter().copied());
            }
        }
        walk(facts, shape.schedule(), &mut required);
        required
    }

    fn residence_of_input(routed: &RoutedStrategy, param: u32) -> ResidenceId {
        let leaf = BoundaryLeaf::Input {
            param,
            leaf: seismic_lang::types::ValuePath::default(),
        };
        routed
            .residences()
            .iter()
            .find(|(_, residence)| residence.source == ResidenceSource::RootInput(leaf.clone()))
            .map(|(id, _)| id)
            .unwrap_or_else(|| panic!("input parameter {param} has an ABI residence"))
    }

    fn result_route<'r>(facts: &OccurrenceFacts<'_>, routed: &'r RoutedStrategy) -> &'r ValueRoute {
        let root = routed.shape().root();
        let boundary = &facts.alternative(key(root.occurrence, root.logical_alternative)).boundary;
        let (_, result) = boundary.results.iter().next().expect("one result leaf");
        let callee = match result {
            InstantiatedResult::Root { callee } | InstantiatedResult::Call { callee, .. } => *callee,
        };
        let [leaf] = facts.leaves(facts.canonical_value(callee).unwrap()) else {
            panic!("the result is one leaf");
        };
        routed.routes().route(*leaf)
    }

    const KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n";

    #[test]
    fn root_leaves_route_to_abi_residences_and_the_domain_is_exact() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        for (_, routed) in routed_strategies(facts) {
            // Every root tensor input has one ABI residence, live for the
            // whole strategy, with no solver choice.
            for param in 0..3 {
                let residence = routed.residences().residence(residence_of_input(&routed, param));
                assert_eq!(residence.choice, ResidenceChoice::Fixed(StorageScope::Abi));
                assert_eq!(residence.lifetime, Lifetime::Whole);
                assert_eq!(residence.replication, Replication::Once);
                assert_eq!(residence.planes.len(), 1);
                assert_eq!(
                    residence.planes.first().bytes,
                    Sym::constant(4 * 8 * 4),
                    "a dense f32 [4, 8] plane is 128 bytes"
                );
            }
            // Every root boundary input leaf routes to its residence with
            // the identity transform.
            let root = routed.shape().root();
            let boundary = &facts.alternative(key(root.occurrence, root.logical_alternative)).boundary;
            for (leaf, input) in &boundary.inputs {
                let callee = match input {
                    InstantiatedInput::Root { callee, .. } | InstantiatedInput::Call { callee, .. } => *callee,
                };
                let BoundaryLeaf::Input { param, .. } = leaf else {
                    panic!("an input leaf");
                };
                for canonical_leaf in facts.leaves(facts.canonical_value(callee).unwrap()) {
                    let ValueRoute::Tensor(route) = routed.routes().route(*canonical_leaf) else {
                        panic!("a tensor input routes as a tensor");
                    };
                    assert_eq!(route.residence, residence_of_input(&routed, *param));
                    assert_eq!(route.transform, ViewTransformTemplate::identity());
                }
            }
            // No ABI residence other than the three inputs: the result is a
            // pass-through of `result` and names that input's residence.
            let abi = routed
                .residences()
                .iter()
                .filter(|(_, residence)| residence.choice == ResidenceChoice::Fixed(StorageScope::Abi))
                .count();
            assert_eq!(abi, 3);
            let ValueRoute::Tensor(result) = result_route(facts, &routed) else {
                panic!("the tensor result routes as a tensor");
            };
            assert_eq!(result.residence, residence_of_input(&routed, 2));
            assert_eq!(result.transform, ViewTransformTemplate::identity());
            assert_eq!(result.access, Access::Exclusive);
            // Aliasing is route equality: no residence is a `RootResult`.
            assert!(routed
                .residences()
                .iter()
                .all(|(_, residence)| !matches!(residence.source, ResidenceSource::RootResult(_))));
            // The route domain is exactly the required leaf set.
            let required = required_leaves(facts, &routed);
            let routed_leaves: BTreeSet<CanonicalLeafId> = routed.routes().iter().map(|(leaf, _)| leaf).collect();
            assert_eq!(routed_leaves, required);
            // Every block interface leaf is routed, and every routed leaf a
            // block produces is one of its outputs.
            for (block, cut) in routed.shape().blocks().entries() {
                let interface = &routed.blocks()[block].interface;
                assert_eq!(interface.inputs.len(), cut.inputs.len());
                assert_eq!(interface.outputs.len(), cut.outputs.len());
                for decl in interface.inputs.iter() {
                    assert!(cut.inputs.contains(&decl.leaf));
                    assert_eq!(routed.routes().route(decl.leaf), &decl.route);
                }
                for decl in interface.outputs.iter() {
                    assert!(cut.outputs.contains(&decl.leaf));
                    assert_eq!(routed.routes().route(decl.leaf), &decl.route);
                }
                assert_eq!(interface.axes.len(), cut.participants.independent_axes.len());
                for (index, axis) in interface.axes.iter().enumerate() {
                    assert_eq!(axis.id, KernelAxisId(index as u32));
                    assert_eq!(axis.binder, Some(cut.participants.independent_axes[index].binder));
                }
                assert!(interface.locals.is_empty());
            }
            // Dense per-strategy allocations.
            assert_eq!(routed.residences().choice_vars(), 0);
            assert_eq!(routed.steps().len(), routed.shape().schedule().steps.len());
        }
    }

    #[test]
    fn the_parallel_block_reads_its_inputs_and_writes_the_result_through_routes() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let mut saw_linear_block = false;
        for (occurrence, routed) in routed_strategies(facts) {
            if occurrence != OccurrenceId(1) {
                // The entry retains the call: no blocks, one call step.
                assert!(routed.blocks().is_empty());
                assert_eq!(routed.steps().len(), 1);
                assert!(matches!(routed.steps()[StepId(0)].anchor, StepAnchor::Call(_)));
                continue;
            }
            assert!(!routed.blocks().is_empty());
            for (block, cut) in routed.shape().blocks().entries() {
                let interface = &routed.blocks()[block].interface;
                // Inputs are the borrowed operands (and anything an
                // earlier block produced); every tensor input names an ABI
                // input residence.
                for decl in interface.inputs.iter() {
                    if let ValueRoute::Tensor(route) = &decl.route {
                        let residence = routed.residences().residence(route.residence);
                        assert!(matches!(residence.source, ResidenceSource::RootInput(_)));
                    }
                }
                if let ParticipantPolicy::Linear { .. } = cut.participants.policy {
                    saw_linear_block = true;
                    assert_eq!(interface.axes.len(), 1);
                    assert_eq!(interface.iteration.extents, vec![ExtentExpr::Static(4)]);
                    assert!(!interface.iteration.serialized);
                } else {
                    assert!(interface.iteration.serialized);
                }
            }
        }
        assert!(saw_linear_block, "the linear peer forms the parallel loop as one block");
    }

    const SCALE: &str = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return f32(x)\n\nfn main[N](x: &tensor[N] f32) -> f32:\n    let y = scale(x)\n    return reduce(y, 0, max)\n";

    #[test]
    fn a_computed_call_result_gets_one_spill_and_a_computed_root_result_is_published() {
        let logical = logical(SCALE, "main", &[("N", 16)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let scale_result_value = {
            let boundary = &facts.alternative(key(OccurrenceId(1), 0)).boundary;
            let (_, result) = boundary.results.iter().next().expect("scale returns one leaf");
            let InstantiatedResult::Call { callee, .. } = result else {
                panic!("scale is called");
            };
            facts.canonical_value(*callee).unwrap()
        };
        assert!(matches!(
            facts.value_kind(scale_result_value),
            GraphValueKind::Tensor { source: TensorSource::Computed, .. }
        ));
        let [result_leaf] = facts.leaves(scale_result_value) else {
            panic!("one leaf");
        };
        for (occurrence, routed) in routed_strategies(facts) {
            let spills: Vec<(ResidenceId, &Residence)> = routed
                .residences()
                .iter()
                .filter(|(_, residence)| matches!(residence.source, ResidenceSource::ComputedSpill(_)))
                .collect();
            if occurrence == OccurrenceId(0) {
                // The caller: the call result crosses from the call step to
                // the reduce block through exactly one spill residence.
                let [(spill, residence)] = spills.as_slice() else {
                    panic!("the caller holds one spill, got {spills:?}");
                };
                assert_eq!(residence.source, ResidenceSource::ComputedSpill(scale_result_value));
                assert_eq!(residence.choice, ResidenceChoice::Fixed(StorageScope::DeviceArena));
                assert_eq!(residence.planes.first().bytes, Sym::constant(16 * 4));
                let ValueRoute::Tensor(route) = routed.routes().route(*result_leaf) else {
                    panic!("the call result routes as a tensor");
                };
                assert_eq!(route.residence, *spill);
                assert_eq!(route.access, Access::Exclusive);
                assert_eq!(route.transform, ViewTransformTemplate::identity());
                // Live from the call step to the launch that consumes it.
                let call = routed
                    .steps()
                    .entries()
                    .find(|(_, step)| matches!(step.anchor, StepAnchor::Call(_)))
                    .map(|(id, _)| id)
                    .expect("one retained call");
                let launch = routed
                    .steps()
                    .entries()
                    .find(|(_, step)| matches!(step.anchor, StepAnchor::Launch(_)))
                    .map(|(id, _)| id)
                    .expect("one launch");
                assert_eq!(residence.lifetime, Lifetime::Steps { first: call, last: launch });
                // The scalar result of `main` is a result field the block publishes.
                let ValueRoute::Scalar(ScalarRoute::ResultField { leaf, endpoint: None }) = result_route(facts, &routed) else {
                    panic!("the scalar result routes to a result field");
                };
                assert_eq!(
                    *leaf,
                    BoundaryLeaf::Result {
                        leaf: seismic_lang::types::ValuePath::default()
                    }
                );
                let blocks: Vec<&RoutedBlock> = routed.blocks().iter().collect();
                let [block] = blocks.as_slice() else {
                    panic!("one block");
                };
                assert!(block
                    .interface
                    .outputs
                    .iter()
                    .any(|decl| matches!(decl.route, ValueRoute::Scalar(ScalarRoute::ResultField { .. }))));
                assert!(block.interface.inputs.iter().any(|decl| decl.leaf == *result_leaf));
                assert_eq!(routed.scalar_slots(), 0);
            } else {
                // The callee: no spill; its computed root result is one ABI
                // result residence that the producing block publishes into.
                assert!(spills.is_empty());
                let results: Vec<(ResidenceId, &Residence)> = routed
                    .residences()
                    .iter()
                    .filter(|(_, residence)| matches!(residence.source, ResidenceSource::RootResult(_)))
                    .collect();
                let [(result, residence)] = results.as_slice() else {
                    panic!("one result residence");
                };
                assert_eq!(residence.choice, ResidenceChoice::Fixed(StorageScope::Abi));
                assert_eq!(residence.lifetime, Lifetime::Whole);
                let ValueRoute::Tensor(route) = routed.routes().route(*result_leaf) else {
                    panic!("the computed result routes as a tensor");
                };
                assert_eq!(route.residence, *result);
                assert!(routed.blocks().iter().any(|block| {
                    block.interface.outputs.iter().any(|decl| decl.leaf == *result_leaf)
                }));
            }
            let required = required_leaves(facts, &routed);
            let routed_leaves: BTreeSet<CanonicalLeafId> = routed.routes().iter().map(|(leaf, _)| leaf).collect();
            assert_eq!(routed_leaves, required);
        }
    }
}
