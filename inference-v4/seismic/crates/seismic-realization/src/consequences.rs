//! Numerical and hard-resource consequences (package M1).
//!
//! `ConsequenceFormer::derive` is an exhaustive fold over sealed ops,
//! residences, and schedule steps. Mapping rules and backends supply typed
//! coefficients and facts; they cannot construct the final consequence set.
//! The exact expressions used by the solver are the ones retained in the
//! physical artifact and checked during native assembly: one identity.
//!
//! # Resource-expression identity
//!
//! Every exported `Sym` is a polynomial over the strategy's tuning
//! parameters (named by their `TuningDeclaration`) plus two reserved atom
//! families for values the sealed blocks reference symbolically:
//!
//! - `@runtime<N>` — the runtime extent `RuntimeExtentId(N)`. Feasibility
//!   and resources bind it to the extent's checked capacity; cost may bind
//!   it to the workload's expected extent.
//! - `@leaf<L>` — the scalar canonical leaf `CanonicalLeafId(L)` (a kernel
//!   input, e.g. a serial loop's runtime range endpoint). P1 binds it from
//!   the leaf's route: a root ABI scalar's retained invocation interval, or
//!   a guarded executor scalar's proved domain.
//!
//! Capacity, never expectation, bounds every resource expression: work and
//! arena totals use `LinearIterationMap::total_symbol` (the checked capacity
//! product) and `@runtime` atoms P1 bounds by capacity. Only `cost` uses
//! `LinearIterationMap::cost_symbol` (the expected total). Numerical-evidence
//! records witness exactly this vocabulary: `numerics::NumericalEvidence`
//! binds the tuning-qualified parameter names and the `@runtime`/`@leaf`
//! atoms its qualification was measured under. Device-arena
//! accounting is per residence (`SolverConstraint::ScopedBytes` over the
//! exact plane expressions); P1's arena packing applies the retained
//! lifetimes, so simultaneously live residences sum and mutually exclusive
//! branch-side residences are never forced to disjoint offsets. The
//! strategy-level `DeviceArenaBytes` range is the conservative capacity
//! envelope over every placement-possible residence: it never undercounts
//! and is identical, term by term, to the packing input. Dynamic-pull
//! contention is unmodelled: the per-pull term prices exactly one counter
//! round trip per claimed coordinate and nothing else.

use crate::failure::{CompilerDefect, Package};
use crate::ids::{BlockId, ChoiceVarId, KernelSsaId, PlanParamId, ResidenceId, StepId};
use crate::kernel::{
    AtomicMode, ClosedKernelBlock, ConstantValue, CoreKernelOp, CostUnit, ExecutableDialect,
    KernelOp, KernelValueRef,
};
use crate::numerics::{self, CountExpr, NumericalTransfer, ReductionTopology};
use crate::residence::{
    ClosedKernelInterface, Lifetime, PullCounter, Replication, ResidenceChoice, RoutedStrategy,
    StepAnchor, StorageScope,
};
use crate::routes::{ScalarRoute, ValueRoute};
use crate::strategy::{
    AlgorithmChoice, CostModel, NumericalChoice, ParticipantPolicy, ShapeSchedule, ShapeStep,
};
use crate::target::TargetLimits;
use seismic_lang::intrinsics::{AtomicOp, IntrinsicId, MathOp, ReduceOp};
use seismic_lang::logical::IdVec;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::types::{DType, ExtentExpr, RuntimeExtentId};
use std::collections::{BTreeMap, BTreeSet};

/// One hard resource scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceScope {
    DeviceArenaBytes,
    KernelLocalBytes,
    ParticipantPrivateBytes,
    WorkgroupBytes,
    ResultBytes,
    StatusBytes,
    ScalarSlotBytes,
    Participants,
    DirectBindings,
    StaticCodeUnits,
}

/// One resource declaration attached to a block or residence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceDeclaration {
    pub scope: ResourceScope,
    pub owner: ResourceOwner,
    /// Bytes/count at capacity, over tuning parameters.
    pub amount: Sym,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceOwner {
    Block(BlockId),
    Residence(ResidenceId),
    Strategy,
}

/// One per-block launch fact set exported to the solver and retained
/// through the physical seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockFacts {
    pub block: BlockId,
    pub participants: Sym,
    pub work_items: Sym,
    pub workgroup_bytes: Sym,
    pub private_bytes_per_participant: Sym,
    pub direct_bindings: u32,
    pub static_code_units: u64,
    pub required_subgroup_width: Option<u32>,
    pub barriers: u32,
    pub capabilities: BTreeSet<IntrinsicId>,
}

/// One solver constraint exported by a strategy. `PartialEq` only: the
/// `Numerical` variant carries float error bounds, which compare
/// bit-stably but are not equivalence classes.
#[derive(Clone, Debug, PartialEq)]
pub enum SolverConstraint {
    /// `expr` within `[lower, upper]` when this strategy is selected.
    Range { expr: Sym, lower: u64, upper: u64 },
    /// A residence scope choice restricted to these options.
    ScopeOptions { var: ChoiceVarId, options: Vec<StorageScope> },
    /// A residence contributes `bytes` to `scope` when placed there.
    ScopedBytes { var: Option<ChoiceVarId>, scope: StorageScope, bytes: Sym },
    /// Numerical acceptance: the strategy is admissible only when the policy
    /// admits this transfer (or evidence qualifies it).
    Numerical(NumericalTransfer),
}

/// The complete consequences of one strategy.
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyConsequences {
    pub numerical: NumericalTransfer,
    pub cost: Sym,
    pub resources: Vec<ResourceDeclaration>,
    pub blocks: IdVec<BlockId, BlockFacts>,
    pub constraints: Vec<SolverConstraint>,
    pub required_capabilities: BTreeSet<IntrinsicId>,
    pub tuning: IdVec<PlanParamId, crate::strategy::TuningDeclaration>,
}

/// A strategy carrying the S1/D1/K1 seals plus its consequences: the only
/// admissible input to `form_plan_space`'s space assembly.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedStrategy<D: ExecutableDialect> {
    routed: RoutedStrategy,
    kernels: IdVec<BlockId, ClosedKernelBlock<D::Intrinsic>>,
    consequences: StrategyConsequences,
}

impl<D: ExecutableDialect> ClosedStrategy<D> {
    pub(crate) fn seal(
        routed: RoutedStrategy,
        kernels: IdVec<BlockId, ClosedKernelBlock<D::Intrinsic>>,
        consequences: StrategyConsequences,
    ) -> ClosedStrategy<D> {
        ClosedStrategy {
            routed,
            kernels,
            consequences,
        }
    }

    pub fn routed(&self) -> &RoutedStrategy {
        &self.routed
    }
    pub fn kernels(&self) -> &IdVec<BlockId, ClosedKernelBlock<D::Intrinsic>> {
        &self.kernels
    }
    pub fn consequences(&self) -> &StrategyConsequences {
        &self.consequences
    }
}

/// The sole constructor of `StrategyConsequences` (package M1).
pub struct ConsequenceFormer;

impl ConsequenceFormer {
    pub fn derive<D: ExecutableDialect>(
        routed: &RoutedStrategy,
        kernels: &IdVec<BlockId, ClosedKernelBlock<D::Intrinsic>>,
        cost_model: &dyn crate::strategy::CostModel,
        limits: &crate::target::TargetLimits,
    ) -> Result<StrategyConsequences, crate::failure::CompilerDefect> {
        let shape = routed.shape();
        if kernels.len() != routed.blocks().len()
            || routed.blocks().ids().any(|id| kernels.get(id).is_none())
        {
            return Err(CompilerDefect::new(
                Package::P1,
                "the sealed kernel set does not cover the routed strategy's blocks exactly",
            ));
        }
        let launch_steps = launch_steps(routed)?;
        let mut constraints = Vec::new();
        let mut resources = Vec::new();
        let (arena_envelope, locals) =
            residence_pass(routed, &launch_steps, &mut constraints, &mut resources)?;
        let multipliers = schedule_multipliers(shape.schedule())?;
        if multipliers.len() != routed.blocks().len() {
            return Err(CompilerDefect::new(
                Package::D1,
                "the structured schedule launches fewer blocks than the strategy holds",
            ));
        }

        let mut block_cost: BTreeMap<BlockId, Sym> = BTreeMap::new();
        let mut blocks: Vec<(BlockId, BlockFacts)> = Vec::with_capacity(routed.blocks().len());
        let mut numerical = NumericalTransfer::Exact;
        let mut required = shape.required_intrinsics().clone();
        let mut status_fields = 0u64;
        for (block, _) in routed.blocks().entries() {
            let (mut facts, per_visit, transfer) =
                analyze_block::<D>(routed, kernels, block, cost_model, limits, &locals, &launch_steps, &mut constraints, &mut resources)?;
            let multiplier = multipliers.get(&block).ok_or_else(|| {
                CompilerDefect::new(
                    Package::D1,
                    format!("block {} is never launched by the schedule", block.0),
                )
            })?;
            // The launch executes once per visit of every enclosing repeat.
            facts.work_items = facts.work_items.mul(multiplier);
            numerical = numerics::compose(&transfer, &numerical);
            required.extend(facts.capabilities.iter().cloned());
            status_fields += kernels[block].status_fields().len() as u64;
            block_cost.insert(block, per_visit);
            blocks.push((block, facts));
        }

        let cost = schedule_cost(shape.schedule(), &Sym::constant(1), &block_cost, cost_model)?;

        // Strategy-level capacity declarations. Slot/status words are i32/u32
        // ABI words; a result field is one u64 word (the widest result scalar
        // block any backend lays out). These are declarations, not target
        // limits; the seal and assembly validate the exact dtypes.
        let slot_bytes = scale_word(u64::from(routed.scalar_slots()), 4)?;
        let result_fields = routed
            .routes()
            .iter()
            .filter(|(_, route)| {
                matches!(route, ValueRoute::Scalar(ScalarRoute::ResultField { .. }))
            })
            .count() as u64;
        let result_bytes = scale_word(result_fields, 8)?;
        let status_bytes = scale_word(status_fields, 4)?;
        resources.push(ResourceDeclaration {
            scope: ResourceScope::ScalarSlotBytes,
            owner: ResourceOwner::Strategy,
            amount: slot_bytes,
        });
        resources.push(ResourceDeclaration {
            scope: ResourceScope::ResultBytes,
            owner: ResourceOwner::Strategy,
            amount: result_bytes,
        });
        resources.push(ResourceDeclaration {
            scope: ResourceScope::StatusBytes,
            owner: ResourceOwner::Strategy,
            amount: status_bytes,
        });
        resources.push(ResourceDeclaration {
            scope: ResourceScope::DeviceArenaBytes,
            owner: ResourceOwner::Strategy,
            amount: arena_envelope.clone(),
        });
        constraints.push(SolverConstraint::Range {
            expr: arena_envelope,
            lower: 0,
            upper: limits.max_device_bytes,
        });
        constraints.push(SolverConstraint::Numerical(numerical.clone()));

        Ok(StrategyConsequences {
            numerical,
            cost,
            resources,
            blocks: IdVec::from_iter(blocks),
            constraints,
            required_capabilities: required,
            tuning: shape.tuning().clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Private derivation
// ---------------------------------------------------------------------------

fn m1(invariant: impl Into<String>) -> CompilerDefect {
    CompilerDefect::new(Package::M1, invariant)
}

fn at(package: Package, invariant: impl Into<String>) -> CompilerDefect {
    CompilerDefect::new(package, invariant)
}

/// The planning symbol of one extent: static extents exact, runtime extents
/// the `@runtime` atom P1 binds (capacity for resources, expected for cost).
fn extent_bound_symbol(extent: &ExtentExpr) -> Result<Sym, CompilerDefect> {
    match extent {
        ExtentExpr::Static(n) => i64::try_from(*n)
            .map(Sym::constant)
            .map_err(|_| m1(format!("static extent {n} exceeds the size domain"))),
        ExtentExpr::Runtime(id) => Ok(Sym::param(&format!("@runtime{}", id.0))),
        ExtentExpr::Sym(sym) => Err(at(
            Package::W1,
            format!("symbolic extent `{sym}` survived specialization"),
        )),
    }
}

/// One plane's bytes padded up to its alignment: every plane occupies a
/// whole number of its alignment units inside the residence.
fn aligned_bytes(bytes: &Sym, alignment: u64) -> Result<Sym, CompilerDefect> {
    let alignment = alignment.max(1);
    let unit = i64::try_from(alignment)
        .map_err(|_| m1("plane alignment exceeds the size domain"))?;
    Ok(bytes
        .add(&Sym::constant(unit - 1))
        .quot(&Sym::constant(unit))
        .scale(unit))
}

fn scale_word(count: u64, bytes: u64) -> Result<Sym, CompilerDefect> {
    let total = count
        .checked_mul(bytes)
        .and_then(|n| i64::try_from(n).ok())
        .ok_or_else(|| m1("word-count byte total exceeds the size domain"))?;
    Ok(Sym::constant(total))
}

/// How a block's serial `Repeat` bound is defined: a constant, or a retained
/// runtime extent. Every other definition is a K1 defect at use.
enum BoundDef {
    Const(i64),
    Runtime(RuntimeExtentId),
}

/// The per-block local-byte envelopes filled by the residence pass.
#[derive(Default, Clone)]
struct LocalBytes {
    workgroup: Option<Sym>,
    private: Option<Sym>,
    kernel_local: Option<Sym>,
}

impl LocalBytes {
    fn workgroup(&self) -> Sym {
        self.workgroup.clone().unwrap_or_else(|| Sym::constant(0))
    }
    fn private(&self) -> Sym {
        self.private.clone().unwrap_or_else(|| Sym::constant(0))
    }
    fn kernel_local(&self) -> Sym {
        self.kernel_local.clone().unwrap_or_else(|| Sym::constant(0))
    }
}

/// The launch step of every block, from the pre-order step table.
fn launch_steps(routed: &RoutedStrategy) -> Result<BTreeMap<BlockId, StepId>, CompilerDefect> {
    let mut out = BTreeMap::new();
    for (id, step) in routed.steps().entries() {
        if let StepAnchor::Launch(block) = &step.anchor {
            if out.insert(*block, id).is_some() {
                return Err(at(
                    Package::D1,
                    format!("block {} appears in two launch steps", block.0),
                ));
            }
        }
    }
    Ok(out)
}

/// The schedule multiplier of every block: the product of the bounds of the
/// retained repeats enclosing its launch.
fn schedule_multipliers(
    schedule: &ShapeSchedule,
) -> Result<BTreeMap<BlockId, Sym>, CompilerDefect> {
    let mut out = BTreeMap::new();
    walk_multipliers(schedule, &Sym::constant(1), &mut out)?;
    Ok(out)
}

fn walk_multipliers(
    schedule: &ShapeSchedule,
    multiplier: &Sym,
    out: &mut BTreeMap<BlockId, Sym>,
) -> Result<(), CompilerDefect> {
    for step in &schedule.steps {
        match step {
            ShapeStep::Launch(block) => {
                if out.insert(*block, multiplier.clone()).is_some() {
                    return Err(at(
                        Package::D1,
                        format!("block {} is launched by two schedule steps", block.0),
                    ));
                }
            }
            ShapeStep::Guard { .. }
            | ShapeStep::PullCounterReset { .. }
            | ShapeStep::Call { .. } => {}
            ShapeStep::If {
                then_schedule,
                else_schedule,
                ..
            } => {
                walk_multipliers(then_schedule, multiplier, out)?;
                walk_multipliers(else_schedule, multiplier, out)?;
            }
            ShapeStep::Repeat { bound, body, .. } => {
                let inner = multiplier.mul(&extent_bound_symbol(bound)?);
                walk_multipliers(body, &inner, out)?;
            }
        }
    }
    Ok(())
}

/// The footprint one residence occupies in `scope` when placed there:
/// workgroup scope is per workgroup, participant scope per participant, the
/// device arena holds every replica. A per-participant or per-subgroup
/// replica set inside one workgroup is bounded by the block's total
/// participants (every replica needs at least one participant); P1's launch
/// geometry tightens that envelope with the per-workgroup count.
fn scope_footprint(
    scope: StorageScope,
    replication: Replication,
    replica_bytes: &Sym,
    replica_count: &Sym,
) -> Result<Sym, CompilerDefect> {
    match scope {
        StorageScope::Abi | StorageScope::DeviceArena => Ok(replica_bytes.mul(replica_count)),
        StorageScope::Workgroup => match replication {
            Replication::Once | Replication::PerWorkgroup => Ok(replica_bytes.clone()),
            Replication::PerParticipant | Replication::PerSubgroup => {
                Ok(replica_bytes.mul(replica_count))
            }
        },
        StorageScope::Participant => match replication {
            Replication::Once | Replication::PerParticipant => Ok(replica_bytes.clone()),
            Replication::PerWorkgroup | Replication::PerSubgroup => Err(at(
                Package::D1,
                "a participant-scoped residence is replicated per workgroup or subgroup",
            )),
        },
    }
}

/// One pass over the residence graph: scope choices, scoped byte
/// constraints, per-residence declarations, the device-arena capacity
/// envelope, and the per-block local-byte envelopes.
fn residence_pass(
    routed: &RoutedStrategy,
    launch_steps: &BTreeMap<BlockId, StepId>,
    constraints: &mut Vec<SolverConstraint>,
    resources: &mut Vec<ResourceDeclaration>,
) -> Result<(Sym, BTreeMap<BlockId, LocalBytes>), CompilerDefect> {
    let step_count = routed.steps().len() as u32;
    let mut arena_envelope = Sym::constant(0);
    let mut locals: BTreeMap<BlockId, LocalBytes> = BTreeMap::new();
    for (rid, residence) in routed.residences().iter() {
        match &residence.lifetime {
            Lifetime::Whole => {}
            Lifetime::Steps { first, last } => {
                if first.0 > last.0 || last.0 >= step_count {
                    return Err(at(
                        Package::D1,
                        format!("residence {} has a lifetime outside the schedule", rid.0),
                    ));
                }
            }
            Lifetime::KernelLocal(block) => {
                if !launch_steps.contains_key(block) {
                    return Err(at(
                        Package::D1,
                        format!(
                            "kernel-local residence {} belongs to block {} which is never launched",
                            rid.0, block.0
                        ),
                    ));
                }
            }
        }
        let mut replica_bytes = Sym::constant(0);
        for plane in residence.planes.iter() {
            replica_bytes = replica_bytes.add(&aligned_bytes(&plane.bytes, plane.alignment)?);
        }
        let replica_count = match residence.replication {
            Replication::Once => Sym::constant(1),
            Replication::PerParticipant | Replication::PerWorkgroup | Replication::PerSubgroup => {
                let Lifetime::KernelLocal(block) = &residence.lifetime else {
                    return Err(at(
                        Package::D1,
                        format!("residence {} is replicated but not kernel-local", rid.0),
                    ));
                };
                routed.blocks()[*block]
                    .interface
                    .iteration
                    .participants
                    .clone()
            }
        };
        let var = match &residence.choice {
            ResidenceChoice::Fixed(_) => None,
            ResidenceChoice::SolverChoice { var, .. } => Some(*var),
        };
        let scopes: Vec<StorageScope> = match &residence.choice {
            ResidenceChoice::Fixed(scope) => vec![*scope],
            ResidenceChoice::SolverChoice { options, var } => {
                constraints.push(SolverConstraint::ScopeOptions {
                    var: *var,
                    options: options.iter().copied().collect(),
                });
                options.iter().copied().collect()
            }
        };
        for scope in scopes {
            let resource_scope = match scope {
                // Caller memory: ABI buffer byte requirements are expressions
                // of actual validated extents owned by the invocation
                // contract (P1), not capacity resources of the target.
                StorageScope::Abi => continue,
                StorageScope::DeviceArena => ResourceScope::DeviceArenaBytes,
                StorageScope::Workgroup => ResourceScope::WorkgroupBytes,
                StorageScope::Participant => ResourceScope::ParticipantPrivateBytes,
            };
            if matches!(scope, StorageScope::Workgroup | StorageScope::Participant)
                && !matches!(residence.lifetime, Lifetime::KernelLocal(_))
            {
                return Err(at(
                    Package::D1,
                    format!(
                        "residence {} is workgroup/participant-scoped without a kernel-local lifetime",
                        rid.0
                    ),
                ));
            }
            let amount = scope_footprint(
                scope,
                residence.replication,
                &replica_bytes,
                &replica_count,
            )?;
            if scope == StorageScope::DeviceArena {
                arena_envelope = arena_envelope.add(&amount);
            }
            constraints.push(SolverConstraint::ScopedBytes {
                var,
                scope,
                bytes: amount.clone(),
            });
            if let Lifetime::KernelLocal(block) = &residence.lifetime {
                let entry = locals.entry(*block).or_default();
                match scope {
                    StorageScope::Workgroup => {
                        entry.workgroup = Some(entry.workgroup().add(&amount));
                    }
                    StorageScope::Participant => {
                        entry.private = Some(entry.private().add(&amount));
                    }
                    StorageScope::DeviceArena | StorageScope::Abi => {}
                }
            }
            resources.push(ResourceDeclaration {
                scope: resource_scope,
                owner: ResourceOwner::Residence(rid),
                amount,
            });
        }
        if let Lifetime::KernelLocal(block) = &residence.lifetime {
            let entry = locals.entry(*block).or_default();
            let total = replica_bytes.mul(&replica_count);
            entry.kernel_local = Some(entry.kernel_local().add(&total));
        }
    }
    Ok((arena_envelope, locals))
}

/// The per-visit cost of one block and the numerical transfer of its ops.
#[allow(clippy::too_many_arguments)]
fn analyze_block<D: ExecutableDialect>(
    routed: &RoutedStrategy,
    kernels: &IdVec<BlockId, ClosedKernelBlock<D::Intrinsic>>,
    block: BlockId,
    cost_model: &dyn CostModel,
    limits: &TargetLimits,
    locals: &BTreeMap<BlockId, LocalBytes>,
    launch_steps: &BTreeMap<BlockId, StepId>,
    constraints: &mut Vec<SolverConstraint>,
    resources: &mut Vec<ResourceDeclaration>,
) -> Result<(BlockFacts, Sym, NumericalTransfer), CompilerDefect> {
    let routed_block = &routed.blocks()[block];
    let kernel = &kernels[block];
    let cut = &routed.shape().blocks()[block];
    let interface = kernel.interface();
    let iteration = &interface.iteration;
    let participants = iteration.participants.clone();
    let local_bytes = locals.get(&block).cloned().unwrap_or_default();

    match &cut.participants.policy {
        ParticipantPolicy::Serial
        | ParticipantPolicy::Linear { .. }
        | ParticipantPolicy::Cooperative { .. } => {
            if !matches!(routed_block.pull_counter, PullCounter::None) {
                return Err(at(
                    Package::D1,
                    format!("block {} carries a pull counter without dynamic-pull participants", block.0),
                ));
            }
        }
        ParticipantPolicy::DynamicPull { .. } => {
            let PullCounter::Counter(_) = routed_block.pull_counter else {
                return Err(at(
                    Package::D1,
                    format!("dynamic-pull block {} has no pull-counter residence", block.0),
                ));
            };
            let launch = launch_steps.get(&block).ok_or_else(|| {
                at(Package::D1, format!("dynamic-pull block {} is never launched", block.0))
            })?;
            if launch.0 == 0 {
                return Err(at(
                    Package::D1,
                    format!(
                        "dynamic-pull block {} has no pull-counter reset step immediately before its launch",
                        block.0
                    ),
                ));
            }
            let previous = &routed.steps()[StepId(launch.0 - 1)].anchor;
            if !matches!(previous, StepAnchor::PullCounterReset(reset) if *reset == block) {
                return Err(at(
                    Package::D1,
                    format!(
                        "dynamic-pull block {} has no pull-counter reset step immediately before its launch",
                        block.0
                    ),
                ));
            }
        }
        ParticipantPolicy::GridCooperative { .. } => {
            if !matches!(routed_block.pull_counter, PullCounter::None) {
                return Err(at(
                    Package::D1,
                    format!("grid-cooperative block {} carries a pull counter", block.0),
                ));
            }
            let Some(cooperative_grid) = &limits.cooperative_grid else {
                return Err(at(
                    Package::S1,
                    format!(
                        "grid-cooperative block {} was admitted on a target without the cooperative-grid facility",
                        block.0
                    ),
                ));
            };
            constraints.push(SolverConstraint::Range {
                expr: participants.clone(),
                lower: 1,
                upper: cooperative_grid.max_resident_participants,
            });
        }
    }

    // Serial `Repeat` bounds resolve through the block's own definitions.
    let mut defs: BTreeMap<KernelSsaId, BoundDef> = BTreeMap::new();
    kernel.walk(&mut |op| {
        if let KernelOp::Core(CoreKernelOp::Const { into, value }) = op {
            if let ConstantValue::Int(n) = value.value {
                defs.insert(*into, BoundDef::Const(n));
            }
        } else if let KernelOp::Core(CoreKernelOp::RuntimeExtent { into, extent }) = op {
            defs.insert(*into, BoundDef::Runtime(*extent));
        }
    });

    let mut acc = BlockAccum {
        // The per-participant work basis of the block's cost. A serial
        // traversal (or a single participant) prices the whole domain on one
        // participant; a parallel traversal splits the domain across its
        // participants, so its wall-time basis is the quotient. Without the
        // division every strategy prices the same total work and the serial
        // feasibility witness wins the selection at any domain size.
        cost_total: {
            let total = iteration.cost_symbol();
            if iteration.serialized || iteration.participants.as_constant() == Some(1) {
                total
            } else {
                Sym::atom(Atom::Quot(
                    Box::new(total),
                    Box::new(iteration.participants.clone()),
                ))
            }
        },
        work_total: iteration.total_symbol(),
        cost: Sym::constant(0),
        work: Sym::constant(0),
        workgroup_bytes: local_bytes.workgroup(),
        private_bytes: local_bytes.private(),
        subgroup_width: None,
        barriers: 0,
        code_units: 0,
        capabilities: BTreeSet::new(),
        numerical: NumericalTransfer::Exact,
        device_float_add: false,
    };
    visit_ops::<D>(
        kernel.ops().as_slice(),
        &Sym::constant(1),
        &cut.algorithm,
        &defs,
        interface,
        cost_model,
        &mut acc,
    )?;

    // Numerical freedoms the strategy records (S1) become transfers here;
    // an emitter can never take one that is not recorded.
    for choice in &cut.numerical {
        match choice {
            NumericalChoice::Reassociate { node } => {
                if !acc.device_float_add {
                    return Err(at(
                        Package::S1,
                        format!(
                            "node {node:?} records a reassociation choice but its block performs no concurrent float-add atomic"
                        ),
                    ));
                }
            }
            NumericalChoice::FastMath { node, operation } => {
                // A backend's fast sequence has no analytical bound here;
                // only evidence keyed to the complete identity can qualify it.
                acc.numerical = numerics::compose(
                    &NumericalTransfer::Unknown {
                        reason: format!(
                            "fast-math `{operation}` on {node:?} has no analytical bound"
                        ),
                    },
                    &acc.numerical,
                );
            }
            NumericalChoice::ReducedPrecision { node: _, dtype } => {
                acc.numerical = numerics::compose(
                    &NumericalTransfer::Round {
                        dtype: *dtype,
                        count: CountExpr::Unbounded,
                    },
                    &acc.numerical,
                );
            }
        }
    }
    if acc.device_float_add
        && !cut
            .numerical
            .iter()
            .any(|choice| matches!(choice, NumericalChoice::Reassociate { .. }))
    {
        return Err(at(
            Package::S1,
            format!(
                "block {} performs a concurrent float-add atomic without a recorded reassociation choice",
                block.0
            ),
        ));
    }

    // Dynamic pull: one counter round trip per claimed coordinate;
    // contention beyond that is unmodelled.
    if matches!(cut.participants.policy, ParticipantPolicy::DynamicPull { .. }) {
        let pull_price = i64::try_from(cost_model.point_cost_ns(&CostUnit::Atomic { dtype: DType::U32 }))
            .map_err(|_| m1("point cost exceeds the cost domain"))?;
        acc.cost = acc.cost.add(&Sym::constant(pull_price).mul(&acc.cost_total));
    }

    // The direct-storage-binding count of the launch: one per interface
    // decl that binds a storage (tensor inputs/outputs and locals carry one
    // view per plane; scalar inputs and scalar outputs bind words, not
    // storages), plus the pull-counter storage when present. P1 seals the
    // storage bindings in the same order; the backend native validation
    // compares against exactly this count.
    let storage_decls = interface
        .inputs
        .iter()
        .filter(|decl| matches!(decl.route, crate::routes::ValueRoute::Tensor(_)))
        .count()
        + interface
            .outputs
            .iter()
            .filter(|decl| matches!(decl.route, crate::routes::ValueRoute::Tensor(_)))
            .count()
        + interface.locals.len();
    let direct_bindings = u32::try_from(
        storage_decls as u64
            + u64::from(matches!(routed_block.pull_counter, PullCounter::Counter(_))),
    )
    .map_err(|_| m1(format!("block {} exceeds the direct-binding domain", block.0)))?;

    constraints.push(SolverConstraint::Range {
        expr: participants.clone(),
        // Nonzero work needs at least one participant; identically zero work
        // is skipped by the retained launch condition.
        lower: u64::from(!acc.work.is_zero()),
        upper: limits.max_participants,
    });
    constraints.push(SolverConstraint::Range {
        expr: acc.workgroup_bytes.clone(),
        lower: 0,
        upper: limits.max_workgroup_bytes,
    });
    constraints.push(SolverConstraint::Range {
        expr: acc.private_bytes.clone(),
        lower: 0,
        upper: limits.max_explicit_private_bytes,
    });
    constraints.push(SolverConstraint::Range {
        expr: Sym::constant(i64::from(direct_bindings)),
        lower: u64::from(direct_bindings),
        upper: u64::from(limits.max_direct_bindings),
    });
    resources.push(ResourceDeclaration {
        scope: ResourceScope::Participants,
        owner: ResourceOwner::Block(block),
        amount: participants.clone(),
    });
    resources.push(ResourceDeclaration {
        scope: ResourceScope::WorkgroupBytes,
        owner: ResourceOwner::Block(block),
        amount: acc.workgroup_bytes.clone(),
    });
    resources.push(ResourceDeclaration {
        scope: ResourceScope::ParticipantPrivateBytes,
        owner: ResourceOwner::Block(block),
        amount: acc.private_bytes.clone(),
    });
    resources.push(ResourceDeclaration {
        scope: ResourceScope::DirectBindings,
        owner: ResourceOwner::Block(block),
        amount: Sym::constant(i64::from(direct_bindings)),
    });
    resources.push(ResourceDeclaration {
        scope: ResourceScope::StaticCodeUnits,
        owner: ResourceOwner::Block(block),
        amount: Sym::constant(
            i64::try_from(acc.code_units)
                .map_err(|_| m1("static code units exceed the size domain"))?,
        ),
    });
    resources.push(ResourceDeclaration {
        scope: ResourceScope::KernelLocalBytes,
        owner: ResourceOwner::Block(block),
        amount: local_bytes.kernel_local(),
    });

    let overhead = i64::try_from(cost_model.launch_overhead_ns())
        .map_err(|_| m1("launch overhead exceeds the cost domain"))?;
    let per_visit = Sym::constant(overhead).add(&acc.cost);
    let facts = BlockFacts {
        block,
        participants,
        work_items: acc.work,
        workgroup_bytes: acc.workgroup_bytes,
        private_bytes_per_participant: acc.private_bytes,
        direct_bindings,
        static_code_units: acc.code_units,
        required_subgroup_width: acc.subgroup_width,
        barriers: acc.barriers,
        capabilities: acc.capabilities,
    };
    Ok((facts, per_visit, acc.numerical))
}

struct BlockAccum {
    /// The block's iteration total priced for cost (expected).
    cost_total: Sym,
    /// The block's iteration total for resources (capacity).
    work_total: Sym,
    cost: Sym,
    work: Sym,
    workgroup_bytes: Sym,
    private_bytes: Sym,
    subgroup_width: Option<u32>,
    barriers: u32,
    code_units: u64,
    capabilities: BTreeSet<IntrinsicId>,
    numerical: NumericalTransfer,
    device_float_add: bool,
}

/// Exhaustive op fold of one block: every static site contributes exactly
/// one cost unit weighted by its multiplicity and exactly one numerical
/// transfer. A site's multiplicity is the block's iteration total times the
/// trip counts of its enclosing serial `Repeat`s (a `Fold` additionally
/// times its folded axis extent, which the op visits internally).
#[allow(clippy::too_many_arguments)]
fn visit_ops<D: ExecutableDialect>(
    ops: &[KernelOp<D::Intrinsic>],
    trips: &Sym,
    algorithm: &AlgorithmChoice,
    defs: &BTreeMap<KernelSsaId, BoundDef>,
    interface: &ClosedKernelInterface,
    cost_model: &dyn CostModel,
    acc: &mut BlockAccum,
) -> Result<(), CompilerDefect> {
    for op in ops {
        let (unit, transfer, fold_extent): (CostUnit, NumericalTransfer, Option<Sym>) = match op {
            KernelOp::Core(CoreKernelOp::Const { .. })
            | KernelOp::Core(CoreKernelOp::RuntimeExtent { .. })
            | KernelOp::Core(CoreKernelOp::Unary { .. })
            | KernelOp::Core(CoreKernelOp::Binary { .. })
            | KernelOp::Core(CoreKernelOp::Compare { .. })
            | KernelOp::Core(CoreKernelOp::Cast { .. })
            | KernelOp::Core(CoreKernelOp::Select { .. })
            | KernelOp::Core(CoreKernelOp::TableLookup { .. })
            | KernelOp::Core(CoreKernelOp::Carry { .. })
            | KernelOp::Core(CoreKernelOp::Publish { .. })
            | KernelOp::Core(CoreKernelOp::Repeat { .. })
            | KernelOp::Core(CoreKernelOp::Branch { .. })
            | KernelOp::Core(CoreKernelOp::Check { .. }) => {
                (CostUnit::Scalar, NumericalTransfer::Exact, None)
            }
            KernelOp::Core(CoreKernelOp::Math { op: math, dtype, .. }) => (
                CostUnit::Math {
                    op: *math,
                    dtype: *dtype,
                },
                NumericalTransfer::Exact,
                None,
            ),
            KernelOp::Core(CoreKernelOp::Fma { dtype, .. }) => (
                CostUnit::Math {
                    op: MathOp::Fma,
                    dtype: *dtype,
                },
                NumericalTransfer::Exact,
                None,
            ),
            KernelOp::Core(CoreKernelOp::Load { dtype, .. }) => {
                (CostUnit::Load { dtype: *dtype }, NumericalTransfer::Exact, None)
            }
            KernelOp::Core(CoreKernelOp::Store { dtype, .. }) => {
                (CostUnit::Store { dtype: *dtype }, NumericalTransfer::Exact, None)
            }
            KernelOp::Core(CoreKernelOp::PackedPlaneRead { repr, dtype, .. })
            | KernelOp::Core(CoreKernelOp::PlaneLoad { repr, dtype, .. })
            | KernelOp::Core(CoreKernelOp::PlaneStore { repr, dtype, .. }) => (
                CostUnit::PlaneAccess {
                    repr: repr.clone(),
                    dtype: *dtype,
                },
                NumericalTransfer::Exact,
                None,
            ),
            KernelOp::Core(CoreKernelOp::Atomic {
                op: atomic,
                dtype,
                mode,
                ..
            }) => (
                CostUnit::Atomic { dtype: *dtype },
                atomic_transfer(*atomic, *dtype, *mode),
                None,
            ),
            KernelOp::Core(CoreKernelOp::Fold {
                op: reduce,
                axis,
                shape: fold_shape,
                schema,
                ..
            }) => {
                let extent = fold_shape.axes.get(*axis as usize).ok_or_else(|| {
                    at(
                        Package::K1,
                        format!("fold axis {axis} is outside the operand rank"),
                    )
                })?;
                (
                    CostUnit::Fold {
                        op: *reduce,
                        dtype: schema.accumulator,
                    },
                    fold_transfer(*reduce, algorithm)?,
                    Some(extent_bound_symbol(extent)?),
                )
            }
            KernelOp::Core(CoreKernelOp::Barrier { .. })
            | KernelOp::Core(CoreKernelOp::GridBarrier) => {
                acc.barriers = acc
                    .barriers
                    .checked_add(1)
                    .ok_or_else(|| m1("launch barrier count overflows u32"))?;
                (CostUnit::Barrier, NumericalTransfer::Exact, None)
            }
            KernelOp::Intrinsic(intrinsic) => {
                let consequences = D::intrinsic_consequences(intrinsic);
                let private = i64::try_from(consequences.private_bytes)
                    .map_err(|_| m1("intrinsic private bytes exceed the size domain"))?;
                let workgroup = i64::try_from(consequences.workgroup_bytes)
                    .map_err(|_| m1("intrinsic workgroup bytes exceed the size domain"))?;
                acc.private_bytes = acc.private_bytes.add(&Sym::constant(private));
                acc.workgroup_bytes = acc.workgroup_bytes.add(&Sym::constant(workgroup));
                if let Some(width) = consequences.required_subgroup_width {
                    if acc.subgroup_width.is_some_and(|existing| existing != width) {
                        return Err(at(
                            Package::K1,
                            format!(
                                "dialect {} requires conflicting subgroup widths in one block",
                                std::any::type_name::<D>()
                            ),
                        ));
                    }
                    acc.subgroup_width = Some(width);
                }
                acc.capabilities.insert(consequences.capability.clone());
                (
                    CostUnit::Intrinsic(consequences.capability),
                    consequences.numerical,
                    None,
                )
            }
        };
        if matches!(
            op,
            KernelOp::Core(CoreKernelOp::Atomic {
                op: AtomicOp::Add,
                dtype: DType::F32 | DType::F16 | DType::BF16,
                mode: AtomicMode::Device,
                ..
            })
        ) {
            acc.device_float_add = true;
        }
        let mut cost_weight = acc.cost_total.mul(trips);
        let mut work_weight = acc.work_total.mul(trips);
        if let Some(extent) = &fold_extent {
            cost_weight = cost_weight.mul(extent);
            work_weight = work_weight.mul(extent);
        }
        let price = i64::try_from(cost_model.point_cost_ns(&unit))
            .map_err(|_| m1("point cost exceeds the cost domain"))?;
        acc.cost = acc.cost.add(&Sym::constant(price).mul(&cost_weight));
        acc.work = acc.work.add(&work_weight);
        acc.code_units += 1;
        acc.numerical = numerics::compose(&transfer, &acc.numerical);

        if let KernelOp::Core(CoreKernelOp::Repeat { start, end, body, .. }) = op {
            let trip = value_bound(*end, defs, interface)?.sub(&value_bound(*start, defs, interface)?);
            visit_ops::<D>(body, &trips.mul(&trip), algorithm, defs, interface, cost_model, acc)?;
        } else if let KernelOp::Core(CoreKernelOp::Branch {
            then_body,
            else_body,
            ..
        }) = op
        {
            visit_ops::<D>(then_body, trips, algorithm, defs, interface, cost_model, acc)?;
            visit_ops::<D>(else_body, trips, algorithm, defs, interface, cost_model, acc)?;
        } else if let KernelOp::Core(CoreKernelOp::Check { guarded, .. }) = op {
            visit_ops::<D>(guarded, trips, algorithm, defs, interface, cost_model, acc)?;
        }
    }
    Ok(())
}

/// The transfer of one atomic site: a serialized update is the reference
/// load/combine/round/store, integers and extremum combines are exact and
/// order-independent; only a `Device` float `add` reassociates the
/// per-visit rounding order (the choice is recorded by S1 and verified by
/// the caller). One round trip per visit is priced; contention is
/// unmodelled.
fn atomic_transfer(op: AtomicOp, dtype: DType, mode: AtomicMode) -> NumericalTransfer {
    if mode == AtomicMode::Device && dtype.is_float() && op == AtomicOp::Add {
        NumericalTransfer::Reassociate {
            op: ReduceOp::Sum,
            topology: ReductionTopology::AtomicCombine,
        }
    } else {
        NumericalTransfer::Exact
    }
}

/// The transfer of one `Fold` site: the registry schema's ascending fold is
/// the reference; a block realized under a reassociating reduction topology
/// (`AlgorithmChoice::Reduction`) visits in a different order. `argmax`
/// never reassociates.
fn fold_transfer(op: ReduceOp, algorithm: &AlgorithmChoice) -> Result<NumericalTransfer, CompilerDefect> {
    match algorithm {
        AlgorithmChoice::Reduction(topology) => {
            if topology.is_reference_order() {
                Ok(NumericalTransfer::Exact)
            } else if op == ReduceOp::Argmax {
                Err(at(
                    Package::S1,
                    "a reassociating reduction topology was applied to `argmax`",
                ))
            } else {
                Ok(NumericalTransfer::Reassociate {
                    op,
                    topology: topology.clone(),
                })
            }
        }
        AlgorithmChoice::Universal
        | AlgorithmChoice::Blocked { .. }
        | AlgorithmChoice::Intrinsic(_) => Ok(NumericalTransfer::Exact),
    }
}

fn value_bound(
    value: KernelValueRef,
    defs: &BTreeMap<KernelSsaId, BoundDef>,
    interface: &ClosedKernelInterface,
) -> Result<Sym, CompilerDefect> {
    match value {
        KernelValueRef::Input(id) => {
            let decl = interface.inputs.get(id).ok_or_else(|| {
                at(
                    Package::K1,
                    format!("repeat bound names kernel input {id:?} absent from the interface"),
                )
            })?;
            Ok(Sym::param(&format!("@leaf{}", decl.leaf.0)))
        }
        KernelValueRef::Ssa(id) => match defs.get(&id) {
            Some(BoundDef::Const(n)) => Ok(Sym::constant(*n)),
            Some(BoundDef::Runtime(extent)) => Ok(Sym::param(&format!("@runtime{}", extent.0))),
            None => Err(at(
                Package::K1,
                format!("repeat bound SSA {id:?} is neither a constant nor a runtime extent"),
            )),
        },
        KernelValueRef::Axis(id) => Err(at(
            Package::K1,
            format!("repeat bound is the axis coordinate {id:?}"),
        )),
    }
}

/// Strategy cost over the structured schedule: launches and steps are
/// weighted by the retained repeat bounds enclosing them.
fn schedule_cost(
    schedule: &ShapeSchedule,
    multiplier: &Sym,
    block_cost: &BTreeMap<BlockId, Sym>,
    cost_model: &dyn CostModel,
) -> Result<Sym, CompilerDefect> {
    let scalar = i64::try_from(cost_model.point_cost_ns(&CostUnit::Scalar))
        .map_err(|_| m1("point cost exceeds the cost domain"))?;
    let overhead = i64::try_from(cost_model.launch_overhead_ns())
        .map_err(|_| m1("launch overhead exceeds the cost domain"))?;
    let mut total = Sym::constant(0);
    for step in &schedule.steps {
        let term = match step {
            ShapeStep::Launch(block) => {
                let per_visit = block_cost.get(block).ok_or_else(|| {
                    at(
                        Package::D1,
                        format!("schedule launches block {} which the strategy does not hold", block.0),
                    )
                })?;
                per_visit.mul(multiplier)
            }
            ShapeStep::Guard { .. } => Sym::constant(scalar).mul(multiplier),
            ShapeStep::PullCounterReset { .. } | ShapeStep::Call { .. } => {
                Sym::constant(overhead).mul(multiplier)
            }
            ShapeStep::If {
                then_schedule,
                else_schedule,
                ..
            } => {
                let taken_then = schedule_cost(then_schedule, multiplier, block_cost, cost_model)?;
                let taken_else = schedule_cost(else_schedule, multiplier, block_cost, cost_model)?;
                branch_cost(taken_then, taken_else)
                    .add(&Sym::constant(scalar).mul(multiplier))
            }
            ShapeStep::Repeat { bound, body, .. } => schedule_cost(
                body,
                &multiplier.mul(&extent_bound_symbol(bound)?),
                block_cost,
                cost_model,
            )?,
        };
        total = total.add(&term);
    }
    Ok(total)
}

/// Cost of two mutually exclusive schedule sides. The exact contribution is
/// the maximum of the sides, which a polynomial cannot express; the maximum
/// is used whenever both sides are concrete, and the sum otherwise is the
/// conservative ranking envelope (cost ranks, it never decides legality).
fn branch_cost(taken_then: Sym, taken_else: Sym) -> Sym {
    match (taken_then.as_constant(), taken_else.as_constant()) {
        (Some(a), Some(b)) => Sym::constant(a.max(b)),
        _ => {
            if taken_then.is_zero() {
                taken_else
            } else if taken_else.is_zero() {
                taken_then
            } else {
                taken_then.add(&taken_else)
            }
        }
    }
}
