//! Closed implementation alternatives, the implementation builder, and the
//! factory boundary (spec §6).
//!
//! An `Implementation<B>` owns, together: identity, semantic coverage,
//! parametric schedule, kernels, global and local allocation topology, finite
//! decisions, hard constraints, numerical transfer, modeled duration, and provenance.
//! All fields are private; construction is through `ImplementationBuilder`,
//! which is created only by `plan_space` and handed to factories.
//!
//! Factories receive a request and a builder. Before construction they may
//! decline; once construction begins they return a closed implementation or
//! a real preparation error. They cannot return a graph, proposal, label,
//! partial placement, side table, or callback.
//!
//! Calls are resolved during construction: `splice_call` constructs every
//! applicable child implementation and splices it under a finite decision,
//! composing guards, constraints, lifetimes, transfers, durations, provenance,
//! and effect ordering. After construction no call exists.
//!
//! W4 owns the internals.

use crate::identity::OwnerToken;
use crate::kernel::{KernelArena, KernelBuilder};
use crate::numerics::NumericalTransfer;
use crate::repr::{Representation, ScalarType};
use crate::schedule::{
    AnyScalarSlot, ClosedSchedule, ParametricSchedule, ScalarSlotId, ScheduleBuilder,
};
use crate::storage::{
    AnyBufferView, BufferViewId, GlobalAllocationId, GlobalAllocationTopology,
    LocalAllocationTopology,
};
use crate::target::{Backend, DeviceContract, ExecutionProfile, PlanningMachine, TargetConstants};
use seismic_lang::entry::{
    AccessKind, AliasRule, CallSchema, CandidateKind, ParameterAccess, ParameterKind,
    SemanticFunction, SemanticNodeView, SemanticProgram, SemanticType, TensorStorage,
};
use seismic_lang::expr::{
    AnyExpr, BinaryOp, BoolExpr, CmpOp, DecisionId, DurationExpr, ExprArena, FiniteDomain, NaryOp,
    NatExpr, NodeView, TargetPredicate, UnaryOp,
};
use seismic_lang::ids::{FamilyId, NodeId, RegionId, SemanticValueId, StableFunctionId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::registry::BackendName;
use seismic_lang::types::DType;
use std::fmt;
use std::sync::Arc;
use std::{cell::RefCell, rc::Rc};

pub(crate) type ConstructionBudget =
    Rc<RefCell<crate::preparation_budget::PreparationBudgetTracker>>;

/// Stable identity of one implementation alternative (§15.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImplementationIdentity {
    pub factory: FactoryIdentity,
    /// Digest over the constructed structure (schedule, kernels, topology,
    /// decisions), independent of the solver assignment.
    pub structure: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FactoryIdentity {
    pub name: &'static str,
    pub revision: &'static str,
}

/// Where an implementation's commands came from, for diagnostics and
/// telemetry only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplementationProvenance {
    pub root: StableFunctionId,
    /// Spliced callees, in splice order.
    pub callees: Vec<StableFunctionId>,
}

/// One root result leaf in exact semantic-contract order. This is sealed when
/// the implementation closes, so executable translation never reconstructs
/// result meaning from topology searches or schema zips.
#[derive(Clone, Debug)]
pub(crate) struct ResultPublication {
    pub path: Vec<u32>,
    pub binding: PublishedResult,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PublishedResult {
    Buffer {
        view: AnyBufferView,
        bytes: NatExpr,
    },
    Scalar {
        slot: AnyScalarSlot,
        kind: PublishedScalarKind,
    },
    Range {
        start: AnyScalarSlot,
        end: AnyScalarSlot,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PublishedScalarKind {
    Value(DType),
    Index,
}

/// Structurally closed implementation produced by a factory. Native code is
/// deliberately absent: only the root preparation boundary may consume this
/// value and form a [`Implementation`].
#[derive(Debug)]
pub struct ImplementationDraft<B: Backend> {
    authority: ConstructionAuthority,
    numerical_role: seismic_lang::entry::NumericalRole,
    identity: ImplementationIdentity,
    semantic_coverage: TargetPredicate,
    schedule: ParametricSchedule,
    kernels: KernelArena<B>,
    global_allocations: GlobalAllocationTopology,
    local_allocations: LocalAllocationTopology,
    launch_layouts: Vec<crate::storage::LaunchLocalLayout>,
    launch_scratch: Vec<crate::storage::LaunchScratchRequirements>,
    launch_abi: Vec<Vec<crate::storage::LaunchAbiRequirement>>,
    decisions: Vec<(DecisionId, &'static str)>,
    hard_constraints: BoolExpr,
    numerical_transfer: NumericalTransfer,
    duration: DurationExpr,
    duration_qualification: Vec<BoolExpr>,
    provenance: ImplementationProvenance,
    result_publications: Vec<ResultPublication>,
}

impl<B: Backend> ImplementationDraft<B> {
    pub fn identity(&self) -> &ImplementationIdentity {
        &self.identity
    }
    pub fn semantic_coverage(&self) -> TargetPredicate {
        self.semantic_coverage
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        &self.schedule
    }
    pub fn kernels(&self) -> &KernelArena<B> {
        &self.kernels
    }
    pub fn global_allocations(&self) -> &GlobalAllocationTopology {
        &self.global_allocations
    }
    pub fn local_allocations(&self) -> &LocalAllocationTopology {
        &self.local_allocations
    }
    pub fn launch_layouts(&self) -> &[crate::storage::LaunchLocalLayout] {
        &self.launch_layouts
    }
    pub fn launch_scratch(&self) -> &[crate::storage::LaunchScratchRequirements] {
        &self.launch_scratch
    }
    pub fn launch_abi(&self) -> &[Vec<crate::storage::LaunchAbiRequirement>] {
        &self.launch_abi
    }
    pub fn decisions(&self) -> &[(DecisionId, &'static str)] {
        &self.decisions
    }
    pub fn hard_constraints(&self) -> BoolExpr {
        self.hard_constraints
    }
    pub fn numerical_transfer(&self) -> &NumericalTransfer {
        &self.numerical_transfer
    }
    pub(crate) fn duration(&self) -> DurationExpr {
        self.duration
    }
    pub(crate) fn duration_qualification(&self) -> &[BoolExpr] {
        &self.duration_qualification
    }
    pub fn provenance(&self) -> &ImplementationProvenance {
        &self.provenance
    }

    pub(crate) fn from_parts(parts: ImplementationParts<B>) -> Self {
        Self {
            authority: parts.authority,
            numerical_role: parts.numerical_role,
            identity: parts.identity,
            semantic_coverage: parts.semantic_coverage,
            schedule: parts.schedule,
            kernels: parts.kernels,
            global_allocations: parts.global_allocations,
            local_allocations: parts.local_allocations,
            launch_layouts: parts.launch_layouts,
            launch_scratch: parts.launch_scratch,
            launch_abi: parts.launch_abi,
            decisions: parts.decisions,
            hard_constraints: parts.hard_constraints,
            numerical_transfer: parts.numerical_transfer,
            duration: parts.duration,
            duration_qualification: parts.duration_qualification,
            provenance: parts.provenance,
            result_publications: parts.result_publications,
        }
    }
    pub(crate) fn result_publications(&self) -> &[ResultPublication] {
        &self.result_publications
    }
    pub(crate) fn into_parts(self) -> ImplementationParts<B> {
        ImplementationParts {
            authority: self.authority,
            numerical_role: self.numerical_role,
            identity: self.identity,
            semantic_coverage: self.semantic_coverage,
            schedule: self.schedule,
            kernels: self.kernels,
            global_allocations: self.global_allocations,
            local_allocations: self.local_allocations,
            launch_layouts: self.launch_layouts,
            launch_scratch: self.launch_scratch,
            launch_abi: self.launch_abi,
            decisions: self.decisions,
            hard_constraints: self.hard_constraints,
            numerical_transfer: self.numerical_transfer,
            duration: self.duration,
            duration_qualification: self.duration_qualification,
            provenance: self.provenance,
            result_publications: self.result_publications,
        }
    }

    /// Stable pre-compilation identities for the native templates in this
    /// draft. These are available before a toolchain is invoked, allowing an
    /// optional implementation to be rejected atomically by the preparation
    /// budget without compile-and-drop behavior.
    pub(crate) fn native_template_identities(&self, target: &DeviceContract<B>) -> Vec<[u8; 32]> {
        self.kernels
            .kernels()
            .enumerate()
            .map(|(ordinal, _)| {
                let mut digest =
                    crate::identity::StructureDigest::new("seismic-native-template-v1");
                digest.bytes(&self.identity.structure);
                digest.hashed(&ordinal);
                digest.bytes(&target.compatibility_identity().fingerprint);
                digest.finish()
            })
            .collect()
    }

    /// Consuming native closure. This is the sole call site for backend
    /// native formation, and it runs before `PlanSpace` can be constructed.
    pub(crate) fn close_native(
        mut self,
        arena: &mut ExprArena,
        target: &DeviceContract<B>,
        execution: &ExecutionProfile<B>,
        constants: &TargetConstants,
    ) -> Result<
        (Implementation<B>, crate::target::NativeArtifactMetrics),
        crate::errors::PreparationError,
    > {
        let mut native_kernels = Vec::with_capacity(self.kernels.kernels().count());
        let mut metrics = crate::target::NativeArtifactMetrics {
            compilation_ns: 0,
            code_bytes: 0,
            metadata_bytes: 0,
        };
        for (_, kernel) in self.kernels.kernels() {
            let native = target
                .form_native_kernel_candidate(arena, kernel)
                .and_then(|candidate| target.reconcile_native_kernel(kernel, candidate))
                .map_err(crate::errors::PreparationError::NativeCompilation)?;
            let artifact = native.contract().artifact;
            metrics.compilation_ns = metrics
                .compilation_ns
                .checked_add(artifact.compilation_ns)
                .ok_or_else(|| {
                    crate::errors::PreparationError::NativeCompilation(
                        crate::errors::NativeCompilationError::ToolchainResourceExhausted(
                            "aggregate native compilation duration exceeds u64 nanoseconds".into(),
                        ),
                    )
                })?;
            metrics.code_bytes = metrics
                .code_bytes
                .checked_add(artifact.code_bytes)
                .ok_or_else(|| {
                    crate::errors::PreparationError::NativeCompilation(
                        crate::errors::NativeCompilationError::ToolchainResourceExhausted(
                            "aggregate native code size exceeds u64 bytes".into(),
                        ),
                    )
                })?;
            metrics.metadata_bytes = metrics
                .metadata_bytes
                .checked_add(artifact.metadata_bytes)
                .ok_or_else(|| {
                    crate::errors::PreparationError::NativeCompilation(
                        crate::errors::NativeCompilationError::ToolchainResourceExhausted(
                            "aggregate native metadata size exceeds u64 bytes".into(),
                        ),
                    )
                })?;
            native_kernels.push(Arc::new(native));
        }

        let mut total_launch_certificate = None;
        if self.authority == ConstructionAuthority::UniversalPortable {
            let unspecialized = execution.derive_closed_duration(
                target,
                arena,
                &self.schedule,
                &self.launch_layouts,
                &self.kernels,
                &native_kernels,
            );
            total_launch_certificate = Some(specialize_universal_launches(
                arena,
                execution,
                constants,
                &mut self.schedule,
                &native_kernels,
                &unspecialized.terms,
            )?);
        }

        let reflected = native_hard_constraints(
            arena,
            target,
            &self.schedule,
            &self.kernels,
            &self.launch_layouts,
            &native_kernels,
            self.authority != ConstructionAuthority::UniversalPortable,
        );
        let combined = arena.and(self.hard_constraints, reflected);
        let side_conditions = arena.side_conditions(AnyExpr::Bool(combined));
        self.hard_constraints = arena.and(side_conditions, combined);

        let duration = execution.derive_closed_duration(
            target,
            arena,
            &self.schedule,
            &self.launch_layouts,
            &self.kernels,
            &native_kernels,
        );
        self.duration = duration.duration;
        self.duration_qualification = if total_launch_certificate.is_some() {
            residual_schedule_qualification(&duration)
        } else {
            duration.qualification
        };

        let mut digest = crate::identity::StructureDigest::new("seismic-implementation-native-v1");
        digest.bytes(&self.identity.structure);
        for native in &native_kernels {
            let contract = native.contract();
            digest.bytes(&contract.identity.compatibility.fingerprint);
            digest.bytes(&contract.identity.artifact_digest);
            digest.bytes(&contract.numerical_identity.fingerprint);
        }
        self.identity.structure = digest.finish();
        Ok((
            Implementation {
                draft: self,
                native_kernels,
                total_launch_certificate,
            },
            metrics,
        ))
    }
}

/// Private construction witness that every semantic launch was rewritten to
/// exact chunks within the reconciled native and measured-service domains.
/// It has no public constructor: possession is the proof consumed by
/// `UniversalImplementation`.
#[derive(Debug)]
struct TotalLaunchCertificate {
    maximum_grid_x: Vec<u64>,
}

fn residual_schedule_qualification(duration: &crate::target::QualifiedDuration) -> Vec<BoolExpr> {
    assert_eq!(duration.qualification.len(), duration.terms.len() * 2);
    duration
        .terms
        .iter()
        .zip(duration.qualification.chunks_exact(2))
        .filter(|(term, _)| term.launch_ordinal.is_none())
        .flat_map(|(_, pair)| pair.iter().copied())
        .collect()
}

fn specialize_universal_launches<B: Backend>(
    arena: &mut ExprArena,
    execution: &ExecutionProfile<B>,
    constants: &TargetConstants,
    schedule: &mut ParametricSchedule,
    native_kernels: &[Arc<crate::target::NativeKernel<B>>],
    terms: &[crate::target::ServiceModelTerm],
) -> Result<TotalLaunchCertificate, crate::errors::PreparationError> {
    let mut fixed = seismic_lang::expr::PartialAssignment::new();
    for (symbol, value) in constants.bindings() {
        fixed.bind(*symbol, *value);
    }
    let launches = schedule.launches().to_vec();
    let mut caps = Vec::with_capacity(launches.len());
    for (ordinal, launch) in launches.iter().enumerate() {
        let semantic = match (launch.parallel_extent, launch.logical_base) {
            (Some(_), Some(_)) => true,
            (None, None) => false,
            _ => {
                return Err(crate::errors::PreparationError::UniversalClosure(format!(
                    "launch {ordinal} has incomplete compiler-owned logical indexing"
                )))
            }
        };
        let native = &native_kernels[launch.kernel.index() as usize];
        if !semantic {
            let mut concrete_grid = [0u64; 3];
            for (axis, value) in launch.grid.iter().enumerate() {
                let value = arena.partial(*value, &fixed);
                let NodeView::NatConst(value) = arena.view(AnyExpr::Nat(value)) else {
                    return Err(crate::errors::PreparationError::UniversalClosure(format!(
                        "fixed launch {ordinal} grid is not constructionally closed"
                    )));
                };
                if value == 0 || value > native.contract().launch.max_grid[axis] {
                    return Err(crate::errors::PreparationError::UniversalClosure(format!(
                        "fixed launch {ordinal} exceeds reflected grid legality"
                    )));
                }
                concrete_grid[axis] = value;
            }
            for term in terms
                .iter()
                .filter(|term| term.launch_ordinal == Some(ordinal as u32))
            {
                let units = arena.partial(term.units, &fixed);
                let NodeView::NatConst(units) = arena.view(AnyExpr::Nat(units)) else {
                    let retained = arena
                        .free_symbols(AnyExpr::Nat(units))
                        .into_iter()
                        .map(|symbol| format!("{symbol:?}:{:?}", arena.symbol_kind(symbol)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(exact_inversion_error_with_reason(
                        ordinal,
                        term.class,
                        &format!(
                            "the fixed launch service demand is not constructionally closed (retained: {retained})"
                        ),
                    ));
                };
                let qualification = execution.service(term.class).qualification;
                if units < qualification.minimum_units || units > qualification.maximum_units {
                    return Err(exact_inversion_error(ordinal, term.class));
                }
            }
            caps.push(concrete_grid[0]);
            continue;
        }
        if launch.mode != crate::schedule::LaunchMode::Independent {
            return Err(crate::errors::PreparationError::UniversalClosure(format!(
                "semantic launch {ordinal} is not independently chunkable"
            )));
        }
        for axis in 1..3 {
            let axis = arena.partial(launch.grid[axis], &fixed);
            if !matches!(arena.view(AnyExpr::Nat(axis)), NodeView::NatConst(1)) {
                return Err(crate::errors::PreparationError::UniversalClosure(format!(
                    "semantic launch {ordinal} is not exactly invertible as a one-dimensional launch"
                )));
            }
        }
        let mut cap = native.contract().launch.max_grid[0];
        if cap == 0 {
            return Err(crate::errors::PreparationError::UniversalClosure(format!(
                "semantic launch {ordinal} has zero reflected grid capacity"
            )));
        }
        let grid = launch.grid[0];
        for term in terms
            .iter()
            .filter(|term| term.launch_ordinal == Some(ordinal as u32))
        {
            let qualification = execution.service(term.class).qualification;
            let units = term.units;
            let first = exact_grid_value(arena, units, grid, 1, &fixed).ok_or_else(|| {
                exact_inversion_error_with_reason(
                    ordinal,
                    term.class,
                    "the one-grid service expression is not exactly evaluable from target constants",
                )
            })?;
            if first.value < qualification.minimum_units
                || first.value > qualification.maximum_units
            {
                return Err(exact_inversion_error_with_reason(
                    ordinal,
                    term.class,
                    &format!(
                        "the one-grid service demand {} is outside the qualified domain {}..={}",
                        first.value, qualification.minimum_units, qualification.maximum_units,
                    ),
                ));
            }
            if !first.depends_on_grid {
                continue;
            }
            let mut low = 1u64;
            let mut high = cap;
            while low < high {
                let middle = low + (high - low + 1) / 2;
                let admissible = exact_grid_value(arena, units, grid, middle, &fixed)
                    .map(|value| {
                        value.value >= qualification.minimum_units
                            && value.value <= qualification.maximum_units
                    })
                    .unwrap_or(false);
                if admissible {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            cap = cap.min(low);
        }
        caps.push(cap);
    }
    for (ordinal, cap) in caps.iter().copied().enumerate() {
        if launches[ordinal].parallel_extent.is_none() {
            continue;
        }
        let cap = arena.nat(cap);
        let id = crate::schedule::LaunchId::new(schedule.owner(), ordinal as u32);
        schedule.chunk_semantic_launch(arena, id, cap);
    }
    Ok(TotalLaunchCertificate {
        maximum_grid_x: caps,
    })
}

fn exact_inversion_error(
    launch: usize,
    class: crate::target::ServiceClassId,
) -> crate::errors::PreparationError {
    crate::errors::PreparationError::UniversalClosure(format!(
        "launch {launch} service {} has no exact portable grid inversion",
        class.stable_name()
    ))
}

fn exact_inversion_error_with_reason(
    launch: usize,
    class: crate::target::ServiceClassId,
    reason: &str,
) -> crate::errors::PreparationError {
    crate::errors::PreparationError::UniversalClosure(format!(
        "launch {launch} service {} has no exact portable grid inversion: {reason}",
        class.stable_name()
    ))
}

#[derive(Clone, Copy)]
struct ExactGridValue {
    value: u64,
    depends_on_grid: bool,
}

#[derive(Clone, Copy)]
struct ExactBoolValue {
    value: bool,
    depends_on_grid: bool,
}

fn exact_bool_value(
    arena: &ExprArena,
    expression: BoolExpr,
    grid: NatExpr,
    grid_value: u64,
    fixed: &seismic_lang::expr::PartialAssignment,
) -> Option<ExactBoolValue> {
    let boolean = |value, depends_on_grid| ExactBoolValue {
        value,
        depends_on_grid,
    };
    match arena.view(AnyExpr::Bool(expression)) {
        NodeView::BoolConst(value) => Some(boolean(value, false)),
        NodeView::Symbol(symbol) => match fixed.get(symbol) {
            Some(seismic_lang::expr::SymbolValue::Bool(value)) => Some(boolean(value, false)),
            _ => None,
        },
        NodeView::Unary {
            op: UnaryOp::Not,
            operand: AnyExpr::Bool(operand),
        } => {
            let operand = exact_bool_value(arena, operand, grid, grid_value, fixed)?;
            Some(boolean(!operand.value, operand.depends_on_grid))
        }
        NodeView::Binary {
            op,
            lhs: AnyExpr::Bool(lhs),
            rhs: AnyExpr::Bool(rhs),
        } if matches!(
            op,
            BinaryOp::And | BinaryOp::Or | BinaryOp::Implies | BinaryOp::Iff
        ) =>
        {
            let lhs = exact_bool_value(arena, lhs, grid, grid_value, fixed)?;
            let rhs = exact_bool_value(arena, rhs, grid, grid_value, fixed)?;
            let value = match op {
                BinaryOp::And => lhs.value && rhs.value,
                BinaryOp::Or => lhs.value || rhs.value,
                BinaryOp::Implies => !lhs.value || rhs.value,
                BinaryOp::Iff => lhs.value == rhs.value,
                _ => unreachable!(),
            };
            Some(boolean(value, lhs.depends_on_grid || rhs.depends_on_grid))
        }
        NodeView::Nary { op, operands } if matches!(op, NaryOp::All | NaryOp::Any) => {
            let mut depends_on_grid = false;
            let mut values = Vec::with_capacity(operands.len());
            for operand in operands {
                let AnyExpr::Bool(operand) = *operand else {
                    return None;
                };
                let operand = exact_bool_value(arena, operand, grid, grid_value, fixed)?;
                depends_on_grid |= operand.depends_on_grid;
                values.push(operand.value);
            }
            Some(boolean(
                match op {
                    NaryOp::All => values.into_iter().all(|value| value),
                    NaryOp::Any => values.into_iter().any(|value| value),
                    _ => unreachable!(),
                },
                depends_on_grid,
            ))
        }
        NodeView::Cmp { op, lhs, rhs } => {
            let (AnyExpr::Nat(lhs), AnyExpr::Nat(rhs)) = (lhs, rhs) else {
                return None;
            };
            let lhs = exact_grid_value(arena, lhs, grid, grid_value, fixed)?;
            let rhs = exact_grid_value(arena, rhs, grid, grid_value, fixed)?;
            Some(boolean(
                match op {
                    CmpOp::Eq => lhs.value == rhs.value,
                    CmpOp::Ne => lhs.value != rhs.value,
                    CmpOp::Lt => lhs.value < rhs.value,
                    CmpOp::Le => lhs.value <= rhs.value,
                    CmpOp::Gt => lhs.value > rhs.value,
                    CmpOp::Ge => lhs.value >= rhs.value,
                },
                lhs.depends_on_grid || rhs.depends_on_grid,
            ))
        }
        _ => None,
    }
}

fn exact_grid_value(
    arena: &ExprArena,
    expression: NatExpr,
    grid: NatExpr,
    grid_value: u64,
    fixed: &seismic_lang::expr::PartialAssignment,
) -> Option<ExactGridValue> {
    fn binary(op: BinaryOp, left: ExactGridValue, right: ExactGridValue) -> Option<ExactGridValue> {
        let value = match op {
            BinaryOp::Add => left.value.checked_add(right.value)?,
            BinaryOp::Mul => left.value.checked_mul(right.value)?,
            BinaryOp::Div if !right.depends_on_grid && right.value != 0 => left.value / right.value,
            BinaryOp::CeilDiv if !right.depends_on_grid && right.value != 0 => {
                left.value.checked_add(right.value - 1)? / right.value
            }
            BinaryOp::Min => left.value.min(right.value),
            BinaryOp::Max => left.value.max(right.value),
            BinaryOp::AlignUp if !right.depends_on_grid && right.value != 0 => left
                .value
                .checked_add(right.value - 1)?
                .checked_div(right.value)?
                .checked_mul(right.value)?,
            _ => return None,
        };
        Some(ExactGridValue {
            value,
            depends_on_grid: left.depends_on_grid || right.depends_on_grid,
        })
    }

    if expression == grid {
        return Some(ExactGridValue {
            value: grid_value,
            depends_on_grid: true,
        });
    }
    match arena.view(AnyExpr::Nat(expression)) {
        NodeView::NatConst(value) => Some(ExactGridValue {
            value,
            depends_on_grid: false,
        }),
        NodeView::Symbol(symbol) => match fixed.get(symbol) {
            Some(seismic_lang::expr::SymbolValue::Nat(value)) => Some(ExactGridValue {
                value,
                depends_on_grid: false,
            }),
            _ => None,
        },
        NodeView::Binary { op, lhs, rhs } => {
            let AnyExpr::Nat(lhs) = lhs else { return None };
            let AnyExpr::Nat(rhs) = rhs else { return None };
            binary(
                op,
                exact_grid_value(arena, lhs, grid, grid_value, fixed)?,
                exact_grid_value(arena, rhs, grid, grid_value, fixed)?,
            )
        }
        NodeView::Nary {
            op: NaryOp::Product,
            operands,
        } => operands.iter().try_fold(
            ExactGridValue {
                value: 1,
                depends_on_grid: false,
            },
            |value, operand| {
                let AnyExpr::Nat(operand) = *operand else {
                    return None;
                };
                binary(
                    BinaryOp::Mul,
                    value,
                    exact_grid_value(arena, operand, grid, grid_value, fixed)?,
                )
            },
        ),
        NodeView::Select {
            cond,
            then,
            otherwise,
        } => {
            let condition = exact_bool_value(arena, cond, grid, grid_value, fixed)?;
            if condition.depends_on_grid {
                return None;
            }
            match condition.value {
                true => {
                    let AnyExpr::Nat(then) = then else {
                        return None;
                    };
                    exact_grid_value(arena, then, grid, grid_value, fixed)
                }
                false => {
                    let AnyExpr::Nat(otherwise) = otherwise else {
                        return None;
                    };
                    exact_grid_value(arena, otherwise, grid, grid_value, fixed)
                }
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod exact_launch_inversion_tests {
    use super::*;
    use seismic_lang::expr::{CmpOp, SymbolSort, SymbolValue};

    #[test]
    fn exact_cap_and_cap_plus_one_preserve_the_logical_work_relation() {
        let mut arena = ExprArena::default();
        let (_, grid_symbol) = arena.target_constant(SymbolSort::Nat);
        let grid = arena.nat_symbol(grid_symbol);
        let participants_per_block = arena.nat(128);
        let units = arena.nat_mul(grid, participants_per_block);
        let fixed = seismic_lang::expr::PartialAssignment::new();

        let at_cap = exact_grid_value(&arena, units, grid, 4, &fixed).unwrap();
        let above_cap = exact_grid_value(&arena, units, grid, 5, &fixed).unwrap();
        assert_eq!(at_cap.value, 512);
        assert_eq!(above_cap.value, 640);
        assert!(at_cap.depends_on_grid && above_cap.depends_on_grid);
    }

    #[test]
    fn repack_occupancy_expression_has_an_exact_grid_inversion() {
        let mut arena = ExprArena::default();
        let (_, grid_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, threads_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, shared_symbol) = arena.target_constant(SymbolSort::Nat);
        let grid = arena.nat_symbol(grid_symbol);
        let threads = arena.nat_symbol(threads_symbol);
        let dynamic_shared = arena.nat_symbol(shared_symbol);

        // Same semantic shape as CUDA's refined representation-repack demand:
        // occupancy is selected by fixed reflected workgroup/shared-memory
        // comparisons, then per-participant demand is scaled by launch work.
        let block_threads = arena.nat(128);
        let shared_limit = arena.nat(49_152);
        let within_shared = arena.nat_cmp(CmpOp::Le, dynamic_shared, shared_limit);
        let eight = arena.nat(8);
        let zero = arena.nat(0);
        let active_for_shared = arena.nat_select(within_shared, eight, zero);
        let matching_threads = arena.nat_cmp(CmpOp::Eq, threads, block_threads);
        let resident_blocks = arena.nat_select(matching_threads, active_for_shared, zero);
        let resident_participants = arena.nat_mul(resident_blocks, threads);
        let one = arena.nat(1);
        let resident_participants = arena.nat_max(resident_participants, one);
        let nominal_participants = arena.nat(1_536);
        let per_participant = arena.nat_ceil_div(nominal_participants, resident_participants);
        let launch_participants = arena.nat_mul(grid, threads);
        let repack_units = arena.nat_mul(per_participant, launch_participants);

        let mut fixed = seismic_lang::expr::PartialAssignment::new();
        fixed.bind(threads_symbol, SymbolValue::Nat(128));
        fixed.bind(shared_symbol, SymbolValue::Nat(0));

        let at_cap = exact_grid_value(&arena, repack_units, grid, 4, &fixed).unwrap();
        let above_cap = exact_grid_value(&arena, repack_units, grid, 5, &fixed).unwrap();
        assert_eq!(at_cap.value, 1_024);
        assert_eq!(above_cap.value, 1_280);
        assert!(at_cap.depends_on_grid && above_cap.depends_on_grid);
    }

    #[test]
    fn grid_dependent_select_is_not_claimed_monotone() {
        let mut arena = ExprArena::default();
        let (_, grid_symbol) = arena.target_constant(SymbolSort::Nat);
        let grid = arena.nat_symbol(grid_symbol);
        let four = arena.nat(4);
        let condition = arena.nat_cmp(CmpOp::Le, grid, four);
        let one = arena.nat(1);
        let zero = arena.nat(0);
        let expression = arena.nat_select(condition, one, zero);
        assert!(exact_grid_value(
            &arena,
            expression,
            grid,
            4,
            &seismic_lang::expr::PartialAssignment::new(),
        )
        .is_none());
    }

    #[test]
    fn total_conjunction_implies_any_exact_conjunct_subset() {
        let mut arena = ExprArena::default();
        let (_, x_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, y_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, z_symbol) = arena.target_constant(SymbolSort::Nat);
        let ten = arena.nat(10);
        let x = arena.nat_symbol(x_symbol);
        let y = arena.nat_symbol(y_symbol);
        let z = arena.nat_symbol(z_symbol);
        let x_bound = arena.nat_cmp(CmpOp::Le, x, ten);
        let y_bound = arena.nat_cmp(CmpOp::Le, y, ten);
        let z_bound = arena.nat_cmp(CmpOp::Le, z, ten);
        let required = arena.and(x_bound, y_bound);
        let target_tail = arena.and(y_bound, z_bound);
        let target = arena.and(x_bound, target_tail);
        let implication = arena.implies(target, required);
        assert!(matches!(
            arena.view(AnyExpr::Bool(implication)),
            NodeView::BoolConst(true)
        ));
    }

    #[test]
    fn wider_same_shape_allocation_certifies_narrower_storage_bound() {
        let mut arena = ExprArena::default();
        let (_, rows_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, columns_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, memory_symbol) = arena.target_constant(SymbolSort::Nat);
        let rows = arena.nat_symbol(rows_symbol);
        let columns = arena.nat_symbol(columns_symbol);
        let memory = arena.nat_symbol(memory_symbol);
        let elements = arena.nat_mul(rows, columns);
        let four = arena.nat(4);
        let two = arena.nat(2);
        let f32_bytes = arena.nat_mul(elements, four);
        let bf16_bytes = arena.nat_mul(two, elements);
        let abi_bound = arena.nat_cmp(CmpOp::Le, f32_bytes, memory);
        let internal_bound = arena.nat_cmp(CmpOp::Le, bf16_bytes, memory);
        let implication = arena.implies(abi_bound, internal_bound);
        assert!(matches!(
            arena.view(AnyExpr::Bool(implication)),
            NodeView::BoolConst(true)
        ));

        let reverse = arena.implies(internal_bound, abi_bound);
        assert!(!matches!(
            arena.view(AnyExpr::Bool(reverse)),
            NodeView::BoolConst(true)
        ));
    }

    #[test]
    fn zero_stride_address_axis_does_not_evaluate_a_partial_extent_span() {
        let mut arena = ExprArena::default();
        let (_, start_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, end_symbol) = arena.target_constant(SymbolSort::Nat);
        let start = arena.nat_symbol(start_symbol);
        let end = arena.nat_symbol(end_symbol);
        let partial_extent = arena.nat_sub(end, start);
        let zero = arena.nat(0);

        let span = internals::addressed_axis_span(&mut arena, partial_extent, zero);
        assert!(matches!(
            arena.view(AnyExpr::Nat(span)),
            NodeView::NatConst(0)
        ));
        let side_conditions = arena.side_conditions(AnyExpr::Nat(span));
        assert!(matches!(
            arena.view(AnyExpr::Bool(side_conditions)),
            NodeView::BoolConst(true)
        ));
    }

    #[test]
    fn zero_per_unit_scratch_does_not_evaluate_a_partial_launch_count() {
        let mut arena = ExprArena::default();
        let (_, start_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, end_symbol) = arena.target_constant(SymbolSort::Nat);
        let start = arena.nat_symbol(start_symbol);
        let end = arena.nat_symbol(end_symbol);
        let partial_count = arena.nat_sub(end, start);
        let zero = arena.nat(0);

        let bytes = internals::scaled_scratch_bytes(&mut arena, zero, partial_count);
        assert!(matches!(
            arena.view(AnyExpr::Nat(bytes)),
            NodeView::NatConst(0)
        ));
        let side_conditions = arena.side_conditions(AnyExpr::Nat(bytes));
        assert!(matches!(
            arena.view(AnyExpr::Bool(side_conditions)),
            NodeView::BoolConst(true)
        ));
    }
}

fn native_hard_constraints<B: Backend>(
    arena: &mut ExprArena,
    target: &DeviceContract<B>,
    schedule: &ParametricSchedule,
    kernels: &KernelArena<B>,
    launch_layouts: &[crate::storage::LaunchLocalLayout],
    native_kernels: &[Arc<crate::target::NativeKernel<B>>],
    include_grid_x: bool,
) -> BoolExpr {
    let mut constraints = Vec::new();
    for (launch, local_layout) in schedule.launches().iter().zip(launch_layouts) {
        let native = &native_kernels[launch.kernel.index() as usize];
        let kernel = kernels.kernel(launch.kernel);
        let domain = &native.contract().launch;
        let required_mode = match launch.mode {
            crate::schedule::LaunchMode::Independent => B::independent_launch_mode(),
            crate::schedule::LaunchMode::CooperativeGrid => {
                B::cooperative_launch_mode(target.facts())
                    .expect("cooperative launch was constructed for an unsupported device")
            }
        };
        assert!(
            domain.modes.contains(&required_mode),
            "reconciled native kernel does not admit its constructed launch mode"
        );
        let threads = arena.nat_product(&launch.workgroup);
        if kernel.interface().uses_subgroup {
            let width =
                arena.nat(u64::from(domain.subgroup_width.expect(
                    "subgroup-using native kernel closed without a subgroup width",
                )));
            constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Ge, threads, width));
            let remainder = arena.nat_rem(threads, width);
            let zero = arena.nat(0);
            constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, remainder, zero));
            if let Some(extent) = launch.parallel_extent {
                let remainder = arena.nat_rem(extent, width);
                constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, remainder, zero));
            }
        }
        let max_threads = arena.nat(domain.max_workgroup_threads);
        constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Le, threads, max_threads));
        for axis in 0..3 {
            let workgroup_max = arena.nat(domain.max_workgroup_size[axis]);
            constraints.push(arena.nat_cmp(
                seismic_lang::expr::CmpOp::Le,
                launch.workgroup[axis],
                workgroup_max,
            ));
            if axis != 0 || include_grid_x {
                let grid_max = arena.nat(domain.max_grid[axis]);
                constraints.push(arena.nat_cmp(
                    seismic_lang::expr::CmpOp::Le,
                    launch.grid[axis],
                    grid_max,
                ));
            }
        }
        let local_max = arena.nat(domain.max_dynamic_local_bytes);
        constraints.push(arena.nat_cmp(
            seismic_lang::expr::CmpOp::Le,
            local_layout.workgroup_bytes,
            local_max,
        ));
        constraints.extend(B::native_launch_constraints(
            target,
            arena,
            launch,
            local_layout,
            kernel,
            native,
        ));
    }
    arena.all(&constraints)
}

fn expression_detail(arena: &ExprArena, expression: AnyExpr, depth: usize) -> String {
    if depth == 0 {
        return format!("{expression:?}");
    }
    match arena.view(expression) {
        NodeView::NatConst(value) => value.to_string(),
        NodeView::IntConst(value) => value.to_string(),
        NodeView::BoolConst(value) => value.to_string(),
        NodeView::ScalarConst { dtype, bits } => format!("{dtype:?}(0x{bits:08x})"),
        NodeView::Symbol(symbol) => format!("{symbol:?}:{:?}", arena.symbol_kind(symbol)),
        NodeView::Unary { op, operand } => {
            format!("{op:?}({})", expression_detail(arena, operand, depth - 1))
        }
        NodeView::Binary { op, lhs, rhs } => format!(
            "({} {op:?} {})",
            expression_detail(arena, lhs, depth - 1),
            expression_detail(arena, rhs, depth - 1)
        ),
        NodeView::Nary { op, operands } => format!(
            "{op:?}({})",
            operands
                .iter()
                .map(|operand| expression_detail(arena, *operand, depth - 1))
                .collect::<Vec<_>>()
                .join(",")
        ),
        NodeView::Select {
            cond,
            then,
            otherwise,
        } => format!(
            "select({},{},{})",
            expression_detail(arena, AnyExpr::Bool(cond), depth - 1),
            expression_detail(arena, then, depth - 1),
            expression_detail(arena, otherwise, depth - 1)
        ),
        NodeView::Cmp { op, lhs, rhs } => format!(
            "({} {op:?} {})",
            expression_detail(arena, lhs, depth - 1),
            expression_detail(arena, rhs, depth - 1)
        ),
        NodeView::In { operand, values } => format!(
            "{} in {values:?}",
            expression_detail(arena, operand, depth - 1)
        ),
        NodeView::Fold {
            op,
            binder,
            start,
            extent,
            body,
        } => format!(
            "fold({op:?},{binder:?},start={},extent={},body={})",
            expression_detail(arena, AnyExpr::Nat(start), depth - 1),
            expression_detail(arena, AnyExpr::Nat(extent), depth - 1),
            expression_detail(arena, AnyExpr::Nat(body), depth - 1)
        ),
        NodeView::Duration(terms) => format!("duration({} terms)", terms.len()),
        NodeView::DurationScale { duration, by } => format!(
            "duration_scale({},{})",
            expression_detail(arena, AnyExpr::Duration(duration), depth - 1),
            expression_detail(arena, AnyExpr::Nat(by), depth - 1)
        ),
    }
}

/// The only implementation state accepted by planning and freezing. Every
/// kernel has a reconciled native artifact and all reflected limits have
/// already been folded into `hard_constraints`.
#[derive(Debug)]
pub struct Implementation<B: Backend> {
    draft: ImplementationDraft<B>,
    native_kernels: Vec<Arc<crate::target::NativeKernel<B>>>,
    total_launch_certificate: Option<TotalLaunchCertificate>,
}

impl<B: Backend> Implementation<B> {
    pub(crate) fn retained_metadata_bytes(&self) -> u64 {
        let draft = &self.draft;
        let mut bytes = std::mem::size_of_val(self)
            .saturating_add(draft.schedule.retained_bytes())
            .saturating_add(draft.kernels.retained_bytes())
            .saturating_add(draft.global_allocations.retained_bytes())
            .saturating_add(draft.local_allocations.retained_bytes())
            .saturating_add(
                draft.launch_layouts.capacity()
                    * std::mem::size_of::<crate::storage::LaunchLocalLayout>(),
            )
            .saturating_add(
                draft
                    .launch_layouts
                    .iter()
                    .map(|layout| {
                        layout.locals.capacity()
                            * std::mem::size_of::<crate::storage::LocalLayout>()
                            + layout
                                .locals
                                .iter()
                                .map(|local| {
                                    (local.extents.capacity() + local.strides.capacity())
                                        * std::mem::size_of::<NatExpr>()
                                })
                                .sum::<usize>()
                    })
                    .sum::<usize>(),
            )
            .saturating_add(
                draft.launch_scratch.capacity()
                    * std::mem::size_of::<crate::storage::LaunchScratchRequirements>(),
            )
            .saturating_add(
                draft.launch_abi.capacity()
                    * std::mem::size_of::<Vec<crate::storage::LaunchAbiRequirement>>(),
            )
            .saturating_add(
                draft
                    .launch_abi
                    .iter()
                    .map(|abi| {
                        abi.capacity() * std::mem::size_of::<crate::storage::LaunchAbiRequirement>()
                    })
                    .sum::<usize>(),
            )
            .saturating_add(
                draft.decisions.capacity() * std::mem::size_of::<(DecisionId, &'static str)>(),
            )
            .saturating_add(
                draft.duration_qualification.capacity() * std::mem::size_of::<BoolExpr>(),
            )
            .saturating_add(
                draft.provenance.callees.capacity() * std::mem::size_of::<StableFunctionId>(),
            )
            .saturating_add(
                draft.result_publications.capacity() * std::mem::size_of::<ResultPublication>(),
            )
            .saturating_add(
                draft
                    .result_publications
                    .iter()
                    .map(|result| result.path.capacity() * std::mem::size_of::<u32>())
                    .sum::<usize>(),
            )
            .saturating_add(
                self.native_kernels.capacity()
                    * std::mem::size_of::<Arc<crate::target::NativeKernel<B>>>(),
            );
        // The transfer owns nested output/effect/operation/evidence vectors;
        // its explicit estimator remains colocated with that representation.
        bytes = bytes.saturating_add(draft.numerical_transfer.retained_bytes());
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }
    pub fn identity(&self) -> &ImplementationIdentity {
        self.draft.identity()
    }
    pub fn semantic_coverage(&self) -> TargetPredicate {
        self.draft.semantic_coverage()
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        self.draft.schedule()
    }
    pub fn kernels(&self) -> &KernelArena<B> {
        self.draft.kernels()
    }
    pub(crate) fn native_kernels(&self) -> &[Arc<crate::target::NativeKernel<B>>] {
        &self.native_kernels
    }
    pub(crate) fn native_numerical_identity(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(b"seismic-native-numerical-set-v1");
        for kernel in &self.native_kernels {
            digest.update(kernel.contract().numerical_identity.fingerprint);
        }
        digest.finalize().into()
    }
    pub fn global_allocations(&self) -> &GlobalAllocationTopology {
        self.draft.global_allocations()
    }
    pub fn local_allocations(&self) -> &LocalAllocationTopology {
        self.draft.local_allocations()
    }
    pub fn launch_layouts(&self) -> &[crate::storage::LaunchLocalLayout] {
        self.draft.launch_layouts()
    }
    pub fn launch_scratch(&self) -> &[crate::storage::LaunchScratchRequirements] {
        self.draft.launch_scratch()
    }
    pub fn launch_abi(&self) -> &[Vec<crate::storage::LaunchAbiRequirement>] {
        self.draft.launch_abi()
    }
    pub fn decisions(&self) -> &[(DecisionId, &'static str)] {
        self.draft.decisions()
    }
    pub fn hard_constraints(&self) -> BoolExpr {
        self.draft.hard_constraints()
    }
    pub fn numerical_transfer(&self) -> &NumericalTransfer {
        self.draft.numerical_transfer()
    }
    pub(crate) fn authority(&self) -> ConstructionAuthority {
        self.draft.authority
    }
    pub(crate) fn numerical_role(&self) -> seismic_lang::entry::NumericalRole {
        self.draft.numerical_role
    }
    pub fn duration(&self) -> DurationExpr {
        self.draft.duration()
    }
    pub(crate) fn duration_qualification(&self) -> &[BoolExpr] {
        self.draft.duration_qualification()
    }
    pub fn provenance(&self) -> &ImplementationProvenance {
        self.draft.provenance()
    }
    pub(crate) fn result_publications(&self) -> &[ResultPublication] {
        self.draft.result_publications()
    }
}

#[derive(Clone, Debug)]
pub struct UniversalImplementation<B: Backend>(Arc<Implementation<B>>);

impl<B: Backend> UniversalImplementation<B> {
    fn new(implementation: Implementation<B>) -> Self {
        Self(Arc::new(implementation))
    }
    pub(crate) fn from_closed_reference(
        mut implementation: Implementation<B>,
        arena: &mut ExprArena,
        target_domain: BoolExpr,
        constants: &TargetConstants,
    ) -> Result<Self, crate::errors::PreparationError> {
        let certificate = implementation
            .total_launch_certificate
            .take()
            .ok_or_else(|| {
                crate::errors::PreparationError::UniversalClosure(
                    "implementation lacks a total launch certificate".into(),
                )
            })?;
        if certificate.maximum_grid_x.len() != implementation.schedule().launches().len()
            || certificate.maximum_grid_x.iter().any(|cap| *cap == 0)
        {
            return Err(crate::errors::PreparationError::UniversalClosure(
                "total launch certificate does not cover the closed schedule".into(),
            ));
        }
        if implementation.authority() != ConstructionAuthority::UniversalPortable
            || implementation.numerical_role() != seismic_lang::entry::NumericalRole::Reference
        {
            return Err(crate::errors::PreparationError::UniversalClosure(
                "implementation lacks checked reference provenance".into(),
            ));
        }
        if !implementation.decisions().is_empty() {
            return Err(crate::errors::PreparationError::UniversalClosure(
                "reference implementation contains finite decisions".into(),
            ));
        }
        if !implementation.numerical_transfer().is_exact() {
            return Err(crate::errors::PreparationError::UniversalClosure(
                "reference numerical transfer is not exact".into(),
            ));
        }
        let duration_qualification = arena.all(implementation.duration_qualification());
        let mut fixed = seismic_lang::expr::PartialAssignment::new();
        for (symbol, value) in constants.bindings() {
            fixed.bind(*symbol, *value);
        }
        let mut close = |expression: BoolExpr| {
            let expression = arena.partial(expression, &fixed);
            if arena.free_symbols(AnyExpr::Bool(expression)).is_empty() {
                if let Ok(value) =
                    arena.eval_bool(expression, &seismic_lang::expr::Assignment::new())
                {
                    return arena.bool(value);
                }
            }
            expression
        };
        let target_domain = close(target_domain);
        let semantic_coverage = close(implementation.semantic_coverage().node());
        let hard_constraints = close(implementation.hard_constraints());
        let duration_qualification = close(duration_qualification);
        let coverage = arena.all(&[semantic_coverage, hard_constraints, duration_qualification]);
        let total = arena.implies(target_domain, coverage);
        if !matches!(arena.view(AnyExpr::Bool(total)), NodeView::BoolConst(true)) {
            let symbol_summary = |expression: BoolExpr| {
                arena
                    .free_symbols(AnyExpr::Bool(expression))
                    .into_iter()
                    .map(|symbol| format!("{symbol:?}:{:?}", arena.symbol_kind(symbol)))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            return Err(crate::errors::PreparationError::UniversalClosure(
                format!(
                    "post-native legality or modeled-duration qualification is not constructionally total over TargetDomain (target={:?} target_symbols=[{}], semantic={:?}, hard={:?} hard_detail={} hard_symbols=[{}], duration={:?}, implication={:?})",
                    arena.view(AnyExpr::Bool(target_domain)),
                    symbol_summary(target_domain),
                    arena.view(AnyExpr::Bool(semantic_coverage)),
                    arena.view(AnyExpr::Bool(hard_constraints)),
                    expression_detail(arena, AnyExpr::Bool(hard_constraints), 8),
                    symbol_summary(hard_constraints),
                    arena.view(AnyExpr::Bool(duration_qualification)),
                    arena.view(AnyExpr::Bool(total)),
                ),
            ));
        }
        Ok(Self::new(implementation))
    }
    pub(crate) fn shared(&self) -> Arc<Implementation<B>> {
        self.0.clone()
    }
}

#[derive(Clone, Debug)]
pub struct OptimizedImplementation<B: Backend>(Arc<Implementation<B>>);

impl<B: Backend> OptimizedImplementation<B> {
    fn new(implementation: Implementation<B>) -> Self {
        Self(Arc::new(implementation))
    }
    pub(crate) fn from_closed(implementation: Implementation<B>) -> Self {
        Self::new(implementation)
    }
    pub(crate) fn as_inner(&self) -> &Implementation<B> {
        &self.0
    }
    pub(crate) fn shared(&self) -> Arc<Implementation<B>> {
        self.0.clone()
    }
}

/// Crate-private carrier for the builder's close.
pub(crate) struct ImplementationParts<B: Backend> {
    pub authority: ConstructionAuthority,
    pub numerical_role: seismic_lang::entry::NumericalRole,
    pub identity: ImplementationIdentity,
    pub semantic_coverage: TargetPredicate,
    pub schedule: ParametricSchedule,
    pub kernels: KernelArena<B>,
    pub global_allocations: GlobalAllocationTopology,
    pub local_allocations: LocalAllocationTopology,
    pub launch_layouts: Vec<crate::storage::LaunchLocalLayout>,
    pub launch_scratch: Vec<crate::storage::LaunchScratchRequirements>,
    pub launch_abi: Vec<Vec<crate::storage::LaunchAbiRequirement>>,
    pub decisions: Vec<(DecisionId, &'static str)>,
    pub hard_constraints: BoolExpr,
    pub numerical_transfer: NumericalTransfer,
    pub duration: DurationExpr,
    pub duration_qualification: Vec<BoolExpr>,
    pub provenance: ImplementationProvenance,
    pub result_publications: Vec<ResultPublication>,
}

/// Constructional coverage class consumed by plan-space construction. It is
/// not inferred from factory names or durations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ConstructionAuthority {
    UniversalPortable,
    Optimized,
}

/// What a factory sees.
pub struct FactoryRequest<'a, B: Backend> {
    /// The function body to implement.
    pub function: &'a SemanticFunction,
    pub program: &'a SemanticProgram,
    pub contract: &'a FunctionContract,
    pub machine: PlanningMachine<'a, B>,
    pub constants: &'a TargetConstants,
    pub precision: &'a PrecisionPolicy,
    /// The checked semantic candidate this construction realizes. Authored
    /// backend lowerings/helpers are not inferred from factory names.
    pub candidate_kind: CandidateKind,
    pub numerical_role: seismic_lang::entry::NumericalRole,
    /// Exact checked source applicability of this semantic candidate.
    pub semantic_coverage: TargetPredicate,
    /// Root entry, or a spliced call site with its argument bindings.
    pub site: CallSite<'a>,
}

/// The site an implementation is constructed for.
#[derive(Clone, Copy, Debug)]
pub enum CallSite<'a> {
    Root,
    Spliced {
        /// The call node in the parent function.
        call: NodeId,
        /// Parent values bound to the callee's parameters, in order.
        arguments: &'a [ValueBinding],
        /// Caller-owned destinations shared by every child alternative.
        results: &'a [ResultBinding],
    },
}

/// How a callee parameter is bound at a spliced call.
#[derive(Clone, Debug)]
pub enum ValueBinding {
    /// A global view of the parent (argument, result, or arena).
    View {
        view: AnyBufferView,
        layout: crate::storage::BufferViewLayout,
    },
    /// A scalar slot or symbol of the parent.
    Scalar(seismic_lang::expr::SymbolId),
    Range {
        start: seismic_lang::expr::SymbolId,
        end: seismic_lang::expr::SymbolId,
    },
}

/// Where a callee result is published in the caller. Every alternative
/// writes the same destination, so no option-dependent value join exists.
#[derive(Clone, Debug)]
pub enum ResultBinding {
    View {
        view: AnyBufferView,
        layout: crate::storage::BufferViewLayout,
    },
    Scalar(AnyScalarSlot),
    Range {
        start: AnyScalarSlot,
        end: AnyScalarSlot,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ScalarPublication {
    Scalar(AnyScalarSlot),
    Range {
        start: AnyScalarSlot,
        end: AnyScalarSlot,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PortablePreflightStatus {
    pub(crate) view: AnyBufferView,
    pub(crate) slot: AnyScalarSlot,
}

/// Core-derived contract of any semantic function, root or nested. It uses
/// semantic value identity; entry ABI identities exist only at root binding.
#[derive(Clone, Debug)]
pub struct FunctionContract {
    parameters: Vec<ContractParameter>,
    results: Vec<ContractResult>,
    effects: ContractEffects,
}

#[derive(Clone, Debug, Default)]
pub struct ContractEffects {
    pub reads: Vec<SemanticValueId>,
    pub writes: Vec<SemanticValueId>,
}

#[derive(Clone, Debug)]
pub struct ContractParameter {
    pub value: SemanticValueId,
    pub access: ParameterAccess,
    pub ty: SemanticType,
}

#[derive(Clone, Debug)]
pub struct ContractResult {
    pub value: SemanticValueId,
    pub ty: SemanticType,
    pub paths: Vec<Vec<u32>>,
}

impl FunctionContract {
    pub fn parameters(&self) -> &[ContractParameter] {
        &self.parameters
    }
    pub fn results(&self) -> &[ContractResult] {
        &self.results
    }
    pub fn effects(&self) -> &ContractEffects {
        &self.effects
    }

    pub(crate) fn derive(function: &SemanticFunction) -> Self {
        let parameters = function
            .parameters()
            .iter()
            .map(|parameter| ContractParameter {
                value: parameter.value,
                access: parameter.access,
                ty: function.value(parameter.value).ty.clone(),
            })
            .collect();
        let results = function
            .results()
            .iter()
            .enumerate()
            .map(|(ordinal, value)| ContractResult {
                value: *value,
                ty: function.value(*value).ty.clone(),
                paths: vec![vec![ordinal as u32]],
            })
            .collect();
        let mut effects = ContractEffects::default();
        fn visit(function: &SemanticFunction, region: RegionId, effects: &mut ContractEffects) {
            for (_, node) in function.nodes(region) {
                for event in node.events() {
                    let Some(value) = event.place() else { continue };
                    let collection = match event.access() {
                        AccessKind::Read => &mut effects.reads,
                        AccessKind::Write(_) | AccessKind::AtomicRmw { .. } => &mut effects.writes,
                        AccessKind::Barrier(_) => continue,
                    };
                    if !collection.contains(&value) {
                        collection.push(value);
                    }
                }
                match node.view() {
                    SemanticNodeView::If {
                        then, otherwise, ..
                    } => {
                        visit(function, then, effects);
                        visit(function, otherwise, effects);
                    }
                    SemanticNodeView::Loop { body, .. } => visit(function, body, effects),
                    _ => {}
                }
            }
        }
        visit(function, function.root(), &mut effects);
        Self {
            parameters,
            results,
            effects,
        }
    }
}

/// Applicability decision before construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Applicability {
    Applicable,
    NotApplicable { reason: String },
}

/// The factory boundary (§6.2). Implemented by the core's universal portable
/// factory and by backend structural factories (W5).
pub trait ImplementationFactory<B: Backend>: Send + Sync {
    fn identity(&self) -> FactoryIdentity;

    /// Structural preconditions, decided without constructing anything.
    fn applicable(&self, request: &FactoryRequest<'_, B>) -> Applicability;

    /// Constructs exactly one closed implementation for an applicable
    /// request. Construction is infallible after applicability; violating
    /// checked semantics or the factory contract is a compiler-author panic.
    /// A factory may construct several alternatives
    /// by being registered several times with different identities, or by
    /// exposing finite decisions inside one implementation.
    fn construct(
        &self,
        request: &FactoryRequest<'_, B>,
        builder: ImplementationBuilder<'_, B>,
    ) -> ImplementationDraft<B>;
}

/// The one way to build an implementation. Created by `plan_space`.
pub struct ImplementationBuilder<'a, B: Backend> {
    inner: internals::Builder<'a, B>,
}

impl<'a, B: Backend> ImplementationBuilder<'a, B> {
    pub(crate) fn new(
        arena: &'a mut ExprArena,
        program: &'a SemanticProgram,
        function: &'a SemanticFunction,
        machine: PlanningMachine<'a, B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
        semantic_coverage: TargetPredicate,
        factory: FactoryIdentity,
        authority: ConstructionAuthority,
        numerical_role: seismic_lang::entry::NumericalRole,
        site: CallSite<'a>,
        root_schema: Option<&'a CallSchema>,
        budget: ConstructionBudget,
    ) -> Self {
        Self {
            inner: internals::Builder::new(
                arena,
                program,
                function,
                machine.device(),
                machine.execution(),
                constants,
                precision,
                semantic_coverage,
                factory,
                authority,
                numerical_role,
                site,
                root_schema,
                budget,
            ),
        }
    }
    pub(crate) fn new_universal(
        arena: &'a mut ExprArena,
        program: &'a SemanticProgram,
        function: &'a SemanticFunction,
        machine: PlanningMachine<'a, B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
        semantic_coverage: TargetPredicate,
        factory: FactoryIdentity,
        numerical_role: seismic_lang::entry::NumericalRole,
        site: CallSite<'a>,
        root_schema: Option<&'a CallSchema>,
        budget: ConstructionBudget,
    ) -> Self {
        Self {
            inner: internals::Builder::new(
                arena,
                program,
                function,
                machine.device(),
                machine.execution(),
                constants,
                precision,
                semantic_coverage,
                factory,
                ConstructionAuthority::UniversalPortable,
                numerical_role,
                site,
                root_schema,
                budget,
            ),
        }
    }
    pub fn arena(&mut self) -> &mut ExprArena {
        self.inner.arena()
    }
    pub fn target(&self) -> &DeviceContract<B> {
        self.inner.target()
    }
    pub(crate) fn portable_target_ref(&self) -> &'a DeviceContract<B> {
        self.inner.target
    }
    /// Resolves the unique authored helper body that may be inlined into the
    /// current kernel segment. Helpers crossing a kernel cut still use normal
    /// call splicing; this surface exists specifically so lane/subgroup-local
    /// values never acquire a schedule ABI.
    pub(crate) fn authored_helper(
        &self,
        family: FamilyId,
        backend: BackendName,
    ) -> &'a SemanticFunction {
        let mut matching = self
            .inner
            .program
            .family(family)
            .candidates()
            .iter()
            .filter(|candidate| {
                candidate.kind == CandidateKind::Helper { backend }
                    && candidate
                        .requires
                        .iter()
                        .all(|capability| self.inner.target.supports_capability(*capability))
                    && matches!(
                        self.inner
                            .arena
                            .view(AnyExpr::Bool(candidate.applicability.node())),
                        NodeView::BoolConst(true)
                    )
            });
        let candidate = matching
            .next()
            .expect("checked authored helper has no unconditionally applicable backend body");
        assert!(
            matching.next().is_none(),
            "checked authored helper resolution is ambiguous"
        );
        self.inner.program.function(candidate.function)
    }
    pub fn constants(&self) -> &TargetConstants {
        self.inner.constants()
    }

    /// Declares a finite decision. The domain must be non-empty.
    pub fn decision(&mut self, name: &'static str, domain: FiniteDomain) -> DecisionId {
        self.inner.decision(name, domain)
    }

    /// Adds a hard constraint (target limits are added by the builder from
    /// the topology; factories add only structural legality).
    pub fn constrain(&mut self, predicate: BoolExpr) {
        self.inner.constrain(predicate)
    }

    // ----- storage -------------------------------------------------------------

    /// The view of a call argument tensor, typed by its representation. A
    /// A representation mismatch after `applicable` is a factory invariant
    /// violation; runtime representation dispatch happens before construction.
    pub fn argument_view<R: Representation>(
        &mut self,
        parameter: SemanticValueId,
    ) -> BufferViewId<R> {
        self.inner.argument_view::<R>(parameter)
    }
    /// The view of a result leaf, allocated by the runtime.
    pub fn result_view<R: Representation>(&mut self, value: SemanticValueId) -> BufferViewId<R> {
        self.inner.result_view::<R>(value)
    }
    /// A new arena allocation.
    pub fn arena_allocation(&mut self, bytes: NatExpr, alignment: u64) -> GlobalAllocationId {
        self.inner.arena_allocation(bytes, alignment)
    }
    /// A typed dense view over an allocation.
    pub fn view<R: Representation>(
        &mut self,
        allocation: GlobalAllocationId,
        offset: NatExpr,
        extents: Vec<NatExpr>,
    ) -> BufferViewId<R> {
        self.inner.view::<R>(allocation, offset, extents)
    }
    /// A strided sub-view of an existing view.
    pub fn subview<R: Representation>(
        &mut self,
        base: BufferViewId<R>,
        offset: NatExpr,
        extents: Vec<NatExpr>,
        strides: Vec<NatExpr>,
    ) -> BufferViewId<R> {
        self.inner.subview::<R>(base, offset, extents, strides)
    }
    /// The view a semantic place value denotes (parameter, owned allocation,
    /// or already-materialized value).
    pub fn place_view<R: Representation>(&mut self, value: SemanticValueId) -> BufferViewId<R> {
        if !self.inner.has_view(value) {
            self.materialize_tensor(value, R::id());
        }
        self.inner.typed_view::<R>(value)
    }
    pub fn bind_view<R: Representation>(&self, view: BufferViewId<R>) -> ValueBinding {
        self.inner.bind_view(view.erase())
    }
    /// The caller-owned publication slot of one scalar result. All spliced
    /// alternatives for the call receive this same destination.
    pub fn result_slot<T: ScalarType>(&mut self, value: SemanticValueId) -> ScalarSlotId<T> {
        self.inner.result_slot::<T>(value)
    }

    // ----- kernels and schedule ----------------------------------------------

    pub fn kernel(&mut self) -> KernelBuilder<'_, B> {
        self.inner.kernel()
    }
    pub fn schedule(&mut self) -> ScheduleBuilder<'_, B> {
        self.inner.schedule()
    }

    pub(crate) fn portable_kernel(&mut self) -> crate::kernel::internals::PortableBuilder<'_, B> {
        self.inner.portable_kernel()
    }
    pub(crate) fn portable_function(&self) -> &SemanticFunction {
        self.inner.function
    }
    /// The checked function reference has the builder's construction
    /// lifetime rather than the borrow of `self`, so a portable lowerer may
    /// inspect it while mutating the builder without an aliasing workaround.
    pub(crate) fn portable_function_ref(&self) -> &'a SemanticFunction {
        self.inner.function
    }
    pub(crate) fn portable_view(&mut self, value: SemanticValueId) -> AnyBufferView {
        if !self.inner.has_view(value) {
            let representation = self.inner.tensor_representation(value);
            self.materialize_tensor(value, representation);
        }
        self.inner.any_view(value)
    }
    pub(crate) fn portable_existing_binding(&self, value: SemanticValueId) -> Option<ValueBinding> {
        self.inner.portable_existing_binding(value)
    }
    pub(crate) fn portable_allocate_tensor(&mut self, value: SemanticValueId) -> AnyBufferView {
        self.inner.portable_allocate_tensor(value)
    }
    pub(crate) fn portable_binding(&self, value: SemanticValueId) -> ValueBinding {
        self.inner.portable_binding(value)
    }
    pub(crate) fn portable_publish(&mut self, value: SemanticValueId) -> ScalarPublication {
        self.inner.portable_publish(value)
    }
    pub(crate) fn portable_define_view(
        &mut self,
        value: SemanticValueId,
        base: AnyBufferView,
        offset: NatExpr,
        extents: Vec<NatExpr>,
        strides: Vec<NatExpr>,
    ) -> AnyBufferView {
        self.inner
            .portable_define_view(value, base, offset, extents, strides)
    }
    pub(crate) fn portable_define_represented_view(
        &mut self,
        value: SemanticValueId,
        base: AnyBufferView,
        offset: NatExpr,
        extents: Vec<NatExpr>,
        strides: Vec<NatExpr>,
    ) -> AnyBufferView {
        self.inner
            .portable_define_represented_view(value, base, offset, extents, strides)
    }
    pub(crate) fn portable_layout(&self, view: AnyBufferView) -> crate::storage::BufferViewLayout {
        self.inner.portable_layout(view)
    }
    pub(crate) fn portable_preflight_status(&mut self) -> PortablePreflightStatus {
        self.inner.portable_preflight_status()
    }
    pub(crate) fn portable_finish_preflight(
        &mut self,
        status: PortablePreflightStatus,
        site: crate::kernel::ops::CheckSite,
    ) {
        self.inner.portable_finish_preflight(status, site)
    }
    pub(crate) fn portable_branch<T, E>(
        &mut self,
        condition: BoolExpr,
        then: impl FnOnce(&mut Self) -> Result<T, E>,
        otherwise: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<(T, T), E> {
        let parent = self.inner.schedule_region;
        let (then_region, else_region) = self.inner.portable_begin_branch(condition);
        self.inner.schedule_region = then_region;
        let then_value = match then(self) {
            Ok(value) => value,
            Err(error) => {
                self.inner.schedule_region = parent;
                return Err(error);
            }
        };
        self.inner.schedule_region = else_region;
        let else_value = match otherwise(self) {
            Ok(value) => value,
            Err(error) => {
                self.inner.schedule_region = parent;
                return Err(error);
            }
        };
        self.inner.schedule_region = parent;
        Ok((then_value, else_value))
    }
    pub(crate) fn portable_repeat<T, E>(
        &mut self,
        start: NatExpr,
        end: NatExpr,
        body: impl FnOnce(&mut Self, crate::schedule::LoopBinding) -> Result<T, E>,
    ) -> Result<T, E> {
        let parent = self.inner.schedule_region;
        let (body_region, binding) = self.inner.portable_begin_repeat(start, end);
        self.inner.schedule_region = body_region;
        let value = match body(self, binding) {
            Ok(value) => value,
            Err(error) => {
                self.inner.schedule_region = parent;
                return Err(error);
            }
        };
        self.inner.schedule_region = parent;
        Ok(value)
    }
    pub(crate) fn materialize_tensor(
        &mut self,
        value: SemanticValueId,
        required: seismic_lang::ids::RepresentationId,
    ) {
        let tensor = self.inner.tensor_semantics(value);
        assert_eq!(
            tensor.representation, required,
            "materialization representation differs from the checked semantic tensor"
        );
        crate::portable::realize_tensor(self, value, tensor)
    }

    /// Resolves a call node: constructs every applicable child alternative
    /// for the callee family through the registry and splices them under a
    /// new finite decision, composing all facts (§6.4). Returns the decision
    /// so the schedule can `choose` on it, and the spliced results' views.
    pub fn splice_call(&mut self, call: NodeId, arguments: &[ValueBinding]) -> SplicedCall<B> {
        self.inner.splice_call(call, arguments)
    }

    /// Closes the implementation. Requires the closed schedule and the
    /// semantic coverage predicate; numerical transfer, resource
    /// constraints, lifetimes, and identity are derived here.
    pub fn close(self, schedule: ClosedSchedule) -> ImplementationDraft<B> {
        self.inner.close(schedule)
    }
}

/// The result of splicing a call.
#[derive(Debug)]
pub struct SplicedCall<B: Backend> {
    pub decision: Option<DecisionId>,
    /// Caller-owned tensor or scalar result destinations.
    pub results: Vec<(SemanticValueId, ResultBinding)>,
    alternatives: Vec<(i64, crate::schedule::ImportedSchedule)>,
    marker: std::marker::PhantomData<B>,
}

impl<B: Backend> SplicedCall<B> {
    /// Inserts the already-closed child alternatives at this exact lexical
    /// schedule region. Consumes the token, so a call is scheduled once.
    pub fn schedule(self, schedule: &mut ScheduleBuilder<'_, B>) {
        schedule.splice(self.decision, self.alternatives)
    }
}

impl<B: Backend> fmt::Debug for ImplementationBuilder<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImplementationBuilder").finish()
    }
}

/// Deterministic root enumeration used by plan-space construction. Exactly
/// one checked reference portable body is constructed through the sealed
/// universal path; every other applicable backend factory is an optimized
/// peer. Declines occur only before a builder exists.
pub(crate) fn construct_root_implementations<B: Backend>(
    arena: &mut ExprArena,
    program: &SemanticProgram,
    schema: &CallSchema,
    machine: PlanningMachine<'_, B>,
    constants: &TargetConstants,
    precision: &PrecisionPolicy,
    budget: ConstructionBudget,
    mut retain_universal: impl FnMut(
        ImplementationDraft<B>,
        &mut ExprArena,
    ) -> Result<(), crate::errors::PreparationError>,
    mut retain_optimized: impl FnMut(
        ImplementationDraft<B>,
        &mut ExprArena,
    ) -> Result<bool, crate::errors::PreparationError>,
) -> Result<(), crate::errors::PreparationError> {
    let target = machine.device();
    let family = program.family(program.root());
    let reference = family.reference();
    let reference_candidate = reference.candidate();
    let reference_function = program.function(reference.function());
    let reference_contract = FunctionContract::derive(reference_function);
    let reference_factory = crate::portable::PortableFactory;
    let reference_request = FactoryRequest {
        function: reference_function,
        program,
        contract: &reference_contract,
        machine,
        constants,
        precision,
        candidate_kind: reference_candidate.kind,
        numerical_role: reference_candidate.numerical,
        semantic_coverage: reference_candidate.applicability,
        site: CallSite::Root,
    };
    if let Applicability::NotApplicable { reason } =
        reference_factory.applicable(&reference_request)
    {
        panic!("checked reference portable body declined construction: {reason}");
    }
    let reference_builder = ImplementationBuilder::new_universal(
        arena,
        program,
        reference_function,
        machine,
        constants,
        precision,
        reference_candidate.applicability,
        <crate::portable::PortableFactory as ImplementationFactory<B>>::identity(
            &reference_factory,
        ),
        reference_candidate.numerical,
        CallSite::Root,
        Some(schema),
        budget.clone(),
    );
    let universal = reference_factory.construct(&reference_request, reference_builder);
    retain_universal(universal, arena)?;
    'candidates: for candidate in std::iter::once(reference_candidate).chain(family.alternatives())
    {
        let is_reference = std::ptr::eq(candidate, reference_candidate);
        let backend_match = match candidate.kind {
            CandidateKind::Portable => true,
            CandidateKind::Lowering { backend } | CandidateKind::Helper { backend } => {
                backend == B::NAME
            }
        };
        if !backend_match
            || candidate
                .requires
                .iter()
                .any(|capability| !target.supports_capability(*capability))
        {
            continue;
        }
        let function = program.function(candidate.function);
        let contract = FunctionContract::derive(function);
        let site = CallSite::Root;
        if !is_reference {
            if !budget.borrow_mut().admit_optional_implementation() {
                break;
            }
            let factory: &dyn ImplementationFactory<B> = match candidate.kind {
                CandidateKind::Portable => &crate::portable::PortableFactory,
                CandidateKind::Lowering { .. } | CandidateKind::Helper { .. } => {
                    &crate::portable::AuthoredSemanticFactory
                }
            };
            let request = FactoryRequest {
                function,
                program,
                contract: &contract,
                machine,
                constants,
                precision,
                candidate_kind: candidate.kind,
                numerical_role: candidate.numerical,
                semantic_coverage: candidate.applicability,
                site,
            };
            match factory.applicable(&request) {
                Applicability::Applicable => {
                    let started = std::time::Instant::now();
                    let builder = ImplementationBuilder::new(
                        arena,
                        program,
                        function,
                        machine,
                        constants,
                        precision,
                        candidate.applicability,
                        factory.identity(),
                        ConstructionAuthority::Optimized,
                        candidate.numerical,
                        site,
                        Some(schema),
                        budget.clone(),
                    );
                    let draft = factory.construct(&request, builder);
                    if !budget
                        .borrow_mut()
                        .record_implementation_construction(started.elapsed())
                    {
                        retain_optimized(draft, arena)?;
                        break;
                    }
                    if !retain_optimized(draft, arena)? {
                        break;
                    }
                }
                Applicability::NotApplicable { reason } => {
                    panic!("checked portable alternative declined construction: {reason}")
                }
            }
        }
        if candidate.kind == CandidateKind::Portable {
            if !budget.borrow_mut().admit_optional_implementation() {
                break;
            }
            let factory = crate::portable::PortableParallelFactory;
            let request = FactoryRequest {
                function,
                program,
                contract: &contract,
                machine,
                constants,
                precision,
                candidate_kind: candidate.kind,
                numerical_role: candidate.numerical,
                semantic_coverage: candidate.applicability,
                site,
            };
            match factory.applicable(&request) {
                Applicability::Applicable => {
                    let started = std::time::Instant::now();
                    let builder =
                        ImplementationBuilder::new(
                            arena,
                            program,
                            function,
                            machine,
                            constants,
                            precision,
                            candidate.applicability,
                            <crate::portable::PortableParallelFactory as ImplementationFactory<
                                B,
                            >>::identity(&factory),
                            ConstructionAuthority::Optimized,
                            candidate.numerical,
                            site,
                            Some(schema),
                            budget.clone(),
                        );
                    let draft = factory.construct(&request, builder);
                    if !budget
                        .borrow_mut()
                        .record_implementation_construction(started.elapsed())
                    {
                        retain_optimized(draft, arena)?;
                        break;
                    }
                    if !retain_optimized(draft, arena)? {
                        break;
                    }
                }
                Applicability::NotApplicable { reason } => {
                    panic!("checked portable body declined parallel construction: {reason}")
                }
            }
        }
        // Backend structural policies are additive alternatives to portable
        // semantics. They never stand in for an authored lowering/helper,
        // whose body is guaranteed above by AuthoredSemanticFactory.
        if candidate.kind != CandidateKind::Portable {
            continue;
        }
        for factory in target.registry().factories() {
            let request = FactoryRequest {
                function,
                program,
                contract: &contract,
                machine,
                constants,
                precision,
                candidate_kind: candidate.kind,
                numerical_role: candidate.numerical,
                semantic_coverage: candidate.applicability,
                site,
            };
            match factory.applicable(&request) {
                Applicability::Applicable => {
                    if !budget.borrow_mut().admit_optional_implementation() {
                        break 'candidates;
                    }
                    let started = std::time::Instant::now();
                    let builder = ImplementationBuilder::new(
                        arena,
                        program,
                        function,
                        machine,
                        constants,
                        precision,
                        candidate.applicability,
                        factory.identity(),
                        ConstructionAuthority::Optimized,
                        candidate.numerical,
                        site,
                        Some(schema),
                        budget.clone(),
                    );
                    let draft = factory.construct(&request, builder);
                    if !budget
                        .borrow_mut()
                        .record_implementation_construction(started.elapsed())
                    {
                        retain_optimized(draft, arena)?;
                        break 'candidates;
                    }
                    if !retain_optimized(draft, arena)? {
                        break 'candidates;
                    }
                }
                Applicability::NotApplicable { .. } => {}
            }
        }
    }
    Ok(())
}

mod internals {
    use super::*;
    use crate::kernel::internals as kernel_internals;
    use crate::numerics::{
        ConditionalNumericalTransfer, ErrorBound, NumericalEffect, OutputTransfer,
    };
    use crate::schedule::ScheduleConstruction;
    use crate::schedule::ScheduleStep;
    use crate::storage::{AllocationLiveness, GlobalBufferKind, TopologyBuilder};
    use seismic_lang::expr::{AnyExpr, CmpOp, RootName, SymbolSort};
    use seismic_lang::registry;
    use std::collections::HashMap;

    struct ConstructedChild<B: Backend> {
        implementation: ImplementationDraft<B>,
        /// Result identities belong to the child's semantic program owner and
        /// therefore cannot be compared with the caller's result identities.
        /// Publication is an ordinal contract.
        results: Vec<SemanticValueId>,
    }

    pub(super) struct Builder<'a, B: Backend> {
        owner: OwnerToken,
        pub(super) arena: &'a mut ExprArena,
        pub(super) program: &'a SemanticProgram,
        pub(super) function: &'a SemanticFunction,
        contract: FunctionContract,
        pub(super) target: &'a DeviceContract<B>,
        execution: &'a ExecutionProfile<B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
        semantic_coverage: TargetPredicate,
        factory: FactoryIdentity,
        authority: ConstructionAuthority,
        numerical_role: seismic_lang::entry::NumericalRole,
        topology: TopologyBuilder,
        value_views: HashMap<SemanticValueId, AnyBufferView>,
        scalar_symbols: HashMap<SemanticValueId, ValueBinding>,
        result_slots: HashMap<SemanticValueId, ScalarPublication>,
        pending_result_paths: HashMap<SemanticValueId, Vec<Vec<u32>>>,
        kernels: Vec<crate::kernel::Kernel<B>>,
        kernel_state: crate::kernel::internals::KernelState<B>,
        schedule: Option<ScheduleConstruction<B>>,
        pub(super) schedule_region: u32,
        decisions: Vec<(DecisionId, &'static str)>,
        constraints: Vec<BoolExpr>,
        child_durations: Vec<DurationExpr>,
        child_duration_qualification: Vec<BoolExpr>,
        callees: Vec<StableFunctionId>,
        conditional_child_transfers: Vec<ConditionalNumericalTransfer>,
        budget: ConstructionBudget,
    }

    impl<'a, B: Backend> Builder<'a, B> {
        pub(super) fn new(
            arena: &'a mut ExprArena,
            program: &'a SemanticProgram,
            function: &'a SemanticFunction,
            target: &'a DeviceContract<B>,
            execution: &'a ExecutionProfile<B>,
            constants: &'a TargetConstants,
            precision: &'a PrecisionPolicy,
            semantic_coverage: TargetPredicate,
            factory: FactoryIdentity,
            authority: ConstructionAuthority,
            numerical_role: seismic_lang::entry::NumericalRole,
            site: CallSite<'a>,
            root_schema: Option<&'a CallSchema>,
            budget: ConstructionBudget,
        ) -> Self {
            let owner = OwnerToken::fresh();
            let mut contract = FunctionContract::derive(function);
            if let Some(schema) = root_schema {
                for result in &mut contract.results {
                    result.paths = schema
                        .results()
                        .iter()
                        .filter_map(|schema| {
                            (schema.value == result.value).then_some(schema.path.clone())
                        })
                        .collect();
                }
                assert!(
                    schema.results().iter().all(|schema| contract
                        .results
                        .iter()
                        .any(|result| result.value == schema.value)),
                    "call schema publishes a value absent from the root function results"
                );
            }
            let disjoint = root_schema
                .map(|schema| {
                    schema
                        .aliases()
                        .iter()
                        .filter_map(|rule| match rule {
                            AliasRule::Disjoint(a, b) => Some((*a, *b)),
                            AliasRule::MayOverlap(_, _) => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut topology =
                TopologyBuilder::new(owner, disjoint, matches!(site, CallSite::Spliced { .. }));
            let mut value_views = HashMap::new();
            let mut scalar_symbols = HashMap::new();
            let mut pending_result_paths: HashMap<SemanticValueId, Vec<Vec<u32>>> = HashMap::new();
            for parameter in &contract.parameters {
                match &parameter.ty {
                    SemanticType::Tensor(tensor) => {
                        let abi = root_schema
                            .and_then(|schema| {
                                schema
                                    .parameters()
                                    .iter()
                                    .find(|p| p.value == parameter.value)
                            })
                            .map(|p| p.id);
                        if let Some(schema) = root_schema {
                            let p = schema
                                .parameters()
                                .iter()
                                .find(|p| p.value == parameter.value)
                                .unwrap_or_else(|| {
                                    panic!("root function parameter is absent from CallSchema")
                                });
                            match &p.kind {
                                ParameterKind::Tensor {
                                    representation,
                                    axes,
                                    ..
                                } => {
                                    assert_eq!(
                                        *representation, tensor.representation,
                                        "root parameter representation differs from semantic contract"
                                    );
                                    assert_eq!(
                                        axes, &tensor.axes,
                                        "root parameter axes differ from semantic contract"
                                    );
                                }
                                _ => {
                                    panic!("tensor semantic parameter is non-tensor in CallSchema")
                                }
                            }
                        }
                        let (kind, imported_layout) = match site {
                            CallSite::Root => (
                                GlobalBufferKind::Argument {
                                    value: parameter.value,
                                    abi,
                                },
                                None,
                            ),
                            CallSite::Spliced { arguments, .. } => {
                                let ordinal = contract
                                    .parameters
                                    .iter()
                                    .position(|p| p.value == parameter.value)
                                    .expect("contract parameter exists");
                                match arguments.get(ordinal) {
                                    Some(ValueBinding::View { view, layout }) => {
                                        assert_eq!(
                                            view.representation, tensor.representation,
                                            "spliced tensor argument representation mismatch"
                                        );
                                        (
                                            GlobalBufferKind::Imported {
                                                value: parameter.value,
                                                source: *view,
                                            },
                                            Some(layout.clone()),
                                        )
                                    }
                                    _ => panic!("spliced tensor parameter has no view binding"),
                                }
                            }
                        };
                        let bytes = crate::storage::tensor_bytes(
                            arena,
                            tensor.representation,
                            &tensor.axes,
                        );
                        let allocation = topology.allocate(
                            kind,
                            bytes,
                            crate::storage::representation_alignment(tensor.representation),
                        );
                        let zero = arena.nat(0);
                        let view = match imported_layout {
                            Some(layout) => topology.strided_view(
                                allocation,
                                tensor.representation,
                                zero,
                                tensor.axes.clone(),
                                layout.strides,
                            ),
                            None => topology.dense_view(
                                arena,
                                allocation,
                                tensor.representation,
                                zero,
                                tensor.axes.clone(),
                            ),
                        };
                        value_views.insert(
                            parameter.value,
                            AnyBufferView::new(owner, view, tensor.representation),
                        );
                    }
                    SemanticType::Scalar(_) | SemanticType::Index { .. } => {
                        let symbol = match site {
                            CallSite::Root => {
                                let schema =
                                    root_schema.expect("root builder requires its CallSchema");
                                let p = schema
                                    .parameters()
                                    .iter()
                                    .find(|p| p.value == parameter.value)
                                    .unwrap_or_else(|| {
                                        panic!("root scalar parameter is absent from CallSchema")
                                    });
                                match &p.kind {
                                    ParameterKind::Scalar { symbol, .. }
                                    | ParameterKind::Index { symbol, .. } => *symbol,
                                    _ => panic!(
                                        "scalar semantic parameter has a non-scalar ABI kind"
                                    ),
                                }
                            }
                            CallSite::Spliced { arguments, .. } => {
                                let ordinal = contract
                                    .parameters
                                    .iter()
                                    .position(|p| p.value == parameter.value)
                                    .expect("contract parameter exists");
                                match arguments.get(ordinal) {
                                    Some(ValueBinding::Scalar(symbol)) => *symbol,
                                    _ => panic!("spliced scalar parameter has no scalar binding"),
                                }
                            }
                        };
                        scalar_symbols.insert(parameter.value, ValueBinding::Scalar(symbol));
                    }
                    SemanticType::Range { .. } => {
                        let binding = match site {
                            CallSite::Root => {
                                let schema =
                                    root_schema.expect("root builder requires its CallSchema");
                                let parameter = schema
                                    .parameters()
                                    .iter()
                                    .find(|candidate| candidate.value == parameter.value)
                                    .unwrap_or_else(|| {
                                        panic!("root range parameter is absent from CallSchema")
                                    });
                                match &parameter.kind {
                                    ParameterKind::Range { start, end, .. } => {
                                        ValueBinding::Range {
                                            start: *start,
                                            end: *end,
                                        }
                                    }
                                    _ => {
                                        panic!("range semantic parameter has a non-range ABI kind")
                                    }
                                }
                            }
                            CallSite::Spliced { arguments, .. } => {
                                let ordinal = contract
                                    .parameters
                                    .iter()
                                    .position(|candidate| candidate.value == parameter.value)
                                    .expect("contract parameter exists");
                                match arguments.get(ordinal) {
                                    Some(ValueBinding::Range { start, end }) => {
                                        ValueBinding::Range {
                                            start: *start,
                                            end: *end,
                                        }
                                    }
                                    _ => panic!("spliced range parameter has no range binding"),
                                }
                            }
                        };
                        scalar_symbols.insert(parameter.value, binding);
                    }
                    SemanticType::Tuple(_) | SemanticType::Opaque { .. } | SemanticType::Void => {}
                }
            }
            for result in &contract.results {
                if let SemanticType::Tensor(tensor) = &result.ty {
                    let abi_path = root_schema
                        .and_then(|schema| {
                            schema.results().iter().find(|r| r.value == result.value)
                        })
                        .map(|r| r.path.clone());
                    if matches!(site, CallSite::Root) {
                        if let Some(view) = value_views.get(&result.value).copied() {
                            if let Some(path) = abi_path {
                                topology.publish_result(arena, view, path);
                            }
                            continue;
                        }
                        if matches!(tensor.storage, TensorStorage::View { .. }) {
                            if let Some(path) = abi_path {
                                pending_result_paths
                                    .entry(result.value)
                                    .or_default()
                                    .push(path);
                            }
                            continue;
                        }
                    }
                    let (kind, imported_layout) = match site {
                        CallSite::Root => (
                            GlobalBufferKind::Result {
                                value: result.value,
                            },
                            None,
                        ),
                        CallSite::Spliced { results, .. } => {
                            let ordinal = contract
                                .results
                                .iter()
                                .position(|r| r.value == result.value)
                                .expect("contract result exists");
                            match results.get(ordinal) {
                                Some(ResultBinding::View { view, layout }) => {
                                    assert_eq!(
                                        view.representation, tensor.representation,
                                        "spliced tensor result representation mismatch"
                                    );
                                    (
                                        GlobalBufferKind::Imported {
                                            value: result.value,
                                            source: *view,
                                        },
                                        Some(layout.clone()),
                                    )
                                }
                                _ => panic!(
                                    "spliced tensor result has no caller-owned view destination"
                                ),
                            }
                        }
                    };
                    let bytes =
                        crate::storage::tensor_bytes(arena, tensor.representation, &tensor.axes);
                    let allocation = topology.allocate(
                        kind,
                        bytes,
                        crate::storage::representation_alignment(tensor.representation),
                    );
                    let zero = arena.nat(0);
                    let view = match imported_layout {
                        Some(layout) => topology.strided_view(
                            allocation,
                            tensor.representation,
                            zero,
                            tensor.axes.clone(),
                            layout.strides,
                        ),
                        None => topology.dense_view(
                            arena,
                            allocation,
                            tensor.representation,
                            zero,
                            tensor.axes.clone(),
                        ),
                    };
                    let view = AnyBufferView::new(owner, view, tensor.representation);
                    value_views.insert(result.value, view);
                    if let (CallSite::Root, Some(path)) = (site, abi_path) {
                        topology.publish_result(arena, view, path);
                    }
                }
            }
            let kernel_zero = arena.nat(0);
            Self {
                owner,
                arena,
                program,
                function,
                contract,
                target,
                execution,
                constants,
                precision,
                semantic_coverage,
                factory,
                authority,
                numerical_role,
                topology,
                value_views,
                scalar_symbols,
                result_slots: HashMap::new(),
                pending_result_paths,
                kernels: Vec::new(),
                kernel_state: crate::kernel::internals::KernelState::new(
                    owner,
                    0,
                    kernel_zero,
                    target.addressable_resources().len(),
                ),
                schedule: Some(ScheduleConstruction::new(owner)),
                schedule_region: 0,
                decisions: Vec::new(),
                constraints: Vec::new(),
                child_durations: Vec::new(),
                child_duration_qualification: Vec::new(),
                callees: Vec::new(),
                conditional_child_transfers: Vec::new(),
                budget,
            }
        }

        pub(super) fn arena(&mut self) -> &mut ExprArena {
            self.arena
        }
        pub(super) fn target(&self) -> &DeviceContract<B> {
            self.target
        }
        pub(super) fn constants(&self) -> &TargetConstants {
            self.constants
        }
        pub(super) fn decision(&mut self, name: &'static str, domain: FiniteDomain) -> DecisionId {
            let decision = self.arena.decision(domain);
            self.decisions.push((decision, name));
            decision
        }
        pub(super) fn constrain(&mut self, predicate: BoolExpr) {
            self.constraints.push(predicate)
        }
        pub(super) fn argument_view<R: Representation>(
            &mut self,
            value: SemanticValueId,
        ) -> BufferViewId<R> {
            assert!(
                self.contract.parameters.iter().any(|p| p.value == value),
                "factory requested a value that is not a function parameter"
            );
            self.typed_view::<R>(value)
        }
        pub(super) fn result_view<R: Representation>(
            &mut self,
            value: SemanticValueId,
        ) -> BufferViewId<R> {
            assert!(
                self.contract.results.iter().any(|r| r.value == value),
                "factory requested a value that is not a function result"
            );
            self.typed_view::<R>(value)
        }
        pub(super) fn arena_allocation(
            &mut self,
            bytes: NatExpr,
            alignment: u64,
        ) -> GlobalAllocationId {
            self.topology
                .allocate(GlobalBufferKind::Arena, bytes, alignment)
        }
        pub(super) fn view<R: Representation>(
            &mut self,
            allocation: GlobalAllocationId,
            offset: NatExpr,
            extents: Vec<NatExpr>,
        ) -> BufferViewId<R> {
            let index = self
                .topology
                .dense_view(self.arena, allocation, R::id(), offset, extents);
            BufferViewId::new(self.owner, index)
        }
        pub(super) fn subview<R: Representation>(
            &mut self,
            base: BufferViewId<R>,
            offset: NatExpr,
            extents: Vec<NatExpr>,
            strides: Vec<NatExpr>,
        ) -> BufferViewId<R> {
            assert_eq!(
                base.owner(),
                self.owner,
                "subview base belongs to another implementation"
            );
            let index = self
                .topology
                .subview(self.arena, base.index(), offset, extents, strides);
            BufferViewId::new(self.owner, index)
        }
        pub(super) fn has_view(&self, value: SemanticValueId) -> bool {
            self.value_views.contains_key(&value)
        }
        pub(super) fn any_view(&self, value: SemanticValueId) -> AnyBufferView {
            self.value_views
                .get(&value)
                .copied()
                .unwrap_or_else(|| panic!("semantic tensor has no realized view"))
        }
        pub(super) fn tensor_semantics(
            &self,
            value: SemanticValueId,
        ) -> seismic_lang::entry::TensorSemantics {
            match &self.function.value(value).ty {
                SemanticType::Tensor(tensor) => tensor.clone(),
                _ => panic!("materialization requested for a non-tensor semantic value"),
            }
        }
        pub(super) fn tensor_representation(
            &self,
            value: SemanticValueId,
        ) -> seismic_lang::ids::RepresentationId {
            self.tensor_semantics(value).representation
        }
        pub(super) fn bind_view(&self, view: AnyBufferView) -> ValueBinding {
            ValueBinding::View {
                view,
                layout: self.topology.view_layout(view).clone(),
            }
        }
        pub(super) fn result_slot<T: ScalarType>(
            &mut self,
            value: SemanticValueId,
        ) -> ScalarSlotId<T> {
            let result = self
                .contract
                .results
                .iter()
                .find(|result| result.value == value)
                .unwrap_or_else(|| {
                    panic!("factory requested a scalar that is not a function result")
                });
            let dtype = match result.ty {
                SemanticType::Scalar(dtype) => dtype,
                SemanticType::Index { .. } if T::SYMBOL_SORT == SymbolSort::Nat => T::DTYPE,
                _ => panic!("factory requested a non-scalar result as a scalar slot"),
            };
            assert_eq!(dtype, T::DTYPE, "factory scalar result dtype mismatch");
            let publication = *self.result_slots.entry(value).or_insert_with(|| {
                ScalarPublication::Scalar(
                    self.schedule
                        .as_mut()
                        .expect("schedule construction exists until close")
                        .slot_any(self.arena, dtype, T::SYMBOL_SORT),
                )
            });
            let ScalarPublication::Scalar(slot) = publication else {
                panic!("factory requested a range publication as a scalar slot")
            };
            ScalarSlotId::from_any(slot)
        }
        pub(super) fn kernel(&mut self) -> KernelBuilder<'_, B> {
            kernel_internals::open(
                self.owner,
                self.arena,
                self.topology.views(),
                &mut self.kernels,
                &mut self.kernel_state,
                self.target.facts(),
                self.target.addressable_resources(),
                self.target.vectors(),
            )
        }
        pub(super) fn schedule(&mut self) -> ScheduleBuilder<'_, B> {
            self.schedule
                .as_mut()
                .expect("schedule construction exists until close")
                .builder_at(self.arena, self.topology.views(), self.schedule_region)
        }
        pub(crate) fn portable_kernel(
            &mut self,
        ) -> crate::kernel::internals::PortableBuilder<'_, B> {
            kernel_internals::open_portable(
                self.owner,
                self.arena,
                self.topology.views(),
                &mut self.kernels,
                &mut self.kernel_state,
                self.target.facts(),
                self.target.addressable_resources(),
                self.target.vectors(),
            )
        }
        pub(crate) fn portable_allocate_tensor(&mut self, value: SemanticValueId) -> AnyBufferView {
            if let Some(view) = self.value_views.get(&value).copied() {
                return view;
            }
            let tensor = match &self.function.value(value).ty {
                SemanticType::Tensor(tensor) => tensor.clone(),
                _ => panic!("portable allocation requested for a non-tensor value"),
            };
            let bytes =
                crate::storage::tensor_bytes(self.arena, tensor.representation, &tensor.axes);
            let allocation = self.topology.allocate(
                GlobalBufferKind::Arena,
                bytes,
                crate::storage::representation_alignment(tensor.representation),
            );
            let zero = self.arena.nat(0);
            let index = self.topology.dense_view(
                self.arena,
                allocation,
                tensor.representation,
                zero,
                tensor.axes,
            );
            let view = AnyBufferView::new(self.owner, index, tensor.representation);
            self.value_views.insert(value, view);
            view
        }
        pub(crate) fn portable_binding(&self, value: SemanticValueId) -> ValueBinding {
            self.portable_existing_binding(value)
                .unwrap_or_else(|| panic!("semantic value has no portable binding"))
        }
        pub(crate) fn portable_existing_binding(
            &self,
            value: SemanticValueId,
        ) -> Option<ValueBinding> {
            if let Some(view) = self.value_views.get(&value).copied() {
                return Some(self.bind_view(view));
            }
            if let Some(binding) = self.scalar_symbols.get(&value) {
                return Some(binding.clone());
            }
            match self.result_slots.get(&value).copied() {
                Some(ScalarPublication::Scalar(slot)) => Some(ValueBinding::Scalar(slot.symbol)),
                Some(ScalarPublication::Range { start, end }) => Some(ValueBinding::Range {
                    start: start.symbol,
                    end: end.symbol,
                }),
                None => None,
            }
        }
        pub(crate) fn portable_publish(&mut self, value: SemanticValueId) -> ScalarPublication {
            if let Some(publication) = self.result_slots.get(&value).copied() {
                return publication;
            }
            let publication = match self.function.value(value).ty {
                SemanticType::Scalar(dtype) => ScalarPublication::Scalar(
                    self.schedule.as_mut().expect("schedule exists").slot_any(
                        self.arena,
                        dtype,
                        SymbolSort::Scalar(dtype),
                    ),
                ),
                SemanticType::Index { .. } => ScalarPublication::Scalar(
                    self.schedule.as_mut().expect("schedule exists").slot_any(
                        self.arena,
                        DType::U32,
                        SymbolSort::Nat,
                    ),
                ),
                SemanticType::Range { .. } => ScalarPublication::Range {
                    start: self.schedule.as_mut().expect("schedule exists").slot_any(
                        self.arena,
                        DType::U32,
                        SymbolSort::Nat,
                    ),
                    end: self.schedule.as_mut().expect("schedule exists").slot_any(
                        self.arena,
                        DType::U32,
                        SymbolSort::Nat,
                    ),
                },
                _ => panic!("portable scalar publication requested for a non-scalar value"),
            };
            self.result_slots.insert(value, publication);
            publication
        }
        pub(crate) fn portable_define_view(
            &mut self,
            value: SemanticValueId,
            base: AnyBufferView,
            offset: NatExpr,
            extents: Vec<NatExpr>,
            strides: Vec<NatExpr>,
        ) -> AnyBufferView {
            assert!(
                !self.value_views.contains_key(&value),
                "semantic tensor value was realized twice"
            );
            let representation = match &self.function.value(value).ty {
                SemanticType::Tensor(tensor) => tensor.representation,
                _ => panic!("view value is not a tensor"),
            };
            assert_eq!(
                representation, base.representation,
                "view transform changed representation without an explicit plane/decode operation"
            );
            let index = self
                .topology
                .subview(self.arena, base.index, offset, extents, strides);
            let view = AnyBufferView::new(self.owner, index, representation);
            self.value_views.insert(value, view);
            view
        }
        pub(crate) fn portable_define_represented_view(
            &mut self,
            value: SemanticValueId,
            base: AnyBufferView,
            offset: NatExpr,
            extents: Vec<NatExpr>,
            strides: Vec<NatExpr>,
        ) -> AnyBufferView {
            assert!(
                !self.value_views.contains_key(&value),
                "semantic tensor value was realized twice"
            );
            let representation = match &self.function.value(value).ty {
                SemanticType::Tensor(tensor) => tensor.representation,
                _ => panic!("view value is not a tensor"),
            };
            let base_layout = self.topology.view_layout(base).clone();
            let absolute = self.arena.nat_add(base_layout.offset, offset);
            let index = self.topology.strided_view(
                base_layout.allocation,
                representation,
                absolute,
                extents,
                strides,
            );
            let view = AnyBufferView::new(self.owner, index, representation);
            self.value_views.insert(value, view);
            view
        }
        pub(crate) fn portable_layout(
            &self,
            view: AnyBufferView,
        ) -> crate::storage::BufferViewLayout {
            self.topology.view_layout(view).clone()
        }
        pub(crate) fn portable_preflight_status(&mut self) -> PortablePreflightStatus {
            let representation = registry::dense(DType::U32);
            let bytes = self.arena.nat(u64::from(DType::U32.bytes()));
            let allocation = self.topology.allocate(
                GlobalBufferKind::Arena,
                bytes,
                crate::storage::representation_alignment(representation),
            );
            let zero = self.arena.nat(0);
            let one = self.arena.nat(1);
            let index =
                self.topology
                    .dense_view(self.arena, allocation, representation, zero, vec![one]);
            let view = AnyBufferView::new(self.owner, index, representation);
            let mut schedule = self.schedule();
            let slot = schedule.slot_any(DType::U32, SymbolSort::Scalar(DType::U32));
            schedule.fill_constant_any(view, seismic_lang::intrinsics::FillConstant::Zero);
            PortablePreflightStatus { view, slot }
        }
        pub(crate) fn portable_finish_preflight(
            &mut self,
            status: PortablePreflightStatus,
            site: crate::kernel::ops::CheckSite,
        ) {
            assert_eq!(status.view.owner(), self.owner);
            assert_eq!(status.slot.owner(), self.owner);
            let zero = self.arena.nat(0);
            let mut schedule = self.schedule();
            schedule.scalar_read_any(status.view, vec![zero], status.slot);
            schedule.check_zero_any(status.slot, site);
        }
        pub(crate) fn portable_begin_branch(&mut self, condition: BoolExpr) -> (u32, u32) {
            self.schedule
                .as_mut()
                .expect("schedule exists")
                .begin_branch(self.schedule_region, condition)
        }
        pub(crate) fn portable_begin_repeat(
            &mut self,
            start: NatExpr,
            end: NatExpr,
        ) -> (u32, crate::schedule::LoopBinding) {
            self.schedule
                .as_mut()
                .expect("schedule exists")
                .begin_repeat(self.arena, self.schedule_region, start, end)
        }
        pub(super) fn splice_call(
            &mut self,
            call: NodeId,
            arguments: &[ValueBinding],
        ) -> SplicedCall<B> {
            let node = self.function.node(call);
            let (family_id, call_inputs, call_outputs) = match node.view() {
                SemanticNodeView::Call {
                    family,
                    inputs,
                    outputs,
                } => (family, inputs, outputs),
                _ => panic!("splice_call requires a semantic Call node"),
            };
            assert_eq!(
                arguments.len(),
                call_inputs.len(),
                "call argument binding count differs from semantic call inputs"
            );
            let mut results = Vec::with_capacity(call_outputs.len());
            for output in call_outputs {
                let binding = match &self.function.value(*output).ty {
                    SemanticType::Tensor(tensor) => {
                        if !self.value_views.contains_key(output) {
                            let bytes = crate::storage::tensor_bytes(
                                self.arena,
                                tensor.representation,
                                &tensor.axes,
                            );
                            let allocation = self.topology.allocate(
                                GlobalBufferKind::Arena,
                                bytes,
                                crate::storage::representation_alignment(tensor.representation),
                            );
                            let zero = self.arena.nat(0);
                            let index = self.topology.dense_view(
                                self.arena,
                                allocation,
                                tensor.representation,
                                zero,
                                tensor.axes.clone(),
                            );
                            self.value_views.insert(
                                *output,
                                AnyBufferView::new(self.owner, index, tensor.representation),
                            );
                        }
                        let view = self.value_views[output];
                        ResultBinding::View {
                            view,
                            layout: self.topology.view_layout(view).clone(),
                        }
                    }
                    SemanticType::Scalar(dtype) => {
                        let publication = *self.result_slots.entry(*output).or_insert_with(|| {
                            ScalarPublication::Scalar(
                                self.schedule.as_mut().expect("schedule exists").slot_any(
                                    self.arena,
                                    *dtype,
                                    SymbolSort::Scalar(*dtype),
                                ),
                            )
                        });
                        let ScalarPublication::Scalar(slot) = publication else {
                            panic!("scalar call result has a range publication")
                        };
                        ResultBinding::Scalar(slot)
                    }
                    SemanticType::Index { .. } => {
                        let publication = *self.result_slots.entry(*output).or_insert_with(|| {
                            ScalarPublication::Scalar(
                                self.schedule.as_mut().expect("schedule exists").slot_any(
                                    self.arena,
                                    DType::U32,
                                    SymbolSort::Nat,
                                ),
                            )
                        });
                        let ScalarPublication::Scalar(slot) = publication else {
                            panic!("index call result has a range publication")
                        };
                        ResultBinding::Scalar(slot)
                    }
                    SemanticType::Range { .. } => {
                        let publication = *self.result_slots.entry(*output).or_insert_with(|| {
                            ScalarPublication::Range {
                                start: self.schedule.as_mut().expect("schedule exists").slot_any(
                                    self.arena,
                                    DType::U32,
                                    SymbolSort::Nat,
                                ),
                                end: self.schedule.as_mut().expect("schedule exists").slot_any(
                                    self.arena,
                                    DType::U32,
                                    SymbolSort::Nat,
                                ),
                            }
                        });
                        let ScalarPublication::Range { start, end } = publication else {
                            panic!("range call result has a scalar publication")
                        };
                        ResultBinding::Range { start, end }
                    }
                    other => {
                        panic!("call result {output:?} has unsupported publication type {other:?}")
                    }
                };
                results.push((*output, binding));
            }
            let result_bindings: Vec<ResultBinding> =
                results.iter().map(|(_, binding)| binding.clone()).collect();
            let family = self.program.family(family_id);
            let reference = family.reference().candidate();
            let machine = self.target.planning_with(self.execution);
            let mut children = Vec::new();
            for candidate in std::iter::once(reference).chain(family.alternatives()) {
                let is_reference = std::ptr::eq(candidate, reference);
                let backend_match = match candidate.kind {
                    CandidateKind::Portable => true,
                    CandidateKind::Lowering { backend } | CandidateKind::Helper { backend } => {
                        backend == B::NAME
                    }
                };
                if !backend_match
                    || candidate
                        .requires
                        .iter()
                        .any(|capability| !self.target.supports_capability(*capability))
                {
                    continue;
                }
                let function = self.program.function(candidate.function);
                let contract = FunctionContract::derive(function);
                if contract.parameters.len() != arguments.len()
                    || contract.results.len() != result_bindings.len()
                {
                    panic!("semantic call family candidate contract arity differs from call node");
                }
                let site = CallSite::Spliced {
                    call,
                    arguments,
                    results: &result_bindings,
                };
                if candidate.kind == CandidateKind::Portable {
                    let factory = crate::portable::PortableFactory;
                    let request = FactoryRequest {
                        function,
                        program: self.program,
                        contract: &contract,
                        machine,
                        constants: self.constants,
                        precision: self.precision,
                        candidate_kind: candidate.kind,
                        numerical_role: candidate.numerical,
                        semantic_coverage: candidate.applicability,
                        site,
                    };
                    if factory.applicable(&request) == Applicability::Applicable {
                        if self.authority == ConstructionAuthority::UniversalPortable
                            && !is_reference
                        {
                            continue;
                        }
                        let construction_started = if is_reference {
                            None
                        } else {
                            if !self.budget.borrow_mut().admit_optional_implementation() {
                                continue;
                            }
                            Some(std::time::Instant::now())
                        };
                        let builder = if self.authority == ConstructionAuthority::UniversalPortable
                        {
                            ImplementationBuilder::new_universal(
                                self.arena,
                                self.program,
                                function,
                                machine,
                                self.constants,
                                self.precision,
                                candidate.applicability,
                                <crate::portable::PortableFactory as ImplementationFactory<B>>::identity(&factory),
                                candidate.numerical,
                                site,
                                None,
                                self.budget.clone(),
                            )
                        } else {
                            ImplementationBuilder::new(
                                self.arena,
                                self.program,
                                function,
                                machine,
                                self.constants,
                                self.precision,
                                candidate.applicability,
                                <crate::portable::PortableFactory as ImplementationFactory<B>>::identity(&factory),
                                ConstructionAuthority::Optimized,
                                candidate.numerical,
                                site,
                                None,
                                self.budget.clone(),
                            )
                        };
                        children.push(ConstructedChild {
                            implementation: factory.construct(&request, builder),
                            results: contract.results.iter().map(|result| result.value).collect(),
                        });
                        if let Some(started) = construction_started {
                            self.budget
                                .borrow_mut()
                                .record_implementation_construction(started.elapsed());
                        }
                        if self.authority != ConstructionAuthority::UniversalPortable {
                            if !self.budget.borrow_mut().admit_optional_implementation() {
                                continue;
                            }
                            let factory = crate::portable::PortableParallelFactory;
                            let started = std::time::Instant::now();
                            let builder = ImplementationBuilder::new(
                                self.arena,
                                self.program,
                                function,
                                machine,
                                self.constants,
                                self.precision,
                                candidate.applicability,
                                <crate::portable::PortableParallelFactory as ImplementationFactory<B>>::identity(&factory),
                                ConstructionAuthority::Optimized,
                                candidate.numerical,
                                site,
                                None,
                                self.budget.clone(),
                            );
                            children.push(ConstructedChild {
                                implementation: factory.construct(&request, builder),
                                results: contract
                                    .results
                                    .iter()
                                    .map(|result| result.value)
                                    .collect(),
                            });
                            self.budget
                                .borrow_mut()
                                .record_implementation_construction(started.elapsed());
                        }
                    }
                } else if self.authority != ConstructionAuthority::UniversalPortable {
                    let factory = crate::portable::AuthoredSemanticFactory;
                    let request = FactoryRequest {
                        function,
                        program: self.program,
                        contract: &contract,
                        machine,
                        constants: self.constants,
                        precision: self.precision,
                        candidate_kind: candidate.kind,
                        numerical_role: candidate.numerical,
                        semantic_coverage: candidate.applicability,
                        site,
                    };
                    match factory.applicable(&request) {
                        Applicability::Applicable => {
                            if !self.budget.borrow_mut().admit_optional_implementation() {
                                continue;
                            }
                            let started = std::time::Instant::now();
                            let builder = ImplementationBuilder::new(
                                self.arena,
                                self.program,
                                function,
                                machine,
                                self.constants,
                                self.precision,
                                candidate.applicability,
                                <crate::portable::AuthoredSemanticFactory as ImplementationFactory<B>>::identity(&factory),
                                ConstructionAuthority::Optimized,
                                candidate.numerical,
                                site,
                                None,
                                self.budget.clone(),
                            );
                            children.push(ConstructedChild {
                                implementation: factory.construct(&request, builder),
                                results: contract
                                    .results
                                    .iter()
                                    .map(|result| result.value)
                                    .collect(),
                            });
                            self.budget
                                .borrow_mut()
                                .record_implementation_construction(started.elapsed());
                        }
                        Applicability::NotApplicable { reason } => panic!(
                            "checked authored backend candidate declined construction: {reason}"
                        ),
                    }
                }
                if self.authority != ConstructionAuthority::UniversalPortable
                    && candidate.kind == CandidateKind::Portable
                {
                    for factory in self.target.registry().factories() {
                        let request = FactoryRequest {
                            function,
                            program: self.program,
                            contract: &contract,
                            machine,
                            constants: self.constants,
                            precision: self.precision,
                            candidate_kind: candidate.kind,
                            numerical_role: candidate.numerical,
                            semantic_coverage: candidate.applicability,
                            site,
                        };
                        if factory.applicable(&request) == Applicability::Applicable {
                            if !self.budget.borrow_mut().admit_optional_implementation() {
                                break;
                            }
                            let started = std::time::Instant::now();
                            let builder = ImplementationBuilder::new(
                                self.arena,
                                self.program,
                                function,
                                machine,
                                self.constants,
                                self.precision,
                                candidate.applicability,
                                factory.identity(),
                                ConstructionAuthority::Optimized,
                                candidate.numerical,
                                site,
                                None,
                                self.budget.clone(),
                            );
                            children.push(ConstructedChild {
                                implementation: factory.construct(&request, builder),
                                results: contract
                                    .results
                                    .iter()
                                    .map(|result| result.value)
                                    .collect(),
                            });
                            self.budget
                                .borrow_mut()
                                .record_implementation_construction(started.elapsed());
                        }
                    }
                }
            }
            assert!(
                !children.is_empty(),
                "checked call family has no applicable portable child implementation"
            );
            if self.authority == ConstructionAuthority::UniversalPortable {
                assert_eq!(
                    children.len(),
                    1,
                    "universal portable call splicing admits exactly the reference portable child"
                );
            }
            let decision = if children.len() == 1 {
                None
            } else {
                let domain = FiniteDomain::new((0..children.len() as i64).collect())
                    .expect("child alternatives are nonempty");
                Some(self.decision("call implementation", domain))
            };
            let mut alternatives = Vec::with_capacity(children.len());
            for (ordinal, child) in children.into_iter().enumerate() {
                let child_results = child.results;
                let parts = child.implementation.into_parts();
                let child_role = parts.numerical_role;
                let child_transfer = parts.numerical_transfer;
                // Spliced argument/result allocations already carry their
                // caller-owned source view. Import consumes those typed proxy
                // allocations directly; no semantic-id side map survives.
                let view_map = self
                    .topology
                    .import(self.arena, parts.global_allocations, &[]);
                let mut child_kernels = kernel_internals::arena_into_kernels(parts.kernels);
                let kernel_ids: Vec<_> = (0..child_kernels.len())
                    .map(|offset| {
                        crate::kernel::KernelId::new(
                            self.owner,
                            (self.kernels.len() + offset) as u32,
                        )
                    })
                    .collect();
                let mut forced_slots = Vec::new();
                for (result_ordinal, _child_value) in child_results.iter().enumerate() {
                    let Some((_, parent)) = results.get(result_ordinal) else {
                        continue;
                    };
                    let child_publication = parts
                        .result_publications
                        .get(result_ordinal)
                        .map(|publication| publication.binding);
                    match (parent, child_publication) {
                        (
                            ResultBinding::Scalar(parent),
                            Some(PublishedResult::Scalar { slot: child, .. }),
                        ) => forced_slots.push((child.index, *parent)),
                        (
                            ResultBinding::Range {
                                start: parent_start,
                                end: parent_end,
                            },
                            Some(PublishedResult::Range { start, end }),
                        ) => {
                            forced_slots.push((start.index, *parent_start));
                            forced_slots.push((end.index, *parent_end));
                        }
                        (ResultBinding::View { .. }, Some(PublishedResult::Buffer { .. })) => {}
                        _ => panic!(
                            "child result #{result_ordinal} publication does not match the caller destination"
                        ),
                    }
                }
                let imported = self.schedule.as_mut().expect("schedule exists").import(
                    self.arena,
                    parts.schedule,
                    &kernel_ids,
                    &view_map,
                    &forced_slots,
                );
                for (offset, kernel) in child_kernels.iter_mut().enumerate() {
                    kernel_internals::data_mut(kernel).rebrand(
                        self.owner,
                        kernel_ids[offset].index(),
                        |view| view_map[view.index as usize],
                        |slot| imported.slots[slot.index as usize],
                    );
                }
                self.kernels.extend(child_kernels);
                self.decisions.extend(parts.decisions);
                self.callees.push(parts.provenance.root);
                self.callees.extend(parts.provenance.callees);
                if let Some(decision) = decision {
                    let selected = self.arena.decision_is(decision, ordinal as i64);
                    self.constraints
                        .push(self.arena.implies(selected, parts.semantic_coverage.node()));
                    self.constraints
                        .push(self.arena.implies(selected, parts.hard_constraints));
                    let zero = self.arena.duration(&[]);
                    self.child_durations.push(self.arena.duration_select(
                        selected,
                        parts.duration,
                        zero,
                    ));
                    let qualifications = parts
                        .duration_qualification
                        .into_iter()
                        .map(|qualification| self.arena.implies(selected, qualification))
                        .collect::<Vec<_>>();
                    self.child_duration_qualification.extend(qualifications);
                } else {
                    self.constraints.push(parts.semantic_coverage.node());
                    self.constraints.push(parts.hard_constraints);
                    self.child_durations.push(parts.duration);
                    self.child_duration_qualification
                        .extend(parts.duration_qualification);
                }
                self.conditional_child_transfers
                    .push(ConditionalNumericalTransfer {
                        selection: decision.map(|decision| (decision, ordinal as i64)),
                        role: child_role,
                        transfer: Box::new(child_transfer),
                    });
                alternatives.push((ordinal as i64, imported));
            }
            SplicedCall {
                decision,
                results,
                alternatives,
                marker: std::marker::PhantomData,
            }
        }
        pub(super) fn close(mut self, closed: ClosedSchedule) -> ImplementationDraft<B> {
            let coverage = self.semantic_coverage;
            if self.authority == ConstructionAuthority::UniversalPortable {
                assert!(
                    self.decisions.is_empty(),
                    "universal portable implementation cannot contain finite decisions"
                );
            }
            let schedule = self
                .schedule
                .take()
                .expect("schedule construction exists until close")
                .finish(closed);
            for (value, paths) in std::mem::take(&mut self.pending_result_paths) {
                let view = self
                    .value_views
                    .get(&value)
                    .copied()
                    .unwrap_or_else(|| panic!("returned tensor view was never realized"));
                for path in paths {
                    self.topology.publish_result(self.arena, view, path);
                }
            }
            self.validate_schedule(&schedule);
            let mut liveness = vec![Vec::new(); self.topology.allocation_count() as usize];
            for (view, at) in schedule.direct_view_uses() {
                let allocation = self.topology.view_layout(*view).allocation;
                liveness[allocation.index() as usize].push(at.clone());
            }
            for (launch_id, at) in schedule.launch_uses() {
                let launch = schedule.launch(*launch_id);
                let kernel = self
                    .kernels
                    .get(launch.kernel.index() as usize)
                    .expect("launch kernel was closed into this implementation");
                for binding in &kernel.interface().bindings {
                    let allocation = self.topology.view_layout(binding.view).allocation;
                    liveness[allocation.index() as usize].push(at.clone());
                }
            }
            let liveness: Vec<AllocationLiveness> =
                liveness.into_iter().map(AllocationLiveness::new).collect();
            let slots = self.derive_reuse(&liveness);
            let launch_layouts: Vec<_> = schedule
                .launches()
                .iter()
                .map(|launch| {
                    let kernel = self
                        .kernels
                        .get(launch.kernel.index() as usize)
                        .expect("launch kernel belongs to this implementation");
                    crate::storage::derive_launch_local_layout(
                        self.arena,
                        kernel.locals(),
                        kernel_internals::data(kernel).intrinsic_resources(),
                    )
                })
                .collect();
            let launch_abi: Vec<_> = schedule
                .launches()
                .iter()
                .map(|launch| {
                    let kernel = self
                        .kernels
                        .get(launch.kernel.index() as usize)
                        .expect("launch kernel belongs to this implementation");
                    self.target
                        .kernel_abi_layout(kernel)
                        .allocations
                        .into_iter()
                        .map(|allocation| crate::storage::LaunchAbiRequirement {
                            role: allocation.role,
                            bytes: self.arena.nat(allocation.bytes),
                            alignment: allocation.alignment,
                        })
                        .collect()
                })
                .collect();
            let launch_scratch = self.derive_launch_scratch(&schedule, &launch_layouts);
            self.derive_constraints(&schedule, &launch_layouts, &launch_scratch, &launch_abi);
            let raw_constraints = self.arena.all(&self.constraints);
            let side_conditions = self.arena.side_conditions(AnyExpr::Bool(raw_constraints));
            let hard_constraints = self.arena.and(side_conditions, raw_constraints);
            let structural_duration = self.derive_structural_duration(&schedule, &launch_layouts);
            self.child_durations.push(structural_duration.duration);
            self.child_duration_qualification
                .extend(structural_duration.qualification);
            let duration = std::mem::take(&mut self.child_durations)
                .into_iter()
                .reduce(|a, b| self.arena.duration_add(a, b))
                .unwrap_or_else(|| self.arena.duration(&[]));
            let duration_qualification = self.child_duration_qualification.clone();
            let numerical_transfer = self.derive_numerics();
            let roots = self.register_roots(
                &schedule,
                &launch_layouts,
                &launch_scratch,
                &launch_abi,
                hard_constraints,
                coverage,
                duration,
                &duration_qualification,
                &numerical_transfer,
            );
            let expression_digest = self.arena.canonical_digest(&roots).bytes();
            let mut digest = crate::identity::StructureDigest::new("seismic-implementation-v2");
            digest.bytes(self.factory.name.as_bytes());
            digest.bytes(self.factory.revision.as_bytes());
            let (reference_math_version, reference_math_digest) =
                crate::kernel::reference_math_identity();
            digest.bytes(reference_math_version.as_bytes());
            digest.bytes(&reference_math_digest);
            digest.bytes(&expression_digest);
            digest_structure(&mut digest, &self, &schedule, &liveness, &slots);
            digest_numerical(&mut digest, &self, &numerical_transfer);
            let identity = ImplementationIdentity {
                factory: self.factory.clone(),
                structure: digest.finish(),
            };
            let local_allocations = LocalAllocationTopology::new(
                self.owner,
                self.kernels.iter().map(|k| k.locals().to_vec()).collect(),
            );
            let mut result_publications = Vec::new();
            for result in &self.contract.results {
                for path in &result.paths {
                    let binding = match &result.ty {
                        SemanticType::Tensor(_) => {
                            let publication = self
                                .topology
                                .result_views()
                                .iter()
                                .find(|publication| &publication.path == path)
                                .expect("closed result tensor path has no published view");
                            PublishedResult::Buffer {
                                view: publication.view,
                                bytes: publication.bytes,
                            }
                        }
                        SemanticType::Scalar(dtype) => {
                            let ScalarPublication::Scalar(slot) = self
                                .result_slots
                                .get(&result.value)
                                .copied()
                                .expect("closed scalar result has no publication")
                            else {
                                panic!("closed scalar result has a range publication")
                            };
                            PublishedResult::Scalar {
                                slot,
                                kind: PublishedScalarKind::Value(*dtype),
                            }
                        }
                        SemanticType::Index { .. } => {
                            let ScalarPublication::Scalar(slot) = self
                                .result_slots
                                .get(&result.value)
                                .copied()
                                .expect("closed index result has no publication")
                            else {
                                panic!("closed index result has a range publication")
                            };
                            PublishedResult::Scalar {
                                slot,
                                kind: PublishedScalarKind::Index,
                            }
                        }
                        SemanticType::Range { .. } => {
                            let ScalarPublication::Range { start, end } = self
                                .result_slots
                                .get(&result.value)
                                .copied()
                                .expect("closed range result has no publication")
                            else {
                                panic!("closed range result has a scalar publication")
                            };
                            PublishedResult::Range { start, end }
                        }
                        SemanticType::Void => continue,
                        SemanticType::Tuple(_) => {
                            panic!("tuple result survived semantic leaf normalization")
                        }
                        SemanticType::Opaque { .. } => {
                            panic!("opaque result crossed the implementation boundary")
                        }
                    };
                    result_publications.push(ResultPublication {
                        path: path.clone(),
                        binding,
                    });
                }
            }
            let kernels = kernel_internals::arena_from_kernels(self.owner, self.kernels);
            let global_allocations = self.topology.close(liveness, slots);
            ImplementationDraft::from_parts(ImplementationParts {
                authority: self.authority,
                numerical_role: self.numerical_role,
                identity,
                semantic_coverage: coverage,
                schedule,
                kernels,
                global_allocations,
                local_allocations,
                launch_layouts,
                launch_scratch,
                launch_abi,
                decisions: self.decisions,
                hard_constraints,
                numerical_transfer,
                duration,
                duration_qualification: self.child_duration_qualification,
                provenance: ImplementationProvenance {
                    root: self.function.stable(),
                    callees: self.callees,
                },
                result_publications,
            })
        }

        pub(super) fn typed_view<R: Representation>(
            &self,
            value: SemanticValueId,
        ) -> BufferViewId<R> {
            let view = self.value_views.get(&value).copied().unwrap_or_else(|| {
                panic!("factory requested a semantic tensor with no realized view")
            });
            assert_eq!(
                view.representation,
                R::id(),
                "factory requested a semantic tensor through the wrong representation type"
            );
            BufferViewId::new(self.owner, view.index)
        }

        fn derive_launch_scratch(
            &mut self,
            schedule: &ParametricSchedule,
            launch_layouts: &[crate::storage::LaunchLocalLayout],
        ) -> Vec<crate::storage::LaunchScratchRequirements> {
            use crate::storage::{LaunchLocalKind, ScratchRequirement};
            use crate::target::LocalRealization;
            assert_eq!(
                schedule.launches().len(),
                launch_layouts.len(),
                "one local layout per launch"
            );
            schedule
                .launches()
                .iter()
                .zip(launch_layouts)
                .map(|(launch, layout)| {
                    let groups = self.arena.nat_product(&launch.grid);
                    let threads = self.arena.nat_product(&launch.workgroup);
                    let participants = self.arena.nat_mul(groups, threads);
                    let kernel = self
                        .kernels
                        .get(launch.kernel.index() as usize)
                        .expect("launch kernel belongs to this implementation");
                    let alignment = |kind| {
                        kernel
                            .locals()
                            .iter()
                            .filter(|local| local.kind == kind)
                            .map(|local| local.alignment)
                            .max()
                            .unwrap_or(1)
                    };
                    let workgroup_alignment = alignment(LaunchLocalKind::Workgroup);
                    let participant_alignment = alignment(LaunchLocalKind::Participant);
                    let register_alignment = alignment(LaunchLocalKind::Register);
                    let requirement =
                        |this: &mut Self, kind: LaunchLocalKind, per_unit: NatExpr, alignment| {
                            let count = match this.target.local_realization().for_kind(kind) {
                                LocalRealization::NativeDynamic
                                | LocalRealization::NativeStatic => return None,
                                LocalRealization::InvocationScratchPerWorkgroup => groups,
                                LocalRealization::InvocationScratchPerParticipant => participants,
                            };
                            Some(ScratchRequirement {
                                bytes: scaled_scratch_bytes(this.arena, per_unit, count),
                                alignment,
                            })
                        };
                    crate::storage::LaunchScratchRequirements {
                        workgroup: requirement(
                            self,
                            LaunchLocalKind::Workgroup,
                            layout.workgroup_bytes,
                            workgroup_alignment,
                        ),
                        participant: requirement(
                            self,
                            LaunchLocalKind::Participant,
                            layout.participant_bytes,
                            participant_alignment,
                        ),
                        register: requirement(
                            self,
                            LaunchLocalKind::Register,
                            layout.register_bytes,
                            register_alignment,
                        ),
                    }
                })
                .collect()
        }

        /// The target-semantic duration model consumes the same closed
        /// schedule and typed kernels that native execution consumes. A
        /// factory cannot provide or override this value.
        fn derive_structural_duration(
            &mut self,
            schedule: &ParametricSchedule,
            launch_layouts: &[crate::storage::LaunchLocalLayout],
        ) -> crate::target::QualifiedDuration {
            self.execution.derive_duration(
                self.target,
                self.arena,
                schedule,
                launch_layouts,
                &self.kernels,
            )
        }

        fn derive_constraints(
            &mut self,
            schedule: &ParametricSchedule,
            launch_layouts: &[crate::storage::LaunchLocalLayout],
            launch_scratch: &[crate::storage::LaunchScratchRequirements],
            launch_abi: &[Vec<crate::storage::LaunchAbiRequirement>],
        ) {
            let max_allocation = self.arena.nat_symbol(
                self.arena
                    .target_constant_symbol(self.constants.max_allocation_bytes),
            );
            // `NatExpr` is already a u64 domain. A `value <= u64::MAX`
            // predicate is therefore a type tautology, not a target legality
            // condition, and retaining it leaves spurious call-dimension
            // obligations in universal closure. Narrower native index domains
            // still require the explicit bound everywhere below.
            let max_index = (self.target.limits().max_index_bits < 64).then(|| {
                self.arena
                    .nat((1u64 << self.target.limits().max_index_bits) - 1)
            });
            for allocation in 0..self.topology.allocation_count() {
                let id = GlobalAllocationId::new(self.owner, allocation);
                let bytes = self.topology.allocation_bytes(id);
                self.constraints
                    .push(self.arena.nat_cmp(CmpOp::Le, bytes, max_allocation));
                if let Some(max_index) = max_index {
                    self.constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, bytes, max_index));
                }
                let required_alignment = self.topology.allocation_alignment(id);
                self.constraints.push(
                    self.arena
                        .bool(required_alignment <= self.target.limits().max_allocation_alignment),
                );
            }
            for layout in self.topology.views() {
                if let Some(max_index) = max_index {
                    self.constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, layout.offset, max_index));
                }
                for extent in &layout.extents {
                    if let Some(max_index) = max_index {
                        self.constraints
                            .push(self.arena.nat_cmp(CmpOp::Le, *extent, max_index));
                    }
                }
                for stride in &layout.strides {
                    if let Some(max_index) = max_index {
                        self.constraints
                            .push(self.arena.nat_cmp(CmpOp::Le, *stride, max_index));
                    }
                }
                // Whole canonical views must use the same tensor-byte
                // expression as allocation and TargetDomain construction.
                // Re-expanding them as a generic strided range introduces an
                // equivalent max/sub/select tree (and subtraction side
                // conditions) that structural universal closure cannot and
                // should not have to rediscover. Nonzero-offset and strided
                // views retain the full addressed-range proof below.
                let addressed = if layout.contiguous
                    && matches!(
                        self.arena.view(AnyExpr::Nat(layout.offset)),
                        NodeView::NatConst(0)
                    ) {
                    crate::storage::tensor_bytes(self.arena, layout.representation, &layout.extents)
                } else {
                    addressed_bytes(self.arena, layout)
                };
                let allocation_bytes = self.topology.allocation_bytes(layout.allocation);
                self.constraints
                    .push(self.arena.nat_cmp(CmpOp::Le, addressed, allocation_bytes));
                if let Some(max_index) = max_index {
                    self.constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, addressed, max_index));
                }
                let view_alignment =
                    crate::storage::representation_alignment(layout.representation);
                let alignment = self.arena.nat(view_alignment);
                let remainder = self.arena.nat_rem(layout.offset, alignment);
                let zero = self.arena.nat(0);
                self.constraints
                    .push(self.arena.nat_cmp(CmpOp::Eq, remainder, zero));
            }
            let max_bindings = self.arena.nat_symbol(
                self.arena
                    .target_constant_symbol(self.constants.max_bindings),
            );
            let max_argument_bytes = self.arena.nat_symbol(
                self.arena
                    .target_constant_symbol(self.constants.max_argument_bytes),
            );
            assert_eq!(
                schedule.launches().len(),
                launch_layouts.len(),
                "one canonical local layout per launch"
            );
            assert_eq!(
                schedule.launches().len(),
                launch_scratch.len(),
                "one scratch realization per launch"
            );
            assert_eq!(
                schedule.launches().len(),
                launch_abi.len(),
                "one ABI allocation set per launch"
            );
            for (((launch, local_layout), scratch), abi_allocations) in schedule
                .launches()
                .iter()
                .zip(launch_layouts)
                .zip(launch_scratch)
                .zip(launch_abi)
            {
                let threads = self.arena.nat_product(&launch.workgroup);
                let max_threads = self.arena.nat(self.target.limits().max_workgroup_threads);
                self.constraints
                    .push(self.arena.nat_cmp(CmpOp::Le, threads, max_threads));
                for axis in 0..3 {
                    let max_workgroup_axis = self
                        .arena
                        .nat(self.target.limits().max_workgroup_size[axis]);
                    self.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        launch.workgroup[axis],
                        max_workgroup_axis,
                    ));
                    let certificate_owns_grid_x = axis == 0
                        && self.authority == ConstructionAuthority::UniversalPortable
                        && launch.parallel_extent.is_some()
                        && launch.logical_base.is_some();
                    if !certificate_owns_grid_x {
                        let max = self.arena.nat(self.target.limits().max_grid[axis]);
                        self.constraints.push(self.arena.nat_cmp(
                            CmpOp::Le,
                            launch.grid[axis],
                            max,
                        ));
                        if let Some(max_index) = max_index {
                            self.constraints.push(self.arena.nat_cmp(
                                CmpOp::Le,
                                launch.grid[axis],
                                max_index,
                            ));
                        }
                    }
                }

                let kernel = self
                    .kernels
                    .get(launch.kernel.index() as usize)
                    .expect("launch kernel belongs to this implementation");
                for local in &local_layout.locals {
                    for value in std::iter::once(local.offset)
                        .chain(local.extents.iter().copied())
                        .chain(local.strides.iter().copied())
                        .chain(std::iter::once(local.bytes))
                    {
                        if let Some(max_index) = max_index {
                            self.constraints
                                .push(self.arena.nat_cmp(CmpOp::Le, value, max_index));
                        }
                    }
                }
                for total in [
                    local_layout.workgroup_bytes,
                    local_layout.participant_bytes,
                    local_layout.register_bytes,
                ] {
                    if let Some(max_index) = max_index {
                        self.constraints
                            .push(self.arena.nat_cmp(CmpOp::Le, total, max_index));
                    }
                }
                for requirement in [&scratch.workgroup, &scratch.participant, &scratch.register]
                    .into_iter()
                    .flatten()
                {
                    self.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        requirement.bytes,
                        max_allocation,
                    ));
                    if let Some(max_index) = max_index {
                        self.constraints.push(self.arena.nat_cmp(
                            CmpOp::Le,
                            requirement.bytes,
                            max_index,
                        ));
                    }
                    self.constraints.push(self.arena.bool(
                        requirement.alignment.is_power_of_two()
                            && requirement.alignment
                                <= self.target.limits().max_allocation_alignment,
                    ));
                }
                let data = kernel_internals::data(kernel);
                for lease in data.addressable_resources() {
                    let capacity = self.arena.nat_symbol(self.arena.target_constant_symbol(
                        self.constants.addressable_resource_capacity(lease.class_id),
                    ));
                    let end = self.arena.nat_add(lease.offset_units, lease.units);
                    self.constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, end, capacity));
                    if let Some(max_index) = max_index {
                        self.constraints
                            .push(self.arena.nat_cmp(CmpOp::Le, end, max_index));
                    }
                }
                let binding_count = self.arena.nat(data.interface().bindings.len() as u64);
                self.constraints
                    .push(self.arena.nat_cmp(CmpOp::Le, binding_count, max_bindings));
                // Core ABI metadata size, not bytes reachable through bound
                // tensors. Buffer/result slots are encoded as machine-width
                // addresses, Nat arguments as u64, and scalar arguments at
                // their declared width.
                let footprint = self.target.kernel_abi_layout(kernel).footprint;
                assert!(
                    footprint.alignment.is_power_of_two(),
                    "backend kernel ABI footprint alignment is a nonzero power of two"
                );
                let argument_bytes = self.arena.nat(footprint.bytes);
                self.constraints.push(self.arena.nat_cmp(
                    CmpOp::Le,
                    argument_bytes,
                    max_argument_bytes,
                ));
                for allocation in abi_allocations {
                    self.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        allocation.bytes,
                        max_allocation,
                    ));
                    if let Some(max_index) = max_index {
                        self.constraints.push(self.arena.nat_cmp(
                            CmpOp::Le,
                            allocation.bytes,
                            max_index,
                        ));
                    }
                    self.constraints.push(self.arena.bool(
                        allocation.alignment.is_power_of_two()
                            && allocation.alignment
                                <= self.target.limits().max_allocation_alignment,
                    ));
                }

                for resource in data.intrinsic_resources() {
                    if resource.requires_subgroup {
                        assert!(
                            data.interface().uses_subgroup,
                            "intrinsic subgroup requirement was not propagated to the kernel interface"
                        );
                    }
                }
                let dynamic_local_limit = self.arena.nat(self.target.limits().max_workgroup_bytes);
                self.constraints.push(self.arena.nat_cmp(
                    CmpOp::Le,
                    local_layout.workgroup_bytes,
                    dynamic_local_limit,
                ));
                if let Some(participant_limit) = self.constants.participant_local_bytes {
                    let participant_limit = self
                        .arena
                        .nat_symbol(self.arena.target_constant_symbol(participant_limit));
                    self.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        local_layout.participant_bytes,
                        participant_limit,
                    ));
                }
            }
        }

        fn derive_reuse(&mut self, liveness: &[AllocationLiveness]) -> Vec<Option<DecisionId>> {
            let count = self.topology.allocation_count() as usize;
            let mut slots = vec![None; count];
            let arenas: Vec<usize> = (0..count)
                .filter(|index| {
                    matches!(
                        self.topology
                            .allocation_kind(GlobalAllocationId::new(self.owner, *index as u32)),
                        GlobalBufferKind::Arena
                    )
                })
                .collect();
            for &index in &arenas {
                assert!(
                    !liveness[index].uses().is_empty(),
                    "arena allocation has no structured schedule use"
                );
            }
            if self.authority == ConstructionAuthority::UniversalPortable {
                return slots;
            }
            for (eligible_ordinal, &allocation) in arenas.iter().enumerate() {
                let domain = FiniteDomain::new((0..=eligible_ordinal as i64).collect())
                    .expect("an arena allocation always has its own reuse slot");
                slots[allocation] = Some(self.decision("arena reuse slot", domain));
            }
            for (right_ordinal, &right) in arenas.iter().enumerate() {
                for (left_ordinal, &left) in arenas[..right_ordinal].iter().enumerate() {
                    let incompatible = !self.reuse_compatible(left, right);
                    if incompatible || lifetimes_interfere(&liveness[left], &liveness[right]) {
                        let left_decision =
                            slots[left].expect("optimized arena allocation has a slot decision");
                        let right_decision =
                            slots[right].expect("optimized arena allocation has a slot decision");
                        for value in 0..=left_ordinal as i64 {
                            let left_is = self.arena.decision_is(left_decision, value);
                            let right_is = self.arena.decision_is(right_decision, value);
                            let both = self.arena.all(&[left_is, right_is]);
                            let distinct = self.arena.not(both);
                            self.constraints.push(distinct);
                        }
                    }
                }
            }
            slots
        }

        fn reuse_compatible(&self, left: usize, right: usize) -> bool {
            let left_id = GlobalAllocationId::new(self.owner, left as u32);
            let right_id = GlobalAllocationId::new(self.owner, right as u32);
            if self.topology.allocation_alignment(left_id)
                != self.topology.allocation_alignment(right_id)
            {
                return false;
            }
            let representations = |allocation: GlobalAllocationId| {
                let mut values: Vec<_> = self
                    .topology
                    .views()
                    .iter()
                    .filter_map(|view| {
                        (view.allocation == allocation).then_some(view.representation)
                    })
                    .collect();
                values.sort();
                values.dedup();
                values
            };
            representations(left_id) == representations(right_id)
        }

        fn validate_schedule(&mut self, schedule: &ParametricSchedule) {
            fn visit<B: Backend>(builder: &mut Builder<'_, B>, steps: &[ScheduleStep]) {
                for step in steps {
                    match step {
                        ScheduleStep::Launch(_)
                        | ScheduleStep::ScalarMove(_)
                        | ScheduleStep::Check(_) => {}
                        ScheduleStep::Copy(copy) => {
                            let source = builder.topology.view_layout(copy.source).clone();
                            let destination =
                                builder.topology.view_layout(copy.destination).clone();
                            assert_eq!(
                                source.representation, destination.representation,
                                "copy representation mismatch"
                            );
                            assert_eq!(
                                source.extents.len(),
                                destination.extents.len(),
                                "copy rank mismatch"
                            );
                            for (a, b) in source
                                .extents
                                .iter()
                                .copied()
                                .zip(destination.extents.iter().copied())
                            {
                                builder
                                    .constraints
                                    .push(builder.arena.nat_cmp(CmpOp::Eq, a, b));
                            }
                            let source_bytes = crate::storage::tensor_bytes(
                                builder.arena,
                                source.representation,
                                &source.extents,
                            );
                            let destination_bytes = crate::storage::tensor_bytes(
                                builder.arena,
                                destination.representation,
                                &destination.extents,
                            );
                            builder.constraints.push(builder.arena.nat_cmp(
                                CmpOp::Eq,
                                source_bytes,
                                destination_bytes,
                            ));
                        }
                        ScheduleStep::Fill(_) => {}
                        ScheduleStep::ScalarRead(_) => {}
                        ScheduleStep::If {
                            then_steps,
                            else_steps,
                            ..
                        } => {
                            visit(builder, then_steps);
                            visit(builder, else_steps);
                        }
                        ScheduleStep::Repeat { body, .. } => visit(builder, body),
                        ScheduleStep::Choose { options, .. } => {
                            for (_, body) in options {
                                visit(builder, body);
                            }
                        }
                    }
                }
            }
            visit(self, schedule.steps());
        }

        fn derive_numerics(&mut self) -> NumericalTransfer {
            let mut effects = Vec::new();
            let mut operations = Vec::new();
            if self.numerical_role == seismic_lang::entry::NumericalRole::Alternative {
                // A distinct authored body is not reference-equivalent merely
                // because it happened to contain no individually approximate
                // primitive. Whole-body equivalence requires operation-level
                // analysis or qualified evidence.
                effects.push(NumericalEffect::AlternativeBody);
            }
            for (kernel_ordinal, kernel) in self.kernels.iter().enumerate() {
                for (ordinal, (fact, multiplicity)) in kernel_internals::data(kernel)
                    .fact_multiplicities()
                    .enumerate()
                {
                    let effect = match fact {
                        crate::kernel::ops::NumericalFact::ContractedFma => {
                            NumericalEffect::Contraction
                        }
                        crate::kernel::ops::NumericalFact::ApproximateMath(op) => {
                            NumericalEffect::ApproximateTranscendental(*op)
                        }
                        crate::kernel::ops::NumericalFact::ReassociatedIntrinsic(id) => {
                            NumericalEffect::BackendIntrinsic(*id)
                        }
                        crate::kernel::ops::NumericalFact::NarrowAccumulator(dtype) => {
                            NumericalEffect::NarrowAccumulator(*dtype)
                        }
                        crate::kernel::ops::NumericalFact::FlushToZero => {
                            NumericalEffect::FlushToZero
                        }
                        crate::kernel::ops::NumericalFact::ReassociatedReduction => {
                            NumericalEffect::ReassociatedReduction
                        }
                    };
                    if !effects.contains(&effect) {
                        effects.push(effect.clone());
                    }
                    operations.push(crate::numerics::NumericalOperation {
                        kernel: kernel_ordinal as u32,
                        ordinal: ordinal as u32,
                        effect,
                        multiplicity,
                    });
                }
            }
            let exact = effects.is_empty();
            let outputs = self
                .contract
                .results
                .iter()
                .flat_map(|result| {
                    let dtype = match &result.ty {
                        SemanticType::Tensor(tensor) => {
                            match &registry::representation_info(tensor.representation).kind {
                                registry::RepresentationKind::Dense(dtype) => *dtype,
                                registry::RepresentationKind::Packed(_) => DType::F32,
                                registry::RepresentationKind::External(_) => {
                                    panic!("external representation cannot be a generic semantic result")
                                }
                            }
                        }
                        SemanticType::Scalar(dtype) => *dtype,
                        SemanticType::Index { .. } | SemanticType::Range { .. } => DType::U32,
                        _ => DType::F32,
                    };
                    result
                        .paths
                        .iter()
                        .cloned()
                        .map(move |path| OutputTransfer {
                            path,
                            dtype,
                            bound: if exact {
                                ErrorBound::Exact
                            } else {
                                ErrorBound::Unknown
                            },
                            specials: crate::numerics::SpecialGuarantees {
                                nan: exact,
                                infinity: exact,
                                signed_zero: exact,
                                subnormal: exact,
                            },
                        })
                })
                .collect();
            NumericalTransfer::new(
                outputs,
                effects,
                operations,
                std::collections::BTreeMap::new(),
                std::mem::take(&mut self.conditional_child_transfers),
            )
        }

        fn register_roots(
            &mut self,
            schedule: &ParametricSchedule,
            launch_layouts: &[crate::storage::LaunchLocalLayout],
            launch_scratch: &[crate::storage::LaunchScratchRequirements],
            launch_abi: &[Vec<crate::storage::LaunchAbiRequirement>],
            hard_constraints: BoolExpr,
            coverage: TargetPredicate,
            duration: DurationExpr,
            duration_qualification: &[BoolExpr],
            transfer: &NumericalTransfer,
        ) -> Vec<seismic_lang::expr::RootId> {
            let mut roots = Vec::new();
            for allocation in 0..self.topology.allocation_count() {
                let id = GlobalAllocationId::new(self.owner, allocation);
                roots.push(self.arena.root(
                    RootName::AllocationBytes { allocation },
                    AnyExpr::Nat(self.topology.allocation_bytes(id)),
                ));
            }
            for (view, layout) in self.topology.views().iter().enumerate() {
                roots.push(self.arena.root(
                    RootName::ViewOffset { view: view as u32 },
                    AnyExpr::Nat(layout.offset),
                ));
                for (axis, extent) in layout.extents.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::ViewExtent {
                            view: view as u32,
                            axis: axis as u32,
                        },
                        AnyExpr::Nat(*extent),
                    ));
                }
                for (axis, stride) in layout.strides.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::ViewStride {
                            view: view as u32,
                            axis: axis as u32,
                        },
                        AnyExpr::Nat(*stride),
                    ));
                }
            }
            for (launch_id, launch) in schedule.launches().iter().enumerate() {
                for axis in 0..3 {
                    roots.push(self.arena.root(
                        RootName::LaunchGrid {
                            launch: launch_id as u32,
                            axis: axis as u8,
                        },
                        AnyExpr::Nat(launch.grid[axis]),
                    ));
                }
                for axis in 0..3 {
                    roots.push(self.arena.root(
                        RootName::Workgroup {
                            launch: launch_id as u32,
                            axis: axis as u8,
                        },
                        AnyExpr::Nat(launch.workgroup[axis]),
                    ));
                }
                roots.push(self.arena.root(
                    RootName::LaunchEmpty {
                        launch: launch_id as u32,
                    },
                    AnyExpr::Bool(launch.empty),
                ));
                let layout = &launch_layouts[launch_id];
                for (local, value) in layout.locals.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::LocalOffset {
                            launch: launch_id as u32,
                            local: local as u32,
                        },
                        AnyExpr::Nat(value.offset),
                    ));
                    for (axis, stride) in value.strides.iter().enumerate() {
                        roots.push(self.arena.root(
                            RootName::LocalStride {
                                launch: launch_id as u32,
                                local: local as u32,
                                axis: axis as u32,
                            },
                            AnyExpr::Nat(*stride),
                        ));
                    }
                }
                for (class, bytes) in [
                    layout.workgroup_bytes,
                    layout.participant_bytes,
                    layout.register_bytes,
                ]
                .into_iter()
                .enumerate()
                {
                    roots.push(self.arena.root(
                        RootName::LocalClassBytes {
                            launch: launch_id as u32,
                            class: class as u8,
                        },
                        AnyExpr::Nat(bytes),
                    ));
                }
                let scratch = &launch_scratch[launch_id];
                for (class, requirement) in
                    [&scratch.workgroup, &scratch.participant, &scratch.register]
                        .into_iter()
                        .enumerate()
                {
                    if let Some(requirement) = requirement {
                        roots.push(self.arena.root(
                            RootName::LaunchScratchBytes {
                                launch: launch_id as u32,
                                class: class as u8,
                            },
                            AnyExpr::Nat(requirement.bytes),
                        ));
                    }
                }
                for (allocation, requirement) in launch_abi[launch_id].iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::LaunchAbiBytes {
                            launch: launch_id as u32,
                            allocation: allocation as u32,
                        },
                        AnyExpr::Nat(requirement.bytes),
                    ));
                }
            }
            for (kernel_index, kernel) in self.kernels.iter().enumerate() {
                let data = kernel_internals::data(kernel);
                for (argument, value) in data.interface().nat_args.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::KernelNatArgument {
                            kernel: kernel_index as u32,
                            argument: argument as u32,
                        },
                        AnyExpr::Nat(*value),
                    ));
                }
                for (argument, (symbol, dtype)) in data.interface().scalar_args.iter().enumerate() {
                    let expression = match dtype {
                        DType::F32 => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::F32>(*symbol),
                        ),
                        DType::F16 => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::F16>(*symbol),
                        ),
                        DType::BF16 => AnyExpr::from(
                            self.arena
                                .scalar_symbol::<seismic_lang::expr::BF16>(*symbol),
                        ),
                        DType::I32 => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::I32>(*symbol),
                        ),
                        DType::U32 => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::U32>(*symbol),
                        ),
                        DType::Bool => AnyExpr::from(
                            self.arena
                                .scalar_symbol::<seismic_lang::expr::BoolScalar>(*symbol),
                        ),
                    };
                    roots.push(self.arena.root(
                        RootName::KernelScalarArgument {
                            kernel: kernel_index as u32,
                            argument: argument as u32,
                        },
                        expression,
                    ));
                }
                for (local, allocation) in data.locals().iter().enumerate() {
                    for (axis, extent) in allocation.extents.iter().enumerate() {
                        roots.push(self.arena.root(
                            RootName::LocalExtent {
                                kernel: kernel_index as u32,
                                local: local as u32,
                                axis: axis as u32,
                            },
                            AnyExpr::Nat(*extent),
                        ));
                    }
                }
                for (lease, resource) in data.addressable_resources().iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::AddressableResourceOffset {
                            kernel: kernel_index as u32,
                            lease: lease as u32,
                        },
                        AnyExpr::Nat(resource.offset_units),
                    ));
                    roots.push(self.arena.root(
                        RootName::AddressableResourceUnits {
                            kernel: kernel_index as u32,
                            lease: lease as u32,
                        },
                        AnyExpr::Nat(resource.units),
                    ));
                }
                for (resource, values) in data.intrinsic_resources().iter().enumerate() {
                    if let Some(value) = values.workgroup_bytes {
                        roots.push(self.arena.root(
                            RootName::IntrinsicWorkgroupBytes {
                                kernel: kernel_index as u32,
                                resource: resource as u32,
                            },
                            AnyExpr::Nat(value),
                        ));
                    }
                    if let Some(value) = values.participant_bytes {
                        roots.push(self.arena.root(
                            RootName::IntrinsicParticipantBytes {
                                kernel: kernel_index as u32,
                                resource: resource as u32,
                            },
                            AnyExpr::Nat(value),
                        ));
                    }
                    if let Some(value) = values.register_bytes {
                        roots.push(self.arena.root(
                            RootName::IntrinsicRegisterBytes {
                                kernel: kernel_index as u32,
                                resource: resource as u32,
                            },
                            AnyExpr::Nat(value),
                        ));
                    }
                }
                for (fact, (_, multiplicity)) in data.fact_multiplicities().enumerate() {
                    if let Some(value) = multiplicity {
                        roots.push(self.arena.root(
                            RootName::NumericalMultiplicity {
                                kernel: kernel_index as u32,
                                fact: fact as u32,
                            },
                            AnyExpr::Nat(value),
                        ));
                    }
                }
            }
            fn schedule_roots(
                arena: &mut ExprArena,
                steps: &[ScheduleStep],
                roots: &mut Vec<seismic_lang::expr::RootId>,
                control: &mut u32,
                repeat: &mut u32,
                scalar_read: &mut u32,
            ) {
                for step in steps {
                    match step {
                        ScheduleStep::Launch(_)
                        | ScheduleStep::Copy(_)
                        | ScheduleStep::Fill(_)
                        | ScheduleStep::ScalarMove(_)
                        | ScheduleStep::Check(_) => {}
                        ScheduleStep::ScalarRead(read) => {
                            let step = *scalar_read;
                            *scalar_read = (*scalar_read)
                                .checked_add(1)
                                .expect("scalar-read root ordinal space exhausted");
                            for (axis, index) in read.index.iter().enumerate() {
                                roots.push(arena.root(
                                    RootName::ScalarReadIndex {
                                        step,
                                        axis: axis as u32,
                                    },
                                    AnyExpr::Nat(*index),
                                ));
                            }
                        }
                        ScheduleStep::If {
                            condition,
                            then_steps,
                            else_steps,
                        } => {
                            let id = *control;
                            *control = (*control)
                                .checked_add(1)
                                .expect("schedule-control root ordinal space exhausted");
                            roots.push(arena.root(
                                RootName::ScheduleCondition { control: id },
                                AnyExpr::Bool(*condition),
                            ));
                            schedule_roots(arena, then_steps, roots, control, repeat, scalar_read);
                            schedule_roots(arena, else_steps, roots, control, repeat, scalar_read);
                        }
                        ScheduleStep::Repeat {
                            start, end, body, ..
                        } => {
                            let id = *repeat;
                            *repeat = (*repeat)
                                .checked_add(1)
                                .expect("schedule-repeat root ordinal space exhausted");
                            roots.push(
                                arena.root(
                                    RootName::RepeatStart { repeat: id },
                                    AnyExpr::Nat(*start),
                                ),
                            );
                            roots.push(
                                arena.root(RootName::RepeatEnd { repeat: id }, AnyExpr::Nat(*end)),
                            );
                            schedule_roots(arena, body, roots, control, repeat, scalar_read);
                        }
                        ScheduleStep::Choose { decision, options } => {
                            let id = *control;
                            *control = (*control)
                                .checked_add(1)
                                .expect("schedule-control root ordinal space exhausted");
                            let value = arena.decision_value(*decision);
                            roots.push(arena.root(
                                RootName::ScheduleChoice { control: id },
                                AnyExpr::Int(value),
                            ));
                            for (_, body) in options {
                                schedule_roots(arena, body, roots, control, repeat, scalar_read);
                            }
                        }
                    }
                }
            }
            let (mut control, mut repeat, mut scalar_read) = (0, 0, 0);
            schedule_roots(
                self.arena,
                schedule.steps(),
                &mut roots,
                &mut control,
                &mut repeat,
                &mut scalar_read,
            );
            roots.push(
                self.arena
                    .root(RootName::HardConstraints, AnyExpr::Bool(hard_constraints)),
            );
            roots.push(
                self.arena
                    .root(RootName::Guard, AnyExpr::Bool(coverage.node())),
            );
            for qualification in duration_qualification {
                roots.push(
                    self.arena
                        .root(RootName::Guard, AnyExpr::Bool(*qualification)),
                );
            }
            roots.push(
                self.arena
                    .root(RootName::Duration, AnyExpr::Duration(duration)),
            );
            fn numerical_roots(
                arena: &mut ExprArena,
                transfer: &NumericalTransfer,
                roots: &mut Vec<seismic_lang::expr::RootId>,
                output: &mut u32,
                child: &mut u32,
                operation: &mut u32,
            ) {
                for value in transfer.outputs() {
                    if let ErrorBound::Analytic { roundings, .. } = &value.bound {
                        roots.push(arena.root(
                            RootName::ErrorBound { output: *output },
                            AnyExpr::Nat(*roundings),
                        ));
                    }
                    *output = output
                        .checked_add(1)
                        .expect("numerical-output root ordinal space exhausted");
                }
                for value in transfer.operations() {
                    if let Some(multiplicity) = value.multiplicity {
                        roots.push(arena.root(
                            RootName::NumericalOperationMultiplicity {
                                operation: *operation,
                            },
                            AnyExpr::Nat(multiplicity),
                        ));
                    }
                    *operation = operation
                        .checked_add(1)
                        .expect("numerical-operation root ordinal space exhausted");
                }
                for value in transfer.children() {
                    if let Some((decision, expected)) = value.selection {
                        let condition = arena.decision_is(decision, expected);
                        roots.push(arena.root(
                            RootName::NumericalCondition { child: *child },
                            AnyExpr::Bool(condition),
                        ));
                    }
                    *child = child
                        .checked_add(1)
                        .expect("numerical-child root ordinal space exhausted");
                    numerical_roots(arena, &value.transfer, roots, output, child, operation);
                }
            }
            let (mut output, mut child, mut operation) = (0, 0, 0);
            numerical_roots(
                self.arena,
                transfer,
                &mut roots,
                &mut output,
                &mut child,
                &mut operation,
            );
            roots
        }
    }

    /// Contribution of one view axis to its furthest addressed element.
    ///
    /// A zero stride means the axis never participates in address formation.
    /// Do not construct or evaluate `(max(extent, 1) - 1)` in that case: the
    /// extent can itself be a partial range subtraction, but its definedness
    /// is irrelevant to this axis's address contribution.
    pub(super) fn addressed_axis_span(
        arena: &mut ExprArena,
        extent: NatExpr,
        stride: NatExpr,
    ) -> NatExpr {
        if matches!(arena.view(AnyExpr::Nat(stride)), NodeView::NatConst(0)) {
            return stride;
        }
        let one = arena.nat(1);
        let nonzero_extent = arena.nat_max(extent, one);
        let last = arena.nat_sub(nonzero_extent, one);
        arena.nat_mul(last, stride)
    }

    /// Scale a per-unit scratch requirement without introducing launch-count
    /// definedness when the implementation requires no scratch bytes at all.
    /// This is a property of scratch realization, not a global relaxation of
    /// strict expression evaluation.
    pub(super) fn scaled_scratch_bytes(
        arena: &mut ExprArena,
        per_unit: NatExpr,
        count: NatExpr,
    ) -> NatExpr {
        if matches!(arena.view(AnyExpr::Nat(per_unit)), NodeView::NatConst(0)) {
            per_unit
        } else {
            arena.nat_mul(per_unit, count)
        }
    }

    /// Exclusive byte end of the furthest element reachable through a view.
    /// A zero-extent view addresses no bytes and therefore ends at its offset.
    fn addressed_bytes(
        arena: &mut ExprArena,
        layout: &crate::storage::BufferViewLayout,
    ) -> NatExpr {
        assert_eq!(
            layout.extents.len(),
            layout.strides.len(),
            "closed view rank and stride count differ"
        );
        let (unit_bytes, logical_extents) =
            match &registry::representation_info(layout.representation).kind {
                registry::RepresentationKind::Dense(dtype) => {
                    (dtype.bytes() as u64, layout.extents.clone())
                }
                registry::RepresentationKind::Packed(packet) => {
                    let mut extents = layout.extents.clone();
                    if let Some(last) = extents.last_mut() {
                        let group = arena.nat(u64::from(packet.group));
                        *last = arena.nat_ceil_div(*last, group);
                    }
                    (u64::from(packet.packet_size), extents)
                }
                registry::RepresentationKind::External(packet) => {
                    let mut extents = layout.extents.clone();
                    if let Some(last) = extents.last_mut() {
                        let group = arena.nat(u64::from(packet.logical_group));
                        *last = arena.nat_ceil_div(*last, group);
                    }
                    (u64::from(packet.packet_size), extents)
                }
            };
        let zero = arena.nat(0);
        let one = arena.nat(1);
        let mut span = zero;
        let mut empty_terms = Vec::with_capacity(logical_extents.len());
        for (extent, stride) in logical_extents
            .iter()
            .copied()
            .zip(layout.strides.iter().copied())
        {
            empty_terms.push(arena.nat_cmp(CmpOp::Eq, extent, zero));
            let axis = addressed_axis_span(arena, extent, stride);
            span = arena.nat_add(span, axis);
        }
        let element_bytes = arena.nat(unit_bytes);
        let span_with_element = arena.nat_add(span, one);
        let payload_end = arena.nat_mul(span_with_element, element_bytes);
        let end = arena.nat_add(layout.offset, payload_end);
        let empty = arena.any(&empty_terms);
        arena.nat_select(empty, layout.offset, end)
    }

    fn lifetimes_interfere(left: &AllocationLiveness, right: &AllocationLiveness) -> bool {
        if left
            .uses()
            .iter()
            .all(|a| right.uses().iter().all(|b| mutually_exclusive(a, b)))
        {
            return false;
        }
        let mut left_points: Vec<_> = left.uses().iter().map(schedule_point).collect();
        let mut right_points: Vec<_> = right.uses().iter().map(schedule_point).collect();
        left_points.sort();
        right_points.sort();
        !(left_points.last() < right_points.first() || right_points.last() < left_points.first())
    }

    fn schedule_point(at: &crate::storage::ScheduleUse) -> Vec<u32> {
        let mut point = Vec::with_capacity(at.region.len() + 1);
        for edge in &at.region {
            let parent_ordinal = match edge {
                crate::storage::ScheduleRegionEdge::IfThen { parent_ordinal, .. }
                | crate::storage::ScheduleRegionEdge::IfElse { parent_ordinal, .. }
                | crate::storage::ScheduleRegionEdge::RepeatBody { parent_ordinal, .. }
                | crate::storage::ScheduleRegionEdge::ChooseOption { parent_ordinal, .. }
                | crate::storage::ScheduleRegionEdge::Imported { parent_ordinal, .. } => {
                    *parent_ordinal
                }
            };
            point.push(parent_ordinal);
        }
        point.push(at.ordinal);
        point
    }

    fn mutually_exclusive(
        a: &crate::storage::ScheduleUse,
        b: &crate::storage::ScheduleUse,
    ) -> bool {
        for (left, right) in a.region.iter().zip(&b.region) {
            use crate::storage::ScheduleRegionEdge::*;
            match (left, right) {
                (IfThen { node: a, .. }, IfElse { node: b, .. })
                | (IfElse { node: a, .. }, IfThen { node: b, .. })
                    if a == b =>
                {
                    return true;
                }
                (
                    ChooseOption {
                        node: a, value: av, ..
                    },
                    ChooseOption {
                        node: b, value: bv, ..
                    },
                ) if a == b && av != bv => return true,
                _ if left != right => return false,
                _ => {}
            }
        }
        false
    }

    fn digest_structure<B: Backend>(
        digest: &mut crate::identity::StructureDigest,
        builder: &Builder<'_, B>,
        schedule: &ParametricSchedule,
        liveness: &[AllocationLiveness],
        slots: &[Option<DecisionId>],
    ) {
        digest.bytes(builder.function.stable().digest());
        digest.bytes(match builder.authority {
            ConstructionAuthority::UniversalPortable => b"universal",
            ConstructionAuthority::Optimized => b"optimized",
        });
        digest.hashed(&builder.decisions.len());
        for (decision, name) in &builder.decisions {
            digest.bytes(name.as_bytes());
            digest.hashed(builder.arena.decision_domain(*decision).values());
        }

        digest.hashed(&builder.topology.allocation_count());
        for index in 0..builder.topology.allocation_count() {
            let id = GlobalAllocationId::new(builder.owner, index);
            match builder.topology.allocation_kind(id) {
                GlobalBufferKind::Argument { value, abi } => {
                    digest.bytes(b"argument");
                    digest.hashed(
                        &builder
                            .contract
                            .parameters
                            .iter()
                            .position(|parameter| parameter.value == *value),
                    );
                    digest.hashed(
                        &abi.as_ref()
                            .and_then(|id| {
                                builder
                                    .contract
                                    .parameters
                                    .iter()
                                    .position(|parameter| parameter.value == *value)
                                    .map(|_| id)
                            })
                            .is_some(),
                    );
                }
                GlobalBufferKind::Result { value } => {
                    digest.bytes(b"result");
                    digest.hashed(
                        &builder
                            .contract
                            .results
                            .iter()
                            .position(|result| result.value == *value),
                    );
                }
                GlobalBufferKind::Imported { value, source } => {
                    digest.bytes(b"imported");
                    digest.hashed(
                        &builder
                            .contract
                            .parameters
                            .iter()
                            .position(|parameter| parameter.value == *value),
                    );
                    digest.hashed(
                        &builder
                            .contract
                            .results
                            .iter()
                            .position(|result| result.value == *value),
                    );
                    digest.bytes(
                        seismic_lang::registry::representation_info(source.representation)
                            .name
                            .as_bytes(),
                    );
                }
                GlobalBufferKind::Arena => digest.bytes(b"arena"),
                GlobalBufferKind::Persistent => digest.bytes(b"persistent"),
            }
            digest.hashed(&builder.topology.allocation_alignment(id));
            digest.hashed(&liveness[index as usize].uses().len());
            for at in liveness[index as usize].uses() {
                digest.hashed(&at.region);
                digest.hashed(&at.ordinal);
            }
            digest.hashed(&slots[index as usize].and_then(|decision| {
                builder
                    .decisions
                    .iter()
                    .position(|(candidate, _)| *candidate == decision)
            }));
        }
        digest.hashed(&builder.topology.views().len());
        for view in builder.topology.views() {
            digest.hashed(&view.allocation.index());
            digest.bytes(
                seismic_lang::registry::representation_info(view.representation)
                    .name
                    .as_bytes(),
            );
            digest.hashed(&view.extents.len());
            digest.hashed(&view.contiguous);
        }
        digest.hashed(&builder.topology.result_views().len());
        for publication in builder.topology.result_views() {
            digest.hashed(&publication.path);
            digest.hashed(&publication.view.index);
        }

        digest.hashed(&builder.kernels.len());
        for kernel in &builder.kernels {
            digest_kernel::<B>(digest, kernel);
        }
        digest.hashed(&schedule.launches().len());
        for launch in schedule.launches() {
            digest.bytes(b"launch-definition");
            digest.hashed(&launch.kernel.index());
            digest.hashed(&launch.mode);
        }
        digest_schedule(digest, builder, schedule.steps());

        for result in &builder.contract.results {
            if let Some(publication) = builder.result_slots.get(&result.value) {
                digest.bytes(b"published-scalar");
                digest.hashed(
                    &builder
                        .contract
                        .results
                        .iter()
                        .position(|candidate| candidate.value == result.value),
                );
                match publication {
                    ScalarPublication::Scalar(slot) => {
                        digest.bytes(b"one");
                        digest.hashed(&slot.index);
                        digest.hashed(&slot.dtype);
                        digest.hashed(&slot.sort);
                    }
                    ScalarPublication::Range { start, end } => {
                        digest.bytes(b"range");
                        digest.hashed(&start.index);
                        digest.hashed(&end.index);
                    }
                }
            }
        }
        for callee in &builder.callees {
            digest.bytes(callee.digest());
        }
    }

    fn digest_numerical<B: Backend>(
        digest: &mut crate::identity::StructureDigest,
        builder: &Builder<'_, B>,
        numerical: &NumericalTransfer,
    ) {
        fn role(
            digest: &mut crate::identity::StructureDigest,
            value: seismic_lang::entry::NumericalRole,
        ) {
            digest.bytes(match value {
                seismic_lang::entry::NumericalRole::Reference => b"reference",
                seismic_lang::entry::NumericalRole::Alternative => b"alternative",
            });
        }
        fn effect(digest: &mut crate::identity::StructureDigest, value: &NumericalEffect) {
            match value {
                NumericalEffect::ReassociatedReduction => digest.bytes(b"reassociated-reduction"),
                NumericalEffect::Contraction => digest.bytes(b"contraction"),
                NumericalEffect::ApproximateTranscendental(op) => {
                    digest.bytes(b"approximate-transcendental");
                    digest.hashed(op);
                }
                NumericalEffect::NarrowAccumulator(dtype) => {
                    digest.bytes(b"narrow-accumulator");
                    digest.hashed(dtype);
                }
                NumericalEffect::FlushToZero => digest.bytes(b"flush-to-zero"),
                NumericalEffect::BackendIntrinsic(id) => {
                    digest.bytes(b"backend-intrinsic");
                    let signature = seismic_lang::registry::intrinsic_signature(*id);
                    let capability = seismic_lang::registry::capability_info(signature.capability);
                    digest.bytes(capability.backend.as_str().as_bytes());
                    digest.bytes(capability.name.as_bytes());
                    digest.bytes(signature.name.as_bytes());
                }
                NumericalEffect::AlternativeBody => digest.bytes(b"alternative-body"),
            }
        }
        fn transfer<B: Backend>(
            digest: &mut crate::identity::StructureDigest,
            builder: &Builder<'_, B>,
            value: &NumericalTransfer,
        ) {
            digest.hashed(&value.outputs().len());
            for output in value.outputs() {
                digest.hashed(&output.path);
                digest.hashed(&output.dtype);
                match &output.bound {
                    ErrorBound::Exact => digest.bytes(b"exact"),
                    ErrorBound::Unknown => digest.bytes(b"unknown"),
                    ErrorBound::Analytic {
                        absolute,
                        relative,
                        ulps,
                        ..
                    } => {
                        digest.bytes(b"analytic");
                        digest.u64(absolute.to_bits());
                        digest.u64(relative.to_bits());
                        digest.u32(*ulps);
                    }
                }
                digest.bool(output.specials.nan);
                digest.bool(output.specials.infinity);
                digest.bool(output.specials.signed_zero);
                digest.bool(output.specials.subnormal);
            }
            digest.hashed(&value.effects().len());
            for item in value.effects() {
                effect(digest, item);
            }
            digest.hashed(&value.operations().len());
            for operation in value.operations() {
                digest.u32(operation.kernel);
                digest.u32(operation.ordinal);
                effect(digest, &operation.effect);
                digest.bool(operation.multiplicity.is_some());
            }
            digest.hashed(&value.input_assumptions().len());
            for (name, range) in value.input_assumptions() {
                digest.bytes(name.as_bytes());
                digest.u64(range.minimum.get().to_bits());
                digest.u64(range.maximum.get().to_bits());
            }
            digest.hashed(&value.children().len());
            for child in value.children() {
                match child.selection {
                    Some((decision, expected)) => {
                        digest.bool(true);
                        let ordinal = builder
                            .decisions
                            .iter()
                            .position(|(candidate, _)| *candidate == decision)
                            .expect("numerical transfer references an unowned decision");
                        digest.hashed(&ordinal);
                        digest.i64(expected);
                    }
                    None => digest.bool(false),
                }
                role(digest, child.role);
                transfer(digest, builder, &child.transfer);
            }
        }
        role(digest, builder.numerical_role);
        transfer(digest, builder, numerical);
    }

    fn digest_kernel<B: Backend>(
        digest: &mut crate::identity::StructureDigest,
        kernel: &crate::kernel::Kernel<B>,
    ) {
        use crate::kernel::ops::{Op, PlaceRef};
        fn value(
            digest: &mut crate::identity::StructureDigest,
            value: crate::kernel::ops::ErasedValue,
        ) {
            digest.hashed(&value.ordinal());
        }
        fn values(
            digest: &mut crate::identity::StructureDigest,
            values: &[crate::kernel::ops::ErasedValue],
        ) {
            digest.hashed(&values.len());
            for item in values {
                value(digest, *item);
            }
        }
        fn place(
            digest: &mut crate::identity::StructureDigest,
            place: crate::kernel::ops::PlaceRef,
        ) {
            match place {
                crate::kernel::ops::PlaceRef::Global { slot } => {
                    digest.bytes(b"global");
                    digest.hashed(&slot.ordinal());
                }
                crate::kernel::ops::PlaceRef::Local { index } => {
                    digest.bytes(b"local");
                    digest.hashed(&index);
                }
            }
        }
        let data = kernel_internals::data(kernel);
        let interface = data.interface();
        digest.hashed(&interface.bindings.len());
        for binding in &interface.bindings {
            digest.hashed(&binding.slot.index());
            digest.hashed(&binding.view.index);
            digest.hashed(&binding.access);
            digest.hashed(&binding.rank);
        }
        digest.hashed(&interface.nat_args.len());
        digest.hashed(
            &interface
                .scalar_args
                .iter()
                .map(|(_, dtype)| *dtype)
                .collect::<Vec<_>>(),
        );
        for (slot, dtype) in &interface.result_slots {
            digest.hashed(&slot.index);
            digest.hashed(dtype);
        }
        digest.hashed(&interface.uses_subgroup);
        digest.hashed(data.value_types());
        for local in data.locals() {
            digest.hashed(&local.kind);
            digest.bytes(
                seismic_lang::registry::representation_info(local.representation)
                    .name
                    .as_bytes(),
            );
            digest.hashed(&local.extents.len());
            digest.hashed(&local.alignment);
        }
        digest.hashed(&data.addressable_resources().len());
        for lease in data.addressable_resources() {
            digest.bytes(lease.class.stable_name.as_bytes());
            digest.bytes(lease.class.unit_name.as_bytes());
            digest.hashed(&lease.class.ownership);
            digest.hashed(&lease.class.realization);
            digest.hashed(&lease.alignment_units);
            digest.hashed(&lease.lifetime);
        }
        digest.hashed(&data.blocks().len());
        for block in data.blocks() {
            digest.hashed(&block.ops.len());
            for op in &block.ops {
                match op {
                    Op::Constant {
                        out,
                        value: constant,
                    } => {
                        digest.bytes(b"constant");
                        value(digest, *out);
                        match constant {
                            crate::kernel::ops::ConstantValue::F32(value) => {
                                digest.bytes(b"f32");
                                digest.hashed(&value.to_bits())
                            }
                            crate::kernel::ops::ConstantValue::F16(value) => {
                                digest.bytes(b"f16");
                                digest.hashed(value)
                            }
                            crate::kernel::ops::ConstantValue::BF16(value) => {
                                digest.bytes(b"bf16");
                                digest.hashed(value)
                            }
                            crate::kernel::ops::ConstantValue::I32(value) => {
                                digest.bytes(b"i32");
                                digest.hashed(value)
                            }
                            crate::kernel::ops::ConstantValue::U32(value) => {
                                digest.bytes(b"u32");
                                digest.hashed(value)
                            }
                            crate::kernel::ops::ConstantValue::Bool(value) => {
                                digest.bytes(b"bool");
                                digest.hashed(value)
                            }
                            crate::kernel::ops::ConstantValue::Index(value) => {
                                digest.bytes(b"index");
                                digest.hashed(value)
                            }
                        }
                    }
                    Op::Binary { op, out, a, b } => {
                        digest.bytes(b"binary");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Unary { op, out, a } => {
                        digest.bytes(b"unary");
                        digest.hashed(op);
                        values(digest, &[*out, *a]);
                    }
                    Op::Bit { op, out, a, b } => {
                        digest.bytes(b"bit");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Fma { out, a, b, c } => {
                        digest.bytes(b"fma");
                        values(digest, &[*out, *a, *b, *c]);
                    }
                    Op::VectorSplat { out, value: scalar } => {
                        digest.bytes(b"vector-splat");
                        values(digest, &[*out, *scalar]);
                    }
                    Op::VectorBinary { op, out, a, b } => {
                        digest.bytes(b"vector-binary");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::VectorUnary { op, out, a } => {
                        digest.bytes(b"vector-unary");
                        digest.hashed(op);
                        values(digest, &[*out, *a]);
                    }
                    Op::VectorBit { op, out, a, b } => {
                        digest.bytes(b"vector-bit");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::VectorFma { out, a, b, c } => {
                        digest.bytes(b"vector-fma");
                        values(digest, &[*out, *a, *b, *c]);
                    }
                    Op::VectorCast { out, a, to } => {
                        digest.bytes(b"vector-cast");
                        digest.hashed(to);
                        values(digest, &[*out, *a]);
                    }
                    Op::VectorLane { out, vector, lane } => {
                        digest.bytes(b"vector-lane");
                        digest.hashed(lane);
                        values(digest, &[*out, *vector]);
                    }
                    Op::VectorReduceAdd { out, vector } => {
                        digest.bytes(b"vector-reduce-add");
                        values(digest, &[*out, *vector]);
                    }
                    Op::Math {
                        op,
                        precision,
                        out,
                        a,
                    } => {
                        digest.bytes(b"math");
                        digest.hashed(op);
                        digest.hashed(precision);
                        values(digest, &[*out, *a]);
                    }
                    Op::Cast { out, a, to } => {
                        digest.bytes(b"cast");
                        digest.hashed(to);
                        values(digest, &[*out, *a]);
                    }
                    Op::Bitcast { out, a, to } => {
                        digest.bytes(b"bitcast");
                        digest.hashed(to);
                        values(digest, &[*out, *a]);
                    }
                    Op::Cmp { op, out, a, b } => {
                        digest.bytes(b"cmp");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Select { out, cond, a, b } => {
                        digest.bytes(b"select");
                        values(digest, &[*out, *cond, *a, *b]);
                    }
                    Op::Logic { op, out, a, b } => {
                        digest.bytes(b"logic");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Not { out, a } => {
                        digest.bytes(b"not");
                        values(digest, &[*out, *a]);
                    }
                    Op::Geometry { out, kind } => {
                        digest.bytes(b"geometry");
                        digest.hashed(kind);
                        value(digest, *out);
                    }
                    Op::NatArg { out, index } => {
                        digest.bytes(b"nat-arg");
                        digest.hashed(index);
                        value(digest, *out);
                    }
                    Op::ScalarArg { out, index } => {
                        digest.bytes(b"scalar-arg");
                        digest.hashed(index);
                        value(digest, *out);
                    }
                    Op::Read {
                        out,
                        place: source,
                        representation,
                        index,
                    } => {
                        digest.bytes(b"read");
                        value(digest, *out);
                        place(digest, *source);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                    }
                    Op::VectorRead {
                        out,
                        place: source,
                        representation,
                        index,
                        axis,
                        active,
                    } => {
                        digest.bytes(b"vector-read");
                        value(digest, *out);
                        place(digest, *source);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        digest.hashed(axis);
                        value(digest, *active);
                    }
                    Op::VectorWrite {
                        place: destination,
                        representation,
                        index,
                        axis,
                        active,
                        value: stored,
                    } => {
                        digest.bytes(b"vector-write");
                        place(digest, *destination);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        digest.hashed(axis);
                        values(digest, &[*active, *stored]);
                    }
                    Op::ReadPlane {
                        out,
                        place: source,
                        plane,
                        index,
                    } => {
                        digest.bytes(b"read-plane");
                        value(digest, *out);
                        place(digest, *source);
                        digest.hashed(plane);
                        values(digest, index);
                    }
                    Op::RepresentationConvertPacket {
                        source,
                        destination,
                        conversion,
                        packet,
                    } => {
                        digest.bytes(b"representation-convert-packet");
                        place(digest, PlaceRef::Global { slot: *source });
                        place(digest, PlaceRef::Global { slot: *destination });
                        let conversion =
                            seismic_lang::registry::representation_conversion_info(*conversion);
                        digest.bytes(
                            seismic_lang::registry::representation_info(conversion.source)
                                .name
                                .as_bytes(),
                        );
                        digest.bytes(
                            seismic_lang::registry::representation_info(conversion.destination)
                                .name
                                .as_bytes(),
                        );
                        value(digest, *packet);
                    }
                    Op::Write {
                        place: destination,
                        representation,
                        index,
                        value: stored,
                    } => {
                        digest.bytes(b"write");
                        place(digest, *destination);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        value(digest, *stored);
                    }
                    Op::Extent {
                        out,
                        place: source,
                        axis,
                    } => {
                        digest.bytes(b"extent");
                        value(digest, *out);
                        place(digest, *source);
                        digest.hashed(axis);
                    }
                    Op::Atomic {
                        op,
                        place: destination,
                        representation,
                        index,
                        value: stored,
                    } => {
                        digest.bytes(b"atomic");
                        digest.hashed(op);
                        place(digest, *destination);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        value(digest, *stored);
                    }
                    Op::StoreSlot {
                        slot,
                        value: stored,
                        election,
                    } => {
                        digest.bytes(b"store-slot");
                        digest.hashed(slot);
                        digest.hashed(election);
                        value(digest, *stored);
                    }
                    Op::Barrier(scope) => {
                        digest.bytes(b"barrier");
                        digest.hashed(scope);
                    }
                    Op::Intrinsic {
                        intrinsic,
                        op,
                        outs,
                        args,
                        mapping_dependencies,
                    } => {
                        digest.bytes(b"intrinsic");
                        let signature = seismic_lang::registry::intrinsic_signature(*intrinsic);
                        digest.bytes(signature.name.as_bytes());
                        let mut identity = crate::identity::IntrinsicIdentityBuilder::new();
                        B::write_intrinsic_identity(op, &mut identity);
                        digest.bytes(&identity.finish());
                        values(digest, outs);
                        values(digest, args);
                        digest.bytes(b"mapping-dependencies");
                        values(digest, mapping_dependencies);
                    }
                    Op::Branch {
                        cond,
                        then,
                        otherwise,
                        outs,
                    } => {
                        digest.bytes(b"branch");
                        value(digest, *cond);
                        digest.hashed(&then.ordinal());
                        digest.hashed(&otherwise.ordinal());
                        values(digest, outs);
                    }
                    Op::Repeat {
                        start,
                        end,
                        binder,
                        carries_in,
                        carry_params,
                        body,
                        outs,
                    } => {
                        digest.bytes(b"repeat");
                        digest.hashed(&binder.ordinal());
                        values(digest, &[*start, *end]);
                        values(digest, carries_in);
                        values(digest, carry_params);
                        digest.hashed(&body.ordinal());
                        values(digest, outs);
                    }
                    Op::Yield { values: yielded } => {
                        digest.bytes(b"yield");
                        values(digest, yielded);
                    }
                }
            }
        }
        for intrinsic in data.intrinsics_used() {
            let signature = seismic_lang::registry::intrinsic_signature(*intrinsic);
            let capability = seismic_lang::registry::capability_info(signature.capability);
            digest.hashed(&capability.backend);
            digest.bytes(capability.name.as_bytes());
            digest.bytes(signature.name.as_bytes());
        }
        digest.hashed(data.resource_facts());
    }

    fn digest_schedule<B: Backend>(
        digest: &mut crate::identity::StructureDigest,
        builder: &Builder<'_, B>,
        steps: &[ScheduleStep],
    ) {
        digest.hashed(&steps.len());
        for step in steps {
            match step {
                ScheduleStep::Launch(id) => {
                    digest.bytes(b"launch");
                    digest.hashed(&id.index());
                }
                ScheduleStep::Copy(copy) => {
                    digest.bytes(b"copy");
                    digest.hashed(&(copy.source.index, copy.destination.index));
                }
                ScheduleStep::Fill(fill) => {
                    digest.bytes(b"fill");
                    digest.hashed(&fill.destination.index);
                    digest.hashed(&fill.value);
                }
                ScheduleStep::ScalarMove(value) => {
                    digest.bytes(b"scalar-move");
                    digest.hashed(&(value.from.index, value.to.index));
                }
                ScheduleStep::ScalarRead(value) => {
                    digest.bytes(b"scalar-read");
                    digest.hashed(&(value.source.index, value.index.len(), value.to.index));
                    digest.hashed(&value.bounds.len());
                }
                ScheduleStep::Check(value) => {
                    digest.bytes(b"check");
                    digest.hashed(&value.condition.index);
                    digest.bytes(match value.expectation {
                        crate::schedule::ScalarCheckExpectation::BoolTrue => b"bool-true",
                        crate::schedule::ScalarCheckExpectation::U32Zero => b"u32-zero",
                    });
                    digest.bytes(value.site.reason.as_bytes());
                    digest.bytes(value.site.path.as_bytes());
                    digest.hashed(&value.site.line);
                }
                ScheduleStep::If {
                    then_steps,
                    else_steps,
                    ..
                } => {
                    digest.bytes(b"if");
                    digest_schedule(digest, builder, then_steps);
                    digest_schedule(digest, builder, else_steps);
                }
                ScheduleStep::Repeat { body, .. } => {
                    digest.bytes(b"repeat");
                    digest_schedule(digest, builder, body);
                }
                ScheduleStep::Choose { decision, options } => {
                    digest.bytes(b"choose");
                    digest.hashed(
                        &builder
                            .decisions
                            .iter()
                            .position(|(candidate, _)| candidate == decision),
                    );
                    for (value, body) in options {
                        digest.hashed(value);
                        digest_schedule(digest, builder, body);
                    }
                }
            }
        }
    }
}
