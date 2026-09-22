//! Candidate-family construction and reconciled, handle-free implementations.
//!
//! An `Implementation<B>` owns, together: identity, semantic coverage,
//! parametric schedule, kernels, global and local allocation topology, finite
//! decisions, hard constraints, numerical transfer, and provenance.
//! All fields are private. Refinement creates `ImplementationBuilder`s and
//! hands them to structural factories; builders close into pure
//! `CandidateFamily` values. Native realization is coordinated by
//! [`crate::realization`]; an `Implementation` retains only immutable reflected
//! descriptions, never executable handles.
//!
//! Factories receive a request and a builder. Before construction they may
//! decline; once construction begins they return a closed implementation or
//! a real preparation error. They cannot return a graph, proposal, label,
//! partial placement, side table, or callback.
//!
//! Calls are resolved during construction: `splice_call` constructs every
//! applicable child implementation and splices it under a finite decision,
//! composing guards, constraints, lifetimes, transfers, provenance,
//! and effect ordering. After construction no call exists.
//!
//! W4 owns the internals.

pub(crate) mod native;

use crate::numerics::NumericalTransfer;
use crate::refinement::{
    CandidateFamily, CandidateFamilyIdentity, CandidateFamilyParts, ConstructionAuthority,
    FactoryIdentity, ImplementationProvenance, PublishedResult, PublishedScalarKind,
    RefinementBudget, ResultPublication,
};
use crate::target::{CompilerRegistry, TargetConstants};
use seismic_ir::construction::Construction;
use seismic_ir::kernel::{KernelArena, KernelBuilder};
use seismic_ir::repr::{Representation, ScalarType};
use seismic_ir::schedule::{
    AnyScalarSlot, ClosedSchedule, ParametricSchedule, ScalarSlotId, ScheduleBuilder,
};
use seismic_ir::storage::{
    AnyBufferView, BufferViewId, GlobalAllocationId, GlobalAllocationTopology,
    LocalAllocationTopology,
};
use seismic_lang::entry::{
    AccessKind, AliasRule, CallSchema, CandidateKind, ParameterAccess, ParameterKind,
    SemanticFunction, SemanticNodeView, SemanticProgram, SemanticType, TensorStorage,
};
use seismic_lang::expr::{
    AnyExpr, BoolExpr, DecisionId, ExprArena, FiniteDomain, NatExpr, NodeView, TargetPredicate,
};
use seismic_lang::ids::{FamilyId, NodeId, RegionId, SemanticValueId, StableFunctionId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::registry::BackendName;
use seismic_lang::types::DType;
use seismic_target::DeviceDescription;
use std::fmt;
use std::sync::Arc;

/// Stable identity of one natively realized implementation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImplementationIdentity {
    pub factory: FactoryIdentity,
    /// Digest over the candidate structure and reconciled native artifacts.
    pub structure: [u8; 32],
}

/// Stable pre-compilation identities for a family's native templates.  This
/// sits on the realization side of the boundary so [`CandidateFamily`] has no
/// native operation.
pub(crate) fn candidate_native_template_identities<B: seismic_target::TargetFamily>(
    family: &CandidateFamily<B>,
    target: &DeviceDescription<B>,
) -> Vec<[u8; 32]> {
    family
        .executable
        .kernels()
        .kernels()
        .enumerate()
        .map(|(ordinal, _)| {
            let mut digest =
                seismic_ir::identity::StructureDigest::new("seismic-native-template-v1");
            digest.bytes(&family.identity.structure);
            digest.hashed(&ordinal);
            digest.bytes(&target.compatibility_identity().fingerprint);
            digest.finish()
        })
        .collect()
}

/// Consumes one pure family into a natively reconciled implementation.
///
/// Universal launch specialization uses only reflected native legality.
/// Performance evaluation is owned by `evaluation` after the whole candidate
/// domain has been sealed and cannot change executable structure.
pub(crate) fn reconcile_candidate_with_descriptions<B: seismic_target::TargetFamily>(
    mut family: CandidateFamily<B>,
    arena: &mut ExprArena,
    target: &DeviceDescription<B>,
    registry: &CompilerRegistry<B>,
    constants: &TargetConstants,
    native_descriptions: Vec<seismic_target::NativeKernelDescription<B>>,
) -> Result<Implementation<B>, crate::errors::PreparationError> {
    let mut total_launch_certificate = None;
    if family.authority == ConstructionAuthority::UniversalPortable {
        let (certificate, chunks) = specialize_universal_launches(
            arena,
            constants,
            family.executable.schedule(),
            &native_descriptions,
        )?;
        family.executable = family.executable.chunk_semantic_launches(arena, chunks);
        total_launch_certificate = Some(certificate);
    }

    let (reflected, native_launch_modes) = native_hard_constraints(
        arena,
        target,
        registry,
        family.executable.schedule(),
        family.executable.kernels(),
        family.executable.launch_layouts(),
        &native_descriptions,
        family.authority != ConstructionAuthority::UniversalPortable,
    );
    let combined = arena.and(family.hard_constraints, reflected);
    let side_conditions = arena.side_conditions(AnyExpr::Bool(combined));
    family.hard_constraints = arena.and(side_conditions, combined);

    let mut digest = seismic_ir::identity::StructureDigest::new("seismic-implementation-native-v1");
    digest.bytes(&family.identity.structure);
    for native in &native_descriptions {
        digest.bytes(&native.identity.compatibility.fingerprint);
        digest.bytes(&native.identity.artifact_digest);
        digest.bytes(&native.numerical_identity.fingerprint);
    }
    let identity = ImplementationIdentity {
        factory: family.identity.factory.clone(),
        structure: digest.finish(),
    };
    Ok(Implementation {
        family,
        identity,
        native_descriptions,
        native_launch_modes,
        total_launch_certificate,
    })
}

/// Private construction witness that every semantic launch was rewritten to
/// exact chunks within its reconciled native grid domain.
/// It has no public constructor: possession is the proof consumed by
/// `UniversalImplementation`.
#[derive(Debug)]
struct TotalLaunchCertificate {
    maximum_grid_x: Vec<u64>,
}

fn specialize_universal_launches<B: seismic_target::TargetFamily>(
    arena: &mut ExprArena,
    constants: &TargetConstants,
    schedule: &ParametricSchedule,
    native_kernels: &[seismic_target::NativeKernelDescription<B>],
) -> Result<
    (
        TotalLaunchCertificate,
        Vec<(seismic_ir::schedule::LaunchId, NatExpr)>,
    ),
    crate::errors::PreparationError,
> {
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
                )));
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
                if value == 0 || value > native.launch.max_grid[axis] {
                    return Err(crate::errors::PreparationError::UniversalClosure(format!(
                        "fixed launch {ordinal} exceeds reflected grid legality"
                    )));
                }
                concrete_grid[axis] = value;
            }
            caps.push(concrete_grid[0]);
            continue;
        }
        if launch.mode != seismic_ir::schedule::LaunchMode::Independent {
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
        let cap = native.launch.max_grid[0];
        if cap == 0 {
            return Err(crate::errors::PreparationError::UniversalClosure(format!(
                "semantic launch {ordinal} has zero reflected grid capacity"
            )));
        }
        caps.push(cap);
    }
    let mut chunks = Vec::new();
    for (ordinal, cap) in caps.iter().copied().enumerate() {
        if launches[ordinal].parallel_extent.is_none() {
            continue;
        }
        let id = schedule.launch_id(ordinal as u32);
        chunks.push((id, arena.nat(cap)));
    }
    Ok((
        TotalLaunchCertificate {
            maximum_grid_x: caps,
        },
        chunks,
    ))
}

#[cfg(test)]
mod implementation_invariant_tests {
    use super::*;
    use seismic_lang::expr::{CmpOp, SymbolSort};

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

fn native_hard_constraints<B: seismic_target::TargetFamily>(
    arena: &mut ExprArena,
    target: &DeviceDescription<B>,
    registry: &CompilerRegistry<B>,
    schedule: &ParametricSchedule,
    kernels: &KernelArena<B>,
    launch_layouts: &[seismic_ir::storage::LaunchLocalLayout],
    native_descriptions: &[seismic_target::NativeKernelDescription<B>],
    include_grid_x: bool,
) -> (BoolExpr, Vec<B::NativeLaunchMode>) {
    let mut constraints = Vec::new();
    let mut native_launch_modes = Vec::with_capacity(schedule.launches().len());
    for (launch, local_layout) in schedule.launches().iter().zip(launch_layouts) {
        let native = &native_descriptions[launch.kernel.index() as usize];
        let kernel = kernels.kernel(launch.kernel);
        let domain = &native.launch;
        let required_mode = match launch.mode {
            seismic_ir::schedule::LaunchMode::Independent => {
                registry.independent_launch_mode().clone()
            }
            seismic_ir::schedule::LaunchMode::CooperativeGrid => registry
                .cooperative_launch_mode(target.facts())
                .expect("cooperative launch was constructed for an unsupported device"),
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
        native_launch_modes.push(required_mode);
        constraints.extend(registry.native_launch_constraints(
            target,
            arena,
            launch,
            local_layout,
            kernel,
            native,
        ));
    }
    assert_eq!(native_launch_modes.len(), schedule.launches().len());
    (arena.all(&constraints), native_launch_modes)
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
pub struct Implementation<B: seismic_target::TargetFamily> {
    family: CandidateFamily<B>,
    identity: ImplementationIdentity,
    native_descriptions: Vec<seismic_target::NativeKernelDescription<B>>,
    native_launch_modes: Vec<B::NativeLaunchMode>,
    total_launch_certificate: Option<TotalLaunchCertificate>,
}

impl<B: seismic_target::TargetFamily> Implementation<B> {
    pub(crate) fn retained_metadata_bytes(&self) -> u64 {
        let family = &self.family;
        let local_allocations = family.executable.local_allocations();
        let mut bytes = std::mem::size_of_val(self)
            .saturating_add(family.executable.schedule().retained_bytes())
            .saturating_add(family.executable.kernels().retained_bytes())
            .saturating_add(family.executable.storage().retained_bytes())
            .saturating_add(local_allocations.retained_bytes())
            .saturating_add(
                family.executable.allocation_constraints().len() * std::mem::size_of::<BoolExpr>(),
            )
            .saturating_add(family.executable.retained_layout_bytes())
            .saturating_add(
                family.launch_scratch.capacity()
                    * std::mem::size_of::<seismic_ir::storage::LaunchScratchRequirements>(),
            )
            .saturating_add(
                family.launch_abi.capacity()
                    * std::mem::size_of::<Vec<seismic_ir::storage::LaunchAbiRequirement>>(),
            )
            .saturating_add(
                family
                    .launch_abi
                    .iter()
                    .map(|abi| {
                        abi.capacity()
                            * std::mem::size_of::<seismic_ir::storage::LaunchAbiRequirement>()
                    })
                    .sum::<usize>(),
            )
            .saturating_add(
                family.decisions.capacity() * std::mem::size_of::<(DecisionId, &'static str)>(),
            )
            .saturating_add(
                family.provenance.callees.capacity() * std::mem::size_of::<StableFunctionId>(),
            )
            .saturating_add(
                family.result_publications.capacity() * std::mem::size_of::<ResultPublication>(),
            )
            .saturating_add(
                family
                    .result_publications
                    .iter()
                    .map(|result| result.path.capacity() * std::mem::size_of::<u32>())
                    .sum::<usize>(),
            )
            .saturating_add(
                self.native_descriptions.capacity()
                    * std::mem::size_of::<seismic_target::NativeKernelDescription<B>>(),
            );
        // The transfer owns nested output/effect/operation/evidence vectors;
        // its explicit estimator remains colocated with that representation.
        bytes = bytes.saturating_add(family.numerical_transfer.retained_bytes());
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }
    pub fn identity(&self) -> &ImplementationIdentity {
        &self.identity
    }
    pub fn semantic_coverage(&self) -> TargetPredicate {
        self.family.semantic_coverage()
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        self.family.schedule()
    }
    pub fn kernels(&self) -> &KernelArena<B> {
        self.family.kernels()
    }
    pub(crate) fn native_descriptions(&self) -> &[seismic_target::NativeKernelDescription<B>] {
        &self.native_descriptions
    }
    pub(crate) fn native_launch_modes(&self) -> &[B::NativeLaunchMode] {
        &self.native_launch_modes
    }
    pub(crate) fn native_numerical_identity(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(b"seismic-native-numerical-set-v1");
        for kernel in &self.native_descriptions {
            digest.update(kernel.numerical_identity.fingerprint);
        }
        digest.finalize().into()
    }
    pub fn global_allocations(&self) -> &GlobalAllocationTopology {
        self.family.global_allocations()
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        self.family.local_allocations()
    }
    pub fn launch_layouts(&self) -> &[seismic_ir::storage::LaunchLocalLayout] {
        self.family.launch_layouts()
    }
    pub fn closed_execution(&self) -> seismic_ir::execution::ClosedExecutionView<'_, B> {
        self.family.executable.view()
    }
    pub fn launch_scratch(&self) -> &[seismic_ir::storage::LaunchScratchRequirements] {
        self.family.launch_scratch()
    }
    pub fn launch_abi(&self) -> &[Vec<seismic_ir::storage::LaunchAbiRequirement>] {
        self.family.launch_abi()
    }
    pub fn decisions(&self) -> &[(DecisionId, &'static str)] {
        self.family.decisions()
    }
    pub fn hard_constraints(&self) -> BoolExpr {
        self.family.hard_constraints()
    }
    pub fn numerical_transfer(&self) -> &NumericalTransfer {
        self.family.numerical_transfer()
    }
    pub(crate) fn authority(&self) -> ConstructionAuthority {
        self.family.authority
    }
    pub(crate) fn numerical_role(&self) -> seismic_lang::entry::NumericalRole {
        self.family.numerical_role
    }
    pub fn provenance(&self) -> &ImplementationProvenance {
        self.family.provenance()
    }
    pub(crate) fn result_publications(&self) -> &[ResultPublication] {
        self.family.result_publications()
    }
}

#[derive(Clone, Debug)]
pub struct UniversalImplementation<B: seismic_target::TargetFamily>(Arc<Implementation<B>>);

impl<B: seismic_target::TargetFamily> UniversalImplementation<B> {
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
        let coverage = arena.all(&[semantic_coverage, hard_constraints]);
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
            return Err(crate::errors::PreparationError::UniversalClosure(format!(
                "post-native legality is not constructionally total over TargetDomain (target={:?} target_symbols=[{}], semantic={:?}, hard={:?} hard_detail={} hard_symbols=[{}], implication={:?})",
                arena.view(AnyExpr::Bool(target_domain)),
                symbol_summary(target_domain),
                arena.view(AnyExpr::Bool(semantic_coverage)),
                arena.view(AnyExpr::Bool(hard_constraints)),
                expression_detail(arena, AnyExpr::Bool(hard_constraints), 8),
                symbol_summary(hard_constraints),
                arena.view(AnyExpr::Bool(total)),
            )));
        }
        Ok(Self::new(implementation))
    }
    pub(crate) fn shared(&self) -> Arc<Implementation<B>> {
        self.0.clone()
    }
    pub(crate) fn as_inner(&self) -> &Implementation<B> {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct OptimizedImplementation<B: seismic_target::TargetFamily>(Arc<Implementation<B>>);

impl<B: seismic_target::TargetFamily> OptimizedImplementation<B> {
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

/// What a factory sees.
pub struct FactoryRequest<'a, B: seismic_target::TargetFamily> {
    /// The function body to implement.
    pub function: &'a SemanticFunction,
    pub program: &'a SemanticProgram,
    pub contract: &'a FunctionContract,
    pub target: &'a DeviceDescription<B>,
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
        layout: seismic_ir::storage::BufferViewLayout,
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
        layout: seismic_ir::storage::BufferViewLayout,
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
pub trait ImplementationFactory<B: seismic_target::TargetFamily>: Send + Sync {
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
    ) -> CandidateFamily<B>;
}

/// The one way to build an implementation. Created while closing a candidate
/// domain.
pub struct ImplementationBuilder<'a, B: seismic_target::TargetFamily> {
    inner: internals::Builder<'a, B>,
}

impl<'a, B: seismic_target::TargetFamily> ImplementationBuilder<'a, B> {
    pub(crate) fn new(
        arena: &'a mut ExprArena,
        program: &'a SemanticProgram,
        function: &'a SemanticFunction,
        target: &'a DeviceDescription<B>,
        registry: &'a CompilerRegistry<B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
        semantic_coverage: TargetPredicate,
        factory: FactoryIdentity,
        authority: ConstructionAuthority,
        numerical_role: seismic_lang::entry::NumericalRole,
        site: CallSite<'a>,
        root_schema: Option<&'a CallSchema>,
        budget: RefinementBudget,
    ) -> Self {
        Self {
            inner: internals::Builder::new(
                arena,
                program,
                function,
                target,
                registry,
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
        target: &'a DeviceDescription<B>,
        registry: &'a CompilerRegistry<B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
        semantic_coverage: TargetPredicate,
        factory: FactoryIdentity,
        numerical_role: seismic_lang::entry::NumericalRole,
        site: CallSite<'a>,
        root_schema: Option<&'a CallSchema>,
        budget: RefinementBudget,
    ) -> Self {
        Self {
            inner: internals::Builder::new(
                arena,
                program,
                function,
                target,
                registry,
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
    pub fn target(&self) -> &DeviceDescription<B> {
        self.inner.target()
    }
    pub(crate) fn portable_target_ref(&self) -> &'a DeviceDescription<B> {
        self.inner.target
    }
    pub(crate) fn portable_registry_ref(&self) -> &'a CompilerRegistry<B> {
        self.inner.registry
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

    pub(crate) fn portable_kernel(
        &mut self,
    ) -> seismic_ir::kernel::dynamic::PortableBuilder<'_, B> {
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
    pub(crate) fn portable_layout(
        &self,
        view: AnyBufferView,
    ) -> seismic_ir::storage::BufferViewLayout {
        self.inner.portable_layout(view)
    }
    pub(crate) fn portable_preflight_status(&mut self) -> PortablePreflightStatus {
        self.inner.portable_preflight_status()
    }
    pub(crate) fn portable_finish_preflight(
        &mut self,
        status: PortablePreflightStatus,
        site: seismic_ir::kernel::ops::CheckSite,
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
        body: impl FnOnce(&mut Self, seismic_ir::schedule::LoopBinding) -> Result<T, E>,
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
    pub fn close(self, schedule: ClosedSchedule) -> CandidateFamily<B> {
        self.inner.close(schedule)
    }
}

/// The result of splicing a call.
#[derive(Debug)]
pub struct SplicedCall<B: seismic_target::TargetFamily> {
    pub decision: Option<DecisionId>,
    /// Caller-owned tensor or scalar result destinations.
    pub results: Vec<(SemanticValueId, ResultBinding)>,
    alternatives: Vec<(i64, seismic_ir::schedule::ImportedSchedule)>,
    marker: std::marker::PhantomData<B>,
}

impl<B: seismic_target::TargetFamily> SplicedCall<B> {
    /// Inserts the already-closed child alternatives at this exact lexical
    /// schedule region. Consumes the token, so a call is scheduled once.
    pub fn schedule(self, schedule: &mut ScheduleBuilder<'_, B>) {
        schedule.splice(self.decision, self.alternatives)
    }
}

impl<B: seismic_target::TargetFamily> fmt::Debug for ImplementationBuilder<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImplementationBuilder").finish()
    }
}

mod internals {
    use super::*;
    use crate::numerics::{
        ConditionalNumericalTransfer, ErrorBound, NumericalEffect, OutputTransfer,
    };
    use seismic_ir::schedule::ScheduleStep;
    use seismic_ir::storage::{AllocationLiveness, GlobalBufferKind};
    use seismic_lang::expr::{AnyExpr, CmpOp, RootName, SymbolSort};
    use seismic_lang::registry;
    use std::collections::HashMap;

    struct ConstructedChild<B: seismic_target::TargetFamily> {
        implementation: CandidateFamily<B>,
        /// Result identities belong to the child's semantic program owner and
        /// therefore cannot be compared with the caller's result identities.
        /// Publication is an ordinal contract.
        results: Vec<SemanticValueId>,
    }

    /// Private move slot for the open construction. Builder closure consumes
    /// the IR owner before the remaining compiler metadata has been finalized;
    /// deref keeps the open-phase implementation uncluttered without exposing
    /// an optional lifecycle in the public IR API.
    struct OpenConstruction<B: seismic_target::TargetFamily>(Option<Construction<B>>);
    impl<B: seismic_target::TargetFamily> OpenConstruction<B> {
        fn new(construction: Construction<B>) -> Self {
            Self(Some(construction))
        }
        fn take(&mut self) -> Construction<B> {
            self.0
                .take()
                .expect("builder construction is consumed exactly once at close")
        }
    }
    impl<B: seismic_target::TargetFamily> std::ops::Deref for OpenConstruction<B> {
        type Target = Construction<B>;
        fn deref(&self) -> &Self::Target {
            self.0
                .as_ref()
                .expect("builder construction is available before close")
        }
    }
    impl<B: seismic_target::TargetFamily> std::ops::DerefMut for OpenConstruction<B> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.0
                .as_mut()
                .expect("builder construction is available before close")
        }
    }

    pub(super) struct Builder<'a, B: seismic_target::TargetFamily> {
        construction: OpenConstruction<B>,
        pub(super) arena: &'a mut ExprArena,
        pub(super) program: &'a SemanticProgram,
        pub(super) function: &'a SemanticFunction,
        contract: FunctionContract,
        pub(super) target: &'a DeviceDescription<B>,
        pub(super) registry: &'a CompilerRegistry<B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
        semantic_coverage: TargetPredicate,
        factory: FactoryIdentity,
        authority: ConstructionAuthority,
        numerical_role: seismic_lang::entry::NumericalRole,
        value_views: HashMap<SemanticValueId, AnyBufferView>,
        scalar_symbols: HashMap<SemanticValueId, ValueBinding>,
        result_slots: HashMap<SemanticValueId, ScalarPublication>,
        pending_result_paths: HashMap<SemanticValueId, Vec<Vec<u32>>>,
        pub(super) schedule_region: u32,
        decisions: Vec<(DecisionId, &'static str)>,
        constraints: Vec<BoolExpr>,
        callees: Vec<StableFunctionId>,
        conditional_child_transfers: Vec<ConditionalNumericalTransfer>,
        budget: RefinementBudget,
    }

    impl<'a, B: seismic_target::TargetFamily> Builder<'a, B> {
        pub(super) fn new(
            arena: &'a mut ExprArena,
            program: &'a SemanticProgram,
            function: &'a SemanticFunction,
            target: &'a DeviceDescription<B>,
            registry: &'a CompilerRegistry<B>,
            constants: &'a TargetConstants,
            precision: &'a PrecisionPolicy,
            semantic_coverage: TargetPredicate,
            factory: FactoryIdentity,
            authority: ConstructionAuthority,
            numerical_role: seismic_lang::entry::NumericalRole,
            site: CallSite<'a>,
            root_schema: Option<&'a CallSchema>,
            budget: RefinementBudget,
        ) -> Self {
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
            let mut construction = Construction::new(
                arena,
                disjoint,
                matches!(site, CallSite::Spliced { .. }),
                target.addressable_resources().len(),
            );
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
                                            view.representation(),
                                            tensor.representation,
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
                        let bytes = seismic_ir::storage::tensor_bytes(
                            arena,
                            tensor.representation,
                            &tensor.axes,
                        );
                        let allocation = construction.storage_mut().allocate(
                            kind,
                            bytes,
                            seismic_ir::storage::representation_alignment(tensor.representation),
                        );
                        let zero = arena.nat(0);
                        let view = match imported_layout {
                            Some(layout) => construction.storage_mut().strided_view(
                                allocation,
                                tensor.representation,
                                zero,
                                tensor.axes.clone(),
                                layout.strides,
                            ),
                            None => construction.storage_mut().dense_view(
                                arena,
                                allocation,
                                tensor.representation,
                                zero,
                                tensor.axes.clone(),
                            ),
                        };
                        value_views.insert(
                            parameter.value,
                            construction.view(view, tensor.representation),
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
                                construction.storage_mut().publish_result(arena, view, path);
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
                                        view.representation(),
                                        tensor.representation,
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
                    let bytes = seismic_ir::storage::tensor_bytes(
                        arena,
                        tensor.representation,
                        &tensor.axes,
                    );
                    let allocation = construction.storage_mut().allocate(
                        kind,
                        bytes,
                        seismic_ir::storage::representation_alignment(tensor.representation),
                    );
                    let zero = arena.nat(0);
                    let view = match imported_layout {
                        Some(layout) => construction.storage_mut().strided_view(
                            allocation,
                            tensor.representation,
                            zero,
                            tensor.axes.clone(),
                            layout.strides,
                        ),
                        None => construction.storage_mut().dense_view(
                            arena,
                            allocation,
                            tensor.representation,
                            zero,
                            tensor.axes.clone(),
                        ),
                    };
                    let view = construction.view(view, tensor.representation);
                    value_views.insert(result.value, view);
                    if let (CallSite::Root, Some(path)) = (site, abi_path) {
                        construction.storage_mut().publish_result(arena, view, path);
                    }
                }
            }
            Self {
                construction: OpenConstruction::new(construction),
                arena,
                program,
                function,
                contract,
                target,
                registry,
                constants,
                precision,
                semantic_coverage,
                factory,
                authority,
                numerical_role,
                value_views,
                scalar_symbols,
                result_slots: HashMap::new(),
                pending_result_paths,
                schedule_region: 0,
                decisions: Vec::new(),
                constraints: Vec::new(),
                callees: Vec::new(),
                conditional_child_transfers: Vec::new(),
                budget,
            }
        }

        pub(super) fn arena(&mut self) -> &mut ExprArena {
            self.arena
        }
        pub(super) fn target(&self) -> &DeviceDescription<B> {
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
            self.construction
                .storage_mut()
                .allocate(GlobalBufferKind::Arena, bytes, alignment)
        }
        pub(super) fn view<R: Representation>(
            &mut self,
            allocation: GlobalAllocationId,
            offset: NatExpr,
            extents: Vec<NatExpr>,
        ) -> BufferViewId<R> {
            let index = self.construction.storage_mut().dense_view(
                self.arena,
                allocation,
                R::id(),
                offset,
                extents,
            );
            self.construction
                .typed_view(self.construction.view(index, R::id()))
        }
        pub(super) fn subview<R: Representation>(
            &mut self,
            base: BufferViewId<R>,
            offset: NatExpr,
            extents: Vec<NatExpr>,
            strides: Vec<NatExpr>,
        ) -> BufferViewId<R> {
            self.construction.assert_view(base.erase());
            let index = self.construction.storage_mut().subview(
                self.arena,
                base.index(),
                offset,
                extents,
                strides,
            );
            self.construction
                .typed_view(self.construction.view(index, R::id()))
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
                layout: self.construction.storage().view_layout(view).clone(),
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
                ScalarPublication::Scalar(self.construction.schedule_state().slot_any(
                    self.arena,
                    dtype,
                    T::SYMBOL_SORT,
                ))
            });
            let ScalarPublication::Scalar(slot) = publication else {
                panic!("factory requested a range publication as a scalar slot")
            };
            ScalarSlotId::from_any(slot)
        }
        pub(super) fn kernel(&mut self) -> KernelBuilder<'_, B> {
            self.construction.kernel(
                self.arena,
                self.target.facts(),
                self.target.addressable_resources(),
                self.target.vectors(),
            )
        }
        pub(super) fn schedule(&mut self) -> ScheduleBuilder<'_, B> {
            self.construction.schedule(self.arena, self.schedule_region)
        }
        pub(crate) fn portable_kernel(
            &mut self,
        ) -> seismic_ir::kernel::dynamic::PortableBuilder<'_, B> {
            self.construction.portable_kernel(
                self.arena,
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
                seismic_ir::storage::tensor_bytes(self.arena, tensor.representation, &tensor.axes);
            let allocation = self.construction.storage_mut().allocate(
                GlobalBufferKind::Arena,
                bytes,
                seismic_ir::storage::representation_alignment(tensor.representation),
            );
            let zero = self.arena.nat(0);
            let index = self.construction.storage_mut().dense_view(
                self.arena,
                allocation,
                tensor.representation,
                zero,
                tensor.axes,
            );
            let view = self.construction.view(index, tensor.representation);
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
                Some(ScalarPublication::Scalar(slot)) => Some(ValueBinding::Scalar(slot.symbol())),
                Some(ScalarPublication::Range { start, end }) => Some(ValueBinding::Range {
                    start: start.symbol(),
                    end: end.symbol(),
                }),
                None => None,
            }
        }
        pub(crate) fn portable_publish(&mut self, value: SemanticValueId) -> ScalarPublication {
            if let Some(publication) = self.result_slots.get(&value).copied() {
                return publication;
            }
            let publication = match self.function.value(value).ty {
                SemanticType::Scalar(dtype) => {
                    ScalarPublication::Scalar(self.construction.schedule_state().slot_any(
                        self.arena,
                        dtype,
                        SymbolSort::Scalar(dtype),
                    ))
                }
                SemanticType::Index { .. } => {
                    ScalarPublication::Scalar(self.construction.schedule_state().slot_any(
                        self.arena,
                        DType::U32,
                        SymbolSort::Nat,
                    ))
                }
                SemanticType::Range { .. } => ScalarPublication::Range {
                    start: self.construction.schedule_state().slot_any(
                        self.arena,
                        DType::U32,
                        SymbolSort::Nat,
                    ),
                    end: self.construction.schedule_state().slot_any(
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
                representation,
                base.representation(),
                "view transform changed representation without an explicit plane/decode operation"
            );
            let index = self.construction.storage_mut().subview(
                self.arena,
                base.index(),
                offset,
                extents,
                strides,
            );
            let view = self.construction.view(index, representation);
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
            let base_layout = self.construction.storage().view_layout(base).clone();
            let absolute = self.arena.nat_add(base_layout.offset, offset);
            let index = self.construction.storage_mut().strided_view(
                base_layout.allocation,
                representation,
                absolute,
                extents,
                strides,
            );
            let view = self.construction.view(index, representation);
            self.value_views.insert(value, view);
            view
        }
        pub(crate) fn portable_layout(
            &self,
            view: AnyBufferView,
        ) -> seismic_ir::storage::BufferViewLayout {
            self.construction.storage().view_layout(view).clone()
        }
        pub(crate) fn portable_preflight_status(&mut self) -> PortablePreflightStatus {
            let representation = registry::dense(DType::U32);
            let bytes = self.arena.nat(u64::from(DType::U32.bytes()));
            let allocation = self.construction.storage_mut().allocate(
                GlobalBufferKind::Arena,
                bytes,
                seismic_ir::storage::representation_alignment(representation),
            );
            let zero = self.arena.nat(0);
            let one = self.arena.nat(1);
            let index = self.construction.storage_mut().dense_view(
                self.arena,
                allocation,
                representation,
                zero,
                vec![one],
            );
            let view = self.construction.view(index, representation);
            let mut schedule = self.schedule();
            let slot = schedule.slot_any(DType::U32, SymbolSort::Scalar(DType::U32));
            schedule.fill_constant_any(view, seismic_lang::intrinsics::FillConstant::Zero);
            PortablePreflightStatus { view, slot }
        }
        pub(crate) fn portable_finish_preflight(
            &mut self,
            status: PortablePreflightStatus,
            site: seismic_ir::kernel::ops::CheckSite,
        ) {
            self.construction.assert_view(status.view);
            self.construction.assert_slot(status.slot);
            let zero = self.arena.nat(0);
            let mut schedule = self.schedule();
            schedule.scalar_read_any(status.view, vec![zero], status.slot);
            schedule.check_zero_any(status.slot, site);
        }
        pub(crate) fn portable_begin_branch(&mut self, condition: BoolExpr) -> (u32, u32) {
            self.construction
                .schedule_state()
                .begin_branch(self.schedule_region, condition)
        }
        pub(crate) fn portable_begin_repeat(
            &mut self,
            start: NatExpr,
            end: NatExpr,
        ) -> (u32, seismic_ir::schedule::LoopBinding) {
            self.construction.schedule_state().begin_repeat(
                self.arena,
                self.schedule_region,
                start,
                end,
            )
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
                            let bytes = seismic_ir::storage::tensor_bytes(
                                self.arena,
                                tensor.representation,
                                &tensor.axes,
                            );
                            let allocation = self.construction.storage_mut().allocate(
                                GlobalBufferKind::Arena,
                                bytes,
                                seismic_ir::storage::representation_alignment(
                                    tensor.representation,
                                ),
                            );
                            let zero = self.arena.nat(0);
                            let index = self.construction.storage_mut().dense_view(
                                self.arena,
                                allocation,
                                tensor.representation,
                                zero,
                                tensor.axes.clone(),
                            );
                            self.value_views.insert(
                                *output,
                                self.construction.view(index, tensor.representation),
                            );
                        }
                        let view = self.value_views[output];
                        ResultBinding::View {
                            view,
                            layout: self.construction.storage().view_layout(view).clone(),
                        }
                    }
                    SemanticType::Scalar(dtype) => {
                        let publication = *self.result_slots.entry(*output).or_insert_with(|| {
                            ScalarPublication::Scalar(self.construction.schedule_state().slot_any(
                                self.arena,
                                *dtype,
                                SymbolSort::Scalar(*dtype),
                            ))
                        });
                        let ScalarPublication::Scalar(slot) = publication else {
                            panic!("scalar call result has a range publication")
                        };
                        ResultBinding::Scalar(slot)
                    }
                    SemanticType::Index { .. } => {
                        let publication = *self.result_slots.entry(*output).or_insert_with(|| {
                            ScalarPublication::Scalar(self.construction.schedule_state().slot_any(
                                self.arena,
                                DType::U32,
                                SymbolSort::Nat,
                            ))
                        });
                        let ScalarPublication::Scalar(slot) = publication else {
                            panic!("index call result has a range publication")
                        };
                        ResultBinding::Scalar(slot)
                    }
                    SemanticType::Range { .. } => {
                        let publication = *self.result_slots.entry(*output).or_insert_with(|| {
                            ScalarPublication::Range {
                                start: self.construction.schedule_state().slot_any(
                                    self.arena,
                                    DType::U32,
                                    SymbolSort::Nat,
                                ),
                                end: self.construction.schedule_state().slot_any(
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
            let target = self.target;
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
                        target,
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
                                target,
                                self.registry,
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
                                target,
                                self.registry,
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
                                target,
                                self.registry,
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
                        target,
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
                                target,
                                self.registry,
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
                    for factory in self.registry.factories() {
                        let request = FactoryRequest {
                            function,
                            program: self.program,
                            contract: &contract,
                            target,
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
                                target,
                                self.registry,
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
                        ) => forced_slots.push((child.index(), *parent)),
                        (
                            ResultBinding::Range {
                                start: parent_start,
                                end: parent_end,
                            },
                            Some(PublishedResult::Range { start, end }),
                        ) => {
                            forced_slots.push((start.index(), *parent_start));
                            forced_slots.push((end.index(), *parent_end));
                        }
                        (ResultBinding::View { .. }, Some(PublishedResult::Buffer { .. })) => {}
                        _ => panic!(
                            "child result #{result_ordinal} publication does not match the caller destination"
                        ),
                    }
                }
                let child_ir =
                    parts
                        .executable
                        .into_ir()
                        .into_importable()
                        .unwrap_or_else(|error| {
                            panic!("spliced child produced a root-only executable: {error:?}")
                        });
                let imported = self
                    .construction
                    .import(self.arena, child_ir, &forced_slots);
                self.decisions.extend(parts.decisions);
                self.callees.push(parts.provenance.root);
                self.callees.extend(parts.provenance.callees);
                if let Some(decision) = decision {
                    let selected = self.arena.decision_is(decision, ordinal as i64);
                    self.constraints
                        .push(self.arena.implies(selected, parts.semantic_coverage.node()));
                    self.constraints
                        .push(self.arena.implies(selected, parts.hard_constraints));
                } else {
                    self.constraints.push(parts.semantic_coverage.node());
                    self.constraints.push(parts.hard_constraints);
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
        pub(super) fn close(mut self, closed: ClosedSchedule) -> CandidateFamily<B> {
            let coverage = self.semantic_coverage;
            if self.authority == ConstructionAuthority::UniversalPortable {
                assert!(
                    self.decisions.is_empty(),
                    "universal portable implementation cannot contain finite decisions"
                );
            }
            for (value, paths) in std::mem::take(&mut self.pending_result_paths) {
                let view = self
                    .value_views
                    .get(&value)
                    .copied()
                    .unwrap_or_else(|| panic!("returned tensor view was never realized"));
                for path in paths {
                    self.construction
                        .storage_mut()
                        .publish_result(self.arena, view, path);
                }
            }
            let allocation_ids = (0..self.construction.storage().allocation_count())
                .map(|ordinal| self.construction.allocation(ordinal))
                .collect::<Vec<_>>();
            let construction = self.construction.take();
            let analyzed = construction.close(closed).analyze_allocations();
            self.validate_schedule(analyzed.schedule(), analyzed.storage());
            let reuse_policy = if self.authority == ConstructionAuthority::UniversalPortable {
                crate::refinement::AllocationReusePolicy::Distinct
            } else {
                crate::refinement::AllocationReusePolicy::Explore
            };
            let (planned, reuse_decisions, reuse_constraints) =
                crate::refinement::refine_allocation_choices(self.arena, analyzed, reuse_policy)
                    .into_parts();
            self.decisions.extend(reuse_decisions);
            self.constraints.extend(reuse_constraints);
            let liveness = planned.liveness();
            let slots = planned.slots();
            let launch_layouts: Vec<_> = planned
                .schedule()
                .launches()
                .iter()
                .map(|launch| {
                    let kernel = planned
                        .kernels()
                        .get(launch.kernel.index() as usize)
                        .expect("launch kernel belongs to this implementation");
                    seismic_ir::storage::derive_launch_local_layout(
                        self.arena,
                        kernel.locals(),
                        kernel.intrinsic_resources(),
                    )
                })
                .collect();
            let launch_abi: Vec<_> = planned
                .schedule()
                .launches()
                .iter()
                .map(|launch| {
                    let kernel = planned
                        .kernels()
                        .get(launch.kernel.index() as usize)
                        .expect("launch kernel belongs to this implementation");
                    self.target
                        .kernel_abi_layout(kernel)
                        .allocations
                        .into_iter()
                        .map(|allocation| seismic_ir::storage::LaunchAbiRequirement {
                            role: allocation.role,
                            bytes: self.arena.nat(allocation.bytes),
                            alignment: allocation.alignment,
                        })
                        .collect()
                })
                .collect();
            let launch_scratch =
                self.derive_launch_scratch(planned.schedule(), planned.kernels(), &launch_layouts);
            self.derive_constraints(
                planned.storage(),
                &allocation_ids,
                planned.schedule(),
                planned.kernels(),
                &launch_layouts,
                &launch_scratch,
                &launch_abi,
            );
            let raw_constraints = self.arena.all(&self.constraints);
            let side_conditions = self.arena.side_conditions(AnyExpr::Bool(raw_constraints));
            let hard_constraints = self.arena.and(side_conditions, raw_constraints);
            let numerical_transfer = self.derive_numerics(planned.kernels());
            let roots = self.register_roots(
                planned.storage(),
                &allocation_ids,
                planned.kernels(),
                planned.schedule(),
                &launch_layouts,
                &launch_scratch,
                &launch_abi,
                hard_constraints,
                coverage,
                &numerical_transfer,
            );
            let expression_digest = self.arena.canonical_digest(&roots).bytes();
            let mut digest =
                seismic_ir::identity::StructureDigest::new("seismic-implementation-v3");
            digest.bytes(self.factory.name.as_bytes());
            digest.bytes(self.factory.revision.as_bytes());
            let (reference_math_version, reference_math_digest) =
                seismic_ir::kernel::reference_math_identity();
            digest.bytes(reference_math_version.as_bytes());
            digest.bytes(&reference_math_digest);
            digest.bytes(&expression_digest);
            digest_structure(
                &mut digest,
                &self,
                planned.storage(),
                &allocation_ids,
                planned.kernels(),
                planned.schedule(),
                liveness,
                slots,
            );
            digest_numerical(&mut digest, &self, &numerical_transfer);
            let identity = CandidateFamilyIdentity {
                factory: self.factory.clone(),
                structure: digest.finish(),
            };
            let mut result_publications = Vec::new();
            for result in &self.contract.results {
                for path in &result.paths {
                    let binding = match &result.ty {
                        SemanticType::Tensor(_) => {
                            let publication = planned
                                .storage()
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
            let executable = planned.finish().close_execution(self.arena);
            CandidateFamily::from_parts(CandidateFamilyParts {
                authority: self.authority,
                numerical_role: self.numerical_role,
                identity,
                semantic_coverage: coverage,
                executable,
                launch_scratch,
                launch_abi,
                decisions: self.decisions,
                hard_constraints,
                numerical_transfer,
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
                view.representation(),
                R::id(),
                "factory requested a semantic tensor through the wrong representation type"
            );
            self.construction.typed_view(view)
        }

        fn derive_launch_scratch(
            &mut self,
            schedule: &ParametricSchedule,
            kernels: &[seismic_ir::kernel::Kernel<B>],
            launch_layouts: &[seismic_ir::storage::LaunchLocalLayout],
        ) -> Vec<seismic_ir::storage::LaunchScratchRequirements> {
            use seismic_ir::storage::{LaunchLocalKind, ScratchRequirement};
            use seismic_ir::target::LocalRealization;
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
                    let kernel = kernels
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
                    seismic_ir::storage::LaunchScratchRequirements {
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

        fn derive_constraints(
            &mut self,
            storage: &seismic_ir::storage::TopologyBuilder,
            allocation_ids: &[GlobalAllocationId],
            schedule: &ParametricSchedule,
            kernels: &[seismic_ir::kernel::Kernel<B>],
            launch_layouts: &[seismic_ir::storage::LaunchLocalLayout],
            launch_scratch: &[seismic_ir::storage::LaunchScratchRequirements],
            launch_abi: &[Vec<seismic_ir::storage::LaunchAbiRequirement>],
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
            assert_eq!(allocation_ids.len(), storage.allocation_count() as usize);
            for &id in allocation_ids {
                let bytes = storage.allocation_bytes(id);
                self.constraints
                    .push(self.arena.nat_cmp(CmpOp::Le, bytes, max_allocation));
                if let Some(max_index) = max_index {
                    self.constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, bytes, max_index));
                }
                let required_alignment = storage.allocation_alignment(id);
                self.constraints.push(
                    self.arena
                        .bool(required_alignment <= self.target.limits().max_allocation_alignment),
                );
            }
            for layout in storage.views() {
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
                    seismic_ir::storage::tensor_bytes(
                        self.arena,
                        layout.representation,
                        &layout.extents,
                    )
                } else {
                    addressed_bytes(self.arena, layout)
                };
                let allocation_bytes = storage.allocation_bytes(layout.allocation);
                self.constraints
                    .push(self.arena.nat_cmp(CmpOp::Le, addressed, allocation_bytes));
                if let Some(max_index) = max_index {
                    self.constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, addressed, max_index));
                }
                let view_alignment =
                    seismic_ir::storage::representation_alignment(layout.representation);
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

                let kernel = kernels
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
                let data = kernel;
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

        fn validate_schedule(
            &mut self,
            schedule: &ParametricSchedule,
            storage: &seismic_ir::storage::TopologyBuilder,
        ) {
            fn visit<B: seismic_target::TargetFamily>(
                builder: &mut Builder<'_, B>,
                storage: &seismic_ir::storage::TopologyBuilder,
                steps: &[ScheduleStep],
            ) {
                for step in steps {
                    match step {
                        ScheduleStep::Launch(_)
                        | ScheduleStep::ScalarMove(_)
                        | ScheduleStep::Check(_) => {}
                        ScheduleStep::Copy(copy) => {
                            let source = storage.view_layout(copy.source).clone();
                            let destination = storage.view_layout(copy.destination).clone();
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
                            let source_bytes = seismic_ir::storage::tensor_bytes(
                                builder.arena,
                                source.representation,
                                &source.extents,
                            );
                            let destination_bytes = seismic_ir::storage::tensor_bytes(
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
                            visit(builder, storage, then_steps);
                            visit(builder, storage, else_steps);
                        }
                        ScheduleStep::Repeat { body, .. } => visit(builder, storage, body),
                        ScheduleStep::Choose { options, .. } => {
                            for (_, body) in options {
                                visit(builder, storage, body);
                            }
                        }
                    }
                }
            }
            visit(self, storage, schedule.steps());
        }

        fn derive_numerics(
            &mut self,
            kernels: &[seismic_ir::kernel::Kernel<B>],
        ) -> NumericalTransfer {
            let mut effects = Vec::new();
            let mut operations = Vec::new();
            if self.numerical_role == seismic_lang::entry::NumericalRole::Alternative {
                // A distinct authored body is not reference-equivalent merely
                // because it happened to contain no individually approximate
                // primitive. Whole-body equivalence requires operation-level
                // analysis or qualified evidence.
                effects.push(NumericalEffect::AlternativeBody);
            }
            for (kernel_ordinal, kernel) in kernels.iter().enumerate() {
                for (ordinal, (fact, multiplicity)) in kernel.fact_multiplicities().enumerate() {
                    let effect = match fact {
                        seismic_ir::kernel::ops::NumericalFact::ContractedFma => {
                            NumericalEffect::Contraction
                        }
                        seismic_ir::kernel::ops::NumericalFact::ApproximateMath(op) => {
                            NumericalEffect::ApproximateTranscendental(*op)
                        }
                        seismic_ir::kernel::ops::NumericalFact::ReassociatedIntrinsic(id) => {
                            NumericalEffect::BackendIntrinsic(*id)
                        }
                        seismic_ir::kernel::ops::NumericalFact::NarrowAccumulator(dtype) => {
                            NumericalEffect::NarrowAccumulator(*dtype)
                        }
                        seismic_ir::kernel::ops::NumericalFact::FlushToZero => {
                            NumericalEffect::FlushToZero
                        }
                        seismic_ir::kernel::ops::NumericalFact::ReassociatedReduction => {
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
            storage: &seismic_ir::storage::TopologyBuilder,
            allocation_ids: &[GlobalAllocationId],
            kernels: &[seismic_ir::kernel::Kernel<B>],
            schedule: &ParametricSchedule,
            launch_layouts: &[seismic_ir::storage::LaunchLocalLayout],
            launch_scratch: &[seismic_ir::storage::LaunchScratchRequirements],
            launch_abi: &[Vec<seismic_ir::storage::LaunchAbiRequirement>],
            hard_constraints: BoolExpr,
            coverage: TargetPredicate,
            transfer: &NumericalTransfer,
        ) -> Vec<seismic_lang::expr::RootId> {
            let mut roots = Vec::new();
            assert_eq!(allocation_ids.len(), storage.allocation_count() as usize);
            for (allocation, &id) in allocation_ids.iter().enumerate() {
                roots.push(self.arena.root(
                    RootName::AllocationBytes {
                        allocation: allocation as u32,
                    },
                    AnyExpr::Nat(storage.allocation_bytes(id)),
                ));
            }
            for (view, layout) in storage.views().iter().enumerate() {
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
            for (kernel_index, kernel) in kernels.iter().enumerate() {
                let data = kernel;
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
        layout: &seismic_ir::storage::BufferViewLayout,
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

    fn digest_structure<B: seismic_target::TargetFamily>(
        digest: &mut seismic_ir::identity::StructureDigest,
        builder: &Builder<'_, B>,
        storage: &seismic_ir::storage::TopologyBuilder,
        allocation_ids: &[GlobalAllocationId],
        kernels: &[seismic_ir::kernel::Kernel<B>],
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

        assert_eq!(allocation_ids.len(), storage.allocation_count() as usize);
        digest.hashed(&storage.allocation_count());
        for (index, &id) in allocation_ids.iter().enumerate() {
            match storage.allocation_kind(id) {
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
                        seismic_lang::registry::representation_info(source.representation())
                            .name
                            .as_bytes(),
                    );
                }
                GlobalBufferKind::Arena => digest.bytes(b"arena"),
                GlobalBufferKind::Persistent => digest.bytes(b"persistent"),
            }
            digest.hashed(&storage.allocation_alignment(id));
            digest.hashed(&liveness[index].uses().len());
            for at in liveness[index].uses() {
                digest.hashed(at.region());
                digest.hashed(&at.ordinal());
            }
            digest.hashed(&slots[index].and_then(|decision| {
                builder
                    .decisions
                    .iter()
                    .position(|(candidate, _)| *candidate == decision)
            }));
        }
        digest.hashed(&storage.views().len());
        for view in storage.views() {
            digest.hashed(&view.allocation.index());
            digest.bytes(
                seismic_lang::registry::representation_info(view.representation)
                    .name
                    .as_bytes(),
            );
            digest.hashed(&view.extents.len());
            digest.hashed(&view.contiguous);
        }
        digest.hashed(&storage.result_views().len());
        for publication in storage.result_views() {
            digest.hashed(&publication.path);
            digest.hashed(&publication.view.index());
        }

        digest.hashed(&kernels.len());
        for kernel in kernels {
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
                        digest.hashed(&slot.index());
                        digest.hashed(&slot.dtype());
                        digest.hashed(&slot.sort());
                    }
                    ScalarPublication::Range { start, end } => {
                        digest.bytes(b"range");
                        digest.hashed(&start.index());
                        digest.hashed(&end.index());
                    }
                }
            }
        }
        for callee in &builder.callees {
            digest.bytes(callee.digest());
        }
    }

    fn digest_numerical<B: seismic_target::TargetFamily>(
        digest: &mut seismic_ir::identity::StructureDigest,
        builder: &Builder<'_, B>,
        numerical: &NumericalTransfer,
    ) {
        fn role(
            digest: &mut seismic_ir::identity::StructureDigest,
            value: seismic_lang::entry::NumericalRole,
        ) {
            digest.bytes(match value {
                seismic_lang::entry::NumericalRole::Reference => b"reference",
                seismic_lang::entry::NumericalRole::Alternative => b"alternative",
            });
        }
        fn effect(digest: &mut seismic_ir::identity::StructureDigest, value: &NumericalEffect) {
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
        fn transfer<B: seismic_target::TargetFamily>(
            digest: &mut seismic_ir::identity::StructureDigest,
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

    fn digest_kernel<B: seismic_target::TargetFamily>(
        digest: &mut seismic_ir::identity::StructureDigest,
        kernel: &seismic_ir::kernel::Kernel<B>,
    ) {
        use seismic_ir::kernel::ops::{Op, PlaceRef};
        fn value(
            digest: &mut seismic_ir::identity::StructureDigest,
            value: seismic_ir::kernel::ops::ErasedValue,
        ) {
            digest.hashed(&value.ordinal());
        }
        fn values(
            digest: &mut seismic_ir::identity::StructureDigest,
            values: &[seismic_ir::kernel::ops::ErasedValue],
        ) {
            digest.hashed(&values.len());
            for item in values {
                value(digest, *item);
            }
        }
        fn place(
            digest: &mut seismic_ir::identity::StructureDigest,
            place: seismic_ir::kernel::ops::PlaceRef,
        ) {
            match place {
                seismic_ir::kernel::ops::PlaceRef::Global { slot } => {
                    digest.bytes(b"global");
                    digest.hashed(&slot.ordinal());
                }
                seismic_ir::kernel::ops::PlaceRef::Local { index } => {
                    digest.bytes(b"local");
                    digest.hashed(&index);
                }
            }
        }
        let data = kernel;
        let interface = data.interface();
        digest.hashed(&interface.bindings.len());
        for binding in &interface.bindings {
            digest.hashed(&binding.slot.index());
            digest.hashed(&binding.view.index());
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
            digest.hashed(&slot.index());
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
                            seismic_ir::kernel::ops::ConstantValue::F32(value) => {
                                digest.bytes(b"f32");
                                digest.hashed(&value.to_bits())
                            }
                            seismic_ir::kernel::ops::ConstantValue::F16(value) => {
                                digest.bytes(b"f16");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::BF16(value) => {
                                digest.bytes(b"bf16");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::I32(value) => {
                                digest.bytes(b"i32");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::U32(value) => {
                                digest.bytes(b"u32");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::Bool(value) => {
                                digest.bytes(b"bool");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::Index(value) => {
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
                        let mut identity = seismic_ir::identity::IntrinsicIdentityBuilder::new();
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

    fn digest_schedule<B: seismic_target::TargetFamily>(
        digest: &mut seismic_ir::identity::StructureDigest,
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
                    digest.hashed(&(copy.source.index(), copy.destination.index()));
                }
                ScheduleStep::Fill(fill) => {
                    digest.bytes(b"fill");
                    digest.hashed(&fill.destination.index());
                    digest.hashed(&fill.value);
                }
                ScheduleStep::ScalarMove(value) => {
                    digest.bytes(b"scalar-move");
                    digest.hashed(&(value.from.index(), value.to.index()));
                }
                ScheduleStep::ScalarRead(value) => {
                    digest.bytes(b"scalar-read");
                    digest.hashed(&(value.source.index(), value.index.len(), value.to.index()));
                    digest.hashed(&value.bounds.len());
                }
                ScheduleStep::Check(value) => {
                    digest.bytes(b"check");
                    digest.hashed(&value.condition.index());
                    digest.bytes(match value.expectation {
                        seismic_ir::schedule::ScalarCheckExpectation::BoolTrue => b"bool-true",
                        seismic_ir::schedule::ScalarCheckExpectation::U32Zero => b"u32-zero",
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
