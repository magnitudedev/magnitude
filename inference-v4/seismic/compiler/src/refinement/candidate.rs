//! Pure, structurally closed candidate families produced by refinement.
//!
//! A candidate family contains every symbolic choice and constraint needed by
//! later planning, but it contains no native artifacts, measured results, or
//! evaluator output.  Refinement is therefore inspectable without invoking a
//! backend toolchain.

use crate::numerics::NumericalTransfer;
use seismic_ir::{
    kernel::KernelArena,
    schedule::{AnyScalarSlot, ParametricSchedule},
    storage::{AnyBufferView, GlobalAllocationTopology, LocalAllocationTopology},
    target::KernelDialect,
};
use seismic_lang::{
    expr::{BoolExpr, DecisionId, NatExpr, TargetPredicate},
    ids::StableFunctionId,
    types::DType,
};
use std::fmt;

/// Stable identity of one candidate family before native realization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandidateFamilyIdentity {
    pub factory: FactoryIdentity,
    /// Digest over the refined structure (schedule, kernels, topology, and
    /// decisions), independent of any solver assignment or native artifact.
    pub structure: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FactoryIdentity {
    pub name: &'static str,
    pub revision: &'static str,
}

/// Where a family's commands came from, for diagnostics and telemetry only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplementationProvenance {
    pub root: StableFunctionId,
    /// Spliced callees, in splice order.
    pub callees: Vec<StableFunctionId>,
}

/// One root result leaf in exact semantic-contract order.  Refinement seals
/// this mapping so executable translation never reconstructs result meaning.
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

/// Constructional coverage class carried explicitly into realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ConstructionAuthority {
    UniversalPortable,
    Optimized,
}

/// The pure output unit of refinement.
///
/// This owns a closed executable IR and its symbolic planning facts.  It has
/// deliberately no operation that can compile, measure, estimate, or solve.
pub struct CandidateFamily<B: KernelDialect> {
    pub(crate) authority: ConstructionAuthority,
    pub(crate) numerical_role: seismic_lang::entry::NumericalRole,
    pub(crate) identity: CandidateFamilyIdentity,
    pub(crate) semantic_coverage: TargetPredicate,
    pub(crate) executable: seismic_ir::execution::ClosedExecutableIr<B>,
    pub(crate) launch_scratch: Vec<seismic_ir::storage::LaunchScratchRequirements>,
    pub(crate) launch_abi: Vec<Vec<seismic_ir::storage::LaunchAbiRequirement>>,
    pub(crate) decisions: Vec<(DecisionId, &'static str)>,
    pub(crate) hard_constraints: BoolExpr,
    pub(crate) numerical_transfer: NumericalTransfer,
    pub(crate) provenance: ImplementationProvenance,
    pub(crate) result_publications: Vec<ResultPublication>,
}

impl<B: KernelDialect> CandidateFamily<B> {
    pub fn identity(&self) -> &CandidateFamilyIdentity {
        &self.identity
    }

    pub fn semantic_coverage(&self) -> TargetPredicate {
        self.semantic_coverage
    }

    pub fn schedule(&self) -> &ParametricSchedule {
        self.executable.schedule()
    }

    pub fn kernels(&self) -> &KernelArena<B> {
        self.executable.kernels()
    }

    pub fn global_allocations(&self) -> &GlobalAllocationTopology {
        self.executable.storage()
    }

    pub fn local_allocations(&self) -> LocalAllocationTopology {
        self.executable.local_allocations()
    }

    pub fn launch_layouts(&self) -> &[seismic_ir::storage::LaunchLocalLayout] {
        self.executable.launch_layouts()
    }

    pub fn launch_scratch(&self) -> &[seismic_ir::storage::LaunchScratchRequirements] {
        &self.launch_scratch
    }

    pub fn launch_abi(&self) -> &[Vec<seismic_ir::storage::LaunchAbiRequirement>] {
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

    pub fn provenance(&self) -> &ImplementationProvenance {
        &self.provenance
    }

    pub(crate) fn result_publications(&self) -> &[ResultPublication] {
        &self.result_publications
    }

    pub(crate) fn from_parts(parts: CandidateFamilyParts<B>) -> Self {
        Self {
            authority: parts.authority,
            numerical_role: parts.numerical_role,
            identity: parts.identity,
            semantic_coverage: parts.semantic_coverage,
            executable: parts.executable,
            launch_scratch: parts.launch_scratch,
            launch_abi: parts.launch_abi,
            decisions: parts.decisions,
            hard_constraints: parts.hard_constraints,
            numerical_transfer: parts.numerical_transfer,
            provenance: parts.provenance,
            result_publications: parts.result_publications,
        }
    }

    pub(crate) fn into_parts(self) -> CandidateFamilyParts<B> {
        CandidateFamilyParts {
            authority: self.authority,
            numerical_role: self.numerical_role,
            identity: self.identity,
            semantic_coverage: self.semantic_coverage,
            executable: self.executable,
            launch_scratch: self.launch_scratch,
            launch_abi: self.launch_abi,
            decisions: self.decisions,
            hard_constraints: self.hard_constraints,
            numerical_transfer: self.numerical_transfer,
            provenance: self.provenance,
            result_publications: self.result_publications,
        }
    }
}

impl<B: KernelDialect> fmt::Debug for CandidateFamily<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CandidateFamily")
            .field("identity", &self.identity)
            .field("authority", &self.authority)
            .field("launches", &self.executable.schedule().launches().len())
            .field("kernels", &self.executable.kernels().kernels().count())
            .finish_non_exhaustive()
    }
}

/// Crate-private carrier used only while a builder seals a family.
pub(crate) struct CandidateFamilyParts<B: KernelDialect> {
    pub authority: ConstructionAuthority,
    pub numerical_role: seismic_lang::entry::NumericalRole,
    pub identity: CandidateFamilyIdentity,
    pub semantic_coverage: TargetPredicate,
    pub executable: seismic_ir::execution::ClosedExecutableIr<B>,
    pub launch_scratch: Vec<seismic_ir::storage::LaunchScratchRequirements>,
    pub launch_abi: Vec<Vec<seismic_ir::storage::LaunchAbiRequirement>>,
    pub decisions: Vec<(DecisionId, &'static str)>,
    pub hard_constraints: BoolExpr,
    pub numerical_transfer: NumericalTransfer,
    pub provenance: ImplementationProvenance,
    pub result_publications: Vec<ResultPublication>,
}
