//! The CPU mapping catalog: no optional rules, the vacuous intrinsic
//! catalog, an uncalibrated ranking-only cost model, and the native fact
//! domains the assembler reflects.
//!
//! The core universal catalog already guarantees at least one complete,
//! resource-scalable strategy for every applicable portable alternative, so
//! the CPU adds no optional families: `rules()` is empty by design, never as
//! a fallback. Legality never lives here; the cost model ranks only.

use crate::intrinsics::{CpuDialect, CpuIntrinsicCatalog};
use seismic_compiler::pipeline::{AssemblyFailure, Backend, EncodedPlan};
use seismic_lang::sir::IntrinsicUse;
use seismic_realization::{
    kernel::{CostUnit, IntrinsicCatalog},
    physical::SealedLaunch,
    strategy::{CostModel, MappingCatalog, MappingRule, NativeFactDomain, NativeFactKind},
    target::{EffectiveTargetProfile, TargetLimits},
};

/// The CPU mapping catalog.
#[derive(Clone, Debug)]
pub struct CpuCatalog {
    intrinsics: CpuIntrinsicCatalog,
    cost: CpuCostModel,
    fact_domains: Vec<NativeFactDomain>,
    limits: TargetLimits,
}

impl CpuCatalog {
    pub fn new(limits: &TargetLimits) -> CpuCatalog {
        CpuCatalog {
            intrinsics: CpuIntrinsicCatalog,
            cost: CpuCostModel,
            fact_domains: vec![
                NativeFactDomain {
                    kind: NativeFactKind::MaxResidentParticipants,
                    min: 1,
                    max: limits.max_participants,
                },
                NativeFactDomain {
                    kind: NativeFactKind::NativeSubgroupWidth,
                    min: 1,
                    max: 1,
                },
            ],
            limits: limits.clone(),
        }
    }
}

impl MappingCatalog<CpuDialect> for CpuCatalog {
    fn rules(&self) -> &[Box<dyn MappingRule>] {
        // The core universal rules are the whole CPU planning surface.
        &[]
    }

    fn intrinsics(&self) -> &dyn IntrinsicCatalog<CpuDialect> {
        &self.intrinsics
    }

    fn cost_model(&self) -> &dyn CostModel {
        &self.cost
    }

    fn native_fact_domains(&self) -> &[NativeFactDomain] {
        &self.fact_domains
    }

    fn limits(&self) -> &TargetLimits {
        &self.limits
    }
}

/// Uncalibrated CPU cost coefficients. Ranking only; never legality.
#[derive(Clone, Debug)]
struct CpuCostModel;

impl CostModel for CpuCostModel {
    fn launch_overhead_ns(&self) -> u64 {
        200
    }

    fn point_cost_ns(&self, op: &CostUnit) -> u64 {
        match op {
            CostUnit::Scalar => 2,
            CostUnit::Load { .. } | CostUnit::Store { .. } => 4,
            // A representation-plane access folds bit extraction over one
            // or two host words.
            CostUnit::PlaneAccess { .. } => 8,
            // The versioned software-math sequences are host calls.
            CostUnit::Math { .. } => 40,
            // One compare/exchange round trip on the element word.
            CostUnit::Atomic { .. } => 50,
            CostUnit::Fold { .. } => 6,
            CostUnit::Barrier => 20,
            // The CPU declares no intrinsic families; an intrinsic cost unit
            // cannot be presented to this model.
            // Cost units are derived from the sealed op stream (M1); the
            // CPU intrinsic enum is uninhabited, so no intrinsic unit can
            // be presented to this model.
            CostUnit::Intrinsic(_) => {
                unreachable!("a cost unit names an intrinsic family on a target with no intrinsics (M1)")
            }
        }
    }
}

impl Backend for crate::mapping::Cpu {
    type Dialect = CpuDialect;
    type Catalog = CpuCatalog;
    type EncodedLaunch = crate::encode::EncodedLaunch;
    type NativeArtifact = crate::native::NativeArtifact;

    fn profile(&self) -> &EffectiveTargetProfile {
        self.profile()
    }

    fn catalog(&self) -> &CpuCatalog {
        self.catalog()
    }

    fn supports_intrinsic(&self, _intrinsic: &IntrinsicUse) -> Result<(), String> {
        // No effective signatures: every authored capability use routes to
        // its portable reference body or is inapplicable on this target.
        Err("the CPU target implements no capability intrinsics".into())
    }

    fn encode(&self, launch: &SealedLaunch<CpuDialect>) -> crate::encode::EncodedLaunch {
        crate::encode::encode(launch)
    }

    fn assemble(
        &self,
        plan: EncodedPlan<CpuDialect, crate::encode::EncodedLaunch>,
    ) -> Result<crate::native::NativeArtifact, AssemblyFailure> {
        crate::native::assemble(plan, self.codegen_policy(), self.catalog())
    }
}
