//! Numerical transfer taxonomy and composition.
//!
//! The taxonomy is hosted in `seismic-realization::numerics` so
//! every crate below the compiler shares the one definition; this module
//! re-exports it. See the host for the semantics: `NumericalTransfer`
//! (Exact/Round/Reassociate/Approximate/Capability/Unknown), symbolic
//! `CountExpr`, `compose`/`compose_all`, `satisfies_policy`, evidence keys,
//! and `unit_roundoff`.
//!
//! The registry-level form (`seismic_lang::intrinsics::NumericalTransfer`)
//! describes authored capability effects and is converted by
//! `NumericalTransfer::from_registry`.

pub use seismic_realization::numerics::{
    accepts_evidence, compose, compose_all, default_tolerance, satisfies_policy, unit_roundoff,
    AssignmentFingerprint, CapabilitySignatureId, CountExpr, EvidenceKey, NumericalEvidence,
    NumericalTransfer, PolicyDecision, ReductionTopology, ToolchainId, WorkloadFingerprint,
};
