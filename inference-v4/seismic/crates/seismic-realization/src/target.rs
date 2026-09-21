//! The effective target profile (frozen by C0; consumed by every package).
//!
//! Hardware, driver, toolchain, and implementation differences form one
//! effective target profile supplied once by the backend. Kernel-visible
//! capabilities are exact intrinsic signatures; all other differences are
//! compiler/backend concerns expressed as limits and the mapping catalog.

use seismic_lang::intrinsics::IntrinsicId;
use std::collections::BTreeSet;

/// Hard resource limits plus the exact effective capability signatures of
/// the compilation target. Legality is decided against this profile during
/// strategy formation and solver export, never after selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveTargetProfile {
    pub backend: String,
    pub capability_fingerprint: String,
    pub toolchain_fingerprint: String,
    pub effective_signatures: BTreeSet<IntrinsicId>,
    pub limits: TargetLimits,
}

/// Target hard limits. Every field is a solver constraint domain (package M1)
/// and is validated identically against reflected native facts (package N1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetLimits {
    pub max_participants: u64,
    pub max_workgroups_axis: [u64; 3],
    pub max_workgroup_bytes: u64,
    pub max_explicit_private_bytes: u64,
    pub max_direct_bindings: u32,
    pub max_device_bytes: u64,
    /// Whether the target can launch one cooperative grid whose participants
    /// all stay resident and can synchronize grid-wide. `None` means the
    /// facility does not exist on this target (Metal, CPU): a grid-cooperative
    /// proposal is then simply never made, never a fallback.
    pub cooperative_grid: Option<CooperativeGrid>,
}

/// The cooperative-grid facility of a target (CUDA cooperative launch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CooperativeGrid {
    /// The maximum number of participants that can be simultaneously
    /// resident, so that a grid-wide barrier cannot deadlock. The solver
    /// bounds a `GridCooperative` launch's participants by it.
    pub max_resident_participants: u64,
}
