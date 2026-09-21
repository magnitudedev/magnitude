//! The CPU target: worker limits, the effective target profile, and the
//! retained native compilation policy.
//!
//! The profile decides legality during strategy formation and solver export.
//! CPU cost is uncalibrated: it affects ranking only, never legality, so the
//! capability fingerprint carries only facts that change semantics —
//! architecture, worker limit, scratch bound, and the versioned software-math
//! identity of the host sequences this crate owns.

use seismic_realization::target::{EffectiveTargetProfile, TargetLimits};
use std::collections::BTreeSet;

pub const TARGET: &str = "cpu";
pub const SCRATCH_BYTES: u64 = 64 << 20;

/// The identity of the versioned `seismic_math` host sequences (`lib.rs`),
/// which the reference interpreter and the CPU encoder share by construction.
pub const SEISMIC_MATH_IDENTITY: &str = "seismic_math";
pub const SEISMIC_MATH_VERSION: u32 = 1;

/// Facts of the host a CPU target executes on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub workers: u64,
    pub max_scratch_bytes: u64,
}

impl Limits {
    pub fn host(workers: u64) -> Self {
        Self {
            workers,
            max_scratch_bytes: SCRATCH_BYTES,
        }
    }
}

/// The CPU backend: profile, catalog, and retained codegen policy.
pub struct Cpu {
    limits: Limits,
    profile: EffectiveTargetProfile,
    codegen: crate::codegen::Policy,
    catalog: crate::catalog::CpuCatalog,
}

impl Cpu {
    pub fn new(limits: Limits) -> Result<Self, String> {
        if limits.workers == 0 || i64::try_from(limits.max_scratch_bytes).is_err() {
            return Err(format!(
                "CPU compilation needs at least one worker and representable scratch; \
                 received {} workers and {} bytes",
                limits.workers, limits.max_scratch_bytes
            ));
        }
        let codegen = crate::codegen::Policy::host()?;
        let profile = target_profile(&limits);
        let catalog = crate::catalog::CpuCatalog::new(&profile.limits);
        Ok(Self {
            limits,
            profile,
            codegen,
            catalog,
        })
    }

    pub fn host(workers: u64) -> Result<Self, String> {
        Self::new(Limits::host(workers))
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn profile(&self) -> &EffectiveTargetProfile {
        &self.profile
    }

    pub(crate) fn codegen_policy(&self) -> &crate::codegen::Policy {
        &self.codegen
    }

    pub(crate) fn catalog(&self) -> &crate::catalog::CpuCatalog {
        &self.catalog
    }
}

pub fn capability_fingerprint(limits: &Limits) -> String {
    format!(
        "seismic-cpu-v4:{}:workers={}:scratch={}:{}-v{}",
        std::env::consts::ARCH,
        limits.workers,
        limits.max_scratch_bytes,
        SEISMIC_MATH_IDENTITY,
        SEISMIC_MATH_VERSION,
    )
}

/// The effective CPU target profile. The CPU backend implements the whole
/// portable matrix and offers no backend intrinsics, so the effective
/// signature set is empty and every authored capability use routes to the
/// portable reference body (or is inapplicable on this target). A CPU launch
/// is one flat participant pool: no workgroup storage, no subgroups, and no
/// cooperative grid facility.
pub fn target_profile(limits: &Limits) -> EffectiveTargetProfile {
    EffectiveTargetProfile {
        backend: TARGET.into(),
        capability_fingerprint: capability_fingerprint(limits),
        toolchain_fingerprint: format!(
            "cranelift-{}-{}",
            env!("SEISMIC_CRANELIFT_VERSION"),
            std::env::consts::ARCH
        ),
        effective_signatures: BTreeSet::new(),
        limits: TargetLimits {
            max_participants: limits.workers,
            // One CPU launch is one flat pool of participants covering the
            // domain by grid stride or pull claims; the derived per-axis
            // workgroup geometry (ceil of work over participants) is
            // unbounded above.
            max_workgroups_axis: [u64::MAX, 1, 1],
            // Workgroup-scope storage does not exist on this target.
            max_workgroup_bytes: 0,
            max_explicit_private_bytes: limits.max_scratch_bytes,
            max_direct_bindings: u32::MAX,
            max_device_bytes: u64::MAX,
            cooperative_grid: None,
        },
    }
}
