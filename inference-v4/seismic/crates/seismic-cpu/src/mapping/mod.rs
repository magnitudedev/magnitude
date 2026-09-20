//! CPU target device: worker limits and the effective target profile.
//!
//! The profile decides legality during alternative construction.
//! CPU cost is uncalibrated: it affects ranking only, never legality, so the
//! capability fingerprint carries only facts that change semantics —
//! architecture, worker count, scratch bound, and the versioned software-math
//! identity.

use seismic_compiler::terminal::SEISMIC_MATH;
use seismic_realization::executable::{EffectiveTargetProfile, TargetLimits};
use std::collections::BTreeSet;

pub const TARGET: &str = "cpu";
pub const SCRATCH_BYTES: u64 = 64 << 20;

/// The one CPU tuning parameter: the physical linear participant count of an
/// independent domain. Every universal launch carries it; the solver selects
/// it in `1..=workers`.
pub const PARTICIPANTS_PARAMETER: &str = "cpu-participants";

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

pub struct Cpu {
    limits: Limits,
    profile: EffectiveTargetProfile,
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
        Ok(Self {
            profile: target_profile(&limits),
            limits,
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
}

pub fn capability_fingerprint(limits: &Limits) -> String {
    format!(
        "seismic-cpu-v3:{}:workers={}:scratch={}:{}-v{}",
        std::env::consts::ARCH,
        limits.workers,
        limits.max_scratch_bytes,
        SEISMIC_MATH.identity,
        SEISMIC_MATH.version,
    )
}

/// The effective CPU target profile. The CPU backend implements the whole
/// portable matrix and offers no backend intrinsics, so the effective
/// signature set is empty and every authored capability use routes to the
/// portable reference body (or is inapplicable on this target).
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
            max_participants: limits.workers as i64,
            max_workgroups_axis: [i64::MAX, 1, 1],
            max_workgroup_bytes: 0,
            max_explicit_private_bytes: limits.max_scratch_bytes as i64,
            max_direct_bindings: i64::MAX,
            max_argument_table_bytes: i64::MAX,
            max_device_bytes: i64::MAX,
        },
    }
}
