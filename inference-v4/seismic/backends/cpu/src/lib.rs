//! The CPU backend (spec §2.3, §24.1 R9): host target discovery and
//! profile, the CPU capability registry (no capability intrinsics;
//! structural factories only), the Cranelift native compiler that consumes
//! closed typed kernels, and the executor that runs the core-owned command
//! vocabulary over host buffers and the worker pool.
//!
//! Execution model. A launch is a grid of workgroups. The worker pool is
//! split into teams of `workgroup threads` workers (the profile bounds a
//! workgroup by the worker count); each team claims workgroups from a
//! shared counter and its members run one thread of the workgroup each.
//! A workgroup barrier is a team barrier; workgroup storage is a per-team
//! scratch and participant storage is a per-worker scratch, both bounded
//! by the documented scratch policy in the profile. A one-thread workgroup
//! degenerates to independent workers claiming work items.
//!
//! Numerics. Floating arithmetic follows the registry reference model:
//! every operation is computed exactly enough to round once at the result
//! dtype; transcendentals go through the versioned host sequences
//! (`seismic_math`); contraction happens only where the kernel IR names
//! `Fma`; narrow floats round through the registry rounding.

mod buffer;
mod codegen;
mod command;
mod compile;
mod emit;
mod executor;
mod factory;
pub mod model;
mod numeric;
mod open;
mod profile;
mod registry;
mod services;
mod workers;

pub use buffer::Buffer;
pub use command::CompiledKernel;
pub use compile::NativeCandidate;
pub use executor::{Device, Executor};
pub use open::{open_host, OpenedCpu};
pub use profile::{HostFacts, HostKernelAbi, ScratchPolicy, SimdTier, BACKEND_REVISION};
pub use registry::registry;
pub use workers::Workers;

use seismic_ir::target::IntrinsicIdentityBuilder;
use seismic_lang::registry::BackendName;
use seismic_target::{
    DeviceDescription, NativeArtifactMetrics, NativeCompilationError, NativeKernelDescription,
    NativeKernelIdentity, NativeKernelReflection, NativeLaunchDomain, NativeNumericalModeIdentity,
    NativeResourceUsage, NativeResources,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

/// The CPU backend marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cpu;

/// The CPU has no capability intrinsic families: every authored capability
/// use is inapplicable on this target and the portable body serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuIntrinsic {}

/// The CPU has only independent launches. A backend-native unit-like type
/// makes cooperative launch unrepresentable in a prepared CPU executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuLaunchMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuNumericalMode;

impl seismic_ir::target::KernelDialect for Cpu {
    const NAME: BackendName = BackendName::Cpu;
    type Intrinsic = CpuIntrinsic;
    type Facts = HostFacts;

    fn write_intrinsic_identity(
        intrinsic: &Self::Intrinsic,
        _identity: &mut IntrinsicIdentityBuilder,
    ) {
        match *intrinsic {}
    }

    fn intrinsic_numerics(
        _facts: &Self::Facts,
        _signature: &seismic_lang::registry::IntrinsicSignature,
        intrinsic: &Self::Intrinsic,
    ) -> seismic_ir::target::IntrinsicNumericalSemantics {
        match *intrinsic {}
    }

    fn intrinsic_addressable_resources(
        intrinsic: &Self::Intrinsic,
    ) -> Vec<seismic_ir::kernel::ops::AddressableResourceHandle> {
        match *intrinsic {}
    }
}

impl seismic_target::TargetFamily for Cpu {
    type KernelAbi = HostKernelAbi;
    type NativeLaunchMode = CpuLaunchMode;
    type NativeNumericalMode = CpuNumericalMode;
    type NativeProperties = ();
}

pub struct CpuNativeCompiler;
static CPU_NATIVE_COMPILER: CpuNativeCompiler = CpuNativeCompiler;

pub fn native_compiler() -> &'static CpuNativeCompiler {
    &CPU_NATIVE_COMPILER
}

impl seismic_target::NativeCompiler<Cpu> for CpuNativeCompiler {
    type Context = ();
    type Candidate = NativeCandidate;
    type Handle = CompiledKernel;
    fn form(
        &self,
        _context: &Self::Context,
        target: &DeviceDescription<Cpu>,
        kernel: &seismic_ir::kernel::Kernel<Cpu>,
        layout: &seismic_ir::target::KernelEmissionLayout,
    ) -> Result<Self::Candidate, NativeCompilationError> {
        compile::compile_kernel(target, kernel, layout)
    }
    fn reflect(
        &self,
        target: &DeviceDescription<Cpu>,
        kernel: &seismic_ir::kernel::Kernel<Cpu>,
        _layout: &seismic_ir::target::KernelEmissionLayout,
        candidate: Self::Candidate,
    ) -> Result<NativeKernelReflection<Cpu, Self::Handle>, NativeCompilationError> {
        let reflection_started = Instant::now();
        let numerical_identity = NativeNumericalModeIdentity {
            fingerprint: Sha256::digest(b"seismic-cpu-strict-ieee-v1").into(),
        };
        let description = NativeKernelDescription {
            identity: NativeKernelIdentity {
                compatibility: target.compatibility_identity().clone(),
                artifact_digest: candidate.artifact_digest,
            },
            abi: target.kernel_abi_layout(kernel),
            launch: NativeLaunchDomain {
                modes: vec![CpuLaunchMode],
                subgroup_width: None,
                cluster: seismic_target::NativeClusterDomain::NotApplicable,
                max_grid: target.limits().max_grid,
                max_workgroup_size: target.limits().max_workgroup_size,
                max_workgroup_threads: target.limits().max_workgroup_threads,
                max_dynamic_local_bytes: target.limits().max_workgroup_bytes,
            },
            resources: NativeResources {
                // CPU scheduling is not occupancy-limited by a device
                // register file. The reflected native frame is the exact
                // per-participant spill/private-stack footprint.
                registers_per_participant: NativeResourceUsage::NotApplicable,
                spill_bytes_per_participant: NativeResourceUsage::Exact(candidate.frame_size),
                static_local_bytes: NativeResourceUsage::NotApplicable,
            },
            numerics: CpuNumericalMode,
            numerical_identity,
            properties: (),
        };
        let metrics = NativeArtifactMetrics {
            compilation_ns: candidate
                .compilation_ns
                .checked_add(
                    u64::try_from(reflection_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                )
                .unwrap_or(u64::MAX),
            code_bytes: candidate.code_bytes,
            metadata_bytes: candidate.metadata_bytes,
        };
        Ok(NativeKernelReflection::new(
            candidate.kernel,
            description,
            metrics,
        ))
    }
}
