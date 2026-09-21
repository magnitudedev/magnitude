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
mod numeric;
mod profile;
mod registry;
mod services;
mod workers;

pub use buffer::Buffer;
pub use command::CompiledKernel;
pub use compile::NativeCandidate;
pub use executor::{Device, Executor};
pub use profile::{HostFacts, HostKernelAbi, ScratchPolicy, SimdTier, BACKEND_REVISION};
pub use registry::registry;
pub use workers::Workers;

use seismic_compiler::errors::NativeCompilationError;
use seismic_compiler::target::{
    Backend, DeviceContract, IntrinsicIdentityBuilder, NativeArtifactMetrics, NativeKernelIdentity,
    NativeKernelReflection, NativeKernelResources, NativeLaunchDomain, NativeNumericalModeIdentity,
    NativeResourceUsage,
};
use seismic_lang::registry::BackendName;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
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

impl Backend for Cpu {
    const NAME: BackendName = BackendName::Cpu;
    type Intrinsic = CpuIntrinsic;
    type Facts = HostFacts;
    type KernelAbi = HostKernelAbi;
    type NativeLaunchMode = CpuLaunchMode;
    type RawNativeCandidate = NativeCandidate;
    type NativeKernelHandle = CompiledKernel;
    type NativeNumericalMode = CpuNumericalMode;
    type Executor = Executor;

    fn independent_launch_mode() -> Self::NativeLaunchMode {
        CpuLaunchMode
    }

    fn cooperative_launch_mode(_facts: &Self::Facts) -> Option<Self::NativeLaunchMode> {
        None
    }

    fn write_intrinsic_identity(
        intrinsic: &Self::Intrinsic,
        _identity: &mut IntrinsicIdentityBuilder,
    ) {
        match *intrinsic {}
    }

    fn emitted_intrinsics() -> BTreeSet<seismic_lang::ids::IntrinsicId> {
        BTreeSet::new()
    }

    fn required_service_classes(
        _facts: &Self::Facts,
        _supported_intrinsics: &BTreeSet<seismic_lang::ids::IntrinsicId>,
    ) -> BTreeSet<seismic_compiler::target::ServiceClassId> {
        services::required_service_classes()
    }

    fn execution_demand(
        target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        emission: &seismic_compiler::target::KernelEmissionLayout,
        op: seismic_compiler::kernel::ops::ClosedOpView<'_, Self>,
    ) -> Vec<seismic_compiler::target::ExecutionDemand> {
        services::execution_demand(target, arena, kernel, emission, op)
    }

    fn intrinsic_numerics(
        _facts: &Self::Facts,
        _signature: &seismic_lang::registry::IntrinsicSignature,
        intrinsic: &Self::Intrinsic,
    ) -> seismic_compiler::target::IntrinsicNumericalSemantics {
        match *intrinsic {}
    }

    fn intrinsic_addressable_resources(
        intrinsic: &Self::Intrinsic,
    ) -> Vec<seismic_compiler::kernel::ops::AddressableResourceHandle> {
        match *intrinsic {}
    }

    fn semantic_intrinsic_requirements(
        _target: &DeviceContract<Self>,
        _arena: &mut seismic_lang::expr::ExprArena,
        signature: &seismic_lang::registry::IntrinsicSignature,
        _parallel_extent: seismic_lang::expr::NatExpr,
    ) -> seismic_compiler::kernel::ops::SemanticIntrinsicLaunchRequirements {
        panic!(
            "CPU static capability registry admitted unimplemented intrinsic {:?}",
            signature.id
        )
    }

    fn lower_semantic_intrinsic(
        _target: &DeviceContract<Self>,
        _domain: &seismic_compiler::kernel::ops::SegmentLaunchDomain,
        call: seismic_compiler::kernel::ops::SemanticIntrinsicCall<'_>,
        _sink: &mut seismic_compiler::kernel::ops::SemanticIntrinsicSink<'_, '_, Self>,
    ) {
        panic!(
            "CPU semantic lowering received unregistered intrinsic {:?}",
            call.signature.id
        )
    }

    fn form_native_kernel_candidate(
        target: &DeviceContract<Self>,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        layout: &seismic_compiler::target::KernelEmissionLayout,
    ) -> Result<Self::RawNativeCandidate, NativeCompilationError> {
        compile::compile_kernel(target, kernel, layout)
    }

    fn reflect_native_kernel(
        target: &DeviceContract<Self>,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        _layout: &seismic_compiler::target::KernelEmissionLayout,
        candidate: Self::RawNativeCandidate,
    ) -> Result<NativeKernelReflection<Self>, NativeCompilationError> {
        let reflection_started = Instant::now();
        let numerical_identity = NativeNumericalModeIdentity {
            fingerprint: Sha256::digest(b"seismic-cpu-strict-ieee-v1").into(),
        };
        Ok(NativeKernelReflection {
            handle: candidate.kernel,
            identity: NativeKernelIdentity {
                compatibility: target.compatibility_identity().clone(),
                artifact_digest: candidate.artifact_digest,
            },
            abi: target.kernel_abi_layout(kernel),
            launch: NativeLaunchDomain {
                modes: vec![CpuLaunchMode],
                subgroup_width: None,
                cluster: seismic_compiler::target::NativeClusterDomain::NotApplicable,
                max_grid: target.limits().max_grid,
                max_workgroup_size: target.limits().max_workgroup_size,
                max_workgroup_threads: target.limits().max_workgroup_threads,
                max_dynamic_local_bytes: target.limits().max_workgroup_bytes,
            },
            resources: NativeKernelResources {
                // CPU scheduling is not occupancy-limited by a device
                // register file. The reflected native frame is the exact
                // per-participant spill/private-stack footprint.
                registers_per_participant: NativeResourceUsage::NotApplicable,
                spill_bytes_per_participant: NativeResourceUsage::Exact(candidate.frame_size),
                static_local_bytes: NativeResourceUsage::NotApplicable,
            },
            numerics: CpuNumericalMode,
            numerical_identity,
            artifact: NativeArtifactMetrics {
                compilation_ns: candidate
                    .compilation_ns
                    .checked_add(
                        u64::try_from(reflection_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    )
                    .unwrap_or(u64::MAX),
                code_bytes: candidate.code_bytes,
                metadata_bytes: candidate.metadata_bytes,
            },
        })
    }
}
