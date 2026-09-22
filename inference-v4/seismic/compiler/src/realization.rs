//! Opaque native realizations retained across evaluation and planning.
//!
//! Candidate domains contain immutable reflected descriptions only. This
//! registry is owned by preparation orchestration and is never passed to an
//! evaluator or planner. The artifact store owns one shared handle for each
//! native artifact identity; a separate implementation manifest records the
//! exact ordered artifact sequence for each implementation.

use crate::errors::PreparationError;
use crate::implementation::ImplementationIdentity;
use crate::refinement::CandidateFamily;
use seismic_target::{DeviceDescriptionIdentity, NativeCompiler, TargetFamily};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) fn realize_candidate<T, C>(
    family: CandidateFamily<T>,
    compiler: &C,
    native_context: &C::Context,
    arena: &mut seismic_lang::expr::ExprArena,
    target: &seismic_target::DeviceDescription<T>,
    registry: &crate::target::CompilerRegistry<T>,
    constants: &crate::target::TargetConstants,
) -> Result<
    (
        crate::implementation::Implementation<T>,
        Vec<seismic_target::NativeKernel<T, C::Handle>>,
        seismic_target::NativeArtifactMetrics,
    ),
    PreparationError,
>
where
    T: TargetFamily,
    C: NativeCompiler<T>,
{
    let (kernels, metrics) = crate::implementation::native::realize(
        compiler,
        native_context,
        family.executable.kernels(),
        target,
    )?;
    let descriptions = kernels
        .iter()
        .map(|kernel| kernel.description().clone())
        .collect();
    let implementation = crate::implementation::reconcile_candidate_with_descriptions(
        family,
        arena,
        target,
        registry,
        constants,
        descriptions,
    )?;
    Ok((implementation, kernels, metrics))
}

/// The sole owner of candidate-native handles before materialization.
///
/// `artifacts` is content-addressed native storage. `manifests` is the exact
/// implementation-to-artifact relation. Insertion validates a complete batch
/// before either map changes, so failed insertion cannot leave partial state.
#[derive(Debug)]
pub(crate) struct RealizationRegistry<T: TargetFamily, H> {
    device: DeviceDescriptionIdentity,
    artifacts: NativeArtifactStore<T, H>,
    manifests: ImplementationManifest,
}

/// Content-addressed native handle ownership. This type alone decides whether
/// a repeated identity denotes the same artifact and performs handle dedupe.
#[derive(Debug)]
struct NativeArtifactStore<T: TargetFamily, H> {
    by_identity:
        HashMap<seismic_target::NativeKernelIdentity, Arc<seismic_target::NativeKernel<T, H>>>,
}

/// Owns implementation membership and the exact semantic-kernel-ordinal to
/// artifact-identity relation. Slice order is semantic kernel ordinal.
#[derive(Debug)]
struct ImplementationManifest {
    by_implementation: HashMap<ImplementationIdentity, Box<[seismic_target::NativeKernelIdentity]>>,
}

impl ImplementationManifest {
    fn new() -> Self {
        Self {
            by_implementation: HashMap::new(),
        }
    }

    fn validate_absent(
        &self,
        implementation: &ImplementationIdentity,
    ) -> Result<(), PreparationError> {
        if self.by_implementation.contains_key(implementation) {
            return Err(PreparationError::InvalidCandidateDomain(format!(
                "duplicate implementation identity from factory `{}`",
                implementation.factory.name
            )));
        }
        Ok(())
    }

    fn artifacts_for<T: TargetFamily, H>(
        kernels: &[seismic_target::NativeKernel<T, H>],
    ) -> Box<[seismic_target::NativeKernelIdentity]> {
        kernels
            .iter()
            .map(|kernel| kernel.description().identity.clone())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Publishes a manifest only after every referenced artifact is retained.
    fn commit(
        &mut self,
        implementation: ImplementationIdentity,
        artifacts: Box<[seismic_target::NativeKernelIdentity]>,
    ) {
        let replaced = self.by_implementation.insert(implementation, artifacts);
        assert!(
            replaced.is_none(),
            "realization manifest changed after successful batch preflight"
        );
    }

    fn resolve_exact<'a>(
        &'a self,
        implementation: &ImplementationIdentity,
        expected: &[seismic_target::NativeKernelIdentity],
    ) -> &'a [seismic_target::NativeKernelIdentity] {
        let manifest = self
            .by_implementation
            .get(implementation)
            .unwrap_or_else(|| {
                panic!("planned implementation is absent from its realization registry")
            });
        assert_eq!(
            manifest.as_ref(),
            expected,
            "planned implementation artifact manifest differs from its realized manifest"
        );
        manifest
    }
}

impl<T: TargetFamily, H> NativeArtifactStore<T, H> {
    fn new() -> Self {
        Self {
            by_identity: HashMap::new(),
        }
    }

    fn validate_batch(
        &self,
        device: &DeviceDescriptionIdentity,
        factory: &str,
        kernels: &[seismic_target::NativeKernel<T, H>],
    ) -> Result<(), PreparationError> {
        let mut staged = HashMap::<
            seismic_target::NativeKernelIdentity,
            &seismic_target::NativeKernelDescription<T>,
        >::new();
        for kernel in kernels {
            let description = kernel.description();
            if description.identity.compatibility.backend != device.backend
                || description.identity.compatibility.fingerprint != device.fingerprint
            {
                return Err(PreparationError::InvalidCandidateDomain(format!(
                    "implementation from factory `{factory}` contains a native artifact for another device"
                )));
            }
            if let Some(existing) = self.by_identity.get(&description.identity) {
                if existing.description() != description {
                    return Err(PreparationError::InvalidCandidateDomain(format!(
                        "native artifact identity collision from factory `{factory}` has conflicting reflected descriptions"
                    )));
                }
            }
            if let Some(earlier) = staged.insert(description.identity.clone(), description) {
                if earlier != description {
                    return Err(PreparationError::InvalidCandidateDomain(format!(
                        "native artifact identity collision within factory `{factory}` has conflicting reflected descriptions"
                    )));
                }
            }
        }
        Ok(())
    }

    fn commit(&mut self, kernels: Vec<seismic_target::NativeKernel<T, H>>) {
        for kernel in kernels {
            let identity = kernel.description().identity.clone();
            self.by_identity
                .entry(identity)
                .or_insert_with(|| Arc::new(kernel));
        }
    }

    fn get(
        &self,
        identity: &seismic_target::NativeKernelIdentity,
    ) -> Option<Arc<seismic_target::NativeKernel<T, H>>> {
        self.by_identity.get(identity).cloned()
    }
}

impl<T: TargetFamily, H> RealizationRegistry<T, H> {
    pub(crate) fn new(device: DeviceDescriptionIdentity) -> Self {
        Self {
            device,
            artifacts: NativeArtifactStore::new(),
            manifests: ImplementationManifest::new(),
        }
    }

    pub(crate) fn insert(
        &mut self,
        implementation: ImplementationIdentity,
        kernels: Vec<seismic_target::NativeKernel<T, H>>,
    ) -> Result<(), PreparationError> {
        // Both owners preflight before either owner changes.
        self.manifests.validate_absent(&implementation)?;
        self.artifacts
            .validate_batch(&self.device, implementation.factory.name, &kernels)?;
        let manifest = ImplementationManifest::artifacts_for(&kernels);
        self.artifacts.commit(kernels);
        self.manifests.commit(implementation, manifest);
        Ok(())
    }

    pub(crate) fn resolve(
        &self,
        device: &DeviceDescriptionIdentity,
        implementation: &ImplementationIdentity,
        expected: &[seismic_target::NativeKernelIdentity],
    ) -> Vec<Arc<seismic_target::NativeKernel<T, H>>> {
        assert_eq!(
            device, &self.device,
            "planned policy and private realization registry have different device identities"
        );
        let manifest = self.manifests.resolve_exact(implementation, expected);
        manifest
            .iter()
            .map(|artifact| {
                self.artifacts.get(artifact).unwrap_or_else(|| {
                    panic!("implementation manifest references an absent native artifact")
                })
            })
            .collect()
    }
}
