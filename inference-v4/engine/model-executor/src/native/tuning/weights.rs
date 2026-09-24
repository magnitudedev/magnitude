//! Real resident weights for tuning rotations. Each weight is imported from
//! its source bytes through the same prepared import entry that loading uses,
//! into a tensor owned by tuning; it is released when its entry is tuned.

use crate::{
    native::{import::ImportKernels, AttestedImport},
    ArtifactComponent, ArtifactComponentKind, ModelLoadPlan, Stored,
    WeightPlan,
};
use magnitude_artifacts::{gguf::GgufArtifact, Package};
use magnitude_model_contracts::{WeightKind, WeightRole, WeightScope};
use seismic::{Device, Tensor};
use std::collections::HashMap;

/// Where tuning reads weights from.
pub trait TuningWeightSource {
    /// The bytes of `weight` in its source representation.
    fn source_bytes(&self, weight: &WeightPlan) -> Result<Vec<u8>, String>;
}

impl TuningWeightSource for Package {
    fn source_bytes(&self, weight: &WeightPlan) -> Result<Vec<u8>, String> {
        let artifact = artifact(self, weight.component)?;
        let stored =
            Stored::from_gguf(artifact, &weight.descriptor).map_err(|error| error.to_string())?;
        crate::programs::native_import::stored_source_bytes(&stored)
            .map_err(|error| error.to_string())
    }
}

fn artifact(package: &Package, component: ArtifactComponent) -> Result<&GgufArtifact, String> {
    let artifact = match component.kind {
        ArtifactComponentKind::Target => package.target(),
        ArtifactComponentKind::Projector => package
            .projector()
            .ok_or("the package has no projector artifact")?,
    };
    if artifact.identity() != component.identity {
        return Err("the package artifact differs from the planned component".into());
    }
    Ok(artifact)
}

/// Zero weights of every planned shape, for programs prepared from a
/// synthetic model definition without an artifact (test fixtures). Every
/// source representation decodes zero bytes to zeros, so each entry tunes
/// on well-defined values at the model's real geometry.
pub struct ZeroTuningWeights;

impl TuningWeightSource for ZeroTuningWeights {
    fn source_bytes(&self, weight: &WeightPlan) -> Result<Vec<u8>, String> {
        let bytes = weight
            .source
            .canonical_byte_len(&[logical_count(&weight.shape)?])
            .map_err(|error| error.to_string())?;
        Ok(vec![0; usize::try_from(bytes).map_err(|_| "weight bytes exceed usize")?])
    }
}

fn logical_count(shape: &[u64]) -> Result<u64, String> {
    shape
        .iter()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| "weight element count overflows".to_owned())
}

pub(crate) struct TuningWeights<'a> {
    device: &'a Device,
    load: &'a ModelLoadPlan,
    source: &'a dyn TuningWeightSource,
    import: &'a ImportKernels,
    resident: HashMap<WeightRole, Tensor>,
}

impl<'a> TuningWeights<'a> {
    pub fn new(
        device: &'a Device,
        load: &'a ModelLoadPlan,
        source: &'a dyn TuningWeightSource,
        import: &'a ImportKernels,
    ) -> Self {
        Self {
            device,
            load,
            source,
            import,
            resident: HashMap::new(),
        }
    }

    /// Drop every imported weight; the next entry imports what it needs.
    pub fn release(&mut self) {
        self.resident.clear();
    }

    fn plan(&self, scope: WeightScope, kind: WeightKind) -> Result<&'a WeightPlan, String> {
        let role = WeightRole { scope, kind };
        self.load
            .weights()
            .find(|weight| weight.role == role)
            .ok_or_else(|| format!("weight {role:?} is absent from the load plan"))
    }

    pub fn shape(&self, scope: WeightScope, kind: WeightKind) -> Result<Vec<u64>, String> {
        Ok(self.plan(scope, kind)?.shape.clone())
    }

    pub fn weight(&mut self, scope: WeightScope, kind: WeightKind) -> Result<Tensor, String> {
        let role = WeightRole { scope, kind };
        if let Some(tensor) = self.resident.get(&role) {
            return Ok(tensor.clone());
        }
        let plan = self.plan(scope, kind)?;
        let bytes = self.source.source_bytes(plan)?;
        let logical = logical_count(&plan.shape)?;
        let source = Tensor::from_host(self.device, plan.source, &[logical], &bytes)
            .map_err(|error| error.to_string())?;
        let destination = Tensor::zeros(self.device, plan.resident, &plan.shape)
            .map_err(|error| error.to_string())?;
        let handle = match (plan.source.dtype(), plan.resident.dtype()) {
            (Some(source), Some(resident)) => self
                .import
                .import_dense
                .get(&(source, resident))
                .cloned()
                .map(AttestedImport::Dense),
            _ => self
                .import
                .repack_weight
                .get(&(plan.source, plan.resident))
                .cloned()
                .map(AttestedImport::Repack),
        }
        .ok_or_else(|| {
            format!(
                "no import entry was prepared for {} -> {}",
                plan.source.name(),
                plan.resident.name()
            )
        })?;
        crate::programs::native_import::import_into(&handle, &source, &destination)
            .map_err(|error| error.to_string())?;
        self.resident.insert(role, destination.clone());
        Ok(destination)
    }
}
