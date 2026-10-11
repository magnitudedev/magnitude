//! Real resident weights for tuning rotations. Each weight is imported the
//! way loading imports it, from the same source ([`ImportSource`]) through
//! the same prepared import entry, into a tensor owned by tuning; it is
//! released when its entry is tuned.

use crate::{
    ArtifactComponent, ArtifactComponentKind, ModelLoadPlan, Stored, WeightPlan,
    native::{AttestedImport, import::ImportKernels},
    resources::{ImportSource, ImportWindow},
};
use magnitude_artifacts::{Package, gguf::GgufArtifact};
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use seismic::{Device, Tensor};
use std::collections::HashMap;

/// Where tuning reads weights from.
pub trait TuningWeightSource {
    /// The bytes of `weight` in its upload representation, on the host:
    /// for a weight the host itself reads or places.
    fn source_bytes(&self, weight: &WeightPlan) -> Result<Vec<u8>, String>;

    /// The tensor an import entry reads `weight` from on `device`: its
    /// source in its upload representation, one row of every element.
    fn import_source(&self, device: &Device, weight: &WeightPlan) -> Result<Tensor, String>;
}

impl TuningWeightSource for Package {
    /// The transformed (and dequantized) source bytes, as loading uploads
    /// them.
    fn source_bytes(&self, weight: &WeightPlan) -> Result<Vec<u8>, String> {
        let artifact = artifact(self, weight.component)?;
        let stored =
            Stored::from_gguf(artifact, &weight.descriptor).map_err(|error| error.to_string())?;
        let bytes = crate::programs::native_import::stored_source_bytes(&stored)
            .map_err(|error| error.to_string())?;
        crate::import_transforms::upload_bytes(
            &weight.descriptor,
            stored.shape(),
            weight.source,
            stored.packed_encoding(),
            weight.upload,
            bytes,
        )
    }

    /// The source as the load's own importer takes it
    /// ([`ImportSource::acquire`]): a view of the artifact's mapped pages
    /// where the device reads them in place, and otherwise an upload
    /// written from the stored range a block at a time. Tuning's import of
    /// a weight therefore holds on the host what the load's import of it
    /// holds, and no copy of the source besides.
    fn import_source(&self, device: &Device, weight: &WeightPlan) -> Result<Tensor, String> {
        let artifact = artifact(self, weight.component)?;
        let stored =
            Stored::from_gguf(artifact, &weight.descriptor).map_err(|error| error.to_string())?;
        let window = (ImportWindow::serves(device) && !weight.host_prepared())
            .then(|| ImportWindow::of(device, &stored))
            .transpose()
            .map_err(|error| error.to_string())?;
        let mut source = ImportSource::acquire(weight, &stored, window.as_ref(), device)
            .map_err(|error| error.to_string())?;
        if let Some(upload) = source.staged_mut() {
            crate::programs::native_import::fill_staged_source(&stored, weight, upload)
                .map_err(|error| error.to_string())?;
        }
        Ok(source.into_tensor())
    }
}

fn artifact(package: &Package, component: ArtifactComponent) -> Result<&GgufArtifact, String> {
    let artifact = match component.kind {
        ArtifactComponentKind::Target => package.target(),
        ArtifactComponentKind::Projector => package
            .projector()
            .ok_or("the package has no projector artifact")?,
        ArtifactComponentKind::Draft => {
            package.draft().ok_or("the package has no draft artifact")?
        }
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
            .upload
            .canonical_byte_len(&[logical_count(&weight.shape)?])
            .map_err(|error| error.to_string())?;
        Ok(vec![
            0;
            usize::try_from(bytes)
                .map_err(|_| "weight bytes exceed usize")?
        ])
    }

    fn import_source(&self, device: &Device, weight: &WeightPlan) -> Result<Tensor, String> {
        Tensor::zeros(device, weight.upload, &[logical_count(&weight.shape)?])
            .map_err(|error| error.to_string())
    }
}

fn logical_count(shape: &[u64]) -> Result<u64, String> {
    shape
        .iter()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| "weight element count overflows".to_owned())
}

/// The load plan's weight of one role.
pub(crate) fn weight_plan(
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
) -> Result<&WeightPlan, String> {
    let role = WeightRole { scope, kind };
    load.qualification_weights()
        .find(|weight| weight.role == role)
        .ok_or_else(|| format!("weight {role:?} is absent from the load plan"))
}

/// Whether a tensor a tuning cache keeps is bound by anything else: dropping
/// the cache's handle alone would give no storage back.
pub(crate) fn held_elsewhere(tensor: &Tensor) -> bool {
    matches!(Tensor::reclaimable_bytes([tensor]), Ok(0))
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

    /// Drop every imported weight that no built input binds. A weight some
    /// point's inputs still hold stays findable here, so the next point or
    /// unit that reads the same role shares its storage: one layer's weight
    /// is on the device once, however many points and units read it, and
    /// leaves when the last of them gives its inputs up.
    pub fn release_unheld(&mut self) {
        self.resident.retain(|_, tensor| held_elsewhere(tensor));
    }

    pub fn load(&self) -> &'a ModelLoadPlan {
        self.load
    }

    fn plan(&self, scope: WeightScope, kind: WeightKind) -> Result<&'a WeightPlan, String> {
        weight_plan(self.load, scope, kind)
    }

    pub fn weight(&mut self, scope: WeightScope, kind: WeightKind) -> Result<Tensor, String> {
        let role = WeightRole { scope, kind };
        if let Some(tensor) = self.resident.get(&role) {
            return Ok(tensor.clone());
        }
        let plan = self.plan(scope, kind)?;
        // A plane the host places is uploaded as placed, as loading
        // uploads it.
        if plan.placed_on_host() {
            let bytes = self.source.source_bytes(plan)?;
            let tensor = Tensor::from_host(self.device, plan.resident, &plan.shape, &bytes)
                .map_err(|error| error.to_string())?;
            self.resident.insert(role, tensor.clone());
            return Ok(tensor);
        }
        let source = self.source.import_source(self.device, plan)?;
        let destination = Tensor::zeros(self.device, plan.resident, &plan.shape)
            .map_err(|error| error.to_string())?;
        let handle = match (plan.upload.dtype(), plan.resident.dtype()) {
            (Some(source), Some(resident)) => self
                .import
                .import_dense
                .get(&(source, resident))
                .cloned()
                .map(AttestedImport::Dense),
            _ => self
                .import
                .repack_weight
                .get(&(plan.upload, plan.resident))
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
