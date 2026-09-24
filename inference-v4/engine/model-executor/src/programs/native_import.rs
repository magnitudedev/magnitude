//! One-shot native weight import. The source upload and resident destination
//! are already allocated from the admitted plan before submit.

use super::{ImportProgram, ReadySubmission};
use crate::{
    DeviceError, ImportLaunchCore, ImportWorkspaceLease, ResidentWeightSlot, Stored, SubmitError,
    ValidatedImportLaunch, WeightStorageIdentity, native::AttestedImport,
};
use magnitude_model_kernels::{import_dense, repack_weight};
use seismic::Tensor;

pub struct NativeImportProgram {
    identity: WeightStorageIdentity,
    handle: AttestedImport,
}

impl NativeImportProgram {
    pub(crate) fn new(identity: WeightStorageIdentity, handle: AttestedImport) -> Self {
        Self { identity, handle }
    }
}

impl ImportProgram for NativeImportProgram {
    type Submission = ReadySubmission<ImportLaunchCore, ImportWorkspaceLease, ResidentWeightSlot>;

    fn submit(
        &mut self,
        mut launch: ValidatedImportLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedImportLaunch)> {
        let result = (|| -> Result<(), SubmitError> {
            let (core, workspace, destination) = launch.submission_parts_mut();
            if core.plan().storage_identity() != self.identity {
                return Err(SubmitError::Invariant(crate::InvariantError {
                    context: "native import program",
                    detail: "launch destination differs from the attested weight slot".into(),
                }));
            }
            let bytes = stored_source_bytes(core.source().stored())?;
            let upload = workspace.upload_mut();
            upload
                .write_from_host(&bytes)
                .map_err(|error| SubmitError::Device(DeviceError::Transfer(error.to_string())))?;
            import_into(&self.handle, upload, destination.tensor())?;
            Ok(())
        })();
        if let Err(error) = result {
            return Err((error, launch));
        }
        let (core, workspace, destination) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, workspace, destination))
    }
}

/// The artifact bytes of one stored source tensor.
pub(crate) fn stored_source_bytes(stored: &Stored) -> Result<Vec<u8>, SubmitError> {
    match stored {
        Stored::Dense(tensor) => tensor
            .read()
            .map_err(|error| SubmitError::Device(DeviceError::Transfer(error.to_string()))),
        Stored::GgmlBlocks {
            source,
            offset,
            nbytes,
            ..
        } => {
            let length = usize::try_from(*nbytes).map_err(|_| {
                SubmitError::Device(DeviceError::Transfer(
                    "source weight exceeds host address range".into(),
                ))
            })?;
            source
                .read(*offset, length)
                .map_err(|error| SubmitError::Device(DeviceError::Transfer(error.to_string())))
        }
    }
}

/// Run one import entry from an uploaded source tensor into its resident
/// destination. The entries see a weight as its `[B, N, K]` view: `K` the
/// packing (last) axis, `N` the row axis, `B` every leading matrix. That view
/// keeps the row geometry of every resident layout (a `rows16` row, an
/// `mma16` row tile) and the source's packets.
pub(crate) fn import_into(
    handle: &AttestedImport,
    source: &Tensor,
    destination: &Tensor,
) -> Result<(), SubmitError> {
    let view = import_view(destination.extents())?;
    let source = shaped(source, &view, "source")?;
    let mut destination = shaped(destination, &view, "resident")?;
    match handle {
        AttestedImport::Dense(handle) => handle
            .call_into(
                import_dense::Args { source: &source },
                import_dense::OutputArgs {
                    value: &mut destination,
                },
            )
            .map(|_| ()),
        AttestedImport::Repack(handle) => handle
            .call_into(
                repack_weight::Args { source: &source },
                repack_weight::OutputArgs {
                    value: &mut destination,
                },
            )
            .map(|_| ()),
    }
    .map_err(|error| SubmitError::Device(DeviceError::Execution(error.to_string())))
}

fn invariant(detail: String) -> SubmitError {
    SubmitError::Invariant(crate::InvariantError {
        context: "native import program",
        detail,
    })
}

/// The `[B, N, K]` import view of a weight of `extents`.
fn import_view(extents: &[u64]) -> Result<[u64; 3], SubmitError> {
    let (&k, leading) = extents
        .split_last()
        .ok_or_else(|| invariant("a resident weight has rank zero".into()))?;
    let (&n, matrices) = leading.split_last().unwrap_or((&1, &[]));
    let b = matrices
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| invariant("resident matrix count overflowed".into()))?;
    Ok([b, n, k])
}

fn shaped(tensor: &Tensor, view: &[u64; 3], role: &str) -> Result<Tensor, SubmitError> {
    tensor.reshape(view).map_err(|error| {
        invariant(format!("the {role} tensor has no [B, N, K] import view: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog, Element};

    #[test]
    fn import_views_keep_the_row_axis_and_packing_axis() {
        assert_eq!(import_view(&[2560]).unwrap(), [1, 1, 2560]);
        assert_eq!(import_view(&[9216, 2560]).unwrap(), [1, 9216, 2560]);
        assert_eq!(import_view(&[256, 512, 2048]).unwrap(), [256, 512, 2048]);
        assert_eq!(import_view(&[2, 3, 5, 256]).unwrap(), [6, 5, 256]);
        assert!(import_view(&[]).is_err());
    }

    /// A shaped packed import (flat source upload, model-shaped resident
    /// destination) in the Metal execution layout decodes exactly as the
    /// host reference of the registered conversion, including a row count
    /// off the 16-row tile.
    #[test]
    fn packed_import_writes_a_shaped_resident_tensor() {
        let catalog = DeviceCatalog::discover().unwrap();
        let Ok(device) = catalog.open_backend(BackendName::Metal) else {
            return;
        };
        let source_element = Element::named("gguf_q5_k").unwrap();
        for layout in [seismic::Layout::Rows16, seismic::Layout::Mma16] {
            let resident_element = Element::stored("q5k", layout).unwrap();
            let shape = [17_u64, 512];
            let source_bytes = (0..17 * 2 * 176)
                .map(|index: usize| (index as u8).wrapping_mul(29).wrapping_add(7))
                .collect::<Vec<_>>();
            let source = Tensor::from_host(&device, source_element, &[17 * 512], &source_bytes)
                .unwrap();
            let resident = Tensor::zeros(&device, resident_element, &shape).unwrap();
            let kernel = repack_weight::native_for_device_with(
                &device,
                repack_weight::Elements {
                    E: source_element,
                    U: resident_element,
                },
                &seismic::NativeSpecialization::new(),
            )
            .unwrap();
            import_into(&AttestedImport::Repack(kernel), &source, &resident).unwrap();
            assert_eq!(resident.extents(), shape);
            assert_eq!(
                resident.read_to_host().unwrap(),
                resident_element
                    .repack_host(source_element, &shape, &source_bytes)
                    .unwrap(),
                "{}",
                resident_element.name()
            );
        }
    }
}
