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
            let bytes = match core.source().stored() {
                Stored::Dense(tensor) => tensor.read().map_err(|error| {
                    SubmitError::Device(DeviceError::Transfer(error.to_string()))
                })?,
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
                    source.read(*offset, length).map_err(|error| {
                        SubmitError::Device(DeviceError::Transfer(error.to_string()))
                    })?
                }
            };
            let upload = workspace.upload_mut();
            upload
                .write_from_host(&bytes)
                .map_err(|error| SubmitError::Device(DeviceError::Transfer(error.to_string())))?;
            // The import entries operate over the logical flat element sequence.
            // Keep the model-shaped destination as the resident owner while the
            // checked flat view supplies the entry's rank-one output argument.
            // Reshape requires identical canonical byte coverage, so a packed
            // destination with row padding cannot be silently imported here.
            let mut flat_destination = flat_import_destination(destination.tensor())?;
            match &self.handle {
                AttestedImport::Dense(handle) => handle
                    .call_into(
                        import_dense::Args { source: upload },
                        import_dense::OutputArgs {
                            value: &mut flat_destination,
                        },
                    )
                    .map(|_| ()),
                AttestedImport::Repack(handle) => handle
                    .call_into(
                        repack_weight::Args { source: upload },
                        repack_weight::OutputArgs {
                            value: &mut flat_destination,
                        },
                    )
                    .map(|_| ()),
            }
            .map_err(|error| SubmitError::Device(DeviceError::Execution(error.to_string())))?;
            Ok(())
        })();
        if let Err(error) = result {
            return Err((error, launch));
        }
        let (core, workspace, destination) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, workspace, destination))
    }
}

fn flat_import_destination(destination: &Tensor) -> Result<Tensor, SubmitError> {
    let elements = destination
        .extents()
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| {
            SubmitError::Invariant(crate::InvariantError {
                context: "native import program",
                detail: "resident logical element count overflowed".into(),
            })
        })?;
    destination.reshape(&[elements]).map_err(|error| {
        SubmitError::Invariant(crate::InvariantError {
            context: "native import program",
            detail: format!("resident layout cannot supply a flat import view: {error}"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog, Element};

    #[test]
    fn packed_import_writes_a_shaped_resident_tensor() {
        let catalog = DeviceCatalog::discover().unwrap();
        let Ok(device) = catalog.open_backend(BackendName::Metal) else {
            return;
        };
        let source_element = Element::named("gguf_q5_k").unwrap();
        let resident_element = Element::named("q5k").unwrap();
        let source_bytes = vec![0_u8; 2 * 176];
        let source = Tensor::from_host(&device, source_element, &[512], &source_bytes).unwrap();
        let resident = Tensor::zeros(&device, resident_element, &[2, 256]).unwrap();
        let mut flat_resident = flat_import_destination(&resident).unwrap();
        let kernel = repack_weight::native_for_device_with(
            &device,
            repack_weight::Elements {
                E: source_element,
                U: resident_element,
            },
        )
        .unwrap();
        let expected = kernel
            .call(repack_weight::Args { source: &source })
            .unwrap()
            .value;
        kernel
            .call_into(
                repack_weight::Args { source: &source },
                repack_weight::OutputArgs {
                    value: &mut flat_resident,
                },
            )
            .unwrap();
        assert_eq!(resident.extents(), [2, 256]);
        assert_eq!(
            resident.read_to_host().unwrap(),
            expected.read_to_host().unwrap()
        );
    }
}
