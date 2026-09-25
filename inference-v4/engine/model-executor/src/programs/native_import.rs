//! One-shot native weight import. The mapped or staged source and resident
//! destination are already allocated from the admitted plan before submit.

use super::{ImportProgram, ReadySubmission};
use crate::{
    native::AttestedImport, DeviceError, ImportLaunchCore, ImportWorkspaceLease,
    ResidentWeightSlot, Stored, SubmitError, ValidatedImportLaunch, WeightStorageIdentity,
};
use magnitude_model_kernels::{import_dense, repack_weight};
use seismic::{NativeTensorBatch, Tensor};
use std::io::{Seek, SeekFrom};

pub struct NativeImportProgram {
    identity: WeightStorageIdentity,
    handle: AttestedImport,
}

impl NativeImportProgram {
    pub(crate) fn new(identity: WeightStorageIdentity, handle: AttestedImport) -> Self {
        Self { identity, handle }
    }

    /// Add a validated import to a device batch. The returned launch keeps
    /// its mapped or staged source and resident destination owned until the
    /// batch has completed.
    pub(crate) fn enqueue(
        &self,
        batch: &mut NativeTensorBatch,
        mut launch: ValidatedImportLaunch,
    ) -> Result<ValidatedImportLaunch, (SubmitError, ValidatedImportLaunch)> {
        let result = (|| -> Result<(), SubmitError> {
            let (core, workspace, destination) = launch.submission_parts_mut();
            if core.plan().storage_identity() != self.identity {
                return Err(SubmitError::Invariant(crate::InvariantError {
                    context: "native import program",
                    detail: "launch destination differs from the attested weight slot".into(),
                }));
            }
            if let Some(upload) = workspace.staged_mut() {
                fill_staged_source(core.source().stored(), upload)?;
            }
            enqueue_into(
                &self.handle,
                batch,
                workspace.source(),
                destination.tensor(),
            )
        })();
        if let Err(error) = result {
            return Err((error, launch));
        }
        Ok(launch)
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
            if let Some(upload) = workspace.staged_mut() {
                fill_staged_source(core.source().stored(), upload)?;
            }
            import_into(&self.handle, workspace.source(), destination.tensor())?;
            Ok(())
        })();
        if let Err(error) = result {
            return Err((error, launch));
        }
        let (core, workspace, destination) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, workspace, destination))
    }
}

fn fill_staged_source(stored: &Stored, upload: &mut Tensor) -> Result<(), SubmitError> {
    let (source, offset, bytes) = stored.file_range();
    if upload.byte_len() != bytes {
        return Err(SubmitError::Device(DeviceError::Transfer(
            "staged source byte length differs from its tensor".into(),
        )));
    }
    let mut reader = source.reader();
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|error| SubmitError::Device(DeviceError::Transfer(error.to_string())))?;
    upload
        .write_from_reader(&mut reader)
        .map_err(|error| SubmitError::Device(DeviceError::Transfer(error.to_string())))
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

fn enqueue_into(
    handle: &AttestedImport,
    batch: &mut NativeTensorBatch,
    source: &Tensor,
    destination: &Tensor,
) -> Result<(), SubmitError> {
    let view = import_view(destination.extents())?;
    let source = shaped(source, &view, "source")?;
    let mut destination = shaped(destination, &view, "resident")?;
    let queued = match handle {
        AttestedImport::Dense(handle) => batch.push(
            handle,
            import_dense::Args { source: &source },
            import_dense::OutputArgs {
                value: &mut destination,
            },
        ),
        AttestedImport::Repack(handle) => batch.push(
            handle,
            repack_weight::Args { source: &source },
            repack_weight::OutputArgs {
                value: &mut destination,
            },
        ),
    };
    queued.map_err(|error| SubmitError::Device(DeviceError::Execution(error.to_string())))
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
        invariant(format!(
            "the {role} tensor has no [B, N, K] import view: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog, Element};

    #[test]
    fn staged_source_streams_the_exact_file_range() {
        use magnitude_artifacts::FileSource;
        use std::{fs, sync::Arc};

        let catalog = DeviceCatalog::discover().unwrap();
        let device = catalog.open_backend(BackendName::Cpu).unwrap();
        let values = (0..(1024 * 1024 + 16))
            .map(|index| (index as u8).wrapping_mul(17))
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "magnitude-staged-import-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file_bytes = vec![0x5a; 37];
        file_bytes.extend_from_slice(&values);
        fs::write(&path, file_bytes).unwrap();
        let file = Arc::new(FileSource::open(&path).unwrap());
        fs::remove_file(&path).unwrap();
        let stored = Stored::Dense(crate::StoredTensor {
            source: file,
            offset: 37,
            nbytes: values.len() as u64,
            dtype: seismic::DType::F32,
            shape: vec![values.len() as u64 / 4],
        });
        // SAFETY: fill_staged_source writes every physical upload byte.
        let mut upload =
            unsafe { Tensor::uninitialized(&device, Element::f32(), &[values.len() as u64 / 4]) }
                .unwrap();
        fill_staged_source(&stored, &mut upload).unwrap();
        assert_eq!(upload.read_to_host().unwrap(), values);
    }

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
            let source =
                Tensor::from_host(&device, source_element, &[17 * 512], &source_bytes).unwrap();
            // SAFETY: this import is a pure result that must write every
            // physical byte, including packed padding.
            let resident =
                unsafe { Tensor::uninitialized(&device, resident_element, &shape) }.unwrap();
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

    #[cfg(target_os = "macos")]
    #[test]
    fn packed_import_reads_a_mapped_artifact_window() {
        use magnitude_artifacts::FileSource;
        use seismic::{HostRegion, ReadOnlyMappedRegion};
        use std::{fs, sync::Arc};

        let catalog = DeviceCatalog::discover().unwrap();
        let Ok(device) = catalog.open_backend(BackendName::Metal) else {
            return;
        };
        let source_element = Element::named("gguf_q5_k").unwrap();
        let resident_element = Element::stored("q5k", seismic::Layout::Rows16).unwrap();
        let shape = [17_u64, 512];
        let bytes = (0..17 * 2 * 176)
            .map(|index: usize| (index as u8).wrapping_mul(29).wrapping_add(7))
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "magnitude-mapped-import-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut artifact = vec![0; 32];
        artifact.extend_from_slice(&bytes);
        fs::write(&path, artifact).unwrap();
        let file = Arc::new(FileSource::open(&path).unwrap());
        fs::remove_file(&path).unwrap();
        let window = file.map_window(32, bytes.len() as u64).unwrap();
        let pointer = std::ptr::NonNull::new(window.as_ref().as_ref().as_ptr() as *mut u8).unwrap();
        let host = unsafe { HostRegion::new(pointer, window.mapped_len(), window.clone()) };
        let region = ReadOnlyMappedRegion::new(&device, host).unwrap();
        let mut source = region
            .tensor(source_element, &[17 * 512], window.data_offset() as u64)
            .unwrap();
        assert!(source.write_from_host(&bytes).is_err());
        drop(region);
        drop(window);
        drop(file);
        // SAFETY: the packed import writes every resident byte before readback.
        let resident = unsafe { Tensor::uninitialized(&device, resident_element, &shape) }.unwrap();
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
        assert_eq!(
            resident.read_to_host().unwrap(),
            resident_element
                .repack_host(source_element, &shape, &bytes)
                .unwrap()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mapped_packed_and_dense_imports_share_one_native_batch() {
        use magnitude_artifacts::FileSource;
        use seismic::{HostRegion, ReadOnlyMappedRegion};
        use std::{fs, sync::Arc};

        let catalog = DeviceCatalog::discover().unwrap();
        let Ok(device) = catalog.open_backend(BackendName::Metal) else {
            return;
        };
        let packed_element = Element::named("gguf_q5_k").unwrap();
        let resident_element = Element::stored("q5k", seismic::Layout::Rows16).unwrap();
        let dense_element = Element::f32();
        let packed_bytes = (0..17 * 2 * 176)
            .map(|index: usize| (index as u8).wrapping_mul(29).wrapping_add(7))
            .collect::<Vec<_>>();
        let dense_bytes = [1.0f32, -2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "magnitude-batched-import-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let mut artifact = vec![0; 32];
        artifact.extend_from_slice(&packed_bytes);
        artifact.extend_from_slice(&dense_bytes);
        fs::write(&path, artifact).unwrap();
        let file = Arc::new(FileSource::open(&path).unwrap());
        fs::remove_file(&path).unwrap();
        let window = file
            .map_window(32, (packed_bytes.len() + dense_bytes.len()) as u64)
            .unwrap();
        let pointer = std::ptr::NonNull::new(window.as_ref().as_ref().as_ptr() as *mut u8).unwrap();
        let host = unsafe { HostRegion::new(pointer, window.mapped_len(), window.clone()) };
        let region = ReadOnlyMappedRegion::new(&device, host).unwrap();
        let packed = region
            .tensor(packed_element, &[17 * 512], window.data_offset() as u64)
            .unwrap();
        let dense = region
            .tensor(
                dense_element,
                &[4],
                (window.data_offset() + packed_bytes.len()) as u64,
            )
            .unwrap();
        // SAFETY: both import entries write their complete physical results.
        let packed_result =
            unsafe { Tensor::uninitialized(&device, resident_element, &[17, 512]) }.unwrap();
        let dense_result = unsafe { Tensor::uninitialized(&device, dense_element, &[4]) }.unwrap();
        let packed_kernel = repack_weight::native_for_device_with(
            &device,
            repack_weight::Elements {
                E: packed_element,
                U: resident_element,
            },
            &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        let dense_kernel = import_dense::native_for_device_with(
            &device,
            import_dense::Elements {
                E: dense_element,
                U: dense_element,
            },
            &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        let mut batch = NativeTensorBatch::new(&device);
        enqueue_into(
            &AttestedImport::Repack(packed_kernel),
            &mut batch,
            &packed,
            &packed_result,
        )
        .unwrap();
        enqueue_into(
            &AttestedImport::Dense(dense_kernel),
            &mut batch,
            &dense,
            &dense_result,
        )
        .unwrap();
        drop(region);
        drop(window);
        drop(file);
        batch.submit().unwrap().wait().unwrap();
        assert_eq!(
            packed_result.read_to_host().unwrap(),
            resident_element
                .repack_host(packed_element, &[17, 512], &packed_bytes)
                .unwrap()
        );
        assert_eq!(dense_result.read_to_host().unwrap(), dense_bytes);
    }
}
