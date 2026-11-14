//! Closed typed-kernel to PTX compilation: the kernel's PTX formed through
//! the device's PTX programs ([`CudaPtxPrograms`]), so a kernel any process
//! formed is loaded from its CUBIN.

use crate::command::CompiledKernel;
use crate::direct::DirectModule;
use crate::driver::set_function_attribute;
use crate::{ptx, Cuda, CudaPtxPrograms};
use seismic_ir::kernel::Kernel;
use seismic_ir::physical_target::KernelEmissionLayout;
use seismic_native_target::{DeviceDescription, Formed, NativeCompilationError, ProgramEntry, ProgramSource};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub struct NativeCandidate {
    /// The kernel's formed PTX program, which holds `entry`.
    pub(crate) formed: Formed<DirectModule>,
    pub(crate) entry: ProgramEntry,
    pub(crate) layout: KernelEmissionLayout,
    pub(crate) configured_dynamic_shared_bytes: i32,
    pub artifact_digest: [u8; 32],
    pub image_bytes: u64,
    pub metadata_bytes: u64,
    pub compilation_ns: u64,
}

pub(crate) fn compile_kernel(
    programs: &CudaPtxPrograms,
    target: &DeviceDescription<Cuda>,
    kernel: &Kernel<Cuda>,
    layout: &KernelEmissionLayout,
) -> Result<NativeCandidate, NativeCompilationError> {
    let started = Instant::now();
    let emitted = ptx::emit(target, kernel, layout)?;
    // The PTX and the device's compatibility identity, which names its
    // driver and architecture, determine the CUBIN.
    let artifact_digest: [u8; 32] = Sha256::digest(emitted.source.as_bytes()).into();
    let entry = ProgramEntry::named(emitted.entry);
    let source = ProgramSource { text: emitted.source, entries: vec![entry.clone()] };
    let formed = programs
        .form(std::slice::from_ref(&source))?
        .pop()
        .expect("one formed program for one source");
    let (module, function) = formed.program(&entry);
    let context = module.context();
    let image_bytes = module.image_bytes();
    let static_shared = module.static_shared_bytes(function);
    let dynamic_shared = target
        .limits()
        .max_workgroup_bytes
        .checked_sub(static_shared)
        .ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(
                "CUDA function static shared memory exceeds the device limit".into(),
            )
        })?;
    let dynamic_shared = i32::try_from(dynamic_shared).map_err(|_| {
        NativeCompilationError::ToolchainFailure(
            "CUDA opt-in workgroup storage limit exceeds driver ABI".into(),
        )
    })?;
    // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8. Reserve the exact
    // remaining device-wide shared-memory domain after reflected static use;
    // reconciliation reads the configured function value back.
    set_function_attribute(context, module.function(function), 8, dynamic_shared).map_err(driver_failure)?;
    let entry_bytes = u64::try_from(entry.symbol.len()).map_err(|_| {
        NativeCompilationError::MalformedToolchainOutput(
            "CUDA native entry name size exceeds u64".into(),
        )
    })?;
    let metadata_bytes = layout_metadata_bytes(layout)?
        .checked_add(entry_bytes)
        .ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(
                "CUDA native metadata exceeds u64".into(),
            )
        })?;
    Ok(NativeCandidate {
        formed,
        entry,
        layout: layout.clone(),
        configured_dynamic_shared_bytes: dynamic_shared,
        artifact_digest,
        image_bytes,
        metadata_bytes,
        compilation_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

fn layout_metadata_bytes(layout: &KernelEmissionLayout) -> Result<u64, NativeCompilationError> {
    let bytes = std::mem::size_of::<CompiledKernel>()
        .checked_add(std::mem::size_of::<KernelEmissionLayout>())
        .and_then(|bytes| {
            bytes.checked_add(
                layout.bindings.len()
                    * std::mem::size_of::<seismic_ir::physical_target::BindingEmissionLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.locals.len()
                    * std::mem::size_of::<seismic_ir::physical_target::LocalEmissionLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.addressable_resources.len()
                    * std::mem::size_of::<
                        seismic_ir::physical_target::AddressableResourceEmissionLayout,
                    >(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.scalar_args.len() * std::mem::size_of::<seismic_ir::repr::ScalarKind>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.result_types.len() * std::mem::size_of::<seismic_ir::repr::ScalarKind>(),
            )
        })
        .ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(
                "CUDA artifact metadata footprint exceeds usize".into(),
            )
        })?;
    u64::try_from(bytes).map_err(|_| {
        NativeCompilationError::MalformedToolchainOutput(
            "CUDA artifact metadata footprint exceeds u64".into(),
        )
    })
}

fn driver_failure(error: crate::driver::DriverError) -> NativeCompilationError {
    NativeCompilationError::ToolchainFailure(error.to_string())
}
