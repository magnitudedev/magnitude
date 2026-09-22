//! Closed typed-kernel to PTX/native-image compilation.

use crate::command::CompiledKernel;
use crate::driver::{
    compile_image, function_attribute, load_module, set_function_attribute, Context, JitError,
    Module,
};
use crate::{ptx, Cuda};
use seismic_ir::kernel::Kernel;
use seismic_ir::target::KernelEmissionLayout;
use seismic_target::{DeviceDescription, NativeCompilationError};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;

pub struct NativeCandidate {
    pub(crate) module: Arc<Module>,
    pub(crate) entry: String,
    pub(crate) layout: KernelEmissionLayout,
    pub artifact_digest: [u8; 32],
    pub image_bytes: u64,
    pub metadata_bytes: u64,
    pub compilation_ns: u64,
}

pub(crate) fn compile_kernel(
    target: &DeviceDescription<Cuda>,
    kernel: &Kernel<Cuda>,
    layout: &KernelEmissionLayout,
) -> Result<NativeCandidate, NativeCompilationError> {
    let started = Instant::now();
    let driver = crate::driver::Driver::load().map_err(NativeCompilationError::ToolchainFailure)?;
    let ordinal = i32::try_from(target.facts().device_ordinal).map_err(|_| {
        NativeCompilationError::ToolchainFailure("CUDA device ordinal exceeds driver ABI".into())
    })?;
    let context = Context::retain(driver, ordinal).map_err(driver_failure)?;
    let emitted = ptx::emit(target, kernel, layout)?;
    let image = compile_image(&context, &emitted.source).map_err(|error| {
        let numbered = emitted
            .source
            .lines()
            .enumerate()
            .take(160)
            .map(|(line, text)| format!("{:>4}: {text}", line + 1))
            .collect::<Vec<_>>()
            .join("\n");
        NativeCompilationError::ToolchainFailure(format!(
            "{}\nGenerated PTX (first 160 lines):\n{numbered}",
            jit_failure(error)
        ))
    })?;
    let artifact_digest: [u8; 32] = Sha256::digest(&image).into();
    let image_bytes = u64::try_from(image.len()).map_err(|_| {
        NativeCompilationError::MalformedToolchainOutput(
            "CUDA native image size exceeds u64".into(),
        )
    })?;
    let (module, function) = load_module(&context, &image, &emitted.entry).map_err(jit_failure)?;
    let static_shared = function_attribute(&context, function, 1).map_err(driver_failure)?;
    let static_shared = u64::try_from(static_shared).map_err(|_| {
        NativeCompilationError::MalformedToolchainOutput(
            "CUDA function reports negative static shared memory".into(),
        )
    })?;
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
    set_function_attribute(&context, function, 8, dynamic_shared).map_err(driver_failure)?;
    let metadata_bytes = layout_metadata_bytes(layout)?;
    Ok(NativeCandidate {
        module: Arc::new(module),
        entry: emitted.entry,
        layout: layout.clone(),
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
                    * std::mem::size_of::<seismic_ir::target::BindingEmissionLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.locals.len()
                    * std::mem::size_of::<seismic_ir::target::LocalEmissionLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.addressable_resources.len()
                    * std::mem::size_of::<seismic_ir::target::AddressableResourceEmissionLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.scalar_args.len() * std::mem::size_of::<seismic_lang::types::DType>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.result_slots.len()
                    * std::mem::size_of::<(
                        seismic_ir::schedule::AnyScalarSlot,
                        seismic_lang::types::DType,
                    )>(),
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

pub(crate) fn jit_failure(error: JitError) -> NativeCompilationError {
    match error {
        JitError::Toolchain { error, log } => {
            NativeCompilationError::ToolchainFailure(format!("{error}; {log}"))
        }
        JitError::Driver(error) => driver_failure(error),
        JitError::MalformedImage(reason) => NativeCompilationError::ToolchainFailure(reason),
    }
}
