//! Typed-kernel to native compilation: the kernel's function emitted as a
//! CLIF program and formed through the device's programs, so a kernel any
//! process of the user formed is loaded, compiling nothing.

use crate::command::CompiledKernel;
use crate::{emit, toolchain, Cpu, CpuPrograms};
use seismic_ir::kernel::Kernel;
use seismic_ir::physical_target::KernelEmissionLayout;
use seismic_native_target::{DeviceDescription, NativeCompilationError};
use std::time::Instant;

pub struct NativeCandidate {
    pub kernel: CompiledKernel,
    pub artifact_digest: [u8; 32],
    pub frame_size: u64,
    pub code_bytes: u64,
    pub metadata_bytes: u64,
    pub compilation_ns: u64,
}

pub(crate) fn compile_kernel(
    programs: &CpuPrograms,
    target: &DeviceDescription<Cpu>,
    kernel: &Kernel<Cpu>,
    layout: &KernelEmissionLayout,
) -> Result<NativeCandidate, NativeCompilationError> {
    let started = Instant::now();
    let emitted = emit::emit(kernel, layout, target.device_info().codegen.call_conv)?;
    let source = toolchain::source(&emitted);
    let formed = programs
        .form(std::slice::from_ref(&source))?
        .pop()
        .expect("one formed source per source");
    let (program, _) = formed.program(&source.entries[0]);
    let (entry, artifact_digest, frame_size, code_bytes) = (
        program.entry(),
        program.digest(),
        program.frame_size(),
        program.code_bytes(),
    );
    let metadata_bytes = layout_metadata_bytes(layout)?;
    Ok(NativeCandidate {
        kernel: CompiledKernel {
            entry,
            _program: formed,
            layout: layout.clone(),
        },
        artifact_digest,
        frame_size,
        code_bytes,
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
            NativeCompilationError::ToolchainFailure(
                "CPU native artifact metadata footprint exceeds usize".into(),
            )
        })?;
    u64::try_from(bytes).map_err(|_| {
        NativeCompilationError::ToolchainFailure(
            "CPU native artifact metadata footprint exceeds u64".into(),
        )
    })
}
