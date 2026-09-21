//! Closed typed-kernel to Metal pipeline compilation.

use crate::command::Pipeline;
use crate::{render, Metal};
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLLibrary};
use seismic_compiler::errors::NativeCompilationError;
use seismic_compiler::kernel::Kernel;
use seismic_compiler::target::{DeviceContract, KernelEmissionLayout};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub struct NativeCandidate {
    pub pipeline: Pipeline,
    pub artifact_digest: [u8; 32],
    pub source_bytes: u64,
    pub metadata_bytes: u64,
    pub compilation_ns: u64,
}

pub(crate) fn compile_kernel(
    target: &DeviceContract<Metal>,
    kernel: &Kernel<Metal>,
    layout: &KernelEmissionLayout,
) -> Result<NativeCandidate, NativeCompilationError> {
    let started = Instant::now();
    let rendered = render::render(0, kernel, layout);
    let source_text = format!(
        "{}{}{}",
        render::LIBRARY_PRELUDE,
        render::SOFTFLOAT_PRELUDE,
        rendered.source
    );
    let artifact_digest: [u8; 32] = Sha256::digest(source_text.as_bytes()).into();
    let source_bytes = u64::try_from(source_text.len()).map_err(|_| {
        NativeCompilationError::MalformedToolchainOutput(
            "Metal source artifact length exceeds u64".into(),
        )
    })?;
    let source = NSString::from_str(&source_text);
    let options = crate::profile::compile_options(target.facts().language);
    let library = target
        .facts()
        .device
        .raw()
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
    let name = NSString::from_str(&rendered.name);
    let function = library.newFunctionWithName(&name).ok_or_else(|| {
        NativeCompilationError::MalformedToolchainOutput(format!(
            "Metal library omitted compiled function `{}`",
            rendered.name
        ))
    })?;
    let state = target
        .facts()
        .device
        .raw()
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
    let metadata_bytes = std::mem::size_of::<Pipeline>()
        .checked_add(
            layout.result_slots.len()
                * std::mem::size_of::<(
                    seismic_compiler::schedule::AnyScalarSlot,
                    seismic_lang::types::DType,
                )>(),
        )
        .and_then(|bytes| {
            bytes.checked_add(
                layout.words.bindings.len()
                    * std::mem::size_of::<seismic_compiler::target::BindingWordLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.words.locals.len()
                    * std::mem::size_of::<seismic_compiler::target::LocalWordLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.words.addressable_resources.len()
                    * std::mem::size_of::<seismic_compiler::target::AddressableResourceWordLayout>(
                    ),
            )
        })
        .ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(
                "Metal artifact metadata footprint exceeds usize".into(),
            )
        })?;
    Ok(NativeCandidate {
        pipeline: Pipeline {
            state,
            words: layout.words.clone(),
            result_slots: layout.result_slots.clone(),
        },
        artifact_digest,
        source_bytes,
        metadata_bytes: u64::try_from(metadata_bytes).map_err(|_| {
            NativeCompilationError::MalformedToolchainOutput(
                "Metal artifact metadata footprint exceeds u64".into(),
            )
        })?,
        compilation_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}
