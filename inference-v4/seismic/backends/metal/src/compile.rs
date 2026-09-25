//! Closed typed-kernel to Metal pipeline compilation.

use crate::command::Pipeline;
use crate::{Metal, render};
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLLibrary};
use seismic_ir::kernel::Kernel;
use seismic_ir::physical_target::KernelEmissionLayout;
use seismic_native_target::{DeviceDescription, NativeCompilationError};
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
    context: &crate::DeviceHandle,
    target: &DeviceDescription<Metal>,
    kernel: &Kernel<Metal>,
    layout: &KernelEmissionLayout,
) -> Result<NativeCandidate, NativeCompilationError> {
    assert_eq!(
        context.registry_id(),
        target.facts().registry_id,
        "native context belongs to another Metal target"
    );
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
    let library = context
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
    let state = context
        .raw()
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
    let metadata_bytes = std::mem::size_of::<Pipeline>()
        .checked_add(
            layout.result_types.len() * std::mem::size_of::<seismic_ir::repr::ScalarKind>(),
        )
        .and_then(|bytes| {
            bytes.checked_add(
                layout.words.bindings.len()
                    * std::mem::size_of::<seismic_ir::physical_target::BindingWordLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.words.locals.len()
                    * std::mem::size_of::<seismic_ir::physical_target::LocalWordLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.words.addressable_resources.len()
                    * std::mem::size_of::<seismic_ir::physical_target::AddressableResourceWordLayout>(),
            )
        })
        .and_then(|bytes| bytes.checked_add(source_text.len()))
        .and_then(|bytes| bytes.checked_add(rendered.name.len()))
        .ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(
                "Metal artifact metadata footprint exceeds usize".into(),
            )
        })?;
    Ok(NativeCandidate {
        pipeline: Pipeline {
            state,
            words: layout.words.clone(),
            source: source_text.into_boxed_str(),
            entry: rendered.name.into_boxed_str(),
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
