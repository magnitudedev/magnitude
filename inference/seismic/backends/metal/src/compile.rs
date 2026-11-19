//! Closed typed-kernel to Metal pipeline formation: the kernel's MSL,
//! formed through the device's programs.

use crate::command::Pipeline;
use crate::{render, DirectPipeline, MetalPrograms};
use seismic_ir::kernel::Kernel;
use seismic_ir::physical_target::{KernelEmissionLayout, KernelWordLayout};
use seismic_native_target::{
    Formed, NativeCompilationError, ProgramEntry, ProgramSource,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

/// A formed kernel before reflection: the formed source, which holds its
/// pipeline alive, and the kernel's entry in it.
pub struct NativeCandidate {
    formed: Formed<Vec<DirectPipeline>>,
    entry: ProgramEntry,
    words: KernelWordLayout,
    pub artifact_digest: [u8; 32],
    pub source_bytes: u64,
    pub metadata_bytes: u64,
    pub compilation_ns: u64,
}

impl NativeCandidate {
    /// The kernel's pipeline.
    pub fn pipeline(&self) -> &DirectPipeline {
        let (pipelines, position) = self.formed.program(&self.entry);
        &pipelines[position]
    }

    /// The executable's handle: the kernel's pipeline, kept alive by its
    /// compile, and its word layout.
    pub(crate) fn into_handle(self) -> Pipeline {
        let (compile, position) = self.formed.entry(&self.entry);
        Pipeline::new(compile.clone(), position, self.words)
    }
}

pub(crate) fn form_kernel(
    programs: &MetalPrograms,
    kernel: &Kernel<crate::Metal>,
    layout: &KernelEmissionLayout,
) -> Result<NativeCandidate, NativeCompilationError> {
    let started = Instant::now();
    let rendered = render::render(0, kernel, layout);
    let text = format!(
        "{}{}{}",
        render::library_prelude(),
        render::SOFTFLOAT_PRELUDE,
        rendered.source
    );
    let artifact_digest: [u8; 32] = Sha256::digest(text.as_bytes()).into();
    let source_bytes = u64::try_from(text.len()).map_err(|_| {
        NativeCompilationError::MalformedToolchainOutput(
            "Metal source artifact length exceeds u64".into(),
        )
    })?;
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
                        * std::mem::size_of::<
                            seismic_ir::physical_target::AddressableResourceWordLayout,
                        >(),
                )
        })
        .and_then(|bytes| bytes.checked_add(rendered.name.len()))
        .ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(
                "Metal artifact metadata footprint exceeds usize".into(),
            )
        })?;
    let entry = ProgramEntry::named(rendered.name);
    let formed = programs
        .form(&[ProgramSource { text, entries: vec![entry.clone()] }])?
        .pop()
        .expect("one formed source for the one source");
    Ok(NativeCandidate {
        formed,
        entry,
        words: layout.words.clone(),
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
