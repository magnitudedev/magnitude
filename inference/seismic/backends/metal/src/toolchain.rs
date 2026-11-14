//! Metal's program toolchain.
//!
//! Compiling a program compiles its source once, with an explicit
//! instantiation for each entry that names a template instance, and forms
//! the pipeline of each entry as any Metal pipeline is formed. Its artifact
//! is made off that path: each pipeline added to a binary archive, formed
//! from the archive alone to prove it serves it, and the archive
//! serialized. The serialized archive holds the library's functions and
//! the pipelines' device code, so a later process opens it, in place where
//! the store keeps it in a file, as both the library and the archive and
//! forms every pipeline from it, compiling nothing. Bytes the OS does not open as an archive, or a pipeline the
//! archive does not serve, are declined rather than compiled: a miss,
//! which the former compiles and stores again.

use crate::direct::DirectPipeline;
use crate::facts::{LanguageVersion, MetalFacts};
use crate::{DeviceHandle, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSError, NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary, MTLPipelineOption,
};
use seismic_native_target::{
    Artifact, Declined, NativeCompilationError, ProgramEntry, ProgramSource, ProgramStatistic,
    Toolchain, ToolchainIdentity,
};

pub struct MetalToolchain {
    device: MetalDevice,
    identity: ToolchainIdentity,
    language: LanguageVersion,
}

impl MetalToolchain {
    pub fn new(device: MetalDevice, facts: &MetalFacts) -> Self {
        Self {
            device,
            language: facts.language,
            identity: ToolchainIdentity {
                namespace: "metal",
                material: material(facts.operating_system(), facts.language),
                stored: format!("{};build {FORMATION_BUILD}", facts.architecture),
            },
        }
    }
}

/// This build's formation code ([`build.rs`](../build.rs)), its compile
/// options among it: part of every stored artifact's key, never of
/// tuning's, with the device's GPU architecture, which the compiled
/// pipelines are code for.
pub const FORMATION_BUILD: &str = env!("SEISMIC_FORMATION_BUILD");

/// Everything besides the source and this build that determines a
/// formation: the OS build, which ships the Metal compiler, and the
/// language version compiled to.
pub fn material(operating_system: &str, language: LanguageVersion) -> String {
    format!("metal;{operating_system};msl-{}", language.label())
}

/// The function name of an entry: a plain kernel name as is, a template
/// instance `kernel<16, 2>` as `kernel$16_2`.
fn host_name(symbol: &str) -> String {
    match symbol.split_once('<') {
        Some((kernel, arguments)) => format!(
            "{kernel}${}",
            arguments
                .trim_end_matches('>')
                .split(',')
                .map(str::trim)
                .collect::<Vec<_>>()
                .join("_")
        ),
        None => symbol.to_owned(),
    }
}

/// A Metal program's artifact: the file of the serialized binary archive
/// of its compile, which holds the library's functions and every
/// pipeline's device code. Metal opens an archive by path alone.
pub struct MetalArchive(ArchiveFile);

impl Artifact for MetalArchive {
    const ENCODING: u32 = 3;
    const IN_PLACE: bool = true;

    /// Any bytes: whether they are an archive that serves the program is
    /// learned by loading them, and bytes that are not are a miss. The
    /// store's file of them is opened in place; without one, they are
    /// written to a file of their own.
    fn decode(bytes: &[u8], file: Option<&std::path::Path>) -> Option<Self> {
        if let Some(path) = file {
            return Some(Self(ArchiveFile { path: path.to_owned(), temporary: false }));
        }
        let file = ArchiveFile::temporary();
        std::fs::write(&file.path, bytes).ok()?;
        Some(Self(file))
    }
}

impl Toolchain for MetalToolchain {
    type Artifact = MetalArchive;
    type Program = Vec<DirectPipeline>;
    type Pending = Formed;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    /// The program of `source`'s entries, its pipelines formed as any
    /// Metal pipeline is (the OS's shader cache serves what it holds), and
    /// what each was formed of, which [`Toolchain::artifact`] archives off
    /// this path: archiving and serializing took ~40 ms a program on an M4
    /// Pro (2026-11-13), as much as a third of a cold compile, and the
    /// program needs none of it.
    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(Formed, Vec<DirectPipeline>), NativeCompilationError> {
        crate::pooled(|| {
            let functions = form_functions(self.device.handle(), source, &source.entries, self.language)?;
            let mut formed = Vec::new();
            let program = self.pipelines(&functions, &source.entries, None, &mut formed)?.ok_or_else(|| {
                NativeCompilationError::MalformedToolchainOutput("a compiled pipeline does not form".into())
            })?;
            Ok((Formed(formed), program))
        })
    }

    /// The binary archive of every pipeline a compile formed, serialized:
    /// each pipeline added, then formed from the archive alone
    /// (`FailOnBinaryArchiveMiss`), so no archive is stored that a load would
    /// have to compile.
    fn artifact(&self, formed: Formed) -> Result<Vec<u8>, NativeCompilationError> {
        crate::pooled(|| {
            let device = self.device.handle().raw();
            let archive = device
                .newBinaryArchiveWithDescriptor_error(&MTLBinaryArchiveDescriptor::new())
                .map_err(failure)?;
            let archives = NSArray::from_retained_slice(std::slice::from_ref(&archive));
            for (function, threads) in &formed.0 {
                let descriptor = pipeline_descriptor(function, *threads);
                archive.addComputePipelineFunctionsWithDescriptor_error(&descriptor).map_err(failure)?;
                descriptor.setBinaryArchives(Some(&archives));
                device
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &descriptor,
                        MTLPipelineOption::FailOnBinaryArchiveMiss,
                        None,
                    )
                    .map_err(|_| {
                        NativeCompilationError::MalformedToolchainOutput(
                            "a binary archive does not serve a pipeline just added to it".into(),
                        )
                    })?;
            }
            let file = ArchiveFile::temporary();
            archive.serializeToURL_error(&file.url()).map_err(failure)?;
            std::fs::read(&file.path).map_err(|error| {
                NativeCompilationError::ToolchainFailure(format!("reading the serialized archive: {error}"))
            })
        })
    }

    /// `entries`' pipelines from `artifact`, opened as their library and as
    /// the archive of their device code; declined where it does not serve
    /// them without compiling.
    fn load(&self, artifact: &MetalArchive, entries: &[ProgramEntry]) -> Result<Vec<DirectPipeline>, Declined> {
        crate::pooled(|| {
            let device = self.device.handle().raw();
            let url = artifact.0.url();
            let descriptor = MTLBinaryArchiveDescriptor::new();
            descriptor.setUrl(Some(&url));
            let archive = device
                .newBinaryArchiveWithDescriptor_error(&descriptor)
                .map_err(|error| Declined(format!("the OS does not open the bytes as a binary archive: {error}")))?;
            let library = device
                .newLibraryWithURL_error(&url)
                .map_err(|error| Declined(format!("the OS does not open the bytes as a library: {error}")))?;
            let functions = entries
                .iter()
                .map(|entry| {
                    library
                        .newFunctionWithName(&NSString::from_str(&host_name(&entry.symbol)))
                        .ok_or_else(|| Declined(format!("the archive's library holds no `{}`", entry.symbol)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.pipelines(&functions, entries, Some(&archive), &mut Vec::new())?
                .ok_or_else(|| Declined("the archive does not serve a pipeline without compiling".into()))
        })
    }

    /// Metal's compiler reports nothing of a pipeline.
    fn statistics(_: &Vec<DirectPipeline>, _: usize) -> &[ProgramStatistic] {
        &[]
    }
}

fn failure(error: Retained<NSError>) -> NativeCompilationError {
    NativeCompilationError::ToolchainFailure(error.to_string())
}

/// What each pipeline of a compile was formed of: its function, and the
/// threads a group of it was formed to admit where its launch needed more
/// than the function's own pipeline does.
pub struct Formed(Vec<(Retained<ProtocolObject<dyn MTLFunction>>, Option<u64>)>);

// Functions are immutable once formed, and documented as thread-safe.
unsafe impl Send for Formed {}

/// The descriptor of `function`'s pipeline, admitting `threads` a group
/// where given.
fn pipeline_descriptor(function: &ProtocolObject<dyn MTLFunction>, threads: Option<u64>) -> Retained<MTLComputePipelineDescriptor> {
    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(function));
    if let Some(threads) = threads {
        descriptor.setMaxTotalThreadsPerThreadgroup(threads as usize);
    }
    descriptor
}

/// A file a binary archive is serialized to or opened from, which Metal
/// names by URL alone: the store's own, or a temporary one, removed when
/// dropped.
struct ArchiveFile {
    path: std::path::PathBuf,
    temporary: bool,
}

impl ArchiveFile {
    fn temporary() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "seismic-metal-{}-{}.metallib",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        Self { path, temporary: true }
    }

    fn url(&self) -> Retained<NSURL> {
        NSURL::fileURLWithPath(&NSString::from_str(&self.path.to_string_lossy()))
    }
}

impl Drop for ArchiveFile {
    fn drop(&mut self) {
        if self.temporary {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl MetalToolchain {
    /// The pipeline of each of `entries`, each function of `functions` in
    /// turn: formed as any Metal pipeline is, without `archive`, and from
    /// `archive` alone with it, never compiled (none when it does not serve
    /// one). What each pipeline was formed of is added to `shapes`.
    ///
    /// Each symbol's pipeline is formed once. An entry's group size is the
    /// one its launch declares: when register allocation left that pipeline
    /// admitting fewer threads, the entry gets a pipeline formed to admit
    /// them; otherwise it shares the symbol's.
    fn pipelines(
        &self,
        functions: &[Retained<ProtocolObject<dyn MTLFunction>>],
        entries: &[ProgramEntry],
        archive: Option<&Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
        shapes: &mut Vec<(Retained<ProtocolObject<dyn MTLFunction>>, Option<u64>)>,
    ) -> Result<Option<Vec<DirectPipeline>>, NativeCompilationError> {
        let device = self.device.handle().raw();
        let archives = archive.map(|archive| NSArray::from_retained_slice(std::slice::from_ref(archive)));
        let mut form = |function: &Retained<ProtocolObject<dyn MTLFunction>>, threads: Option<u64>| {
            shapes.push((function.clone(), threads));
            match (&archives, threads) {
                (None, None) => device.newComputePipelineStateWithFunction_error(function).map(Some).map_err(failure),
                (None, Some(_)) => device
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &pipeline_descriptor(function, threads),
                        MTLPipelineOption::empty(),
                        None,
                    )
                    .map(Some)
                    .map_err(failure),
                (Some(archives), _) => {
                    let descriptor = pipeline_descriptor(function, threads);
                    descriptor.setBinaryArchives(Some(archives));
                    Ok(device
                        .newComputePipelineStateWithDescriptor_options_reflection_error(
                            &descriptor,
                            MTLPipelineOption::FailOnBinaryArchiveMiss,
                            None,
                        )
                        .ok())
                }
            }
        };
        let mut formed: Vec<(&str, Retained<ProtocolObject<dyn MTLComputePipelineState>>)> = Vec::new();
        let mut pipelines = Vec::with_capacity(entries.len());
        for (entry, function) in entries.iter().zip(functions) {
            let natural = match formed.iter().find(|(symbol, _)| *symbol == entry.symbol) {
                Some((_, state)) => state.clone(),
                None => {
                    let Some(state) = form(function, None)? else {
                        return Ok(None);
                    };
                    formed.push((&entry.symbol, state.clone()));
                    state
                }
            };
            let threads = entry
                .group_size
                .map(|size| size.iter().map(|&axis| u64::from(axis)).product::<u64>());
            let state = match threads {
                Some(threads) if (natural.maxTotalThreadsPerThreadgroup() as u64) < threads => {
                    let Some(state) = form(function, Some(threads))? else {
                        return Ok(None);
                    };
                    state
                }
                _ => natural,
            };
            pipelines.push(DirectPipeline::from_state(state));
        }
        Ok(Some(pipelines))
    }
}

/// The text of `source`'s library: its text with an explicit instantiation
/// for each entry that names a template instance.
fn library_text(source: &ProgramSource) -> String {
    let mut text = source.text.clone();
    let mut instantiated = Vec::new();
    for entry in &source.entries {
        // Entries of one template instance that differ only in group size
        // share its instantiation.
        if entry.symbol.contains('<') && !instantiated.contains(&&entry.symbol) {
            instantiated.push(&entry.symbol);
            text.push_str(&format!(
                "\ntemplate [[host_name(\"{}\")]] [[kernel]] decltype({symbol}) {symbol};\n",
                host_name(&entry.symbol),
                symbol = entry.symbol
            ));
        }
    }
    text
}

/// `source`'s library ([`library_text`]) and the function of each of
/// `entries`, entries of `source`, in their order: everything a Metal
/// program is formed from before its pipelines, and what
/// [`MetalToolchain::compile`] forms them of. Outside the toolchain it is
/// the offline coverage check's compile-only formation (whether a program
/// forms, no pipelines), which the closed exemption list names.
pub fn form_functions(
    device: &DeviceHandle,
    source: &ProgramSource,
    entries: &[ProgramEntry],
    language: LanguageVersion,
) -> Result<Vec<Retained<ProtocolObject<dyn MTLFunction>>>, NativeCompilationError> {
    let text = library_text(source);
    crate::pooled(|| {
        let options = crate::profile::compile_options(language);
        let library = device
            .raw()
            .newLibraryWithSource_options_error(&NSString::from_str(&text), Some(&options))
            .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
        entries
            .iter()
            .map(|entry| {
                let name = host_name(&entry.symbol);
                library
                    .newFunctionWithName(&NSString::from_str(&name))
                    .ok_or_else(|| {
                        NativeCompilationError::MalformedToolchainOutput(format!(
                            "Metal library does not define kernel `{name}`"
                        ))
                    })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_native_target::ProgramEntry;

    #[test]
    fn native_tensor_operations_use_the_probed_language() {
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device, crate::test_support::store()).unwrap();
        if !facts.facts().properties().tensor_ops.holds() {
            return;
        }
        let source = ProgramSource {
            text: "#include <metal_stdlib>\n#include <metal_tensor>\n\
                   #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n\
                   using namespace metal;\n\
                   kernel void tensor_probe(device float* out [[buffer(0)]]) { \
                   typedef extents<int32_t, 16, 16> shape; \
                   typedef tensor<threadgroup half, shape, tensor_inline> tile; \
                   constexpr auto d = mpp::tensor_ops::matmul2d_descriptor(16, 16, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate); \
                   mpp::tensor_ops::matmul2d<d, execution_simdgroups<1>> op; \
                   auto acc = op.get_destination_cooperative_tensor<tile, tile, float>(); \
                   out[0] = float(acc.get_capacity()); }\n".into(),
            entries: vec![ProgramEntry::named("tensor_probe")],
        };
        let toolchain = MetalToolchain::new(device, facts.facts());
        assert_eq!(toolchain.compile(&source).unwrap().1.len(), 1);
    }

    /// A stored archive serves its compile's pipelines to another toolchain
    /// on the device, any subset of its entries, a pipeline formed for a
    /// declared group size included, compiling nothing (every pipeline is
    /// formed under `FailOnBinaryArchiveMiss`).
    #[test]
    fn a_stored_archive_serves_its_pipelines_without_compiling() {
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device, crate::test_support::store()).unwrap().facts().clone();
        let source = ProgramSource {
            text: "#include <metal_stdlib>\nusing namespace metal;\n\
                   template <uint TILE> kernel void probe(device float* x [[buffer(0)]], \
                   uint i [[thread_position_in_grid]]) { x[i] *= TILE; }\n"
                .into(),
            entries: vec![
                ProgramEntry::named("probe<2>"),
                ProgramEntry { symbol: "probe<4>".into(), group_size: Some([1024, 1, 1]), constants: Vec::new() },
            ],
        };
        let compiling = MetalToolchain::new(device.clone(), &facts);
        let stored = compiling.artifact(compiling.compile(&source).unwrap().0).unwrap();
        let toolchain = MetalToolchain::new(device, &facts);
        let artifact = MetalArchive::decode(&stored, None).unwrap();
        assert_eq!(toolchain.load(&artifact, &source.entries).expect("the archive serves its compile").len(), 2);
        let wanted = &source.entries[1..];
        let loaded = toolchain.load(&artifact, wanted).expect("the archive serves a subset");
        assert!(loaded[0].max_threads_per_threadgroup() >= 1024);
        // A store's file serves in place, and stays the store's.
        let kept = std::env::temp_dir().join(format!("seismic-metal-kept-{}.metallib", std::process::id()));
        std::fs::write(&kept, &stored).unwrap();
        let opened = MetalArchive::decode(&stored, Some(&kept)).unwrap();
        assert_eq!(toolchain.load(&opened, wanted).expect("the file serves in place").len(), 1);
        drop(opened);
        std::fs::remove_file(kept).expect("opening in place leaves the store's file");
    }

    /// Bytes that are no archive, or an archive without a requested entry,
    /// are declined: a miss, which the former compiles again.
    #[test]
    fn what_an_archive_does_not_serve_is_a_miss() {
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device, crate::test_support::store()).unwrap().facts().clone();
        let toolchain = MetalToolchain::new(device, &facts);
        let source = ProgramSource {
            text: "kernel void probe(device float* x [[buffer(0)]]) { x[0] = 1; }\n".into(),
            entries: vec![ProgramEntry::named("probe")],
        };
        let damaged = MetalArchive::decode(b"not an archive", None).unwrap();
        assert!(toolchain.load(&damaged, &source.entries).is_err());
        let artifact = MetalArchive::decode(&toolchain.artifact(toolchain.compile(&source).unwrap().0).unwrap(), None).unwrap();
        assert!(toolchain.load(&artifact, &[ProgramEntry::named("absent")]).is_err());
    }

    #[test]
    fn template_instances_are_named_by_their_arguments() {
        assert_eq!(host_name("probe"), "probe");
        assert_eq!(host_name("probe<4>"), "probe$4");
        assert_eq!(host_name("probe<16, 2>"), "probe$16_2");
    }

    /// An entry's group size is its launch's, which its pipeline admits;
    /// entries of one template instance share its instantiation.
    #[test]
    fn an_entry_group_size_bounds_its_pipeline() {
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device, crate::test_support::store())
            .unwrap()
            .facts()
            .clone();
        let toolchain = MetalToolchain::new(device, &facts);
        let source = ProgramSource {
            text: "#include <metal_stdlib>\nusing namespace metal;\n\
                   kernel void plain(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1; }\n\
                   template <uint TILE> kernel void probe(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] *= TILE; }\n"
                .into(),
            entries: vec![
                ProgramEntry {
                    symbol: "plain".into(),
                    group_size: Some([256, 1, 1]),
                    constants: Vec::new(),
                },
                ProgramEntry::named("plain"),
                ProgramEntry {
                    symbol: "probe<4>".into(),
                    group_size: Some([128, 1, 1]),
                    constants: Vec::new(),
                },
                ProgramEntry {
                    symbol: "probe<4>".into(),
                    group_size: Some([512, 1, 1]),
                    constants: Vec::new(),
                },
            ],
        };
        let pipelines = toolchain.compile(&source).unwrap().1;
        assert_eq!(pipelines.len(), 4);
        for (pipeline, entry) in pipelines.iter().zip(&source.entries) {
            let declared = entry.group_size.map_or(1, |size| size.iter().product::<u32>());
            assert!(pipeline.max_threads_per_threadgroup() >= u64::from(declared));
        }
    }

    /// One compile forms every requested instance of a templated kernel.
    #[test]
    fn every_requested_instance_forms_a_pipeline() {
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device, crate::test_support::store())
            .unwrap()
            .facts()
            .clone();
        let toolchain = MetalToolchain::new(device, &facts);
        let source = ProgramSource {
            text: "#include <metal_stdlib>\nusing namespace metal;\n\
                   template <uint TILE> kernel void probe(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] *= TILE; }\n\
                   kernel void plain(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1; }\n"
                .into(),
            entries: vec![
                ProgramEntry::named("probe<4>"),
                ProgramEntry::named("probe<8>"),
                ProgramEntry::named("plain"),
            ],
        };
        assert_eq!(toolchain.compile(&source).unwrap().1.len(), 3);
    }
}
