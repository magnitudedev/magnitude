//! Metal's program toolchain.
//!
//! Compiling a program compiles its source once, with an explicit
//! instantiation for each entry that names a template instance, and forms a
//! pipeline per entry. Its artifact is the source and its entries, not
//! compiled bytes ([`Load::PlatformCached`]): loading forms the library of
//! exactly that source and entry list again, whatever entries it is asked
//! for, so the OS's shader cache serves what the compile formed. That is
//! measured, not assumed: a load of Qwen3.6 35B Q4 in a process after the
//! one that compiled it re-forms its 161 libraries and 202 pipelines in
//! 0.5 s on an M4 Pro, against 70 s compiling them. The former counts and
//! times every re-formation, so a platform that stops serving them shows
//! as their time.

use crate::direct::DirectPipeline;
use crate::facts::{LanguageVersion, MetalFacts};
use crate::{DeviceHandle, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary};
use seismic_native_target::{
    Artifact, Load, NativeCompilationError, ProgramEntry, ProgramSource, ProgramStatistic,
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
            },
        }
    }
}

/// Everything besides the source that determines a formation: the OS build,
/// which ships the Metal compiler, and the language version compiled to.
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

/// A Metal program's artifact: the source text and its entries, which the
/// OS's shader cache serves the compiled forms of.
pub struct MetalSource {
    source: ProgramSource,
}

impl Artifact for MetalSource {
    const ENCODING: u32 = 2;

    /// The text, then each entry: its symbol, whether it states a group
    /// size and that size, and its constants, as little-endian words and
    /// length-prefixed strings.
    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let word = |bytes: &mut Vec<u8>, value: u32| bytes.extend_from_slice(&value.to_le_bytes());
        let text = |bytes: &mut Vec<u8>, text: &str| {
            word(bytes, u32::try_from(text.len()).expect("a source fits u32"));
            bytes.extend_from_slice(text.as_bytes());
        };
        text(&mut bytes, &self.source.text);
        for entry in &self.source.entries {
            text(&mut bytes, &entry.symbol);
            word(&mut bytes, u32::from(entry.group_size.is_some()));
            for axis in entry.group_size.into_iter().flatten() {
                word(&mut bytes, axis);
            }
            word(&mut bytes, u32::try_from(entry.constants.len()).expect("an entry's constants fit u32"));
            for constant in &entry.constants {
                word(&mut bytes, *constant);
            }
        }
        bytes
    }

    fn decode(mut bytes: &[u8]) -> Option<Self> {
        fn word(bytes: &mut &[u8]) -> Option<u32> {
            let value = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
            *bytes = &bytes[4..];
            Some(value)
        }
        fn text(bytes: &mut &[u8]) -> Option<String> {
            let length = word(bytes)? as usize;
            let text = std::str::from_utf8(bytes.get(..length)?).ok()?.to_owned();
            *bytes = &bytes[length..];
            Some(text)
        }
        let source = text(&mut bytes)?;
        let mut entries = Vec::new();
        while !bytes.is_empty() {
            let symbol = text(&mut bytes)?;
            let group_size = match word(&mut bytes)? {
                0 => None,
                1 => Some([word(&mut bytes)?, word(&mut bytes)?, word(&mut bytes)?]),
                _ => return None,
            };
            let constants = (0..word(&mut bytes)?)
                .map(|_| word(&mut bytes))
                .collect::<Option<Vec<_>>>()?;
            entries.push(ProgramEntry {
                symbol,
                group_size,
                constants,
            });
        }
        Some(Self {
            source: ProgramSource { text: source, entries },
        })
    }
}

impl Toolchain for MetalToolchain {
    type Artifact = MetalSource;
    type Program = Vec<DirectPipeline>;
    const LOAD: Load = Load::PlatformCached;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    /// The program compiled of `source`, which the OS cache then holds
    /// for a later load of the artifact.
    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(MetalSource, Vec<DirectPipeline>), NativeCompilationError> {
        let program = crate::pooled(|| self.pipelines(source, &source.entries))?;
        Ok((
            MetalSource {
                source: source.clone(),
            },
            program,
        ))
    }

    fn load(
        &self,
        artifact: &MetalSource,
        entries: &[ProgramEntry],
    ) -> Result<Vec<DirectPipeline>, NativeCompilationError> {
        crate::pooled(|| self.pipelines(&artifact.source, entries))
    }

    /// Metal's compiler reports nothing of a pipeline.
    fn statistics(_: &Vec<DirectPipeline>, _: usize) -> &[ProgramStatistic] {
        &[]
    }
}

impl MetalToolchain {
    /// Compile `source`'s library and form the pipeline of each of
    /// `entries`, entries of `source`.
    fn pipelines(
        &self,
        source: &ProgramSource,
        entries: &[ProgramEntry],
    ) -> Result<Vec<DirectPipeline>, NativeCompilationError> {
        let device = self.device.handle().raw();
        let functions = form_functions(self.device.handle(), source, entries, self.language)?;
        let failure =
            |error: Retained<NSError>| NativeCompilationError::ToolchainFailure(error.to_string());
        // Each symbol's pipeline is formed once. An entry's group size is the
        // one its launch declares: when register allocation left that
        // pipeline admitting fewer threads, the entry gets a pipeline formed
        // to admit them; otherwise it shares the symbol's.
        let mut formed: Vec<(&str, Retained<ProtocolObject<dyn MTLComputePipelineState>>)> =
            Vec::new();
        let mut pipelines = Vec::with_capacity(entries.len());
        for (entry, function) in entries.iter().zip(&functions) {
            let natural = match formed.iter().find(|(symbol, _)| *symbol == entry.symbol) {
                Some((_, state)) => state.clone(),
                None => {
                    let state = device
                        .newComputePipelineStateWithFunction_error(function)
                        .map_err(failure)?;
                    formed.push((&entry.symbol, state.clone()));
                    state
                }
            };
            let threads = entry
                .group_size
                .map(|size| size.iter().map(|&axis| u64::from(axis)).product::<u64>());
            let state = match threads {
                Some(threads)
                    if (natural.maxTotalThreadsPerThreadgroup() as u64) < threads =>
                {
                    let descriptor = objc2_metal::MTLComputePipelineDescriptor::new();
                    descriptor.setComputeFunction(Some(function));
                    descriptor.setMaxTotalThreadsPerThreadgroup(threads as usize);
                    device
                        .newComputePipelineStateWithDescriptor_options_reflection_error(
                            &descriptor,
                            objc2_metal::MTLPipelineOption::empty(),
                            None,
                        )
                        .map_err(failure)?
                }
                _ => natural,
            };
            pipelines.push(DirectPipeline::from_state(state));
        }
        Ok(pipelines)
    }
}

/// The text of `source`'s library: its text with an explicit instantiation
/// for each entry that names a template instance. It depends on the whole
/// entry list, so a load forms it of every entry its artifact's compile
/// formed, not only of those it asks for.
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
/// program is formed from before its pipelines.
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

/// Whether this host's Metal compiler forms tensor-operation types and
/// cooperative destinations (Metal 4), whatever the GPU executes: what
/// forming a `SEISMIC_HAS_TENSOR_OPS` source needs.
pub fn forms_tensor_operations(device: &DeviceHandle, language: LanguageVersion) -> bool {
    let source = ProgramSource {
        text: "#include <metal_stdlib>\n#include <metal_tensor>\n\
               #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n\
               kernel void seismic_tensor_headers(device float* out [[buffer(0)]]) { \
               using shape = metal::extents<int32_t, 16, 16>; \
               using tile = metal::tensor<threadgroup half, shape, metal::tensor_inline>; \
               constexpr auto d = mpp::tensor_ops::matmul2d_descriptor(16, 16, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate); \
               mpp::tensor_ops::matmul2d<d, metal::execution_simdgroups<1>> op; \
               auto acc = op.get_destination_cooperative_tensor<tile, tile, float>(); \
               out[0] = float(acc.get_capacity()); }\n"
            .to_owned(),
        entries: vec![seismic_native_target::ProgramEntry::named("seismic_tensor_headers")],
    };
    crate::pooled(|| form_functions(device, &source, &source.entries, language).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_native_target::ProgramEntry;

    #[test]
    fn native_tensor_operations_use_the_probed_language() {
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device).unwrap();
        if !facts.facts().tensor_ops {
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

    /// The artifact keeps the source and its entries whole: a stated group
    /// size of zeros is not read back as none.
    #[test]
    fn an_artifact_comes_back_as_it_went() {
        let source = ProgramSource {
            text: "kernel void probe() {}".into(),
            entries: vec![
                ProgramEntry::named("probe<2>"),
                ProgramEntry {
                    symbol: "probe<4>".into(),
                    group_size: Some([64, 1, 1]),
                    constants: vec![3, 9],
                },
                ProgramEntry {
                    symbol: "probe<8>".into(),
                    group_size: Some([0, 0, 0]),
                    constants: Vec::new(),
                },
            ],
        };
        let decoded = MetalSource::decode(&MetalSource { source: source.clone() }.encode()).unwrap();
        assert_eq!(decoded.source, source);
    }

    /// A load of some of an artifact's entries forms the library its
    /// compile formed, every instantiation of the compile included, so the
    /// OS's shader cache serves it; a library of the requested entries
    /// alone would be another text, which the cache has never seen.
    #[test]
    fn a_subset_of_an_artifact_loads_from_the_library_its_compile_formed() {
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device)
            .unwrap()
            .facts()
            .clone();
        let toolchain = MetalToolchain::new(device, &facts);
        let source = ProgramSource {
            text: "#include <metal_stdlib>\nusing namespace metal;\n\
                   template <uint TILE> kernel void probe(device float* x [[buffer(0)]], \
                   uint i [[thread_position_in_grid]]) { x[i] *= TILE; }\n"
                .into(),
            entries: vec![ProgramEntry::named("probe<2>"), ProgramEntry::named("probe<4>")],
        };
        let artifact = MetalSource::decode(&toolchain.compile(&source).unwrap().0.encode()).unwrap();
        let wanted = [ProgramEntry::named("probe<4>")];
        assert_eq!(toolchain.load(&artifact, &wanted).unwrap().len(), 1);
        let formed = library_text(&artifact.source);
        assert_eq!(formed, library_text(&source));
        assert_ne!(
            formed,
            library_text(&ProgramSource {
                text: source.text.clone(),
                entries: wanted.to_vec(),
            })
        );
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
        let facts = crate::profile::open_device(&device)
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
        let facts = crate::profile::open_device(&device)
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
