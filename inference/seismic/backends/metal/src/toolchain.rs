//! Metal's program toolchain.
//!
//! Compiling a program compiles its source once, with an explicit
//! instantiation for each entry that names a template instance, and forms a
//! pipeline per entry. Metal keeps no compiled bytes: the OS caches compiled
//! libraries and pipelines itself, in a shader cache this toolchain can
//! neither read nor query. What it keeps for a source text is a record of
//! the entry lists it has compiled of that text ([`compiled`]): that this
//! OS build compiled exactly this program before, for a process using the
//! same store. Whether the OS still holds what it compiled then is not
//! knowable here.

use crate::direct::DirectPipeline;
use crate::facts::{LanguageVersion, MetalFacts};
use crate::{DeviceHandle, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary};
use seismic_native_target::{
    NativeCompilationError, ProgramCache, ProgramSource, Toolchain, ToolchainIdentity,
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
            // The OS build ships the Metal compiler.
            identity: ToolchainIdentity {
                namespace: "metal",
                material: format!(
                    "metal;{};msl-{}",
                    facts.operating_system(),
                    facts.language.label()
                ),
            },
        }
    }
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

/// One line of a source text's record: the entries a compile of it formed,
/// in order, each with the group size its pipeline was formed to admit. The
/// compiled text and every pipeline are functions of the source text and
/// this list.
fn compiled(source: &ProgramSource) -> String {
    source
        .entries
        .iter()
        .map(|entry| match entry.group_size {
            Some([x, y, z]) => format!("{}@{x}x{y}x{z}", entry.symbol),
            None => entry.symbol.clone(),
        })
        .collect::<Vec<_>>()
        .join("\t")
}

/// Whether the record `kept` of a source text names the entry list `line`.
fn recorded(kept: &[u8], line: &str) -> bool {
    std::str::from_utf8(kept).is_ok_and(|record| record.lines().any(|kept| kept == line))
}

impl Toolchain for MetalToolchain {
    type Program = Vec<DirectPipeline>;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    fn keeps(&self, source: &ProgramSource, kept: &[u8]) -> bool {
        recorded(kept, &compiled(source))
    }

    fn compile(
        &self,
        source: &ProgramSource,
        cache: Option<&dyn ProgramCache>,
    ) -> Result<Vec<DirectPipeline>, NativeCompilationError> {
        let pipelines = crate::pooled(|| self.pipelines(source))?;
        if let Some(cache) = cache {
            let line = compiled(source);
            let kept = cache.get().unwrap_or_default();
            if !recorded(&kept, &line) {
                let mut record = String::from_utf8(kept).unwrap_or_default();
                record.push_str(&line);
                record.push('\n');
                cache.put(record.as_bytes());
            }
        }
        Ok(pipelines)
    }
}

impl MetalToolchain {
    /// Compile `source` and form each entry's pipeline.
    fn pipelines(
        &self,
        source: &ProgramSource,
    ) -> Result<Vec<DirectPipeline>, NativeCompilationError> {
        let device = self.device.handle().raw();
        let functions = form_functions(self.device.handle(), source, self.language)?;
        let failure =
            |error: Retained<NSError>| NativeCompilationError::ToolchainFailure(error.to_string());
        // Each symbol's pipeline is formed once. An entry's group size is the
        // one its launch declares: when register allocation left that
        // pipeline admitting fewer threads, the entry gets a pipeline formed
        // to admit them; otherwise it shares the symbol's.
        let mut formed: Vec<(&str, Retained<ProtocolObject<dyn MTLComputePipelineState>>)> =
            Vec::new();
        let mut pipelines = Vec::with_capacity(source.entries.len());
        for (entry, function) in source.entries.iter().zip(&functions) {
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

/// `source`'s library, compiled with an explicit instantiation for each
/// entry that names a template instance, and each entry's function in entry
/// order: everything a Metal program is formed from before its pipelines.
pub fn form_functions(
    device: &DeviceHandle,
    source: &ProgramSource,
    language: LanguageVersion,
) -> Result<Vec<Retained<ProtocolObject<dyn MTLFunction>>>, NativeCompilationError> {
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
    crate::pooled(|| {
        let options = crate::profile::compile_options(language);
        let library = device
            .raw()
            .newLibraryWithSource_options_error(&NSString::from_str(&text), Some(&options))
            .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
        source
            .entries
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
    crate::pooled(|| form_functions(device, &source, language).is_ok())
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
        assert_eq!(toolchain.compile(&source, None).unwrap().len(), 1);
    }

    /// Metal keeps no compiled bytes: it records each entry list it
    /// compiled of a text, and only a compile of exactly a recorded list is
    /// one the OS compiled before.
    #[test]
    fn a_compile_records_its_entry_list_and_only_that_list_is_kept() {
        struct Kept(std::cell::RefCell<Option<Vec<u8>>>);
        impl ProgramCache for Kept {
            fn get(&self) -> Option<Vec<u8>> {
                self.0.borrow().clone()
            }
            fn put(&self, bytes: &[u8]) {
                *self.0.borrow_mut() = Some(bytes.to_vec());
            }
        }
        let device = crate::test_support::metal_device();
        let facts = crate::profile::open_device(&device)
            .unwrap()
            .facts()
            .clone();
        let toolchain = MetalToolchain::new(device, &facts);
        let of = |symbols: &[&str]| ProgramSource {
            text: "#include <metal_stdlib>\nusing namespace metal;\n\
                   template <uint TILE> kernel void probe(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] *= TILE; }\n"
                .into(),
            entries: symbols.iter().copied().map(ProgramEntry::named).collect(),
        };
        let kept = Kept(std::cell::RefCell::new(None));
        let two = of(&["probe<2>", "probe<4>"]);
        toolchain.compile(&two, Some(&kept)).unwrap();
        let record = kept.get().expect("the compile was recorded");
        assert!(toolchain.keeps(&two, &record));
        // Another instantiation list of the same text is another compile.
        let one = of(&["probe<4>"]);
        assert!(!toolchain.keeps(&one, &record));
        toolchain.compile(&one, Some(&kept)).unwrap();
        let record = kept.get().unwrap();
        assert!(toolchain.keeps(&one, &record) && toolchain.keeps(&two, &record));
        // Compiling a recorded list again records nothing more.
        toolchain.compile(&two, Some(&kept)).unwrap();
        assert_eq!(kept.get().unwrap(), record);
        assert!(!toolchain.keeps(&two, b"\xff not a record"));
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
        let pipelines = toolchain.compile(&source, None).unwrap();
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
        assert_eq!(toolchain.compile(&source, None).unwrap().len(), 3);
    }
}
