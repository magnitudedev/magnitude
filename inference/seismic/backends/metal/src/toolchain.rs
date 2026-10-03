//! Metal's program toolchain.
//!
//! Compiling a program compiles its source once, with an explicit
//! instantiation for each entry that names a template instance, and forms a
//! pipeline per entry. Metal keeps no artifacts: the OS caches compiled
//! libraries itself.

use crate::direct::DirectPipeline;
use crate::facts::MetalFacts;
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
}

impl MetalToolchain {
    pub fn new(device: MetalDevice, facts: &MetalFacts) -> Self {
        Self {
            device,
            // The OS build ships the Metal compiler.
            identity: ToolchainIdentity {
                namespace: "metal",
                material: format!("metal;{}", facts.operating_system()),
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

impl Toolchain for MetalToolchain {
    type Program = Vec<DirectPipeline>;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    fn compile(
        &self,
        source: &ProgramSource,
        _cache: Option<&dyn ProgramCache>,
    ) -> Result<Vec<DirectPipeline>, NativeCompilationError> {
        let device = self.device.handle().raw();
        let functions = form_functions(self.device.handle(), source)?;
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
    let options = objc2_metal::MTLCompileOptions::new();
    options.setMathMode(objc2_metal::MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(objc2_metal::MTLMathFloatingPointFunctions::Precise);
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
}

/// Whether this host's Metal compiler forms a source that includes the
/// tensor-operation headers (Metal 4), whatever the GPU executes: what
/// forming a `SEISMIC_HAS_TENSOR_OPS` source needs.
pub fn forms_tensor_operations(device: &DeviceHandle) -> bool {
    let source = ProgramSource {
        text: "#include <metal_stdlib>\n#include <metal_tensor>\n\
               #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n\
               kernel void seismic_tensor_headers(device float* out [[buffer(0)]]) { out[0] = 0.0f; }\n"
            .to_owned(),
        entries: vec![seismic_native_target::ProgramEntry::named("seismic_tensor_headers")],
    };
    form_functions(device, &source).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_native_target::ProgramEntry;

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
