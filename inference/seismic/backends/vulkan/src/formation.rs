//! Formation (§7.2): GLSL → SPIR-V with glslang, the seal pass,
//! SPIRV-Tools validation and dead-code cleanup ([`compile`]); then, for a
//! module, the `OpFmaKHR` binding of `fma` on devices with
//! `VK_KHR_shader_fma` (§16.1) and one compute pipeline per entry with its
//! workgroup size and group-memory views as specialization constants, a
//! required subgroup size of 32 where the device takes one, and a pipeline
//! cache of its own, whose data is the program's artifact ([`DirectModule::compile`]).

use crate::device::Device;
use crate::facts::SUBGROUP_WIDTH;
use crate::seal::{self, SEAL_VERSION};
use ash::vk;
use seismic_native_target::{NativeCompilationError, ProgramEntry, ProgramStatistic, StatisticValue};
use spirv_tools::opt::Optimizer;
use spirv_tools::val::Validator;

/// The GLSL compiler this crate links (pinned in its manifest).
pub const GLSLANG_RELEASE: &str = "glslang 0.8.1+275822a";

/// Everything besides the source that determines a formed module: the
/// compiler, target, seal passes and the device facts they and the driver
/// consume (§7.5), and what every pipeline on the device is created with
/// ([`created_with`]).
pub(crate) fn formation(device: &Device) -> String {
    format!(
        "{};{};create {}",
        compiler(),
        device.facts().formation_identity(device.properties()),
        created_with(device).as_raw()
    )
}

/// What every pipeline on `device` is created with besides what its
/// operation adds: statistics capture, where the device reports statistics.
/// A driver may key a pipeline's cache data by it, so a load creates its
/// pipelines with it as their compile did, and it is part of the formation
/// identity, so no load meets data created without it.
fn created_with(device: &Device) -> vk::PipelineCreateFlags {
    if device.inner.executables.is_some() {
        vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
    } else {
        vk::PipelineCreateFlags::empty()
    }
}

/// The part of every formation identity the compiler determines: glslang,
/// the SPIR-V target and the seal passes. Offline formation, which forms no
/// pipeline on a device, is identified by this and the formation facts.
pub fn compiler() -> String {
    format!("vulkan;{GLSLANG_RELEASE};spirv 1.6;seal {SEAL_VERSION}")
}

fn toolchain(error: impl std::fmt::Display) -> NativeCompilationError {
    NativeCompilationError::ToolchainFailure(error.to_string())
}

/// Compile the launch `kernel` of `source` to SPIR-V sealed for
/// `environment`, validated and cleaned. A module holding a construct the
/// emitted contract forbids ([`seal::contract`]) is refused, as glslang
/// refuses invalid source.
pub fn compile(
    source: &str,
    kernel: &str,
    environment: seal::Environment,
) -> Result<Vec<u32>, NativeCompilationError> {
    let compiler = glslang::Compiler::acquire().ok_or_else(|| {
        NativeCompilationError::ToolchainFailure("glslang could not be initialized".into())
    })?;
    let text = glslang::ShaderSource::from(source);
    let defines = [("SEISMIC_KERNEL", Some(kernel))];
    let input = glslang::ShaderInput::new(
        &text,
        glslang::ShaderStage::Compute,
        &options(),
        Some(&defines[..]),
        None,
    )
    .map_err(toolchain)?;
    let shader = glslang::Shader::new(compiler, input).map_err(toolchain)?;
    let compiled = shader.compile().map_err(toolchain)?;
    let sealed = seal::seal(&compiled, environment)
        .map_err(|error| NativeCompilationError::MalformedToolchainOutput(error.0))?;
    validate(&sealed).map_err(NativeCompilationError::MalformedToolchainOutput)?;
    let cleaned = clean(&sealed)?;
    let violations = seal::contract(&cleaned)
        .map_err(|error| NativeCompilationError::MalformedToolchainOutput(error.0))?;
    if !violations.is_empty() {
        // Written as glslang writes a diagnostic, so readers that keep a
        // toolchain failure's error lines keep it.
        return Err(toolchain(format!(
            "ERROR: `{kernel}` violates the Vulkan kernel contract: {}",
            violations
                .iter()
                .map(|violation| violation.0.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }
    Ok(cleaned)
}

/// The glslang options of every formation.
fn options() -> glslang::CompilerOptions {
    glslang::CompilerOptions {
        source_language: glslang::SourceLanguage::GLSL,
        target: glslang::Target::Vulkan {
            version: glslang::VulkanVersion::Vulkan1_3,
            spirv_version: glslang::SpirvVersion::SPIRV1_6,
        },
        version_profile: None,
        messages: glslang::ShaderMessage::DEFAULT,
    }
}

/// glslang's preprocessed text of `source` for the launch `kernel`, as
/// [`compile`] preprocesses it. glslang parses what it preprocesses, so
/// `source` must parse as a compute shader; a preprocessing failure (an
/// active `#error` among them) is a toolchain failure.
pub fn preprocess(source: &str, kernel: &str) -> Result<String, NativeCompilationError> {
    let compiler = glslang::Compiler::acquire().ok_or_else(|| {
        NativeCompilationError::ToolchainFailure("glslang could not be initialized".into())
    })?;
    let text = glslang::ShaderSource::from(source);
    let defines = [("SEISMIC_KERNEL", Some(kernel))];
    let input = glslang::ShaderInput::new(
        &text,
        glslang::ShaderStage::Compute,
        &options(),
        Some(&defines[..]),
        None,
    )
    .map_err(toolchain)?;
    let shader = glslang::Shader::new(compiler, input).map_err(toolchain)?;
    Ok(shader.get_preprocessed_code())
}

fn target() -> spirv_tools::TargetEnv {
    spirv_tools::TargetEnv::Vulkan_1_3
}

fn validator_options() -> spirv_tools::val::ValidatorOptions {
    spirv_tools::val::ValidatorOptions {
        scalar_block_layout: true,
        ..Default::default()
    }
}

/// SPIRV-Tools validation for the Vulkan 1.3 environment.
pub fn validate(module: &[u32]) -> Result<(), String> {
    spirv_tools::val::create(Some(target()))
        .validate(module, Some(validator_options()))
        .map_err(|error| error.to_string())
}

/// Dead-code cleanup only: the driver compiler does all optimization, and
/// the performance passes would rewrite float math.
fn clean(module: &[u32]) -> Result<Vec<u32>, NativeCompilationError> {
    let mut optimizer = spirv_tools::opt::create(Some(target()));
    optimizer
        .register_pass(spirv_tools::opt::Passes::AggressiveDCE)
        .register_pass(spirv_tools::opt::Passes::DeadVariableElimination)
        .register_pass(spirv_tools::opt::Passes::RemoveUnusedInterfaceVariables)
        .register_pass(spirv_tools::opt::Passes::EliminateDeadFunctions);
    let mut messages = Vec::new();
    let options = spirv_tools::opt::Options {
        validator_options: Some(validator_options()),
        preserve_spec_constants: true,
        ..Default::default()
    };
    optimizer
        .optimize(
            module,
            &mut |message: spirv_tools::error::Message| messages.push(message.message),
            Some(options),
        )
        .map(|binary| binary.as_words().to_vec())
        .map_err(|error| {
            NativeCompilationError::MalformedToolchainOutput(format!(
                "{error}: {}",
                messages.join("; ")
            ))
        })
}

/// The pipelines of one specialization, one per launch in order.
pub struct DirectModule {
    device: Device,
    pipelines: Vec<vk::Pipeline>,
    /// Each pipeline's executables' statistics, as the driver names them;
    /// empty where the device reports none (`Facts::pipeline_statistics`).
    /// Read once every pipeline is held ([`DirectModule::read_statistics`]).
    statistics: Vec<Vec<ProgramStatistic>>,
}

impl DirectModule {
    /// The pipelines of `entries`, each from its sealed SPIR-V with its
    /// constants (the workgroup size, ids 0..3, then the rest), as the
    /// driver compiled them, and their pipeline cache data, which
    /// [`DirectModule::load`] may create them from without compiling. A
    /// pipeline-creation failure is a toolchain failure. Every pipeline's
    /// statistics are read from the driver, here and on a load alike, so
    /// the artifact carries none.
    pub(crate) fn compile(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
    ) -> Result<(Self, Vec<u8>), NativeCompilationError> {
        let (mut module, data) = Self::created(device, entries, &[], vk::PipelineCreateFlags::empty())?
            .expect("a creation that may compile never reports that it would have to");
        module.read_statistics()?;
        Ok((module, data))
    }

    /// The pipelines of `entries` from `pipelines`, the data
    /// [`DirectModule::compile`] made of them on this device and driver,
    /// or `None` where the driver does not serve one of them from it
    /// without compiling (`FAIL_ON_PIPELINE_COMPILE_REQUIRED`): the
    /// pipeline cache is a hint the driver may decline, so a declined load
    /// is a miss, never a failure.
    pub(crate) fn load(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        pipelines: &[u8],
    ) -> Result<Option<Self>, NativeCompilationError> {
        let Some((mut module, _)) = Self::created(
            device,
            entries,
            pipelines,
            vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED,
        )?
        else {
            return Ok(None);
        };
        module.read_statistics()?;
        Ok(Some(module))
    }

    /// The pipelines of `entries`, created in a cache that starts as
    /// `kept`, and what the cache then holds; `None` where `flags` forbid
    /// compiling and a pipeline would need it.
    fn created(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        kept: &[u8],
        flags: vk::PipelineCreateFlags,
    ) -> Result<Option<(Self, Vec<u8>)>, NativeCompilationError> {
        let mut module = Self {
            device: device.clone(),
            pipelines: Vec::with_capacity(entries.len()),
            statistics: Vec::new(),
        };
        let vulkan = &device.inner.device;
        let cache = unsafe {
            vulkan.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(kept),
                None,
            )
        }
        .map_err(|error| toolchain(format!("vkCreatePipelineCache failed: {error}")))?;
        let created = module.create(cache, entries, flags).and_then(|served| {
            served
                .then(|| unsafe { vulkan.get_pipeline_cache_data(cache) })
                .transpose()
                .map_err(|error| toolchain(format!("vkGetPipelineCacheData failed: {error}")))
        });
        unsafe { vulkan.destroy_pipeline_cache(cache, None) };
        Ok(created?.map(|data| (module, data)))
    }

    /// Create every pipeline of `entries` in order; `false`, with the
    /// pipelines created so far held for the module to destroy, at the
    /// first that `flags` forbid compiling and that would need it.
    fn create(
        &mut self,
        cache: vk::PipelineCache,
        entries: &[(&[u32], &ProgramEntry)],
        flags: vk::PipelineCreateFlags,
    ) -> Result<bool, NativeCompilationError> {
        let device = self.device.clone();
        for (words, entry) in entries {
            // The `OpFmaKHR` binding follows validation, which the linked
            // validator does not know.
            let pipeline = if device.properties().fma_32.holds() {
                let bound = seal::fma_khr(words)
                    .map_err(|error| NativeCompilationError::MalformedToolchainOutput(error.0))?;
                self.pipeline(cache, &bound, entry, flags)?
            } else {
                self.pipeline(cache, words, entry, flags)?
            };
            let Some(pipeline) = pipeline else {
                return Ok(false);
            };
            self.pipelines.push(pipeline);
        }
        Ok(true)
    }

    fn pipeline(
        &self,
        cache: vk::PipelineCache,
        words: &[u32],
        entry: &ProgramEntry,
        flags: vk::PipelineCreateFlags,
    ) -> Result<Option<vk::Pipeline>, NativeCompilationError> {
        let threads = entry.group_size.ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(format!(
                "`{}` carries no workgroup size",
                entry.symbol
            ))
        })?;
        let inner = &self.device.inner;
        let device = &inner.device;
        let shader = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
        }
        .map_err(|error| {
            toolchain(format!(
                "vkCreateShaderModule of `{}` failed: {error}",
                entry.symbol
            ))
        })?;
        let values = threads.iter().chain(&entry.constants).collect::<Vec<_>>();
        let entries = (0..values.len() as u32)
            .map(|id| vk::SpecializationMapEntry {
                constant_id: id,
                offset: id * 4,
                size: 4,
            })
            .collect::<Vec<_>>();
        let data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(SUBGROUP_WIDTH);
        // Full subgroups are required whenever the X extent allows it (a
        // multiple of the hardware subgroup); smaller groups run one partial
        // subgroup, as a partial CUDA warp does.
        let lanes = self.device.properties().subgroup_width.get();
        let subgroups = if u64::from(threads[0]) % lanes == 0 {
            vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
        } else {
            vk::PipelineShaderStageCreateFlags::empty()
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .flags(subgroups)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main")
            .specialization_info(&specialization);
        let stage = if self.device.facts().requires_subgroup_size() {
            stage.push_next(&mut subgroup)
        } else {
            stage
        };
        let info = vk::ComputePipelineCreateInfo::default()
            .flags(flags | created_with(&self.device))
            .stage(stage)
            .layout(inner.layout);
        let created = {
            let _creating = crate::device::creating_pipelines();
            unsafe { device.create_compute_pipelines(cache, &[info], None) }
        };
        unsafe { device.destroy_shader_module(shader, None) };
        // `PIPELINE_COMPILE_REQUIRED` is a success code: the pipeline
        // returned with it is null, and nothing was compiled.
        match created {
            Ok(pipelines) if pipelines[0] != vk::Pipeline::null() => Ok(Some(pipelines[0])),
            Ok(_) | Err((_, vk::Result::PIPELINE_COMPILE_REQUIRED)) => Ok(None),
            Err((_, error)) => Err(toolchain(format!(
                "vkCreateComputePipelines of `{}` failed: {error}",
                entry.symbol
            ))),
        }
    }

    pub(crate) fn pipeline_handle(&self, launch: usize) -> vk::Pipeline {
        self.pipelines[launch]
    }

    /// The statistics of launch `launch`'s pipeline, as the driver reports
    /// them.
    pub fn statistics(&self, launch: usize) -> &[ProgramStatistic] {
        &self.statistics[launch]
    }

    /// Read every pipeline's statistics. The module holds its pipelines
    /// first, so a failure to read them destroys them with it.
    fn read_statistics(&mut self) -> Result<(), NativeCompilationError> {
        self.statistics = self
            .pipelines
            .iter()
            .map(|pipeline| self.statistics_of(*pipeline))
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    /// Every statistic of every executable of `pipeline`, named and
    /// described as the driver reports it (an executable's name prefixes
    /// the statistic's where the pipeline has more than one); none where
    /// the device reports no statistics.
    ///
    /// The NVIDIA proprietary driver (595, RTX 3090) reports "Register
    /// Count", "Binary Size", "Stack Size", "Local Memory Size" and "Shared
    /// Memory Size". Two of them are not values: "Local Memory Size" reads
    /// 2^36 for every pipeline (a sentinel) and "Stack Size" reads 0 for
    /// every pipeline. Its spill signal is "Register Count" at the 255
    /// ceiling: the projection GEMM's 64x64 sub-tiles reach it, with binaries
    /// 2-3x larger, and run 6-7x slower than the best tile.
    fn statistics_of(&self, pipeline: vk::Pipeline) -> Result<Vec<ProgramStatistic>, NativeCompilationError> {
        let Some(executables) = &self.device.inner.executables else {
            return Ok(Vec::new());
        };
        let failed = |error| toolchain(format!("pipeline executable statistics: {error}"));
        let text = |raw: Result<&std::ffi::CStr, _>| {
            raw.expect("the driver null-terminates a pipeline executable's text").to_string_lossy().into_owned()
        };
        let properties = unsafe {
            executables.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipeline))
        }
        .map_err(failed)?;
        let several = properties.len() > 1;
        let mut statistics = Vec::new();
        for (index, executable) in properties.iter().enumerate() {
            let prefix = if several { format!("{}: ", text(executable.name_as_c_str())) } else { String::new() };
            let index = u32::try_from(index).expect("a pipeline's executables are counted in u32");
            let info = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(index);
            for statistic in unsafe { executables.get_pipeline_executable_statistics(&info) }.map_err(failed)? {
                // SAFETY: `format` names the union's member the driver wrote.
                let value = unsafe {
                    match statistic.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => StatisticValue::Bool(statistic.value.b32 == vk::TRUE),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => StatisticValue::Int(statistic.value.i64),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => StatisticValue::Uint(statistic.value.u64),
                        vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => StatisticValue::Float(statistic.value.f64),
                        other => panic!("the driver reported pipeline statistic format {other:?}, which the extension does not define"),
                    }
                };
                statistics.push(ProgramStatistic {
                    name: format!("{prefix}{}", text(statistic.name_as_c_str())),
                    description: text(statistic.description_as_c_str()),
                    value,
                });
            }
        }
        Ok(statistics)
    }
}

impl Drop for DirectModule {
    fn drop(&mut self) {
        // Submissions that use the pipelines keep the module alive until
        // they complete.
        for pipeline in &self.pipelines {
            unsafe { self.device.inner.device.destroy_pipeline(*pipeline, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `body` as the launch `probe` of a module with one u64 buffer.
    fn form(globals: &str, body: &str) -> Result<Vec<u32>, NativeCompilationError> {
        let source = format!(
            "#version 460\n\
             #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n\
             #extension GL_KHR_cooperative_matrix : require\n\
             #extension GL_KHR_memory_scope_semantics : require\n\
             layout(local_size_x = 32) in;\n\
             layout(set = 0, binding = 0, std430) buffer Output {{ uint64_t x[]; }} data;\n\
             layout(set = 0, binding = 1, std430) buffer Floats {{ float x[]; }} floats;\n\
             {globals}\n\
             void probe() {{ {body} }}\n\
             void main() {{ SEISMIC_KERNEL(); }}\n"
        );
        let environment = seal::Environment {
            float16: false,
            rounding_rte_32: true,
            denorm_preserve_32: false,
        };
        compile(&source, "probe", environment)
    }

    fn refused(globals: &str, body: &str, construct: &str) {
        match form(globals, body) {
            Err(NativeCompilationError::ToolchainFailure(message)) => assert!(
                message.starts_with("ERROR: `probe` violates") && message.contains(construct),
                "{body}: {message}"
            ),
            other => panic!("{body}: expected a contract violation, got {other:?}"),
        }
    }

    fn formed(globals: &str, body: &str) {
        if let Err(error) = form(globals, body) {
            panic!("{body}: {error}");
        }
    }

    const OUT: &str = "data.x[gl_GlobalInvocationID.x]";

    #[test]
    fn a_64_bit_division_is_refused_wherever_it_runs() {
        refused(
            "",
            &format!("{OUT} = data.x[0] / data.x[1];"),
            "64-bit OpUDiv",
        );
        // A literal divisor still divides at run time.
        refused(
            "",
            &format!("{OUT} = data.x[0] / uint64_t(3);"),
            "64-bit OpUDiv",
        );
        refused(
            "",
            &format!("{OUT} = uint64_t(int64_t(data.x[0]) % int64_t(data.x[1]));"),
            "64-bit OpSMod",
        );
        refused(
            "",
            &format!(
                "u64vec2 a = u64vec2(data.x[0], data.x[1]), b = u64vec2(data.x[2], data.x[3]); {OUT} = (a / b).x;"
            ),
            "64-bit OpUDiv",
        );
    }

    #[test]
    fn folded_and_32_bit_division_form() {
        formed("", &format!("{OUT} = uint64_t(123) / uint64_t(3);"));
        formed(
            "",
            &format!("{OUT} = uint64_t(uint(data.x[0]) / uint(data.x[1]) + uint(data.x[2]) % 7u);"),
        );
    }

    #[test]
    fn a_constant_table_in_per_invocation_storage_is_refused() {
        let table = "const uint64_t t[4] = uint64_t[4](3, 5, 7, 11);";
        refused(
            "",
            &format!("{table} {OUT} = t[uint(data.x[0]) & 3u];"),
            "constant table",
        );
        let large = (0..256)
            .map(|value| (value * value + 3).to_string())
            .collect::<Vec<_>>();
        refused(
            "",
            &format!(
                "const uint64_t t[256] = uint64_t[256]({}); {OUT} = t[uint(data.x[0]) & 255u];",
                large.join(", ")
            ),
            "constant table",
        );
        refused(
            "",
            &format!(
                "const uint64_t t[2][2] = uint64_t[2][2](uint64_t[2](3, 5), uint64_t[2](7, 11)); uint i = uint(data.x[0]); {OUT} = t[i & 1u][(i >> 1) & 1u];"
            ),
            "constant table",
        );
        // Rows equal to each other are still tables.
        refused(
            "",
            &format!(
                "const uint64_t t[2][2] = uint64_t[2][2](uint64_t[2](3, 5), uint64_t[2](3, 5)); uint i = uint(data.x[0]); {OUT} = t[i & 1u][(i >> 1) & 1u];"
            ),
            "constant table",
        );
        refused(
            "",
            &format!("{table} uint64_t u[4] = t; {OUT} = u[uint(data.x[0]) & 3u];"),
            "constant table",
        );
        // A table passed to a function is copied into its parameter.
        refused(
            "const uint64_t T[4] = uint64_t[4](3, 5, 7, 11);\n\
             uint64_t pick(const uint64_t t[4], uint i) { return t[i]; }",
            &format!("{OUT} = pick(T, uint(data.x[0]) & 3u);"),
            "constant table",
        );
    }

    #[test]
    fn working_state_and_vectors_form() {
        formed(
            "",
            &format!("uint64_t t[4] = uint64_t[4](0, 0, 0, 0); t[uint(data.x[0]) & 3u] = data.x[1]; {OUT} = t[uint(data.x[2]) & 3u];"),
        );
        formed(
            "",
            &format!("uint64_t t[4] = uint64_t[4](7, 7, 7, 7); t[uint(data.x[1]) & 3u] = data.x[2]; {OUT} = t[uint(data.x[0]) & 3u];"),
        );
        formed(
            "",
            &format!("float t[8]; for (uint i = 0u; i < 8u; i++) t[i] = 0.0; t[uint(data.x[1]) & 7u] = 1.0; {OUT} = uint64_t(uint(t[uint(data.x[0]) & 7u]));"),
        );
        formed(
            "",
            &format!("uvec4 t = uvec4(0); t[uint(data.x[0]) & 3u] = uint(data.x[1]); {OUT} = t[uint(data.x[2]) & 3u];"),
        );
        formed(
            "",
            &format!("u64vec4 t = u64vec4(3, 5, 7, 11); {OUT} = t[uint(data.x[0]) & 3u];"),
        );
        // Constant indexing folds the table away.
        formed(
            "",
            &format!("const uint64_t t[4] = uint64_t[4](3, 5, 7, 11); {OUT} = t[2] + data.x[0];"),
        );
    }

    /// A splat whose equal elements are distinct constants is still a splat:
    /// equality is of values, not ids.
    #[test]
    fn splats_compare_by_value() {
        let module = form(
            "",
            &format!("uint64_t t[4] = uint64_t[4](7, 7, 7, 7); t[uint(data.x[1]) & 3u] = data.x[2]; {OUT} = t[uint(data.x[0]) & 3u];"),
        )
        .expect("a splat forms");
        // Give the composite's first element a fresh constant of equal value.
        let mut at = 5;
        let (mut seven, mut composite) = (None, None);
        while at < module.len() {
            let (words, opcode) = ((module[at] >> 16) as usize, module[at] & 0xffff);
            if opcode == 43 && module[at + 3] == 7 && words == 5 {
                seven = Some(at);
            }
            if opcode == 44 && seven.is_some_and(|seven| module[at + 3] == module[seven + 2]) {
                composite = Some(at);
            }
            at += words;
        }
        let (seven, composite) = (
            seven.expect("the constant 7"),
            composite.expect("the splat"),
        );
        let fresh = module[3];
        let mut patched = module.clone();
        patched[3] += 1;
        patched[composite + 3 + 1] = fresh;
        let copy = [
            module[seven],
            module[seven + 1],
            fresh,
            module[seven + 3],
            module[seven + 4],
        ];
        let after = seven + 5;
        patched.splice(after..after, copy);
        assert_eq!(seal::contract(&patched), Ok(vec![]));
        // An unequal element makes it a table.
        let mut unequal = patched.clone();
        unequal[after + 3] = 8;
        assert_eq!(seal::contract(&unequal).expect("parses").len(), 1);
    }

    #[test]
    fn a_float_converted_to_a_64_bit_integer_is_refused() {
        let value = "uintBitsToFloat(uint(data.x[0]))";
        refused("", &format!("{OUT} = uint64_t({value});"), "64-bit OpConvertFToU");
        refused(
            "",
            &format!("{OUT} = uint64_t(int64_t({value}));"),
            "64-bit OpConvertFToS",
        );
        refused(
            "",
            &format!("{OUT} = u64vec2({value}, 1.0).x;"),
            "64-bit OpConvertFToU",
        );
        formed("", &format!("{OUT} = uint64_t(uint({value}));"));
        formed("", &format!("{OUT} = uint64_t(int({value}));"));
    }

    const ACCUMULATOR: &str = "coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>";

    #[test]
    fn a_cooperative_matrix_constant_two_functions_read_is_refused() {
        let store = "coopMatStore(c, floats.x, 0, 16, gl_CooperativeMatrixLayoutRowMajor);";
        refused(
            &format!(
                "{ACCUMULATOR} zero_a() {{ return {ACCUMULATOR}(0.0); }}\n\
                 {ACCUMULATOR} zero_b() {{ return {ACCUMULATOR}(0.0); }}"
            ),
            &format!("{ACCUMULATOR} c = data.x[0] == 0ul ? zero_a() : zero_b(); {store}"),
            "cooperative-matrix constant",
        );
        // One function holds the constant; its callers read its result.
        formed(
            &format!(
                "{ACCUMULATOR} zero() {{ return {ACCUMULATOR}(0.0); }}\n\
                 {ACCUMULATOR} cleared() {{ return zero(); }}"
            ),
            &format!("{ACCUMULATOR} c = data.x[0] == 0ul ? zero() : cleared(); {store}"),
        );
    }
}
