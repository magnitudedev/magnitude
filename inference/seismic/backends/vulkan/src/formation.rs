//! Formation (§7.2): GLSL → SPIR-V with glslang, the seal pass,
//! SPIRV-Tools validation and dead-code cleanup ([`compile`]); then, for a
//! module, the `OpFmaKHR` binding of `fma` on devices with
//! `VK_KHR_shader_fma` (§16.1) and one compute pipeline per entry with its
//! workgroup size and group-memory views as specialization constants, a
//! required subgroup size of 32 where the device takes one, and a pipeline
//! cache of its own, whose data is the program's artifact ([`DirectModule::compile`]).

use crate::device::Device;
use crate::facts::{SubgroupWidth, SUBGROUP_WIDTH};
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
/// `environment`, validated and cleaned.
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
    clean(&sealed)
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
    /// [`DirectModule::load`] creates them from without compiling. That
    /// the data serves such a load is not proven here: the toolchain proves
    /// it for every compile, before its artifact is stored. A
    /// pipeline-creation failure is a toolchain failure. Every pipeline's
    /// statistics are read from the driver, here and on a load alike, so
    /// the artifact carries none.
    pub(crate) fn compile(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
    ) -> Result<(Self, Vec<u8>), NativeCompilationError> {
        let (mut module, data) = Self::created(device, entries, &[], vk::PipelineCreateFlags::empty())?;
        module.read_statistics()?;
        Ok((module, data))
    }

    /// The pipelines of `entries` from `pipelines`, the data
    /// [`DirectModule::compile`] made of them on this device and driver.
    /// Nothing is compiled: a pipeline the data does not hold fails
    /// (`FAIL_ON_PIPELINE_COMPILE_REQUIRED`).
    pub(crate) fn load(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        pipelines: &[u8],
    ) -> Result<Self, NativeCompilationError> {
        let mut module = Self::loaded(device, entries, pipelines)?;
        module.read_statistics()?;
        Ok(module)
    }

    /// Whether `pipelines` serves a load of `entries` without compiling:
    /// the load's pipelines are created and destroyed, and their
    /// statistics, which the compile that made the data read, are not
    /// read again.
    pub(crate) fn serves(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        pipelines: &[u8],
    ) -> Result<(), NativeCompilationError> {
        Self::loaded(device, entries, pipelines).map(drop)
    }

    /// The pipelines of `entries` from `pipelines`, their statistics not
    /// yet read.
    fn loaded(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        pipelines: &[u8],
    ) -> Result<Self, NativeCompilationError> {
        Self::created(
            device,
            entries,
            pipelines,
            vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED,
        )
        .map(|(module, _)| module)
    }

    /// The pipelines of `entries`, created in a cache that starts as
    /// `kept`, and what the cache then holds.
    fn created(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
        kept: &[u8],
        flags: vk::PipelineCreateFlags,
    ) -> Result<(Self, Vec<u8>), NativeCompilationError> {
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
        let created = module.create(cache, entries, flags);
        let data = created.and_then(|()| {
            unsafe { vulkan.get_pipeline_cache_data(cache) }
                .map_err(|error| toolchain(format!("vkGetPipelineCacheData failed: {error}")))
        });
        unsafe { vulkan.destroy_pipeline_cache(cache, None) };
        Ok((module, data?))
    }

    fn create(
        &mut self,
        cache: vk::PipelineCache,
        entries: &[(&[u32], &ProgramEntry)],
        flags: vk::PipelineCreateFlags,
    ) -> Result<(), NativeCompilationError> {
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
            self.pipelines.push(pipeline);
        }
        Ok(())
    }

    fn pipeline(
        &self,
        cache: vk::PipelineCache,
        words: &[u32],
        entry: &ProgramEntry,
        flags: vk::PipelineCreateFlags,
    ) -> Result<vk::Pipeline, NativeCompilationError> {
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
        let width = self.device.facts().subgroup_width();
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
        let stage = if width == SubgroupWidth::Required32 {
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
            Ok(pipelines) if pipelines[0] != vk::Pipeline::null() => Ok(pipelines[0]),
            Ok(_) | Err((_, vk::Result::PIPELINE_COMPILE_REQUIRED)) => Err(toolchain(format!(
                "the kept pipeline of `{}` does not serve it without compiling",
                entry.symbol
            ))),
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
