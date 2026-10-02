//! Formation (§7.2): GLSL → SPIR-V with glslang, the seal pass,
//! SPIRV-Tools validation and dead-code cleanup ([`compile`]); then, for a
//! module, the `OpFmaKHR` binding of `fma` on devices with
//! `VK_KHR_shader_fma` (§16.1) and one compute pipeline per entry with its
//! workgroup size and group-memory views as specialization constants, a
//! required subgroup size of 32 where the device takes one, and the device's
//! pipeline cache ([`DirectModule::new`]).

use crate::device::Device;
use crate::facts::{SubgroupWidth, SUBGROUP_WIDTH};
use crate::seal::{self, SEAL_VERSION};
use ash::vk;
use seismic_native_target::{NativeCompilationError, ProgramEntry};
use spirv_tools::opt::Optimizer;
use spirv_tools::val::Validator;

/// The GLSL compiler this crate links (pinned in its manifest).
pub const GLSLANG_RELEASE: &str = "glslang 0.8.1+275822a";

/// Everything besides the source that determines a formed module: the
/// compiler, target, seal passes and the device facts they and the driver
/// consume (§7.5).
pub(crate) fn formation(device: &Device) -> String {
    format!(
        "vulkan;{GLSLANG_RELEASE};spirv 1.6;seal {SEAL_VERSION};{}",
        device.facts().formation_identity()
    )
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
    let options = glslang::CompilerOptions {
        source_language: glslang::SourceLanguage::GLSL,
        target: glslang::Target::Vulkan {
            version: glslang::VulkanVersion::Vulkan1_3,
            spirv_version: glslang::SpirvVersion::SPIRV1_6,
        },
        version_profile: None,
        messages: glslang::ShaderMessage::DEFAULT,
    };
    let defines = [("SEISMIC_KERNEL", Some(kernel))];
    let input = glslang::ShaderInput::new(
        &text,
        glslang::ShaderStage::Compute,
        &options,
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
}

impl DirectModule {
    /// The pipelines of `entries`, each from its sealed SPIR-V, with its
    /// constants: the workgroup size (ids 0..3), then the rest (ids 3..). A
    /// pipeline-creation failure is a toolchain failure.
    pub(crate) fn new(
        device: &Device,
        entries: &[(&[u32], &ProgramEntry)],
    ) -> Result<Self, NativeCompilationError> {
        let mut module = Self {
            device: device.clone(),
            pipelines: Vec::with_capacity(entries.len()),
        };
        for (words, entry) in entries {
            // The `OpFmaKHR` binding follows validation, which the linked
            // validator does not know.
            let pipeline = if device.facts().shader_fma.float32 {
                let bound = seal::fma_khr(words)
                    .map_err(|error| NativeCompilationError::MalformedToolchainOutput(error.0))?;
                module.pipeline(&bound, entry)?
            } else {
                module.pipeline(words, entry)?
            };
            module.pipelines.push(pipeline);
        }
        Ok(module)
    }

    fn pipeline(
        &self,
        words: &[u32],
        entry: &ProgramEntry,
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
        let flags = if threads[0] % width.lanes() == 0 {
            vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
        } else {
            vk::PipelineShaderStageCreateFlags::empty()
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .flags(flags)
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
            .stage(stage)
            .layout(inner.layout);
        let created = unsafe { device.create_compute_pipelines(inner.cache, &[info], None) };
        unsafe { device.destroy_shader_module(shader, None) };
        created.map(|pipelines| pipelines[0]).map_err(|(_, error)| {
            toolchain(format!(
                "vkCreateComputePipelines of `{}` failed: {error}",
                entry.symbol
            ))
        })
    }

    pub(crate) fn pipeline_handle(&self, launch: usize) -> vk::Pipeline {
        self.pipelines[launch]
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
