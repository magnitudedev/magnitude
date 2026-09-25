//! Seismic's Vulkan backend: the direct native route only
//! (`specs/26-09-24/vulkan-native-backend.md`). Authored GLSL compute
//! kernels are formed to SPIR-V with the statically linked glslang, sealed
//! for Seismic's numerical environment, validated with SPIRV-Tools and run on
//! one queue of the opened device through ash.
//!
//! There is no planned route: this crate implements no compiler target.

mod device;
pub mod direct;
mod facts;
pub mod formation;
mod instance;
mod memory;
mod probe;
pub mod seal;

pub use device::{Device, MemoryBudget, OpenError, Rounded};
pub use facts::{Calibration, Description, DeviceType, Facts, Limits, SUBGROUP_WIDTH};
pub use instance::LoaderError;
pub use memory::Buffer;

/// Enumerate the physical devices: no logical device is created. GPUs come
/// first, so the first device of the backend is a GPU when there is one.
pub fn discover() -> Result<Vec<Description>, LoaderError> {
    let instance = instance::instance()?;
    let mut descriptions = instance
        .physical_devices()?
        .into_iter()
        .map(|physical| facts::describe(instance, physical))
        .collect::<Vec<_>>();
    descriptions.sort_by_key(|description| !description.facts.is_gpu());
    Ok(descriptions)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every device meeting the discovery floor opens, or fails only the
    /// multiply-add probe (lavapipe); storage round-trips through host
    /// writes and reads.
    #[test]
    fn devices_meeting_the_floor_open_and_round_trip_storage() {
        let devices = match discover() {
            Ok(devices) => devices,
            Err(error) => {
                eprintln!("no Vulkan loader: {error}");
                return;
            }
        };
        for description in devices {
            eprintln!(
                "{} ({}, {}): floor {:?}, matrix {}, denorm-preserve {}, rte32 {}, shader fma {:?}, uuid {}",
                description.facts.name,
                description.facts.driver_name,
                description.facts.driver_info,
                description.floor,
                description.facts.matrix,
                description.facts.denorm_preserve_32,
                description.facts.rounding_rte_32,
                description.facts.shader_fma,
                facts::hex(&description.facts.uuid)
            );
            if description.floor.is_err() {
                continue;
            }
            let device = match Device::open(description.facts.uuid) {
                Ok(device) => device,
                Err(error @ (OpenError::MultiplyAdd { .. } | OpenError::Rounding { .. })) => {
                    eprintln!("  {error}");
                    continue;
                }
                Err(error) => panic!("a device meeting the floor opens: {error}"),
            };
            let buffer = device.allocate(1 << 20, 256).expect("allocation");
            let bytes = (0..1u32 << 18)
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            device.write(&buffer, 0, &bytes).expect("host write");
            let mut back = vec![0u8; bytes.len()];
            device.read(&buffer, 0, &mut back).expect("host read");
            assert_eq!(back, bytes);
            let budget = device.memory_budget();
            assert!(budget.heap_budget_bytes > 0);
            // Formed `fma` (OpFmaKHR where bound) is fused beside a sealed
            // `a*b+c`, which is rounded twice (§7.3, §13.1).
            probe::multiply_add(&device).expect("fma is fused and a*b+c is not contracted");
            // fp32 rounds to nearest even whether RTE 32 is declared or
            // probed at open.
            probe::rounding(&device)
                .expect("fp32 add, multiply and conversion round to nearest even");
        }
    }

    #[test]
    fn sealing_decorates_every_float_operation_and_sets_the_environment() {
        let source = "#version 460\nlayout(local_size_x = 1) in;\nlayout(std430, binding = 0) buffer b { float v[]; };\nvoid main() { v[3] = v[0] * v[1] + v[2] - v[0] / v[2]; v[4] = fma(v[0], v[1], v[2]); }\n";
        let compiler = glslang::Compiler::acquire().expect("glslang");
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
        let input = glslang::ShaderInput::new::<(&str, Option<&str>)>(
            &text,
            glslang::ShaderStage::Compute,
            &options,
            None,
            None,
        )
        .expect("input");
        let compiled = glslang::Shader::new(compiler, input)
            .expect("shader")
            .compile()
            .expect("compile");
        let (missing, environment) = seal::unsealed(&compiled).expect("module");
        assert!(missing >= 4 && !environment);
        let declared = seal::Environment {
            rounding_rte_32: true,
            denorm_preserve_32: true,
        };
        let sealed = seal::seal(&compiled, declared).expect("seal");
        assert_eq!(seal::unsealed(&sealed).expect("module"), (0, true));
        // The probed environment leaves out only `RoundingModeRTE 32`
        // (4 words).
        let probed = seal::seal(
            &compiled,
            seal::Environment {
                rounding_rte_32: false,
                ..declared
            },
        )
        .expect("seal");
        assert_eq!(seal::unsealed(&probed).expect("module"), (0, false));
        assert_eq!(probed.len(), sealed.len() - 4);
        formation::validate(&probed).expect("a sealed module validates");
        formation::validate(&sealed).expect("a sealed module validates");
        // `fma` becomes `OpFmaKHR` (opcode 4427, six words) with its
        // capability and extension; nothing else changes.
        let bound = seal::fma_khr(&sealed).expect("fma binding");
        let fma_khr = (6 << 16) | 4427;
        assert_eq!(bound.iter().filter(|word| **word == fma_khr).count(), 1);
        assert!(bound.windows(2).any(|pair| pair == [(2 << 16) | 17, 6030]));
        // + capability (2) + extension (4) - the shorter instruction (8 -> 6).
        assert_eq!(bound.len(), sealed.len() + 2 + 4 - 2);
    }
}
