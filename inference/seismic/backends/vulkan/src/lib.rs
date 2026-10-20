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
pub mod toolchain;

pub use device::{Device, HeapBudget, MemoryBudget, OpenError, Rounded};
pub use facts::{
    Allocation, Calibration, Description, DeviceType, Facts, FormationFacts, Limits, MemoryHeap,
    MemoryLayout, MemoryType, SubgroupWidth, SUBGROUP_WIDTH,
};
pub use instance::LoaderError;
pub use memory::{AllocationAccount, Buffer, PlacedStorage};

/// Enumerate the physical devices against the whole floor, so a device
/// discovery reports available opens: one meeting the static floor whose
/// numerics only a probe establishes is opened briefly, once per process,
/// to run it (`device::numerical_floor`). GPUs come first, so the first
/// device of the backend is a GPU when there is one.
pub fn discover() -> Result<Vec<Description>, LoaderError> {
    let instance = instance::instance()?;
    let mut descriptions = instance
        .physical_devices()?
        .into_iter()
        .map(|physical| {
            let mut description = facts::describe(instance, physical);
            if description.floor.is_ok() {
                description.floor = device::numerical_floor(instance, physical, &description.facts)
                    .map_err(|error| {
                        error.floor_reason().unwrap_or_else(|| {
                            format!("its numerical probes could not run: {error}")
                        })
                    });
            }
            description
        })
        .collect::<Vec<_>>();
    descriptions.sort_by_key(|description| !description.facts.is_gpu());
    Ok(descriptions)
}

/// Every heap's `VK_EXT_memory_budget` budget and usage of the device of
/// `uuid`, in heap order, read from the physical device without a logical
/// one; `None` when no device has that UUID. The extension is in the floor,
/// so every device meeting it reports budgets.
pub fn heap_budgets(uuid: [u8; 16]) -> Result<Option<Vec<HeapBudget>>, LoaderError> {
    let instance = instance::instance()?;
    for physical in instance.physical_devices()? {
        let mut id = ash::vk::PhysicalDeviceIDProperties::default();
        let mut properties = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        unsafe {
            instance
                .raw()
                .get_physical_device_properties2(physical, &mut properties)
        };
        if id.device_uuid == uuid {
            return Ok(Some(device::heap_budgets(instance, physical)));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every device discovery reports available opens, and one it reports
    /// unavailable (below the static floor, or failing a numerical probe)
    /// fails to open for the same reason; storage round-trips through host
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
                "{} ({}, {}): floor {:?}, subgroup {:?}, f16 {}, matrix {}, denorm-preserve {}, rte32 {}, shader fma {:?}, uuid {}",
                description.facts.name,
                description.facts.driver_name,
                description.facts.driver_info,
                description.floor,
                description.facts.subgroup_width,
                description.facts.float16,
                description.facts.matrix,
                description.facts.denorm_preserve_32,
                description.facts.rounding_rte_32,
                description.facts.shader_fma,
                facts::hex(&description.facts.uuid)
            );
            if let Err(reason) = &description.floor {
                let error = Device::open(description.facts.uuid)
                    .err()
                    .expect("a device discovery reports unavailable does not open");
                eprintln!("  {error}");
                if let Some(open_reason) = error.floor_reason() {
                    assert_eq!(&open_reason, reason);
                }
                continue;
            }
            let device = Device::open(description.facts.uuid).unwrap_or_else(|error| {
                panic!("a device discovery reports available opens: {error}")
            });
            let buffer = device.allocate(1 << 20, 256, None).expect("allocation");
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
        let (missing, environment) = seal::unsealed(&compiled, &[16, 32]).expect("module");
        assert!(missing >= 4 && !environment);
        let declared = seal::Environment {
            float16: true,
            rounding_rte_32: true,
            denorm_preserve_32: true,
        };
        let sealed = seal::seal(&compiled, declared).expect("seal");
        assert_eq!(
            seal::unsealed(&sealed, &[16, 32]).expect("module"),
            (0, true)
        );
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
        assert_eq!(
            seal::unsealed(&probed, &[16, 32]).expect("module"),
            (0, false)
        );
        assert_eq!(probed.len(), sealed.len() - 4);
        // Without fp16 arithmetic, the two fp16 modes (8 words) are left out
        // and the fp32 environment is complete.
        let fp32 = seal::seal(
            &compiled,
            seal::Environment {
                float16: false,
                ..declared
            },
        )
        .expect("seal");
        assert_eq!(seal::unsealed(&fp32, &[32]).expect("module"), (0, true));
        assert_eq!(seal::unsealed(&fp32, &[16]).expect("module"), (0, false));
        assert_eq!(fp32.len(), sealed.len() - 8);
        formation::validate(&probed).expect("a sealed module validates");
        formation::validate(&sealed).expect("a sealed module validates");
        formation::validate(&fp32).expect("a sealed module validates");
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
