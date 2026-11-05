//! Seismic's Vulkan backend: the direct native route only. Authored GLSL compute
//! kernels are formed to SPIR-V with the statically linked glslang, sealed
//! for Seismic's numerical environment, validated with SPIRV-Tools and run on
//! one queue of the opened device through ash.
//!
//! There is no planned route: this crate implements no compiler target.

mod device;
pub mod direct;
mod facts;
pub mod formation;
mod graph_cost;
mod instance;
mod memory;
pub mod probe;
pub mod reservation;
pub mod seal;
pub mod toolchain;
pub mod topology;
pub mod transfer;

pub use device::{Device, DeviceHandle, HeapBudget, MemoryBudget, OpenError};
pub use facts::{
    Allocation, Calibration, Description, DeviceType, Facts, FormationTarget, Limits, MemoryHeap,
    MemoryLayout, MemoryType, Probed, Properties, ReservationForm, Reserving, SUBGROUP_WIDTH,
};
pub use graph_cost::graph_cost;
pub use reservation::{ReservedBuffer, VulkanResizer};
pub use instance::LoaderError;
pub use memory::{AllocationAccount, Buffer, MemoryAccounts};

/// Enumerate the physical devices against the floor, from their reported
/// properties alone: discovery opens no logical device. GPUs come first, so
/// the first device of the backend is a GPU when there is one.
pub fn discover() -> Result<Vec<Description>, LoaderError> {
    let instance = instance::instance()?;
    let mut descriptions = instance
        .physical_devices()?
        .into_iter()
        .map(|physical| facts::describe(&instance, physical))
        .collect::<Vec<_>>();
    descriptions.sort_by_key(|description| !description.facts.is_gpu());
    Ok(descriptions)
}

/// Accounts that bound and charge nothing for every kind of a device of
/// `facts`: a test's, and the probe process's, which allocates only its
/// probes' few buffers.
pub(crate) fn unbounded_accounts(facts: &Facts) -> MemoryAccounts {
    let unbounded = || -> std::sync::Arc<dyn seismic_compiler::executable::MemoryAccount> {
        std::sync::Arc::new(seismic_compiler::executable::Unbounded)
    };
    let kinds = (0..facts.memory.types.len() as u32)
        .chain([topology::IMPORTED_KIND])
        .map(|kind| AllocationAccount { kind, account: unbounded() })
        .collect();
    MemoryAccounts { host: unbounded(), kinds, driver: unbounded() }
}

/// The devices meeting the floor a test exercises; none without a loader,
/// which a run that selected Vulkan (`SEISMIC_TEST_BACKEND`) refuses, as it
/// refuses another backend's selection or finding no such device.
#[cfg(test)]
pub(crate) fn test_devices() -> Vec<Description> {
    let selected = std::env::var("SEISMIC_TEST_BACKEND").ok();
    assert!(selected.as_deref().is_none_or(|name| name == "vulkan"), "this test exercises Vulkan");
    let devices = match discover() {
        Ok(devices) => devices,
        Err(error) => {
            assert!(selected.is_none(), "selected Vulkan discovery failed: {error}");
            eprintln!("no Vulkan loader: {error}");
            return Vec::new();
        }
    };
    let devices = devices.into_iter().filter(|description| description.floor.is_ok()).collect::<Vec<_>>();
    assert!(selected.is_none() || !devices.is_empty(), "no eligible selected Vulkan device");
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every device discovery reports available opens, and one it reports
    /// unavailable (below the floor) fails to open for the same reason;
    /// storage round-trips through host writes and reads.
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
                "{} ({}, {}): unprobed properties {:?}, subgroup size required {:?}, matrix path {}, shader fma {:?}, uuid {}",
                description.facts.name,
                description.facts.driver_name,
                description.facts.driver_info,
                description.properties(&Probed::default()).as_ref().map(Properties::identity),
                description.floor.is_ok().then(|| description.facts.requires_subgroup_size()),
                description.facts.matrix_path,
                description.facts.shader_fma(),
                facts::hex(&description.facts.uuid)
            );
            if let Err(reason) = &description.floor {
                let error = Device::open(description.facts.uuid, crate::unbounded_accounts(&description.facts), &Default::default())
                    .err()
                    .expect("a device discovery reports unavailable does not open");
                eprintln!("  {error}");
                if let Some(open_reason) = error.floor_reason() {
                    assert_eq!(&open_reason, reason);
                }
                continue;
            }
            let device = Device::open(description.facts.uuid, crate::unbounded_accounts(&description.facts), &Default::default()).unwrap_or_else(|error| {
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
        // Where the driver cannot compile it, the environment leaves out
        // only `RoundingModeRTE 32` (4 words).
        let without_rte = seal::seal(
            &compiled,
            seal::Environment {
                rounding_rte_32: false,
                ..declared
            },
        )
        .expect("seal");
        assert_eq!(
            seal::unsealed(&without_rte, &[16, 32]).expect("module"),
            (0, false)
        );
        assert_eq!(without_rte.len(), sealed.len() - 4);
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
        formation::validate(&without_rte).expect("a sealed module validates");
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
