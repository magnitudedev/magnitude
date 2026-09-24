//! Physical-device facts and the device floor (Vulkan backend spec §4).
//!
//! Discovery and opening read the same facts through [`describe`]: the floor
//! decides availability (the first missing item is the reason), and the gated
//! facts select prefix macros, the numerical environment and the formation and
//! tuning identities.

use crate::instance::Instance;
use ash::vk;
use std::ffi::CStr;

/// Minimum group memory, invocations, grid and group extents and push
/// constants of the floor.
const FLOOR_SHARED_BYTES: u32 = 32 * 1024;
const FLOOR_INVOCATIONS: u32 = 1024;
const FLOOR_GROUP_COUNT: u32 = 65_535;
const FLOOR_GROUP_SIZE: [u32; 3] = [1024, 1024, 64];
const FLOOR_PUSH_CONSTANT_BYTES: u32 = 128;
/// The subgroup width every pipeline requires (§6.3).
pub const SUBGROUP_WIDTH: u32 = 32;

/// PCI vendor of AMD, whose Windows driver misreports cooperative matrix on
/// RDNA2 (§4).
const AMD_VENDOR: u32 = 0x1002;

/// `VK_KHR_shader_fma` (newer than ash 0.38).
pub(crate) const SHADER_FMA_EXTENSION: &CStr = c"VK_KHR_shader_fma";

/// `VkPhysicalDeviceShaderFmaFeaturesKHR` (newer than ash 0.38).
#[repr(C)]
pub(crate) struct ShaderFmaFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    pub(crate) float16: vk::Bool32,
    pub(crate) float32: vk::Bool32,
    pub(crate) float64: vk::Bool32,
}

impl Default for ShaderFmaFeatures {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_579_000),
            p_next: std::ptr::null_mut(),
            float16: vk::FALSE,
            float32: vk::FALSE,
            float64: vk::FALSE,
        }
    }
}

// SAFETY: a Vulkan structure with `sType` and `pNext` leading, extending
// both chains per the extension's registry entry.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for ShaderFmaFeatures {}
unsafe impl vk::ExtendsDeviceCreateInfo for ShaderFmaFeatures {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    Discrete,
    Integrated,
    /// A software device (lavapipe).
    Cpu,
    Other,
}

/// Limits the runtime checks launches and allocations against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub max_shared_bytes: u64,
    pub max_invocations: u64,
    pub max_group_count: [u64; 3],
    pub max_group_size: [u64; 3],
    /// Largest single allocation: min(`maxMemoryAllocationSize`,
    /// `maxBufferSize`).
    pub max_allocation_bytes: u64,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f64,
    /// Valid bits of the chosen queue family's timestamps.
    pub timestamp_valid_bits: u32,
}

/// The widths `OpFmaKHR` supports (`VK_KHR_shader_fma`); all false without
/// the extension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderFma {
    pub float16: bool,
    pub float32: bool,
    pub float64: bool,
}

/// How a device pairs its timestamps with the host clock (§8.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Calibration {
    Khr,
    Ext,
}

/// Everything Seismic uses about one physical device.
#[derive(Clone, Debug)]
pub struct Facts {
    pub name: String,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`.
    pub uuid: [u8; 16],
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: DeviceType,
    /// `VkDriverId` (raw).
    pub driver_id: i32,
    pub driver_name: String,
    pub driver_info: String,
    pub driver_version: u32,
    pub api_version: u32,
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: Limits,
    /// The queue family work is submitted to: compute-only when one exists.
    pub queue_family: u32,
    pub dedicated_compute_queue: bool,
    /// Subgroup-scope 16x16x16 cooperative matrix with f16xf16->f32 and
    /// s8xs8->s32 (`SEISMIC_HAS_MATRIX`), never on RDNA2.
    pub matrix: bool,
    /// `DenormPreserve 32` is emitted when true (§7.3).
    pub denorm_preserve_32: bool,
    /// `RoundingModeRTE 32` is emitted when true (§7.3). False on the NVIDIA
    /// proprietary driver, whose compiler traps on fp32 `OpFDiv` under it
    /// (580.159.03, §16.1); opening such a device probes that its default
    /// fp32 rounding is round-to-nearest-even.
    pub rounding_rte_32: bool,
    pub mixed_dot_accelerated: bool,
    /// `VK_KHR_shader_fma` with fp32 `OpFmaKHR` (fp16/fp64 as reported):
    /// `seismic_fma_rn` is bound to `OpFmaKHR` (§16.1).
    pub shader_fma: ShaderFma,
    pub f32_atomic_add: bool,
    pub shared_int64_atomics: bool,
    pub calibration: Option<Calibration>,
    /// Largest device-local heap.
    pub device_local_bytes: u64,
    /// Largest device-local heap that also has host-visible coherent memory
    /// (ReBAR/SAM, or unified memory); 0 when there is none.
    pub host_visible_device_local_bytes: u64,
}

impl Facts {
    pub fn is_gpu(&self) -> bool {
        matches!(self.device_type, DeviceType::Discrete | DeviceType::Integrated)
    }

    /// The formation-identity part the device determines (§7.5), including
    /// the sealed environment and how `seismic_fma_rn` is bound.
    pub fn formation_identity(&self) -> String {
        format!(
            "driver {} {};cache {};matrix {};denorm {};rte32 {};fma {}",
            self.driver_id,
            self.driver_version,
            hex(&self.pipeline_cache_uuid),
            u8::from(self.matrix),
            if self.denorm_preserve_32 { "preserve" } else { "default" },
            if self.rounding_rte_32 { "declared" } else { "probed" },
            if self.shader_fma.float32 { "khr" } else { "glsl" }
        )
    }

    /// The environment the seal pass declares for this device.
    pub fn environment(&self) -> crate::seal::Environment {
        crate::seal::Environment {
            rounding_rte_32: self.rounding_rte_32,
            denorm_preserve_32: self.denorm_preserve_32,
        }
    }

    /// Tuning identity (§7.5): vendor and device, name, driver ID and
    /// version, so a driver update changes the key.
    pub fn tuning_identity(&self) -> String {
        format!(
            "vulkan;vendor {:04x};device {:04x};{};driver {} {} ({})",
            self.vendor_id,
            self.device_id,
            self.name,
            self.driver_id,
            self.driver_version,
            self.driver_info
        )
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One enumerated physical device: its facts, and whether it meets the floor
/// (`Err` names the first missing item).
#[derive(Clone, Debug)]
pub struct Description {
    pub facts: Facts,
    pub floor: Result<(), String>,
}

/// Device extensions of the floor.
pub(crate) const FLOOR_EXTENSIONS: [&CStr; 2] = [
    ash::khr::workgroup_memory_explicit_layout::NAME,
    ash::ext::memory_budget::NAME,
];

fn text(raw: &[std::ffi::c_char]) -> String {
    // SAFETY: Vulkan fixed-size name arrays are NUL-terminated.
    unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Describe one physical device.
pub(crate) fn describe(instance: &Instance, physical: vk::PhysicalDevice) -> Description {
    let raw = instance.raw();
    let extensions = unsafe { raw.enumerate_device_extension_properties(physical) }
        .unwrap_or_default()
        .iter()
        .map(|extension| text(&extension.extension_name))
        .collect::<Vec<_>>();
    let has = |name: &CStr| extensions.iter().any(|extension| extension.as_bytes() == name.to_bytes());

    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
    let mut v13 = vk::PhysicalDeviceVulkan13Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut v11)
        .push_next(&mut v12)
        .push_next(&mut v13);
    unsafe { raw.get_physical_device_properties2(physical, &mut properties) };
    let base = properties.properties;

    let mut explicit = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
    let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
    let mut cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut shader_fma = ShaderFmaFeatures::default();
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f11)
        .push_next(&mut f12)
        .push_next(&mut f13);
    if has(ash::khr::workgroup_memory_explicit_layout::NAME) {
        features = features.push_next(&mut explicit);
    }
    if has(ash::ext::shader_atomic_float::NAME) {
        features = features.push_next(&mut atomic_float);
    }
    if has(ash::khr::cooperative_matrix::NAME) {
        features = features.push_next(&mut cooperative);
    }
    if has(SHADER_FMA_EXTENSION) {
        features = features.push_next(&mut shader_fma);
    }
    unsafe { raw.get_physical_device_features2(physical, &mut features) };
    let core = features.features;

    let families = unsafe { raw.get_physical_device_queue_family_properties(physical) };
    let compute_only = families.iter().position(|family| {
        family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && family.timestamp_valid_bits > 0
    });
    let universal = families.iter().position(|family| {
        family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.timestamp_valid_bits > 0
    });
    let queue_family = compute_only.or(universal);

    let memory = unsafe { raw.get_physical_device_memory_properties(physical) };
    let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
    let types = &memory.memory_types[..memory.memory_type_count as usize];
    let device_local_bytes = heaps
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    let host_visible_device_local_bytes = types
        .iter()
        .filter(|kind| {
            kind.property_flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .map(|kind| heaps[kind.heap_index as usize].size)
        .max()
        .unwrap_or(0);

    let mixed_dot_accelerated =
        v13.integer_dot_product4x8_bit_packed_mixed_signedness_accelerated == vk::TRUE;
    let matrix = cooperative.cooperative_matrix == vk::TRUE
        && matrix_shapes(instance, physical)
        // RDNA2 has no WMMA units whatever its driver reports; accelerated
        // mixed-signedness packed dot identifies RDNA3+ (§4, §16.3).
        && (base.vendor_id != AMD_VENDOR || mixed_dot_accelerated);
    let calibration = if has(ash::khr::calibrated_timestamps::NAME) {
        Some(Calibration::Khr)
    } else if has(ash::ext::calibrated_timestamps::NAME) {
        Some(Calibration::Ext)
    } else {
        None
    };
    let limits = base.limits;
    let facts = Facts {
        name: text(&base.device_name),
        uuid: v11.device_uuid,
        vendor_id: base.vendor_id,
        device_id: base.device_id,
        device_type: match base.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
            vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
            _ => DeviceType::Other,
        },
        driver_id: v12.driver_id.as_raw(),
        driver_name: text(&v12.driver_name),
        driver_info: text(&v12.driver_info),
        driver_version: base.driver_version,
        api_version: base.api_version,
        pipeline_cache_uuid: base.pipeline_cache_uuid,
        limits: Limits {
            max_shared_bytes: limits.max_compute_shared_memory_size.into(),
            max_invocations: limits.max_compute_work_group_invocations.into(),
            max_group_count: limits.max_compute_work_group_count.map(u64::from),
            max_group_size: limits.max_compute_work_group_size.map(u64::from),
            max_allocation_bytes: v11.max_memory_allocation_size.min(v13.max_buffer_size),
            timestamp_period: f64::from(limits.timestamp_period),
            timestamp_valid_bits: queue_family
                .map_or(0, |family| families[family].timestamp_valid_bits),
        },
        queue_family: queue_family.map_or(0, |family| family as u32),
        dedicated_compute_queue: compute_only.is_some(),
        matrix,
        denorm_preserve_32: v12.shader_denorm_preserve_float32 == vk::TRUE,
        rounding_rte_32: v12.driver_id != vk::DriverId::NVIDIA_PROPRIETARY,
        mixed_dot_accelerated,
        shader_fma: ShaderFma {
            float16: shader_fma.float16 == vk::TRUE,
            float32: shader_fma.float32 == vk::TRUE,
            float64: shader_fma.float64 == vk::TRUE,
        },
        f32_atomic_add: atomic_float.shader_buffer_float32_atomic_add == vk::TRUE,
        shared_int64_atomics: f12.shader_shared_int64_atomics == vk::TRUE,
        calibration,
        device_local_bytes,
        host_visible_device_local_bytes,
    };

    let on = |value: vk::Bool32| value == vk::TRUE;
    let subgroup_operations = vk::SubgroupFeatureFlags::BASIC
        | vk::SubgroupFeatureFlags::VOTE
        | vk::SubgroupFeatureFlags::ARITHMETIC
        | vk::SubgroupFeatureFlags::BALLOT
        | vk::SubgroupFeatureFlags::SHUFFLE
        | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE
        | vk::SubgroupFeatureFlags::CLUSTERED
        | vk::SubgroupFeatureFlags::QUAD;
    let requirements: [(&str, bool); 38] = [
        ("Vulkan 1.3", vk::api_version_major(base.api_version) > 1 || vk::api_version_minor(base.api_version) >= 3),
        ("a compute queue with timestamps", queue_family.is_some()),
        ("bufferDeviceAddress", on(f12.buffer_device_address)),
        ("shaderInt64", on(core.shader_int64)),
        ("shaderInt16", on(core.shader_int16)),
        ("shaderInt8", on(f12.shader_int8)),
        ("shaderFloat16", on(f12.shader_float16)),
        ("storageBuffer16BitAccess", on(f11.storage_buffer16_bit_access)),
        ("storageBuffer8BitAccess", on(f12.storage_buffer8_bit_access)),
        ("scalarBlockLayout", on(f12.scalar_block_layout)),
        ("timelineSemaphore", on(f12.timeline_semaphore)),
        ("synchronization2", on(f13.synchronization2)),
        ("maintenance4", on(f13.maintenance4)),
        ("pipelineCreationCacheControl", on(f13.pipeline_creation_cache_control)),
        ("vulkanMemoryModel", on(f12.vulkan_memory_model)),
        ("vulkanMemoryModelDeviceScope", on(f12.vulkan_memory_model_device_scope)),
        ("shaderSubgroupExtendedTypes", on(f12.shader_subgroup_extended_types)),
        ("shaderIntegerDotProduct", on(f13.shader_integer_dot_product)),
        ("subgroupSizeControl", on(f13.subgroup_size_control)),
        ("computeFullSubgroups", on(f13.compute_full_subgroups)),
        ("compute in requiredSubgroupSizeStages", v13.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)),
        ("subgroup size 32", v13.min_subgroup_size <= SUBGROUP_WIDTH && SUBGROUP_WIDTH <= v13.max_subgroup_size),
        ("subgroup operations in compute", v11.subgroup_supported_stages.contains(vk::ShaderStageFlags::COMPUTE)),
        ("basic, vote, arithmetic, ballot, shuffle, shuffle-relative, clustered and quad subgroup operations", v11.subgroup_supported_operations.contains(subgroup_operations)),
        ("VK_KHR_workgroup_memory_explicit_layout", has(FLOOR_EXTENSIONS[0])),
        ("workgroupMemoryExplicitLayout (base, 8-bit, 16-bit, scalar)", on(explicit.workgroup_memory_explicit_layout) && on(explicit.workgroup_memory_explicit_layout8_bit_access) && on(explicit.workgroup_memory_explicit_layout16_bit_access) && on(explicit.workgroup_memory_explicit_layout_scalar_block_layout)),
        ("VK_EXT_memory_budget", has(FLOOR_EXTENSIONS[1])),
        ("RTE rounding for fp16 and fp32", on(v12.shader_rounding_mode_rte_float16) && on(v12.shader_rounding_mode_rte_float32)),
        ("SignedZeroInfNanPreserve for fp16 and fp32", on(v12.shader_signed_zero_inf_nan_preserve_float16) && on(v12.shader_signed_zero_inf_nan_preserve_float32)),
        ("32 KiB of shared memory", limits.max_compute_shared_memory_size >= FLOOR_SHARED_BYTES),
        ("1024 invocations per workgroup", limits.max_compute_work_group_invocations >= FLOOR_INVOCATIONS),
        ("65535 workgroups on every axis", limits.max_compute_work_group_count.iter().all(|count| *count >= FLOOR_GROUP_COUNT)),
        ("workgroup size (1024, 1024, 64)", limits.max_compute_work_group_size.iter().zip(FLOOR_GROUP_SIZE).all(|(size, floor)| *size >= floor)),
        ("128 bytes of push constants", limits.max_push_constants_size >= FLOOR_PUSH_CONSTANT_BYTES),
        ("a GPU or CPU device type", !matches!(facts.device_type, DeviceType::Other)),
        ("a device-local heap", device_local_bytes > 0),
        ("host-visible coherent memory", types.iter().any(|kind| kind.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))),
        ("a nonzero allocation limit", facts.limits.max_allocation_bytes > 0),
    ];
    let floor = match requirements.iter().find(|(_, met)| !met) {
        Some((missing, _)) => Err(format!("missing {missing}")),
        None => Ok(()),
    };
    Description { facts, floor }
}

/// Whether the device offers subgroup-scope 16x16x16 cooperative matrix
/// with f16 x f16 -> f32 and s8 x s8 -> s32.
fn matrix_shapes(instance: &Instance, physical: vk::PhysicalDevice) -> bool {
    let loader = ash::khr::cooperative_matrix::Instance::new(instance.entry(), instance.raw());
    let Ok(shapes) = (unsafe { loader.get_physical_device_cooperative_matrix_properties(physical) })
    else {
        return false;
    };
    let offers = |input: vk::ComponentTypeKHR, output: vk::ComponentTypeKHR| {
        shapes.iter().any(|shape| {
            shape.scope == vk::ScopeKHR::SUBGROUP
                && (shape.m_size, shape.n_size, shape.k_size) == (16, 16, 16)
                && shape.a_type == input
                && shape.b_type == input
                && shape.c_type == output
                && shape.result_type == output
        })
    };
    offers(vk::ComponentTypeKHR::FLOAT16, vk::ComponentTypeKHR::FLOAT32)
        && offers(vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32)
}
