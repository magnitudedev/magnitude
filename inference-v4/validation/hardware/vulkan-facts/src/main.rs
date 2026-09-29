//! Vulkan device facts and driver-behaviour probes for the Vulkan backend spec (§4 floor, §16).
//!
//! `vulkan-facts` prints one JSON document with, per physical device:
//! - identity, API/driver version, UUIDs, subgroup, integer-dot, float-control and limit facts;
//! - the §4 floor features and the gated facts (cooperative-matrix shapes, atomics, ReBAR, queues);
//! - memory heaps, types and budgets;
//! - device probes (skipped for CPU devices unless `--all`):
//!   - `fma`: witness inputs through `fma()`, `a*b+c` and `precise a*b+c` (fused result 2^-24,
//!     two roundings 0), plus the driver's ISA lines for the multiply/add if it exposes them;
//!   - `shared_spec`: a shared array sized by a specialization constant at several sizes: whether
//!     the result is correct and the driver's reported shared (LDS) size.
use ash::vk;
use serde_json::{json, Value};
use std::ffi::{c_char, CStr};

type Error = Box<dyn std::error::Error>;

fn text(chars: &[c_char]) -> String {
    let bytes: Vec<u8> = chars.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn uuid(bytes: &[u8]) -> String {
    let h: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let h = h.concat();
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

fn b(v: vk::Bool32) -> bool {
    v == vk::TRUE
}

fn compile(source: &str) -> Result<Vec<u32>, Error> {
    let compiler = glslang::Compiler::acquire().ok_or("glslang compiler unavailable")?;
    let source = glslang::ShaderSource::from(source);
    let options = glslang::CompilerOptions {
        source_language: glslang::SourceLanguage::GLSL,
        target: glslang::Target::Vulkan {
            version: glslang::VulkanVersion::Vulkan1_3,
            spirv_version: glslang::SpirvVersion::SPIRV1_6,
        },
        version_profile: None,
        messages: glslang::ShaderMessage::DEFAULT,
    };
    let input = glslang::ShaderInput::new(
        &source,
        glslang::ShaderStage::Compute,
        &options,
        None::<&[(&str, Option<&str>)]>,
        None,
    )?;
    let shader = glslang::Shader::new(compiler, input)?;
    Ok(shader.compile()?)
}

const WATCHED_EXTENSIONS: &[&str] = &[
    "VK_KHR_cooperative_matrix",
    "VK_NV_cooperative_matrix2",
    "VK_KHR_shader_fma",
    "VK_KHR_workgroup_memory_explicit_layout",
    "VK_EXT_memory_budget",
    "VK_EXT_memory_priority",
    "VK_EXT_pageable_device_local_memory",
    "VK_EXT_external_memory_host",
    "VK_KHR_pipeline_executable_properties",
    "VK_KHR_calibrated_timestamps",
    "VK_EXT_calibrated_timestamps",
    "VK_KHR_shader_float_controls2",
    "VK_KHR_shader_bfloat16",
    "VK_EXT_shader_float8",
    "VK_EXT_shader_atomic_float",
    "VK_EXT_shader_atomic_float2",
    "VK_KHR_shader_subgroup_rotate",
    "VK_KHR_shader_expect_assume",
    "VK_KHR_maintenance5",
    "VK_KHR_maintenance6",
    "VK_KHR_maintenance7",
    "VK_KHR_maintenance8",
    "VK_KHR_maintenance9",
    "VK_EXT_descriptor_buffer",
    "VK_KHR_shader_relaxed_extended_instruction",
    "VK_EXT_shader_replicated_composites",
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_shader_integer_dot_product",
];

fn main() -> Result<(), Error> {
    let all = std::env::args().any(|a| a == "--all");
    let entry = unsafe { ash::Entry::load()? };
    let app = vk::ApplicationInfo::default()
        .application_name(c"vulkan-facts")
        .api_version(vk::API_VERSION_1_3);
    let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)? };
    let loader_version = unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);
    let mut devices = Vec::new();
    for pd in unsafe { instance.enumerate_physical_devices()? } {
        devices.push(describe(&entry, &instance, pd, all)?);
    }
    let report = json!({
        "loader_api": version(loader_version),
        "glslang_sys": "0.8.1+275822a",
        "devices": devices,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    unsafe { instance.destroy_instance(None) };
    Ok(())
}

fn version(v: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))
}

fn describe(entry: &ash::Entry, instance: &ash::Instance, pd: vk::PhysicalDevice, all: bool) -> Result<Value, Error> {
    let extensions: Vec<String> = unsafe { instance.enumerate_device_extension_properties(pd)? }
        .iter()
        .map(|e| text(&e.extension_name))
        .collect();
    let has = |name: &str| extensions.iter().any(|e| e == name);

    let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
    let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
    let mut props = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut p11)
        .push_next(&mut p12)
        .push_next(&mut p13);
    unsafe { instance.get_physical_device_properties2(pd, &mut props) };
    let base = props.properties;
    let limits = base.limits;

    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut explicit = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
    let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
    let mut executable = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f11)
        .push_next(&mut f12)
        .push_next(&mut f13);
    if has("VK_KHR_workgroup_memory_explicit_layout") {
        features = features.push_next(&mut explicit);
    }
    if has("VK_KHR_cooperative_matrix") {
        features = features.push_next(&mut coop);
    }
    if has("VK_EXT_shader_atomic_float") {
        features = features.push_next(&mut atomic_float);
    }
    if has("VK_KHR_pipeline_executable_properties") {
        features = features.push_next(&mut executable);
    }
    unsafe { instance.get_physical_device_features2(pd, &mut features) };
    let f10 = features.features;

    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut memory2 = vk::PhysicalDeviceMemoryProperties2::default();
    if has("VK_EXT_memory_budget") {
        memory2 = memory2.push_next(&mut budget);
    }
    unsafe { instance.get_physical_device_memory_properties2(pd, &mut memory2) };
    let memory = memory2.memory_properties;
    let heaps: Vec<Value> = (0..memory.memory_heap_count as usize)
        .map(|i| {
            let h = memory.memory_heaps[i];
            json!({
                "index": i,
                "size_gib": h.size as f64 / (1u64 << 30) as f64,
                "flags": format!("{:?}", h.flags),
                "budget_gib": budget.heap_budget[i] as f64 / (1u64 << 30) as f64,
                "usage_gib": budget.heap_usage[i] as f64 / (1u64 << 30) as f64,
            })
        })
        .collect();
    let types: Vec<Value> = (0..memory.memory_type_count as usize)
        .map(|i| {
            let t = memory.memory_types[i];
            json!({ "index": i, "heap": t.heap_index, "flags": format!("{:?}", t.property_flags) })
        })
        .collect();
    let rebar_gib: f64 = (0..memory.memory_type_count as usize)
        .map(|i| memory.memory_types[i])
        .filter(|t| {
            t.property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE)
        })
        .map(|t| memory.memory_heaps[t.heap_index as usize].size as f64 / (1u64 << 30) as f64)
        .fold(0.0, f64::max);

    let queues: Vec<Value> = unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .enumerate()
        .map(|(i, q)| {
            json!({ "index": i, "flags": format!("{:?}", q.queue_flags), "count": q.queue_count,
                    "timestamp_valid_bits": q.timestamp_valid_bits })
        })
        .collect();

    let matrix: Vec<Value> = if has("VK_KHR_cooperative_matrix") {
        let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
        unsafe { loader.get_physical_device_cooperative_matrix_properties(pd)? }
            .iter()
            .map(|m| {
                json!(format!(
                    "{}x{}x{} A {:?} B {:?} C {:?} R {:?} sat {} scope {:?}",
                    m.m_size, m.n_size, m.k_size, m.a_type, m.b_type, m.c_type, m.result_type,
                    b(m.saturating_accumulation), m.scope
                ))
            })
            .collect()
    } else {
        Vec::new()
    };

    let floor = json!({
        "bufferDeviceAddress": b(f12.buffer_device_address),
        "shaderInt64": b(f10.shader_int64),
        "shaderInt16": b(f10.shader_int16),
        "shaderInt8": b(f12.shader_int8),
        "shaderFloat16": b(f12.shader_float16),
        "storageBuffer16BitAccess": b(f11.storage_buffer16_bit_access),
        "storageBuffer8BitAccess": b(f12.storage_buffer8_bit_access),
        "scalarBlockLayout": b(f12.scalar_block_layout),
        "timelineSemaphore": b(f12.timeline_semaphore),
        "synchronization2": b(f13.synchronization2),
        "maintenance4": b(f13.maintenance4),
        "pipelineCreationCacheControl": b(f13.pipeline_creation_cache_control),
        "vulkanMemoryModel": b(f12.vulkan_memory_model),
        "vulkanMemoryModelDeviceScope": b(f12.vulkan_memory_model_device_scope),
        "shaderSubgroupExtendedTypes": b(f12.shader_subgroup_extended_types),
        "shaderIntegerDotProduct": b(f13.shader_integer_dot_product),
        "subgroupSizeControl": b(f13.subgroup_size_control),
        "computeFullSubgroups": b(f13.compute_full_subgroups),
        "workgroupMemoryExplicitLayout": b(explicit.workgroup_memory_explicit_layout),
        "workgroupMemoryExplicitLayout8BitAccess": b(explicit.workgroup_memory_explicit_layout8_bit_access),
        "workgroupMemoryExplicitLayout16BitAccess": b(explicit.workgroup_memory_explicit_layout16_bit_access),
        "workgroupMemoryExplicitLayoutScalarBlockLayout": b(explicit.workgroup_memory_explicit_layout_scalar_block_layout),
        "VK_EXT_memory_budget": has("VK_EXT_memory_budget"),
    });

    let facts = json!({
        "name": text(&base.device_name),
        "type": format!("{:?}", base.device_type),
        "vendor_id": format!("{:#06x}", base.vendor_id),
        "device_id": format!("{:#06x}", base.device_id),
        "api": version(base.api_version),
        "driver_id": format!("{:?}", p12.driver_id),
        "driver_name": text(&p12.driver_name),
        "driver_info": text(&p12.driver_info),
        "driver_version_raw": base.driver_version,
        "device_uuid": uuid(&p11.device_uuid),
        "driver_uuid": uuid(&p11.driver_uuid),
        "pipeline_cache_uuid": uuid(&base.pipeline_cache_uuid),
        "subgroup": {
            "size": p11.subgroup_size,
            "min": p13.min_subgroup_size,
            "max": p13.max_subgroup_size,
            "required_stages": format!("{:?}", p13.required_subgroup_size_stages),
            "max_compute_workgroup_subgroups": p13.max_compute_workgroup_subgroups,
            "stages": format!("{:?}", p11.subgroup_supported_stages),
            "operations": format!("{:?}", p11.subgroup_supported_operations),
        },
        "integer_dot": {
            "4x8_packed_unsigned_accelerated": b(p13.integer_dot_product4x8_bit_packed_unsigned_accelerated),
            "4x8_packed_signed_accelerated": b(p13.integer_dot_product4x8_bit_packed_signed_accelerated),
            "4x8_packed_mixed_signedness_accelerated": b(p13.integer_dot_product4x8_bit_packed_mixed_signedness_accelerated),
            "8bit_mixed_signedness_accelerated": b(p13.integer_dot_product8_bit_mixed_signedness_accelerated),
            "accumulating_saturating_4x8_packed_mixed_accelerated":
                b(p13.integer_dot_product_accumulating_saturating4x8_bit_packed_mixed_signedness_accelerated),
        },
        "float_controls": {
            "denorm_behavior_independence": format!("{:?}", p12.denorm_behavior_independence),
            "rounding_mode_independence": format!("{:?}", p12.rounding_mode_independence),
            "rte_f16": b(p12.shader_rounding_mode_rte_float16),
            "rte_f32": b(p12.shader_rounding_mode_rte_float32),
            "szinp_f16": b(p12.shader_signed_zero_inf_nan_preserve_float16),
            "szinp_f32": b(p12.shader_signed_zero_inf_nan_preserve_float32),
            "denorm_preserve_f16": b(p12.shader_denorm_preserve_float16),
            "denorm_preserve_f32": b(p12.shader_denorm_preserve_float32),
            "denorm_flush_f32": b(p12.shader_denorm_flush_to_zero_float32),
        },
        "limits": {
            "max_compute_shared_memory_size": limits.max_compute_shared_memory_size,
            "max_compute_work_group_invocations": limits.max_compute_work_group_invocations,
            "max_compute_work_group_count": limits.max_compute_work_group_count,
            "max_compute_work_group_size": limits.max_compute_work_group_size,
            "max_push_constants_size": limits.max_push_constants_size,
            "max_memory_allocation_size": p11.max_memory_allocation_size,
            "max_buffer_size": p13.max_buffer_size,
            "max_memory_allocation_count": limits.max_memory_allocation_count,
            "max_storage_buffer_range": limits.max_storage_buffer_range,
            "min_storage_buffer_offset_alignment": limits.min_storage_buffer_offset_alignment,
            "non_coherent_atom_size": limits.non_coherent_atom_size,
            "timestamp_period": limits.timestamp_period,
            "timestamp_compute_and_graphics": b(limits.timestamp_compute_and_graphics),
            "sparse_binding": b(f10.sparse_binding),
            "sparse_residency_buffer": b(f10.sparse_residency_buffer),
            "sparse_address_space_size_gib": limits.sparse_address_space_size / (1u64 << 30),
            "residency_non_resident_strict": b(base.sparse_properties.residency_non_resident_strict),
        },
        "gated": {
            "cooperative_matrix_feature": b(coop.cooperative_matrix),
            "cooperative_matrix_shapes": matrix,
            "shader_buffer_float32_atomic_add": b(atomic_float.shader_buffer_float32_atomic_add),
            "shader_shared_float32_atomic_add": b(atomic_float.shader_shared_float32_atomic_add),
            "shader_buffer_int64_atomics": b(f12.shader_buffer_int64_atomics),
            "shader_shared_int64_atomics": b(f12.shader_shared_int64_atomics),
            "host_visible_device_local_heap_gib": rebar_gib,
            "pipeline_executable_info": b(executable.pipeline_executable_info),
        },
        "floor": floor,
        "watched_extensions": WATCHED_EXTENSIONS.iter().map(|e| json!({ "name": e, "present": has(e) })).collect::<Vec<_>>(),
        "extension_count": extensions.len(),
        "memory_heaps": heaps,
        "memory_types": types,
        "queue_families": queues,
    });

    let cpu = base.device_type == vk::PhysicalDeviceType::CPU;
    let probes = if cpu && !all {
        json!("skipped (CPU device; pass --all)")
    } else {
        let capture = has("VK_KHR_pipeline_executable_properties") && b(executable.pipeline_executable_info);
        match run_probes(instance, pd, capture, has("VK_KHR_shader_fma")) {
            Ok(v) => v,
            Err(e) => json!({ "error": e.to_string() }),
        }
    };
    let mut facts = facts;
    facts["probes"] = probes;
    Ok(facts)
}

struct Context<'a> {
    instance: &'a ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    executable: Option<ash::khr::pipeline_executable_properties::Device>,
    shader_fma: bool,
    family: u32,
}

const BUFFER_BYTES: u64 = 1 << 20;

/// `VkPhysicalDeviceShaderFmaFeaturesKHR` (newer than ash 0.38).
#[repr(C)]
struct ShaderFmaFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    float16: vk::Bool32,
    float32: vk::Bool32,
    float64: vk::Bool32,
}
const STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR: i32 = 1000579000;

fn run_probes(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    capture: bool,
    shader_fma: bool,
) -> Result<Value, Error> {
    let family = unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or("no compute queue")? as u32;
    let priorities = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
    let mut executable_features = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
    let mut fma_features = ShaderFmaFeatures {
        s_type: vk::StructureType::from_raw(STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR),
        p_next: std::ptr::null_mut(),
        float16: vk::FALSE,
        float32: vk::TRUE,
        float64: vk::FALSE,
    };
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default().maintenance4(true);
    let mut extension_names = Vec::new();
    if shader_fma {
        f13.p_next = (&mut fma_features as *mut ShaderFmaFeatures).cast();
        extension_names.push(c"VK_KHR_shader_fma".as_ptr());
    }
    let mut info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info).push_next(&mut f13);
    if capture {
        extension_names.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
        info = info.push_next(&mut executable_features);
    }
    info = info.enabled_extension_names(&extension_names);
    let device = unsafe { instance.create_device(pd, &info, None)? };
    let queue = unsafe { device.get_device_queue(family, 0) };
    let executable = capture.then(|| ash::khr::pipeline_executable_properties::Device::new(instance, &device));
    let context = Context {
        instance,
        memory_types: unsafe { instance.get_physical_device_memory_properties(pd) },
        device,
        queue,
        executable,
        shader_fma,
        family,
    };
    let result = probes(&context);
    unsafe { context.device.destroy_device(None) };
    result
}

/// SPIR-V opcodes and ids used by the `OpFmaKHR` rewrite.
const OP_EXTENSION: u32 = 10;
const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
const OP_CAPABILITY: u32 = 17;
const OP_FMA_KHR: u32 = 4427;
const CAPABILITY_FMA_KHR: u32 = 6030;
const GLSL_STD_450_FMA: u32 = 50;

fn string_words(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Rewrites every `GLSL.std.450 Fma` into `OpFmaKHR` (SPV_KHR_fma) — what glslang cannot emit from GLSL.
fn rewrite_fma_khr(code: &[u32]) -> Vec<u32> {
    let mut out = code[..5].to_vec();
    let mut glsl = None;
    let mut inserted = false;
    let mut i = 5;
    while i < code.len() {
        let count = (code[i] >> 16) as usize;
        let op = code[i] & 0xffff;
        let words = &code[i..i + count];
        if !inserted && op != OP_CAPABILITY {
            out.extend([(2 << 16) | OP_CAPABILITY, CAPABILITY_FMA_KHR]);
            let name = string_words("SPV_KHR_fma");
            out.push(((1 + name.len() as u32) << 16) | OP_EXTENSION);
            out.extend(name);
            inserted = true;
        }
        if op == OP_EXT_INST_IMPORT && text_words(&words[2..]) == "GLSL.std.450" {
            glsl = Some(words[1]);
        }
        if op == OP_EXT_INST && Some(words[3]) == glsl && words[4] == GLSL_STD_450_FMA {
            out.extend([(6 << 16) | OP_FMA_KHR, words[1], words[2], words[5], words[6], words[7]]);
        } else {
            out.extend_from_slice(words);
        }
        i += count;
    }
    out
}

fn text_words(words: &[u32]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).take_while(|b| *b != 0).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn probes(c: &Context) -> Result<Value, Error> {
    let _ = c.instance;
    // Witness: a = b = 1 + 2^-12, c = -(1 + 2^-11). Exact a*b + c = 2^-24; a*b rounds to 1 + 2^-11,
    // so two roundings give 0. Each form reads its own inputs so the compiler cannot merge them.
    let a = 1.0f32 + 2f32.powi(-12);
    let cc = -(1.0f32 + 2f32.powi(-11));
    let fma_source = r#"#version 460
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { float v[]; };
void main() {
    v[12] = fma(v[0], v[1], v[2]);
    v[13] = v[3] * v[4] + v[5];
    precise float p = v[6] * v[7] + v[8];
    v[14] = p;
}
"#;
    let mut input = vec![0u32; 16];
    for form in 0..3 {
        input[form * 3] = a.to_bits();
        input[form * 3 + 1] = a.to_bits();
        input[form * 3 + 2] = cc.to_bits();
    }
    let fused = 2f32.powi(-24);
    let classify = |x: f32| {
        if x == fused {
            "fused (2^-24)"
        } else if x == 0.0 {
            "two roundings (0)"
        } else {
            "unexpected"
        }
    };
    let isa_lines = |isa: &str| -> Vec<String> {
        isa.lines()
            .filter(|l| {
                let l = l.to_ascii_lowercase();
                ["v_fma", "v_fmac", "v_mul_f32", "v_add_f32", "v_mad", "v_pk_fma", "v_dual"].iter().any(|k| l.contains(k))
            })
            .map(|l| l.trim().to_string())
            .collect()
    };
    let glsl_code = compile(fma_source)?;
    let (out, isa, _stats) = dispatch(c, &glsl_code, &[], &input, input.len())?;
    let f = |i: usize| f32::from_bits(out[i]);
    let mut fma = json!({
        "fma() as GLSL.std.450 Fma": { "value": f(12), "class": classify(f(12)) },
        "a*b+c": { "value": f(13), "class": classify(f(13)) },
        "precise a*b+c": { "value": f(14), "class": classify(f(14)) },
        "isa_lines": isa_lines(&isa),
    });
    if c.shader_fma {
        let khr = rewrite_fma_khr(&glsl_code);
        // VULKAN_FACTS_DUMP=<dir> writes both modules for `spirv-val --target-env vulkan1.3`.
        if let Ok(dir) = std::env::var("VULKAN_FACTS_DUMP") {
            let bytes = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(format!("{dir}/fma-glsl.spv"), bytes(&glsl_code))?;
            std::fs::write(format!("{dir}/fma-khr.spv"), bytes(&khr))?;
        }
        let (out, isa, _stats) = dispatch(c, &khr, &[], &input, input.len())?;
        let x = f32::from_bits(out[12]);
        fma["fma() rewritten to OpFmaKHR"] = json!({ "value": x, "class": classify(x), "isa_lines": isa_lines(&isa) });
    } else {
        fma["fma() rewritten to OpFmaKHR"] = json!("VK_KHR_shader_fma not available");
    }
    // The same fma() next to a `precise` a*b+c over the same operands: a compiler free to treat
    // GLSL.std.450 Fma as mul+add may share the product and split the fma.
    let shared_operands = r#"#version 460
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { float v[]; };
void main() {
    float a = v[0], b = v[1], c = v[2];
    v[12] = fma(a, b, c);
    precise float p = a * b + c;
    v[13] = p;
}
"#;
    let shared_code = compile(shared_operands)?;
    let mut forms = vec![("GLSL.std.450 Fma", shared_code.clone())];
    if c.shader_fma {
        forms.push(("OpFmaKHR", rewrite_fma_khr(&shared_code)));
    }
    let mut next_to_precise = serde_json::Map::new();
    for (name, code) in forms {
        let (out, isa, _stats) = dispatch(c, &code, &[], &input, input.len())?;
        let x = f32::from_bits(out[12]);
        let p = f32::from_bits(out[13]);
        next_to_precise.insert(
            name.to_string(),
            json!({ "fma": { "value": x, "class": classify(x) }, "precise": { "value": p, "class": classify(p) },
                    "isa_lines": isa_lines(&isa) }),
        );
    }
    fma["fma() next to precise a*b+c on the same operands"] = Value::Object(next_to_precise);

    let shared_source = r#"#version 460
layout(local_size_x = 256) in;
layout(constant_id = 0) const uint N = 16;
shared uint s[N];
layout(std430, binding = 0) buffer B { uint o[]; };
void main() {
    uint t = gl_LocalInvocationID.x;
    for (uint i = t; i < N; i += 256u) s[i] = i * 3u + 1u;
    barrier();
    for (uint i = t; i < N; i += 256u) o[i] = s[N - 1u - i];
}
"#;
    let shared_code = compile(shared_source)?;
    let mut shared = Vec::new();
    for n in [16u32, 1000, 4096, 8192, 16384] {
        let input = vec![0u32; n as usize];
        match dispatch(c, &shared_code, &n.to_ne_bytes(), &input, n as usize) {
            Ok((out, _isa, stats)) => {
                let correct = (0..n as usize).all(|i| out[i] == (n - 1 - i as u32) * 3 + 1);
                let lds: Vec<Value> = stats
                    .iter()
                    .filter(|(k, _)| {
                        let k = k.to_ascii_lowercase();
                        k.contains("lds") || k.contains("shared")
                    })
                    .map(|(k, v)| json!({ k.clone(): v }))
                    .collect();
                shared.push(json!({ "n_words": n, "bytes": n * 4, "correct": correct, "driver_stats": lds }));
            }
            Err(e) => shared.push(json!({ "n_words": n, "bytes": n * 4, "error": e.to_string() })),
        }
    }
    Ok(json!({ "fma": fma, "shared_spec": shared }))
}

/// Builds a one-binding compute pipeline, runs one workgroup over `input` (u32 words) and returns
/// the first `read` words, the ISA text (if captured) and the driver statistics.
fn dispatch(
    c: &Context,
    code: &[u32],
    spec_bytes: &[u8],
    input: &[u32],
    read: usize,
) -> Result<(Vec<u32>, String, Vec<(String, String)>), Error> {
    let d = &c.device;
    unsafe {
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)?;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding), None)?;
        let set_layouts = [set_layout];
        let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)?;
        let map = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
        let spec = vk::SpecializationInfo::default().map_entries(&map).data(spec_bytes);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        if !spec_bytes.is_empty() {
            stage = stage.specialization_info(&spec);
        }
        let flags = if c.executable.is_some() {
            vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR
        } else {
            vk::PipelineCreateFlags::empty()
        };
        let pipeline = d
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout).flags(flags)],
                None,
            )
            .map_err(|(_, e)| e)?[0];

        let (isa, stats) = match &c.executable {
            Some(x) => capture(x, pipeline)?,
            None => (String::new(), Vec::new()),
        };

        let buffer = d.create_buffer(
            &vk::BufferCreateInfo::default().size(BUFFER_BYTES).usage(vk::BufferUsageFlags::STORAGE_BUFFER),
            None,
        )?;
        let req = d.get_buffer_memory_requirements(buffer);
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = (0..c.memory_types.memory_type_count)
            .find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && c.memory_types.memory_types[i as usize].property_flags.contains(wanted)
            })
            .ok_or("no host-visible coherent memory type")?;
        let memory = d.allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(type_index),
            None,
        )?;
        d.bind_buffer_memory(buffer, memory, 0)?;
        let mapped = d.map_memory(memory, 0, BUFFER_BYTES, vk::MemoryMapFlags::empty())? as *mut u32;
        std::ptr::copy_nonoverlapping(input.as_ptr(), mapped, input.len());

        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }];
        let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)?;
        let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts))?[0];
        let buffer_info = [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        d.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info)],
            &[],
        );
        let command_pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.family), None)?;
        let command = d.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(command_pool).command_buffer_count(1),
        )?[0];
        d.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
        d.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        d.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
        d.cmd_dispatch(command, 1, 1, 1);
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        d.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
        d.end_command_buffer(command)?;
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let commands = [command];
        d.queue_submit(c.queue, &[vk::SubmitInfo::default().command_buffers(&commands)], fence)?;
        d.wait_for_fences(&[fence], true, u64::MAX)?;
        let out = std::slice::from_raw_parts(mapped, read).to_vec();

        d.destroy_fence(fence, None);
        d.destroy_command_pool(command_pool, None);
        d.destroy_descriptor_pool(pool, None);
        d.unmap_memory(memory);
        d.destroy_buffer(buffer, None);
        d.free_memory(memory, None);
        d.destroy_pipeline(pipeline, None);
        d.destroy_pipeline_layout(layout, None);
        d.destroy_descriptor_set_layout(set_layout, None);
        d.destroy_shader_module(module, None);
        Ok((out, isa, stats))
    }
}

fn capture(
    x: &ash::khr::pipeline_executable_properties::Device,
    pipeline: vk::Pipeline,
) -> Result<(String, Vec<(String, String)>), Error> {
    unsafe {
        let info = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(0);
        let stats = x
            .get_pipeline_executable_statistics(&info)?
            .iter()
            .map(|s| {
                let value = match s.format {
                    vk::PipelineExecutableStatisticFormatKHR::BOOL32 => (s.value.b32 != 0).to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
                    _ => s.value.f64.to_string(),
                };
                (text(&s.name), value)
            })
            .collect();
        // Two-call pattern: sizes first, then the data.
        let mut count = 0u32;
        (x.fp().get_pipeline_executable_internal_representations_khr)(x.device(), &info, &mut count, std::ptr::null_mut())
            .result()?;
        let mut reps = vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
        (x.fp().get_pipeline_executable_internal_representations_khr)(x.device(), &info, &mut count, reps.as_mut_ptr())
            .result()?;
        let mut buffers: Vec<Vec<u8>> = reps.iter().map(|r| vec![0u8; r.data_size]).collect();
        for (r, buf) in reps.iter_mut().zip(buffers.iter_mut()) {
            r.p_data = buf.as_mut_ptr().cast();
        }
        (x.fp().get_pipeline_executable_internal_representations_khr)(x.device(), &info, &mut count, reps.as_mut_ptr())
            .result()?;
        let mut isa = String::new();
        for (r, buf) in reps.iter().zip(buffers.iter()) {
            let name = text(&r.name);
            if r.is_text == vk::TRUE {
                let body = CStr::from_bytes_until_nul(buf).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                if name.to_ascii_lowercase().contains("assembly") || name.to_ascii_lowercase().contains("isa") {
                    isa.push_str(&body);
                }
            }
        }
        Ok((isa, stats))
    }
}
