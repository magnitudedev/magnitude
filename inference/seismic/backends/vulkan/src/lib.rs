//! Seismic's Vulkan backend: the direct native route only. Authored GLSL compute
//! kernels are formed to SPIR-V with the statically linked glslang, sealed
//! for Seismic's numerical environment, validated with SPIRV-Tools and run on
//! one queue of the opened device through ash.
//!
//! There is no planned route: this crate implements no compiler target.

#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests start threads of their own; the code they test starts its threads through seismic-threads"
    )
)]

mod device;
pub mod direct;
mod facts;
pub mod formation;
mod graph_cost;
mod instance;
mod memory;
mod numbering;
pub mod probe;
pub mod seal;
pub mod toolchain;
pub mod topology;
pub mod transfer;
mod uniformity;

pub use device::{Device, DeviceHandle, HeapBudget, MemoryBudget, OpenError};
pub use facts::{
    Allocation, Calibration, Description, DeviceType, Facts, FormationTarget, Limits, MemoryHeap,
    MemoryLayout, MemoryType, Probed, Properties, SUBGROUP_WIDTH,
};
pub use graph_cost::{graph_cost, UNMEASURED as UNMEASURED_GRAPH_COST};

/// The programs of one opened device under its toolchain: the one way the
/// backend's code forms a program (the runtime's program former, over the
/// process's program store).
pub type VulkanPrograms = std::sync::Arc<dyn seismic_native_target::Programs<toolchain::VulkanToolchain>>;
pub use instance::LoaderError;
pub use memory::{Buffer, MemoryStores};

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

/// One tally, bounding nothing, as every store of a device of `facts`: a
/// test's, and the probe process's, which allocates only its probes' few
/// buffers and keeps no count of its stores apart.
pub fn unbounded_stores(facts: &Facts) -> MemoryStores {
    let tally: std::sync::Arc<dyn seismic_memory::entry::Store> =
        std::sync::Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam));
    let kinds = (0..facts.memory.types.len() as u32).map(|kind| (kind, tally.clone())).collect();
    MemoryStores { kinds, host: tally }
}

/// The devices meeting the floor a test exercises; none without a loader,
/// or where the runner requests another backend
/// (`seismic_lang::registry::requested_test_backend`). A run that requests
/// Vulkan refuses a failed discovery or finding no such device.
#[cfg(test)]
pub(crate) fn test_devices() -> Vec<Description> {
    seismic_lang::registry::test_device_reached("discovers the Vulkan test devices");
    let requested = seismic_lang::registry::requested_test_backend();
    if requested.is_some_and(|backend| backend != seismic_lang::registry::BackendName::Vulkan) {
        return Vec::new();
    }
    let selected = requested.is_some();
    let devices = match discover() {
        Ok(devices) => devices,
        Err(error) => {
            assert!(!selected, "requested Vulkan discovery failed: {error}");
            eprintln!("no Vulkan loader: {error}");
            return Vec::new();
        }
    };
    let devices = devices.into_iter().filter(|description| description.floor.is_ok()).collect::<Vec<_>>();
    assert!(!selected || !devices.is_empty(), "no eligible requested Vulkan device");
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
                let error = Device::open(description.facts.uuid, crate::unbounded_stores(&description.facts), &Default::default())
                    .err()
                    .expect("a device discovery reports unavailable does not open");
                eprintln!("  {error}");
                if let Some(open_reason) = error.floor_reason() {
                    assert_eq!(&open_reason, reason);
                }
                continue;
            }
            let device = Device::open(description.facts.uuid, crate::unbounded_stores(&description.facts), &Default::default()).unwrap_or_else(|error| {
                panic!("a device discovery reports available opens: {error}")
            });
            let buffer = device.allocate(1 << 20, 256, &device.own()).expect("allocation");
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
            subgroup_uniform_control_flow: true,
            maximal_reconvergence: true,
        };
        // Sealed as for a module that relies on its selections rejoining.
        let sealed = seal::seal(&compiled, declared, true, false).expect("seal");
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
            true,
            false,
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
            true,
            false,
        )
        .expect("seal");
        assert_eq!(seal::unsealed(&fp32, &[32]).expect("module"), (0, true));
        assert_eq!(seal::unsealed(&fp32, &[16]).expect("module"), (0, false));
        assert_eq!(fp32.len(), sealed.len() - 8);
        // `SubgroupUniformControlFlowKHR` (4421) is declared on the entry
        // point; without the extension on the device, or where the module
        // does not rely on it, the mode (3 words) and its extension (11) are
        // left out.
        assert!(sealed.windows(3).any(|words| words[0] == ((3 << 16) | 16) && words[2] == 4421));
        let without_uniform = seal::seal(
            &compiled,
            seal::Environment {
                subgroup_uniform_control_flow: false,
                ..declared
            },
            true,
            false,
        )
        .expect("seal");
        assert_eq!(without_uniform.len(), sealed.len() - 14);
        assert_eq!(seal::seal(&compiled, declared, false, false).expect("seal"), without_uniform);
        formation::validate(&without_rte).expect("a sealed module validates");
        formation::validate(&sealed).expect("a sealed module validates");
        formation::validate(&fp32).expect("a sealed module validates");
        formation::validate(&without_uniform).expect("a sealed module validates");
        // `fma` becomes `OpFmaKHR` (opcode 4427, six words) with its
        // capability and extension; nothing else changes.
        let bound = seal::fma_khr(&sealed).expect("fma binding");
        let fma_khr = (6 << 16) | 4427;
        assert_eq!(bound.iter().filter(|word| **word == fma_khr).count(), 1);
        assert!(bound.windows(2).any(|pair| pair == [(2 << 16) | 17, 6030]));
        // + capability (2) + extension (4) - the shorter instruction (8 -> 6).
        assert_eq!(bound.len(), sealed.len() + 2 + 4 - 2);
    }

    /// Formation holds a kernel to the execution model (`uniformity`):
    /// each kernel forms, or is refused for the violation named.
    #[test]
    fn formation_holds_kernels_to_the_execution_model() {
        // `n` is a trip count every invocation reads alike; `arguments` an
        // argument block.
        let header = "#version 460\n\
             #extension GL_KHR_shader_subgroup_basic : require\n\
             #extension GL_KHR_shader_subgroup_arithmetic : require\n\
             #extension GL_KHR_shader_subgroup_ballot : require\n\
             #extension GL_KHR_shader_subgroup_shuffle : require\n\
             #extension GL_EXT_buffer_reference : require\n\
             #extension GL_EXT_scalar_block_layout : require\n\
             #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n\
             layout(local_size_x = 64) in;\n\
             layout(std430, binding = 0) buffer b { float v[]; };\n\
             layout(push_constant) uniform k { uint n; uint64_t arguments; };\n";
        let environment = seal::Environment {
            float16: false,
            rounding_rte_32: true,
            denorm_preserve_32: false,
            subgroup_uniform_control_flow: true,
            maximal_reconvergence: false,
        };
        // Each invocation starts at its lane: they leave at different
        // iterations.
        let strided = "float s = 0.0; for (uint i = gl_SubgroupInvocationID; i < n; i += 32u) s += v[i];";
        let lane = "gl_SubgroupID * 32u + gl_SubgroupInvocationID";
        // A buffer the launch only reads, through the input block.
        let input = format!(
            "layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer {} {{ uint64_t w[]; }};\n\
             layout(buffer_reference, scalar, buffer_reference_align = 4) buffer f32 {{ float x[]; }};\n\
             f32 source() {{ return f32({}(arguments).w[0]); }}\n",
            formation::INPUT_BLOCK,
            formation::INPUT_BLOCK
        );
        // Buffers through argument words the launch may write.
        let written = format!(
            "layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer {} {{ uint64_t w[]; }};\n\
             layout(buffer_reference, scalar, buffer_reference_align = 4) buffer f32 {{ float x[]; }};\n\
             #define WORD(i) ({}(arguments).w[i])\n\
             #define BUFFER(i) f32(WORD(i))\n",
            formation::ARGUMENT_BLOCK,
            formation::ARGUMENT_BLOCK
        );
        // An address table of a port the launch may write through (a slab
        // view's slabs), which it only reads (declared writable here only to
        // try a write).
        let table = format!(
            "layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer {} {{ uint64_t w[]; }};\n\
             layout(buffer_reference, scalar, buffer_reference_align = 8) buffer u64 {{ uint64_t a[]; }};\n\
             #define TABLE(i) ({}(arguments).w[i])\n",
            formation::ADDRESS_TABLE_BLOCK,
            formation::ADDRESS_TABLE_BLOCK
        );
        // A row of a table that a constant `table` of zero leaves out, as
        // `projection_row` does: a branch the module never takes.
        let row = "struct Q { uint64_t table; uint m; };\n\
             uint row(Q q) { if (q.table == 0ul) return q.m; return floatBitsToUint(f32(q.table).x[q.m]); }\n";
        // The device's tables, which no launch writes (declared writable
        // here only to try a write).
        let tables = uniformity::TABLES_BLOCK;
        let tables = format!(
            "layout(buffer_reference, std430, buffer_reference_align = 16) buffer {tables} {{ float t[]; }};\n\
             #define TABLES {tables}(arguments)\n"
        );
        // A 32-lane hardware join, as the prelude defines it.
        let join = "bool seismic_hardware_any(const bool value) { return value; }\n";
        let join_all = "bool seismic_hardware_all(const bool value) { return value; }\n";
        // The input block, read beside the argument block.
        let input_block = format!(
            "layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer {} {{ uint64_t w[]; }};\n",
            formation::INPUT_BLOCK
        );
        let input_word = |offset: &str| format!("{}(arguments{offset}).w[0]", formation::INPUT_BLOCK);
        let address_store = "a store of a buffer's address, or of a value derived from one, to workgroup or device memory";
        let mixing = "an instruction combining the local or global invocation id with the lane order of subgroups";
        for (body, refused) in [
            (
                "void main() { float s = 0.0; for (uint i = 0u; i < n; i++) s += v[i]; v[gl_SubgroupID] = subgroupAdd(s); }".to_owned(),
                None,
            ),
            // The strided loop written with a uniform trip count.
            (
                "void main() { float s = 0.0; for (uint base = 0u; base < n; base += 32u) { uint i = base + gl_SubgroupInvocationID; s += i < n ? v[min(i, n - 1u)] : 0.0; } v[gl_SubgroupID] = subgroupAdd(s); }".to_owned(),
                None,
            ),
            (
                format!("void main() {{ {strided} v[gl_SubgroupID] = subgroupAdd(s); }}"),
                Some("OpGroupNonUniformFAdd after a lane-varying loop"),
            ),
            (
                "void main() { float s = v[0]; if (gl_SubgroupInvocationID < 7u) s = subgroupAdd(v[gl_SubgroupInvocationID]); v[1] = s; }".to_owned(),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A selection header's instructions run before its branch.
            (
                "void main() { v[gl_SubgroupID] = subgroupAdd(v[1]); if (gl_SubgroupInvocationID == 0u) v[1] = 2.0; }".to_owned(),
                None,
            ),
            (
                "void main() { if (gl_SubgroupInvocationID == 0u) v[1] = 2.0; barrier(); if (gl_SubgroupInvocationID == 0u) v[2] = v[1]; }".to_owned(),
                None,
            ),
            // The selection rejoins at its merge.
            (
                "void main() { if (gl_SubgroupInvocationID == 0u) v[1] = 2.0; v[2u + gl_SubgroupID] = subgroupAdd(v[gl_SubgroupInvocationID]); }".to_owned(),
                None,
            ),
            (
                "void main() { if (gl_SubgroupInvocationID < 3u) return; v[0] = subgroupAdd(v[1]); }".to_owned(),
                Some("OpGroupNonUniformFAdd after a lane-dependent exit"),
            ),
            (
                "void main() { if (gl_SubgroupInvocationID < 3u) { v[0] = 1.0; barrier(); } v[1] = 2.0; }".to_owned(),
                Some("OpControlBarrier inside a divergent selection"),
            ),
            // Every invocation reaches a barrier after the loop.
            (format!("void main() {{ {strided} barrier(); v[{lane}] = s; }}"), None),
            // A legal workgroup barrier makes control uniform again; a
            // subgroup barrier does not, and is itself refused there.
            (format!("void main() {{ {strided} barrier(); v[gl_SubgroupID] = subgroupAdd(s); }}"), None),
            (
                format!("void main() {{ {strided} subgroupBarrier(); v[gl_SubgroupID] = subgroupAdd(s); }}"),
                Some("OpGroupNonUniformFAdd after a lane-varying loop"),
            ),
            (
                format!("void main() {{ {strided} subgroupBarrier(); v[gl_SubgroupID] = s; }}"),
                Some("OpControlBarrier of a subgroup after a lane-varying loop"),
            ),
            // Where control is uniform, a subgroup barrier is legal.
            ("void main() { v[0] = 1.0; subgroupBarrier(); v[gl_SubgroupID] = subgroupAdd(v[1]); }".to_owned(), None),
            // Not after a lane-dependent exit, where the barrier is refused.
            (
                "void main() { if (gl_SubgroupInvocationID < 3u) return; barrier(); v[0] = subgroupAdd(v[1]); }".to_owned(),
                Some("OpControlBarrier after a lane-dependent exit"),
            ),
            (
                "void main() { v[0] = subgroupShuffle(v[gl_LocalInvocationIndex], 3u); }".to_owned(),
                Some("OpGroupNonUniformShuffle on a value derived from the local or global invocation id"),
            ),
            (format!("void main() {{ v[0] = subgroupShuffle(v[{lane}], 3u); }}"), None),
            // The reduction is inlined into the entry point.
            (
                format!("float total(float x) {{ return subgroupAdd(x); }}\nvoid main() {{ {strided} v[0] = total(s); }}"),
                Some("OpGroupNonUniformFAdd after a lane-varying loop"),
            ),
            // A call rejoins at its return, as a selection at its merge.
            (
                "float pick(float x) { if (gl_SubgroupInvocationID < 3u) return x; return v[1]; }\nvoid main() { v[2] = subgroupAdd(pick(v[0])); }".to_owned(),
                None,
            ),
            // A broadcast's lane must be uniform; a shuffle from a uniform
            // lane gives a uniform value.
            (
                format!("void main() {{ float b = subgroupBroadcast(v[{lane}], gl_SubgroupInvocationID & 1u); if (b > 0.0) v[gl_SubgroupID] = subgroupAdd(v[1]); }}"),
                Some("OpGroupNonUniformBroadcast with a divergent id"),
            ),
            (
                format!("void main() {{ float b = subgroupShuffle(v[{lane}], 3u); if (b > 0.0) v[gl_SubgroupID] = subgroupAdd(v[1]); }}"),
                None,
            ),
            // Workgroup memory read where a store may race: invocations of a
            // subgroup agree on it only in lockstep.
            (
                format!("shared float red[64];\nvoid main() {{ uint t = {lane}; red[t] = v[t]; if (gl_SubgroupInvocationID < 16u) red[t] += red[t + 16u]; float total = red[gl_SubgroupID * 32u]; if (total > 0.0) v[0] = subgroupAdd(v[1]); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            (
                format!("shared float red[64];\nvoid main() {{ uint t = {lane}; red[t] = v[t]; if (gl_SubgroupInvocationID < 16u) red[t] += red[t + 16u]; barrier(); float total = red[gl_SubgroupID * 32u]; if (total > 0.0) v[0] = subgroupAdd(v[1]); }}"),
                None,
            ),
            // So is device memory the launch writes, until a barrier whose
            // semantics order it.
            (
                format!("void main() {{ v[{lane}] = 1.0; barrier(); if (v[0] > 0.0) {{ float s = subgroupAdd(v[2]); barrier(); v[1] = s; }} }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A barrier never orders another workgroup's stores, so a
            // buffer the launch writes is read apart even past one.
            (
                format!("void main() {{ v[{lane}] = 1.0; memoryBarrierBuffer(); barrier(); if (v[0] > 0.0) v[1] = subgroupAdd(v[2]); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // One invocation stages the word through workgroup memory, which
            // every invocation then reads alike.
            (
                "shared float word;\nvoid main() { if (gl_LocalInvocationIndex == 0u) word = v[0]; barrier(); if (word > 0.0) v[1u + gl_WorkGroupID.x] = subgroupAdd(v[2]); }".to_owned(),
                None,
            ),
            // A loop holding a barrier that runs at least once (its first
            // test holds on entry) has no path past the barrier: a read
            // before it and a store after it, of the same word, do not race.
            // With a bound the launch sets, it may run none, and they do.
            (
                "shared float s[2];\nvoid main() { if (gl_LocalInvocationIndex == 0u) s[0] = v[0]; barrier(); float t = s[0]; for (uint i = 0u; i < 3u; ++i) barrier(); s[0] = v[1]; if (t > 0.0) v[2u + gl_SubgroupID] = subgroupAdd(v[1]); }".to_owned(),
                None,
            ),
            (
                "shared float s[2];\nvoid main() { if (gl_LocalInvocationIndex == 0u) s[0] = v[0]; barrier(); float t = s[0]; for (uint i = 0u; i < n; ++i) barrier(); s[0] = v[1]; if (t > 0.0) v[2u + gl_SubgroupID] = subgroupAdd(v[1]); }".to_owned(),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A store of the workgroup on a path that never comes before
            // the read, nor in its barrier interval, keeps nothing the read
            // sees; one that may come before it keeps the local id of the
            // invocation that placed it.
            (
                format!("{written}void main() {{ if (gl_WorkGroupID.x == 0u) {{ BUFFER(1).x[gl_WorkGroupID.x] = float(gl_LocalInvocationIndex); return; }} barrier(); BUFFER(2).x[{lane}] = BUFFER(1).x[{lane}]; }}"),
                None,
            ),
            (
                format!("{written}void main() {{ if (gl_WorkGroupID.x == 0u) BUFFER(1).x[gl_WorkGroupID.x] = float(gl_LocalInvocationIndex); barrier(); BUFFER(2).x[{lane}] = BUFFER(1).x[{lane}]; }}"),
                Some(mixing),
            ),
            // Stored where its local id places it, a value is a function of
            // its place: read by lane, it holds nothing of how local ids
            // meet lanes (staging).
            (
                format!("{written}void main() {{ BUFFER(1).x[gl_LocalInvocationIndex] = float(gl_LocalInvocationIndex); barrier(); BUFFER(2).x[{lane}] = BUFFER(1).x[{lane}]; }}"),
                None,
            ),
            // What another workgroup stores derives from none of the
            // reader's places; a store of its own workgroup before the read,
            // which no barrier here orders, may be its own subgroup's. (The
            // word is read back: a workgroup variable nothing reads is
            // removed with its stores before the check.)
            (
                "shared float word;\nvoid main() { v[1u + gl_WorkGroupID.x] = subgroupAdd(v[2]); barrier(); if (gl_LocalInvocationIndex == 0u) word = v[0]; barrier(); v[1] = word; }".to_owned(),
                Some(mixing),
            ),
            // A store and a load of workgroup memory in one barrier interval
            // race, though no path joins them: subgroups run side by side.
            (
                "shared float tile[1];\nvoid main() { if (gl_SubgroupID == 0u) tile[0] = v[0]; else if (tile[0] > 0.0) v[1] = subgroupAdd(v[2]); }".to_owned(),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // The tables are read alike, and never written.
            (format!("{tables}void main() {{ v[{lane}] = 1.0; if (TABLES.t[0] > 0.0) v[1] = subgroupAdd(v[2]); }}"), None),
            (format!("{tables}void main() {{ TABLES.t[0] = 1.0; }}"), Some("a write to memory the launch only reads")),
            // Nothing in the launch writes a buffer it only reads, which it
            // may not write.
            (
                format!("{input}void main() {{ v[{lane}] = 1.0; if (source().x[0] > 0.0) v[1] = subgroupAdd(v[2]); }}"),
                None,
            ),
            (format!("{input}void main() {{ source().x[0] = 1.0; }}"), Some("a write to memory the launch only reads")),
            // A buffer the module never stores through is one the launch
            // only reads (an earlier launch's result or scratch); one it
            // stores through races its loads.
            (
                format!("{written}void main() {{ BUFFER(1).x[{lane}] = 1.0; if (BUFFER(0).x[0] > 0.0) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                None,
            ),
            (
                format!("{written}void main() {{ BUFFER(0).x[{lane}] = 1.0; if (BUFFER(0).x[0] > 0.0) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A store at a buffer's address plus an offset read from memory
            // (a gathered GEMV's products, at offsets its table holds) stays
            // in that buffer: the table it read is still one the launch
            // only reads.
            (
                format!("{written}void main() {{ uint at = floatBitsToUint(BUFFER(0).x[1u + {lane}]); f32(WORD(1) + uint64_t(at) * 4ul).x[0] = 1.0; if (BUFFER(0).x[0] > 0.0) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                None,
            ),
            // An address read from memory may point anywhere, at any offset.
            (
                format!("{written}layout(buffer_reference, scalar, buffer_reference_align = 8) buffer u64 {{ uint64_t a[]; }};\nvoid main() {{ f32(u64(WORD(0)).a[0] + uint64_t({lane}) * 4ul).x[0] = 1.0; if (BUFFER(2).x[0] > 0.0) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A value chosen by control derived from the local id, or read
            // from memory such a value is stored to, derives from it.
            (
                "void main() { float t = 0.0; if ((gl_LocalInvocationIndex & 31u) == 0u) t = 1.0; v[gl_SubgroupID] = subgroupAdd(t); }".to_owned(),
                Some("OpGroupNonUniformFAdd on a value derived from the local or global invocation id"),
            ),
            (
                format!("shared uint slot[64];\nvoid main() {{ slot[{lane}] = gl_LocalInvocationIndex; barrier(); uint j = slot[{lane}]; v[0] = subgroupShuffle(v[1], j & 31u); }}"),
                Some("OpGroupNonUniformShuffle on a value derived from the local or global invocation id"),
            ),
            // So is one read from a buffer such a value is stored to.
            (
                "void main() { v[gl_WorkGroupID.x] = float(gl_LocalInvocationIndex); uint j = uint(v[0]); v[1] = subgroupShuffle(v[2], j & 31u); }".to_owned(),
                Some("OpGroupNonUniformShuffle on a value derived from the local or global invocation id"),
            ),
            // A value of the lane order, stored where the local id says, or
            // read where the lane order says and stored where the local id
            // says, depends on how the device maps one onto the other.
            (
                "void main() { v[gl_LocalInvocationIndex] = subgroupExclusiveAdd(float(n)); }".to_owned(),
                Some("an instruction combining the local or global invocation id with the lane order of subgroups"),
            ),
            (
                format!("{written}void main() {{ BUFFER(1).x[gl_LocalInvocationIndex] = BUFFER(0).x[{lane}]; }}"),
                Some("an instruction combining the local or global invocation id with the lane order of subgroups"),
            ),
            // Indexed by the lane order alone, it does not; nor does one
            // value for the whole subgroup, stored where the local id says.
            (format!("{written}void main() {{ BUFFER(1).x[{lane}] = BUFFER(0).x[{lane}] + subgroupExclusiveAdd(1.0); }}"), None),
            // One value for the whole subgroup shows which subgroup combined
            // it where its operand derives from the subgroup, or differs
            // otherwise than by lane.
            (
                format!("{written}void main() {{ BUFFER(1).x[gl_LocalInvocationIndex] = subgroupAdd(BUFFER(0).x[{lane}]); }}"),
                Some(mixing),
            ),
            ("void main() { v[gl_LocalInvocationIndex] = subgroupAdd(float(gl_SubgroupID)) / 32.0; }".to_owned(), Some(mixing)),
            // Where it differs by lane alone, every full subgroup combines
            // the same values.
            ("void main() { v[gl_LocalInvocationIndex] = subgroupAdd(float(gl_SubgroupInvocationID)); }".to_owned(), None),
            // But one value every invocation reads, past a barrier from the
            // store, holds no lane order, whoever stored it.
            (
                "shared uint word;\nvoid main() { if (gl_LocalInvocationIndex == 0u) word = n; barrier(); v[gl_SubgroupID] = subgroupAdd(float(word)); }".to_owned(),
                None,
            ),
            // A cooperative-matrix load's pointer must be uniform.
            (
                "#extension GL_KHR_cooperative_matrix : require\n\
                 #extension GL_KHR_memory_scope_semantics : require\n\
                 shared float tile[512];\n\
                 void main() { coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c; coopMatLoad(c, tile, gl_SubgroupInvocationID * 16u, 16u, gl_CooperativeMatrixLayoutRowMajor); coopMatStore(c, v, gl_SubgroupID * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor); }".to_owned(),
                Some("OpCooperativeMatrixLoadKHR with a divergent pointer"),
            ),
            // So must a store's stride.
            (
                "#extension GL_KHR_cooperative_matrix : require\n\
                 #extension GL_KHR_memory_scope_semantics : require\n\
                 shared float tile[256];\n\
                 void main() { coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c; coopMatLoad(c, tile, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor); coopMatStore(c, v, gl_SubgroupID * 256u, 16u + gl_SubgroupInvocationID, gl_CooperativeMatrixLayoutRowMajor); }".to_owned(),
                Some("OpCooperativeMatrixStoreKHR with a divergent stride"),
            ),
            // A quad broadcast's index and a read invocation's lane must be
            // uniform.
            (
                format!("#extension GL_KHR_shader_subgroup_quad : require\nvoid main() {{ v[{lane}] = subgroupQuadBroadcast(v[{lane}], gl_SubgroupInvocationID & 3u); }}"),
                Some("OpGroupNonUniformQuadBroadcast with a divergent index"),
            ),
            (
                format!("#extension GL_ARB_shader_ballot : require\nvoid main() {{ v[{lane}] = readInvocationARB(v[{lane}], gl_SubgroupInvocationID & 1u); }}"),
                Some("OpSubgroupReadInvocationKHR with a divergent id"),
            ),
            // An undefined value adds nothing to a phi of it: a kernel whose
            // result depends on one is wrong whatever it holds.
            ("void main() { float s; if (n > 3u) s = 1.0; if (s > 0.0) v[gl_SubgroupID] = subgroupAdd(v[1]); }".to_owned(), None),
            // Nor to an array written element by element, which promotion
            // leaves in memory (`attention_absorb`'s keys scored).
            (
                format!("{join}void main() {{ bool scored[4]; for (uint j = 0u; j < 4u; ++j) scored[j] = seismic_hardware_any(j < n && ((n >> j) & 1u) != 0u); for (uint j = 0u; j < 4u; ++j) if (scored[j]) v[gl_SubgroupID * 4u + j] = subgroupAdd(v[{lane}]); }}"),
                None,
            ),
            // Nor to the value a function returns inside a loop, which
            // merging its returns leaves unset on the loop's other exit
            // (`ss_piece_at`, an early exit before barriers).
            (
                "bool piece_at(int piece, out int local) { local = 0; for (uint i = 0u; i < n; ++i) { const int count = int(i) + 1; if (piece < count) { local = piece; return true; } piece -= count; } return false; }\nvoid main() { int local; if (!piece_at(int(gl_WorkGroupID.x), local)) return; barrier(); v[gl_LocalInvocationIndex] = float(local); }".to_owned(),
                None,
            ),
            // A 32-lane hardware join of a value that differs within the
            // subgroup is refused (one of a value held alike forms, below).
            (
                format!("{join}void main() {{ if (seismic_hardware_any(v[{lane}] > 0.0)) v[gl_SubgroupID] = subgroupAdd(v[1]); }}"),
                Some("`seismic_hardware_any` of a value that differs within the hardware subgroup"),
            ),
            // An instruction the check does not read is refused, not
            // malformed.
            (
                "layout(binding = 1, r32f) uniform image2D picture;\nvoid main() { imageStore(picture, ivec2(0), vec4(v[0])); }".to_owned(),
                Some("opcode 99, which the execution-model check does not read"),
            ),
            // A value a lane-varying loop computes alike in every iteration
            // is uniform after it.
            (
                "void main() { float s = 0.0; uint u = 0u; for (uint i = gl_SubgroupInvocationID; ; i += 32u) { u = n + 1u; if (i >= n) break; s += v[i]; } barrier(); if (u > 3u) v[gl_SubgroupID] = subgroupAdd(s); }".to_owned(),
                None,
            ),
            // A phi that copies a value is that value: promotion leaves one
            // of `first` in the lane-varying loop that only reads it, which
            // that loop carries out unchanged (`gated_delta_step`'s rows).
            (
                "void main() { for (uint first = 0u; first < n; first += 8u) { for (uint i = gl_SubgroupInvocationID; i < 8u; i += 32u) { float s = 0.0; for (uint k = 0u; k < 4u; k++) s += v[k]; v[first + i] = s; } barrier(); v[first + gl_SubgroupID] = subgroupAdd(1.0); } }".to_owned(),
                None,
            ),
            // One the loop changes is carried out divergent.
            (
                "void main() { for (uint first = 0u; first < n; first += 8u) { uint c = first; for (uint i = gl_SubgroupInvocationID; i < 8u; i += 32u) { float s = 0.0; for (uint k = 0u; k < 4u; k++) s += v[k]; v[first + i] = s; c++; } barrier(); if (c > first) v[first + gl_SubgroupID] = subgroupAdd(1.0); } }".to_owned(),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A variable holds what the stores reaching a load put in it,
            // and a struct's members apart.
            (
                format!("void main() {{ float a[4] = float[4](0.0, 0.0, 0.0, 0.0); float k = a[n & 3u]; if (k == 0.0) v[gl_SubgroupID] = subgroupAdd(v[1]); a[n & 3u] = v[{lane}]; v[2] = a[1]; }}"),
                None,
            ),
            (
                format!("struct P {{ float a; uint m; }};\nfloat f(P p) {{ float s = 0.0; for (uint i = 0u; i < p.m; i++) s += v[i]; return s + p.a; }}\nvoid main() {{ v[gl_SubgroupID] = subgroupAdd(f(P(v[{lane}], n))); }}"),
                None,
            ),
            // An address computed from two buffers' addresses names both,
            // however it combines them: a store through it writes word 1's
            // buffer, which the launch then reads apart.
            (
                format!("{written}void main() {{ f32(WORD(0) + (WORD(1) - WORD(0)) * uint64_t(n)).x[{lane}] = 1.0; if (BUFFER(1).x[0] > 0.0) BUFFER(0).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // No address, nor a delta between addresses, is stored to
            // device or workgroup memory.
            (
                format!("{written}layout(buffer_reference, scalar, buffer_reference_align = 8) buffer u64 {{ uint64_t a[]; }};\nvoid main() {{ u64(WORD(1)).a[{lane}] = WORD(2) - WORD(0); }}"),
                Some(address_store),
            ),
            (
                format!("{written}shared uint64_t at[64];\nvoid main() {{ at[{lane}] = WORD(0); barrier(); BUFFER(1).x[{lane}] = float(uint(at[{lane}])); }}"),
                Some(address_store),
            ),
            (
                format!("{written}{input_block}void main() {{ BUFFER(1).x[{lane}] = float({}); }}", input_word("")),
                Some(address_store),
            ),
            // The launch's words, read through the arguments past the
            // addresses, carry none.
            (format!("{written}{input_block}void main() {{ BUFFER(1).x[{lane}] = float({}); }}", input_word(" + 24ul")), None),
            // A workgroup barrier needs every subgroup of the workgroup:
            // not under control, nor after an exit, steered by the
            // subgroup's id.
            (
                "void main() { if (gl_SubgroupID == 1u) return; barrier(); v[gl_WorkGroupID.x] = 1.0; }".to_owned(),
                Some("OpControlBarrier of the workgroup under control that differs between subgroups"),
            ),
            (
                "void main() { if (gl_SubgroupID == 0u) { v[0] = 1.0; barrier(); } else { v[1] = 1.0; barrier(); } }".to_owned(),
                Some("OpControlBarrier of the workgroup under control that differs between subgroups"),
            ),
            // A count every subgroup sums alike from its lanes steers the
            // whole workgroup alike.
            (
                "void main() { uint total = subgroupAdd(gl_SubgroupInvocationID); if (n >= total) return; barrier(); v[gl_LocalInvocationIndex] = 1.0; }".to_owned(),
                None,
            ),
            // An atomic store's value and its pointer combine the orders as
            // any store's do.
            (
                "#extension GL_KHR_memory_scope_semantics : require\nlayout(std430, binding = 1) buffer c { uint u[]; };\nvoid main() { atomicStore(u[gl_LocalInvocationIndex], gl_SubgroupInvocationID, gl_ScopeDevice, gl_StorageSemanticsNone, gl_SemanticsRelaxed); }".to_owned(),
                Some(mixing),
            ),
            (
                format!("#extension GL_KHR_memory_scope_semantics : require\nlayout(std430, binding = 1) buffer c {{ uint u[]; }};\nvoid main() {{ atomicStore(u[{lane}], gl_SubgroupInvocationID, gl_ScopeDevice, gl_StorageSemanticsNone, gl_SemanticsRelaxed); }}"),
                None,
            ),
            // Nor does a variable before it is written.
            ("void main() { float s[2]; if (s[n & 1u] > 0.0) v[gl_SubgroupID] = subgroupAdd(v[1]); }".to_owned(), None),
            // A kernel's own function under the joins' prefix is not one.
            (
                "uint seismic_hardware_twice(const uint value) { return value * 2u; }\nvoid main() { v[gl_SubgroupID] = float(seismic_hardware_twice(n)); }".to_owned(),
                Some("`seismic_hardware_twice`, named as a 32-lane hardware join"),
            ),
            // A load of memory the launch only reads, at an address the same
            // in every iteration, is the same in every iteration of a
            // lane-varying loop; one of memory it writes is not.
            (
                format!("{input}void main() {{ float s = 0.0; float w = 0.0; for (uint i = gl_SubgroupInvocationID; ; i += 32u) {{ w = source().x[0]; if (i >= n) break; s += v[i]; }} barrier(); if (w > 0.0) v[gl_SubgroupID] = subgroupAdd(s); }}"),
                None,
            ),
            (
                "void main() { float s = 0.0; float w = 0.0; for (uint i = gl_SubgroupInvocationID; ; i += 32u) { w = v[0]; if (i >= n) break; s += v[i]; } barrier(); if (w > 0.0) v[gl_SubgroupID] = subgroupAdd(s); }".to_owned(),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // Inside a lane-varying loop, neither a subgroup operation nor
            // a barrier has its invocations together.
            (
                "void main() { for (uint i = gl_SubgroupInvocationID; i < n; i += 32u) v[i] = subgroupAdd(v[i]); }".to_owned(),
                Some("OpGroupNonUniformFAdd inside a lane-varying loop"),
            ),
            (
                "void main() { for (uint i = gl_SubgroupInvocationID; i < n; i += 32u) { v[i] = 1.0; barrier(); } }".to_owned(),
                Some("OpControlBarrier inside a lane-varying loop"),
            ),
            // A loop left under control that differs between subgroups
            // carries out values that do: subgroups leave it at different
            // iterations.
            (
                "void main() { uint c = 0u; for (;;) { if (gl_SubgroupID + c >= n) break; c++; } if (c > 1u) barrier(); }".to_owned(),
                Some("OpControlBarrier of the workgroup under control that differs between subgroups"),
            ),
            (
                "void main() { uint c = 0u; for (;;) { if (gl_SubgroupID + c >= n) break; c++; } v[gl_LocalInvocationIndex] = float(c); }".to_owned(),
                Some(mixing),
            ),
            // One left under a join of the whole subgroup's lanes carries
            // out values every subgroup holds alike.
            (
                format!("#extension GL_KHR_shader_subgroup_vote : require\n{join_all}void main() {{ uint c = 0u; for (;;) {{ if (seismic_hardware_all(subgroupAll(gl_SubgroupInvocationID + c >= n))) break; c++; }} barrier(); if (c > 1u) barrier(); v[gl_LocalInvocationIndex] = float(c); }}"),
                None,
            ),
            // The orders combine through the control a store runs under, and
            // through workgroup memory: a local id placed by lane order is
            // another invocation's. A lane placed by lane order is a
            // function of its place, whoever placed it, so reading it by
            // local id combines nothing.
            ("void main() { if (gl_SubgroupInvocationID == 0u) v[gl_LocalInvocationIndex] = 1.0; }".to_owned(), Some(mixing)),
            (
                format!("shared float lanes[64];\nvoid main() {{ lanes[{lane}] = float(gl_LocalInvocationIndex); barrier(); v[gl_LocalInvocationIndex] = lanes[gl_LocalInvocationIndex]; }}"),
                Some(mixing),
            ),
            (
                format!("shared float lanes[64];\nvoid main() {{ lanes[{lane}] = float(gl_SubgroupInvocationID); barrier(); v[gl_LocalInvocationIndex] = lanes[gl_LocalInvocationIndex]; }}"),
                None,
            ),
            // A store races a load it reaches across a loop's back edge, in
            // the barrier interval the next iteration starts in.
            (
                format!("shared float t[64];\nvoid main() {{ for (uint k = 0u; k < n; k++) {{ float w = t[0]; barrier(); if (w > 0.0) v[gl_SubgroupID] = subgroupAdd(v[1]); t[{lane}] = 1.0; }} }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A workgroup barrier whose semantics do not order workgroup
            // memory rejoins the workgroup, but ends no barrier interval.
            (
                format!("#extension GL_KHR_memory_scope_semantics : require\nshared float red[64];\nvoid main() {{ uint t = {lane}; red[t] = v[t]; controlBarrier(gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease); float total = red[gl_SubgroupID * 32u]; if (total > 0.0) v[0] = subgroupAdd(v[1]); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A broadcast's and a read invocation's lane may be any uniform
            // value.
            (format!("void main() {{ v[gl_SubgroupID] = subgroupBroadcast(v[{lane}], n & 31u); }}"), None),
            (
                format!("#extension GL_ARB_shader_ballot : require\nvoid main() {{ v[gl_SubgroupID] = readInvocationARB(v[{lane}], n & 31u); }}"),
                None,
            ),
            // A cooperative-matrix load and store of uniform operands.
            (
                "#extension GL_KHR_cooperative_matrix : require\n\
                 #extension GL_KHR_memory_scope_semantics : require\n\
                 shared float tile[256];\n\
                 void main() { coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c; coopMatLoad(c, tile, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor); coopMatStore(c, v, gl_SubgroupID * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor); }".to_owned(),
                None,
            ),
            // A constant that decides a `&&` alone decides it whatever the
            // other operand holds: false and a divergent condition take
            // the one way.
            (
                format!("{written}void gate(uint k) {{ bool b = BUFFER(1).x[{lane}] > 0.0; if (k != 0u && b) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}\nvoid main() {{ gate(0u); }}"),
                None,
            ),
            (
                format!("{written}void gate(uint k) {{ bool b = BUFFER(1).x[{lane}] > 0.0; if (k != 0u && b) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}\nvoid main() {{ gate(1u); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A branch on a constant, passed in a struct through a call,
            // takes one way: the row read at address 0 the other would take
            // is never read, so it reads nothing the launch writes.
            (
                format!("{written}{row}void main() {{ BUFFER(1).x[{lane}] = 1.0; if (row(Q(0ul, n)) > 3u) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                None,
            ),
            // Where the table is a launch's value, either way may be taken.
            (
                format!("{written}{row}void main() {{ BUFFER(1).x[{lane}] = 1.0; if (row(Q(uint64_t(n), n)) > 3u) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // So it may where it is a specialization constant, which
            // pipeline creation sets after the check.
            (
                format!("{written}layout(constant_id = 7) const uint S = 0u;\n{row}void main() {{ BUFFER(1).x[{lane}] = 1.0; if (row(Q(uint64_t(S), n)) > 3u) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A struct passed to a call under divergent control is stored to
            // its parameter there, as uniform as what is stored: the callee's
            // loop over its count is not lane-varying.
            (
                "struct P { uint parts; float scale; };\nfloat total(P p) { float s = 0.0; for (uint i = 0u; i < p.parts; i++) s += v[i]; return s * p.scale; }\nvoid main() { float t = 0.0; if (gl_SubgroupInvocationID < 7u) t = total(P(n, 2.0)); v[gl_SubgroupID] = subgroupAdd(t); }".to_owned(),
                None,
            ),
            // But a variable stored under divergent control differs where
            // the invocations that stored and those that did not meet.
            (
                "struct C { uint count; float pad; };\nvoid main() { C c = C(0u, 0.0); if (gl_SubgroupInvocationID < 7u) c.count = n; float s = 0.0; for (uint i = 0u; i < c.count; i++) s += v[i]; v[gl_SubgroupID] = subgroupAdd(s); }".to_owned(),
                Some("OpGroupNonUniformFAdd after a lane-varying loop"),
            ),
            // What a lane-varying loop stores to a variable kept in memory
            // differs where the loop is left, though each store is uniform.
            (
                "struct S { uint a; float b; };\nvoid main() { S s = S(0u, 0.0); for (uint i = gl_SubgroupInvocationID; ; i += 32u) { s.a += 1u; if (i >= n) break; } barrier(); if (s.a > 3u) v[gl_SubgroupID] = subgroupAdd(v[1]); }".to_owned(),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // A condition undefined along some of the ways reaching it is
            // any value there, so it takes every way: not folded to the
            // defined one, the way it would not take checked too. (Its
            // divergence is the defined value's: an undefined value adds
            // nothing to what it joins.)
            (
                "void main() { bool f; if (gl_SubgroupInvocationID < 7u) f = true; if (!f) v[0] = subgroupShuffle(v[gl_LocalInvocationIndex], 3u); }".to_owned(),
                Some("OpGroupNonUniformShuffle on a value derived from the local or global invocation id"),
            ),
            // An array of structs keeps its elements' members apart: a
            // member written at a divergent index leaves the others as they
            // were.
            (
                format!("struct E {{ uint count; float value; }};\nvoid main() {{ E e[2]; e[0].count = n; e[1].count = n; e[gl_SubgroupInvocationID & 1u].value = v[{lane}]; float s = 0.0; for (uint i = 0u; i < e[n & 1u].count; i++) s += v[i]; v[gl_SubgroupID] = subgroupAdd(s); }}"),
                None,
            ),
            // The member written there differs, in every element.
            (
                "struct E { uint count; float value; };\nvoid main() { E e[2]; e[0].count = n; e[1].count = n; e[gl_SubgroupInvocationID & 1u].count = 0u; float s = 0.0; for (uint i = 0u; i < e[0].count; i++) s += v[i]; v[gl_SubgroupID] = subgroupAdd(s); }".to_owned(),
                Some("OpGroupNonUniformFAdd after a lane-varying loop"),
            ),
            // A condition on an address holds no address, so it may be
            // stored anywhere; a value computed from the address may not.
            (
                format!("{written}shared bool aligned;\nvoid main() {{ if (gl_LocalInvocationIndex == 0u) aligned = (WORD(1) & 15ul) == 0ul; barrier(); BUFFER(1).x[{lane}] = aligned ? 1.0 : 0.0; }}"),
                None,
            ),
            (
                format!("{written}shared uint misaligned;\nvoid main() {{ if (gl_LocalInvocationIndex == 0u) misaligned = uint(WORD(1) & 15ul); barrier(); BUFFER(1).x[{lane}] = float(misaligned); }}"),
                Some(address_store),
            ),
            // An address read from an address table points into its port's
            // own bytes: a store through it writes no other buffer, which
            // the launch still reads alike.
            (
                format!("{written}{table}void main() {{ f32(u64(TABLE(2)).a[n]).x[{lane}] = 1.0; if (BUFFER(0).x[0] > 0.0) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                None,
            ),
            // The port's bytes it writes are read apart.
            (
                format!("{written}{table}void main() {{ f32(u64(TABLE(2)).a[n]).x[{lane}] = 1.0; if (f32(u64(TABLE(2)).a[n]).x[0] > 0.0) BUFFER(1).x[gl_SubgroupID] = subgroupAdd(2.0); }}"),
                Some("OpGroupNonUniformFAdd inside a divergent selection"),
            ),
            // The table itself the launch only reads.
            (format!("{table}void main() {{ u64(TABLE(2)).a[0] = 0ul; }}"), Some("a write to memory the launch only reads")),
        ] {
            match formation::compile(&format!("{header}{body}\n"), "main", environment) {
                Ok(_) => assert!(refused.is_none(), "{body}: formed, expected {refused:?}"),
                Err(seismic_native_target::NativeCompilationError::ToolchainFailure(message)) => {
                    assert!(refused.is_some_and(|refused| message.contains(refused)), "{body}: {message}")
                }
                Err(error) => panic!("{body}: {error:?}"),
            }
        }
        // A refusal names its first instance, and where what it refuses
        // comes from.
        match formation::compile(
            &format!("{header}void main() {{ v[0] = subgroupShuffle(v[gl_LocalInvocationIndex], 3u); }}\n"),
            "main",
            environment,
        ) {
            Err(seismic_native_target::NativeCompilationError::ToolchainFailure(message)) => {
                for part in [
                    "1 × OpGroupNonUniformShuffle on a value derived from the local or global invocation id in `main`, first at %",
                    " = OpGroupNonUniformShuffle, the local id from %",
                    " = OpLoad (`gl_LocalInvocationIndex`)",
                ] {
                    assert!(message.contains(part), "{part}: {message}");
                }
            }
            other => panic!("expected the shuffle refused, got {other:?}"),
        }
        // A read a store may race names that store; ordered by a barrier, the
        // same read is uniform (the race cases above).
        match formation::compile(
            &format!("{header}shared float red[64];\nvoid main() {{ uint t = {lane}; red[t] = v[t]; float total = red[gl_SubgroupID * 32u]; if (total > 0.0) v[0] = subgroupAdd(v[1]); }}\n"),
            "main",
            environment,
        ) {
            Err(seismic_native_target::NativeCompilationError::ToolchainFailure(message)) => {
                for part in [" = OpLoad (`red`), which OpStore %", " may race"] {
                    assert!(message.contains(part), "{part}: {message}");
                }
            }
            other => panic!("expected the racing read refused, got {other:?}"),
        }
        // A module declares `SubgroupUniformControlFlowKHR` (4421) where it
        // relies on it, a subgroup operation after a divergent selection's
        // merge, and the device has it.
        let declares = |module: &[u32]| module.windows(3).any(|words| words[0] == ((3 << 16) | 16) && words[2] == 4421);
        for (body, relies) in [
            ("void main() { if (gl_SubgroupInvocationID < 7u) v[1] = 2.0; else v[2] = 3.0; v[3u + gl_SubgroupID] = subgroupAdd(v[gl_SubgroupInvocationID]); }", true),
            ("void main() { if (n > 7u) v[1] = 2.0; else v[2] = 3.0; v[3u + gl_SubgroupID] = subgroupAdd(v[gl_SubgroupInvocationID]); }", false),
        ] {
            let source = format!("{header}{body}\n");
            let module = formation::compile(&source, "main", environment).expect(body);
            assert_eq!(declares(&module), relies, "{body}");
            let without = seal::Environment { subgroup_uniform_control_flow: false, ..environment };
            assert!(!declares(&formation::compile(&source, "main", without).expect(body)), "{body}");
        }
        // A replicated constant (`OpConstantCompositeReplicateEXT`, 4461) is
        // a constant like any other.
        let module = formation::compile(
            &format!("{header}void main() {{ vec4 p = vec4(2.0); p.y = v[0]; v[gl_SubgroupID] = subgroupAdd(p.x + p.y + p[n & 3u]); }}\n"),
            "main",
            environment,
        )
        .expect("module");
        let mut replicated = Vec::with_capacity(module.len());
        replicated.extend_from_slice(&module[..5]);
        let mut patched = 0;
        for instruction in seal::instructions(&module).expect("module") {
            let words = &module[instruction.start..instruction.start + instruction.words];
            // `OpConstantComposite` of equal constituents.
            if instruction.opcode == 44 && words.len() > 4 && words[3..].iter().all(|constituent| *constituent == words[3]) {
                replicated.extend([(4 << 16) | 4461, words[1], words[2], words[3]]);
                patched += 1;
            } else {
                replicated.extend_from_slice(words);
            }
        }
        assert!(patched > 0, "the module holds a splat constant");
        assert_eq!(uniformity::check(&replicated, environment.maximal_reconvergence).map(|findings| findings.violations), Ok(Vec::new()));
        // Formation ships a hardware join as its argument: no call, and no
        // function, is left of it.
        let module = formation::compile(
            &format!("{header}{join}void main() {{ if (seismic_hardware_any(n > 3u)) v[gl_SubgroupID] = subgroupAdd(v[1]); }}\n"),
            "main",
            environment,
        )
        .expect("module");
        let opcodes = seal::instructions(&module).expect("module").iter().map(|instruction| instruction.opcode).collect::<Vec<_>>();
        // `OpFunctionCall`, and `OpFunction` but the entry point's.
        assert!(!opcodes.contains(&57) && opcodes.iter().filter(|opcode| **opcode == 54).count() == 1);
        // A module of more than one entry point is not one formation makes.
        let module = formation::compile(&format!("{header}void main() {{ v[0] = 1.0; }}\n"), "main", environment).expect("module");
        let entry = seal::instructions(&module)
            .expect("module")
            .into_iter()
            .find(|instruction| instruction.opcode == 15)
            .expect("an entry point");
        let entry_point = &module[entry.start..entry.start + entry.words];
        let module = [&module[..entry.start], entry_point, &module[entry.start..]].concat();
        assert_eq!(
            uniformity::check(&module, environment.maximal_reconvergence),
            Err(seal::MalformedModule("more than one entry point".into()))
        );
        // Only a module that reads its subgroups requires them full.
        for (body, reads) in [
            ("void main() { v[gl_LocalInvocationIndex] = 1.0; }", false),
            ("void main() { v[gl_SubgroupID] = 1.0; }", true),
            ("void main() { v[0] = subgroupAdd(v[1]); }", true),
        ] {
            let module = formation::compile(&format!("{header}{body}\n"), "main", environment).expect(body);
            assert_eq!(uniformity::reads_subgroups(&module).expect("module"), reads, "{body}");
        }
    }
}
