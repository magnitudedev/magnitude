//! The Vulkan dialect of the generated prefix (Vulkan backend spec §5.2):
//! `#version` and the floor's extensions, the device feature macros, the
//! device library, the argument block, the workgroup size and the typed
//! group-memory views, and the suffix that calls the launch's kernel.

/// Device features a Vulkan prefix exposes as `SEISMIC_HAS_*` macros (and
/// the extensions they need). They are part of the rendered source, so of
/// the formation and tuning identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VulkanFeatures {
    /// Subgroup 16x16x16 cooperative matrix, f16->f32 and s8->s32.
    pub(crate) matrix: bool,
    /// Accumulator arrays of more than 4 matrix fragments compile correctly.
    pub(crate) wide_accumulators: bool,
    /// Mixed-signedness packed 4x8 dot product is accelerated.
    pub(crate) mixed_dot: bool,
    pub(crate) f32_atomic_add: bool,
    pub(crate) shared_int64_atomics: bool,
}

/// The typed views of a Vulkan launch's one group-memory region, in
/// specialization-constant order after the workgroup size (ids 3..): view
/// name suffix, GLSL element type and element bytes. Every view covers the
/// whole region, and all of them alias it.
const SHARED_VIEWS: [(&str, &str, u64); 8] = [
    ("F32", "float", 4),
    ("F16", "float16_t", 2),
    ("BF16", "uint16_t", 2),
    ("U8", "uint8_t", 1),
    ("U16", "uint16_t", 2),
    ("U32", "uint", 4),
    ("I32", "int", 4),
    ("UVEC4", "uvec4", 16),
];

/// First specialization-constant id of the shared views; ids 0..2 are the
/// workgroup size.
const SHARED_VIEW_CONSTANT: usize = 3;

/// Lengths of the shared views of a region of `shared_bytes` (at least one
/// element each), in specialization-constant order.
pub(crate) fn shared_view_lengths(shared_bytes: u64) -> Vec<u64> {
    SHARED_VIEWS
        .iter()
        .map(|(_, _, bytes)| shared_bytes.div_ceil(*bytes).max(1))
        .collect()
}

/// Group memory a pipeline reserves for `shared_bytes`: its largest view.
pub(crate) fn shared_footprint(shared_bytes: u64) -> u64 {
    SHARED_VIEWS
        .iter()
        .zip(shared_view_lengths(shared_bytes))
        .map(|((_, _, bytes), length)| bytes * length)
        .max()
        .expect("there are shared views")
}

/// The Vulkan device library (`vulkan_prelude.glsl`): element references,
/// conversions, rounded arithmetic, subgroup shuffles and fixed-order
/// reductions, packed dot products.
const PRELUDE: &str = include_str!("../vulkan_prelude.glsl");

/// Extensions every Vulkan source enables: the device floor's.
const EXTENSIONS: [&str; 22] = [
    "GL_GOOGLE_cpp_style_line_directive",
    "GL_EXT_buffer_reference",
    "GL_EXT_buffer_reference2",
    "GL_EXT_scalar_block_layout",
    "GL_EXT_shader_explicit_arithmetic_types",
    "GL_EXT_shader_16bit_storage",
    "GL_EXT_shader_8bit_storage",
    "GL_EXT_shared_memory_block",
    "GL_EXT_control_flow_attributes",
    "GL_EXT_integer_dot_product",
    "GL_EXT_spirv_intrinsics",
    "GL_KHR_memory_scope_semantics",
    "GL_KHR_shader_subgroup_basic",
    "GL_KHR_shader_subgroup_vote",
    "GL_KHR_shader_subgroup_arithmetic",
    "GL_KHR_shader_subgroup_ballot",
    "GL_KHR_shader_subgroup_shuffle",
    "GL_KHR_shader_subgroup_shuffle_relative",
    "GL_KHR_shader_subgroup_clustered",
    "GL_KHR_shader_subgroup_quad",
    "GL_EXT_shader_subgroup_extended_types_int8",
    "GL_EXT_shader_subgroup_extended_types_float16",
];

/// `#version`, the extensions, the device feature macros and the prelude.
pub(super) fn header(features: VulkanFeatures) -> String {
    let mut header = String::from("#version 460\n");
    let gated = [
        (features.matrix, "GL_KHR_cooperative_matrix"),
        (features.f32_atomic_add, "GL_EXT_shader_atomic_float"),
        (features.shared_int64_atomics, "GL_EXT_shader_atomic_int64"),
    ];
    for extension in EXTENSIONS
        .iter()
        .chain(gated.iter().filter(|(on, _)| *on).map(|(_, name)| name))
    {
        header.push_str(&format!("#extension {extension} : require\n"));
    }
    for (name, on) in [
        ("MATRIX", features.matrix),
        ("WIDE_ACCUMULATORS", features.wide_accumulators),
        ("MIXED_DOT", features.mixed_dot),
        ("F32_ATOMIC_ADD", features.f32_atomic_add),
        ("SHARED_INT64_ATOMICS", features.shared_int64_atomics),
    ] {
        header.push_str(&format!("#define SEISMIC_HAS_{name} {}\n", u8::from(on)));
    }
    header.push_str(PRELUDE);
    header
}

/// The argument block (buffer addresses, the scalar-result address, then
/// the words: there is no words buffer), read-only flags, workgroup size and
/// group-memory views of a source with `buffers` buffers.
pub(super) fn tail(buffers: usize, read_only: impl Iterator<Item = bool>) -> String {
    let mut tail = format!("#define SEISMIC_BUFFER_SCALAR_RESULTS {buffers}\n");
    tail.push_str(
        "layout(push_constant, scalar) uniform seismic_push_t { uint64_t seismic_arguments; };\n\
         layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer seismic_argument_block_t { uint64_t w[]; };\n\
         #define SEISMIC_PTR(index) (seismic_argument_block_t(seismic_arguments).w[index])\n",
    );
    tail.push_str(&format!(
        "#define SEISMIC_SCALAR_RESULTS (seismic_argument_block_t(seismic_arguments).w[{buffers}])\n\
         #define seismic_words (seismic_argument_block_t(seismic_arguments + {}ul).w)\n",
        (buffers + 1) * 8
    ));
    tail.push_str("#define SEISMIC_READONLY_(index) SEISMIC_READONLY_##index\n#define SEISMIC_READONLY(index) SEISMIC_READONLY_(index)\n");
    for (index, read_only) in read_only.enumerate() {
        tail.push_str(&format!("#define SEISMIC_READONLY_{index} {read_only}\n"));
    }
    tail.push_str("layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\n");
    for (ordinal, (name, ty, bytes)) in SHARED_VIEWS.iter().enumerate() {
        let lower = name.to_ascii_lowercase();
        tail.push_str(&format!(
            "layout(constant_id = {}) const uint SEISMIC_SHARED_{name} = {}u;\nshared seismic_shared_{lower}_t {{ {ty} seismic_shared_{lower}[SEISMIC_SHARED_{name}]; }};\n",
            SHARED_VIEW_CONSTANT + ordinal,
            SHARED_DEFAULT_BYTES / bytes
        ));
    }
    tail
}

/// The group-memory region the compiler sees before specialization: the
/// largest any floor device has (64 KiB), so constant indices into the
/// views compile. Pipelines always specialize the lengths to the launch's
/// `shared_bytes`; drivers allocate only the specialized size.
const SHARED_DEFAULT_BYTES: u64 = 64 << 10;

/// Calls the launch's kernel, which formation names with `SEISMIC_KERNEL`.
pub(super) const SUFFIX: &str = "\nvoid main() {\n    SEISMIC_KERNEL();\n}\n";

/// The prelude's correctly rounded helpers against the host oracle on the
/// first Vulkan GPU (§13.1 Numerics): `seismic_div_rn`, `seismic_sqrt_rn`,
/// `seismic_fma_rn` and the bf16 conversion over edge cases (zeros,
/// subnormals, extremes, infinities, NaN) and random operands.
#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;
    use seismic_vulkan::direct::{DirectBatch, DirectLaunch};
    use seismic_vulkan::formation::{DirectModule, Kernel};

    const KERNEL: &str = "
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(push_constant, scalar) uniform seismic_push_t { uint64_t seismic_arguments; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer words_t { uint64_t w[]; };
void numerics() {
    // The scalar-result slot carries the data buffer: [cases][count * 3 in][count * 4 out].
    const uint64_t base = words_t(seismic_arguments).w[0];
    const uint count = seismic_u32(base)[0].v;
    const uint i = gl_GlobalInvocationID.x;
    if (i >= count)
        return;
    seismic_u32 data = seismic_u32(base + 4ul);
    const float a = uintBitsToFloat(data[3u * i].v);
    const float b = uintBitsToFloat(data[3u * i + 1u].v);
    const float c = uintBitsToFloat(data[3u * i + 2u].v);
    const uint result = 3u * count + 4u * i;
    data[result].v = floatBitsToUint(seismic_div_rn(a, b));
    // |a| by its bits: a float `abs` would flush a subnormal where the
    // device flushes (NVIDIA).
    data[result + 1u].v = floatBitsToUint(seismic_sqrt_rn(uintBitsToFloat(floatBitsToUint(a) & 0x7fffffffu)));
    data[result + 2u].v = floatBitsToUint(seismic_fma_rn(a, b, c));
    data[result + 3u].v = uint(seismic_f32_to_bf16(a));
}
void main() {
    numerics();
}
";

    fn bf16(value: f32) -> u32 {
        let bits = value.to_bits();
        if value.is_nan() {
            return 0x7fff;
        }
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16
    }

    fn same(expected: f32, actual: u32) -> bool {
        (expected.is_nan() && f32::from_bits(actual).is_nan()) || expected.to_bits() == actual
    }

    #[test]
    fn rounded_helpers_match_the_host_oracle() {
        let Some(description) = seismic_vulkan::discover().ok().and_then(|devices| {
            devices
                .into_iter()
                .find(|device| device.floor.is_ok() && device.facts.is_gpu())
        }) else {
            eprintln!("no Vulkan GPU meets the floor");
            return;
        };
        let device = seismic_vulkan::Device::open(description.facts.uuid).expect("device opens");
        let features = VulkanFeatures {
            matrix: false,
            wide_accumulators: false,
            mixed_dot: false,
            f32_atomic_add: false,
            shared_int64_atomics: false,
        };
        let source = format!("{}{KERNEL}", header(features));
        let module = DirectModule::form(
            &device,
            &source,
            &[Kernel {
                name: "numerics",
                threads: [64, 1, 1],
                constants: &[],
            }],
            |_| None,
            |_, _| {},
        )
        .expect("the numerics kernel forms");

        let specials = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            3.0,
            0.1,
            1e-45,
            -1e-45,
            1.1754942e-38,
            1.1754944e-38,
            2.3509887e-38,
            3.4028235e38,
            -3.4028235e38,
            1e30,
            1e-30,
            7.0e-39,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1.0000001,
            0.99999994,
            16777215.0,
            1.5e-44,
        ];
        let mut cases = Vec::new();
        for a in specials {
            for b in specials {
                cases.push((
                    a,
                    b,
                    specials[(a.to_bits() ^ b.to_bits()) as usize % specials.len()],
                ));
            }
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        while cases.len() < 1 << 16 {
            let word = next();
            // Uniform bits, and values of moderate exponent (the fast path).
            let a = f32::from_bits(word as u32);
            let b = f32::from_bits((word >> 32) as u32);
            let c = f32::from_bits(next() as u32);
            cases.push((a, b, c));
            let scale = |value: f32, word: u64| {
                value.abs().max(1e-30).min(1e30) * if word & 1 == 0 { 1.0 } else { -1.0 }
            };
            cases.push((scale(a, word), scale(b, word >> 1), scale(c, word >> 2)));
        }
        let count = cases.len() as u32;
        let mut bytes = count.to_le_bytes().to_vec();
        for (a, b, c) in &cases {
            for value in [a, b, c] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let total = bytes.len() + cases.len() * 16;
        let data = device.allocate(total as u64, 256).expect("allocation");
        device.write(&data, 0, &bytes).expect("upload");
        let mut batch = DirectBatch::new(&device).expect("batch");
        batch
            .launch(&DirectLaunch {
                module: &module,
                function: 0,
                buffers: &[],
                words: &[],
                scalar_results: (&data, 0),
                groups: [u64::from(count.div_ceil(64)), 1, 1],
            })
            .expect("launch");
        batch
            .commit()
            .and_then(|submission| submission.finish())
            .expect("run");
        let mut results = vec![0u8; cases.len() * 16];
        device
            .read(&data, bytes.len() as u64, &mut results)
            .expect("download");
        // Without `DenormPreserve 32` the driver's default applies (NVIDIA:
        // flush to zero, §7.3): `fma` with a subnormal operand or result is
        // then outside the oracle. The integer paths preserve subnormals.
        let flushes = !device.facts().denorm_preserve_32;
        let subnormal = |value: f32| value.is_subnormal();
        let mut failures = Vec::new();
        for (index, (a, b, c)) in cases.iter().enumerate() {
            let fma_defined = !flushes
                || ![*a, *b, *c, a.mul_add(*b, *c), a * b]
                    .into_iter()
                    .any(subnormal);
            let word = |slot: usize| {
                let at = index * 16 + slot * 4;
                u32::from_le_bytes(results[at..at + 4].try_into().expect("four bytes"))
            };
            let checks = [
                ("div", same(a / b, word(0)), (a / b).to_bits(), word(0)),
                (
                    "sqrt",
                    same(a.abs().sqrt(), word(1)),
                    a.abs().sqrt().to_bits(),
                    word(1),
                ),
                (
                    "fma",
                    !fma_defined || same(a.mul_add(*b, *c), word(2)),
                    a.mul_add(*b, *c).to_bits(),
                    word(2),
                ),
                ("bf16", bf16(*a) == word(3), bf16(*a), word(3)),
            ];
            for (name, ok, expected, actual) in checks {
                if !ok {
                    failures.push(format!(
                        "{name}({:#010x}, {:#010x}, {:#010x}): expected {expected:#010x}, got {actual:#010x}",
                        a.to_bits(),
                        b.to_bits(),
                        c.to_bits()
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} cases differ on {}:\n{}",
            failures.len(),
            cases.len(),
            device.facts().name,
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
