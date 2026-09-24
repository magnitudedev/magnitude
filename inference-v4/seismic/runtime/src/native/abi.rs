//! The authored-native ABI: argument word layout and the generated source
//! prefix of Metal and CUDA native implementations.
//!
//! Buffer order is every tensor parameter, every tensor result, then every
//! scratch buffer, in declaration order. The argument words follow them
//! (`setBytes` on Metal, one by-value struct parameter on CUDA), and the
//! scalar-result slots come last.

use seismic_lang::checked::{NativeImplementation, NativeSpecialization};
use seismic_lang::entry::{CallSchema, ElementBindings, ParameterKind, ResultKind};
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry;

/// Source dialects with a generated prefix. CPU implementations are Rust and
/// receive a generated context instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dialect {
    Metal,
    Cuda,
}

/// Number of 64-bit argument words of an entry schema. The layout depends
/// only on the schema: every dimension, then each parameter (a tensor's
/// extents then strides; one word per scalar or index; two per range), then
/// each tensor result's extents and strides.
pub(crate) fn word_count(schema: &CallSchema) -> usize {
    schema.dimensions().len()
        + schema
            .parameters()
            .iter()
            .map(|parameter| match &parameter.kind {
                ParameterKind::Tensor { axes, .. } => axes.len() * 2,
                ParameterKind::Range { .. } => 2,
                ParameterKind::Scalar { .. } | ParameterKind::Index { .. } => 1,
            })
            .sum::<usize>()
        + schema
            .results()
            .iter()
            .map(|result| match &result.kind {
                ResultKind::Tensor { axes, .. } => axes.len() * 2,
                ResultKind::Range { .. } | ResultKind::Scalar(_) | ResultKind::Index { .. } => 0,
            })
            .sum::<usize>()
}

/// Number of 64-bit scalar-result slots.
pub(crate) fn scalar_word_count(schema: &CallSchema) -> usize {
    schema
        .results()
        .iter()
        .map(|result| match &result.kind {
            ResultKind::Range { .. } => 2,
            ResultKind::Scalar(_) | ResultKind::Index { .. } => 1,
            ResultKind::Tensor { .. } => 0,
        })
        .sum()
}

pub(crate) fn native_macro(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// The complete formation source: generated prefix followed by the asset.
pub(crate) fn render_source(
    dialect: Dialect,
    schema: &CallSchema,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: &str,
) -> String {
    let mut prefix = match dialect {
        Dialect::Metal => String::from("#include <metal_stdlib>\nusing namespace metal;\n"),
        Dialect::Cuda => String::from(CUDA_HELPERS),
    };
    for (name, representation) in bindings.iter() {
        render_representation(
            &mut prefix,
            &format!("SEISMIC_ELEMENT_{}", native_macro(name)),
            representation,
        );
    }
    for (name, value) in specialization.params() {
        prefix.push_str(&format!(
            "#define SEISMIC_TUNE_{} {value}\n",
            native_macro(name)
        ));
    }
    let mut buffer = 0usize;
    let mut word = 0usize;
    for dimension in schema.dimensions() {
        let name = native_macro(&dimension.name);
        match specialization.static_value(&dimension.name) {
            Some(value) => {
                let word_type = match dialect {
                    Dialect::Metal => "ulong",
                    Dialect::Cuda => "unsigned long long",
                };
                prefix.push_str(&format!(
                    "#define SEISMIC_DIM_{name} (({word_type}){value})\n"
                ));
            }
            None => prefix.push_str(&format!(
                "#define SEISMIC_DIM_{name} (seismic_words[{word}])\n"
            )),
        }
        word += 1;
    }
    for (ordinal, parameter) in schema.parameters().iter().enumerate() {
        let name = native_macro(&parameter.name);
        let named = schema
            .parameters()
            .iter()
            .filter(|candidate| candidate.name == parameter.name)
            .count()
            == 1;
        match &parameter.kind {
            ParameterKind::Tensor {
                axes,
                representation,
                ..
            } => {
                if named {
                    prefix.push_str(&format!("#define SEISMIC_BUFFER_{name} {buffer}\n"));
                }
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal}_BUFFER {buffer}\n"
                ));
                render_representation(
                    &mut prefix,
                    &format!("SEISMIC_PARAM_{ordinal}"),
                    *representation,
                );
                if named {
                    render_representation(&mut prefix, &format!("SEISMIC_{name}"), *representation);
                }
                buffer += 1;
                for axis in 0..axes.len() {
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{ordinal}_EXTENT_{axis} (seismic_words[{}])\n",
                        word + axis
                    ));
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{ordinal}_STRIDE_{axis} (seismic_words[{}])\n",
                        word + axes.len() + axis
                    ));
                    if named {
                        prefix.push_str(&format!(
                            "#define SEISMIC_{name}_EXTENT_{axis} SEISMIC_PARAM_{ordinal}_EXTENT_{axis}\n#define SEISMIC_{name}_STRIDE_{axis} SEISMIC_PARAM_{ordinal}_STRIDE_{axis}\n"
                        ));
                    }
                }
                word += axes.len() * 2;
            }
            ParameterKind::Scalar { .. } | ParameterKind::Index { .. } => {
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal} (seismic_words[{word}])\n"
                ));
                if named {
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{name} SEISMIC_PARAM_{ordinal}\n"
                    ));
                }
                word += 1;
            }
            ParameterKind::Range { .. } => {
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal}_START (seismic_words[{word}])\n"
                ));
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal}_END (seismic_words[{}])\n",
                    word + 1
                ));
                if named {
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{name}_START SEISMIC_PARAM_{ordinal}_START\n#define SEISMIC_PARAM_{name}_END SEISMIC_PARAM_{ordinal}_END\n"
                    ));
                }
                word += 2;
            }
        }
    }
    let mut scalar_word = 0usize;
    for (ordinal, result) in schema.results().iter().enumerate() {
        if let ResultKind::Tensor {
            axes,
            representation,
        } = &result.kind
        {
            prefix.push_str(&format!(
                "#define SEISMIC_RESULT_{ordinal}_BUFFER {buffer}\n"
            ));
            render_representation(
                &mut prefix,
                &format!("SEISMIC_RESULT_{ordinal}"),
                *representation,
            );
            buffer += 1;
            for axis in 0..axes.len() {
                prefix.push_str(&format!(
                    "#define SEISMIC_RESULT_{ordinal}_EXTENT_{axis} (seismic_words[{}])\n",
                    word + axis
                ));
                prefix.push_str(&format!(
                    "#define SEISMIC_RESULT_{ordinal}_STRIDE_{axis} (seismic_words[{}])\n",
                    word + axes.len() + axis
                ));
            }
            word += axes.len() * 2;
        } else {
            prefix.push_str(&format!(
                "#define SEISMIC_RESULT_{ordinal}_WORD {scalar_word}\n"
            ));
            scalar_word += if matches!(result.kind, ResultKind::Range { .. }) {
                2
            } else {
                1
            };
        }
    }
    for scratch in &implementation.scratch {
        prefix.push_str(&format!(
            "#define SEISMIC_BUFFER_SCRATCH_{} {buffer}\n",
            native_macro(&scratch.name)
        ));
        buffer += 1;
    }
    prefix.push_str(&format!("#define SEISMIC_BUFFER_WORDS {buffer}\n"));
    prefix.push_str(&format!(
        "#define SEISMIC_BUFFER_SCALAR_RESULTS {}\n",
        buffer + 1
    ));
    if dialect == Dialect::Cuda {
        prefix.push_str(&format!(
            "struct seismic_words_t {{ unsigned long long w[{}]; }};\n#define seismic_words (seismic_words_value.w)\n",
            word.max(1)
        ));
        let buffers = (0..buffer)
            .map(|index| format!("unsigned char* __restrict__ seismic_buffer_{index}, "))
            .collect::<String>();
        prefix.push_str(&format!(
            "#define SEISMIC_KERNEL_PARAMS {buffers}seismic_words_t seismic_words_value, unsigned long long* seismic_scalar_results\n"
        ));
        prefix.push_str("#define SEISMIC_PTR_(index) seismic_buffer_##index\n#define SEISMIC_PTR(index) SEISMIC_PTR_(index)\n");
        prefix.push_str("#define SEISMIC_SCALAR_RESULTS seismic_scalar_results\n");
    }
    debug_assert_eq!(word, word_count(schema));
    prefix.push_str(asset);
    prefix
}

/// CUDA helpers every native source receives in place of vendor headers.
/// Conversions use the same rounding as the compiler's PTX emitter.
const CUDA_HELPERS: &str = r#"
__device__ __forceinline__ float seismic_bf16_to_f32(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}
__device__ __forceinline__ unsigned short seismic_f32_to_bf16(float value) {
    unsigned short result;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}
__device__ __forceinline__ float seismic_f16_to_f32(unsigned short value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}
__device__ __forceinline__ unsigned short seismic_f32_to_f16(float value) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}
"#;

fn render_representation(out: &mut String, prefix: &str, representation: RepresentationId) {
    let info = registry::representation_info(representation);
    out.push_str(&format!(
        "#define {prefix}_REPRESENTATION_{} 1\n",
        native_macro(info.name)
    ));
    out.push_str(&format!(
        "#define {prefix}_DECODED_{} 1\n",
        native_macro(info.decoded.name())
    ));
    match &info.kind {
        registry::RepresentationKind::Dense(dtype) => {
            out.push_str(&format!("#define {prefix}_KIND_DENSE 1\n"));
            out.push_str(&format!(
                "#define {prefix}_PACKET_SIZE {}\n#define {prefix}_PACKET_ALIGNMENT {}\n#define {prefix}_LOGICAL_GROUP 1\n#define {prefix}_PLANE_COUNT 0\n",
                dtype.bytes(),
                dtype.bytes()
            ));
        }
        registry::RepresentationKind::Packed(layout) => {
            out.push_str(&format!("#define {prefix}_KIND_PACKED 1\n"));
            out.push_str(&format!(
                "#define {prefix}_PACKET_SIZE {}\n#define {prefix}_PACKET_ALIGNMENT {}\n#define {prefix}_LOGICAL_GROUP {}\n#define {prefix}_PLANE_COUNT {}\n",
                layout.packet_size,
                layout.packet_alignment,
                layout.group,
                layout.planes.len()
            ));
            for (ordinal, plane) in layout.planes.iter().enumerate() {
                let plane_prefix = format!("{prefix}_PLANE_{ordinal}");
                out.push_str(&format!(
                    "#define {plane_prefix}_NAME_{} 1\n#define {plane_prefix}_OFFSET {}\n#define {plane_prefix}_BYTES_PER_GROUP {}\n#define {plane_prefix}_ALIGNMENT {}\n#define {plane_prefix}_GROUP {}\n#define {plane_prefix}_FIELDS {}\n#define {plane_prefix}_ENTRY_BITS {}\n#define {plane_prefix}_STORAGE_{} 1\n",
                    native_macro(plane.name),
                    plane.offset,
                    plane.bytes_per_group,
                    plane.alignment,
                    plane.group,
                    plane.fields,
                    plane.entry_bits,
                    native_macro(plane.storage_dtype.name())
                ));
                render_plane_encoding(out, &plane_prefix, &plane.encoding);
            }
        }
        registry::RepresentationKind::External(layout) => {
            out.push_str(&format!("#define {prefix}_KIND_EXTERNAL 1\n"));
            out.push_str(&format!(
                "#define {prefix}_PACKET_SIZE {}\n#define {prefix}_PACKET_ALIGNMENT {}\n#define {prefix}_LOGICAL_GROUP {}\n#define {prefix}_PLANE_COUNT 0\n",
                layout.packet_size, layout.packet_alignment, layout.logical_group
            ));
        }
    }
}

fn render_plane_encoding(out: &mut String, prefix: &str, encoding: &registry::PlaneEncoding) {
    match encoding {
        registry::PlaneEncoding::Dense(dtype) => {
            out.push_str(&format!(
                "#define {prefix}_ENCODING_DENSE 1\n#define {prefix}_ENCODING_DTYPE_{} 1\n",
                native_macro(dtype.name())
            ));
        }
        registry::PlaneEncoding::Packed {
            bits,
            interpretation,
        } => {
            out.push_str(&format!(
                "#define {prefix}_ENCODING_PACKED 1\n#define {prefix}_ENCODING_BITS {bits}\n"
            ));
            render_code_interpretation(out, prefix, interpretation);
        }
        registry::PlaneEncoding::FloatCode { format } => {
            let name = match format {
                registry::FloatCodeFormat::E2M1 => "E2M1",
                registry::FloatCodeFormat::E4M3 => "E4M3",
                registry::FloatCodeFormat::UE4M3 => "UE4M3",
            };
            out.push_str(&format!(
                "#define {prefix}_ENCODING_FLOAT_CODE 1\n#define {prefix}_ENCODING_FLOAT_CODE_{name} 1\n#define {prefix}_ENCODING_BITS {}\n",
                format.bits()
            ));
        }
    }
}

fn render_code_interpretation(
    out: &mut String,
    prefix: &str,
    interpretation: &registry::CodeInterpretation,
) {
    match interpretation {
        registry::CodeInterpretation::Unsigned => {
            out.push_str(&format!("#define {prefix}_CODE_UNSIGNED 1\n"));
        }
        registry::CodeInterpretation::TwosComplement => {
            out.push_str(&format!("#define {prefix}_CODE_TWOS_COMPLEMENT 1\n"));
        }
        registry::CodeInterpretation::Offset(offset) => {
            out.push_str(&format!(
                "#define {prefix}_CODE_OFFSET 1\n#define {prefix}_CODE_OFFSET_VALUE {offset}\n"
            ));
        }
        registry::CodeInterpretation::Table(values) => {
            out.push_str(&format!(
                "#define {prefix}_CODE_TABLE 1\n#define {prefix}_CODE_TABLE_COUNT {}\n",
                values.len()
            ));
            for (ordinal, value) in values.iter().enumerate() {
                out.push_str(&format!("#define {prefix}_CODE_TABLE_{ordinal} {value}\n"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::registry::BackendName;

    const PROBE: &str = "fn probe[N, K](x: &tensor[N, K] E) -> tensor[1] f32:\n    let mut output = tensor[1] f32\n    for i in 0..1:\n        output[i] = f32(x[0, 0])\n    return output\n\nnative probe for cuda from \"probe.cu\":\n    static (K)\n    params (TILE in [4, 8])\n    scratch partials bytes (N * 4)\n    launch probe:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (TILE, 1, 1)\n";

    fn render(dialect: Dialect, representation: &str) -> String {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "probe.seismic".to_owned(),
            text: PROBE.to_owned(),
        }]))
        .expect("probe source checks");
        let entry = module.entries()[0].id;
        let binding = registry::representation(representation).expect("registered representation");
        let bindings = ElementBindings::new().bind("E", binding);
        let logical = module
            .entry(entry, &bindings)
            .expect("probe entry monomorphizes");
        let implementation = module
            .native_implementation(entry, BackendName::Cuda)
            .expect("native implementation");
        let specialization = NativeSpecialization::new()
            .with_static("K", 64)
            .with_param("TILE", 8);
        render_source(
            dialect,
            logical.schema(),
            &bindings,
            implementation,
            &specialization,
            "\n// asset\n",
        )
    }

    #[test]
    fn prefix_describes_dense_element_parameter_and_tensor_abi() {
        let source = render(Dialect::Metal, "f16");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_F16 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_DENSE 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_DECODED_F16 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 2\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 1\n"));
        assert!(source.contains("#define SEISMIC_X_REPRESENTATION_F16 1\n"));
        assert!(source.contains("#define SEISMIC_RESULT_0_REPRESENTATION_F32 1\n"));
    }

    #[test]
    fn prefix_describes_packed_planes_and_encoding() {
        let source = render(Dialect::Metal, "q8g32");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_Q8G32 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_PACKED 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 36\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_COUNT 2\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_NAME_WORDS 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_BYTES_PER_GROUP 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_ENCODING_PACKED 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_CODE_UNSIGNED 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_NAME_SCALE 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_OFFSET 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_STORAGE_F32 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_ENCODING_DENSE 1\n"));
    }

    #[test]
    fn prefix_describes_external_packets() {
        let source = render(Dialect::Metal, "gguf_q4_k");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q4_K 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_EXTERNAL 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 144\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 256\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_COUNT 0\n"));
    }

    #[test]
    fn specialization_renders_constants_and_scratch_follows_results() {
        let source = render(Dialect::Metal, "f32");
        assert!(source.contains("#define SEISMIC_DIM_N (seismic_words[0])\n"));
        assert!(source.contains("#define SEISMIC_DIM_K ((ulong)64)\n"));
        assert!(source.contains("#define SEISMIC_TUNE_TILE 8\n"));
        // x is buffer 0, the result buffer 1, scratch 2, words 3.
        assert!(source.contains("#define SEISMIC_RESULT_0_BUFFER 1\n"));
        assert!(source.contains("#define SEISMIC_BUFFER_SCRATCH_PARTIALS 2\n"));
        assert!(source.contains("#define SEISMIC_BUFFER_WORDS 3\n"));
        assert!(source.contains("#define SEISMIC_BUFFER_SCALAR_RESULTS 4\n"));
        assert!(!source.contains("seismic_words_t"));
    }

    #[test]
    fn cuda_prefix_declares_kernel_parameters_and_helpers() {
        let source = render(Dialect::Cuda, "f32");
        assert!(source.contains("seismic_bf16_to_f32"));
        assert!(!source.contains("#include"));
        // Two dimensions, x (rank 2) and the result (rank 1): 2 + 4 + 2 words.
        assert!(source.contains("struct seismic_words_t { unsigned long long w[8]; };\n"));
        assert!(source.contains("#define SEISMIC_KERNEL_PARAMS unsigned char* __restrict__ seismic_buffer_0, unsigned char* __restrict__ seismic_buffer_1, unsigned char* __restrict__ seismic_buffer_2, seismic_words_t seismic_words_value, unsigned long long* seismic_scalar_results\n"));
        assert!(source.contains("#define SEISMIC_PTR(index) SEISMIC_PTR_(index)\n"));
    }
}
