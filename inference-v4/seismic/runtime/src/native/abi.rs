//! The authored-native ABI: argument word layout and the generated source
//! prefix of Metal, CUDA and Vulkan native implementations.
//!
//! Buffer order is every tensor parameter, every tensor result, then every
//! scratch buffer, in declaration order. The argument words follow them
//! (`setBytes` on Metal, one by-value struct parameter on CUDA), and the
//! scalar-result slots come last. On Vulkan one argument block in device
//! memory holds every buffer address, then the scalar-result address, then
//! the words; the single push constant is the block's address.

use super::plan::ParameterAddress;
use seismic_lang::checked::{NativeImplementation, NativeSpecialization};
use seismic_lang::entry::{
    CallSchema, ElementBindings, LogicalEntry, ParameterKind, ResultKind, TensorAccess,
};
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry;

#[cfg(any(not(target_os = "macos"), test))]
pub(crate) mod vulkan;

/// Source dialects with a generated prefix. CPU implementations are Rust and
/// receive a generated context instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dialect {
    Metal,
    Cuda,
    /// GLSL 4.60 compute for Vulkan 1.3, with the device's feature macros.
    /// Vulkan is not built on macOS; its prefix still renders in tests there.
    #[cfg(any(not(target_os = "macos"), test))]
    Vulkan(vulkan::VulkanFeatures),
}

/// A 64-bit constant in the dialect's syntax.
fn literal(dialect: Dialect, value: u64) -> String {
    match dialect {
        Dialect::Metal => format!("((ulong){value})"),
        Dialect::Cuda => format!("((unsigned long long){value})"),
        #[cfg(any(not(target_os = "macos"), test))]
        Dialect::Vulkan(_) => format!("uint64_t({value}ul)"),
    }
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

/// A launch-scoped implementation passes its non-code choices in the native
/// argument words. Each launch source sees its own local names; all launches
/// share the same word layout so one call can dispatch several of them.
pub(crate) fn runtime_parameters(implementation: &NativeImplementation) -> Vec<ParameterAddress> {
    if !implementation.launch_scoped() {
        return Vec::new();
    }
    implementation
        .params
        .iter()
        .filter(|parameter| !parameter.code)
        .map(|parameter| ParameterAddress::Entry(parameter.name.clone()))
        .chain(
            implementation
                .launches
                .iter()
                .enumerate()
                .flat_map(|(ordinal, launch)| {
                    launch
                        .params
                        .iter()
                        .filter(|parameter| !parameter.code)
                        .map(move |parameter| ParameterAddress::Launch {
                            ordinal,
                            name: parameter.name.clone(),
                        })
                }),
        )
        .collect()
}

pub(crate) fn native_word_count(
    schema: &CallSchema,
    implementation: &NativeImplementation,
) -> usize {
    word_count(schema) + runtime_parameters(implementation).len()
}

/// Argument slots an implementation binds: its buffers (tensor parameters,
/// tensor results, scratch), the argument words and the scalar-result slots.
pub(crate) fn buffer_slots(schema: &CallSchema, implementation: &NativeImplementation) -> usize {
    let tensors = schema
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.kind, ParameterKind::Tensor { .. }))
        .count()
        + schema
            .results()
            .iter()
            .filter(|result| matches!(result.kind, ResultKind::Tensor { .. }))
            .count();
    tensors + implementation.scratch.len() + 2
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

/// The geometry of one tensor argument fixed by the specialization's static
/// dimensions (S10): each axis extent the static dimensions determine, and
/// the canonical strides when every extent is static.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StaticTensor {
    pub extents: Vec<Option<u64>>,
    pub strides: Option<Vec<u64>>,
}

/// Per tensor parameter and tensor result (by schema ordinal; `None` for
/// non-tensors): the static geometry the prefix renders as constants. A
/// parameter with static strides must be bound with exactly those canonical
/// strides; results are always canonical.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StaticGeometry {
    pub parameters: Vec<Option<StaticTensor>>,
    pub results: Vec<Option<StaticTensor>>,
}

pub(crate) fn static_geometry(
    logical: &LogicalEntry,
    specialization: &NativeSpecialization,
) -> StaticGeometry {
    let schema = logical.schema();
    let mut values = InvocationValues::new();
    for dimension in schema.dimensions() {
        if let Some(value) = specialization.static_value(&dimension.name) {
            values.bind(dimension.symbol, SymbolValue::Nat(value.into()));
        }
    }
    let tensor = |representation: RepresentationId, axes: &[seismic_lang::expr::NatExpr]| {
        let extents = axes
            .iter()
            .map(|axis| {
                logical
                    .arena()
                    .compile_nat(*axis)
                    .evaluate_u64(&values)
                    .ok()
            })
            .collect::<Vec<_>>();
        let strides = extents
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .and_then(|extents| crate::layout::canonical(representation, &extents).ok())
            .map(|layout| layout.strides);
        Some(StaticTensor { extents, strides })
    };
    StaticGeometry {
        parameters: schema
            .parameters()
            .iter()
            .map(|parameter| match &parameter.kind {
                ParameterKind::Tensor {
                    representation,
                    axes,
                    ..
                } => tensor(*representation, axes),
                ParameterKind::Scalar { .. }
                | ParameterKind::Index { .. }
                | ParameterKind::Range { .. } => None,
            })
            .collect(),
        results: schema
            .results()
            .iter()
            .map(|result| match &result.kind {
                ResultKind::Tensor {
                    representation,
                    axes,
                } => tensor(*representation, axes),
                ResultKind::Scalar(_) | ResultKind::Index { .. } | ResultKind::Range { .. } => None,
            })
            .collect(),
    }
}

/// The complete formation source: generated prefix followed by the asset.
pub(crate) fn render_source(
    dialect: Dialect,
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: &str,
) -> String {
    render_source_with(
        dialect,
        logical,
        bindings,
        implementation,
        specialization,
        asset,
        None,
    )
}

/// One Metal launch source. The asset guards each templated kernel with its
/// `SEISMIC_FORMING_<kernel>` name. Code variants become explicit template
/// instantiations in one library; runtime parameter values never enter it.
pub(crate) fn render_metal_launch_source(
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: &str,
    launch_source: &super::plan::LaunchSource,
) -> String {
    let kernel = &implementation.launches[launch_source.ordinal].kernel;
    let mut source = render_source_with(
        Dialect::Metal,
        logical,
        bindings,
        implementation,
        specialization,
        asset,
        Some(kernel),
    );
    for variant in &launch_source.code_variants {
        if variant.is_empty() {
            continue;
        }
        let values = variant
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = variant
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("_");
        source.push_str(&format!(
            "\ntemplate [[host_name(\"{kernel}${suffix}\")]] [[kernel]] decltype({kernel}<{values}>) {kernel}<{values}>;\n"
        ));
    }
    source
}

/// One CUDA launch source. The asset guards templated kernels with their
/// `SEISMIC_FORMING_<kernel>` names; NVRTC requests the required instances by
/// name expression and supplies their lowered linker names after compilation.
pub(crate) fn render_cuda_launch_source(
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: &str,
    ordinal: usize,
) -> String {
    render_source_with(
        Dialect::Cuda,
        logical,
        bindings,
        implementation,
        specialization,
        asset,
        Some(&implementation.launches[ordinal].kernel),
    )
}

fn render_source_with(
    dialect: Dialect,
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: &str,
    forming: Option<&str>,
) -> String {
    let schema = logical.schema();
    let geometry = static_geometry(logical, specialization);
    let mut prefix = match dialect {
        Dialect::Metal => String::from("#include <metal_stdlib>\nusing namespace metal;\n"),
        Dialect::Cuda => String::from(CUDA_HELPERS),
        #[cfg(any(not(target_os = "macos"), test))]
        Dialect::Vulkan(features) => vulkan::header(features),
    };
    for (name, representation) in bindings.iter() {
        render_representation(
            &mut prefix,
            &format!("SEISMIC_ELEMENT_{}", native_macro(name)),
            representation,
        );
    }
    if let Some(kernel) = forming {
        prefix.push_str(&format!(
            "#define SEISMIC_FORMING_{} 1\n",
            native_macro(kernel)
        ));
    } else {
        for (name, value) in specialization.params() {
            prefix.push_str(&format!(
                "#define SEISMIC_TUNE_{} {value}\n",
                native_macro(name)
            ));
        }
    }
    let mut buffer = 0usize;
    let mut word = 0usize;
    for dimension in schema.dimensions() {
        let name = native_macro(&dimension.name);
        match specialization.static_value(&dimension.name) {
            Some(value) => {
                prefix.push_str(&format!(
                    "#define SEISMIC_DIM_{name} {}\n",
                    literal(dialect, value)
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
                let tensor = TensorAbi {
                    representation: *representation,
                    rank: axes.len(),
                    word,
                    fixed: geometry.parameters[ordinal]
                        .as_ref()
                        .expect("a tensor parameter has static geometry"),
                    dialect,
                };
                tensor.render(&mut prefix, &format!("SEISMIC_PARAM_{ordinal}"));
                if named {
                    tensor.render(&mut prefix, &format!("SEISMIC_{name}"));
                }
                buffer += 1;
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
            TensorAbi {
                representation: *representation,
                rank: axes.len(),
                word,
                fixed: geometry.results[ordinal]
                    .as_ref()
                    .expect("a tensor result has static geometry"),
                dialect,
            }
            .render(&mut prefix, &format!("SEISMIC_RESULT_{ordinal}"));
            buffer += 1;
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
    let runtime_parameters = if forming.is_some() {
        runtime_parameters(implementation)
    } else {
        Vec::new()
    };
    if let Some(kernel) = forming {
        let current = implementation
            .launches
            .iter()
            .position(|launch| launch.kernel == kernel)
            .expect("forming guard names a declared launch");
        let reads = implementation.launches[current].parameters();
        for (offset, address) in runtime_parameters.iter().enumerate() {
            let name = match address {
                ParameterAddress::Entry(name) if reads.contains(name) => Some(name),
                ParameterAddress::Launch { ordinal, name } if *ordinal == current => Some(name),
                _ => None,
            };
            if let Some(name) = name {
                prefix.push_str(&format!(
                    "#define SEISMIC_RUNTIME_{} (seismic_words[{}])\n",
                    native_macro(name),
                    word + offset,
                ));
            }
        }
        word += runtime_parameters.len();
    }
    for scratch in &implementation.scratch {
        prefix.push_str(&format!(
            "#define SEISMIC_BUFFER_SCRATCH_{} {buffer}\n",
            native_macro(&scratch.name)
        ));
        buffer += 1;
    }
    // Tensor parameters lead the buffer order; a shared (`&`) parameter is
    // read-only. Results and scratch follow.
    let read_only = schema
        .parameters()
        .iter()
        .filter_map(|parameter| match &parameter.kind {
            ParameterKind::Tensor { access, .. } => Some(*access == TensorAccess::Shared),
            ParameterKind::Scalar { .. }
            | ParameterKind::Index { .. }
            | ParameterKind::Range { .. } => None,
        })
        .chain(std::iter::repeat(false))
        .take(buffer);
    #[cfg(any(not(target_os = "macos"), test))]
    if let Dialect::Vulkan(_) = dialect {
        prefix.push_str(&vulkan::tail(buffer, read_only));
        debug_assert_eq!(word, word_count(schema) + runtime_parameters.len());
        prefix.push_str(asset);
        prefix.push_str(vulkan::SUFFIX);
        return prefix;
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
        // A read-only buffer is `const`, so its loads are eligible for the
        // non-coherent path.
        let buffers = read_only
            .enumerate()
            .map(|(index, read_only)| {
                let qualifier = if read_only { "const " } else { "" };
                format!("{qualifier}unsigned char* __restrict__ seismic_buffer_{index}, ")
            })
            .collect::<String>();
        prefix.push_str(&format!(
            "#define SEISMIC_KERNEL_PARAMS {buffers}seismic_words_t seismic_words_value, unsigned long long* seismic_scalar_results\n"
        ));
        prefix.push_str("#define SEISMIC_PTR_(index) seismic_buffer_##index\n#define SEISMIC_PTR(index) SEISMIC_PTR_(index)\n");
        prefix.push_str("#define SEISMIC_SCALAR_RESULTS seismic_scalar_results\n");
    }
    debug_assert_eq!(word, word_count(schema) + runtime_parameters.len());
    prefix.push_str(asset);
    prefix
}

/// The CUDA device library every native source receives in place of vendor
/// headers: conversions, explicit-rounding arithmetic and the inline-PTX
/// primitives (tensor-core MMA, `ldmatrix`, `cp.async`, non-coherent loads,
/// L2 prefetch, warp shuffles and reductions, `dp4a`).
const CUDA_HELPERS: &str = include_str!("cuda_prelude.cuh");

/// The ABI of one tensor parameter or result: its storage macros, extents,
/// strides and, for a row layout, its row geometry.
struct TensorAbi<'a> {
    representation: RepresentationId,
    rank: usize,
    /// First argument word of the tensor's extents (strides follow them).
    word: usize,
    /// Static geometry rendered as constants (S10).
    fixed: &'a StaticTensor,
    dialect: Dialect,
}

impl TensorAbi<'_> {
    fn render(&self, out: &mut String, prefix: &str) {
        render_representation(out, prefix, self.representation);
        let constant = |value| literal(self.dialect, value);
        for axis in 0..self.rank {
            match self.fixed.extents[axis] {
                Some(extent) => out.push_str(&format!(
                    "#define {prefix}_EXTENT_{axis} {}\n",
                    constant(extent)
                )),
                None => out.push_str(&format!(
                    "#define {prefix}_EXTENT_{axis} (seismic_words[{}])\n",
                    self.word + axis
                )),
            }
            match &self.fixed.strides {
                Some(strides) => out.push_str(&format!(
                    "#define {prefix}_STRIDE_{axis} {}\n",
                    constant(strides[axis])
                )),
                None => out.push_str(&format!(
                    "#define {prefix}_STRIDE_{axis} (seismic_words[{}])\n",
                    self.word + self.rank + axis
                )),
            }
        }
        if let registry::RepresentationKind::PackedRows(rows) =
            &registry::representation_info(self.representation).kind
        {
            let last = self
                .rank
                .checked_sub(1)
                .expect("checked row-layout tensors have a packing axis");
            match self.fixed.extents[last] {
                Some(extent) => render_static_rows(out, prefix, rows, extent, self.dialect),
                None => render_symbolic_rows(out, prefix, rows, &format!("{prefix}_EXTENT_{last}")),
            }
        }
    }
}

/// Row geometry of a row-layout tensor whose packing axis is static.
fn render_static_rows(
    out: &mut String,
    prefix: &str,
    rows: &registry::PackedRowLayout,
    extent: u64,
    dialect: Dialect,
) {
    const ADMITTED: &str = "static row geometry was admitted by its canonical layout";
    let constant = |value: Option<u64>| literal(dialect, value.expect(ADMITTED));
    out.push_str(&format!(
        "#define {prefix}_ROW_GROUPS {}\n#define {prefix}_ROW_STRIDE_BYTES {}\n",
        constant(rows.row_groups(extent)),
        constant(rows.row_stride_bytes(extent)),
    ));
    for (index, plane) in rows.planes.iter().enumerate() {
        out.push_str(&format!(
            "#define {prefix}_PLANE_{name}_ROW_OFFSET {}\n#define {prefix}_PLANE_{name}_BYTES_PER_ROW {}\n",
            constant(rows.plane_row_offset(index, extent)),
            constant(rows.plane_bytes_per_row(index, extent)),
            name = native_macro(plane.name),
        ));
    }
}

/// Row geometry of a row-layout tensor as expressions of its packing-axis
/// extent macro `extent` (the same byte math as `PackedRowLayout`).
fn render_symbolic_rows(
    out: &mut String,
    prefix: &str,
    rows: &registry::PackedRowLayout,
    extent: &str,
) {
    let align = registry::ROW_ALIGNMENT;
    let group = rows.group();
    let groups = format!("(({extent} + {}) / {group})", group - 1);
    let groups = match rows.group_multiple {
        1 => groups,
        multiple => format!("(({groups} + {}) / {multiple} * {multiple})", multiple - 1),
    };
    out.push_str(&format!("#define {prefix}_ROW_GROUPS {groups}\n"));
    let mut end = String::from("0");
    for plane in &rows.planes {
        let name = native_macro(plane.name);
        out.push_str(&format!(
            "#define {prefix}_PLANE_{name}_ROW_OFFSET (({end} + {}) / {align} * {align})\n#define {prefix}_PLANE_{name}_BYTES_PER_ROW ({prefix}_ROW_GROUPS * {})\n",
            align - 1,
            plane.bytes_per_group,
        ));
        end = format!("({prefix}_PLANE_{name}_ROW_OFFSET + {prefix}_PLANE_{name}_BYTES_PER_ROW)");
    }
    out.push_str(&format!(
        "#define {prefix}_ROW_STRIDE_BYTES (({end} + {}) / {align} * {align})\n",
        align - 1
    ));
}

fn render_representation(out: &mut String, prefix: &str, representation: RepresentationId) {
    let info = registry::representation_info(representation);
    out.push_str(&format!(
        "#define {prefix}_REPRESENTATION_{} 1\n",
        native_macro(info.representation)
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
        registry::RepresentationKind::PackedRows(rows) => {
            out.push_str(&format!(
                "#define {prefix}_KIND_PACKED 1\n#define {prefix}_LAYOUT_{} 1\n#define {prefix}_LOGICAL_GROUP {}\n#define {prefix}_GROUP_MULTIPLE {}\n#define {prefix}_ROW_ALIGNMENT {}\n#define {prefix}_TILE_ROWS {}\n#define {prefix}_CODE_BITS {}\n",
                native_macro(info.layout.as_str()),
                rows.group(),
                rows.group_multiple,
                registry::ROW_ALIGNMENT,
                rows.tile_rows(),
                rows.packet.planes[0].entry_bits,
            ));
            if info.layout == registry::Layout::Mma16 {
                out.push_str(&format!(
                    "#define {prefix}_MMA_KBLOCK {}\n#define {prefix}_MMA_LANES {}\n",
                    registry::MMA_KBLOCK,
                    registry::MMA_LANES
                ));
            }
            for plane in &rows.planes {
                let plane_prefix = format!("{prefix}_PLANE_{}", native_macro(plane.name));
                out.push_str(&format!(
                    "#define {plane_prefix} 1\n#define {plane_prefix}_BYTES_PER_GROUP {}\n",
                    plane.bytes_per_group
                ));
                if let registry::RowPlaneContent::Codes { shift, bits } = plane.content {
                    out.push_str(&format!(
                        "#define {plane_prefix}_CODE_SHIFT {shift}\n#define {plane_prefix}_CODE_BITS {bits}\n"
                    ));
                }
            }
        }
        registry::RepresentationKind::Packed(layout) => {
            out.push_str(&format!(
                "#define {prefix}_KIND_PACKED 1\n#define {prefix}_LAYOUT_PACKET 1\n"
            ));
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
        render_with(
            dialect,
            representation,
            NativeSpecialization::new().with_static("K", 64),
        )
    }

    fn render_with(
        dialect: Dialect,
        representation: &str,
        specialization: NativeSpecialization,
    ) -> String {
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
        let specialization = specialization.with_param("TILE", 8);
        render_source(
            dialect,
            &logical,
            &bindings,
            implementation,
            &specialization,
            "\n// asset\n",
        )
    }

    #[test]
    fn metal_launch_source_contains_only_its_form_and_code_instantiations() {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "probe.seismic".to_owned(),
            text: PROBE.to_owned(),
        }]))
        .unwrap();
        let entry = module.entries()[0].id;
        let bindings = ElementBindings::new().bind("E", registry::representation("f32").unwrap());
        let logical = module.entry(entry, &bindings).unwrap();
        let implementation = module
            .native_implementation(entry, BackendName::Cuda)
            .unwrap();
        let specialization = NativeSpecialization::new()
            .with_static("K", 64)
            .with_param("TILE", 4);
        let source = render_metal_launch_source(
            &logical,
            &bindings,
            implementation,
            &specialization,
            "\n#ifdef SEISMIC_FORMING_PROBE\ntemplate <uint TILE> kernel void probe() {}\n#endif\n",
            &super::super::plan::LaunchSource {
                ordinal: 0,
                code_variants: vec![vec![4], vec![8]],
            },
        );
        assert!(source.contains("#define SEISMIC_FORMING_PROBE 1\n"));
        assert!(!source.contains("SEISMIC_TUNE_TILE"));
        assert!(source.contains("host_name(\"probe$4\")"));
        assert!(source.contains("host_name(\"probe$8\")"));
    }

    #[test]
    fn scoped_runtime_words_have_stable_entry_then_launch_offsets() {
        let text = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (SPLIT in [1, 2])\n    launch small when N < 16:\n        params (WIDTH in [32, 64], code ROWS in [1, 2])\n        reads (SPLIT)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n    launch large when N >= 16:\n        params (WIDTH in [64, 128])\n        threadgroups (ceil_div(N, WIDTH), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n";
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "scale.seismic".into(),
            text: text.into(),
        }]))
        .unwrap();
        let entry = module.entry_named("scale").unwrap();
        let bindings = ElementBindings::default();
        let logical = module.entry(entry, &bindings).unwrap();
        let implementation = module
            .native_implementation(entry, BackendName::Metal)
            .unwrap();
        let defaults = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        let base = word_count(logical.schema());
        assert_eq!(
            native_word_count(logical.schema(), implementation),
            base + 3
        );
        let source = render_metal_launch_source(
            &logical,
            &bindings,
            implementation,
            &defaults,
            "\n#ifdef SEISMIC_FORMING_SMALL\ntemplate <uint ROWS> kernel void small() {}\n#endif\n",
            &super::super::plan::LaunchSource {
                ordinal: 0,
                code_variants: vec![vec![1], vec![2]],
            },
        );
        assert!(source.contains(&format!(
            "#define SEISMIC_RUNTIME_SPLIT (seismic_words[{base}])\n"
        )));
        assert!(source.contains(&format!(
            "#define SEISMIC_RUNTIME_WIDTH (seismic_words[{}])\n",
            base + 1
        )));
        assert!(!source.contains(&format!(
            "#define SEISMIC_RUNTIME_WIDTH (seismic_words[{}])\n",
            base + 2
        )));
    }

    #[test]
    fn prefix_names_the_layout_and_row_geometry_of_row_storage() {
        // K is static (256): the row geometry renders as constants.
        let source = render_with(
            Dialect::Metal,
            "q5k@rows16",
            NativeSpecialization::new().with_static("K", 256),
        );
        for line in [
            "#define SEISMIC_ELEMENT_E_REPRESENTATION_Q5K 1\n",
            "#define SEISMIC_ELEMENT_E_KIND_PACKED 1\n",
            "#define SEISMIC_ELEMENT_E_LAYOUT_ROWS16 1\n",
            "#define SEISMIC_ELEMENT_E_TILE_ROWS 1\n",
            "#define SEISMIC_ELEMENT_E_PLANE_CODES_HI_BYTES_PER_GROUP 32\n",
            "#define SEISMIC_ELEMENT_E_PLANE_CODES_HI_CODE_SHIFT 4\n",
            "#define SEISMIC_X_LAYOUT_ROWS16 1\n",
            "#define SEISMIC_X_ROW_STRIDE_BYTES ((ulong)192)\n",
            "#define SEISMIC_X_PLANE_CODES_LO_ROW_OFFSET ((ulong)0)\n",
            "#define SEISMIC_X_PLANE_CODES_HI_ROW_OFFSET ((ulong)128)\n",
            "#define SEISMIC_X_PLANE_SCALES_ROW_OFFSET ((ulong)160)\n",
            "#define SEISMIC_X_PLANE_SCALES_BYTES_PER_ROW ((ulong)12)\n",
            "#define SEISMIC_X_PLANE_SUPERS_ROW_OFFSET ((ulong)176)\n",
            "#define SEISMIC_PARAM_0_ROW_STRIDE_BYTES ((ulong)192)\n",
        ] {
            assert!(source.contains(line), "missing {line:?}");
        }
        assert!(!source.contains("SEISMIC_X_PACKET_SIZE"));
        let mma = render(Dialect::Cuda, "q8g32s@mma16");
        assert!(mma.contains("#define SEISMIC_X_LAYOUT_MMA16 1\n"));
        assert!(mma.contains("#define SEISMIC_X_TILE_ROWS 16\n"));
        assert!(mma.contains("#define SEISMIC_X_MMA_KBLOCK 64\n"));
        let packet = render(Dialect::Metal, "q5k");
        assert!(packet.contains("#define SEISMIC_X_LAYOUT_PACKET 1\n"));
        assert!(!packet.contains("ROW_STRIDE_BYTES"));
    }

    #[test]
    fn symbolic_row_geometry_matches_the_registry_byte_math() {
        // Evaluate the rendered expressions for a dynamic K with a tiny
        // arithmetic reader over the macro text.
        let source = render_with(Dialect::Metal, "q6k@rows16", NativeSpecialization::new());
        assert!(source.contains("#define SEISMIC_X_EXTENT_1 (seismic_words["));
        let registry::RepresentationKind::PackedRows(rows) =
            &registry::representation_info(registry::representation("q6k@rows16").unwrap()).kind
        else {
            unreachable!()
        };
        let macros: std::collections::HashMap<&str, &str> = source
            .lines()
            .filter_map(|line| line.strip_prefix("#define "))
            .filter_map(|line| line.split_once(' '))
            .collect();
        fn eval(text: &str, macros: &std::collections::HashMap<&str, &str>, k: u64) -> u64 {
            let mut tokens = Vec::new();
            let mut chars = text.chars().peekable();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    let mut word = String::new();
                    while let Some(&c) = chars
                        .peek()
                        .filter(|c| c.is_ascii_alphanumeric() || **c == '_')
                    {
                        word.push(c);
                        chars.next();
                    }
                    tokens.push(word);
                } else {
                    if !c.is_whitespace() {
                        tokens.push(c.to_string());
                    }
                    chars.next();
                }
            }
            fn expr(
                tokens: &[String],
                at: &mut usize,
                macros: &std::collections::HashMap<&str, &str>,
                k: u64,
            ) -> u64 {
                let mut value = term(tokens, at, macros, k);
                while *at < tokens.len() && (tokens[*at] == "+" || tokens[*at] == "-") {
                    let op = tokens[*at].clone();
                    *at += 1;
                    let rhs = term(tokens, at, macros, k);
                    value = if op == "+" { value + rhs } else { value - rhs };
                }
                value
            }
            fn term(
                tokens: &[String],
                at: &mut usize,
                macros: &std::collections::HashMap<&str, &str>,
                k: u64,
            ) -> u64 {
                let mut value = atom(tokens, at, macros, k);
                while *at < tokens.len() && (tokens[*at] == "*" || tokens[*at] == "/") {
                    let op = tokens[*at].clone();
                    *at += 1;
                    let rhs = atom(tokens, at, macros, k);
                    value = if op == "*" { value * rhs } else { value / rhs };
                }
                value
            }
            fn atom(
                tokens: &[String],
                at: &mut usize,
                macros: &std::collections::HashMap<&str, &str>,
                k: u64,
            ) -> u64 {
                let token = tokens[*at].clone();
                *at += 1;
                if token == "(" {
                    let value = expr(tokens, at, macros, k);
                    *at += 1;
                    value
                } else if let Ok(value) = token.parse() {
                    value
                } else if token == "SEISMIC_X_EXTENT_1" {
                    k
                } else {
                    eval(macros[token.as_str()], macros, k)
                }
            }
            let mut at = 0;
            expr(&tokens, &mut at, macros, k)
        }
        for k in [256u64, 512, 2560, 9216] {
            assert_eq!(
                eval(macros["SEISMIC_X_ROW_STRIDE_BYTES"], &macros, k),
                rows.row_stride_bytes(k).unwrap()
            );
            for (index, plane) in rows.planes.iter().enumerate() {
                let name = native_macro(plane.name);
                assert_eq!(
                    eval(
                        macros[format!("SEISMIC_X_PLANE_{name}_ROW_OFFSET").as_str()],
                        &macros,
                        k
                    ),
                    rows.plane_row_offset(index, k).unwrap()
                );
                assert_eq!(
                    eval(
                        macros[format!("SEISMIC_X_PLANE_{name}_BYTES_PER_ROW").as_str()],
                        &macros,
                        k
                    ),
                    rows.plane_bytes_per_row(index, k).unwrap()
                );
            }
        }
    }

    #[test]
    fn static_tensors_render_constant_canonical_strides() {
        let source = render(Dialect::Metal, "f32");
        // N is dynamic, so x keeps runtime words; the result is fully static.
        assert!(source.contains("#define SEISMIC_X_STRIDE_0 (seismic_words["));
        assert!(source.contains("#define SEISMIC_X_EXTENT_0 (seismic_words["));
        assert!(source.contains("#define SEISMIC_X_EXTENT_1 ((ulong)64)\n"));
        assert!(source.contains("#define SEISMIC_RESULT_0_EXTENT_0 ((ulong)1)\n"));
        assert!(source.contains("#define SEISMIC_RESULT_0_STRIDE_0 ((ulong)1)\n"));
        let fixed = render_with(
            Dialect::Cuda,
            "f32",
            NativeSpecialization::new()
                .with_static("K", 64)
                .with_static("N", 3),
        );
        assert!(fixed.contains("#define SEISMIC_X_EXTENT_0 ((unsigned long long)3)\n"));
        assert!(fixed.contains("#define SEISMIC_X_STRIDE_0 ((unsigned long long)64)\n"));
        assert!(fixed.contains("#define SEISMIC_X_STRIDE_1 ((unsigned long long)1)\n"));
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

    const VULKAN_FEATURES: vulkan::VulkanFeatures = vulkan::VulkanFeatures {
        matrix: false,
        wide_accumulators: true,
        mixed_dot: true,
        f32_atomic_add: false,
        shared_int64_atomics: false,
    };

    #[test]
    fn vulkan_prefix_declares_the_argument_block_views_and_suffix() {
        let source = render(Dialect::Vulkan(VULKAN_FEATURES), "f32");
        assert!(source.starts_with("#version 460\n"));
        for line in [
            "#extension GL_EXT_buffer_reference2 : require\n",
            "#extension GL_EXT_control_flow_attributes : require\n",
            "#extension GL_EXT_integer_dot_product : require\n",
            "#extension GL_EXT_shader_explicit_arithmetic_types : require\n",
            "#define SEISMIC_HAS_MATRIX 0\n",
            "#define SEISMIC_HAS_MIXED_DOT 1\n",
            "#define SEISMIC_DIM_K uint64_t(64ul)\n",
            "#define SEISMIC_X_EXTENT_1 uint64_t(64ul)\n",
            "#define SEISMIC_X_STRIDE_0 (seismic_words[",
            "#define SEISMIC_BUFFER_SCRATCH_PARTIALS 2\n",
            "#define SEISMIC_BUFFER_SCALAR_RESULTS 3\n",
            "#define SEISMIC_SCALAR_RESULTS (seismic_argument_block_t(seismic_arguments).w[3])\n",
            "#define seismic_words (seismic_argument_block_t(seismic_arguments + 32ul).w)\n",
            "#define SEISMIC_READONLY_0 true\n",
            "#define SEISMIC_READONLY_1 false\n",
            "#define SEISMIC_READONLY_2 false\n",
            "layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\n",
            "layout(constant_id = 3) const uint SEISMIC_SHARED_F32 = 16384u;\n",
            "layout(constant_id = 10) const uint SEISMIC_SHARED_UVEC4 = 4096u;\n",
            "shared seismic_shared_uvec4_t { uvec4 seismic_shared_uvec4[SEISMIC_SHARED_UVEC4]; };\n",
        ] {
            assert!(source.contains(line), "missing {line:?}");
        }
        assert!(!source.contains("SEISMIC_BUFFER_WORDS"));
        assert!(!source.contains("GL_KHR_cooperative_matrix"));
        assert!(source.ends_with("\n// asset\n\nvoid main() {\n    SEISMIC_KERNEL();\n}\n"));
        let rows = render_with(
            Dialect::Vulkan(VULKAN_FEATURES),
            "q5k@rows16",
            NativeSpecialization::new().with_static("K", 256),
        );
        assert!(rows.contains("#define SEISMIC_X_ROW_STRIDE_BYTES uint64_t(192ul)\n"));
        let matrix = render(
            Dialect::Vulkan(vulkan::VulkanFeatures {
                matrix: true,
                ..VULKAN_FEATURES
            }),
            "f32",
        );
        assert!(matrix.contains("#extension GL_KHR_cooperative_matrix : require\n#"));
        assert!(matrix.contains("#define SEISMIC_HAS_MATRIX 1\n"));
    }

    #[test]
    fn vulkan_shared_views_cover_the_region() {
        assert_eq!(
            vulkan::shared_view_lengths(10),
            vec![3, 5, 5, 10, 5, 3, 3, 1]
        );
        assert_eq!(vulkan::shared_footprint(10), 16);
        assert_eq!(vulkan::shared_footprint(0), 16);
        assert_eq!(vulkan::shared_footprint(256), 256);
    }

    /// The Vulkan prefix of every representation kind, with an asset using
    /// the ABI and every prelude helper, compiles, seals and validates, and
    /// takes the `OpFmaKHR` binding.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn vulkan_prefix_compiles_for_every_representation_kind() {
        const ASSET: &str = "
void probe() {
    seismic_f32 result = seismic_f32(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const uint lane = gl_LocalInvocationID.x;
    seismic_shared_f32[lane] = float(SEISMIC_X_STRIDE_0 + SEISMIC_DIM_N) * float(SEISMIC_TUNE_TILE);
    barrier();
    float value = seismic_subgroup_sum_f32(seismic_shared_f32[lane]);
    value = seismic_fma_rn(value, seismic_div_rn(value, 3.0), seismic_sqrt_rn(value));
    value += seismic_bf16_to_f32(seismic_f32_to_bf16(value)) + seismic_unpack_f16x2(seismic_pack_f16x2(value, 1.0)).x;
    value += float(seismic_dp4a_s8(int(lane), 3, 1) + seismic_dp4a_u8s8(lane, -3, 0) + seismic_redux_add_s32(1));
    if (SEISMIC_READONLY(SEISMIC_BUFFER_X) && lane == 0u)
        result[0].v = value + seismic_tanh_approx(value);
    seismic_u64(SEISMIC_SCALAR_RESULTS)[0].v = seismic_words[0];
}
";
        for representation in [
            "f32",
            "bf16",
            "q6k@rows16",
            "q5k@rows16",
            "q8g32",
            "gguf_q4_k",
        ] {
            let source = render_with(
                Dialect::Vulkan(VULKAN_FEATURES),
                representation,
                NativeSpecialization::new(),
            )
            .replace("\n// asset\n", ASSET);
            let environment = seismic_vulkan::seal::Environment {
                rounding_rte_32: true,
                denorm_preserve_32: true,
            };
            let module = seismic_vulkan::formation::compile(&source, "probe", environment)
                .unwrap_or_else(|error| panic!("{representation}: {error:?}"));
            seismic_vulkan::seal::fma_khr(&module).expect("the fma binding applies");
        }
    }

    #[test]
    fn cuda_prefix_declares_kernel_parameters_and_helpers() {
        let source = render(Dialect::Cuda, "f32");
        assert!(source.contains("seismic_bf16_to_f32"));
        assert!(!source.contains("#include"));
        // Two dimensions, x (rank 2) and the result (rank 1): 2 + 4 + 2 words.
        assert!(source.contains("struct seismic_words_t { unsigned long long w[8]; };\n"));
        // The shared parameter `x` is read-only; the result and scratch are not.
        assert!(source.contains("#define SEISMIC_KERNEL_PARAMS const unsigned char* __restrict__ seismic_buffer_0, unsigned char* __restrict__ seismic_buffer_1, unsigned char* __restrict__ seismic_buffer_2, seismic_words_t seismic_words_value, unsigned long long* seismic_scalar_results\n"));
        assert!(source.contains("#define SEISMIC_PTR(index) SEISMIC_PTR_(index)\n"));
    }
}
