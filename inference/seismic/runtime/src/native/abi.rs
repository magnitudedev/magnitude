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
use seismic_lang::expr::SymbolValue;
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry;
use seismic_lang::repack::{alignment, WordTable};

#[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
pub(crate) mod vulkan;

/// Source dialects with a generated prefix. CPU implementations are Rust and
/// receive a generated context instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// MSL, with the device's feature macros.
    Metal(MetalFeatures),
    /// CUDA C++, with the device's fact macros.
    Cuda(CudaFeatures),
    /// GLSL 4.60 compute for Vulkan 1.3, with the device's feature macros.
    /// Vulkan is not built on macOS; its prefix still renders in tests there.
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    Vulkan(vulkan::VulkanFeatures),
}

impl Dialect {
    /// Whether the dialect evaluates a conversion's word table at compile
    /// time (`seismic_lang::repack::evaluates_tables`); Vulkan's GLSL takes
    /// the word form as code (`vulkan::render_conversion_code`).
    fn evaluates_tables(self) -> bool {
        seismic_lang::repack::evaluates_tables(match self {
            Self::Metal(_) => registry::SourceDialect::Metal,
            Self::Cuda(_) => registry::SourceDialect::Cuda,
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Vulkan(_) => registry::SourceDialect::Vulkan,
        })
    }
}

/// The Metal device features a native source tests as `SEISMIC_HAS_<NAME>`,
/// from the opened device's facts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MetalFeatures {
    /// Metal 4 tensor operations (`mpp::tensor_ops::matmul2d`) on the GPU's
    /// matrix hardware; the prefix includes their headers.
    pub(crate) tensor_ops: bool,
}

/// The standard library, the tensor-operation headers when enabled, and the
/// device feature macros.
fn metal_header(features: MetalFeatures) -> String {
    let mut header = String::from("#include <metal_stdlib>\n");
    if features.tensor_ops {
        header.push_str(
            "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        );
    }
    header.push_str("using namespace metal;\n");
    header.push_str(&format!(
        "#define SEISMIC_HAS_TENSOR_OPS {}\n",
        u8::from(features.tensor_ops)
    ));
    header
}

/// The CUDA device facts a native source reads as `SEISMIC_<NAME>`, from the
/// opened device's facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CudaFeatures {
    /// Streaming multiprocessors (`SEISMIC_MULTIPROCESSORS`): a kernel that
    /// states its grid sizes its blocks' residency to the waves they fill.
    pub(crate) multiprocessors: u32,
}

impl CudaFeatures {
    /// The facts a source renders at without an opened device (formation
    /// coverage, a kernel's identity): a source reads the multiprocessor
    /// count only as a constant, which no preprocessor group tests, so one
    /// count renders every group the same way as any other.
    pub(crate) const UNOPENED: Self = Self { multiprocessors: 48 };
}

/// The helpers and the device fact macros.
fn cuda_header(features: CudaFeatures) -> String {
    let mut header = String::from(CUDA_HELPERS);
    header.push_str(&format!(
        "#define SEISMIC_MULTIPROCESSORS {}\n",
        features.multiprocessors
    ));
    header
}

/// A 64-bit constant in the dialect's syntax.
fn literal(dialect: Dialect, value: u64) -> String {
    match dialect {
        Dialect::Metal(_) => format!("((ulong){value})"),
        Dialect::Cuda(_) => format!("((unsigned long long){value})"),
        #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
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

pub(crate) use seismic_lang::native_name::native_macro;

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

/// One launch's source (Metal or CUDA). The asset guards each templated
/// kernel with its `SEISMIC_FORMING_<kernel>` name. Code variants are not in
/// the text: they are the program's entries, template instances the
/// toolchain requests, so every variant of a launch shares one source.
/// Runtime parameter values never enter it.
pub(crate) fn render_launch_source(
    dialect: Dialect,
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: &str,
    ordinal: usize,
) -> String {
    render_source_with(
        dialect,
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
        Dialect::Metal(features) => metal_header(features),
        Dialect::Cuda(features) => cuda_header(features),
        #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
        Dialect::Vulkan(features) => vulkan::header(features),
    };
    render_axis_constants(&mut prefix, dialect);
    for (name, representation) in bindings.iter() {
        render_representation(
            &mut prefix,
            &format!("SEISMIC_ELEMENT_{}", native_macro(name)),
            representation,
            dialect,
        );
    }
    let mut converts = false;
    for (from, source) in bindings.iter() {
        for (to, destination) in bindings.iter() {
            if let Some(conversion) = registry::representation_conversion(source, destination) {
                render_conversion(
                    &mut prefix,
                    &format!(
                        "SEISMIC_CONVERSION_{}_{}",
                        native_macro(from),
                        native_macro(to)
                    ),
                    conversion,
                    dialect,
                );
                converts = true;
            }
        }
    }
    // The op codes of every conversion's word form, `SEISMIC_REPACK_OP_<NAME>`,
    // where the table is evaluated.
    if converts && dialect.evaluates_tables() {
        for op in seismic_lang::repack::WordOp::ALL {
            prefix.push_str(&format!(
                "#define SEISMIC_REPACK_OP_{} {}\n",
                op.name(),
                op.code()
            ));
        }
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
                // GLSL's preprocessor evaluates no typed literal: an `#if`
                // guard on a static extent reads it as a plain integer.
                #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
                if matches!(dialect, Dialect::Vulkan(_)) {
                    prefix.push_str(&format!("#define SEISMIC_STATIC_{name} {value}\n"));
                }
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
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
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
    if matches!(dialect, Dialect::Cuda(_)) {
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
        render_representation(out, prefix, self.representation, self.dialect);
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
                None => render_symbolic_rows(
                    out,
                    prefix,
                    rows,
                    &format!("{prefix}_EXTENT_{last}"),
                    self.dialect,
                ),
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
    render_row_roles(out, prefix, rows, dialect);
}

/// The row planes a native library addresses by role
/// (`RowPlaneInfo::role`), every role defined, 0 where the layout has none:
/// `_ROW_<ROLE>` (row offset) and `_ROW_<ROLE>_BYTES` (bytes per row).
fn render_row_roles(out: &mut String, prefix: &str, rows: &registry::PackedRowLayout, dialect: Dialect) {
    for role in registry::RowPlaneRole::ALL {
        let role_name = native_macro(role.name());
        match rows.planes.iter().find(|plane| plane.role() == role) {
            Some(plane) => out.push_str(&format!(
                "#define {prefix}_ROW_{role_name} {prefix}_PLANE_{name}_ROW_OFFSET\n#define {prefix}_ROW_{role_name}_BYTES {prefix}_PLANE_{name}_BYTES_PER_ROW\n",
                name = native_macro(plane.name)
            )),
            None => out.push_str(&format!(
                "#define {prefix}_ROW_{role_name} {zero}\n#define {prefix}_ROW_{role_name}_BYTES {zero}\n",
                zero = literal(dialect, 0)
            )),
        }
    }
}

/// Row geometry of a row-layout tensor as expressions of its packing-axis
/// extent macro `extent` (the same byte math as `PackedRowLayout`).
fn render_symbolic_rows(
    out: &mut String,
    prefix: &str,
    rows: &registry::PackedRowLayout,
    extent: &str,
    dialect: Dialect,
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
    render_row_roles(out, prefix, rows, dialect);
}

/// The registered conversion between two bound elements, as its word form
/// (`seismic_lang::repack::WordTable`, whose docs give the table's format): for
/// every dialect the flag, `_WORDS`, a record's words; for a row layout, per
/// row-plane role (`RowPlaneRole::ALL`) `_<ROLE>_BYTES`, its share's bytes (0
/// without the role); and per code role (codes and high; the `packet` layout's
/// codes are its plane 0) `_<ROLE>_UNIT_WORDS`, a unit's words of the code
/// planes' span (`WordSpan`). Metal and CUDA, which evaluate the table at
/// compile time, also take `_TABLE`, the table's words as a list of literals
/// each dialect's array initializer takes; `_STACK`, the words of an
/// evaluation's stack (0 without values, when no evaluation is instantiated);
/// per row-plane role `_<ROLE>_WORD`, its share's first record word (0 without
/// the role); and the span's `_DIRECTORY`, `_CLASSES`, `_CLASS_COUNT` and per
/// code role `_<ROLE>_FIRST`. CUDA also takes its kernel's tile plans
/// (`render_tile_plans`). Vulkan, whose GLSL evaluates no table at compile
/// time, takes the word form as code (`vulkan::render_conversion_code`). The op
/// codes the table's op words hold are the Metal or CUDA source's, once
/// (`render_source_with`).
fn render_conversion(
    out: &mut String,
    prefix: &str,
    conversion: &registry::RepresentationConversion,
    dialect: Dialect,
) {
    let words = WordTable::of(conversion);
    let evaluated = dialect.evaluates_tables();
    out.push_str(&format!("#define {prefix} 1\n"));
    if evaluated {
        let table = words
            .table()
            .iter()
            .map(|word| format!("{word:#x}u"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("#define {prefix}_TABLE {table}\n"));
    }
    out.push_str(&format!("#define {prefix}_WORDS {}\n", words.words()));
    if evaluated {
        out.push_str(&format!("#define {prefix}_STACK {}\n", words.depth()));
    }
    if !words.roles().is_empty() {
        for role in registry::RowPlaneRole::ALL {
            let (first, bytes) = words
                .roles()
                .iter()
                .find(|(candidate, _, _)| *candidate == role)
                .map_or((0, 0), |(_, first, bytes)| (*first, *bytes));
            let role = native_macro(role.name());
            if evaluated {
                out.push_str(&format!("#define {prefix}_{role}_WORD {first}\n"));
            }
            out.push_str(&format!("#define {prefix}_{role}_BYTES {bytes}\n"));
        }
    }
    // The code planes' span: its unit directory, class list and class
    // count, and per code role its segment's first record word and a unit's
    // words (0 and 0 without the role).
    let span = words.span();
    if evaluated {
        out.push_str(&format!(
            "#define {prefix}_DIRECTORY {}\n#define {prefix}_CLASSES {}\n#define {prefix}_CLASS_COUNT {}\n",
            span.directory, span.classes, span.class_count
        ));
    }
    for role in registry::RowPlaneRole::CODES {
        let (first, unit_words) = span
            .segments
            .iter()
            .find(|segment| segment.role == role)
            .map_or((0, 0), |segment| (segment.first, segment.unit_words));
        let role = native_macro(role.name());
        if evaluated {
            out.push_str(&format!("#define {prefix}_{role}_FIRST {first}\n"));
        }
        out.push_str(&format!(
            "#define {prefix}_{role}_UNIT_WORDS {unit_words}\n"
        ));
    }
    if let Dialect::Cuda(_) = dialect {
        render_tile_plans(out, prefix, words, conversion.source_layout);
    }
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    if !evaluated {
        vulkan::render_conversion_code(out, prefix, words, conversion.source_layout);
    }
}

/// The CUDA kernel's tile plans of a conversion's word form (`WordTable`):
/// the source bytes a warp stages for a tile and where, laid out here so the
/// kernel's front end evaluates no plan. A tile's words, a unit's (plan `c`
/// for a unit of class `c`, its representative's block) or at a group's
/// first tile the fixed words (plan `_CLASS_COUNT`, the fixed blocks), read
/// some of its rows' source bytes: part 0, the bytes of runs that stay where
/// the table puts them, and part 1 + s, the representative's bytes of the
/// runs of slot s, which a unit moves by s's byte displacement. A run's
/// bytes are its group's members', from the lowest to the highest, so a
/// member's staged bits lie as far from its representative's as in the
/// packet (a unit moves each member within the bytes it spans). The warp
/// stages exactly those, coalesced, part by part, each maximal run of used
/// bytes (a segment) at a position aligned to its copy unit: the widest
/// that the source's packet alignment (at most 4, the widest unit a kernel
/// reads a source packet in), the segment's first byte and length and its
/// part's displacements allow.
///
/// `_PLAN_SEGMENTS` are the segments, each `{part, first byte, length,
/// position, copy unit}`, plan by plan from `_PLAN_SEGMENT_FIRST[plan]`
/// (one more entry, their end); `_RUN_POSITIONS` the staged bit of each run
/// a class's representative visits, in visiting order, class by class from
/// `_RUN_FIRST[class]` (one more entry, their end); `_FIXED_POSITIONS` the
/// staged byte of each source byte the fixed words read (0 for any other);
/// `_STAGE_WORDS` the words of a staged row, the most any plan stages (the
/// kernel pads a row of a shared array to odd words, so each starts in its
/// own bank).
fn render_tile_plans(
    out: &mut String,
    prefix: &str,
    words: &WordTable,
    source: &registry::ExternalPacketLayout,
) {
    let packet_size = source.packet_size;
    let span = words.span();
    let mut segments = Vec::new();
    let mut segment_first = Vec::new();
    let mut positions: Vec<Vec<u32>> = Vec::new();
    let mut fixed = Vec::new();
    let mut most = 0;
    for class in (0..span.class_count).map(Some).chain([None]) {
        let blocks = match class {
            Some(class) => vec![words.class(class).representative],
            None => (span.units..words.blocks()).collect(),
        };
        // Each run's part and first bit, and the bytes each part uses.
        let mut used: Vec<Vec<bool>> = Vec::new();
        let mut visits = Vec::new();
        for block in blocks {
            let mut run = 0;
            for group in words.groups(block) {
                for (source, parameter) in group.runs() {
                    let part = class.map_or(0, |class| {
                        let entry = words.class(class).entry(run);
                        if entry.still {
                            0
                        } else {
                            1 + entry.slot as usize
                        }
                    });
                    // Its members' first bits (a unit moves each within the
                    // bytes it spans).
                    let firsts = (0..group.count).map(|member| {
                        source
                            .bit
                            .wrapping_add(group.displacement(member, parameter) as u32)
                    });
                    let low = firsts.clone().min().unwrap() / 8;
                    let high = (firsts.max().unwrap() + source.length - 1) / 8;
                    if used.len() <= part {
                        used.resize(part + 1, vec![false; packet_size as usize]);
                    }
                    used[part][low as usize..=high as usize].fill(true);
                    visits.push((part, source.bit));
                    run += 1;
                }
            }
        }
        segment_first.push(segments.len());
        let mut position = vec![vec![0u32; packet_size as usize]; used.len()];
        let mut cursor = 0u32;
        for (part, used) in used.iter().enumerate() {
            // The alignment of every byte displacement the class's units give
            // the part's slot.
            let moved = class
                .filter(|_| part > 0)
                .map_or(16, |class| words.slot_alignment(class, part as u32 - 1));
            for byte in 0..packet_size {
                if !used[byte as usize] {
                    continue;
                }
                if byte == 0 || !used[byte as usize - 1] {
                    let length = (byte..packet_size)
                        .take_while(|at| used[*at as usize])
                        .count() as u32;
                    let unit = [byte, length, moved]
                        .into_iter()
                        .map(alignment)
                        .fold(source.packet_alignment, u32::min);
                    cursor = cursor.div_ceil(unit) * unit;
                    segments.push([part as u32, byte, length, cursor, unit]);
                }
                position[part][byte as usize] = cursor;
                cursor += 1;
            }
        }
        most = most.max(cursor);
        match class {
            Some(_) => positions.push(
                visits
                    .into_iter()
                    .map(|(part, bit)| 8 * position[part][bit as usize / 8] + bit % 8)
                    .collect(),
            ),
            // (Every byte 0 where the fixed words read none.)
            None => {
                fixed = position
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| vec![0; packet_size as usize])
            }
        }
    }
    segment_first.push(segments.len());
    let list = |values: &mut dyn Iterator<Item = String>| values.collect::<Vec<_>>().join(", ");
    out.push_str(&format!(
        "#define {prefix}_PLAN_SEGMENTS {}\n#define {prefix}_PLAN_SEGMENT_FIRST {}\n",
        list(
            &mut segments
                .iter()
                .map(|[part, first, length, position, unit]| {
                    format!("{{{part}, {first}, {length}, {position}, {unit}}}")
                })
        ),
        list(&mut segment_first.iter().map(|first| format!("{first}"))),
    ));
    let mut first = 0;
    out.push_str(&format!(
        "#define {prefix}_RUN_POSITIONS {}\n#define {prefix}_RUN_FIRST {}\n#define {prefix}_FIXED_POSITIONS {}\n#define {prefix}_STAGE_WORDS {}\n",
        list(&mut positions.iter().flatten().map(|position| format!("{position}"))),
        list(&mut positions.iter().map(Vec::len).chain([0]).map(|runs| {
            first += runs;
            format!("{}", first - runs)
        })),
        list(&mut fixed.iter().map(|position| format!("{position}"))),
        most.div_ceil(4),
    ));
}

fn render_representation(
    out: &mut String,
    prefix: &str,
    representation: RepresentationId,
    dialect: Dialect,
) {
    let info = registry::representation_info(representation);
    // The representation as a flag and as a name a library can paste onto
    // its own names.
    let name = native_macro(info.representation);
    out.push_str(&format!(
        "#define {prefix}_REPRESENTATION_{name} 1\n#define {prefix}_REPRESENTATION {name}\n"
    ));
    if let registry::RepresentationKind::PackedRows(_) = info.kind {
        let coded = Coded::of(representation);
        for (axis, value) in CODING_AXES {
            out.push_str(&format!("#define {prefix}_{axis} {}\n", value(&coded)));
        }
    }
    // GLSL's preprocessor pastes no tokens in `#if`: Vulkan libraries read a
    // tensor's axes through its prefix.
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    if matches!(dialect, Dialect::Vulkan(_)) {
        render_axes_tuple(out, prefix, info);
    }
    #[cfg(not(any(not(target_os = "macos"), test, feature = "coverage")))]
    let _ = dialect;
    out.push_str(&format!(
        "#define {prefix}_DECODED_{} 1\n",
        native_macro(info.decoded.name())
    ));
    // The operand recipe of products over the representation, as
    // `operands(W)` gives it to declarations.
    if info.operands == registry::OperandRecipe::Nvfp4Rows {
        out.push_str(&format!("#define {prefix}_OPERANDS_NVFP4 1\n"));
    }
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
                "#define {prefix}_KIND_PACKED 1\n#define {prefix}_LAYOUT_{} 1\n#define {prefix}_LOGICAL_GROUP {}\n#define {prefix}_GROUP_MULTIPLE {}\n#define {prefix}_ROW_ALIGNMENT {}\n#define {prefix}_TILE_ROWS {}\n",
                native_macro(info.layout.as_str()),
                rows.group(),
                rows.group_multiple,
                registry::ROW_ALIGNMENT,
                rows.tile_rows(),
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
                match plane.content {
                    registry::RowPlaneContent::Codes { bits, .. } => {
                        out.push_str(&format!("#define {plane_prefix}_CODE_BITS {bits}\n"))
                    }
                    registry::RowPlaneContent::Grids { bits, .. }
                    | registry::RowPlaneContent::Records { bits, .. } => {
                        out.push_str(&format!("#define {plane_prefix}_CODE_BITS {bits}\n"));
                        // A records tile's factors follow every group's
                        // entries (`PackedRowLayout::record_byte`).
                        if matches!(plane.content, registry::RowPlaneContent::Records { .. }) {
                            out.push_str(&format!(
                                "#define {plane_prefix}_ENTRY_BYTES {}\n#define {plane_prefix}_FACTORS_OFFSET ({prefix}_ROW_GROUPS * {})\n",
                                rows.record_entry_bytes(),
                                rows.record_entry_bytes()
                            ));
                        }
                        // The entry bit carrying the coefficient codes.
                        if let Some((_, carried)) = rows.carried() {
                            out.push_str(&format!(
                                "#define {plane_prefix}_CARRIED_BIT {carried}\n"
                            ));
                        }
                    }
                    registry::RowPlaneContent::Groups { .. } => {}
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

/// One axis of the native decode ABI: its values are the constants
/// `SEISMIC_<AXIS>_<NAME>`, numbered from 1 in `ALL` order (an axis a form
/// lacks is 0). `render_axis_constants` defines them.
trait Axis: Copy + PartialEq + 'static {
    const AXIS: &'static str;
    const ALL: &'static [Self];
    fn name(self) -> &'static str;

    fn constant(self) -> String {
        format!("SEISMIC_{}_{}", Self::AXIS, self.name())
    }
}

fn render_axis_values<A: Axis>(out: &mut String) {
    for (index, value) in A::ALL.iter().enumerate() {
        out.push_str(&format!("#define {} {}\n", value.constant(), index + 1));
    }
}

impl Axis for registry::RepresentationForm {
    const AXIS: &'static str = "FORM";
    const ALL: &'static [Self] = &registry::RepresentationForm::ALL;
    fn name(self) -> &'static str {
        registry::RepresentationForm::name(self)
    }
}

impl Axis for registry::CodeKind {
    const AXIS: &'static str = "CODES";
    const ALL: &'static [Self] = &registry::CodeKind::ALL;
    fn name(self) -> &'static str {
        registry::CodeKind::name(self)
    }
}

/// A code interpretation's axis values: the interpretation, its offset, its
/// codebook (`SEISMIC_CODEBOOK_<NAME>`), and its grid (`SEISMIC_GRID_<NAME>`)
/// with the grid's index bits and values a point; 0 where it has none (and
/// every value 0 for no interpretation).
struct Interpretation {
    codes: Option<registry::CodeKind>,
    offset: i32,
    table: String,
    grid: String,
    index_bits: u32,
    lanes: u32,
}

impl Interpretation {
    fn none() -> Self {
        Self {
            codes: None,
            offset: 0,
            table: "0".to_owned(),
            grid: "0".to_owned(),
            index_bits: 0,
            lanes: 0,
        }
    }

    fn of(code: &registry::CodeInterpretation) -> Self {
        let id = |kind: &str, name: &str| format!("SEISMIC_{kind}_{}", native_macro(name));
        let kind = Self {
            codes: Some(code.kind()),
            ..Self::none()
        };
        match code {
            registry::CodeInterpretation::Unsigned
            | registry::CodeInterpretation::TwosComplement
            | registry::CodeInterpretation::Odd => kind,
            registry::CodeInterpretation::Offset(offset) => Self {
                offset: *offset,
                ..kind
            },
            registry::CodeInterpretation::Table(codebook) => Self {
                table: id("CODEBOOK", codebook.name),
                ..kind
            },
            registry::CodeInterpretation::Grid(grid) => Self {
                grid: id("GRID", grid.grid.name),
                index_bits: grid.index_bits(),
                lanes: grid.lanes(),
                ..kind
            },
        }
    }

    fn codes(&self) -> String {
        self.codes.map_or_else(|| "0".to_owned(), Axis::constant)
    }
}

impl Axis for registry::CoefficientKind {
    const AXIS: &'static str = "COEFF";
    const ALL: &'static [Self] = &registry::CoefficientKind::ALL;
    fn name(self) -> &'static str {
        registry::CoefficientKind::name(self)
    }
}

impl Axis for registry::ScaleEncoding {
    const AXIS: &'static str = "SCALE";
    const ALL: &'static [Self] = &registry::ScaleEncoding::ALL;
    fn name(self) -> &'static str {
        registry::ScaleEncoding::name(self)
    }
}

/// A row-layout packed element's decode axes (`registry::decode_axes`) as
/// the ABI's flat values, 0 where its coefficients have none: a direct
/// coefficient's scale is `scale`; a hierarchical one's factor is `scale`,
/// its local codes are `local_bits`-bit `local`, its minimum's sign is
/// `bias_sign` (1 without one), and carried local codes sit at entry bit
/// `carried_bit`, which is past the entry's indices and signs (0: none
/// carried).
struct Coded {
    low: u32,
    high: u32,
    code: Interpretation,
    code_exponent: i32,
    coeff: registry::CoefficientKind,
    scale: registry::ScaleEncoding,
    bias: bool,
    bias_sign: i32,
    group: u32,
    factor_group: u32,
    exponent: i32,
    local_bits: u32,
    local: Interpretation,
    carried_bit: u32,
}

impl Coded {
    fn of(id: RepresentationId) -> Self {
        let axes = registry::decode_axes(id).expect("packed storage has decode axes");
        let flat = |coeff, scale, bias, group| Self {
            low: axes.low,
            high: axes.high,
            code: Interpretation::of(&axes.code),
            code_exponent: axes.code_exponent,
            coeff,
            scale,
            bias,
            bias_sign: 1,
            group,
            factor_group: 0,
            exponent: 0,
            local_bits: 0,
            local: Interpretation::none(),
            carried_bit: 0,
        };
        match &axes.coefficients {
            registry::CoefficientAxes::Direct { scale, bias, group } => {
                flat(axes.coefficients.kind(), *scale, *bias, *group)
            }
            registry::CoefficientAxes::Hierarchical {
                factor,
                factor_group,
                exponent,
                group,
                locals,
            } => {
                let hierarchical = Self {
                    factor_group: *factor_group,
                    exponent: *exponent,
                    ..flat(axes.coefficients.kind(), *factor, false, *group)
                };
                match locals {
                    registry::LocalCodes::Plane {
                        bits,
                        interpretation,
                        minimum,
                    } => Self {
                        bias: minimum.is_some(),
                        bias_sign: minimum.map_or(1, registry::MinimumSign::value),
                        local_bits: *bits,
                        local: Interpretation::of(interpretation),
                        ..hierarchical
                    },
                    registry::LocalCodes::Carried {
                        bit,
                        bits,
                        interpretation,
                    } => Self {
                        local_bits: *bits,
                        local: Interpretation::of(interpretation),
                        carried_bit: *bit,
                        ..hierarchical
                    },
                }
            }
        }
    }
}

/// The decode axes of a packed row-layout element, each defined as
/// `<prefix>_<AXIS>` (`render_representation`), in the order of the axes
/// tuple (`render_axes_tuple`).
const CODING_AXES: [(&str, fn(&Coded) -> String); 20] = [
    ("CODE_LOW_BITS", |c| c.low.to_string()),
    ("CODE_HIGH_BITS", |c| c.high.to_string()),
    ("CODES", |c| c.code.codes()),
    ("CODES_OFFSET", |c| c.code.offset.to_string()),
    ("CODES_TABLE", |c| c.code.table.clone()),
    ("CODES_GRID", |c| c.code.grid.clone()),
    ("CODES_GRID_INDEX_BITS", |c| c.code.index_bits.to_string()),
    ("CODES_GRID_LANES", |c| c.code.lanes.to_string()),
    ("CODES_EXPONENT", |c| c.code_exponent.to_string()),
    ("COEFF", |c| c.coeff.constant()),
    ("COEFF_SCALE", |c| c.scale.constant()),
    ("COEFF_BIAS", |c| u32::from(c.bias).to_string()),
    ("COEFF_BIAS_SIGN", |c| c.bias_sign.to_string()),
    ("COEFF_GROUP", |c| c.group.to_string()),
    ("COEFF_FACTOR_GROUP", |c| c.factor_group.to_string()),
    ("COEFF_EXPONENT", |c| c.exponent.to_string()),
    ("COEFF_LOCAL_BITS", |c| c.local_bits.to_string()),
    ("COEFF_LOCAL", |c| c.local.codes()),
    ("COEFF_LOCAL_OFFSET", |c| c.local.offset.to_string()),
    ("COEFF_CARRIED", |c| c.carried_bit.to_string()),
];

/// The tensor's prefix as a function-like macro over its axes, emitted for
/// Vulkan sources (`render_representation`): `<prefix>(F)` =
/// `F(<TUPLE_AXES>, <CODING_AXES>)`, its representation's identity
/// (registry position + 1), its form (`SEISMIC_FORM_*`), a dense form's
/// dtype (`SEISMIC_DTYPE_*`), its storage group (`_LOGICAL_GROUP`) and a row
/// form's decode axes, 0 where it has none. A selector `SEISMIC_AXIS_<AXIS>`
/// takes one of them, and `SEISMIC_AXES_NONE(F)` is the tuple of no tensor
/// (`render_axis_constants`). Through the prefix a library reaches every
/// value by expansion alone, so it can evaluate them in `#if`, where glslang
/// pastes no tokens.
#[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
fn render_axes_tuple(out: &mut String, prefix: &str, info: &registry::RepresentationInfo) {
    let id = info.id.index() + 1;
    let form = info.kind.form();
    let dtype = match &info.kind {
        registry::RepresentationKind::Dense(dtype) => {
            format!("SEISMIC_DTYPE_{}", native_macro(dtype.name()))
        }
        registry::RepresentationKind::PackedRows(_)
        | registry::RepresentationKind::Packed(_)
        | registry::RepresentationKind::External(_) => "0".to_owned(),
    };
    let axes: Vec<String> = CODING_AXES
        .iter()
        .map(|(axis, _)| match form {
            registry::RepresentationForm::Rows => format!("{prefix}_{axis}"),
            registry::RepresentationForm::Dense
            | registry::RepresentationForm::Packet
            | registry::RepresentationForm::External => "0".to_owned(),
        })
        .collect();
    out.push_str(&format!(
        "#define {prefix}(F) F({id}, {}, {dtype}, {prefix}_LOGICAL_GROUP, {})\n",
        form.constant(),
        axes.join(", ")
    ));
}

/// The leading axes of a tensor's axes tuple (`render_axes_tuple`), before
/// its decode axes.
#[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
const TUPLE_AXES: [&str; 4] = ["ID", "FORM", "DTYPE", "LOGICAL_GROUP"];

/// The axis values every native source defines (storage forms, dtypes, code
/// interpretations, coefficient structures and scale encodings), which a
/// tensor's axes take; on Vulkan also the axes tuple's selectors
/// (`render_axes_tuple`). Codebook and grid ids are their tables header's
/// (seismic/lang/build.rs).
fn render_axis_constants(out: &mut String, dialect: Dialect) {
    render_axis_values::<registry::CodeKind>(out);
    render_axis_values::<registry::CoefficientKind>(out);
    render_axis_values::<registry::ScaleEncoding>(out);
    // The axes tuple's form and dtype values (`render_axes_tuple`).
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    if matches!(dialect, Dialect::Vulkan(_)) {
        render_axis_values::<registry::RepresentationForm>(out);
        for (index, dtype) in seismic_lang::types::DType::ALL.into_iter().enumerate() {
            out.push_str(&format!(
                "#define SEISMIC_DTYPE_{} {}\n",
                native_macro(dtype.name()),
                index + 1
            ));
        }
        let names: Vec<&str> = TUPLE_AXES
            .into_iter()
            .chain(CODING_AXES.iter().map(|(axis, _)| *axis))
            .collect();
        let parameters = names.join(", ");
        for name in &names {
            out.push_str(&format!("#define SEISMIC_AXIS_{name}({parameters}) {name}\n"));
        }
        out.push_str(&format!(
            "#define SEISMIC_AXES_NONE(F) F({})\n",
            vec!["0"; names.len()].join(", ")
        ));
    }
    #[cfg(not(any(not(target_os = "macos"), test, feature = "coverage")))]
    let _ = dialect;
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{SourceFile, SourceSet, check_source};
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
    fn a_launch_source_contains_only_its_form() {
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
        let source = render_launch_source(
            METAL,
            &logical,
            &bindings,
            implementation,
            &specialization,
            "\n#ifdef SEISMIC_FORMING_PROBE\ntemplate <uint TILE> kernel void probe() {}\n#endif\n",
            0,
        );
        assert!(source.contains("#define SEISMIC_FORMING_PROBE 1\n"));
        assert!(!source.contains("SEISMIC_TUNE_TILE"));
        assert!(!source.contains("host_name"));
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
        let source = render_launch_source(
            METAL,
            &logical,
            &bindings,
            implementation,
            &defaults,
            "\n#ifdef SEISMIC_FORMING_SMALL\ntemplate <uint ROWS> kernel void small() {}\n#endif\n",
            0,
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
            METAL,
            "q5k@rows16",
            NativeSpecialization::new().with_static("K", 256),
        );
        for line in [
            "#define SEISMIC_ELEMENT_E_REPRESENTATION_Q5K 1\n",
            "#define SEISMIC_ELEMENT_E_KIND_PACKED 1\n",
            "#define SEISMIC_ELEMENT_E_LAYOUT_ROWS16 1\n",
            "#define SEISMIC_ELEMENT_E_TILE_ROWS 1\n",
            "#define SEISMIC_ELEMENT_E_PLANE_CODES_HI_BYTES_PER_GROUP 32\n",
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
        let mma = render(CUDA, "q8g32s@mma16");
        assert!(mma.contains("#define SEISMIC_X_LAYOUT_MMA16 1\n"));
        assert!(mma.contains("#define SEISMIC_X_TILE_ROWS 16\n"));
        assert!(mma.contains("#define SEISMIC_X_MMA_KBLOCK 64\n"));
        let packet = render(METAL, "q5k");
        assert!(packet.contains("#define SEISMIC_X_LAYOUT_PACKET 1\n"));
        assert!(!packet.contains("ROW_STRIDE_BYTES"));
    }

    #[test]
    fn symbolic_row_geometry_matches_the_registry_byte_math() {
        // Evaluate the rendered expressions for a dynamic K with a tiny
        // arithmetic reader over the macro text.
        let source = render_with(METAL, "q6k@rows16", NativeSpecialization::new());
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
        let source = render(METAL, "f32");
        // N is dynamic, so x keeps runtime words; the result is fully static.
        assert!(source.contains("#define SEISMIC_X_STRIDE_0 (seismic_words["));
        assert!(source.contains("#define SEISMIC_X_EXTENT_0 (seismic_words["));
        assert!(source.contains("#define SEISMIC_X_EXTENT_1 ((ulong)64)\n"));
        assert!(source.contains("#define SEISMIC_RESULT_0_EXTENT_0 ((ulong)1)\n"));
        assert!(source.contains("#define SEISMIC_RESULT_0_STRIDE_0 ((ulong)1)\n"));
        let fixed = render_with(
            CUDA,
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
        let source = render(METAL, "f16");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_F16 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_DENSE 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_DECODED_F16 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 2\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 1\n"));
        assert!(source.contains("#define SEISMIC_X_REPRESENTATION_F16 1\n"));
        assert!(source.contains("#define SEISMIC_RESULT_0_REPRESENTATION_F32 1\n"));
    }

    /// A row element's decode axes, one form of each coefficient kind: q4k
    /// (hierarchical, a minimum), q8g32s (direct, no local codes) and iq2xxs
    /// (a grid carrying its local codes).
    #[test]
    fn prefix_names_the_decode_axes_of_row_storage() {
        for (representation, axes) in [
            (
                "q4k@rows16",
                &[
                    ("CODE_LOW_BITS", "4"),
                    ("CODE_HIGH_BITS", "0"),
                    ("CODES", "SEISMIC_CODES_UNSIGNED"),
                    ("COEFF", "SEISMIC_COEFF_HIER"),
                    ("COEFF_SCALE", "SEISMIC_SCALE_F16"),
                    ("COEFF_BIAS", "1"),
                    ("COEFF_BIAS_SIGN", "-1"),
                    ("COEFF_GROUP", "32"),
                    ("COEFF_FACTOR_GROUP", "256"),
                    ("COEFF_LOCAL_BITS", "6"),
                    ("COEFF_LOCAL", "SEISMIC_CODES_UNSIGNED"),
                ][..],
            ),
            (
                "q8g32s@rows16",
                &[
                    ("CODE_LOW_BITS", "8"),
                    ("CODES", "SEISMIC_CODES_TWOS"),
                    ("COEFF", "SEISMIC_COEFF_DIRECT"),
                    ("COEFF_BIAS", "0"),
                    ("COEFF_GROUP", "32"),
                    ("COEFF_LOCAL", "0"),
                ][..],
            ),
            (
                "iq2xxs@rows16",
                &[
                    ("CODE_LOW_BITS", "16"),
                    ("CODES", "SEISMIC_CODES_GRID"),
                    ("CODES_GRID", "SEISMIC_GRID_IQ2XXS"),
                    ("CODES_GRID_INDEX_BITS", "8"),
                    ("CODES_GRID_LANES", "8"),
                    ("COEFF", "SEISMIC_COEFF_HIER"),
                    ("COEFF_LOCAL_BITS", "4"),
                    ("COEFF_LOCAL", "SEISMIC_CODES_ODD"),
                    ("COEFF_CARRIED", "15"),
                ][..],
            ),
        ] {
            let source = render(CUDA, representation);
            for (axis, value) in axes {
                let line = format!("#define SEISMIC_ELEMENT_E_{axis} {value}\n");
                assert!(source.contains(&line), "{representation}: missing {line:?}");
            }
        }
    }

    #[test]
    fn prefix_describes_packed_planes() {
        let source = render(METAL, "q8g32");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_Q8G32 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_PACKED 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 36\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_COUNT 2\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_NAME_WORDS 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_BYTES_PER_GROUP 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_NAME_SCALE 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_OFFSET 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_STORAGE_F32 1\n"));
    }

    #[test]
    fn prefix_describes_external_packets() {
        let source = render(METAL, "gguf_q4_k");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q4_K 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_EXTERNAL 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 144\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 256\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_COUNT 0\n"));
    }

    #[test]
    fn specialization_renders_constants_and_scratch_follows_results() {
        let source = render(METAL, "f32");
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

    const METAL: Dialect = Dialect::Metal(MetalFeatures { tensor_ops: false });

    #[test]
    fn metal_prefix_includes_tensor_operations_only_when_the_device_has_them() {
        let plain = render(METAL, "f32");
        assert!(plain.starts_with("#include <metal_stdlib>\nusing namespace metal;\n"));
        assert!(plain.contains("#define SEISMIC_HAS_TENSOR_OPS 0\n"));
        assert!(!plain.contains("metal_tensor"));
        let tensor = render(Dialect::Metal(MetalFeatures { tensor_ops: true }), "f32");
        for line in [
            "#include <metal_tensor>\n",
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
            "#define SEISMIC_HAS_TENSOR_OPS 1\n",
        ] {
            assert!(tensor.contains(line), "missing {line:?}");
        }
    }

    const VULKAN_FEATURES: vulkan::VulkanFeatures = vulkan::VulkanFeatures {
        subgroup_lanes: 32,
        float16: true,
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
            "#define SEISMIC_STATIC_K 64\n",
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
        assert!(source.contains("#define SEISMIC_SUBGROUP_LANES 32\n"));
        assert!(source.contains("#define SEISMIC_HAS_FLOAT16 1\n"));
        assert!(source.contains("GL_EXT_shader_subgroup_extended_types_float16"));
        let fp32_wide = render(
            Dialect::Vulkan(vulkan::VulkanFeatures {
                subgroup_lanes: 64,
                float16: false,
                ..VULKAN_FEATURES
            }),
            "f32",
        );
        assert!(fp32_wide.contains("#define SEISMIC_SUBGROUP_LANES 64\n"));
        assert!(fp32_wide.contains("#define SEISMIC_HAS_FLOAT16 0\n"));
        assert!(!fp32_wide.contains("GL_EXT_shader_subgroup_extended_types_float16"));
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
                float16: true,
                rounding_rte_32: true,
                denorm_preserve_32: true,
            };
            let module = seismic_vulkan::formation::compile(&source, "probe", environment)
                .unwrap_or_else(|error| panic!("{representation}: {error:?}"));
            seismic_vulkan::seal::fma_khr(&module).expect("the fma binding applies");
        }
    }

    /// Every registered conversion's word form, as the Vulkan prefix
    /// renders it, compiles, seals and validates.
    #[test]
    fn vulkan_conversion_code_compiles_for_every_conversion() {
        const ASSET: &str = "
layout(local_size_x = 1) in;
void probe() {
    const uint64_t packet = uint64_t(gl_WorkGroupID.x) * 4096ul;
    uint words[SEISMIC_CONVERSION_E_U_WORDS];
    SEISMIC_CONVERSION_E_U_UNIT(packet, gl_WorkGroupID.z, words);
    uint value = words[0];
    uint shares[SEISMIC_CONVERSION_E_U_WORDS];
    SEISMIC_CONVERSION_E_U_SCALES_SHARE(packet, shares);
    value |= shares[0];
    SEISMIC_CONVERSION_E_U_SUPERS_SHARE(packet, shares);
    value |= shares[0];
    seismic_u32(uint64_t(gl_WorkGroupID.y) * 4ul).v = value;
}
";
        for conversion in registry::representation_conversions() {
            let mut source = vulkan::header(VULKAN_FEATURES);
            render_conversion(
                &mut source,
                "SEISMIC_CONVERSION_E_U",
                conversion,
                Dialect::Vulkan(VULKAN_FEATURES),
            );
            source.push_str(ASSET);
            source.push_str(vulkan::SUFFIX);
            let environment = seismic_vulkan::seal::Environment {
                float16: true,
                rounding_rte_32: true,
                denorm_preserve_32: true,
            };
            seismic_vulkan::formation::compile(&source, "probe", environment).unwrap_or_else(
                |error| {
                    panic!(
                        "{} -> {}: {error:?}",
                        registry::representation_info(conversion.source).name,
                        registry::representation_info(conversion.destination).name
                    )
                },
            );
        }
    }

    const CUDA: Dialect = Dialect::Cuda(CudaFeatures::UNOPENED);

    #[test]
    fn cuda_prefix_declares_kernel_parameters_and_helpers() {
        let source = render(CUDA, "f32");
        assert!(source.contains("seismic_bf16_to_f32"));
        assert!(source.contains("#define SEISMIC_MULTIPROCESSORS 48\n"));
        assert!(!source.contains("#include"));
        // Two dimensions, x (rank 2) and the result (rank 1): 2 + 4 + 2 words.
        assert!(source.contains("struct seismic_words_t { unsigned long long w[8]; };\n"));
        // The shared parameter `x` is read-only; the result and scratch are not.
        assert!(source.contains("#define SEISMIC_KERNEL_PARAMS const unsigned char* __restrict__ seismic_buffer_0, unsigned char* __restrict__ seismic_buffer_1, unsigned char* __restrict__ seismic_buffer_2, seismic_words_t seismic_words_value, unsigned long long* seismic_scalar_results\n"));
        assert!(source.contains("#define SEISMIC_PTR(index) SEISMIC_PTR_(index)\n"));
    }
}
