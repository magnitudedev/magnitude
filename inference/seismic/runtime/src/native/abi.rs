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
use seismic_lang::checked::{
    LaunchGrid, NativeAsset, NativeImplementation, NativeSpecialization, RecordPort,
};
use seismic_lang::entry::{
    CallSchema, ElementBindings, LogicalEntry, ParameterKind, ResultKind, TensorAccess,
};
use seismic_lang::expr::SymbolValue;
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::ids::RepresentationId;
use seismic_lang::native_parameter::native_parameter_names;
use seismic_lang::registry;
use seismic_lang::repack::WordTable;

mod conversion;
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
    /// The backend the dialect's sources run on.
    pub(crate) fn backend(self) -> registry::BackendName {
        match self {
            Self::Metal(_) => registry::BackendName::Metal,
            Self::Cuda(_) => registry::BackendName::Cuda,
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Vulkan(_) => registry::BackendName::Vulkan,
        }
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
    pub(crate) multiprocessors: u64,
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

/// Argument slots an implementation binds: its bound buffers ([`bound`]),
/// the argument words and the scalar-result slots.
pub(crate) fn buffer_slots(schema: &CallSchema, implementation: &NativeImplementation) -> usize {
    bound(schema, implementation).iter().filter(|bound| **bound).count() + 2
}

/// Whether each of the entry's buffers, in ABI order ([`buffer_writes`]),
/// is bound in the kernel's argument table: every one but a host port,
/// which the kernel never accesses. A host port is still one of the call's
/// buffers, which its hazards order by ([`host_writes`]).
pub(crate) fn bound(schema: &CallSchema, implementation: &NativeImplementation) -> Vec<bool> {
    let parameters = schema.parameters().iter().filter_map(|parameter| match &parameter.kind {
        ParameterKind::Tensor { access, .. } => Some(!matches!(access, TensorAccess::Host { .. })),
        _ => None,
    });
    let rest = buffer_writes(schema, implementation).len() - parameters.clone().count();
    parameters.chain(std::iter::repeat_n(true, rest)).collect()
}

/// Whether each of the entry's buffers, in ABI order ([`buffer_writes`]),
/// is a record port of launch `launch` (`NativeImplementation::launch_records`), bound as
/// the table of its view's records' device addresses: a tensor parameter,
/// or a tensor result, which follow the tensor parameters.
pub(crate) fn records(schema: &CallSchema, implementation: &NativeImplementation, launch: usize) -> Vec<bool> {
    let mut records = vec![false; buffer_writes(schema, implementation).len()];
    let tensor = |kind: &ResultKind| matches!(kind, ResultKind::Tensor { .. });
    for port in implementation.launch_records(launch) {
        let buffer = match port {
            RecordPort::Parameter(name) => {
                tensor_buffer(schema, name).expect("a checked record port is a tensor parameter")
            }
            RecordPort::Result(ordinal) => {
                tensor_buffer_count(schema)
                    + schema.results()[..*ordinal as usize]
                        .iter()
                        .filter(|result| tensor(&result.kind))
                        .count()
            }
        };
        records[buffer] = true;
    }
    records
}

/// Tensor parameters: the leading buffers.
fn tensor_buffer_count(schema: &CallSchema) -> usize {
    schema
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.kind, ParameterKind::Tensor { .. }))
        .count()
}

/// The ABI buffer of tensor parameter `name`: tensor parameters lead the
/// buffers, in schema order.
pub(crate) fn tensor_buffer(schema: &CallSchema, name: &str) -> Option<usize> {
    schema
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.kind, ParameterKind::Tensor { .. }))
        .position(|parameter| parameter.name == name)
}

/// The name of the tensor parameter ABI buffer `buffer` is, one leading the
/// buffers ([`tensor_buffer`]).
pub(crate) fn tensor_buffer_name(schema: &CallSchema, buffer: usize) -> &str {
    &schema
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.kind, ParameterKind::Tensor { .. }))
        .nth(buffer)
        .expect("a tensor parameter's buffer leads the buffers")
        .name
}

/// The ABI buffer holding launch grid `grid`: a tensor parameter, or a
/// scratch buffer, which follow every tensor parameter and result.
pub(crate) fn grid_buffer(
    schema: &CallSchema,
    implementation: &NativeImplementation,
    grid: &LaunchGrid,
) -> usize {
    match grid {
        LaunchGrid::Parameter(name) => {
            tensor_buffer(schema, name).expect("a checked launch grid is a tensor parameter")
        }
        LaunchGrid::Scratch(name) => {
            let scratch = implementation.scratch.len();
            buffer_writes(schema, implementation).len() - scratch
                + implementation
                    .scratch
                    .iter()
                    .position(|buffer| &buffer.name == name)
                    .expect("a checked launch grid is a scratch buffer")
        }
    }
}

/// Whether the entry writes each of its buffers, in ABI order: a tensor
/// parameter as its access declares, then every tensor result and every
/// scratch buffer.
pub(crate) fn buffer_writes(schema: &CallSchema, implementation: &NativeImplementation) -> Vec<bool> {
    let parameters = schema.parameters().iter().filter_map(|parameter| match &parameter.kind {
        ParameterKind::Tensor { access, .. } => {
            Some(matches!(access, TensorAccess::Owned | TensorAccess::Mutable))
        }
        _ => None,
    });
    let results = schema
        .results()
        .iter()
        .filter(|result| matches!(result.kind, ResultKind::Tensor { .. }))
        .map(|_| true);
    let scratch = implementation.scratch.iter().map(|_| true);
    parameters.chain(results).chain(scratch).collect()
}

/// Whether the host rewrites each of the entry's buffers once it sees the
/// launch (`&host mut`), in ABI order as [`buffer_writes`]. The device never
/// writes such a buffer; a launch is ordered as though it did. (A `&host`
/// port the host only reads is ordered as the read every bound buffer is.)
pub(crate) fn host_writes(schema: &CallSchema, implementation: &NativeImplementation) -> Vec<bool> {
    let parameters = schema.parameters().iter().filter_map(|parameter| match &parameter.kind {
        ParameterKind::Tensor { access, .. } => {
            Some(matches!(access, TensorAccess::Host { write: true }))
        }
        _ => None,
    });
    let rest = buffer_writes(schema, implementation).len() - parameters.clone().count();
    parameters.chain(std::iter::repeat_n(false, rest)).collect()
}

/// Number of 64-bit scalar-result slots.
pub(crate) fn scalar_word_count(schema: &CallSchema) -> usize {
    schema
        .results()
        .iter()
        .map(|result| result.kind.class().scalar_words())
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
    asset: NativeAsset<'_>,
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
    asset: NativeAsset<'_>,
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
    asset: NativeAsset<'_>,
    forming: Option<&str>,
) -> String {
    let schema = logical.schema();
    let geometry = static_geometry(logical, specialization);
    // The launch being formed, by ordinal.
    let forming_launch = forming.map(|kernel| {
        implementation
            .launches
            .iter()
            .position(|launch| launch.kernel == kernel)
            .expect("forming guard names a declared launch")
    });
    // The launch whose record ports are the source's: the forming one's,
    // else any (an entry formed whole is not launch-scoped).
    let formed = forming_launch.unwrap_or(0);
    let mut prefix = match dialect {
        Dialect::Metal(features) => metal_header(features),
        Dialect::Cuda(features) => cuda_header(features),
        #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
        Dialect::Vulkan(features) => vulkan::header(features),
    };
    // Whether a call that parks waits inside its kernel, spinning on its
    // awaited words, or at a submission boundary before it: the backend's
    // fact (`BackendName::parks_in_kernels`), stated there alone.
    prefix.push_str(&format!(
        "#define SEISMIC_PARKS_IN_KERNELS {}\n",
        u8::from(dialect.backend().parks_in_kernels())
    ));
    render_axis_constants(&mut prefix, dialect);
    for (name, representation) in bindings.iter() {
        render_representation(
            &mut prefix,
            &format!("SEISMIC_ELEMENT_{}", native_macro(name)),
            representation,
            dialect,
        );
    }
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
            }
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
    let names = native_parameter_names(
        &schema
            .parameters()
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect::<Vec<_>>(),
    );
    for ((ordinal, parameter), names) in schema.parameters().iter().enumerate().zip(&names) {
        match &parameter.kind {
            // A host port takes no argument slot: the kernel never
            // accesses it. Its words keep their places.
            ParameterKind::Tensor {
                axes,
                access: TensorAccess::Host { .. },
                ..
            } => word += axes.len() * 2,
            ParameterKind::Tensor {
                axes,
                representation,
                ..
            } => {
                for symbol in names.tensor_buffers() {
                    prefix.push_str(&format!("#define {symbol} {buffer}\n"));
                }
                let tensor = TensorAbi {
                    representation: *representation,
                    rank: axes.len(),
                    word,
                    fixed: geometry.parameters[ordinal]
                        .as_ref()
                        .expect("a tensor parameter has static geometry"),
                    dialect,
                };
                // A record parameter's records are the host's to land while
                // the work that reads them runs: host-written, where the
                // backend's plain loads would not read them
                // (`BackendName::plain_loads_read_host_writes`). The source's
                // record ports are its launch's; an entry not launch-scoped
                // lists the same at every launch.
                let host_written = !dialect.backend().plain_loads_read_host_writes()
                    && implementation
                        .launch_records(formed)
                        .contains(&RecordPort::Parameter(parameter.name.clone()));
                for symbol in names.tensor_prefixes() {
                    tensor.render(&mut prefix, &symbol);
                    if host_written {
                        prefix.push_str(&format!("#define {symbol}_HOST_WRITTEN 1\n"));
                    }
                }
                buffer += 1;
                word += axes.len() * 2;
            }
            ParameterKind::Scalar { .. } | ParameterKind::Index { .. } => {
                let (symbol, alias) = names.word();
                prefix.push_str(&format!("#define {symbol} (seismic_words[{word}])\n"));
                if let Some(alias) = alias {
                    prefix.push_str(&format!("#define {alias} {symbol}\n"));
                }
                word += 1;
            }
            ParameterKind::Range { .. } => {
                let ([start, end], aliases) = names.range();
                prefix.push_str(&format!("#define {start} (seismic_words[{word}])\n"));
                prefix.push_str(&format!("#define {end} (seismic_words[{}])\n", word + 1));
                if let Some([start_alias, end_alias]) = aliases {
                    prefix.push_str(&format!(
                        "#define {start_alias} {start}\n#define {end_alias} {end}\n"
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
            scalar_word += result.kind.class().scalar_words();
        }
    }
    let runtime_parameters = if forming.is_some() {
        runtime_parameters(implementation)
    } else {
        Vec::new()
    };
    if let Some(current) = forming_launch {
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
    // Bound tensor parameters lead the buffer order; a shared (`&`) one is
    // read-only. Results and scratch follow.
    let read_only = schema
        .parameters()
        .iter()
        .filter_map(|parameter| match &parameter.kind {
            ParameterKind::Tensor {
                access: TensorAccess::Host { .. },
                ..
            } => None,
            ParameterKind::Tensor { access, .. } => Some(matches!(access, TensorAccess::Shared)),
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
        constant_macros(&mut prefix, asset);
        prefix.push_str(asset.text);
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
    // The launch being formed, when it dispatches a written grid authored
    // as a block.
    let written = forming_launch
        .filter(|ordinal| {
            implementation.backend.authors_written_grid_blocks() && implementation.launches[*ordinal].grid.is_some()
        });
    if let (Dialect::Cuda(_), Some(_)) = (dialect, written) {
        prefix.push_str(WRITTEN_GRID_BLOCK);
    }
    debug_assert_eq!(word, word_count(schema) + runtime_parameters.len());
    constant_macros(&mut prefix, asset);
    if matches!(dialect, Dialect::Cuda(_)) {
        let tensors = schema.parameters().iter().filter_map(|parameter| match &parameter.kind {
            ParameterKind::Tensor { representation, .. } => Some(*representation),
            ParameterKind::Scalar { .. } | ParameterKind::Index { .. } | ParameterKind::Range { .. } => None,
        });
        render_tables_read(&mut prefix, bindings.iter().map(|(_, representation)| representation).chain(tensors));
    }
    prefix.push_str(asset.text);
    if let (Dialect::Cuda(_), Some(ordinal)) = (dialect, written) {
        written_grid_loop(&mut prefix, schema, implementation, ordinal, buffer);
    }
    prefix
}

/// CUDA has no indirect launch, so a launch with a written grid
/// (`NativeLaunch::grid`) is authored as `__device__ void` the block
/// function `registry::written_grid_block` names
/// (`BackendName::authors_written_grid_blocks`), taking
/// `SEISMIC_KERNEL_PARAMS`: the body of one block. It runs as the kernel
/// [`written_grid_loop`] appends: a grid-stride loop of the device's
/// resident blocks (the runtime sizes its grid,
/// `NativePrepared::launches_at`) over the written grid's blocks, x fastest.
/// Within the body, and every library function it calls, `blockIdx` and
/// `gridDim` are the block's place in the written grid and that grid.
const WRITTEN_GRID_BLOCK: &str = "static __shared__ uint3 seismic_written_block, seismic_written_grid;\n#define blockIdx seismic_written_block\n#define gridDim seismic_written_grid\n";

/// The kernel of written-grid launch `ordinal` ([`WRITTEN_GRID_BLOCK`]). It
/// reads the grid once its predecessor completed (a programmatic one too).
/// Each block's body starts after its place is published and ends at a
/// barrier, as a block's exit would, before the next reuses its shared
/// memory.
fn written_grid_loop(
    out: &mut String,
    schema: &CallSchema,
    implementation: &NativeImplementation,
    ordinal: usize,
    buffers: usize,
) {
    let launch = &implementation.launches[ordinal];
    let grid = launch.grid.as_ref().expect("a written-grid launch");
    // The grid's argument: its buffer among the bound ones.
    let argument = bound(schema, implementation)[..grid_buffer(schema, implementation, grid)]
        .iter()
        .filter(|bound| **bound)
        .count();
    let (header, instance) = match implementation.code_parameters(ordinal).is_empty() {
        true => ("", ""),
        false => ("template <auto... SEISMIC_CODE>\n", "<SEISMIC_CODE...>"),
    };
    let arguments = (0..buffers)
        .map(|index| format!("seismic_buffer_{index}, "))
        .collect::<String>();
    out.push_str(&format!(
        "\n#undef blockIdx\n#undef gridDim\n{header}__global__ void {kernel}(SEISMIC_KERNEL_PARAMS) {{\n    seismic_dependency_wait();\n    const int *grid = reinterpret_cast<const int *>(seismic_buffer_{argument});\n    const unsigned x = unsigned(grid[0]), y = unsigned(grid[1]), z = unsigned(grid[2]);\n    const unsigned long long blocks = (unsigned long long)x * y * z;\n    for (unsigned long long block = blockIdx.x; block < blocks; block += gridDim.x) {{\n        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {{\n            seismic_written_block.x = unsigned(block % x);\n            seismic_written_block.y = unsigned(block / x % y);\n            seismic_written_block.z = unsigned(block / x / y);\n            seismic_written_grid.x = x;\n            seismic_written_grid.y = y;\n            seismic_written_grid.z = z;\n        }}\n        __syncthreads();\n        {block}{instance}({arguments}seismic_words_value, seismic_scalar_results);\n        __syncthreads();\n    }}\n}}\n",
        kernel = launch.kernel,
        block = registry::written_grid_block(&launch.kernel),
    ));
}

/// `#define SEISMIC_CONST_<NAME> <value>` for each module constant the asset
/// names. A source that names none renders as before the module declared
/// any, so the kernel digests tuning results are stored under keep.
fn constant_macros(prefix: &mut String, asset: NativeAsset<'_>) {
    for (name, value) in asset.constants {
        let symbol = format!("SEISMIC_CONST_{}", native_macro(name));
        if names_symbol(asset.text, &symbol) {
            prefix.push_str(&format!("#define {symbol} {value}\n"));
        }
    }
}

/// Whether `text` has `symbol` as a whole identifier.
fn names_symbol(text: &str, symbol: &str) -> bool {
    let identifier = |character: char| character.is_ascii_alphanumeric() || character == '_';
    text.match_indices(symbol).any(|(at, _)| {
        !text[..at].chars().next_back().is_some_and(identifier)
            && !text[at + symbol.len()..].chars().next().is_some_and(identifier)
    })
}

/// The tables a CUDA source's tensors read that its tables header declares
/// only where read (seismic/lang/build.rs): the magnitude operands of the
/// grid of each records plane (`packets::Records`),
/// `SEISMIC_TABLES_READ_MAGNITUDES(ID)`.
fn render_tables_read(out: &mut String, representations: impl Iterator<Item = RepresentationId>) {
    let mut grids: Vec<String> = representations
        .filter_map(|representation| match &registry::representation_info(representation).kind {
            registry::RepresentationKind::PackedRows(rows) => rows.planes.iter().find_map(|plane| match plane.content {
                registry::RowPlaneContent::Records { grid, .. } => Some(grid_id(grid)),
                _ => None,
            }),
            _ => None,
        })
        .collect();
    grids.sort_unstable();
    grids.dedup();
    let read = grids.iter().map(|grid| format!("(ID) == {grid}")).collect::<Vec<_>>();
    out.push_str(&format!(
        "#define SEISMIC_TABLES_READ_MAGNITUDES(ID) ({})\n",
        if read.is_empty() { "0".to_owned() } else { read.join(" || ") }
    ));
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
/// `_ROW_<ROLE>` (row offset), `_ROW_<ROLE>_BYTES` (bytes per row) and
/// `_ROW_<ROLE>_TAIL` (the gap from its end to the next plane's start, or
/// the row's end, as the layout places them).
fn render_row_roles(out: &mut String, prefix: &str, rows: &registry::PackedRowLayout, dialect: Dialect) {
    for role in registry::RowPlaneRole::ALL {
        let role_name = native_macro(role.name());
        match rows.planes.iter().position(|plane| plane.role() == role) {
            Some(index) => out.push_str(&format!(
                "#define {prefix}_ROW_{role_name} {prefix}_PLANE_{name}_ROW_OFFSET\n#define {prefix}_ROW_{role_name}_BYTES {prefix}_PLANE_{name}_BYTES_PER_ROW\n#define {prefix}_ROW_{role_name}_TAIL ({next} - ({prefix}_PLANE_{name}_ROW_OFFSET + {prefix}_PLANE_{name}_BYTES_PER_ROW))\n",
                name = native_macro(rows.planes[index].name),
                next = rows.planes.get(index + 1).map_or_else(
                    || format!("{prefix}_ROW_STRIDE_BYTES"),
                    |next| format!("{prefix}_PLANE_{}_ROW_OFFSET", native_macro(next.name))
                ),
            )),
            None => out.push_str(&format!(
                "#define {prefix}_ROW_{role_name} {zero}\n#define {prefix}_ROW_{role_name}_BYTES {zero}\n#define {prefix}_ROW_{role_name}_TAIL {zero}\n",
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

/// A registered conversion between two bound elements, as the word form
/// (`WordTable`) of its recipe: `{prefix}` defined, `{prefix}_WORDS` the
/// record's words; for a row layout per group role (`RowPlaneRole::ALL`)
/// `{prefix}_{ROLE}_WORD` and `_BYTES`, its share's first record word and
/// bytes (0 and 0 without the role); `{prefix}_UNIT_COLUMNS`, the columns of
/// a unit, and per code role `{prefix}_{ROLE}_FIRST` and `_UNIT_WORDS`
/// (`WordSpan`); the alignments a kernel stores the
/// record's bytes in (`render_record_alignments`); then the word form as
/// code (`conversion::render_conversion_code`).
fn render_conversion(
    out: &mut String,
    prefix: &str,
    conversion: &registry::RepresentationConversion,
    dialect: Dialect,
) {
    let words = WordTable::of(conversion);
    out.push_str(&format!("#define {prefix} 1\n"));
    out.push_str(&format!("#define {prefix}_WORDS {}\n", words.words()));
    if !words.roles().is_empty() {
        for role in registry::RowPlaneRole::ALL {
            let (first, bytes) = words
                .roles()
                .iter()
                .find(|(candidate, _, _)| *candidate == role)
                .map_or((0, 0), |(_, first, bytes)| (*first, *bytes));
            let role = native_macro(role.name());
            out.push_str(&format!("#define {prefix}_{role}_WORD {first}\n"));
            out.push_str(&format!("#define {prefix}_{role}_BYTES {bytes}\n"));
        }
    }
    // Per code role its segment of the span: its first record word and a
    // unit's words (0 and 0 without the role).
    let span = words.span();
    out.push_str(&format!("#define {prefix}_UNIT_COLUMNS {}\n", span.columns));
    for role in registry::RowPlaneRole::CODES {
        let (first, unit_words) = span
            .segments
            .iter()
            .find(|segment| segment.role == role)
            .map_or((0, 0), |segment| (segment.first, segment.unit_words));
        let role = native_macro(role.name());
        out.push_str(&format!("#define {prefix}_{role}_FIRST {first}\n"));
        out.push_str(&format!(
            "#define {prefix}_{role}_UNIT_WORDS {unit_words}\n"
        ));
    }
    render_record_alignments(out, prefix, words, conversion.destination_layout);
    conversion::render_conversion_code(out, prefix, words, conversion.source_layout, dialect);
}

/// The widest unit (`repack::alignment`, to 16 bytes) each range of record
/// bytes a kernel stores together keeps: its start, its length and its
/// destination's alignment allow. For a row layout, per role
/// (`RowPlaneRole::ALL`, from a row plane's 16-byte-aligned start)
/// `{prefix}_{ROLE}_ALIGNMENT`: a code role's unit's words, a group role's
/// share; and per code role `{prefix}_{ROLE}_REST_ALIGNMENT`: its share's
/// bytes past its units' words (an mma16 record's factors); per code role
/// `{prefix}_{ROLE}_CHUNK_BYTES` and `_CHUNK_ALIGNMENT`, a unit's 32-column
/// chunk placed as a unit of the plane (`{prefix}_CHUNK_COLUMNS` its columns);
/// and in `mma16`, `{prefix}_{ROLE}_FRAGMENT_ALIGNMENT`, a lane's two `k16`
/// steps of the role's codes, and `{prefix}_KBLOCK_BYTES` and
/// `_KBLOCK_ALIGNMENT`, a row's k-block of grid entries, which sit at a
/// multiple of it and need not be a power of two (iq3s's eight 3-byte
/// entries: 24). For the packet
/// layout (from a packet's start): `{prefix}_CODES_ALIGNMENT`, a unit's
/// codes; `{prefix}_LEADING_ALIGNMENT` and `{prefix}_TRAILING_ALIGNMENT`,
/// the record's bytes before and after the codes (to the packet's end).
fn render_record_alignments(
    out: &mut String,
    prefix: &str,
    words: &WordTable,
    layout: registry::ConversionLayout,
) {
    use seismic_lang::repack::alignment;
    let span = words.span();
    let segment = |role| {
        span.segments
            .iter()
            .find(|segment| segment.role == role)
            .map_or((0, 0), |segment| (segment.first, segment.unit_words))
    };
    match layout {
        registry::ConversionLayout::Packet(packet) => {
            let size = packet.packet_size;
            let (first, unit_words) = segment(registry::RowPlaneRole::Codes);
            let end = 4 * (first + span.units * unit_words);
            let stored = (4 * words.words()).min(size);
            let base = alignment(size).min(alignment(4 * first));
            out.push_str(&format!(
                "#define {prefix}_CODES_ALIGNMENT {}\n#define {prefix}_LEADING_ALIGNMENT {}\n#define {prefix}_TRAILING_ALIGNMENT {}\n",
                base.min(alignment(4 * unit_words)),
                base,
                alignment(size)
                    .min(alignment(end))
                    .min(alignment(stored.checked_sub(end).expect("the codes are within the packet"))),
            ));
        }
        registry::ConversionLayout::Rows(rows) => {
            let mma16 = rows.layout == registry::Layout::Mma16;
            out.push_str(&format!(
                "#define {prefix}_CHUNK_COLUMNS {}\n",
                registry::ROWS32_CHUNK
            ));
            if mma16 {
                let (_, unit_words) = segment(registry::RowPlaneRole::Codes);
                let kblock = 4 * unit_words * registry::MMA_KBLOCK as u32 / span.columns;
                out.push_str(&format!(
                    "#define {prefix}_KBLOCK_BYTES {kblock}\n#define {prefix}_KBLOCK_ALIGNMENT {}\n",
                    alignment(kblock)
                ));
            }
            for role in registry::RowPlaneRole::ALL {
                let bytes = words
                    .roles()
                    .iter()
                    .find(|(candidate, _, _)| *candidate == role)
                    .map_or(0, |(_, _, bytes)| *bytes);
                let name = native_macro(role.name());
                if registry::RowPlaneRole::CODES.contains(&role) {
                    let (_, unit_words) = segment(role);
                    out.push_str(&format!(
                        "#define {prefix}_{name}_ALIGNMENT {}\n#define {prefix}_{name}_REST_ALIGNMENT {}\n",
                        alignment(4 * unit_words),
                        alignment(
                            bytes
                                .checked_sub(4 * span.units * unit_words)
                                .expect("a code role's share holds its units' words")
                        ),
                    ));
                    let chunk = 4 * unit_words * registry::ROWS32_CHUNK as u32 / span.columns;
                    out.push_str(&format!(
                        "#define {prefix}_{name}_CHUNK_BYTES {chunk}\n#define {prefix}_{name}_CHUNK_ALIGNMENT {}\n",
                        alignment(chunk)
                    ));
                    if mma16 {
                        out.push_str(&format!(
                            "#define {prefix}_{name}_FRAGMENT_ALIGNMENT {}\n",
                            alignment(2 * unit_words)
                        ));
                    }
                } else {
                    out.push_str(&format!("#define {prefix}_{name}_ALIGNMENT {}\n", alignment(bytes)));
                }
            }
        }
    }
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
    if let registry::RepresentationKind::PackedRows(_) = &info.kind {
        let coded = Coded::of(representation);
        for (axis, value) in CODING_AXES {
            out.push_str(&format!("#define {prefix}_{axis} {}\n", value(&coded)));
        }
        for (form, value) in seismic_lang::forms::forms(dialect.backend(), representation) {
            out.push_str(&format!("#define {prefix}_FORM_{form} {value}\n"));
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
                    "#define {prefix}_MMA_KBLOCK {}\n#define {prefix}_MMA_LANES {}\n#define {prefix}_MMA_STEP {}\n#define {prefix}_MMA_STEP_SLOTS {}\n",
                    registry::MMA_KBLOCK,
                    registry::MMA_LANES,
                    registry::MMA_STEP,
                    registry::MMA_STEP_SLOTS,
                ));
                use registry::PackedRowLayout as Rows;
                for (name, rule) in [
                    ("MMA_FRAGMENT_ROW", Rows::MMA_FRAGMENT_ROW_RULE),
                    ("MMA_FRAGMENT_COLUMN", Rows::MMA_FRAGMENT_COLUMN_RULE),
                    ("MMA_FRAGMENT_BYTE", Rows::MMA_FRAGMENT_BYTE_RULE),
                    ("RECORD_ENTRY", Rows::RECORD_ENTRY_RULE),
                    ("RECORD_FACTOR", Rows::RECORD_FACTOR_RULE),
                ] {
                    // A long rule's text may hold line breaks; a macro's can't.
                    out.push_str(&format!("#define {prefix}_{name}{}\n", rule.replace('\n', " ")));
                }
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
                                "#define {plane_prefix}_ENTRY_BYTES {}\n#define {plane_prefix}_FACTORS_OFFSET ({prefix}_ROW_GROUPS * {})\n#define {plane_prefix}_ENTRY_BITS {}\n#define {plane_prefix}_KBLOCK_ENTRIES {}\n",
                                rows.record_entry_bytes(),
                                rows.record_entry_bytes(),
                                registry::GRID_VALUES * bits,
                                registry::MMA_KBLOCK / u64::from(registry::GRID_VALUES),
                            ));
                        }
                        // The entry bit carrying the coefficient codes.
                        if let Some((_, carried)) = rows.carried() {
                            out.push_str(&format!(
                                "#define {plane_prefix}_CARRIED_BIT {carried}\n"
                            ));
                        }
                    }
                    // An entry's bits past its whole bytes: `ENTRY_BITS` an
                    // entry.
                    registry::RowPlaneContent::GridHigh { bits, .. } => {
                        out.push_str(&format!("#define {plane_prefix}_ENTRY_BITS {bits}\n"))
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
/// codebook (`SEISMIC_CODEBOOK_<NAME>`), and its grid (`grid`,
/// `SEISMIC_GRID_<NAME>`) with the grid's index bits, values a point, signs
/// (`SEISMIC_GRID_SIGNS_<NAME>`, their first bit and their count) and delta
/// (`delta`, its exponent, and its bit); 0 where it has none (and every
/// value 0 for no interpretation).
struct Interpretation {
    codes: Option<registry::CodeKind>,
    offset: i32,
    table: String,
    grid: Option<&'static registry::GridCodes>,
}

impl Interpretation {
    fn none() -> Self {
        Self {
            codes: None,
            offset: 0,
            table: "0".to_owned(),
            grid: None,
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
            | registry::CodeInterpretation::TwosComplement => kind,
            registry::CodeInterpretation::Offset(offset)
            | registry::CodeInterpretation::Odd(offset) => Self {
                offset: *offset,
                ..kind
            },
            registry::CodeInterpretation::Table(codebook) => Self {
                table: id("CODEBOOK", codebook.name),
                ..kind
            },
        }
    }

    /// The axis values of a decoder's value codes: their interpretation's,
    /// a grid's, or E4M3 bytes' (their kind alone).
    fn of_values(codes: &registry::ValueCodes) -> Self {
        match codes {
            registry::ValueCodes::Fields(code) => Self::of(code),
            registry::ValueCodes::Grid(grid) => Self {
                codes: Some(registry::CodeKind::Grid),
                grid: Some(grid),
                ..Self::none()
            },
            registry::ValueCodes::E4m3 => Self {
                codes: Some(registry::CodeKind::E4m3),
                ..Self::none()
            },
        }
    }

    fn codes(&self) -> String {
        self.codes.map_or_else(|| "0".to_owned(), Axis::constant)
    }

    /// A fact of its grid, 0 where it has none.
    fn grid_value(&self, value: fn(&registry::GridCodes) -> u32) -> String {
        self.grid.map_or(0, value).to_string()
    }
}

/// A grid's id (`SEISMIC_GRID_<NAME>`, its tables header's).
fn grid_id(grid: &registry::GridCodes) -> String {
    format!("SEISMIC_GRID_{}", native_macro(grid.grid.name))
}

impl Axis for registry::GridSigns {
    const AXIS: &'static str = "GRID_SIGNS";
    const ALL: &'static [Self] = &registry::GridSigns::ALL;
    fn name(self) -> &'static str {
        registry::GridSigns::name(self)
    }
}

impl Axis for registry::CoefficientKind {
    const AXIS: &'static str = "COEFF";
    const ALL: &'static [Self] = &registry::CoefficientKind::ALL;
    fn name(self) -> &'static str {
        registry::CoefficientKind::name(self)
    }
}

impl Axis for registry::BiasKind {
    const AXIS: &'static str = "BIAS_KIND";
    const ALL: &'static [Self] = &registry::BiasKind::ALL;
    fn name(self) -> &'static str {
        registry::BiasKind::name(self)
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
/// its local codes are `local_bits`-bit `local`, its bias term is of kind
/// `bias` (`SEISMIC_BIAS_KIND_*`), a minimum's sign `bias_sign` (1 without
/// one) and a signed bias's exponent `signed_exponent`, and carried local
/// codes sit at entry bit `carried_bit`, which is past the entry's indices,
/// signs and delta (0: none carried).
struct Coded {
    low: u32,
    high: u32,
    code: Interpretation,
    code_exponent: i32,
    coeff: registry::CoefficientKind,
    scale: registry::ScaleEncoding,
    bias: Option<registry::BiasKind>,
    bias_sign: i32,
    group: u32,
    factor_group: u32,
    exponent: i32,
    local_bits: u32,
    /// The bits of a local code read as its scale code
    /// (`registry::LocalCodes::scale_bits`).
    local_scale_bits: u32,
    local: Interpretation,
    signed_exponent: i32,
    carried_bit: u32,
    /// The code domain and the local coefficient code domain, where its
    /// local scales are codes (`registry::profile::Profile`).
    codes: registry::profile::Domain,
    locals: Option<registry::profile::Domain>,
}

impl Coded {
    fn of(id: RepresentationId) -> Self {
        let axes = registry::decode_axes(id).expect("packed storage has decode axes");
        let profile = axes.profile();
        let codes = profile.codes;
        let locals = match profile.local.scale {
            registry::profile::LocalScale::Code(locals) => Some(locals),
            registry::profile::LocalScale::Stored(_) => None,
        };
        let values = axes.coding.codes();
        let flat = |coeff, scale, bias, group| Self {
            low: axes.low,
            high: axes.high,
            code: Interpretation::of_values(&values),
            code_exponent: axes.code_exponent,
            coeff,
            scale,
            bias,
            bias_sign: 1,
            group,
            factor_group: 0,
            exponent: 0,
            local_bits: 0,
            local_scale_bits: 0,
            local: Interpretation::none(),
            signed_exponent: 0,
            carried_bit: 0,
            codes,
            locals,
        };
        match &axes.coding {
            registry::CodingAxes::Direct {
                scale, bias, group, ..
            } => flat(
                axes.coding.kind(),
                *scale,
                bias.then_some(registry::BiasKind::Minimum),
                *group,
            ),
            registry::CodingAxes::Hierarchical {
                factor,
                factor_group,
                exponent,
                group,
                bits,
                interpretation,
                locals,
            } => {
                let (bias_sign, signed_exponent) = match locals.bias() {
                    None => (1, 0),
                    Some(registry::LocalBias::Minimum(sign)) => (sign.value(), 0),
                    Some(registry::LocalBias::Signed { exponent }) => (1, exponent),
                };
                Self {
                    factor_group: *factor_group,
                    exponent: *exponent,
                    bias: locals.bias().map(registry::LocalBias::kind),
                    bias_sign,
                    signed_exponent,
                    local_bits: *bits,
                    local_scale_bits: locals.scale_bits(*bits),
                    local: Interpretation::of(interpretation),
                    carried_bit: match locals {
                        registry::LocalCodes::Plane { .. } => 0,
                        registry::LocalCodes::Carried { grid, .. } => grid.carried_bit(),
                    },
                    ..flat(axes.coding.kind(), *factor, None, *group)
                }
            }
        }
    }
}

/// The decode axes of a packed row-layout element, each defined as
/// `<prefix>_<AXIS>` (`render_representation`), in the order of the axes
/// tuple (`render_axes_tuple`).
const CODING_AXES: [(&str, fn(&Coded) -> String); 32] = [
    ("CODE_LOW_BITS", |c| c.low.to_string()),
    ("CODE_HIGH_BITS", |c| c.high.to_string()),
    ("CODES", |c| c.code.codes()),
    ("CODES_OFFSET", |c| c.code.offset.to_string()),
    ("CODES_TABLE", |c| c.code.table.clone()),
    ("CODES_GRID", |c| {
        c.code.grid.map_or_else(|| "0".to_owned(), grid_id)
    }),
    ("CODES_GRID_INDEX_BITS", |c| {
        c.code.grid_value(registry::GridCodes::index_bits)
    }),
    ("CODES_GRID_LANES", |c| {
        c.code.grid_value(registry::GridCodes::lanes)
    }),
    ("CODES_EXPONENT", |c| c.code_exponent.to_string()),
    ("COEFF", |c| c.coeff.constant()),
    ("COEFF_SCALE", |c| c.scale.constant()),
    ("COEFF_BIAS", |c| c.bias.map_or_else(|| "0".to_owned(), Axis::constant)),
    ("COEFF_BIAS_SIGN", |c| c.bias_sign.to_string()),
    ("COEFF_GROUP", |c| c.group.to_string()),
    ("COEFF_FACTOR_GROUP", |c| c.factor_group.to_string()),
    ("COEFF_EXPONENT", |c| c.exponent.to_string()),
    ("COEFF_LOCAL_BITS", |c| c.local_bits.to_string()),
    ("COEFF_LOCAL_SCALE_BITS", |c| c.local_scale_bits.to_string()),
    ("COEFF_LOCAL", |c| c.local.codes()),
    ("COEFF_LOCAL_OFFSET", |c| c.local.offset.to_string()),
    ("COEFF_CARRIED", |c| c.carried_bit.to_string()),
    ("CODES_LEAST", |c| c.codes.least.to_string()),
    ("CODES_GREATEST", |c| c.codes.greatest.to_string()),
    ("COEFF_LOCAL_LEAST", |c| {
        c.locals.map_or(0, |locals| locals.least).to_string()
    }),
    ("COEFF_LOCAL_GREATEST", |c| {
        c.locals.map_or(0, |locals| locals.greatest).to_string()
    }),
    ("CODES_GRID_SIGNS", |c| {
        c.code
            .grid
            .map_or_else(|| "0".to_owned(), |grid| grid.signs.constant())
    }),
    ("CODES_GRID_DELTA", |c| {
        c.code.grid_value(|grid| u32::from(grid.delta.is_some()))
    }),
    ("CODES_GRID_DELTA_EXPONENT", |c| {
        c.code.grid_value(|grid| grid.delta.unwrap_or(0))
    }),
    // An entry's first sign bit, its sign bits and its delta bit
    // (`registry::GridCodes::{sign_bit, delta_bit}`).
    ("CODES_GRID_SIGN_BIT", |c| {
        c.code.grid_value(registry::GridCodes::sign_bit)
    }),
    ("CODES_GRID_SIGN_BITS", |c| {
        c.code.grid_value(|grid| grid.signs.bits())
    }),
    ("CODES_GRID_DELTA_BIT", |c| {
        c.code.grid_value(|grid| {
            if grid.delta.is_some() {
                grid.delta_bit()
            } else {
                0
            }
        })
    }),
    ("COEFF_SIGNED_BIAS_EXPONENT", |c| {
        c.signed_exponent.to_string()
    }),
];

/// The tensor's prefix as a function-like macro over its axes, emitted for
/// Vulkan sources (`render_representation`): `<prefix>(F)` =
/// `F(<TUPLE_AXES>, <CODING_AXES>)`, its representation's identity
/// (its registry key, nonzero), its form (`SEISMIC_FORM_*`), a dense form's
/// dtype (`SEISMIC_DTYPE_*`), its storage group (`_LOGICAL_GROUP`) and a row
/// form's decode axes, 0 where it has none. A selector `SEISMIC_AXIS_<AXIS>`
/// takes one of them, and `SEISMIC_AXES_NONE(F)` is the tuple of no tensor
/// (`render_axis_constants`). Through the prefix a library reaches every
/// value by expansion alone, so it can evaluate them in `#if`, where glslang
/// pastes no tokens.
#[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
fn render_axes_tuple(out: &mut String, prefix: &str, info: &registry::RepresentationInfo) {
    let id = info.id.key();
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
    render_axis_values::<registry::BiasKind>(out);
    render_axis_values::<registry::ScaleEncoding>(out);
    render_axis_values::<registry::GridSigns>(out);
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

    const PROBE: &str = "fn probe[N, K](x: &tensor[N, K] E) -> tensor[1] f32:\n    let mut output = tensor[1] f32\n    for i in 0..1:\n        output[i] = f32(x[0, 0])\n    return output\n\nnative probe for cuda from \"probe.cu\":\n    static (K)\n    params (TILE in [4, 8] default 4)\n    scratch partials bytes (N * 4)\n    launch probe:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (TILE, 1, 1)\n";

    fn render(dialect: Dialect, representation: &str) -> String {
        render_with(
            dialect,
            representation,
            NativeSpecialization::new().with_static("K", 64),
        )
    }

    /// An asset of a module that declares no constant.
    fn bare(text: &str) -> NativeAsset<'_> {
        NativeAsset { text, constants: &[] }
    }

    fn render_with(
        dialect: Dialect,
        representation: &str,
        specialization: NativeSpecialization,
    ) -> String {
        render_asset(dialect, representation, specialization, bare("\n// asset\n"))
    }

    /// A rendered source defines each module constant its asset names as a
    /// whole identifier, and no other: a source naming none renders as it
    /// would in a module without constants.
    #[test]
    fn a_source_defines_the_module_constants_it_names() {
        let constants = [("FIRST".to_owned(), 2), ("SECOND".to_owned(), 3)];
        let at = |text| {
            let asset = NativeAsset { text, constants: &constants };
            render_asset(CUDA, "f32", NativeSpecialization::new().with_static("K", 64), asset)
        };
        let source = at("\nint x = SEISMIC_CONST_FIRST;\n");
        assert!(source.contains("#define SEISMIC_CONST_FIRST 2\n"));
        assert!(!source.contains("SEISMIC_CONST_SECOND"));
        assert!(!at("\nint x = SEISMIC_CONST_FIRSTLY;\n").contains("#define SEISMIC_CONST_FIRST "));
        assert_eq!(at("\n// asset\n"), render(CUDA, "f32"));
    }

    fn render_asset(
        dialect: Dialect,
        representation: &str,
        specialization: NativeSpecialization,
        asset: NativeAsset<'_>,
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
            asset,
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
            bare("\n#ifdef SEISMIC_FORMING_PROBE\ntemplate <uint TILE> kernel void probe() {}\n#endif\n"),
            0,
        );
        assert!(source.contains("#define SEISMIC_FORMING_PROBE 1\n"));
        assert!(!source.contains("SEISMIC_TUNE_TILE"));
        assert!(!source.contains("host_name"));
    }

    /// A host port is never the device's write, and is the host's write
    /// where it is `&host mut`; an entry's results and scratch are the
    /// device's writes.
    /// A host port is one of the call's buffers, which hazards order by (a
    /// `&host mut` one as its writer), but no argument of the kernel: the
    /// buffers after it bind one slot lower, and it has no buffer macro.
    #[test]
    fn host_ports_are_the_hosts_writes_and_take_no_argument() {
        let text = "fn request[N](served: &host tensor[N] i32, staged: &host mut tensor[N] i32, out: &mut tensor[1] i32):\n    out[0] = 1\n\nnative request for metal from \"request.metal\":\n    scratch counts bytes (4)\n    launch request:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "request.seismic".into(),
            text: text.into(),
        }]))
        .unwrap();
        let entry = module.entry_named("request").unwrap();
        let bindings = ElementBindings::default();
        let logical = module.entry(entry, &bindings).unwrap();
        let implementation = module
            .native_implementation(entry, BackendName::Metal)
            .unwrap();
        assert_eq!(
            buffer_writes(logical.schema(), implementation),
            [false, false, true, true]
        );
        assert_eq!(
            host_writes(logical.schema(), implementation),
            [false, true, false, false]
        );
        assert_eq!(bound(logical.schema(), implementation), [false, false, true, true]);
        assert_eq!(buffer_slots(logical.schema(), implementation), 4);
        let source = render_source(
            METAL,
            &logical,
            &bindings,
            implementation,
            &NativeSpecialization::new(),
            bare(""),
        );
        assert!(!source.contains("SEISMIC_BUFFER_SERVED"));
        assert!(!source.contains("SEISMIC_BUFFER_STAGED"));
        assert!(source.contains("#define SEISMIC_BUFFER_OUT 0\n"));
        assert!(source.contains("#define SEISMIC_BUFFER_SCRATCH_COUNTS 1\n"));
        assert!(source.contains("#define SEISMIC_BUFFER_WORDS 2\n"));
    }

    #[test]
    fn scoped_runtime_words_have_stable_entry_then_launch_offsets() {
        let text = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (SPLIT in [1, 2] default 1)\n    launch small when N < 16:\n        params (WIDTH in [32, 64] default 32, ROWS in [1, 2] default 1)\n        reads (SPLIT)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n    launch large when N >= 16:\n        params (WIDTH in [64, 128] default 64)\n        threadgroups (ceil_div(N, WIDTH), 1, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n";
        // Its source's `small` takes `ROWS` as a template parameter.
        let module = seismic_lang::native_source::check_fixture(
            SourceSet::new(vec![SourceFile {
                path: "scale.seismic".into(),
                text: text.into(),
            }]),
            &["ROWS"],
        )
        .unwrap();
        let entry = module.entry_named("scale").unwrap();
        let bindings = ElementBindings::default();
        let logical = module.entry(entry, &bindings).unwrap();
        let implementation = module
            .native_implementation(entry, BackendName::Metal)
            .unwrap();
        let defaults = implementation
            .device_independent().unwrap().default_specialization(&NativeSpecialization::new())
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
            bare("\n#ifdef SEISMIC_FORMING_SMALL\ntemplate <uint ROWS> kernel void small() {}\n#endif\n"),
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
        assert!(mma.contains("#define SEISMIC_X_MMA_FRAGMENT_ROW(lane, slot) ((lane) / 4 + 8 * ((slot) % 2))\n"));
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
    /// (hierarchical, a minimum), q8g32s (direct, no local codes), iq2xxs
    /// (a grid carrying its local codes) and iq1s (a signed bias, its scale
    /// code all but the local code's sign bit).
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
                    ("COEFF_BIAS", "SEISMIC_BIAS_KIND_MINIMUM"),
                    ("COEFF_BIAS_SIGN", "-1"),
                    ("COEFF_GROUP", "32"),
                    ("COEFF_FACTOR_GROUP", "256"),
                    ("COEFF_LOCAL_BITS", "6"),
                    ("COEFF_LOCAL_SCALE_BITS", "6"),
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
                    ("COEFF_LOCAL_SCALE_BITS", "4"),
                    ("COEFF_LOCAL", "SEISMIC_CODES_ODD"),
                    ("COEFF_CARRIED", "15"),
                ][..],
            ),
            (
                "iq1s@rows16",
                &[
                    ("COEFF_BIAS", "SEISMIC_BIAS_KIND_SIGNED"),
                    ("COEFF_LOCAL_BITS", "4"),
                    ("COEFF_LOCAL_SCALE_BITS", "3"),
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

    /// A record parameter is host-written where the backend's plain loads
    /// would not read the host's writes (Metal), under each of its prefixes;
    /// a record result and a port bound whole are not, and where plain loads
    /// read them (Vulkan) no port is.
    #[test]
    fn a_record_parameter_is_host_written_where_plain_loads_miss_host_writes() {
        let source = |dialect: Dialect| {
            let backend = dialect.backend();
            let module = check_source(SourceSet::new(vec![SourceFile {
                path: "scale.seismic".to_owned(),
                text: format!(
                    "fn scale[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut result = zeros_like(x)\n    parallel for row in 0..M:\n        parallel for column in 0..N:\n            result[row, column] = x[row, column] * y[row, column]\n    return result\n\nnative scale for {} from \"scale.src\":\n    records (x, result 0)\n    launch scale:\n        threadgroups (M, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
                    backend.as_str()
                ),
            }]))
            .expect("record ports check");
            let entry = module.entries()[0].id;
            let bindings = ElementBindings::new();
            let logical = module.entry(entry, &bindings).expect("entry monomorphizes");
            let implementation = module.native_implementation(entry, backend).expect("native implementation");
            render_source(dialect, &logical, &bindings, implementation, &NativeSpecialization::new(), bare("\n"))
        };
        let metal = source(METAL);
        for line in ["#define SEISMIC_PARAM_0_HOST_WRITTEN 1\n", "#define SEISMIC_X_HOST_WRITTEN 1\n"] {
            assert!(metal.contains(line), "missing {line:?}");
        }
        assert_eq!(metal.matches("_HOST_WRITTEN").count(), 2);
        assert!(!source(Dialect::Vulkan(VULKAN_FEATURES)).contains("_HOST_WRITTEN"));
    }

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
    SEISMIC_CONVERSION_E_U_FIXED(packet, words);
    value |= words[SEISMIC_CONVERSION_E_U_WORDS - 1];
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
