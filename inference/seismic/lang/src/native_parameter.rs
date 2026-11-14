//! The macro names an entry parameter defines in an authored native source.
//! The runtime renders these names with their values; the build admits
//! exactly these names when it validates an asset.
//!
//! Every parameter has ordinal names (`SEISMIC_PARAM_<n>`). A parameter
//! whose name is unique among the entry's parameters also has named
//! aliases. A host port defines no name at all: the kernel never accesses
//! it, so callers skip it.

use crate::checked::ElementSummary;
use crate::ids::RepresentationId;
use crate::native_name::native_macro;
use crate::registry;

/// How a native kernel reaches a tensor: through its raw buffer, or (a floating dense tensor it
/// writes) only through its element owner's handle (`Out`), which defines no buffer index. This is
/// the element's part of which tensors have handles; [`parameter_reach`] adds a parameter's
/// access and record port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnerReach {
    Raw,
    Owner,
    /// Decided at formation by the element parameter's binding.
    Bound,
}

/// The reach of a tensor of `representation`: handles for F32, BF16 and F16 dense tensors.
pub fn representation_reach(representation: RepresentationId) -> OwnerReach {
    match &registry::representation_info(representation).kind {
        registry::RepresentationKind::Dense(dtype) if dtype.is_float() => OwnerReach::Owner,
        _ => OwnerReach::Raw,
    }
}

/// The reach of a tensor parameter whose element has reach `element`. A tensor the kernel mutates
/// has its element's reach, except a record port: it is lent records the repack's conversions land
/// and the streaming copies move whole, so its representation's owner is the conversion, not a
/// handle. A tensor the kernel only reads keeps its buffer index. This is the single statement of
/// which parameters have handles: the runtime's ABI and the build's symbol check both call it.
pub fn parameter_reach(element: OwnerReach, mutated: bool, record_port: bool) -> OwnerReach {
    if mutated && !record_port {
        element
    } else {
        OwnerReach::Raw
    }
}

/// The reach of a tensor whose element is `element`.
pub fn element_reach(element: &ElementSummary) -> OwnerReach {
    match element {
        ElementSummary::Fixed(name) => registry::representation(name)
            .map_or(OwnerReach::Raw, representation_reach),
        ElementSummary::Parameter(_) => OwnerReach::Bound,
    }
}

/// Result `ordinal`'s owner-handle macro: the handle expression (CUDA, Vulkan), or with
/// `declaration` (Metal) the macro that declares the kernel parameter.
pub fn result_handle(ordinal: usize, declaration: bool) -> String {
    if declaration {
        format!("SEISMIC_RESULT_{ordinal}_PARAM")
    } else {
        format!("SEISMIC_RESULT_{ordinal}_OUT")
    }
}

/// The names of one parameter, at its ordinal among the entry's parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeParameterNames {
    ordinal: usize,
    /// `native_macro` of the parameter's name, when it is unique.
    named: Option<String>,
}

/// The names of each parameter, from every parameter's name in ABI order.
pub fn native_parameter_names(names: &[&str]) -> Vec<NativeParameterNames> {
    names
        .iter()
        .enumerate()
        .map(|(ordinal, name)| NativeParameterNames {
            ordinal,
            named: (names.iter().filter(|candidate| *candidate == name).count() == 1)
                .then(|| native_macro(name)),
        })
        .collect()
}

impl NativeParameterNames {
    /// A tensor's buffer-index macros: `SEISMIC_BUFFER_<NAME>` when named,
    /// then `SEISMIC_PARAM_<n>_BUFFER`.
    pub fn tensor_buffers(&self) -> impl Iterator<Item = String> + '_ {
        self.named
            .iter()
            .map(|name| format!("SEISMIC_BUFFER_{name}"))
            .chain(std::iter::once(format!("SEISMIC_PARAM_{}_BUFFER", self.ordinal)))
    }

    /// The owner-handle macros of a floating tensor a kernel mutates: `SEISMIC_OUT_<NAME>` when
    /// named, then `SEISMIC_OUT_<n>` (CUDA, Vulkan: the handle expression; Metal with `declaration`: the
    /// `_PARAM(name)` macro that declares the kernel parameter).
    pub fn tensor_handles(&self, declaration: bool) -> impl Iterator<Item = String> + '_ {
        let suffix = if declaration { "_PARAM" } else { "" };
        self.named
            .iter()
            .map(move |name| format!("SEISMIC_OUT_{name}{suffix}"))
            .chain(std::iter::once(format!("SEISMIC_OUT_{}{suffix}", self.ordinal)))
    }

    /// The prefixes of a tensor's storage, extent and stride macros:
    /// `SEISMIC_PARAM_<n>`, then `SEISMIC_<NAME>` when named.
    pub fn tensor_prefixes(&self) -> impl Iterator<Item = String> + '_ {
        std::iter::once(format!("SEISMIC_PARAM_{}", self.ordinal))
            .chain(self.named.iter().map(|name| format!("SEISMIC_{name}")))
    }

    /// A scalar or index word's macro, `SEISMIC_PARAM_<n>`, and its alias
    /// `SEISMIC_PARAM_<NAME>` when named.
    pub fn word(&self) -> (String, Option<String>) {
        (
            format!("SEISMIC_PARAM_{}", self.ordinal),
            self.named.as_ref().map(|name| format!("SEISMIC_PARAM_{name}")),
        )
    }

    /// A range's start and end macros, `SEISMIC_PARAM_<n>_START` and
    /// `_END`, and their aliases `SEISMIC_PARAM_<NAME>_START` and `_END`
    /// when named.
    pub fn range(&self) -> ([String; 2], Option<[String; 2]>) {
        let pair = |prefix: &str| [format!("{prefix}_START"), format!("{prefix}_END")];
        (
            pair(&format!("SEISMIC_PARAM_{}", self.ordinal)),
            self.named
                .as_ref()
                .map(|name| pair(&format!("SEISMIC_PARAM_{name}"))),
        )
    }
}
