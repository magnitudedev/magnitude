//! The macro names an entry parameter defines in an authored native source.
//! The runtime renders these names with their values; the build admits
//! exactly these names when it validates an asset.
//!
//! Every parameter has ordinal names (`SEISMIC_PARAM_<n>`). A parameter
//! whose name is unique among the entry's parameters also has named
//! aliases. A host port defines no name at all: the kernel never accesses
//! it, so callers skip it.

use crate::native_name::native_macro;

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
