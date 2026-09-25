//! The retained Cranelift code-generation policy: a target-profile fact
//! (`HostFacts::codegen`) gathered once at discovery and reconstructed
//! into an ISA at native compilation. Nothing is rediscovered late: the
//! triple and every flag are part of the profile identity.

use cranelift_codegen::{
    isa,
    settings::{self, Configurable},
};
use seismic_compiler::errors::TargetError;
use seismic_native_target::NativeCompilationError;
use std::collections::BTreeMap;

/// The host code-generation policy: compiler version, target triple,
/// calling convention, and every shared and ISA flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodegenPolicy {
    pub compiler: String,
    pub triple: String,
    pub call_conv: isa::CallConv,
    pub shared_flags: BTreeMap<String, String>,
    pub isa_flags: BTreeMap<String, String>,
}

// Cranelift floating instructions are strict operation boundaries: only its
// explicit `fma` instruction contracts. Keep NaN canonicalization disabled,
// and pair this codegen contract with the worker's strict FP environment.
const SHARED_FLAGS: [(&str, &str); 5] = [
    ("use_colocated_libcalls", "false"),
    ("is_pic", "false"),
    ("opt_level", "speed"),
    ("machine_code_cfg_info", "true"),
    ("enable_nan_canonicalization", "false"),
];

impl CodegenPolicy {
    /// The host's policy. Fails typed when Cranelift has no backend for the
    /// host or the flag set is rejected.
    pub fn host() -> Result<Self, TargetError> {
        let mut flags = settings::builder();
        for (name, value) in SHARED_FLAGS {
            flags.set(name, value).map_err(|error| {
                TargetError::UnsupportedToolchain(format!("cranelift flag `{name}`: {error}"))
            })?;
        }
        let isa = cranelift_native::builder()
            .map_err(|reason| {
                TargetError::UnsupportedDevice(format!(
                    "cranelift has no backend for this host: {reason}"
                ))
            })?
            .finish(settings::Flags::new(flags))
            .map_err(|error| TargetError::UnsupportedToolchain(error.to_string()))?;
        if isa.pointer_bits() != 64 {
            return Err(TargetError::UnsupportedDevice(format!(
                "the CPU backend requires 64-bit pointers; the host has {}",
                isa.pointer_bits()
            )));
        }
        Ok(Self::of_isa(isa.as_ref()))
    }

    fn of_isa(isa: &dyn isa::TargetIsa) -> Self {
        Self {
            compiler: format!("cranelift-codegen/{}", env!("SEISMIC_CRANELIFT_VERSION")),
            triple: isa.triple().to_string(),
            call_conv: isa.default_call_conv(),
            shared_flags: isa
                .flags()
                .iter()
                .map(|flag| (flag.name.into(), flag.value_string()))
                .collect(),
            isa_flags: isa
                .isa_flags()
                .into_iter()
                .map(|flag| (flag.name.into(), flag.value_string()))
                .collect(),
        }
    }

    /// Reconstructs the ISA this policy describes. Every failure is a
    /// toolchain failure: the policy was produced by this same toolchain.
    pub(crate) fn isa(&self) -> Result<isa::OwnedTargetIsa, NativeCompilationError> {
        let toolchain = |message: String| NativeCompilationError::ToolchainFailure(message);
        let mut shared = settings::builder();
        for (name, value) in &self.shared_flags {
            shared
                .set(name, value)
                .map_err(|error| toolchain(format!("cranelift flag `{name}`: {error}")))?;
        }
        let triple = self
            .triple
            .parse()
            .map_err(|error| toolchain(format!("target triple `{}`: {error}", self.triple)))?;
        let mut builder = isa::lookup(triple).map_err(|error| toolchain(error.to_string()))?;
        for (name, value) in &self.isa_flags {
            builder
                .set(name, value)
                .map_err(|error| toolchain(format!("cranelift isa flag `{name}`: {error}")))?;
        }
        builder
            .finish(settings::Flags::new(shared))
            .map_err(|error| toolchain(error.to_string()))
    }
}
