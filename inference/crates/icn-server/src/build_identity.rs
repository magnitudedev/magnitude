use icn_contracts::bootstrap_protocol::IcnBinaryIdentity;
use sha2::{Digest, Sha256};

pub(crate) const BINDINGS_REVISION: &str = env!("ICN_BINDINGS_REVISION");
pub(crate) const NATIVE_BACKEND_REVISION: &str = env!("ICN_NATIVE_BACKEND_REVISION");
pub(crate) const TARGET: &str = env!("ICN_BUILD_TARGET");
pub(crate) const PROFILE: &str = env!("ICN_BUILD_PROFILE");
pub(crate) const RUSTC_VERSION: &str = env!("ICN_RUSTC_VERSION");

pub(crate) fn enabled_backends() -> Vec<&'static str> {
    let mut backends = vec!["cpu"];
    // The pinned bindings enable Metal through a target-specific dependency on
    // Apple Silicon even when the top-level Cargo feature is not repeated.
    if cfg!(feature = "metal") || (cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        backends.push("metal");
    }
    if cfg!(feature = "cuda") || cfg!(feature = "cuda-no-vmm") {
        backends.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        backends.push("vulkan");
    }
    backends
}

pub(crate) fn identity() -> IcnBinaryIdentity {
    IcnBinaryIdentity {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        api_version: 1,
        native_build: native_build(),
        backend_module_abi: backend_module_abi(),
        capabilities: [
            "hardware",
            "model_catalog",
            "model_installed",
            "model_assessment",
            "model_downloads",
            "model_residency",
            "chat_streaming",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        target: TARGET.to_owned(),
        profile: PROFILE.to_owned(),
        rustc: RUSTC_VERSION.to_owned(),
        backends: enabled_backends().into_iter().map(str::to_owned).collect(),
    }
}

pub(crate) fn native_build() -> String {
    let mut digest = Sha256::new();
    digest.update(BINDINGS_REVISION.as_bytes());
    digest.update([0]);
    digest.update(NATIVE_BACKEND_REVISION.as_bytes());
    format!("native_{:x}", digest.finalize())
}

pub(crate) fn backend_module_abi() -> String {
    format!("llama-backend-{BINDINGS_REVISION}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_identity_contains_full_pins_and_cpu_backend() {
        assert_eq!(BINDINGS_REVISION.len(), 40);
        assert_eq!(NATIVE_BACKEND_REVISION.len(), 40);
        assert!(enabled_backends().contains(&"cpu"));

        let identity = identity();
        assert_eq!(identity.native_build, native_build());
        assert!(!identity.target.is_empty());
    }
}
