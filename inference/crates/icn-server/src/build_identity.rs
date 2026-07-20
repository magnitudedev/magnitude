use serde_json::{Value, json};

pub(crate) const BINDINGS_REVISION: &str = env!("ICN_BINDINGS_REVISION");
pub(crate) const LLAMA_CPP_REVISION: &str = env!("ICN_LLAMA_CPP_REVISION");
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
    if cfg!(feature = "cuda") {
        backends.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        backends.push("vulkan");
    }
    backends
}

pub(crate) fn json() -> Value {
    let native_build = format!(
        "bindings:{};llama_cpp:{}",
        BINDINGS_REVISION, LLAMA_CPP_REVISION
    );
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "api_version": 1,
        "native_build": native_build,
        "capabilities": [
            "hardware",
            "model_inventory",
            "model_preview",
            "model_download",
            "runtime_model_control",
            "chat_streaming"
        ],
        "bindings_revision": BINDINGS_REVISION,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "target": TARGET,
        "profile": PROFILE,
        "rustc": RUSTC_VERSION,
        "backends": enabled_backends(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_identity_contains_full_pins_and_cpu_backend() {
        assert_eq!(BINDINGS_REVISION.len(), 40);
        assert_eq!(LLAMA_CPP_REVISION.len(), 40);
        assert!(enabled_backends().contains(&"cpu"));

        let identity = json();
        assert_eq!(identity["bindings_revision"], BINDINGS_REVISION);
        assert_eq!(identity["llama_cpp_revision"], LLAMA_CPP_REVISION);
        assert!(
            identity["target"]
                .as_str()
                .is_some_and(|value| !value.is_empty())
        );
    }
}
