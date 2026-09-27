//! The service executable's build identity: target, toolchain, engine build and backends.

use magnitude_engine::worker::protocol::EngineBuild;
use magnitude_service_contracts::ExecutionBackend;
use magnitude_service_contracts::bootstrap_protocol::IcnBinaryIdentity;

pub const TARGET: &str = env!("SERVICE_BUILD_TARGET");
pub const PROFILE: &str = env!("SERVICE_BUILD_PROFILE");
pub const RUSTC_VERSION: &str = env!("SERVICE_RUSTC_VERSION");

/// The backends this host's engine build executes on (integration spec §11.1).
pub fn compiled_backends() -> Vec<ExecutionBackend> {
    if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            vec![ExecutionBackend::Cpu, ExecutionBackend::Metal]
        } else {
            vec![ExecutionBackend::Cpu]
        }
    } else {
        vec![
            ExecutionBackend::Cpu,
            ExecutionBackend::Cuda,
            ExecutionBackend::Vulkan,
        ]
    }
}

pub fn identity() -> IcnBinaryIdentity {
    IcnBinaryIdentity {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        api_version: 1,
        native_build: native_build(),
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
        backends: compiled_backends(),
    }
}

/// The engine build identity: engine version plus the digest of the engine sources, which
/// includes the native kernel bundle.
pub fn native_build() -> String {
    EngineBuild::current().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_identity_names_the_engine_build_and_cpu_backend() {
        assert!(compiled_backends().contains(&ExecutionBackend::Cpu));
        let identity = identity();
        assert_eq!(identity.native_build, native_build());
        assert!(!identity.target.is_empty());
    }
}
