//! Linux startup admission against a real cgroup limit, with abundant host RAM.
//! Run serially with MAGNITUDE_TEST_GGUF and a writable, empty
//! MAGNITUDE_TEST_CGROUP. The tests move only their own process into that group.
#![cfg(target_os = "linux")]

use magnitude_engine::{
    build_native_domain,
    composition::{EngineConfiguration, ResolvedEngineConfiguration},
    error::LoadError,
    options::{ModelPolicy, PackageOptions, ProjectorSelection, standard_service_limits},
};
use magnitude_executor::{
    ExecutionPath,
    platform::{DeviceRequest, MemoryReserves},
};
use seismic::{BackendName, DeviceCatalog, MemoryPoolKind};
use std::path::PathBuf;

fn model() -> ResolvedEngineConfiguration {
    EngineConfiguration {
        package: PackageOptions {
            target: std::env::var_os("MAGNITUDE_TEST_GGUF")
                .expect("set MAGNITUDE_TEST_GGUF")
                .into(),
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy::default(),
        context_tokens: Some(1024),
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device: DeviceRequest::Backend(BackendName::Cpu),
        kernel_cache: None,
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap()
}

struct Window {
    group: PathBuf,
    parent: PathBuf,
    _ballast: Vec<u8>,
}

impl Window {
    fn new(room: u64) -> Self {
        let group = PathBuf::from(
            std::env::var_os("MAGNITUDE_TEST_CGROUP").expect("set MAGNITUDE_TEST_CGROUP"),
        );
        let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap();
        let path = membership
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .unwrap();
        let parent = PathBuf::from("/sys/fs/cgroup").join(path.trim_start_matches('/'));
        assert_ne!(group, parent);
        std::fs::write(group.join("memory.max"), "max").unwrap();
        std::fs::write(group.join("cgroup.procs"), std::process::id().to_string()).unwrap();
        // A stable cgroup capacity that fits the model, but most of it is in
        // use. Touch each page so the cgroup really charges the ballast.
        let mut ballast = vec![0u8; 2 * 1024 * 1024 * 1024];
        for byte in ballast.iter_mut().step_by(4096) {
            *byte = 1;
        }
        let catalog = DeviceCatalog::discover().unwrap();
        let reserve = MemoryReserves::standard()
            .for_domain(
                MemoryPoolKind::HostRam,
                catalog.topology().host_pool().capacity_bytes,
            )
            .planning_bytes;
        let used = std::fs::read_to_string(group.join("memory.current"))
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap();
        std::fs::write(
            group.join("memory.max"),
            (used + reserve + room).to_string(),
        )
        .unwrap();
        eprintln!("startup cgroup: used={used}, reserve={reserve}, added room={room}");
        Self {
            group,
            parent,
            _ballast: ballast,
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        std::fs::write(self.group.join("memory.max"), "max").unwrap();
        std::fs::write(
            self.parent.join("cgroup.procs"),
            std::process::id().to_string(),
        )
        .unwrap();
    }
}

#[test]
#[ignore = "requires a real model and delegated cgroup"]
fn zero_startup_headroom_is_memory_pressure() {
    let resolved = model();
    let _window = Window::new(0);
    let error = match build_native_domain(&resolved.manifest, resolved.host.shared_package()) {
        Ok(_) => panic!("load crossed its planning reserve"),
        Err(error) => error,
    };
    eprintln!("zero headroom: {error}");
    assert!(error.retryable(), "{error:?}");
    assert!(
        matches!(error, LoadError::MemoryPressure { .. }),
        "{error:?}"
    );
}

#[test]
#[ignore = "requires a real model and delegated cgroup"]
fn qualification_deficit_is_retryable_and_recovers() {
    let resolved = model();
    {
        let _window = Window::new(
            std::env::var("MAGNITUDE_TEST_STARTUP_ROOM")
                .map(|value| value.parse().unwrap())
                .unwrap_or(128 * 1024 * 1024),
        );
        let error = match build_native_domain(&resolved.manifest, resolved.host.shared_package()) {
            Ok(_) => panic!("qualification exceeded available startup memory"),
            Err(error) => error,
        };
        eprintln!("qualification deficit: {error}");
        assert!(error.retryable(), "{error:?}");
        assert!(
            matches!(error, LoadError::InsufficientMemory { .. }),
            "{error:?}"
        );
    }
    // The same model succeeds when the temporary constraint is removed.
    build_native_domain(&resolved.manifest, resolved.host.shared_package())
        .unwrap_or_else(|error| panic!("recovery: {error}"));
}

#[test]
#[ignore = "requires a real model; reports a load under the enclosing VM or cgroup"]
fn probe_model_load() {
    let resolved = model();
    let catalog = DeviceCatalog::discover().unwrap();
    eprintln!("before load: {:?}", catalog.host_memory_status().unwrap());
    match build_native_domain(&resolved.manifest, resolved.host.shared_package()) {
        Ok(_) => eprintln!("probe: model loaded and warmed up"),
        Err(error) => panic!("probe: {error}"),
    }
}
