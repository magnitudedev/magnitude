use seismic::BackendName;
use seismic::coverage::{Toolchain, untaken};

/// Planning gives a weight any representation of its backend's resident
/// layout, so a weight slot that takes one of them takes every one; one it
/// lacks fails only when a model needing it forms. Preprocessing decides, so
/// this checks Vulkan on every host (glslang is linked), and CUDA and Metal
/// where this host has their toolchains.
#[test]
fn weight_slots_take_every_resident_representation() {
    let module = magnitude_kernels::module().unwrap();
    let mut untaken_by_backend = Vec::new();
    for backend in [BackendName::Vulkan, BackendName::Cuda, BackendName::Metal] {
        let toolchain = match Toolchain::open(backend) {
            Ok(toolchain) => toolchain,
            Err(error) => {
                assert_ne!(
                    backend,
                    BackendName::Vulkan,
                    "Vulkan forms on every host: {error}"
                );
                continue;
            }
        };
        for found in untaken(module, &toolchain).unwrap() {
            untaken_by_backend.push(format!("{}: {found}", backend.as_str()));
        }
    }
    assert!(
        untaken_by_backend.is_empty(),
        "untaken representations:\n{}",
        untaken_by_backend.join("\n")
    );
}
