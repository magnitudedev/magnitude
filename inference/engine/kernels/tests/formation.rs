//! Every native kernel of this crate forms with each GPU backend's real
//! toolchain, and every preprocessor group of its sources is formed under
//! some configuration its declaration admits.
//!
//! A backend whose toolchain this host lacks (NVRTC off Linux and Windows,
//! Metal off macOS) is skipped with a note, and Metal configurations the
//! host's compiler cannot form are reported; `MAGNITUDE_REQUIRE_KERNEL_FORMATION=1`
//! (set by CI) makes either a failure.

use seismic::coverage::{self, CoverageError};
use seismic::BackendName;

fn required() -> bool {
    std::env::var_os("MAGNITUDE_REQUIRE_KERNEL_FORMATION").is_some()
}

fn cover(backend: BackendName) {
    let module = magnitude_kernels::module().expect("the kernel bundle decodes");
    let coverage = match coverage::cover(module, backend) {
        Ok(coverage) => coverage,
        Err(CoverageError::ToolchainUnavailable(reason)) if !required() => {
            eprintln!("skipping {}: {reason}", backend.as_str());
            return;
        }
        Err(error) => panic!("{} coverage did not run: {error}", backend.as_str()),
    };
    eprintln!("{}: {coverage}", backend.as_str());
    assert!(
        coverage.complete(),
        "{} kernels are not completely formed:\n{coverage}",
        backend.as_str()
    );
    assert!(
        coverage.unformable.is_empty() || !required(),
        "{} configurations this host cannot form:\n{coverage}",
        backend.as_str()
    );
}

#[test]
fn vulkan_kernels_form() {
    cover(BackendName::Vulkan);
}

#[test]
fn cuda_kernels_form() {
    cover(BackendName::Cuda);
}

#[test]
fn metal_kernels_form() {
    cover(BackendName::Metal);
}
