//! Every native implementation of the kernel module forms, at its base
//! configuration and in parallel, with the backend's real toolchain, so a
//! formation-time refusal of any kernel or library source fails here whether
//! or not a model run happens to launch it. The search of every authored
//! preprocessor group is `seismic::coverage::cover`'s, which costs hours.

use seismic::coverage::{form_bases, InProcess, Progress, Toolchain};
use seismic::BackendName;

fn every_entry_forms(backend: BackendName) {
    let module = magnitude_kernels::module().unwrap();
    let toolchain = Toolchain::open(backend).unwrap_or_else(|error| panic!("{backend:?} forms on this host: {error}"));
    let coverage = form_bases(&module, &toolchain, &InProcess, &Progress::default()).unwrap();
    assert!(
        coverage.failures.is_empty() && coverage.unimplemented.is_empty(),
        "{coverage}"
    );
}

#[test]
fn every_vulkan_entry_forms() {
    every_entry_forms(BackendName::Vulkan);
}

#[cfg(target_os = "macos")]
#[test]
fn every_metal_entry_forms() {
    every_entry_forms(BackendName::Metal);
}

#[cfg(target_os = "linux")]
#[test]
fn every_cuda_entry_forms() {
    every_entry_forms(BackendName::Cuda);
}
