//! What the kernel tests share.

use seismic_lang::checked::{CheckedModule, SourceSet};

pub mod bound;
pub mod recurrent_bound;

/// `sources`, held in memory, checked with each native implementation's
/// source read from the crate's `kernels/`, where the declarations name
/// them.
pub fn check(sources: SourceSet) -> CheckedModule {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let kernels = std::path::Path::new(&manifest).join("kernels");
    seismic_lang::source::check_beside(sources, &kernels).expect("the kernel sources check")
}
