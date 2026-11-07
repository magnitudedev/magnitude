//! The identity of every source the coverage search's result depends on: the
//! kernel crate's dependency closure, which holds the standard library its
//! bundle is built from and Seismic (the search, rendering, the registry and
//! the backends). With the toolchain identity, it keys a completed coverage
//! run; engine and service sources do not enter it.

fn main() {
    println!(
        "cargo:rustc-env=KERNEL_FORMATION_SEARCH_SOURCES={}",
        magnitude_source_identity::of_closure("magnitude-kernels")
    );
}
