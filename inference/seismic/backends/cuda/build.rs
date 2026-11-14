//! The formation build identity (`toolchain::FORMATION_BUILD`): the sources
//! and locked versions of this backend's dependency closure and the compiler,
//! which hold all its formation code. It keys every artifact the backend's
//! toolchain stores, so a stored artifact is never served to a build whose
//! formation code differs from the one that compiled it.

fn main() {
    println!(
        "cargo:rustc-env=SEISMIC_FORMATION_BUILD={}",
        magnitude_source_identity::of_closure(env!("CARGO_PKG_NAME"))
    );
}
