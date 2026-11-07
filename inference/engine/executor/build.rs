//! The executor build identity (`magnitude_executor::EXECUTOR_BUILD`): the
//! sources and locked versions of the executor's own dependency closure and the
//! compiler, which are all that can change what it computes and stores.

fn main() {
    println!(
        "cargo:rustc-env=MAGNITUDE_EXECUTOR_BUILD={}",
        magnitude_source_identity::of_closure(env!("CARGO_PKG_NAME"))
    );
}
