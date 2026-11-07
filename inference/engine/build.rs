//! The engine build identity (`worker::protocol::EngineBuild`): the sources and
//! locked versions of the engine's dependency closure, which holds both ends of
//! the worker protocol and everything a worker computes, and the compiler.

fn main() {
    println!(
        "cargo:rustc-env=MAGNITUDE_ENGINE_BUILD={}",
        magnitude_source_identity::of_closure(env!("CARGO_PKG_NAME"))
    );
}
