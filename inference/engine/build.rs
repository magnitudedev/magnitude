//! The engine Rust build identity (`MAGNITUDE_ENGINE_RUST_BUILD`): the sources
//! and locked versions of the engine's dependency closure and the compiler,
//! excluding GPU shader directories whose content is folded into
//! [`worker::protocol::EngineBuild`] at runtime via
//! `magnitude_kernels_bundle::identity()`.

const GPU_SHADER_DIRS: &[&str] = &[
    "engine/kernels/kernels/metal",
    "engine/kernels/kernels/cuda",
    "engine/kernels/kernels/vulkan",
];

fn main() {
    println!(
        "cargo:rustc-env=MAGNITUDE_ENGINE_RUST_BUILD={}",
        magnitude_source_identity::of_closure_excluding_paths(env!("CARGO_PKG_NAME"), GPU_SHADER_DIRS)
    );
}
