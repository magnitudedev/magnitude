//! Derives the engine build identity from the workspace sources that implement
//! its worker protocol, execution, and native kernel bundle.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn is_build_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some(
            "rs" | "toml"
                | "seismic"
                | "metal"
                | "cu"
                | "cuh"
                | "glsl"
                | "comp"
                | "h"
                | "hpp"
                | "c"
                | "cc"
                | "cpp"
        )
    )
}

fn collect(directory: &Path, files: &mut Vec<PathBuf>) {
    let mut entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()))
        .map(|entry| entry.expect("engine source directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                collect(&path, files);
            }
        } else if is_build_source(&path) {
            files.push(path);
        }
    }
}

fn main() {
    let engine_root =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let root = engine_root
        .parent()
        .expect("engine is in the inference workspace");
    let mut files = Vec::new();
    for directory in ["engine", "seismic", "seismic-std", "solver"] {
        let path = root.join(directory);
        println!("cargo:rerun-if-changed={}", path.display());
        collect(&path, &mut files);
    }
    for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        let path = root.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        files.push(path);
    }
    let mut digest = Sha256::new();
    for file in files {
        let relative = file
            .strip_prefix(root)
            .expect("source under the inference root");
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        digest.update(std::fs::read(&file).expect("engine source file"));
        digest.update([0]);
    }
    let hash = digest
        .finalize()
        .iter()
        .take(12)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!(
        "cargo:rustc-env=MAGNITUDE_ENGINE_BUILD={}+src.{hash}",
        std::env::var("CARGO_PKG_VERSION").expect("package version")
    );
}
