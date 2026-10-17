//! The digest of every source the coverage search's result depends on: the
//! kernel crate and the standard library its bundle is built from, Seismic
//! (the search, rendering, the registry and the backends) and the locked
//! dependency versions. With the toolchain identity, it keys a completed
//! coverage run; engine and service sources do not enter it.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const SOURCES: &[&str] = &[
    "../../engine/kernels",
    "../../seismic-std",
    "../../seismic",
    "../../Cargo.lock",
];

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let mut files = Vec::new();
    for source in SOURCES {
        let path = manifest.join(source);
        println!("cargo:rerun-if-changed={}", path.display());
        collect(&path, &mut files);
    }
    files.sort();
    let mut digest = Sha256::new();
    for file in &files {
        let relative = file
            .strip_prefix(&manifest)
            .unwrap_or(file)
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let contents = std::fs::read(file).expect("a source file reads");
        for part in [relative.as_bytes(), &contents] {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
    }
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    println!("cargo:rustc-env=KERNEL_FORMATION_SEARCH_SOURCES={hex}");
}

/// Every file under `path`, skipping build output and hidden directories.
fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        files.push(path.to_owned());
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir()
            && (name == "target"
                || name.starts_with('.')
                || name == "__pycache__"
                || name == "node_modules")
        {
            continue;
        }
        collect(&path, files);
    }
}
