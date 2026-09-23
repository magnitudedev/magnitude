//! Derive the toolchain fingerprint from the actual locked Cranelift
//! version, so it can never silently drift from the dependency set.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Re-run when the lockfile changes.
    println!("cargo:rerun-if-changed=../../../Cargo.lock");
    let version = locked_cranelift_version()
        .unwrap_or_else(|message| panic!("seismic-cpu toolchain fingerprint: {message}"));
    println!("cargo:rustc-env=SEISMIC_CRANELIFT_VERSION={version}");
}

/// The locked `cranelift-codegen` version, read from the workspace lockfile.
fn locked_cranelift_version() -> Result<String, String> {
    let manifest = env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| "the manifest directory is unknown".to_string())?;
    let lock = Path::new(&manifest)
        .ancestors()
        .nth(3)
        .ok_or_else(|| "the workspace root is absent".to_string())?
        .join("Cargo.lock");
    let text = fs::read_to_string(&lock)
        .map_err(|error| format!("cannot read {}: {error}", lock.display()))?;
    let mut in_package = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("name = \"cranelift-codegen\"") {
            in_package = true;
        } else if in_package && line.starts_with("version = ") {
            return Ok(line
                .trim_start_matches("version = ")
                .trim_matches('"')
                .to_string());
        } else if line.starts_with("[[package]]") {
            in_package = false;
        }
    }
    Err("cranelift-codegen is absent from the lockfile".to_string())
}
