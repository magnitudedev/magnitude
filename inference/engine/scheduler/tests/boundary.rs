use std::path::{Path, PathBuf};

/// This crate's directory, as cargo names it to the running test.
fn manifest_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
}

/// The packages this crate depends on, dev and build included, by package
/// name, as Cargo resolves its manifest: whatever key or rename declares
/// them.
fn dependencies() -> Vec<String> {
    let output = std::process::Command::new(std::env::var_os("CARGO").expect("cargo sets CARGO"))
        .args(["metadata", "--format-version", "1", "--no-deps", "--offline", "--manifest-path"])
        .arg(manifest_dir().join("Cargo.toml"))
        .output()
        .expect("cargo metadata runs");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).expect("cargo metadata is JSON");
    let package = metadata["packages"]
        .as_array()
        .expect("packages")
        .iter()
        .find(|package| package["name"] == env!("CARGO_PKG_NAME"))
        .expect("this package");
    package["dependencies"]
        .as_array()
        .expect("dependencies")
        .iter()
        .map(|dependency| dependency["name"].as_str().expect("a dependency's package name").to_owned())
        .collect()
}

#[test]
fn direct_dependencies_preserve_the_service_boundary() {
    let dependencies = dependencies();
    assert!(dependencies.iter().any(|name| name == "magnitude-executor"), "{dependencies:?}");
    for forbidden in [
        "seismic",
        "magnitude-artifacts",
        "magnitude-chat",
        "magnitude-templates",
        "magnitude-family-qwen35",
    ] {
        assert!(
            !dependencies.iter().any(|name| name == forbidden),
            "service must not depend on {forbidden}"
        );
    }
}

/// Every Rust source under `directory`, recursively.
fn sources(directory: &Path) -> Vec<(std::path::PathBuf, String)> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(directory).expect("a source directory") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            found.extend(sources(&path));
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = std::fs::read_to_string(&path).expect("a readable source");
            found.push((path, source));
        }
    }
    found
}

#[test]
fn service_sources_do_not_import_forbidden_implementation_layers() {
    let sources = sources(&manifest_dir().join("src"));
    assert!(sources.iter().any(|(path, _)| path.ends_with("owner.rs")), "the walk reads the crate's sources");
    for (path, source) in &sources {
        for forbidden in [
            "seismic",
            "magnitude_artifacts",
            "magnitude_family_qwen35",
            "crate::chat",
            "crate::templates",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} imported {forbidden}",
                path.display()
            );
        }
    }
}
