//! Generates one public-API test per scenario, backend and selection
//! (`$OUT_DIR/generated_public.rs`, included by `tests/public.rs`).
//!
//! This enumerates files and reads only the `# backends:` header line; every
//! test parses its whole scenario with `seismic_corpus::scenario::parse`, the
//! owner of the header grammar, so a malformed header fails its tests.
use std::fmt::Write;
use std::path::Path;

/// Top directories whose scenarios also get a `__ranked` test (design A10 §2.2.3).
const RANKED_DIRECTORIES: &[&str] = &["history", "composition", "areas"];

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest).join("scenarios");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut scenarios = Vec::new();
    collect(&root, &root, &mut scenarios);
    scenarios.sort();

    let mut generated = String::new();
    for (name, text) in &scenarios {
        let identifier = test_identifier(name);
        let top = name.split('/').next().expect("split yields one component");
        let mut selections = vec!["Baseline"];
        if RANKED_DIRECTORIES.contains(&top) {
            selections.push("Ranked");
        }
        for (backend, variant, cfg) in declared_backends(name, text) {
            for selection in &selections {
                writeln!(
                    generated,
                    "{cfg}#[test]\n#[allow(non_snake_case)]\nfn {identifier}__{backend}__{}() {{\n    \
                     common::run_public({name:?}, seismic::BackendName::{variant}, common::Selection::{selection});\n}}",
                    selection.to_lowercase(),
                )
                .expect("writing to a String cannot fail");
            }
        }
    }
    let out = Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("generated_public.rs");
    std::fs::write(&out, generated).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
}

fn collect(root: &Path, directory: &Path, scenarios: &mut Vec<(String, String)>) {
    let entries =
        std::fs::read_dir(directory).unwrap_or_else(|e| panic!("{}: {e}", directory.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("{}: {e}", directory.display()))
            .path();
        if path.is_dir() {
            collect(root, &path, scenarios);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "seismic")
        {
            let name = path
                .with_extension("")
                .strip_prefix(root)
                .expect("walked path lies under the root")
                .components()
                .map(|c| c.as_os_str().to_str().expect("scenario paths are UTF-8"))
                .collect::<Vec<_>>()
                .join("/");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            scenarios.push((name, text));
        }
    }
}

/// `history/h46-source-arithmetic-lanes` -> `history__h46_source_arithmetic_lanes`.
fn test_identifier(name: &str) -> String {
    let identifier = name.replace('/', "__").replace('-', "_");
    assert!(
        identifier
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
            && identifier.starts_with(|c: char| c.is_ascii_alphabetic()),
        "scenario `{name}`: paths use letters, digits, `-`, `_` and `/` and start with a letter"
    );
    identifier
}

/// `(test suffix, BackendName variant, cfg attribute)` for each backend of
/// the scenario's `# backends:` line (both when absent).
fn declared_backends(name: &str, text: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    let cpu = ("cpu", "Cpu", "");
    let metal = ("metal", "Metal", "#[cfg(target_os = \"macos\")]\n");
    let declared = text
        .lines()
        .take_while(|line| line.trim().is_empty() || line.starts_with('#'))
        .find_map(|line| line.strip_prefix("# backends: "));
    let Some(declared) = declared else {
        return vec![cpu, metal];
    };
    declared
        .split_ascii_whitespace()
        .map(|word| match word {
            "cpu" => cpu,
            "metal" => metal,
            _ => panic!("scenario `{name}`: unknown backend `{word}`"),
        })
        .collect()
}
