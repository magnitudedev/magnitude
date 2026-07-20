use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let pin_path = manifest_dir.join("../../native-pin.toml");
    println!("cargo:rerun-if-changed={}", pin_path.display());

    let pin = fs::read_to_string(&pin_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", pin_path.display()));
    let bindings_revision = table_value(&pin, "llama_cpp_rs", "revision")
        .unwrap_or_else(|| panic!("missing llama_cpp_rs.revision in {}", pin_path.display()));
    let native_backend_revision = table_value(&pin, "llama_cpp", "revision")
        .unwrap_or_else(|| panic!("missing llama_cpp.revision in {}", pin_path.display()));

    emit("ICN_BINDINGS_REVISION", &bindings_revision);
    emit("ICN_NATIVE_BACKEND_REVISION", &native_backend_revision);
    emit(
        "ICN_BUILD_TARGET",
        &env::var("TARGET").expect("TARGET must be set"),
    );
    emit(
        "ICN_BUILD_PROFILE",
        &env::var("PROFILE").expect("PROFILE must be set"),
    );
    emit("ICN_RUSTC_VERSION", &rustc_version());
}

fn table_value(source: &str, wanted_table: &str, wanted_key: &str) -> Option<String> {
    let mut table = None;
    for raw_line in source.lines() {
        let line = raw_line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            table = Some(name.trim());
            continue;
        }
        if table != Some(wanted_table) {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == wanted_key {
            return value
                .trim()
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .map(str::to_owned);
        }
    }
    None
}

fn rustc_version() -> String {
    let rustc = env::var_os("RUSTC").expect("RUSTC must be set");
    let output = Command::new(&rustc)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| {
            panic!("failed to execute {}: {error}", Path::new(&rustc).display())
        });
    assert!(output.status.success(), "rustc --version failed");
    String::from_utf8(output.stdout)
        .expect("rustc --version must be UTF-8")
        .trim()
        .to_owned()
}

fn emit(name: &str, value: &str) {
    println!("cargo:rustc-env={name}={value}");
}
