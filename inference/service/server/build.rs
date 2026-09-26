use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    emit(
        "SERVICE_BUILD_TARGET",
        &env::var("TARGET").expect("TARGET must be set"),
    );
    emit(
        "SERVICE_BUILD_PROFILE",
        &env::var("PROFILE").expect("PROFILE must be set"),
    );
    emit("SERVICE_RUSTC_VERSION", &rustc_version());
    // The installation layout keeps runtime-loaded libraries (NVRTC) in `runtime/` beside `bin/`.
    // Apple executables own no libraries and carry no rpath (design/release/platform-contracts.md).
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../runtime");
    }
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
