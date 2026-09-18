use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command.status().expect("start native template build tool");
    assert!(
        status.success(),
        "native template build failed: {command:?}"
    );
}

fn main() {
    let source = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../../inference-v3/native/templates")
        .canonicalize()
        .expect("owned native template sources");
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-env-changed=CMAKE_TOOLCHAIN_FILE");
    let build = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("native");
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    let toolchain = env::var_os("CMAKE_TOOLCHAIN_FILE");
    assert!(
        target == host || toolchain.is_some(),
        "cross-compiling templates requires CMAKE_TOOLCHAIN_FILE for the Rust target"
    );
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&source)
        .arg("-B")
        .arg(&build)
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DTEMPLATES_STATIC=ON",
        ]);
    if let Some(toolchain) = toolchain {
        configure.arg(format!(
            "-DCMAKE_TOOLCHAIN_FILE={}",
            PathBuf::from(toolchain).display()
        ));
    }
    run(&mut configure);
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build)
        .args(["--config", "Release", "--target", "templates", "--parallel"])
        .arg(env::var("NUM_JOBS").unwrap_or_else(|_| "1".into())));
    println!("cargo:rustc-link-search=native={}", build.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("Release").display()
    );
    println!("cargo:rustc-link-lib=static=templates");
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if !target.contains("msvc") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
