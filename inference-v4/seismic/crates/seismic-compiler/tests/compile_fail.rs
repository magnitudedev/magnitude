use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Case {
    name: &'static str,
    diagnostic: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "checked_module_literal",
        diagnostic: "field `inner` of struct `CheckedModule` is private",
    },
    Case {
        name: "planning_authority",
        diagnostic: "module `expression` is private",
    },
    Case {
        name: "kernel_handle_literal",
        diagnostic: "of struct `ScalarId` are private",
    },
    Case {
        name: "raw_assignment_freeze",
        diagnostic: "struct `RawAssignment` is private",
    },
    Case {
        name: "native_candidate_literal",
        diagnostic: "of struct `NativeKernelCandidate` are private",
    },
    Case {
        name: "unreflected_candidate_use",
        diagnostic: "found reference `&NativeKernelCandidate<B>`",
    },
    Case {
        name: "implementation_builder_literal",
        diagnostic: "field `inner` of struct `ImplementationBuilder` is private",
    },
];

#[test]
fn invalid_public_transitions_do_not_compile() {
    let compiler = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lang = compiler
        .parent()
        .expect("compiler crate has a crates directory")
        .join("seismic-lang");
    let root = std::env::temp_dir().join(format!(
        "seismic-compiler-compile-fail-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root).expect("remove stale compile-fail directory");
    }
    fs::create_dir_all(&root).expect("create compile-fail directory");

    for case in CASES {
        run_case(&root, &compiler, &lang, case);
    }

    fs::remove_dir_all(&root).expect("remove compile-fail directory");
}

fn run_case(root: &Path, compiler: &Path, lang: &Path, case: &Case) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("ui")
        .join(format!("{}.rs", case.name));
    let directory = root.join(case.name);
    fs::create_dir_all(directory.join("src")).expect("create fixture source directory");
    fs::copy(&fixture, directory.join("src/main.rs")).expect("copy compile-fail fixture");
    let manifest = format!(
        "[package]\nname = \"seismic-compile-fail-{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[workspace]\n\n[dependencies]\nseismic-compiler = {{ path = {:?} }}\nseismic-lang = {{ path = {:?} }}\n",
        case.name, compiler, lang
    );
    fs::write(directory.join("Cargo.toml"), manifest).expect("write fixture manifest");

    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["check", "--quiet", "--offline"])
        .current_dir(&directory)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .output()
        .expect("run fixture cargo check");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "compile-fail fixture `{}` unexpectedly compiled",
        case.name
    );
    assert!(
        stderr.contains(case.diagnostic),
        "compile-fail fixture `{}` failed for the wrong reason; expected {:?}\n{}",
        case.name,
        case.diagnostic,
        stderr
    );
}
