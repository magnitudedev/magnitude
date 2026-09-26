use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Case {
    name: &'static str,
    diagnostic: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "erased_view_mutation",
        diagnostic: "field `index` of struct `AnyBufferView` is private",
    },
    Case {
        name: "scalar_slot_mutation",
        diagnostic: "field `index` of struct `AnyScalarSlot` is private",
    },
    Case {
        name: "ir_owner_token",
        diagnostic: "struct `OwnerToken` is private",
    },
    Case {
        name: "checked_module_literal",
        diagnostic: "cannot construct `CheckedModule` with struct literal syntax due to private fields",
    },
    Case {
        name: "planning_authority",
        diagnostic: "module `expression` is private",
    },
    Case {
        name: "kernel_handle_literal",
        diagnostic: "cannot construct `PortableValue` with struct literal syntax due to private fields",
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
        diagnostic: "found reference `&NativeKernelCandidate",
    },
    Case {
        name: "implementation_builder_literal",
        diagnostic: "struct `ImplementationBuilder` is private",
    },
    Case {
        name: "executable_kernel_enumeration",
        diagnostic: "no method named `kernels`",
    },
    Case {
        name: "executable_schedule_access",
        diagnostic: "no method named `schedule`",
    },
    Case {
        name: "execution_environment_literal",
        diagnostic:
            "cannot construct `ExecutionEnvironment<'_, _, _, _>` with struct literal syntax due to private fields",
    },
    Case {
        name: "selection_function_literal",
        diagnostic: "cannot construct `SelectionFunction` with struct literal syntax due to private fields",
    },
    Case {
        name: "candidate_index_literal",
        diagnostic: "private fields",
    },
    Case {
        name: "selection_policy_literal",
        diagnostic: "cannot construct `SelectionPolicy` with struct literal syntax due to private fields",
    },
];

#[test]
fn invalid_public_transitions_do_not_compile() {
    let compiler = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lang = compiler
        .parent()
        .expect("compiler crate has a subsystem directory")
        .join("lang");
    let native_target = compiler
        .parent()
        .expect("compiler crate has a subsystem directory")
        .join("native-target");
    let root = std::env::temp_dir().join(format!(
        "seismic-compiler-compile-fail-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root).expect("remove stale compile-fail directory");
    }
    fs::create_dir_all(&root).expect("create compile-fail directory");

    let mismatches: Vec<String> = CASES
        .iter()
        .filter_map(|case| run_case(&root, &compiler, &lang, &native_target, case))
        .collect();

    fs::remove_dir_all(&root).expect("remove compile-fail directory");
    assert!(
        mismatches.is_empty(),
        "{} of {} compile-fail fixtures mismatched:\n\n{}",
        mismatches.len(),
        CASES.len(),
        mismatches.join("\n\n")
    );
}

/// Checks one fixture against the shared target directory and describes the mismatch, if any.
fn run_case(
    root: &Path,
    compiler: &Path,
    lang: &Path,
    native_target: &Path,
    case: &Case,
) -> Option<String> {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("ui")
        .join(format!("{}.rs", case.name));
    let directory = root.join(case.name);
    fs::create_dir_all(directory.join("src")).expect("create fixture source directory");
    fs::copy(&fixture, directory.join("src/main.rs")).expect("copy compile-fail fixture");
    let manifest = format!(
        "[package]\nname = \"seismic-compile-fail-{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[workspace]\n\n[dependencies]\nseismic-compiler = {{ path = {:?} }}\nseismic-lang = {{ path = {:?} }}\nseismic-ir = {{ path = {:?} }}\nseismic-native-target = {{ path = {:?} }}\n",
        case.name,
        compiler,
        lang,
        compiler.parent().unwrap().join("ir"),
        native_target,
    );
    fs::write(directory.join("Cargo.toml"), manifest).expect("write fixture manifest");

    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["check", "--quiet", "--offline"])
        .current_dir(&directory)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .output()
        .expect("run fixture cargo check");
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        Some(format!(
            "compile-fail fixture `{}` unexpectedly compiled",
            case.name
        ))
    } else if !stderr.contains(case.diagnostic) {
        Some(format!(
            "compile-fail fixture `{}` failed for the wrong reason; expected {:?}\n{}",
            case.name, case.diagnostic, stderr
        ))
    } else {
        None
    }
}
