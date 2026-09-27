use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn workflow_draft_cannot_bypass_admission() {
    let runtime = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = std::env::temp_dir().join(format!(
        "seismic-runtime-compile-fail-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root).expect("remove stale compile-fail directory");
    }
    fs::create_dir_all(&root).expect("create compile-fail directory");

    run_fixture(&root, &runtime);

    fs::remove_dir_all(&root).expect("remove compile-fail directory");
}

fn run_fixture(root: &Path, runtime: &Path) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("ui")
        .join("workflow_draft_submit.rs");
    let directory = root.join("workflow-draft-submit");
    fs::create_dir_all(directory.join("src")).expect("create fixture source directory");
    fs::copy(&fixture, directory.join("src/main.rs")).expect("copy compile-fail fixture");
    let manifest = format!(
        "[package]\nname = \"seismic-runtime-workflow-draft-submit\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[workspace]\n\n[dependencies]\nseismic-runtime = {{ path = {:?} }}\n",
        runtime
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
        "workflow draft unexpectedly bypassed admission"
    );
    assert!(
        stderr.contains("no method named `submit` found for struct `WorkflowDraftAny`"),
        "workflow draft failed for the wrong reason\n{stderr}"
    );
}
