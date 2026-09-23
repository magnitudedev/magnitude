//! No source file of the in-scope trees is excluded from version control.
//!
//! `**/target/` in `inference-v4/.gitignore` ignores Cargo build output, and
//! source directories named `target` are re-included explicitly. A new source
//! directory with that name, or any other rule that swallows source, would be
//! lost silently at commit time; this test names every such file instead.
//! A directory is Cargo build output exactly when Cargo wrote `CACHEDIR.TAG`
//! into it.
use crate::common::corpus_path;
use std::path::Path;
use std::process::Command;

/// Scanned trees, relative to the workspace root `inference-v4/`.
const TREES: &[&str] = &["seismic", "seismic-std", "engine"];

const SOURCE_EXTENSIONS: &[&str] = &["rs", "seismic", "metal", "toml"];

#[test]
fn no_source_file_is_ignored() {
    let workspace = corpus_path("../..");
    let output = Command::new("git")
        .current_dir(&workspace)
        .args(["ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--"])
        .args(TREES)
        .output()
        .expect("git runs in the workspace checkout");
    assert!(output.status.success(), "git ls-files failed: {}", String::from_utf8_lossy(&output.stderr));
    let lost: Vec<String> = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .filter(|path| {
            Path::new(path)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| SOURCE_EXTENSIONS.contains(&extension))
        })
        .filter(|path| !inside_build_output(&workspace, Path::new(path)))
        .collect();
    assert!(
        lost.is_empty(),
        "source files are git-ignored and would never be committed \
         (add a `!` exception to inference-v4/.gitignore): {lost:#?}"
    );
}

/// Whether some ancestor of `path` is a Cargo build directory.
fn inside_build_output(workspace: &Path, path: &Path) -> bool {
    path.ancestors()
        .skip(1)
        .any(|ancestor| workspace.join(ancestor).join("CACHEDIR.TAG").is_file())
}
