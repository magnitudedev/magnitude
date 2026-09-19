//! Read-only observations of modules already mapped by the native loader.

use std::fs::{self, File};
use std::io::{Read, Take};
use std::path::Path;

use anyhow::{Context, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MAX_MODULE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Backing-file identity for an already loaded module in the immutable installation.
/// This does not claim a hash of relocated in-memory instructions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct LoadedBackendModule {
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
}

#[cfg(unix)]
fn already_loaded(path: &Path) -> Result<libloading::Library, libloading::Error> {
    // SAFETY: NOLOAD cannot initialize an absent library. The returned handle temporarily
    // retains an existing mapping, and Drop releases only this additional reference.
    unsafe {
        libloading::os::unix::Library::open(Some(path), libc::RTLD_NOLOAD | libc::RTLD_LAZY)
            .map(Into::into)
    }
}

#[cfg(windows)]
fn already_loaded(path: &Path) -> Result<libloading::Library, libloading::Error> {
    libloading::os::windows::Library::open_already_loaded(path).map(Into::into)
}

pub(crate) fn observe(directory: &Path) -> anyhow::Result<Vec<LoadedBackendModule>> {
    let directory = directory.canonicalize()?;
    let mut modules = Vec::new();
    for (index, entry) in fs::read_dir(&directory)?.enumerate() {
        ensure!(index < 64, "backend directory exceeds observation limit");
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "backend module is not a regular file"
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("backend module filename is not UTF-8"))?;
        ensure!(
            name.len() <= 256,
            "backend module filename exceeds observation limit"
        );
        let Ok(_mapping) = already_loaded(&path) else {
            continue;
        };
        ensure!(
            metadata.len() > 0 && metadata.len() <= MAX_MODULE_BYTES,
            "backend module exceeds observation byte limit"
        );
        let file = File::open(&path)?;
        let before = file.metadata()?;
        let mut reader: Take<&File> = (&file).take(MAX_MODULE_BYTES + 1);
        let mut digest = Sha256::new();
        let bytes = std::io::copy(&mut reader, &mut digest)
            .context("failed to hash loaded module backing file")?;
        let after = file.metadata()?;
        let current = fs::symlink_metadata(&path)?;
        ensure!(
            bytes == before.len()
                && bytes == metadata.len()
                && bytes == current.len()
                && bytes <= MAX_MODULE_BYTES
                && before.modified()? == after.modified()?
                && metadata.modified()? == current.modified()?
                && !current.file_type().is_symlink(),
            "backend module changed during observation"
        );
        modules.push(LoadedBackendModule {
            name,
            sha256: format!("{:x}", digest.finalize()),
            bytes,
        });
    }
    modules.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(modules)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn observes_only_the_loaded_full_path_without_loading_other_files() {
        let root = tempfile::tempdir().unwrap();
        let selected = root.path().join("selected");
        let other = root.path().join("other");
        fs::create_dir(&selected).unwrap();
        fs::create_dir(&other).unwrap();
        let source = root.path().join("fixture.c");
        fs::write(
            &source,
            "int execution_module_fixture(void) { return 42; }\n",
        )
        .unwrap();
        let name = if cfg!(target_os = "macos") {
            "libfixture.dylib"
        } else {
            "libfixture.so"
        };
        let module = selected.join(name);
        let result = Command::new("cc")
            .args(if cfg!(target_os = "macos") {
                vec!["-dynamiclib"]
            } else {
                vec!["-shared", "-fPIC"]
            })
            .arg(&source)
            .arg("-o")
            .arg(&module)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        fs::copy(&module, other.join(name)).unwrap();
        assert!(observe(&selected).unwrap().is_empty());
        // SAFETY: the test owns and compiled this trivial library, which has no constructors.
        let loaded = unsafe { libloading::Library::new(&module) }.unwrap();
        assert!(
            observe(&other).unwrap().is_empty(),
            "a matching basename is not loaded identity"
        );
        let records = observe(&selected).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, name);
        let content = fs::read(&module).unwrap();
        assert_eq!(records[0].bytes, content.len() as u64);
        assert_eq!(records[0].sha256, format!("{:x}", Sha256::digest(content)));
        // The observation released only its own reference; the actual owner still works.
        let symbol =
            unsafe { loaded.get::<unsafe extern "C" fn() -> i32>(b"execution_module_fixture") }
                .unwrap();
        assert_eq!(unsafe { symbol() }, 42);
    }
}
