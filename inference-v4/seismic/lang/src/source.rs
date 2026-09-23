//! Shared filesystem source collection and native-asset capture.
use crate::checked::{check_source, CheckedModule, SourceError, SourceFile, SourceSet};
use crate::registry::BackendName;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Source(SourceError),
    Invalid(String),
}
impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Source(e) => e.fmt(f),
            Self::Invalid(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for LoadError {}
impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub fn collect(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), LoadError> {
    let meta = std::fs::metadata(path)?;
    if meta.is_file() {
        if path.extension().is_none_or(|e| e != "seismic") {
            return Err(LoadError::Invalid(format!(
                "{} is not a .seismic source",
                path.display()
            )));
        }
        files.push(path.canonicalize()?);
    } else if meta.is_dir() {
        for child in std::fs::read_dir(path)? {
            let child = child?;
            let kind = child.file_type()?;
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() || child.path().extension().is_some_and(|e| e == "seismic") {
                collect(&child.path(), files)?;
            }
        }
    } else {
        return Err(LoadError::Invalid(format!(
            "{} is not a file or directory",
            path.display()
        )));
    }
    Ok(())
}

pub fn load(
    paths: &[PathBuf],
    mut prelude: SourceSet,
) -> Result<(CheckedModule, Vec<PathBuf>), LoadError> {
    if paths.is_empty() && prelude.files().is_empty() {
        return Err(LoadError::Invalid("no source paths provided".into()));
    }
    let mut files = Vec::new();
    for path in paths {
        collect(path, &mut files)?;
    }
    files.sort();
    files.dedup();
    if files.is_empty() && prelude.files().is_empty() {
        return Err(LoadError::Invalid("no .seismic files found".into()));
    }
    for path in &files {
        prelude.push(SourceFile {
            path: path.to_string_lossy().replace('\\', "/"),
            text: std::fs::read_to_string(path)?,
        });
    }
    let mut module = check_source(prelude).map_err(LoadError::Source)?;
    files.extend(capture_assets(&mut module, None)?);
    Ok((module, files))
}

/// `base` is required for native assets in an inline source snapshot.
pub fn capture_assets(
    module: &mut CheckedModule,
    base: Option<&Path>,
) -> Result<Vec<PathBuf>, LoadError> {
    let definitions: Vec<_> = module
        .entries()
        .iter()
        .flat_map(|entry| {
            BackendName::ALL.into_iter().filter_map(|backend| {
                module
                    .native_implementation(entry.id, backend)
                    .map(|native| {
                        (
                            native.entry,
                            native.backend,
                            native.declared_in.clone(),
                            native.source_path.clone(),
                        )
                    })
            })
        })
        .collect();
    let mut paths = Vec::new();
    for (entry, backend, declared_in, source_path) in definitions {
        let declaring = Path::new(&declared_in);
        let root = match base {
            Some(base) => base,
            None if declaring.is_absolute() => declaring.parent().expect("absolute source parent"),
            None => {
                return Err(LoadError::Invalid(
                    "inline native assets require base_dir".into(),
                ))
            }
        };
        let path = root.join(source_path).canonicalize()?;
        module
            .capture_native_asset(entry, backend, std::fs::read_to_string(&path)?)
            .map_err(LoadError::Invalid)?;
        paths.push(path);
    }
    Ok(paths)
}
