//! Program assembly: collect sources, parse, check the closed linked program.
use super::check::{self, resolve::Located};
use super::sir::Program;
use super::syntax;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SourceFile {
    pub path: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct FileDiagnostic {
    pub path: String,
    pub rendered: String,
}

impl FileDiagnostic {
    pub fn render(&self) -> String {
        self.rendered.clone()
    }
}

/// Recursively collect files whose extension is exactly `.seismic`, sorted and deduplicated.
pub fn collect_files(paths: &[PathBuf]) -> Result<Vec<SourceFile>, String> {
    fn walk(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        if path.is_dir() {
            let mut entries = Vec::new();
            for entry in std::fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))? {
                entries.push(
                    entry
                        .map_err(|e| format!("{}: {e}", path.display()))?
                        .path(),
                );
            }
            entries.sort();
            for entry in entries {
                walk(&entry, out)?;
            }
        } else if path
            .extension()
            .is_some_and(|extension| extension == "seismic")
        {
            out.push(path.to_path_buf());
        } else if !path.exists() {
            return Err(format!("{}: no such file or directory", path.display()));
        }
        Ok(())
    }
    let mut found = Vec::new();
    for path in paths {
        walk(path, &mut found)?;
    }
    found.sort();
    found.dedup();
    let mut files = Vec::new();
    for path in found {
        let display = path.display().to_string();
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{display}: {e}"))?;
        files.push(SourceFile {
            path: display,
            text,
        });
    }
    Ok(files)
}

/// Parse and check every file as one closed program.
pub fn compile(files: &[SourceFile]) -> Result<Program, Vec<FileDiagnostic>> {
    let mut diagnostics: Vec<Located> = Vec::new();
    let mut parsed = Vec::new();
    for (index, file) in files.iter().enumerate() {
        match syntax::parse(&file.text) {
            Ok(ast) => parsed.push((index, ast)),
            Err(diagnostic) => diagnostics.push(Located {
                file: index,
                diagnostic,
            }),
        }
    }
    let (definitions, families) = check::check_program(&parsed, &mut diagnostics);
    if diagnostics.is_empty() {
        return Ok(Program {
            definitions,
            families,
            files: files
                .iter()
                .map(|f| (f.path.clone(), f.text.clone()))
                .collect(),
        });
    }
    diagnostics.sort_by_key(|d| (d.file, d.diagnostic.span.start));
    diagnostics.dedup_by(|a, b| a.file == b.file && a.diagnostic == b.diagnostic);
    Err(diagnostics
        .into_iter()
        .map(|d| {
            let file = &files[d.file];
            FileDiagnostic {
                path: file.path.clone(),
                rendered: d.diagnostic.render(&file.path, &file.text),
            }
        })
        .collect())
}
