//! Libraries and whole-program checks.
//!
//! A library is a set of files. A program is a list of libraries compiled
//! together against a backend list: one global namespace, every construct with
//! a full row of lowerings, every lowering bound to its construct, and every
//! body checked in its file's scope.

use crate::ast;
use crate::check::{self, Checked, Env, Signature};
use crate::hir::{Function, Lowering};
use crate::span::Diagnostic;
use crate::{parse, scope_of_path, Scope};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SourceFile {
    pub path: PathBuf,
    pub text: String,
    pub scope: Scope,
}

#[derive(Debug)]
pub struct FileDiagnostic {
    pub path: PathBuf,
    pub text: String,
    pub diagnostic: Diagnostic,
}

impl FileDiagnostic {
    pub fn render(&self) -> String {
        self.diagnostic.render(&self.path.display().to_string(), &self.text)
    }
}

pub struct Program {
    pub functions: Vec<Function>,
    pub lowerings: Vec<Lowering>,
    pub signatures: HashMap<String, Signature>,
}

/// Collect every `.seismic.*` file under the given paths (files or directories).
pub fn collect_files(paths: &[PathBuf]) -> Result<Vec<SourceFile>, String> {
    let mut found = Vec::new();
    fn walk(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        if path.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))?.flatten().map(|e| e.path()).collect();
            entries.sort();
            for e in entries {
                walk(&e, out)?;
            }
        } else if path.file_name().map(|n| n.to_string_lossy().contains(".seismic.")).unwrap_or(false) {
            out.push(path.to_path_buf());
        } else if !path.exists() {
            return Err(format!("{}: no such file or directory", path.display()));
        }
        Ok(())
    }
    for p in paths {
        walk(p, &mut found)?;
    }
    found.sort();
    found.dedup();
    let mut files = Vec::new();
    for path in found {
        let display = path.display().to_string();
        let Some(scope) = scope_of_path(&display) else {
            return Err(format!("{display}: file name must be `<name>.seismic.portable` or `<name>.seismic.<backend>`"));
        };
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{display}: {e}"))?;
        files.push(SourceFile { path, text, scope });
    }
    Ok(files)
}

/// Compile a set of files together. Returns the program, or every diagnostic found.
pub fn compile(files: &[SourceFile], backends: &[String]) -> Result<Program, Vec<FileDiagnostic>> {
    let mut errors: Vec<FileDiagnostic> = Vec::new();
    let mut parsed: Vec<(usize, ast::File)> = Vec::new();
    for (i, f) in files.iter().enumerate() {
        match parse(&f.text) {
            Ok(file) => parsed.push((i, file)),
            Err(d) => errors.push(FileDiagnostic { path: f.path.clone(), text: f.text.clone(), diagnostic: d }),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Scope rules on declarations, and the global namespace.
    let mut signatures: HashMap<String, Signature> = HashMap::new();
    let mut declared_in: HashMap<String, usize> = HashMap::new();
    for (i, file) in &parsed {
        let f = &files[*i];
        for decl in &file.decls {
            match decl {
                ast::Decl::Fn(d) | ast::Decl::Construct(d) => {
                    let is_construct = matches!(decl, ast::Decl::Construct(_));
                    if is_construct && f.scope != Scope::Portable {
                        errors.push(FileDiagnostic { path: f.path.clone(), text: f.text.clone(), diagnostic: Diagnostic::new(d.name.span, "`construct` is only allowed in portable files") });
                        continue;
                    }
                    if f.scope != Scope::Portable {
                        // Backend-scoped helper: registered under a backend-qualified key so it never
                        // collides with, or is callable from, the portable namespace.
                        continue;
                    }
                    match check::signature_of(d, is_construct) {
                        Ok(sig) => {
                            if let Some(prev) = declared_in.get(&sig.name) {
                                errors.push(FileDiagnostic {
                                    path: f.path.clone(),
                                    text: f.text.clone(),
                                    diagnostic: Diagnostic::new(d.name.span, format!("`{}` is already declared in {}", sig.name, files[*prev].path.display())),
                                });
                                continue;
                            }
                            declared_in.insert(sig.name.clone(), *i);
                            signatures.insert(sig.name.clone(), sig);
                        }
                        Err(diag) => errors.push(FileDiagnostic { path: f.path.clone(), text: f.text.clone(), diagnostic: diag }),
                    }
                }
                ast::Decl::Lower(_) => {}
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Check every body in its scope.
    let mut functions = Vec::new();
    let mut lowerings = Vec::new();
    for (i, file) in &parsed {
        let f = &files[*i];
        let env = Env { signatures: &signatures, scope: f.scope.clone() };
        let Checked { functions: fs, lowerings: ls, diagnostics } = check::check_file(file, &env);
        for d in diagnostics {
            errors.push(FileDiagnostic { path: f.path.clone(), text: f.text.clone(), diagnostic: d });
        }
        functions.extend(fs);
        lowerings.extend(ls);
    }

    // Rows and coverage: every construct, every backend.
    let mut by_cell: BTreeMap<(String, String), Vec<&Lowering>> = BTreeMap::new();
    for l in &lowerings {
        by_cell.entry((l.construct.clone(), l.backend.clone())).or_default().push(l);
    }
    for sig in signatures.values() {
        if !sig.is_construct {
            continue;
        }
        let file = &files[declared_in[&sig.name]];
        for backend in backends {
            let cell = by_cell.get(&(sig.name.clone(), backend.clone()));
            match cell {
                None => errors.push(FileDiagnostic {
                    path: file.path.clone(),
                    text: file.text.clone(),
                    diagnostic: Diagnostic::new(sig.span, format!("construct `{}` has no lowering for backend `{backend}`; add a `{}.seismic.{backend}` file with `lower {}:` bodies or `lower {}: portable`", sig.name, sig.name, sig.name, sig.name)),
                }),
                Some(blocks) => {
                    let covered = blocks.iter().any(|b| b.residual.is_empty() && b.elem_bindings.is_empty());
                    if !covered {
                        let domains: Vec<String> = blocks.iter().map(|b| b.residual.iter().map(|r| format!("{r} >= 0")).collect::<Vec<_>>().join(" and ")).collect();
                        errors.push(FileDiagnostic {
                            path: file.path.clone(),
                            text: file.text.clone(),
                            diagnostic: Diagnostic::new(
                                sig.span,
                                format!("construct `{}` on `{backend}`: no lowering covers the whole domain (blocks apply where {}); add `lower {}: portable` or an unconditional body", sig.name, domains.join(" | "), sig.name),
                            ),
                        });
                    }
                }
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(Program { functions, lowerings, signatures })
}
