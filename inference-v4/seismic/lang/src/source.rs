//! Shared filesystem source collection and native-asset capture.
//!
//! A Metal, CUDA or Vulkan native asset may include files of its backend's
//! shared device library: `#include "common/<name>.h"` (Metal),
//! `#include "common/<name>.cuh"` (CUDA) or `#include "common/<name>.glsl"`
//! (Vulkan), resolved in the asset's own directory. Capture inlines each included file once, at its first
//! directive, so the captured asset (and with it the checked bundle and every
//! identity derived from it) contains exactly the text that is compiled.
//! Every other include, vendor and system headers included, is rejected.
use crate::checked::{check_source, CheckedModule, SourceError, SourceFile, SourceSet};
use crate::ids::EntryId;
use crate::registry::BackendName;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Source(SourceError),
    Invalid(String),
    NativeInclude(NativeIncludeError),
}
impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Source(e) => e.fmt(f),
            Self::Invalid(e) => e.fmt(f),
            Self::NativeInclude(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for LoadError {}
impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A rejected `#include` directive in a native asset or an included file.
#[derive(Debug)]
pub struct NativeIncludeError {
    /// The file containing the directive.
    pub path: PathBuf,
    /// One-based line of the directive.
    pub line: usize,
    pub directive: String,
    pub reason: NativeIncludeReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeIncludeReason {
    /// `#include <…>`: vendor and system headers are not part of the
    /// packaged runtime; the generated prefix provides the toolchain header.
    System,
    /// A quoted path other than `common/<name>.<backend extension>`.
    OutsideCommon,
    /// Not a `#include "…"` directive (`#import`, macro-expanded, …).
    Malformed,
    /// The named `common/` file does not exist or is not readable.
    Missing,
}

impl std::fmt::Display for NativeIncludeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self.reason {
            NativeIncludeReason::System => "vendor and system headers are not admitted",
            NativeIncludeReason::OutsideCommon => {
                "only `common/<name>.h` (Metal), `common/<name>.cuh` (CUDA) or `common/<name>.glsl` (Vulkan) may be included"
            }
            NativeIncludeReason::Malformed => "only `#include \"common/<name>\"` directives are admitted",
            NativeIncludeReason::Missing => "the included file does not exist in the backend's common directory",
        };
        write!(
            f,
            "native include {}:{}: `{}`: {reason}",
            self.path.display(),
            self.line,
            self.directive.trim()
        )
    }
}

/// One authored file of a captured native asset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeAssetFile {
    pub path: PathBuf,
    pub text: String,
}

/// The authored files behind one captured native asset: the asset and the
/// `common/` files it includes (transitively, first-inclusion order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedAsset {
    pub entry: EntryId,
    pub backend: BackendName,
    pub asset: NativeAssetFile,
    pub includes: Vec<NativeAssetFile>,
}

impl CapturedAsset {
    /// Every file the captured asset was built from.
    pub fn paths(&self) -> impl Iterator<Item = &Path> {
        std::iter::once(self.asset.path.as_path())
            .chain(self.includes.iter().map(|file| file.path.as_path()))
    }
}

/// A checked module loaded from the filesystem.
pub struct Loaded {
    pub module: CheckedModule,
    /// The `.seismic` files, canonical and sorted.
    pub sources: Vec<PathBuf>,
    pub assets: Vec<CapturedAsset>,
}

impl Loaded {
    /// Every file the module depends on: sources, assets and includes.
    pub fn dependencies(&self) -> impl Iterator<Item = &Path> {
        self.sources
            .iter()
            .map(PathBuf::as_path)
            .chain(self.assets.iter().flat_map(CapturedAsset::paths))
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

pub fn load(paths: &[PathBuf], mut prelude: SourceSet) -> Result<Loaded, LoadError> {
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
    let assets = capture_assets(&mut module, None)?;
    Ok(Loaded {
        module,
        sources: files,
        assets,
    })
}

/// `base` is required for native assets in an inline source snapshot.
pub fn capture_assets(
    module: &mut CheckedModule,
    base: Option<&Path>,
) -> Result<Vec<CapturedAsset>, LoadError> {
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
    let mut captured = Vec::new();
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
        let asset = NativeAssetFile {
            text: std::fs::read_to_string(&path)?,
            path,
        };
        // CPU assets are Rust and have no include set.
        let (source, includes) = match backend {
            BackendName::Cpu => (asset.text.clone(), Vec::new()),
            BackendName::Metal => expand_includes(&asset, "h")?,
            BackendName::Cuda => expand_includes(&asset, "cuh")?,
            BackendName::Vulkan => expand_includes(&asset, "glsl")?,
        };
        module
            .capture_native_asset(entry, backend, source)
            .map_err(LoadError::Invalid)?;
        captured.push(CapturedAsset {
            entry,
            backend,
            asset,
            includes,
        });
    }
    Ok(captured)
}

/// Inlines every `common/<name>.<extension>` include of a Metal, CUDA or
/// Vulkan asset, each file once at its first directive (later directives naming it
/// expand to an empty line).
/// `#line` markers keep toolchain diagnostics attributed to the authored
/// files; their labels are relative, so the expansion is host-independent.
fn expand_includes(
    asset: &NativeAssetFile,
    extension: &'static str,
) -> Result<(String, Vec<NativeAssetFile>), LoadError> {
    let directory = asset
        .path
        .parent()
        .expect("a canonical asset path has a parent directory");
    let label = asset
        .path
        .file_name()
        .expect("a canonical asset path names a file")
        .to_string_lossy()
        .into_owned();
    let mut expansion = Expansion {
        directory,
        extension,
        includes: Vec::new(),
    };
    let source = expansion.expand(&asset.path, &label, &asset.text)?;
    Ok((source, expansion.includes))
}

struct Expansion<'a> {
    directory: &'a Path,
    extension: &'static str,
    includes: Vec<NativeAssetFile>,
}

impl Expansion<'_> {
    fn expand(&mut self, path: &Path, label: &str, text: &str) -> Result<String, LoadError> {
        let mut out = String::with_capacity(text.len());
        for (index, line) in text.split_inclusive('\n').enumerate() {
            let Some(directive) = include_directive(line) else {
                out.push_str(line);
                continue;
            };
            let failure = |reason| {
                LoadError::NativeInclude(NativeIncludeError {
                    path: path.to_path_buf(),
                    line: index + 1,
                    directive: line.trim_end().to_owned(),
                    reason,
                })
            };
            let name = self.resolve(directive).map_err(failure)?;
            let included = self.directory.join("common").join(&name);
            let included = included
                .canonicalize()
                .map_err(|_| failure(NativeIncludeReason::Missing))?;
            if included.parent() != Some(self.directory.join("common").canonicalize()?.as_path()) {
                // A symlink leaving the common directory.
                return Err(failure(NativeIncludeReason::OutsideCommon));
            }
            if self.includes.iter().any(|file| file.path == included) {
                out.push('\n');
                continue;
            }
            let text = std::fs::read_to_string(&included)
                .map_err(|_| failure(NativeIncludeReason::Missing))?;
            self.includes.push(NativeAssetFile {
                path: included.clone(),
                text: text.clone(),
            });
            let included_label = format!("common/{name}");
            out.push_str(&format!("#line 1 \"{included_label}\"\n"));
            out.push_str(&self.expand(&included, &included_label, &text)?);
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&format!("#line {} \"{label}\"\n", index + 2));
        }
        Ok(out)
    }

    /// The file name inside `common/` a directive names.
    fn resolve(&self, directive: &str) -> Result<String, NativeIncludeReason> {
        let Some(operand) = directive.strip_prefix("include") else {
            return Err(NativeIncludeReason::Malformed);
        };
        if !operand.starts_with(char::is_whitespace) && !operand.starts_with(['"', '<']) {
            return Err(NativeIncludeReason::Malformed);
        }
        let operand = operand.trim();
        if operand.starts_with('<') {
            return Err(NativeIncludeReason::System);
        }
        let Some(quoted) = operand.strip_prefix('"') else {
            return Err(NativeIncludeReason::Malformed);
        };
        let Some((target, rest)) = quoted.split_once('"') else {
            return Err(NativeIncludeReason::Malformed);
        };
        let rest = rest.trim();
        if !(rest.is_empty() || rest.starts_with("//")) {
            return Err(NativeIncludeReason::Malformed);
        }
        let Some(name) = target.strip_prefix("common/") else {
            return Err(NativeIncludeReason::OutsideCommon);
        };
        let Some(stem) = name.strip_suffix(self.extension).and_then(|stem| stem.strip_suffix('.'))
        else {
            return Err(NativeIncludeReason::OutsideCommon);
        };
        if stem.is_empty()
            || !stem
                .bytes()
                .all(|byte| byte == b'_' || byte == b'-' || byte.is_ascii_alphanumeric())
        {
            return Err(NativeIncludeReason::OutsideCommon);
        }
        Ok(name.to_owned())
    }
}

/// The text after `#` of an include-like preprocessor directive.
fn include_directive(line: &str) -> Option<&str> {
    let directive = line.trim_start().strip_prefix('#')?.trim_start();
    ["include", "import"]
        .iter()
        .any(|keyword| directive.starts_with(keyword))
        .then_some(directive)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "seismic-include-{name}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(root.join("metal/common")).expect("fixture directory");
        root.canonicalize().expect("canonical fixture")
    }

    fn asset(root: &Path, text: &str) -> NativeAssetFile {
        let path = root.join("metal/kernel.metal");
        std::fs::write(&path, text).expect("asset");
        NativeAssetFile {
            path,
            text: text.to_owned(),
        }
    }

    #[test]
    fn inlines_common_files_once_in_first_inclusion_order() {
        let root = fixture("order");
        std::fs::write(root.join("metal/common/a.h"), "#include \"common/b.h\"\nA\n").unwrap();
        std::fs::write(root.join("metal/common/b.h"), "B\n").unwrap();
        let asset = asset(
            &root,
            "#include \"common/a.h\"\n#include \"common/b.h\" // again\nBODY\n",
        );
        let (source, includes) = expand_includes(&asset, "h").unwrap();
        assert_eq!(
            source,
            "#line 1 \"common/a.h\"\n#line 1 \"common/b.h\"\nB\n#line 2 \"common/a.h\"\nA\n#line 2 \"kernel.metal\"\n\nBODY\n"
        );
        let names = includes
            .iter()
            .map(|file| file.path.file_name().unwrap().to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, ["a.h", "b.h"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn vulkan_assets_inline_common_glsl_files() {
        let root = fixture("vulkan");
        std::fs::create_dir_all(root.join("vulkan/common")).unwrap();
        std::fs::write(root.join("vulkan/common/reduce.glsl"), "float reduce_sum(float x) { return x; }\n").unwrap();
        std::fs::write(root.join("vulkan/common/other.h"), "X\n").unwrap();
        let path = root.join("vulkan/kernel.comp");
        let text = "#include \"common/reduce.glsl\"\nvoid kernel() {}\n";
        std::fs::write(&path, text).unwrap();
        let asset = NativeAssetFile { path, text: text.to_owned() };
        let (source, includes) = expand_includes(&asset, "glsl").unwrap();
        assert_eq!(
            source,
            "#line 1 \"common/reduce.glsl\"\nfloat reduce_sum(float x) { return x; }\n#line 2 \"kernel.comp\"\nvoid kernel() {}\n"
        );
        assert_eq!(includes.len(), 1);
        let wrong = NativeAssetFile {
            path: asset.path.clone(),
            text: "#include \"common/other.h\"\n".to_owned(),
        };
        match expand_includes(&wrong, "glsl") {
            Err(LoadError::NativeInclude(error)) => {
                assert_eq!(error.reason, NativeIncludeReason::OutsideCommon)
            }
            other => panic!("unexpected {other:?}"),
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_everything_but_common_backend_files() {
        let root = fixture("reject");
        std::fs::write(root.join("metal/common/a.h"), "A\n").unwrap();
        std::fs::write(root.join("metal/outside.h"), "X\n").unwrap();
        for (text, reason) in [
            ("#include <metal_stdlib>\n", NativeIncludeReason::System),
            ("  #  include \"outside.h\"\n", NativeIncludeReason::OutsideCommon),
            ("#include \"common/../outside.h\"\n", NativeIncludeReason::OutsideCommon),
            ("#include \"common/a.cuh\"\n", NativeIncludeReason::OutsideCommon),
            ("#include \"common/missing.h\"\n", NativeIncludeReason::Missing),
            ("#import \"common/a.h\"\n", NativeIncludeReason::Malformed),
            ("#include COMMON_HEADER\n", NativeIncludeReason::Malformed),
        ] {
            let asset = asset(&root, &format!("// header\n{text}"));
            match expand_includes(&asset, "h") {
                Err(LoadError::NativeInclude(error)) => {
                    assert_eq!(error.reason, reason, "{text}");
                    assert_eq!(error.line, 2);
                    assert_eq!(error.path, asset.path);
                }
                other => panic!("{text}: unexpected {other:?}"),
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
