//! Shared filesystem source collection and native-asset capture.
//!
//! A Metal, CUDA or Vulkan native asset may include library files written for
//! its backend (`.h` Metal, `.cuh` CUDA, `.glsl` Vulkan): `#include "<path>"`,
//! resolved relative to the file containing the directive, whose canonical
//! target must lie inside one of the load's source roots. Capture inlines each
//! included file once, at its first directive, so the captured asset (and with
//! it the checked bundle and every identity derived from it) contains exactly
//! the text that is compiled. Vendor and system headers, absolute paths and
//! files outside the source roots are rejected.
//!
//! `#include <seismic/<name>>` names a file of Seismic's native library for the
//! asset's backend ([`NATIVE_LIBRARY`]): dense element types, packed weight
//! decoders and weight slot bindings, owned by Seismic because Seismic owns
//! representation semantics. Library files are embedded in this crate, inlined
//! once like any include, and may include only other library files.
use crate::checked::{check_source, CheckedModule, SourceError, SourceFile, SourceSet};
use crate::ids::EntryId;
use crate::registry::BackendName;
use std::path::{Path, PathBuf};

/// Seismic's native library: `(name, text)`, included as `<seismic/<name>>`. A
/// name's extension selects its backend (`.h` Metal, `.cuh` CUDA, `.glsl`
/// Vulkan).
pub const NATIVE_LIBRARY: &[(&str, &str)] = &[
    ("element.h", include_str!("../native-library/metal/element.h")),
    ("packets.h", include_str!("../native-library/metal/packets.h")),
    ("element.cuh", include_str!("../native-library/cuda/element.cuh")),
    ("packets.cuh", include_str!("../native-library/cuda/packets.cuh")),
    ("element.glsl", include_str!("../native-library/vulkan/element.glsl")),
    ("packets.glsl", include_str!("../native-library/vulkan/packets.glsl")),
];

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
    /// `#include <…>` other than `<seismic/…>`: vendor and system headers are
    /// not part of the packaged runtime; the generated prefix provides the
    /// toolchain header.
    System,
    /// A relative include inside a Seismic library file, which may include
    /// only other library files.
    LibraryRelative,
    /// Not a `#include "…"` directive (`#import`, macro-expanded, …).
    Malformed,
    /// An absolute path: includes are relative to the including file.
    Absolute,
    /// A path without the backend's library extension.
    Extension,
    /// The resolved file lies outside every source root of the load.
    OutsideSourceRoots,
    /// The named file does not exist or is not readable.
    Missing,
}

impl std::fmt::Display for NativeIncludeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self.reason {
            NativeIncludeReason::System => {
                "vendor and system headers are not admitted; `<seismic/…>` names Seismic's native library"
            }
            NativeIncludeReason::LibraryRelative => {
                "a Seismic library file may include only `<seismic/…>` files"
            }
            NativeIncludeReason::Malformed => "only `#include \"<relative path>\"` directives are admitted",
            NativeIncludeReason::Absolute => "include paths are relative to the including file",
            NativeIncludeReason::Extension => {
                "only `.h` (Metal), `.cuh` (CUDA) or `.glsl` (Vulkan) files of the asset's backend may be included"
            }
            NativeIncludeReason::OutsideSourceRoots => {
                "the included file lies outside the build's source roots"
            }
            NativeIncludeReason::Missing => "the included file does not exist (in the source roots or Seismic's native library)",
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

/// The files behind one captured native asset: the asset, the authored files
/// it includes, and the Seismic library files it includes (each transitively,
/// in first-inclusion order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedAsset {
    pub entry: EntryId,
    pub backend: BackendName,
    pub asset: NativeAssetFile,
    pub includes: Vec<NativeAssetFile>,
    /// Names in [`NATIVE_LIBRARY`]; embedded, so not filesystem dependencies.
    pub library: Vec<&'static str>,
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
    let roots = source_roots(paths)?;
    let assets = capture_assets(&mut module, None, &roots)?;
    Ok(Loaded {
        module,
        sources: files,
        assets,
    })
}

/// The directories native includes may resolve into: each source directory,
/// and the containing directory of each source file, canonical.
pub fn source_roots(paths: &[PathBuf]) -> Result<Vec<PathBuf>, LoadError> {
    let mut roots = Vec::new();
    for path in paths {
        let path = path.canonicalize()?;
        let root = if path.is_dir() {
            path
        } else {
            path.parent()
                .expect("a canonical file path has a parent directory")
                .to_path_buf()
        };
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    Ok(roots)
}

/// `base` is required for native assets in an inline source snapshot.
/// `roots` bound where native includes may resolve (see [`source_roots`]).
pub fn capture_assets(
    module: &mut CheckedModule,
    base: Option<&Path>,
    roots: &[PathBuf],
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
        let expansion = match backend {
            BackendName::Cpu => Expanded {
                source: asset.text.clone(),
                includes: Vec::new(),
                library: Vec::new(),
            },
            BackendName::Metal => expand_includes(&asset, "h", roots)?,
            BackendName::Cuda => expand_includes(&asset, "cuh", roots)?,
            BackendName::Vulkan => expand_includes(&asset, "glsl", roots)?,
        };
        module
            .capture_native_asset(entry, backend, expansion.source)
            .map_err(LoadError::Invalid)?;
        captured.push(CapturedAsset {
            entry,
            backend,
            asset,
            includes: expansion.includes,
            library: expansion.library,
        });
    }
    Ok(captured)
}

/// An asset's source with every include inlined, and the files inlined.
#[derive(Debug)]
struct Expanded {
    source: String,
    includes: Vec<NativeAssetFile>,
    library: Vec<&'static str>,
}

/// Inlines every include of a Metal, CUDA or Vulkan asset, each file once at
/// its first directive (later directives naming it expand to an empty line).
/// `#line` markers keep toolchain diagnostics attributed to the authored
/// files; their labels are paths relative to the asset's directory (library
/// files: `seismic/<name>`), so the expansion is host-independent.
fn expand_includes(
    asset: &NativeAssetFile,
    extension: &'static str,
    roots: &[PathBuf],
) -> Result<Expanded, LoadError> {
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
        asset_directory: directory,
        extension,
        roots,
        includes: Vec::new(),
        library: Vec::new(),
    };
    let source = expansion.expand(&asset.path, &label, &asset.text, false)?;
    Ok(Expanded {
        source,
        includes: expansion.includes,
        library: expansion.library,
    })
}

struct Expansion<'a> {
    asset_directory: &'a Path,
    extension: &'static str,
    roots: &'a [PathBuf],
    includes: Vec<NativeAssetFile>,
    library: Vec<&'static str>,
}

/// What an include directive names.
enum Include {
    /// A path relative to the including file.
    Relative(String),
    /// A file of [`NATIVE_LIBRARY`]: its name and text.
    Library(&'static str, &'static str),
}

/// `to` relative to the directory `from`, with `/` separators. Both are
/// canonical.
fn relative_label(from: &Path, to: &Path) -> String {
    let from: Vec<_> = from.components().collect();
    let to: Vec<_> = to.components().collect();
    let shared = from
        .iter()
        .zip(&to)
        .take_while(|(left, right)| left == right)
        .count();
    std::iter::repeat_n("..".to_owned(), from.len() - shared)
        .chain(
            to[shared..]
                .iter()
                .map(|component| component.as_os_str().to_string_lossy().into_owned()),
        )
        .collect::<Vec<_>>()
        .join("/")
}

impl Expansion<'_> {
    /// `in_library`: `text` is a Seismic library file, whose `path` is only a
    /// diagnostic label.
    fn expand(
        &mut self,
        path: &Path,
        label: &str,
        text: &str,
        in_library: bool,
    ) -> Result<String, LoadError> {
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
            let target = match self.resolve(directive).map_err(failure)? {
                Include::Library(name, library_text) => {
                    if !self.library.contains(&name) {
                        self.library.push(name);
                        let library_label = format!("seismic/{name}");
                        out.push_str(&format!("#line 1 \"{library_label}\"\n"));
                        out.push_str(&self.expand(
                            &PathBuf::from(format!("<seismic>/{name}")),
                            &library_label,
                            library_text,
                            true,
                        )?);
                        if !out.ends_with('\n') {
                            out.push('\n');
                        }
                        out.push_str(&format!("#line {} \"{label}\"\n", index + 2));
                    } else {
                        out.push('\n');
                    }
                    continue;
                }
                Include::Relative(_) if in_library => {
                    return Err(failure(NativeIncludeReason::LibraryRelative))
                }
                Include::Relative(target) => target,
            };
            let directory = path
                .parent()
                .expect("a canonical native file path has a parent directory");
            let included = directory
                .join(&target)
                .canonicalize()
                .map_err(|_| failure(NativeIncludeReason::Missing))?;
            // Canonical: `..` and symlinks are followed before the boundary check.
            if !self.roots.iter().any(|root| included.starts_with(root)) {
                return Err(failure(NativeIncludeReason::OutsideSourceRoots));
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
            let included_label = relative_label(self.asset_directory, &included);
            out.push_str(&format!("#line 1 \"{included_label}\"\n"));
            out.push_str(&self.expand(&included, &included_label, &text, false)?);
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&format!("#line {} \"{label}\"\n", index + 2));
        }
        Ok(out)
    }

    /// What a directive names: a relative path, or a Seismic library file.
    fn resolve(&self, directive: &str) -> Result<Include, NativeIncludeReason> {
        let Some(operand) = directive.strip_prefix("include") else {
            return Err(NativeIncludeReason::Malformed);
        };
        if !operand.starts_with(char::is_whitespace) && !operand.starts_with(['"', '<']) {
            return Err(NativeIncludeReason::Malformed);
        }
        let operand = operand.trim();
        let (target, rest, angled) = if let Some(angled) = operand.strip_prefix('<') {
            let Some((target, rest)) = angled.split_once('>') else {
                return Err(NativeIncludeReason::Malformed);
            };
            (target, rest, true)
        } else if let Some(quoted) = operand.strip_prefix('"') {
            let Some((target, rest)) = quoted.split_once('"') else {
                return Err(NativeIncludeReason::Malformed);
            };
            (target, rest, false)
        } else {
            return Err(NativeIncludeReason::Malformed);
        };
        let rest = rest.trim();
        if !(rest.is_empty() || rest.starts_with("//")) {
            return Err(NativeIncludeReason::Malformed);
        }
        if target.is_empty() {
            return Err(NativeIncludeReason::Malformed);
        }
        let extension_matches = |name: &str| {
            Path::new(name).extension().and_then(|extension| extension.to_str()) == Some(self.extension)
        };
        if angled {
            let Some(name) = target.strip_prefix("seismic/") else {
                return Err(NativeIncludeReason::System);
            };
            if !extension_matches(name) {
                return Err(NativeIncludeReason::Extension);
            }
            return NATIVE_LIBRARY
                .iter()
                .find(|(library_name, _)| *library_name == name)
                .map(|(library_name, text)| Include::Library(library_name, text))
                .ok_or(NativeIncludeReason::Missing);
        }
        if Path::new(target).is_absolute() {
            return Err(NativeIncludeReason::Absolute);
        }
        if !extension_matches(target) {
            return Err(NativeIncludeReason::Extension);
        }
        Ok(Include::Relative(target.to_owned()))
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
    fn inlines_included_files_once_in_first_inclusion_order() {
        let root = fixture("order");
        std::fs::write(root.join("metal/common/a.h"), "#include \"b.h\"\nA\n").unwrap();
        std::fs::write(root.join("metal/common/b.h"), "B\n").unwrap();
        let asset = asset(
            &root,
            "#include \"common/a.h\"\n#include \"common/b.h\" // again\nBODY\n",
        );
        let Expanded { source, includes, .. } = expand_includes(&asset, "h", &[root.clone()]).unwrap();
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
    fn nested_includes_resolve_relative_to_their_own_file() {
        let root = fixture("relative");
        std::fs::create_dir_all(root.join("metal/attention")).unwrap();
        std::fs::write(
            root.join("metal/attention/softmax.h"),
            "#include \"../common/element.h\"\nSOFTMAX\n",
        )
        .unwrap();
        std::fs::write(root.join("metal/common/element.h"), "ELEMENT\n").unwrap();
        let asset = asset(&root, "#include \"attention/softmax.h\"\nBODY\n");
        let Expanded { source, includes, .. } = expand_includes(&asset, "h", &[root.clone()]).unwrap();
        assert_eq!(
            source,
            "#line 1 \"attention/softmax.h\"\n#line 1 \"common/element.h\"\nELEMENT\n#line 2 \"attention/softmax.h\"\nSOFTMAX\n#line 2 \"kernel.metal\"\nBODY\n"
        );
        assert_eq!(includes.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn vulkan_assets_inline_glsl_files() {
        let root = fixture("vulkan");
        std::fs::create_dir_all(root.join("vulkan/common")).unwrap();
        std::fs::write(root.join("vulkan/common/reduce.glsl"), "float reduce_sum(float x) { return x; }\n").unwrap();
        std::fs::write(root.join("vulkan/common/other.h"), "X\n").unwrap();
        let path = root.join("vulkan/kernel.comp");
        let text = "#include \"common/reduce.glsl\"\nvoid kernel() {}\n";
        std::fs::write(&path, text).unwrap();
        let asset = NativeAssetFile { path, text: text.to_owned() };
        let Expanded { source, includes, .. } = expand_includes(&asset, "glsl", &[root.clone()]).unwrap();
        assert_eq!(
            source,
            "#line 1 \"common/reduce.glsl\"\nfloat reduce_sum(float x) { return x; }\n#line 2 \"kernel.comp\"\nvoid kernel() {}\n"
        );
        assert_eq!(includes.len(), 1);
        let wrong = NativeAssetFile {
            path: asset.path.clone(),
            text: "#include \"common/other.h\"\n".to_owned(),
        };
        match expand_includes(&wrong, "glsl", &[root.clone()]) {
            Err(LoadError::NativeInclude(error)) => {
                assert_eq!(error.reason, NativeIncludeReason::Extension)
            }
            other => panic!("unexpected {other:?}"),
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn admits_any_file_of_the_backend_inside_the_source_roots() {
        let root = fixture("admit");
        std::fs::write(root.join("metal/outside_common.h"), "X\n").unwrap();
        let asset = asset(&root, "  #  include \"outside_common.h\"\n");
        let Expanded { source, .. } = expand_includes(&asset, "h", &[root.clone()]).unwrap();
        assert!(source.starts_with("#line 1 \"outside_common.h\"\nX\n"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inlines_seismic_library_files_once_without_filesystem_dependencies() {
        let root = fixture("library");
        std::fs::write(root.join("metal/common/lib.h"), "#include <seismic/element.h>\nLIB\n").unwrap();
        // packets.h includes element.h itself; each library file is inlined once.
        let asset = asset(
            &root,
            "#include <seismic/packets.h>\n#include \"common/lib.h\"\n#include <seismic/element.h>\nBODY\n",
        );
        let Expanded { source, includes, library } =
            expand_includes(&asset, "h", &[root.clone()]).unwrap();
        assert_eq!(library, ["packets.h", "element.h"]);
        assert_eq!(includes.len(), 1, "library files are not filesystem includes");
        assert!(source.starts_with("#line 1 \"seismic/packets.h\"\n"));
        assert!(source.contains("#line 1 \"seismic/element.h\"\n"));
        assert!(!source.lines().any(|line| line.trim_start().starts_with("#include")));
        assert_eq!(source.matches("struct Rows16 {").count(), 1);
        assert_eq!(source.matches("#define ELEMENT_OF(prefix)").count(), 1);
        assert!(source.ends_with("\nBODY\n"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn every_library_file_expands_for_its_backend() {
        for (name, _) in NATIVE_LIBRARY {
            let extension = Path::new(name).extension().unwrap().to_str().unwrap();
            let asset = NativeAssetFile {
                path: std::env::temp_dir().join("kernel"),
                text: format!("#include <seismic/{name}>\n"),
            };
            let Expanded { library, .. } = expand_includes(&asset, extension, &[])
                .unwrap_or_else(|error| panic!("{name}: {error:?}"));
            assert_eq!(library.first(), Some(name), "{name}");
        }
    }

    #[test]
    fn rejects_system_absolute_foreign_missing_and_escaping_includes() {
        let root = fixture("reject");
        std::fs::write(root.join("metal/common/a.h"), "A\n").unwrap();
        // A file beside the fixture root, outside it; reachable by `..` and by a symlink.
        let outside = root.with_extension("outside.h");
        std::fs::write(&outside, "X\n").unwrap();
        let outside_name = outside.file_name().unwrap().to_str().unwrap().to_owned();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("metal/common/link.h")).unwrap();
        let escape = format!("#include \"../../{outside_name}\"\n");
        let mut cases = vec![
            ("#include <metal_stdlib>\n".to_owned(), NativeIncludeReason::System),
            (escape, NativeIncludeReason::OutsideSourceRoots),
            ("#include \"/tmp/a.h\"\n".to_owned(), NativeIncludeReason::Absolute),
            ("#include \"common/a.cuh\"\n".to_owned(), NativeIncludeReason::Extension),
            ("#include \"common/missing.h\"\n".to_owned(), NativeIncludeReason::Missing),
            ("#import \"common/a.h\"\n".to_owned(), NativeIncludeReason::Malformed),
            ("#include COMMON_HEADER\n".to_owned(), NativeIncludeReason::Malformed),
            ("#include \"\"\n".to_owned(), NativeIncludeReason::Malformed),
            ("#include <seismic/missing.h>\n".to_owned(), NativeIncludeReason::Missing),
            ("#include <seismic/element.cuh>\n".to_owned(), NativeIncludeReason::Extension),
            ("#include <seismic/element.h\n".to_owned(), NativeIncludeReason::Malformed),
        ];
        #[cfg(unix)]
        cases.push((
            "#include \"common/link.h\"\n".to_owned(),
            NativeIncludeReason::OutsideSourceRoots,
        ));
        for (text, reason) in cases {
            let asset = asset(&root, &format!("// header\n{text}"));
            match expand_includes(&asset, "h", &[root.clone()]) {
                Err(LoadError::NativeInclude(error)) => {
                    assert_eq!(error.reason, reason, "{text}");
                    assert_eq!(error.line, 2);
                    assert_eq!(error.path, asset.path);
                }
                other => panic!("{text}: unexpected {other:?}"),
            }
        }
        std::fs::remove_file(outside).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_roots_are_directories_and_the_parents_of_files() {
        let root = fixture("roots");
        let file = root.join("metal/ops.seismic");
        std::fs::write(&file, "").unwrap();
        let roots = source_roots(&[root.clone(), file, root.join("metal")]).unwrap();
        assert_eq!(roots, [root.clone(), root.join("metal")]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
