use seismic_build::{BackendName, NativeCoverage};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Every entry is implemented on each of these backends, one asset per
/// backend at `kernels/<backend>/<entry>.<extension>`.
const BACKENDS: [BackendName; 4] = [BackendName::Cpu, BackendName::Metal, BackendName::Cuda, BackendName::Vulkan];

/// Files of the per-backend trees that exist on only some backends: (path
/// relative to `kernels/<backend>/`, without extension; the backends that have
/// it; why). Every other file — entry asset or library file, at any depth —
/// exists on all of `BACKENDS`. An exception whose actual presence differs,
/// including one now on every backend, fails the build so this list cannot go
/// stale.
const TREE_EXCEPTIONS: &[(&str, &[BackendName], &str)] = &[
    (
        "lib/attention/prefill",
        &[BackendName::Cuda, BackendName::Vulkan],
        "prefill attention body (CUDA tensor-core flash, Vulkan tiled); Metal keeps its prefill in `attention.h`",
    ),
    (
        "lib/attention/flash",
        &[BackendName::Vulkan],
        "cooperative-matrix tile pieces shared by the prefill and vision attention bodies",
    ),
    (
        "lib/core/precise",
        &[BackendName::Vulkan],
        "bounded-error `exp`: Vulkan permits 3 + 2|x| ulp where Metal and CUDA do not",
    ),
    (
        "lib/core/rotary",
        &[BackendName::Vulkan],
        "rotary sin/cos shared by attention and vision; inside `attention.h` / `attention.cuh` elsewhere",
    ),
];

fn main() {
    let artifacts = seismic_build::Build::new("kernels")
        .source("kernels")
        // Keep newly introduced stage roots explicit so Cargo notices their
        // first addition; the recursive directory source deduplicates them.
        .source("kernels/draft.seismic")
        .source("kernels/readout.seismic")
        .source("kernels/vision.seismic")
        .source("kernels/recurrent.seismic")
        .run()
        .unwrap_or_else(|error| {
            panic!("checking engine Seismic sources and generating bindings failed: {error}")
        });

    let kernels = Path::new("kernels")
        .canonicalize()
        .expect("the kernels directory exists");
    let mut violations = entry_parity(&kernels, &artifacts.natives);
    violations.extend(tree_parity(&kernels));
    if !violations.is_empty() {
        panic!(
            "native backend parity failed ({} violations):\n  - {}",
            violations.len(),
            violations.join("\n  - ")
        );
    }
}

const fn entry_extension(backend: BackendName) -> &'static str {
    match backend {
        BackendName::Cpu => "rs",
        BackendName::Metal => "metal",
        BackendName::Cuda => "cu",
        BackendName::Vulkan => "comp",
    }
}

const fn library_extension(backend: BackendName) -> &'static str {
    match backend {
        BackendName::Cpu => "rs",
        BackendName::Metal => "h",
        BackendName::Cuda => "cuh",
        BackendName::Vulkan => "glsl",
    }
}

/// Every entry with a native implementation has one on each of `BACKENDS`,
/// at the path named after the entry.
fn entry_parity(kernels: &Path, natives: &[NativeCoverage]) -> Vec<String> {
    let mut by_entry = BTreeMap::<&str, BTreeMap<BackendName, &Path>>::new();
    for native in natives {
        by_entry
            .entry(&native.entry)
            .or_default()
            .insert(native.backend, &native.asset);
    }
    let mut violations = Vec::new();
    for (entry, assets) in by_entry {
        for backend in BACKENDS {
            let expected = kernels
                .join(backend.as_str())
                .join(format!("{entry}.{}", entry_extension(backend)));
            match assets.get(&backend) {
                None => violations.push(format!(
                    "entry `{entry}` has no {} implementation (expected `native {entry} for {} from \"{}\"`)",
                    backend.as_str(),
                    backend.as_str(),
                    relative(kernels, &expected).display()
                )),
                Some(asset) if *asset != expected => violations.push(format!(
                    "entry `{entry}`: {} asset is `{}`, expected `{}`",
                    backend.as_str(),
                    relative(kernels, asset).display(),
                    relative(kernels, &expected).display()
                )),
                Some(_) => {}
            }
        }
    }
    violations
}

/// The per-backend trees `kernels/<backend>/` of `BACKENDS` hold the same
/// files — entry assets and library files, at any depth — by relative path
/// without extension, apart from `TREE_EXCEPTIONS`.
fn tree_parity(kernels: &Path) -> Vec<String> {
    let mut presence = BTreeMap::<String, BTreeSet<BackendName>>::new();
    let mut violations = Vec::new();
    for backend in BACKENDS {
        let tree = kernels.join(backend.as_str());
        let mut files = Vec::new();
        collect_files(&tree, &mut files);
        for file in files {
            let path = relative(&tree, &file);
            let extension = path.extension().and_then(|value| value.to_str());
            if extension != Some(entry_extension(backend)) && extension != Some(library_extension(backend)) {
                violations.push(format!(
                    "`{}` is neither a `.{}` entry asset nor a `.{}` library file",
                    relative(kernels, &file).display(),
                    entry_extension(backend),
                    library_extension(backend)
                ));
                continue;
            }
            let stem = path.with_extension("").to_string_lossy().replace('\\', "/");
            presence.entry(stem).or_default().insert(backend);
        }
    }
    let exceptions: BTreeMap<&str, (BTreeSet<BackendName>, &str)> = TREE_EXCEPTIONS
        .iter()
        .map(|(stem, backends, reason)| (*stem, (backends.iter().copied().collect(), *reason)))
        .collect();
    for (stem, backends) in &presence {
        let complete = backends.len() == BACKENDS.len();
        match exceptions.get(stem.as_str()) {
            None if complete => {}
            None => violations.push(format!(
                "`{stem}` exists on {} but not on {}; add the counterparts or list it in TREE_EXCEPTIONS with a reason",
                names(backends),
                names(&BACKENDS.iter().copied().filter(|backend| !backends.contains(backend)).collect())
            )),
            Some((declared, _)) if declared != backends => violations.push(format!(
                "TREE_EXCEPTIONS lists `{stem}` on {}, but it exists on {}; update or remove the exception",
                names(declared),
                names(backends)
            )),
            Some(_) => {}
        }
    }
    for (stem, (declared, _)) in &exceptions {
        if !presence.contains_key(*stem) {
            violations.push(format!(
                "TREE_EXCEPTIONS lists `{stem}` on {}, but no backend has it; remove the exception",
                names(declared)
            ));
        }
    }
    violations
}

/// Every file under `directory`, recursively. Each visited directory is a
/// rerun trigger, so adding or removing a file reruns the check.
fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", directory.display());
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()))
            .path();
        if path.is_dir() {
            collect_files(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn relative<'a>(base: &Path, path: &'a Path) -> &'a Path {
    path.strip_prefix(base).unwrap_or(path)
}

fn names(backends: &BTreeSet<BackendName>) -> String {
    backends
        .iter()
        .map(|backend| backend.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}
