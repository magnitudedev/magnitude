use seismic_build::{BackendName, NativeCoverage};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Every entry is implemented on each of these backends, one asset per
/// backend at `kernels/<backend>/<entry>.<extension>`.
const BACKENDS: [BackendName; 3] = [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan];

/// `common/` files that exist on only some backends: (path without extension,
/// the backends that have it, why). Every other `common/` file exists on all
/// of `BACKENDS`. An entry whose actual presence differs, including one now on
/// every backend, fails the build so this list cannot go stale.
const COMMON_EXCEPTIONS: &[(&str, &[BackendName], &str)] = &[
    (
        "attention_prefill",
        &[BackendName::Cuda],
        "tensor-core flash prefill body; Metal keeps its prefill in `attention.h`, Vulkan in `prefill.glsl`",
    ),
    (
        "prefill",
        &[BackendName::Vulkan],
        "prefill attention body; Metal keeps it in `attention.h`, CUDA in `attention_prefill.cuh`",
    ),
    (
        "flash",
        &[BackendName::Vulkan],
        "cooperative-matrix tile pieces shared by the prefill and vision attention bodies",
    ),
    (
        "history",
        &[BackendName::Vulkan],
        "dense and affine K8/V4 history forms; the affine part of Metal `attention.h` and CUDA `attention.cuh`",
    ),
    (
        "activation",
        &[BackendName::Vulkan],
        "`ELEMENT_ACT` names A's ABI symbols, which only entries binding A admit; `element::Act` elsewhere",
    ),
    (
        "precise",
        &[BackendName::Vulkan],
        "bounded-error `exp`: Vulkan permits 3 + 2|x| ulp where Metal and CUDA do not",
    ),
    (
        "rotary",
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
    violations.extend(common_parity(&kernels));
    if !violations.is_empty() {
        panic!(
            "native backend parity failed ({} violations):\n  - {}",
            violations.len(),
            violations.join("\n  - ")
        );
    }
}

const fn extension(backend: BackendName) -> &'static str {
    match backend {
        BackendName::Cpu => "rs",
        BackendName::Metal => "metal",
        BackendName::Cuda => "cu",
        BackendName::Vulkan => "comp",
    }
}

const fn common_extension(backend: BackendName) -> &'static str {
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
                .join(format!("{entry}.{}", extension(backend)));
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

/// The `common/` trees of `BACKENDS` hold the same files, relative path
/// without extension, apart from `COMMON_EXCEPTIONS`.
fn common_parity(kernels: &Path) -> Vec<String> {
    let mut presence = BTreeMap::<String, BTreeSet<BackendName>>::new();
    let mut violations = Vec::new();
    for backend in BACKENDS {
        let common = kernels.join(backend.as_str()).join("common");
        // A file added to or removed from a tree reruns the check.
        println!("cargo:rerun-if-changed={}", common.display());
        let mut files = Vec::new();
        collect_files(&common, &mut files);
        for file in files {
            let path = relative(&common, &file);
            if path.extension().and_then(|value| value.to_str()) != Some(common_extension(backend)) {
                violations.push(format!(
                    "`{}` is not a `.{}` file",
                    relative(kernels, &file).display(),
                    common_extension(backend)
                ));
                continue;
            }
            let stem = path.with_extension("").to_string_lossy().into_owned();
            presence.entry(stem).or_default().insert(backend);
        }
    }
    let exceptions: BTreeMap<&str, (BTreeSet<BackendName>, &str)> = COMMON_EXCEPTIONS
        .iter()
        .map(|(stem, backends, reason)| (*stem, (backends.iter().copied().collect(), *reason)))
        .collect();
    for (stem, backends) in &presence {
        let complete = backends.len() == BACKENDS.len();
        match exceptions.get(stem.as_str()) {
            None if complete => {}
            None => violations.push(format!(
                "`common/{stem}` exists on {} but not on {}; add the counterparts or list it in COMMON_EXCEPTIONS with a reason",
                names(backends),
                names(&BACKENDS.iter().copied().filter(|backend| !backends.contains(backend)).collect())
            )),
            Some((declared, _)) if declared != backends => violations.push(format!(
                "COMMON_EXCEPTIONS lists `common/{stem}` on {}, but it exists on {}; update or remove the exception",
                names(declared),
                names(backends)
            )),
            Some(_) => {}
        }
    }
    for (stem, (declared, _)) in &exceptions {
        if !presence.contains_key(*stem) {
            violations.push(format!(
                "COMMON_EXCEPTIONS lists `common/{stem}` on {}, but no backend has it; remove the exception",
                names(declared)
            ));
        }
    }
    violations
}

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) {
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
