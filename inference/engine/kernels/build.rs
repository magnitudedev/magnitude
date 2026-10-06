use seismic_build::{BackendName, NativeCoverage};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Every entry is implemented on each of these backends, one asset per
/// backend at `kernels/<backend>/<entry>.<extension>`, apart from
/// `ENTRY_EXCEPTIONS`.
const BACKENDS: [BackendName; 4] = [
    BackendName::Cpu,
    BackendName::Metal,
    BackendName::Cuda,
    BackendName::Vulkan,
];

/// Entries implemented on only some backends: (entry; the backends that
/// implement it; why). The executor selects such an entry only where the
/// checked bundle declares it. An exception whose actual coverage differs,
/// including one now on every backend, fails the build so this list cannot go
/// stale; its assets are exempt from `tree_parity` alike.
const ENTRY_EXCEPTIONS: &[(&str, &[BackendName], &str)] = &[
    (
        "routed_route_shared",
        &[BackendName::Metal],
        "decode routing that also expands the shared expert in the router launch; the other backends run `routed_route`, then `routed_expand`, until the form is measured there",
    ),
    (
        "gated_delta_project_convolved",
        &[BackendName::Metal],
        "the convolved recurrent decode form's projection, which also convolves and publishes the windows; the other backends run `gated_delta_project`, then `gated_delta_step`, until the form is measured there",
    ),
    (
        "gated_delta_step_convolved",
        &[BackendName::Metal],
        "the convolved recurrent decode form's step over the projection's convolved channels; the other backends convolve in `gated_delta_step`",
    ),    (
        "readout_top_rows",
        &[BackendName::Metal, BackendName::Cuda],
        "level one of the progressive head's certified selection (decode rows)",
    ),
    (
        "readout_refine_rows",
        &[BackendName::Metal, BackendName::Cuda],
        "level two of the progressive head's certified selection (decode rows)",
    ),
    (
        "readout_exact_rows",
        &[BackendName::Metal, BackendName::Cuda],
        "level three of the progressive head's certified selection (decode rows)",
    ),
    (
        "readout_planes_rows",
        &[BackendName::Metal, BackendName::Cuda],
        "the full exact readout over the progressive head's planes (decode rows)",
    ),
];

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
        "lib/core/arrive",
        &[BackendName::Metal],
        "last-threadgroup arrival over `sync` scratch; only Metal natives declare sync scratch so far",
    ),
    (
        "lib/core/gumbel",
        &[BackendName::Metal, BackendName::Cuda],
        "the sampler's Gumbel noise, shared by `sample_rows` and the certified selection's bounds",
    ),
    (
        "lib/readout/progressive",
        &[BackendName::Metal, BackendName::Cuda],
        "the progressive head's plane decoding and certified levels",
    ),
    (
        "lib/core/precise",
        &[BackendName::Vulkan],
        "bounded-error `exp`: Vulkan permits 3 + 2|x| ulp where Metal and CUDA do not",
    ),
    (
        "lib/projection/packing",
        &[BackendName::Metal],
        "the token-packing GEMM of the projection entries' PACK form, on simdgroup matrices",
    ),
    (
        "lib/core/rotary",
        &[BackendName::Vulkan],
        "rotary sin/cos shared by attention and vision; inside `attention.h` / `attention.cuh` elsewhere",
    ),
    (
        "lib/routed/expert_gate",
        &[BackendName::Vulkan],
        "representation ladder of `expert_gate` (GLSL has no templates); Metal and CUDA bind it through `packets` slots",
    ),
    (
        "lib/routed/expert_up",
        &[BackendName::Vulkan],
        "representation ladder of `expert_up` (GLSL has no templates); Metal and CUDA bind it through `packets` slots",
    ),
    (
        "lib/routed/expert_down",
        &[BackendName::Vulkan],
        "representation ladder of `expert_down` (GLSL has no templates); Metal and CUDA bind it through `packets` slots",
    ),
    (
        "lib/recurrent/convolution",
        &[BackendName::Metal],
        "the gated-delta convolution and window publication, apart from `recurrent.h` because the Metal-only `gated_delta_step_convolved` binds no window; the other backends keep them in their recurrent library",
    ),
    (
        "lib/routed/router",
        &[BackendName::Metal],
        "decode router logits shared by the Metal route and select GEMVs over the projection GEMV body (any activation element, unlike `routed.h`); CUDA and Vulkan keep their router GEMVs inline",
    ),
    (
        "lib/routed/route",
        &[BackendName::Metal],
        "decode router threadgroups of the `routed_route` contract, shared by `routed_route` and the Metal-only `routed_route_shared`",
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
        .source("kernels/short_conv.seismic")
        .source("kernels/state_space.seismic")
        .source("kernels/residual.seismic")
        .source("kernels/functions.seismic")
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

/// Every entry with a native implementation has one on each of `BACKENDS`
/// (on exactly its listed backends, for an `ENTRY_EXCEPTIONS` entry), at the
/// path named after the entry.
fn entry_parity(kernels: &Path, natives: &[NativeCoverage]) -> Vec<String> {
    let mut by_entry = BTreeMap::<&str, BTreeMap<BackendName, &Path>>::new();
    for native in natives {
        by_entry
            .entry(&native.entry)
            .or_default()
            .insert(native.backend, &native.asset);
    }
    let mut violations = Vec::new();
    for (entry, _, _) in ENTRY_EXCEPTIONS {
        if !by_entry.contains_key(entry) {
            violations.push(format!(
                "ENTRY_EXCEPTIONS lists `{entry}`, but no backend implements it; remove the exception"
            ));
        }
    }
    for (entry, assets) in by_entry {
        let exception = ENTRY_EXCEPTIONS
            .iter()
            .find(|(candidate, _, _)| *candidate == entry);
        if let Some((_, declared, _)) = exception {
            let declared = declared.iter().copied().collect::<BTreeSet<_>>();
            let actual = assets.keys().copied().collect::<BTreeSet<_>>();
            if declared != actual {
                violations.push(format!(
                    "ENTRY_EXCEPTIONS lists `{entry}` on {}, but it is implemented on {}; update or remove the exception",
                    names(&declared),
                    names(&actual)
                ));
            }
        }
        for backend in BACKENDS {
            if exception.is_some() && !assets.contains_key(&backend) {
                continue;
            }
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
            if extension != Some(entry_extension(backend))
                && extension != Some(library_extension(backend))
            {
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
        .chain(ENTRY_EXCEPTIONS)
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
