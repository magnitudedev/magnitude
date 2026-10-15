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
    ),
    (
        "packing_bound",
        &[BackendName::Metal],
        "a weight's packing bound, which only Metal's token-packing projection forms read",
    ),
    (
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
    (
        "routed_route_streamed",
        &[BackendName::Metal],
        "expert streaming's routing with in-step slot resolution and prefetch hints, published to the host through a qualified system-coherent handshake; Metal is the first backend qualified for it (specs/26-10-11/expert-streaming.md 4.3)",
    ),
    (
        "routed_bank_fill",
        &[BackendName::Metal],
        "expert streaming's park and conversion of a streamed layer's absent experts into their bank slots, and their commit; Metal first, with `routed_route_streamed`",
    ),
    (
        "routed_scratch_request",
        &[BackendName::Metal],
        "expert streaming's scratch fill request, published to the host mid-step through system-coherent memory; Metal first, with `routed_route_streamed`",
    ),
    (
        "routed_scratch_wait",
        &[BackendName::Metal],
        "expert streaming's parked scratch fill rounds; Metal first, with `routed_route_streamed`",
    ),
    (
        "routed_convert_wide",
        &[BackendName::Metal],
        "expert streaming's scratch-round convert of the round's staged records into their scratch slots; Metal first, with `routed_route_streamed`",
    ),
    (
        "routed_copy",
        &[BackendName::Metal],
        "expert streaming's copy of listed records between the bank and the scratch bank; Metal first, with `routed_route_streamed`",
    ),
    (
        "routed_seed_commit",
        &[BackendName::Metal],
        "expert streaming's commit of the experts a scratch fill seeded into the bank, published to the host through system-coherent memory; Metal first, with `routed_route_streamed`",
    ),
    (
        "routed_step",
        &[BackendName::Metal],
        "expert streaming's device step sequence, read by the request, stamp and hint words; Metal first, with `routed_route_streamed`",
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
        "lib/attention/inputs",
        &[BackendName::Cuda],
        "query and visibility ABI macros are separate from CUDA shared attention helpers so cache-publication entries bind only K/V state inputs",
    ),
    (
        "lib/attention/decode_merge",
        &[BackendName::Cuda],
        "decode output helpers are separate from CUDA's shared attention header so append-only entries do not require output bindings",
    ),
    (
        "lib/attention/prefill",
        &[BackendName::Cuda, BackendName::Vulkan],
        "prefill attention body (CUDA tensor-core flash, Vulkan tiled); Metal keeps its prefill in `attention.h`",
    ),
    (
        "lib/core/matrix",
        &[BackendName::Vulkan],
        "cooperative-matrix fragment types and the one zero accumulator every Vulkan matrix body clears with",
    ),
    (
        "lib/attention/flash",
        &[BackendName::Vulkan],
        "cooperative-matrix tile pieces shared by the prefill and vision attention bodies",
    ),
    (
        "lib/streaming/convert",
        &[BackendName::Metal],
        "expert streaming's convert kernel body, shared by its two Metal entries",
    ),
    (
        "lib/repack/bind",
        &[BackendName::Metal],
        "`repack_weight`'s binding type of the Metal repack templates",
    ),
    (
        "lib/streaming/bind_convert",
        &[BackendName::Metal],
        "`routed_convert_wide`'s binding type of the Metal repack templates",
    ),
    (
        "lib/streaming/bind_gate",
        &[BackendName::Metal],
        "the gate conversion's binding type of the Metal repack templates in `routed_bank_fill`, which holds three conversions",
    ),
    (
        "lib/streaming/bind_up",
        &[BackendName::Metal],
        "the up conversion's binding type of the Metal repack templates in `routed_bank_fill`",
    ),
    (
        "lib/streaming/bind_down",
        &[BackendName::Metal],
        "the down conversion's binding type of the Metal repack templates in `routed_bank_fill`",
    ),
    (
        "lib/streaming/window",
        &[BackendName::Metal],
        "expert streaming's windowed conversion of staged records, shared by the Metal convert entries and the bank fill",
    ),
    (
        "lib/streaming/control",
        &[BackendName::Metal],
        "expert streaming's system-coherent handshake words and fences, shared by the Metal-only streaming entries",
    ),
    (
        "lib/repack/repack",
        &[BackendName::Metal],
        "the GGUF -> resident conversion body shared by `repack_weight` and the Metal-only streaming conversions",
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
        "lib/streams/down_weight",
        &[BackendName::Vulkan],
        "representation ladder of a mix site's `down_weight` (GLSL has no templates); Metal and CUDA bind it through `packets` slots",
    ),
    (
        "lib/streams/inject_weight",
        &[BackendName::Vulkan],
        "representation ladder of a mix site's `inject_weight` (GLSL has no templates); Metal and CUDA bind it through `packets` slots",
    ),
    (
        "lib/streams/up_weight",
        &[BackendName::Vulkan],
        "representation ladder of a mix site's `up_weight` (GLSL has no templates); Metal and CUDA bind it through `packets` slots",
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
        .source("kernels/clustered_readout.seismic")
        .source("kernels/vision.seismic")
        .source("kernels/recurrent.seismic")
        .source("kernels/short_conv.seismic")
        .source("kernels/state_space.seismic")
        .source("kernels/residual.seismic")
        .source("kernels/streams.seismic")
        .source("kernels/ngram.seismic")
        .source("kernels/indexer.seismic")
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
    violations.extend(repack_bindings(&kernels.join("metal/lib")));
    if !violations.is_empty() {
        panic!(
            "native backend parity failed ({} violations):\n  - {}",
            violations.len(),
            violations.join("\n  - ")
        );
    }
    let incoherent = metal_arrival_coherence(&kernels.join("metal"));
    if !incoherent.is_empty() {
        panic!(
            "Metal arrival handoffs are not device-coherent ({} violations):\n  - {}",
            incoherent.len(),
            incoherent.join("\n  - ")
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

/// A Metal entry that counts arrivals over atomic scratch, so that the last of
/// its threadgroups finishes the work (`arrive::last`), declares every other
/// writable scratch buffer device-coherent (`arrive::coherent_device`): the
/// last threadgroup's plain loads of another threadgroup's stores are
/// otherwise served from a stale cache on M1 Max-class GPUs
/// (`lib/core/arrive.h`). Read-only (`const`) scratch was written by an
/// earlier launch and needs nothing.
fn metal_arrival_coherence(metal: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    let entries = std::fs::read_dir(metal)
        .unwrap_or_else(|error| panic!("reading {}: {error}", metal.display()));
    for entry in entries {
        let path = entry.expect("a Metal kernel directory entry").path();
        if path.extension().and_then(|value| value.to_str()) != Some("metal") {
            continue;
        }
        let mut seen = BTreeSet::new();
        let text = expand_includes(metal, &path, &mut seen);
        let declarations = scratch_declarations(&text);
        // An entry that binds atomic scratch counts arrivals over it (the
        // `sync` scratch of `arrive::last`).
        if !declarations
            .iter()
            .any(|declaration| declaration.contains("atomic_"))
        {
            continue;
        }
        let name = relative(metal, &path).display().to_string();
        for declaration in declarations {
            let coherent = declaration.contains("coherent");
            let atomic = declaration.contains("atomic_");
            let read_only = declaration.contains("const ");
            if !(coherent || atomic || read_only) {
                violations.push(format!(
                    "`metal/{name}` hands work to its last-arriving threadgroup but declares writable scratch `{}` without `arrive::coherent_device` (lib/core/arrive.h)",
                    declaration.split_whitespace().collect::<Vec<_>>().join(" ")
                ));
            }
        }
    }
    violations
}

/// `path`'s text with each local `#include "..."` replaced by the included
/// file's (relative to the including file, else to the Metal root), each file
/// once.
fn expand_includes(metal: &Path, path: &Path, seen: &mut BTreeSet<PathBuf>) -> String {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !seen.insert(path.clone()) {
        return String::new();
    }
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let mut expanded = String::new();
    for line in text.lines() {
        let included = line
            .trim_start()
            .strip_prefix("#include \"")
            .and_then(|rest| rest.split('"').next());
        match included {
            Some(name) => {
                let beside = path.parent().map(|parent| parent.join(name));
                let file = beside
                    .filter(|candidate| candidate.exists())
                    .unwrap_or_else(|| metal.join(name));
                if file.exists() {
                    expanded.push_str(&expand_includes(metal, &file, seen));
                }
            }
            None => {
                expanded.push_str(line);
                expanded.push('\n');
            }
        }
    }
    expanded
}

/// The parameter declarations bound to scratch buffers
/// (`[[buffer(SEISMIC_BUFFER_SCRATCH_...)]]`), from the preceding parameter
/// separator.
fn scratch_declarations(text: &str) -> Vec<String> {
    let mut declarations = Vec::new();
    let marker = "[[buffer(SEISMIC_BUFFER_SCRATCH_";
    let mut from = 0;
    while let Some(found) = text[from..].find(marker) {
        let at = from + found;
        let start = text[..at]
            .rfind([',', '('])
            .map_or(0, |position| position + 1);
        declarations.push(text[start..at].replace('\\', " ").trim().to_owned());
        from = at + marker.len();
    }
    declarations
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

/// The repack templates (`repack/repack.h`, `streaming/window.h`) read a
/// conversion's elements and geometry from a binding type `B`, so one kernel
/// holds several conversions. Every `B::` member those headers read must be
/// defined by each binding: one a binding misses fails only the kernels
/// that instantiate it.
fn repack_bindings(lib: &Path) -> Vec<String> {
    let read = |path: &str| {
        let path = lib.join(path);
        println!("cargo:rerun-if-changed={}", path.display());
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    };
    let mut members = BTreeSet::new();
    for header in ["repack/repack.h", "streaming/window.h"] {
        let text = read(header);
        let mut rest = text.as_str();
        while let Some(at) = rest.find("B::") {
            let tail = &rest[at + 3..];
            let end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(tail.len());
            if end > 0 {
                members.insert(tail[..end].to_owned());
            }
            rest = &tail[end..];
        }
    }
    let mut violations = Vec::new();
    for binding in [
        "repack/bind.h",
        "streaming/bind_convert.h",
        "streaming/bind_gate.h",
        "streaming/bind_up.h",
        "streaming/bind_down.h",
    ] {
        let text = read(binding);
        for member in &members {
            if !text.contains(&format!(" {member}(")) {
                violations.push(format!("metal/lib/{binding} does not define {member}, which the repack templates read"));
            }
        }
    }
    violations
}
