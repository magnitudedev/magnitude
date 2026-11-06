use seismic_build::{BackendName, NativeCoverage};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

include!("tests/gguf_formats/widths.rs");

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
        &[BackendName::Metal, BackendName::Cuda, BackendName::Vulkan],
        "decode routing that also expands the shared expert, the decode form of the backends that stream decode; the other backends run `routed_route`, then `routed_expand`",
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
];

/// Expert streaming's entries (routed.seismic's streaming section), on the
/// backends that stream (`STREAMING_BACKENDS`): the CPU backend streams no
/// experts.
const STREAMING_ENTRIES: &[&str] = &[
    "routed_route_streamed",
    "routed_select_streamed",
    "routed_tier_wait",
    "routed_site_request",
    "routed_down_project",
    "routed_down_sited",
    "routed_output_project",
    "routed_output_sited",
    "routed_step",
    "routed_scratch_request",
    "routed_scratch_wait",
    "routed_scratch_grids",
    "routed_copy",
    "routed_list_grid",
    "routed_pass_map",
    "routed_list_stage",
    "routed_move",
    "routed_convert",
    "routed_seed_commit",
];

const STREAMING_BACKENDS: &[BackendName] =
    &[BackendName::Metal, BackendName::Cuda, BackendName::Vulkan];

/// Every entry exception: `ENTRY_EXCEPTIONS`, then `STREAMING_ENTRIES`.
fn entry_exceptions() -> impl Iterator<Item = (&'static str, &'static [BackendName], &'static str)>
{
    let streaming = "expert streaming's entry, on the backends that stream";
    ENTRY_EXCEPTIONS.iter().copied().chain(
        STREAMING_ENTRIES
            .iter()
            .map(move |&entry| (entry, STREAMING_BACKENDS, streaming)),
    )
}

/// Files of the per-backend trees that exist on only some backends: (path
/// relative to `kernels/<backend>/`, without extension; the backends that have
/// it; why). Every other file — entry asset or library file, at any depth —
/// exists on all of `BACKENDS`, apart from `STREAMING_LIBRARIES` and the
/// assets of `entry_exceptions`. An exception whose actual presence differs,
/// including one now on every backend, fails the build so this list cannot go
/// stale.
const TREE_EXCEPTIONS: &[(&str, &[BackendName], &str)] = &[
    (
        "lib/routed/resolve",
        &[BackendName::Cuda],
        "the streamed routes' resolve over their bank ports is separate from CUDA's streaming shim so the other streaming entries bind none of them",
    ),
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
        "lib/core/arrive",
        &[BackendName::Metal, BackendName::Cuda],
        "last-threadgroup arrival over `sync` scratch, for the GPU natives that declare sync scratch",
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
        "lib/routed/shared",
        &[BackendName::Metal, BackendName::Cuda],
        "the shared expert's expansion in a decode route's launch, shared by `routed_route_shared` and `routed_route_streamed`, on the backends that declare them",
    ),
    (
        "lib/routed/logits",
        &[BackendName::Metal],
        "the prefill router logits tile on F32 simdgroup matrices, shared by the Metal route and select GEMMs; CUDA and Vulkan keep their router GEMMs inline",
    ),
    (
        "lib/routed/route_body",
        &[BackendName::Vulkan],
        "Vulkan resident router arithmetic rebound to the current or next-layer router's ports",
    ),
    (
        "lib/routed/selection_body",
        &[BackendName::Vulkan],
        "Vulkan resident selection arithmetic rebound to the current or next-layer router's ports",
    ),
    (
        "lib/routed/route_shared",
        &[BackendName::Vulkan],
        "Vulkan shared expert projection over the resident paired GEMV implementation",
    ),
    (
        "lib/routed/streamed",
        &[BackendName::Vulkan],
        "Vulkan workgroup adapters to the shared streaming resolution and hint protocol",
    ),
];

/// Expert streaming's library files, on the backends that stream
/// (`STREAMING_BACKENDS`): (path as in `TREE_EXCEPTIONS`; why it is a file of
/// its own). The CPU backend streams no experts and keeps the resident bodies
/// these share inline in its entries.
const STREAMING_LIBRARIES: &[(&str, &str)] = &[
    (
        "lib/routed/down",
        "the general decode combine's projections, shared by `routed_down` and `routed_down_project`, apart from `routed.h` so the entries that include that bind none of the combine's ports",
    ),
    (
        "lib/routed/down_sum",
        "the general decode combine's slot-order sum, shared by `routed_down` and `routed_down_sited`, apart from the projections so the sited sum binds no weights",
    ),
    (
        "lib/routed/output",
        "the fused decode output's projections, shared by `routed_output` and `routed_output_project`, apart from `routed.h` so the entries that include that bind none of the combine's ports",
    ),
    (
        "lib/routed/output_sum",
        "the fused decode output's join, shared by `routed_output` and `routed_output_sited`, apart from the projections so the sited join binds no weights",
    ),
    (
        "lib/routed/sited",
        "the wait for the CPU's answer of the streamed decode sums, apart from them so the resident combines bind no answer port",
    ),
    (
        "lib/routed/route",
        "decode router threadgroups of the `routed_route` contract, shared by `routed_route`, `routed_route_shared` and `routed_route_streamed`",
    ),
    (
        "lib/repack/repack",
        "the GGUF -> resident tile conversion over a binding (templates; on Vulkan, macros), so `repack_weight` and the streaming convert hold one conversion each per tensor pair; the CPU keeps it in `repack_weight`",
    ),
    (
        "lib/streaming/control",
        "expert streaming's backend shim (system-coherent accesses, system-scope fences, barriers, the spin wait, byte copies) under the protocol the GPU backends share in `kernels/streaming/`, for the streaming entries",
    ),
    (
        "lib/routed/selection",
        "the decode launches' bodies of the `routed_select` contract, shared by `routed_select` and `routed_select_streamed`",
    ),
    (
        "lib/streaming/window",
        "expert streaming's windowed conversion of host-staged records over `lib/repack`, used by `routed_convert`",
    ),
];

/// Every tree exception: `TREE_EXCEPTIONS`, then `STREAMING_LIBRARIES`.
fn tree_exceptions() -> impl Iterator<Item = (&'static str, &'static [BackendName], &'static str)>
{
    TREE_EXCEPTIONS.iter().copied().chain(
        STREAMING_LIBRARIES
            .iter()
            .map(|&(path, reason)| (path, STREAMING_BACKENDS, reason)),
    )
}

fn main() {
    // The one source is the `kernels` directory, which the build reruns on
    // (`cargo:rerun-if-changed` of a directory watches every file under it,
    // a new one included).
    let artifacts = seismic_build::Build::new("kernels")
        .source("kernels")
        .run()
        .unwrap_or_else(|error| {
            panic!("checking engine Seismic sources and generating bindings failed: {error}")
        });

    gguf_format_tests(&PathBuf::from(
        std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"),
    ));

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
    let incoherent = metal_arrival_coherence(&kernels.join("metal"));
    if !incoherent.is_empty() {
        panic!(
            "Metal arrival handoffs are not device-coherent ({} violations):\n  - {}",
            incoherent.len(),
            incoherent.join("\n  - ")
        );
    }
}

/// Writes `gguf_format_tests.rs`: the macro `gguf_format_tests!`, whose arm
/// for a layout (`rows16; every: [..]; partial_block: [..]; own_storage:
/// [..]`) expands to `mod gguf` with one module for each storage the
/// registry converts GGUF formats into in that layout, holding one module
/// for each such format, in ggml id order, a format under a scale grid
/// named for it too (`gguf::q2g256::tq2_0`, `gguf::mxfp8g32::f8_e4m3_e8m0`).
/// A format's module has a test for each of the `every` functions, and for
/// each `partial_block` function when whole blocks of it fill a
/// `PARTIAL_BLOCK_WIDTHS` width (`partial_block_widths`). A storage's
/// module has a test for each `own_storage` function, of its first format,
/// when it is a rounded element (an element of another operand recipe
/// shares its storage). Each test calls `crate::<function>(format)` with the
/// format's `crate::Format`.
fn gguf_format_tests(out: &Path) {
    use magnitude_ggml::Encoding;
    use seismic_lang::registry::{self, GgufScales, Layout, OperandRecipe};
    let mut arms = String::new();
    for layout in Layout::ALL {
        // Each storage's modules, in the order its first format comes.
        let mut storages: Vec<(String, String)> = Vec::new();
        for (encoding, scales) in Encoding::ALL
            .into_iter()
            .flat_map(|encoding| [GgufScales::Own, GgufScales::E8m0Grid].map(|scales| (encoding, scales)))
        {
            let (block, suffix) = match scales {
                GgufScales::Own => (encoding.block_elements(), ""),
                GgufScales::E8m0Grid => (GgufScales::GRID_BLOCK, "_e8m0"),
            };
            let Some(source) = registry::gguf_source(encoding, scales, block) else {
                continue;
            };
            for operands in OperandRecipe::ALL {
                let Some(element) = registry::conversion_destination(source, layout, operands)
                else {
                    continue;
                };
                let name = format!(
                    "{}{}",
                    registry::representation_info(element).representation,
                    operands.suffix()
                );
                let index = storages
                    .iter()
                    .position(|(known, _)| *known == name)
                    .unwrap_or_else(|| {
                        let own = if operands == OperandRecipe::Rounded {
                            format!(
                                " crate::gguf_format_tests!(@tests {encoding:?} {scales:?} {operands:?}: $($own)*);"
                            )
                        } else {
                            String::new()
                        };
                        storages.push((name, own));
                        storages.len() - 1
                    });
                let mut tests = String::from("$($every)*");
                if partial_block_widths(block as usize).next().is_some() {
                    tests.push_str(" $($partial)*");
                }
                storages[index].1.push_str(&format!(
                    " pub mod {}{suffix} {{ crate::gguf_format_tests!(@tests {encoding:?} {scales:?} {operands:?}: {tests}); }}",
                    encoding.ggml_name().to_ascii_lowercase()
                ));
            }
        }
        let modules: String = storages
            .iter()
            .map(|(name, body)| format!("            pub mod {name} {{{body} }}\n"))
            .collect();
        arms.push_str(&format!(
            "    ({}; every: [$($every:ident)*]; partial_block: [$($partial:ident)*]; own_storage: [$($own:ident)*] $(;)?) => {{\n        mod gguf {{\n{modules}        }}\n    }};\n",
            layout.as_str()
        ));
    }
    let text = format!(
        "/// Generated by `build.rs` (`gguf_format_tests`) from the registry.\nmacro_rules! gguf_format_tests {{\n{arms}    (@tests $encoding:ident $scales:ident $operands:ident: $($test:ident)*) => {{\n        $(\n            #[test]\n            fn $test() {{\n                crate::$test(crate::Format {{\n                    encoding: ::magnitude_ggml::Encoding::$encoding,\n                    scales: ::seismic::GgufScales::$scales,\n                    operands: ::seismic::OperandRecipe::$operands,\n                }});\n            }}\n        )*\n    }};\n}}\n"
    );
    std::fs::write(out.join("gguf_format_tests.rs"), text)
        .unwrap_or_else(|error| panic!("writing gguf_format_tests.rs: {error}"));
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
/// path named after the entry, and every file directly in a backend's tree
/// is an entry's asset.
fn entry_parity(kernels: &Path, natives: &[NativeCoverage]) -> Vec<String> {
    let mut by_entry = BTreeMap::<&str, BTreeMap<BackendName, &Path>>::new();
    for native in natives {
        by_entry
            .entry(&native.entry)
            .or_default()
            .insert(native.backend, &native.asset);
    }
    let mut violations = Vec::new();
    for backend in BACKENDS {
        let tree = kernels.join(backend.as_str());
        let assets = std::fs::read_dir(&tree)
            .unwrap_or_else(|error| panic!("reading {}: {error}", tree.display()));
        for asset in assets {
            let path = asset.expect("a kernel directory entry").path();
            let entry = path.file_stem().and_then(|stem| stem.to_str());
            if path.is_file() && !entry.is_some_and(|entry| by_entry.contains_key(entry)) {
                violations.push(format!("`{}` is no entry's asset", relative(kernels, &path).display()));
            }
        }
    }
    for (entry, _, _) in entry_exceptions() {
        if !by_entry.contains_key(entry) {
            violations.push(format!(
                "the entry exceptions list `{entry}`, but no backend implements it; remove the exception"
            ));
        }
    }
    for (entry, assets) in by_entry {
        let exception = entry_exceptions().find(|(candidate, _, _)| *candidate == entry);
        if let Some((_, declared, _)) = exception {
            let declared = declared.iter().copied().collect::<BTreeSet<_>>();
            let actual = assets.keys().copied().collect::<BTreeSet<_>>();
            if declared != actual {
                violations.push(format!(
                    "the entry exceptions list `{entry}` on {}, but it is implemented on {}; update or remove the exception",
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
/// without extension, apart from `tree_exceptions` and `entry_exceptions`.
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
    let exceptions: BTreeMap<&str, (BTreeSet<BackendName>, &str)> = tree_exceptions()
        .chain(entry_exceptions())
        .map(|(stem, backends, reason)| (stem, (backends.iter().copied().collect(), reason)))
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
                "`{stem}` is excepted on {}, but it exists on {}; update or remove the exception",
                names(declared),
                names(backends)
            )),
            Some(_) => {}
        }
    }
    for (stem, (declared, _)) in &exceptions {
        if !presence.contains_key(*stem) {
            violations.push(format!(
                "`{stem}` is excepted on {}, but no backend has it; remove the exception",
                names(declared)
            ));
        }
    }
    violations
}

/// A Metal kernel that counts arrivals over atomic scratch, so that the last
/// of its threadgroups finishes the work (`arrive::last`), declares every
/// other writable scratch buffer device-coherent (`arrive::coherent_device`):
/// the last threadgroup's plain loads of another threadgroup's stores are
/// otherwise served from a stale cache on M1 Max-class GPUs
/// (`lib/core/arrive.h`). Read-only (`const`) scratch was written by an
/// earlier launch and needs nothing, and so does the scratch of another
/// launch of the entry that counts none.
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
        let name = relative(metal, &path).display().to_string();
        // Each kernel's parameters, from its `kernel` to the next one's.
        for kernel in text.split("kernel void").skip(1) {
            let declarations = scratch_declarations(kernel);
            // A kernel that binds atomic scratch counts arrivals over it
            // (the `sync` scratch of `arrive::last`).
            if !declarations
                .iter()
                .any(|declaration| declaration.contains("atomic_"))
            {
                continue;
            }
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
