//! Program preparation of pinned models on the host's accelerator: every
//! parameterized entry is tuned on real resident weights of the model, each
//! distinct case once, with no defect. This composes a model family with the
//! executor, so it lives at the engine layer (the executor stays
//! family-agnostic). The 4B covers the dense, attention, recurrent-state and
//! selection cases; the 35B covers the routed ones (route, group, expand,
//! output, experts, combine). With a kernel cache, a second load uses every
//! stored tuning result (and, on CUDA, every stored image) and tunes nothing.
//!
//! `cargo test --release -p magnitude-engine --test native_tuning -- --ignored
//! --test-threads 1 --nocapture` (CUDA: with `SEISMIC_NVRTC_DIRECTORY`).

use magnitude_artifacts::Package;
use magnitude_model_executor::{
    platform::{self, DeviceRequest, PlatformConfig},
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, KernelCache,
    PlannedMethod, ResourceBudget, ResourceLimits, TunedEntry, TuningContext, TuningOrigin,
    UnreportedTuning, DEFAULT_KERNEL_CACHE_BYTES,
};
use magnitude_model_state::KvCodec;
use seismic::{BackendName, DeviceCatalog};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

/// The pinned 4B file on the Metal hosts and on the CUDA host.
const PINNED_4B: [&str; 2] = [
    "/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
    "/models/unsloth-qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf",
];
/// The pinned 35B-A3B file (same hub path on every host).
const PINNED_35B: [&str; 1] = [
    "/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-Q4_K_M.gguf",
];

/// Prepare every program of the first existing file of `candidates` on the
/// automatically selected accelerator `loads` times, with `cache` as the
/// kernel cache, and return each load's tunings. Every entry of the first
/// load must tune without defect.
fn prepare_every_entry(
    candidates: &[&str],
    storage_gib: u64,
    cache: Option<Arc<KernelCache>>,
    loads: usize,
) -> (BackendName, Vec<Vec<TunedEntry>>) {
    let home = std::env::var("HOME").unwrap();
    let path = candidates
        .iter()
        .map(|relative| PathBuf::from(home.clone() + *relative))
        .find(|path| path.exists())
        .unwrap_or_else(|| panic!("none of {candidates:?} exists under {home}"));
    let catalog = DeviceCatalog::discover().unwrap();
    let selected =
        platform::select_device(&catalog, ExecutionPath::Native, DeviceRequest::Automatic)
            .unwrap_or_else(|error| panic!("no accelerator: {error}"));
    let package = Package::open_without_projector(&path).unwrap();
    let definition = magnitude_model_qwen35::inspect_package(&package).unwrap();
    let limits = ResourceLimits {
        active_requests: 1,
        in_flight_requests: 1,
        branch_checkpoints: 1,
        max_batch_rows: 512,
        max_projected_rows: 8,
        max_images_per_request: 1,
        lookahead: false,
    };
    let storage_bytes = storage_gib << 30;
    let draft = ExecutionPlanner::prepare(
        &selected,
        &package.manifest(),
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        ExecutionPath::Native,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
        ResourceBudget {
            storage_bytes,
            retention_bytes: 0,
            safety_reserve_bytes: 1 << 30,
        },
    )
    .unwrap();
    let opened = platform::open_selected(
        &catalog,
        draft.device().selector(),
        PlatformConfig {
            path: ExecutionPath::Native,
            storage_bytes,
            artifacts: cache
                .clone()
                .map(|cache| cache as Arc<dyn seismic::ArtifactStore>),
        },
    )
    .unwrap();
    let backend = opened.device().backend();
    let mut tunings = Vec::with_capacity(loads);
    for load in 0..loads {
        let began = std::time::Instant::now();
        let programs = AttestedPrograms::prepare_draft(
            &draft,
            opened.device(),
            TuningContext {
                definition: &definition,
                weights: &package,
                observer: &UnreportedTuning,
                cache: cache.as_deref(),
            },
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let seconds = began.elapsed().as_secs_f64();
        let mut cases = HashSet::new();
        for tuned in programs.tuned() {
            eprintln!(
                "{:.2} s {} [{}] {:?} {:?} ({} measured, {} excluded, {} defects)",
                tuned.seconds,
                tuned.entry,
                tuned.bindings,
                tuned.origin,
                tuned.overall.params,
                tuned.measured,
                tuned.excluded,
                tuned.defects
            );
            assert!(
                cases.insert((tuned.entry, tuned.bindings.clone())),
                "{} [{}] was tuned twice",
                tuned.entry,
                tuned.bindings
            );
        }
        eprintln!(
            "{backend:?} {} load {load}: prepared in {seconds:.2} s, {:.2} s tuning {} entries",
            path.file_name().unwrap().to_string_lossy(),
            programs.tuned().iter().map(|tuned| tuned.seconds).sum::<f64>(),
            programs.tuned().len()
        );
        tunings.push(programs.tuned().to_vec());
    }
    let defective = tunings[0]
        .iter()
        .filter(|tuned| tuned.measured == 0 || tuned.defects > 0)
        .map(|tuned| format!("{} [{}]: {:?}", tuned.entry, tuned.bindings, tuned.first_defect))
        .collect::<Vec<_>>();
    assert!(defective.is_empty(), "{defective:#?}");
    (backend, tunings)
}

#[test]
#[ignore = "reads the pinned 4B GGUF; needs Metal or CUDA"]
fn the_pinned_4b_tunes_every_entry_once_on_real_weights() {
    prepare_every_entry(&PINNED_4B, 16, None, 1);
}

#[test]
#[ignore = "reads the pinned 35B GGUF; needs Metal or CUDA and ~24 GiB"]
fn the_pinned_35b_tunes_every_entry_once_on_real_weights() {
    prepare_every_entry(&PINNED_35B, 22, None, 1);
}

/// Tuning spec §C and §E4 check 2: the first load with an empty kernel cache
/// searches every unit and stores its result (and, on CUDA, every formed
/// image); the second load uses every stored result, choosing the same
/// configurations without forming or measuring anything to tune.
#[test]
#[ignore = "reads the pinned 4B GGUF; needs Metal or CUDA"]
fn the_pinned_4b_uses_stored_tuning_on_its_second_load() {
    let root = std::env::temp_dir().join(format!("magnitude-kernel-cache-4b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = Arc::new(KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap());
    let (backend, loads) = prepare_every_entry(&PINNED_4B, 16, Some(cache), 2);
    let files = |directory: &str| std::fs::read_dir(root.join(directory)).unwrap().count();
    let (first, second) = (&loads[0], &loads[1]);
    assert!(first.iter().all(|tuned| tuned.origin == TuningOrigin::Searched));
    assert_eq!(files("tuning"), first.len());
    assert!(second.iter().all(|tuned| tuned.origin == TuningOrigin::Stored
        && tuned.time.forming_seconds == 0.0
        && tuned.time.measuring_seconds == 0.0
        && tuned.time.validating_seconds == 0.0));
    let choices = |load: &[TunedEntry]| {
        load.iter()
            .map(|tuned| (tuned.entry, tuned.bindings.clone(), tuned.overall.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(choices(first), choices(second));
    if backend == BackendName::Cuda {
        assert!(files("cuda") > 0, "CUDA images are stored");
    } else {
        assert_eq!(files("cuda"), 0);
    }
    std::fs::remove_dir_all(root).unwrap();
}
