//! The embedder's artifact store around native formation. Each compile's
//! artifact is stored (CUDA's CUBIN, Vulkan's SPIR-V with its pipeline cache
//! data, Metal's binary archive of its library and pipelines),
//! with a pointer per entry naming the artifact that holds it. A later
//! process loads every program from the store and compiles nothing; a
//! damaged pointer, or a damaged artifact, is a miss, compiled and stored
//! again, and the store's other entries are kept. The CPU's native entries
//! (compiled into the binary) keep nothing.
//!
//! A process after the first forms nothing at all: every device opened
//! with default options forms through the user store, so a second process
//! over the same cache directory compiles nothing, misses no stored load and
//! runs no probe ([`a_later_process_compiles_nothing`]).

use seismic::{
    ArtifactKey, ArtifactStore, BackendName, Device, DeviceOptions, Element, NativeSpecialization,
    ProgramStore, Tensor,
};
use seismic_native_tests::{add_one, split_sum};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A store in memory, counting its lookups, hits and writes of programs.
/// The probe answers a device opens with (Metal's) it keeps apart, uncounted
/// and undamaged: they are not the programs these tests observe.
#[derive(Default)]
struct RecordingStore {
    store: Mutex<HashMap<(String, ArtifactKey), Vec<u8>>>,
    probes: Mutex<HashMap<ArtifactKey, Vec<u8>>>,
    counts: Mutex<Counts>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    gets: usize,
    hits: usize,
    puts: usize,
}

impl RecordingStore {
    fn counts(&self) -> Counts {
        *self.counts.lock().unwrap()
    }

    /// Everything the store holds, by namespace and key.
    fn stored(&self) -> HashMap<(String, ArtifactKey), Vec<u8>> {
        self.store.lock().unwrap().clone()
    }

    /// Replace every stored artifact with bytes no toolchain loads.
    fn corrupt(&self) {
        for bytes in self.store.lock().unwrap().values_mut() {
            *bytes = b"not an artifact".to_vec();
        }
    }
}

impl ArtifactStore for RecordingStore {
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        if namespace == seismic::PROBE_NAMESPACE {
            return self.probes.lock().unwrap().get(key).cloned();
        }
        let found = self.store.lock().unwrap().get(&(namespace.to_owned(), key.clone())).cloned();
        let mut counts = self.counts.lock().unwrap();
        counts.gets += 1;
        counts.hits += usize::from(found.is_some());
        found
    }

    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
        if namespace == seismic::PROBE_NAMESPACE {
            self.probes.lock().unwrap().insert(key.clone(), bytes.to_vec());
            return;
        }
        self.store.lock().unwrap().insert((namespace.to_owned(), key.clone()), bytes.to_vec());
        self.counts.lock().unwrap().puts += 1;
    }
}

/// The available devices of `backends`, opened from fresh `Devices` with
/// `store`, so nothing formed by an earlier open is reused in this process.
/// Their graph cost is given, so opening forms no graph-cost kernel: what
/// they form is what the test forms.
fn devices_of(backends: &[BackendName], store: &Arc<ProgramStore>) -> Vec<Device> {
    backends
        .iter()
        .flat_map(|backend| {
            // A backend this build lacks has no graph-cost bound, and no
            // device to open.
            if cfg!(target_os = "macos") && *backend == BackendName::Vulkan {
                return Vec::new();
            }
            seismic_native_tests::devices_with(
                &[*backend],
                DeviceOptions {
                    artifacts: Some(store.clone()),
                    graph_cost: Some(seismic::unmeasured_graph_cost(*backend)),
                },
            )
        })
        .collect()
}

/// The input of `split_sum` at N = 1000, all ones.
const N: u64 = 1000;

fn ones(device: &Device) -> Tensor {
    let bytes = (0..N).flat_map(|_| 1.0f32.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), &[N], &bytes).expect("host tensor")
}

/// The value of `split_sum`'s result `sum`, which sums [`ones`].
fn checked_sum(device: &Device, sum: &Tensor) {
    let sum = f32::from_le_bytes(sum.read_to_host().expect("read")[..4].try_into().unwrap());
    assert_eq!(sum, N as f32, "{:?}", device.backend());
}

/// Form and run the defaults of `split_sum` at N = 1000.
fn form_and_run(device: &Device) {
    let n = N;
    let implementation = split_sum::native_implementation(device)
        .expect("bundle")
        .expect("split_sum has an implementation on every backend");
    let defaults = implementation
        .default_specialization(&NativeSpecialization::new().with_static("N", n))
        .expect("statics");
    let kernel = split_sum::native_for_device(device, &defaults)
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
    let x = ones(device);
    checked_sum(device, &kernel.call(split_sum::Args { x: &x }).expect("call").value);
}

/// What forming `split_sum` on fresh devices of `backend` over `store`
/// did: the programs compiled and the stored loads the platform declined,
/// summed over the devices; none when no device of `backend` is available.
fn formed(backend: BackendName, store: &Arc<RecordingStore>) -> Option<(u64, u64)> {
    let programs = Arc::new(ProgramStore::new(store.clone()));
    let devices = devices_of(&[backend], &programs);
    devices.iter().for_each(form_and_run);
    // Every artifact is stored before the store is read, as a process
    // leaves them when its devices drop.
    programs.settle();
    (!devices.is_empty()).then(|| {
        devices.iter().fold((0, 0), |(compiles, misses), device| {
            let formed = device.programs_formed();
            (compiles + formed.compiles, misses + formed.load_misses)
        })
    })
}

/// Form `add_one` from its Seismic IR both ways a process prepares it, and
/// run it. Analytical preparation opens the device's analytical context,
/// whose service probes are formed through the device's programs before any
/// is timed; whether their timings then qualify is the host's, not the
/// store's, so its verdict is reported and never asserted. Preparation by
/// feedback with no search time times nothing: it forms the entry's one
/// candidate, which is run. The entry has no loop, so its IR leaves no
/// mapping to choose and every process forms the same kernel whichever
/// estimates its analytical context measured.
fn form_and_run_planned(device: &Device) {
    let analytical = seismic::PreparationOptions::analytical(seismic::PrecisionPolicy::Exact);
    match add_one::for_device(device, analytical) {
        Ok(_) => println!("{CHILD} analytical {:?} prepared", device.backend()),
        Err(error) => println!("{CHILD} analytical {:?} not prepared: {error}", device.backend()),
    }
    let options = seismic::PreparationOptions::feedback(
        seismic::PrecisionPolicy::Exact,
        seismic::FeedbackOptions { search_time: std::time::Duration::ZERO, ..Default::default() },
    );
    let kernel = add_one::for_device(device, options).unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
    let x = Tensor::from_host(device, Element::f32(), &[1], &1.0f32.to_le_bytes()).expect("host tensor");
    let result = kernel.call(add_one::Args { x: &x }).expect("call").value;
    let result = f32::from_le_bytes(result.read_to_host().expect("read")[..4].try_into().unwrap());
    assert_eq!(result, 2.0, "{:?}", device.backend());
}

/// The stored entries that are pointers: the artifact key each names, its
/// pointer's first word.
fn pointers(stored: &HashMap<(String, ArtifactKey), Vec<u8>>) -> Vec<(String, String)> {
    stored
        .iter()
        .filter_map(|((namespace, _), bytes)| {
            let text = std::str::from_utf8(bytes).ok()?.split(' ').next()?;
            let named = stored
                .keys()
                .any(|(held, key)| held == namespace && key.as_str() == text);
            named.then(|| (namespace.clone(), text.to_owned()))
        })
        .collect()
}

/// Form `split_sum` on a fresh device of `backend` three times: into an
/// empty store, from the store, and after the store's entries are damaged.
fn formation_uses_the_store(backend: BackendName) {
    let store = Arc::new(RecordingStore::default());
    let Some((compiles, _)) = formed(backend, &store) else {
        return;
    };
    let first = store.counts();
    if backend == BackendName::Cpu {
        assert_eq!(first, Counts::default(), "{backend:?} keeps nothing");
        return;
    }
    // Every program is a miss, compiled once and stored: its artifact, and
    // a pointer per entry naming it. Each key is written once.
    assert!(compiles > 0);
    assert_eq!(first.hits, 0, "{backend:?}");
    let kept = store.stored();
    assert_eq!(first.puts, kept.len(), "{backend:?}");
    let pointers = pointers(&kept);
    let artifacts = kept.len() - pointers.len();
    assert_eq!(artifacts as u64, compiles, "{backend:?}: one artifact a compile");
    assert!(pointers.len() >= artifacts, "{backend:?}: every entry is named");

    // A fresh device, as a later process is, loads every program from the
    // store: nothing compiles and nothing is written.
    let (compiled, misses) = formed(backend, &store).expect("the device opens again");
    assert_eq!(compiled, 0, "{backend:?}");
    assert_eq!(misses, 0, "{backend:?}");
    assert!(store.counts().hits > 0, "{backend:?}");
    assert_eq!(store.counts().puts, first.puts, "{backend:?}");
    assert_eq!(store.stored(), kept, "{backend:?}");

    // A damaged pointer is a miss: every program is compiled and stored
    // again.
    store.corrupt();
    let (compiled, _) = formed(backend, &store).expect("the device opens again");
    assert_eq!(compiled, compiles, "{backend:?}: compiled again");
    assert_eq!(store.counts().puts, 2 * first.puts, "{backend:?}");
}

/// Form `split_sum` on fresh devices of `backend` into an empty store, damage
/// one byte in the middle of one stored artifact, and form it again, twice:
/// the damaged artifact is a counted miss, its program compiled and the
/// artifact stored again, every other entry kept as it was; then the
/// rewritten artifact loads.
fn a_damaged_artifact_is_a_miss_and_is_rewritten(backend: BackendName) {
    let store = Arc::new(RecordingStore::default());
    if formed(backend, &store).is_none() {
        return;
    }
    let kept = store.stored();
    let named = pointers(&kept);
    let damaged = kept
        .keys()
        .find(|(namespace, key)| named.iter().any(|(held, text)| held == namespace && text == key.as_str()))
        .expect("an artifact is stored")
        .clone();
    let middle = kept[&damaged].len() / 2;
    store.store.lock().unwrap().get_mut(&damaged).expect("stored")[middle] ^= 0x5a;
    let (compiled, misses) = formed(backend, &store).expect("the device opens again");
    assert_eq!((compiled, misses), (1, 1), "{backend:?}: the damaged artifact is a counted miss, compiled again");
    let rewritten = store.stored();
    assert_eq!(rewritten.len(), kept.len(), "{backend:?}");
    for (key, bytes) in &kept {
        if *key != damaged {
            assert_eq!(&rewritten[key], bytes, "{backend:?}: every other entry is kept");
        }
    }
    assert_eq!(formed(backend, &store), Some((0, 0)), "{backend:?}: the rewritten artifact loads");
}

#[test]
fn metal_damaged_artifacts_are_misses() {
    a_damaged_artifact_is_a_miss_and_is_rewritten(BackendName::Metal);
}

#[test]
fn cuda_damaged_artifacts_are_misses() {
    a_damaged_artifact_is_a_miss_and_is_rewritten(BackendName::Cuda);
}

#[test]
fn vulkan_damaged_artifacts_are_misses() {
    a_damaged_artifact_is_a_miss_and_is_rewritten(BackendName::Vulkan);
}

#[test]
fn cpu_formation_never_uses_the_store() {
    formation_uses_the_store(BackendName::Cpu);
}

#[test]
fn metal_programs_are_kept_in_the_embedders_store() {
    formation_uses_the_store(BackendName::Metal);
}

#[test]
fn cuda_programs_are_kept_in_the_embedders_store() {
    formation_uses_the_store(BackendName::Cuda);
}

#[test]
fn vulkan_programs_are_kept_in_the_embedders_store() {
    formation_uses_the_store(BackendName::Vulkan);
}

/// The environment variable that marks a child process of
/// [`a_later_process_compiles_nothing`], and the prefix of each line it
/// reports.
const CHILD: &str = "SEISMIC_STORE_CHILD";

/// What one child process reported: each device's compiles and stored loads
/// missed, across all its toolchains, the device probes the process ran
/// (`seismic::probe_runs`), and each device's analytical verdict, which the
/// host's timings decide.
#[derive(Debug, Default)]
struct Child {
    devices: Vec<(String, u64, u64)>,
    probe_runs: u64,
    probe_misses: u64,
    analytical: Vec<String>,
}

/// This test's binary run again as a child, its cache directory `cache`:
/// what it reported ([`form_every_program`]).
fn child(cache: &std::path::Path) -> Child {
    let output = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args(["a_later_process_compiles_nothing", "--exact", "--nocapture", "--test-threads", "1"])
        .env(CHILD, "1")
        .env("XDG_CACHE_HOME", cache)
        .output()
        .expect("the child runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the child failed ({}):\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let mut reported = Child::default();
    for line in stdout.lines() {
        // A report can follow the test harness's own `test … ... ` on its
        // line, which the harness ends only after the test.
        let words = line.split_whitespace().skip_while(|word| *word != CHILD).collect::<Vec<_>>();
        let count = |word: &str| word.parse::<u64>().expect("a reported count");
        match words[..] {
            [CHILD, "device", backend, compiles, misses] => {
                reported.devices.push((backend.to_owned(), count(compiles), count(misses)))
            }
            [CHILD, "probes", runs, misses] => (reported.probe_runs, reported.probe_misses) = (count(runs), count(misses)),
            [CHILD, "analytical", ref verdict @ ..] => reported.analytical.push(verdict.join(" ")),
            _ => {}
        }
    }
    assert!(!reported.devices.is_empty(), "the child reported no device:\n{stdout}");
    reported
}

/// A child's work: open each selected backend's device with default
/// options, so it forms through the user store (Metal's discovery and
/// probes, graph cost and the analytical context included); form native
/// entries (but on the CPU, whose native entries are compiled into the
/// binary) and one Seismic-IR kernel both ways it is prepared (but on
/// Vulkan, which has no planned route); report what each device formed, and the device probes the
/// process ran, once every program is stored.
fn form_every_program() {
    use BackendName::{Cpu, Cuda, Metal, Vulkan};
    let devices = seismic_native_tests::devices(&[Metal, Cuda, Vulkan, Cpu]);
    for device in &devices {
        if device.backend() != Cpu {
            form_and_run(device);
        }
        if device.backend() != Vulkan {
            form_and_run_planned(device);
        }
    }
    ProgramStore::user().settle();
    for device in &devices {
        let formed = device.programs_formed();
        println!("{CHILD} device {:?} {} {}", device.backend(), formed.compiles, formed.load_misses);
    }
    println!("{CHILD} probes {} {}", seismic::probe_runs(), seismic::probe_misses());
}

/// Two processes over one fresh cache directory, each opening its devices
/// with default options and forming the same programs: the first compiles,
/// and the second compiles nothing, misses no stored load and runs no
/// probe, on every toolchain of every device.
#[test]
fn a_later_process_compiles_nothing() {
    if std::env::var_os(CHILD).is_some() {
        return form_every_program();
    }
    let cache = std::env::temp_dir().join(format!("seismic-later-process-{}", std::process::id()));
    if cache.exists() {
        std::fs::remove_dir_all(&cache).expect("an earlier run's cache is removed");
    }
    std::fs::create_dir_all(&cache).expect("the cache directory");
    let first = child(&cache);
    let second = child(&cache);
    std::fs::remove_dir_all(&cache).expect("the cache is removed");
    // Each device's counts, the evidence of which backends this run covered.
    println!("first process: {first:?}\nsecond process: {second:?}");
    let compiled = first.devices.iter().map(|(_, compiles, _)| compiles).sum::<u64>();
    assert!(compiled > 0, "the first process compiles: {first:?}");
    assert_eq!(
        first.devices.iter().map(|(backend, ..)| backend).collect::<Vec<_>>(),
        second.devices.iter().map(|(backend, ..)| backend).collect::<Vec<_>>(),
        "both processes open the same devices"
    );
    for (backend, compiles, misses) in &second.devices {
        assert_eq!((*compiles, *misses), (0, 0), "{backend}: the second process compiles nothing and misses no load");
    }
    assert_eq!((second.probe_runs, second.probe_misses), (0, 0), "the second process runs no probe and finds none damaged: {second:?}");
}
