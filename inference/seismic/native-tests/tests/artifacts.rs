//! The embedder's artifact store around native formation. Each compile's
//! artifact is stored (CUDA's CUBIN, Vulkan's SPIR-V with its pipeline cache
//! data, Metal's source and entries, whose compiled forms the OS caches),
//! with a pointer per entry naming the artifact that holds it. A later
//! process loads every program from the store and compiles nothing; a
//! damaged pointer is a miss, compiled and stored again. CPU (compiled into
//! the binary) keeps nothing.

use seismic::{
    ArtifactKey, ArtifactStore, BackendName, Device, DeviceOptions, Element,
    NativeSpecialization, Tensor,
};
use seismic_native_tests::split_sum;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// An in-memory store that counts lookups, hits and writes.
#[derive(Default)]
struct RecordingStore {
    entries: Mutex<HashMap<(String, ArtifactKey), Vec<u8>>>,
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
        self.entries.lock().unwrap().clone()
    }

    /// Replace every stored artifact with bytes no toolchain loads.
    fn corrupt(&self) {
        for bytes in self.entries.lock().unwrap().values_mut() {
            *bytes = b"not an artifact".to_vec();
        }
    }
}

impl ArtifactStore for RecordingStore {
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        let found = self
            .entries
            .lock()
            .unwrap()
            .get(&(namespace.to_owned(), key.clone()))
            .cloned();
        let mut counts = self.counts.lock().unwrap();
        counts.gets += 1;
        counts.hits += usize::from(found.is_some());
        found
    }

    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
        self.entries
            .lock()
            .unwrap()
            .insert((namespace.to_owned(), key.clone()), bytes.to_vec());
        self.counts.lock().unwrap().puts += 1;
    }
}

/// The available devices of `backends`, opened from fresh `Devices` with
/// `store`, so nothing formed by an earlier open is reused in this process.
fn devices_of(backends: &[BackendName], store: &Arc<RecordingStore>) -> Vec<Device> {
    seismic_native_tests::devices_with(
        backends,
        DeviceOptions {
            artifacts: Some(store.clone()), graph_cost: None
        },
    )
}

/// Form and run the defaults of `split_sum` at N = 1000.
fn form_and_run(device: &Device) {
    let n = 1000u64;
    let implementation = split_sum::native_implementation(device)
        .expect("bundle")
        .expect("split_sum has an implementation on every backend");
    let defaults = implementation
        .default_specialization(&NativeSpecialization::new().with_static("N", n))
        .expect("statics");
    let kernel = split_sum::native_for_device(device, &defaults)
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
    let bytes = (0..n)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect::<Vec<_>>();
    let x = Tensor::from_host(device, Element::f32(), &[n], &bytes).expect("host tensor");
    let sum = kernel.call(split_sum::Args { x: &x }).expect("call").value;
    let sum = f32::from_le_bytes(sum.read_to_host().expect("read")[..4].try_into().unwrap());
    assert_eq!(sum, n as f32, "{:?}", device.backend());
}

/// What forming `split_sum` on fresh devices of `backend` over `store`
/// did: the programs compiled and the platform-cache loads, summed over the
/// devices; none when no device of `backend` is available.
fn formed(backend: BackendName, store: &Arc<RecordingStore>) -> Option<(u64, u64)> {
    let devices = devices_of(&[backend], store);
    devices.iter().for_each(form_and_run);
    (!devices.is_empty()).then(|| {
        devices.iter().fold((0, 0), |(compiles, loads), device| {
            let formed = device.programs_formed();
            (compiles + formed.compiles, loads + formed.platform_loads)
        })
    })
}

/// The stored entries that are pointers: the artifact key each names.
fn pointers(stored: &HashMap<(String, ArtifactKey), Vec<u8>>) -> Vec<(String, String)> {
    stored
        .iter()
        .filter_map(|((namespace, _), bytes)| {
            let text = std::str::from_utf8(bytes).ok()?;
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
    // store: nothing compiles and nothing is written. Only Metal's loads
    // form their programs again through the platform's cache.
    let (compiled, loads) = formed(backend, &store).expect("the device opens again");
    assert_eq!(compiled, 0, "{backend:?}");
    assert_eq!(loads > 0, backend == BackendName::Metal, "{backend:?}");
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
