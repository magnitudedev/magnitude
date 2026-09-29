//! The embedder's artifact store around native formation: CUDA keeps CUBINs
//! and Vulkan keeps SPIR-V, looked up before the toolchain runs and kept
//! after; a damaged entry is a miss and is compiled and stored again. Metal
//! (whose compiles the OS caches) and CPU (compiled into the binary) keep
//! nothing.

use seismic::{
    ArtifactKey, ArtifactStore, Availability, BackendName, Device, DeviceCatalog, DeviceOptions,
    Element, NativeSpecialization, Tensor,
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

/// The available devices of `backends`, opened from a fresh catalog with
/// `store`, so nothing formed by an earlier open is reused in this process.
fn devices_of(backends: &[BackendName], store: &Arc<RecordingStore>) -> Vec<Device> {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let topology = catalog.topology();
    backends
        .iter()
        .copied()
        .filter_map(|backend| {
            topology
                .devices()
                .iter()
                .find(|device| {
                    device.backend == backend
                        && matches!(device.availability, Availability::Available)
                })
                .map(|device| device.id)
        })
        .map(|id| {
            catalog
                .open_with(
                    id,
                    DeviceOptions {
                        artifacts: Some(store.clone()),
                    },
                )
                .expect("available device opens")
        })
        .collect()
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

/// Form `split_sum` on a fresh device of `backend` three times: into an
/// empty store, from the store, and after the store's entries are damaged.
fn formation_uses_the_store(backend: BackendName) {
    let store = Arc::new(RecordingStore::default());
    let formed = |store: &Arc<RecordingStore>| {
        let devices = devices_of(&[backend], store);
        devices.iter().for_each(form_and_run);
        !devices.is_empty()
    };
    if !formed(&store) {
        return;
    }
    let first = store.counts();
    if matches!(backend, BackendName::Cpu | BackendName::Metal) {
        assert_eq!(first, Counts::default(), "{backend:?} keeps nothing");
        return;
    }
    // Every program is a miss, compiled and stored.
    let programs = first.gets;
    assert!(programs > 0);
    assert_eq!(
        first,
        Counts {
            gets: programs,
            hits: 0,
            puts: programs
        }
    );

    // A fresh device forms the same programs from the store: nothing compiles.
    formed(&store);
    assert_eq!(
        store.counts(),
        Counts {
            gets: 2 * programs,
            hits: programs,
            puts: programs
        }
    );

    // A damaged entry is a miss: compiled and stored again.
    store.corrupt();
    formed(&store);
    assert_eq!(
        store.counts(),
        Counts {
            gets: 3 * programs,
            hits: 2 * programs,
            puts: 2 * programs
        }
    );
}

#[test]
fn cpu_formation_never_uses_the_store() {
    formation_uses_the_store(BackendName::Cpu);
}

#[test]
fn metal_formation_never_uses_the_store() {
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
