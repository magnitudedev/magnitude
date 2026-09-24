//! The embedder's artifact store around native formation (tuning spec §C1):
//! CUDA images are looked up before NVRTC runs and kept after it; a stored
//! image the driver refuses is a miss and is formed and stored again. Metal
//! and CPU formation never consult the store.

use seismic::{
    ArtifactKey, ArtifactKind, ArtifactStore, Availability, BackendName, Device, DeviceCatalog,
    DeviceOptions, Element, NativeSpecialization, Tensor,
};
use seismic_native_tests::split_sum;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// An in-memory store that counts lookups, hits and writes.
#[derive(Default)]
struct RecordingStore {
    entries: Mutex<HashMap<(ArtifactKind, ArtifactKey), Vec<u8>>>,
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

    /// Replace every stored image with bytes no driver loads.
    fn corrupt(&self) {
        for bytes in self.entries.lock().unwrap().values_mut() {
            *bytes = b"not a cubin".to_vec();
        }
    }
}

impl ArtifactStore for RecordingStore {
    fn get(&self, kind: ArtifactKind, key: &ArtifactKey) -> Option<Vec<u8>> {
        let found = self.entries.lock().unwrap().get(&(kind, key.clone())).cloned();
        let mut counts = self.counts.lock().unwrap();
        counts.gets += 1;
        counts.hits += usize::from(found.is_some());
        found
    }

    fn put(&self, kind: ArtifactKind, key: &ArtifactKey, bytes: &[u8]) {
        self.entries
            .lock()
            .unwrap()
            .insert((kind, key.clone()), bytes.to_vec());
        self.counts.lock().unwrap().puts += 1;
    }
}

/// Every available backend, opened from a fresh catalog with `store`, so
/// nothing formed by an earlier open is reused in this process.
fn devices(store: &Arc<RecordingStore>) -> Vec<Device> {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let topology = catalog.topology();
    [BackendName::Cpu, BackendName::Metal, BackendName::Cuda]
        .into_iter()
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
    let bytes = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect::<Vec<_>>();
    let x = Tensor::from_host(device, Element::f32(), &[n], &bytes).expect("host tensor");
    let sum = kernel.call(split_sum::Args { x: &x }).expect("call").value;
    let sum = f32::from_le_bytes(sum.read_to_host().expect("read")[..4].try_into().unwrap());
    assert_eq!(sum, n as f32, "{:?}", device.backend());
}

#[test]
fn cuda_images_are_kept_in_the_embedders_store_and_a_refused_image_is_a_miss() {
    let store = Arc::new(RecordingStore::default());
    let backends = devices(&store).iter().map(Device::backend).collect::<Vec<_>>();
    let cuda = backends.contains(&BackendName::Cuda);

    // First formation: a miss, formed by NVRTC and stored.
    for device in devices(&store) {
        form_and_run(&device);
    }
    let first = store.counts();
    if !cuda {
        assert_eq!(first, Counts::default(), "only CUDA formation uses the store");
        return;
    }
    assert_eq!(first, Counts { gets: 1, hits: 0, puts: 1 });

    // A fresh device forms the same source and formation: the stored image
    // is loaded and NVRTC does not run (nothing is stored again).
    for device in devices(&store) {
        form_and_run(&device);
    }
    assert_eq!(store.counts(), Counts { gets: 2, hits: 1, puts: 1 });

    // A stored image the driver refuses is a miss: formed and stored again.
    store.corrupt();
    for device in devices(&store) {
        form_and_run(&device);
    }
    assert_eq!(store.counts(), Counts { gets: 3, hits: 2, puts: 2 });
}
