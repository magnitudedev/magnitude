//! The embedder's artifact store around native formation: CUDA keeps CUBINs
//! and Vulkan keeps SPIR-V, looked up before the toolchain runs and kept
//! after; a damaged entry is a miss and is compiled and stored again. Metal,
//! whose compiles the OS caches where no toolchain reads them, keeps no
//! program: its formation records each entry list it compiled of a source
//! text, so a later process knows that request was compiled before. CPU
//! (compiled into the binary) keeps nothing.

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
    if backend == BackendName::Cpu {
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

/// Metal stores no compiled program: the OS caches what it compiles, where
/// the toolchain can neither read nor query it. What formation keeps in the
/// store is that an entry list of a source text was compiled, in two
/// records of one line each: the toolchain's, under the text and the entry
/// list (what makes a later request for exactly that list one compiled
/// before), and the former's list of the entry lists kept of the text (how
/// a request for some of a kept list's entries is found). A fresh device
/// over the same store, as a later process is, finds its request recorded:
/// it is formed as kept, and nothing is written.
#[test]
fn metal_formation_records_each_compiled_entry_list_once_and_stores_no_program() {
    let store = Arc::new(RecordingStore::default());
    let formed = |store: &Arc<RecordingStore>| {
        let devices = devices_of(&[BackendName::Metal], store);
        devices.iter().for_each(form_and_run);
        !devices.is_empty()
    };
    if !formed(&store) {
        return;
    }
    // Into an empty store, each compile reads four times and finds nothing
    // (the former: the record of exactly its entry list, then the text's
    // entry lists; the toolchain: its record, to add to; the former: the
    // text's entry lists, to add to) and writes its two records.
    let first = store.counts();
    let compiles = first.puts / 2;
    assert!(compiles > 0);
    assert_eq!(
        first,
        Counts {
            gets: 4 * compiles,
            hits: 0,
            puts: 2 * compiles
        }
    );
    // What is stored is those records and nothing else: each is the text of
    // one line, the entries of the one list compiled, so no stored byte is
    // of a compiled program. The store's keys do not say which record is
    // which; their lines do (the former's give each entry's group size and
    // constants after `|`, the toolchain's a group size after `@`).
    let kept = store.stored();
    assert_eq!(kept.len(), 2 * compiles);
    let (mut recorded, mut listed) = (Vec::new(), Vec::new());
    for ((namespace, _), bytes) in &kept {
        assert_eq!(namespace, "metal");
        let text = std::str::from_utf8(bytes).expect("a record is text, not a compiled program");
        let line = text.strip_suffix('\n').expect("a record ends its line");
        assert!(
            !line.is_empty() && !line.contains('\n'),
            "one entry list, recorded once: {text:?}"
        );
        let symbols = |mark: char| {
            line.split('\t')
                .map(|entry| entry.split(mark).next().unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        if line.contains('|') {
            listed.push(symbols('|'));
        } else {
            recorded.push(symbols('@'));
        }
    }
    recorded.sort();
    listed.sort();
    assert_eq!(recorded.len(), compiles);
    assert_eq!(recorded, listed, "both records name the entries compiled");
    assert!(recorded.iter().all(|entries| !entries.is_empty()));

    // A fresh device forms the same request: the former finds the record of
    // exactly its entry list, which names it, so the formation is one kept
    // (it looks no further), and the toolchain finds its line recorded.
    // Nothing is written, and the store holds what it held.
    assert!(formed(&store));
    assert_eq!(
        store.counts(),
        Counts {
            gets: 6 * compiles,
            hits: 2 * compiles,
            puts: 2 * compiles
        }
    );
    assert_eq!(store.stored(), kept);
}

#[test]
fn cuda_programs_are_kept_in_the_embedders_store() {
    formation_uses_the_store(BackendName::Cuda);
}

#[test]
fn vulkan_programs_are_kept_in_the_embedders_store() {
    formation_uses_the_store(BackendName::Vulkan);
}
