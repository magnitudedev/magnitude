//! Physical pressure uses the same topology and observations as admission.
use super::*;
use magnitude_executor::platform::{observe_domains, select_device};
use magnitude_executor::memory::ledger::own_lag;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use seismic::{Device, DeviceTopology, Element, StoreId, StoreIdentity, StoreKind, Tensor};

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum PressureStore { Host, Device }

impl PressureStore {
    pub fn name(self) -> &'static str {
        match self { Self::Host => "host", Self::Device => "device" }
    }
}

/// Several access paths may name one store; spanning stores is not targeted pressure.
fn unique_store<T: Copy + Eq>(stores: impl IntoIterator<Item = T>) -> Result<T, String> {
    let mut stores = stores.into_iter();
    let first = stores.next().ok_or("device storage has no physical placement")?;
    if stores.any(|store| store != first) {
        return Err("device allocation spans distinct physical stores; targeted device pressure is unsupported".into());
    }
    Ok(first)
}

/// The physical store `selected` names of `executor`'s in `topology`: host
/// RAM, or the one store its device-only storage lies in.
fn pressed(topology: &DeviceTopology, executor: &seismic::Executor, selected: PressureStore) -> Result<StoreId, String> {
    match selected {
        PressureStore::Host => Ok(topology.host_ram().id()),
        PressureStore::Device => unique_store(executor.paths(seismic::Contract::DeviceOnly).iter().map(|path|
            topology.access_path(*path).expect("placement belongs to topology").store())),
    }
}

/// The store `selected` names of `executor`'s, as the owner's memory
/// readings name it (`store`) and [`Target::snapshot`] does (`name`).
pub fn store_name(devices: &Devices, executor: &seismic::Executor, selected: PressureStore) -> Result<String, String> {
    let topology = devices.topology();
    let store = pressed(&topology, executor, selected)?;
    Ok(format!("{:?}", topology.store_identity(store).expect("placement store has identity")))
}

/// The least allocation a press on a device's own memory takes
/// ([`Target::press`]): a megabyte, below a bank's least block.
const PRESS_LEAST: u64 = 1 << 20;

pub struct Target {
    devices: Devices,
    device: Device,
    store: StoreId,
    /// When the latest holding [`Self::take`] made was in place.
    taken: Option<Instant>,
}

impl Target {
    pub fn open(backend: BackendName, selected: PressureStore) -> Result<Self, String> {
        let devices = Devices::discover(&[]).map_err(|error| error.to_string())?;
        let chosen = select_device(&devices, ExecutionPath::Native, DeviceRequest::Backend(backend), &ReleaseRate::of_machine(None))
            .map_err(|error| error.to_string())?;
        let id = devices.resolve(chosen.info.selector()).map_err(|error| error.to_string())?;
        // Opened as a load opens its device.
        let device = devices.open(id).map_err(|error| error.to_string())?;
        let store = pressed(device.topology(), device.info(), selected)?;
        Ok(Self { devices, device, store, taken: None })
    }

    pub fn reading(&self) -> Result<magnitude_executor::platform::DomainReading, String> {
        observe_domains(&self.devices, &self.device)
            .map_err(|error| error.to_string())?.into_iter().find(|reading| reading.domain == self.store)
            .ok_or_else(|| "selected pressure store has no admission observation".into())
    }

    /// What the store gives once it shows this process's own holdings, the
    /// headroom the occupier sizes its holding from: read at or past the
    /// store's lag for a growth after the latest [`Self::take`], as the
    /// engine's ledger judges its own changes shown
    /// ([`own_lag`]). Others' figures stand as read, as the engine reads them.
    pub fn settled_headroom(&self) -> Result<u64, String> {
        if let Some(taken) = self.taken {
            let store = self.device.topology().store_identity(self.store).expect("selected store has identity");
            let lag = own_lag(&store, &self.devices.device_status(&self.device).sample.host).growth;
            std::thread::sleep((taken + lag).saturating_duration_since(Instant::now()));
        }
        self.reading().map(|reading| reading.headroom_bytes)
    }

    pub fn snapshot(&self) -> Result<Value, String> {
        let reading = self.reading()?;
        // `name`: as the owner's memory readings name the store.
        Ok(json!({ "store": reading.identity, "name": format!("{:?}", reading.identity),
            "headroom_bytes": reading.headroom_bytes }))
    }

    /// The press where the store gives `headroom` and its bank shrinks by
    /// `block` ([`super::occupy`]), with the bytes it holds: in host RAM
    /// everything the store gives and the block past it, which the OS finds
    /// by displacing a working set; in a device's own memory everything its
    /// driver admits, which refuses any byte past what is free, so the
    /// press takes allocations until the driver refuses the least one
    /// ([`PRESS_LEAST`]), whatever the reading's figure (an RTX 3090's
    /// headroom read 1.75 MiB above what its driver then allocated).
    pub fn press(&mut self, headroom: u64, block: u64, seed: usize) -> Result<(Held, u64), String> {
        if self.device.topology().store(self.store).expect("selected store").kind() == &StoreKind::HostRam {
            let bytes = headroom + block;
            return self.take(bytes, seed).map(|held| (held, bytes));
        }
        let mut tensors = Vec::new();
        let (mut taken, mut chunk) = (0u64, self.largest()?.min(headroom.max(PRESS_LEAST)));
        while chunk >= PRESS_LEAST {
            match Tensor::zeros(&self.device, Element::bool(), &[chunk]) {
                Ok(tensor) => {
                    tensors.push(tensor);
                    taken += chunk;
                }
                Err(seismic::TensorError::Execution(error)) if error.capacity_refusal().is_some() => chunk /= 2,
                Err(error) => return Err(error.to_string()),
            }
        }
        self.taken = Some(Instant::now());
        Ok((Held::Device { _tensors: tensors }, taken))
    }

    /// The device's largest single allocation.
    fn largest(&self) -> Result<u64, String> {
        match self.device.info().limits().max_allocation.bytes() {
            0 => Err("the device admits no allocation".into()),
            largest => Ok(largest),
        }
    }

    /// `bytes` of the selected store, host RAM's written from `seed` ([`written`]).
    pub fn take(&mut self, bytes: u64, seed: usize) -> Result<Held, String> {
        let held = self.hold(bytes, seed)?;
        self.taken = Some(Instant::now());
        Ok(held)
    }

    fn hold(&self, bytes: u64, seed: usize) -> Result<Held, String> {
        if self.device.topology().store(self.store).expect("selected store").kind() == &StoreKind::HostRam {
            Ok(Held::Host { _words: Hot::keep(written(bytes, seed)?)? })
        } else {
            // Ordinary backed storage, not a reservation or a virtual allocation.
            // Its entire placement was proved to resolve to this selected store.
            // No allocation exceeds the device's largest single allocation.
            let largest = self.largest()?;
            let mut tensors = Vec::with_capacity(usize::try_from(bytes.div_ceil(largest)).map_err(|error| error.to_string())?);
            let mut left = bytes;
            while left > 0 {
                let chunk = left.min(largest);
                tensors.push(Tensor::zeros(&self.device, Element::bool(), &[chunk]).map_err(|error| error.to_string())?);
                left -= chunk;
            }
            Ok(Held::Device { _tensors: tensors })
        }
    }
}

// Fields own the occupied memory until the protocol releases it.
pub enum Held { Host { _words: Hot }, Device { _tensors: Vec<Tensor> } }

/// The words of a 4 KiB page: the smallest page any host maps, so reading
/// one word of each reaches every page of a holding on every host.
const PAGE_WORDS: usize = 4096 / std::mem::size_of::<u64>();

/// Host words kept in use, as an application's working set is: one thread
/// in the class that yields every processor to other work ([`yielding`])
/// reads a word of each page in turn until the holding is dropped. The OS
/// then reclaims none of them without the holder faulting it back, which
/// the engine reads as others' harm; untouched, a swap-backed host pages
/// them out at no one's cost and its free memory comes back, so no press
/// stands. The thread stops before the words are freed.
pub struct Hot {
    words: Arc<Vec<u64>>,
    stop: Arc<AtomicBool>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl Hot {
    fn keep(words: Vec<u64>) -> Result<Self, String> {
        let words = Arc::new(words);
        let stop = Arc::new(AtomicBool::new(false));
        let (classed, class) = std::sync::mpsc::sync_channel(1);
        let reader = std::thread::spawn({
            let (words, stop) = (words.clone(), stop.clone());
            move || {
                let yields = yielding();
                let refused = yields.is_err();
                let _ = classed.send(yields);
                // An empty holding has no page to keep.
                if refused || words.is_empty() {
                    return;
                }
                while !stop.load(Ordering::Relaxed) {
                    for page in words.chunks(PAGE_WORDS) {
                        std::hint::black_box(page[0]);
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                    }
                }
            }
        });
        let held = Self { words, stop, reader: Some(reader) };
        class.recv().map_err(|_| "the reading thread ended before taking its class".to_owned())??;
        Ok(held)
    }
}

impl Drop for Hot {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(reader) = self.reader.take() {
            reader.join().expect("the reading thread panicked");
        }
        debug_assert_eq!(Arc::strong_count(&self.words), 1, "the reader released its words");
    }
}

#[cfg(test)]
mod tests {
    use super::{Hot, unique_store};

    /// A hot holding keeps its words as written while its reader runs, and
    /// dropping it stops the reader before the words go; an empty one keeps
    /// nothing and stops as well.
    #[test]
    fn a_hot_holding_keeps_its_words_and_stops_its_reader_on_drop() {
        let words: Vec<u64> = (0..3 * 512 + 7).collect();
        let hot = Hot::keep(words.clone()).unwrap();
        assert_eq!(*hot.words, words);
        drop(hot);
        drop(Hot::keep(Vec::new()).unwrap());
    }

    #[test]
    fn physical_placement_accepts_aliases_but_never_spanning_stores() {
        assert_eq!(unique_store([3, 3, 3]).unwrap(), 3);
        assert!(unique_store([3, 4]).is_err());
        assert!(unique_store::<u8>([]).is_err());
    }
}

pub fn holding_stores(devices: &Devices, executor: &seismic::Executor, kind: seismic::Contract)
    -> Vec<StoreIdentity>
{
    let topology = devices.topology();
    let mut stores = Vec::new();
    for path in executor.paths(kind) {
        let store = topology.access_path(*path).expect("placement belongs to topology").store();
        let identity = topology.store_identity(store).expect("placement store has identity");
        if !stores.contains(&identity) { stores.push(identity); }
    }
    stores
}
