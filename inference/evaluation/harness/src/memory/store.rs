//! Physical pressure uses the same topology and observations as admission.
use super::*;
use seismic::{Device, StoreId, StoreIdentity, StoreKind};

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

pub struct Target {
    devices: Devices,
    device: Device,
    store: StoreId,
    /// Host RAM's room as the pressed load reads it while it serves
    /// (`HostRoom::Load`), its free headroom what the host can give
    /// without displacing more of other programs' data than it held
    /// displaced as this target opened, before the approach, the load
    /// serving already. Read as allocatable alone, RAM the host freed by
    /// swapping under the approach is room the load does not count, and a
    /// press sized from it lands deeper below the load's line than its
    /// deficit.
    room: HostRoom,
}

impl Target {
    pub fn open(backend: BackendName, selected: PressureStore) -> Result<Self, String> {
        let devices = Devices::discover(&[]).map_err(|error| error.to_string())?;
        let chosen = select_device(&devices, ExecutionPath::Native, DeviceRequest::Backend(backend), &MemoryReserves::of_machine(None))
            .map_err(|error| error.to_string())?;
        let id = devices.resolve(chosen.info.selector()).map_err(|error| error.to_string())?;
        let device = devices.open(id).map_err(|error| error.to_string())?;
        let topology = device.topology();
        let store = match selected {
            PressureStore::Host => topology.host_ram().id(),
            PressureStore::Device => unique_store(device.info().paths(seismic::Placing::Fixed(seismic::Contract::DeviceOnly)).iter().map(|path|
                topology.access_path(*path).expect("placement belongs to topology").store()))?,
        };
        let host = devices.status(topology, device.info().id()).sample.host_ram.map_err(|error| error.to_string())?;
        let room = HostRoom::serving(&host, std::time::Instant::now());
        Ok(Self { devices, device, store, room })
    }

    pub fn reading(&self) -> Result<magnitude_executor::platform::DomainReading, String> {
        observe_domains(&self.devices, &self.device, &MemoryReserves::of_machine(None), self.room)
            .map_err(|error| error.to_string())?.into_iter().find(|reading| reading.domain == self.store)
            .ok_or_else(|| "selected pressure store has no admission observation".into())
    }

    pub fn snapshot(&self) -> Result<Value, String> {
        let reading = self.reading()?;
        // `name`: as the owner's memory readings name the store.
        Ok(json!({ "store": reading.identity, "name": format!("{:?}", reading.identity), "headroom_bytes": reading.headroom_bytes,
            "ceiling_bytes": reading.ceiling_bytes, "planning_bytes": reading.thresholds.planning_bytes,
            "band": format!("{:?}", reading.band) }))
    }

    /// `bytes` of the selected store, host RAM's written from `seed` ([`written`]).
    pub fn take(&self, bytes: u64, seed: usize) -> Result<Held, String> {
        if self.device.topology().store(self.store).expect("selected store").kind() == &StoreKind::HostRam {
            Ok(Held::Host { _words: written(bytes, seed)? })
        } else {
            // Ordinary backed storage, not a reservation or a virtual allocation.
            // Its entire placement was proved to resolve to this selected store.
            // No allocation exceeds the device's largest single allocation.
            let largest = self.device.info().limits().max_allocation.bytes();
            if largest == 0 {
                return Err("the device admits no allocation".into());
            }
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
pub enum Held { Host { _words: Vec<u64> }, Device { _tensors: Vec<Tensor> } }

#[cfg(test)]
mod tests {
    use super::unique_store;
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
    for path in executor.paths(seismic::Placing::Fixed(kind)) {
        let store = topology.access_path(*path).expect("placement belongs to topology").store();
        let identity = topology.store_identity(store).expect("placement store has identity");
        if !stores.contains(&identity) { stores.push(identity); }
    }
    stores
}
