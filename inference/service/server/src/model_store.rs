//! The model store and the machine it serves, opened together: the store's served paths exist
//! once it is open, and the devices are discovered with them, so the topology holds the volumes
//! downloads write and loads read.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use magnitude_executor::platform::VolumeReserves;
use magnitude_service_contracts::HardwareVolumeRole;
use magnitude_service_models::{InventoryConfig, ManagedModelStore, ModelCache};
use seismic::{Devices, VolumeId};

/// An open model store and the devices discovered with the paths it serves.
pub struct ServedStore {
    pub store: Arc<ManagedModelStore>,
    pub devices: Arc<Devices>,
    pub volumes: ServedVolumes,
}

/// The volumes the service keeps its paths on: the reserve of each, which downloads' space and
/// the hardware report read, and what the service keeps on each.
pub struct ServedVolumes {
    pub reserves: Arc<VolumeReserves>,
    roles: BTreeMap<VolumeId, Vec<HardwareVolumeRole>>,
}

impl ServedVolumes {
    pub fn new(reserves: Arc<VolumeReserves>) -> Self {
        Self {
            reserves,
            roles: BTreeMap::new(),
        }
    }

    /// The volume holding the existing directory `path` keeps `role`.
    pub fn tag(&mut self, path: &Path, role: HardwareVolumeRole) -> std::io::Result<()> {
        let roles = self.roles.entry(seismic::volume_identity(path)?).or_default();
        if !roles.contains(&role) {
            roles.push(role);
            roles.sort();
        }
        Ok(())
    }

    /// What the service keeps on `volume`; empty for a volume it keeps nothing on.
    pub fn roles(&self, volume: &VolumeId) -> Vec<HardwareVolumeRole> {
        self.roles.get(volume).cloned().unwrap_or_default()
    }
}

/// Open the store of `config`, its downloads held to the space its volume leaves above its
/// reserve ([`VolumeReserves::space`], its prior the volume's write rate as the kernel cache
/// keeps it),
/// then discover the devices with the paths it serves ([`ManagedModelStore::served_paths`]),
/// the model store's volume and the kernel cache's tagged with their roles.
pub async fn open_served_store(config: InventoryConfig) -> anyhow::Result<ServedStore> {
    let kernels = ModelCache::new(&config.cache_root).kernel_directory();
    let reserves = Arc::new(VolumeReserves::new(Some(kernels)));
    let space = Arc::new(Arc::clone(&reserves).space(config.root.clone()));
    let (model_store, kernel_cache) = (config.root.clone(), config.cache_root.clone());
    let store = Arc::new(
        ManagedModelStore::open(config, space)
            .await
            .context("failed to initialize model inventory")?,
    );
    let served = store.served_paths();
    let devices = magnitude_service_utils::blocking::run(move || Devices::discover(&served))
        .await
        .context("device discovery failed")?;
    let volumes = magnitude_service_utils::blocking::run(move || {
        let mut volumes = ServedVolumes::new(reserves);
        volumes.tag(&model_store, HardwareVolumeRole::ModelStore)?;
        volumes.tag(&kernel_cache, HardwareVolumeRole::KernelCache)?;
        Ok::<_, std::io::Error>(volumes)
    })
    .await
    .context("failed to identify the served volumes")?;
    Ok(ServedStore {
        store,
        devices: Arc::new(devices),
        volumes,
    })
}
