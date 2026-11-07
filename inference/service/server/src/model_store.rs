//! The model store and the machine it serves, opened together: the store's served paths exist
//! once it is open, and the devices are discovered with them, so the topology holds the volumes
//! downloads write and loads read.

use std::sync::Arc;

use anyhow::Context;
use magnitude_executor::platform::VolumeReserves;
use magnitude_service_models::{InventoryConfig, ManagedModelStore, ModelCache};
use seismic::Devices;

/// An open model store and the devices discovered with the paths it serves.
pub struct ServedStore {
    pub store: Arc<ManagedModelStore>,
    pub devices: Arc<Devices>,
    /// The reserves of the volumes the store writes: its downloads' space and the hardware
    /// report read the one reserve of each.
    pub volumes: Arc<VolumeReserves>,
}

/// Open the store of `config`, its downloads held to the space its volume leaves above its
/// reserve ([`VolumeReserves::space`], its prior the volume's write rate as the kernel cache
/// keeps it),
/// then discover the devices with the paths it serves ([`ManagedModelStore::served_paths`]).
pub async fn open_served_store(config: InventoryConfig) -> anyhow::Result<ServedStore> {
    let kernels = ModelCache::new(&config.cache_root).kernel_directory();
    let volumes = Arc::new(VolumeReserves::new(Some(kernels)));
    let space = Arc::new(Arc::clone(&volumes).space(config.root.clone()));
    let store = Arc::new(
        ManagedModelStore::open(config, space)
            .await
            .context("failed to initialize model inventory")?,
    );
    let served = store.served_paths();
    let devices = magnitude_service_utils::blocking::run(move || Devices::discover(&served))
        .await
        .context("device discovery failed")?;
    Ok(ServedStore {
        store,
        devices: Arc::new(devices),
        volumes,
    })
}
