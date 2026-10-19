//! The model store and the machine it serves, opened together: the store's served paths exist
//! once it is open, and the devices are discovered with them, so the topology holds the volumes
//! downloads write and loads read.

use std::sync::Arc;

use anyhow::Context;
use magnitude_executor::platform::{MemoryReserves, volume_space};
use magnitude_service_models::{InventoryConfig, ManagedModelStore};
use seismic::Devices;

/// An open model store and the devices discovered with the paths it serves.
pub struct ServedStore {
    pub store: Arc<ManagedModelStore>,
    pub devices: Arc<Devices>,
}

/// Open the store of `config`, its downloads held to the space its volume fits above `reserves`
/// (the engine's fit owner, [`volume_space`]), then discover the devices with the paths it
/// serves ([`ManagedModelStore::served_paths`]).
pub async fn open_served_store(
    config: InventoryConfig,
    reserves: MemoryReserves,
) -> anyhow::Result<ServedStore> {
    let space = Arc::new(volume_space(config.root.clone(), reserves));
    let store = Arc::new(
        ManagedModelStore::open(config, space)
            .await
            .context("failed to initialize model inventory")?,
    );
    let served = store.served_paths();
    let devices = crate::spawn_blocking_traced(move || Devices::discover(&served))
        .await
        .context("device discovery task failed")?
        .context("device discovery failed")?;
    Ok(ServedStore {
        store,
        devices: Arc::new(devices),
    })
}
