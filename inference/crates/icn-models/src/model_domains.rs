use std::sync::{Arc, Weak};

use futures_util::stream::{BoxStream, StreamExt};
use icn_contracts::InventoryError;
use icn_contracts::models::{
    CatalogPackageRemover, ModelDomainInvalidation, RecommendableModelCatalog,
};

use crate::ManagedModelDownloads;
use crate::catalog_installations::ManagedCatalogInstallations;
use crate::catalog_models::ManagedCatalogModels;
use crate::discovered_models::ManagedDiscoveredModels;
use crate::inventory::{InstalledPackagesObserver, ManagedModelStore};

#[derive(Clone)]
pub struct ModelDomainResolver {
    pub(crate) inventory: Arc<ManagedModelStore>,
    pub(crate) catalog: RecommendableModelCatalog,
}

impl ModelDomainResolver {
    #[must_use]
    pub fn new(inventory: Arc<ManagedModelStore>, catalog: RecommendableModelCatalog) -> Arc<Self> {
        Arc::new(Self { inventory, catalog })
    }
}

struct ModelInventoryObserver {
    catalog: Weak<ManagedCatalogModels>,
    discovered: Weak<ManagedDiscoveredModels>,
}

impl InstalledPackagesObserver for ModelInventoryObserver {
    fn installed_packages_changed(&self, revision: u64) {
        if let Some(catalog) = self.catalog.upgrade() {
            catalog.installed_packages_changed(revision);
        }
        if let Some(discovered) = self.discovered.upgrade() {
            discovered.installed_packages_changed(revision);
        }
    }
}

pub struct ManagedModelServices {
    pub catalog: Arc<ManagedCatalogModels>,
    pub discovered: Arc<ManagedDiscoveredModels>,
    pub installations: Arc<ManagedCatalogInstallations>,
}

pub fn managed_model_services(
    resolver: Arc<ModelDomainResolver>,
    downloads: Arc<ManagedModelDownloads>,
    remover: Arc<dyn CatalogPackageRemover>,
) -> Result<ManagedModelServices, InventoryError> {
    let installations = ManagedCatalogInstallations::new(resolver.clone(), downloads, remover);
    let discovered = ManagedDiscoveredModels::new(resolver.clone());
    let catalog = ManagedCatalogModels::new(resolver, installations.clone(), {
        let discovered = Arc::downgrade(&discovered);
        move |catalog| {
            Arc::new(ModelInventoryObserver {
                catalog,
                discovered,
            })
        }
    });
    catalog
        .resolver()
        .inventory
        .set_installed_packages_observer(Arc::downgrade(catalog.inventory_observer()))?;
    Ok(ManagedModelServices {
        catalog,
        discovered,
        installations,
    })
}

pub(crate) fn domain_changes(
    initial_revision: u64,
    receiver: tokio::sync::broadcast::Receiver<ModelDomainInvalidation>,
) -> BoxStream<'static, ModelDomainInvalidation> {
    let changes = futures_util::stream::unfold(receiver, |mut receiver| async move {
        loop {
            match receiver.recv().await {
                Ok(event) => return Some((event, receiver)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Box::pin(
        futures_util::stream::once(async move {
            ModelDomainInvalidation {
                revision: initial_revision,
            }
        })
        .chain(changes),
    )
}
