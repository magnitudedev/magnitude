use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

use futures_util::stream::{BoxStream, StreamExt};
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    CatalogModelOptimizer, CatalogPackageRemover, ModelDomainInvalidation,
    RecommendableModelCatalog,
};

use crate::ManagedModelDownloads;
use crate::catalog_installations::ManagedCatalogInstallations;
use crate::catalog_models::ManagedCatalogModels;
use crate::custom_models::{LocalModelsContext, ManagedLocalModels};
use crate::inventory::{InstalledPackagesObserver, ManagedModelStore};
use crate::upstream_updates::UpstreamUpdates;

pub struct ModelDomainResolver {
    pub(crate) inventory: Arc<ManagedModelStore>,
    pub(crate) catalog: RecommendableModelCatalog,
    pub(crate) local: LocalModelsContext,
    pub(crate) catalog_changes: CatalogChanges,
    pub(crate) upstream: UpstreamUpdates,
}

/// What republishes the catalog list: the inventory, and what it shows that the inventory doesn't
/// hold (which catalog models serve without their drafter), counted by `generation`.
pub(crate) struct CatalogChanges {
    pub(crate) generation: AtomicU64,
    pub(crate) changes: tokio::sync::watch::Sender<u64>,
}

impl ModelDomainResolver {
    #[must_use]
    pub fn new(inventory: Arc<ManagedModelStore>, catalog: RecommendableModelCatalog) -> Arc<Self> {
        Arc::new(Self {
            inventory,
            catalog,
            local: LocalModelsContext::new(),
            catalog_changes: CatalogChanges {
                generation: AtomicU64::new(0),
                changes: tokio::sync::watch::channel(0).0,
            },
            upstream: UpstreamUpdates::new(),
        })
    }
}

struct ModelInventoryObserver {
    catalog: Weak<ManagedCatalogModels>,
    local: Weak<ManagedLocalModels>,
}

impl InstalledPackagesObserver for ModelInventoryObserver {
    fn installed_packages_changed(&self, revision: u64) {
        if let Some(catalog) = self.catalog.upgrade() {
            catalog.installed_packages_changed(revision);
        }
        if let Some(local) = self.local.upgrade() {
            local.installed_packages_changed(revision);
        }
    }
}

pub struct ManagedModelServices {
    pub catalog: Arc<ManagedCatalogModels>,
    pub local: Arc<ManagedLocalModels>,
    pub installations: Arc<ManagedCatalogInstallations>,
}

pub fn managed_model_services(
    resolver: Arc<ModelDomainResolver>,
    downloads: Arc<ManagedModelDownloads>,
    remover: Arc<dyn CatalogPackageRemover>,
    optimizer: Arc<dyn CatalogModelOptimizer>,
) -> Result<ManagedModelServices, InventoryError> {
    let installations =
        ManagedCatalogInstallations::new(resolver.clone(), downloads, remover.clone(), optimizer);
    crate::upstream_updates::start(&resolver, remover.clone());
    let local = ManagedLocalModels::new(resolver.clone(), remover);
    let catalog = ManagedCatalogModels::new(resolver, installations.clone(), {
        let local = Arc::downgrade(&local);
        move |catalog| Arc::new(ModelInventoryObserver { catalog, local })
    });
    catalog
        .resolver()
        .inventory
        .set_installed_packages_observer(Arc::downgrade(catalog.inventory_observer()))?;
    Ok(ManagedModelServices {
        catalog,
        local,
        installations,
    })
}

pub(crate) fn domain_changes(
    initial_revision: u64,
    receiver: tokio::sync::watch::Receiver<u64>,
) -> BoxStream<'static, ModelDomainInvalidation> {
    let changes = futures_util::stream::unfold(receiver, |mut receiver| async move {
        receiver.changed().await.ok()?;
        let revision = *receiver.borrow_and_update();
        Some((ModelDomainInvalidation { revision }, receiver))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn domain_changes_coalesces_bursts_without_dropping_latest_revision() {
        let (tx, _rx) = tokio::sync::watch::channel(0u64);
        let mut stream = domain_changes(1, tx.subscribe());
        assert_eq!(stream.next().await.expect("initial").revision, 1);

        for rev in 2..=128 {
            tx.send_replace(rev);
        }

        let next = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
            .await
            .expect("coalesced wakeup arrived")
            .expect("stream open");
        assert_eq!(next.revision, 128);
    }
}
