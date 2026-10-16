//! The native kernels the supported catalog models request on one backend, as assessment
//! inventories them.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, anyhow};
use magnitude_engine::assessment::{KernelInventory, kernel_inventory};
use magnitude_engine::options::standard_service_limits;
use magnitude_service_contracts::models::CatalogSupport;
use magnitude_service_models::{load_release_catalog, servable_model_bundle_key_for_bundle};
use seismic::{BackendName, KernelRequest};

use super::assessor::engine_material;
use super::environment::serving_policy;

/// One distinct request and the first supported model, as `model:variant`, that makes it.
pub struct CatalogKernelRequest {
    pub request: KernelRequest,
    pub model: String,
}

pub struct CatalogKernelRequests {
    pub requests: Vec<CatalogKernelRequest>,
    /// Supported models assessment finds unsupported on the backend, each with why.
    pub unsupported: Vec<(String, String)>,
}

/// Every distinct kernel request of every supported model in the release catalog of
/// `planner_bundle`, on `backend`.
pub fn catalog_kernel_requests(
    planner_bundle: &Path,
    backend: BackendName,
) -> anyhow::Result<CatalogKernelRequests> {
    let catalog = load_release_catalog(planner_bundle)
        .map_err(|error| anyhow!("{error:?}"))
        .with_context(|| format!("loading the release catalog {}", planner_bundle.display()))?;
    let mut requests = Vec::new();
    let mut unsupported = Vec::new();
    let mut seen = HashSet::new();
    for model in &catalog.catalog().models {
        if !matches!(model.support, CatalogSupport::Supported) {
            continue;
        }
        let name = format!("{}:{}", model.model_id, model.variant_id);
        let key = servable_model_bundle_key_for_bundle(&model.configuration.bundle);
        let resolved = catalog
            .resolve_bundle(&key)
            .map_err(|error| anyhow!("`{name}`: {error:?}"))?
            .ok_or_else(|| anyhow!("`{name}` has no planner inputs"))?;
        let package = engine_material(&resolved).map_err(|error| anyhow!("`{name}`: {error:?}"))?;
        let inventory = kernel_inventory(
            &package,
            &serving_policy(),
            &standard_service_limits(),
            backend,
        )
        .map_err(|error| anyhow!("`{name}` on {}: {error}", backend.as_str()))?;
        match inventory {
            KernelInventory::Requests(model_requests) => {
                for request in model_requests {
                    if seen.insert(request.clone()) {
                        requests.push(CatalogKernelRequest {
                            request,
                            model: name.clone(),
                        });
                    }
                }
            }
            KernelInventory::Unsupported(reason) => unsupported.push((name, format!("{reason:?}"))),
        }
    }
    Ok(CatalogKernelRequests {
        requests,
        unsupported,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A supported catalog variant the engine refuses on any backend (weight types it can't
    /// import, such as the `UD-Q2_K_XL` variants' Q2_K tensors) is a release defect: it must leave
    /// the catalog until the engine runs it.
    #[test]
    #[ignore = "requires MAGNITUDE_CATALOG_BUNDLE, the release planner bundle (bun run icn:catalog:build-bundle)"]
    fn every_supported_catalog_variant_runs_on_every_backend() {
        let bundle = std::path::PathBuf::from(
            std::env::var_os("MAGNITUDE_CATALOG_BUNDLE").expect("set MAGNITUDE_CATALOG_BUNDLE"),
        );
        let unsupported = BackendName::ALL
            .into_iter()
            .flat_map(|backend| {
                catalog_kernel_requests(&bundle, backend)
                    .unwrap_or_else(|error| panic!("{}: {error:#}", backend.as_str()))
                    .unsupported
                    .into_iter()
                    .map(move |(model, reason)| format!("{} {model}: {reason}", backend.as_str()))
            })
            .collect::<Vec<_>>();
        assert!(unsupported.is_empty(), "{unsupported:#?}");
    }
}
