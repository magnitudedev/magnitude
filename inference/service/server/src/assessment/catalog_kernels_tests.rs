//! Every native kernel every supported catalog model requests on every GPU
//! backend assessment accepts it on is admissible, implemented and forms
//! with the backend's real toolchain.
//!
//! The catalog is the release header bundle: `MAGNITUDE_PLANNER_BUNDLE`, or
//! `inference/target/catalog-inputs/model-planner-inputs.bundle` as
//! `bun run icn:catalog:build-bundle` writes it. A missing bundle or a
//! backend toolchain this host lacks is skipped with a note;
//! `MAGNITUDE_REQUIRE_KERNEL_FORMATION=1` (set by CI) makes either a failure.

use super::assessor::engine_material;
use super::environment::serving_policy;
use magnitude_engine::assessment::{KernelInventory, kernel_inventory};
use magnitude_engine::options::standard_service_limits;
use magnitude_service_contracts::models::CatalogSupport;
use magnitude_service_models::{load_release_catalog, servable_model_bundle_key_for_bundle};
use seismic::coverage::{CoverageError, RequestFormer};
use seismic::{BackendName, KernelRequest};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

fn required() -> bool {
    std::env::var_os("MAGNITUDE_REQUIRE_KERNEL_FORMATION").is_some()
}

fn bundle() -> PathBuf {
    std::env::var_os("MAGNITUDE_PLANNER_BUNDLE").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/catalog-inputs/model-planner-inputs.bundle")
        },
        PathBuf::from,
    )
}

/// The requests of every supported catalog model on `backend`, each with the
/// first model that makes it.
fn requests(backend: BackendName) -> Vec<(KernelRequest, String)> {
    let catalog = load_release_catalog(&bundle()).expect("the release catalog loads");
    let mut requests = Vec::new();
    let mut seen = HashSet::new();
    for model in &catalog.catalog().models {
        if !matches!(model.support, CatalogSupport::Supported) {
            continue;
        }
        let name = format!("{}:{}", model.model_id, model.variant_id);
        let key = servable_model_bundle_key_for_bundle(&model.configuration.bundle);
        let resolved = catalog
            .resolve_bundle(&key)
            .expect("a catalog bundle resolves")
            .expect("every catalog bundle has planner inputs");
        let package = engine_material(&resolved).expect("a catalog bundle has engine material");
        let inventory = kernel_inventory(
            &package,
            &serving_policy(),
            &standard_service_limits(),
            backend,
        )
        .unwrap_or_else(|error| panic!("`{name}` on {}: {error}", backend.as_str()));
        let model_requests = match inventory {
            KernelInventory::Requests(requests) => requests,
            KernelInventory::Unsupported(unsupported) => {
                eprintln!(
                    "`{name}` is unsupported on {}: {unsupported:?}",
                    backend.as_str()
                );
                continue;
            }
        };
        for request in model_requests {
            if seen.insert(request.clone()) {
                requests.push((request, name.clone()));
            }
        }
    }
    requests
}

fn form(backend: BackendName) {
    if !bundle().exists() && !required() {
        eprintln!(
            "skipping: no release header bundle at {}",
            bundle().display()
        );
        return;
    }
    let former = match RequestFormer::open(backend) {
        Ok(former) => former,
        Err(CoverageError::ToolchainUnavailable(reason)) if !required() => {
            eprintln!("skipping {}: {reason}", backend.as_str());
            return;
        }
        Err(error) => panic!("{} requests cannot form: {error}", backend.as_str()),
    };
    let module = magnitude_kernels::module().expect("the engine kernel bundle decodes");
    let requests = requests(backend);
    let kernels = requests
        .iter()
        .map(|(request, _)| request.clone())
        .collect::<Vec<_>>();
    let mut failures = BTreeMap::new();
    for ((request, model), formed) in requests.iter().zip(former.form_all(module, &kernels)) {
        if let Err(failure) = formed {
            failures.insert(request.to_string(), format!("{failure} (first requested by `{model}`)"));
        }
    }
    eprintln!(
        "{}: formed {} of {} catalog kernel requests",
        backend.as_str(),
        requests.len() - failures.len(),
        requests.len()
    );
    assert!(
        failures.is_empty(),
        "{} catalog kernel requests do not form:\n{}",
        failures.len(),
        failures
            .iter()
            .map(|(request, failure)| format!("{request}: {failure}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn vulkan_catalog_kernels_form() {
    form(BackendName::Vulkan);
}

#[test]
fn cuda_catalog_kernels_form() {
    form(BackendName::Cuda);
}

#[test]
fn metal_catalog_kernels_form() {
    form(BackendName::Metal);
}
