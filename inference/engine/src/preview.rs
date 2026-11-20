//! Read-only load planning (integration spec §4.2, §7.3): which executor a
//! load of a resolved configuration would use and how much it claims at
//! startup, charged along the executor's placement. Opens no
//! device, reads no weight payload and allocates nothing; a load repeats
//! device resolution and makes live claims.

use crate::error::{classify_plan, classify_platform, PlanOutcome, PreviewError};
use crate::options::ExecutionManifest;
use crate::planning::{plan_execution, ExecutionPlanningError};
use seismic::{BackendName, Devices, ExecutorSelector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadPreview {
    /// The served context ceiling.
    pub context_tokens: u64,
    /// The executor a load would open, by its cross-process identity.
    pub device: ExecutorSelector,
    pub backend: BackendName,
    /// The load's startup peak claim, across the stores of the executor's
    /// placement: resident weights, initial state, prepared resources and
    /// the startup transient. History growth is elastic and claimed as it
    /// grows, so it is not included.
    pub required_bytes: u64,
}

fn internal(reason: impl Into<String>) -> PreviewError {
    PreviewError::Internal {
        reason: reason.into(),
    }
}

/// An operation's preview once the plan's prices are measured: the one way
/// a pending plan (`PreviewError::PricesPending`, every pending figure a
/// plan prices on) is met, by every host. While pending, `request` asks the
/// measurement job for the figures and `measured` resolves once a job has
/// run (a host's own job in its process, or its scheduler's), and the plan
/// is taken again: pending never ends the operation. Only `request` or
/// `measured` failing does, and a host's `measured` fails where its job
/// measured nothing of what is pending, so nothing waits forever on a job
/// that cannot measure it. Async, so a host that waits on its scheduler
/// can drop the wait; a host that measures in its own thread resolves each
/// step at once ([`now`]).
pub async fn priced<T, E, P, R, M>(
    mut preview: impl FnMut() -> P,
    mut request: impl FnMut() -> R,
    mut measured: impl FnMut() -> M,
) -> Result<Result<T, PreviewError>, E>
where
    P: std::future::Future<Output = Result<T, PreviewError>>,
    R: std::future::Future<Output = Result<(), E>>,
    M: std::future::Future<Output = Result<(), E>>,
{
    loop {
        match preview().await {
            Err(PreviewError::PricesPending { reason }) => {
                tracing::info!(%reason, "the load waits for the machine's standing figures");
                request().await?;
                measured().await?;
            }
            previewed => return Ok(previewed),
        }
    }
}

/// The output of `future`, which a host that does each step in its own
/// thread made of values already computed: it is ready at its first poll.
pub fn now<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    match future.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
        std::task::Poll::Ready(output) => output,
        std::task::Poll::Pending => panic!("a step done in this thread is ready at once"),
    }
}

impl ExecutionManifest {
    /// Preview a load of this manifest on `catalog`. Only the manifest is read, so a host that
    /// shares its host artifacts elsewhere can still preview.
    pub fn preview(&self, catalog: &Devices) -> Result<LoadPreview, PreviewError> {
        let manifest = self;
        let selected = crate::planning::select_load_device(catalog, manifest)
            .map_err(|error| PreviewError::from(classify_platform(error)))?;
        let backend = selected.info.backend();
        // The bounds planning charged the draft with: its graphs are
        // certified once, there.
        let planned = plan_execution::<magnitude_executor::AssessmentMemoryCharge>(manifest, &selected).map_err(|error| match error {
            ExecutionPlanningError::Plan(error) => classify_plan(error, backend).into(),
            error => PreviewError::from(PlanOutcome::Internal(error.to_string())),
        })?;
        let required_bytes = planned
            .charge()
            .startup_required_bytes(&selected.reading.topology, &selected.info)
            .map_err(internal)?;
        Ok(LoadPreview {
            context_tokens: manifest.definition.decoder.context_limit,
            device: selected.info.selector(),
            backend,
            required_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{fixtures, platform::{DeviceRequest, ReleaseRate}, ExecutionPath};
    use std::cell::Cell;

    /// A load on a host with nothing of its executor's costs measured plans
    /// as every host's first load does: its plan is pending, it asks the
    /// job for the standing figures and waits, the job stores them (here
    /// fixed, as a test's measurements are), and the plan is taken again
    /// and priced. Pending ends nothing; the request is stored for the job.
    #[test]
    fn a_load_on_a_host_with_nothing_measured_requests_its_prices_then_plans() {
        let cache = tempfile::tempdir().unwrap();
        let definition = fixtures::tiny_definition();
        let model = crate::options::ModelPolicy::default().resolve(&definition).unwrap();
        let manifest = ExecutionManifest::new(
            fixtures::tiny_manifest(&definition),
            definition,
            model,
            crate::options::standard_service_limits(),
            ExecutionPath::Native,
            DeviceRequest::Backend(BackendName::Cpu),
            Some(cache.path().to_owned()),
            Some(cache.path().to_owned()),
            ReleaseRate::of_machine(None),
        )
        .unwrap();
        let catalog = Devices::discover(&[]).unwrap();
        let selected = crate::planning::select_load_device(&catalog, &manifest).unwrap();
        assert!(matches!(manifest.preview(&catalog), Err(PreviewError::PricesPending { .. })), "nothing is measured yet");
        let (requests, jobs) = (Cell::new(0), Cell::new(0));
        let previewed = now(priced(
            || std::future::ready(manifest.preview(&catalog)),
            || {
                requests.set(requests.get() + 1);
                std::future::ready(manifest.request_prices(&catalog))
            },
            || {
                jobs.set(jobs.get() + 1);
                // The request is stored for the job before it runs.
                let records = magnitude_executor::KernelCache::measurement_records(cache.path());
                let costs = magnitude_executor::assessment::ExecutorCosts::stored(&catalog.topology(), &selected.info, &records).unwrap().outstanding();
                let outstanding = magnitude_executor::assessment::PointRequests::new(cache.path())
                    .outstanding(&catalog.topology(), &selected.info, cache.path(), &records, costs)
                    .unwrap();
                assert!(!outstanding.is_empty(), "the job has the load's figures to measure");
                crate::execution::store_fixed_costs(&catalog.topology(), &selected.info, cache.path());
                std::future::ready(match jobs.get() {
                    1 => Ok(()),
                    _ => Err("a job that stored the costs left the plan pending".to_owned()),
                })
            },
        ));
        let preview = previewed.expect("the job ran").expect("the plan is priced once its costs are stored");
        assert_eq!(preview.backend, BackendName::Cpu);
        assert_eq!((requests.get(), jobs.get()), (1, 1), "asked once, measured once, planned again");
    }

    /// A step the host's measuring fails ends the operation with its
    /// failure; a preview that fails otherwise is returned as it failed,
    /// never waited on.
    #[test]
    fn only_a_failed_measurement_or_a_plan_that_fails_ends_a_priced_preview() {
        let failed = now(priced(
            || std::future::ready(Err::<(), _>(PreviewError::PricesPending { reason: "nothing is measured".into() })),
            || std::future::ready(Ok(())),
            || std::future::ready(Err("the job measured nothing")),
        ));
        assert_eq!(failed, Err("the job measured nothing"));
        let unsupported = now(priced(
            || std::future::ready(Err::<(), _>(PreviewError::Internal { reason: "no plan".into() })),
            || std::future::ready(Err::<(), &str>("never asked")),
            || std::future::ready(Err("never measured")),
        ));
        assert_eq!(unsupported, Ok(Err(PreviewError::Internal { reason: "no plan".into() })));
    }

    /// `LoadPreview::required_bytes` delegates directly to the planned
    /// charge's `startup_required_bytes` for the selected device.
    #[test]
    fn preview_required_bytes_matches_planned_charge_startup_required_bytes() {
        let cache = tempfile::tempdir().unwrap();
        let definition = fixtures::tiny_definition();
        let model = crate::options::ModelPolicy::default().resolve(&definition).unwrap();
        let manifest = ExecutionManifest::new(
            fixtures::tiny_manifest(&definition),
            definition,
            model,
            crate::options::standard_service_limits(),
            ExecutionPath::Native,
            DeviceRequest::Backend(BackendName::Cpu),
            Some(cache.path().to_owned()),
            Some(cache.path().to_owned()),
            ReleaseRate::of_machine(None),
        )
        .unwrap();
        let catalog = Devices::discover(&[]).unwrap();
        let selected = crate::planning::select_load_device(&catalog, &manifest).unwrap();
        crate::execution::store_fixed_costs(&catalog.topology(), &selected.info, cache.path());
        let preview = manifest.preview(&catalog).unwrap();
        let planned = plan_execution::<magnitude_executor::AssessmentMemoryCharge>(&manifest, &selected).unwrap();
        let expected = planned
            .charge()
            .startup_required_bytes(&selected.reading.topology, &selected.info)
            .unwrap();
        assert_eq!(preview.required_bytes, expected);
    }
}
