//! Every probe of each available Vulkan device, each in a probe process of
//! this test's executable, as the measurement job runs them: each keeps its
//! result, a driver crash after the probe began among them (kept as
//! unanswerable); none fails for a reason that says nothing of the device.
//! Prints what each probe found.
//!
//! Built with `harness = false`: like every executable that runs the
//! measurement job, this one serves the probe processes it starts.

fn main() {
    seismic_runtime::probes::serve_probe_if_asked();
    #[cfg(not(target_os = "macos"))]
    every_probe_answers_in_its_process();
}

#[cfg(not(target_os = "macos"))]
fn every_probe_answers_in_its_process() {
    use seismic_runtime::artifacts::MemoryStore;
    use seismic_topology::{Availability, ExecutorSelector};

    let devices = seismic_runtime::devices::Devices::discover(&[]).expect("the machine is discovered");
    let topology = devices.topology();
    let store = MemoryStore::default();
    for executor in topology.executors() {
        if !matches!(executor.selector(), ExecutorSelector::Vulkan { .. }) || *executor.availability() != Availability::Available {
            continue;
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
        let run = devices.probe(executor.id(), &store, deadline).expect("the executor's probes run");
        for kept in &run.kept {
            eprintln!("{}: {}: {}\n{}", executor.selector(), kept.probe, kept.result.line(), kept.report);
        }
        assert!(run.failures.is_empty(), "{}: {:?}", executor.selector(), run.failures);
    }
}
