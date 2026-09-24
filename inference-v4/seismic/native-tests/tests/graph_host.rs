//! Host cost of one native graph run (S2): attach plus submit of a
//! 300-node graph whose only per-run change is one external binding. The
//! kernels are tiny, so on GPU backends the timed host work is binding and
//! encoding; on the CPU backend submission also executes the nodes.
//!
//! Run with `cargo test --release -p seismic-native-tests --test graph_host
//! -- --ignored --nocapture`.

use seismic::{
    Availability, BackendName, Device, DeviceCatalog, Element, NativeGraphFamily,
    NativeSpecialization, Tensor,
};
use seismic_native_tests::scale_rows;
use std::time::Instant;

const NODES: usize = 300;
const WARM: usize = 20;
const RUNS: usize = 200;

fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let topology = catalog.topology();
    [BackendName::Cpu, BackendName::Metal, BackendName::Cuda]
        .into_iter()
        .filter(|backend| {
            topology.devices().iter().any(|device| {
                device.backend == *backend && matches!(device.availability, Availability::Available)
            })
        })
        .map(|backend| catalog.open_backend(backend).expect("available device opens"))
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "host-time micro-benchmark; run explicitly with --ignored --nocapture"]
fn host_time_per_run_of_a_300_node_graph() {
    for device in devices() {
        let (m, n) = (1u64, 8u64);
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[m, n]).unwrap();
        let mut value = graph
            .enqueue(&kernel, scale_rows::WorkflowArgs { x: input.tensor().into(), factor: 1.0 })
            .unwrap()
            .value;
        for _ in 1..NODES {
            value = graph
                .enqueue(&kernel, scale_rows::WorkflowArgs { x: (&value).into(), factor: 1.0 })
                .unwrap()
                .value;
        }
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        // The plan writes no input, so the slot holds no upload region.
        let mut slot = family.new_slot(1).unwrap();
        let mut output_slot = Some(family.new_output_slot().unwrap());
        let inputs = (0..2)
            .map(|index| {
                let bytes = (0..m * n)
                    .flat_map(|element| ((element + index * 100) as f32).to_le_bytes())
                    .collect::<Vec<_>>();
                Tensor::from_host(&device, Element::f32(), &[m, n], &bytes).unwrap()
            })
            .collect::<Vec<_>>();
        let mut attach_seconds = Vec::with_capacity(RUNS);
        let mut submit_seconds = Vec::with_capacity(RUNS);
        let mut completions = Vec::with_capacity(RUNS + WARM);
        for run in 0..WARM + RUNS {
            let started = Instant::now();
            let outputs = output_slot.take().unwrap().activate(&plan).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(&input, &inputs[run % 2]).unwrap();
            let mut active = slot.activate(&plan).unwrap();
            let ready = active.attach(bindings, outputs).unwrap();
            let attached = Instant::now();
            let (outputs, completion) = ready.submit().unwrap();
            let submitted = Instant::now();
            output_slot = Some(outputs.recycle().unwrap());
            completions.push(completion);
            if run >= WARM {
                attach_seconds.push((attached - started).as_secs_f64());
                submit_seconds.push((submitted - attached).as_secs_f64());
            }
        }
        for completion in completions {
            completion.wait().unwrap();
        }
        let attach = median(&mut attach_seconds);
        let submit = median(&mut submit_seconds);
        eprintln!(
            "graph host time {:?}: {NODES} nodes, median per run: attach {:.1} us, submit {:.1} us, total {:.1} us ({:.2} us/node)",
            device.backend(),
            attach * 1e6,
            submit * 1e6,
            (attach + submit) * 1e6,
            (attach + submit) * 1e6 / NODES as f64
        );
    }
}
