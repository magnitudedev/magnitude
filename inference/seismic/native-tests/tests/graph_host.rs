//! Host cost of one native graph run (S2): attach plus submit of a
//! 300-node graph whose only per-run change is one external binding. The
//! kernels are tiny, so on GPU backends the timed host work is binding and
//! encoding; on the CPU backend submission also executes the nodes.
//!
//! Run with `cargo test --release -p seismic-native-tests --test graph_host
//! -- --ignored --nocapture`.

use seismic::{
    Device, Element, HostHeap, NativeGraphFamily,
    NativeGraphMetadata, NativeSpecialization, Tensor,
};
use seismic_native_tests::{accumulate, scale_rows};
use std::time::Instant;

const NODES: usize = 300;
const WARM: usize = 20;
const RUNS: usize = 200;

/// The host heap of the plan sealed from `nodes` chained `scale_rows` calls
/// and one `accumulate` on two external ports (a disjoint pair a run
/// checks), and the heap its topology plans before any kernel is formed.
fn sealed_and_planned_heap(device: &Device, nodes: usize) -> (HostHeap, HostHeap) {
    let (m, n) = (3u64, 8u64);
    let scale = scale_rows::native_for_device(
        device,
        &NativeSpecialization::new().with_param("ROWS", 1),
    )
    .unwrap();
    let accumulate = accumulate::native_for_device(
        device,
        &NativeSpecialization::new()
            .with_param("BIAS", 0)
            .with_param("WIDTH", 32),
    )
    .unwrap();
    let mut graph = device.native_graph();
    let input = graph.port(Element::f32(), &[m, n]).unwrap();
    let mut state = graph.port(Element::f32(), &[n]).unwrap();
    let addend = graph.port(Element::f32(), &[n]).unwrap();
    let mut value = input.tensor().clone();
    for _ in 0..nodes {
        value = graph
            .enqueue(
                &scale,
                scale_rows::WorkflowArgs {
                    x: (&value).into(),
                    factor: seismic::Argument::from(2.0),
                },
            )
            .unwrap()
            .value;
    }
    graph
        .enqueue(
            &accumulate,
            accumulate::WorkflowArgs {
                state: state.tensor_mut().into(),
                x: addend.tensor().into(),
            },
        )
        .unwrap();
    graph.export(&value).unwrap();
    let plan = graph.seal().unwrap();
    let sealed = plan.host_heap().plus(plan.topology_heap()).unwrap();

    let mut graph = NativeGraphMetadata::new(device.backend());
    let input = graph.port(Element::f32(), &[m, n]).unwrap();
    let mut state = graph.port(Element::f32(), &[n]).unwrap();
    let addend = graph.port(Element::f32(), &[n]).unwrap();
    let mut value = input.tensor().clone();
    for _ in 0..nodes {
        value = graph
            .enqueue::<scale_rows::Entry>(
                &[],
                &[("M", m), ("N", n)],
                scale_rows::WorkflowArgs {
                    x: (&value).into(),
                    factor: seismic::Argument::from(2.0),
                },
            )
            .unwrap()
            .value;
    }
    graph
        .enqueue::<accumulate::Entry>(
            &[],
            &[("N", n)],
            accumulate::WorkflowArgs {
                state: state.tensor_mut().into(),
                x: addend.tensor().into(),
            },
        )
        .unwrap();
    graph.export(&value).unwrap();
    let planned = graph.plan_heap().plus(graph.topology_heap()).unwrap();
    assert_eq!(plan.host_heap(), graph.plan_heap());
    assert_eq!(plan.topology_heap(), graph.topology_heap());
    (sealed, planned)
}

/// A sealed plan's host heap is the sum of its vectors' capacities, and the
/// plan's topology gives the same bytes in the same allocations from its
/// entries' schemas and declarations alone: every vector of a sealed plan
/// holds exactly its length, and each added node adds one node's heap.
#[test]
fn a_sealed_plan_holds_the_host_heap_its_topology_plans() {
    for device in seismic_native_tests::every_device() {
        let heaps = [1usize, 2, 9].map(|nodes| sealed_and_planned_heap(&device, nodes));
        for (nodes, (sealed, planned)) in [1usize, 2, 9].into_iter().zip(heaps) {
            assert_eq!(
                sealed,
                planned,
                "{:?}: {nodes} nodes",
                device.backend()
            );
            assert!(sealed.bytes > 0 && sealed.allocations > 0);
        }
        let node = heaps[1].0.bytes - heaps[0].0.bytes;
        assert_eq!(
            heaps[2].0.bytes - heaps[1].0.bytes,
            7 * node,
            "{:?}: a node's heap does not depend on the graph's length",
            device.backend()
        );
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "host-time micro-benchmark; run explicitly with --ignored --nocapture"]
fn host_time_per_run_of_a_300_node_graph() {
    for device in seismic_native_tests::every_device() {
        let charges = seismic::Charges::tally(&device);
        let (m, n) = (1u64, 8u64);
        let kernel = scale_rows::native_for_device(
            &device,
            &NativeSpecialization::new().with_param("ROWS", 1),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[m, n]).unwrap();
        let mut value = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: seismic::Argument::from(1.0),
                },
            )
            .unwrap()
            .value;
        for _ in 1..NODES {
            value = graph
                .enqueue(
                    &kernel,
                    scale_rows::WorkflowArgs {
                        x: (&value).into(),
                        factor: seismic::Argument::from(1.0),
                    },
                )
                .unwrap()
                .value;
        }
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()], 1).unwrap();
        // The plan writes no input, so the slot holds no upload region.
        let mut slot = family.new_slot(&charges, &[family.upload_bytes()]).unwrap();
        let mut output_slot = Some(family.new_output_slot(&charges).unwrap());
        let inputs = (0..2)
            .map(|index| {
                let bytes = (0..m * n)
                    .flat_map(|element| ((element + index * 100) as f32).to_le_bytes())
                    .collect::<Vec<_>>();
                Tensor::from_host(&charges, Element::f32(), &[m, n], &bytes).unwrap()
            })
            .collect::<Vec<_>>();
        let mut attach_seconds = Vec::with_capacity(RUNS);
        let mut submit_seconds = Vec::with_capacity(RUNS);
        let mut completions = Vec::with_capacity(RUNS + WARM);
        for run in 0..WARM + RUNS {
            let started = Instant::now();
            let outputs = output_slot.take().unwrap().activate(&plan).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(&charges, &input, &inputs[run % 2]).unwrap();
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

/// Inputs have a completion lifetime even if the caller releases every tensor
/// handle before submission. A weak census handle does not supply that lifetime.
#[test]
fn graph_completion_retains_inputs_until_its_outcome_is_taken() {
    for device in seismic_native_tests::every_device() {
        let charges = seismic::Charges::tally(&device);
        let kernel = scale_rows::native_for_device(
            &device, &NativeSpecialization::new().with_param("ROWS", 1),
        ).unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[1, 8]).unwrap();
        let value = graph.enqueue(&kernel, scale_rows::WorkflowArgs {
            x: input.tensor().into(), factor: seismic::Argument::from(2.0),
        }).unwrap().value;
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()], 1).unwrap();
        let mut slot = family.new_slot(&charges, &[family.upload_bytes()]).unwrap();
        let source = Tensor::from_host(&charges, Element::f32(), &[1, 8], &[0; 32]).unwrap();
        let observer = source.observe_storage();
        let mut bindings = plan.bindings();
        bindings.set(&charges, &input, &source).unwrap();
        drop(source);
        let mut active = slot.activate(&plan).unwrap();
        let ready = active.attach(bindings, plan.new_outputs(&charges).unwrap()).unwrap();
        let (outputs, completion) = ready.submit().unwrap();
        assert_eq!(observer.charged_bytes(), Some(32));
        completion.wait().unwrap();
        assert_eq!(observer.charged_bytes(), None);
        drop(outputs);
    }
}

/// Plans sealed from one `NativeGraphLayout` share their immutable
/// `SealedTopology` allocation via `Arc` and each plan's `host_heap()` equals
/// the layout's `plan_heap()`.
#[test]
fn plans_of_one_layout_share_topology() {
    for device in seismic_native_tests::every_device() {
        let layout = NativeGraphMetadata::certify_group([1u64, 4].into_iter().map(|m| {
            let mut metadata = NativeGraphMetadata::new(device.backend());
            let input = metadata.port(Element::f32(), &[m, 8u64]).unwrap();
            let value = metadata
                .enqueue::<scale_rows::Entry>(
                    &[],
                    &[("M", m), ("N", 8u64)],
                    scale_rows::WorkflowArgs {
                        x: input.tensor().into(),
                        factor: seismic::Argument::from(2.0),
                    },
                )
                .unwrap()
                .value;
            metadata.export(&value).unwrap();
            metadata
        }))
        .unwrap();

        let seal_for = |m: u64| {
            let kernel = scale_rows::native_for_device(
                &device,
                &NativeSpecialization::new().with_param("ROWS", 1),
            )
            .unwrap();
            let mut graph = device.native_graph_with_layout(&layout);
            let input = graph.port(Element::f32(), &[m, 8]).unwrap();
            let value = graph
                .enqueue(
                    &kernel,
                    scale_rows::WorkflowArgs {
                        x: input.tensor().into(),
                        factor: seismic::Argument::from(2.0),
                    },
                )
                .unwrap()
                .value;
            graph.export(&value).unwrap();
            graph.seal().unwrap()
        };

        let plan_a = seal_for(1);
        let plan_b = seal_for(4);
        assert!(plan_a.shares_topology(&plan_b), "{:?}: plans should share topology", device.backend());
        assert_eq!(plan_a.topology_identity(), plan_b.topology_identity());
        assert_eq!(plan_a.host_heap(), layout.plan_heap(), "{:?}: plan_a heap", device.backend());
        assert_eq!(plan_b.host_heap(), layout.plan_heap(), "{:?}: plan_b heap", device.backend());
        assert_eq!(plan_a.topology_heap(), layout.topology_heap(), "{:?}: plan_a topology heap", device.backend());
        assert_eq!(plan_b.topology_heap(), layout.topology_heap(), "{:?}: plan_b topology heap", device.backend());
        assert!(layout.topology_heap().bytes > 0 && layout.topology_heap().allocations > 0);
        // Combined family heap charges per-plan slices twice and the shared topology once.
        let planned_total = layout.plan_heap().times(2).unwrap().plus(layout.topology_heap()).unwrap();
        let held_total = plan_a.host_heap().plus(plan_b.host_heap()).unwrap().plus(plan_a.topology_heap()).unwrap();
        assert_eq!(held_total, planned_total, "{:?}: combined planned == held with shared topology", device.backend());
    }
}
