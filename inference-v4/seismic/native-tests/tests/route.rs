//! The direct-native route on every backend this host can open: static
//! dimensions, tuning parameters, `where`, scratch, several launches, shared
//! memory, conditional (`when`) launches and scratch, graphs with
//! host-written inputs, asynchronous submission, measurement and tuning.

use seismic::{
    Availability, BackendName, CallError, Device, DeviceCatalog, Element, Exclusion,
    InvocationError, MeasureOptions, NativeGraphFamily, NativeSpecialization, Outcome, SearchPlan,
    SearchSettings, SearchStop, Strategy, SurveyPlan, Tensor, TraceDetail, TuneError,
    TuningInitializer, TuningMethod, TuningPoint, Validation,
};
use seismic_native_tests::{accumulate, gated_sum, scale_rows, split_sum};

/// A search whose budget covers every configuration of the test entries.
fn search(samples: usize) -> Strategy {
    Strategy::Search(SearchPlan {
        budget: 100,
        settings: SearchSettings {
            improvement: 0.01,
            restarts: 2,
            confirmed: 3,
            default_margin: 0.02,
            samples,
            confirmation_samples: samples,
        },
        min_sample_seconds: 0.0002,
        start: Vec::new(),
        deadline: None,
    })
}

/// Every backend the catalog reports available on this host; each must open.
fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let topology = catalog.topology();
    let devices = [BackendName::Cpu, BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]
        .into_iter()
        .filter(|backend| {
            topology.devices().iter().any(|device| {
                device.backend == *backend && matches!(device.availability, Availability::Available)
            })
        })
        .map(|backend| {
            catalog
                .open_backend(backend)
                .unwrap_or_else(|error| panic!("available {backend:?} device must open: {error}"))
        })
        .collect::<Vec<_>>();
    eprintln!(
        "native route backends: {:?}",
        devices.iter().map(Device::backend).collect::<Vec<_>>()
    );
    devices
}

fn f32_tensor(device: &Device, extents: &[u64], values: &[f32]) -> Tensor {
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), extents, &bytes).expect("host tensor")
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .expect("host read")
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("f32 word")))
        .collect()
}

/// Values whose every partial sum is exact in f32, so all configurations
/// agree bit for bit with the ordered reference.
fn exact_values(n: usize) -> Vec<f32> {
    (0..n).map(|index| ((index % 7) as f32 - 3.0) * 0.25).collect()
}

fn statics(n: u64) -> NativeSpecialization {
    NativeSpecialization::new().with_static("N", n)
}

#[test]
fn every_admissible_split_sum_configuration_matches_the_reference() {
    for device in devices() {
        let n = 1000usize;
        let values = exact_values(n);
        let expected: f32 = values.iter().sum();
        let x = f32_tensor(&device, &[n as u64], &values);
        let implementation = split_sum::native_implementation(&device)
            .expect("bundle")
            .expect("split_sum has an implementation on every backend");
        let configurations = implementation.admissible(&statics(n as u64)).expect("statics");
        // Three part counts by two widths, all admissible at N = 1000.
        assert_eq!(configurations.len(), 6, "{}", device.backend().as_str());
        for configuration in configurations {
            let kernel = split_sum::native_for_device(&device, &configuration)
                .unwrap_or_else(|error| panic!("{configuration:?} on {:?}: {error}", device.backend()));
            let result = kernel.call(split_sum::Args { x: &x }).expect("call");
            assert_eq!(
                read_f32(&result.value),
                [expected],
                "{configuration:?} on {:?}",
                device.backend()
            );
        }
    }
}

#[test]
fn where_filters_configurations_at_small_static_values() {
    for device in devices() {
        let implementation = split_sum::native_implementation(&device).unwrap().unwrap();
        let admissible = implementation.admissible(&statics(2)).unwrap();
        // PARTS <= N keeps parts 1 and 2 at N = 2.
        assert_eq!(admissible.len(), 4);
        assert!(admissible.iter().all(|configuration| configuration.param("PARTS") != Some(4)));
        let inadmissible = statics(2).with_param("PARTS", 4).with_param("WIDTH", 32);
        assert!(split_sum::native_for_device(&device, &inadmissible).is_err());
    }
}

#[test]
fn specialization_errors_are_typed_at_preparation() {
    for device in devices() {
        let missing_static = NativeSpecialization::new()
            .with_param("PARTS", 1)
            .with_param("WIDTH", 32);
        assert!(split_sum::native_for_device(&device, &missing_static).is_err());
        let outside = statics(64).with_param("PARTS", 3).with_param("WIDTH", 32);
        let error = split_sum::native_for_device(&device, &outside)
            .err()
            .expect("3 is outside PARTS's domain");
        assert!(error.to_string().contains("does not admit 3"), "{error}");
        let missing_param = statics(64).with_param("PARTS", 1);
        assert!(split_sum::native_for_device(&device, &missing_param).is_err());
    }
}

#[test]
fn a_call_whose_static_dimension_differs_is_rejected() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(64).with_param("PARTS", 2).with_param("WIDTH", 32),
        )
        .unwrap();
        let x = f32_tensor(&device, &[63], &exact_values(63));
        match kernel.call(split_sum::Args { x: &x }) {
            Err(CallError::Invocation(InvocationError::StaticDimension {
                dimension,
                expected,
                ..
            })) => {
                assert_eq!(dimension, "N");
                assert_eq!(expected, 64);
            }
            other => panic!("expected a static-dimension error, got {:?}", other.err()),
        }
    }
}

/// A graph whose input the host writes: two runs submitted back to back
/// without waiting must each read their own input.
#[test]
fn graph_runs_submit_without_waiting_and_keep_their_inputs() {
    for device in devices() {
        let (m, n) = (5u64, 37u64);
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 2))
                .unwrap();
        let mut graph = device.native_graph();
        let input = graph.input_for(&kernel, "x", &[("M", m), ("N", n)]).unwrap();
        let first = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 2.0,
                },
            )
            .unwrap();
        let second = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: (&first.value).into(),
                    factor: 3.0,
                },
            )
            .unwrap();
        graph.export(&second.value).unwrap();
        let plan = graph.seal().unwrap();
        assert!(plan.upload_bytes() >= m * n * 4);
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        // One upload region per run kept in flight below.
        let mut slot = family.new_slot(3).unwrap();
        let mut runs = Vec::new();
        for run in 0..3u64 {
            let values = (0..m * n)
                .map(|index| (index + run * 1000) as f32)
                .collect::<Vec<_>>();
            let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
            let mut active = slot.activate(&plan).unwrap();
            active.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = active
                .attach(plan.bindings(), outputs)
                .unwrap()
                .submit()
                .unwrap();
            runs.push((values, outputs, completion));
        }
        for (values, outputs, completion) in runs {
            completion.wait().unwrap();
            let result = read_f32(&outputs.exported(&second.value).unwrap());
            let expected = values.iter().map(|value| value * 6.0).collect::<Vec<_>>();
            assert_eq!(result, expected, "{:?}", device.backend());
        }
    }
}

#[test]
fn measurement_reports_device_time() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(4096).with_param("PARTS", 4).with_param("WIDTH", 64),
        )
        .unwrap();
        let rotation = (0..3)
            .map(|_| f32_tensor(&device, &[4096], &exact_values(4096)))
            .collect::<Vec<_>>();
        let measurement = kernel
            .measure(
                rotation.iter().map(|x| split_sum::Args { x }).collect(),
                &MeasureOptions {
                    samples: 3,
                    min_sample_seconds: 0.0005,
                },
            )
            .unwrap();
        assert_eq!(measurement.samples.len(), 3);
        assert!(measurement.median > 0.0, "{:?}", device.backend());
        assert!(measurement.rotation_bytes >= 3 * 4096 * 4);
    }
}

#[test]
fn tuning_searches_from_the_defaults_and_validates_its_choice() {
    for device in devices() {
        let n = 4096u64;
        let inputs = (0..2)
            .map(|_| f32_tensor(&device, &[n], &exact_values(n as usize)))
            .collect::<Vec<_>>();
        fn points(inputs: &[Tensor]) -> Vec<TuningPoint<'_, split_sum::Entry>> {
            vec![
                TuningPoint {
                    label: "short".into(),
                    weight: 1.0,
                    class: None,
                    rotation: inputs.iter().map(|x| split_sum::Args { x }).collect(),
                    initialize: None,
                },
                TuningPoint {
                    label: "long".into(),
                    weight: 3.0,
                    class: None,
                    rotation: inputs.iter().map(|x| split_sum::Args { x }).collect(),
                    initialize: None,
                },
            ]
        }
        let result = split_sum::native_tune(
            &device,
            &statics(n),
            points(&inputs),
            Validation::BitExact,
            search(3),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        // The budget covers the whole domain: the search reaches every
        // configuration once, starting from the defaults.
        assert_eq!(result.configurations.len(), 6);
        assert!(matches!(
            result.method,
            TuningMethod::Search { stop: SearchStop::Exhausted, .. }
        ));
        assert_eq!(
            result.configurations[0].configuration.params,
            [("PARTS".to_owned(), 1), ("WIDTH".to_owned(), 32)].into_iter().collect()
        );
        assert!(result
            .configurations
            .iter()
            .all(|record| matches!(record.outcome, Outcome::Measured { .. })));
        // The chosen configuration was validated, and the finalists
        // re-measured.
        assert!(result.configurations.iter().any(|record| {
            record.configuration == result.overall
                && matches!(record.outcome, Outcome::Measured { validated: true, .. })
        }));
        assert!(result.configurations.iter().any(|record| matches!(
            &record.outcome,
            Outcome::Measured { confirmed, .. } if confirmed.len() == 2
        )));
        // The tuned configuration prepares and runs.
        let kernel = split_sum::native_for_device(&device, &result.overall.specialization()).unwrap();
        let value = kernel.call(split_sum::Args { x: &inputs[0] }).unwrap().value;
        assert_eq!(read_f32(&value), [exact_values(n as usize).iter().sum::<f32>()]);

        // A survey measures and validates every configuration with every
        // sample recorded.
        let survey = split_sum::native_tune(
            &device,
            &statics(n),
            points(&inputs),
            Validation::BitExact,
            Strategy::Survey(SurveyPlan {
                samples: 5,
                min_sample_seconds: 0.0002,
                domains: Default::default(),
            }),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        assert_eq!(survey.configurations.len(), 6);
        for record in &survey.configurations {
            match &record.outcome {
                Outcome::Measured {
                    points, validated, ..
                } => {
                    assert!(validated);
                    assert!(points.iter().all(|point| point.samples.len() == 5));
                }
                Outcome::Excluded(exclusion) => {
                    panic!("{:?}: {exclusion:?}", device.backend())
                }
            }
        }
    }
}

/// A graph node whose port contradicts the kernel's static dimension is
/// rejected when the graph is sealed, not when it runs.
#[test]
fn a_graph_whose_static_dimension_differs_is_rejected_at_seal() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(64).with_param("PARTS", 2).with_param("WIDTH", 32),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let input = graph.input_for(&kernel, "x", &[("N", 63)]).unwrap();
        let result = graph
            .enqueue(&kernel, split_sum::WorkflowArgs { x: input.tensor().into() })
            .unwrap();
        graph.export(&result.value).unwrap();
        match graph.seal() {
            Err(CallError::Invocation(InvocationError::StaticDimension {
                dimension,
                expected,
                ..
            })) => {
                assert_eq!(dimension, "N");
                assert_eq!(expected, 64);
            }
            Err(other) => panic!("{:?}: expected a static-dimension error, got {other}", device.backend()),
            Ok(_) => panic!("{:?}: a contradicted static dimension sealed", device.backend()),
        }
    }
}

/// Node scratch lives in the slot workspace: two scratch-using nodes and a
/// node without scratch run correctly across repeated runs of one slot.
#[test]
fn graph_nodes_with_scratch_run_from_the_slot_workspace() {
    for device in devices() {
        let n = 1000u64;
        let kernel = split_sum::native_for_device(
            &device,
            &statics(n).with_param("PARTS", 4).with_param("WIDTH", 64),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let first_input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
        let second_input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
        let first = graph
            .enqueue(&kernel, split_sum::WorkflowArgs { x: first_input.tensor().into() })
            .unwrap();
        let second = graph
            .enqueue(&kernel, split_sum::WorkflowArgs { x: second_input.tensor().into() })
            .unwrap();
        graph.export(&first.value).unwrap();
        graph.export(&second.value).unwrap();
        let plan = graph.seal().unwrap();
        // Each node's four f32 partials; the second node may reuse the first's.
        assert!(plan.workspace_bytes() >= 16, "{:?}", device.backend());
        let mut slot = plan.new_slot().unwrap();
        for run in 0..3usize {
            let first_values = (0..n as usize)
                .map(|index| ((index + run) % 5) as f32 - 2.0)
                .collect::<Vec<_>>();
            let second_values = exact_values(n as usize);
            let bytes = |values: &[f32]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            slot.write_input(&first_input, &bytes(&first_values)).unwrap();
            slot.write_input(&second_input, &bytes(&second_values)).unwrap();
            let (outputs, completion) = slot
                .attach(plan.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(
                read_f32(&outputs.exported(&first.value).unwrap()),
                [first_values.iter().sum::<f32>()],
                "{:?} run {run}",
                device.backend()
            );
            assert_eq!(
                read_f32(&outputs.exported(&second.value).unwrap()),
                [second_values.iter().sum::<f32>()],
                "{:?} run {run}",
                device.backend()
            );
        }
    }
}

/// The factor of node `node` of a scale chain: exact in f32.
fn chain_factor(node: usize) -> f32 {
    match node {
        0 => 1.0,
        _ if node % 2 == 0 => 0.5,
        _ => 2.0,
    }
}

/// `values` as a scale chain of `nodes` nodes computes them.
fn chained(values: &[f32], nodes: usize) -> Vec<f32> {
    values
        .iter()
        .map(|value| (0..nodes).fold(*value, |value, node| value * chain_factor(node)))
        .collect()
}

fn scale_chain(
    device: &Device,
    nodes: usize,
    m: u64,
    n: u64,
) -> (seismic::NativeGraphPlan, seismic::NativePort, seismic::WorkflowTensor) {
    let kernel =
        scale_rows::native_for_device(device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let mut graph = device.native_graph();
    let input = graph.input_for(&kernel, "x", &[("M", m), ("N", n)]).unwrap();
    let mut value = graph
        .enqueue(&kernel, scale_rows::WorkflowArgs { x: input.tensor().into(), factor: chain_factor(0) })
        .unwrap()
        .value;
    for node in 1..nodes {
        value = graph
            .enqueue(&kernel, scale_rows::WorkflowArgs { x: (&value).into(), factor: chain_factor(node) })
            .unwrap()
            .value;
    }
    graph.export(&value).unwrap();
    (graph.seal().unwrap(), input, value)
}

/// NR6: all launches of one graph submission are encoded as one unit (one
/// serial encoder), and give the bits of the per-launch-encoder form.
#[test]
fn a_graph_submission_encodes_every_node_into_one_encoder() {
    for device in devices() {
        let nodes = 8;
        let (m, n) = (3u64, 50u64);
        let (plan, input, value) = scale_chain(&device, nodes, m, n);
        let values = (0..m * n).map(|index| index as f32 * 0.25).collect::<Vec<_>>();
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let mut slot = plan.new_slot().unwrap();
        let mut run = |detail: TraceDetail| {
            let trace = device.trace_submissions(detail).unwrap();
            slot.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = slot
                .attach(plan.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            let submissions = trace.collect().unwrap();
            (read_f32(&outputs.exported(&value).unwrap()), submissions)
        };
        let (serial, serial_trace) = run(TraceDetail::Submissions);
        let (separate, separate_trace) = run(TraceDetail::Launches);
        assert_eq!(serial, chained(&values, nodes), "{:?}", device.backend());
        assert_eq!(serial, separate, "{:?}", device.backend());
        for trace in [&serial_trace, &separate_trace] {
            assert_eq!(trace.len(), 1, "{:?}: one submission per run", device.backend());
            assert_eq!(trace[0].launches.len(), nodes);
        }
        if device.backend() != BackendName::Cpu {
            // Only the per-launch form has a timed unit (encoder) per launch.
            assert!(serial_trace[0].launches.iter().all(|launch| launch.device.is_none()));
            assert!(separate_trace[0].launches.iter().all(|launch| launch.device.is_some()));
        }
    }
}

/// A sequence submits its runs as one unit, in queue order: a step of an
/// entry run and five block runs chained through two alternating output
/// leases (each recycled while a queued run still reads it, as an engine
/// step does) gives the separately submitted result, and repeated steps
/// (the same storage every time) each read their own input.
#[test]
fn a_sequence_submits_chained_runs_as_one_unit_in_queue_order() {
    for device in devices() {
        let (m, n) = (3u64, 64u64);
        let blocks = 5;
        let (entry, input, entry_value) = scale_chain(&device, 1, m, n);
        let scale =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let x = graph.port(Element::f32(), &[m, n]).unwrap();
        let tripled = graph
            .enqueue(&scale, scale_rows::WorkflowArgs { x: x.tensor().into(), factor: 3.0 })
            .unwrap()
            .value;
        graph.export(&tripled).unwrap();
        let block = graph.seal().unwrap();
        let entry_family = NativeGraphFamily::new(&[entry.clone()]).unwrap();
        let block_family = NativeGraphFamily::new(&[block.clone()]).unwrap();
        let mut entry_slot = entry_family.new_slot(1).unwrap();
        let mut block_slot = block_family.new_slot(1).unwrap();
        let mut entry_output = Some(entry_family.new_output_slot().unwrap());
        let mut block_outputs = [
            Some(block_family.new_output_slot().unwrap()),
            Some(block_family.new_output_slot().unwrap()),
        ];
        for step in 0..3u64 {
            let values = (0..m * n)
                .map(|index| (index + step * 7) as f32 * 0.25)
                .collect::<Vec<_>>();
            let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            let trace = device.trace_submissions(TraceDetail::Submissions).unwrap();
            let mut sequence = device.native_sequence();
            let mut active = entry_slot.activate(&entry).unwrap();
            active.write_input(&input, &bytes).unwrap();
            let entry_outputs = active
                .attach(
                    entry.bindings(),
                    entry_output.take().unwrap().activate(&entry).unwrap(),
                )
                .unwrap()
                .queue(&mut sequence)
                .unwrap();
            let mut hidden = entry_outputs.exported(&entry_value).unwrap();
            let mut pending: [Option<seismic::NativeGraphOutputs>; 2] = [None, None];
            for index in 0..blocks {
                let parity = index % 2;
                let mut bindings = block.bindings();
                bindings.set(&x, &hidden).unwrap();
                let outputs = block_slot
                    .activate(&block)
                    .unwrap()
                    .attach(
                        bindings,
                        block_outputs[parity].take().unwrap().activate(&block).unwrap(),
                    )
                    .unwrap()
                    .queue(&mut sequence)
                    .unwrap();
                hidden = outputs.exported(&tripled).unwrap();
                // The previous output is still read by the run just queued.
                if let Some(previous) = pending[1 - parity].take() {
                    block_outputs[1 - parity] = Some(previous.recycle().unwrap());
                }
                pending[parity] = Some(outputs);
            }
            let completion = sequence.submit().unwrap();
            completion.wait().unwrap();
            let submissions = trace.collect().unwrap();
            assert_eq!(submissions.len(), 1, "{:?}: one submission per step", device.backend());
            assert_eq!(submissions[0].launches.len(), 1 + blocks);
            let expected = values
                .iter()
                .map(|value| value * 3f32.powi(blocks as i32))
                .collect::<Vec<_>>();
            assert_eq!(read_f32(&hidden), expected, "step {step} on {:?}", device.backend());
            drop(hidden);
            entry_output = Some(entry_outputs.recycle().unwrap());
            for (parity, outputs) in pending.into_iter().enumerate() {
                if let Some(outputs) = outputs {
                    block_outputs[parity] = Some(outputs.recycle().unwrap());
                }
            }
        }
    }
}

/// Runs of one plan that bind the same storage (one workspace, one upload
/// region, one recycled output arena) replay the first run's formed launches
/// (on CUDA, one graph). Each still reads the input written for it.
#[test]
fn runs_rebinding_the_same_storage_read_their_own_inputs() {
    for device in devices() {
        let nodes = 6;
        let (m, n) = (4u64, 70u64);
        let (plan, input, value) = scale_chain(&device, nodes, m, n);
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(1).unwrap();
        let mut output_slot = family.new_output_slot().unwrap();
        for run in 0..4u64 {
            let values = (0..m * n)
                .map(|index| (index + run * 100) as f32 * 0.5)
                .collect::<Vec<_>>();
            let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            let mut active = slot.activate(&plan).unwrap();
            active.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = active
                .attach(plan.bindings(), output_slot.activate(&plan).unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(
                read_f32(&outputs.exported(&value).unwrap()),
                chained(&values, nodes),
                "run {run} on {:?}",
                device.backend()
            );
            output_slot = outputs.recycle().unwrap();
        }
    }
}

/// NR5: a host read of an exported output issued before its run completes
/// waits for the run and returns the final bytes.
#[test]
fn a_host_read_before_completion_returns_the_final_bytes() {
    for device in devices() {
        let (m, n) = (64u64, 1024u64);
        let nodes = 64;
        let (plan, input, value) = scale_chain(&device, nodes, m, n);
        let values = (0..m * n).map(|index| (index % 1000) as f32).collect::<Vec<_>>();
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let mut slot = plan.new_slot().unwrap();
        slot.write_input(&input, &bytes).unwrap();
        let (outputs, completion) = slot
            .attach(plan.bindings(), plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        let read = read_f32(&outputs.exported(&value).unwrap());
        assert!(completion.is_complete(), "{:?}: the read waited", device.backend());
        completion.wait().unwrap();
        assert_eq!(read, chained(&values, nodes), "{:?}", device.backend());
    }
}

/// S8: a standalone call's scratch comes from the prepared kernel's
/// invocation workspace, which reports it.
#[test]
fn standalone_scratch_is_charged_to_the_invocation_workspace() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(64).with_param("PARTS", 4).with_param("WIDTH", 32),
        )
        .unwrap();
        let before = kernel.invocation_workspace_bytes();
        let x = f32_tensor(&device, &[64], &exact_values(64));
        for _ in 0..2 {
            let value = kernel.call(split_sum::Args { x: &x }).unwrap().value;
            assert_eq!(read_f32(&value), [exact_values(64).iter().sum::<f32>()]);
            // Four f32 partials, reused by the second call.
            assert_eq!(kernel.invocation_workspace_bytes(), before + 16, "{:?}", device.backend());
        }
    }
}

fn zeroed(device: &Device, n: u64) -> Tensor {
    f32_tensor(device, &[n], &vec![0.0; n as usize])
}

/// S6: a point binding the same `&mut` tensor for every configuration
/// without an initializer is rejected before anything runs.
#[test]
fn tuning_rejects_shared_mutable_state_without_an_initializer() {
    for device in devices() {
        let n = 256u64;
        let mut state = zeroed(&device, n);
        let x = f32_tensor(&device, &[n], &exact_values(n as usize));
        let points = vec![TuningPoint {
            label: "rows".into(),
            weight: 1.0,
            class: None,
            rotation: vec![accumulate::Args { state: &mut state, x: &x }],
            initialize: None,
        }];
        match accumulate::native_tune(
            &device,
            &NativeSpecialization::new(),
            points,
            Validation::BitExact,
            search(2),
        ) {
            Err(TuneError::SharedMutableState { point, parameter }) => {
                assert_eq!(point, "rows");
                assert_eq!(parameter, "state");
            }
            Err(other) => panic!("{:?}: unexpected {other}", device.backend()),
            Ok(_) => panic!("{:?}: shared mutable state was tuned", device.backend()),
        }
    }
}

/// S6: validation compares `&mut` parameters, so a mapping parameter that
/// changes their bits is a misclassified-parameter defect; the initializer
/// makes every configuration start from the same state.
#[test]
fn tuning_validates_in_place_parameters_and_excludes_misclassified_ones() {
    for device in devices() {
        let n = 256u64;
        let mut state = zeroed(&device, n);
        let mut restore = state.clone();
        let zeros = vec![0u8; n as usize * 4];
        let initialize: TuningInitializer<'_> = Box::new(move || restore.write_from_host(&zeros));
        let x = f32_tensor(&device, &[n], &exact_values(n as usize));
        let points = vec![TuningPoint {
            label: "rows".into(),
            weight: 1.0,
            class: None,
            rotation: vec![accumulate::Args { state: &mut state, x: &x }],
            initialize: Some(initialize),
        }];
        let result = accumulate::native_tune(
            &device,
            &NativeSpecialization::new(),
            points,
            Validation::BitExact,
            search(2),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        // A BIAS 1 configuration is either validated, and then excluded as a
        // misclassified parameter, or ranked below the chosen configuration
        // and never validated; it is never chosen.
        for record in &result.configurations {
            match (record.configuration.params["BIAS"], &record.outcome) {
                (0, Outcome::Measured { .. }) | (1, Outcome::Measured { validated: false, .. }) => {}
                (1, Outcome::Excluded(Exclusion::MisclassifiedParameter { point, reference })) => {
                    assert_eq!(point, "rows");
                    assert_eq!(reference.params["BIAS"], 0);
                }
                (bias, outcome) => panic!("{:?}: BIAS {bias} gave {outcome:?}", device.backend()),
            }
        }
        assert_eq!(result.overall.params["BIAS"], 0);
    }
}

/// Static bindings are checked once by `bind_static`; a run checks only the
/// bindings it adds, and still rejects unbound, mismatched and illegally
/// aliased external ports before anything is encoded.
#[test]
fn external_bindings_are_checked_at_attach() {
    for device in devices() {
        let (m, n) = (2u64, 40u64);
        let scale =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let x = graph.port(Element::f32(), &[m, n]).unwrap();
        let doubled = graph
            .enqueue(&scale, scale_rows::WorkflowArgs { x: x.tensor().into(), factor: 2.0 })
            .unwrap()
            .value;
        graph.export(&doubled).unwrap();
        let plan = graph.seal().unwrap();
        let values = (0..m * n).map(|index| index as f32).collect::<Vec<_>>();
        let input = f32_tensor(&device, &[m, n], &values);
        let wrong = f32_tensor(&device, &[m, n + 1], &vec![0.0; (m * (n + 1)) as usize]);
        assert!(matches!(
            plan.bind_static(&[(&x, &wrong)]),
            Err(CallError::Workflow(seismic::WorkflowError::NativePortMismatch { port: 0 }))
        ));
        let bound = plan.bind_static(&[(&x, &input)]).unwrap();
        let mut slot = plan.new_slot().unwrap();
        for _ in 0..2 {
            let (outputs, completion) = slot
                .attach(bound.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            let expected = values.iter().map(|value| value * 2.0).collect::<Vec<_>>();
            assert_eq!(read_f32(&outputs.exported(&doubled).unwrap()), expected);
        }
        assert!(matches!(
            slot.attach(plan.bindings(), plan.new_outputs().unwrap()).err(),
            Some(CallError::Workflow(seismic::WorkflowError::NativePortUnbound { port: 0 }))
        ));
        let mut bindings = plan.bindings();
        bindings.set(&x, &wrong).unwrap();
        assert!(matches!(
            slot.attach(bindings, plan.new_outputs().unwrap()).err(),
            Some(CallError::Workflow(seismic::WorkflowError::NativePortMismatch { port: 0 }))
        ));

        // `accumulate` requires `state` and `x` to be disjoint; both are
        // external ports, so only the run's bindings can violate it.
        let accumulate = accumulate::native_for_device(
            &device,
            &NativeSpecialization::new().with_param("BIAS", 0).with_param("WIDTH", 32),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let mut state = graph.port(Element::f32(), &[n]).unwrap();
        let addend = graph.port(Element::f32(), &[n]).unwrap();
        graph
            .enqueue(
                &accumulate,
                accumulate::WorkflowArgs { state: state.tensor_mut().into(), x: addend.tensor().into() },
            )
            .unwrap();
        let plan = graph.seal().unwrap();
        let mut slot = plan.new_slot().unwrap();
        let shared = f32_tensor(&device, &[n], &vec![1.0; n as usize]);
        let mut bindings = plan.bindings();
        bindings.set(&state, &shared).unwrap();
        bindings.set(&addend, &shared).unwrap();
        match slot.attach(bindings, plan.new_outputs().unwrap()).err() {
            Some(CallError::Invocation(InvocationError::IllegalAliasing { first, second })) => {
                assert_eq!([first.as_str(), second.as_str()], ["state", "x"]);
            }
            other => panic!("{:?}: expected illegal aliasing, got {other:?}", device.backend()),
        }
        let separate = f32_tensor(&device, &[n], &vec![2.0; n as usize]);
        let mut bindings = plan.bindings();
        bindings.set(&state, &shared).unwrap();
        bindings.set(&addend, &separate).unwrap();
        let (_, completion) = slot
            .attach(bindings, plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        completion.wait().unwrap();
        assert_eq!(read_f32(&shared), vec![3.0; n as usize], "{:?}", device.backend());
    }
}

/// `gated_sum` by the portable body (the reference interpreter).
fn portable_gated_sum(values: &[f32]) -> f32 {
    use seismic_lang::{
        checked::{check_source, SourceFile, SourceSet},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        types::DType,
    };
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "conditional.seismic".into(),
        text: include_str!("../fixtures/conditional.seismic").into(),
    }]))
    .expect("the conditional fixture checks");
    let logical = module
        .entry(module.entry_named("gated_sum").unwrap(), &ElementBindings::default())
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let x = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![values.len()],
        values.iter().map(|value| f64::from(*value)).collect(),
    ));
    let outcome = interpreter.run(&[Arg::Tensor(x)]).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("gated_sum portable body failed: {failure}");
    }
    let result = outcome.results().next().expect("one result");
    let OutcomeValue::Tensor(sum) = result.value() else {
        panic!("gated_sum returns a tensor")
    };
    sum.read(0).unwrap() as f32
}

fn gated(device: &Device, small: u64) -> seismic::NativeKernel<gated_sum::Entry> {
    gated_sum::native_for_device(device, &NativeSpecialization::new().with_param("SMALL", small))
        .unwrap_or_else(|error| panic!("{:?}: SMALL {small}: {error}", device.backend()))
}

/// Exactly one of `gated_sum`'s two launches is active for every shape: the
/// call matches the portable body standalone and in a sealed graph, and a
/// launch-detail trace shows one device interval and one empty launch, by
/// declaration ordinal. At `N == SMALL` the inactive launch's grid would
/// divide by zero, and below `SMALL` it and the scratch size would underflow.
#[test]
fn conditional_launches_run_only_the_active_launch() {
    for device in devices() {
        // (N, SMALL, active launch ordinal). Vulkan forms group memory at
        // preparation, so SMALL = 2^20 (4 MiB staged) does not prepare there
        // (`inactive_launches_are_exempt_from_device_limits`).
        let cases = [(10u64, 64u64, 0usize), (64, 64, 0), (200, 64, 1), (200, 1_048_576, 0)];
        let cases = cases
            .into_iter()
            .filter(|(_, small, _)| device.backend() != BackendName::Vulkan || *small == 64);
        for (n, small, active) in cases {
            let values = exact_values(n as usize);
            let expected = portable_gated_sum(&values);
            let kernel = gated(&device, small);
            let x = f32_tensor(&device, &[n], &values);
            let trace = device.trace_submissions(TraceDetail::Launches).unwrap();
            let value = kernel.call(gated_sum::Args { x: &x }).unwrap().value;
            let submissions = trace.collect().unwrap();
            let case = format!("{:?} N {n} SMALL {small}", device.backend());
            assert_eq!(read_f32(&value), [expected], "{case}");
            assert_eq!(submissions.len(), 1, "{case}");
            let launches = &submissions[0].launches;
            assert_eq!(
                launches.iter().map(|launch| launch.launch).collect::<Vec<_>>(),
                [0, 1],
                "{case}: inactive launches keep their ordinal"
            );
            if device.backend() != BackendName::Cpu {
                assert!(launches[active].device.is_some(), "{case}");
                assert!(launches[1 - active].device.is_none(), "{case}");
            }

            let mut graph = device.native_graph();
            let input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
            let sum = graph
                .enqueue(&kernel, gated_sum::WorkflowArgs { x: input.tensor().into() })
                .unwrap()
                .value;
            graph.export(&sum).unwrap();
            let plan = graph.seal().unwrap();
            let mut slot = plan.new_slot().unwrap();
            let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            slot.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = slot
                .attach(plan.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(read_f32(&outputs.exported(&sum).unwrap()), [expected], "{case}: graph");
        }
    }
}

/// An inactive launch is not checked against device limits: the staged
/// launch's group memory (512 KiB here) exceeds every device's limit, which
/// is an error only when that launch is active. Vulkan fixes group memory
/// when the kernel is prepared, so there the staged launch's 4 MiB region is
/// refused at preparation, whether or not a call would activate it.
#[test]
fn inactive_launches_are_exempt_from_device_limits() {
    for device in devices() {
        let n = 1u64 << 17;
        let values = exact_values(n as usize);
        let x = f32_tensor(&device, &[n], &values);
        let value = gated(&device, 64).call(gated_sum::Args { x: &x }).unwrap().value;
        // Every partial sum of these values is exact, so any order agrees
        // with the portable body's.
        assert_eq!(read_f32(&value), [values.iter().sum::<f32>()], "{:?}", device.backend());
        if device.backend() == BackendName::Cpu {
            // The CPU route has no group-memory limit.
            continue;
        }
        if device.backend() == BackendName::Vulkan {
            let error = gated_sum::native_for_device(
                &device,
                &NativeSpecialization::new().with_param("SMALL", 1_048_576),
            )
            .err()
            .expect("a 4 MiB group-memory region is refused at preparation");
            assert!(
                error.to_string().contains("launch `gated_staged` needs")
                    && error.to_string().contains("the device allows"),
                "{error}"
            );
            continue;
        }
        match gated(&device, 1_048_576).call(gated_sum::Args { x: &x }) {
            Err(CallError::Execution(error)) => assert!(
                error.to_string().contains("launch `gated_staged` needs")
                    && error.to_string().contains("the device allows"),
                "{:?}: {error}",
                device.backend()
            ),
            Err(other) => panic!("{:?}: expected a device-limit error, got {other}", device.backend()),
            Ok(_) => panic!("{:?}: an active launch beyond the group-memory limit ran", device.backend()),
        }
    }
}

/// An inactive scratch buffer keeps its slot at the minimum charge, and its
/// size is not evaluated (it would underflow below `SMALL`).
#[test]
fn inactive_scratch_is_charged_the_minimum() {
    for device in devices() {
        let kernel = gated(&device, 64);
        let before = kernel.invocation_workspace_bytes();
        let call = |n: u64| {
            let values = exact_values(n as usize);
            let x = f32_tensor(&device, &[n], &values);
            let value = kernel.call(gated_sum::Args { x: &x }).unwrap().value;
            assert_eq!(read_f32(&value), [portable_gated_sum(&values)], "{:?} N {n}", device.backend());
        };
        call(10);
        assert_eq!(kernel.invocation_workspace_bytes(), before + 1, "{:?}", device.backend());
        call(200);
        assert_eq!(kernel.invocation_workspace_bytes(), before + 800, "{:?}", device.backend());

        let workspace = |n: u64| {
            let mut graph = device.native_graph();
            let input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
            let sum = graph
                .enqueue(&kernel, gated_sum::WorkflowArgs { x: input.tensor().into() })
                .unwrap()
                .value;
            graph.export(&sum).unwrap();
            graph.seal().unwrap().workspace_bytes()
        };
        assert!(workspace(10) < 800, "{:?}", device.backend());
        assert!(workspace(200) >= 800, "{:?}", device.backend());
    }
}

/// A reserved tensor backs only its leading committed rows. Recommitting
/// keeps them (device work bound afterwards reads them) and zero-fills the
/// rows it adds, including rows released by a shrink and backed again (CUDA
/// resizes its reserved address range in place, reusing granules).
#[test]
fn a_reserved_tensor_recommits_keeping_its_rows() {
    for device in devices() {
        let backend = device.backend();
        // 1 KiB rows: several 2 MiB CUDA granules are mapped, released and
        // mapped again.
        let (rows, n) = (8192u64, 256u64);
        let (grow, shrink, regrow) = (6000u64, 4u64, 5000u64);
        let row_values = |count: u64| {
            (0..count * n)
                .map(|index| (index % 97) as f32 * 0.5 + 1.0)
                .collect::<Vec<_>>()
        };
        let bytes = |values: &[f32]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let with_zeros = |values: &[f32], count: u64| {
            let mut all = values.to_vec();
            all.resize((count * n) as usize, 0.0);
            all
        };
        let reserved = Tensor::reserved(&device, Element::f32(), &[rows, n], 8).unwrap();
        assert_eq!(reserved.committed_rows(), 8, "{backend:?}");
        assert_eq!(reserved.resizes_in_place(), backend == BackendName::Cuda, "{backend:?}");
        assert!(reserved.read_to_host().is_err(), "{backend:?}: host access past the committed rows");
        let first = row_values(8);
        reserved.slice_leading(0, 8).unwrap().write_from_host(&bytes(&first)).unwrap();

        let grown = reserved.recommitted(grow).unwrap();
        drop(reserved);
        assert_eq!(grown.committed_rows(), grow, "{backend:?}");
        assert_eq!(read_f32(&grown.slice_leading(0, grow).unwrap()), with_zeros(&first, grow), "{backend:?}");
        grown
            .slice_leading(8, grow)
            .unwrap()
            .write_from_host(&bytes(&row_values(grow - 8)))
            .unwrap();

        let scale =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let x = graph.port(Element::f32(), &[8, n]).unwrap();
        let tripled = graph
            .enqueue(&scale, scale_rows::WorkflowArgs { x: x.tensor().into(), factor: 3.0 })
            .unwrap()
            .value;
        graph.export(&tripled).unwrap();
        let plan = graph.seal().unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&x, &grown.slice_leading(0, 8).unwrap()).unwrap();
        let (outputs, completion) = plan
            .new_slot()
            .unwrap()
            .attach(bindings, plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        completion.wait().unwrap();
        assert_eq!(
            read_f32(&outputs.exported(&tripled).unwrap()),
            first.iter().map(|value| value * 3.0).collect::<Vec<_>>(),
            "{backend:?}"
        );

        let shrunk = grown.recommitted(shrink).unwrap();
        drop(grown);
        assert_eq!(shrunk.committed_rows(), shrink, "{backend:?}");
        assert_eq!(shrunk.resizes_in_place(), backend == BackendName::Cuda, "{backend:?}");
        let regrown = shrunk.recommitted(regrow).unwrap();
        drop(shrunk);
        assert_eq!(
            read_f32(&regrown.slice_leading(0, regrow).unwrap()),
            with_zeros(&first[..(shrink * n) as usize], regrow),
            "{backend:?}: rows released and backed again read zero"
        );
    }
}
