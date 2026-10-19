//! Timeline nodes of a sealed graph (`seismic::Timeline`): a gate holds the
//! work after it until the host raises its timeline to the run's value, and
//! a signal raises its timeline once the work before it has completed. Run
//! on every available device; the CPU backend runs a submission on the
//! submitting thread, so its gate blocks that thread until another opens it.

use seismic::{
    Availability, BackendName, Device, Devices, Element, NativeGraphFamily, NativeGraphMetadata,
    NativeSpecialization, Tensor, Timeline, TimelineNode,
};
use seismic_native_tests::scale_rows;
use std::time::Duration;

fn devices() -> Vec<Device> {
    let devices = Devices::discover(&[]).expect("device discovery");
    let topology = devices.topology();
    [
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
    ]
    .into_iter()
    .filter(|backend| {
        topology.executors().iter().any(|device| {
            device.backend() == *backend && matches!(device.availability(), Availability::Available)
        })
    })
    .map(|backend| {
        devices
            .open_backend(backend)
            .expect("available device opens")
    })
    .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// Two reads of one host-shared input around a gate: the read before it
/// sees what the input held at submission, the read after it what the host
/// wrote before opening the gate; the signal after both is reached only
/// once the gate opened and the work completed.
#[test]
fn a_gate_holds_the_work_after_it_until_the_host_opens_it() {
    let (m, n) = (1u64, 8u64);
    for device in devices() {
        let backend = device.backend();
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[m, n]).unwrap();
        let before = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 1.0,
                },
            )
            .unwrap()
            .value;
        let gate = graph.timeline(TimelineNode::Wait);
        let after = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 2.0,
                },
            )
            .unwrap()
            .value;
        let signal = graph.timeline(TimelineNode::Signal);
        graph.export(&before).unwrap();
        graph.export(&after).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(1).unwrap();
        let output_slot = family.new_output_slot().unwrap();

        let first = (0..n).map(|value| value as f32).collect::<Vec<_>>();
        let second = (0..n).map(|value| 100.0 + value as f32).collect::<Vec<_>>();
        let shared = Tensor::shared(&device, Element::f32(), &[m, n]).unwrap();
        // SAFETY: no device work uses the tensor yet.
        unsafe { shared.write_shared(0, &f32_bytes(&first)).unwrap() };
        let (opened, done) = (Timeline::new(&device).unwrap(), Timeline::new(&device).unwrap());
        // Two runs: the second's values are past the first's.
        for step in 1..=2u64 {
            let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(&input, &shared).unwrap();
            bindings.set_timeline(gate, &opened, step).unwrap();
            bindings.set_timeline(signal, &done, step).unwrap();
            let mut active = slot.activate(&plan).unwrap();
            let ready = active.attach(bindings, outputs).unwrap();
            let opener = {
                let (opened, done, shared, second) =
                    (opened.clone(), done.clone(), shared.clone(), second.clone());
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(50));
                    // The device holds at the gate: nothing after it ran.
                    let held = !done.reached(step);
                    // SAFETY: the work after the gate, which reads the
                    // input, waits until the gate opens; the work before it
                    // read the input 50 ms ago.
                    unsafe { shared.write_shared(0, &f32_bytes(&second)).unwrap() };
                    opened.signal(step).unwrap();
                    held
                })
            };
            let (outputs, completion) = ready.submit().unwrap();
            assert!(done.wait(step, Duration::from_secs(10)), "{backend:?}: signal");
            completion.wait().unwrap();
            assert!(opener.join().unwrap(), "{backend:?}: the gate held the device");
            assert_eq!(
                read_f32(&outputs.exported(&before).unwrap()),
                if step == 1 { first.clone() } else { second.clone() },
                "{backend:?} step {step}: before the gate"
            );
            assert_eq!(
                read_f32(&outputs.exported(&after).unwrap()),
                second.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
                "{backend:?} step {step}: after the gate"
            );
            drop(outputs);
        }
        drop(output_slot);
    }
}

/// A gate the host opened before the device reaches it costs the run
/// nothing but its node: the run completes without the host's help.
#[test]
fn an_open_gate_lets_the_run_through() {
    let (m, n) = (2u64, 4u64);
    for device in devices() {
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[m, n]).unwrap();
        let gate = graph.timeline(TimelineNode::Wait);
        let value = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 3.0,
                },
            )
            .unwrap()
            .value;
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(1).unwrap();
        let values = (0..m * n).map(|value| value as f32).collect::<Vec<_>>();
        let tensor = Tensor::from_host(&device, Element::f32(), &[m, n], &f32_bytes(&values))
            .unwrap();
        let opened = Timeline::new(&device).unwrap();
        opened.signal(7).unwrap();
        let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&input, &tensor).unwrap();
        bindings.set_timeline(gate, &opened, 5).unwrap();
        let mut active = slot.activate(&plan).unwrap();
        let (outputs, completion) = active.attach(bindings, outputs).unwrap().submit().unwrap();
        completion.wait().unwrap();
        assert_eq!(
            read_f32(&outputs.exported(&value).unwrap()),
            values.iter().map(|value| 3.0 * value).collect::<Vec<_>>(),
            "{:?}",
            device.backend()
        );
    }
}

/// A run's every timeline node must be bound, to a timeline of its device.
#[test]
fn a_run_binds_every_timeline_node() {
    let device = Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let kernel =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let mut graph = device.native_graph();
    let input = graph.port(Element::f32(), &[1, 4]).unwrap();
    let gate = graph.timeline(TimelineNode::Wait);
    let value = graph
        .enqueue(
            &kernel,
            scale_rows::WorkflowArgs {
                x: input.tensor().into(),
                factor: 1.0,
            },
        )
        .unwrap()
        .value;
    graph.export(&value).unwrap();
    let plan = graph.seal().unwrap();
    let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
    let mut slot = family.new_slot(1).unwrap();
    let tensor = Tensor::zeros(&device, Element::f32(), &[1, 4]).unwrap();
    let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
    let mut bindings = plan.bindings();
    bindings.set(&input, &tensor).unwrap();
    let mut active = slot.activate(&plan).unwrap();
    assert!(active.attach(bindings, outputs).is_err());
    let mut bindings = plan.bindings();
    let timeline = Timeline::new(&device).unwrap();
    bindings.set_timeline(gate, &timeline, 1).unwrap();
    assert!(bindings.set_timeline(gate, &timeline, 1).is_err());
}

/// A plan with timeline nodes holds the host heap its topology plans.
#[test]
fn timeline_nodes_are_in_the_planned_host_heap() {
    let device = Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let kernel =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let mut graph = device.native_graph();
    let mut metadata = NativeGraphMetadata::new(device.backend());
    let input = graph.port(Element::f32(), &[1, 4]).unwrap();
    let planned_input = metadata.port(Element::f32(), &[1, 4]).unwrap();
    graph.timeline(TimelineNode::Wait);
    metadata.timeline(TimelineNode::Wait);
    let value = graph
        .enqueue(
            &kernel,
            scale_rows::WorkflowArgs {
                x: input.tensor().into(),
                factor: 1.0,
            },
        )
        .unwrap()
        .value;
    let planned = metadata
        .enqueue::<scale_rows::Entry>(
            &[],
            &[("M", 1), ("N", 4)],
            scale_rows::WorkflowArgs {
                x: planned_input.tensor().into(),
                factor: 1.0,
            },
        )
        .unwrap()
        .value;
    graph.timeline(TimelineNode::Signal);
    metadata.timeline(TimelineNode::Signal);
    graph.export(&value).unwrap();
    metadata.export(&planned).unwrap();
    assert_eq!(graph.seal().unwrap().host_heap(), metadata.plan_heap());
}
