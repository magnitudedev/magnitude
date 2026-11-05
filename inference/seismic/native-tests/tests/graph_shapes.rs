//! Graphs by shape (`seismic::Device::graph_replays`): runs of one plan on
//! every place and slot replay that plan's graphs, recorded at their first
//! use (on CUDA one for each run in flight at once, elsewhere one) and
//! updated to the storage each run binds. Run on every available device.

use seismic::{
    BackendName, Device, Element, GraphReplays, NativeGraphClassSlice, NativeGraphFamily, NativeGraphMetadata,
    NativeSpecialization, Tensor,
};
use seismic_native_tests::{gated_sum, scale_rows};

fn devices() -> Vec<Device> {
    seismic_native_tests::devices(&[
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
    ])
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

/// A chain over `PLACES` places, each run on a slot of its own and binding
/// rows of its own, submitted as one sequence three times, each chain's
/// outputs in fresh storage. On CUDA each slot instantiates its graph as it
/// is created and every chain updates each place's graph, idle by then, to
/// its storage, recording none; elsewhere one graph serves the chain,
/// recorded at its first run and updated at every run binding other storage
/// than it holds. A later chain records nothing. Every place's result is its own, bit for bit, as a graph
/// recorded for it alone would give: the exact products of its rows.
#[test]
fn a_chains_places_replay_their_shapes_graphs_bit_for_bit() {
    const PLACES: usize = 7;
    let (m, n) = (2u64, 8u64);
    for device in devices() {
        let backend = device.backend();
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[m, n]).unwrap();
        let landing = graph.port(Element::f32(), &[m, n]).unwrap();
        let scaled = graph
            .enqueue(&kernel, scale_rows::WorkflowArgs { x: input.tensor().into(), factor: 2.0 })
            .unwrap()
            .value;
        graph.store(&scaled, &landing).unwrap();
        let after = graph
            .enqueue(&kernel, scale_rows::WorkflowArgs { x: landing.tensor().into(), factor: 3.0 })
            .unwrap()
            .value;
        graph.export(&after).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let created = device.graph_replays();
        let mut slots = (0..PLACES)
            .map(|_| family.new_slot(&[family.upload_bytes()]).unwrap())
            .collect::<Vec<_>>();
        let created = device.graph_replays().since(created);
        // Each place's input rows, and the rows of one tensor it lands in.
        let inputs = (0..PLACES)
            .map(|place| (0..m * n).map(|value| (100 * place as u64 + value) as f32).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let sources = inputs
            .iter()
            .map(|values| Tensor::from_host(&device, Element::f32(), &[m, n], &f32_bytes(values)).unwrap())
            .collect::<Vec<_>>();
        let rows = PLACES as u64 * m;
        let landed = Tensor::from_host(&device, Element::f32(), &[rows, n], &f32_bytes(&vec![-1.0; (rows * n) as usize]))
            .unwrap();
        let chain = |slots: &mut [seismic::NativeGraphFamilySlot]| {
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
            let mut exported = Vec::new();
            for (place, slot) in slots.iter_mut().enumerate() {
                let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
                let mut bindings = plan.bindings();
                bindings.set(&input, &sources[place]).unwrap();
                let at = place as u64 * m;
                bindings.set(&landing, &landed.slice_leading(at, at + m).unwrap()).unwrap();
                let mut active = slot.activate(&plan).unwrap();
                let outputs = active.attach(bindings, outputs).unwrap().queue(&mut sequence).unwrap();
                exported.push(outputs.exported(&after).unwrap());
            }
            sequence.submit().unwrap().wait().unwrap();
            exported
        };
        let mut counts = Vec::new();
        let mut exported = Vec::new();
        for _ in 0..3 {
            let before = device.graph_replays();
            exported.push(chain(&mut slots));
            counts.push(device.graph_replays().since(before));
        }
        for exported in exported {
            for (place, result) in exported.iter().enumerate() {
                let expected = inputs[place].iter().map(|value| 3.0 * (2.0 * value)).collect::<Vec<_>>();
                assert_eq!(read_f32(result), expected, "{backend:?}: place {place}");
            }
        }
        let doubled = inputs.iter().flatten().map(|value| 2.0 * value).collect::<Vec<_>>();
        assert_eq!(read_f32(&landed), doubled, "{backend:?}: every place's landed rows");
        let places = PLACES as u64;
        let replayed = |recorded, updated| GraphReplays { recorded, updated };
        match backend {
            // A graph per place, instantiated with its slot and updated to
            // each chain's storage.
            BackendName::Cuda => {
                assert_eq!(created, replayed(places, 0), "{backend:?}: the slots' graphs");
                assert_eq!(counts, [replayed(0, places); 3], "{backend:?}: every chain records nothing");
            }
            // A Vulkan division holds its launches' addresses: recorded
            // bound to place 0, updated at the other places, and at every
            // place of each later chain, which begins where the one before
            // ended. It is formed again once its launches' times are
            // measured, by the second chain.
            BackendName::Vulkan => {
                assert_eq!(counts[0], replayed(1, places - 1), "{backend:?}: the first chain");
                let again = u64::from(backend == BackendName::Vulkan);
                assert!(counts[1].recorded <= again && counts[1].updated == places, "{backend:?}: the second chain {:?}", counts[1]);
                assert_eq!(counts[2], replayed(0, places), "{backend:?}: the third chain records nothing");
            }
            // A Metal schedule is host data, not a recording, and the CPU
            // records no graph.
            BackendName::Metal | BackendName::Cpu => {
                assert!(counts.iter().all(|counts| *counts == GraphReplays::default()), "{backend:?}: {counts:?}");
            }
        }
    }
}

/// Serving's mix, interleaved: decode steps of two row classes on two
/// slots, and a prompt chain of another class over seven places, all of one
/// layout's classes. Each shape is recorded at its first use and never
/// again: a second round records nothing. On CUDA every class and place
/// replays the layout's graphs, one for each of the chain's runs in flight
/// at once; on Vulkan each class its plan's. Every result is its own, bit
/// for bit.
/// Steps of a served load, each submitted while the steps before it run:
/// a step makes a launched plan of rows its request brings (a conditioning
/// overlay's, a boundary capture's) and a slot of it with that work in
/// flight, and queues its run between two runs of a graphed plan whose slot
/// was made before any flight. On CUDA a launched plan's slot instantiates
/// nothing, so making it with work queued breaks no invariant, and its run
/// launches call by call in its place among the replayed graphs; elsewhere
/// it records at a shape's first use as any plan. Every step's rows are its
/// own, bit for bit.
#[test]
fn a_launched_plans_slots_are_made_and_run_while_earlier_steps_are_in_flight() {
    // Enough device work per graphed run that the steps before each one
    // are still running as it is made.
    const BUSY_NODES: usize = 16;
    let (busy_rows, n) = (4096u64, 256u64);
    for device in devices() {
        let backend = device.backend();
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1)).unwrap();
        let mut graph = device.native_graph();
        let busy_input = graph.port(Element::f32(), &[busy_rows, n]).unwrap();
        let mut value = graph.enqueue(&kernel, scale_rows::WorkflowArgs { x: busy_input.tensor().into(), factor: 1.0 }).unwrap().value;
        for _ in 1..BUSY_NODES {
            value = graph.enqueue(&kernel, scale_rows::WorkflowArgs { x: (&value).into(), factor: 1.0 }).unwrap().value;
        }
        graph.export(&value).unwrap();
        let busy = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[busy.clone()]).unwrap();
        // Made before any flight, as a load makes its slots.
        let mut busy_slot = family.new_slot(&[family.upload_bytes()]).unwrap();
        let mut busy_outputs = Some(family.new_output_slot().unwrap());
        let busy_x = Tensor::zeros(&device, Element::f32(), &[busy_rows, n]).unwrap();
        let mut busy_run = |sequence: &mut seismic::NativeGraphSequence| {
            let outputs = busy_outputs.take().unwrap().activate(&busy).unwrap();
            let mut bindings = busy.bindings();
            bindings.set(&busy_input, &busy_x).unwrap();
            let outputs = busy_slot.activate(&busy).unwrap().attach(bindings, outputs).unwrap().queue(sequence).unwrap();
            busy_outputs = Some(outputs.recycle().unwrap());
        };
        // Steps already in flight before the first launched slot is made.
        let mut completions = (0..8)
            .map(|_| {
                let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
                busy_run(&mut sequence);
                sequence.submit().unwrap()
            })
            .collect::<Vec<_>>();
        let mut steps = Vec::new();
        for (step, rows) in [1u64, 3, 7, 1, 64, 3].into_iter().enumerate() {
            let mut graph = device.native_graph();
            let source = graph.port(Element::f32(), &[rows, n]).unwrap();
            let destination = graph.port(Element::f32(), &[rows, n]).unwrap();
            let scaled = graph
                .enqueue(&kernel, scale_rows::WorkflowArgs { x: source.tensor().into(), factor: 2.0 })
                .unwrap()
                .value;
            graph.store(&scaled, &destination).unwrap();
            let plan = graph.seal_launched().unwrap();
            let values = (0..rows * n).map(|value| (step as u64 * 100_000 + value) as f32).collect::<Vec<_>>();
            let bound = Tensor::from_host(&device, Element::f32(), &[rows, n], &f32_bytes(&values)).unwrap();
            let landed = Tensor::zeros(&device, Element::f32(), &[rows, n]).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(&source, &bound).unwrap();
            bindings.set(&destination, &landed).unwrap();
            // Made with the steps before this one in flight.
            let mut slot = plan.new_slot().unwrap();
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
            busy_run(&mut sequence);
            slot.attach(bindings, plan.new_outputs().unwrap()).unwrap().queue(&mut sequence).unwrap();
            busy_run(&mut sequence);
            completions.push(sequence.submit().unwrap());
            steps.push((values, landed));
        }
        for completion in completions {
            completion.wait().unwrap();
        }
        for (values, landed) in &steps {
            assert_eq!(read_f32(landed), values.iter().map(|value| 2.0 * value).collect::<Vec<_>>(), "{backend:?}");
        }
    }
}

#[test]
fn interleaved_classes_slots_and_places_record_each_shape_once() {
    const PLACES: usize = 7;
    let n = 8u64;
    let classes = [1u64, 2, 4];
    for device in devices() {
        let backend = device.backend();
        let mut template = NativeGraphMetadata::new_template(backend);
        let input = template.port_with_class_extent(Element::f32(), &[4, n], 0, "M").unwrap();
        let landing = template.port_with_class_extent(Element::f32(), &[4, n], 0, "M").unwrap();
        let scaled = template
            .enqueue::<scale_rows::Entry>(&[], &[("M", 4), ("N", n)], scale_rows::WorkflowArgs { x: input.tensor().into(), factor: 2.0 })
            .unwrap()
            .value;
        template.store(&scaled, &landing).unwrap();
        let layout = template
            .seal_template()
            .unwrap()
            .certify(&[NativeGraphClassSlice::new().dimension("M", classes)])
            .unwrap();
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1)).unwrap();
        let sealed = classes
            .iter()
            .map(|&m| {
                let mut graph = device.native_graph_with_layout(&layout);
                let input = graph.port(Element::f32(), &[m, n]).unwrap();
                let landing = graph.port(Element::f32(), &[m, n]).unwrap();
                let scaled = graph
                    .enqueue(&kernel, scale_rows::WorkflowArgs { x: input.tensor().into(), factor: 2.0 })
                    .unwrap()
                    .value;
                graph.store(&scaled, &landing).unwrap();
                (graph.seal().unwrap(), input, landing)
            })
            .collect::<Vec<_>>();
        let plans = sealed.iter().map(|(plan, ..)| plan.clone()).collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).unwrap();
        let mut slots = (0..PLACES).map(|_| family.new_slot(&[family.upload_bytes()]).unwrap()).collect::<Vec<_>>();
        // A run of class `class` on slot `slot`, its rows the input's
        // doubled into the returned tensor.
        let mut runs = Vec::new();
        let mut queue = |sequence: &mut seismic::NativeGraphSequence, slots: &mut [seismic::NativeGraphFamilySlot], class: usize, slot: usize, seed: u64| {
            let (plan, input, landing) = &sealed[class];
            let m = classes[class];
            let values = (0..m * n).map(|value| (seed * 100 + value) as f32).collect::<Vec<_>>();
            let source = Tensor::from_host(&device, Element::f32(), &[m, n], &f32_bytes(&values)).unwrap();
            let landed = Tensor::zeros(&device, Element::f32(), &[m, n]).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(input, &source).unwrap();
            bindings.set(landing, &landed).unwrap();
            let outputs = family.new_output_slot().unwrap().activate(plan).unwrap();
            let mut active = slots[slot].activate(plan).unwrap();
            active.attach(bindings, outputs).unwrap().queue(sequence).unwrap();
            runs.push((values, landed));
        };
        let mut rounds = Vec::new();
        for round in 0..2u64 {
            let before = device.graph_replays();
            // Decode steps of classes 1 and 2 alternating two slots, each
            // its own submission as a step and its successor are.
            for step in 0..6u64 {
                let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
                queue(&mut sequence, &mut slots, (step % 2) as usize, (step % 2) as usize, round * 1000 + step);
                sequence.submit().unwrap().wait().unwrap();
            }
            // A prompt chain of class 4 over every place, then decode again.
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
            for place in 0..PLACES {
                queue(&mut sequence, &mut slots, 2, place, round * 1000 + 100 + place as u64);
            }
            sequence.submit().unwrap().wait().unwrap();
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
            queue(&mut sequence, &mut slots, 0, 0, round * 1000 + 200);
            sequence.submit().unwrap().wait().unwrap();
            rounds.push(device.graph_replays().since(before));
        }
        for (values, landed) in &runs {
            assert_eq!(read_f32(landed), values.iter().map(|value| 2.0 * value).collect::<Vec<_>>(), "{backend:?}");
        }
        let first = match backend {
            // Instantiated with the slots, before the rounds.
            BackendName::Cuda => 0,
            // A plan's graph per class.
            BackendName::Vulkan => classes.len() as u64,
            BackendName::Metal | BackendName::Cpu => 0,
        };
        assert_eq!(rounds[0].recorded, first, "{backend:?}: recorded at each shape's first use");
        assert_eq!(rounds[1].recorded, 0, "{backend:?}: serving records nothing once each shape has run");
    }
}

/// Classes of one layout whose calls make different launches: `gated_sum`
/// stages at most `SMALL` elements and mirrors more, so its class of 32
/// makes the staged launch and its class of 128 the mirrored one. On CUDA
/// both replay the layout's one graph, a node for each launch either
/// makes, each run enabling its own: every slot instantiates one graph,
/// whose nodes are the two launches, and serving, alternating the classes
/// on every slot, records nothing. Every sum is its own, bit for bit.
#[test]
fn a_layouts_classes_making_different_launches_replay_one_graph() {
    const SLOTS: usize = 3;
    let classes = [32u64, 128];
    for device in devices() {
        let backend = device.backend();
        let mut template = NativeGraphMetadata::new_template(backend);
        let input = template.port_with_class_extent(Element::f32(), &[128], 0, "N").unwrap();
        let sum = template
            .enqueue::<gated_sum::Entry>(&[], &[("N", 128)], gated_sum::WorkflowArgs { x: input.tensor().into() })
            .unwrap()
            .value;
        template.export(&sum).unwrap();
        let layout = template
            .seal_template()
            .unwrap()
            .certify(&[NativeGraphClassSlice::new().dimension("N", classes)])
            .unwrap();
        let kernel = gated_sum::native_for_device(&device, &NativeSpecialization::new().with_param("SMALL", 64)).unwrap();
        let sealed = classes
            .iter()
            .map(|&n| {
                let mut graph = device.native_graph_with_layout(&layout);
                let input = graph.port(Element::f32(), &[n]).unwrap();
                let sum = graph.enqueue(&kernel, gated_sum::WorkflowArgs { x: input.tensor().into() }).unwrap().value;
                graph.export(&sum).unwrap();
                (graph.seal().unwrap(), input, sum)
            })
            .collect::<Vec<_>>();
        let plans = sealed.iter().map(|(plan, ..)| plan.clone()).collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).unwrap();
        let created = device.graph_replays();
        let mut slots = (0..SLOTS).map(|_| family.new_slot(&[family.upload_bytes()]).unwrap()).collect::<Vec<_>>();
        let created = device.graph_replays().since(created);
        let served = device.graph_replays();
        for round in 0..4u64 {
            for (at, slot) in slots.iter_mut().enumerate() {
                let class = (round as usize + at) % classes.len();
                let (plan, input, sum) = &sealed[class];
                let values = (0..classes[class]).map(|value| (round * 7 + at as u64 + value) as f32).collect::<Vec<_>>();
                let source = Tensor::from_host(&device, Element::f32(), &[classes[class]], &f32_bytes(&values)).unwrap();
                let mut bindings = plan.bindings();
                bindings.set(input, &source).unwrap();
                let outputs = family.new_output_slot().unwrap().activate(plan).unwrap();
                let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
                let outputs = slot.activate(plan).unwrap().attach(bindings, outputs).unwrap().queue(&mut sequence).unwrap();
                sequence.submit().unwrap().wait().unwrap();
                let expected = values.iter().sum::<f32>();
                assert_eq!(read_f32(&outputs.exported(sum).unwrap()), [expected], "{backend:?}: round {round}, slot {at}, N {}", classes[class]);
            }
        }
        let served = device.graph_replays().since(served);
        if backend == BackendName::Cuda {
            // One graph of the layout for each slot, both launches its nodes.
            assert_eq!(created.recorded, SLOTS as u64, "{backend:?}: the slots' graphs");
            assert_eq!(family.slot_graphs().graphs, 1, "{backend:?}: a slot holds the layout's one graph");
            assert_eq!(family.slot_graphs().launches, 2, "{backend:?}: the graph's nodes are both classes' launches");
            assert_eq!(served.recorded, 0, "{backend:?}: serving records nothing");
        }
    }
}

/// What a device states its recorded graphs hold
/// (`seismic::Device::graph_cost`), printed for the recorded machines'
/// fixtures: pooled driver memory on CUDA, command memory and argument
/// chunks on Vulkan.
#[test]
fn the_device_states_its_graph_cost() {
    for device in devices() {
        let cost = device.graph_cost();
        eprintln!("graph cost {:?}: {cost:?}", device.backend());
        match device.backend() {
            BackendName::Cuda => assert!(matches!(cost, seismic::GraphCost::Pooled { .. }), "{cost:?}"),
            BackendName::Vulkan => assert!(matches!(cost, seismic::GraphCost::Chunked { .. }), "{cost:?}"),
            BackendName::Metal | BackendName::Cpu => assert_eq!(cost, seismic::GraphCost::None),
        }
    }
}
