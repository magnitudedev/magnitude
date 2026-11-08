//! Span marks of a sealed graph (`NativeGraph::span_mark`): a sampled run
//! stamps the device's time at each mark as it passes it, so the device time
//! of the calls between two marks is read once its work completes
//! (`NativeSpans::seconds`); a mark divides no graph; a backend that times no
//! span (Metal, the CPU) refuses a sampled run and runs an unsampled one as
//! ever.

use seismic::{BackendName, Element, NativeGraphFamily, NativeSpecialization, Tensor};
use seismic_native_tests::scale_rows;

/// A graph of `first` then `second` calls scaling a large row, marked at
/// its start, between the two and at its end. The mark inside is a node of
/// the one graph, on every run: sampled, each span is its part's device time,
/// the whole their sum, and twice the calls read twice the time; unsampled
/// before and after, it runs as any graph does. A backend without span timing
/// refuses the sampled run.
#[test]
fn a_sampled_run_reads_the_device_time_between_its_marks() {
    let (rows, width) = (1u64, 1u64 << 20);
    let (first, second) = (8usize, 16usize);
    for device in seismic_native_tests::every_device() {
        let backend = device.backend();
        let kernel = scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1)).unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[rows, width]).unwrap();
        let mut value = input.tensor().clone();
        let mut marks = vec![graph.span_mark()];
        for calls in [first, second] {
            for _ in 0..calls {
                value = graph.enqueue(&kernel, scale_rows::WorkflowArgs { x: (&value).into(), factor: 1.0 }).unwrap().value;
            }
            marks.push(graph.span_mark());
        }
        assert_eq!(marks, [0, 1, 2]);
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()], 1).unwrap();
        assert_eq!(family.graphs().decode.graphs, 1, "{backend:?}: a mark adds no graph");
        let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();
        let tensor = Tensor::zeros(&device, Element::f32(), &[rows, width]).unwrap();
        let run = |sampled: bool, slot: &mut seismic::NativeGraphFamilySlot| {
            let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(&input, &tensor).unwrap();
            let mut active = slot.activate(&plan).unwrap();
            let ready = active.attach(bindings, outputs).unwrap();
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
            let spans = match sampled {
                true => Some(ready.queue_sampled(&mut sequence)),
                false => {
                    ready.queue(&mut sequence).unwrap();
                    None
                }
            };
            (spans, sequence)
        };
        // Unsampled: a run as any, its launch costs learned.
        for _ in 0..2 {
            let (_, sequence) = run(false, &mut slot);
            sequence.submit().unwrap().wait().unwrap();
        }
        for round in 0..2 {
            let (spans, sequence) = run(true, &mut slot);
            let spans = match (backend, spans.expect("sampled")) {
                (BackendName::Cuda | BackendName::Vulkan, sampled) => sampled.unwrap().1,
                (_, refused) => {
                    assert!(refused.is_err(), "{backend:?} times no span");
                    break;
                }
            };
            sequence.submit().unwrap().wait().unwrap();
            let (part, rest, whole) = (spans.seconds(0, 1).unwrap(), spans.seconds(1, 2).unwrap(), spans.seconds(0, 2).unwrap());
            eprintln!("{backend:?} round {round}: {first} calls {:.1} µs, {second} calls {:.1} µs, both {:.1} µs", part * 1e6, rest * 1e6, whole * 1e6);
            assert!(part > 0. && rest > 0., "{backend:?}: {part} {rest}");
            assert!((whole - (part + rest)).abs() <= 1e-6 + 1e-3 * whole, "{backend:?}: {whole} against {part} + {rest}");
            let ratio = rest / part;
            assert!((1.7..2.3).contains(&ratio), "{backend:?}: twice the calls read {ratio} times the time");
            // Unsampled after sampling: the marks back at rest.
            let (_, sequence) = run(false, &mut slot);
            sequence.submit().unwrap().wait().unwrap();
        }
    }
}
