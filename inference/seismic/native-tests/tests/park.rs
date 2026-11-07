//! A call that parks (`&await`, `seismic::timeline::Park`): the host answers it
//! while it waits, on every backend, and a park node pairs with exactly such
//! a call. Where the backend's devices see host writes while work runs, the
//! call's kernel waits for the answer itself and the park node binds no
//! operation; elsewhere (Vulkan) the park is a submission boundary: the
//! device reaches the park's value, taken when its run is queued, once the
//! work before it completed, and the call runs once the host raised the
//! timeline past it. A run that leaves the call's launch out takes none.

use seismic::{
    BackendName, Element, NativeGraphFamily, NativeSpecialization, Tensor,
    Timeline,
};
use seismic_native_tests::{parked_scale, scale_rows};
use std::time::Duration;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[test]
fn the_host_answers_a_parked_call() {
    let n = 8u64;
    for device in seismic_native_tests::devices(&[BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]) {
        let backend = device.backend();
        let kernel = parked_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
        let mut graph = device.native_graph();
        let answer = graph.port(Element::i32(), &[1]).unwrap();
        let skip = graph.port(Element::i32(), &[1]).unwrap();
        let input = graph.port(Element::f32(), &[n]).unwrap();
        let park = graph.timeline(seismic::timeline::Park);
        let value = graph
            .enqueue(
                &kernel,
                parked_scale::WorkflowArgs {
                    answer: answer.tensor().into(),
                    skip: skip.tensor().into(),
                    x: input.tensor().into(),
                },
            )
            .unwrap()
            .value;
        let after = graph.timeline(seismic::timeline::Signal);
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();

        let values = (0..n).map(|value| value as f32).collect::<Vec<_>>();
        let x = Tensor::from_host(&device, Element::f32(), &[n], &f32_bytes(&values)).unwrap();
        let word = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        let skipped = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        // Three runs, the second leaving the call out: it takes no park
        // value, and the third's lies past the first's.
        let (parks, done) = (
            Timeline::<seismic::timeline::Parks>::new(&device).unwrap(),
            Timeline::<seismic::timeline::Reached>::new(&device).unwrap(),
        );
        let mut last = 0;
        for step in 1..=3u64 {
            let skip_step = step == 2;
            // SAFETY: no device work uses the words yet.
            unsafe {
                word.write_shared(0, &[0; 4]).unwrap();
                skipped.write_shared(0, &i32::from(skip_step).to_le_bytes()).unwrap();
            }
            let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(&answer, &word).unwrap();
            bindings.set(&skip, &skipped).unwrap();
            bindings.set(&input, &x).unwrap();
            bindings.set_park(park, &parks).unwrap();
            bindings.set_timeline(after, &done, step).unwrap();
            let mut active = slot.activate(&plan).unwrap();
            let ready = active.attach(bindings, outputs).unwrap();
            let (outputs, completion) = ready.submit().unwrap();
            let parked = outputs.park(park);
            assert_eq!(
                parked.is_some(),
                !backend.parks_in_kernels() && !skip_step,
                "{backend:?} step {step}: a park takes a value where it binds operations and its call runs"
            );
            // The call waits for the host's answer: on a backend whose
            // kernels park, from its submission; elsewhere once the work
            // before it completed, and it holds there.
            if let Some(value) = parked {
                assert!(value > last, "{backend:?} step {step}: park values rise");
                last = value;
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while !parks.reached(value).unwrap() {
                    assert!(std::time::Instant::now() < deadline, "{backend:?} step {step}: the device parks");
                    std::thread::yield_now();
                }
                assert!(!done.reached(step).unwrap(), "{backend:?} step {step}: the park held the call");
            }
            word.store_release(0, 2).unwrap();
            if let Some(value) = parked {
                parks.answer(value, seismic::ParkAnswer { by: seismic::AnsweredBy::Arrival, park: "test", layer: 0 }).unwrap();
            }
            assert!(done.wait(step, Duration::from_secs(10)).unwrap(), "{backend:?} step {step}: the run ends");
            completion.wait().unwrap();
            if skip_step {
                continue;
            }
            assert_eq!(
                outputs
                    .exported(&value)
                    .unwrap()
                    .read_to_host()
                    .unwrap()
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                values.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
                "{backend:?} step {step}: the call read the host's answer"
            );
        }
    }
}

/// A publication (`seismic::timeline::Publish`) follows a call, which it
/// publishes. Where parks are submission boundaries it takes the park
/// timeline's next value as its run is queued, the run's output as a park's
/// is, and the device reaches it once the call completed, waiting for
/// nothing; where kernels park it takes none. One before any call is
/// refused.
#[test]
fn a_publication_ends_a_submission_after_its_call_and_waits_for_nothing() {
    let n = 8u64;
    for device in seismic_native_tests::devices(&[BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]) {
        let backend = device.backend();
        let rows = scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1)).unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[1, 4]).unwrap();
        graph.timeline(seismic::timeline::Publish);
        let call = scale_rows::WorkflowArgs { x: input.tensor().into(), factor: 1.0 };
        assert!(graph.enqueue(&rows, call).is_err(), "{backend:?}: a publication before any call");

        let kernel = parked_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
        let mut graph = device.native_graph();
        let published = graph.port(Element::f32(), &[1, 4]).unwrap();
        let answer = graph.port(Element::i32(), &[1]).unwrap();
        let skip = graph.port(Element::i32(), &[1]).unwrap();
        let input = graph.port(Element::f32(), &[n]).unwrap();
        let scaled = graph
            .enqueue(&rows, scale_rows::WorkflowArgs { x: published.tensor().into(), factor: 3.0 })
            .unwrap()
            .value;
        let publication = graph.timeline(seismic::timeline::Publish);
        let park = graph.timeline(seismic::timeline::Park);
        let value = graph
            .enqueue(
                &kernel,
                parked_scale::WorkflowArgs {
                    answer: answer.tensor().into(),
                    skip: skip.tensor().into(),
                    x: input.tensor().into(),
                },
            )
            .unwrap()
            .value;
        graph.export(&scaled).unwrap();
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();
        let x = Tensor::from_host(&device, Element::f32(), &[n], &f32_bytes(&vec![1.0; n as usize])).unwrap();
        let row = Tensor::from_host(&device, Element::f32(), &[1, 4], &f32_bytes(&[1.0; 4])).unwrap();
        let word = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        let skipped = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        let parks = Timeline::new(&device).unwrap();
        let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&published, &row).unwrap();
        bindings.set(&answer, &word).unwrap();
        bindings.set(&skip, &skipped).unwrap();
        bindings.set(&input, &x).unwrap();
        bindings.set_park(publication, &parks).unwrap();
        bindings.set_park(park, &parks).unwrap();
        let mut active = slot.activate(&plan).unwrap();
        let (outputs, completion) = active.attach(bindings, outputs).unwrap().submit().unwrap();
        let (published, parked) = (outputs.park(publication), outputs.park(park));
        assert_eq!(parked.is_some(), !backend.parks_in_kernels(), "{backend:?}: a park binds operations");
        assert_eq!(published.is_some(), parked.is_some(), "{backend:?}: a publication takes a value where a park does");
        if let (Some(published), Some(parked)) = (published, parked) {
            // The publication took the value before the park's, which is the
            // host's one way to know the call's writes are its to read, and
            // the device passes it on its own, reaching the park.
            assert!(published < parked, "{backend:?}: the publication's value precedes the park's");
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !parks.reached(parked).unwrap() {
                assert!(std::time::Instant::now() < deadline, "{backend:?}: the device passes the publication");
                std::thread::yield_now();
            }
            assert!(parks.reached(published).unwrap(), "{backend:?}: the device reached the publication before the park");
        }
        word.store_release(0, 2).unwrap();
        if let Some(parked) = parked {
            parks.answer(parked, seismic::ParkAnswer { by: seismic::AnsweredBy::Arrival, park: "test", layer: 0 }).unwrap();
        }
        completion.wait().unwrap();
        let read = |port| {
            outputs
                .exported(port)
                .unwrap()
                .read_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(read(&scaled), vec![3.0; 4], "{backend:?}: the published call ran");
        assert_eq!(read(&value), vec![2.0; n as usize], "{backend:?}: the parked call read its answer");
    }
}

/// A park node precedes exactly a call that parks: neither may be left
/// without the other.
#[test]
fn a_park_pairs_with_a_call_that_parks() {
    for device in seismic_native_tests::devices(&[BackendName::Cpu, BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]) {
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        // A park before a call that does not park.
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[1, 4]).unwrap();
        graph.timeline(seismic::timeline::Park);
        let call = scale_rows::WorkflowArgs {
            x: input.tensor().into(),
            factor: 1.0,
        };
        assert!(graph.enqueue(&kernel, call).is_err());
        // A park that ends the graph.
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[1, 4]).unwrap();
        let call = scale_rows::WorkflowArgs {
            x: input.tensor().into(),
            factor: 1.0,
        };
        let value = graph.enqueue(&kernel, call).unwrap().value;
        graph.export(&value).unwrap();
        graph.timeline(seismic::timeline::Park);
        assert!(graph.seal().is_err());
    }
}

/// A park answered before its run is submitted (`set_answered`) binds no
/// operation on any backend: the call reads the answer already in place.
/// Only a park's port may be bound so, which its type says
/// (`seismic::Timeline`).
#[test]
fn a_park_answered_before_the_run_waits_for_nothing() {
    let n = 4u64;
    for device in seismic_native_tests::devices(&[BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]) {
        let backend = device.backend();
        let kernel = parked_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
        let mut graph = device.native_graph();
        let answer = graph.port(Element::i32(), &[1]).unwrap();
        let skip = graph.port(Element::i32(), &[1]).unwrap();
        let input = graph.port(Element::f32(), &[n]).unwrap();
        let park = graph.timeline(seismic::timeline::Park);
        let value = graph
            .enqueue(
                &kernel,
                parked_scale::WorkflowArgs {
                    answer: answer.tensor().into(),
                    skip: skip.tensor().into(),
                    x: input.tensor().into(),
                },
            )
            .unwrap()
            .value;
        let after = graph.timeline(seismic::timeline::Signal);
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let mut slot = plan.new_slot().unwrap();
        let values = (0..n).map(|value| value as f32).collect::<Vec<_>>();
        let x = Tensor::from_host(&device, Element::f32(), &[n], &f32_bytes(&values)).unwrap();
        let word = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        word.store_release(0, 3).unwrap();
        let run = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        run.store_release(0, 0).unwrap();
        let done = Timeline::<seismic::timeline::Reached>::new(&device).unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&answer, &word).unwrap();
        bindings.set(&skip, &run).unwrap();
        bindings.set(&input, &x).unwrap();
        bindings.set_answered(park).unwrap();
        bindings.set_timeline(after, &done, 1).unwrap();
        let (outputs, completion) = slot.attach(bindings, plan.new_outputs().unwrap()).unwrap().submit().unwrap();
        completion.wait().unwrap();
        assert_eq!(
            outputs
                .exported(&value)
                .unwrap()
                .read_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
            values.iter().map(|value| 3.0 * value).collect::<Vec<_>>(),
            "{backend:?}: the call read the answer in place"
        );
    }
}

/// Standalone calls expose no submission-boundary answer capability. All
/// three public preparation routes must refuse before queuing work.
#[test]
fn standalone_awaited_preparation_requires_kernel_parking() {
    for device in seismic_native_tests::devices(&[BackendName::Vulkan]) {
        let kernel = parked_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
        let answer = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        let skip = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        let x = Tensor::from_host(&device, Element::f32(), &[1], &1.0f32.to_le_bytes()).unwrap();
        let mut value = Tensor::shared(&device, Element::f32(), &[1]).unwrap();
        let args = || parked_scale::Args { answer: &answer, skip: &skip, x: &x };
        let refused = |result: Result<(), seismic::CallError>| assert!(matches!(
            result, Err(seismic::CallError::Workflow(seismic::WorkflowError::ParkRequiresGraph))
        ));
        refused(kernel.call(args()).map(|_| ()));
        refused(kernel.point(vec![args()]).map(|_| ()));
        let mut batch = seismic::NativeTensorBatch::new(&device);
        refused(batch.push(&kernel, args(), parked_scale::OutputArgs { value: &mut value }));
    }
}
