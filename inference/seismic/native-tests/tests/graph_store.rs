//! Stored results (`NativeGraph::store`): a node's result written into the
//! tensor a run binds to an external port, in place of graph storage, which
//! later nodes of the run then read through that port. Run on every
//! available device.

use seismic::{
    BackendName, Devices, Element, NativeGraphClassSlice, NativeGraphFamily,
    NativeGraphMetadata, NativeSpecialization, Tensor,
};
use seismic_native_tests::scale_rows;

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

/// The stored result lands in the rows of a larger tensor the run binds,
/// and a node after it reads them through the port; the graph places no
/// storage for it.
#[test]
fn a_stored_result_is_written_into_the_bound_tensor() {
    let (m, n) = (2u64, 8u64);
    for device in seismic_native_tests::every_device() {
        let backend = device.backend();
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[m, n]).unwrap();
        let landing = graph.port(Element::f32(), &[m, n]).unwrap();
        let scaled = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 2.0,
                },
            )
            .unwrap()
            .value;
        graph.store(&scaled, &landing).unwrap();
        let after = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: landing.tensor().into(),
                    factor: 3.0,
                },
            )
            .unwrap()
            .value;
        graph.export(&after).unwrap();
        let plan = graph.seal().unwrap();
        // Only the exported result has storage of its own.
        assert_eq!(plan.output_bytes(), m * n * 4, "{backend:?}");
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();

        let values = (0..m * n).map(|value| value as f32).collect::<Vec<_>>();
        let source = Tensor::from_host(&device, Element::f32(), &[m, n], &f32_bytes(&values))
            .unwrap();
        // The bound rows are rows 1 and 2 of four; the others are untouched.
        let whole = Tensor::from_host(
            &device,
            Element::f32(),
            &[4, n],
            &f32_bytes(&vec![-1.0; 4 * n as usize]),
        )
        .unwrap();
        let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&input, &source).unwrap();
        bindings
            .set(&landing, &whole.slice_leading(1, 3).unwrap())
            .unwrap();
        let mut active = slot.activate(&plan).unwrap();
        let (outputs, completion) = active.attach(bindings, outputs).unwrap().submit().unwrap();
        completion.wait().unwrap();
        let doubled = values.iter().map(|value| 2.0 * value).collect::<Vec<_>>();
        let untouched = vec![-1.0; n as usize];
        assert_eq!(
            read_f32(&whole),
            [untouched.clone(), doubled.clone(), untouched].concat(),
            "{backend:?}: the stored rows"
        );
        assert_eq!(
            read_f32(&outputs.exported(&after).unwrap()),
            doubled.iter().map(|value| 3.0 * value).collect::<Vec<_>>(),
            "{backend:?}: read after the store"
        );
    }
}

/// Every class of one template seals into its layout with the template's
/// store: each class is a draft of its own, and the store is the same edge
/// in each.
#[test]
fn a_templates_classes_seal_their_store_into_its_layout() {
    let n = 8u64;
    for device in seismic_native_tests::every_device() {
        let backend = device.backend();
        let mut template = NativeGraphMetadata::new_template(backend);
        let input = template
            .port_with_class_extent(Element::f32(), &[4, n], 0, "M")
            .unwrap();
        let landing = template
            .port_with_class_extent(Element::f32(), &[4, n], 0, "M")
            .unwrap();
        let scaled = template
            .enqueue::<scale_rows::Entry>(
                &[],
                &[("M", 4), ("N", n)],
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 2.0,
                },
            )
            .unwrap()
            .value;
        template.store(&scaled, &landing).unwrap();
        let layout = template
            .seal_template()
            .unwrap()
            .certify(&[NativeGraphClassSlice::new().dimension("M", [2, 4])])
            .unwrap();
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
                .unwrap();
        for m in [2u64, 4] {
            let mut graph = device.native_graph_with_layout(&layout);
            let input = graph.port(Element::f32(), &[m, n]).unwrap();
            let landing = graph.port(Element::f32(), &[m, n]).unwrap();
            let scaled = graph
                .enqueue(
                    &kernel,
                    scale_rows::WorkflowArgs {
                        x: input.tensor().into(),
                        factor: 2.0,
                    },
                )
                .unwrap()
                .value;
            graph.store(&scaled, &landing).unwrap();
            let plan = graph
                .seal()
                .unwrap_or_else(|error| panic!("{backend:?} M = {m}: {error:?}"));
            let values = (0..m * n).map(|value| value as f32).collect::<Vec<_>>();
            let source =
                Tensor::from_host(&device, Element::f32(), &[m, n], &f32_bytes(&values)).unwrap();
            let stored = Tensor::zeros(&device, Element::f32(), &[m, n]).unwrap();
            let mut bindings = plan.bindings();
            bindings.set(&input, &source).unwrap();
            bindings.set(&landing, &stored).unwrap();
            let mut slot = plan.new_slot().unwrap();
            let (_, completion) = slot
                .attach(bindings, plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(
                read_f32(&stored),
                values.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
                "{backend:?} M = {m}"
            );
        }
    }
}

/// A result is stored only into an external port that describes it
/// exactly, once, and is then not exported.
#[test]
fn a_store_takes_one_exact_external_port() {
    let device = Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let kernel =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let draft = |extents: &[u64], export: bool, twice: bool| {
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[2, 8]).unwrap();
        let landing = graph.port(Element::f32(), extents).unwrap();
        let value = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 2.0,
                },
            )
            .unwrap()
            .value;
        let stored = graph.store(&value, &landing).is_ok()
            && (!twice || graph.store(&value, &landing).is_ok());
        if export {
            graph.export(&value).unwrap();
        }
        stored && graph.seal().is_ok()
    };
    assert!(draft(&[2, 8], false, false));
    assert!(!draft(&[2, 4], false, false), "a port of another shape");
    assert!(!draft(&[2, 8], true, false), "an exported result");
    assert!(!draft(&[2, 8], false, true), "a second store");
}
