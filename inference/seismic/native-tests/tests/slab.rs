use seismic::{BackendName, Element, NativeSpecialization, NativeTensorBatch, SlabFill, SlabRegion, SlabTensor, Tensor};
use seismic_native_tests::{pool, scale_rows, slab_scale};

#[test]
fn native_cpu_reads_rows_across_slab_boundary() {
    native_reads_rows_across_slab_boundary(BackendName::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_reads_rows_across_slab_boundary() {
    native_reads_rows_across_slab_boundary(BackendName::Metal);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a CUDA host"]
fn native_cuda_reads_rows_across_slab_boundary() {
    native_reads_rows_across_slab_boundary(BackendName::Cuda);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a Vulkan host"]
fn native_vulkan_reads_rows_across_slab_boundary() {
    native_reads_rows_across_slab_boundary(BackendName::Vulkan);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a Vulkan host"]
fn native_vulkan_reads_two_row_slabs() {
    native_reads_rows_with_slab_size(BackendName::Vulkan, 2);
}

#[test]
fn native_cpu_slab_port_refuses_a_plain_tensor() {
    slab_port_refuses_a_plain_tensor(BackendName::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_slab_port_refuses_a_plain_tensor() {
    slab_port_refuses_a_plain_tensor(BackendName::Metal);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a CUDA host"]
fn native_cuda_slab_port_refuses_a_plain_tensor() {
    slab_port_refuses_a_plain_tensor(BackendName::Cuda);
}

/// A slab port (`&slab`) binds a slab view: its natives read its binding as
/// a slab address table, so a plain tensor's elements would be taken for
/// slab addresses. A call refuses one, as a native graph's port binding
/// does, before anything is encoded.
fn slab_port_refuses_a_plain_tensor(backend: BackendName) {
    let Some(device) = seismic_native_tests::fresh_device(backend) else { return };
    let kernel = slab_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let plain = Tensor::from_host(&device, Element::f32(), &[4, 1], &[0; 16]).unwrap();
    let refused = seismic::WorkflowError::PlainSlabBinding { parameter: "x".into() };
    match kernel.call(slab_scale::Args { x: &plain, slab_rows: 4, factor: 2.0, delay_ms: 0 }) {
        Err(seismic::CallError::Workflow(error)) => assert_eq!(error, refused),
        other => panic!("a plain tensor at a slab port: {:?}", other.map(|_| ())),
    }

    let mut graph = device.native_graph();
    let port = graph.port(Element::f32(), &[4, 1]).unwrap();
    let scaled = graph
        .enqueue(
            &kernel,
            slab_scale::WorkflowArgs {
                x: port.tensor().into(),
                slab_rows: seismic::Argument::from(4),
                factor: seismic::Argument::from(2.0),
                delay_ms: seismic::Argument::from(0),
            },
        )
        .unwrap()
        .value;
    graph.export(&scaled).unwrap();
    let plan = graph.seal().unwrap();
    let mut bindings = plan.bindings();
    assert_eq!(
        bindings.set(&port, &plain),
        Err(seismic::WorkflowError::PlainSlabBinding { parameter: "port 0".into() })
    );
}

/// An empty leading slice of a slab view names no slab, so it is a view of
/// no rows whether or not any slab is held, as an empty slice of a plain
/// tensor is; a slice past the extent is still refused.
#[test]
fn an_empty_slice_of_a_slab_view_holds_no_slab() {
    let Some(device) = seismic_native_tests::fresh_device(BackendName::Cpu) else { return };
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        2,
        2,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    let logical = slabs.logical_region(0).unwrap();
    for held in [false, true] {
        if held {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        let mut empty = logical.slice_leading(1, 1).unwrap();
        assert_eq!(empty.extents(), &[0, 1]);
        assert!(empty.read_to_host().unwrap().is_empty());
        empty.write_from_host(&[]).unwrap();
    }
    assert!(logical.slice_leading(2, 3).is_err());

    // A one-row-per-slab empty slice forms a zero-count record view over the
    // store's address table without allocating a view table, even after the
    // device's record tables are sealed.
    let record_slabs = SlabTensor::in_pool(
        &pool(&device),
        1,
        4,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![16],
        }],
    )
    .unwrap();
    device.seal_record_tables();
    let empty_records = record_slabs.logical_region(0).unwrap().slice_leading(2, 2).unwrap();
    assert_eq!(empty_records.extents(), &[0, 16]);
    empty_records.hold_record_table().unwrap();
}

fn native_reads_rows_across_slab_boundary(backend: BackendName) {
    native_reads_rows_with_slab_size(backend, 256);
}

fn native_reads_rows_with_slab_size(backend: BackendName, slab_rows: u64) {
    let Some(device) = seismic_native_tests::fresh_device(backend) else { return };
    let tail_rows = slab_rows.min(44);
    let logical_rows = slab_rows + tail_rows;
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        slab_rows,
        logical_rows,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    for (start, rows) in [(0, slab_rows), (slab_rows, tail_rows)] {
        let values = (start..start + rows)
            .flat_map(|row| (row as f32).to_le_bytes())
            .collect::<Vec<_>>();
        slabs
            .region_rows(0, start, rows)
            .unwrap()
            .write_from_host(&values)
            .unwrap();
    }
    let logical = slabs.logical_region(0).unwrap();
    let kernel = slab_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let result = kernel
        .call(slab_scale::Args {
            x: &logical,
            slab_rows: slab_rows as u32,
            factor: 2.0,
            delay_ms: 0,
        })
        .unwrap()
        .value;
    let values = result.read_to_host().unwrap();
    for (row, bytes) in values.chunks_exact(4).enumerate() {
        assert_eq!(
            f32::from_le_bytes(bytes.try_into().unwrap()),
            row as f32 * 2.0
        );
    }
}

#[test]
fn native_cpu_reads_reused_interior_slab_index() {
    native_reads_reused_interior_slab_index(BackendName::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_reads_reused_interior_slab_index() {
    native_reads_reused_interior_slab_index(BackendName::Metal);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a CUDA host"]
fn native_cuda_reads_reused_interior_slab_index() {
    native_reads_reused_interior_slab_index(BackendName::Cuda);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a Vulkan host"]
fn native_vulkan_reads_reused_interior_slab_index() {
    native_reads_reused_interior_slab_index(BackendName::Vulkan);
}

fn native_reads_reused_interior_slab_index(backend: BackendName) {
    let Some(device) = seismic_native_tests::fresh_device(backend) else { return };
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        256,
        3 * 256,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    for _ in 0..3 {
        slabs.add_slab(SlabFill::Zeroed).unwrap();
    }
    assert!(slabs.free_slab(1).unwrap());
    assert_eq!(slabs.add_slab(SlabFill::Zeroed).unwrap(), 1);
    for index in 0..3 {
        let start = index * 256;
        let values = (start..start + 256)
            .flat_map(|row| (row as f32).to_le_bytes())
            .collect::<Vec<_>>();
        slabs
            .region_rows(0, start, 256)
            .unwrap()
            .write_from_host(&values)
            .unwrap();
    }
    let logical = slabs.logical_region(0).unwrap();
    let kernel = slab_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let output = kernel
        .call(slab_scale::Args {
            x: &logical,
            slab_rows: 256,
            factor: 2.0,
            delay_ms: 0,
        })
        .unwrap()
        .value
        .read_to_host()
        .unwrap();
    for (row, bytes) in output.chunks_exact(4).enumerate() {
        assert_eq!(
            f32::from_le_bytes(bytes.try_into().unwrap()),
            row as f32 * 2.0
        );
    }
}

/// The kernel does one read and one write for either slab count. Only the
/// number of backed slabs changes. Run explicitly
/// on an otherwise idle Metal device and compare the three printed samples.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "manual Metal submission-overhead measurement"]
fn metal_submission_cost_with_one_and_one_hundred_slabs() {
    use seismic_native_tests::slab_probe;
    use std::time::Instant;

    fn sample(device: &seismic::Device, slab_count: usize) -> (u128, u128) {
        // Keep the logical shape, address-table size and compiled kernel
        // identical. Only the number of backed slabs changes.
        let rows = 100 * 256;
        let slabs = SlabTensor::in_pool(
            &pool(device),
            256,
            rows,
            &[SlabRegion {
                element: Element::f32(),
                row_shape: vec![1],
            }],
        )
        .unwrap();
        for _ in 0..slab_count {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        slabs
            .region_rows(0, 0, 1)
            .unwrap()
            .write_from_host(&1f32.to_le_bytes())
            .unwrap();
        let logical = slabs.logical_region(0).unwrap();
        let kernel = slab_probe::native_for_device(device, &NativeSpecialization::new()).unwrap();
        let mut graph = device.native_graph();
        let input = graph.port(Element::f32(), &[rows, 1]).unwrap();
        let output = graph
            .enqueue(
                &kernel,
                slab_probe::WorkflowArgs {
                    x: input.tensor().into(),
                },
            )
            .unwrap()
            .value;
        graph.export(&output).unwrap();
        let plan = graph.seal().unwrap();
        let mut queue_samples = Vec::new();
        let mut submit_samples = Vec::new();
        for index in 0..600 {
            let mut bindings = plan.bindings();
            bindings.set(&input, &logical).unwrap();
            let outputs = plan.new_outputs().unwrap();
            let mut slot = plan.new_slot().unwrap();
            let ready = slot.attach(bindings, outputs).unwrap();
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
            let start = Instant::now();
            let outputs = ready.queue(&mut sequence).unwrap();
            let queue_nanos = start.elapsed().as_nanos();
            let start = Instant::now();
            let completion = sequence.submit().unwrap();
            let submit_nanos = start.elapsed().as_nanos();
            completion.wait().unwrap();
            assert_eq!(
                outputs.exported(&output).unwrap().read_to_host().unwrap(),
                1f32.to_le_bytes()
            );
            if index >= 100 {
                queue_samples.push(queue_nanos);
                submit_samples.push(submit_nanos);
            }
        }
        queue_samples.sort_unstable();
        submit_samples.sort_unstable();
        (
            queue_samples[queue_samples.len() / 2],
            submit_samples[submit_samples.len() / 2],
        )
    }

    let Some(device) = seismic_native_tests::fresh_device(BackendName::Metal) else { return };
    let one_before = sample(&device, 1);
    let hundred = sample(&device, 100);
    let one_after = sample(&device, 1);
    eprintln!(
        "Metal queue/submit median ns: 1 slab before={one_before:?}, 100 slabs={hundred:?}, 1 slab after={one_after:?}"
    );
}

/// A slab freed while a submitted graph run reads it frees at once: its
/// slot names the pool's zero slab from then, and the run reads the rows it
/// had, whole, until it completes.
#[test]
fn a_slab_freed_beside_a_submitted_read_keeps_its_rows_for_it() {
    free_after_submitted_read(BackendName::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_frees_slab_after_submitted_read() {
    free_after_submitted_read(BackendName::Metal);
}

fn free_after_submitted_read(backend: BackendName) {
    let Some(device) = seismic_native_tests::fresh_device(backend) else { return };
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        256,
        300,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    for (start, rows) in [(0, 256), (256, 44)] {
        let values = (start..start + rows)
            .flat_map(|row| (row as f32).to_le_bytes())
            .collect::<Vec<_>>();
        slabs
            .region_rows(0, start, rows)
            .unwrap()
            .write_from_host(&values)
            .unwrap();
    }
    let kernel = slab_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let mut graph = device.native_graph();
    let input = graph.port(Element::f32(), &[300, 1]).unwrap();
    let output = graph
        .enqueue(
            &kernel,
            slab_scale::WorkflowArgs {
                x: input.tensor().into(),
                slab_rows: seismic::Argument::from(256),
                factor: seismic::Argument::from(2.0),
                delay_ms: seismic::Argument::from(80),
            },
        )
        .unwrap()
        .value;
    graph.export(&output).unwrap();
    let plan = graph.seal().unwrap();
    let mut bindings = plan.bindings();
    let logical = slabs.logical_region(0).unwrap();
    bindings.set(&input, &logical).unwrap();
    let mut slot = plan.new_slot().unwrap();
    let (outputs, completion) = slot
        .attach(bindings, plan.new_outputs().unwrap())
        .unwrap()
        .submit()
        .unwrap();
    drop(logical);
    assert!(slabs.free_slab(0).unwrap());
    completion.wait().unwrap();
    assert_eq!(slabs.add_slab(SlabFill::Zeroed).unwrap(), 0);
    let values = outputs.exported(&output).unwrap().read_to_host().unwrap();
    for (row, bytes) in values.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes(bytes.try_into().unwrap());
        let expected = row as f32 * 2.0;
        if row < 256 {
            assert!(
                got == expected || got == 0.0,
                "freed slot 0 row {row} reads whole as old ({expected}) or absent (0.0), got {got}"
            );
        } else {
            assert_eq!(got, expected);
        }
    }
}

/// A queued graph run holds off no slab change: its store's table is read as
/// the run runs, so a slab frees beside it.
#[test]
fn a_queued_graph_run_holds_off_no_slab_change() {
    let Some(device) = seismic_native_tests::fresh_device(BackendName::Cpu) else { return };
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        256,
        300,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();

    let kernel = slab_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let mut graph = device.native_graph();
    let input = graph.port(Element::f32(), &[300, 1]).unwrap();
    let output = graph
        .enqueue(
            &kernel,
            slab_scale::WorkflowArgs {
                x: input.tensor().into(),
                slab_rows: seismic::Argument::from(256),
                factor: seismic::Argument::from(2.0),
                delay_ms: seismic::Argument::from(0),
            },
        )
        .unwrap()
        .value;
    graph.export(&output).unwrap();
    let plan = graph.seal().unwrap();
    let mut bindings = plan.bindings();
    let logical = slabs.logical_region(0).unwrap();
    bindings.set(&input, &logical).unwrap();
    let mut slot = plan.new_slot().unwrap();
    let ready = slot.attach(bindings, plan.new_outputs().unwrap()).unwrap();
    drop(logical);
    let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
    ready.queue(&mut sequence).unwrap();
    assert!(slabs.free_slab(0).unwrap());
    assert_eq!(slabs.add_slab(SlabFill::Zeroed).unwrap(), 0);
    drop(sequence);
}

#[test]
fn freeing_a_slab_after_a_submitted_batch_read() {
    free_after_batch_read(BackendName::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_frees_slab_after_a_submitted_batch_read() {
    free_after_batch_read(BackendName::Metal);
}

/// A slab freed while a batch's read is in flight keeps its rows for that
/// read: its granule is claimable again only once the batch completes.
fn free_after_batch_read(backend: BackendName) {
    let Some(device) = seismic_native_tests::fresh_device(backend) else { return };
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        256,
        300,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    for (start, rows) in [(0, 256), (256, 44)] {
        let values = (start..start + rows)
            .flat_map(|row| (row as f32).to_le_bytes())
            .collect::<Vec<_>>();
        slabs
            .region_rows(0, start, rows)
            .unwrap()
            .write_from_host(&values)
            .unwrap();
    }
    let kernel = slab_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let logical = slabs.logical_region(0).unwrap();
    let mut value = Tensor::zeros(&device, Element::f32(), &[300, 1]).unwrap();
    let mut batch = NativeTensorBatch::new(&device);
    batch
        .push(
            &kernel,
            slab_scale::Args {
                x: &logical,
                slab_rows: 256,
                factor: 2.0,
                delay_ms: 80,
            },
            slab_scale::OutputArgs { value: &mut value },
        )
        .unwrap();
    let completion = batch.submit().unwrap();
    drop(logical);
    assert!(slabs.free_slab(0).unwrap());
    completion.wait().unwrap();
    assert_eq!(slabs.add_slab(SlabFill::Zeroed).unwrap(), 0);
    let values = value.read_to_host().unwrap();
    for (row, bytes) in values.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes(bytes.try_into().unwrap());
        let expected = row as f32 * 2.0;
        if row < 256 {
            assert!(
                got == expected || got == 0.0,
                "freed slot 0 row {row} reads whole as old ({expected}) or absent (0.0), got {got}"
            );
        } else {
            assert_eq!(got, expected);
        }
    }
}


#[test]
fn plain_port_refuses_a_slab_view() {
    for device in seismic_native_tests::devices(&[
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
    ]) {
        plain_port_refuses_a_non_empty_slab_view(device);
    }
}

fn plain_port_refuses_a_non_empty_slab_view(device: seismic::Device) {
    let kernel =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        2,
        4,
        &[SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    slabs.add_slab(SlabFill::Zeroed).unwrap();
    let logical = slabs.logical_region(0).unwrap();
    let subview = logical.slice_leading(0, 4).unwrap();
    let refused = seismic::WorkflowError::SlabPlainBinding {
        parameter: "x".into(),
    };
    match kernel.call(scale_rows::Args {
        x: &subview,
        factor: 2.0,
    }) {
        Err(seismic::CallError::Workflow(error)) => assert_eq!(error, refused),
        other => panic!("a non-empty slab view at a plain port: {:?}", other.map(|_| ())),
    }
    let empty = logical.slice_leading(2, 2).unwrap();
    match kernel.call(scale_rows::Args {
        x: &empty,
        factor: 2.0,
    }) {
        Err(seismic::CallError::Workflow(seismic::WorkflowError::SlabPlainBinding { .. })) => {
            panic!("an empty slab view has no elements and is permitted at a plain port")
        }
        Err(seismic::CallError::Invocation(seismic::InvocationError::OutsideTargetDomain)) => {}
        other => panic!("unexpected scale_rows empty result: {:?}", other.map(|_| ())),
    }

    let mut graph = device.native_graph();
    let port = graph.port(Element::f32(), &[4, 1]).unwrap();
    let scaled = graph
        .enqueue(
            &kernel,
            scale_rows::WorkflowArgs {
                x: port.tensor().into(),
                factor: seismic::Argument::from(2.0),
            },
        )
        .unwrap()
        .value;
    graph.export(&scaled).unwrap();
    let plan = graph.seal().unwrap();
    let mut bindings = plan.bindings();
    assert_eq!(
        bindings.set(&port, &subview),
        Err(seismic::WorkflowError::SlabPlainBinding {
            parameter: "port 0".into(),
        })
    );
}

#[test]
fn multi_slab_subviews_compose_and_scope_slabs() {
    for device in seismic_native_tests::devices(&[
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
    ]) {
        multi_slab_subviews_compose_and_scope_slabs_on(device);
    }
}

fn multi_slab_subviews_compose_and_scope_slabs_on(device: seismic::Device) {
    // 4 slots, 4 rows per slab, 14 logical rows (slots 0..3 hold 4 rows each,
    // slot 3 holds 2 tail rows), and two regions so region 1 starts past
    // region 0's table entries.
    let slabs = SlabTensor::in_pool(
        &pool(&device),
        4,
        14,
        &[
            SlabRegion {
                element: Element::f32(),
                row_shape: vec![1],
            },
            SlabRegion {
                element: Element::f32(),
                row_shape: vec![2],
            },
        ],
    )
    .unwrap();
    for _ in 0..4 {
        slabs.add_slab(SlabFill::Zeroed).unwrap();
    }
    let mut logical = slabs.logical_region(1).unwrap();

    // Unaligned cross-slab slices are refused; slab-aligned multi-slab slices
    // (including the partial final slab) form SlabView sub-views.
    assert!(logical.slice_leading(2, 6).is_err());
    assert!(logical.slice_leading(0, 5).is_err());

    let head = logical.slice_leading(0, 4).unwrap();
    let mut mid_and_tail = logical.slice_leading(4, 14).unwrap();
    assert_eq!(head.extents(), &[4, 2]);
    assert_eq!(mid_and_tail.extents(), &[10, 2]);

    // Free slot 0 (outside `mid_and_tail`, which spans slots 1..4): host I/O
    // on `mid_and_tail` succeeds while `logical` refuses host writes because
    // slot 0 is free.
    assert!(slabs.free_slab(0).unwrap());
    let ten_rows = (0..20)
        .map(|index| (index as f32) + 10.0)
        .collect::<Vec<_>>();
    let ten_bytes = ten_rows
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    assert!(logical
        .write_from_host(&vec![0u8; 14 * 2 * 4])
        .is_err());
    mid_and_tail.write_from_host(&ten_bytes).unwrap();
    assert_eq!(mid_and_tail.read_to_host().unwrap(), ten_bytes);

    // A whole-slab sub-view (multi-slab or 1-slab) holds no granule: freeing
    // slot 1 succeeds while both `mid_and_tail` and `single_whole` live, and
    // host writes to `mid_and_tail` then notice the free slot inside 1..4.
    let _single_whole = mid_and_tail.slice_leading(0, 4).unwrap();
    assert!(slabs.free_slab(1).unwrap());
    assert!(mid_and_tail.write_from_host(&ten_bytes).is_err());
    assert_eq!(slabs.add_slab(SlabFill::Zeroed).unwrap(), 0);
    assert_eq!(slabs.add_slab(SlabFill::Zeroed).unwrap(), 1);
    mid_and_tail.write_from_host(&ten_bytes).unwrap();

    // Nested whole-slab sub-view (`mid_and_tail` rows 4..10 -> logical rows
    // 8..14, slots 2..4) and intra-slab slice (`nested` rows 1..3 -> logical
    // rows 9..11 inside slot 2) compose offsets cleanly.
    let nested = mid_and_tail.slice_leading(4, 10).unwrap();
    assert_eq!(nested.extents(), &[6, 2]);
    assert_eq!(nested.read_to_host().unwrap(), ten_bytes[4 * 2 * 4..]);

    let reshaped = nested.reshape(&[6, 1, 2]).unwrap();
    assert_eq!(reshaped.extents(), &[6, 1, 2]);
    assert_eq!(reshaped.read_to_host().unwrap(), ten_bytes[4 * 2 * 4..]);

    let intra = nested.slice_leading(1, 3).unwrap();
    assert_eq!(intra.extents(), &[2, 2]);
    assert_eq!(
        intra.read_to_host().unwrap(),
        ten_bytes[5 * 2 * 4..7 * 2 * 4]
    );
    // `intra` is a plain single-slab view holding slot 2's granule: freeing
    // slot 2 from the store repoints its table while `intra` keeps reading
    // its rows until dropped.
    assert!(slabs.free_slab(2).unwrap());
    assert_eq!(
        intra.read_to_host().unwrap(),
        ten_bytes[5 * 2 * 4..7 * 2 * 4]
    );
    drop(intra);
    assert_eq!(slabs.add_slab(SlabFill::Zeroed).unwrap(), 2);
    mid_and_tail.write_from_host(&ten_bytes).unwrap();

    // Native `&slab` kernel reads the exact sub-view slots (`first..first+count`)
    // from region 1's table offset for both `mid_and_tail` and `nested`.
    let kernel = slab_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let scaled_mid = kernel
        .call(slab_scale::Args {
            x: &mid_and_tail,
            slab_rows: 4,
            factor: 3.0,
            delay_ms: 0,
        })
        .unwrap();
    let got_mid = scaled_mid
        .value
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let want_mid = ten_rows.iter().map(|v| v * 3.0).collect::<Vec<_>>();
    assert_eq!(got_mid, want_mid);

    let scaled_nested = kernel
        .call(slab_scale::Args {
            x: &nested,
            slab_rows: 4,
            factor: 2.0,
            delay_ms: 0,
        })
        .unwrap();
    let got_nested = scaled_nested
        .value
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let want_nested = ten_rows[8..].iter().map(|v| v * 2.0).collect::<Vec<_>>();
    assert_eq!(got_nested, want_nested);
}
