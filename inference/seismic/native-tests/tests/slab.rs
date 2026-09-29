use seismic::{BackendName, DeviceCatalog, Element, NativeSpecialization, SlabRegion, SlabTensor};
use seismic_native_tests::{scale_rows, slab_probe, slab_scale};

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

fn native_reads_rows_across_slab_boundary(backend: BackendName) {
    native_reads_rows_with_slab_size(backend, 256);
}

fn native_reads_rows_with_slab_size(backend: BackendName, slab_rows: u64) {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(backend)
        .unwrap();
    let tail_rows = slab_rows.min(44);
    let logical_rows = slab_rows + tail_rows;
    let mut slabs = SlabTensor::new(
        &device,
        slab_rows,
        logical_rows,
        vec![SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab().unwrap();
    slabs.add_slab().unwrap();
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
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(backend)
        .unwrap();
    let mut slabs = SlabTensor::new(
        &device,
        256,
        3 * 256,
        vec![SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    for _ in 0..3 {
        slabs.add_slab().unwrap();
    }
    let old = slabs.slab(1).unwrap().observe_storage();
    slabs.free_slab(1).unwrap();
    assert_eq!(slabs.add_slab().unwrap(), 1);
    assert_eq!(old.charged_bytes(), None);
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
/// number of retained and resident slab allocations changes. Run explicitly
/// on an otherwise idle Metal device and compare the three printed samples.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "manual Metal submission-overhead measurement"]
fn metal_submission_cost_with_one_and_one_hundred_slabs() {
    use std::time::Instant;

    fn sample(device: &seismic::Device, slab_count: usize) -> (u128, u128) {
        // Keep the logical shape, address-table size and compiled kernel
        // identical. Only the number of backed slabs changes.
        let rows = 100 * 256;
        let mut slabs = SlabTensor::new(
            device,
            256,
            rows,
            vec![SlabRegion {
                element: Element::f32(),
                row_shape: vec![1],
            }],
        )
        .unwrap();
        for _ in 0..slab_count {
            slabs.add_slab().unwrap();
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
            let mut sequence = device.native_sequence();
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

    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let one_before = sample(&device, 1);
    let hundred = sample(&device, 100);
    let one_after = sample(&device, 1);
    eprintln!(
        "Metal queue/submit median ns: 1 slab before={one_before:?}, 100 slabs={hundred:?}, 1 slab after={one_after:?}"
    );
}

#[test]
fn freeing_a_slab_waits_for_submitted_table_and_slab_reads() {
    free_after_submitted_read(BackendName::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_frees_slab_after_submitted_read() {
    free_after_submitted_read(BackendName::Metal);
}

fn free_after_submitted_read(backend: BackendName) {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(backend)
        .unwrap();
    let mut slabs = SlabTensor::new(
        &device,
        256,
        300,
        vec![SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab().unwrap();
    slabs.add_slab().unwrap();
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
                slab_rows: 256,
                factor: 2.0,
                delay_ms: 80,
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
    let observer = slabs.free_slab(0).unwrap().unwrap();
    completion.wait().unwrap();
    assert_eq!(observer.charged_bytes(), None);
    assert_eq!(slabs.add_slab().unwrap(), 0);
    let values = outputs.exported(&output).unwrap().read_to_host().unwrap();
    for (row, bytes) in values.chunks_exact(4).enumerate() {
        assert_eq!(
            f32::from_le_bytes(bytes.try_into().unwrap()),
            row as f32 * 2.0
        );
    }
}

#[test]
fn queued_graph_keeps_slab_placement_fixed_until_submission_or_drop() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let mut slabs = SlabTensor::new(
        &device,
        256,
        300,
        vec![SlabRegion {
            element: Element::f32(),
            row_shape: vec![1],
        }],
    )
    .unwrap();
    slabs.add_slab().unwrap();
    slabs.add_slab().unwrap();

    let kernel =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let mut graph = device.native_graph();
    let input = graph.port(Element::f32(), &[300, 1]).unwrap();
    let output = graph
        .enqueue(
            &kernel,
            scale_rows::WorkflowArgs {
                x: input.tensor().into(),
                factor: 2.0,
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
    assert!(slabs.free_slab(0).is_err());

    let mut sequence = device.native_sequence();
    ready.queue(&mut sequence).unwrap();
    assert!(slabs.free_slab(0).is_err());
    drop(sequence);
    assert!(slabs.free_slab(0).unwrap().is_some());
}
