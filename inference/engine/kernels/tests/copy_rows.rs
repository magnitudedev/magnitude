use magnitude_kernels::copy_rows;

fn slabbed_rows(
    device: &seismic::Device,
    element: seismic::Element,
    rows: u64,
    row_shape: &[u64],
    bytes: &[u8],
    slab_rows: u64,
) -> seismic::SlabTensor {
    let mut slabs = seismic::SlabTensor::new(
        device,
        slab_rows,
        rows,
        vec![seismic::SlabRegion {
            element,
            row_shape: row_shape.to_vec(),
        }],
    )
    .unwrap();
    let row_bytes = bytes.len() / rows as usize;
    for index in 0..rows.div_ceil(slab_rows) {
        slabs.add_slab().unwrap();
        let start = index * slab_rows;
        let count = slab_rows.min(rows - start);
        slabs
            .region_rows(0, start, count)
            .unwrap()
            .write_from_host(
                &bytes[start as usize * row_bytes..(start + count) as usize * row_bytes],
            )
            .unwrap();
    }
    slabs
}

#[test]
fn portable_copy_moves_only_the_indexed_dense_rows() {
    use seismic_lang::{
        checked::{check_source, SourceFile, SourceSet},
        entry::ElementBindings,
        interp::{Arg, Interpreter, TensorData},
        registry,
        types::DType,
    };
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "state.seismic".into(),
        text: include_str!("../kernels/state.seismic").into(),
    }]))
    .unwrap();
    let logical = module
        .entry(
            module.entry_named("copy_rows").unwrap(),
            &ElementBindings::new().bind("A", registry::dense(DType::U32)),
        )
        .unwrap();
    let source = (0..24).map(f64::from).collect::<Vec<_>>();
    let mut interpreter = Interpreter::new(&logical);
    let rows = interpreter.add_tensor(TensorData::dense(DType::U32, vec![4, 2, 3], source));
    let from = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![3.0, 1.0]));
    let to = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![0.0, 2.0]));
    let outcome = interpreter
        .run(&[
            Arg::Tensor(rows),
            Arg::Tensor(from),
            Arg::Tensor(to),
            Arg::Scalar(seismic_lang::reference_math::ReferenceScalar::U32(2)),
        ])
        .unwrap();
    let input = outcome.inputs().next().unwrap();
    let rows = input.tensor();
    let data = (0..rows.element_count())
        .map(|index| rows.read(index).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(&data[0..6], &[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
    assert_eq!(&data[6..12], &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    assert_eq!(&data[12..18], &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    assert_eq!(&data[18..24], &[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
}

#[test]
fn generated_surface_exposes_native_and_planned_dense_plane_bindings() {
    fn planned(
        device: &seismic::Device,
        elements: copy_rows::Elements,
    ) -> Result<seismic::Kernel<copy_rows::Entry>, seismic::LoadError> {
        copy_rows::for_device_with(
            device,
            seismic::PreparationOptions::analytical(seismic::PrecisionPolicy::Exact),
            elements,
        )
    }
    fn native(
        device: &seismic::Device,
        elements: copy_rows::Elements,
    ) -> Result<seismic::NativeKernel<copy_rows::Entry>, seismic::LoadError> {
        copy_rows::native_for_device_with(device, elements, &seismic::NativeSpecialization::new())
    }
    let _ = (planned, native);
}

#[test]
fn native_copies_indexed_word_plane_rows_bit_exactly_on_every_device() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    for backend in [
        seismic::BackendName::Metal,
        seismic::BackendName::Cuda,
        seismic::BackendName::Vulkan,
        seismic::BackendName::Cpu,
    ] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        copy_word_plane_rows(&device);
    }
}

fn copy_word_plane_rows(device: &seismic::Device) {
    let element = seismic::Element::u32();
    let source = (0_u32..24).collect::<Vec<_>>();
    let source_bytes = source
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let indices = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let rows = slabbed_rows(device, element, 4, &[2, 3], &source_bytes, 2);
    let mut rows_region = rows.logical_region(0).unwrap();
    let from =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[3, 1]))
            .unwrap();
    let to = seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[0, 2]))
        .unwrap();
    copy_rows::native_for_device_with(
        &device,
        copy_rows::Elements { A: element },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(copy_rows::Args {
        rows: &mut rows_region,
        from: &from,
        to: &to,
        slab_rows: 2,
    })
    .unwrap();
    let actual = rows_region
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(&actual[0..6], &source[18..24], "{:?}", device.backend());
    assert_eq!(&actual[6..12], &source[6..12], "{:?}", device.backend());
    assert_eq!(&actual[12..18], &source[6..12], "{:?}", device.backend());
    assert_eq!(&actual[18..24], &source[18..24], "{:?}", device.backend());
}

/// The host's GPU (Metal on macOS, else Vulkan when present), then the CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let gpu = if cfg!(target_os = "macos") {
        Some(catalog.open_backend(seismic::BackendName::Metal).unwrap())
    } else {
        catalog.open_backend(seismic::BackendName::Vulkan).ok()
    };
    gpu.into_iter()
        .chain(std::iter::once(
            catalog.open_backend(seismic::BackendName::Cpu).unwrap(),
        ))
        .collect()
}

fn assert_native_two_byte_plane_copy(
    device: &seismic::Device,
    element: seismic::Element,
    source: &[u16],
) {
    let source_bytes = source
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let indices = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let rows = slabbed_rows(device, element, 4, &[2, 3], &source_bytes, 2);
    let mut rows_region = rows.logical_region(0).unwrap();
    let from =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[3, 1]))
            .unwrap();
    let to = seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[0, 2]))
        .unwrap();

    copy_rows::native_for_device_with(
        &device,
        copy_rows::Elements { A: element },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(copy_rows::Args {
        rows: &mut rows_region,
        from: &from,
        to: &to,
        slab_rows: 2,
    })
    .unwrap();

    let actual = rows_region
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(&actual[0..6], &source[18..24], "{:?}", device.backend());
    assert_eq!(&actual[6..12], &source[6..12], "{:?}", device.backend());
    assert_eq!(&actual[12..18], &source[6..12], "{:?}", device.backend());
    assert_eq!(&actual[18..24], &source[18..24], "{:?}", device.backend());
}

#[test]
fn native_copies_f16_and_bf16_plane_rows_bit_exactly() {
    for device in devices() {
        native_copies_f16_and_bf16_plane_rows_bit_exactly_on(&device);
    }
}

fn native_copies_f16_and_bf16_plane_rows_bit_exactly_on(device: &seismic::Device) {
    // Deliberately include NaNs, infinities, signed zero, and ordinary values:
    // copy_rows moves dense state packets and must never numerically convert them.
    let f16 = [
        0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e01, 0x3555, 0x0400, 0x7bff, 0x1234,
        0xabcd, 0x0101, 0x0202, 0x0303, 0x0404, 0x0505, 0x0606, 0x0707, 0x0808, 0x0909, 0x0a0a,
        0x0b0b, 0x0c0c,
    ];
    let bf16 = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0x3eaa, 0x0080, 0x7f7f, 0x1234,
        0xabcd, 0x1010, 0x2020, 0x3030, 0x4040, 0x5050, 0x6060, 0x7070, 0x8080, 0x9090, 0xa0a0,
        0xb0b0, 0xc0c0,
    ];
    assert_native_two_byte_plane_copy(device, seismic::Element::f16(), &f16);
    assert_native_two_byte_plane_copy(device, seismic::Element::bf16(), &bf16);
}

#[test]
fn native_copy_across_slab_boundary() {
    for device in devices() {
        native_copy_across_slab_boundary_on(&device);
    }
}

fn native_copy_across_slab_boundary_on(device: &seismic::Device) {
    let element = seismic::Element::u32();
    let words = |values: &[u32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let rows = slabbed_rows(
        device,
        element,
        3,
        &[1, 4],
        &words(&[2, 3, 5, 7, 99, 99, 99, 99, 99, 99, 99, 99]),
        2,
    );
    let mut rows_region = rows.logical_region(0).unwrap();
    let from =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[1], &0_i32.to_le_bytes())
            .unwrap();
    let to =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[1], &2_i32.to_le_bytes())
            .unwrap();
    copy_rows::native_for_device_with(
        &device,
        copy_rows::Elements { A: element },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call_into(
        copy_rows::Args {
            rows: &mut rows_region,
            from: &from,
            to: &to,
            slab_rows: 2,
        },
        copy_rows::OutputArgs,
    )
    .unwrap();
    let actual = rows_region
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(&actual[0..4], &[2, 3, 5, 7], "{:?}", device.backend());
    assert_eq!(&actual[4..8], &[99; 4], "{:?}", device.backend());
    assert_eq!(&actual[8..12], &[2, 3, 5, 7], "{:?}", device.backend());
}
