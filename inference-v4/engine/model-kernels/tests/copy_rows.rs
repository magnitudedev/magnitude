use magnitude_model_kernels::{copy_rows, gather_rows};

#[cfg(target_os = "macos")]
#[test]
fn native_gather_preserves_requested_order_and_duplicates() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let source = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[3, 4],
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let rows = seismic::Tensor::from_host(
        &device,
        seismic::Element::i32(),
        &[3],
        &[
            2_i32.to_le_bytes(),
            0_i32.to_le_bytes(),
            2_i32.to_le_bytes(),
        ]
        .concat(),
    )
    .unwrap();
    let actual = gather_rows::native_for_device(&device)
        .unwrap()
        .call(gather_rows::Args {
            source: &source,
            rows: &rows,
        })
        .unwrap()
        .value
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]
    );
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
    let src = interpreter.add_tensor(TensorData::dense(DType::U32, vec![4, 2, 3], source));
    let dst = interpreter.add_tensor(TensorData::dense(DType::U32, vec![4, 2, 3], vec![99.0; 24]));
    let from = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![3.0, 1.0]));
    let to = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![0.0, 2.0]));
    interpreter
        .run(&[
            Arg::Tensor(src),
            Arg::Tensor(dst),
            Arg::Tensor(from),
            Arg::Tensor(to),
        ])
        .unwrap();
    let TensorData::Dense { data, .. } = &interpreter.tensors[dst] else {
        panic!()
    };
    assert_eq!(&data[0..6], &[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
    assert_eq!(&data[6..12], &[99.0; 6]);
    assert_eq!(&data[12..18], &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    assert_eq!(&data[18..24], &[99.0; 6]);
}

#[test]
fn generated_surface_exposes_native_and_planned_dense_plane_bindings() {
    fn planned(
        device: &seismic::Device,
        elements: copy_rows::Elements,
    ) -> Result<seismic::Kernel<copy_rows::Entry>, seismic::LoadError> {
        copy_rows::for_device_with(device, seismic::PrecisionPolicy::Exact, elements)
    }
    fn native(
        device: &seismic::Device,
        elements: copy_rows::Elements,
    ) -> Result<seismic::NativeKernel<copy_rows::Entry>, seismic::LoadError> {
        copy_rows::native_for_device_with(device, elements)
    }
    let _ = (planned, native);
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_copies_indexed_u32_plane_rows_bit_exactly() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let source = (0_u32..24).collect::<Vec<_>>();
    let source_bytes = source
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let sentinel = vec![99_u32; 24];
    let sentinel_bytes = sentinel
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let indices = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let src =
        seismic::Tensor::from_host(&device, seismic::Element::u32(), &[4, 2, 3], &source_bytes)
            .unwrap();
    let mut dst = seismic::Tensor::from_host(
        &device,
        seismic::Element::u32(),
        &[4, 2, 3],
        &sentinel_bytes,
    )
    .unwrap();
    let from =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[3, 1]))
            .unwrap();
    let to = seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[0, 2]))
        .unwrap();
    copy_rows::native_for_device_with(
        &device,
        copy_rows::Elements {
            A: seismic::Element::u32(),
        },
    )
    .unwrap()
    .call(copy_rows::Args {
        src: &src,
        dst: &mut dst,
        from: &from,
        to: &to,
    })
    .unwrap();
    let actual = dst
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(&actual[0..6], &source[18..24]);
    assert_eq!(&actual[6..12], &[99; 6]);
    assert_eq!(&actual[12..18], &source[6..12]);
    assert_eq!(&actual[18..24], &[99; 6]);
}

#[cfg(target_os = "macos")]
fn assert_native_two_byte_plane_copy(element: seismic::Element, source: &[u16], sentinel: u16) {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let source_bytes = source
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let sentinel_values = vec![sentinel; source.len()];
    let sentinel_bytes = sentinel_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let indices = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let src = seismic::Tensor::from_host(&device, element, &[4, 2, 3], &source_bytes).unwrap();
    let mut dst =
        seismic::Tensor::from_host(&device, element, &[4, 2, 3], &sentinel_bytes).unwrap();
    let from =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[3, 1]))
            .unwrap();
    let to = seismic::Tensor::from_host(&device, seismic::Element::i32(), &[2], &indices(&[0, 2]))
        .unwrap();

    copy_rows::native_for_device_with(&device, copy_rows::Elements { A: element })
        .unwrap()
        .call(copy_rows::Args {
            src: &src,
            dst: &mut dst,
            from: &from,
            to: &to,
        })
        .unwrap();

    let actual = dst
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(&actual[0..6], &source[18..24]);
    assert_eq!(&actual[6..12], &[sentinel; 6]);
    assert_eq!(&actual[12..18], &source[6..12]);
    assert_eq!(&actual[18..24], &[sentinel; 6]);
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_copies_f16_and_bf16_plane_rows_bit_exactly() {
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
    assert_native_two_byte_plane_copy(seismic::Element::f16(), &f16, 0xdead);
    assert_native_two_byte_plane_copy(seismic::Element::bf16(), &bf16, 0xbeef);
}

#[cfg(target_os = "macos")]
#[test]
fn native_copy_into_larger_aggregate_uses_independent_row_bounds() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let words = |values: &[u32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let src = seismic::Tensor::from_host(
        &device,
        seismic::Element::u32(),
        &[1, 1, 4],
        &words(&[2, 3, 5, 7]),
    )
    .unwrap();
    let mut dst = seismic::Tensor::from_host(
        &device,
        seismic::Element::u32(),
        &[3, 1, 4],
        &words(&[99; 12]),
    )
    .unwrap();
    let from =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[1], &0_i32.to_le_bytes())
            .unwrap();
    let to =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[1], &2_i32.to_le_bytes())
            .unwrap();
    copy_rows::native_for_device_with(
        &device,
        copy_rows::Elements {
            A: seismic::Element::u32(),
        },
    )
    .unwrap()
    .call_into(
        copy_rows::Args {
            src: &src,
            dst: &mut dst,
            from: &from,
            to: &to,
        },
        copy_rows::OutputArgs,
    )
    .unwrap();
    let actual = dst
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(&actual[0..8], &[99; 8]);
    assert_eq!(&actual[8..12], &[2, 3, 5, 7]);
}
