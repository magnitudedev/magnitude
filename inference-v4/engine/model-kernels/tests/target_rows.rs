use magnitude_model_kernels::{
    qwen_conditioning_overlay, qwen_dense_expand, qwen_dense_expand_demanded, qwen_dense_output,
    qwen_dense_output_demanded, qwen_dense_rows, qwen_dense_rows_demanded, qwen_embedding_rows,
    qwen_features_rows, qwen_readout_rows, qwen_selected_rows,
};

const M: usize = 3;
const D: usize = 4;
const V: usize = 5;
const O: usize = 2;
const F: usize = 3;
const EPSILON: f32 = 1e-5;
type DenseFixtures = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(target_os = "macos")]
#[test]
fn packed_q8_residents_execute_dense_embedding_and_readout_with_bf16_activations() {
    use magnitude_model_kernels::repack_weight;

    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let q8_external = seismic::Element::named("gguf_q8_0").unwrap();
    let q8_resident = seismic::Element::named("q8g32s").unwrap();
    let resident = |groups: usize| {
        let mut bytes = Vec::with_capacity(groups * 34);
        for _ in 0..groups {
            bytes.extend_from_slice(&seismic_lang::registry::f16_bits(1.0).to_le_bytes());
            bytes.extend((0..32).map(|index| if index == 0 { 1u8 } else { 0u8 }));
        }
        let source =
            seismic::Tensor::from_host(&device, q8_external, &[(groups * 32) as u64], &bytes)
                .unwrap();
        repack_weight::native_for_device_with(
            &device,
            repack_weight::Elements {
                E: q8_external,
                U: q8_resident,
            },
        )
        .unwrap()
        .call(repack_weight::Args { source: &source })
        .unwrap()
        .value
    };
    let f32_tensor = |shape: &[u64], values: &[f32]| {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        seismic::Tensor::from_host(&device, seismic::Element::f32(), shape, &bytes).unwrap()
    };
    let bf16_tensor = |shape: &[u64], values: &[f32]| {
        let bytes = values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        seismic::Tensor::from_host(&device, seismic::Element::bf16(), shape, &bytes).unwrap()
    };

    let residual_values = (0..32)
        .map(|index| if index == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let residual = f32_tensor(&[1, 32], &residual_values);
    let norm = bf16_tensor(&[32], &vec![1.0; 32]);
    let gate = resident(32).reshape(&[32, 32]).unwrap();
    let up = resident(32).reshape(&[32, 32]).unwrap();
    let down = resident(32).reshape(&[32, 32]).unwrap();
    let dense = qwen_dense_rows::native_for_device_with(
        &device,
        qwen_dense_rows::Elements {
            NW: seismic::Element::bf16(),
            GW: q8_resident,
            UW: q8_resident,
            DW: q8_resident,
            A: seismic::Element::bf16(),
        },
    )
    .unwrap()
    .call(qwen_dense_rows::Args {
        residual: &residual,
        norm: &norm,
        gate_weight: &gate,
        up_weight: &up,
        down_weight: &down,
        eps: 1e-5,
    })
    .unwrap()
    .value;
    assert!(read_f32(&dense).iter().all(|value| value.is_finite()));

    let table = resident(2).reshape(&[2, 32]).unwrap();
    let token =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[1], &0i32.to_le_bytes())
            .unwrap();
    let embedded = qwen_embedding_rows::native_for_device_with(
        &device,
        qwen_embedding_rows::Elements {
            EW: q8_resident,
            A: seismic::Element::bf16(),
        },
    )
    .unwrap()
    .call(qwen_embedding_rows::Args {
        table: &table,
        tokens: &token,
    })
    .unwrap();
    assert!(read_f32(&embedded.r1).iter().all(|value| value.is_finite()));

    let out_rows =
        seismic::Tensor::from_host(&device, seismic::Element::i32(), &[1], &0i32.to_le_bytes())
            .unwrap();
    let readout = qwen_readout_rows::native_for_device_with(
        &device,
        qwen_readout_rows::Elements {
            NW: seismic::Element::bf16(),
            OW: q8_resident,
            A: seismic::Element::bf16(),
        },
    )
    .unwrap()
    .call(qwen_readout_rows::Args {
        hidden: &residual,
        norm: &norm,
        weight: &table,
        out_rows: &out_rows,
        epsilon: 1e-5,
    })
    .unwrap();
    assert!(read_f32(&readout.r1).iter().all(|value| value.is_finite()));
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read_f32(tensor: &seismic::Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-5,
            "value {index}: {actual} != {expected}"
        );
    }
}

fn fixtures() -> DenseFixtures {
    let hidden = vec![
        1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5, 3.0, -2.0, 1.0, 0.25,
    ];
    let norm = vec![1.0, 0.5, 1.5, -1.0];
    let gate = vec![
        0.2, -0.1, 0.3, 0.4, -0.3, 0.5, 0.1, -0.2, 0.6, 0.2, -0.4, 0.3,
    ];
    let up = vec![
        -0.2, 0.4, 0.1, 0.3, 0.5, -0.1, 0.2, 0.6, -0.3, 0.1, 0.2, 0.4,
    ];
    let down = vec![
        0.4, -0.2, 0.1, 0.3, 0.5, -0.4, -0.1, 0.2, 0.6, 0.7, -0.3, 0.2,
    ];
    (hidden, norm, gate, up, down)
}

fn normalized(hidden: &[f32], norm: &[f32], row: usize) -> Vec<f32> {
    let source = &hidden[row * D..(row + 1) * D];
    let inverse = (source.iter().map(|value| value * value).sum::<f32>() / D as f32 + EPSILON)
        .sqrt()
        .recip();
    source
        .iter()
        .zip(norm)
        .map(|(value, norm)| value * inverse * norm)
        .collect()
}

fn dense_reference(
    hidden: &[f32],
    norm: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    rows: &[usize],
) -> Vec<f32> {
    let mut result = Vec::with_capacity(rows.len() * D);
    for &row in rows {
        let x = normalized(hidden, norm, row);
        let product = (0..F)
            .map(|feature| {
                let g = (0..D)
                    .map(|source| x[source] * gate[feature * D + source])
                    .sum::<f32>();
                let u = (0..D)
                    .map(|source| x[source] * up[feature * D + source])
                    .sum::<f32>();
                g / (1.0 + (-g).exp()) * u
            })
            .collect::<Vec<_>>();
        for column in 0..D {
            result.push(
                hidden[row * D + column]
                    + (0..F)
                        .map(|feature| product[feature] * down[column * F + feature])
                        .sum::<f32>(),
            );
        }
    }
    result
}

#[test]
fn checked_entries_expose_both_planned_and_native_routes() {
    let _ = qwen_embedding_rows::for_device_with;
    let _ = qwen_embedding_rows::native_for_device_with;
    let _ = qwen_dense_rows::for_device_with;
    let _ = qwen_dense_rows::native_for_device_with;
    let _ = qwen_dense_rows_demanded::for_device_with;
    let _ = qwen_dense_rows_demanded::native_for_device_with;
    let _ = qwen_readout_rows::for_device_with;
    let _ = qwen_readout_rows::native_for_device_with;
    let _ = qwen_features_rows::for_device_with;
    let _ = qwen_features_rows::native_for_device_with;
    let _ = qwen_selected_rows::for_device_with;
    let _ = qwen_selected_rows::native_for_device_with;
    let _ = qwen_conditioning_overlay::for_device;
    let _ = qwen_conditioning_overlay::native_for_device;
}

#[cfg(target_os = "macos")]
#[test]
fn conditioning_overlay_copies_the_complete_row_table() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let values = [1.0f32, -2.0, 3.5, 4.0, 0.25, -6.0];
    let input = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[2, 3],
        &f32_bytes(&values),
    )
    .unwrap();
    let mut out = seismic::Tensor::zeros(&device, seismic::Element::f32(), &[2, 3]).unwrap();
    qwen_conditioning_overlay::native_for_device(&device)
        .unwrap()
        .call(qwen_conditioning_overlay::Args {
            input: &input,
            out: &mut out,
        })
        .unwrap();
    assert_eq!(read_f32(&out), values);
}

#[test]
fn checked_portable_partition_executes_every_new_entry_in_the_interpreter() {
    use seismic_lang::{
        checked::{check_source, SourceFile},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "target_rows.seismic".into(),
        text: include_str!("../kernels/target_rows.seismic").into(),
    });
    sources.push(SourceFile {
        path: "dense_rows.seismic".into(),
        text: include_str!("../kernels/dense_rows.seismic").into(),
    });
    sources.push(SourceFile {
        path: "readout.seismic".into(),
        text: include_str!("../kernels/readout.seismic").into(),
    });
    let module = check_source(sources).unwrap();
    let f32_element = registry::dense(DType::F32);
    let f16_element = registry::dense(DType::F16);
    let (hidden, norm, gate, up, down) = fixtures();
    let weight = (0..V * D)
        .map(|index| index as f32 * 0.03 - 0.2)
        .collect::<Vec<_>>();
    let run = |name: &str,
                   bindings: Vec<(&str, _)>,
                   tensors: Vec<(DType, Vec<usize>, Vec<f64>)>,
                   scalars: Vec<f32>| {
        let elements = bindings
            .into_iter()
            .fold(ElementBindings::new(), |bindings, (name, value)| {
                bindings.bind(name, value)
            });
        let logical = module
            .entry(module.entry_named(name).unwrap(), &elements)
            .unwrap();
        let mut interpreter = Interpreter::new(&logical);
        let mut args = tensors
            .into_iter()
            .map(|(dtype, shape, data)| {
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, data)))
            })
            .collect::<Vec<_>>();
        args.extend(
            scalars
                .into_iter()
                .map(|value| Arg::Scalar(ReferenceScalar::F32(value.to_bits()))),
        );
        let outcome = interpreter
            .run(&args)
            .unwrap_or_else(|error| panic!("{name} interpreter error: {error}"));
        // Deterministic: no reassociation is allowed. An entered parallel
        // region only describes failure prefixes, and this outcome returns.
        assert!(
            outcome
                .relation()
                .is_none_or(|relation| relation.associations().is_empty()),
            "{name} must be deterministic"
        );
        match outcome.termination() {
            SourceTermination::Returned(_) => assert!(outcome.results().len() > 0),
            SourceTermination::Failed(failure) => panic!("{name} failed: {failure}"),
        }
    };
    let floats = |shape: Vec<usize>, values: &[f32]| {
        (
            DType::F32,
            shape,
            values.iter().map(|value| f64::from(*value)).collect(),
        )
    };
    let ints = |values: &[i32]| {
        (
            DType::I32,
            vec![values.len()],
            values.iter().map(|value| f64::from(*value)).collect(),
        )
    };
    run(
        "qwen_embedding_rows",
        vec![("EW", f32_element), ("A", f16_element)],
        vec![floats(vec![V, D], &weight), ints(&[0, 2, 4])],
        vec![],
    );
    run(
        "qwen_dense_rows",
        vec![
            ("NW", f32_element),
            ("GW", f32_element),
            ("UW", f32_element),
            ("DW", f32_element),
            ("A", f16_element),
        ],
        vec![
            floats(vec![M, D], &hidden),
            floats(vec![D], &norm),
            floats(vec![F, D], &gate),
            floats(vec![F, D], &up),
            floats(vec![D, F], &down),
        ],
        vec![EPSILON],
    );
    for rows in [&[2, 0][..], &[1][..]] {
        run(
            "qwen_dense_rows_demanded",
            vec![
                ("NW", f32_element),
                ("GW", f32_element),
                ("UW", f32_element),
                ("DW", f32_element),
                ("A", f16_element),
            ],
            vec![
                floats(vec![M, D], &hidden),
                floats(vec![D], &norm),
                floats(vec![F, D], &gate),
                floats(vec![F, D], &up),
                floats(vec![D, F], &down),
                ints(rows),
            ],
            vec![EPSILON],
        );
        run(
            "qwen_readout_rows",
            vec![("NW", f32_element), ("OW", f32_element), ("A", f16_element)],
            vec![
                floats(vec![M, D], &hidden),
                floats(vec![D], &norm),
                floats(vec![V, D], &weight),
                ints(rows),
            ],
            vec![EPSILON],
        );
        run(
            "qwen_features_rows",
            vec![("NW", f32_element), ("A", f16_element)],
            vec![
                floats(vec![M, D], &hidden),
                floats(vec![D], &norm),
                ints(rows),
            ],
            vec![EPSILON],
        );
        run(
            "qwen_selected_rows",
            vec![("NW", f32_element), ("OW", f32_element), ("A", f16_element)],
            vec![
                floats(vec![M, D], &hidden),
                floats(vec![D], &norm),
                floats(vec![V, D], &weight),
                ints(rows),
                ints(&[4, 1]),
            ],
            vec![EPSILON],
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_target_rows_match_host_oracles() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let f32e = seismic::Element::f32();
    let tensor = |shape: &[u64], values: &[f32]| {
        seismic::Tensor::from_host(&device, f32e, shape, &f32_bytes(values)).unwrap()
    };
    let indices = |values: &[i32]| {
        seismic::Tensor::from_host(
            &device,
            seismic::Element::i32(),
            &[values.len() as u64],
            &i32_bytes(values),
        )
        .unwrap()
    };
    let (hidden_values, norm_values, gate_values, up_values, down_values) = fixtures();
    let hidden = tensor(&[M as u64, D as u64], &hidden_values);
    let norm = tensor(&[D as u64], &norm_values);
    let gate = tensor(&[F as u64, D as u64], &gate_values);
    let up = tensor(&[F as u64, D as u64], &up_values);
    let down = tensor(&[D as u64, F as u64], &down_values);
    let dense = qwen_dense_rows::native_for_device_with(
        &device,
        qwen_dense_rows::Elements {
            NW: f32e,
            GW: f32e,
            UW: f32e,
            DW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_dense_rows::Args {
        residual: &hidden,
        norm: &norm,
        gate_weight: &gate,
        up_weight: &up,
        down_weight: &down,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    assert_close(
        &read_f32(&dense),
        &dense_reference(
            &hidden_values,
            &norm_values,
            &gate_values,
            &up_values,
            &down_values,
            &[0, 1, 2],
        ),
    );
    let out_rows = indices(&[2, 0]);
    let demanded = qwen_dense_rows_demanded::native_for_device_with(
        &device,
        qwen_dense_rows_demanded::Elements {
            NW: f32e,
            GW: f32e,
            UW: f32e,
            DW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_dense_rows_demanded::Args {
        residual: &hidden,
        norm: &norm,
        gate_weight: &gate,
        up_weight: &up,
        down_weight: &down,
        out_rows: &out_rows,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    assert_close(
        &read_f32(&demanded),
        &dense_reference(
            &hidden_values,
            &norm_values,
            &gate_values,
            &up_values,
            &down_values,
            &[2, 0],
        ),
    );
    let expand = qwen_dense_expand::native_for_device_with(
        &device,
        qwen_dense_expand::Elements {
            NW: f32e,
            GW: f32e,
            UW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_dense_expand::Args {
        residual: &hidden,
        norm: &norm,
        gate_weight: &gate,
        up_weight: &up,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    let staged = qwen_dense_output::native_for_device_with(
        &device,
        qwen_dense_output::Elements { DW: f32e, A: f32e },
    )
    .unwrap()
    .call(qwen_dense_output::Args {
        residual: &hidden,
        product: &expand,
        down_weight: &down,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&staged), &read_f32(&dense));
    let demanded_expand = qwen_dense_expand_demanded::native_for_device_with(
        &device,
        qwen_dense_expand_demanded::Elements {
            NW: f32e,
            GW: f32e,
            UW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_dense_expand_demanded::Args {
        residual: &hidden,
        norm: &norm,
        gate_weight: &gate,
        up_weight: &up,
        out_rows: &out_rows,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    let demanded_staged = qwen_dense_output_demanded::native_for_device_with(
        &device,
        qwen_dense_output_demanded::Elements { DW: f32e, A: f32e },
    )
    .unwrap()
    .call(qwen_dense_output_demanded::Args {
        residual: &hidden,
        product: &demanded_expand,
        down_weight: &down,
        out_rows: &out_rows,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&demanded_staged), &read_f32(&demanded));

    let table_values = (0..V * D)
        .map(|index| index as f32 * 0.05 - 0.3)
        .collect::<Vec<_>>();
    let table = tensor(&[V as u64, D as u64], &table_values);
    let tokens = indices(&[4, 1, 3]);
    let embedded = qwen_embedding_rows::native_for_device_with(
        &device,
        qwen_embedding_rows::Elements { EW: f32e, A: f32e },
    )
    .unwrap()
    .call(qwen_embedding_rows::Args {
        table: &table,
        tokens: &tokens,
    })
    .unwrap();
    let expected_embedding = [4usize, 1, 3]
        .into_iter()
        .flat_map(|row| table_values[row * D..(row + 1) * D].iter().copied())
        .collect::<Vec<_>>();
    assert_close(&read_f32(&embedded.r0), &expected_embedding);
    assert_close(&read_f32(&embedded.r1), &expected_embedding);

    let expected_features = [2usize, 0]
        .into_iter()
        .flat_map(|row| normalized(&hidden_values, &norm_values, row))
        .collect::<Vec<_>>();
    let features = qwen_features_rows::native_for_device_with(
        &device,
        qwen_features_rows::Elements { NW: f32e, A: f32e },
    )
    .unwrap()
    .call(qwen_features_rows::Args {
        hidden: &hidden,
        norm: &norm,
        out_rows: &out_rows,
        epsilon: EPSILON,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&features), &expected_features);
    let logits2 = |row: usize, vocabulary: usize| {
        (0..D)
            .map(|source| {
                expected_features[row * D + source] * table_values[vocabulary * D + source]
            })
            .sum::<f32>()
    };
    let readout = qwen_readout_rows::native_for_device_with(
        &device,
        qwen_readout_rows::Elements {
            NW: f32e,
            OW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_readout_rows::Args {
        hidden: &hidden,
        norm: &norm,
        weight: &table,
        out_rows: &out_rows,
        epsilon: EPSILON,
    })
    .unwrap();
    assert_close(&read_f32(&readout.r0), &expected_features);
    let expected_logits = (0..O)
        .flat_map(|row| (0..V).map(move |vocabulary| logits2(row, vocabulary)))
        .collect::<Vec<_>>();
    assert_close(&read_f32(&readout.r1), &expected_logits);
    let selected_ids = indices(&[4, 1]);
    let selected = qwen_selected_rows::native_for_device_with(
        &device,
        qwen_selected_rows::Elements {
            NW: f32e,
            OW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_selected_rows::Args {
        hidden: &hidden,
        norm: &norm,
        weight: &table,
        out_rows: &out_rows,
        selected: &selected_ids,
        epsilon: EPSILON,
    })
    .unwrap();
    assert_close(&read_f32(&selected.r0), &expected_features);
    let expected_selected = (0..O)
        .flat_map(|row| {
            [4usize, 1]
                .into_iter()
                .map(move |vocabulary| logits2(row, vocabulary))
        })
        .collect::<Vec<_>>();
    assert_close(&read_f32(&selected.r1), &expected_selected);

    // A distinct one-row partition exercises the same checked entries without
    // relying on the two-row launch geometry above.
    let one_row = indices(&[1]);
    let one_features_expected = normalized(&hidden_values, &norm_values, 1);
    let one_features = qwen_features_rows::native_for_device_with(
        &device,
        qwen_features_rows::Elements { NW: f32e, A: f32e },
    )
    .unwrap()
    .call(qwen_features_rows::Args {
        hidden: &hidden,
        norm: &norm,
        out_rows: &one_row,
        epsilon: EPSILON,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&one_features), &one_features_expected);
    let one_demanded = qwen_dense_rows_demanded::native_for_device_with(
        &device,
        qwen_dense_rows_demanded::Elements {
            NW: f32e,
            GW: f32e,
            UW: f32e,
            DW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_dense_rows_demanded::Args {
        residual: &hidden,
        norm: &norm,
        gate_weight: &gate,
        up_weight: &up,
        down_weight: &down,
        out_rows: &one_row,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    assert_close(
        &read_f32(&one_demanded),
        &dense_reference(
            &hidden_values,
            &norm_values,
            &gate_values,
            &up_values,
            &down_values,
            &[1],
        ),
    );
    let one_readout = qwen_readout_rows::native_for_device_with(
        &device,
        qwen_readout_rows::Elements {
            NW: f32e,
            OW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_readout_rows::Args {
        hidden: &hidden,
        norm: &norm,
        weight: &table,
        out_rows: &one_row,
        epsilon: EPSILON,
    })
    .unwrap();
    assert_close(&read_f32(&one_readout.r0), &one_features_expected);
    let one_logits = (0..V)
        .map(|vocabulary| {
            (0..D)
                .map(|source| one_features_expected[source] * table_values[vocabulary * D + source])
                .sum()
        })
        .collect::<Vec<f32>>();
    assert_close(&read_f32(&one_readout.r1), &one_logits);
    let one_selected = qwen_selected_rows::native_for_device_with(
        &device,
        qwen_selected_rows::Elements {
            NW: f32e,
            OW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_selected_rows::Args {
        hidden: &hidden,
        norm: &norm,
        weight: &table,
        out_rows: &one_row,
        selected: &selected_ids,
        epsilon: EPSILON,
    })
    .unwrap();
    assert_close(&read_f32(&one_selected.r0), &one_features_expected);
    assert_close(&read_f32(&one_selected.r1), &[one_logits[4], one_logits[1]]);

    // Production uses BF16 activations with dense norm weights and may bind
    // F16 dense projections. Qualify that ABI path separately from the F32
    // arithmetic oracle above.
    let f16_tensor = |shape: &[u64], values: &[f32]| {
        let bytes = values
            .iter()
            .flat_map(|value| seismic_lang::registry::f16_bits(*value).to_le_bytes())
            .collect::<Vec<_>>();
        seismic::Tensor::from_host(&device, seismic::Element::f16(), shape, &bytes).unwrap()
    };
    let bf16_tensor = |shape: &[u64], values: &[f32]| {
        let bytes = values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        seismic::Tensor::from_host(&device, seismic::Element::bf16(), shape, &bytes).unwrap()
    };
    let norm_bf16 = bf16_tensor(&[D as u64], &norm_values);
    let gate_f16 = f16_tensor(&[F as u64, D as u64], &gate_values);
    let up_f16 = f16_tensor(&[F as u64, D as u64], &up_values);
    let down_f16 = f16_tensor(&[D as u64, F as u64], &down_values);
    let production = qwen_dense_rows::native_for_device_with(
        &device,
        qwen_dense_rows::Elements {
            NW: seismic::Element::bf16(),
            GW: seismic::Element::f16(),
            UW: seismic::Element::f16(),
            DW: seismic::Element::f16(),
            A: seismic::Element::bf16(),
        },
    )
    .unwrap()
    .call(qwen_dense_rows::Args {
        residual: &hidden,
        norm: &norm_bf16,
        gate_weight: &gate_f16,
        up_weight: &up_f16,
        down_weight: &down_f16,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    assert!(read_f32(&production).iter().all(|value| value.is_finite()));
    let demanded_production = qwen_dense_rows_demanded::native_for_device_with(
        &device,
        qwen_dense_rows_demanded::Elements {
            NW: seismic::Element::bf16(),
            GW: seismic::Element::f16(),
            UW: seismic::Element::f16(),
            DW: seismic::Element::f16(),
            A: seismic::Element::bf16(),
        },
    )
    .unwrap()
    .call(qwen_dense_rows_demanded::Args {
        residual: &hidden,
        norm: &norm_bf16,
        gate_weight: &gate_f16,
        up_weight: &up_f16,
        down_weight: &down_f16,
        out_rows: &one_row,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    assert!(read_f32(&demanded_production)
        .iter()
        .all(|value| value.is_finite()));
    let norm_f16 = f16_tensor(&[D as u64], &norm_values);
    let gate_bf16 = bf16_tensor(&[F as u64, D as u64], &gate_values);
    let up_bf16 = bf16_tensor(&[F as u64, D as u64], &up_values);
    let down_bf16 = bf16_tensor(&[D as u64, F as u64], &down_values);
    let bf16_weights = qwen_dense_rows::native_for_device_with(
        &device,
        qwen_dense_rows::Elements {
            NW: seismic::Element::f16(),
            GW: seismic::Element::bf16(),
            UW: seismic::Element::bf16(),
            DW: seismic::Element::bf16(),
            A: seismic::Element::bf16(),
        },
    )
    .unwrap()
    .call(qwen_dense_rows::Args {
        residual: &hidden,
        norm: &norm_f16,
        gate_weight: &gate_bf16,
        up_weight: &up_bf16,
        down_weight: &down_bf16,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    assert!(read_f32(&bf16_weights)
        .iter()
        .all(|value| value.is_finite()));
}

#[cfg(target_os = "macos")]
#[test]
fn native_dense_stages_tile_weights_across_prefill_rows() {
    const ROWS: usize = 9;
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let f32e = seismic::Element::f32();
    let tensor = |shape: &[u64], values: &[f32]| {
        seismic::Tensor::from_host(&device, f32e, shape, &f32_bytes(values)).unwrap()
    };
    let hidden_values = (0..ROWS * D)
        .map(|index| (index as f32 - 13.0) * 0.125)
        .collect::<Vec<_>>();
    let (_, norm_values, gate_values, up_values, down_values) = fixtures();
    let hidden = tensor(&[ROWS as u64, D as u64], &hidden_values);
    let norm = tensor(&[D as u64], &norm_values);
    let gate = tensor(&[F as u64, D as u64], &gate_values);
    let up = tensor(&[F as u64, D as u64], &up_values);
    let down = tensor(&[D as u64, F as u64], &down_values);

    let product = qwen_dense_expand::native_for_device_with(
        &device,
        qwen_dense_expand::Elements {
            NW: f32e,
            GW: f32e,
            UW: f32e,
            A: f32e,
        },
    )
    .unwrap()
    .call(qwen_dense_expand::Args {
        residual: &hidden,
        norm: &norm,
        gate_weight: &gate,
        up_weight: &up,
        eps: EPSILON,
    })
    .unwrap()
    .value;
    let actual = qwen_dense_output::native_for_device_with(
        &device,
        qwen_dense_output::Elements { DW: f32e, A: f32e },
    )
    .unwrap()
    .call(qwen_dense_output::Args {
        residual: &hidden,
        product: &product,
        down_weight: &down,
    })
    .unwrap()
    .value;
    let rows = (0..ROWS).collect::<Vec<_>>();
    assert_close(
        &read_f32(&actual),
        &dense_reference(
            &hidden_values,
            &norm_values,
            &gate_values,
            &up_values,
            &down_values,
            &rows,
        ),
    );
}
