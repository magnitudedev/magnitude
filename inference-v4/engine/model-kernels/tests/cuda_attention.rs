//! The CUDA `gated_attention_decode` / `gated_attention_prefill` against the
//! same cases, host model and portable-body pin as the Metal tests (included
//! verbatim; their Metal tests skip without a Metal device), plus timings.

include!("attention.rs");

fn cuda() -> Option<Device> {
    DeviceCatalog::discover()
        .ok()?
        .open_backend(BackendName::Cuda)
        .ok()
}

fn cuda_decode(
    device: &Device,
    geometry: Geometry,
    (parts, warps): (u64, u64),
) -> seismic::NativeKernel<gated_attention_decode::Entry> {
    gated_attention_decode::native_for_device_with(
        device,
        gated_attention_decode::Elements { A: Element::bf16() },
        &statics(geometry)
            .with_param("PARTS", parts)
            .with_param("WARPS", warps),
    )
    .unwrap()
}

fn cuda_prefill(
    device: &Device,
    geometry: Geometry,
    warps: u64,
) -> seismic::NativeKernel<gated_attention_prefill::Entry> {
    gated_attention_prefill::native_for_device_with(
        device,
        gated_attention_prefill::Elements { A: Element::bf16() },
        &statics(geometry).with_param("WARPS", warps),
    )
    .unwrap()
}

const DECODE_CONFIGS: [(u64, u64); 4] = [(12, 4), (24, 8), (48, 4), (12, 8)];

#[test]
fn cuda_decode_matches_portable_body() {
    let Some(device) = cuda() else { return };
    let case = Case::new(SMALL, 64, 2, &decode_rows(5), 11);
    check_host_model_against_portable_body("gated_attention_decode", &case);
    let expected = case.expected();
    for config in DECODE_CONFIGS {
        let kernel = cuda_decode(&device, SMALL, config);
        let mut bound = Bound::new(&device, &case);
        let gated = run_decode(&kernel, &mut bound, &case);
        check(
            &format!("cuda small decode {config:?}"),
            &case,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn cuda_prefill_matches_portable_body() {
    let Some(device) = cuda() else { return };
    let case = Case::new(SMALL, 128, 2, &prefill_rows(20, 23), 23);
    check_host_model_against_portable_body("gated_attention_prefill", &case);
    let expected = case.expected();
    for warps in [4, 2] {
        let kernel = cuda_prefill(&device, SMALL, warps);
        let mut bound = Bound::new(&device, &case);
        let gated = run_prefill(&kernel, &mut bound, &case);
        check(
            &format!("cuda small prefill WARPS={warps}"),
            &case,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn cuda_qwen_geometry_decode_and_prefill_match_host_model() {
    let Some(device) = cuda() else { return };
    for context in [256, 4096, 16384] {
        let case = Case::new(QWEN, context + 128, 2, &decode_rows(context as i32 - 40), 5);
        let expected = case.expected();
        for config in DECODE_CONFIGS {
            let kernel = cuda_decode(&device, QWEN, config);
            let mut bound = Bound::new(&device, &case);
            let gated = run_decode(&kernel, &mut bound, &case);
            check(
                &format!("cuda decode context {context} {config:?}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
    for (rows, history) in [(40, 300), (128, 1000)] {
        let case = Case::new(
            QWEN,
            history as usize + 256,
            2,
            &prefill_rows(rows, history),
            7,
        );
        let expected = case.expected();
        for warps in [4, 2] {
            let kernel = cuda_prefill(&device, QWEN, warps);
            let mut bound = Bound::new(&device, &case);
            let gated = run_prefill(&kernel, &mut bound, &case);
            check(
                &format!("cuda prefill {rows} rows WARPS={warps}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

/// Device time per call at the 4B geometry: decode of one row at contexts
/// 256 / 4k / 16k and a 128-row prefill chunk after 1k of history. Each
/// rotation entry owns its histories, so rotations exceed the 24 MiB L2 at
/// long contexts.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn cuda_attention_timings() {
    let Some(device) = cuda() else { return };
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.002,
    };
    for context in [256usize, 4096, 16384] {
        let rows = vec![Row {
            spans: vec![(0, context as i32 - 1)],
            fresh: (0, 1),
            destination: context as i32 - 1,
            position: context as i32 - 1,
        }];
        let case = Case::new(QWEN, context, 1, &rows, 3);
        let rotation = if context >= 4096 { 3 } else { 1 };
        let mut bounds = (0..rotation)
            .map(|_| Bound::new(&device, &case))
            .collect::<Vec<_>>();
        for config in DECODE_CONFIGS {
            let kernel = cuda_decode(&device, QWEN, config);
            let args = bounds
                .iter_mut()
                .map(|bound| gated_attention_decode::Args {
                    query_gate: &bound.query_gate,
                    key: &bound.key,
                    value: &bound.value,
                    query_norm: &bound.query_norm,
                    key_norm: &bound.key_norm,
                    rotary_components: &bound.components,
                    coordinates: &bound.coordinates,
                    visible: &bound.visible,
                    fresh: &bound.fresh,
                    destinations: &bound.destinations,
                    history_key: &mut bound.history_key,
                    history_value: &mut bound.history_value,
                    rotary_frequencies: &bound.frequencies,
                    epsilon: case.epsilon,
                    scale: case.scale,
                })
                .collect();
            let measured = kernel.measure(args, &options).unwrap();
            let bytes = (context * QWEN.kv * QWEN.w() * 2 * 2) as f64;
            println!(
                "decode context {context} PARTS,WARPS {config:?}: {:.1} us, KV read {:.0} GB/s",
                measured.median * 1e6,
                bytes / measured.median / 1e9
            );
        }
    }
    for (rows, history) in [(128usize, 1024i32), (128, 0)] {
        let spans = if history > 0 {
            vec![(0, history)]
        } else {
            vec![]
        };
        let rows_spec = (0..rows)
            .map(|row| Row {
                spans: spans.clone(),
                fresh: (0, row as i32 + 1),
                destination: history + row as i32,
                position: history + row as i32,
            })
            .collect::<Vec<_>>();
        let case = Case::new(QWEN, history as usize + rows, 1, &rows_spec, 9);
        let mut bound = Bound::new(&device, &case);
        for warps in [4, 2] {
            let kernel = cuda_prefill(&device, QWEN, warps);
            let args = vec![gated_attention_prefill::Args {
                query_gate: &bound.query_gate,
                key: &bound.key,
                value: &bound.value,
                query_norm: &bound.query_norm,
                key_norm: &bound.key_norm,
                rotary_components: &bound.components,
                coordinates: &bound.coordinates,
                visible: &bound.visible,
                fresh: &bound.fresh,
                destinations: &bound.destinations,
                history_key: &mut bound.history_key,
                history_value: &mut bound.history_value,
                rotary_frequencies: &bound.frequencies,
                epsilon: case.epsilon,
                scale: case.scale,
            }];
            let measured = kernel.measure(args, &options).unwrap();
            println!(
                "prefill {rows} rows after {history} history WARPS={warps}: {:.1} us",
                measured.median * 1e6
            );
        }
    }
}
