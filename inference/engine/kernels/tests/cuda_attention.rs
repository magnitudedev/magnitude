//! The CUDA `attention_decode` / `attention_prefill` in Qwen's form against
//! the same cases, host model and portable-body pin as the Metal tests
//! (included verbatim; their Metal tests skip without a Metal device), plus
//! timings.

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
    (parts, warps, slices): (u64, u64, u64),
) -> seismic::NativeKernel<attention_decode::Entry> {
    attention_decode::native_for_device_with(
        device,
        attention_decode::Elements { A: Element::bf16() },
        &qwen_form(statics(geometry), geometry)
            .with_param("PARTS", parts)
            .with_param("WARPS", warps)
            .with_param("SLICES", slices),
    )
    .unwrap()
}

fn cuda_prefill(
    device: &Device,
    geometry: Geometry,
    (warps, split_groups): (u64, u64),
) -> seismic::NativeKernel<attention_prefill::Entry> {
    attention_prefill::native_for_device_with(
        device,
        attention_prefill::Elements { A: Element::bf16() },
        &qwen_form(statics(geometry), geometry)
            .with_param("WARPS", warps)
            .with_param("SPLIT_GROUPS", split_groups),
    )
    .unwrap()
}

/// (PARTS, WARPS, SLICES); every tested group divides by 2.
const DECODE_CONFIGS: [(u64, u64, u64); 4] = [(12, 4, 1), (24, 8, 2), (48, 4, 1), (12, 8, 1)];
/// (WARPS, SPLIT_GROUPS).
const PREFILL_CONFIGS: [(u64, u64); 3] = [(4, 1), (2, 1), (4, 256)];

#[test]
fn cuda_decode_matches_portable_body() {
    let Some(device) = cuda() else { return };
    let case = Case::new(SMALL, 64, 2, &decode_rows(5), 11);
    check_host_model_against_portable_body("attention_decode", &case);
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

/// Visible spans crossing slab edges (the device measurement's binding: one
/// span over several slabs) at every decode configuration.
#[test]
fn cuda_decode_reads_spans_crossing_history_slabs() {
    let Some(device) = cuda() else { return };
    let rows = [
        Row { spans: vec![(0, 200)], fresh: (0, 1), destination: 200, position: 200 },
        Row { spans: vec![(7, 150), (170, 199)], fresh: (1, 2), destination: 201, position: 199 },
    ];
    let case = Case::new(SMALL, 224, 2, &rows, 43);
    let expected = case.expected();
    for config in DECODE_CONFIGS {
        let kernel = cuda_decode(&device, SMALL, config);
        // A reused middle slab puts consecutive slabs at unrelated addresses.
        for reused in [false, true] {
            let mut bound = Bound::new_with_options(&device, &case, 32, reused);
            let gated = run_decode(&kernel, &mut bound, &case);
            check(
                &format!("cuda crossing-span decode {config:?} reused={reused}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

#[test]
fn cuda_prefill_matches_portable_body() {
    let Some(device) = cuda() else { return };
    let case = Case::new(SMALL, 128, 2, &prefill_rows(20, 23), 23);
    check_host_model_against_portable_body("attention_prefill", &case);
    let expected = case.expected();
    for config in PREFILL_CONFIGS {
        let kernel = cuda_prefill(&device, SMALL, config);
        let mut bound = Bound::new(&device, &case);
        let gated = run_prefill(&kernel, &mut bound, &case);
        check(
            &format!("cuda small prefill {config:?}"),
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
        for config in PREFILL_CONFIGS {
            let kernel = cuda_prefill(&device, QWEN, config);
            let mut bound = Bound::new(&device, &case);
            let gated = run_prefill(&kernel, &mut bound, &case);
            check(
                &format!("cuda prefill {rows} rows {config:?}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

#[test]
fn cuda_eight_query_group_decode_and_prefill_match_host_model() {
    let Some(device) = cuda() else { return };
    for context in [256, 4096] {
        let case = Case::new(
            QWEN35B,
            context + 128,
            2,
            &decode_rows(context as i32 - 40),
            5,
        );
        let expected = case.expected();
        for config in DECODE_CONFIGS {
            let kernel = cuda_decode(&device, QWEN35B, config);
            let mut bound = Bound::new(&device, &case);
            let gated = run_decode(&kernel, &mut bound, &case);
            check(
                &format!("cuda group-8 decode context {context} {config:?}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
    for (rows, history) in [(40, 300), (128, 1000), (64, 4096)] {
        let case = Case::new(
            QWEN35B,
            history as usize + 256,
            2,
            &prefill_rows(rows, history),
            7,
        );
        let expected = case.expected();
        for config in PREFILL_CONFIGS {
            let kernel = cuda_prefill(&device, QWEN35B, config);
            let mut bound = Bound::new(&device, &case);
            let gated = run_prefill(&kernel, &mut bound, &case);
            check(
                &format!("cuda group-8 prefill {rows} rows {config:?}"),
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
                .map(|bound| args!(attention_decode, bound, case))
                .collect();
            let measured = kernel.measure(args, &options).unwrap();
            let bytes = (context * QWEN.kv * QWEN.w() * 2 * 2) as f64;
            println!(
                "decode context {context} PARTS,WARPS,SLICES {config:?}: {:.1} us, KV read {:.0} GB/s",
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
        for config in PREFILL_CONFIGS {
            let kernel = cuda_prefill(&device, QWEN, config);
            let measured = kernel.measure(vec![args!(attention_prefill, bound, case)], &options).unwrap();
            println!(
                "prefill {rows} rows after {history} history WARPS,SPLIT_GROUPS {config:?}: {:.1} us",
                measured.median * 1e6
            );
        }
    }
}
