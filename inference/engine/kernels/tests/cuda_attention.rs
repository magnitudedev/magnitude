//! The CUDA `attention_decode` / `attention_prefill` in Qwen's form against
//! the same cases, host model and portable-body pin as the Metal tests
//! (included verbatim; their Metal tests skip without a Metal device).

include!("attention.rs");

fn cuda() -> Option<Device> {
    seismic::Devices::discover(&[])
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
            .with_static("SK", 0)
            .with_param("PARTS", parts)
            .with_param("WARPS", warps)
            .with_param("SLICES", slices)
            .with_param("MATRIX", 0)
            .with_param("STAGES", 2)
            .with_param("COLUMNS", 1)
            .with_param("QREG", 0),
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
        &visible_prefill(device, qwen_form(statics(geometry), geometry))
            .with_static("SK", 0)
            .with_param("WARPS", warps)
            .with_param("SPLIT_GROUPS", split_groups)
            .with_param("STAGES", 2)
            .with_param("COLUMNS", 1)
            .with_param("QREG", 0),
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

/// Visible spans crossing slab edges (one span over several slabs) at every
/// decode configuration.
#[test]
fn cuda_decode_reads_spans_crossing_history_slabs() {
    let Some(device) = cuda() else { return };
    let rows = [
        Row {
            spans: vec![(0, 200)],
            fresh: (0, 1),
            destination: 200,
            position: 200,
        },
        Row {
            spans: vec![(7, 150), (170, 199)],
            fresh: (1, 2),
            destination: 201,
            position: 199,
        },
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
