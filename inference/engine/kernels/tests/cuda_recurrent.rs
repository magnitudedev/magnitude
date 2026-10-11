//! The CUDA `gated_delta_step` / `gated_delta_chunk` against the same
//! cases, portable-body oracle and host model as the Metal tests (included
//! verbatim; their Metal tests skip without a Metal device), plus timings.

include!("recurrent_stages.rs");

fn cuda() -> Option<Device> {
    let selected = std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() == Some("cuda");
    let catalog = match DeviceCatalog::discover() {
        Ok(catalog) => catalog,
        Err(error) if selected => panic!("selected CUDA backend discovery failed: {error}"),
        Err(_) => return None,
    };
    match catalog.open_backend(BackendName::Cuda) {
        Ok(device) => Some(device),
        Err(error) if selected => panic!("selected CUDA backend did not open: {error}"),
        Err(_) => None,
    }
}

#[test]
fn cuda_slab_bank_crossing_matches_host_model() {
    let Some(device) = cuda() else { return };
    let (_, case) = tape_cases().remove(0);
    let host = case.host();
    for reused in [false, true] {
        let step = case.cuda_with_reused_slab(&device, Element::f32(), Mapping::Step(4, 8), reused);
        check("cuda slab tape step", &case, &step, &host, (5e-4, 2e-5));
        let chunk = case.cuda_with_reused_slab(&device, Element::f32(), Mapping::Chunk(32), reused);
        check("cuda slab tape chunk", &case, &chunk, &host, (5e-4, 2e-5));
    }
}

/// A CUDA mapping: the step's (ROWS, WARPS) or the chunk's ROWS.
#[derive(Clone, Copy, Debug)]
enum Mapping {
    Step(u64, u64),
    Chunk(u64),
}

impl Case {
    fn cuda_step(
        &self,
        device: &Device,
        activation: Element,
        (rows, warps): (u64, u64),
    ) -> seismic::NativeKernel<gated_delta_step::Entry> {
        gated_delta_step::native_for_device_with(
            device,
            gated_delta_step::Elements {
                RN: Element::f32(),
                A: activation,
            },
            &NativeSpecialization::new()
                .with_static("NK", self.geometry.key_heads as u64)
                .with_static("NV", self.geometry.value_heads as u64)
                .with_static("W", self.geometry.width as u64)
                .with_static("C", self.geometry.convolution as u64)
                .with_launch_param(0, "ROWS", rows)
                .with_launch_param(0, "WARPS", warps),
        )
        .unwrap()
    }

    fn cuda_chunk(
        &self,
        device: &Device,
        activation: Element,
        rows: u64,
    ) -> seismic::NativeKernel<gated_delta_chunk::Entry> {
        gated_delta_chunk::native_for_device_with(
            device,
            gated_delta_chunk::Elements {
                RN: Element::f32(),
                A: activation,
            },
            &NativeSpecialization::new()
                .with_static("NK", self.geometry.key_heads as u64)
                .with_static("NV", self.geometry.value_heads as u64)
                .with_static("W", self.geometry.width as u64)
                .with_static("C", self.geometry.convolution as u64)
                .with_launch_param(1, "ROWS", rows),
        )
        .unwrap()
    }

    fn cuda(&self, device: &Device, activation: Element, mapping: Mapping) -> Outcome {
        self.cuda_with_reused_slab(device, activation, mapping, false)
    }

    fn cuda_with_reused_slab(
        &self,
        device: &Device,
        activation: Element,
        mapping: Mapping,
        reused: bool,
    ) -> Outcome {
        let mut t = self.tensors_with_reused_slab(device, activation, reused);
        let gated = match mapping {
            Mapping::Step(rows, warps) => {
                self.cuda_step(device, activation, (rows, warps))
                    .call(t.step_args(self))
                    .unwrap()
                    .value
            }
            Mapping::Chunk(rows) => {
                self.cuda_chunk(device, activation, rows)
                    .call(t.chunk_args(self))
                    .unwrap()
                    .value
            }
        };
        Outcome {
            gated: read(&gated),
            window: read(&t.window),
            delta: read(&t.delta),
            tape: read(&t.tape),
        }
    }
}

const STEPS: [(u64, u64); 4] = [(2, 4), (4, 4), (2, 8), (4, 8)];
#[test]
fn cuda_step_and_chunk_match_the_portable_body() {
    let Some(device) = cuda() else { return };
    for (label, case) in small_cases() {
        let oracle = case.oracle();
        for mapping in STEPS {
            let step = case.cuda(&device, Element::f32(), Mapping::Step(mapping.0, mapping.1));
            check(
                &format!("{label}: cuda step {mapping:?}"),
                &case,
                &step,
                &oracle,
                (2e-5, 2e-6),
            );
        }
        // The chunked form runs its state products on f16 tensor-core
        // operands (2^-11 relative rounding) with F32 accumulation, so it is
        // held to f16-operand tolerances.
        for rows in CHUNK_ROWS
            .into_iter()
            .filter(|rows| *rows <= case.geometry.width as u64)
        {
            let chunked = case.cuda(&device, Element::f32(), Mapping::Chunk(rows));
            check(
                &format!("{label}: cuda chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                (1e-2, 1e-3),
            );
        }
    }
}

#[test]
fn cuda_mapping_never_changes_bits_and_stop_equals_a_shorter_run() {
    let Some(device) = cuda() else { return };
    let slot = |rows, stop| slot(rows, stop, 1, 2);
    let full = Case::new(SMALL, 6, vec![slot(6, 3)], false, 11);
    let reference = full.cuda(&device, Element::f32(), Mapping::Step(2, 4));
    for mapping in STEPS {
        let outcome = full.cuda(&device, Element::f32(), Mapping::Step(mapping.0, mapping.1));
        assert!(
            reference
                .gated
                .iter()
                .zip(&outcome.gated)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && reference
                    .delta
                    .iter()
                    .zip(&outcome.delta)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "step {mapping:?} changed result bits"
        );
    }
    // The chunk's ROWS never changes bits either (W 128 admits every ROWS).
    let wide = Geometry {
        key_heads: 2,
        value_heads: 4,
        width: 128,
        convolution: 4,
        banks: 3,
        tape: 4,
    };
    let long = Case::new(wide, 100, vec![slot(100, 70)], true, 12);
    let first = long.cuda(&device, Element::f32(), Mapping::Chunk(CHUNK_ROWS[0]));
    for rows in &CHUNK_ROWS[1..] {
        let other = long.cuda(&device, Element::f32(), Mapping::Chunk(*rows));
        assert!(
            first
                .gated
                .iter()
                .zip(&other.gated)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && first
                    .delta
                    .iter()
                    .zip(&other.delta)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "chunk ROWS {rows} changed result bits"
        );
    }
    let mut prefix = Case::new(SMALL, 6, vec![slot(6, 3)], false, 11);
    prefix.rows = 3;
    prefix.slots = vec![slot(3, 3)];
    prefix.projection.truncate(3 * SMALL.projection_width());
    let short = prefix.cuda(&device, Element::f32(), Mapping::Step(2, 4));
    assert!(
        short
            .delta
            .iter()
            .zip(&reference.delta)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row state differs from the state of a shorter run"
    );
    assert!(
        short
            .window
            .iter()
            .zip(&reference.window)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row window differs from the window of a shorter run"
    );
}

/// MTP verify: a slot of at most 16 rows gets the step's bits from the chunk
/// entry too, whatever its peers (here a 40-row slot on the chunked path), so
/// a request's verify rows never depend on the class.
#[test]
fn cuda_chunk_short_slots_get_the_step_bits() {
    let Some(device) = cuda() else { return };
    let geometry = Geometry {
        banks: 11,
        ..QWEN_4B
    };
    let slots = vec![
        slot(4, 1, 1, 6),
        slot(16, 9, 2, 7),
        slot(40, 40, 3, 8),
        slot(1, 1, 4, 9),
        slot(7, 0, 5, 10),
    ];
    let case = Case::new(geometry, 70, slots, false, 31).with_bf16_activations();
    let step = case.cuda(&device, Element::bf16(), Mapping::Step(2, 4));
    let host = case.host();
    let g = case.geometry;
    let row_elements = g.value_heads * g.width;
    for rows in CHUNK_ROWS {
        let chunk = case.cuda(&device, Element::bf16(), Mapping::Chunk(rows));
        let mut first = 0;
        for s in &case.slots {
            let range = first * row_elements..(first + s.rows) * row_elements;
            first += s.rows;
            if s.rows > 16 {
                continue;
            }
            let bank = s.following * g.delta_bank()..(s.following + 1) * g.delta_bank();
            assert!(
                step.gated[range.clone()]
                    .iter()
                    .zip(&chunk.gated[range])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                    && step.delta[bank.clone()]
                        .iter()
                        .zip(&chunk.delta[bank])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "chunk ROWS {rows}: a {}-row slot differs from the step",
                s.rows
            );
        }
        check(
            &format!("4B verify mix: cuda chunk ROWS {rows}"),
            &case,
            &chunk,
            &host,
            (1.5e-2, 3e-3),
        );
    }
}

#[test]
#[ignore = "focused CUDA tuning; run explicitly on an idle device"]
fn cuda_recurrent_scoped_tuning_completes() {
    let Some(device) = cuda() else { return };
    let geometry = Geometry {
        banks: 3,
        ..QWEN_4B
    };
    let case = Case::new(geometry, 32, vec![slot(32, 32, 1, 2)], false, 41).with_bf16_activations();
    let mut tensors = case.tensors(&device, Element::bf16());
    let mut window = tensors.window.clone();
    let mut delta = tensors.delta.clone();
    let mut tape = tensors.tape.clone();
    let initial_window = window.read_to_host().unwrap();
    let initial_delta = delta.read_to_host().unwrap();
    let initial_tape = tape.read_to_host().unwrap();
    let statics = NativeSpecialization::new()
        .with_static("NK", geometry.key_heads as u64)
        .with_static("NV", geometry.value_heads as u64)
        .with_static("W", geometry.width as u64)
        .with_static("C", geometry.convolution as u64);
    let search = || {
        seismic::Strategy::Search(seismic::SearchPlan {
            allowance: std::time::Duration::from_secs(600),
            admission: std::time::Duration::from_secs(600),
            required: std::time::Duration::from_secs(600),
            settings: seismic::SearchSettings {
                improvement: 0.01,
                restarts: 2,
                confirmed: 3,
                default_margin: 0.02,
                samples: 3,
                confirmation_samples: 5,
            },
            min_sample_seconds: 0.0002,
            start: Vec::new(),
        })
    };
    let points = vec![seismic::TuningPoint {
        label: "rows32".into(),
        weight: 1.0,
        class: None,
        cost: 1.0,
        required: false,
        rotation: vec![tensors.chunk_args(&case)],
        initialize: Some(Box::new(move || {
            window.write_from_host(&initial_window)?;
            delta.write_from_host(&initial_delta)?;
            tape.write_from_host(&initial_tape)
        })),
        written: Default::default(),
    }];
    let result = gated_delta_chunk::native_tune_with(
        &device,
        gated_delta_chunk::Elements {
            RN: Element::f32(),
            A: Element::bf16(),
        },
        &statics,
        points,
        seismic::PrecisionPolicy::bounded(seismic::precision::Tolerance {
            absolute: seismic::precision::Limit::new(0.01).unwrap(),
            relative: seismic::precision::Limit::new(0.05).unwrap(),
            relative_floor: seismic::precision::Limit::ZERO,
            ulps: None,
        }),
        search(),
        seismic::TuningReference::Portable,
    )
    .unwrap();
    assert!(matches!(
        result.method,
        seismic::TuningMethod::Factored {
            groups: 1,
            candidates: 3,
            complete: true,
            ..
        }
    ));
    assert_eq!(result.rejections().count(), 0);
    println!(
        "chunk CUDA choice {:?}, time {:?}",
        result.overall.launches, result.time
    );
}

#[test]
fn cuda_tape_cases_match_the_portable_body() {
    let Some(device) = cuda() else { return };
    for (label, case) in tape_cases() {
        let oracle = case.oracle();
        for mapping in STEPS {
            let step = case.cuda(&device, Element::f32(), Mapping::Step(mapping.0, mapping.1));
            check(
                &format!("{label}: cuda step {mapping:?}"),
                &case,
                &step,
                &oracle,
                (2e-5, 2e-6),
            );
        }
        for rows in CHUNK_ROWS
            .into_iter()
            .filter(|rows| *rows <= case.geometry.width as u64)
        {
            let chunked = case.cuda(&device, Element::f32(), Mapping::Chunk(rows));
            check(
                &format!("{label}: cuda chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                (1e-2, 1e-3),
            );
        }
    }
}

#[test]
fn cuda_tape_versions_equal_runs_that_stopped_there() {
    let Some(device) = cuda() else { return };
    for geometry in [SMALL, QWEN_4B] {
        tape_versions_equal_stopped_runs("cuda step", geometry, &|case| {
            case.cuda(&device, Element::bf16(), Mapping::Step(2, 4))
        });
        tape_versions_equal_stopped_runs("cuda chunk", geometry, &|case| {
            case.cuda(&device, Element::bf16(), Mapping::Chunk(32))
        });
    }
}

const QWEN_4B: Geometry = Geometry {
    key_heads: 16,
    value_heads: 32,
    width: 128,
    convolution: 4,
    banks: 5,
    tape: 0,
};

#[test]
fn cuda_real_4b_geometry_step_and_chunk_agree_with_the_host_model() {
    let Some(device) = cuda() else { return };
    for (label, rows, slots) in [
        ("decode, one slot", 1, vec![slot(1, 1, 1, 3)]),
        (
            "verify, two slots",
            8,
            vec![slot(4, 2, 1, 3), slot(3, 3, 0, 4)],
        ),
        ("prefill 128", 128, vec![slot(128, 128, 1, 3)]),
        (
            "prefill 512, two slots",
            512,
            vec![slot(300, 211, 1, 3), slot(212, 212, 2, 4)],
        ),
    ] {
        let case = Case::new(QWEN_4B, rows, slots, false, 21).with_bf16_activations();
        let host = case.host();
        let step = case.cuda(&device, Element::bf16(), Mapping::Step(2, 4));
        check(
            &format!("4B {label}: cuda step"),
            &case,
            &step,
            &host,
            (1.5e-2, 3e-3),
        );
        if rows >= 16 {
            for rows in CHUNK_ROWS {
                let chunked = case.cuda(&device, Element::bf16(), Mapping::Chunk(rows));
                check(
                    &format!("4B {label}: cuda chunk ROWS {rows}"),
                    &case,
                    &chunked,
                    &host,
                    (1.5e-2, 3e-3),
                );
            }
        }
    }
}

/// The chunk's prepare block convolves q, k and the GROUP = NV / NK value
/// heads of its key head: every group size forms and runs. Groups of 1 to 4
/// at small width against the portable body, tiled and grouped head mappings
/// (a group of 1 maps heads identically either way).
#[test]
fn cuda_chunk_every_value_head_group_matches_the_portable_body() {
    let Some(device) = cuda() else { return };
    for (value_heads, grouped) in [(4, false), (8, true), (12, false), (12, true), (16, false)] {
        // Four key heads: tape rows (NV * W + NK * W + NV floats) stay
        // 16-byte multiples.
        let geometry = Geometry {
            key_heads: 4,
            value_heads,
            ..SMALL
        };
        // Two pieces (16 + 1 rows) before the stop row, three rows after it.
        let case = Case::new(geometry, 20, vec![slot(20, 17, 1, 3)], grouped, 5);
        let oracle = case.oracle();
        let chunked = case.cuda(&device, Element::f32(), Mapping::Chunk(32));
        check(
            &format!("NV {value_heads} grouped {grouped}: cuda chunk"),
            &case,
            &chunked,
            &oracle,
            (1e-2, 1e-3),
        );
    }
}

/// The real Qwen3.5 key-head count against the host model at every catalog
/// value-head count: 16 (0.8B, 2B), 32 (4B, 9B), 48 (27B) and 64 (122B).
#[test]
fn cuda_chunk_every_catalog_value_head_count_matches_the_host_model() {
    let Some(device) = cuda() else { return };
    for value_heads in [16, 32, 48, 64] {
        let geometry = Geometry {
            value_heads,
            ..QWEN_4B
        };
        let case =
            Case::new(geometry, 128, vec![slot(128, 100, 1, 3)], false, 23).with_bf16_activations();
        let host = case.host();
        for rows in CHUNK_ROWS {
            let chunked = case.cuda(&device, Element::bf16(), Mapping::Chunk(rows));
            check(
                &format!("NK 16 NV {value_heads} prefill 128: cuda chunk ROWS {rows}"),
                &case,
                &chunked,
                &host,
                (1.5e-2, 3e-3),
            );
        }
    }
}

/// Device time per call at the 4B geometry (one layer), each slot in its own
/// bank pair: decode, MTP verify (2-16 rows per slot, alone and 8 batched)
/// and prefill, the step up to 16 rows per slot and the chunk from 2 rows. Calls
/// rotate over 16 argument sets (state arenas beyond the 24 MiB L2), so state
/// traffic comes from DRAM as across a model's layers.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn cuda_recurrent_timings() {
    let Some(device) = cuda() else { return };
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.002,
    };
    // Slots of `per_slot` rows each: decode (1 row), MTP verify (2-16 rows
    // per request, alone and 8 requests batched) and prefill.
    for (slots, per_slot) in [
        (1usize, 1usize),
        (8, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (8, 4),
        (8, 8),
        (1, 32),
        (1, 128),
        (1, 512),
    ] {
        let label = format!("{slots} x {per_slot} rows");
        let slots = (0..slots)
            .map(|s| slot(per_slot, per_slot, 1 + s, 1 + slots + s))
            .collect::<Vec<_>>();
        let banks = slots.iter().map(|s| s.following).max().unwrap() + 1;
        let geometry = Geometry { banks, ..QWEN_4B };
        let rows = slots.len() * per_slot;
        let case = Case::new(geometry, rows, slots, false, 5).with_bf16_activations();
        let mut rotation = (0..16)
            .map(|_| case.tensors(&device, Element::bf16()))
            .collect::<Vec<_>>();
        if per_slot <= 16 {
            for mapping in STEPS {
                let kernel = case.cuda_step(&device, Element::bf16(), mapping);
                let args = rotation.iter_mut().map(|t| t.step_args(&case)).collect();
                let measured = kernel.measure(args, &options).unwrap();
                println!("{label}: step {mapping:?} {:.1} us", measured.median * 1e6);
            }
        }
        if rows >= 2 {
            for mapping in CHUNK_ROWS {
                let kernel = case.cuda_chunk(&device, Element::bf16(), mapping);
                let args = rotation.iter_mut().map(|t| t.chunk_args(&case)).collect();
                let measured = kernel.measure(args, &options).unwrap();
                println!(
                    "{label}: chunk ROWS {mapping} {:.1} us",
                    measured.median * 1e6
                );
            }
        }
    }
}

/// Three calls each of the 4B step at 1 row (ROWS 4, WARPS 8) and the chunk at
/// 512 rows (ROWS 128), each on its own argument set, for a profiler
/// (`ncu --kernel-name <entry> --launch-skip 2 --launch-count 1`).
#[test]
#[ignore = "profiling; run explicitly under a profiler"]
fn cuda_recurrent_profile_launches() {
    let Some(device) = cuda() else { return };
    let slot = |rows| slot(rows, rows, 1, 2);
    let geometry = Geometry {
        banks: 3,
        ..QWEN_4B
    };
    let decode = Case::new(geometry, 1, vec![slot(1)], false, 5).with_bf16_activations();
    let step = decode.cuda_step(&device, Element::bf16(), (4, 8));
    for _ in 0..3 {
        let mut t = decode.tensors(&device, Element::bf16());
        step.call(t.step_args(&decode)).unwrap();
    }
    let prefill = Case::new(geometry, 512, vec![slot(512)], false, 5).with_bf16_activations();
    let chunk = prefill.cuda_chunk(&device, Element::bf16(), 128);
    for _ in 0..3 {
        let mut t = prefill.tensors(&device, Element::bf16());
        chunk.call(t.chunk_args(&prefill)).unwrap();
    }
}
