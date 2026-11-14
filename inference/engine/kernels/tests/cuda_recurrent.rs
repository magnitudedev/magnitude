//! The CUDA `gated_delta_step` / `gated_delta_chunk` against the same
//! cases, portable-body oracle and host model as the Metal tests (included
//! verbatim; their Metal tests skip without a Metal device).

include!("recurrent_stages.rs");

/// CUDA's step and chunk entries, for their bounds.
const STEP: Native = Native::Step(BackendName::Cuda);
const CHUNK: Native = Native::Chunk(BackendName::Cuda);

fn cuda() -> Option<Device> {
    seismic_native_tests::devices(&[BackendName::Cuda]).into_iter().next()
}

#[test]
fn cuda_slab_bank_crossing_matches_host_model() {
    let Some(device) = cuda() else { return };
    let (_, case) = tape_cases().remove(0);
    let host = case.host();
    for reused in [false, true] {
        let step = case.cuda_with_reused_slab(&device, Element::f32(), Mapping::Step(4, 8), reused);
        check("cuda slab tape step", &case, &step, &host, against_host(STEP));
        let chunk = case.cuda_with_reused_slab(&device, Element::f32(), Mapping::Chunk(32), reused);
        check("cuda slab tape chunk", &case, &chunk, &host, against_host(CHUNK));
    }
}

/// A CUDA mapping: the step's (ROWS, WARPS) or the chunk's ROWS.
#[derive(Clone, Copy, Debug)]
enum Mapping {
    Step(u64, u64),
    Chunk(u64),
}

impl Case {
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
        let statics = NativeSpecialization::new()
            .with_static("NK", self.geometry.key_heads as u64)
            .with_static("NV", self.geometry.value_heads as u64)
            .with_static("W", self.geometry.width as u64)
            .with_static("C", self.geometry.convolution as u64);
        let (chunk, specialization) = match mapping {
            Mapping::Step(rows, warps) => (
                false,
                statics
                    .with_launch_param(0, "ROWS", rows)
                    .with_launch_param(0, "WARPS", warps),
            ),
            Mapping::Chunk(rows) => (true, statics.with_launch_param(1, "ROWS", rows)),
        };
        let gated = t.run(device, self, activation, chunk, &specialization);
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
                between(STEP, ORACLE),
            );
        }
        // The chunked form runs its state products on f16 tensor-core
        // operands with F32 accumulation: its bound carries their roundings
        // (`recurrent_bound`).
        // One mapping per distinct kernel path: ROWS never changes result bits.
        for rows in CHUNK_ROWS
            .into_iter()
            .filter(|rows| *rows <= case.geometry.width as u64)
            .take(1)
        {
            let chunked = case.cuda(&device, Element::f32(), Mapping::Chunk(rows));
            check(
                &format!("{label}: cuda chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                between(CHUNK, ORACLE),
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
        decay_width: 1,
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
        slot(SEQUENTIAL_ROWS, 9, 2, 7),
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
            if s.rows > SEQUENTIAL_ROWS {
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
            against_host(CHUNK),
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

            settings: seismic::SearchSettings {
                error_rate: 0.05,
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
        workload: seismic::WorkloadRotation::All,
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
        // Eight rounding steps of the `bf16` activations: the engine's
        // standard α.
        seismic::NumericalRequirement::RoundingSteps {
            alpha: 8.,
            activation: Some(seismic::DType::BF16),
        },
        search(),
    )
    .unwrap();
    assert!(matches!(
        result.method,
        seismic::TuningMethod::Search {
            groups: 1,
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
                between(STEP, ORACLE),
            );
        }
        // One mapping per distinct kernel path: ROWS never changes result bits.
        for rows in CHUNK_ROWS
            .into_iter()
            .filter(|rows| *rows <= case.geometry.width as u64)
            .take(1)
        {
            let chunked = case.cuda(&device, Element::f32(), Mapping::Chunk(rows));
            check(
                &format!("{label}: cuda chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                between(CHUNK, ORACLE),
            );
        }
    }
}

#[test]
fn cuda_tape_versions_equal_runs_that_stopped_there() {
    let Some(device) = cuda() else { return };
    let per_head = |case: Case| case.with_bf16_activations();
    for geometry in [SMALL, QWEN_4B] {
        tape_versions_equal_stopped_runs("cuda step", geometry, &per_head, STEP, &|case| {
            case.cuda(&device, Element::bf16(), Mapping::Step(2, 4))
        });
        tape_versions_equal_stopped_runs("cuda chunk", geometry, &per_head, CHUNK, &|case| {
            case.cuda(&device, Element::bf16(), Mapping::Chunk(32))
        });
    }
}

const QWEN_4B: Geometry = Geometry {
    key_heads: 16,
    value_heads: 32,
    width: 128,
    decay_width: 1,
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
            against_host(STEP),
        );
        if rows >= 16 {
            for rows in CHUNK_ROWS {
                let chunked = case.cuda(&device, Element::bf16(), Mapping::Chunk(rows));
                check(
                    &format!("4B {label}: cuda chunk ROWS {rows}"),
                    &case,
                    &chunked,
                    &host,
                    against_host(CHUNK),
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
            between(CHUNK, ORACLE),
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
        for rows in CHUNK_ROWS.into_iter().take(1) {
            let chunked = case.cuda(&device, Element::bf16(), Mapping::Chunk(rows));
            check(
                &format!("NK 16 NV {value_heads} prefill 128: cuda chunk ROWS {rows}"),
                &case,
                &chunked,
                &host,
                against_host(CHUNK),
            );
        }
    }
}

/// The CUDA entries for the per-channel checks. The step is the sequential F32
/// form. The chunk runs its state products on f16 operands (keys and state
/// as high + low pairs), with the decays inside the operands (Q o exp G,
/// K o exp G, K o exp(G_last - G)) instead of F32 row factors after them;
/// `recurrent_bound` derives its bounds from those roundings.
fn cuda_per_channel_entries(device: &Device) -> PerChannelEntries<'_> {
    PerChannelEntries {
        backend: "cuda",
        kind: BackendName::Cuda,
        step: Box::new(move |case: &Case, activation| {
            case.cuda(device, activation, Mapping::Step(STEPS[0].0, STEPS[0].1))
        }),
        chunks: CHUNK_ROWS
            .into_iter()
            .take(1)
            .map(|rows| {
                let run: Box<dyn Fn(&Case, Element) -> Outcome + '_> =
                    Box::new(move |case: &Case, activation| {
                        case.cuda(device, activation, Mapping::Chunk(rows.min(case.geometry.width as u64)))
                    });
                (format!("cuda chunk ROWS {rows}"), run)
            })
            .collect(),
    }
}

#[test]
fn cuda_per_channel_decay_matches_the_portable_body() {
    let Some(device) = cuda() else { return };
    per_channel_decay_matches_the_portable_body_on(&cuda_per_channel_entries(&device));
}

#[test]
fn cuda_kda_geometry_step_and_chunk_agree_with_the_host_model() {
    let Some(device) = cuda() else { return };
    kda_geometry_step_and_chunk_agree_with_the_host_model_on(&cuda_per_channel_entries(&device));
}

#[test]
fn cuda_kda_prefill_then_decode_equals_decode_alone() {
    let Some(device) = cuda() else { return };
    kda_prefill_then_decode_equals_decode_alone_on(&cuda_per_channel_entries(&device));
}
