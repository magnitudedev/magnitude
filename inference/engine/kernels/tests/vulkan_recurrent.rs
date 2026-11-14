//! The Vulkan `gated_delta_step` / `gated_delta_chunk` against the same
//! cases, portable-body oracle and host model as the Metal tests (included
//! verbatim; their Metal tests skip without a Metal device). The step and the
//! chunk's row-sequential slots (at most 16 rows, zero stop) share their state
//! bits, and their gated bits beyond 8 rows (to 8 rows the step gates in the
//! GEMV staging's lane order); the chunk's scanned rows are held to the oracle
//! and host model, and no mapping changes their bits.

include!("recurrent_stages.rs");

/// Vulkan's step and chunk entries, for their bounds: the chunk is the
/// step's row-sequential arithmetic with the state on lanes, so the two
/// share a form.
const STEP: Native = Native::Step(BackendName::Vulkan);
const CHUNK: Native = Native::Chunk(BackendName::Vulkan);

fn vulkan() -> Option<Device> {
    seismic_native_tests::devices(&[BackendName::Vulkan]).into_iter().next()
}

#[test]
fn vulkan_slab_bank_crossing_matches_host_model() {
    let Some(device) = vulkan() else { return };
    let (_, case) = tape_cases().remove(0);
    let host = case.host();
    for reused in [false, true] {
        let step =
            case.vulkan_with_reused_slab(&device, Element::f32(), false, MAPPINGS[0], reused);
        check("vulkan slab tape step", &case, &step, &host, against_host(STEP));
        let chunk =
            case.vulkan_with_reused_slab(&device, Element::f32(), true, MAPPINGS[3], reused);
        check("vulkan slab tape chunk", &case, &chunk, &host, against_host(CHUNK));
    }
}

/// The step's and the chunk's (ROWS, WARPS) mappings.
const MAPPINGS: [(u64, u64); 4] = [(2, 4), (4, 4), (2, 8), (4, 8)];

impl Case {
    fn vulkan(
        &self,
        device: &Device,
        activation: Element,
        chunk: bool,
        (rows, warps): (u64, u64),
    ) -> Outcome {
        self.vulkan_with_reused_slab(device, activation, chunk, (rows, warps), false)
    }

    fn vulkan_with_reused_slab(
        &self,
        device: &Device,
        activation: Element,
        chunk: bool,
        (rows, warps): (u64, u64),
        reused: bool,
    ) -> Outcome {
        let specialization = self.specialization(device, rows).with_param("WARPS", warps);
        let mut t = self.tensors_with_reused_slab(device, activation, reused);
        let gated = t.run(device, self, activation, chunk, &specialization);
        Outcome {
            gated: read(&gated),
            window: read(&t.window),
            delta: read(&t.delta),
            tape: read(&t.tape),
        }
    }

    /// Every slot advances row-sequentially in the chunk entry too.
    fn sequential_in_chunk(&self) -> bool {
        self.slots
            .iter()
            .all(|slot| slot.rows <= SEQUENTIAL_ROWS || slot.stop == 0)
    }
}

/// Whether `a` and `b` carry the same state and window bits and, with
/// `gated`, the same gated bits. The step gates a class of at most 8 rows in
/// the GEMV staging's lane order, the chunk in its stage order, so their
/// gated bits are compared only beyond 8 rows.
fn same_bits(a: &Outcome, b: &Outcome, gated: bool) -> bool {
    let equal = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
    (!gated || equal(&a.gated, &b.gated))
        && equal(&a.delta, &b.delta)
        && equal(&a.window, &b.window)
}

#[test]
fn vulkan_step_and_chunk_match_the_portable_body() {
    let Some(device) = vulkan() else { return };
    for (label, case) in small_cases() {
        if case.geometry.width % 32 != 0 {
            continue;
        }
        let oracle = case.oracle();
        let mut reference: Option<Outcome> = None;
        // One mapping per distinct kernel path: ROWS and WARPS never change result
        // bits (`vulkan_mapping_never_changes_bits_and_stop_equals_a_shorter_run`).
        for mapping in MAPPINGS
            .into_iter()
            .filter(|(rows, warps)| case.geometry.width as u64 % (rows * warps) == 0)
            .take(1)
        {
            let step = case.vulkan(&device, Element::f32(), false, mapping);
            check(
                &format!("{label}: vulkan step {mapping:?}"),
                &case,
                &step,
                &oracle,
                between(STEP, ORACLE),
            );
            let chunk = case.vulkan(&device, Element::f32(), true, mapping);
            check(
                &format!("{label}: vulkan chunk {mapping:?}"),
                &case,
                &chunk,
                &oracle,
                between(CHUNK, ORACLE),
            );
            if case.sequential_in_chunk() {
                assert!(
                    same_bits(&step, &chunk, case.rows > 8),
                    "{label}: vulkan chunk {mapping:?} differs from the step"
                );
            }
            match &reference {
                Some(reference) => {
                    assert!(
                        same_bits(reference, &chunk, true),
                        "{label}: vulkan chunk {mapping:?} changed result bits"
                    )
                }
                None => reference = Some(chunk),
            }
        }
    }
}

#[test]
fn vulkan_mapping_never_changes_bits_and_stop_equals_a_shorter_run() {
    let Some(device) = vulkan() else { return };
    let wide = Geometry {
        key_heads: 2,
        value_heads: 4,
        width: 128,
        decay_width: 1,
        convolution: 4,
        banks: 3,
        tape: 0,
    };
    let slot = |rows, stop| slot(rows, stop, 1, 2);
    let full = Case::new(wide, 6, vec![slot(6, 3)], false, 11);
    let reference = full.vulkan(&device, Element::f32(), false, MAPPINGS[0]);
    for mapping in MAPPINGS {
        for chunk in [false, true] {
            let outcome = full.vulkan(&device, Element::f32(), chunk, mapping);
            assert!(
                same_bits(&reference, &outcome, !chunk),
                "{} {mapping:?} changed result bits",
                if chunk { "chunk" } else { "step" }
            );
        }
    }
    let mut prefix = Case::new(wide, 6, vec![slot(6, 3)], false, 11);
    prefix.rows = 3;
    prefix.slots = vec![slot(3, 3)];
    prefix.projection.truncate(3 * wide.projection_width());
    let short = prefix.vulkan(&device, Element::f32(), false, MAPPINGS[0]);
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
fn vulkan_real_4b_geometry_step_and_chunk_agree_with_the_host_model() {
    let Some(device) = vulkan() else { return };
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
        let step = case.vulkan(&device, Element::bf16(), false, MAPPINGS[0]);
        check(
            &format!("4B {label}: vulkan step"),
            &case,
            &step,
            &host,
            against_host(STEP),
        );
        let mut reference: Option<Outcome> = None;
        for mapping in MAPPINGS {
            let chunk = case.vulkan(&device, Element::bf16(), true, mapping);
            if case.sequential_in_chunk() {
                assert!(
                    same_bits(&step, &chunk, case.rows > 8),
                    "4B {label}: vulkan chunk {mapping:?} differs from the step"
                );
                continue;
            }
            check(
                &format!("4B {label}: vulkan chunk {mapping:?}"),
                &case,
                &chunk,
                &host,
                against_host(CHUNK),
            );
            check(
                &format!("4B {label}: vulkan chunk {mapping:?} vs step"),
                &case,
                &chunk,
                &step,
                between(CHUNK, STEP),
            );
            match &reference {
                Some(reference) => {
                    assert!(
                        same_bits(reference, &chunk, true),
                        "4B {label}: vulkan chunk {mapping:?} changed result bits"
                    )
                }
                None => reference = Some(chunk),
            }
        }
    }
}

/// MTP verify: a slot of at most 16 rows gets the step's bits from the chunk
/// entry too, whatever its peers (here a 40-row slot on the scanned path).
#[test]
fn vulkan_chunk_short_slots_get_the_step_bits() {
    let Some(device) = vulkan() else { return };
    let geometry = Geometry {
        key_heads: 16,
        value_heads: 32,
        width: 128,
        decay_width: 1,
        convolution: 4,
        banks: 11,
        tape: 0,
    };
    let slots = vec![
        slot(4, 1, 1, 6),
        slot(SEQUENTIAL_ROWS, 9, 2, 7),
        slot(40, 40, 3, 8),
        slot(1, 1, 4, 9),
        slot(7, 0, 5, 10),
    ];
    let case = Case::new(geometry, 70, slots, false, 31).with_bf16_activations();
    let step = case.vulkan(&device, Element::bf16(), false, MAPPINGS[0]);
    let host = case.host();
    let row_elements = geometry.value_heads * geometry.width;
    for mapping in MAPPINGS {
        let chunk = case.vulkan(&device, Element::bf16(), true, mapping);
        let mut first = 0;
        for s in &case.slots {
            let range = first * row_elements..(first + s.rows) * row_elements;
            first += s.rows;
            if s.rows > SEQUENTIAL_ROWS {
                continue;
            }
            let bank =
                s.following * geometry.delta_bank()..(s.following + 1) * geometry.delta_bank();
            assert!(
                step.gated[range.clone()]
                    .iter()
                    .zip(&chunk.gated[range])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                    && step.delta[bank.clone()]
                        .iter()
                        .zip(&chunk.delta[bank])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "chunk {mapping:?}: a {}-row slot differs from the step",
                s.rows
            );
        }
        check(
            &format!("4B verify mix: vulkan chunk {mapping:?}"),
            &case,
            &chunk,
            &host,
            against_host(CHUNK),
        );
    }
}

#[test]
fn vulkan_tape_cases_match_the_portable_body() {
    let Some(device) = vulkan() else { return };
    for (label, case) in tape_cases() {
        let oracle = case.oracle();
        // One mapping per distinct kernel path: ROWS and WARPS never change result
        // bits (`vulkan_mapping_never_changes_bits_and_stop_equals_a_shorter_run`).
        for mapping in MAPPINGS
            .into_iter()
            .filter(|(rows, warps)| case.geometry.width as u64 % (rows * warps) == 0)
            .take(1)
        {
            let step = case.vulkan(&device, Element::f32(), false, mapping);
            check(
                &format!("{label}: vulkan step {mapping:?}"),
                &case,
                &step,
                &oracle,
                between(STEP, ORACLE),
            );
            let chunk = case.vulkan(&device, Element::f32(), true, mapping);
            check(
                &format!("{label}: vulkan chunk {mapping:?}"),
                &case,
                &chunk,
                &oracle,
                between(CHUNK, ORACLE),
            );
            if case.sequential_in_chunk() {
                assert!(
                    step.tape
                        .iter()
                        .zip(&chunk.tape)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label}: vulkan chunk {mapping:?} records a different tape"
                );
            }
        }
    }
}

#[test]
fn vulkan_tape_versions_equal_runs_that_stopped_there() {
    let Some(device) = vulkan() else { return };
    let per_head = |case: Case| case.with_bf16_activations();
    for geometry in [SMALL, QWEN_4B] {
        tape_versions_equal_stopped_runs("vulkan step", geometry, &per_head, STEP, &|case| {
            case.vulkan(&device, Element::bf16(), false, MAPPINGS[0])
        });
        tape_versions_equal_stopped_runs("vulkan chunk", geometry, &per_head, CHUNK, &|case| {
            case.vulkan(&device, Element::bf16(), true, MAPPINGS[3])
        });
    }
}

/// The Vulkan entries for the per-channel checks: the step and the chunk (a
/// row-sequential scan with the step's arithmetic per row) are sequential F32
/// forms, every (ROWS, WARPS) mapping the width admits.
fn vulkan_per_channel_entries(device: &Device) -> PerChannelEntries<'_> {
    PerChannelEntries {
        backend: "vulkan",
        kind: BackendName::Vulkan,
        step: Box::new(move |case: &Case, activation| case.vulkan(device, activation, false, MAPPINGS[0])),
        chunks: MAPPINGS
            .into_iter()
            .take(1)
            .map(|mapping| {
                let run: Box<dyn Fn(&Case, Element) -> Outcome + '_> =
                    Box::new(move |case: &Case, activation| case.vulkan(device, activation, true, mapping));
                (format!("vulkan chunk {mapping:?}"), run)
            })
            .collect(),
    }
}

#[test]
fn vulkan_per_channel_decay_matches_the_portable_body() {
    let Some(device) = vulkan() else { return };
    per_channel_decay_matches_the_portable_body_on(&vulkan_per_channel_entries(&device));
}

#[test]
fn vulkan_kda_geometry_step_and_chunk_agree_with_the_host_model() {
    let Some(device) = vulkan() else { return };
    kda_geometry_step_and_chunk_agree_with_the_host_model_on(&vulkan_per_channel_entries(&device));
}

#[test]
fn vulkan_kda_prefill_then_decode_equals_decode_alone() {
    let Some(device) = vulkan() else { return };
    kda_prefill_then_decode_equals_decode_alone_on(&vulkan_per_channel_entries(&device));
}
