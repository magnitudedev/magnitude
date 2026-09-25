//! The Vulkan `gated_delta_step` / `gated_delta_chunk` against the same
//! cases, portable-body oracle and host model as the Metal tests (included
//! verbatim; their Metal tests skip without a Metal device). The step and the
//! chunk's row-sequential slots (at most 16 rows, zero stop) share their bits;
//! the chunk's scanned rows are held to the oracle and host model, and no
//! mapping changes their bits.

include!("recurrent_stages.rs");

fn vulkan() -> Option<Device> {
    DeviceCatalog::discover()
        .ok()
        .and_then(|catalog| catalog.open_backend(BackendName::Vulkan).ok())
}

/// The step's and the chunk's (ROWS, WARPS) mappings.
const MAPPINGS: [(u64, u64); 4] = [(2, 4), (4, 4), (2, 8), (4, 8)];

impl Case {
    fn vulkan(&self, device: &Device, activation: Element, chunk: bool, (rows, warps): (u64, u64)) -> Outcome {
        let specialization = self.specialization(device, rows).with_param("WARPS", warps);
        let mut t = self.tensors(device, activation);
        let mixed = if chunk {
            gated_delta_chunk::native_for_device_with(
                device,
                gated_delta_chunk::Elements { A: activation },
                &specialization,
            )
            .unwrap()
            .call(t.chunk_args(self))
            .unwrap()
            .value
        } else {
            gated_delta_step::native_for_device_with(
                device,
                gated_delta_step::Elements { A: activation },
                &specialization,
            )
            .unwrap()
            .call(t.step_args(self))
            .unwrap()
            .value
        };
        Outcome {
            mixed: read(&mixed),
            window: read(&t.window),
            delta: read(&t.delta),
            tape: read(&t.tape),
        }
    }

    /// Every slot advances row-sequentially in the chunk entry too.
    fn sequential_in_chunk(&self) -> bool {
        self.slots.iter().all(|slot| slot.rows <= 16 || slot.stop == 0)
    }
}

fn same_bits(a: &Outcome, b: &Outcome) -> bool {
    let equal = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
    equal(&a.mixed, &b.mixed) && equal(&a.delta, &b.delta) && equal(&a.window, &b.window)
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
        for mapping in MAPPINGS.into_iter().filter(|(rows, warps)| case.geometry.width as u64 % (rows * warps) == 0) {
            let step = case.vulkan(&device, Element::f32(), false, mapping);
            check(&format!("{label}: vulkan step {mapping:?}"), &case, &step, &oracle, (2e-5, 2e-6));
            let chunk = case.vulkan(&device, Element::f32(), true, mapping);
            check(&format!("{label}: vulkan chunk {mapping:?}"), &case, &chunk, &oracle, (5e-4, 2e-5));
            if case.sequential_in_chunk() {
                assert!(same_bits(&step, &chunk), "{label}: vulkan chunk {mapping:?} differs from the step");
            }
            match &reference {
                Some(reference) => {
                    assert!(same_bits(reference, &chunk), "{label}: vulkan chunk {mapping:?} changed result bits")
                }
                None => reference = Some(chunk),
            }
        }
    }
}

#[test]
fn vulkan_mapping_never_changes_bits_and_stop_equals_a_shorter_run() {
    let Some(device) = vulkan() else { return };
    let wide = Geometry { key_heads: 2, value_heads: 4, width: 128, convolution: 4, banks: 3, tape: 0 };
    let slot = |rows, stop| slot(rows, stop, 1, 2);
    let full = Case::new(wide, 6, vec![slot(6, 3)], false, 11);
    let reference = full.vulkan(&device, Element::f32(), false, MAPPINGS[0]);
    for mapping in MAPPINGS {
        for chunk in [false, true] {
            let outcome = full.vulkan(&device, Element::f32(), chunk, mapping);
            assert!(same_bits(&reference, &outcome), "{} {mapping:?} changed result bits", if chunk { "chunk" } else { "step" });
        }
    }
    let mut prefix = Case::new(wide, 6, vec![slot(6, 3)], false, 11);
    prefix.rows = 3;
    prefix.slots = vec![slot(3, 3)];
    prefix.projection.truncate(3 * wide.projection_width());
    let short = prefix.vulkan(&device, Element::f32(), false, MAPPINGS[0]);
    assert!(
        short.delta.iter().zip(&reference.delta).all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row state differs from the state of a shorter run"
    );
    assert!(
        short.window.iter().zip(&reference.window).all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row window differs from the window of a shorter run"
    );
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
fn vulkan_real_4b_geometry_step_and_chunk_agree_with_the_host_model() {
    let Some(device) = vulkan() else { return };
    for (label, rows, slots) in [
        ("decode, one slot", 1, vec![slot(1, 1, 1, 3)]),
        ("verify, two slots", 8, vec![slot(4, 2, 1, 3), slot(3, 3, 0, 4)]),
        ("prefill 128", 128, vec![slot(128, 128, 1, 3)]),
        ("prefill 512, two slots", 512, vec![slot(300, 211, 1, 3), slot(212, 212, 2, 4)]),
    ] {
        let case = Case::new(QWEN_4B, rows, slots, false, 21).with_bf16_activations();
        let host = case.host();
        let step = case.vulkan(&device, Element::bf16(), false, MAPPINGS[0]);
        check(&format!("4B {label}: vulkan step"), &case, &step, &host, (1.5e-2, 3e-3));
        let mut reference: Option<Outcome> = None;
        for mapping in MAPPINGS {
            let chunk = case.vulkan(&device, Element::bf16(), true, mapping);
            if case.sequential_in_chunk() {
                assert!(same_bits(&step, &chunk), "4B {label}: vulkan chunk {mapping:?} differs from the step");
                continue;
            }
            check(&format!("4B {label}: vulkan chunk {mapping:?}"), &case, &chunk, &host, (1.5e-2, 3e-3));
            let (max, rms) = errors(&chunk.mixed, &step.mixed);
            println!("4B {label}: vulkan chunk {mapping:?} vs step: max {max:.3e} rms {rms:.3e}");
            // Both are within one BF16 ulp of the host model per element, so
            // they may differ by two ulps of the largest output.
            assert!(max <= 4e-2 && rms <= 3e-3);
            match &reference {
                Some(reference) => {
                    assert!(same_bits(reference, &chunk), "4B {label}: vulkan chunk {mapping:?} changed result bits")
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
    let geometry = Geometry { key_heads: 16, value_heads: 32, width: 128, convolution: 4, banks: 11, tape: 0 };
    let slots = vec![slot(4, 1, 1, 6), slot(16, 9, 2, 7), slot(40, 40, 3, 8), slot(1, 1, 4, 9), slot(7, 0, 5, 10)];
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
            if s.rows > 16 {
                continue;
            }
            let bank = s.following * geometry.delta_bank()..(s.following + 1) * geometry.delta_bank();
            assert!(
                step.mixed[range.clone()].iter().zip(&chunk.mixed[range]).all(|(a, b)| a.to_bits() == b.to_bits())
                    && step.delta[bank.clone()].iter().zip(&chunk.delta[bank]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "chunk {mapping:?}: a {}-row slot differs from the step",
                s.rows
            );
        }
        check(&format!("4B verify mix: vulkan chunk {mapping:?}"), &case, &chunk, &host, (1.5e-2, 3e-3));
    }
}

#[test]
fn vulkan_tape_cases_match_the_portable_body() {
    let Some(device) = vulkan() else { return };
    for (label, case) in tape_cases() {
        let oracle = case.oracle();
        for mapping in MAPPINGS.into_iter().filter(|(rows, warps)| case.geometry.width as u64 % (rows * warps) == 0) {
            let step = case.vulkan(&device, Element::f32(), false, mapping);
            check(&format!("{label}: vulkan step {mapping:?}"), &case, &step, &oracle, (2e-5, 2e-6));
            let chunk = case.vulkan(&device, Element::f32(), true, mapping);
            check(&format!("{label}: vulkan chunk {mapping:?}"), &case, &chunk, &oracle, (5e-4, 2e-5));
            if case.sequential_in_chunk() {
                assert!(
                    step.tape.iter().zip(&chunk.tape).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label}: vulkan chunk {mapping:?} records a different tape"
                );
            }
        }
    }
}

#[test]
fn vulkan_tape_versions_equal_runs_that_stopped_there() {
    let Some(device) = vulkan() else { return };
    for geometry in [SMALL, QWEN_4B] {
        tape_versions_equal_stopped_runs("vulkan step", geometry, &|case| {
            case.vulkan(&device, Element::bf16(), false, MAPPINGS[0])
        });
        tape_versions_equal_stopped_runs("vulkan chunk", geometry, &|case| {
            case.vulkan(&device, Element::bf16(), true, MAPPINGS[3])
        });
    }
}
