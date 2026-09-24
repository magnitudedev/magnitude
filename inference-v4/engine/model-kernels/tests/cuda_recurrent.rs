//! The CUDA `qwen_recurrent_step` / `qwen_recurrent_chunk` against the same
//! cases, portable-body oracle and host model as the Metal tests (included
//! verbatim; their Metal tests skip without a Metal device), plus timings.

include!("recurrent_stages.rs");

fn cuda() -> Option<Device> {
    DeviceCatalog::discover()
        .ok()
        .and_then(|catalog| catalog.open_backend(BackendName::Cuda).ok())
}

/// A CUDA mapping: the step's (ROWS, WARPS) or the chunk's ROWS.
#[derive(Clone, Copy, Debug)]
enum Mapping {
    Step(u64, u64),
    Chunk(u64),
}

impl Case {
    fn cuda_step(&self, device: &Device, activation: Element, (rows, warps): (u64, u64))
        -> seismic::NativeKernel<qwen_recurrent_step::Entry> {
        qwen_recurrent_step::native_for_device_with(
            device,
            qwen_recurrent_step::Elements { A: activation },
            &self.statics().with_param("ROWS", rows).with_param("WARPS", warps),
        )
        .unwrap()
    }

    fn cuda_chunk(&self, device: &Device, activation: Element, rows: u64)
        -> seismic::NativeKernel<qwen_recurrent_chunk::Entry> {
        qwen_recurrent_chunk::native_for_device_with(
            device,
            qwen_recurrent_chunk::Elements { A: activation },
            &self.statics().with_param("ROWS", rows),
        )
        .unwrap()
    }

    fn cuda(&self, device: &Device, activation: Element, mapping: Mapping) -> Outcome {
        let mut t = self.tensors(device, activation);
        let mixed = match mapping {
            Mapping::Step(rows, warps) => self
                .cuda_step(device, activation, (rows, warps))
                .call(t.step_args(self))
                .unwrap()
                .value,
            Mapping::Chunk(rows) => self
                .cuda_chunk(device, activation, rows)
                .call(t.chunk_args(self))
                .unwrap()
                .value,
        };
        Outcome {
            mixed: read(&mixed),
            window: read(&t.window),
            delta: read(&t.delta),
        }
    }
}

const STEPS: [(u64, u64); 4] = [(2, 4), (4, 4), (2, 8), (4, 8)];
/// ROWS mappings of the chunk (those up to the geometry's W apply).
const CHUNK_ROWS: [u64; 4] = [128, 64, 32, 16];

#[test]
fn cuda_step_and_chunk_match_the_portable_body() {
    let Some(device) = cuda() else { return };
    for (label, case) in small_cases() {
        let oracle = case.oracle();
        for mapping in STEPS {
            let step = case.cuda(&device, Element::f32(), Mapping::Step(mapping.0, mapping.1));
            check(&format!("{label}: cuda step {mapping:?}"), &case, &step, &oracle, (2e-5, 2e-6));
        }
        // The chunked form runs its state products on f16 tensor-core
        // operands (2^-11 relative rounding) with F32 accumulation, so it is
        // held to f16-operand tolerances.
        for rows in CHUNK_ROWS.into_iter().filter(|rows| *rows <= case.geometry.width as u64) {
            let chunked = case.cuda(&device, Element::f32(), Mapping::Chunk(rows));
            check(&format!("{label}: cuda chunk ROWS {rows}"), &case, &chunked, &oracle, (1e-2, 1e-3));
        }
    }
}

#[test]
fn cuda_mapping_never_changes_bits_and_stop_equals_a_shorter_run() {
    let Some(device) = cuda() else { return };
    let slot = |rows, stop| SlotCase { rows, stop, previous: 1, following: 2 };
    let full = Case::new(SMALL, 6, vec![slot(6, 3)], false, 11);
    let reference = full.cuda(&device, Element::f32(), Mapping::Step(2, 4));
    for mapping in STEPS {
        let outcome = full.cuda(&device, Element::f32(), Mapping::Step(mapping.0, mapping.1));
        assert!(
            reference.mixed.iter().zip(&outcome.mixed).all(|(a, b)| a.to_bits() == b.to_bits())
                && reference.delta.iter().zip(&outcome.delta).all(|(a, b)| a.to_bits() == b.to_bits()),
            "step {mapping:?} changed result bits"
        );
    }
    // The chunk's ROWS never changes bits either.
    let long = Case::new(SMALL, 100, vec![SlotCase { rows: 100, stop: 70, previous: 1, following: 2 }], true, 12);
    let first = long.cuda(&device, Element::f32(), Mapping::Chunk(32));
    let second = long.cuda(&device, Element::f32(), Mapping::Chunk(16));
    assert!(
        first.mixed.iter().zip(&second.mixed).all(|(a, b)| a.to_bits() == b.to_bits())
            && first.delta.iter().zip(&second.delta).all(|(a, b)| a.to_bits() == b.to_bits()),
        "chunk ROWS changed result bits"
    );
    let mut prefix = Case::new(SMALL, 6, vec![slot(6, 3)], false, 11);
    prefix.rows = 3;
    prefix.slots = vec![slot(3, 3)];
    prefix.projection.truncate(3 * SMALL.projection_width());
    let short = prefix.cuda(&device, Element::f32(), Mapping::Step(2, 4));
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
};

#[test]
fn cuda_real_4b_geometry_step_and_chunk_agree_with_the_host_model() {
    let Some(device) = cuda() else { return };
    let slot = |rows, stop, previous, following| SlotCase { rows, stop, previous, following };
    for (label, rows, slots) in [
        ("decode, one slot", 1, vec![slot(1, 1, 1, 3)]),
        ("verify, two slots", 8, vec![slot(4, 2, 1, 3), slot(3, 3, 0, 4)]),
        ("prefill 128", 128, vec![slot(128, 128, 1, 3)]),
        ("prefill 512, two slots", 512, vec![slot(300, 211, 1, 3), slot(212, 212, 2, 4)]),
    ] {
        let case = Case::new(QWEN_4B, rows, slots, false, 21).with_bf16_activations();
        let host = case.host();
        let step = case.cuda(&device, Element::bf16(), Mapping::Step(2, 4));
        check(&format!("4B {label}: cuda step"), &case, &step, &host, (1.5e-2, 3e-3));
        if rows >= 16 {
            for rows in CHUNK_ROWS {
                let chunked = case.cuda(&device, Element::bf16(), Mapping::Chunk(rows));
                check(&format!("4B {label}: cuda chunk ROWS {rows}"), &case, &chunked, &host, (1.5e-2, 3e-3));
            }
        }
    }
}

/// Device time per call at the 4B geometry (one layer): step at 1 and 8
/// rows, chunk at 32 to 512 rows with each launch's time, each slot in its
/// own bank pair. Calls rotate over 16 argument sets (state arenas beyond the
/// 24 MiB L2), so state traffic comes from DRAM as across a model's layers.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn cuda_recurrent_timings() {
    let Some(device) = cuda() else { return };
    let options = seismic::MeasureOptions { samples: 15, min_sample_seconds: 0.002 };
    let slot = |rows, previous, following| SlotCase { rows, stop: rows, previous, following };
    for (label, rows, slots) in [
        ("step 1 row", 1usize, vec![slot(1, 1, 2)]),
        ("step 8 rows (8 slots)", 8, (0..8).map(|s| slot(1, 1 + s, 9 + s)).collect::<Vec<_>>()),
        ("chunk 32 rows", 32, vec![slot(32, 1, 2)]),
        ("chunk 64 rows", 64, vec![slot(64, 1, 2)]),
        ("chunk 128 rows", 128, vec![slot(128, 1, 2)]),
        ("chunk 512 rows", 512, vec![slot(512, 1, 2)]),
    ] {
        let banks = slots.iter().map(|s| s.following).max().unwrap() + 1;
        let geometry = Geometry { banks, ..QWEN_4B };
        let case = Case::new(geometry, rows, slots, false, 5).with_bf16_activations();
        let mut rotation = (0..16).map(|_| case.tensors(&device, Element::bf16())).collect::<Vec<_>>();
        if label.starts_with("step") {
            for mapping in STEPS {
                let kernel = case.cuda_step(&device, Element::bf16(), mapping);
                let args = rotation.iter_mut().map(|t| t.step_args(&case)).collect();
                let measured = kernel.measure(args, &options).unwrap();
                println!("{label} {mapping:?}: {:.1} us", measured.median * 1e6);
            }
        } else {
            for mapping in CHUNK_ROWS {
                let kernel = case.cuda_chunk(&device, Element::bf16(), mapping);
                let args = rotation.iter_mut().map(|t| t.chunk_args(&case)).collect();
                let measured = kernel.measure(args, &options).unwrap();
                let launches = launch_medians(&device, || {
                    for t in rotation.iter_mut() {
                        kernel.call(t.chunk_args(&case)).unwrap();
                    }
                });
                println!("{label} ROWS {mapping}: {:.1} us (launches {launches} us)", measured.median * 1e6);
            }
        }
    }
}
