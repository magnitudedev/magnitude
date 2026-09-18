use seismic_lang::{
    Scope,
    lower::lower,
    program::{SourceFile, compile},
};
use seismic_metal::{execution::Config, msl::emit_storage_selected};
use seismic_realization::dispatch::TilePlacement;
use std::collections::HashMap;

fn program(text: &str, width: i64) -> seismic_lang::lowered_ir::LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "storage.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    lower(
        &program,
        "evaluate",
        "metal",
        &HashMap::from([("M".into(), 3), ("N".into(), width)]),
    )
    .unwrap()
}
const POINTWISE: &str = "fn evaluate[M,N](x: tensor[M,N] f32, out: tensor[M,N] f32):\n  for row in parallel:\n    t = load(x[row])\n    y = tile[N] f32\n    for i in owned(y): y[i] = t[i] * 2.0 + 1.0\n    store(y,out[row])\n";

#[test]
fn emitted_storage_is_the_selected_form_and_cross_owner_reads_reject() {
    for placement in [
        TilePlacement::Replicated,
        TilePlacement::Distributed,
        TilePlacement::GroupShared,
    ] {
        let lowered = program(POINTWISE, 65);
        let mut count = 0;
        let emitted = emit_storage_selected(
            &lowered,
            Config {
                loads: seismic_realization::LoadStrategy::Materialize,
                ..Default::default()
            },
            &mut |decision| {
                count += 1;
                assert!(decision.alternatives.contains(&placement));
                Ok(placement.clone())
            },
        )
        .unwrap();
        assert_eq!(count, 2);
        assert!(
            emitted
                .launches
                .iter()
                .flat_map(|l| &l.tiles)
                .all(|t| t.placement == placement)
        );
    }
    let lowered = program(
        "fn evaluate[M,N](x: tensor[M,N] f32, out: tensor[M,N] f32):\n  for row in parallel:\n    t = load(x[row])\n    for i in owned(t): t[i] = t[i] + 1.0\n    y = tile[N] f32\n    for i in owned(y): y[i] = t[(i+1)%N]\n    store(y,out[row])\n",
        65,
    );
    let error = emit_storage_selected(
        &lowered,
        Config {
            loads: seismic_realization::LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |_| Ok(TilePlacement::Distributed),
    )
    .unwrap_err();
    assert!(error.contains("incompatible"), "{error}");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_storage_forms_preserve_pointwise_results_and_tails() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let facts = device.info();
    for width in [1, 7, 32, 33, 65] {
        let lowered = program(POINTWISE, width);
        let input: Vec<_> = (0..3 * width).map(|i| (i as f32 - 75.0) * 0.25).collect();
        let input_bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        for placement in [
            TilePlacement::Replicated,
            TilePlacement::Distributed,
            TilePlacement::GroupShared,
        ] {
            let emitted = emit_storage_selected(
                &lowered,
                Config {
                    loads: seismic_realization::LoadStrategy::Materialize,
                    sg_per_tg: 2,
                    max_threads_per_threadgroup: facts.max_threads_per_threadgroup as i64,
                    max_threadgroup_bytes: facts.max_threadgroup_bytes as i64,
                    ..Default::default()
                },
                &mut |_| Ok(placement.clone()),
            )
            .unwrap();
            let pipeline = device.compile(emitted).unwrap();
            let x = device.buffer_from(&input_bytes).unwrap();
            let out = device.buffer(input_bytes.len()).unwrap();
            device.run(&pipeline, &[&x, &out], &[], 1).unwrap();
            for (bytes, x) in out.read(input_bytes.len()).chunks_exact(4).zip(&input) {
                assert_eq!(
                    f32::from_le_bytes(bytes.try_into().unwrap()),
                    x * 2.0 + 1.0,
                    "{placement:?}, width={width}"
                );
            }
        }
    }
}

#[test]
fn load_strategy_is_explicit_and_independent_of_tile_size() {
    use seismic_realization::LoadStrategy;
    for width in [1, 7, 32, 33, 65] {
        let lowered = program(POINTWISE, width);
        for loads in [
            LoadStrategy::Materialize,
            LoadStrategy::BorrowProvenReadOnly,
        ] {
            let emitted = seismic_metal::msl::emit_with(
                &lowered,
                Config {
                    loads,
                    ..Default::default()
                },
            )
            .unwrap();
            let snapshot = emitted
                .launches
                .iter()
                .flat_map(|l| &l.tiles)
                .any(|t| t.symbol == "t");
            assert_eq!(
                snapshot,
                loads == LoadStrategy::Materialize,
                "width {width}, {loads:?}"
            );
        }
    }
    // A store may alias the source of the snapshot. Borrowing is not legal merely
    // because that source has a different parameter name from the store target.
    let lowered = program(
        "fn evaluate[M,N](x: tensor[M,N] f32, out: tensor[M,N] f32):\n  for row in parallel:\n    t = load(x[row])\n    zeros = tile[N] f32\n    for i in owned(zeros): zeros[i] = 0.0\n    store(zeros,out[row])\n    store(t,out[row])\n",
        65,
    );
    let emitted = seismic_metal::msl::emit_with(
        &lowered,
        Config {
            loads: LoadStrategy::BorrowProvenReadOnly,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        emitted
            .launches
            .iter()
            .flat_map(|l| &l.tiles)
            .any(|t| t.symbol == "t")
    );
}

#[test]
fn storage_domains_are_available_from_ir_and_widening_recomputes_ownership() {
    use seismic_metal::storage::StorageAnalysis;
    let text = "fn evaluate[M,N](x: tensor[M,N] f32, out: tensor[M,N] f32):\n  for row in parallel:\n    t = load(x[row])\n    for i in owned(t): t[i] = t[i] + 1.0\n    y = tile[N] f32\n    for i in owned(y): y[i] = t[(i+1)%N]\n    store(y,out[row])\n";
    let lowered = program(text, 65);
    // Obtain domains before invoking any emitter.
    let analysis = StorageAnalysis::new(&lowered.vars, &lowered.body);
    let t = lowered.vars.iter().rposition(|v| v.name == "t").unwrap();
    let domain = analysis
        .decision(t, 65, seismic_lang::types::DType::F32)
        .unwrap();
    assert!(domain.cross_lane_read);
    assert!(domain.select(TilePlacement::Distributed).is_err());
    let declaration = domain.select(TilePlacement::GroupShared).unwrap();
    assert_eq!(declaration.capacity, 65);
    assert_eq!(
        declaration
            .bytes(&seismic_realization::dispatch::GroupDispatch::new(3, 32, 4).unwrap())
            .unwrap(),
        (0, 1040)
    );
    assert!(
        analysis
            .decision(t, 65, seismic_lang::types::DType::F16)
            .is_err()
    );
    assert_eq!(
        domain.alternatives,
        vec![TilePlacement::Replicated, TilePlacement::GroupShared]
    );
    assert!(
        analysis
            .decision(lowered.vars.len(), 65, seismic_lang::types::DType::F32)
            .is_err()
    );
    assert!(
        analysis
            .decision(t, -1, seismic_lang::types::DType::F32)
            .is_err()
    );
    let mut cross_reads = 0;
    emit_storage_selected(
        &lowered,
        Config {
            per_item: 3,
            ..Default::default()
        },
        &mut |decision| {
            if decision.cross_lane_read {
                cross_reads += 1;
                assert!(!decision.alternatives.contains(&TilePlacement::Distributed));
                Ok(TilePlacement::GroupShared)
            } else {
                Ok(TilePlacement::Distributed)
            }
        },
    )
    .unwrap();
    assert_eq!(
        cross_reads, 3,
        "each widened snapshot retains its cross-lane dependency"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn widened_storage_choices_preserve_cross_lane_reads() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let text = "fn evaluate[M,N](x: tensor[M,N] f32, out: tensor[M,N] f32):\n  for row in parallel:\n    t = load(x[row])\n    for i in owned(t): t[i] = t[i] + 1.0\n    y = tile[N] f32\n    for i in owned(y): y[i] = t[(i+1)%N]\n    store(y,out[row])\n";
    for width in [7, 32, 65] {
        let lowered = program(text, width);
        let emitted = emit_storage_selected(
            &lowered,
            Config {
                per_item: 3,
                ..Default::default()
            },
            &mut |decision| {
                Ok(if decision.cross_lane_read {
                    TilePlacement::GroupShared
                } else {
                    TilePlacement::Distributed
                })
            },
        )
        .unwrap();
        let pipeline = device.compile(emitted).unwrap();
        let values: Vec<_> = (0..3 * width).map(|i| i as f32).collect();
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x = device.buffer_from(&bytes).unwrap();
        let out = device.buffer(bytes.len()).unwrap();
        device.run(&pipeline, &[&x, &out], &[], 1).unwrap();
        for (i, bytes) in out.read(bytes.len()).chunks_exact(4).enumerate() {
            let row = i / width as usize;
            let col = i % width as usize;
            assert_eq!(
                f32::from_le_bytes(bytes.try_into().unwrap()),
                values[row * width as usize + (col + 1) % width as usize] + 1.0
            );
        }
    }
}

#[test]
fn storage_is_resolved_before_printing_and_reused_without_selection() {
    let lowered = program(POINTWISE, 65);
    let mut decisions = Vec::new();
    let execution = seismic_metal::execution::prepare_storage_selected(
        &lowered,
        Config {
            loads: seismic_realization::LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |domain| {
            decisions.push(domain.variable);
            Ok(TilePlacement::GroupShared)
        },
    )
    .unwrap();
    assert_eq!(decisions.len(), 2);
    let planned = execution.storage().declarations();
    assert_eq!(planned.len(), 2);
    for declaration in planned.values() {
        assert_eq!(declaration.capacity, 65);
        assert_eq!(declaration.placement, TilePlacement::GroupShared);
    }
    for _ in 0..2 {
        let emitted = seismic_metal::msl::emit_execution(&execution).unwrap();
        for declaration in emitted.launches.iter().flat_map(|l| &l.tiles) {
            assert!(planned.values().any(|p| p == declaration));
        }
    }
    assert_eq!(decisions.len(), 2);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn tile_reassignment_snapshots_overlapping_views_before_writing() {
    let lowered = program(
        "fn evaluate[M,N](x: tensor[N,N] f32, out: tensor[N,N] f32):\n  a = load(x)\n  a = a.T\n  for step in range(0,2):\n    a = a.T\n  store(a,out)\n",
        7,
    );
    let device = seismic_metal::runtime::Device::open().unwrap();
    let facts = device.info();
    let values: Vec<_> = (0..49).map(|i| i as f32 * 0.25).collect();
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    for placement in [TilePlacement::Replicated, TilePlacement::GroupShared] {
        let emitted = emit_storage_selected(
            &lowered,
            Config {
                loads: seismic_realization::LoadStrategy::Materialize,
                max_threads_per_threadgroup: facts.max_threads_per_threadgroup as i64,
                max_threadgroup_bytes: facts.max_threadgroup_bytes as i64,
                ..Default::default()
            },
            &mut |_| Ok(placement.clone()),
        )
        .unwrap();
        let pipeline = device.compile(emitted).unwrap();
        let input = device.buffer_from(&bytes).unwrap();
        let output = device.buffer(bytes.len()).unwrap();
        device.run(&pipeline, &[&input, &output], &[], 1).unwrap();
        for (i, value) in output.read(bytes.len()).chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_le_bytes(value.try_into().unwrap()),
                values[(i % 7) * 7 + i / 7],
                "{placement:?} at {i}"
            );
        }
    }
}

const CROSS_PUBLICATION: &str = "fn evaluate(x:tensor[65] f32,out:tensor[65] f32):\n  a=load(x)\n  b=tile[65] f32\n  for i in owned(a): b[i]=a[i]+1.0\n  c=tile[65] f32\n  for i in owned(c): c[i]=b[(i+1)%65]\n  store(c,out)\n";

#[test]
fn storage_domains_couple_element_publication_and_distributed_read_owners() {
    let f = program(CROSS_PUBLICATION, 65);
    for first in [TilePlacement::Replicated, TilePlacement::Distributed, TilePlacement::GroupShared] {
        let mut seen = 0;
        let execution = seismic_metal::execution::prepare_storage_selected(&f, Config { loads: seismic_realization::LoadStrategy::Materialize, ..Default::default() }, &mut |d| {
            seen += 1;
            Ok(if d.name.starts_with("a") { first.clone() }
            else if d.name.starts_with("b") {
                let required = if first == TilePlacement::Replicated { TilePlacement::Replicated } else { TilePlacement::GroupShared };
                assert_eq!(d.alternatives, vec![required.clone()], "publication must agree with its owner");
                required
            } else { TilePlacement::Replicated })
        }).unwrap();
        assert_eq!(seen, 3);
        if first == TilePlacement::Distributed {
            let a = f.vars.iter().position(|v| v.name.starts_with("a")).unwrap();
            assert!(execution.memory().launches()[0].barriers.iter().any(|(site, barrier)| site.variable == a
                && site.purpose == seismic_metal::memory::BarrierPurpose::Owned
                && barrier.memory == seismic_metal::memory::MemorySpace::Threadgroup), "the shared destination must be published even when its loop owner is private distributed storage");
        }
    }
    let error = seismic_metal::execution::prepare_storage_selected(&f, Config { loads: seismic_realization::LoadStrategy::Materialize, ..Default::default() }, &mut |d| Ok(if d.name.starts_with("a") { TilePlacement::GroupShared } else { TilePlacement::Replicated })).err().unwrap();
    assert!(error.contains("incompatible"), "{error}");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_cross_tile_publication_preserves_cooperative_and_replicated_values() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let input: Vec<f32> = (0..65).map(|i| i as f32 / 8.0 - 4.0).collect();
    let x = device.buffer_from(&input.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let out = device.buffer(65 * 4).unwrap();
    let f = program(CROSS_PUBLICATION, 65);
    for first in [TilePlacement::Replicated, TilePlacement::Distributed, TilePlacement::GroupShared] {
        let execution = seismic_metal::execution::prepare_storage_selected(&f, Config { loads: seismic_realization::LoadStrategy::Materialize, ..Default::default() }, &mut |d| Ok(if d.name.starts_with("a") { first.clone() } else if d.name.starts_with("b") && first != TilePlacement::Replicated { TilePlacement::GroupShared } else { TilePlacement::Replicated })).unwrap();
        let kernel = device.compile(seismic_metal::msl::emit_execution(&execution).unwrap()).unwrap();
        device.run(&kernel, &[&x, &out], &[], 1).unwrap();
        let expected = (0..65).flat_map(|i| (input[(i + 1) % 65] + 1.0).to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(out.read(65 * 4), expected, "{first:?}");
    }
}
