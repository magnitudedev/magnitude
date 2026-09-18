use seismic_lang::{
    ir::{StmtKind, VarKind},
    lower::{lower_with, Options},
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::{
    execution::{prepare, Config},
    msl::emit_execution,
};

fn lower(text: &str) -> LoweredIr {
    let p = compile(
        &[SourceFile {
            path: "execution.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    lower_with(
        &p,
        "evaluate",
        "metal",
        &Default::default(),
        &Options { piece: Some(17) },
    )
    .unwrap()
}

#[test]
fn widening_is_present_in_ir_before_emission() {
    let original = lower("fn evaluate(x: tensor[6,65] f32, out: tensor[6,65] f32):\n  for row in parallel:\n    t = load(x[row])\n    for i in owned(t): t[i] = t[i] + 1.0\n    store(t,out[row])\n");
    let original_debug = format!("{original:?}");
    let execution = prepare(
        &original,
        Config {
            per_item: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(execution.phases()[0].dispatch.work_items, 2);
    assert!(execution.function().vars.len() > original.vars.len());
    let StmtKind::Parallel { body, .. } = &execution.function().body[0].kind else {
        panic!()
    };
    let analysis = seismic_metal::storage::StorageAnalysis::new(&execution.function().vars, body);
    let loads = body
        .iter()
        .filter_map(|s| match &s.kind {
            StmtKind::Assign { target, value, .. }
                if matches!(value.kind, seismic_lang::ir::ExprKind::Load { .. }) =>
            {
                let seismic_lang::ir::ExprKind::Var(v) = target.kind else {
                    return None;
                };
                Some(
                    analysis
                        .decision(v, 65, seismic_lang::types::DType::F32)
                        .unwrap(),
                )
            }
            _ => None,
        })
        .count();
    assert_eq!(loads, 3);
    let before = format!("{:?}", execution.function());
    let first = emit_execution(&execution).unwrap();
    let second = emit_execution(&execution).unwrap();
    assert_eq!(first.source, second.source);
    assert_eq!(before, format!("{:?}", execution.function()));
    assert_eq!(original_debug, format!("{original:?}"));
    assert_eq!(
        first.launches[0].dispatch.as_ref(),
        Some(&execution.phases()[0].dispatch)
    );
    assert!(prepare(
        &original,
        Config {
            per_item: 4,
            ..Default::default()
        }
    )
    .is_err());
}

const SPLIT_THEN_READ: &str = "fn evaluate(x: tensor[2,65] f32, middle: tensor[2] f32, out: tensor[2] f32):\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    for t in load(x[row,0:65], over=0):\n      acc[0] += reduce(t,0,sum)\n    store(acc,middle[row:row+1])\n  for row in parallel:\n    t = load(middle[row:row+1])\n    for i in owned(t): t[i] = t[i] * 2.0 + 1.0\n    store(t,out[row:row+1])\n";

#[test]
fn split_indices_and_phase_dependencies_exist_before_emission() {
    let original = lower(SPLIT_THEN_READ);
    let execution = prepare(
        &original,
        Config {
            split: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(execution.phases().len(), 2);
    let split = execution.phases()[0].split.as_ref().unwrap();
    assert!(split.part >= original.vars.len());
    assert!(matches!(
        execution.function().vars[split.part].kind,
        VarKind::Index(_)
    ));
    assert_eq!(execution.phases()[0].dispatch.work_items, 6);
    assert_eq!(execution.phases()[1].dispatch.work_items, 2);
    assert!(execution.phases()[1].split.is_none());
    let emitted = emit_execution(&execution).unwrap();
    assert_eq!(
        emitted
            .launches
            .iter()
            .map(|l| l.kernel.as_str())
            .collect::<Vec<_>>(),
        ["evaluate_0", "evaluate_0_merge", "evaluate_1"]
    );
    assert!(!emitted.launches[0].after_barrier);
    assert!(emitted.launches[1..].iter().all(|l| l.after_barrier));
}

#[test]
fn reduction_bindings_preserve_the_split_loop_and_original_domain_inputs() {
    let text = SPLIT_THEN_READ.replace("    acc = tile[1] f32", "    probe = load(x[row,0:1])\n    unused = 1.0 + reduce(probe,0,sum)\n    acc = tile[1] f32")
        .replace("x[row,0:65]", "x[row,reduce(probe,0,argmax):65]");
    let original = lower(&text);
    let execution = prepare(
        &original,
        Config {
            split: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let split = execution.phases()[0].split.as_ref().unwrap();
    let StmtKind::Parallel { body, .. } = &execution.function().body[0].kind else {
        panic!()
    };
    assert!(matches!(
        body[split.loop_at].kind,
        StmtKind::LoadLoop { .. }
    ));
    assert!(!split.validation_bindings.is_empty());
    let before = format!("{:?}", execution.function());
    emit_execution(&execution).unwrap();
    assert_eq!(before, format!("{:?}", execution.function()));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn split_merge_finishes_before_the_next_phase_reads_its_output() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    for parts in [2, 3, 7, 100] {
        let execution = prepare(
            &lower(SPLIT_THEN_READ),
            Config {
                split: parts,
                ..Default::default()
            },
        )
        .unwrap();
        let pipeline = device.compile(emit_execution(&execution).unwrap()).unwrap();
        let values: Vec<f32> = (0..130).map(|i| i as f32 - 30.0).collect();
        let input = device
            .buffer_from(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let sentinel = [f32::NAN; 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let middle = device.buffer_from(&sentinel).unwrap();
        let output = device.buffer_from(&sentinel).unwrap();
        device
            .run(&pipeline, &[&input, &middle, &output], &[], 1)
            .unwrap();
        for (row, bytes) in output.read(8).chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_le_bytes(bytes.try_into().unwrap()),
                values[row * 65..(row + 1) * 65].iter().sum::<f32>() * 2.0 + 1.0
            );
        }
    }
}

#[test]
fn split_requires_a_proven_identity_and_chunk_independent_merge() {
    for text in [
        SPLIT_THEN_READ.replace("acc[i] = 0.0", "acc[i] = 1.0"),
        SPLIT_THEN_READ.replace("reduce(t,0,sum)", "reduce(t,0,max)"),
        SPLIT_THEN_READ.replace("reduce(t,0,sum)", "reduce(t,0,sum,ordered=true)"),
        SPLIT_THEN_READ.replace("acc[0] +=", "acc[0] *= "),
        SPLIT_THEN_READ.replace("reduce(t,0,sum)", "reduce(t,0,sum) + acc[0]"),
    ] {
        assert!(
            prepare(
                &lower(&text),
                Config {
                    split: 3,
                    ..Default::default()
                }
            )
            .is_err(),
            "{text}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn split_preserves_dynamic_domains_including_empty_and_invalid_ranges() {
    let text = SPLIT_THEN_READ
        .replace(
            "middle: tensor[2]",
            "limits: tensor[2] i32, middle: tensor[2]",
        )
        .replace("x[row,0:65]", "x[row,limits[0]:limits[1]]");
    let execution = prepare(
        &lower(&text),
        Config {
            split: 7,
            ..Default::default()
        },
    )
    .unwrap();
    let device = seismic_metal::runtime::Device::open().unwrap();
    let pipeline = device.compile(emit_execution(&execution).unwrap()).unwrap();
    let values: Vec<f32> = (0..130).map(|i| i as f32).collect();
    let input = device
        .buffer_from(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let middle = device.buffer(8).unwrap();
    let output = device.buffer(8).unwrap();
    for (lo, hi) in [
        (0i32, 65i32),
        (3, 64),
        (65, 65),
        (60, 61),
        (-1, 2),
        (2, 1),
        (0, 66),
    ] {
        let limits = device
            .buffer_from(
                &[lo, hi]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let result = device.run(&pipeline, &[&input, &limits, &middle, &output], &[], 1);
        if lo < 0 || hi < lo || hi > 65 {
            assert!(result.is_err(), "{lo}:{hi}");
            continue;
        }
        result.unwrap();
        for (row, bytes) in output.read(8).chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_le_bytes(bytes.try_into().unwrap()),
                values[row * 65 + lo as usize..row * 65 + hi as usize]
                    .iter()
                    .sum::<f32>()
                    * 2.0
                    + 1.0,
                "{lo}:{hi}"
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn split_keeps_each_streams_own_slice_origin() {
    let text = "fn evaluate(x: tensor[65] f32, out: tensor[1] f32):\n  a = tile[1] f32\n  b = tile[1] f32\n  for i in owned(a): a[i] = 0.0\n  for i in owned(b): b[i] = 0.0\n  for u, v in load((x[0:64],x[1:65]), over=0):\n    a[0] += reduce(u,0,sum)\n    b[0] += reduce(v,0,sum)\n  a[0] += b[0]\n  store(a,out)\n";
    let execution = prepare(
        &lower(text),
        Config {
            split: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let device = seismic_metal::runtime::Device::open().unwrap();
    let pipeline = device.compile(emit_execution(&execution).unwrap()).unwrap();
    let input = device
        .buffer_from(
            &(0..65)
                .flat_map(|v| (v as f32).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let output = device.buffer(4).unwrap();
    device.run(&pipeline, &[&input, &output], &[], 1).unwrap();
    assert_eq!(
        f32::from_le_bytes(output.read(4).try_into().unwrap()),
        4096.0
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn multidimensional_work_mapping_preserves_coordinates_and_empty_domains() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    for middle in [0, 6] {
        let text = format!("fn evaluate(x: tensor[2,{middle},5] f32, out: tensor[2,{middle},5] f32):\n  for a, b in parallel:\n    t = load(x[a,b])\n    for i in owned(t): t[i] = t[i] + 1.0\n    store(t,out[a,b])\n");
        let original = lower(&text);
        for per_item in [1, 2, 3, 6] {
            let execution = prepare(
                &original,
                Config {
                    per_item,
                    sg_per_tg: 2,
                    ..Default::default()
                },
            )
            .unwrap();
            let mapping = &execution.phases()[0].mapping;
            assert_eq!(mapping.work_items(), (2 * middle / per_item) as u64);
            let pipeline = device.compile(emit_execution(&execution).unwrap()).unwrap();
            let count = (2 * middle * 5) as usize;
            let values: Vec<f32> = (0..count.max(1)).map(|i| i as f32).collect();
            let input = device
                .buffer_from(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let sentinel = vec![-999.0f32; count.max(1)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let output = device.buffer_from(&sentinel).unwrap();
            device.run(&pipeline, &[&input, &output], &[], 1).unwrap();
            for (at, bytes) in output.read(sentinel.len()).chunks_exact(4).enumerate() {
                let expected = if count == 0 { -999.0 } else { values[at] + 1.0 };
                assert_eq!(
                    f32::from_le_bytes(bytes.try_into().unwrap()),
                    expected,
                    "middle={middle}, per_item={per_item}, at={at}"
                );
            }
        }
    }
}

#[test]
fn duplicated_source_spans_have_distinct_execution_decisions() {
    let mut original = lower("fn evaluate(x: tensor[65] f32, out: tensor[1] f32):\n  a = load(x)\n  r = reduce(a,0,sum)\n  y = tile[1] f32\n  for i in owned(y): y[i] = r\n  store(y,out)\n");
    // Inlining and expansion can preserve source provenance for distinct operations.
    let repeated = original.body[1].clone();
    original.body.insert(2, repeated);
    assert_eq!(original.body[1].span, original.body[2].span);
    let execution = prepare(&original, Config::default()).unwrap();
    let reductions = execution.reductions().selections();
    assert_eq!(reductions.len(), 2);
    let sites = reductions.keys().copied().collect::<Vec<_>>();
    assert_eq!(sites[0].output, sites[1].output);
    assert_ne!(sites[0].operation, sites[1].operation);
    emit_execution(&execution).unwrap();
}

#[test]
fn operation_ids_cover_main_and_split_validation_bodies_without_collisions() {
    use seismic_lang::ir::{OperationId, Stmt};
    fn collect(body: &[Stmt], ids: &mut std::collections::HashSet<OperationId>) {
        for statement in body {
            assert!(ids.insert(statement.id.expect("normalized operation identity")));
            match &statement.kind {
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => collect(body, ids),
                StmtKind::If { then, els, .. } => {
                    collect(then, ids);
                    collect(els, ids);
                }
                _ => {}
            }
        }
    }
    let text = SPLIT_THEN_READ
        .replace(
            "    acc = tile[1] f32",
            "    probe = load(x[row,0:1])\n    acc = tile[1] f32",
        )
        .replace("x[row,0:65]", "x[row,reduce(probe,0,argmax):65]");
    let original = lower(&text);
    let execution = prepare(
        &original,
        Config {
            split: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let mut ids = std::collections::HashSet::new();
    collect(&execution.function().body, &mut ids);
    let body_count = ids.len();
    for phase in execution.phases() {
        if let Some(split) = &phase.split {
            collect(&split.validation_bindings, &mut ids);
        }
    }
    assert!(ids.len() > body_count);
    let repeated = prepare(
        &original,
        Config {
            split: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(execution.function().body, repeated.function().body);
    emit_execution(&execution).unwrap();
}
