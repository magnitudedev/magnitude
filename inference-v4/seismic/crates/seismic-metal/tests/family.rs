use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::{
    execution::{prepare, Config},
    family::GroupFamily,
    msl::{emit_execution, emit_with},
};
use std::collections::HashMap;
#[test]
fn grouping_domain_matches_each_emitted_candidate_and_rejects_the_boundary() {
    let text="fn shift[M,N](x: tensor[M,N] f32, out: tensor[M,N] f32):\n  for row in parallel:\n    t = load(x[row])\n    for i in owned(t): t[i] = t[i] + 1.0\n    y = tile[N] f32\n    for i in owned(y): y[i] = t[(i + 1) % N]\n    store(y,out[row])\n";
    let program = compile(
        &[SourceFile {
            path: "family.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    for (rows, width, capacity) in [(5, 64, 1024), (11, 128, 2048), (1, 64, 256)] {
        let f = lower(
            &program,
            "shift",
            "metal",
            &HashMap::from([("M".into(), rows), ("N".into(), width)]),
        )
        .unwrap();
        let config = Config {
            max_threads_per_threadgroup: 256,
            max_threadgroup_bytes: capacity,
            ..Default::default()
        };
        let family = GroupFamily::derive(
            prepare(
                &f,
                Config {
                    sg_per_tg: 1,
                    ..config.clone()
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert!(family
            .constraints
            .iter()
            .any(|c| c.resource == "threadgroup_bytes"));
        let groups = family.groupings().collect::<Vec<_>>();
        let maximum = groups.last().unwrap().items_per_group;
        assert_eq!(maximum, (capacity / (width * 4)) as u64);
        for group in groups {
            let emitted = emit_execution(&family.select(group.items_per_group).unwrap()).unwrap();
            for ((actual, dispatch), shared) in emitted
                .launches
                .iter()
                .zip(&group.launches)
                .zip(&group.shared_bytes_per_group)
            {
                assert_eq!(actual.dispatch.as_ref(), Some(dispatch));
                assert_eq!(actual.declared_threadgroup_bytes, *shared);
            }
        }
        assert!(family.select(0).is_err());
        assert!(family.select(maximum + 1).is_err());
        assert!(emit_with(
            &f,
            Config {
                sg_per_tg: (maximum + 1) as i64,
                ..config.clone()
            }
        )
        .is_err());
    }
}

fn lower_source(text: &str) -> seismic_lang::lowered_ir::LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "grouping.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    lower(&program, "evaluate", "metal", &Default::default()).unwrap()
}

const SPLIT: &str = "fn evaluate(x: tensor[2,65] f32, middle: tensor[2] f32, out: tensor[2] f32):\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    for t in load(x[row,0:65], over=0):\n      acc[0] += reduce(t,0,sum)\n    store(acc,middle[row:row+1])\n  for row in parallel:\n    t = load(middle[row:row+1])\n    for i in owned(t): t[i] = t[i] * 2.0 + 1.0\n    store(t,out[row:row+1])\n";

fn split_family() -> GroupFamily {
    let lowered = lower_source(SPLIT);
    let execution = seismic_metal::execution::prepare_storage_selected(
        &lowered,
        Config {
            split: 3,
            sg_per_tg: 1,
            max_threads_per_threadgroup: 256,
            ..Default::default()
        },
        &mut |_| Ok(seismic_realization::dispatch::TilePlacement::GroupShared),
    )
    .unwrap();
    GroupFamily::derive(execution).unwrap()
}

#[test]
fn split_grouping_preserves_selected_operations_and_launch_order() {
    let family = split_family();
    let baseline = family.execution();
    assert_eq!(baseline.memory().launches().len(), 3);
    let identities = |execution: &seismic_metal::execution::Execution| {
        execution
            .memory()
            .launches()
            .iter()
            .map(|launch| launch.arrays.iter().map(|a| a.id).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    for grouping in family.groupings() {
        let execution = family.select(grouping.items_per_group).unwrap();
        assert_eq!(identities(&execution), identities(baseline));
        assert_eq!(
            execution
                .memory()
                .launches()
                .iter()
                .map(|l| l.barriers.keys().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            baseline
                .memory()
                .launches()
                .iter()
                .map(|l| l.barriers.keys().collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        let emitted = emit_execution(&execution).unwrap();
        assert_eq!(
            emitted
                .launches
                .iter()
                .map(|l| l.kernel.as_str())
                .collect::<Vec<_>>(),
            ["evaluate_0", "evaluate_0_merge", "evaluate_1"]
        );
        for ((launch, dispatch), shared) in emitted
            .launches
            .iter()
            .zip(&grouping.launches)
            .zip(&grouping.shared_bytes_per_group)
        {
            assert_eq!(launch.dispatch.as_ref(), Some(dispatch));
            assert_eq!(launch.declared_threadgroup_bytes, *shared);
        }
    }
}

#[test]
fn padding_exclusions_do_not_discard_legal_groupings() {
    let lowered = lower_source("fn evaluate(x: tensor[65535,65537,1] f32, out: tensor[65535,65537,1] f32):\n  for row,col in parallel:\n    t = load(x[row,col])\n    store(t,out[row,col])\n");
    let execution = prepare(
        &lowered,
        Config {
            sg_per_tg: 1,
            max_threads_per_threadgroup: 256,
            ..Default::default()
        },
    )
    .unwrap();
    let family = GroupFamily::derive(execution).unwrap();
    let legal = family
        .groupings()
        .map(|g| g.items_per_group)
        .collect::<Vec<_>>();
    assert_eq!(legal, [1, 3, 5]);
    assert!(family.select(2).err().unwrap().contains("slot index width"));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_split_groupings_preserve_phase_handoffs() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let family = split_family();
    let input: Vec<f32> = (0..130).map(|i| i as f32 * 0.25 - 7.0).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let expected: Vec<f32> = input
        .chunks(65)
        .map(|row| row.iter().sum::<f32>() * 2.0 + 1.0)
        .collect();
    for groups in [1, 2, 4, 8] {
        let selected = family.select(groups).unwrap();
        let kernel = device.compile(emit_execution(&selected).unwrap()).unwrap();
        let x = device.buffer_from(&bytes).unwrap();
        let middle = device.buffer(8).unwrap();
        let out = device.buffer(8).unwrap();
        device.run(&kernel, &[&x, &middle, &out], &[], 1).unwrap();
        let actual: Vec<f32> = out
            .read(8)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected, "grouping={groups}");
    }
}
