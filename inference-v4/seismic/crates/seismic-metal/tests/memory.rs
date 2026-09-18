use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::memory::{BarrierPurpose, MemorySpace, Purpose};
use seismic_metal::{
    execution::{prepare_storage_selected, Config},
    msl::emit_execution,
};
use seismic_realization::{dispatch::TilePlacement, LoadStrategy};

fn prepare(text: &str, placement: TilePlacement) -> seismic_metal::execution::Execution {
    let program = compile(
        &[SourceFile {
            path: "memory.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    prepare_storage_selected(
        &lowered,
        Config {
            loads: LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |_| Ok(placement.clone()),
    )
    .unwrap()
}

#[test]
fn synchronization_sites_follow_selected_memory_ownership() {
    let text = "fn evaluate(x: tensor[65] f32, out: tensor[65] f32):\n  a = load(x)\n  a = load(x)\n  b = a\n  for i in owned(b): b[i] = b[i] + 1.0\n  store(b,out)\n";
    for placement in [
        TilePlacement::Replicated,
        TilePlacement::Distributed,
        TilePlacement::GroupShared,
    ] {
        let execution = prepare(text, placement.clone());
        let barriers = &execution.memory().launches()[0].barriers;
        let emitted = emit_execution(&execution).unwrap();
        if placement == TilePlacement::GroupShared {
            assert_eq!(barriers.len(), 5);
            assert!(barriers
                .values()
                .all(|b| b.memory == MemorySpace::Threadgroup));
            assert_eq!(
                barriers
                    .keys()
                    .filter(|s| s.purpose == BarrierPurpose::Snapshot(Purpose::Value))
                    .count(),
                2
            );
            assert_eq!(
                barriers
                    .keys()
                    .filter(|s| s.purpose == BarrierPurpose::Copy)
                    .count(),
                2
            );
            assert_eq!(
                barriers
                    .keys()
                    .filter(|s| s.purpose == BarrierPurpose::Owned)
                    .count(),
                1
            );
        } else {
            assert!(barriers.is_empty());
        }
        assert_eq!(
            emitted.source.matches("simdgroup_barrier(").count(),
            barriers.len()
        );
    }
}

fn fragment_execution() -> seismic_metal::execution::Execution {
    let portable = "construct transfer(x: tile[8,8] f32, out: tile[8,8] f32):\n  for i,j in owned(out): out[i,j] = x[i,j]\nfn evaluate(x: tensor[8,8] f32, out: tensor[8,8] f32):\n  a = load(x)\n  b = tile[8,8] f32\n  transfer(a,b)\n  store(b,out)\n";
    let backend = "lower transfer(x: tile[8,8] f32, out: tile[8,8] f32):\n  tmp = tile[8,8] f32\n  a = simdgroup_matrix(f32)\n  simdgroup_load(a,x,0,0)\n  simdgroup_store(a,tmp,0,0)\n  b = simdgroup_matrix(f32)\n  simdgroup_load(b,tmp,0,0)\n  simdgroup_store(b,out,0,0)\n";
    let program = compile(
        &[
            SourceFile {
                path: "fragment.seismic.portable".into(),
                scope: Scope::Portable,
                text: portable.into(),
            },
            SourceFile {
                path: "fragment.seismic.metal".into(),
                scope: Scope::Backend("metal".into()),
                text: backend.into(),
            },
        ],
        &[],
    )
    .unwrap();
    let lowered = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    seismic_metal::execution::prepare(&lowered, Config::default()).unwrap()
}

#[test]
fn fragment_publication_orders_owned_shared_storage() {
    let execution = fragment_execution();
    let launch = &execution.memory().launches()[0];
    assert_eq!(launch.fragments.len(), 2);
    let stores: Vec<_> = launch
        .barriers
        .iter()
        .filter(|(s, _)| s.purpose == BarrierPurpose::IntrinsicStore)
        .collect();
    assert_eq!(stores.len(), 2);
    assert!(stores
        .iter()
        .all(|(_, b)| b.memory == MemorySpace::Threadgroup));
    let source = emit_execution(&execution).unwrap().source;
    assert_eq!(
        source
            .matches("simdgroup_barrier(mem_flags::mem_threadgroup)")
            .count(),
        launch.barriers.len()
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_fragment_publication_preserves_cross_lane_values() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 5.0).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let execution = fragment_execution();
    let kernel = device.compile(emit_execution(&execution).unwrap()).unwrap();
    let x = device.buffer_from(&bytes).unwrap();
    let out = device.buffer(bytes.len()).unwrap();
    device.run(&kernel, &[&x, &out], &[], 1).unwrap();
    assert_eq!(out.read(bytes.len()), bytes);
}

fn two_split_phases() -> seismic_metal::execution::Execution {
    let text = "fn evaluate(x: tensor[2,65] f32, middle: tensor[2] f32, out: tensor[2] f32):\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    for t in load(x[row,0:65], over=0):\n      acc[0] += reduce(t,0,sum)\n    store(acc,middle[row:row+1])\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    for t in load(middle[row:row+1], over=0):\n      acc[0] += reduce(t,0,sum)\n    store(acc,out[row:row+1])\n";
    let program = compile(
        &[SourceFile {
            path: "scratch.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    seismic_metal::execution::prepare(
        &lowered,
        Config {
            split: 3,
            sg_per_tg: 1,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn scratch_sizes_and_lifetimes_exist_before_emission() {
    let execution = two_split_phases();
    let memory = execution.memory();
    assert_eq!(
        memory
            .launches()
            .iter()
            .map(|l| l.predecessor)
            .collect::<Vec<_>>(),
        [None, Some(0), Some(1), Some(2)]
    );
    assert_eq!(memory.scratch().len(), 2);
    for (phase, scratch) in memory.scratch().iter().enumerate() {
        assert_eq!(scratch.index, phase);
        assert_eq!(scratch.phase, phase);
        assert_eq!(
            (scratch.producer, scratch.consumer),
            (phase * 2, phase * 2 + 1)
        );
        assert_eq!(
            (scratch.work_items, scratch.parts, scratch.elements_per_item),
            (2, 3, 1)
        );
        assert_eq!(scratch.bytes, 24);
        assert_eq!(scratch.dtype, seismic_lang::types::DType::F32);
    }
    let family = seismic_metal::family::GroupFamily::derive(execution).unwrap();
    for grouping in [1, 2, 4] {
        let selected = family.select(grouping).unwrap();
        assert_eq!(
            selected.memory().scratch(),
            family.execution().memory().scratch()
        );
        let emitted = emit_execution(&selected).unwrap();
        assert_eq!(emitted.scratch, [24, 24]);
        assert_eq!(
            emitted
                .launches
                .iter()
                .map(|l| l.after_barrier)
                .collect::<Vec<_>>(),
            [false, true, true, true]
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_separate_scratch_handoffs_preserve_two_split_phases() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let family = seismic_metal::family::GroupFamily::derive(two_split_phases()).unwrap();
    let input: Vec<f32> = (0..130).map(|i| i as f32 * 0.25 - 3.0).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let expected: Vec<u8> = input
        .chunks(65)
        .flat_map(|row| row.iter().sum::<f32>().to_le_bytes())
        .collect();
    for grouping in [1, 4] {
        let selected = family.select(grouping).unwrap();
        let kernel = device.compile(emit_execution(&selected).unwrap()).unwrap();
        let x = device.buffer_from(&bytes).unwrap();
        let middle = device.buffer(8).unwrap();
        let out = device.buffer(8).unwrap();
        device.run(&kernel, &[&x, &middle, &out], &[], 1).unwrap();
        assert_eq!(out.read(8), expected, "grouping={grouping}");
    }
}

fn snapshot_in_owned(output: TilePlacement) -> Result<seismic_metal::execution::Execution, String> {
    let text = "fn evaluate(x: tensor[65] f32, out: tensor[1] f32):\n  y = tile[1] f32\n  for i in owned(y):\n    y[i] = 0.0\n    t = load(x)\n    y[i] += reduce(t,0,sum)\n  store(y,out)\n";
    let program = compile(
        &[SourceFile {
            path: "partial-memory.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    prepare_storage_selected(
        &lowered,
        Config {
            loads: LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |decision| {
            Ok(if decision.name == "y" {
                output.clone()
            } else {
                TilePlacement::GroupShared
            })
        },
    )
}

#[test]
fn shared_snapshot_inside_partial_owned_domain_rejects_before_emission() {
    for output in [TilePlacement::Distributed, TilePlacement::GroupShared] {
        let error = snapshot_in_owned(output)
            .err()
            .expect("partial snapshot barrier must be rejected");
        assert!(
            error.contains("memory barrier") && error.contains("full-lane participation"),
            "{error}"
        );
    }
    let execution = snapshot_in_owned(TilePlacement::Replicated).unwrap();
    assert_eq!(execution.memory().launches()[0].barriers.len(), 1);
    emit_execution(&execution).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_shared_snapshot_in_replicated_owned_domain_preserves_sum() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let execution = snapshot_in_owned(TilePlacement::Replicated).unwrap();
    let input: Vec<f32> = (0..65).map(|i| i as f32).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kernel = device.compile(emit_execution(&execution).unwrap()).unwrap();
    let x = device.buffer_from(&bytes).unwrap();
    let out = device.buffer(4).unwrap();
    device.run(&kernel, &[&x, &out], &[], 1).unwrap();
    assert_eq!(out.read(4), input.iter().sum::<f32>().to_le_bytes());
}

fn selected_backend_body(body: &str) -> Result<seismic_metal::execution::Execution, String> {
    let portable = "construct transfer(x: tile[32] f32, out: tile[1] f32):\n  out[0] = x[0]\nfn evaluate(x: tensor[32] f32, out: tensor[1] f32):\n  a = load(x)\n  b = tile[1] f32\n  for i in owned(b): b[i] = 0.0\n  transfer(a,b)\n  store(b,out)\n";
    let backend = format!("lower transfer(x: tile[32] f32, out: tile[1] f32):\n{body}");
    let program = compile(
        &[
            SourceFile {
                path: "collective.seismic.portable".into(),
                scope: Scope::Portable,
                text: portable.into(),
            },
            SourceFile {
                path: "collective.seismic.metal".into(),
                scope: Scope::Backend("metal".into()),
                text: backend,
            },
        ],
        &[],
    )
    .map_err(|e| format!("{e:?}"))?;
    let lowered =
        lower(&program, "evaluate", "metal", &Default::default()).map_err(|e| format!("{e:?}"))?;
    seismic_metal::execution::prepare(&lowered, Config::default())
}

#[test]
fn collective_implementations_are_admitted_and_consumed_before_native_compilation() {
    let execution = selected_backend_body("  out[0] = simd_sum(simd_max(0.0))\n").unwrap();
    let launch = &execution.memory().launches()[0];
    assert_eq!(launch.collectives.len(), 2);
    let names: Vec<_> = launch
        .collectives
        .values()
        .map(|c| c.implementation.metal_builtin().unwrap())
        .collect();
    assert_eq!(names, ["simd_max", "simd_sum"]);
    emit_execution(&execution).unwrap();
    let invalid = selected_backend_body(
        "  fragment = simdgroup_matrix(i32)\n  for i in owned(out): out[i] = 0.0\n",
    )
    .err()
    .unwrap();
    assert!(invalid.contains("matrices do not support i32"), "{invalid}");
}

#[test]
fn lane_dependent_collective_control_is_rejected_during_preparation() {
    let invalid = selected_backend_body(
        "  for k in lanes(32,1):\n    if k == 0:\n      out[0] = simd_sum(x[k])\n",
    )
    .err()
    .unwrap();
    assert!(invalid.contains("full-lane participation"), "{invalid}");
}

fn tensor_control(body: &str) -> Result<seismic_metal::execution::Execution, String> {
    let portable = "construct transfer(x: tensor[32] f32, out: tile[1] f32):\n  out[0] = x[0]\nfn evaluate(x: tensor[32] f32, out: tensor[1] f32):\n  b = tile[1] f32\n  for i in owned(b): b[i] = 0.0\n  transfer(x,b)\n  store(b,out)\n";
    let backend = format!("lower transfer(x: tensor[32] f32, out: tile[1] f32):\n{body}");
    let program = compile(
        &[
            SourceFile {
                path: "tensor-control.seismic.portable".into(),
                scope: Scope::Portable,
                text: portable.into(),
            },
            SourceFile {
                path: "tensor-control.seismic.metal".into(),
                scope: Scope::Backend("metal".into()),
                text: backend,
            },
        ],
        &[],
    )
    .map_err(|e| format!("{e:?}"))?;
    let lowered =
        lower(&program, "evaluate", "metal", &Default::default()).map_err(|e| format!("{e:?}"))?;
    seismic_metal::execution::prepare(&lowered, Config::default())
}

#[test]
fn common_tensor_addresses_prove_uniform_control_without_assuming_private_values_agree() {
    let execution = tensor_control("  if x[0] > 0.0:\n    out[0] = simd_sum(1.0)\n")
        .expect("the same stable tensor read agrees in every lane");
    assert_eq!(execution.memory().launches()[0].collectives.len(), 1);
    emit_execution(&execution).unwrap();

    for body in [
        "  for k in lanes(32,1):\n    if x[k] > 0.0:\n      out[0] = simd_sum(1.0)\n",
        "  partial = tile[1] f32\n  for i in owned(partial): partial[i] = 0.0\n  for k in lanes(32,1): partial[0] = f32(k)\n  if partial[0] > 0.0:\n    out[0] = simd_sum(1.0)\n",
        "  store(out,x[0:1])\n  if x[0] > 0.0:\n    out[0] = simd_sum(1.0)\n",
        "  for step in range(2):\n    if x[0] > 0.0:\n      out[0] = simd_sum(1.0)\n    store(out,x[0:1])\n",
    ] {
        let error = tensor_control(body)
            .err()
            .expect("varying or potentially unpublished memory cannot prove collective control");
        assert!(error.contains("full-lane participation"), "{body}\n{error}");
    }
}
