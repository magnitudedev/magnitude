use seismic_accounting::{
    schedule::{CapacityUnit, Resource, Timebase},
    workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload},
};
use seismic_lang::{
    Scope,
    program::{SourceFile, compile},
};
use seismic_metal::{
    execution::{self, Config},
    model::{self, Hardware, Service, Timing, Units},
    terminal::Primitive,
};
fn source() -> seismic_lang::lowered_ir::LoweredIr {
    let p=compile(&[SourceFile{path:"terminal.seismic.portable".into(),scope:Scope::Portable,text:"fn evaluate(out: tensor[2] f32):\n  y = tile[2] f32\n  for i in owned(y): y[i] = 3.0\n  store(y,out)\n".into()}],&[]).unwrap();
    seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap()
}
fn workload() -> ScalarWorkload {
    ScalarWorkload {
        identity: "two outputs".into(),
        allocations: vec![Allocation {
            id: 1,
            bytes: 8,
            alignment: 4,
            known_bytes: Default::default(),
        }],
        buffers: vec![BufferBinding {
            allocation: 1,
            offset: 0,
            bytes: 8,
        }],
        scalars: vec![],
    }
}
fn synthetic(keys: Vec<Primitive>) -> Hardware {
    Hardware {
        identity: "test pooled MSL machine; not physical Metal timing".into(),
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![Resource {
            name: "service".into(),
            capacity: 1024,
            unit: CapacityUnit::Slots,
        }],
        resident_groups: 2,
        resident_shared_bytes: 65536,
        timings: keys
            .into_iter()
            .map(|primitive| Timing {
                primitive,
                latency: 1,
                services: vec![Service {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: Units::PerLane(1),
                }],
            })
            .collect(),
    }
}
#[test]
fn selected_storage_implementations_have_terminal_resource_mappings() {
    let f = source();
    let mut all = Vec::new();
    let mut executions = Vec::new();
    for placement in [
        seismic_realization::dispatch::TilePlacement::Replicated,
        seismic_realization::dispatch::TilePlacement::Distributed,
        seismic_realization::dispatch::TilePlacement::GroupShared,
    ] {
        let e = execution::prepare_storage_selected(&f, Config::default(), &mut |_| {
            Ok(placement.clone())
        })
        .unwrap();
        let r = model::requirements(&e).unwrap();
        assert!(r.unmapped.is_empty(), "{:?}", r.unmapped);
        for p in r.primitives {
            if !all.contains(&p) {
                all.push(p);
            }
        }
        executions.push(e);
    }
    let h = synthetic(all);
    for e in &executions {
        let m = model::execution(
            e,
            &h,
            &workload(),
            DerivationLimits {
                instructions: 100000,
                operations: 100000,
            },
        )
        .unwrap();
        assert!(m.unmapped.is_empty(), "{:?}", m.unmapped);
        assert!(
            m.operations
                .iter()
                .any(|o| o.name.contains("Write { space: Device"))
        );
        assert!(m.lower_bound().unwrap() > 0);
    }
}
#[test]
fn automatic_choices_cover_typed_decomposition_and_storage_without_source_cost_callbacks() {
    use seismic_compiler::tuner::Preparation;
    use seismic_metal::tuning::{self, Capacities, Form};
    let source = source();
    let capacities = Capacities {
        max_threads_per_threadgroup: 64,
        max_threadgroup_bytes: 4096,
    };
    let mut pending = vec![vec![]];
    let mut executions = Vec::new();
    let mut keys = Vec::new();
    let mut visits = 0;
    while let Some(path) = pending.pop() {
        visits += 1;
        assert!(visits < 200);
        match tuning::expand(&source, &Form::Automatic, &capacities, &path).unwrap() {
            Preparation::Choice { alternatives, .. } => {
                for i in 0..alternatives.len() {
                    let mut p = path.clone();
                    p.push(i);
                    pending.push(p);
                }
            }
            Preparation::Execution(e) => {
                let r = model::requirements(&e).unwrap();
                assert!(r.unmapped.is_empty(), "{:?}", r.unmapped);
                for p in r.primitives {
                    if !keys.contains(&p) {
                        keys.push(p);
                    }
                }
                executions.push(e);
            }
            Preparation::Infeasible(_) => {}
        }
    }
    assert!(executions.len() >= 12, "{}", executions.len());
    let h = synthetic(keys);
    for e in executions {
        let m = model::execution(
            &e,
            &h,
            &workload(),
            DerivationLimits {
                instructions: 100000,
                operations: 100000,
            },
        )
        .unwrap();
        assert!(m.unmapped.is_empty(), "{:?}", m.unmapped);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a Metal device"]
fn packed_five_and_six_bit_planes_decode_without_expansion() {
    use seismic_lang::{interp::TensorData, repr, types::DType};
    let device = seismic_metal::runtime::Device::open().unwrap();
    for name in ["q4k", "q5k", "q6k"] {
        let representation = repr::lookup(name).unwrap();
        let count = 512usize;
        let mut planes = Vec::new();
        for plane in representation.planes() {
            let mut bytes = vec![0; plane.bytes(count as u64).unwrap() as usize];
            match plane.encoding {
                repr::PlaneEncoding::Packed { bits, .. } => {
                    for i in 0..plane.entries(count as u64).unwrap() as usize {
                        repr::write_packed(
                            &mut bytes,
                            i,
                            bits,
                            (i as u32).wrapping_mul(37).wrapping_add(11),
                        );
                    }
                }
                repr::PlaneEncoding::Dense(dtype) => {
                    for i in 0..plane.entries(count as u64).unwrap() as usize {
                        let value = if i % 2 == 0 { 0.125f32 } else { -0.25 };
                        let at = i * dtype.bytes() as usize;
                        match dtype {
                            DType::F16 => bytes[at..at + 2].copy_from_slice(
                                &seismic_lang::numeric::f16_bits(value).to_le_bytes(),
                            ),
                            DType::F32 => bytes[at..at + 4].copy_from_slice(&value.to_le_bytes()),
                            _ => panic!("unexpected factor"),
                        }
                    }
                }
            }
            planes.push(bytes);
        }
        let input = TensorData::Packed {
            repr: representation,
            shape: vec![2, 256],
            planes,
        };
        let text = format!(
            "fn evaluate(x: tensor[2,256] {name}, out: tensor[2,256] f32):\n  for row in parallel:\n    a = load(x[row])\n    y = tile[256] f32\n    for i in owned(y): y[i] = a[i]\n    store(y,out[row])\n"
        );
        let p = compile(
            &[SourceFile {
                path: "packed.seismic.portable".into(),
                text,
                scope: Scope::Portable,
            }],
            &[],
        )
        .unwrap();
        let f = seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap();
        for loads in [
            seismic_realization::LoadStrategy::BorrowProvenReadOnly,
            seismic_realization::LoadStrategy::Materialize,
        ] {
            let selected = execution::prepare(
                &f,
                Config {
                    loads,
                    ..Default::default()
                },
            )
            .unwrap();
            let r = model::requirements(&selected).unwrap();
            assert!(r.unmapped.is_empty(), "{name}: {:?}", r.unmapped);
            let kernel = device
                .compile(seismic_metal::msl::emit_execution(&selected).unwrap())
                .unwrap();
            let mut buffers = input
                .device_bytes()
                .iter()
                .map(|b| device.buffer_from(b).unwrap())
                .collect::<Vec<_>>();
            buffers.push(device.buffer(count * 4).unwrap());
            device
                .run(&kernel, &buffers.iter().collect::<Vec<_>>(), &[], 1)
                .unwrap();
            let bytes = buffers.last().unwrap().read(count * 4);
            for (i, bytes) in bytes.chunks_exact(4).enumerate() {
                let actual = f32::from_le_bytes(bytes.try_into().unwrap());
                assert_eq!(
                    actual,
                    input.get(i) as f32,
                    "{name}, {loads:?}, element {i}"
                );
            }
        }
    }
}

fn reused(placement: seismic_realization::dispatch::TilePlacement) -> execution::Execution {
    let p=compile(&[SourceFile{path:"lifetime.seismic.portable".into(),scope:Scope::Portable,text:"fn evaluate(out: tensor[4,65] f32):\n  for t in range(0,2):\n    a = tile[65] f32\n    for i in owned(a): a[i] = f32(t) + 1.0\n    store(a,out[t*2])\n    b = tile[65] f32\n    for i in owned(b): b[i] = f32(t) + 11.0\n    store(b,out[t*2+1])\n".into()}],&[]).unwrap();
    let f = seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap();
    execution::prepare_with_allocation_choices(
        &f,
        Config::default(),
        &mut |_, _| Ok(seismic_lang::ir::LoadMode::Materialize),
        &mut |_| Ok(placement.clone()),
        &mut |r| Ok(r.diagnostic()),
        &mut |a| Ok(a.alternatives[0]),
    )
    .unwrap()
}
#[test]
fn allocation_choice_reuses_only_completed_values_and_accounts_actual_backing() {
    use seismic_realization::dispatch::TilePlacement;
    for placement in [
        TilePlacement::Replicated,
        TilePlacement::Distributed,
        TilePlacement::GroupShared,
    ] {
        let selected = reused(placement.clone());
        let memory = &selected.memory().launches()[0];
        assert_eq!(memory.arrays.len(), 2);
        assert_eq!(memory.slots.len(), 1);
        assert!(
            !memory.arrays[0]
                .lifetime
                .overlaps(memory.arrays[1].lifetime)
        );
        let reuse = memory
            .barriers
            .keys()
            .filter(|b| matches!(b.purpose, seismic_metal::memory::BarrierPurpose::Reuse(_)))
            .count();
        assert_eq!(
            reuse,
            if placement == TilePlacement::GroupShared {
                2
            } else {
                0
            }
        );
        let emitted = seismic_metal::msl::emit_execution(&selected).unwrap();
        assert_eq!(
            emitted.launches[0].declared_threadgroup_bytes,
            memory.shared_bytes_per_group
        );
        let requirements = model::requirements(&selected).unwrap();
        assert!(
            requirements.unmapped.is_empty(),
            "{:?}",
            requirements.unmapped
        );
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_lifetime_reuse_and_nondivisor_decompositions_preserve_outputs() {
    use seismic_realization::dispatch::TilePlacement;
    let device = seismic_metal::runtime::Device::open().unwrap();
    for placement in [
        TilePlacement::Replicated,
        TilePlacement::Distributed,
        TilePlacement::GroupShared,
    ] {
        let selected = reused(placement);
        let kernel = device
            .compile(seismic_metal::msl::emit_execution(&selected).unwrap())
            .unwrap();
        let output = device.buffer(4 * 65 * 4).unwrap();
        device.run(&kernel, &[&output], &[], 1).unwrap();
        for (i, b) in output.read(4 * 65 * 4).chunks_exact(4).enumerate() {
            let row = i / 65;
            let expected = [1.0, 11.0, 2.0, 12.0][row];
            assert_eq!(
                f32::from_le_bytes(b.try_into().unwrap()),
                expected,
                "reuse element {i}"
            );
        }
    }
    let p=compile(&[SourceFile{path:"tails.seismic.portable".into(),scope:Scope::Portable,text:"fn evaluate(x: tensor[5,7] f32, out: tensor[5,7] f32):\n  for row in parallel:\n    a = load(x[row])\n    b = tile[7] f32\n    for i in owned(b): b[i] = a[i] * 2.0\n    store(b,out[row])\n".into()}],&[]).unwrap();
    let f = seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap();
    let data = (0..35)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let input = device.buffer_from(&data).unwrap();
    for config in [
        Config {
            per_item: 2,
            ..Default::default()
        },
        Config {
            per_item: 3,
            ..Default::default()
        },
        Config {
            per_item: 4,
            ..Default::default()
        },
        Config {
            tile_piece: Some(2),
            ..Default::default()
        },
        Config {
            tile_piece: Some(3),
            ..Default::default()
        },
        Config {
            tile_piece: Some(6),
            ..Default::default()
        },
    ] {
        let selected = execution::prepare(&f, config.clone()).unwrap();
        let kernel = device
            .compile(seismic_metal::msl::emit_execution(&selected).unwrap())
            .unwrap();
        let output = device.buffer(data.len()).unwrap();
        device.run(&kernel, &[&input, &output], &[], 1).unwrap();
        for (i, b) in output.read(data.len()).chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_le_bytes(b.try_into().unwrap()),
                (i * 2) as f32,
                "{config:?}, element {i}"
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_joint_axis_mapping_partition_and_coupled_reduction() {
    use seismic_realization::{
        LoadStrategy,
        dispatch::{TilePlacement, WorkMapping},
    };
    let device = seismic_metal::runtime::Device::open().unwrap();
    let text = "fn evaluate(x: tensor[3,5,7] f32, out: tensor[3,5,7] f32):\n  for batch,row in parallel:\n    a = load(x[batch,row])\n    b = tile[7] f32\n    for i in owned(b):\n      product = a[i] * 2.0\n      b[i] = product + 1.0\n    store(b,out[batch,row])\n";
    let p = compile(
        &[SourceFile {
            path: "joint.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let f = seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap();
    let data = (0..105)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let input = device.buffer_from(&data).unwrap();
    for piece in [None, Some(3)] {
        let mappings = vec![
            match piece {
                None => WorkMapping::new(&[3, 5], &[2, 3]),
                Some(_) => WorkMapping::new(&[3, 5, 3], &[2, 3, 2]),
            }
            .unwrap(),
        ];
        let selected = execution::prepare_with_mappings(
            &f,
            Config {
                tile_piece: piece,
                loads: LoadStrategy::Materialize,
                ..Default::default()
            },
            Some(&mappings),
            &mut |_, _| Ok(seismic_lang::ir::LoadMode::Materialize),
            &mut |_| Ok(TilePlacement::Replicated),
            &mut |r| Ok(r.diagnostic()),
            &mut |a| Ok(a.alternatives[0]),
        )
        .unwrap();
        let kernel = device
            .compile(seismic_metal::msl::emit_execution(&selected).unwrap())
            .unwrap();
        let output = device.buffer(data.len()).unwrap();
        device.run(&kernel, &[&input, &output], &[], 1).unwrap();
        for (i, b) in output.read(data.len()).chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_le_bytes(b.try_into().unwrap()),
                (2 * i + 1) as f32,
                "joint mapping {piece:?}, {i}"
            );
        }
    }
    let text = "fn merge(a: tile[1] f32, b: tile[1] f32, c: tile[1] f32, d: tile[1] f32, x: tile[1] f32, y: tile[1] f32):\n  for i in owned(x): x[i] = a[i] + c[i]\n  for i in owned(y): y[i] = b[i] + d[i] + a[i] * c[i]\nfn evaluate(input_a: tensor[5,1] f32, input_b: tensor[5,1] f32, out: tensor[2] f32):\n  a = load(input_a)\n  b = load(input_b)\n  x = tile[1] f32\n  y = tile[1] f32\n  for i in owned(x): x[i] = 10.0\n  for i in owned(y): y[i] = 7.0\n  reduce((a,b),0,merge,into=(x,y),ordered=false)\n  z = tile[2] f32\n  for i in owned(z): z[i] = 0.0\n  z[0] = x[0]\n  z[1] = y[0]\n  store(z,out)\n";
    let p = compile(
        &[SourceFile {
            path: "coupled.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    for tree in [
        seismic_lang::reduction::structured::Tree::Ordered,
        seismic_lang::reduction::structured::Tree::Pairwise,
    ] {
        let f = seismic_lang::lower::lower_selected(
            &p,
            "evaluate",
            "metal",
            &Default::default(),
            &Default::default(),
            &Default::default(),
            &mut |d| {
                Ok(
                    if matches!(
                        d.kind,
                        seismic_lang::lowered_ir::DecisionKind::Reduction { .. }
                    ) {
                        seismic_lang::lowered_ir::Alternative::ReductionTree(tree)
                    } else {
                        d.alternatives.get(0).unwrap()
                    },
                )
            },
        )
        .unwrap();
        let selected = execution::prepare_with_allocation_choices(
            &f,
            Config::default(),
            &mut |_, s| {
                Ok(if s.can_borrow {
                    seismic_lang::ir::LoadMode::Borrow
                } else {
                    seismic_lang::ir::LoadMode::Materialize
                })
            },
            &mut |_| Ok(TilePlacement::Replicated),
            &mut |r| Ok(r.diagnostic()),
            &mut |a| Ok(a.alternatives[0]),
        )
        .unwrap();
        let requirements = model::requirements(&selected).unwrap();
        assert!(
            requirements.unmapped.is_empty(),
            "{tree:?}: {:?}",
            requirements.unmapped
        );
        let kernel = device
            .compile(seismic_metal::msl::emit_execution(&selected).unwrap())
            .unwrap();
        let a = device
            .buffer_from(
                &(1..=5)
                    .flat_map(|i| (i as f32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let b = device
            .buffer_from(
                &[2.0f32; 5]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let out = device.buffer(8).unwrap();
        device.run(&kernel, &[&a, &b, &out], &[], 1).unwrap();
        let actual = out
            .read(8)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, [25.0, 252.0], "{tree:?}");
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_integer_wrapping_and_partial_lane_rounds() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let p = compile(&[SourceFile {
        path: "integer_edges.seismic.portable".into(),
        scope: Scope::Portable,
        text: "fn evaluate(input: tensor[2] i32, out: tensor[6] i32, unsigned_out: tensor[1] u32):\n  a = load(input)\n  b = tile[6] i32\n  for i in owned(b): b[i] = 0\n  b[0] = a[0] + 1\n  b[1] = a[1] - 1\n  b[2] = a[1] * -1\n  b[3] = -a[1]\n  b[4] = i32(u32(-1))\n  b[5] = abs(a[1])\n  store(b,out)\n  u = tile[1] u32\n  for i in owned(u): u[i] = u32(-1)\n  store(u,unsigned_out)\n".into(),
    }], &[]).unwrap();
    let f = seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap();
    let selected = execution::prepare(&f, Config::default()).unwrap();
    let requirements = model::requirements(&selected).unwrap();
    assert!(
        requirements.unmapped.is_empty(),
        "{:?}",
        requirements.unmapped
    );
    let kernel = device
        .compile(seismic_metal::msl::emit_execution(&selected).unwrap())
        .unwrap();
    let input = device
        .buffer_from(
            &[i32::MAX, i32::MIN]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let output = device.buffer(24).unwrap();
    let unsigned = device.buffer(4).unwrap();
    device
        .run(&kernel, &[&input, &output, &unsigned], &[], 1)
        .unwrap();
    let actual = output
        .read(24)
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        [i32::MIN, i32::MAX, i32::MIN, i32::MIN, -1, i32::MIN]
    );
    assert_eq!(
        u32::from_le_bytes(unsigned.read(4).try_into().unwrap()),
        u32::MAX
    );

    let p = compile(&[
        SourceFile {path: "lane_tail.seismic.portable".into(), scope: Scope::Portable,
            text: "construct transfer(input: tensor[67] f32, out: tile[67] f32):\n  for i in owned(out): out[i] = input[i] * 2.0\nfn evaluate(input: tensor[67] f32, out: tensor[67] f32):\n  tile_out = tile[67] f32\n  for i in owned(tile_out): tile_out[i] = 0.0\n  transfer(input,tile_out)\n  store(tile_out,out)\n".into()},
        SourceFile {path: "lane_tail.seismic.metal".into(), scope: Scope::Backend("metal".into()),
            text: "lower transfer(input: tensor[67] f32, out: tile[67] f32):\n  for i in owned(out): out[i] = 0.0\n  for i in lanes(67,2): out[i] = input[i] * 2.0\n".into()},
    ], &[]).unwrap();
    let f = seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap();
    let selected = execution::prepare(&f, Config::default()).unwrap();
    let requirements = model::requirements(&selected).unwrap();
    assert!(
        requirements.unmapped.is_empty(),
        "{:?}",
        requirements.unmapped
    );
    let kernel = device
        .compile(seismic_metal::msl::emit_execution(&selected).unwrap())
        .unwrap();
    let input = device
        .buffer_from(
            &(0..67)
                .flat_map(|i| (i as f32).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let output = device.buffer(67 * 4).unwrap();
    device.run(&kernel, &[&input, &output], &[], 1).unwrap();
    for (i, b) in output.read(67 * 4).chunks_exact(4).enumerate() {
        assert_eq!(
            f32::from_le_bytes(b.try_into().unwrap()),
            (2 * i) as f32,
            "lane tail {i}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_selected_fold_participants_preserve_fma_segments_and_bound_storage() {
    use seismic_lang::{
        lowered_ir::{Alternative, DecisionKind},
        reduction::structured::Tree,
    };
    use seismic_metal::execution::{FoldOwnership, prepare_with_participants};
    let text = r#"
fn merge[M](left:tile[M] f32,right:tile[M] f32,out:tile[M] f32):
  for i in owned(out): out[i] = left[i] + right[i]
fn accumulate[M](state:tile[M] f32,a:tile[M] f32,b:tile[M] f32,out:tile[M] f32):
  for i in owned(out): out[i] = fma(a[i],b[i],state[i])
fn evaluate(a:tensor[67,3] f32,b:tensor[67,3] f32,out:tensor[3] f32):
  ta = load(a)
  tb = load(b)
  state = tile[3] f32
  zero = tile[3] f32
  for i in owned(state): state[i] = 10.0
  for i in owned(zero): zero[i] = 0.0
  reduce((ta,tb),0,merge,into=(state,),step=accumulate,identity=(zero,),ordered=false)
  store(state,out)
"#;
    let p = compile(
        &[SourceFile {
            path: "participant_fold.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let device = seismic_metal::runtime::Device::open().unwrap();
    let a = device
        .buffer_from(
            &(0..201)
                .flat_map(|i| (i as f32).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let b = device
        .buffer_from(
            &[2.0f32; 201]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for tree in [Tree::Pairwise, Tree::Explicit] {
        for segment in [3, 7, 17, 67] {
            let f = seismic_lang::lower::lower_selected(
                &p,
                "evaluate",
                "metal",
                &Default::default(),
                &Default::default(),
                &Default::default(),
                &mut |d| {
                    Ok(match d.kind {
                        DecisionKind::Reduction { .. } => Alternative::ReductionTree(tree),
                        DecisionKind::ReductionSegments { .. } => {
                            Alternative::ReductionSegment(segment)
                        }
                        _ => d.alternatives.get(0).unwrap(),
                    })
                },
            )
            .unwrap();
            let first = seismic_metal::choices::expand(&f, Config::default(), &[]).unwrap();
            assert!(matches!(
                first,
                seismic_metal::choices::Expansion::Choice(seismic_metal::choices::Domain {
                    decision: seismic_metal::choices::Decision::Fold(_)
                })
            ));
            let selected = prepare_with_participants(
                &f,
                Config::default(),
                None,
                &mut |_| Ok(FoldOwnership::Participants),
                &mut |_, site| {
                    Ok(if site.can_borrow {
                        seismic_lang::ir::LoadMode::Borrow
                    } else {
                        seismic_lang::ir::LoadMode::Materialize
                    })
                },
                &mut |s| Ok(s.diagnostic()),
                &mut |r| Ok(r.diagnostic()),
                &mut |a| Ok(a.alternatives[0]),
            )
            .unwrap();
            let requirements = model::requirements(&selected).unwrap();
            assert!(
                requirements.unmapped.is_empty(),
                "{tree:?}/{segment}: {:?}",
                requirements.unmapped
            );
            assert!(
                selected.memory().launches()[0]
                    .arrays
                    .iter()
                    .all(|a| a.declaration.capacity <= 3),
                "logical input/leaves must stay unmaterialized"
            );
            let emitted = seismic_metal::msl::emit_execution(&selected).unwrap();
            assert!(emitted.source.contains("simd_shuffle"));
            let hardware = synthetic(requirements.primitives);
            let workload = ScalarWorkload {
                identity: "participant fold".into(),
                allocations: [(1, 804), (2, 804), (3, 12)]
                    .into_iter()
                    .map(|(id, bytes)| Allocation {
                        id,
                        bytes,
                        alignment: 4,
                        known_bytes: Default::default(),
                    })
                    .collect(),
                buffers: [(1, 804), (2, 804), (3, 12)]
                    .into_iter()
                    .map(|(allocation, bytes)| BufferBinding {
                        allocation,
                        offset: 0,
                        bytes,
                    })
                    .collect(),
                scalars: vec![],
            };
            let modeled = model::execution(
                &selected,
                &hardware,
                &workload,
                DerivationLimits {
                    instructions: 2_000_000,
                    operations: 2_000_000,
                },
            )
            .unwrap();
            assert!(
                modeled.unmapped.is_empty(),
                "{tree:?}/{segment}: {:?}",
                modeled.unmapped
            );
            let kernel = device.compile(emitted).unwrap();
            let out = device.buffer(12).unwrap();
            device.run(&kernel, &[&a, &b, &out], &[], 1).unwrap();
            let actual = out
                .read(12)
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let expected = (0..3)
                .map(|column| {
                    10.0 + (0..67)
                        .map(|row| 2.0 * ((row * 3 + column) as f32))
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{tree:?}/{segment}");
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn native_matrix_intrinsics_use_the_same_typed_terminal_contract_as_accounting() {
    let p = compile(
        &[
            SourceFile {
                path: "matrix.seismic.portable".into(),
                scope: Scope::Portable,
                text: r#"
construct product(a:tile[8,8] f32,b:tile[8,8] f32,c:tile[8,8] f32):
  for i,j in owned(c):
    for k in range(8): c[i,j] = fma(a[i,k],b[k,j],c[i,j])
fn evaluate(x:tensor[8,8] f32,y:tensor[8,8] f32,out:tensor[8,8] f32):
  a=load(x)
  b=load(y)
  c=tile[8,8] f32
  for i,j in owned(c): c[i,j]=1.0
  product(a,b,c)
  store(c,out)
"#
                .into(),
            },
            SourceFile {
                path: "matrix.seismic.metal".into(),
                scope: Scope::Backend("metal".into()),
                text: r#"
lower product(a:tile[8,8] f32,b:tile[8,8] f32,c:tile[8,8] f32):
  af=simdgroup_matrix(f32)
  bf=simdgroup_matrix(f32)
  cf=simdgroup_matrix(f32)
  simdgroup_load(af,a,0,0)
  simdgroup_load(bf,b,0,0)
  simdgroup_load(cf,c,0,0)
  simdgroup_multiply_accumulate(cf,af,bf,cf)
  simdgroup_store(cf,c,0,0)
"#
                .into(),
            },
        ],
        &[],
    )
    .unwrap();
    let function =
        seismic_lang::lower::lower(&p, "evaluate", "metal", &Default::default()).unwrap();
    let device = seismic_metal::runtime::Device::open().unwrap();
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        let execution = execution::prepare(
            &function,
            Config {
                loads,
                ..Default::default()
            },
        )
        .unwrap();
        let requirements = model::requirements(&execution).unwrap();
        assert!(
            requirements.unmapped.is_empty(),
            "{:?}",
            requirements.unmapped
        );
        assert!(
            requirements
                .primitives
                .iter()
                .any(|p| matches!(p, Primitive::MatrixMultiplyAccumulate { .. }))
        );
        let workload = ScalarWorkload {
            identity: "8x8 source matrix operation".into(),
            allocations: (1..=3)
                .map(|id| Allocation {
                    id,
                    bytes: 256,
                    alignment: 4,
                    known_bytes: Default::default(),
                })
                .collect(),
            buffers: (1..=3)
                .map(|allocation| BufferBinding {
                    allocation,
                    offset: 0,
                    bytes: 256,
                })
                .collect(),
            scalars: vec![],
        };
        let model = model::execution(
            &execution,
            &synthetic(requirements.primitives),
            &workload,
            DerivationLimits {
                instructions: 100000,
                operations: 100000,
            },
        )
        .unwrap();
        assert!(model.unmapped.is_empty(), "{:?}", model.unmapped);
        let emitted = seismic_metal::msl::emit_execution(&execution).unwrap();
        assert!(
            emitted
                .terminal
                .launches()
                .iter()
                .flatten()
                .any(|site| matches!(
                    site.statement,
                    seismic_metal::terminal::Statement::MatrixMultiplyAccumulate { .. }
                ))
        );
        let pipeline = device.compile(emitted).unwrap();
        let x = (0..64).map(|i| (i as f32 - 30.) / 8.).collect::<Vec<_>>();
        let y = (0..64)
            .map(|i| ((i * 3 % 13) as f32 - 6.) / 4.)
            .collect::<Vec<_>>();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let a = device.buffer_from(&bytes(&x)).unwrap();
        let b = device.buffer_from(&bytes(&y)).unwrap();
        let out = device.buffer_from(&[0; 256]).unwrap();
        device.run(&pipeline, &[&a, &b, &out], &[], 1).unwrap();
        for (i, chunk) in out.read(256).chunks_exact(4).enumerate() {
            let expected =
                (0..8).fold(1f32, |s, k| x[(i / 8) * 8 + k].mul_add(y[k * 8 + i % 8], s));
            assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), expected);
        }
    }
}

#[test]
fn construction_limits_are_typed_separately_from_analysis_errors() {
    use seismic_accounting::workload::{DerivationError, DerivationLimit};
    let execution = execution::prepare_storage_selected(&source(), Config::default(), &mut |_| {
        Ok(seismic_realization::dispatch::TilePlacement::Replicated)
    })
    .unwrap();
    let hardware = synthetic(model::requirements(&execution).unwrap().primitives);
    for (limits, expected) in [
        (
            DerivationLimits {
                instructions: 1,
                operations: 100_000,
            },
            DerivationLimit::Instructions(1),
        ),
        (
            DerivationLimits {
                instructions: 100_000,
                operations: 1,
            },
            DerivationLimit::Operations(1),
        ),
    ] {
        let Err(error) = model::execution(&execution, &hardware, &workload(), limits) else {
            panic!("construction unexpectedly fit the limit");
        };
        assert_eq!(error, DerivationError::Exhausted(expected));
    }
}
