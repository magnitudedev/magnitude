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
    ScalarWorkload { integer_domains: Vec::new(),
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
fn structured_launch_derivation_matches_flat_exact_oracle() {
    let execution = execution::prepare(&source(), Config::default()).unwrap();
    let mut hardware = synthetic(model::requirements(&execution).unwrap().primitives);
    hardware.resources[0].capacity = 1;
    for timing in &mut hardware.timings { for service in &mut timing.services { service.units = Units::PerSubgroup(1); } }
    let limits = DerivationLimits { instructions: 100_000, operations: 100_000 };
    let flat = model::execution(&execution, &hardware, &workload(), limits).unwrap();
    let structured = model::structured_execution(&execution, &hardware, &workload(), limits).unwrap();
    assert!(structured.unmapped.is_empty(), "{:?}", structured.unmapped);
    let reference = flat.solve(100_000).unwrap();
    assert!(reference.is_optimal());
    let witness = structured.compact_witness().unwrap().unwrap();
    assert!(witness.is_optimal());
    assert_eq!(witness.completion(), reference.schedule().completion);
    let expanded = structured.expand(100_000).unwrap().solve(100_000).unwrap();
    assert!(expanded.is_optimal());
    assert_eq!(expanded.schedule().completion, witness.completion());
}
#[test]
fn invariant_dispatch_groups_are_derived_without_enumeration() {
    for count in [3, 8, 11, 1_000_000_003] {
        let program = compile(&[SourceFile { path: "groups.seismic.portable".into(), scope: Scope::Portable,
            text: "fn write[N](out: tensor[N] f32):\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = 3.0\n    store(y, out[row:row+1])\n".into() }], &[]).unwrap();
        let lowered = seismic_lang::lower::lower(&program, "write", "metal", &std::collections::HashMap::from([("N".into(), count)])).unwrap();
        let execution = execution::prepare(&lowered, Config::default()).unwrap();
        let hardware = synthetic(model::requirements(&execution).unwrap().primitives);
        let mut workload = workload();
        workload.allocations[0].bytes = count as u64 * 4;
        workload.buffers[0].bytes = count as u64 * 4;
        let limits = DerivationLimits { instructions: 1000, operations: 1000 };
        let structured = model::structured_execution(&execution, &hardware, &workload, limits).unwrap();
        assert!(structured.unmapped.is_empty(), "{:?}", structured.unmapped);
        let relaxed = model::invocation_relaxation(&execution, &workload, limits).unwrap();
        assert!(relaxed.is_complete(), "symbolic dispatch count: {:?}", relaxed.unmapped);
        assert!(relaxed.visits < 1000);
        assert_eq!(relaxed.operations.iter().filter(|term| matches!(term.primitive, Primitive::Launch))
            .map(|term| term.instances).sum::<u64>(), 1);
        assert_eq!(relaxed.operations.iter().filter(|term| matches!(term.primitive, Primitive::Group))
            .map(|term| term.instances).sum::<u64>(), execution.phases()[0].dispatch.groups);
        assert_eq!(relaxed.operations.iter().filter(|term| matches!(term.primitive, Primitive::Write { space: seismic_metal::terminal::Space::Device, .. }))
            .map(|term| term.instances * term.lanes).sum::<u64>(), count as u64);
        let grouped = seismic_metal::family::GroupFamily::derive(execution.clone()).unwrap().select(3).unwrap();
        let grouped_count = model::invocation_relaxation(&grouped, &workload, limits).unwrap();
        assert!(grouped_count.is_complete(), "partial group count: {:?}", grouped_count.unmapped);
        assert_eq!(grouped_count.operations.iter().filter(|term| matches!(term.primitive, Primitive::Group))
            .map(|term| term.instances).sum::<u64>(), (count as u64).div_ceil(3));
        assert_eq!(grouped_count.operations.iter().filter(|term| matches!(term.primitive, Primitive::Write { space: seismic_metal::terminal::Space::Device, .. }))
            .map(|term| term.instances * term.lanes).sum::<u64>(), count as u64);
        if count < 100 {
            let flat = model::execution(&execution, &hardware, &workload, DerivationLimits { instructions: 1000, operations: 1000 }).unwrap();
            let expanded = structured.expand(1000).unwrap();
            let counts = |model: &seismic_accounting::schedule::Model| {
                let mut counts = std::collections::BTreeMap::new();
                for op in model.operations.iter().filter(|op| op.latency > 0) {
                    *counts.entry(format!("{}:{:?}", op.latency, op.reservations)).or_insert(0) += 1;
                }
                counts
            };
            assert_eq!(counts(&expanded), counts(&flat));
            assert_eq!(expanded.lower_bound().unwrap(), flat.lower_bound().unwrap());
            structured.compact_witness().unwrap().unwrap().expand(1000).unwrap();
            // Both symbolic geometry and any necessary concrete refinement
            // must preserve the transaction model.
            let mut transactions = hardware.clone();
            for timing in &mut transactions.timings {
                if matches!(timing.primitive, Primitive::Write { space: seismic_metal::terminal::Space::Device, .. }) {
                    for service in &mut timing.services { service.units = Units::PerTransaction { bytes: 16, units: 1 }; }
                }
            }
            workload.allocations[0].alignment = 16;
            let concrete = model::execution(&execution, &transactions, &workload, limits).unwrap();
            let retry = model::structured_execution(&execution, &transactions, &workload, limits).unwrap();
            assert!(retry.unmapped.is_empty(), "count {count}: {:?}; flat {:?}", retry.unmapped, concrete.unmapped);
            assert_eq!(counts(&retry.expand(1000).unwrap()), counts(&concrete));
            if count == 3 {
                let reference = flat.solve(100_000).unwrap();
                let candidate = expanded.solve(100_000).unwrap();
                assert!(reference.is_optimal() && candidate.is_optimal());
                assert_eq!(candidate.schedule().completion, reference.schedule().completion);
            }
        } else {
            let mut transactions = hardware.clone();
            for timing in &mut transactions.timings {
                if matches!(timing.primitive, Primitive::Write { space: seismic_metal::terminal::Space::Device, .. }) {
                    for service in &mut timing.services { service.units = Units::PerTransaction { bytes: 4, units: 1 }; }
                }
            }
            workload.allocations[0].alignment = 16;
            let compact = model::structured_execution(&execution, &transactions, &workload, limits).unwrap();
            assert!(compact.unmapped.is_empty(), "symbolic transactions: {:?}", compact.unmapped);
            assert!(matches!(model::execution(&execution, &hardware, &workload, limits), Err(seismic_accounting::workload::DerivationError::Exhausted(_))));
        }
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
    participant_fold_correspondence(false);
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_seed_root_fold_preserves_selected_tree_and_bound_storage() {
    participant_fold_correspondence(true);
}
#[cfg(target_os = "macos")]
fn participant_fold_correspondence(root_seed: bool) {
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
    let n = if root_seed { 64 } else { 67 };
    let text = text.replace("67", &n.to_string());
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
    let values = (0..n * 3)
        .map(|i| match (i / 3) % 4 {
            0 => 1.0e20_f32,
            1 => [1.0, 0.25, 3.0][i % 3],
            2 => -1.0e20_f32,
            _ => [-0.5, 4.0, 0.125][i % 3],
        })
        .collect::<Vec<_>>();
    let a = device
        .buffer_from(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let b = device
        .buffer_from(
            &vec![2.0f32; n * 3]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let ownerships = if root_seed {
        vec![FoldOwnership::ParticipantsRootSeed, FoldOwnership::ParticipantsWavefrontRootSeed]
    } else {
        vec![FoldOwnership::Participants, FoldOwnership::ParticipantsInsertSeed,
            FoldOwnership::ParticipantsWavefront, FoldOwnership::ParticipantsWavefrontInsertSeed]
    };
    let trees = if root_seed { vec![Tree::SeedThenPairwise] } else { vec![Tree::Pairwise, Tree::Explicit] };
    for ownership in ownerships {
        for &tree in &trees {
            if tree == Tree::Explicit && matches!(ownership, FoldOwnership::ParticipantsWavefront | FoldOwnership::ParticipantsWavefrontInsertSeed) { continue; }
            for segment in [1, 2, 3, 7, 17, n as i64] {
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
                let selected = prepare_with_participants(
                    &f,
                    Config::default(),
                    None,
                    &mut |_| Ok(ownership),
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
                        .all(|a| a.declaration.capacity
                            <= 3 * (((n as u64).div_ceil(segment as u64) + u64::from(!root_seed)).div_ceil(32))),
                    "private state must scale with leaves per lane, not total leaves"
                );
                let emitted = seismic_metal::msl::emit_execution(&selected).unwrap();
                assert!(emitted.source.contains("simd_shuffle"));
                let hardware = synthetic(requirements.primitives);
                let workload = ScalarWorkload { integer_domains: Vec::new(),
                    identity: "participant fold".into(),
                    allocations: [(1, (n * 12) as u64), (2, (n * 12) as u64), (3, 12)]
                        .into_iter()
                        .map(|(id, bytes)| Allocation {
                            id,
                            bytes,
                            alignment: 4,
                            known_bytes: Default::default(),
                        })
                        .collect(),
                    buffers: [(1, (n * 12) as u64), (2, (n * 12) as u64), (3, 12)]
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
                // Evaluate the same selected tree independently. Cancellation makes
                // changing a child order or adding padded identity leaves observable.
                let reference = seismic_lang::program::Program {
                    functions: vec![seismic_lang::ir::Function {
                        name: f.name.clone(),
                        is_construct: false,
                        shape_params: vec![],
                        elem_params: vec![],
                        params: f.params.clone(),
                        index_params: f.index_params.clone(),
                        vars: f.vars.clone(),
                        body: f.body.clone(),
                    }],
                    lowerings: vec![],
                    signatures: Default::default(),
                };
                let mut interpreter = seismic_lang::interp::Interpreter::new(&reference);
                let x = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    seismic_lang::types::DType::F32,
                    vec![n, 3],
                    values.iter().map(|&v| f64::from(v)).collect(),
                ));
                let y = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    seismic_lang::types::DType::F32,
                    vec![n, 3],
                    vec![2.0; n * 3],
                ));
                let z = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    seismic_lang::types::DType::F32,
                    vec![3],
                    vec![0.0; 3],
                ));
                interpreter
                    .run(
                        "evaluate",
                        &[
                            seismic_lang::interp::Arg::Tensor(x),
                            seismic_lang::interp::Arg::Tensor(y),
                            seismic_lang::interp::Arg::Tensor(z),
                        ],
                        &Default::default(),
                    )
                    .unwrap();
                let expected = (0..3)
                    .map(|i| interpreter.tensors[z].get(i) as f32)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{ownership:?}/{tree:?}/{segment}");
            }
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
        let workload = ScalarWorkload { integer_domains: Vec::new(),
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

#[test]
fn compact_terminal_counts_match_schedule_operations_and_preserve_incompleteness() {
    let f = source();
    for placement in [
        seismic_realization::dispatch::TilePlacement::Replicated,
        seismic_realization::dispatch::TilePlacement::Distributed,
        seismic_realization::dispatch::TilePlacement::GroupShared,
    ] {
        let e = execution::prepare_storage_selected(&f, Config::default(), &mut |_| Ok(placement.clone())).unwrap();
        let mut h = synthetic(model::requirements(&e).unwrap().primitives);
        h.resources.push(Resource { name: "subgroup issue".into(), capacity: 4, unit: CapacityUnit::Slots });
        for timing in &mut h.timings {
            timing.services.push(Service { resource: 1, offset: 0, duration: 1, units: Units::PerSubgroup(1) });
        }
        let limits = DerivationLimits { instructions: 100_000, operations: 100_000 };
        let schedule = model::execution(&e, &h, &workload(), limits).unwrap();
        let counts = model::invocation_account(&e, &workload(), limits).unwrap();
        assert!(counts.is_complete(), "{:?}", counts.unmapped);
        let instances: u64 = counts.operations.iter().map(|t| t.instances).sum();
        let lanes: u64 = counts.operations.iter().map(|t| t.instances * t.lanes).sum();
        let service = |resource| schedule.operations.iter().flat_map(|op| &op.reservations)
            .filter(|r| r.resource == resource).map(|r| r.units * r.duration).sum::<u64>();
        assert_eq!(instances, service(1));
        assert_eq!(lanes, service(0));
        assert_eq!(counts.visits, instances);
        assert!(counts.operations.len() < schedule.operations.len());
        let demand = counts.demand(&h).unwrap().unwrap();
        assert_eq!(demand.lower_bound().unwrap(), lanes.div_ceil(1024).max(instances.div_ceil(4)).max(1));
        assert!(demand.lower_bound().unwrap() <= schedule.lower_bound().unwrap());
        let partial = model::invocation_account(&e, &workload(), DerivationLimits { instructions: 2, operations: 100_000 }).unwrap();
        assert_eq!(partial.exhausted, Some(seismic_accounting::workload::DerivationLimit::Instructions(2)));
        assert!(!partial.is_complete());
        assert!(partial.demand(&h).unwrap().is_none());
        let bounded = model::invocation_account(&e, &workload(), DerivationLimits { instructions: 100_000, operations: 1 }).unwrap();
        assert_eq!(bounded.exhausted, Some(seismic_accounting::workload::DerivationLimit::Operations(1)));
        assert_eq!(bounded.operations.len(), 1);
        h.timings.clear();
        assert!(counts.demand(&h).unwrap().is_none());
    }
}

#[test]
fn indirect_gather_domains_preserve_guard_and_transaction_geometry() {
    use seismic_accounting::workload::{IntegerDomain, IntegerInput, IntegerRange};
    let program = compile(&[SourceFile { path: "indirect-domain.seismic.portable".into(), scope: Scope::Portable,
        text: "fn gather(table: tensor[16, 8] f32, token: tensor[1] i32, out: tensor[8] f32):\n  x = load(table[token[0]])\n  store(x, out)\n".into() }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "gather", "metal", &Default::default()).unwrap();
    let execution = execution::prepare(&lowered, Config::default()).unwrap();
    let mut hardware = synthetic(model::requirements(&execution).unwrap().primitives);
    for timing in &mut hardware.timings {
        if matches!(timing.primitive, Primitive::Read { space: seismic_metal::terminal::Space::Device, .. } | Primitive::Write { space: seismic_metal::terminal::Space::Device, .. }) {
            for service in &mut timing.services { service.units = Units::PerTransaction { bytes: 4, units: 1 }; }
        }
    }
    let sizes = [512, 4, 32];
    let workload = ScalarWorkload { identity: "varying indirect row".into(),
        allocations: sizes.iter().enumerate().map(|(id, &bytes)| Allocation { id: id as u64, bytes, alignment: 256, known_bytes: Default::default() }).collect(),
        buffers: sizes.iter().enumerate().map(|(id, &bytes)| BufferBinding { allocation: id as u64, offset: 0, bytes }).collect(),
        scalars: vec![], integer_domains: vec![IntegerDomain { input: IntegerInput::Allocation { allocation: 1, offset: 0 }, bytes: 4, signed: true,
            range: IntegerRange { min: 0, max: 15, stride: 1 } }] };
    let limits = DerivationLimits { instructions: 100_000, operations: 10_000 };
    let domain = model::structured_execution(&execution, &hardware, &workload, limits).unwrap();
    assert!(domain.unmapped.is_empty(), "{:?}", domain.unmapped);
    let normalized = |model: &seismic_accounting::schedule::Model| {
        model.operations.iter().filter(|op| op.latency > 0).map(|op| (op.latency, op.reservations.clone())).collect::<Vec<_>>()
    };
    for token in [0i32, 3, 15] {
        let mut exact = workload.clone();
        exact.integer_domains.clear();
        exact.allocations[1].known_bytes = token.to_le_bytes().into_iter().enumerate().map(|(i, byte)| (i as u64, byte)).collect();
        let exact = model::structured_execution(&execution, &hardware, &exact, limits).unwrap();
        assert!(exact.unmapped.is_empty(), "{:?}", exact.unmapped);
        assert_eq!(normalized(&domain.expand(10_000).unwrap()), normalized(&exact.expand(10_000).unwrap()));
    }
    let mut invalid = workload;
    invalid.integer_domains[0].range.max = 16;
    assert!(!model::structured_execution(&execution, &hardware, &invalid, limits).unwrap().unmapped.is_empty());
}
