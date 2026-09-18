use seismic_accounting::selection::{Choices, IntegerRange};
use seismic_compiler::tuner::Preparation;
use seismic_cuda::{
    DeviceInfo,
    tuning::{self, BlockChoice, DispatchChoice, FoldChoice, FoldImplementation},
};
use seismic_lang::{
    Scope,
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind, LoweredIr},
    normalize::loads,
    program::{SourceFile, compile},
    reduction::structured::Tree,
};
use seismic_realization::Dispatch;
use std::collections::HashMap;

const SOURCE: &str = r#"
fn add[S](left:tile[S] f32,right:tile[S] f32,out:tile[S] f32):
  for i in owned(out): out[i] = left[i] + right[i]
fn accumulate[S](state:tile[S] f32,a:tile[S] f32,b:tile[S] f32,out:tile[S] f32):
  for i in owned(out): out[i] = fma(a[i],b[i],state[i])
fn fold[N](a:tensor[3,N,2] f32,b:tensor[3,N,2] f32,out:tensor[3,2] f32):
  for row in parallel:
    ta = load(a[row])
    tb = load(b[row])
    state = tile[2] f32
    zero = tile[2] f32
    for i in owned(state): state[i] = 7.0 + f32(i)
    for i in owned(zero): zero[i] = 0.0
    reduce((ta,tb),0,add,into=(state,),step=accumulate,identity=(zero,),ordered=false)
    store(state,out[row])
"#;
fn lowered(n: i64, segment: i64, tree: Tree) -> LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "fold.seismic.portable".into(),
            scope: Scope::Portable,
            text: SOURCE.into(),
        }],
        &["cuda".into()],
    )
    .unwrap();
    lower_selected(
        &program,
        "fold",
        "cuda",
        &HashMap::from([("N".into(), n)]),
        &HashMap::new(),
        &Options::default(),
        &mut |d| {
            Ok(match d.kind {
                DecisionKind::Reduction { .. } => Alternative::ReductionTree(tree),
                DecisionKind::ReductionSegments { .. } => Alternative::ReductionSegment(segment),
                _ => d.alternatives.get(0).unwrap(),
            })
        },
    )
    .unwrap()
}
fn device() -> DeviceInfo {
    DeviceInfo {
        name: "synthetic full-warp device".into(),
        compute_capability: (8, 0),
        driver_version: 0,
        max_threads_per_block: 64,
        max_grid_x: 8,
        warp_size: 32,
        multiprocessors: 1,
        global_memory_bytes: 4096,
        l2_cache_bytes: 0,
        max_threads_per_multiprocessor: 128,
        registers_32bit_per_multiprocessor: 4096,
        shared_bytes_per_multiprocessor: 4096,
    }
}
fn prepare(f: &LoweredIr, device: &DeviceInfo) -> Vec<seismic_cuda::execution::Execution> {
    prepare_ownership(f, device, FoldImplementation::Subgroup)
}
fn prepare_ownership(
    f: &LoweredIr,
    device: &DeviceInfo,
    ownership: FoldImplementation,
) -> Vec<seismic_cuda::execution::Execution> {
    let mut path = vec![];
    loop {
        match tuning::prepare(f, device, &path).unwrap() {
            Preparation::Execution(e) => return e,
            Preparation::Infeasible(e) => panic!("{e:?}"),
            Preparation::Unresolved(e) => panic!("{e}"),
            Preparation::Choice { alternatives, .. } => {
                let i = if let Some(c) = alternatives.owner::<loads::Choice>() {
                    c.modes()
                        .iter()
                        .position(|&m| m == seismic_lang::ir::LoadMode::Borrow)
                        .unwrap()
                } else if let Some(c) = alternatives.owner::<FoldChoice>() {
                    (0..c.len()).find(|&i| c.get(i) == Some(ownership)).unwrap()
                } else if let Some(c) = alternatives.owner::<DispatchChoice>() {
                    (0..c.len())
                        .find(|&i| c.get(i) == Some(Dispatch::ParallelRoot))
                        .unwrap()
                } else if let Some(c) = alternatives.owner::<IntegerRange<BlockChoice>>() {
                    c.index(2).unwrap()
                } else {
                    panic!("unexpected CUDA choice")
                };
                path.push(i);
            }
        }
    }
}
#[test]
fn selected_fold_uses_adjacent_exchange_and_bounded_private_state() {
    let mut sizes = vec![];
    for (n, s) in [(35, 2), (1023, 33)] {
        let f = lowered(n, s, Tree::Pairwise);
        let e = prepare(&f, &device());
        assert_eq!(e[0].dispatch().lanes_per_item, 32);
        let ptx = seismic_cuda::ptx::print(e[0].target_plan());
        assert!(ptx.contains("shfl.sync.idx"));
        assert!(!ptx.contains("shfl.sync.bfly"));
        sizes.push(e[0].program().scratch_bytes);
    }
    assert_eq!(
        sizes[0], sizes[1],
        "logical input/leaves must not become replicated private arrays"
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn native_selected_fold_preserves_seed_segments_tree_and_tail() {
    participant_fold_correspondence(false);
}
#[test]
#[ignore = "requires CUDA hardware"]
fn native_seed_root_fold_preserves_seed_segments_tree_and_tail() {
    participant_fold_correspondence(true);
}
fn participant_fold_correspondence(root_seed: bool) {
    let device = seismic_cuda::Device::open(0).unwrap();
    let ownerships = if root_seed {
        vec![FoldImplementation::SubgroupRootSeed, FoldImplementation::SubgroupWavefrontRootSeed]
    } else {
        vec![FoldImplementation::Subgroup, FoldImplementation::SubgroupInsertSeed,
            FoldImplementation::SubgroupWavefront, FoldImplementation::SubgroupWavefrontInsertSeed]
    };
    let trees = if root_seed { vec![Tree::SeedThenPairwise] } else { vec![Tree::Pairwise, Tree::Explicit] };
    for ownership in ownerships {
        for (n, segment) in [
            (1, 1),
            (13, 1),
            (35, 2),
            (67, 4),
            (93, 3),
            (67, 1),
            (512, 16),
            (64, 1),
            (1057, 1),
        ] {
            for &tree in &trees {
                let wavefront = matches!(ownership, FoldImplementation::SubgroupWavefront | FoldImplementation::SubgroupWavefrontInsertSeed | FoldImplementation::SubgroupWavefrontRootSeed);
                if (tree == Tree::Explicit && wavefront) || (n == 1057 && !wavefront) { continue; }
                let f = lowered(n, segment, tree);
                let execution = prepare_ownership(&f, &device.info, ownership);
                let a = (0..3 * n * 2)
                    .map(|i| ((i % 19) as f32 - 9.0) / 8.0)
                    .collect::<Vec<_>>();
                let b = (0..3 * n * 2)
                    .map(|i| ((i % 11) as f32 - 5.0) / 16.0)
                    .collect::<Vec<_>>();
                // Evaluate the exact selected lowered source, including its seed/tree, with
                // independent interpreter arithmetic rather than another warp algorithm.
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
                    signatures: HashMap::new(),
                };
                let mut interpreter = seismic_lang::interp::Interpreter::new(&reference);
                let x = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    seismic_lang::types::DType::F32,
                    vec![3, n as usize, 2],
                    a.iter().map(|&x| x as f64).collect(),
                ));
                let y = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    seismic_lang::types::DType::F32,
                    vec![3, n as usize, 2],
                    b.iter().map(|&x| x as f64).collect(),
                ));
                let z = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    seismic_lang::types::DType::F32,
                    vec![3, 2],
                    vec![0.0; 6],
                ));
                interpreter
                    .run(
                        "fold",
                        &[
                            seismic_lang::interp::Arg::Tensor(x),
                            seismic_lang::interp::Arg::Tensor(y),
                            seismic_lang::interp::Arg::Tensor(z),
                        ],
                        &HashMap::new(),
                    )
                    .unwrap();
                let mut kernel = device.compile_executions(execution).unwrap();
                let x = device
                    .buffer_from(&a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                    .unwrap();
                let y = device
                    .buffer_from(&b.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                    .unwrap();
                let z_native = device.buffer(24).unwrap();
                kernel
                    .execute(&[x, y, z_native.clone()], &[], false)
                    .unwrap();
                let mut bytes = vec![0; 24];
                z_native.read(&mut bytes).unwrap();
                for (i, bytes) in bytes.chunks_exact(4).enumerate() {
                    assert_eq!(
                        f32::from_le_bytes(bytes.try_into().unwrap()),
                        interpreter.tensors[z].get(i) as f32,
                        "{ownership:?},n{n},segment{segment},{tree:?},i{i}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn native_standard_packed_matmul_reaches_same_generic_fold_family() {
    use seismic_lang::{
        interp::{Arg, Interpreter, Rng, TensorData},
        repr,
        types::{DType, Elem},
    };
    let device = seismic_cuda::Device::open(0).unwrap();
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "packed_fold.seismic.portable".into(),
        scope: Scope::Portable,
        text: r#"
fn packed(x:tensor[3,256] f32,weight:tensor[5,256] W,out:tensor[3,5] f32):
  for row,col in parallel:
    a = load(x[row:row+1])
    b = load(weight[col:col+1])
    c = tile[1,1] f32
    for i,j in owned(c): c[i,j] = 3.0 + f32(col)
    matmul(a,b,c)
    store(c,out[row:row+1,col:col+1])
"#
        .into(),
    });
    let p = compile(&sources, &["cpu".into(), "cuda".into(), "metal".into()]).unwrap();
    for name in ["q4k", "q5k", "q6k"] {
        for decoded in [false, true] {
            let f = lower_selected(
                &p,
                "packed",
                "cuda",
                &HashMap::new(),
                &HashMap::from([("W".into(), Elem::Repr(name.into()))]),
                &Options::default(),
                &mut |d| {
                    Ok(match d.kind {
                        DecisionKind::Representation { .. } => {
                            if decoded {
                                Alternative::Decoded
                            } else {
                                Alternative::Encoded
                            }
                        }
                        DecisionKind::Reduction { .. } => {
                            Alternative::ReductionTree(Tree::Pairwise)
                        }
                        DecisionKind::ReductionSegments { .. } => Alternative::ReductionSegment(16),
                        _ => d.alternatives.get(0).unwrap(),
                    })
                },
            )
            .unwrap();
            let selected = prepare(&f, &device.info);
            assert_eq!(selected[0].dispatch().lanes_per_item, 32);
            let x = TensorData::random_dense(&mut Rng(0x54321222), DType::F32, vec![3, 256]);
            let w = TensorData::random_packed(
                &mut Rng(0x1477aa3),
                repr::lookup(name).unwrap(),
                vec![5, 256],
            );
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
                signatures: HashMap::new(),
            };
            let mut interpreter = Interpreter::new(&reference);
            let a = interpreter.add_tensor(x.clone());
            let b = interpreter.add_tensor(w.clone());
            let c =
                interpreter.add_tensor(TensorData::dense(DType::F32, vec![3, 5], vec![0.0; 15]));
            interpreter
                .run(
                    "packed",
                    &[Arg::Tensor(a), Arg::Tensor(b), Arg::Tensor(c)],
                    &HashMap::new(),
                )
                .unwrap();
            let mut kernel = device.compile_executions(selected).unwrap();
            let input = device.buffer_from(&x.device_bytes()[0]).unwrap();
            let weights = w
                .device_bytes()
                .iter()
                .map(|b| device.buffer_from(b).unwrap())
                .collect::<Vec<_>>();
            let out = device.buffer(60).unwrap();
            let buffers = kernel
                .buffers()
                .iter()
                .map(|binding| match binding.parameter.as_str() {
                    "x" => input.clone(),
                    "out" => out.clone(),
                    _ => weights[repr::lookup(name)
                        .unwrap()
                        .plane_index(&binding.plane)
                        .unwrap()]
                    .clone(),
                })
                .collect::<Vec<_>>();
            kernel.execute(&buffers, &[], false).unwrap();
            let mut bytes = vec![0; 60];
            out.read(&mut bytes).unwrap();
            for (i, bytes) in bytes.chunks_exact(4).enumerate() {
                let actual = f32::from_le_bytes(bytes.try_into().unwrap());
                let expected = interpreter.tensors[c].get(i) as f32;
                assert_eq!(actual, expected, "{name} element{i}");
            }
        }
    }
}

#[test]
fn resource_analysis_tracks_dynamic_exchange_from_retained_operations() {
    use seismic_accounting::{schedule, workload};
    use seismic_cuda::model;
    let execution = prepare(&lowered(35, 2, Tree::Pairwise), &device());
    let e = &execution[0];
    let hardware = model::CudaHardware {
        identity: "synthetic unit PTX service; no native timing claim".into(),
        scope: model::Scope::HypotheticalInstructionPreservingPtxV1,
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1_000_000_000,
        },
        execution_units: 1,
        warp_width: 32,
        cohorts: model::CohortPolicy::LowestPosition,
        internal_alignment: 256,
        resources: vec![model::Resource {
            name: "issue".into(),
            scope: model::ResourceScope::Device,
            capacity: 1,
            unit: schedule::CapacityUnit::Slots,
        }],
        timings: model::requirements(e)
            .into_iter()
            .filter_map(|r| {
                if let model::Requirement::Instruction(primitive) = r {
                    Some(model::PrimitiveTiming {
                        primitive,
                        latency: model::Ticks::Fixed(1),
                        reservations: vec![model::Reservation {
                            resource: 0,
                            offset: 0,
                            duration: model::Ticks::Fixed(1),
                            units: model::Amount::fixed(1),
                        }],
                    })
                } else {
                    None
                }
            })
            .collect(),
        block_residency: vec![],
        per_unit_residency: vec![model::UnitResidency {
            name: "resident blocks".into(),
            capacity: 2,
            units_per_block: model::Amount::fixed(1),
        }],
    };
    let workload = workload::ScalarWorkload { integer_domains: Vec::new(),
        identity: "disjoint unknown float inputs".into(),
        allocations: e
            .program()
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| workload::Allocation {
                id: i as u64,
                bytes: b.bytes as u64,
                alignment: 256,
                known_bytes: Default::default(),
            })
            .collect(),
        buffers: e
            .program()
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| workload::BufferBinding {
                allocation: i as u64,
                offset: 0,
                bytes: b.bytes as u64,
            })
            .collect(),
        scalars: vec![],
    };
    let derived = model::derive_cuda(
        e,
        &hardware,
        &workload,
        &model::Placement::HomogeneousResidentSlots,
        workload::DerivationLimits {
            instructions: 1_000_000,
            operations: 100_000,
        },
    )
    .unwrap();
    let exchanges = derived
        .events
        .iter()
        .filter(|event| {
            matches!(
                event.requirement,
                model::Requirement::Instruction(seismic_cuda::ptx::Primitive::Shuffle { .. })
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(exchanges.len(), 36);
    assert!(
        exchanges
            .iter()
            .all(|e| e.active_lanes.len() == 32 && !e.predecessors.is_empty())
    );
    assert!(derived.model.lower_bound().unwrap() > 36);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn native_ordinary_reductions_use_the_same_selected_participant_form() {
    use seismic_lang::{
        interp::{Arg, Interpreter, TensorData},
        types::DType,
    };
    let device = seismic_cuda::Device::open(0).unwrap();
    for operation in ["sum", "min", "max"] {
        let text = format!(
            "fn ordinary(x:tensor[3,35] f32,out:tensor[3] f32):\n  for row in parallel:\n    a = load(x[row])\n    value = reduce(a,0,{operation})\n    result = tile[1] f32\n    for i in owned(result): result[i] = value\n    store(result,out[row:row+1])\n"
        );
        let program = compile(
            &[SourceFile {
                path: "ordinary.seismic.portable".into(),
                scope: Scope::Portable,
                text,
            }],
            &["cuda".into()],
        )
        .unwrap();
        let function = lower_selected(
            &program,
            "ordinary",
            "cuda",
            &HashMap::new(),
            &HashMap::new(),
            &Options::default(),
            &mut |d| {
                Ok(match d.kind {
                    DecisionKind::Reduction { .. } => Alternative::ReductionTree(Tree::Pairwise),
                    DecisionKind::ReductionSegments { .. } => Alternative::ReductionSegment(2),
                    _ => d.alternatives.get(0).unwrap(),
                })
            },
        )
        .unwrap();
        let execution = prepare(&function, &device.info);
        assert_eq!(execution[0].dispatch().lanes_per_item, 32);
        let mut values = (0..105).map(|i| (i as f32 - 53.) / 8.).collect::<Vec<_>>();
        values[0] = -0.;
        values[1] = 0.;
        values[9] = f32::NAN;
        values[35] = f32::INFINITY;
        values[70] = f32::NEG_INFINITY;
        let reference = seismic_lang::program::Program {
            functions: vec![seismic_lang::ir::Function {
                name: function.name.clone(),
                is_construct: false,
                shape_params: vec![],
                elem_params: vec![],
                params: function.params.clone(),
                index_params: function.index_params.clone(),
                vars: function.vars.clone(),
                body: function.body.clone(),
            }],
            lowerings: vec![],
            signatures: HashMap::new(),
        };
        let mut interpreter = Interpreter::new(&reference);
        let input = interpreter.add_tensor(TensorData::dense(
            DType::F32,
            vec![3, 35],
            values.iter().map(|&x| x as f64).collect(),
        ));
        let output = interpreter.add_tensor(TensorData::dense(DType::F32, vec![3], vec![0.; 3]));
        interpreter
            .run(
                "ordinary",
                &[Arg::Tensor(input), Arg::Tensor(output)],
                &HashMap::new(),
            )
            .unwrap();
        let mut kernel = device.compile_executions(execution).unwrap();
        let input = device
            .buffer_from(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let result = device.buffer(12).unwrap();
        kernel
            .execute(&[input, result.clone()], &[], false)
            .unwrap();
        let mut bytes = vec![0; 12];
        result.read(&mut bytes).unwrap();
        for (row, bytes) in bytes.chunks_exact(4).enumerate() {
            let actual = f32::from_le_bytes(bytes.try_into().unwrap());
            let expected = interpreter.tensors[output].get(row) as f32;
            if expected.is_nan() {
                assert!(actual.is_nan(), "{operation} row{row}");
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{operation} row{row}");
            }
        }
    }
}
