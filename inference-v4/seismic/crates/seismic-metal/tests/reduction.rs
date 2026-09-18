use seismic_lang::types::DType;
use seismic_metal::reduction::{Algorithm, ReductionDomain};
use seismic_realization::dispatch::{GroupDispatch, TilePlacement};

#[test]
fn reduction_domains_follow_ownership_and_numerical_contracts() {
    for placement in [
        TilePlacement::Replicated,
        TilePlacement::GroupShared,
        TilePlacement::Distributed,
    ] {
        let reduction =
            ReductionDomain::new(&[3, 64], 0, DType::F32, false, placement.clone(), 32).unwrap();
        assert_eq!(reduction.input_capacity(), 192);
        assert_eq!(reduction.output_capacity(), 64);
        assert!(reduction.algorithms().contains(&Algorithm::Ordered));
        assert_eq!(
            reduction.algorithms().contains(&Algorithm::LaneLocal),
            placement == TilePlacement::Distributed
        );
        assert_eq!(
            reduction.algorithms().contains(&Algorithm::Collective),
            placement == TilePlacement::Distributed
        );
        for &algorithm in reduction.algorithms() {
            let output = reduction
                .output(algorithm, "result".into())
                .unwrap()
                .unwrap();
            let layout = output
                .layout(&GroupDispatch::new(1, 32, 4).unwrap())
                .unwrap();
            assert_eq!(
                layout.private_elements_per_lane,
                reduction.output_slots(algorithm).unwrap()
            );
        }
    }
    let crossing = ReductionDomain::new(
        &[3, 65],
        0,
        DType::F32,
        false,
        TilePlacement::Distributed,
        32,
    )
    .unwrap();
    assert!(!crossing.algorithms().contains(&Algorithm::LaneLocal));
    assert!(crossing.output(Algorithm::LaneLocal, "bad".into()).is_err());
    for (dtype, ordered) in [
        (DType::F32, true),
        (DType::F16, false),
        (DType::BF16, false),
    ] {
        let reduction =
            ReductionDomain::new(&[3, 65], 0, dtype, ordered, TilePlacement::Distributed, 32)
                .unwrap();
        assert_eq!(reduction.algorithms(), [Algorithm::Ordered]);
    }
}

#[test]
fn empty_reduction_axes_and_empty_outputs_are_distinct() {
    let identity =
        ReductionDomain::new(&[0], 0, DType::F32, false, TilePlacement::Distributed, 32).unwrap();
    assert!(identity.scalar_output());
    assert_eq!(identity.output_capacity(), 1);
    assert_eq!(identity.algorithms(), [Algorithm::Ordered]);
    assert!(identity
        .output(Algorithm::Ordered, "scalar".into())
        .unwrap()
        .is_none());
    let empty = ReductionDomain::new(
        &[3, 0],
        0,
        DType::F32,
        false,
        TilePlacement::Distributed,
        32,
    )
    .unwrap();
    assert_eq!(empty.output_capacity(), 0);
    assert!(!empty.scalar_output());
    assert_eq!(empty.output_slots(Algorithm::Ordered).unwrap(), 1);
    assert!(
        ReductionDomain::new(&[3], 1, DType::F32, false, TilePlacement::Distributed, 32).is_err()
    );
    assert!(
        ReductionDomain::new(&[-1], 0, DType::F32, false, TilePlacement::Distributed, 32).is_err()
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn boolean_reductions_use_boolean_identities_and_no_numeric_collective() {
    use seismic_lang::{
        lower::lower,
        program::{compile, SourceFile},
        Scope,
    };
    use seismic_metal::{execution::Config, msl::emit_storage_selected, runtime::Device};
    let device = Device::open().unwrap();
    let values: Vec<u8> = (0..15)
        .map(|i| u8::from(i % 5 == 1 || (i % 5 == 2 && i / 5 == 0)))
        .collect();
    for op in ["sum", "max", "min"] {
        let source = format!("fn evaluate(x: tensor[3,5] bool, out: tensor[5] bool):\n  t = load(x)\n  y = reduce(t,0,{op})\n  store(y,out)\n");
        let program = compile(
            &[SourceFile {
                path: "boolean-reduction.seismic.portable".into(),
                scope: Scope::Portable,
                text: source,
            }],
            &[],
        )
        .unwrap();
        let ir = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
        for placement in [
            TilePlacement::Replicated,
            TilePlacement::GroupShared,
            TilePlacement::Distributed,
        ] {
            let contract =
                ReductionDomain::new(&[3, 5], 0, DType::Bool, false, placement.clone(), 32)
                    .unwrap();
            assert_eq!(contract.algorithms(), [Algorithm::Ordered]);
            let emitted = emit_storage_selected(
                &ir,
                Config {
                    loads: seismic_realization::LoadStrategy::Materialize,
                    ..Default::default()
                },
                &mut |_| Ok(placement.clone()),
            )
            .unwrap();
            let pipeline = device.compile(emitted).unwrap();
            let input = device.buffer_from(&values).unwrap();
            let result = device.buffer(5).unwrap();
            device.run(&pipeline, &[&input, &result], &[], 1).unwrap();
            let expected: Vec<u8> = (0..5)
                .map(|c| {
                    u8::from(if op == "min" {
                        (0..3).all(|r| values[r * 5 + c] != 0)
                    } else {
                        (0..3).any(|r| values[r * 5 + c] != 0)
                    })
                })
                .collect();
            assert_eq!(result.read(5), expected, "{op}, {placement:?}");
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn reduction_forms_match_reference_and_report_output_allocations() {
    use seismic_lang::{
        lower::lower,
        program::{compile, SourceFile},
        Scope,
    };
    use seismic_metal::{execution::Config, msl::emit_storage_selected, runtime::Device};
    let device = Device::open().unwrap();
    for (rows, cols, axis) in [(3, 64, 0), (3, 65, 0), (4, 33, 1), (0, 4, 0), (3, 0, 0)] {
        let output_count = if axis == 0 { cols } else { rows };
        for ordered in [false, true] {
            let source = format!("fn evaluate(x: tensor[{rows},{cols}] f32, out: tensor[{output_count}] f32):\n  t = load(x)\n  y = reduce(t,{axis},sum,ordered={})\n  store(y,out)\n",if ordered {"true"}else{"false"});
            let program = compile(
                &[SourceFile {
                    path: "reduction.seismic.portable".into(),
                    scope: Scope::Portable,
                    text: source,
                }],
                &[],
            )
            .unwrap();
            let ir = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
            let values: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32 - 3.).collect();
            for placement in [
                TilePlacement::Replicated,
                TilePlacement::GroupShared,
                TilePlacement::Distributed,
            ] {
                let emitted = emit_storage_selected(
                    &ir,
                    Config {
                        loads: seismic_realization::LoadStrategy::Materialize,
                        ..Default::default()
                    },
                    &mut |_| Ok(placement.clone()),
                )
                .unwrap();
                let output = emitted
                    .launches
                    .iter()
                    .flat_map(|l| &l.tiles)
                    .find(|t| t.symbol == "y")
                    .unwrap();
                assert_eq!(output.capacity, output_count as u64);
                let pipeline = device.compile(emitted).unwrap();
                let input_bytes = if values.is_empty() {
                    vec![0; 4]
                } else {
                    values.iter().flat_map(|v| v.to_le_bytes()).collect()
                };
                let input = device.buffer_from(&input_bytes).unwrap();
                let sentinel = vec![-999.0f32; output_count.max(1)]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                let result = device.buffer_from(&sentinel).unwrap();
                device.run(&pipeline, &[&input, &result], &[], 1).unwrap();
                for (i, bytes) in result.read(sentinel.len()).chunks_exact(4).enumerate() {
                    let expected = if output_count == 0 {
                        -999.
                    } else if axis == 0 {
                        (0..rows).map(|r| values[r * cols + i]).sum()
                    } else {
                        values[i * cols..(i + 1) * cols].iter().sum()
                    };
                    assert_eq!(
                        f32::from_le_bytes(bytes.try_into().unwrap()),
                        expected,
                        "{rows}x{cols}, axis={axis}, {placement:?}, ordered={ordered}"
                    );
                }
            }
        }
    }
}

fn nested_reduction() -> seismic_lang::lowered_ir::LoweredIr {
    use seismic_lang::{
        lower::lower,
        program::{compile, SourceFile},
        Scope,
    };
    let program = compile(&[SourceFile {
        path: "planned-reduction.seismic.portable".into(), scope: Scope::Portable,
        text: "fn evaluate(x: tensor[3,64] f32, out: tensor[1] f32):\n  a = load(x)\n  y = reduce(a,0,sum)\n  z = tile[1] f32\n  for i in owned(z): z[i] = reduce(y,0,sum)\n  store(z,out)\n".into(),
    }], &[]).unwrap();
    lower(&program, "evaluate", "metal", &Default::default()).unwrap()
}

fn selected_nested(algorithm: Algorithm) -> seismic_metal::execution::Execution {
    seismic_metal::execution::prepare_selected(
        &nested_reduction(),
        Default::default(),
        &mut |_| Ok(TilePlacement::Distributed),
        &mut |decision| {
            Ok(if decision.domain.input_capacity() == 192 {
                algorithm
            } else {
                Algorithm::Ordered
            })
        },
    )
    .unwrap()
}

#[test]
fn reduction_selection_precedes_emission_and_propagates_output_ownership() {
    for algorithm in [
        Algorithm::Ordered,
        Algorithm::LaneLocal,
        Algorithm::Collective,
    ] {
        let execution = selected_nested(algorithm);
        let selections = execution.reductions().selections();
        assert_eq!(selections.len(), 2);
        let first = selections
            .values()
            .find(|s| s.decision.domain.input_capacity() == 192)
            .unwrap();
        let second = selections
            .values()
            .find(|s| s.decision.domain.input_capacity() == 64)
            .unwrap();
        assert_eq!(first.algorithm, algorithm);
        assert!(first.decision.materialize_input);
        assert!(!second.decision.materialize_input);
        assert_eq!(
            second
                .decision
                .domain
                .algorithms()
                .contains(&Algorithm::Collective),
            algorithm == Algorithm::LaneLocal
        );
        seismic_metal::msl::emit_execution(&execution).unwrap();
    }
    let result = seismic_metal::execution::prepare_selected(
        &nested_reduction(),
        Default::default(),
        &mut |_| Ok(TilePlacement::Replicated),
        &mut |_| Ok(Algorithm::Collective),
    );
    assert!(result.err().unwrap().contains("violates"));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn every_selected_reduction_algorithm_executes_the_same_composition() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let values: Vec<f32> = (0..192).map(|i| (i % 7) as f32 - 3.0).collect();
    let expected: f32 = values.iter().sum();
    for algorithm in [
        Algorithm::Ordered,
        Algorithm::LaneLocal,
        Algorithm::Collective,
    ] {
        let execution = selected_nested(algorithm);
        let emitted = seismic_metal::msl::emit_execution(&execution).unwrap();
        let pipeline = device.compile(emitted).unwrap();
        let input = device
            .buffer_from(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.buffer(4).unwrap();
        device.run(&pipeline, &[&input, &output], &[], 1).unwrap();
        assert_eq!(
            f32::from_le_bytes(output.read(4).try_into().unwrap()),
            expected,
            "{algorithm:?}"
        );
    }
}

#[test]
fn partially_active_owned_domain_cannot_select_a_cross_lane_reduction() {
    use seismic_lang::{
        lower::lower,
        program::{compile, SourceFile},
        Scope,
    };
    let program = compile(&[SourceFile {
        path: "masked-reduction.seismic.portable".into(), scope: Scope::Portable,
        text: "fn evaluate(x: tensor[64] f32, out: tensor[1] f32):\n  a = load(x)\n  z = tile[1] f32\n  for i in owned(z): z[i] = 0.0\n  for i in owned(z):\n    if i == 0: z[i] = reduce(a,0,sum)\n  store(z,out)\n".into(),
    }], &[]).unwrap();
    let ir = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    let result = seismic_metal::execution::prepare_selected(
        &ir,
        Default::default(),
        &mut |_| Ok(TilePlacement::Distributed),
        &mut |_| Ok(Algorithm::Collective),
    );
    assert!(result.err().unwrap().contains("full-lane participation"));
}

#[test]
fn argmax_domains_include_direct_reads_and_check_lane_participation() {
    for source in [
        None,
        Some(TilePlacement::Replicated),
        Some(TilePlacement::Distributed),
        Some(TilePlacement::GroupShared),
    ] {
        let domain =
            ReductionDomain::argmax(&[3, 65], 0, DType::F32, source.clone(), true, 32).unwrap();
        assert_eq!(domain.output_capacity(), 65);
        assert_eq!(
            domain
                .output(Algorithm::Ordered, "winner".into())
                .unwrap()
                .unwrap()
                .dtype,
            DType::I32
        );
        assert_eq!(
            domain.algorithms().contains(&Algorithm::Collective),
            source.is_none() || source == Some(TilePlacement::Distributed)
        );
        let partial =
            ReductionDomain::argmax(&[3, 65], 0, DType::F32, source.clone(), false, 32).unwrap();
        assert!(!partial.algorithms().contains(&Algorithm::Collective));
        assert_eq!(
            partial.algorithms().contains(&Algorithm::Ordered),
            source != Some(TilePlacement::Distributed)
        );
    }
    assert!(ReductionDomain::argmax(&[0, 65], 0, DType::F32, None, true, 32).is_err());
    assert_eq!(
        ReductionDomain::argmax(&[3, 0], 0, DType::F32, None, true, 32)
            .unwrap()
            .output_capacity(),
        0
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn argmax_plans_preserve_nan_ties_empty_outputs_and_large_shapes() {
    use seismic_lang::{
        lower::lower,
        program::{compile, SourceFile},
        Scope,
    };
    use seismic_metal::{
        execution::{prepare_selected, Config},
        msl::emit_execution,
        runtime::Device,
    };
    use seismic_realization::LoadStrategy;
    let device = Device::open().unwrap();
    for (rows, cols, axis) in [(3, 65, 0), (4, 33, 1), (3, 0, 0), (0, 3, 1)] {
        let count = if axis == 0 { cols } else { rows };
        let program = compile(&[SourceFile {
            path: "argmax-plan.seismic.portable".into(), scope: Scope::Portable,
            text: format!("fn evaluate(x: tensor[{rows},{cols}] f32, out: tensor[{count}] i32):\n  a = load(x)\n  y = reduce(a,{axis},argmax)\n  store(y,out)\n"),
        }], &[]).unwrap();
        let ir = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let (out, k) = if axis == 0 {
                    (i % cols, i / cols)
                } else {
                    (i / cols, i % cols)
                };
                match out % 5 {
                    0 => {
                        if k == 0 {
                            f32::NAN
                        } else {
                            f32::NEG_INFINITY
                        }
                    }
                    1 => f32::NAN,
                    2 => 7.0,
                    3 => {
                        if k == 0 {
                            f32::NEG_INFINITY
                        } else {
                            9.0
                        }
                    }
                    _ => (k % 3) as f32,
                }
            })
            .collect();
        let expected: Vec<i32> = (0..count)
            .map(|o| {
                let mut best = f32::NEG_INFINITY;
                let mut at = 0;
                for k in 0..if axis == 0 { rows } else { cols } {
                    let x = values[if axis == 0 {
                        k * cols + o
                    } else {
                        o * cols + k
                    }];
                    if x > best {
                        best = x;
                        at = k as i32;
                    }
                }
                at
            })
            .collect();
        for loads in [
            LoadStrategy::Materialize,
            LoadStrategy::BorrowProvenReadOnly,
        ] {
            for placement in [
                TilePlacement::Replicated,
                TilePlacement::Distributed,
                TilePlacement::GroupShared,
            ] {
                let source = if loads == LoadStrategy::BorrowProvenReadOnly {
                    None
                } else {
                    Some(placement.clone())
                };
                let domain = ReductionDomain::argmax(
                    &[rows as i64, cols as i64],
                    axis,
                    DType::F32,
                    source,
                    true,
                    32,
                )
                .unwrap();
                for &algorithm in domain.algorithms() {
                    let execution = prepare_selected(
                        &ir,
                        Config {
                            loads,
                            ..Default::default()
                        },
                        &mut |_| Ok(placement.clone()),
                        &mut |decision| {
                            assert_eq!(
                                decision.contract.operation,
                                seismic_lang::ir::ReduceOp::Argmax
                            );
                            Ok(algorithm)
                        },
                    )
                    .unwrap();
                    let selected = execution.reductions().selections().values().next().unwrap();
                    assert!(!selected.decision.materialize_input);
                    let emitted = emit_execution(&execution).unwrap();
                    assert!(emitted
                        .launches
                        .iter()
                        .flat_map(|l| &l.tiles)
                        .any(|t| t.symbol == "y"
                            && t.dtype == DType::I32
                            && t.capacity == count as u64));
                    let pipeline = device.compile(emitted).unwrap();
                    let mut bytes = values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    if bytes.is_empty() {
                        bytes.resize(4, 0);
                    }
                    let input = device.buffer_from(&bytes).unwrap();
                    let output = device.buffer_from(&vec![0xcd; (count * 4).max(4)]).unwrap();
                    device.run(&pipeline, &[&input, &output], &[], 1).unwrap();
                    if count == 0 {
                        assert_eq!(output.read(4), [0xcd; 4]);
                    } else {
                        let actual = output
                            .read(count * 4)
                            .chunks_exact(4)
                            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                            .collect::<Vec<_>>();
                        assert_eq!(
                            actual, expected,
                            "{rows}x{cols} axis {axis}, {loads:?}, {placement:?}, {algorithm:?}"
                        );
                    }
                }
            }
        }
    }
}
