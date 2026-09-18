//! Computed dynamic views retain both materialization and captured recomputation.
use seismic_accounting::quantity::Count;
use seismic_lang::{
    Scope,
    interp::{Arg, Interpreter, TensorData},
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind},
    program::{Program, SourceFile, compile},
    types::DType,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};

fn program(selection: &str) -> Program {
    compile(
        &[SourceFile {
            path: "dynamic_projection.seismic.portable".into(),
            scope: Scope::Portable,
            text: format!(
                "fn evaluate(x:tensor[4] i32,start:i32,end:i32,out:tensor[1] i32):\n  a = load(x)\n  s = tile[4] i32\n  for i in owned(s): s[i] = a[i] + i*10\n{selection}  result = tile[1] i32\n  for i in owned(result): result[i] = reduce(selected,0,sum,ordered=true)\n  store(result,out)\n"
            ),
        }],
        &[],
    )
    .unwrap()
}

fn ranked_program(selection: &str) -> Program {
    compile(&[SourceFile {
        path: "ranked_projection.seismic.portable".into(), scope: Scope::Portable,
        text: format!("fn evaluate(x:tensor[2,2] i32,start:i32,end:i32,out:tensor[1] i32):\n  a = load(x)\n  s = tile[2,2] i32\n  for i,j in owned(s): s[i,j] = a[i,j] + i*10 + j*100\n{selection}  result = tile[1] i32\n  for i in owned(result): result[i] = reduce(selected,0,sum,ordered=true)\n  store(result,out)\n"),
    }], &[]).unwrap()
}

fn reference(program: &Program, start: i32, end: i32) -> Result<i32, String> {
    let mut interpreter = Interpreter::new(program);
    let input = interpreter.add_tensor(TensorData::dense(
        DType::I32,
        program
            .functions
            .iter()
            .find(|f| f.name == "evaluate")
            .unwrap()
            .params[0]
            .1
            .shaped()
            .unwrap()
            .shape
            .iter()
            .map(|n| n.as_constant().unwrap() as usize)
            .collect(),
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let output = interpreter.add_tensor(TensorData::dense(DType::I32, vec![1], vec![0.0]));
    interpreter.run(
        "evaluate",
        &[
            Arg::Tensor(input),
            Arg::Scalar(f64::from(start)),
            Arg::Scalar(f64::from(end)),
            Arg::Tensor(output),
        ],
        &Default::default(),
    )?;
    Ok(interpreter.tensors[output].get(0) as i32)
}

fn exercise(
    device: &Device,
    candidate: &Candidate,
    build: fn(&str) -> Program,
    selections: &[&str],
    invocations: &[(i32, i32)],
) {
    for &selection in selections {
        let program = build(selection);
        for (recompute, piece) in [false, true]
            .into_iter()
            .flat_map(|recompute| [None, Some(1), Some(3)].map(|piece| (recompute, piece)))
        {
            let mut computed_projection = false;
            let lowered = lower_selected(
                &program,
                "evaluate",
                device.backend(),
                &Default::default(),
                &Default::default(),
                &Options {
                    piece,
                    ..Default::default()
                },
                &mut |decision| {
                    if matches!(decision.kind, DecisionKind::Producer { .. })
                        && decision.alternatives.contains(&Alternative::Recompute)
                    {
                        computed_projection |= matches!(&decision.kind,
                            DecisionKind::Producer { name, .. } if name == "s");
                        Ok(if recompute {
                            Alternative::Recompute
                        } else {
                            Alternative::Materialize
                        })
                    } else {
                        Ok(decision.alternatives.get(0).unwrap())
                    }
                },
            )
            .unwrap_or_else(|error| {
                panic!("{selection}, recompute={recompute}, piece={piece:?}: {error}")
            });
            assert!(
                computed_projection,
                "computed view has no recomputation alternative: {selection}"
            );
            let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
            let input = device
                .buffer_from(&(1i32..=4).flat_map(i32::to_le_bytes).collect::<Vec<_>>())
                .unwrap();
            let output = device.buffer(4).unwrap();
            for &(start, end) in invocations {
                let context =
                    format!("{selection}, recompute={recompute}, piece={piece:?}, {start}:{end}");
                let expected = reference(&program, start, end);
                let result = kernel.execute(
                    &[input.clone(), output.clone()],
                    &[f64::from(start), f64::from(end)],
                );
                match expected {
                    Ok(expected) => {
                        result.unwrap_or_else(|error| panic!("{context}: {error}"));
                        let mut bytes = [0; 4];
                        output.read(&mut bytes).unwrap();
                        assert_eq!(i32::from_le_bytes(bytes), expected, "{context}");
                    }
                    Err(_) => {
                        assert!(result.is_err(), "erased endpoint failure: {context}");
                        // Failed native completion can have partial effects.
                        // The next valid invocation must still match reference.
                    }
                }
            }
        }
    }
}

fn clamped_windows(device: &Device, candidate: &Candidate) {
    exercise(
        device,
        candidate,
        program,
        &[
            "  selected = s[start:end]\n",
            "  selected = s[:end]\n",
            "  selected = s[start:]\n",
            "  selected = s[start:end][start:end]\n",
            "  window = s[start:end]\n  selected = window[:]\n",
        ],
        &[(1, 3), (-10, 20), (3, 1), (9, 12), (-9, -1), (0, 0), (0, 4)],
    );
}

fn endpoint_snapshots_and_failures(device: &Device, candidate: &Candidate) {
    exercise(
        device,
        candidate,
        program,
        &[
            "  lo = start\n  hi = end\n  window = s[lo:hi]\n  lo = 0\n  hi = 4\n  selected = window[:]\n",
            "  lo = start\n  hi = end\n  first = s[lo:hi]\n  lo = 0\n  hi = 4\n  selected = s[lo:hi]\n",
            "  lo = start\n  hi = end\n  first = s[lo:hi]\n  lo = start - 1\n  hi = end + 1\n  second = s[lo:hi]\n  selected = tile[2] i32\n  for i in owned(selected): selected[i] = 0\n  selected[0] = reduce(first,0,sum,ordered=true) + extent(first,0)\n  selected[1] = reduce(second,0,sum,ordered=true) + extent(second,0)\n",
            "  choice = tile[1] i32\n  for i in owned(choice): choice[i] = 0\n  if start >= 0:\n    first = s[start:end]\n    choice[0] = reduce(first,0,sum,ordered=true)\n  else:\n    second = s[start:end]\n    choice[0] = reduce(second,0,sum,ordered=true)\n  selected = choice[:]\n",
            "  selected = s[a[start]:end]\n",
            "  selected = s[start:a[end]]\n",
            "  selected = s[start/end:0]\n",
        ],
        &[(1, 3), (-1, 0), (4, -1), (1, 3), (0, 0), (0, 2)],
    );
}

fn ranked_views(device: &Device, candidate: &Candidate) {
    exercise(
        device,
        candidate,
        ranked_program,
        &[
            "  selected = s[start,:end]\n",
            "  selected = s[:end,start]\n",
            "  selected = s.T[start,:end]\n",
            "  point = start-start\n  selected = s[start:end,:][point,:end]\n",
            "  window = s.T[:end,:]\n  selected = window[start,:]\n",
            "  selected = s[start,:end/start]\n",
        ],
        &[(1, 2), (0, 2), (-1, 2), (2, 0), (1, 0), (0, -3), (1, 2)],
    );
}

#[test]
fn cpu_computed_ranked_views_preserve_coordinate_maps_and_guards() {
    ranked_views(
        &Device::cpu(),
        &Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
fn cpu_computed_dynamic_slice_keeps_materialization_available() {
    clamped_windows(
        &Device::cpu(),
        &Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
fn cpu_computed_dynamic_slice_preserves_endpoint_snapshots_and_failures() {
    endpoint_snapshots_and_failures(
        &Device::cpu(),
        &Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_computed_dynamic_slice_keeps_materialization_available() {
    let device = Device::metal().unwrap();
    let candidate = Candidate::Metal(Default::default());
    clamped_windows(&device, &candidate);
    endpoint_snapshots_and_failures(&device, &candidate);
    ranked_views(&device, &candidate);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_computed_dynamic_slice_keeps_materialization_available() {
    let device = Device::cuda(0).unwrap();
    let candidate = Candidate::Cuda {
        options: seismic_realization::ScalarOptions {
            dispatch: seismic_realization::Dispatch::Sequential,
            loads: LoadStrategy::Materialize,
        },
        threads_per_block: 32,
    };
    clamped_windows(&device, &candidate);
    endpoint_snapshots_and_failures(&device, &candidate);
    ranked_views(&device, &candidate);
}

fn computed_window_storage(
    device: &Device,
    candidate: &Candidate,
    sizes: &[i64],
    piece: Option<i64>,
) {
    let program = compile(&[SourceFile {
        path: "computed_history.seismic.portable".into(), scope: Scope::Portable,
        text: "fn evaluate[N](x:tensor[N] f32,start:i32,end:i32,out:tensor[1] f32):\n  raw = load(x)\n  computed = tile[N] f32\n  for i in owned(computed): computed[i] = raw[i] * 2.0 + f32(i)\n  window = computed[start:end]\n  result = tile[1] f32\n  for i in owned(result): result[i] = reduce(window,0,sum,ordered=true)\n  store(result,out)\n".into(),
    }], &[]).unwrap();
    let mut storage = Vec::new();
    for &n in sizes {
        let lowered = lower_selected(
            &program,
            "evaluate",
            device.backend(),
            &std::collections::HashMap::from([("N".into(), n)]),
            &Default::default(),
            &Options {
                piece,
                ..Default::default()
            },
            &mut |decision| {
                Ok(match &decision.kind {
                    DecisionKind::Producer { .. }
                        if decision.alternatives.contains(&Alternative::Recompute) =>
                    {
                        Alternative::Recompute
                    }
                    _ => decision.alternatives.get(0).unwrap(),
                })
            },
        )
        .unwrap();
        if let Some(capacity) = piece {
            assert!(lowered.decisions.iter().any(|decision| {
                matches!(decision.domain.kind, DecisionKind::Stream { maximum, .. } if maximum == n)
                    && decision.selected == Alternative::StreamCapacity(capacity.min(n))
            }), "computed window has no streaming choice for capacity {capacity}");
        }
        let execution = seismic_runtime::execution::Execution::prepare(
            &lowered,
            candidate.clone(),
            &device.facts(),
        )
        .unwrap();
        use seismic_runtime::execution::Account;
        let quantities = match execution.account().unwrap() {
            Account::CpuScalarIr { account, .. } => {
                vec![Count::Exact(account.scratch_bytes_per_invocation)]
            }
            Account::Cuda(phases) => phases
                .iter()
                .map(|phase| Count::Exact(phase.scalar_ir().scratch_bytes_per_invocation))
                .collect(),
            #[cfg(target_os = "macos")]
            Account::MetalStorage { account, .. } => {
                let mut quantities = vec![account.retained_scratch_bytes];
                for launch in account.launches {
                    quantities.extend([
                        launch.declared_private_array_bytes_per_lane,
                        launch.declared_shared_array_bytes_per_group,
                        launch.declared_fragment_payload_bytes_per_subgroup,
                    ]);
                }
                quantities
            }
        };
        assert!(quantities.iter().all(|n| matches!(n, Count::Exact(_))));
        storage.push(quantities);
        let data = seismic_lang::demand::data_variables(&lowered.body);
        for (variable, definition) in lowered.vars.iter().enumerate() {
            if matches!(definition.name.as_str(), "raw" | "computed")
                || piece.is_some() && definition.name == "window"
            {
                assert!(
                    !data.contains(&variable),
                    "full producer retains data storage"
                );
            }
        }
        let input = device
            .buffer_from(
                &(0..n)
                    .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.buffer(4).unwrap();
        let mut kernel = device.compile_execution(execution).unwrap();
        for (start, end) in [(2, 17), (-3, 1000), (20, 3), (0, 0)] {
            kernel
                .execute(
                    &[input.clone(), output.clone()],
                    &[start as f64, end as f64],
                )
                .unwrap();
            let mut bytes = [0; 4];
            output.read(&mut bytes).unwrap();
            let end = end.clamp(0, n);
            let start = start.clamp(0, end);
            let expected =
                (start..end).fold(0.0f32, |sum, i| sum + (i as f32 + 1.0) * 2.0 + i as f32);
            assert_eq!(f32::from_le_bytes(bytes), expected);
        }
    }
    for (pair, sizes) in storage.windows(2).zip(sizes.windows(2)) {
        if piece.is_some() {
            assert_eq!(
                pair[0], pair[1],
                "streamed storage grows with input capacity: {storage:?}"
            );
        } else {
            // The whole-window diagnostic choice may use one N-capacity result;
            // source loads and computed producers must not add more N-sized data.
            assert_eq!(
                pair[1][0],
                pair[0][0]
                    .clone()
                    .add(&Count::Exact(((sizes[1] - sizes[0]) * 4) as u64)),
                "extra history-sized allocation: {storage:?}"
            );
        }
    }
}

#[test]
fn cpu_computed_window_omits_full_producer_storage() {
    computed_window_storage(
        &Device::cpu(),
        &Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
        &[32, 64, 128],
        None,
    );
}

#[test]
fn cpu_computed_window_has_bounded_streaming_storage() {
    let device = Device::cpu();
    let candidate = Candidate::Cpu {
        loads: LoadStrategy::Materialize,
    };
    for capacity in 1..=4 {
        computed_window_storage(&device, &candidate, &[4], Some(capacity));
    }
    computed_window_storage(&device, &candidate, &[32, 64, 128], Some(7));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_computed_window_has_bounded_streaming_storage() {
    computed_window_storage(
        &Device::metal().unwrap(),
        &Candidate::Metal(Default::default()),
        &[32, 64, 128],
        Some(7),
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_computed_window_has_bounded_streaming_storage() {
    computed_window_storage(
        &Device::cuda(0).unwrap(),
        &Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::Sequential,
                loads: LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
        &[32, 64, 128],
        Some(7),
    );
}
