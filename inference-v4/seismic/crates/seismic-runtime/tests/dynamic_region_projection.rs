//! Captured views restrict existing pure regions without changing their snapshots.
use seismic_accounting::quantity::Count;
use seismic_lang::{
    Scope,
    interp::{Arg, Interpreter, TensorData},
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind, LoweredIr},
    program::{Program, SourceFile, compile},
    types::DType,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{
    Candidate, Device,
    execution::{Account, Execution},
};
use std::collections::HashMap;

fn program(selection: &str, ranked: bool, fallible: bool) -> Program {
    let (dimensions, coordinates, source, ordinal) = if ranked {
        ("2,N", "i,j", "x[i,j]", "i*10+j")
    } else {
        ("N", "i", "x[i]", "i")
    };
    let source = if fallible { "x[start]" } else { source };
    compile(&[SourceFile {
        path: "dynamic_region_projection.seismic.portable".into(),
        scope: Scope::Portable,
        text: format!(
            "fn evaluate[N](x:tensor[{dimensions}] f32,start:i32,end:i32,out:tensor[1] f32):\n  s = tile[{dimensions}] f16\n  for {coordinates} in owned(s): s[{coordinates}] = f16({source} * 0.125 + f32({ordinal}) * 0.0625)\n  for iteration in range(3):\n    for {coordinates} in owned(s):\n      previous = f32(s[{coordinates}])\n      rounded = f16(previous + {source} * 0.0333)\n      s[{coordinates}] = f16(f32(rounded) + f32({ordinal}) * 0.001)\n{selection}  result = tile[1] f32\n  for i in owned(result): result[i] = f32(reduce(selected,0,sum,ordered=true))\n  store(result,out)\n"
        ),
    }], &[]).unwrap()
}

fn input(n: i64, ranked: bool) -> Vec<f32> {
    (0..n * if ranked { 2 } else { 1 })
        .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
        .collect()
}

fn reference(
    program: &Program,
    n: i64,
    ranked: bool,
    start: i32,
    end: i32,
) -> Result<Vec<u8>, String> {
    let mut vm = Interpreter::new(program);
    let shape = if ranked {
        vec![2, n as usize]
    } else {
        vec![n as usize]
    };
    let x = vm.add_tensor(TensorData::dense(
        DType::F32,
        shape,
        input(n, ranked).into_iter().map(f64::from).collect(),
    ));
    let out = vm.add_tensor(TensorData::dense(DType::F32, vec![1], vec![0.0]));
    vm.run(
        "evaluate",
        &[
            Arg::Tensor(x),
            Arg::Scalar(start.into()),
            Arg::Scalar(end.into()),
            Arg::Tensor(out),
        ],
        &HashMap::from([("N".into(), n)]),
    )?;
    Ok(vm.tensors[out].device_bytes().remove(0))
}

fn lower(
    program: &Program,
    backend: &str,
    n: i64,
    piece: Option<i64>,
    recompute: bool,
) -> (LoweredIr, bool) {
    let mut offered = false;
    let lowered = lower_selected(
        program,
        "evaluate",
        backend,
        &HashMap::from([("N".into(), n)]),
        &HashMap::new(),
        &Options {
            piece,
            ..Default::default()
        },
        &mut |decision| {
            if let DecisionKind::Producer { name, .. } = &decision.kind {
                if decision.alternatives.contains(&Alternative::Recompute) {
                    offered |= name == "s";
                    return Ok(if recompute {
                        Alternative::Recompute
                    } else {
                        Alternative::Materialize
                    });
                }
            }
            Ok(decision.alternatives.get(0).unwrap())
        },
    )
    .unwrap();
    (lowered, offered)
}

fn exercise(
    device: &Device,
    candidate: &Candidate,
    ranked: bool,
    selections: &[&str],
    invocations: &[(i32, i32)],
    fallible: bool,
) {
    for selection in selections {
        let program = program(selection, ranked, fallible);
        for recompute in [false, true] {
            for piece in [None, Some(1), Some(3)] {
                let (lowered, offered) = lower(&program, device.backend(), 4, piece, recompute);
                assert_eq!(offered, !fallible, "region admission: {selection}");
                let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
                let x = device
                    .buffer_from(
                        &input(4, ranked)
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let out = device.buffer(4).unwrap();
                for &(start, end) in invocations {
                    let context = format!(
                        "{selection}, ranked={ranked}, recompute={recompute}, piece={piece:?}, {start}:{end}"
                    );
                    let actual =
                        kernel.execute(&[x.clone(), out.clone()], &[start.into(), end.into()]);
                    match reference(&program, 4, ranked, start, end) {
                        Ok(expected) => {
                            actual.unwrap_or_else(|error| panic!("{context}: {error}"));
                            let mut bytes = [0; 4];
                            out.read(&mut bytes).unwrap();
                            assert_eq!(bytes.as_slice(), expected, "{context}");
                        }
                        Err(_) => assert!(actual.is_err(), "removed source failure: {context}"),
                    }
                }
            }
        }
    }
}

fn views_and_snapshots(device: &Device, candidate: &Candidate) {
    exercise(
        device,
        candidate,
        false,
        &[
            "  selected = s[start:end]\n",
            "  selected = s[start:end][start:end]\n",
            "  lo = start\n  hi = end\n  window = s[lo:hi]\n  lo = 0\n  hi = 4\n  selected = window[:]\n",
            "  lo = start\n  hi = end\n  first = s[lo:hi]\n  lo = 0\n  hi = extent(s,0)\n  second = s[lo:hi]\n  selected = tile[2] f16\n  for i in owned(selected): selected[i] = f16(0.0)\n  selected[0] = reduce(first,0,sum,ordered=true) + f16(extent(first,0))\n  selected[1] = reduce(second,0,sum,ordered=true) + f16(extent(second,0))\n",
            "  selected = s[start/end:0]\n",
            "  selected = s[i32(x[start]):end]\n",
        ],
        &[(1, 3), (-3, 9), (3, 1), (0, 0), (9, 10), (1, 3)],
        false,
    );
    exercise(
        device,
        candidate,
        true,
        &[
            "  selected = s[start,:end]\n",
            "  selected = s[:end,start]\n",
            "  selected = s.T[start,:end]\n",
            "  point = start-start\n  selected = s[start:end,:][point,:end]\n",
            "  window = s.T[:end,:]\n  selected = window[start,:]\n",
            "  selected = s[start,:end/start]\n",
        ],
        &[(1, 2), (0, 2), (-1, 2), (2, 0), (1, 0), (0, -3), (1, 2)],
        false,
    );
    // A dynamic point in the full producer remains observable even when its
    // demanded output view is empty. Valid invocations recover after failures.
    exercise(
        device,
        candidate,
        false,
        &["  selected = s[start:end]\n"],
        &[(1, 3), (-1, -1), (4, 4), (1, 3)],
        true,
    );
}

fn storage(execution: &Execution) -> Vec<Count> {
    match execution.account().unwrap() {
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
    }
}

fn bounded_storage(device: &Device, candidate: &Candidate) {
    let source = program("  selected = s[start:end]\n", false, false);
    let mut accounts = Vec::new();
    for n in [32, 64, 128] {
        let (lowered, offered) = lower(&source, device.backend(), n, Some(7), true);
        assert!(offered);
        assert!(lowered.decisions.iter().any(|decision|
            matches!(decision.domain.kind, DecisionKind::Stream { maximum, .. } if maximum == n)
                && decision.selected == Alternative::StreamCapacity(7)));
        let data = seismic_lang::demand::data_variables(&lowered.body);
        for (id, variable) in lowered.vars.iter().enumerate() {
            if matches!(variable.name.as_str(), "s" | "selected") {
                assert!(
                    !data.contains(&id),
                    "full region output remains live: {}",
                    variable.name
                );
            }
        }
        let execution = Execution::prepare(&lowered, candidate.clone(), &device.facts()).unwrap();
        accounts.push(storage(&execution));
        let mut kernel = device.compile_execution(execution).unwrap();
        let x = device
            .buffer_from(
                &input(n, false)
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let out = device.buffer(4).unwrap();
        for (start, end) in [(2, 17), (-3, 1000), (20, 3), (0, 0)] {
            kernel
                .execute(&[x.clone(), out.clone()], &[start.into(), end.into()])
                .unwrap();
            let mut actual = [0; 4];
            out.read(&mut actual).unwrap();
            assert_eq!(
                actual.as_slice(),
                reference(&source, n, false, start, end).unwrap(),
                "N={n}, {start}:{end}"
            );
        }
    }
    assert!(
        accounts
            .iter()
            .flatten()
            .all(|quantity| matches!(quantity, Count::Exact(_)))
    );
    assert!(
        accounts.windows(2).all(|pair| pair[0] == pair[1]),
        "region storage grows with source capacity: {accounts:?}"
    );
}

#[test]
fn cpu_dynamic_regions_preserve_views_snapshots_and_failures() {
    views_and_snapshots(
        &Device::cpu(),
        &Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
fn cpu_dynamic_regions_have_bounded_consumer_storage() {
    bounded_storage(
        &Device::cpu(),
        &Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_dynamic_regions_preserve_views_and_bounded_storage() {
    let device = Device::metal().unwrap();
    let candidate = Candidate::Metal(Default::default());
    views_and_snapshots(&device, &candidate);
    bounded_storage(&device, &candidate);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_dynamic_regions_preserve_views_and_bounded_storage() {
    let device = Device::cuda(0).unwrap();
    let candidate = Candidate::Cuda {
        options: seismic_realization::ScalarOptions {
            dispatch: seismic_realization::Dispatch::Sequential,
            loads: LoadStrategy::Materialize,
        },
        threads_per_block: 32,
    };
    views_and_snapshots(&device, &candidate);
    bounded_storage(&device, &candidate);
}
