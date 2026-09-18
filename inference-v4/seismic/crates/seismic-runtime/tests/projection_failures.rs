//! Restricting a producer's demanded values must preserve its dynamic failures.
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

fn program(expression: &str, region: bool) -> Program {
    let producer = if region {
        format!(
            "  for i in owned(s): s[i] = 0\n  for iteration in range(2):\n    for i in owned(s): s[i] = ({expression}) + s[i]\n"
        )
    } else {
        format!("  for i in owned(s): s[i] = {expression}\n")
    };
    compile(
        &[SourceFile {
            path: "projection_failures.seismic.portable".into(),
            scope: Scope::Portable,
            text: format!(
                "fn evaluate(values:tensor[4] i32,controls:tensor[4] i32,out:tensor[2] i32):\n  a = load(values)\n  b = load(controls)\n  s = tile[4] i32\n{producer}  selected = s[2:4]\n  store(selected,out)\n"
            ),
        }],
        &[],
    )
    .unwrap()
}

fn encode(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|n| n.to_le_bytes()).collect()
}

fn reference(program: &Program, values: &[i32; 4], controls: &[i32; 4]) -> Result<Vec<u8>, String> {
    let mut interpreter = Interpreter::new(program);
    let values = interpreter.add_tensor(TensorData::dense(
        DType::I32,
        vec![4],
        values.iter().map(|&n| f64::from(n)).collect(),
    ));
    let controls = interpreter.add_tensor(TensorData::dense(
        DType::I32,
        vec![4],
        controls.iter().map(|&n| f64::from(n)).collect(),
    ));
    let output = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![0.0; 2]));
    interpreter.run(
        "evaluate",
        &[
            Arg::Tensor(values),
            Arg::Tensor(controls),
            Arg::Tensor(output),
        ],
        &Default::default(),
    )?;
    Ok((0..2)
        .flat_map(|i| (interpreter.tensors[output].get(i) as i32).to_le_bytes())
        .collect())
}

fn exercise(
    expression: &str,
    values: [i32; 4],
    controls: [i32; 4],
    invalid_prefixes: &[(i32, i32)],
) {
    exercise_on(
        &Device::cpu(),
        &Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
        expression,
        values,
        controls,
        invalid_prefixes,
    );
}

fn exercise_on(
    device: &Device,
    candidate: &Candidate,
    expression: &str,
    values: [i32; 4],
    controls: [i32; 4],
    invalid_prefixes: &[(i32, i32)],
) {
    for region in [false, true] {
        let program = program(expression, region);
        for recompute in [false, true] {
            let mut projected_s = false;
            let function = lower_selected(
                &program,
                "evaluate",
                device.backend(),
                &Default::default(),
                &Default::default(),
                &Options::default(),
                &mut |decision| {
                    if let DecisionKind::Producer { name, .. } = &decision.kind {
                        if decision.alternatives.contains(&Alternative::Recompute) {
                            projected_s |= name == "s" && recompute;
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
            if invalid_prefixes.is_empty() && recompute {
                assert!(
                    projected_s,
                    "positive control must select producer projection: {expression}, region={region}"
                );
            }
            let mut kernel = device.compile(&function, candidate.clone()).unwrap();
            let input = device.buffer_from(&encode(&values)).unwrap();
            let control = device.buffer_from(&encode(&controls)).unwrap();
            let output = device.buffer(8).unwrap();
            let buffers = [input.clone(), control.clone(), output.clone()];

            // The suffix stays valid in every invocation. Only an omitted
            // prefix changes; succeeding there would erase a source failure.
            let mut invocations = vec![(values, controls, false)];
            for &(value, count) in invalid_prefixes {
                let mut invalid_values = values;
                let mut invalid_controls = controls;
                invalid_values[0] = value;
                invalid_controls[0] = count;
                invocations.push((invalid_values, invalid_controls, true));
                invocations.push((values, controls, false));
            }
            for (values, controls, fails) in invocations {
                input.write(&encode(&values)).unwrap();
                control.write(&encode(&controls)).unwrap();
                let expected = reference(&program, &values, &controls);
                let actual = kernel.execute(&buffers, &[]);
                let context = format!(
                    "{expression}, region={region}, recompute={recompute}, projected_s={projected_s}, prefix=({}, {})",
                    values[0], controls[0]
                );
                assert_eq!(expected.is_err(), fails, "source reference: {context}");
                if fails {
                    assert!(
                        actual.is_err(),
                        "projection erased dynamic failure: {context}"
                    );
                } else {
                    actual.unwrap_or_else(|e| panic!("{context}: {e}"));
                    let mut bytes = vec![0; 8];
                    output.read(&mut bytes).unwrap();
                    assert_eq!(bytes, expected.unwrap(), "{context}");
                }
            }
        }
    }
}

#[test]
fn pure_suffix_producer_still_exposes_projection() {
    exercise("a[i] + b[i]", [12, -15, 21, 28], [3, 3, 3, 4], &[]);
}

#[test]
fn cpu_dynamic_slices_clamp_instead_of_introducing_point_failures() {
    exercise(
        "reduce(a[b[i]:b[i]+1],0,sum,ordered=true)",
        [12, -15, 21, 28],
        [-10, 10, -1, 3],
        &[],
    );
    exercise(
        "reduce(a[b[i]:a[i]],0,sum,ordered=true)",
        [-4, 9, 1, 7],
        [2, -7, 3, -1],
        &[],
    );
}

#[test]
fn cpu_projection_preserves_dynamic_point_failures_in_omitted_prefix() {
    exercise(
        "a[b[i]]",
        [12, -15, 21, 28],
        [0, 3, 2, 1],
        &[(12, -1), (12, 4), (12, i32::MAX)],
    );
}

#[test]
fn cpu_projection_preserves_integer_division_failures_in_omitted_prefix() {
    for operation in ["/", "%"] {
        exercise(
            &format!("a[i] {operation} b[i]"),
            [12, -15, 21, 28],
            [3, 3, 3, 4],
            &[(12, 0), (i32::MIN, -1)],
        );
    }
}

#[test]
fn cpu_projection_preserves_shift_failures_in_omitted_prefix() {
    for operation in ["<<", ">>"] {
        exercise(
            &format!("a[i] {operation} b[i]"),
            [12, -15, 21, 28],
            [0, 1, 2, 3],
            &[(12, -1), (12, 32), (12, i32::MAX)],
        );
    }
}

fn viewed_fold_input_is_a_snapshot(device: Device, candidate: Candidate) {
    for source in ["state", "state[:,:]"] {
        let program = compile(
            &[SourceFile {
                path: "fold_snapshot.seismic.portable".into(),
                scope: Scope::Portable,
                text: r#"
fn add[M,N](left:tile[M,N] f32,right:tile[M,N] f32,out:tile[M,N] f32):
  for i,j in owned(out): out[i,j] = left[i,j] + right[i,j]
fn accumulate[M,N](state:tile[M,N] f32,input:tile[N] f32,out:tile[M,N] f32):
  for i,j in owned(out): out[i,j] = state[i,j] + input[j]
fn evaluate(x:tensor[4,1] f32,out:tensor[4,1] f32):
  state = load(x)
  zero = tile[4,1] f32
  for i,j in owned(zero): zero[i,j] = 0.0
  reduce((INPUT,),0,add,into=(state,),step=accumulate,identity=(zero,),ordered=true)
  store(state,out)
"#
                .replace("INPUT", source),
            }],
            &[],
        )
        .unwrap();
        let input = device
            .buffer_from(
                &(1..=4)
                    .flat_map(|v| (v as f32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let expected = (11..=14)
            .flat_map(|v| (v as f32).to_le_bytes())
            .collect::<Vec<_>>();
        for capacity in 1..=4 {
            let lowered = lower_selected(
                &program,
                "evaluate",
                device.backend(),
                &Default::default(),
                &Default::default(),
                &Options {
                    piece: Some(capacity),
                    ..Default::default()
                },
                &mut |decision| Ok(decision.alternatives.get(0).unwrap()),
            )
            .unwrap();
            assert!(
                lowered.decisions.iter().any(|decision| {
                    matches!(
                        decision.domain.kind,
                        DecisionKind::Stream { maximum: 4, .. }
                    ) && decision.selected == Alternative::StreamCapacity(capacity)
                }),
                "missing fold decomposition for capacity {capacity}"
            );
            let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
            let output = device.buffer(16).unwrap();
            kernel
                .execute(&[input.clone(), output.clone()], &[])
                .unwrap();
            let mut actual = [0; 16];
            output.read(&mut actual).unwrap();
            assert_eq!(
                actual.as_slice(),
                expected,
                "source {source}, capacity {capacity}"
            );
        }
    }
}

#[test]
fn cpu_fold_snapshots_view_input_before_mutating_its_state() {
    viewed_fold_input_is_a_snapshot(
        Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

fn backend_cases(device: Device, candidate: Candidate) {
    exercise_on(
        &device,
        &candidate,
        "a[i] + b[i]",
        [12, -15, 21, 28],
        [3, 3, 3, 4],
        &[],
    );
    exercise_on(
        &device,
        &candidate,
        "a[b[i]]",
        [12, -15, 21, 28],
        [0, 3, 2, 1],
        &[(12, -1), (12, 4)],
    );
    for operation in ["/", "%"] {
        exercise_on(
            &device,
            &candidate,
            &format!("a[i] {operation} b[i]"),
            [12, -15, 21, 28],
            [3, 3, 3, 4],
            &[(12, 0), (i32::MIN, -1)],
        );
    }
    for operation in ["<<", ">>"] {
        exercise_on(
            &device,
            &candidate,
            &format!("a[i] {operation} b[i]"),
            [12, -15, 21, 28],
            [0, 1, 2, 3],
            &[(12, -1), (12, 32)],
        );
    }
    exercise_on(
        &device,
        &candidate,
        "reduce(a[b[i]:b[i]+1],0,sum,ordered=true)",
        [12, -15, 21, 28],
        [-10, 10, -1, 3],
        &[],
    );
    exercise_on(
        &device,
        &candidate,
        "reduce(a[b[i]:a[i]],0,sum,ordered=true)",
        [-4, 9, 1, 7],
        [2, -7, 3, -1],
        &[],
    );
    viewed_fold_input_is_a_snapshot(device, candidate);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_projection_preserves_failures_and_clamped_windows() {
    backend_cases(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::Sequential,
                loads: LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_projection_preserves_failures_and_clamped_windows() {
    backend_cases(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
