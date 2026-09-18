//! Observable state outside a projected producer survives nested lexical scopes.
use seismic_lang::{
    Scope,
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind},
    program::{SourceFile, compile},
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};

fn exercise(device: Device, candidate: Candidate) {
    for control in ["for outer in range(1):", "if true:"] {
        let source = format!(
            "fn evaluate(out:tensor[2] f32,observed:tensor[1] f32):\n  counter = 11.0\n  s = tile[4] f32\n  for init in owned(s): s[init] = 0.0\n  {control}\n    for i in owned(s):\n      counter = f32(i)\n      s[i] = f32(i) * 2.0\n  selected = s[1:3]\n  observation = tile[1] f32\n  for j in owned(observation): observation[j] = counter\n  store(selected,out)\n  store(observation,observed)\n"
        );
        let program = compile(
            &[SourceFile {
                path: "region_effects.seismic.portable".into(),
                scope: Scope::Portable,
                text: source,
            }],
            &[],
        )
        .unwrap();
        for recompute in [false, true] {
            let mut offered = 0;
            let function = lower_selected(
                &program,
                "evaluate",
                device.backend(),
                &Default::default(),
                &Default::default(),
                &Options::default(),
                &mut |decision| {
                    if matches!(decision.kind, DecisionKind::Producer { .. })
                        && decision.alternatives.contains(&Alternative::Recompute)
                    {
                        offered += 1;
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
            .unwrap();
            assert!(offered > 0, "the public lowering must exercise projection");
            let mut kernel = device.compile(&function, candidate.clone()).unwrap();
            let output = device.buffer(8).unwrap();
            let observed = device.buffer(4).unwrap();
            kernel
                .execute(&[output.clone(), observed.clone()], &[])
                .unwrap();
            let mut values = [0; 8];
            output.read(&mut values).unwrap();
            assert_eq!(
                values,
                [2f32.to_le_bytes(), 4f32.to_le_bytes()].concat().as_slice(),
                "{control}, recompute={recompute}"
            );
            let mut value = [0; 4];
            observed.read(&mut value).unwrap();
            assert_eq!(
                f32::from_le_bytes(value),
                3.0,
                "escaping counter: {control}, recompute={recompute}"
            );
        }
    }
}

#[test]
fn cpu_nested_projected_region_preserves_escaping_scalar_state() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_nested_projected_region_preserves_escaping_scalar_state() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_nested_projected_region_preserves_escaping_scalar_state() {
    exercise(
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
