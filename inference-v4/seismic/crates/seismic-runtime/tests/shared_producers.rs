//! Sharing prepared values preserves snapshot versions and stored precision.
use seismic_lang::{
    Scope,
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind},
    program::{SourceFile, compile},
    types::{DType, Elem},
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};

fn exercise(device: Device, candidate: Candidate) {
    for scenario in [
        "shared",
        "source_mutated",
        "result_mutated",
        "escaping_scalar",
    ] {
        let before = if scenario == "source_mutated" {
            "  for i in owned(a): a[i] = a[i] + 0.125\n"
        } else {
            ""
        };
        let after = if scenario == "result_mutated" {
            "  for i in owned(left): left[i] = f16(f32(left[i]) + 0.25)\n"
        } else {
            ""
        };
        let second = if scenario == "escaping_scalar" {
            "  for i in owned(right):\n    last = a[i] * 1.0003\n    right[i] = f16(last)\n"
        } else {
            "  prepare(a,right)\n"
        };
        let extra = if scenario == "escaping_scalar" {
            " + last"
        } else {
            ""
        };
        let source = format!(
            r#"
fn prepare[N](a:tile[N] f32,b:tile[N] f16):
  for i in owned(b):
    intermediate = a[i] * 1.0003
    b[i] = f16(intermediate)
fn evaluate(x:tensor[65] f32,out:tensor[65] f32):
  a = load(x)
  last = 0.0
  left = tile[65] f16
  for i in owned(left): left[i] = f16(a[i] * 1.0003)
{before}  right = tile[65] f16
{second}{after}  result = tile[65] f32
  for i in owned(result): result[i] = f32(left[i]) + f32(right[i]){extra}
  store(result,out)
"#
        );
        let p = compile(
            &[SourceFile {
                path: "prepared.seismic.portable".into(),
                scope: Scope::Portable,
                text: source,
            }],
            &[],
        )
        .unwrap();
        let mut reference = None;
        for share in [false, true] {
            let mut domains = 0;
            let ir = lower_selected(
                &p,
                "evaluate",
                device.backend(),
                &Default::default(),
                &Default::default(),
                &Options::default(),
                &mut |d| {
                    Ok(
                        if matches!(d.kind, DecisionKind::Intermediate { .. })
                            && d.alternatives.contains(&Alternative::RetainLocal)
                        {
                            domains += 1;
                            if share {
                                Alternative::RetainLocal
                            } else {
                                Alternative::Materialize
                            }
                        } else {
                            d.alternatives.get(0).unwrap()
                        },
                    )
                },
            )
            .unwrap();
            assert_eq!(
                domains > 0,
                scenario == "shared",
                "{scenario}: wrong sharing domain"
            );
            let mut allocations = 0;
            fn count(body: &[seismic_lang::ir::Stmt], total: &mut usize) {
                use seismic_lang::ir::{ExprKind, StmtKind};
                for s in body {
                    match &s.kind {
                        StmtKind::Assign { value, .. }
                            if matches!(
                                &value.kind,
                                ExprKind::TileAlloc {
                                    dtype: Elem::Dtype(DType::F16),
                                    ..
                                }
                            ) =>
                        {
                            *total += 1
                        }
                        StmtKind::Range { body, .. }
                        | StmtKind::Owned { body, .. }
                        | StmtKind::Parallel { body, .. }
                        | StmtKind::LoadLoop { body, .. }
                        | StmtKind::Lanes { body, .. } => count(body, total),
                        StmtKind::If { then, els, .. } => {
                            count(then, total);
                            count(els, total)
                        }
                        _ => {}
                    }
                }
            }
            count(&ir.body, &mut allocations);
            assert_eq!(
                allocations,
                if share && scenario == "shared" { 1 } else { 2 }
            );
            let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
            let input = (0..65)
                .map(|i| (i as f32 - 32.0) / 19.0)
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let x = device.buffer_from(&input).unwrap();
            let out = device.buffer(65 * 4).unwrap();
            kernel.execute(&[x, out.clone()], &[]).unwrap();
            let mut bytes = vec![0; 65 * 4];
            out.read(&mut bytes).unwrap();
            if let Some(expected) = &reference {
                assert_eq!(
                    &bytes, expected,
                    "{scenario}: sharing changed snapshot values"
                );
            } else {
                reference = Some(bytes);
            }
        }
    }
}

#[test]
fn cpu_prepared_value_sharing_respects_snapshots_rounding_and_scalar_escape() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_prepared_value_sharing_respects_snapshots_rounding_and_scalar_escape() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}

fn guarded(device: Device, candidate: Candidate) {
    for scenario in ["shared", "changed_input", "changed_guard", "different_false"] {
        let between = match scenario {
            "changed_input" => "  for i in owned(a): a[i]=a[i]+0.25\n",
            "changed_guard" => "  gate=gate-1\n",
            _ => "",
        };
        let false_value = if scenario == "different_false" { "-0.0" } else { "0.0" };
        let source = format!("fn evaluate(x:tensor[5] f32,out:tensor[8] f32,limit:i32):\n  a=load(x)\n  gate=limit\n  left=tile[8] f16\n  for i in owned(left):\n    if i<5 and i<gate: left[i]=f16(a[i]*1.0003)\n    else: left[i]=0.0\n{between}  right=tile[8] f16\n  for j in owned(right):\n    if j<5 and j<gate:\n      converted=f16(a[j]*1.0003)\n      right[j]=converted\n    else: right[j]={false_value}\n  result=tile[8] f32\n  for i in owned(result): result[i]=f32(left[i])+f32(right[i])\n  store(result,out)\n");
        let p = compile(&[SourceFile { path: "guarded_producer.seismic.portable".into(), scope: Scope::Portable, text: source }], &[]).unwrap();
        let input = device.buffer_from(&(0..5).flat_map(|i| (i as f32 / 7.0 - 0.25).to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let out = device.buffer(32).unwrap();
        let mut expected = Vec::new();
        for share in [false, true] {
            let mut domains = 0;
            let ir = lower_selected(&p, "evaluate", device.backend(), &Default::default(), &Default::default(), &Options::default(), &mut |decision| {
                Ok(if matches!(decision.kind, DecisionKind::Intermediate { .. }) && decision.alternatives.contains(&Alternative::RetainLocal) {
                    domains += 1;
                    if share { Alternative::RetainLocal } else { Alternative::Materialize }
                } else { decision.alternatives.get(0).unwrap() })
            }).unwrap();
            assert_eq!(domains > 0, scenario == "shared", "{scenario}");
            let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
            for (at, limit) in [-1.0, 0.0, 3.0, 8.0].into_iter().enumerate() {
                kernel.execute(&[input.clone(), out.clone()], &[limit]).unwrap();
                let mut actual = vec![0; 32]; out.read(&mut actual).unwrap();
                if share { assert_eq!(actual, expected[at], "{scenario} limit {limit}"); }
                else { expected.push(actual); }
            }
        }
    }
}

#[test]
fn cpu_conditional_preparation_shares_complete_lazy_branches_only() {
    guarded(Device::cpu(), Candidate::Cpu { loads: LoadStrategy::Materialize });
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_conditional_preparation_shares_complete_lazy_branches_only() {
    guarded(Device::metal().unwrap(), Candidate::Metal(Default::default()));
}
