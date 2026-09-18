//! Equivalent view spelling must preserve the bounded producer choice family.
use seismic_lang::{
    Scope,
    interp::round_to,
    ir::{ExprKind, Stmt, StmtKind},
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind},
    program::{SourceFile, compile},
    types::DType,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};

struct ViewCase {
    name: &'static str,
    source: &'static str,
    output_shape: &'static str,
    coordinates: &'static [(usize, usize)],
}
const RECTANGLE: &[(usize, usize)] = &[(1, 2), (1, 3), (2, 2), (2, 3)];
const CASES: &[ViewCase] = &[
    ViewCase {
        name: "direct rectangle",
        source: "  selected = s[1:3,2:4]\n",
        output_shape: "2,2",
        coordinates: RECTANGLE,
    },
    ViewCase {
        name: "successive slice aliases",
        source: "  rows = s[1:3,:]\n  selected = rows[:,2:4]\n",
        output_shape: "2,2",
        coordinates: RECTANGLE,
    },
    ViewCase {
        name: "nested slices",
        source: "  selected = s[1:4,1:5][0:2,1:3]\n",
        output_shape: "2,2",
        coordinates: RECTANGLE,
    },
    ViewCase {
        name: "transposed rectangle",
        source: "  transposed = s.T\n  selected = transposed[2:4,1:3]\n",
        output_shape: "2,2",
        coordinates: &[(1, 2), (2, 2), (1, 3), (2, 3)],
    },
    ViewCase {
        name: "point-index rank reduction",
        source: "  selected = s[2,1:4]\n",
        output_shape: "3",
        coordinates: &[(2, 1), (2, 2), (2, 3)],
    },
    ViewCase {
        name: "point-index alias then slice",
        source: "  row_view = s[2,:]\n  selected = row_view[1:4]\n",
        output_shape: "3",
        coordinates: &[(2, 1), (2, 2), (2, 3)],
    },
];

fn full_intermediate(body: &[Stmt]) -> bool {
    body.iter().any(|s| match &s.kind {
        StmtKind::Assign { value, .. } => matches!(
            &value.kind,
            ExprKind::TileAlloc { shape, .. }
                if shape.iter().map(|n| n.as_constant()).collect::<Vec<_>>() == [Some(4), Some(5)]
        ),
        StmtKind::Owned { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Parallel { body, .. }
        | StmtKind::LoadLoop { body, .. }
        | StmtKind::Lanes { body, .. } => full_intermediate(body),
        StmtKind::If { then, els, .. } => full_intermediate(then) || full_intermediate(els),
        StmtKind::Reduction(r) => r.bodies().any(|b| full_intermediate(b)),
        _ => false,
    })
}

fn exercise(device: Device, candidate: Candidate) {
    let values: Vec<f32> = (0..4)
        .flat_map(|row| {
            [1000.25f32, 0.0003, -1000.125, 1.0001, -0.3333].map(|x| x + row as f32 * 0.0625)
        })
        .collect();
    for case in CASES {
        let source = format!(
            "fn evaluate(x:tensor[4,5] f32,out:tensor[{}] f32):\n  t = load(x)\n  s = tile[4,5] f32\n  for i,j in owned(s): s[i,j] = f32(i-j)*0.125\n  for iteration in range(2):\n    for i,j in owned(s):\n      row = t[i,:]\n      total = reduce(row,0,sum,ordered=true)\n      s[i,j] = f32(f16(total + f32(i*11+j))) + s[i,j]*0.5\n{}  store(selected,out)\n",
            case.output_shape, case.source
        );
        let program = compile(
            &[SourceFile {
                path: "region_views.seismic.portable".into(),
                scope: Scope::Portable,
                text: source,
            }],
            &[],
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        let expected: Vec<u8> = case
            .coordinates
            .iter()
            .flat_map(|&(row, col)| {
                let total = values[row * 5..row * 5 + 5]
                    .iter()
                    .fold(0f32, |sum, &value| sum + value);
                let rounded = round_to(DType::F16, (total + (row * 11 + col) as f32) as f64) as f32;
                let mut state = (row as i32 - col as i32) as f32 * 0.125;
                for _ in 0..2 {
                    state = rounded + state * 0.5;
                }
                state.to_le_bytes()
            })
            .collect();
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
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            assert!(offered > 0, "{} lost the producer choice", case.name);
            assert_eq!(
                full_intermediate(&function.body),
                !recompute,
                "{}: full source allocation, recompute={recompute}",
                case.name
            );
            assert!(function.decisions.iter().any(|record| {
                matches!(record.domain.kind, DecisionKind::Producer { .. })
                    && record.selected
                        == if recompute {
                            Alternative::Recompute
                        } else {
                            Alternative::Materialize
                        }
            }));
            let mut kernel = device
                .compile(&function, candidate.clone())
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            let input = device
                .buffer_from(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let output = device.buffer(expected.len()).unwrap();
            kernel.execute(&[input, output.clone()], &[]).unwrap();
            let mut actual = vec![0; expected.len()];
            output.read(&mut actual).unwrap();
            assert_eq!(actual, expected, "{}, recompute={recompute}", case.name);
        }
    }
}

#[test]
fn cpu_equivalent_views_preserve_region_choices_and_values() {
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
fn metal_equivalent_views_preserve_region_choices_and_values() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
