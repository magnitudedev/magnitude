//! Proven reshape dimensions still evaluate their source expressions and guards.
use seismic_lang::{
    Scope,
    interp::{Arg, Interpreter, TensorData},
    lower::lower,
    program::{Program, SourceFile, compile},
    types::DType,
};
use seismic_realization::{Dispatch, LoadStrategy, ScalarOptions};
use seismic_runtime::{Candidate, Device};

#[derive(Clone, Copy, Debug)]
enum Use {
    Extent,
    Load,
    Indexed,
    Nested,
    Captured,
    Empty,
}
impl Use {
    fn all() -> [Self; 6] {
        [
            Self::Extent,
            Self::Load,
            Self::Indexed,
            Self::Nested,
            Self::Captured,
            Self::Empty,
        ]
    }
    fn program(self) -> Program {
        let reshape = "reshape(x,(extent(x[controls[0],:],0),extent(x[controls[1],:],0)/2))";
        let (prepare, value) = match self {
            Self::Extent => (String::new(),format!("extent({reshape},0)")),
            Self::Load => (format!("  captured = load({reshape})\n"),"captured[0,0]".into()),
            Self::Indexed => (String::new(),format!("{reshape}[controls[2],0]")),
            Self::Nested => (String::new(),"reshape(x[controls[0],:],(extent(x[controls[1],:],0)/2,2))[controls[2],1]".into()),
            Self::Captured => (format!("  captured = {reshape}\n  changed = tile[3] i32\n  for j in owned(changed): changed[j] = -1\n  store(changed,controls)\n"),"extent(captured,0)".into()),
            Self::Empty => ("  captured = load(reshape(x[:0,:],(0,extent(x[controls[0],:],0)+extent(x[controls[1],:],0)-4)))\n".into(),"extent(captured,0)".into()),
        };
        let text = format!(
            "fn evaluate(x:tensor[2,4] i32,controls:tensor[3] i32,out:tensor[1] i32):\n{prepare}  result = tile[1] i32\n  for i in owned(result): result[i] = {value}\n  store(result,out)\n"
        );
        compile(
            &[SourceFile {
                path: "reshape_guards.seismic.portable".into(),
                scope: Scope::Portable,
                text,
            }],
            &[],
        )
        .unwrap()
    }
    fn expected(self, controls: [i32; 3]) -> Option<i32> {
        if !(0..2).contains(&controls[0]) || !(0..2).contains(&controls[1]) {
            return None;
        }
        Some(match self {
            Self::Extent | Self::Captured => 4,
            Self::Load => 11,
            Self::Indexed => {
                if !(0..4).contains(&controls[2]) {
                    return None;
                }
                11 + controls[2] * 2
            }
            Self::Nested => {
                if !(0..2).contains(&controls[2]) {
                    return None;
                }
                12 + controls[0] * 4 + controls[2] * 2
            }
            Self::Empty => 0,
        })
    }
}
const CASES: [[i32; 3]; 7] = [
    [0, 1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [1, 2, 1],
    [0, 0, 9],
    [-1, 2, 9],
    [0, 0, 0],
];
fn encode(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn exercise(device: &Device, candidate: &Candidate) {
    for usage in Use::all() {
        let ir = lower(
            &usage.program(),
            "evaluate",
            device.backend(),
            &Default::default(),
        )
        .unwrap();
        let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
        let input = device
            .buffer_from(&encode(&(11..19).collect::<Vec<_>>()))
            .unwrap();
        let controls = device.buffer(12).unwrap();
        let out = device.buffer(4).unwrap();
        for values in CASES {
            controls.write(&encode(&values)).unwrap();
            let result = kernel.execute(&[input.clone(), controls.clone(), out.clone()], &[]);
            let expected = usage.expected(values);
            assert_eq!(
                result.is_ok(),
                expected.is_some(),
                "{usage:?} {values:?}: {result:?}"
            );
            // A failed call need not roll back prior effects; the next valid
            // invocation must still recover and produce its defined result.
            if let Some(expected) = expected {
                let mut bytes = [0; 4];
                out.read(&mut bytes).unwrap();
                assert_eq!(i32::from_le_bytes(bytes), expected, "{usage:?} {values:?}");
            }
        }
    }
}

#[test]
fn interpreter_reshape_dimensions_evaluate_nested_guards() {
    for usage in Use::all() {
        let program = usage.program();
        let mut interpreter = Interpreter::new(&program);
        let input = interpreter.add_tensor(TensorData::dense(
            DType::I32,
            vec![2, 4],
            (11..19).map(f64::from).collect(),
        ));
        let controls = interpreter.add_tensor(TensorData::dense(DType::I32, vec![3], vec![0.0; 3]));
        let out = interpreter.add_tensor(TensorData::dense(DType::I32, vec![1], vec![0.0]));
        for values in CASES {
            for (i, value) in values.iter().enumerate() {
                interpreter.tensors[controls].set(i, f64::from(*value));
            }
            let result = interpreter.run(
                "evaluate",
                &[Arg::Tensor(input), Arg::Tensor(controls), Arg::Tensor(out)],
                &Default::default(),
            );
            let expected = usage.expected(values);
            assert_eq!(
                result.is_ok(),
                expected.is_some(),
                "{usage:?} {values:?}: {result:?}"
            );
            if let Some(expected) = expected {
                assert_eq!(interpreter.tensors[out].get(0) as i32, expected);
            }
        }
    }
}
#[test]
fn cpu_reshape_dimensions_preserve_guards_data_and_captured_views() {
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        exercise(&Device::cpu(), &Candidate::Cpu { loads });
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_reshape_dimensions_preserve_guards_data_and_captured_views() {
    let device = Device::metal().unwrap();
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        exercise(
            &device,
            &Candidate::Metal(seismic_metal::execution::Config {
                loads,
                ..Default::default()
            }),
        );
    }
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_reshape_dimensions_preserve_guards_data_and_captured_views() {
    let device = Device::cuda(0).unwrap();
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        exercise(
            &device,
            &Candidate::Cuda {
                options: ScalarOptions {
                    dispatch: Dispatch::Sequential,
                    loads,
                },
                threads_per_block: 32,
            },
        );
    }
}
