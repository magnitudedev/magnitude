use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::Device;
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;

fn exercise(device: Device) {
    for (dtype, lo, hi, cases) in [
        (
            "i32",
            i64::from(i32::MIN),
            i64::from(i32::MAX),
            vec![
                vec![],
                vec![i64::from(i32::MAX), 1, -1],
                vec![i64::from(i32::MIN), -1, 1],
                vec![-5, 8, -2],
                [vec![i64::from(i32::MAX), 1, -1], vec![0; 62]].concat(),
            ],
        ),
        (
            "u32",
            0,
            i64::from(u32::MAX),
            vec![vec![], vec![i64::from(u32::MAX), 1, 7], vec![0, 1, 7]],
        ),
        (
            "bool",
            0,
            1,
            vec![vec![], vec![0, 0, 0], vec![1, 1, 1], vec![0, 1, 0]],
        ),
    ] {
        for values in cases {
            for operation in ["sum", "min", "max"] {
                let expected = match operation {
                    "sum" => values.iter().fold(0i64, |a, b| (a + b).clamp(lo, hi)),
                    "min" => values.iter().copied().fold(hi, i64::min),
                    _ => values.iter().copied().fold(lo, i64::max),
                };
                let source = format!("fn evaluate(x: tensor[{}] {dtype}, out: tensor[1] {dtype}):\n  a = load(x)\n  value = reduce(a,0,{operation})\n  y = tile[1] {dtype}\n  for i in owned(y): y[i] = value\n  store(y,out)\n", values.len());
                let program = compile(
                    &[SourceFile {
                        path: "typed-reduction.seismic.portable".into(),
                        text: source,
                        scope: Scope::Portable,
                    }],
                    &[],
                )
                .unwrap();
                let ty = match dtype {
                    "i32" => seismic_lang::types::DType::I32,
                    "u32" => seismic_lang::types::DType::U32,
                    _ => seismic_lang::types::DType::Bool,
                };
                let mut interpreter = seismic_lang::interp::Interpreter::new(&program);
                let source = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    ty,
                    vec![values.len()],
                    values.iter().map(|v| *v as f64).collect(),
                ));
                let target = interpreter.add_tensor(seismic_lang::interp::TensorData::dense(
                    ty,
                    vec![1],
                    vec![0.0],
                ));
                interpreter
                    .run(
                        "evaluate",
                        &[
                            seismic_lang::interp::Arg::Tensor(source),
                            seismic_lang::interp::Arg::Tensor(target),
                        ],
                        &Default::default(),
                    )
                    .unwrap();
                assert_eq!(
                    interpreter.tensors[target].get(0),
                    expected as f64,
                    "reference {dtype} {operation} {values:?}"
                );
                let invocation = seismic_runtime::tuner::Input::Portable {
                    program: &program,
                    entry: "evaluate",
                    shapes: &Default::default(),
                    elements: &Default::default(),
                    options: &Default::default(),
                };
                let mut bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|&v| {
                        if dtype == "bool" {
                            vec![v as u8]
                        } else {
                            (v as u32).to_le_bytes().to_vec()
                        }
                    })
                    .collect();
                if bytes.is_empty() {
                    bytes.resize(4, 0);
                }
                let input = device.buffer_from(&bytes).unwrap();
                let output = device.buffer(4).unwrap();
                let mut kernel = automatic_hardware::compile(
                    &device,
                    invocation,
                    &[input.clone(), output.clone()],
                    &[],
                )
                .unwrap();
                kernel.execute(&[input, output.clone()], &[]).unwrap();
                let mut actual = vec![0; if dtype == "bool" { 1 } else { 4 }];
                output.read(&mut actual).unwrap();
                let actual = if dtype == "bool" {
                    i64::from(actual[0])
                } else if dtype == "i32" {
                    i64::from(i32::from_le_bytes(actual.try_into().unwrap()))
                } else {
                    i64::from(u32::from_le_bytes(actual.try_into().unwrap()))
                };
                assert_eq!(actual, expected, "{dtype} {operation} {values:?}");
            }
        }
    }
}
#[test]
fn cpu_typed_reductions() {
    exercise(Device::cpu());
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_typed_reductions() {
    exercise(Device::metal().unwrap());
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_typed_reductions() {
    exercise(Device::cuda(0).unwrap());
}
