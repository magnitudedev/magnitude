use seismic_lang::{
    interp::{Arg, Interpreter, TensorData},
    program::{compile, SourceFile},
    types::DType,
    Scope,
};
use seismic_runtime::Device;
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;
use std::collections::HashMap;
fn exercise(device: Device) {
    for (dtype, name, values) in [
        (
            DType::U32,
            "u32",
            vec![0u32, 1, 0x80000000, 0xffffffff, 0x76543210],
        ),
        (
            DType::I32,
            "i32",
            vec![0u32, 1, 0x80000000, 0xffffffff, 0x76543210],
        ),
    ] {
        let text=format!("fn shift(x: tensor[5] {name}, out: tensor[5] {name}, amount: i32):\n  for row in parallel:\n    a = load(x[row:row+1])\n    z = tile[1] {name}\n    for i in owned(z): z[i] = a[i] >> amount\n    store(z,out[row:row+1])\n");
        let program = compile(
            &[SourceFile {
                path: "shift.seismic.portable".into(),
                scope: Scope::Portable,
                text,
            }],
            &[],
        )
        .unwrap();
        let invocation = seismic_runtime::tuner::Input::Portable {
            program: &program,
            entry: "shift",
            shapes: &Default::default(),
            elements: &Default::default(),
            options: &Default::default(),
        };
        let input = device
            .buffer_from(
                &values
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let out = device.buffer(20).unwrap();
        let mut kernel =
            automatic_hardware::compile(&device, invocation, &[input.clone(), out.clone()], &[0.0])
                .unwrap();
        for amount in [0, 1, 15, 31, -1, 32, 255, 3] {
            let mut interpreter = Interpreter::new(&program);
            let source = interpreter.add_tensor(TensorData::dense(
                dtype,
                vec![5],
                values
                    .iter()
                    .map(|n| {
                        if dtype == DType::I32 {
                            (*n as i32) as f64
                        } else {
                            *n as f64
                        }
                    })
                    .collect(),
            ));
            let target = interpreter.add_tensor(TensorData::dense(dtype, vec![5], vec![0.; 5]));
            let reference = interpreter.run(
                "shift",
                &[
                    Arg::Tensor(source),
                    Arg::Tensor(target),
                    Arg::Scalar(amount as f64),
                ],
                &HashMap::new(),
            );
            if (0..32).contains(&amount) {
                kernel = automatic_hardware::compile(
                    &device,
                    invocation,
                    &[input.clone(), out.clone()],
                    &[f64::from(amount)],
                )
                .unwrap();
            }
            let result = kernel.execute(&[input.clone(), out.clone()], &[amount as f64]);
            if !(0..32).contains(&amount) {
                assert!(result.is_err());
                assert!(reference.is_err());
                continue;
            }
            result.unwrap();
            reference.unwrap();
            let expected = values
                .iter()
                .flat_map(|n| {
                    if dtype == DType::I32 {
                        ((*n as i32) >> amount).to_le_bytes()
                    } else {
                        (n >> amount).to_le_bytes()
                    }
                })
                .collect::<Vec<_>>();
            let mut actual = vec![0; 20];
            out.read(&mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }
}
#[test]
fn cpu_shifts() {
    exercise(Device::cpu());
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_shifts() {
    exercise(Device::metal().unwrap());
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_shifts() {
    exercise(Device::cuda(0).unwrap());
}
