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
const SOURCE: &str = "
fn row_read(x: tensor[3, 4] f32, which: i32, out: tensor[1, 4] f32):
  for row in parallel:
    t = load(x[which])
    store(t, out[row])

fn element_read(x: tensor[12] f32, which: i32, out: tensor[1] f32):
  for row in parallel:
    y = tile[1] f32
    for i in owned(y): y[i] = x[which]
    store(y,out[row:row+1])
";
fn exercise(device: Device) {
    let program = compile(
        &[SourceFile {
            path: "point.seismic.portable".into(),
            scope: Scope::Portable,
            text: SOURCE.into(),
        }],
        &[],
    )
    .unwrap();
    for (name, width, limit, shape) in [
        ("row_read", 4, 3, vec![3, 4]),
        ("element_read", 1, 12, vec![12]),
    ] {
        let invocation = seismic_runtime::tuner::Input::Portable {
            program: &program,
            entry: name,
            shapes: &Default::default(),
            elements: &Default::default(),
            options: &Default::default(),
        };
        let input = device
            .buffer_from(
                &(0..12)
                    .flat_map(|i| (i as f32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.buffer(width * 4).unwrap();
        let mut kernel =
            automatic_hardware::compile(&device, invocation, &[input.clone(), output.clone()], &[0.0])
                .unwrap();
        for which in [1, -1, limit, 0, i32::MAX, limit - 1] {
            if (0..limit).contains(&which) {
                kernel = automatic_hardware::compile(
                    &device,
                    invocation,
                    &[input.clone(), output.clone()],
                    &[f64::from(which)],
                )
                .unwrap();
            }
            let result = kernel.execute(&[input.clone(), output.clone()], &[f64::from(which)]);
            let mut interpreter = Interpreter::new(&program);
            let source = interpreter.add_tensor(TensorData::dense(
                DType::F32,
                shape.clone(),
                (0..12).map(f64::from).collect(),
            ));
            let out = interpreter.add_tensor(TensorData::dense(
                DType::F32,
                if width == 4 { vec![1, 4] } else { vec![1] },
                vec![0.; width],
            ));
            let reference = interpreter.run(
                name,
                &[
                    Arg::Tensor(source),
                    Arg::Scalar(f64::from(which)),
                    Arg::Tensor(out),
                ],
                &HashMap::new(),
            );
            if which < 0 || which >= limit {
                assert!(result.is_err(), "{name}: {which}");
                assert!(reference.is_err());
            } else {
                result.unwrap();
                reference.unwrap();
                let mut actual = vec![0; width * 4];
                output.read(&mut actual).unwrap();
                let expected = (which as usize * width..(which as usize + 1) * width)
                    .flat_map(|i| (i as f32).to_le_bytes())
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{name}: {which}");
            }
        }
    }
    let unsafe_parallel="fn scatter(x: tensor[1] f32, which: i32, out: tensor[4] f32):\n  for row in parallel:\n    t = load(x)\n    store(t,out[which:which+1])\n";
    assert!(compile(
        &[SourceFile {
            path: "scatter.seismic.portable".into(),
            scope: Scope::Portable,
            text: unsafe_parallel.into()
        }],
        &[]
    )
    .is_err());
}
#[test]
fn cpu_point_index() {
    exercise(Device::cpu());
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_point_index() {
    exercise(Device::metal().unwrap());
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_point_index() {
    exercise(Device::cuda(0).unwrap());
}
