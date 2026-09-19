use seismic_lang::{
    lower::{lower_specialized, Options},
    program::{compile, SourceFile},
    types::{DType, Elem},
    Scope,
};
use seismic_runtime::Device;
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;
use std::collections::HashMap;
fn exercise(device: Device) {
    let source = "fn leaf[N](x: tensor[N] T, out: tensor[N] U):\n  for row in parallel:\n    a = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = f32(a[i]) * 2.0\n    store(y,out[row:row+1])\n\nfn entry[N](x: tensor[N] T, out: tensor[N] f32):\n  leaf(x,out)\n";
    let source = format!("{source}\nfn composition(a: tensor[4] f16, b: tensor[4] bf16, out: tensor[8] f32):\n  leaf(a, out[0:4])\n  leaf(b, out[4:8])\n");
    let program = compile(
        &[SourceFile {
            path: "generic.seismic.portable".into(),
            scope: Scope::Portable,
            text: source,
        }],
        &[],
    )
    .unwrap();
    let shapes = HashMap::from([("N".into(), 4)]);
    assert!(lower_specialized(
        &program,
        "entry",
        device.backend(),
        &shapes,
        &HashMap::new(),
        &Options::default()
    )
    .is_err());
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let elements = HashMap::from([("T".into(), Elem::Dtype(dtype))]);
        let invocation = seismic_runtime::tuner::Input::Portable {
            program: &program,
            entry: "entry",
            shapes: &shapes,
            elements: &elements,
            options: &Options::default(),
        };
        let bytes = match dtype {
            DType::F32 => [1f32, -2., 0.5, 4.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
            DType::BF16 => [0x3f80u16, 0xc000, 0x3f00, 0x4080]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            DType::F16 => [0x3c00u16, 0xc000, 0x3800, 0x4400]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            _ => unreachable!(),
        };
        let input = device.buffer_from(&bytes).unwrap();
        let output = device.buffer(16).unwrap();
        let mut kernel =
            automatic_hardware::compile(&device, invocation, &[input.clone(), output.clone()], &[])
                .unwrap();
        assert_eq!(kernel.buffers()[0].bytes, input.len());
        kernel.execute(&[input, output.clone()], &[]).unwrap();
        let mut got = [0u8; 16];
        output.read(&mut got).unwrap();
        assert_eq!(
            &got,
            [2f32, -4., 1., 8.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
    // The same generic body publishes all supported floating output formats.
    // Values straddling narrow-format rounding boundaries must actually round.
    for (dtype, expected) in [
        (
            DType::F32,
            [2.0078125f32, -4.015625, (1.0 + 1.0 / 2048.0), 8.03125]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            DType::BF16,
            [0x4000u16, 0xc080, 0x3f80, 0x4100]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
        (
            DType::F16,
            [0x4004u16, 0xc404, 0x3c00, 0x4804]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
    ] {
        let invocation = seismic_runtime::tuner::Input::Portable {
            program: &program,
            entry: "leaf",
            shapes: &shapes,
            elements: &HashMap::from([
                ("T".into(), Elem::Dtype(DType::F32)),
                ("U".into(), Elem::Dtype(dtype)),
            ]),
            options: &Options::default(),
        };
        let input = device
            .buffer_from(
                &[
                    (1.0f32 + 1.0 / 256.0),
                    -2.0078125,
                    (0.5 + 1.0 / 4096.0),
                    4.015625,
                ]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.buffer(expected.len()).unwrap();
        let mut kernel =
            automatic_hardware::compile(&device, invocation, &[input.clone(), output.clone()], &[])
                .unwrap();
        assert_eq!(kernel.buffers()[0].bytes, input.len());
        kernel.execute(&[input, output.clone()], &[]).unwrap();
        let mut got = vec![0; expected.len()];
        output.read(&mut got).unwrap();
        assert_eq!(got, expected, "generic publication {dtype:?}");
    }
    for target in [Elem::Repr("q4g64".into()), Elem::Dtype(DType::I32)] {
        assert!(lower_specialized(
            &program,
            "leaf",
            device.backend(),
            &HashMap::from([("N".into(), 64)]),
            &HashMap::from([("T".into(), Elem::Dtype(DType::F32)), ("U".into(), target)]),
            &Options::default()
        )
        .is_err());
    }
    struct Bind(HashMap<String, seismic_runtime::Buffer>);
    impl seismic_runtime::plan::Bindings for Bind {
        fn buffer(&self, root: &str, plane: &str) -> Option<&seismic_runtime::Buffer> {
            assert!(plane.is_empty());
            self.0.get(root)
        }
        fn scalar(&self, _: &str) -> Option<f64> {
            None
        }
    }
    let lowered = lower_specialized(
        &program,
        "composition",
        device.backend(),
        &HashMap::new(),
        &HashMap::new(),
        &Options::default(),
    )
    .unwrap();
    let mut compiler = seismic_runtime::plan::PlanCompiler::new(
        &device,
        &program,
        automatic_hardware::settings(&device, &lowered),
    );
    let mut compiled = compiler
        .compile_entry(
            "composition",
            &HashMap::new(),
            &HashMap::new(),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(
        compiled.kernel_count(),
        0,
        "selection waits for actual bindings"
    );
    let a = device
        .buffer_from(
            &[0x3c00u16; 4]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let b = device
        .buffer_from(
            // BF16 3.0 has a different F16 interpretation (2.125), so this
            // catches accidental reuse of the first call's specialization.
            &[0x4040u16; 4]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = device.buffer(32).unwrap();
    compiled
        .execute(&Bind(HashMap::from([
            ("a".into(), a),
            ("b".into(), b),
            ("out".into(), out.clone()),
        ])))
        .unwrap();
    assert_eq!(
        compiled.kernel_count(),
        1,
        "the enclosing program compiles together while retaining each call element type"
    );
    let mut got = [0; 32];
    out.read(&mut got).unwrap();
    assert_eq!(
        &got,
        [2f32, 2., 2., 2., 6., 6., 6., 6.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
            .as_slice()
    );
    for elem in [
        Elem::Param("other".into()),
        Elem::Dtype(DType::I32),
        Elem::Repr("missing".into()),
    ] {
        assert!(lower_specialized(
            &program,
            "entry",
            device.backend(),
            &shapes,
            &HashMap::from([("T".into(), elem)]),
            &Options::default()
        )
        .is_err());
    }
}
#[test]
fn cpu_generic() {
    exercise(Device::cpu());
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_generic() {
    exercise(Device::metal().unwrap());
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_generic() {
    exercise(Device::cuda(0).unwrap());
}
