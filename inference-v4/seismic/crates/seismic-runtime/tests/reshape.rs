use seismic_lang::{
    lower::{lower, Options},
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{
    plan::{Bindings, CompiledPlan},
    Buffer, Candidate, Device,
};
use std::collections::HashMap;
const SOURCE: &str = "
fn read_shape[M, N](x: tensor[M * N] f32, out: tensor[M, N] f32):
  for row in parallel:
    t = load(reshape(x, (M, N))[row])
    store(t, out[row])

fn write_shape[M, N](x: tensor[M, N] f32, out: tensor[M * N] f32):
  for row in parallel:
    t = load(x[row])
    store(t, reshape(out, (M, N))[row])

fn snapshot(x: tensor[4] f32, out: tensor[1, 4] f32):
  for row in parallel:
    old = load(reshape(x, (1, 4))[row])
    zero = tile[4] f32
    for i in owned(zero): zero[i] = 0.0
    store(zero, reshape(x, (1, 4))[row])
    store(old, out[row])

fn composition(x: tensor[20] f32, middle: tensor[3, 4] f32, out: tensor[20] f32):
  read_shape(x[4:16], middle)
  write_shape(middle, out[4:16])

fn reshaped_composition(x: tensor[20] f32, out: tensor[12] f32):
  write_shape(reshape(x[4:16], (3, 4)), out)

fn bad_stride(x: tensor[3, 4] f32, out: tensor[6] f32):
  for row in parallel:
    t = load(reshape(x[0:3, 0:2], (6,))[row:row+1])
    store(t, out[row:row+1])
";
fn bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn exercise(device: Device, candidate: Candidate) {
    let program = compile(
        &[SourceFile {
            path: "reshape.seismic.portable".into(),
            scope: Scope::Portable,
            text: SOURCE.into(),
        }],
        &[],
    )
    .unwrap_or_else(|errors| {
        panic!(
            "{}",
            errors
                .iter()
                .map(|e| e.render())
                .collect::<Vec<_>>()
                .join("\n")
        )
    });
    let values = (0..20).map(|n| n as f32 + 0.125).collect::<Vec<_>>();
    struct Bound(HashMap<String, Buffer>);
    impl Bindings for Bound {
        fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
            assert!(plane.is_empty());
            self.0.get(root)
        }
        fn scalar(&self, _: &str) -> Option<f64> {
            None
        }
    }
    for name in ["composition", "reshaped_composition"] {
        let plan = seismic_lang::plan::plan(&program, name, &HashMap::new()).unwrap();
        let mut compiled = CompiledPlan::compile_diagnostic(
            &device,
            &program,
            &plan,
            &Options::default(),
            candidate.clone(),
        )
        .unwrap();
        let length = if name == "composition" { 20 } else { 12 };
        let out = device.buffer_from(&bytes(&vec![-9.; length])).unwrap();
        let bound = Bound(HashMap::from([
            ("x".into(), device.buffer_from(&bytes(&values)).unwrap()),
            ("middle".into(), device.buffer(48).unwrap()),
            ("out".into(), out.clone()),
        ]));
        compiled.execute(&bound).unwrap();
        let mut actual = vec![0; length * 4];
        out.read(&mut actual).unwrap();
        let expected = if length == 20 {
            [&[-9.; 4][..], &values[4..16], &[-9.; 4][..]].concat()
        } else {
            values[4..16].to_vec()
        };
        assert_eq!(actual, bytes(&expected));
        let account =
            seismic_accounting::derive(&program, name, &HashMap::new(), &Default::default(), 10000)
                .unwrap();
        assert!(
            account.memory.is_exact(),
            "{:?}",
            account.memory.unavailable
        );
        let read = account
            .memory
            .accesses
            .iter()
            .find(|(id, _)| *id == "x")
            .unwrap()
            .1;
        assert_eq!(read.reads.ranges().collect::<Vec<_>>(), vec![16..64]);
    }
    let snapshot = lower(&program, "snapshot", device.backend(), &HashMap::new()).unwrap();
    let mut kernel = device.compile(&snapshot, candidate.clone()).unwrap();
    let input = device.buffer_from(&bytes(&values[..4])).unwrap();
    let output = device.buffer(16).unwrap();
    kernel
        .execute(&[input.clone(), output.clone()], &[])
        .unwrap();
    let mut actual = vec![0; 16];
    output.read(&mut actual).unwrap();
    assert_eq!(actual, bytes(&values[..4]));
    input.read(&mut actual).unwrap();
    assert_eq!(actual, bytes(&[0.; 4]));
    let bad = lower(&program, "bad_stride", device.backend(), &HashMap::new()).unwrap();
    assert!(device.compile(&bad, candidate.clone()).is_err());
    let account = seismic_accounting::memory::derive(
        &program,
        "bad_stride",
        &HashMap::new(),
        &Default::default(),
        10000,
    )
    .unwrap();
    assert!(!account.is_exact());
    assert!(account.unavailable.iter().any(|e| e.contains("contiguous")));
    let source="fn wrong(x: tensor[12] f32, out: tensor[11] f32):\n  t = load(reshape(x,(11,)))\n  store(t,out)\n";
    assert!(compile(
        &[SourceFile {
            path: "bad.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into()
        }],
        &[]
    )
    .is_err());
}
#[test]
fn cpu_reshape() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
        },
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_reshape() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_reshape() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
            },
            threads_per_block: 32,
        },
    );
}
