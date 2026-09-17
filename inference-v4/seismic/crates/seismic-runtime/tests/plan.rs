use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{
    plan::{Bindings, PlanCompiler},
    Buffer, Candidate, Device,
};
use std::collections::HashMap;
struct Inputs {
    buffers: HashMap<String, Buffer>,
    gain: Option<f64>,
}
impl Bindings for Inputs {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        assert_eq!(plane, "");
        self.buffers.get(root)
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        assert_eq!(name, "gain");
        self.gain
    }
}
fn exercise(device: Device, candidate: Candidate) {
    let text = "fn transform[N](x: tensor[N] T, out: tensor[N] f32, gain: f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = t[i] * gain\n    store(y,out[row:row+1])\n\nfn chain(x: tensor[6] f32, tmp: tensor[6] f32, out: tensor[6] f32, gain: f32):\n  transform(x[1:5], tmp[1:5], 2.0)\n  transform(tmp[1:5], out[1:5], gain)\n";
    let program = compile(
        &[SourceFile {
            path: "plan.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let plan = seismic_lang::plan::plan(&program, "chain", &HashMap::new()).unwrap();
    let mut compiler = PlanCompiler::new(&device, &program, Default::default(), candidate);
    let mut compiled = compiler.compile(&plan).unwrap();
    let shared = compiler.compile(&plan).unwrap();
    assert_eq!(compiler.kernel_count(), 1);
    drop(compiler);
    assert_eq!(compiled.step_count(), 2);
    assert_eq!(compiled.kernel_count(), 1);
    let input = [99f32, 1., 2., 3., 4., 99.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    // The binding itself is a view; composition adds its element offset to it.
    let backing = device.buffer_from(&[0xa5; 32]).unwrap();
    let out = backing.view(4..28).unwrap();
    let tmp = device.buffer_from(&[0xa5; 24]).unwrap();
    let mut bindings = Inputs {
        buffers: HashMap::from([
            ("x".into(), device.buffer_from(&input).unwrap()),
            ("tmp".into(), tmp.clone()),
            ("out".into(), out),
        ]),
        gain: None,
    };
    drop(device);
    assert!(compiled
        .execute(&bindings)
        .unwrap_err()
        .contains("unbound scalar"));
    let mut untouched = [0; 24];
    tmp.read(&mut untouched).unwrap();
    assert_eq!(untouched, [0xa5; 24]);
    bindings.gain = Some(3.);
    compiled.execute(&bindings).unwrap();
    let mut actual = [0; 32];
    backing.read(&mut actual).unwrap();
    assert_eq!(&actual[..8], &[0xa5; 8]);
    assert_eq!(&actual[24..], &[0xa5; 8]);
    assert_eq!(
        &actual[8..24],
        &[6f32, 12., 18., 24.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    bindings.gain = Some(4.);
    let mut submission = compiled.prepare(&bindings).unwrap();
    submission.append(shared.prepare(&bindings).unwrap());
    assert_eq!(submission.len(), 4);
    drop(bindings);
    drop(compiled);
    drop(shared);
    let observed = submission.execute_batched().unwrap();
    assert!(observed.host_seconds > 0.);
    submission.execute_batched().unwrap();
    backing.read(&mut actual).unwrap();
    assert_eq!(
        &actual[8..24],
        &[8f32, 16., 24., 32.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}
#[test]
fn cpu_composed_resident_plan() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_composed_resident_plan() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_composed_resident_plan() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
