#![cfg(target_os = "macos")]
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
    x: Buffer,
    out: Buffer,
    which: f64,
}
impl Bindings for Inputs {
    fn buffer(&self, root: &str, _: &str) -> Option<&Buffer> {
        match root {
            "x" => Some(&self.x),
            "out" => Some(&self.out),
            _ => None,
        }
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        (name == "which").then_some(self.which)
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_batch_failure_and_preflight() {
    let source = "fn read(x: tensor[4] f32, out: tensor[1] f32, which: i32):\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = x[which]\n    store(y,out[row:row+1])\n\nfn chain(x: tensor[4] f32, out: tensor[1] f32, which: i32):\n  read(x,out,which)\n  read(x,out,0)\n";
    let program = compile(
        &[SourceFile {
            path: "submission.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let plan = seismic_lang::plan::plan(&program, "chain", &HashMap::new()).unwrap();
    let device = Device::metal().unwrap();
    let mut compiler = PlanCompiler::diagnostic(
        &device,
        &program,
        Default::default(),
        Candidate::Metal(Default::default()),
    );
    let compiled = compiler.compile(&plan).unwrap();
    let mut inputs = Inputs {
        x: device
            .buffer_from(
                &[1f32, 2., 3., 4.]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        out: device.buffer_from(&[0xa5; 4]).unwrap(),
        which: 4.,
    };
    // A successful later dispatch must never clear an earlier dynamic failure.
    assert!(compiled
        .prepare(&inputs)
        .unwrap()
        .execute_batched()
        .unwrap_err()
        .contains("out-of-bounds"));
    inputs.which = 2.;
    compiled
        .prepare(&inputs)
        .unwrap()
        .execute_batched()
        .unwrap();
    let mut bytes = [0; 4];
    inputs.out.read(&mut bytes).unwrap();
    assert_eq!(bytes, 1f32.to_le_bytes());
    // Every pipeline's ownership must be checked before the first write.
    let other = Device::metal().unwrap();
    let mut other_compiler = PlanCompiler::diagnostic(
        &other,
        &program,
        Default::default(),
        Candidate::Metal(Default::default()),
    );
    let other_plan = other_compiler.compile(&plan).unwrap();
    inputs.out.write(&[0xa5; 4]).unwrap();
    let mut combined = compiled.prepare(&inputs).unwrap();
    combined.append(other_plan.prepare(&inputs).unwrap());
    assert!(combined
        .execute_batched()
        .unwrap_err()
        .contains("different device"));
    inputs.out.read(&mut bytes).unwrap();
    assert_eq!(bytes, [0xa5; 4]);
}
