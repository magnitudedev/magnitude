use seismic_engine::execution::{Composition, CompositionSpec};
use seismic_lang::{
    Scope,
    program::{SourceFile, compile},
};
use seismic_runtime::{Candidate, Device, plan::PlanCompiler};
use std::collections::{HashMap, HashSet};
fn names(xs: &[&str]) -> HashSet<String> {
    xs.iter().map(|s| (*s).into()).collect()
}
fn spec() -> CompositionSpec {
    CompositionSpec {
        entry: "chain".into(),
        shapes: HashMap::from([("N".into(), 4)]),
        elements: HashMap::new(),
        weights: HashMap::new(),
        external: names(&["x", "out"]),
        intermediates: names(&["tmp"]),
        scalars: HashMap::from([("gain".into(), 2.)]),
    }
}
#[test]
fn composition_requires_explicit_ownership_and_exact_runtime_bindings() {
    let program=compile(&[SourceFile{path:"composition.seismic.portable".into(),scope:Scope::Portable,text:"fn scale[N](x: tensor[N] f32,out: tensor[N] f32,gain: f32):\n  for row in parallel:\n    x1=load(x[row:row+1])\n    y=tile[1] f32\n    for i in owned(y): y[i]=x1[i]*gain\n    store(y,out[row:row+1])\nfn chain[N](x: tensor[N] f32,tmp: tensor[N] f32,out: tensor[N] f32,gain: f32):\n  scale(x,tmp,gain)\n  scale(tmp,out,gain)\n".into()}],&[]).unwrap();
    let device = Device::cpu();
    let candidate = Candidate::Cpu {
        loads: seismic_realization::LoadStrategy::Materialize,
    };
    let mut compiler =
        PlanCompiler::diagnostic(&device, &program, Default::default(), candidate.clone());
    let mut invalid = spec();
    invalid.intermediates.clear();
    assert!(Composition::compile(&mut compiler, invalid).is_err());
    let mut invalid = spec();
    invalid.intermediates.insert("x".into());
    assert!(Composition::compile(&mut compiler, invalid).is_err());
    let mut invalid = spec();
    invalid.scalars.insert("unknown".into(), 1.);
    assert!(Composition::compile(&mut compiler, invalid).is_err());
    let mut composition = Composition::compile(&mut compiler, spec()).unwrap();
    let mut second = Composition::compile(&mut compiler, spec()).unwrap();
    assert_eq!(compiler.kernel_count(), 1);
    drop(compiler);
    let x = device
        .buffer_from(
            &[1f32, 2., 3., 4.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = device.buffer_from(&[0xa5; 16]).unwrap();
    let tensors = HashMap::from([("x".into(), x), ("out".into(), out.clone())]);
    assert!(
        composition
            .execute(&tensors, &HashMap::from([("gain".into(), 3.)]))
            .is_err()
    );
    assert!(
        composition
            .execute(&HashMap::new(), &HashMap::new())
            .is_err()
    );
    let mut actual = [0; 16];
    out.read(&mut actual).unwrap();
    assert_eq!(actual, [0xa5; 16]);
    let observations = composition
        .execute_observed(&tensors, &HashMap::new())
        .unwrap();
    assert_eq!(observations.len(), 1);
    assert!(observations.iter().all(|o| o.entry == "chain"
        && o.execution.host_seconds > 0.
        && o.execution.device_seconds.is_none()));
    second.execute(&tensors, &HashMap::new()).unwrap();
    out.read(&mut actual).unwrap();
    assert_eq!(
        actual.as_slice(),
        [4f32, 8., 12., 16.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}
