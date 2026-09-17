#![cfg(target_os = "macos")]
#[path = "../../../../validation/support/scalar_cases.rs"]
mod scalar_cases;
#[test]
#[ignore = "requires a Metal device"]
fn value_semantics_and_precision_boundaries() {
    use seismic_metal::{msl, runtime::Device};
    let device = Device::open().unwrap();
    let info = device.info();
    scalar_cases::exercise(
        |lowered, values, scalars| {
            let emitted = msl::emit_with(
                lowered,
                msl::Config {
                    max_threads_per_threadgroup: info.max_threads_per_threadgroup as i64,
                    max_threadgroup_bytes: info.max_threadgroup_bytes as i64,
                    ..Default::default()
                },
            )
            .unwrap();
            let scalars = emitted.encode_scalars(scalars).unwrap();
            let pipeline = device.compile(emitted).unwrap();
            let buffers = values
                .iter()
                .map(|v| device.buffer_from(v).unwrap())
                .collect::<Vec<_>>();
            device
                .run(&pipeline, &buffers.iter().collect::<Vec<_>>(), &scalars, 1)
                .unwrap();
            for (value, buffer) in values.iter_mut().zip(&buffers) {
                *value = buffer.read(value.len());
            }
        },
        "metal",
    );
}

#[test]
#[ignore = "requires a Metal device"]
fn raw_scalar_bytes_obey_index_and_boolean_contracts() {
    use seismic_lang::{
        program::{compile, SourceFile},
        Scope,
    };
    use seismic_metal::{msl, runtime::Device};
    let program=compile(&[SourceFile{path:"bounded.seismic.portable".into(),scope:Scope::Portable,text:"fn bounded(pos: index[4], flag: bool, out: tensor[1] i32):\n  t = tile[1] i32\n  for i in owned(t): t[i] = pos\n  store(t,out)\n".into()}],&[]).unwrap();
    let lowered =
        seismic_lang::lower::lower(&program, "bounded", "metal", &Default::default()).unwrap();
    let emitted = msl::emit_with(&lowered, Default::default()).unwrap();
    let valid = emitted.encode_scalars(&[3.0, 1.0]).unwrap();
    let device = Device::open().unwrap();
    let pipeline = device.compile(emitted).unwrap();
    let buffer = device.buffer_from(&[0; 4]).unwrap();
    for (offset, bytes) in [
        (0, 4i32.to_le_bytes().to_vec()),
        (0, (-1i32).to_le_bytes().to_vec()),
        (4, vec![2]),
    ] {
        let mut invalid = valid.clone();
        invalid[offset..offset + bytes.len()].copy_from_slice(&bytes);
        assert!(device.run(&pipeline, &[&buffer], &invalid, 1).is_err());
        assert_eq!(buffer.read(4), [0; 4]);
    }
    device.run(&pipeline, &[&buffer], &valid, 1).unwrap();
    assert_eq!(buffer.read(4), 3i32.to_le_bytes());
}

#[path = "../../../../validation/support/attention_cases.rs"]
mod attention_cases;
#[test]
#[ignore = "requires a Metal device"]
fn standard_streaming_attention_matches_independent_reference() {
    use seismic_metal::{msl, runtime::Device};
    let device = Device::open().unwrap();
    let info = device.info();
    attention_cases::exercise(
        |lowered, values, scalars| {
            let emitted = msl::emit_with(
                lowered,
                msl::Config {
                    max_threads_per_threadgroup: info.max_threads_per_threadgroup as i64,
                    max_threadgroup_bytes: info.max_threadgroup_bytes as i64,
                    ..Default::default()
                },
            )?;
            let scalars = emitted.encode_scalars(scalars)?;
            let pipeline = device.compile(emitted)?;
            let buffers = values
                .iter()
                .map(|v| device.buffer_from(v))
                .collect::<Result<Vec<_>, _>>()?;
            device.run(&pipeline, &buffers.iter().collect::<Vec<_>>(), &scalars, 1)?;
            for (value, buffer) in values.iter_mut().zip(&buffers) {
                *value = buffer.read(value.len());
            }
            Ok(())
        },
        "metal",
    );
}

#[path = "../../../../validation/support/stream_cases.rs"]
mod stream_cases;
#[test]
#[ignore = "requires a Metal device"]
fn runtime_stream_domains_and_piece_tails() {
    use seismic_metal::{msl, runtime::Device};
    let device = Device::open().unwrap();
    let info = device.info();
    stream_cases::exercise(
        |lowered, values| {
            let emitted = msl::emit_with(
                lowered,
                msl::Config {
                    max_threads_per_threadgroup: info.max_threads_per_threadgroup as i64,
                    max_threadgroup_bytes: info.max_threadgroup_bytes as i64,
                    ..Default::default()
                },
            )?;
            let pipeline = device.compile(emitted)?;
            let buffers = values
                .iter()
                .map(|v| device.buffer_from(v))
                .collect::<Result<Vec<_>, _>>()?;
            device.run(&pipeline, &buffers.iter().collect::<Vec<_>>(), &[], 1)?;
            for (value, buffer) in values.iter_mut().zip(&buffers) {
                *value = buffer.read(value.len());
            }
            Ok(())
        },
        "metal",
    );
}

#[test]
#[ignore = "requires a Metal device"]
fn plan_reports_device_bounds_errors_and_resets_status_between_runs() {
    use seismic_lang::{
        program::{compile, SourceFile},
        Scope,
    };
    use seismic_metal::{
        msl,
        plan_exec::{compile_plan, Bindings},
        runtime::{Buffer, Device},
    };
    use std::collections::HashMap;
    struct Inputs(HashMap<String, Buffer>);
    impl Bindings for Inputs {
        fn buffer(&self, root: &str, part: &str) -> Option<&Buffer> {
            if part.is_empty() {
                self.0.get(root)
            } else {
                None
            }
        }
        fn scalar(&self, _: &str) -> Option<f64> {
            None
        }
    }
    let text="fn stream[T](x: tensor[T] f32, visible: tensor[2] i32, out: tensor[1] f32):\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  for t in load(x[visible[0]:visible[1]], over=0):\n    acc[0] += reduce(t, 0, sum)\n  store(acc, out)\n\nfn composition(x: tensor[4] f32, visible: tensor[2] i32, out: tensor[1] f32):\n  stream(x,visible,out)\n";
    let p = compile(
        &[SourceFile {
            path: "plan.seismic.portable".into(),
            text: text.into(),
            scope: Scope::Portable,
        }],
        &[],
    )
    .unwrap();
    let plan = seismic_lang::plan::plan(&p, "composition", &HashMap::new()).unwrap();
    let device = Device::open().unwrap();
    let compiled = compile_plan(&device, &p, &plan, msl::Config::default()).unwrap();
    let inputs = Inputs(HashMap::from([
        (
            "x".into(),
            device
                .buffer_from(
                    &[1f32, 2., 3., 4.]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
        ),
        (
            "visible".into(),
            device
                .buffer_from(
                    &[0i32, 5]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
        ),
        ("out".into(), device.buffer_from(&[0; 4]).unwrap()),
    ]));
    assert!(device
        .run_plan(&compiled, &inputs)
        .unwrap_err()
        .contains("out-of-bounds"));
    inputs.0["visible"].write(
        &[0i32, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    device.run_plan(&compiled, &inputs).unwrap();
    assert_eq!(inputs.0["out"].read(4), 10f32.to_le_bytes());
    assert!(device
        .run_plan_steps_repeated(&compiled, &inputs, 0..1, 0)
        .is_err());
    assert!(device.run_plan_steps(&compiled, &inputs, 0..2).is_err());
}

#[test]
#[ignore = "requires a Metal device"]
fn altered_domain_is_rejected_without_touching_backing_canaries() {
    use seismic_lang::{
        hir::StmtKind,
        program::{compile, SourceFile},
        sym::Sym,
        Scope,
    };
    use seismic_metal::{msl, runtime::Device};
    let p=compile(&[SourceFile {path:"copy.seismic.portable".into(),scope:Scope::Portable,text:"fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    store(t,out[row:row+1])\n".into()}],&[]).unwrap();
    let mut l = seismic_lang::lower::lower(
        &p,
        "copy",
        "metal",
        &std::collections::HashMap::from([("N".into(), 4)]),
    )
    .unwrap();
    let StmtKind::Parallel { extents, .. } = &mut l.body[0].kind else {
        panic!()
    };
    extents[0] = Sym::constant(5);
    let device = Device::open().unwrap();
    let pipeline = device
        .compile(msl::emit_with(&l, Default::default()).unwrap())
        .unwrap();
    let input = device
        .buffer_from(
            &[1f32; 8]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let sentinel = [-7f32; 8]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let output = device.buffer_from(&sentinel).unwrap();
    assert!(device
        .run(&pipeline, &[&input, &output], &[], 1)
        .unwrap_err()
        .contains("out-of-bounds"));
    assert_eq!(&output.read(32)[16..], &sentinel[16..]);
}

#[test]
#[ignore = "requires a Metal device"]
fn resident_subviews_preserve_offsets_owners_and_canaries() {
    use seismic_lang::{
        program::{compile, SourceFile},
        Scope,
    };
    use seismic_metal::{msl, runtime::Device};
    let p=compile(&[SourceFile{path:"copy.seismic.portable".into(),scope:Scope::Portable,text:"fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    store(t,out[row:row+1])\n".into()}],&[]).unwrap();
    let l = seismic_lang::lower::lower(
        &p,
        "copy",
        "metal",
        &std::collections::HashMap::from([("N".into(), 4)]),
    )
    .unwrap();
    let device = Device::open().unwrap();
    let other = Device::open().unwrap();
    let pipeline = device
        .compile(msl::emit_with(&l, Default::default()).unwrap())
        .unwrap();
    let root = device
        .buffer_from(
            &[7f32, 1., 2., 3., 4., 9.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let input = root.view(4..20).unwrap();
    drop(root);
    let backing = device.buffer_from(&[0xa5; 24]).unwrap();
    let output = backing.view(4..20).unwrap();
    device.run(&pipeline, &[&input, &output], &[], 1).unwrap();
    assert_eq!(
        output.read(16),
        [1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(&backing.read(24)[..4], &[0xa5; 4]);
    assert_eq!(&backing.read(24)[20..], &[0xa5; 4]);
    assert!(device
        .run(&pipeline, &[&input, &backing.view(1..17).unwrap()], &[], 1)
        .unwrap_err()
        .contains("alignment"));
    let foreign = other.buffer(16).unwrap();
    assert!(device.run(&pipeline, &[&foreign, &output], &[], 1).is_err());
    assert!(other.run(&pipeline, &[&input, &output], &[], 1).is_err());
    assert!(backing.view(0..25).is_err());
    assert_eq!(device.buffer(0).unwrap().len(), 0);
    drop(device);
    assert_eq!(
        output.read(16),
        [1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}
