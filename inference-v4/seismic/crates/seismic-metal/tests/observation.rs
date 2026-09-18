#![cfg(target_os = "macos")]
use seismic_lang::{
    Scope,
    program::{SourceFile, compile},
};
use seismic_metal::{msl, runtime::Device};

#[test]
#[ignore = "requires Metal stage timestamp counters"]
fn stage_capture_preserves_dispatch_dependencies_and_failure_status() {
    let program = compile(
        &[SourceFile {
            path: "observation.seismic.portable".into(),
            scope: Scope::Portable,
            text: r#"
fn evaluate(x:tensor[5,64] f32,middle:tensor[5,64] f32,out:tensor[5] f32,p:i32):
  for row in parallel:
    a = load(x[row])
    y = tile[64] f32
    for i in owned(y): y[i] = a[p]
    store(y,middle[row])
  for row in parallel:
    y = tile[1] f32
    for i in owned(y): y[i] = middle[row,0]
    store(y,out[row:row+1])
"#
            .into(),
        }],
        &[],
    )
    .unwrap();
    let lowered =
        seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    let emitted = msl::emit(&lowered).unwrap();
    assert_eq!(emitted.launches.len(), 2);
    let valid = emitted.encode_scalars(&[1.0]).unwrap();
    let invalid = emitted.encode_scalars(&[64.0]).unwrap();
    let device = Device::open().unwrap();
    let pipeline = device.compile(emitted).unwrap();
    let x = device
        .buffer_from(
            &(0..320)
                .flat_map(|n| (n as f32).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let middle = device.buffer(320 * 4).unwrap();
    let out = device.buffer(5 * 4).unwrap();
    assert!(
        device
            .profile(&pipeline, &[&x, &middle, &out], &invalid)
            .is_err(),
        "a later dispatch must not erase an earlier bounds failure"
    );
    let observed = device
        .profile(&pipeline, &[&x, &middle, &out], &valid)
        .unwrap();
    assert!(observed.command_seconds > 0.0);
    assert_eq!(observed.dispatches.len(), 2);
    for (index, dispatch) in observed.dispatches.iter().enumerate() {
        assert_eq!(dispatch.invocation, 0);
        assert_eq!(dispatch.launch, index);
        assert_eq!(dispatch.kernel, pipeline.emitted.launches[index].kernel);
        assert!(dispatch.elapsed_ns > 0);
    }
    assert!(
        observed.dispatches[1].started_ns
            >= observed.dispatches[0].started_ns + observed.dispatches[0].elapsed_ns
    );
    assert_eq!(
        out.read(20),
        (0..5)
            .flat_map(|n| ((n * 64 + 1) as f32).to_le_bytes())
            .collect::<Vec<_>>()
    );
}
