use seismic_lang::{
    ir::LoadMode,
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::{
    execution::{self, Config, FoldOwnership},
    msl,
    terminal::Primitive,
};
use seismic_realization::dispatch::TilePlacement;

fn selected(width: usize) -> execution::Execution {
    let program = compile(
        &[SourceFile {
            path: "traversal.seismic.portable".into(),
            scope: Scope::Portable,
            text: r#"
fn evaluate(x:tensor[7] f32,out:tensor[1] f32,p:i32):
  a=load(x)
  state=tile[1] f32
  for j in owned(state): state[j]=3.0
  for i in range(7):
    value = a[i+p]
    state[0]=fma(value,1.0,state[0])
  store(state,out)
"#
            .into(),
        }],
        &[],
    )
    .unwrap();
    let function = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    let mut visited = 0;
    let execution = execution::prepare_with_traversals(
        &function,
        Config {
            sg_per_tg: 1,
            ..Default::default()
        },
        None,
        &mut |_| Ok(FoldOwnership::Serial),
        &mut |_, _| Ok(LoadMode::Materialize),
        &mut |_| Ok(TilePlacement::Replicated),
        &mut |d| Ok(d.diagnostic()),
        &mut |d| Ok(d.new_slot),
        &mut |d| {
            assert_eq!(d.iterations, 7);
            visited += 1;
            Ok(width)
        },
    )
    .unwrap();
    assert!(visited > 0);
    execution
}

#[test]
fn traversal_counts_preserve_work_and_expose_removed_control() {
    use seismic_accounting::workload::{
        Allocation, BufferBinding, DerivationLimits, ScalarWorkload,
    };
    let mut counts = Vec::new();
    for width in 1..=7 {
        let execution = selected(width);
        let emitted = msl::prepare_execution(&execution).unwrap();
        let workload = ScalarWorkload { integer_domains: Vec::new(),
            identity: "traversal".into(),
            allocations: [28, 4]
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| Allocation {
                    id: id as u64,
                    bytes,
                    alignment: 4,
                    known_bytes: Default::default(),
                })
                .collect(),
            buffers: [28, 4]
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| BufferBinding {
                    allocation: id as u64,
                    offset: 0,
                    bytes,
                })
                .collect(),
            scalars: seismic_lang::abi::ScalarLayout::words(&emitted.scalars)
                .unwrap()
                .encode(&[0.0])
                .unwrap(),
        };
        let account = seismic_metal::model::invocation_account(
            &execution,
            &workload,
            DerivationLimits {
                instructions: 100_000,
                operations: 1000,
            },
        )
        .unwrap();
        assert!(account.is_complete(), "{width}: {:?}", account.unmapped);
        let fmas: u64 = account
            .operations
            .iter()
            .filter(|c| matches!(&c.primitive, Primitive::Builtin { name, .. } if name == "fma"))
            .map(|c| c.instances * c.lanes)
            .sum();
        assert_eq!(fmas, 7 * 32, "width {width}");
        counts.push(
            account
                .operations
                .iter()
                .filter(|c| c.primitive == Primitive::Branch)
                .map(|c| c.instances)
                .sum::<u64>(),
        );
    }
    assert!(counts[6] < counts[0], "{counts:?}");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_traversals_preserve_fma_order_tail_scopes_and_failures() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let values = [16777216.0f32, 1.0, -16777216.0, 3.0, -3.0, 0.5, -0.5];
    let expected = values
        .iter()
        .fold(3.0f32, |state, &value| value.mul_add(1.0, state));
    let x = device
        .buffer_from(
            &values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = device.buffer(4).unwrap();
    for width in 1..=7 {
        let emitted = msl::emit_execution(&selected(width)).unwrap();
        let valid = emitted.encode_scalars(&[0.0]).unwrap();
        let invalid = emitted.encode_scalars(&[1.0]).unwrap();
        let kernel = device.compile(emitted).unwrap();
        assert!(
            device.run(&kernel, &[&x, &out], &invalid, 1).is_err(),
            "width {width}"
        );
        device.run(&kernel, &[&x, &out], &valid, 1).unwrap();
        assert_eq!(out.read(4), expected.to_le_bytes(), "width {width}");
    }
}
