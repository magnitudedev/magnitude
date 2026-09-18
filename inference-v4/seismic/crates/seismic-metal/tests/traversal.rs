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

#[test]
fn retained_traversal_bounds_cover_every_width_and_reuse_accounting() {
    use seismic_accounting::{selection::{Choices, Domain}, schedule::{CapacityUnit, Resource, Timebase}, workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload}};
    use seismic_compiler::tuner::Backend as _;
    use seismic_metal::{choices::{Decision, Expansion}, model::{self, Hardware, Service, Timing, Units}, tuning};
    let baseline = selected(1);
    let config = Config { sg_per_tg: 1, max_threads_per_threadgroup: 32, ..Default::default() };
    let mut preparation = seismic_metal::choices::expand(baseline.source(), config.clone(), &[]).unwrap();
    let root = loop {
        let Expansion::Choice(choice) = preparation else { panic!("missing retained loop choice"); };
        if matches!(choice.decision(), Decision::Traversal(_)) { break choice; }
        let alternative = if matches!(choice.decision(), Decision::Storage(_)) {
            seismic_metal::choices::Alternative::Storage(TilePlacement::Replicated)
        } else { choice.diagnostic(seismic_realization::LoadStrategy::Materialize) };
        let index = choice.index(&alternative).unwrap();
        preparation = choice.refine(index).unwrap();
    };
    let mut primitives = std::collections::HashSet::new();
    for width in 1..=7 { primitives.extend(model::requirements(&selected(width)).unwrap().primitives); }
    let hardware = Hardware {
        identity: "synthetic traversal services".into(), timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
        resources: vec![Resource { name: "service".into(), capacity: 32, unit: CapacityUnit::Slots }],
        resident_groups: 1, resident_shared_bytes: 32768,
        timings: primitives.into_iter().map(|primitive| Timing { primitive, latency: 1, services: vec![Service { resource: 0, offset: 0, duration: 1, units: Units::PerLane(1) }] }).collect(),
    };
    let backend = tuning::Backend::with_conditions(tuning::Conditions {
        target: "fixture".into(), capacities: tuning::Capacities { max_threads_per_threadgroup: 32, max_threadgroup_bytes: 32768 },
        form: tuning::Form::Fixed(Default::default()), hardware: hardware.clone(),
    }).unwrap();
    let emitted = msl::prepare_execution(&baseline).unwrap();
    let workload = ScalarWorkload { integer_domains: Vec::new(),
        identity: "traversal region".into(),
        allocations: [28,4].into_iter().enumerate().map(|(id,bytes)| Allocation { id:id as u64,bytes,alignment:4,known_bytes:Default::default() }).collect(),
        buffers: [28,4].into_iter().enumerate().map(|(id,bytes)| BufferBinding { allocation:id as u64,offset:0,bytes }).collect(),
        scalars: seismic_lang::abi::ScalarLayout::words(&emitted.scalars).unwrap().encode(&[0.0]).unwrap(),
    };
    let limits = DerivationLimits { instructions:100_000,operations:1000 };
    let domain = Domain::new(root.clone()).unwrap();
    let bound = backend.relax(&domain,0..root.len(),&workload,limits).unwrap().unwrap().lower_bound().unwrap();
    assert!(bound > 1, "body demand must improve the launch-only bound");
    let cached = backend.relax(&domain,0..1,&workload,DerivationLimits {instructions:0,operations:0}).unwrap().unwrap().lower_bound().unwrap();
    assert_eq!(bound,cached,"retained stage must reuse the already-derived body account");
    let mut pending = (0..root.len()).map(|i| root.refine(i).unwrap()).collect::<Vec<_>>();
    let mut complete = 0;
    while let Some(next) = pending.pop() {
        match next {
            Expansion::Choice(choice) => {
                assert!(matches!(choice.decision(), Decision::Traversal(_)));
                pending.extend((0..choice.len()).map(|i| choice.refine(i).unwrap()));
            },
            Expansion::Execution {execution,..} => {
                let account = model::invocation_account(&execution,&workload,limits).unwrap();
                let demand = account.demand(&hardware).unwrap().expect("complete mapped execution");
                assert!(bound <= demand.lower_bound().unwrap());
                complete += 1;
            },
            Expansion::Infeasible {..} => panic!("unexpected capacity failure"),
        }
    }
    assert!(complete >= 7);
}
