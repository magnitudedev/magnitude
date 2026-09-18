use seismic_accounting::{
    schedule::{CapacityUnit, Resource, Timebase},
    workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload},
};
use seismic_lang::{program::{compile, SourceFile}, Scope};
use seismic_metal::{execution::{self, Config}, model::{self, Hardware, Service, Timing, Units}, terminal::{Primitive, Space}};

fn selected(stride: u64) -> execution::Execution {
    let source = compile(&[SourceFile {
        path: "geometry.seismic.portable".into(), scope: Scope::Portable,
        text: format!("fn evaluate(x:tensor[1024] f32,out:tensor[32] f32):\n  a=load(x)\n  y=tile[32] f32\n  for i in owned(y): y[i]=a[i*{stride}]\n  store(y,out)\n"),
    }], &[]).unwrap();
    let function = seismic_lang::lower::lower(&source, "evaluate", "metal", &Default::default()).unwrap();
    execution::prepare_storage_selected(&function, Config {
        sg_per_tg: 1, loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
        ..Default::default()
    }, &mut |_| Ok(seismic_realization::dispatch::TilePlacement::Distributed)).unwrap()
}
fn workload(alignment: u64, offset: u64) -> ScalarWorkload {
    ScalarWorkload { integer_domains: Vec::new(),
        identity: format!("synthetic aligned bindings {alignment}/{offset}"),
        allocations: [4096 + offset, 128].into_iter().enumerate().map(|(id, bytes)| Allocation {
            id: id as u64, bytes, alignment, known_bytes: Default::default(),
        }).collect(),
        buffers: vec![BufferBinding { allocation: 0, offset, bytes: 4096 }, BufferBinding { allocation: 1, offset: 0, bytes: 128 }],
        scalars: vec![],
    }
}
fn hardware(execution: &execution::Execution) -> Hardware {
    Hardware {
        identity: "hypothetical aligned 128-byte scalar-access service; no cache reuse".into(),
        timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
        resources: vec![Resource { name: "128-byte device request blocks".into(), capacity: 1, unit: CapacityUnit::Slots }],
        resident_groups: 1, resident_shared_bytes: 0,
        timings: model::requirements(execution).unwrap().primitives.into_iter().map(|primitive| {
            let device = matches!(primitive, Primitive::Read { space: Space::Device, .. } | Primitive::Write { space: Space::Device, .. });
            Timing { primitive, latency: u64::from(device), services: if device { vec![Service {
                resource: 0, offset: 0, duration: 1, units: Units::PerTransaction { bytes: 128, units: 1 },
            }] } else { vec![] } }
        }).collect(),
    }
}
const LIMITS: DerivationLimits = DerivationLimits { instructions: 100_000, operations: 100_000 };

#[test]
fn active_addresses_distinguish_coalescing_and_binding_residue() {
    for (stride, offset, read_blocks) in [(1, 0, 1), (1, 4, 2), (32, 0, 32), (32, 4, 32)] {
        let execution = selected(stride);
        let workload = workload(128, offset);
        let account = model::invocation_account(&execution, &workload, LIMITS).unwrap();
        assert!(account.is_complete(), "{:?}", account.unmapped);
        let reads: u64 = account.operations.iter().filter(|o| matches!(o.primitive, Primitive::Read { space: Space::Device, .. }))
            .map(|o| o.instances * o.access.as_ref().expect("known device address").transactions(128).unwrap()).sum();
        assert_eq!(reads, read_blocks, "stride {stride} offset {offset}");
        let hardware = hardware(&execution);
        let demand = account.demand(&hardware).unwrap().unwrap();
        assert_eq!(demand.lower_bound().unwrap(), read_blocks + 1);
        let schedule = model::execution(&execution, &hardware, &workload, LIMITS).unwrap();
        assert!(schedule.unmapped.is_empty(), "{:?}", schedule.unmapped);
        let service: u64 = schedule.operations.iter().flat_map(|o| &o.reservations).filter(|r| r.resource == 0).map(|r| r.units).sum();
        assert_eq!(service, read_blocks + 1);
    }
}

#[test]
fn weaker_alignment_leaves_transaction_service_unmapped() {
    let execution = selected(1);
    let workload = workload(4, 0);
    let hardware = hardware(&execution);
    let account = model::invocation_account(&execution, &workload, LIMITS).unwrap();
    assert!(account.is_complete()); // Scalar requests are still completely known.
    assert!(account.demand(&hardware).unwrap().is_none());
    let schedule = model::execution(&execution, &hardware, &workload, LIMITS).unwrap();
    assert!(schedule.unmapped.iter().any(|m| m.contains("access geometry")));
}

#[test]
fn known_routes_survive_local_storage_and_determine_gather_geometry() {
    use seismic_realization::dispatch::TilePlacement;
    let program = compile(&[SourceFile {
        path: "routes.seismic.portable".into(), scope: Scope::Portable,
        text: "fn evaluate(x:tensor[1024] f32,route:tensor[32] i32,out:tensor[32] f32):\n  a=load(x)\n  r=load(route)\n  y=tile[32] f32\n  for i in owned(y): y[i]=a[r[i]*32]\n  store(y,out)\n".into(),
    }], &[]).unwrap();
    let function = seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    for placement in [TilePlacement::Distributed, TilePlacement::GroupShared] {
        let execution = execution::prepare_with_choices(&function, Config { sg_per_tg: 1, ..Default::default() },
            &mut |index, site| Ok(if site.can_borrow && index == 0 { seismic_lang::ir::LoadMode::Borrow } else { seismic_lang::ir::LoadMode::Materialize }),
            &mut |_| Ok(placement.clone()), &mut |r| Ok(r.diagnostic())).unwrap();
        for repeated in [false, true] {
            let mut workload = ScalarWorkload { integer_domains: Vec::new(),
                identity: format!("known gather routes {repeated}"),
                allocations: [4096, 128, 128].into_iter().enumerate().map(|(id, bytes)| Allocation { id: id as u64, bytes, alignment: 128, known_bytes: Default::default() }).collect(),
                buffers: [4096, 128, 128].into_iter().enumerate().map(|(id, bytes)| BufferBinding { allocation: id as u64, offset: 0, bytes }).collect(),
                scalars: vec![],
            };
            workload.allocations[1].known_bytes = (0..32i32).flat_map(|i| (if repeated { 0i32 } else { i }).to_le_bytes()).enumerate().map(|(i, b)| (i as u64, b)).collect();
            let account = model::invocation_account(&execution, &workload, LIMITS).unwrap();
            assert!(account.is_complete(), "{placement:?}: {:?}", account.unmapped);
            assert!(account.operations.iter().filter(|o| matches!(o.primitive, Primitive::Read { space: Space::Device, .. })).all(|o| o.access.is_some()));
            let gather: u64 = account.operations.iter().filter(|o| matches!(o.primitive, Primitive::Read { space: Space::Device, ty: seismic_metal::terminal::Type::F32 }))
                .map(|o| o.instances * o.access.as_ref().unwrap().transactions(128).unwrap()).sum();
            assert_eq!(gather, if repeated { 1 } else { 32 }, "{placement:?}");
            workload.allocations[1].known_bytes.remove(&0);
            let unknown = model::invocation_account(&execution, &workload, LIMITS).unwrap();
            assert!(!unknown.is_complete(), "a partially unknown route must not invent an address");
        }
    }
}

#[test]
fn writable_binding_alias_does_not_supply_stale_route_values() {
    let program = compile(&[SourceFile {
        path: "alias_routes.seismic.portable".into(), scope: Scope::Portable,
        text: "fn evaluate(x:tensor[32] i32,out:tensor[32] i32):\n  a=load(x)\n  y=tile[32] i32\n  for i in owned(y):\n    if a[i]>0: y[i]=1\n    else: y[i]=0\n  store(y,out)\n".into(),
    }], &[]).unwrap();
    let function = seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    let execution = execution::prepare_storage_selected(&function, Config { sg_per_tg: 1, ..Default::default() }, &mut |_| Ok(seismic_realization::dispatch::TilePlacement::Distributed)).unwrap();
    let mut workload = ScalarWorkload { integer_domains: Vec::new(),
        identity: "read-only input known, separate output".into(),
        allocations: (0..2).map(|id| Allocation { id, bytes: 128, alignment: 128, known_bytes: (0..128).map(|i| (i, 0)).collect() }).collect(),
        buffers: (0..2).map(|allocation| BufferBinding { allocation, offset: 0, bytes: 128 }).collect(), scalars: vec![],
    };
    assert!(model::invocation_account(&execution, &workload, LIMITS).unwrap().is_complete());
    workload.identity = "same physical backing is also published".into();
    workload.buffers[1].allocation = 0;
    let account = model::invocation_account(&execution, &workload, LIMITS).unwrap();
    assert!(!account.is_complete());
    assert!(account.unmapped.iter().any(|g| g.contains("predicate")), "{:?}", account.unmapped);
}
