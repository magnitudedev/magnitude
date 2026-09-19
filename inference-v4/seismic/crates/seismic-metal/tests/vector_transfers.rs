use seismic_lang::{program::{compile, SourceFile}, Scope};
use seismic_metal::{execution::{self, Config, FoldOwnership}, model, terminal::{Primitive, Space}};
use seismic_accounting::workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload};

fn selected(dtype: &str, width: u8, traversal: bool) -> execution::Execution {
    let program = compile(&[SourceFile { path: "vector_copy.seismic.portable".into(), scope: Scope::Portable,
        text: format!("fn evaluate(x:tensor[11] {dtype},out:tensor[11] {dtype}):\n  a=load(x)\n  store(a,out)\n"),
    }], &[]).unwrap();
    prepare(&program, width, traversal)
}
fn prepare(program: &seismic_lang::program::Program, width: u8, traversal: bool) -> execution::Execution {
    let function = seismic_lang::lower::lower(program, "evaluate", "metal", &Default::default()).unwrap();
    let mut copies = 0;
    let execution = execution::prepare_with_transfers(&function, Config { sg_per_tg: 1, ..Default::default() }, None,
        &mut |_| Ok(FoldOwnership::Serial), &mut |_, _| Ok(seismic_lang::ir::LoadMode::Materialize),
        &mut |_| Ok(seismic_realization::dispatch::TilePlacement::Replicated), &mut |r| Ok(r.diagnostic()), &mut |a| Ok(a.new_slot),
        &mut |choice| { copies += 1; assert_eq!(choice.maximum, 4); Ok(width) },
        &mut |choice| Ok(if traversal { choice.iterations } else { 1 }),
    ).unwrap();
    assert!(copies > 0, "snapshot has a typed vector-copy choice");
    execution
}
#[test]
fn vector_payloads_and_scalar_tails_are_accounted_from_the_selected_transfer() {
    for width in 1..=4 {
        for traversal in [false, true] {
            let execution = selected("f32", width, traversal);
            let workload = ScalarWorkload { integer_domains: Vec::new(), identity: "unaligned view of aligned backing".into(),
                allocations: (0..2).map(|id| Allocation { id, bytes: 48, alignment: 16, known_bytes: Default::default() }).collect(),
                buffers: (0..2).map(|allocation| BufferBinding { allocation, offset: 4, bytes: 44 }).collect(), scalars: vec![],
            };
            let account = model::invocation_account(&execution, &workload, DerivationLimits { instructions: 100_000, operations: 100_000 }).unwrap();
            assert!(account.is_complete(), "{:?}", account.unmapped);
            let mut vector_calls = 0;
            let mut scalar_calls = 0;
            let mut bytes = 0;
            for operation in &account.operations {
                match operation.primitive {
                    Primitive::VectorRead { space: Space::Device, ty, components } => {
                        vector_calls += operation.instances;
                        bytes += operation.instances * operation.lanes * ty.bytes() * u64::from(components);
                        assert!(operation.access.as_ref().unwrap().transactions(16).is_some());
                    }
                    Primitive::Read { space: Space::Device, ty } => { scalar_calls += operation.instances; bytes += operation.instances * operation.lanes * ty.bytes(); }
                    _ => {}
                }
            }
            assert_eq!(bytes, 11 * 4 * 32);
            assert_eq!(vector_calls, if width == 1 { 0 } else { 11 / u64::from(width) });
            assert_eq!(scalar_calls, if width == 1 { 11 } else { 11 % u64::from(width) });
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_vector_snapshots_preserve_scalar_results_tails_and_alignment() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    for (dtype, bytes) in [("f32", 4), ("i32", 4), ("u32", 4), ("f16", 2), ("bf16", 2)] {
        let payload = (0..11 * bytes).map(|i| (i * 37 + 11) as u8).collect::<Vec<_>>();
        let mut backing = vec![0; bytes]; backing.extend(&payload);
        let input = device.buffer_from(&backing).unwrap().view(bytes..backing.len()).unwrap();
        let output = device.buffer_from(&vec![0; backing.len()]).unwrap().view(bytes..backing.len()).unwrap();
        let baseline = selected(dtype, 1, true);
        let baseline = device.compile(seismic_metal::msl::prepare_execution(&baseline).unwrap().clone()).unwrap();
        device.run(&baseline, &[&input, &output], &[], 1).unwrap();
        let expected = output.read(payload.len());
        // Metal's existing scalar BF16 copy flushes the subnormal in this
        // payload. This test establishes vector/scalar correspondence, not a
        // claim that native BF16 copies preserve every storage bit.
        if dtype != "bf16" { assert_eq!(expected, payload); }
        for width in 2..=4 {
            let execution = selected(dtype, width, true);
            let emitted = seismic_metal::msl::prepare_execution(&execution).unwrap();
            assert!(emitted.source.contains(&format!("packed_{dtype}")) || emitted.source.contains("packed_float") || emitted.source.contains("packed_int") || emitted.source.contains("packed_uint") || emitted.source.contains("packed_half") || emitted.source.contains("packed_bfloat"));
            let pipeline = device.compile(emitted.clone()).unwrap();
            device.run(&pipeline, &[&input, &output], &[], 1).unwrap();
            assert_eq!(output.read(payload.len()), expected, "{dtype} width {width}");
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn guarded_vectors_preserve_partial_chunks_and_invalid_view_failures() {
    let program = compile(&[SourceFile { path: "guarded_copy.seismic.portable".into(), scope: Scope::Portable,
        text: "fn evaluate(x:tensor[11] f32,out:tensor[11] f32,n:i32,p:i32):\n  a=load(x[:n])\n  if n>0: a[p]=a[p]\n  store(a,out[:n])\n".into(),
    }], &[]).unwrap();
    let device = seismic_metal::runtime::Device::open().unwrap();
    let payload = (0..11).flat_map(|i| (i as f32 + 0.25).to_le_bytes()).collect::<Vec<_>>();
    let input = device.buffer_from(&payload).unwrap();
    let output = device.buffer_from(&[0x55; 44]).unwrap();
    for width in 2..=4 {
        let execution = prepare(&program, width, true);
        let emitted = seismic_metal::msl::prepare_execution(&execution).unwrap();
        let pipeline = device.compile(emitted.clone()).unwrap();
        for n in [0, 1, 2, 4, 7, 10, 11] {
            output.write(&[0x55; 44]);
            device.run(&pipeline, &[&input, &output], &emitted.encode_scalars(&[n as f64, 0.0]).unwrap(), 1).unwrap();
            let actual = output.read(44);
            assert_eq!(&actual[..n * 4], &payload[..n * 4], "width {width} n {n}");
            assert!(actual[n * 4..].iter().all(|b| *b == 0x55));
            let workload = ScalarWorkload { integer_domains: Vec::new(), identity: format!("partial vector {n}"),
                allocations: (0..2).map(|id| Allocation { id, bytes: 44, alignment: 4, known_bytes: Default::default() }).collect(),
                buffers: (0..2).map(|allocation| BufferBinding { allocation, offset: 0, bytes: 44 }).collect(),
                scalars: seismic_lang::abi::ScalarLayout::words(&emitted.scalars).unwrap().encode(&[n as f64, 0.0]).unwrap(),
            };
            let account = model::invocation_account(&execution, &workload, DerivationLimits { instructions: 100_000, operations: 100_000 }).unwrap();
            assert!(account.is_complete(), "{:?}", account.unmapped);
            let bytes: u64 = account.operations.iter().map(|o| match o.primitive {
                Primitive::VectorRead { components, .. } => o.instances * o.lanes * u64::from(components) * 4,
                Primitive::Read { space: Space::Device, .. } => o.instances * o.lanes * 4,
                _ => 0,
            }).sum();
            assert_eq!(bytes, n as u64 * 4 * 32);
        }
        for (n, m) in [(11., -1.), (11., 11.)] {
            assert!(device.run(&pipeline, &[&input, &output], &emitted.encode_scalars(&[n, m]).unwrap(), 1).is_err());
        }
    }
}
