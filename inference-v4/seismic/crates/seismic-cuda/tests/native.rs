use seismic_lang::{
    interp::{Arg, Interpreter, Rng, TensorData},
    program::{collect_files, compile, Program},
    repr,
    types::{Elem, Ty},
};
use seismic_realization::Dispatch;
use std::{collections::HashMap, path::PathBuf};
fn standard() -> Program {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib");
    compile(
        &collect_files(&[root]).unwrap(),
        &["cpu".into(), "metal".into(), "cuda".into()],
    )
    .unwrap_or_else(|e| {
        panic!(
            "{}",
            e.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n")
        )
    })
}
fn compare(name: &str, shapes: &[(&str, i64)], scalars: &[(&str, f64)]) {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        compare_candidate(name, shapes, scalars, loads, Dispatch::ParallelRoot);
    }
}
fn compare_candidate(
    name: &str,
    shapes: &[(&str, i64)],
    scalars: &[(&str, f64)],
    loads: seismic_realization::LoadStrategy,
    dispatch: Dispatch,
) {
    let program = standard();
    let shapes = shapes
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect::<HashMap<_, _>>();
    let scalars = scalars
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect::<HashMap<_, _>>();
    let f = program.functions.iter().find(|f| f.name == name).unwrap();
    let elements = f
        .elem_params
        .iter()
        .map(|p| (p.clone(), Elem::Dtype(seismic_lang::types::DType::BF16)))
        .collect();
    let mut rng = Rng(71);
    let mut data = HashMap::new();
    let mut interpreter = Interpreter::new(&program);
    let mut arguments = Vec::new();
    for (name, ty) in &f.params {
        match ty {
            Ty::Tensor(sh) => {
                let dims = sh
                    .shape
                    .iter()
                    .map(|s| s.eval(&|p| shapes.get(p).copied()).unwrap() as usize)
                    .collect();
                let tensor = match &sh.elem {
                    Elem::Dtype(d) => TensorData::random_dense(&mut rng, *d, dims),
                    Elem::Repr(r) => {
                        TensorData::random_packed(&mut rng, repr::lookup(r).unwrap(), dims)
                    }
                    Elem::Param(_) => {
                        TensorData::random_dense(&mut rng, seismic_lang::types::DType::BF16, dims)
                    }
                };
                let id = interpreter.add_tensor(tensor.clone());
                arguments.push(Arg::Tensor(id));
                data.insert(name.clone(), (tensor, id));
            }
            Ty::Scalar(_) => arguments.push(Arg::Scalar(scalars[name])),
            _ => panic!(),
        }
    }
    interpreter.run(name, &arguments, &shapes).unwrap();
    let lowered = seismic_lang::lower::lower_specialized(
        &program,
        name,
        "cuda",
        &shapes,
        &elements,
        &Default::default(),
    )
    .unwrap();
    let device = seismic_cuda::Device::open(0).unwrap();
    let mut kernel = device
        .compile_sequence(
            &lowered,
            seismic_realization::ScalarOptions { dispatch, loads },
            32,
        )
        .unwrap();
    eprintln!(
        "device={:?}; phases={}; resources={:?}",
        device.info,
        kernel.phase_count(),
        kernel
            .realizations()
            .map(|(program, native)| (program.scratch_bytes, native))
            .collect::<Vec<_>>()
    );
    let specs = kernel.buffers().to_vec();
    let mut buffers = specs
        .iter()
        .map(|s| {
            let parts = data[&s.parameter].0.device_bytes();
            parts[match s.plane.as_str() {
                "" | "words" => 0,
                "scale" => 1,
                _ => 2,
            }]
            .clone()
        })
        .collect::<Vec<_>>();
    let scalar_values = kernel
        .scalars()
        .iter()
        .map(|parameter| scalars[&parameter.name])
        .collect::<Vec<_>>();
    let resident = buffers
        .iter()
        .map(|bytes| device.buffer_from(bytes).unwrap())
        .collect::<Vec<_>>();
    kernel.execute(&resident, &scalar_values, false).unwrap();
    for (native, host) in resident.iter().zip(&mut buffers) {
        native.read(host).unwrap();
    }
    for (slot, bytes) in specs.iter().zip(buffers) {
        if slot.plane.is_empty() {
            let (mut got, id) = data[&slot.parameter].clone();
            got.load_device_bytes(&bytes);
            let expected = &interpreter.tensors[id];
            for i in 0..got.shape().iter().product() {
                let (a, b) = (got.get(i), expected.get(i));
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 0.002 + 0.005 * b.abs(),
                    "{name} {}[{i}] expected {b}, got {a}\n{}",
                    slot.parameter,
                    kernel.ptx_sources().collect::<Vec<_>>().join("\n")
                );
            }
        }
    }
}
#[test]
#[ignore = "requires a CUDA SM 8.0+ device; execute explicitly on a qualification host"]
fn packed_projection_native_matches_reference() {
    compare("projection", &[("N", 37), ("K", 64)], &[]);
}
#[test]
#[ignore = "requires a CUDA SM 8.0+ device; execute explicitly on a qualification host"]
fn row_norm_native_matches_reference() {
    compare("rms_norm", &[("R", 3), ("W", 17)], &[("eps", 1e-6)]);
}

fn copy_program(n: i64) -> seismic_lang::lowered_ir::LoweredIr {
    use seismic_lang::{program::SourceFile, Scope};
    let p=compile(&[SourceFile {path:"copy.seismic.portable".into(),scope:Scope::Portable,text:"fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for i in parallel:\n    t = load(x[i:i+1])\n    store(t, out[i:i+1])\n".into()}],&[]).unwrap_or_else(|e|panic!("{e:?}"));
    seismic_lang::lower::lower(&p, "copy", "cuda", &HashMap::from([("N".into(), n)])).unwrap()
}
#[test]
#[ignore = "requires CUDA hardware"]
fn native_bindings_bounds_and_repeated_submission() {
    use seismic_realization::Dispatch;
    let device = seismic_cuda::Device::open(0).unwrap();
    let lowered = copy_program(37);
    let mut kernel = device
        .compile(&lowered, Dispatch::ParallelRoot, 32)
        .unwrap();
    assert!(kernel.launch().unwrap_err().contains("uploaded"));
    let mut too_small = vec![0u8; 4];
    let mut output = vec![0; 37 * 4];
    assert!(kernel
        .run(&mut [&mut too_small, &mut output], &[])
        .unwrap_err()
        .contains("needs"));
    let mut input = (0..37)
        .flat_map(|i| (i as f32 - 7.0).to_le_bytes())
        .collect::<Vec<_>>();
    kernel.run(&mut [&mut input, &mut output], &[]).unwrap();
    assert_eq!(output, input);
    for _ in 0..3 {
        kernel.launch().unwrap();
    }
    kernel.download(&mut [&mut input, &mut output]).unwrap();
    assert_eq!(output, input);
    input.fill(0);
    kernel.run(&mut [&mut input, &mut output], &[]).unwrap();
    assert_eq!(output, input);
    let mut invalid = lowered;
    let seismic_lang::ir::StmtKind::Parallel { extents, .. } = &mut invalid.body[0].kind else {
        panic!()
    };
    extents[0] = seismic_lang::sym::Sym::constant(38);
    let mut kernel = device
        .compile(&invalid, Dispatch::ParallelRoot, 32)
        .unwrap();
    assert!(kernel
        .run(&mut [&mut input, &mut output], &[])
        .unwrap_err()
        .contains("work item 37 failed"));
}
#[test]
#[ignore = "requires CUDA hardware"]
fn empty_domain_does_not_launch_or_touch_memory() {
    let device = seismic_cuda::Device::open(0).unwrap();
    let mut kernel = device
        .compile(
            &copy_program(0),
            seismic_realization::Dispatch::ParallelRoot,
            32,
        )
        .unwrap();
    assert_eq!(kernel.work_items(), 0);
    kernel.run(&mut [&mut [], &mut []], &[]).unwrap();
}
#[test]
#[ignore = "requires CUDA hardware"]
fn sequential_snapshot_keeps_loaded_values_after_store() {
    use seismic_lang::{program::SourceFile, Scope};
    let p=compile(&[SourceFile{path:"snapshot.seismic.portable".into(),scope:Scope::Portable,text:"fn snapshot[N](x: tensor[N] f32, out: tensor[N] f32):\n  t = load(x)\n  zero = tile[N] f32\n  for i in owned(zero): zero[i] = 0.0\n  store(zero, x)\n  store(t, out)\n".into()}],&[]).unwrap_or_else(|e|panic!("{e:?}"));
    let lowered =
        seismic_lang::lower::lower(&p, "snapshot", "cuda", &HashMap::from([("N".into(), 17)]))
            .unwrap();
    let device = seismic_cuda::Device::open(0).unwrap();
    let mut kernel = device
        .compile(&lowered, seismic_realization::Dispatch::Sequential, 32)
        .unwrap();
    let mut input = (0..17)
        .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
        .collect::<Vec<_>>();
    let expected = input.clone();
    let mut output = vec![0; 17 * 4];
    kernel.run(&mut [&mut input, &mut output], &[]).unwrap();
    assert_eq!(output, expected);
    assert!(input.iter().all(|b| *b == 0));
}

#[test]
#[ignore = "requires CUDA hardware"]
fn gated_projection_uses_qualified_exp_implementation() {
    compare("gated_projection", &[("N", 37), ("K", 64)], &[]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn exponential_full_range_and_special_values() {
    use seismic_lang::{program::SourceFile, Scope};
    let p=compile(&[SourceFile{path:"exp.seismic.portable".into(),scope:Scope::Portable,text:"fn exponential[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    y = tile[1] f32\n    for j in owned(y): y[j] = exp(t[j])\n    store(y, out[row:row+1])\n".into()}],&[]).unwrap_or_else(|e|panic!("{e:?}"));
    let mut values = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7f800001),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    // Dense coverage of the finite output interval plus deterministic raw-bit
    // samples, including threshold neighbors and subnormal outputs.
    for i in 0..200_001 {
        values.push(-105.0 + 195.0 * (i as f32 / 200_000.0));
    }
    for bits in [
        0x39000000u32,
        0x3eb17218,
        0x3f851592,
        0x42aeac50,
        0x42b17218,
        0x42cff1b5,
    ] {
        for delta in -4i64..=4 {
            let x = f32::from_bits((i64::from(bits) + delta) as u32);
            values.push(x);
            values.push(-x);
        }
    }
    let mut state = 0x5eedu32;
    for _ in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values.push(f32::from_bits(state));
    }
    let lowered = seismic_lang::lower::lower(
        &p,
        "exponential",
        "cuda",
        &HashMap::from([("N".into(), values.len() as i64)]),
    )
    .unwrap();
    let device = seismic_cuda::Device::open(0).unwrap();
    let mut kernel = device
        .compile(&lowered, seismic_realization::Dispatch::ParallelRoot, 128)
        .unwrap();
    let mut input = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let mut output = vec![0u8; input.len()];
    kernel.run(&mut [&mut input, &mut output], &[]).unwrap();
    let mut max_ulp = 0;
    for (x, bytes) in values.iter().zip(output.chunks_exact(4)) {
        let actual = f32::from_le_bytes(bytes.try_into().unwrap());
        // f64 exp rounded once provides a more precise independent oracle than
        // comparing against the f32 algorithm adapted by the backend.
        let expected = (*x as f64).exp() as f32;
        if expected.is_nan() {
            assert!(actual.is_nan());
            continue;
        }
        if expected.is_infinite() || expected == 0.0 {
            assert_eq!(actual, expected, "exp({x})");
            continue;
        }
        let ulp = actual.to_bits().abs_diff(expected.to_bits());
        max_ulp = max_ulp.max(ulp);
        assert!(
            actual.is_finite() && actual >= 0.0 && ulp <= 2,
            "exp({x}): expected {expected}, got {actual}, ULP {ulp}"
        );
    }
    eprintln!(
        "CUDA exp: {} inputs, max {max_ulp} ULP versus f64 reference rounded to f32",
        values.len()
    );
}

#[path = "../../../../validation/support/scalar_cases.rs"]
mod scalar_cases;
#[test]
#[ignore = "requires CUDA hardware"]
fn value_semantics_and_precision_boundaries() {
    let device = seismic_cuda::Device::open(0).unwrap();
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        scalar_cases::exercise(
            |lowered, buffers, scalars| {
                let mut kernel = device
                    .compile_candidate(
                        lowered,
                        seismic_realization::ScalarOptions {
                            dispatch: seismic_realization::Dispatch::Sequential,
                            loads,
                        },
                        32,
                    )
                    .unwrap();
                kernel
                    .run(
                        &mut buffers
                            .iter_mut()
                            .map(Vec::as_mut_slice)
                            .collect::<Vec<_>>(),
                        scalars,
                    )
                    .unwrap();
            },
            "cuda",
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn event_timing_preserves_inputs_and_context_lifetime() {
    use seismic_lang::{program::SourceFile, Scope};
    let p=compile(&[SourceFile {path:"timing.seismic.portable".into(),scope:Scope::Portable,text:"fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    store(t,out[row:row+1])\n".into()}],&[]).unwrap();
    let l = seismic_lang::lower::lower(&p, "copy", "cuda", &HashMap::from([("N".into(), 1025)]))
        .unwrap();
    let device = seismic_cuda::Device::open(0).unwrap();
    let mut kernel = device
        .compile(&l, seismic_realization::Dispatch::ParallelRoot, 64)
        .unwrap();
    drop(device);
    let original = (0..1025)
        .flat_map(|i| (i as f32 - 512.0).to_le_bytes())
        .collect::<Vec<_>>();
    let mut values = [original.clone(), vec![0; original.len()]];
    assert!(kernel.launch_timed().is_err());
    kernel
        .upload(
            &values.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
            &[],
        )
        .unwrap();
    for _ in 0..3 {
        let duration = kernel.launch_timed().unwrap();
        assert!(duration.is_finite() && duration >= 0.0);
        kernel
            .download(&mut values.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(values[0], original);
        assert_eq!(values[1], original);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn bounded_embedding_index_matches_reference() {
    compare("embedding", &[("V", 4), ("K", 64)], &[("token", 3.0)]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_phase_argmax_sequential_candidate_matches_reference() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        compare_candidate(
            "argmax_row",
            &[("V", 128), ("B", 64)],
            &[],
            loads,
            Dispatch::Sequential,
        );
    }
}

#[path = "../../../../validation/support/stream_cases.rs"]
mod stream_cases;
#[test]
#[ignore = "requires CUDA hardware"]
fn runtime_stream_domains_and_piece_tails() {
    let device = seismic_cuda::Device::open(0).unwrap();
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        stream_cases::exercise(
            |lowered, buffers| {
                let mut kernel = device.compile_candidate(
                    lowered,
                    seismic_realization::ScalarOptions {
                        dispatch: Dispatch::Sequential,
                        loads,
                    },
                    32,
                )?;
                kernel.run(
                    &mut buffers
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    &[],
                )
            },
            "cuda",
        );
    }
}
#[path = "../../../../validation/support/attention_cases.rs"]
mod attention_cases;
#[test]
#[ignore = "requires CUDA hardware"]
fn standard_streaming_attention_matches_independent_reference() {
    let device = seismic_cuda::Device::open(0).unwrap();
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        attention_cases::exercise(
            |lowered, buffers, scalars| {
                let mut kernel = device.compile_candidate(
                    lowered,
                    seismic_realization::ScalarOptions {
                        dispatch: Dispatch::ParallelRoot,
                        loads,
                    },
                    32,
                )?;
                kernel.run(
                    &mut buffers
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    scalars,
                )
            },
            "cuda",
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn resident_views_chain_without_host_roundtrips_and_retain_owners() {
    let device = seismic_cuda::Device::open(0).unwrap();
    let lowered = copy_program(37);
    let mut first = device
        .compile(&lowered, Dispatch::ParallelRoot, 32)
        .unwrap();
    let mut second = device
        .compile(&lowered, Dispatch::ParallelRoot, 32)
        .unwrap();
    let expected = (0..37)
        .flat_map(|i| (i as f32 - 17.0).to_le_bytes())
        .collect::<Vec<_>>();
    let mut initial = vec![0x5a; expected.len() + 8];
    initial[4..4 + expected.len()].copy_from_slice(&expected);
    let backing = device.buffer_from(&initial).unwrap();
    let input = backing.view(4..4 + expected.len()).unwrap();
    let intermediate = device.buffer(expected.len()).unwrap();
    let final_backing = device.buffer_from(&vec![0xa5; expected.len() + 8]).unwrap();
    let output = final_backing.view(4..4 + expected.len()).unwrap();
    first.bind(&[input, intermediate.clone()], &[]).unwrap();
    second.bind(&[intermediate, output], &[]).unwrap();
    drop(backing);
    drop(device);
    first.launch().unwrap();
    second.launch().unwrap();
    let mut got = vec![0; expected.len() + 8];
    final_backing.read(&mut got).unwrap();
    assert_eq!(&got[4..4 + expected.len()], expected.as_slice());
    assert_eq!(&got[..4], &[0xa5; 4]);
    assert_eq!(&got[4 + expected.len()..], &[0xa5; 4]);
    assert!(final_backing.view(1..usize::MAX).is_err());
    assert!(final_backing
        .view(std::ops::Range { start: 4, end: 3 })
        .is_err());
    let short = final_backing.view(0..4).unwrap();
    assert!(short.write(&[0; 5]).is_err());
    assert!(short.read(&mut [0; 5]).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn resident_bindings_reject_cross_context_misalignment_and_stale_launches() {
    let device = seismic_cuda::Device::open(0).unwrap();
    let other = seismic_cuda::Device::open(0).unwrap();
    let mut kernel = device
        .compile(&copy_program(4), Dispatch::ParallelRoot, 32)
        .unwrap();
    let input = device.buffer_from(&[0; 20]).unwrap();
    let out = device.buffer(16).unwrap();
    kernel.bind(&[input.clone(), out.clone()], &[]).unwrap();
    kernel.launch().unwrap();
    let foreign = other.buffer(16).unwrap();
    assert!(kernel
        .bind(&[foreign, out.clone()], &[])
        .unwrap_err()
        .contains("different context"));
    assert!(kernel.launch().is_err());
    assert!(kernel
        .bind(&[input.view(1..17).unwrap(), out.clone()], &[])
        .unwrap_err()
        .contains("alignment"));
    assert!(kernel
        .bind(&[input.view(0..15).unwrap(), out.clone()], &[])
        .is_err());
    kernel.bind(&[input, out], &[]).unwrap();
    kernel.launch().unwrap();
}

#[test]
#[ignore = "requires CUDA hardware"]
fn logarithm_and_trigonometry_full_range_and_special_values() {
    use seismic_lang::{program::SourceFile, Scope};
    let mut values = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7f800001),
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
        1.0,
        -1.0,
    ];
    let mut state = 0x54adee19u32;
    for _ in 0..400_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values.push(f32::from_bits(state));
    }
    for i in -50_000..=50_000 {
        values.push(i as f32 * 0.03125);
    }
    // All normal exponent boundaries plus neighborhoods of range-reduction
    // thresholds and many multiples of pi/2, where absolute errors can hide ULPs.
    for exponent in 1u32..255 {
        for delta in -3i64..=3 {
            let x = f32::from_bits(((exponent << 23) as i64 + delta) as u32);
            values.extend([x, -x]);
        }
    }
    for bits in [
        0x39800000u32,
        0x3f490fda,
        0x4016cbe3,
        0x407b53d1,
        0x40afeddf,
        0x40e231d5,
        0x4dc90fdb,
        0x3f800000,
    ] {
        for delta in -16i64..=16 {
            let x = f32::from_bits((bits as i64 + delta) as u32);
            values.extend([x, -x]);
        }
    }
    for n in 1..10_000 {
        let bits = (f64::from(n) * std::f64::consts::FRAC_PI_2) as f32;
        for delta in -1i64..=1 {
            let x = f32::from_bits((i64::from(bits.to_bits()) + delta) as u32);
            values.extend([x, -x]);
        }
    }
    let device = seismic_cuda::Device::open(0).unwrap();
    for operation in ["log", "sin", "cos"] {
        let text=format!("fn math[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = {operation}(t[i])\n    store(y,out[row:row+1])\n");
        let program = compile(
            &[SourceFile {
                path: "math.seismic.portable".into(),
                scope: Scope::Portable,
                text,
            }],
            &[],
        )
        .unwrap();
        let lowered = seismic_lang::lower::lower(
            &program,
            "math",
            "cuda",
            &HashMap::from([("N".into(), values.len() as i64)]),
        )
        .unwrap();
        let mut kernel = device
            .compile_candidate(
                &lowered,
                seismic_realization::ScalarOptions {
                    dispatch: seismic_realization::Dispatch::ParallelRoot,
                    loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
                },
                128,
            )
            .unwrap();
        let input = device
            .buffer_from(
                &values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.buffer(values.len() * 4).unwrap();
        kernel.bind(&[input, output.clone()], &[]).unwrap();
        kernel.launch().unwrap();
        let mut got = vec![0; values.len() * 4];
        output.read(&mut got).unwrap();
        let mut maximum = 0;
        for (input, bytes) in values.iter().zip(got.chunks_exact(4)) {
            let actual = f32::from_le_bytes(bytes.try_into().unwrap());
            let x = f64::from(*input);
            let expected = match operation {
                "log" => x.ln(),
                "sin" => x.sin(),
                "cos" => x.cos(),
                _ => unreachable!(),
            } as f32;
            if expected.is_nan() {
                assert!(actual.is_nan(), "{operation}({input}) returned {actual}");
                continue;
            }
            if expected.is_infinite() || expected == 0.0 {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{operation}({input})");
                continue;
            }
            let ordered = |x: f32| {
                let bits = x.to_bits();
                if bits >> 31 != 0 {
                    !bits
                } else {
                    bits | 0x80000000
                }
            };
            let ulp = ordered(actual).abs_diff(ordered(expected));
            maximum = maximum.max(ulp);
            assert!(
                actual.is_finite() && ulp <= 2,
                "{operation}({input}, bits {:08x}): got {actual}, expected {expected}, ULP {ulp}",
                input.to_bits()
            );
        }
        eprintln!(
            "CUDA {operation}: {} inputs, max {maximum} ULP versus f64 reference rounded to f32",
            values.len()
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn rotary_and_recurrent_preparation_use_native_math() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        for dispatch in [Dispatch::Sequential, Dispatch::ParallelRoot] {
            compare_candidate(
                "attention_prepare",
                &[("H", 4), ("G", 2), ("R", 4), ("S", 4), ("T", 8)],
                &[("pos", 3.), ("theta", 10000.), ("eps", 1e-6)],
                loads,
                dispatch,
            );
            compare_candidate(
                "recurrent_prepare",
                &[("NK", 1), ("NV", 2), ("W", 8), ("C", 4)],
                &[("eps", 1e-6)],
                loads,
                dispatch,
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn native_artifacts_do_not_allocate_invocation_storage() {
    let device = seismic_cuda::Device::open(0).unwrap();
    // The invocation would need hundreds of GB of scratch, independent of tensors.
    let lowered = copy_program(1i64 << 36);
    let artifacts = device.compile_artifacts(&lowered, seismic_realization::ScalarOptions {
        dispatch: Dispatch::ParallelRoot,
        loads: seismic_realization::LoadStrategy::Materialize,
    }, 128).unwrap();
    assert_eq!(artifacts.len(), 1);
    let artifact = &artifacts[0];
    assert_eq!(artifact.work_items, 1u64 << 36);
    assert_eq!(artifact.blocks, 1u32 << 29);
    assert!(artifact.image.cubin.starts_with(b"\x7fELF"));
    assert_eq!(artifact.image.driver_version, device.info.driver_version);
    assert_eq!(artifact.image.compute_capability, device.info.compute_capability);
    assert!(artifact.native.registers_per_thread > 0);
    assert!(artifact.native.max_active_blocks_per_multiprocessor > 0);
    // Inspection and execution use one compiler, retaining the image actually loaded.
    let small = copy_program(17);
    let options = seismic_realization::ScalarOptions {
        dispatch: Dispatch::ParallelRoot,
        loads: seismic_realization::LoadStrategy::Materialize,
    };
    let inspected = device.compile_artifacts(&small, options, 128).unwrap();
    let executed = device.compile_candidate(&small, options, 128).unwrap();
    assert_eq!(inspected[0].ptx, executed.ptx);
    assert_eq!(inspected[0].image.cubin, executed.native_image().cubin);
}
