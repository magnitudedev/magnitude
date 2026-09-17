use seismic_lang::{
    interp::{Arg, Interpreter, Rng, TensorData},
    program::{collect_files, compile, Program},
    repr,
    types::{Elem, Ty},
};
use std::{collections::HashMap, path::PathBuf};
fn standard() -> Program {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib");
    compile(
        &collect_files(&[root]).unwrap(),
        &["cpu".into(), "metal".into()],
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
        compare_candidate(name, shapes, scalars, loads);
    }
}
fn compare_candidate(
    name: &str,
    shapes: &[(&str, i64)],
    scalars: &[(&str, f64)],
    loads: seismic_realization::LoadStrategy,
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
        "cpu",
        &shapes,
        &elements,
        &Default::default(),
    )
    .unwrap();
    let mut kernel = seismic_cpu::compile_candidate(&lowered, loads).unwrap();
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
    kernel
        .run(
            &mut buffers
                .iter_mut()
                .map(|b| b.as_mut_slice())
                .collect::<Vec<_>>(),
            &scalar_values,
        )
        .unwrap();
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
                    kernel.ir
                );
            }
        }
    }
}
#[test]
fn packed_projection_native_matches_reference() {
    compare("projection", &[("N", 7), ("K", 64)], &[]);
}
#[test]
fn row_norm_native_matches_reference() {
    compare("rms_norm", &[("R", 3), ("W", 17)], &[("eps", 1e-6)]);
}
#[test]
fn fused_gated_projection_native_matches_reference() {
    compare("gated_projection", &[("N", 7), ("K", 64)], &[]);
}
#[test]
fn undersized_bindings_fail_before_native_execution() {
    let p = standard();
    let lowered = seismic_lang::lower::lower(
        &p,
        "projection",
        "cpu",
        &HashMap::from([("N".into(), 1), ("K".into(), 64)]),
    )
    .unwrap();
    let mut kernel = seismic_cpu::compile(&lowered).unwrap();
    let mut buffers = kernel
        .buffers()
        .iter()
        .map(|b| vec![0; b.bytes])
        .collect::<Vec<_>>();
    buffers[0].clear();
    let error = kernel
        .run(
            &mut buffers
                .iter_mut()
                .map(|b| b.as_mut_slice())
                .collect::<Vec<_>>(),
            &[],
        )
        .unwrap_err();
    assert!(error.contains("needs"));
}

#[test]
fn load_is_a_snapshot_across_later_source_writes() {
    use seismic_lang::{program::SourceFile, Scope};
    let source="fn snapshot[N](x: tensor[N] f32, out: tensor[N] f32):\n  t = load(x)\n  zero = tile[N] f32\n  for i in owned(zero): zero[i] = 0.0\n  store(zero, x)\n  store(t, out)\n";
    let program = compile(
        &[SourceFile {
            path: "snapshot.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap_or_else(|e| {
        panic!(
            "{}",
            e.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n")
        )
    });
    let lowered = seismic_lang::lower::lower(
        &program,
        "snapshot",
        "cpu",
        &HashMap::from([("N".into(), 3)]),
    )
    .unwrap();
    let mut kernel = seismic_cpu::compile(&lowered).unwrap();
    let original = [1.0f32, -2.0, 3.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    let mut input = original.clone();
    let mut output = vec![0; 12];
    kernel.run(&mut [&mut input, &mut output], &[]).unwrap();
    assert_eq!(output, original);
    assert_eq!(input, vec![0; 12]);
}

#[test]
fn repeated_native_invocation_reinitializes_logical_tiles() {
    let p = standard();
    let lowered = seismic_lang::lower::lower(
        &p,
        "projection",
        "cpu",
        &HashMap::from([("N".into(), 3), ("K".into(), 64)]),
    )
    .unwrap();
    let mut kernel = seismic_cpu::compile(&lowered).unwrap();
    let mut buffers = kernel
        .buffers()
        .iter()
        .map(|b| vec![0; b.bytes])
        .collect::<Vec<_>>();
    for _ in 0..4 {
        kernel
            .run(
                &mut buffers
                    .iter_mut()
                    .map(|b| b.as_mut_slice())
                    .collect::<Vec<_>>(),
                &[],
            )
            .unwrap();
        assert!(buffers.last().unwrap().iter().all(|b| *b == 0));
    }
}

#[test]
fn altered_loop_domain_cannot_escape_native_buffer_bounds() {
    use seismic_lang::{hir::StmtKind, program::SourceFile, sym::Sym, Scope};
    let p = compile(&[SourceFile {
        path: "copy.seismic.portable".into(), scope: Scope::Portable,
        text: "fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for i in parallel:\n    t = load(x[i:i+1])\n    store(t, out[i:i+1])\n".into(),
    }], &[]).unwrap_or_else(|e| panic!("{e:?}"));
    let mut lowered =
        seismic_lang::lower::lower(&p, "copy", "cpu", &HashMap::from([("N".into(), 4)])).unwrap();
    let StmtKind::Parallel { extents, .. } = &mut lowered.body[0].kind else {
        panic!()
    };
    extents[0] = Sym::constant(5);
    let mut kernel = seismic_cpu::compile(&lowered).unwrap();
    let mut input = vec![0; 16];
    let mut output = vec![0; 16];
    let error = kernel.run(&mut [&mut input, &mut output], &[]).unwrap_err();
    assert!(error.contains("out-of-bounds"));
}

#[path = "../../../../validation/support/scalar_cases.rs"]
mod scalar_cases;
#[test]
fn value_semantics_and_precision_boundaries() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        scalar_cases::exercise(
            |lowered, buffers, scalars| {
                let mut kernel = seismic_cpu::compile_candidate(lowered, loads).unwrap();
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
            "cpu",
        );
    }
}

#[test]
fn bounded_embedding_index_matches_reference() {
    compare("embedding", &[("V", 4), ("K", 64)], &[("token", 3.0)]);
}
#[test]
fn invalid_index_is_rejected_even_without_memory_indexing() {
    use seismic_lang::{program::SourceFile, Scope};
    let p=compile(&[SourceFile{path:"bounded.seismic.portable".into(),scope:Scope::Portable,text:"fn bounded(pos: index[4], out: tensor[1] i32):\n  t = tile[1] i32\n  for i in owned(t): t[i] = pos\n  store(t,out)\n".into()}],&[]).unwrap();
    let l = seismic_lang::lower::lower(&p, "bounded", "cpu", &HashMap::new()).unwrap();
    let mut kernel = seismic_cpu::compile(&l).unwrap();
    let mut out = [0u8; 4];
    for invalid in [-1.0, 4.0, 0.5] {
        assert!(kernel.run(&mut [&mut out], &[invalid]).is_err());
        assert_eq!(out, [0; 4]);
    }
    kernel.run(&mut [&mut out], &[3.0]).unwrap();
    assert_eq!(out, 3i32.to_le_bytes());
    let mut interpreter = Interpreter::new(&p);
    let id = interpreter.add_tensor(TensorData::dense(
        seismic_lang::types::DType::I32,
        vec![1],
        vec![0.0],
    ));
    assert!(interpreter
        .run(
            "bounded",
            &[Arg::Scalar(4.0), Arg::Tensor(id)],
            &HashMap::new()
        )
        .is_err());
}

#[test]
fn two_phase_argmax_matches_reference() {
    compare("argmax_row", &[("V", 128), ("B", 64)], &[]);
}

#[path = "../../../../validation/support/stream_cases.rs"]
mod stream_cases;
#[test]
fn runtime_stream_domains_and_piece_tails() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        stream_cases::exercise(
            |lowered, buffers| {
                let mut kernel = seismic_cpu::compile_candidate(lowered, loads)?;
                kernel.run(
                    &mut buffers
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    &[],
                )
            },
            "cpu",
        );
    }
}

#[path = "../../../../validation/support/attention_cases.rs"]
mod attention_cases;
#[test]
fn standard_streaming_attention_matches_independent_reference() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        attention_cases::exercise(
            |lowered, buffers, scalars| {
                let mut kernel = seismic_cpu::compile_candidate(lowered, loads)?;
                kernel.run(
                    &mut buffers
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    scalars,
                )
            },
            "cpu",
        );
    }
}

#[test]
fn resident_views_retain_storage_and_preserve_snapshot_aliases() {
    use seismic_cpu::Buffer;
    use seismic_lang::{program::SourceFile, Scope};
    let p = compile(
        &[SourceFile {
            path: "copy.seismic.portable".into(),
            scope: Scope::Portable,
            text:
                "fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  t = load(x)\n  store(t,out)\n"
                    .into(),
        }],
        &[],
    )
    .unwrap();
    let l =
        seismic_lang::lower::lower(&p, "copy", "cpu", &HashMap::from([("N".into(), 4)])).unwrap();
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        let mut kernel = seismic_cpu::compile_candidate(&l, loads).unwrap();
        let data = [1f32, 2., 3., 4., 5., 6.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let root = Buffer::from_bytes(&data).unwrap();
        let source = root.view(0..16).unwrap();
        let destination = root.view(4..20).unwrap();
        let retained = root.clone();
        drop(root);
        kernel.run_resident(&[source, destination], &[]).unwrap();
        let mut got = vec![0; 24];
        retained.read(&mut got).unwrap();
        assert_eq!(
            got,
            [1f32, 1., 2., 3., 4., 6.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(kernel
            .run_resident(
                &[retained.view(1..17).unwrap(), retained.view(4..20).unwrap()],
                &[]
            )
            .unwrap_err()
            .contains("alignment"));
        assert!(retained.view(0..25).is_err());
        assert!(retained.view(0..4).unwrap().write(&[0; 5]).is_err());
    }
}
