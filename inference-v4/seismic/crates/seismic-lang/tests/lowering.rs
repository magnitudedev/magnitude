use seismic_lang::{
    hir::StmtKind,
    lower::{lower_with, Options},
    program::{compile, SourceFile},
    Scope,
};
use std::{collections::HashMap, path::PathBuf};

fn stream_capacity(piece: Option<i64>, dynamic: bool) -> Result<Option<i64>, String> {
    let slice = if dynamic {
        "x[visible[0]:visible[1]]"
    } else {
        "x"
    };
    let text = format!("fn stream[T](x: tensor[T] f32, visible: tensor[2] i32, out: tensor[1] f32):\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  for t in load({slice}, over=0):\n    acc[0] += reduce(t, 0, sum)\n  store(acc, out)\n");
    let program = compile(
        &[SourceFile {
            path: PathBuf::from("stream.seismic.portable"),
            text,
            scope: Scope::Portable,
        }],
        &["cpu".into()],
    )
    .map_err(|e| format!("{e:?}"))?;
    let lowered = lower_with(
        &program,
        "stream",
        "cpu",
        &HashMap::from([("T".into(), 137)]),
        &Options { piece },
    )?;
    lowered
        .body
        .iter()
        .find_map(|stmt| match &stmt.kind {
            StmtKind::LoadLoop { capacity, .. } => Some(Ok(*capacity)),
            _ => None,
        })
        .unwrap_or_else(|| Err("missing stream".into()))
}

#[test]
fn dynamic_stream_capacity_is_explicit_or_structurally_bounded() {
    assert_eq!(stream_capacity(None, true).unwrap(), Some(137));
    assert_eq!(stream_capacity(Some(17), true).unwrap(), Some(17));
    assert_eq!(stream_capacity(None, false).unwrap(), None);
    assert_eq!(stream_capacity(Some(17), false).unwrap(), Some(17));
}

#[test]
fn nonpositive_piece_capacity_is_rejected_before_arithmetic() {
    for dynamic in [false, true] {
        for capacity in [0, -1, i64::MIN] {
            assert!(stream_capacity(Some(capacity), dynamic)
                .unwrap_err()
                .contains("must be positive"));
        }
    }
}

#[test]
fn reference_empty_stream_has_no_body_effects() {
    use seismic_lang::{
        interp::{Arg, Interpreter, TensorData},
        types::DType,
    };
    let text = include_str!("../../../../validation/programs/scalar-semantics.seismic.portable");
    let p = compile(
        &[SourceFile {
            path: "semantics.seismic.portable".into(),
            text: text.into(),
            scope: Scope::Portable,
        }],
        &[],
    )
    .unwrap();
    let mut interpreter = Interpreter::new(&p);
    let x = interpreter.add_tensor(TensorData::dense(DType::F32, vec![1], vec![7.0]));
    let out = interpreter.add_tensor(TensorData::dense(DType::F32, vec![1], vec![9.0]));
    interpreter
        .run(
            "empty_stream",
            &[Arg::Tensor(x), Arg::Tensor(out)],
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(interpreter.tensors[out].get(0), 0.0);
}
