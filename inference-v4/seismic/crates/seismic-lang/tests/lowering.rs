use seismic_lang::{
    ir::StmtKind,
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

fn alternatives_program() -> seismic_lang::program::Program {
    compile(
        &[
            SourceFile {
                path: "choices.seismic.portable".into(),
                scope: Scope::Portable,
                text: "construct leaf[N](x: tile[N] f32):\n  for i in owned(x): x[i] = x[i] + 1.0\nconstruct outer[N](x: tile[N] f32):\n  leaf(x)\nfn entry[N](x: tensor[N] f32, out: tensor[N] f32):\n  t = load(x)\n  outer(t)\n  store(t, out)\n".into(),
            },
            SourceFile {
                path: "choices.seismic.cpu".into(),
                scope: Scope::Backend("cpu".into()),
                text: "lower leaf: portable\nlower leaf[N](x: tile[N] f32):\n  for i in owned(x): x[i] = 1.0 + x[i]\nlower outer: portable\nlower outer[N](x: tile[N] f32):\n  for i in owned(x): x[i] = x[i] + 1.0\n".into(),
            },
        ],
        &["cpu".into()],
    ).unwrap()
}

#[test]
fn all_applicable_bodies_are_visible_and_replayed_without_fallback() {
    use seismic_lang::lower::lower_selected;
    use seismic_lang::lowered_ir::{Choice, Alternative, DecisionKind};
    let program = alternatives_program();
    let mut visited = Vec::new();
    let lowered = lower_selected(&program, "entry", "cpu",
        &HashMap::from([("N".into(), 8)]), &HashMap::new(), &Options::default(),
        &mut |decision| {
            visited.push(decision.clone());
            Ok(Alternative::Body(Choice::Portable))
        }).unwrap();
    assert_eq!(visited.iter().map(|d| match &d.kind { DecisionKind::Construct { name, .. } => name.as_str(), _ => panic!("unexpected producer") }).collect::<Vec<_>>(), ["outer", "leaf"]);
    assert!(visited.iter().all(|d| d.alternatives == [Alternative::Body(Choice::Block(1)), Alternative::Body(Choice::Portable)]));
    assert!(lowered.selections.iter().all(|s| s.choice == Choice::Portable));
    let mut visited = Vec::new();
    lower_selected(&program, "entry", "cpu",
        &HashMap::from([("N".into(), 8)]), &HashMap::new(), &Options::default(),
        &mut |decision| {
            visited.push(match &decision.kind { DecisionKind::Construct { name, .. } => name.clone(), _ => panic!("unexpected producer") });
            Ok(Alternative::Body(Choice::Block(1)))
        }).unwrap();
    assert_eq!(visited, ["outer"]); // This expansion contains no nested construct.
    for invalid in [Choice::Block(0), Choice::Block(usize::MAX)] {
        let error = lower_selected(&program, "entry", "cpu",
            &HashMap::from([("N".into(), 8)]), &HashMap::new(), &Options::default(),
            &mut |_| Ok(Alternative::Body(invalid.clone()))).unwrap_err();
        assert!(error.contains("not applicable"), "{error}");
    }
}

#[test]
fn expansion_space_covers_branches_with_different_nested_decisions() {
    use seismic_lang::lowered_ir::{Choice, Alternative};
    use seismic_lang::lower::alternatives::{Space, Specialization};
    let program = alternatives_program();
    let shapes = HashMap::from([("N".into(), 8)]);
    let elements = HashMap::new();
    let options = Options::default();
    let mut space = Space::new(Specialization {
        program: &program, entry: "entry", backend: "cpu",
        shapes: &shapes, elements: &elements, options: &options,
    });
    assert!(!space.exhausted());
    let mut paths = Vec::new();
    for attempt in space.by_ref() {
        let lowered = attempt.result.unwrap();
        assert_eq!(lowered.decisions, attempt.steps);
        let replay = |records: &[seismic_lang::lowered_ir::DecisionRecord]| {
            seismic_lang::lower::alternatives::replay(Specialization {
                program: &program, entry: "entry", backend: "cpu",
                shapes: &shapes, elements: &elements, options: &options,
            }, records)
        };
        let repeated = replay(&attempt.steps).unwrap();
        assert_eq!(format!("{:?}", lowered.body), format!("{:?}", repeated.body));
        assert!(replay(&[]).is_err());
        let mut extra = attempt.steps.clone();
        extra.push(extra[0].clone());
        assert!(replay(&extra).unwrap_err().contains("unused"));
        let mut corrupted = attempt.steps.clone();
        corrupted[0].domain.alternatives.clear();
        assert!(replay(&corrupted).unwrap_err().contains("domain changed"));
        paths.push(attempt.steps.into_iter().map(|s| s.selected).collect::<Vec<_>>());
    }
    assert!(space.exhausted());
    assert_eq!(paths, [
        vec![Alternative::Body(Choice::Block(1))],
        vec![Alternative::Body(Choice::Portable), Alternative::Body(Choice::Block(1))],
        vec![Alternative::Body(Choice::Portable), Alternative::Body(Choice::Portable)],
    ]);
    assert!(space.next().is_none());
}
