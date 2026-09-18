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
    let text = format!("fn stream[T](x: tensor[T] f32, visible: tensor[2] i32, out: tensor[1] f32):\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  t = load({slice})\n  acc[0] = reduce(t,0,sum,ordered=true)\n  store(acc, out)\n");
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
        &Options { piece, ..Default::default() },
    )?;
    use seismic_lang::lowered_ir::{DecisionKind,Alternative};
    let selected=lowered.decisions.iter().find_map(|d|match (&d.domain.kind,&d.selected){
        (DecisionKind::Stream{..},Alternative::StreamCapacity(n))=>Some(*n),_=>None
    }).ok_or("missing derived stream capacity")?;
    if dynamic {
        assert!(lowered.body.iter().any(|s|matches!(s.kind,StmtKind::LoadLoop{..})));
        Ok(Some(selected))
    }else if selected<137 {
        assert!(lowered.body.iter().any(|s|matches!(s.kind,StmtKind::Range{..})));
        Ok(Some(selected))
    }else{Ok(None)}
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
fn stream_piece_domains_are_independent_and_symbolic() {
    use seismic_lang::{lower::alternatives::{expand, Expansion, Specialization}, lowered_ir::{Alternative, DecisionKind}};
    let program = compile(&[SourceFile {
        path: "pieces.seismic.portable".into(), scope: Scope::Portable,
        text: "fn pieces[N](x: tensor[N] f32, y: tensor[3] f32, out:tensor[2] f32):\n  a = load(x)\n  b = load(y)\n  result = tile[2] f32\n  for i in owned(result): result[i] = 0.0\n  result[0] = reduce(a,0,sum,ordered=true)\n  result[1] = reduce(b,0,sum,ordered=true)\n  store(result,out)\n".into(),
    }], &[]).unwrap();
    let shapes = HashMap::from([("N".into(), 1_000_000_000)]);
    let elements = HashMap::new();
    let options = Options::default();
    let request = || Specialization { program: &program, entry: "pieces", backend: "cpu", shapes: &shapes, elements: &elements, options: &options };
    let mut path=Vec::new();let mut capacities=Vec::new();
    let function=loop {
        match expand(request(),&path).unwrap(){
            Expansion::Choice(domain)=>{
                let index=match domain.kind {
                    DecisionKind::Stream{maximum:1_000_000_000,..}=>{
                        assert_eq!(domain.alternatives.capacity_interval(),Some((1_000_000_000,1)));
                        assert_eq!(domain.alternatives.get(999_999_999),Some(Alternative::StreamCapacity(1)));
                        capacities.push(1);999_999_999
                    }
                    DecisionKind::Stream{maximum:3,..}=>{capacities.push(2);1}
                    _=>0,
                };path.push(index);
            }
            Expansion::Lowered{function,consumed}=>{assert_eq!(consumed,path.len());break function;}
            Expansion::RetainedChoice(_)=>path.push(0),
        }
    };
    assert_eq!(capacities,vec![1,2]);
    assert!(function.vars.len()<1000,"logical domain must remain bounded rather than unrolled");
    assert_eq!(function.body.iter().filter(|s|matches!(s.kind,StmtKind::Range{..})).count(),2);
    let mut space = seismic_lang::lower::alternatives::Space::new(request());
    assert!(space.next().unwrap().result.is_ok());
    assert!(space.next().unwrap().result.is_ok());
    assert!(!space.exhausted());
}

#[test]
fn reference_empty_window_has_no_body_effects() {
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
            "empty_window",
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
    assert!(visited.iter().all(|d| d.alternatives == vec![Alternative::Body(Choice::Block(1)), Alternative::Body(Choice::Portable)].into()));
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
        corrupted[0].domain.alternatives = Vec::new().into();
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

#[test]
fn pure_temporaries_helper_extraction_and_equal_indices_preserve_producer_choices() {
    use seismic_lang::{lower::lower_selected,lowered_ir::{Alternative,DecisionKind},interp::{Interpreter,TensorData,Arg},ir::Function,types::DType,program::Program};
    let helper="fn write[N](x:tile[N] f32,t:tile[N] f32):\n  for i in owned(t):\n    value = x[i] * 2.0\n    t[i+0] = value + 1.0\n";
    let forms=[
        "  for i in owned(t): t[i] = x[i] * 2.0 + 1.0\n",
        "  for i in owned(t):\n    value = x[i] * 2.0\n    t[0+i] = value + 1.0\n",
        "  write(x,t)\n",
    ];
    let mut families=Vec::new();
    for form in forms {
        let text=format!("{helper}fn entry[N](a:tensor[N] f32,out:tensor[N] f32):\n  x = load(a)\n  t = tile[N] f32\n{form}  y = tile[N] f32\n  for i in owned(y): y[i] = t[i] * t[i]\n  store(y,out)\n");
        let p=compile(&[SourceFile{path:"producer.seismic.portable".into(),scope:Scope::Portable,text}],&[]).unwrap();
        let mut families_for_form=Vec::new();
        for retain in [false,true] {
            let mut choices=Vec::new();
            let lowered=lower_selected(&p,"entry","cpu",&HashMap::from([("N".into(),7)]),&HashMap::new(),&Options::default(),&mut |d| {
                if let DecisionKind::Producer{ty,..}=&d.kind {
                    choices.push((ty.clone(),d.alternatives.clone()));
                    return Ok(if retain {Alternative::Materialize}else{Alternative::Recompute});
                }
                Ok(d.alternatives.get(0).unwrap())
            }).unwrap();
            let p=Program{functions:vec![Function{name:lowered.name,is_construct:false,shape_params:vec![],elem_params:vec![],params:lowered.params,index_params:lowered.index_params,vars:lowered.vars,body:lowered.body}],lowerings:vec![],signatures:HashMap::new()};
            let mut vm=Interpreter::new(&p);
            let x=vm.add_tensor(TensorData::dense(DType::F32,vec![7],(0..7).map(|i|i as f64).collect()));
            let out=vm.add_tensor(TensorData::dense(DType::F32,vec![7],vec![0.0;7]));
            vm.run("entry",&[Arg::Tensor(x),Arg::Tensor(out)],&HashMap::new()).unwrap();
            for i in 0..7 {assert_eq!(vm.tensors[out].get(i),((i*2+1)*(i*2+1)) as f64);}
            assert!(!choices.is_empty(),"producer family disappeared after source refactoring");
            families_for_form.push(choices);
        }
        families.push(families_for_form);
    }
    assert!(families.windows(2).all(|f|f[0]==f[1]));
}
