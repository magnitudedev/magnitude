use seismic_lang::{
    interp::{Arg, Interpreter, TensorData},
    normalize::bind_values,
    program::{compile, Program, SourceFile},
    types::DType,
    Scope,
};

fn program(text: &str) -> Program {
    compile(
        &[SourceFile {
            path: "normalize.seismic.portable".into(),
            text: text.into(),
            scope: Scope::Portable,
        }],
        &[],
    )
    .unwrap()
}

const SCOPED: &str = "fn evaluate(x: tensor[4] f32, out: tensor[2] f32, enabled: bool):\n  a = load(x)\n  acc = 10.0\n  if enabled:\n    acc += reduce(a,0,sum)\n    for i in owned(a): a[i] = a[i] + 1.0\n    acc += reduce(a,0,sum) * 2.0\n  y = tile[2] f32\n  for i in owned(y):\n    if i == 0: y[i] = acc\n    else: y[i] = reduce(a,0,sum)\n  store(y,out)\n";

fn run(p: &Program, enabled: bool) -> Vec<f64> {
    let mut interpreter = Interpreter::new(p);
    let x = interpreter.add_tensor(TensorData::dense(DType::F32, vec![4], vec![1., 2., 3., 4.]));
    let out = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![-999.; 2]));
    interpreter
        .run(
            "evaluate",
            &[
                Arg::Tensor(x),
                Arg::Tensor(out),
                Arg::Scalar(if enabled { 1. } else { 0. }),
            ],
            &Default::default(),
        )
        .unwrap();
    (0..2).map(|i| interpreter.tensors[out].get(i)).collect()
}

#[test]
fn reduction_bindings_preserve_control_scope_and_reads_across_mutation() {
    let mut p = program(SCOPED);
    assert_eq!(run(&p, false), [10., 10.]);
    assert_eq!(run(&p, true), [48., 14.]);
    let f = &mut p.functions[0];
    let original_vars = f.vars.len();
    let original_statements = f.body.len();
    let positions = bind_values(&mut f.body, &mut f.vars);
    assert_eq!(positions.len(), original_statements);
    assert!(f.vars.len() > original_vars);
    let normalized = f.clone();
    let second_positions = bind_values(&mut f.body, &mut f.vars);
    assert_eq!(*f, normalized);
    assert_eq!(second_positions, (0..f.body.len()).collect::<Vec<_>>());
    assert_eq!(run(&p, false), [10., 10.]);
    assert_eq!(run(&p, true), [48., 14.]);
}

#[test]
fn nested_tile_reductions_bind_dependencies_before_consumers() {
    let mut p = program("fn evaluate(x: tensor[2,3] f32, out: tensor[1] f32):\n  a = load(x)\n  y = tile[1] f32\n  for i in owned(y): y[i] = reduce(reduce(a,1,sum),0,sum)\n  store(y,out)\n");
    for normalize in [false, true] {
        if normalize {
            let f = &mut p.functions[0];
            bind_values(&mut f.body, &mut f.vars);
        }
        let mut interpreter = Interpreter::new(&p);
        let x = interpreter.add_tensor(TensorData::dense(
            DType::F32,
            vec![2, 3],
            vec![1., 2., 3., 4., 5., 6.],
        ));
        let out = interpreter.add_tensor(TensorData::dense(DType::F32, vec![1], vec![0.]));
        interpreter
            .run(
                "evaluate",
                &[Arg::Tensor(x), Arg::Tensor(out)],
                &Default::default(),
            )
            .unwrap();
        assert_eq!(interpreter.tensors[out].get(0), 21.);
    }
}

#[test]
fn selected_loads_preserve_snapshots_and_can_be_reselected() {
    use seismic_lang::{
        ir::{ExprKind, LoadMode, StmtKind},
        normalize::select_loads,
    };
    let cases = [
        (
            "a = load(x)\n  z = reduce(a,0,sum)\n  store(a,out)",
            LoadMode::Materialize,
        ),
        ("a = load(x)\n  z = reduce(a,0,sum)", LoadMode::Borrow),
        (
            "a = load(x)\n  a = load(x)\n  z = reduce(a,0,sum)",
            LoadMode::Materialize,
        ),
        (
            "a = load(x)\n  for i in owned(a): a[i] = 0.0\n  z = reduce(a,0,sum)",
            LoadMode::Materialize,
        ),
    ];
    for (body, expected) in cases {
        let mut p = program(&format!(
            "fn evaluate(x: tensor[4] f32, out: tensor[4] f32):\n  {body}\n"
        ));
        let f = &mut p.functions[0];
        bind_values(&mut f.body, &mut f.vars);
        for (borrow, expected) in [(true, expected), (false, LoadMode::Materialize)] {
            select_loads(&mut f.body, borrow);
            let mut count = 0;
            for statement in &f.body {
                if let StmtKind::Assign { value, .. } = &statement.kind {
                    if let ExprKind::Load { mode, .. } = value.kind {
                        assert_eq!(mode, expected, "{body}");
                        count += 1;
                    }
                }
            }
            assert!(count > 0);
        }
    }
}

#[test]
fn load_space_covers_mixed_modes_and_forces_mutated_snapshots() {
    use seismic_lang::{
        ir::LoadMode::{Borrow, Materialize},
        normalize::loads::{self, Expansion},
    };
    let p = program("fn evaluate(x: tensor[4] f32, y: tensor[4] f32, out: tensor[1] f32):\n  a = load(x)\n  sa = reduce(a,0,sum)\n  b = load(y)\n  sb = reduce(b,0,sum)\n  c = load(x)\n  for i in owned(c): c[i] = 0.0\n  sc = reduce(c,0,sum)\n  z = tile[1] f32\n  for i in owned(z): z[i] = sa + sb + sc\n  store(z,out)\n");
    let lowered = seismic_lang::lower::lower(&p, "evaluate", "cpu", &Default::default()).unwrap();
    assert!(matches!(
        loads::expand(&lowered, &[]).unwrap(),
        Expansion::Choice(loads::Choice { site: 0, .. })
    ));
    let mut observed = Vec::new();
    for a in 0..2 {
        for b in 0..2 {
            let Expansion::Selected {
                mut function,
                consumed,
            } = loads::expand(&lowered, &[a, b]).unwrap()
            else {
                panic!()
            };
            assert_eq!(consumed, 2);
            let modes = loads::selected(&function.body)
                .unwrap()
                .into_iter()
                .map(|d| d.mode)
                .collect::<Vec<_>>();
            assert_eq!(modes[2], Materialize);
            observed.push(modes.clone());
            let before = function.body.clone();
            let mut invalid = modes;
            invalid[2] = Borrow;
            assert!(loads::resolve(&mut function.body, &invalid).is_err());
            assert_eq!(function.body, before);
        }
    }
    assert_eq!(
        observed,
        vec![
            vec![Materialize, Materialize, Materialize],
            vec![Materialize, Borrow, Materialize],
            vec![Borrow, Materialize, Materialize],
            vec![Borrow, Borrow, Materialize]
        ]
    );
}

#[test]
fn selected_loads_keep_reference_interpreter_semantics() {
    let mut p = program(SCOPED);
    let f = &mut p.functions[0];
    bind_values(&mut f.body, &mut f.vars);
    seismic_lang::normalize::select_loads(&mut f.body, true);
    assert_eq!(run(&p, false), [10., 10.]);
    assert_eq!(run(&p, true), [48., 14.]);
}

#[test]
fn stream_modes_are_selected_before_emission_and_respect_mutation() {
    use seismic_lang::{
        ir::{LoadMode, StmtKind},
        normalize::select_loads,
    };
    for (use_tile, expected) in [
        ("z = reduce(t,0,sum)", LoadMode::Borrow),
        ("for i in owned(t): t[i] = 0.0", LoadMode::Materialize),
        ("store(load(out),out)", LoadMode::Materialize),
    ] {
        let mut p = program(&format!("fn evaluate(x: tensor[4] f32, out: tensor[4] f32):\n  t = load(x)\n  {use_tile}\n"));
        let f = &mut p.functions[0];
        // Exercise the execution node directly; it is never authored in source.
        let load = f.body.remove(0);
        let StmtKind::Assign {target, value, ..} = load.kind else {panic!()};
        let seismic_lang::ir::ExprKind::Var(binding) = target.kind else {panic!()};
        let seismic_lang::ir::ExprKind::Builtin {args,..} = value.kind else {panic!()};
        let body = std::mem::take(&mut f.body);
        f.body.push(seismic_lang::ir::Stmt {id:None, span:load.span, kind:StmtKind::LoadLoop {
            domain:seismic_lang::ir::IterationDomain{view:args[0].clone(),axis:0}, offset:None, modes:None, vars:vec![binding], views:args, axes:vec![0],
            piece:seismic_lang::sym::Atom::Param("$fixture_piece".into()), capacity:Some(4), body,
        }});
        let StmtKind::LoadLoop { modes, .. } = &f.body[0].kind else {
            panic!("expected stream")
        };
        assert_eq!(*modes, None);
        bind_values(&mut f.body, &mut f.vars);
        for (borrow, expected) in [(true, expected), (false, LoadMode::Materialize)] {
            select_loads(&mut f.body, borrow);
            let StmtKind::LoadLoop { modes, .. } = &f.body[0].kind else {
                panic!("expected stream")
            };
            assert_eq!(modes.as_deref(), Some([expected].as_slice()));
        }
    }
}

#[test]
fn serial_work_domain_preserves_the_body_and_is_idempotent() {
    use seismic_lang::{ir::StmtKind, normalize::work_domain};
    let p = program(SCOPED);
    let original = p.functions[0].body.clone();
    let mut body = original.clone();
    work_domain(&mut body);
    let [statement] = body.as_slice() else {
        panic!("expected one work domain")
    };
    let StmtKind::Parallel {
        vars,
        extents,
        body: inner,
    } = &statement.kind
    else {
        panic!("expected parallel domain")
    };
    assert!(vars.is_empty());
    assert!(extents.is_empty());
    assert_eq!(*inner, original);
    let once = body.clone();
    work_domain(&mut body);
    assert_eq!(body, once);
}

#[test]
fn lifting_reductions_preserves_mutation_and_control_dependencies() {
    let mut p = program(SCOPED);
    let f = &mut p.functions[0];
    bind_values(&mut f.body, &mut f.vars);
    seismic_lang::normalize::lift_owned_reductions(&mut f.body);
    assert_eq!(run(&p, false), [10., 10.]);
    assert_eq!(run(&p, true), [48., 14.]);
}
