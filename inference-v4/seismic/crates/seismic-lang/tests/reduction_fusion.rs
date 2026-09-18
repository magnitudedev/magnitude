use seismic_lang::{
    Scope,
    interp::{Arg, Interpreter, TensorData},
    ir::{Function, Stmt, StmtKind},
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind, LoweredIr},
    program::{Program, SourceFile, compile},
    reduction::structured::Tree,
    types::DType,
};

fn evaluate(ir: &LoweredIr) -> Vec<f32> {
    let program = Program {
        functions: vec![Function {
            name: ir.name.clone(),
            is_construct: false,
            shape_params: vec![],
            elem_params: vec![],
            params: ir.params.clone(),
            index_params: ir.index_params.clone(),
            vars: ir.vars.clone(),
            body: ir.body.clone(),
        }],
        lowerings: vec![],
        signatures: Default::default(),
    };
    let mut interpreter = Interpreter::new(&program);
    let input = (0..67)
        .map(|i| match i % 4 {
            0 => 1.0e20_f32,
            1 => 0.25,
            2 => -1.0e20,
            _ => -0.5,
        })
        .map(f64::from)
        .collect();
    let x = interpreter.add_tensor(TensorData::dense(DType::F32, vec![67, 1], input));
    let y = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![0.0; 2]));
    interpreter
        .run(
            "evaluate",
            &[Arg::Tensor(x), Arg::Tensor(y)],
            &Default::default(),
        )
        .unwrap();
    (0..2)
        .map(|i| interpreter.tensors[y].get(i) as f32)
        .collect()
}

fn folds(body: &[Stmt], fields: &mut Vec<(usize, usize)>) {
    for s in body {
        match &s.kind {
            StmtKind::Reduction(r) => fields.push((r.state.len(), r.inputs.len())),
            StmtKind::Range { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::Parallel { body, .. }
            | StmtKind::LoadLoop { body, .. }
            | StmtKind::Lanes { body, .. } => folds(body, fields),
            StmtKind::If { then, els, .. } => {
                folds(then, fields);
                folds(els, fields);
            }
            _ => {}
        }
    }
}

#[test]
fn product_folds_preserve_each_seed_tree_and_destructive_helper_snapshot() {
    for destructive in [false, true] {
        let mutate = if destructive {
            "  input[0] = input[0] * 0.5\n"
        } else {
            ""
        };
        let source = format!(
            r#"
fn add(left:tile[1] f32,right:tile[1] f32,out:tile[1] f32):
  for i in owned(out): out[i] = left[0] + right[0]
fn first(state:tile[1] f32,input:tile[1] f32,out:tile[1] f32):
{mutate}  for i in owned(out): out[i] = fma(input[0],2.0,state[0])
fn second(state:tile[1] f32,input:tile[1] f32,out:tile[1] f32):
  for i in owned(out): out[i] = fma(input[0],-3.0,state[0])
fn evaluate(x:tensor[67,1] f32,y:tensor[2] f32):
  input = load(x)
  left = tile[1] f32
  for i in owned(left): left[i] = 10.0
  zero = tile[1] f32
  for i in owned(zero): zero[i] = 0.0
  reduce((input,),0,add,into=(left,),step=first,identity=(zero,),ordered=false)
  right = tile[1] f32
  for i in owned(right): right[i] = -7.0
  reduce((input,),0,add,into=(right,),step=second,identity=(zero,),ordered=false)
  output = tile[2] f32
  for i in owned(output):
    if i == 0: output[i] = left[0]
    else: output[i] = right[0]
  store(output,y)
"#
        );
        let program = compile(
            &[SourceFile {
                path: "product.seismic.portable".into(),
                scope: Scope::Portable,
                text: source,
            }],
            &[],
        )
        .unwrap();
        for tree in [Tree::Ordered, Tree::Pairwise, Tree::Explicit] {
            let mut expected = None;
            for fuse in [false, true] {
                let mut decisions = 0;
                let mut viewed = 0;
                let ir = lower_selected(
                    &program,
                    "evaluate",
                    "cpu",
                    &Default::default(),
                    &Default::default(),
                    &Options::default(),
                    &mut |d| {
                        Ok(match d.kind {
                            DecisionKind::Reduction { .. } => Alternative::ReductionTree(tree),
                            DecisionKind::ReductionSegments { .. } => {
                                Alternative::ReductionSegment(3)
                            }
                            DecisionKind::FoldOperand { .. } => {
                                viewed += 1;
                                Alternative::StepOperand(seismic_lang::reduction::structured::StepOperand::View)
                            }
                            DecisionKind::ReductionFusion { .. } => {
                                decisions += 1;
                                if fuse {
                                    Alternative::Fuse
                                } else {
                                    Alternative::Separate
                                }
                            }
                            _ => d.alternatives.get(0).unwrap(),
                        })
                    },
                )
                .unwrap();
                assert_eq!(decisions, 1);
                assert_eq!(viewed, if destructive || fuse { 1 } else { 2 });
                let mut fields = vec![];
                folds(&ir.body, &mut fields);
                assert_eq!(
                    fields,
                    if fuse {
                        vec![(2, if destructive { 2 } else { 1 })]
                    } else {
                        vec![(1, 1), (1, 1)]
                    }
                );
                let actual = evaluate(&ir);
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected, "{tree:?}, destructive={destructive}");
                } else {
                    expected = Some(actual);
                }
            }
        }
    }
}
