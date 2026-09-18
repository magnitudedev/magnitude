use seismic_lang::{
    interp::{Arg, Interpreter, TensorData},
    ir::{Function, StmtKind},
    lower::{lower_selected, Options},
    lowered_ir::{Alternative, DecisionKind},
    program::{compile, Program, SourceFile},
    reduction::structured::Tree,
    types::DType,
    Scope,
};
use std::collections::HashMap;

fn source(ordered: bool) -> String {
    format!(r#"
fn merge[M](la: tile[M] f32, lb: tile[M] f32, ra: tile[M] f32, rb: tile[M] f32, oa: tile[M] f32, ob: tile[M] f32):
  for i in owned(oa): oa[i] = la[i] + ra[i]
  for i in owned(ob): ob[i] = lb[i] + rb[i] + la[i] * ra[i]
fn coupled[N](a: tensor[N,1] f32, b: tensor[N,1] f32, out: tensor[2] f32):
  ta = load(a)
  tb = load(b)
  sa = tile[1] f32
  sb = tile[1] f32
  for i in owned(sa): sa[i] = 10.0
  for i in owned(sb): sb[i] = 7.0
  reduce((ta,tb), 0, merge, into=(sa,sb), ordered={ordered})
  result = tile[2] f32
  for i in owned(result): result[i] = 0.0
  result[0] = sa[0]
  result[1] = sb[0]
  store(result,out)
"#)
}
fn program(text: String) -> Program {
    compile(&[SourceFile {path:"coupled.seismic.portable".into(),text,scope:Scope::Portable}], &[])
        .unwrap_or_else(|e| panic!("{}",e.iter().map(|e|e.render()).collect::<Vec<_>>().join("\n")))
}
fn run(program: &Program, n: usize) -> [f64;2] {
    let mut interpreter = Interpreter::new(program);
    let a = interpreter.add_tensor(TensorData::dense(DType::F32,vec![n,1], (1..=n).map(|i|i as f64).collect()));
    let b = interpreter.add_tensor(TensorData::dense(DType::F32,vec![n,1], vec![2.0;n]));
    let out = interpreter.add_tensor(TensorData::dense(DType::F32,vec![2],vec![0.0;2]));
    interpreter.run("coupled", &[Arg::Tensor(a),Arg::Tensor(b),Arg::Tensor(out)], &HashMap::from([("N".into(),n as i64)])).unwrap();
    [interpreter.tensors[out].get(0),interpreter.tensors[out].get(1)]
}
#[test]
fn coupled_state_and_single_seed_are_preserved_through_realization() {
    for n in [0,1,2,3,4,7,8,19] {
        let p = program(source(false));
        let reference = run(&p,n);
        let sum = (1..=n).sum::<usize>() as f64;
        let squares = (1..=n).map(|i|i*i).sum::<usize>() as f64;
        assert_eq!(reference,[10.0+sum,7.0+2.0*n as f64+10.0*sum+(sum*sum-squares)/2.0]);
        for tree in [Tree::Ordered,Tree::Pairwise] {
            let lowered = lower_selected(&p,"coupled","cpu",&HashMap::from([("N".into(),n as i64)]),&HashMap::new(),&Options::default(), &mut |d| {
                Ok(if matches!(d.kind,DecisionKind::Reduction{..}) && d.alternatives.contains(&Alternative::ReductionTree(tree)) {
                    Alternative::ReductionTree(tree)
                } else {d.alternatives.get(0).unwrap()})
            }).unwrap();
            assert!(lowered.body.iter().any(|s|matches!(s.kind,StmtKind::Reduction(_))));
            for lowered in [lowered.clone(),seismic_lang::reduction::structured::materialize(&lowered).unwrap()] {
                let lowered_program = Program {functions:vec![Function{name:lowered.name,is_construct:false,shape_params:vec![],elem_params:vec![],params:lowered.params,index_params:lowered.index_params,vars:lowered.vars,body:lowered.body}],lowerings:vec![],signatures:HashMap::new()};
                assert_eq!(run(&lowered_program,n),reference,"extent {n}, tree {tree:?}");
            }
        }
    }
}
#[test]
fn ordered_source_excludes_reassociation() {
    let p = program(source(true));
    let result = lower_selected(&p,"coupled","cpu",&HashMap::from([("N".into(),8)]),&HashMap::new(),&Options::default(), &mut |d| {
        Ok(if matches!(d.kind,DecisionKind::Reduction{..}) {Alternative::ReductionTree(Tree::Pairwise)} else {d.alternatives.get(0).unwrap()})
    });
    assert!(result.unwrap_err().contains("not applicable"));
}
#[test]
fn invalid_merge_and_state_are_rejected_at_source_boundary() {
    for text in [
        source(false).replace("into=(sa,sb)","into=(sa,sa)"),
        source(false).replace("oa[i] = la[i] + ra[i]","oa[i] = oa[i] + ra[i]"),
        source(false).replace("tb = load(b)","tb = tile[N,2] f32\n  for i,j in owned(tb): tb[i,j] = 0.0"),
    ] {
        assert!(compile(&[SourceFile{path:"invalid.seismic.portable".into(),text,scope:Scope::Portable}],&[]).is_err());
    }
}

#[test]
fn all_order_preserving_binary_trees_are_reachable() {
    use seismic_lang::lower::alternatives::{Space,Specialization};
    let p=program(source(false));
    let shapes=HashMap::from([("N".into(),4)]);
    let elements=HashMap::new();
    let options=Options::default();
    let space=Space::new(Specialization{program:&p,entry:"coupled",backend:"cpu",shapes:&shapes,elements:&elements,options:&options});
    let reference=run(&p,4);
    let mut trees=std::collections::BTreeSet::new();
    for attempt in space {
        let lowered=attempt.result.unwrap();
        let Some(reduction)=lowered.body.iter().find_map(|s|if let StmtKind::Reduction(r)=&s.kind {Some(r)}else{None}) else {continue;};
        if reduction.tree!=Some(Tree::Explicit) {continue;}
        trees.insert(reduction.branches.iter().map(|b|(b.start,b.cut,b.end)).collect::<Vec<_>>());
        let p=Program{functions:vec![Function{name:lowered.name,is_construct:false,shape_params:vec![],elem_params:vec![],params:lowered.params,index_params:lowered.index_params,vars:lowered.vars,body:lowered.body}],lowerings:vec![],signatures:HashMap::new()};
        assert_eq!(run(&p,4),reference);
    }
    // Five leaves: the non-neutral seed and four input states. C_4 = 14.
    assert_eq!(trees.len(),14);
}

fn fold_source(ordered:bool)->String {format!(r#"
fn add[M](left:tile[M] f32,right:tile[M] f32,out:tile[M] f32):
  for i in owned(out): out[i] = left[i] + right[i]
fn accumulate[M](state:tile[M] f32,a:tile[M] f32,b:tile[M] f32,out:tile[M] f32):
  for i in owned(out): out[i] = fma(a[i],b[i],state[i])
fn coupled[N](a:tensor[N,1] f32,b:tensor[N,1] f32,out:tensor[2] f32):
  ta = load(a)
  tb = load(b)
  state = tile[1] f32
  zero = tile[1] f32
  for i in owned(state): state[i] = 10.0
  for i in owned(zero): zero[i] = 0.0
  reduce((ta,tb),0,add,into=(state,),step=accumulate,identity=(zero,),ordered={ordered})
  result = tile[2] f32
  for i in owned(result): result[i] = state[0]
  store(result,out)
"#)}

#[test]
fn mergeable_fold_preserves_step_arithmetic_and_covers_all_segment_extents() {
    for n in [0,1,2,3,7,19] {
        let p=program(fold_source(false));
        let reference=run(&p,n);
        assert_eq!(reference,[10.0+(n*(n+1)) as f64;2]);
        for tree in [Tree::Ordered,Tree::Pairwise,Tree::Explicit] {
            for segment in 1..=n.max(1) {
                let lowered=lower_selected(&p,"coupled","cpu",&HashMap::from([("N".into(),n as i64)]),&HashMap::new(),&Options::default(),&mut |d| {
                    let choice=match d.kind {
                        DecisionKind::Reduction{..} if d.alternatives.contains(&Alternative::ReductionTree(tree))=>Alternative::ReductionTree(tree),
                        DecisionKind::ReductionSegments{..}=>Alternative::ReductionSegment(segment as i64),
                        _=>d.alternatives.get(0).unwrap(),
                    };Ok(choice)
                }).unwrap();
                let reduction=lowered.body.iter().find_map(|s|if let StmtKind::Reduction(r)=&s.kind {Some(r)}else{None}).unwrap();
                assert!(reduction.step.as_ref().unwrap().implementation.is_some());
                let lowered=seismic_lang::reduction::structured::materialize(&lowered).unwrap();
                let p=Program{functions:vec![Function{name:lowered.name,is_construct:false,shape_params:vec![],elem_params:vec![],params:lowered.params,index_params:lowered.index_params,vars:lowered.vars,body:lowered.body}],lowerings:vec![],signatures:HashMap::new()};
                assert_eq!(run(&p,n),reference,"n={n} segment={segment} tree={tree:?}");
            }
        }
    }
    let p=program(fold_source(true));
    let lowered=lower_selected(&p,"coupled","cpu",&HashMap::from([("N".into(),19)]),&HashMap::new(),&Options::default(),&mut |d| {
        assert!(!matches!(d.kind,DecisionKind::ReductionSegments{..}));
        if matches!(d.kind,DecisionKind::Reduction{..}) {assert_eq!(d.alternatives.len(),1);}
        Ok(d.alternatives.get(0).unwrap())
    }).unwrap();
    assert!(lowered.body.iter().any(|s|matches!(s.kind,StmtKind::Reduction(_))));
}
