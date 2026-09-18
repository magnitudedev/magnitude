//! Retained dense input windows are ordinary private storage. Their read savings
//! and tail guards are derived from the same execution used by native emission.
use seismic_lang::{program::{compile, SourceFile}, Scope, lower::lower_selected,
    lowered_ir::{Alternative, DecisionKind}, reduction::structured::{PreparationScope, StepOperand, StepState, Tree}};
use seismic_metal::{execution::{prepare_with_participants, Config, FoldOwnership}, model, terminal::{Primitive, Space}};
use seismic_accounting::workload::{ScalarWorkload, Allocation, BufferBinding, DerivationLimits};

fn selected(scope: Option<PreparationScope>) -> seismic_metal::execution::Execution {
    let p = compile(&[SourceFile { path: "snapshot.seismic.portable".into(), scope: Scope::Portable, text: r#"
fn merge(left:tile[3] f32,right:tile[3] f32,out:tile[3] f32):
  for i in owned(out): out[i]=left[i]+right[i]
fn step(state:tile[3] f32,a:tile[1] f32,w:tile[3] f32,out:tile[3] f32):
  for i in owned(out): out[i]=fma(a[0],w[i],state[i])
fn evaluate(x:tensor[1,129] f32,w:tensor[3,129] f32,out:tensor[3] f32):
  a=load(x)
  b=load(w)
  state=tile[3] f32
  zero=tile[3] f32
  for i in owned(state): state[i]=1.25
  for i in owned(zero): zero[i]=0.0
  reduce((a,b),1,merge,into=(state,),step=step,identity=(zero,),ordered=false)
  store(state,out)
"#.into() }], &[]).unwrap();
    let ir = lower_selected(&p, "evaluate", "metal", &Default::default(), &Default::default(), &Default::default(), &mut |d| Ok(match d.kind {
        DecisionKind::Reduction { .. } => Alternative::ReductionTree(Tree::SeedThenPairwise),
        DecisionKind::ReductionSegments { .. } => Alternative::ReductionSegment(64),
        DecisionKind::ReductionInput { input: 0, .. } if scope.is_some() => Alternative::InputSnapshot(scope.unwrap()),
        DecisionKind::FoldPreparation { .. } => Alternative::PreparationWindow(17),
        DecisionKind::FoldOperand { .. } => Alternative::StepOperand(StepOperand::View),
        DecisionKind::FoldState { .. } => Alternative::StepState(StepState::Retained),
        _ => d.alternatives.get(0).unwrap(),
    })).unwrap();
    prepare_with_participants(&ir, Config { sg_per_tg: 1, ..Default::default() }, None,
        &mut |_| Ok(FoldOwnership::ParticipantsRootSeed),
        &mut |_, site| Ok(if site.can_borrow { seismic_lang::ir::LoadMode::Borrow } else { seismic_lang::ir::LoadMode::Materialize }),
        &mut |s| Ok(s.diagnostic()), &mut |r| Ok(r.diagnostic()), &mut |a| Ok(a.alternatives[0])).unwrap()
}

#[test]
fn dense_fold_snapshots_remove_repeated_activation_reads_and_exclude_tail_reads() {
    for scope in [None, Some(PreparationScope::Segment), Some(PreparationScope::Window)] {
        let execution = selected(scope);
        let sizes = [129 * 4, 3 * 129 * 4, 12];
        let workload = ScalarWorkload { integer_domains: Vec::new(), identity: format!("dense snapshots {scope:?}"), allocations: sizes.iter().enumerate().map(|(id, &bytes)| Allocation { id: id as u64, bytes, alignment: 16, known_bytes: Default::default() }).collect(), buffers: sizes.iter().enumerate().map(|(id, &bytes)| BufferBinding { allocation: id as u64, offset: 0, bytes }).collect(), scalars: vec![] };
        let account = model::invocation_account(&execution, &workload, DerivationLimits { instructions: 1_000_000, operations: 1_000_000 }).unwrap();
        assert!(account.is_complete(), "{:?}", account.unmapped);
        let reads: u64 = account.operations.iter().filter(|op| matches!(op.primitive, Primitive::Read { space: Space::Device, .. })).map(|op| op.lanes * op.instances).sum();
        assert_eq!(reads, 129 * if scope.is_some() { 4 } else { 6 }, "{scope:?}");
        if let Some(scope) = scope {
            let capacities = execution.function().vars.iter().filter(|v| v.name.starts_with("fold_snapshot_")).map(|v| v.ty.shaped().unwrap().shape.last().unwrap().as_constant().unwrap()).collect::<Vec<_>>();
            assert_eq!(capacities, if scope == PreparationScope::Segment { vec![64] } else { vec![17, 13] });
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_dense_fold_snapshots_preserve_tail_segments_and_tree() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let x: Vec<f32> = (0..129).map(|i| match i % 4 { 0 => 1.0e20, 1 => 0.25, 2 => -1.0e20, _ => -0.5 }).collect();
    let w: Vec<f32> = (0..3 * 129).map(|i| (i % 11 - 5) as f32 / 16.0).collect();
    let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let x_buffer = device.buffer_from(&bytes(&x)).unwrap();
    let w_buffer = device.buffer_from(&bytes(&w)).unwrap();
    let output = device.buffer(12).unwrap();
    let expected: Vec<f32> = (0..3).map(|row| {
        let parts: Vec<f32> = (0..129).step_by(64).map(|start| (start..(start + 64).min(129)).fold(0.0f32, |sum, k| x[k].mul_add(w[row * 129 + k], sum))).collect();
        1.25 + ((parts[0] + parts[1]) + parts[2])
    }).collect();
    for scope in [None, Some(PreparationScope::Segment), Some(PreparationScope::Window)] {
        let execution = selected(scope);
        let kernel = device.compile(seismic_metal::msl::emit_execution(&execution).unwrap()).unwrap();
        device.run(&kernel, &[&x_buffer, &w_buffer, &output], &[], 1).unwrap();
        assert_eq!(output.read(12), bytes(&expected), "{scope:?}");
    }
}
