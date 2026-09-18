use seismic_engine::{
    generation::{Generation, Options, Readiness, Sampling, WorkKind},
    inputs::{InputLayout, TokenId},
    models::sequence::OwnedSequence,
    state::{ComponentSpec, StateStore},
};
use seismic_lang::types::DType;
use seismic_runtime::Device;
use std::{collections::BTreeSet, rc::Rc};
fn setup() -> (Generation, OwnedSequence, Rc<StateStore>) {
    let store = StateStore::new(
        Rc::new(Device::cpu()),
        16,
        32,
        vec![4],
        vec![ComponentSpec {
            shape: vec![1],
            dtype: DType::F32,
        }],
    )
    .unwrap();
    let sequence = OwnedSequence::new(store.create().unwrap());
    let g = Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        Options {
            max_tokens: 8,
            output_capacity: 4,
            context_limit: 16,
            vocabulary: 16,
            stop_tokens: BTreeSet::from([TokenId(15)]),
            sampling: Sampling::Greedy,
            seed: 1,
            forced_quantum: 0,
        },
        None,
    )
    .unwrap();
    (g, sequence, store)
}
fn advance(g: &mut Generation, sequence: &OwnedSequence, allowance: usize) -> WorkKind {
    let Readiness::Ready(p) = g.ready(allowance).unwrap() else {
        panic!("ready")
    };
    let row = sequence
        .prepare_completed(p.position(), p.tokens().len(), |s| {
            let mut next = s.begin(p.tokens().len())?;
            next.execute(|_| Ok(()))?;
            next.commit()?;
            Ok(p.needs_sample().then_some(TokenId(7)))
        })
        .unwrap();
    let kind = p.kind();
    g.attach(p, row).unwrap();
    g.reconcile().unwrap();
    kind
}
#[test]
fn checkpoint_rejects_unresolved_and_mismatched_state_and_forks_have_new_identity() {
    let (mut g, sequence, store) = setup();
    let Readiness::Ready(p) = g.ready(2).unwrap() else {
        panic!("ready")
    };
    let row = sequence
        .prepare_completed(0, 2, |s| {
            let mut next = s.begin(2)?;
            next.execute(|_| Ok(()))?;
            next.commit()?;
            Ok(Some(TokenId(7)))
        })
        .unwrap();
    assert!(g.checkpoint(&sequence).is_err());
    g.attach(p.clone(), row).unwrap();
    assert!(g.checkpoint(&sequence).is_err());
    g.reconcile().unwrap();
    assert!(g
        .checkpoint(&OwnedSequence::new(store.create().unwrap()))
        .is_err());
    let checkpoint = g.checkpoint(&sequence).unwrap();
    let (mut fork, fork_state) = checkpoint.fork().unwrap();
    let Readiness::Ready(original) = g.ready(1).unwrap() else {
        panic!("ready")
    };
    let row = fork_state
        .prepare_completed(2, 1, |s| {
            let mut next = s.begin(1)?;
            next.execute(|_| Ok(()))?;
            next.commit()?;
            Ok(Some(TokenId(7)))
        })
        .unwrap();
    assert!(fork.attach(original, row).is_err());
    assert!(!fork_state.pending());
    assert_eq!(fork_state.position(), 2);
    advance(&mut fork, &fork_state, 1);
    assert_eq!(g.generated().len(), 1);
    assert_eq!(fork.generated().len(), 2);
    g.evicted().unwrap();
    assert!(g.checkpoint(&sequence).is_err());
}
#[test]
fn checkpoint_during_replay_preserves_recovery_boundary_without_duplicate_output() {
    let (mut g, sequence, store) = setup();
    advance(&mut g, &sequence, 2);
    g.take(1).unwrap();
    g.evicted().unwrap();
    drop(sequence);
    let sequence = OwnedSequence::new(store.create().unwrap());
    g.restored().unwrap();
    assert_eq!(advance(&mut g, &sequence, 1), WorkKind::Replay);
    let checkpoint = g.checkpoint(&sequence).unwrap();
    let (mut fork, numerical) = checkpoint.fork().unwrap();
    assert_eq!(advance(&mut fork, &numerical, 1), WorkKind::Replay);
    assert_eq!(fork.output_len(), 0);
    assert_eq!(fork.generated(), &[TokenId(7)]);
    assert_eq!(advance(&mut fork, &numerical, 1), WorkKind::Decode);
    assert_eq!(fork.take(1).unwrap()[0].index, 1);
    assert_eq!(g.processed(), 1);
    assert_eq!(sequence.position(), 1);
}
