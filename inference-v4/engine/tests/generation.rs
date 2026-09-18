use seismic_engine::models::sequence::Advance;
use seismic_engine::{
    generation::{
        Constraint, FinishReason, Generation, Options, Proposal, Readiness, Sampling, WaitReason,
        WorkKind,
    },
    inputs::{InputLayout, TokenId},
};
use std::{cell::Cell, collections::BTreeSet, rc::Rc};

struct Row {
    complete: Rc<Cell<bool>>,
    committed: Rc<Cell<bool>>,
    dropped: Rc<Cell<bool>>,
    selected: Option<TokenId>,
    commit_failure: bool,
}
impl Advance for Row {
    fn is_complete(&self) -> bool {
        self.complete.get()
    }
    fn selected(&mut self) -> Result<Option<TokenId>, String> {
        assert!(self.complete.get());
        Ok(self.selected)
    }
    fn commit(&mut self) -> Result<(), String> {
        assert!(self.complete.get());
        if self.commit_failure {
            return Err("commit failed".into());
        }
        self.committed.set(true);
        Ok(())
    }
}
impl Drop for Row {
    fn drop(&mut self) {
        self.dropped.set(true);
    }
}
fn row(selected: Option<u32>) -> Row {
    Row {
        complete: Rc::new(Cell::new(true)),
        committed: Rc::new(Cell::new(false)),
        dropped: Rc::new(Cell::new(false)),
        selected: selected.map(TokenId),
        commit_failure: false,
    }
}
fn options() -> Options {
    Options {
        max_tokens: 8,
        output_capacity: 2,
        context_limit: 32,
        vocabulary: 100,
        stop_tokens: BTreeSet::from([TokenId(99)]),
        sampling: Sampling::Greedy,
        seed: 42,
        forced_quantum: 4,
    }
}
fn generation(constraint: Option<Box<dyn Constraint>>) -> Generation {
    Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        options(),
        constraint,
    )
    .unwrap()
}
fn ready(generation: &Generation, allowance: usize) -> Proposal {
    match generation.ready(allowance).unwrap() {
        Readiness::Ready(p) => p,
        other => panic!("not ready: {other:?}"),
    }
}
fn finish(generation: &mut Generation, proposal: Proposal, selected: Option<u32>) {
    generation
        .attach(proposal, Box::new(row(selected)))
        .unwrap();
    assert!(generation.reconcile().unwrap());
}
#[derive(Clone)]
struct Grammar {
    accepted: Vec<TokenId>,
    forced: Vec<TokenId>,
    reject: Option<TokenId>,
}
impl Constraint for Grammar {
    fn fork(&self) -> Box<dyn Constraint> {
        Box::new(self.clone())
    }
    fn mask(&self) -> Result<std::sync::Arc<[u32]>, String> {
        Ok(vec![u32::MAX; 4].into())
    }
    fn position(&self) -> usize {
        self.accepted.len()
    }
    fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String> {
        if tokens.iter().any(|t| Some(*t) == self.reject) {
            return Err("grammar rejected token".into());
        }
        let mut accepted = self.accepted.clone();
        accepted.extend_from_slice(tokens);
        Ok(Box::new(Self {
            accepted,
            forced: self.forced.clone(),
            reject: self.reject,
        }))
    }
    fn forced(&self, limit: usize) -> Result<Vec<TokenId>, String> {
        Ok(self
            .forced
            .iter()
            .skip(self.accepted.len())
            .take(limit)
            .copied()
            .collect())
    }
}

#[test]
fn pure_proposals_completion_output_credit_and_stop_are_independent() {
    let mut g = generation(None);
    let p = ready(&g, 1);
    assert_eq!(p, ready(&g, 1));
    assert_eq!(g.processed(), 0);
    assert!(!p.needs_sample());
    finish(&mut g, p, None);
    let p = ready(&g, 1);
    assert!(p.needs_sample());
    let row = row(Some(10));
    row.complete.set(false);
    let complete = row.complete.clone();
    let committed = row.committed.clone();
    g.attach(p, Box::new(row)).unwrap();
    assert!(!g.reconcile().unwrap());
    assert!(!committed.get());
    assert_eq!(g.ready(1).unwrap(), Readiness::Wait(WaitReason::Completion));
    complete.set(true);
    assert!(g.reconcile().unwrap());
    assert_eq!(g.generated(), &[TokenId(10)]);
    let p = ready(&g, 1);
    assert_eq!(p.kind(), WorkKind::Decode);
    assert_eq!(p.tokens(), &[TokenId(10)]);
    assert_eq!(p.sample_position(), 1);
    assert_eq!(p.seed(), 42);
    finish(&mut g, p, Some(11));
    assert_eq!(g.ready(1).unwrap(), Readiness::Wait(WaitReason::Output));
    assert_eq!(g.take(1).unwrap()[0].index, 0);
    let p = ready(&g, 1);
    finish(&mut g, p, Some(99));
    assert_eq!(g.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(g.usage().prompt_tokens, 2);
    assert_eq!(g.usage().completion_tokens, 3);
    let output = g.take(10).unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].token, TokenId(11));
    assert_eq!(output[0].index, 1);
}

#[test]
fn grammar_failure_does_not_commit_and_commit_failure_does_not_install_grammar() {
    for fail_commit in [false, true] {
        let mut g = generation(Some(Box::new(Grammar {
            accepted: vec![],
            forced: vec![],
            reject: Some(TokenId(12)),
        })));
        let p = ready(&g, 2);
        let mut row = row(Some(if fail_commit { 11 } else { 12 }));
        row.commit_failure = fail_commit;
        let committed = row.committed.clone();
        g.attach(p, Box::new(row)).unwrap();
        assert!(g.reconcile().is_err());
        assert!(!committed.get());
        assert_eq!(g.constraint_position(), Some(0));
        assert_eq!(g.processed(), 0);
        assert!(g.generated().is_empty());
        assert_eq!(g.finish_reason(), Some(FinishReason::Failed));
    }
}

#[test]
fn cancellation_retains_submitted_work_until_completion_and_keeps_accepted_output() {
    let mut g = generation(None);
    let p = ready(&g, 2);
    finish(&mut g, p, Some(10));
    let p = ready(&g, 1);
    let row = row(Some(11));
    row.complete.set(false);
    let complete = row.complete.clone();
    let committed = row.committed.clone();
    let dropped = row.dropped.clone();
    g.attach(p, Box::new(row)).unwrap();
    g.cancel();
    assert!(!g.reconcile().unwrap());
    assert!(!dropped.get());
    assert!(g.evicted().is_err());
    complete.set(true);
    assert!(g.reconcile().unwrap());
    assert!(dropped.get());
    assert!(!committed.get());
    assert_eq!(g.generated(), &[TokenId(10)]);
    assert_eq!(g.take(1).unwrap()[0].token, TokenId(10));
    assert_eq!(g.finish_reason(), Some(FinishReason::Cancelled));
}

#[test]
fn replay_preserves_queue_grammar_and_pending_input_without_sampling() {
    let mut g = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![],
        reject: None,
    })));
    let p = ready(&g, 2);
    finish(&mut g, p, Some(10));
    g.evicted().unwrap();
    assert_eq!(g.ready(1).unwrap(), Readiness::Wait(WaitReason::Residency));
    g.restored().unwrap();
    let p = ready(&g, 1);
    assert_eq!(p.kind(), WorkKind::Replay);
    assert!(!p.needs_sample());
    finish(&mut g, p, None);
    let p = ready(&g, 1);
    finish(&mut g, p, None);
    assert_eq!(g.constraint_position(), Some(1));
    assert_eq!(g.generated(), &[TokenId(10)]);
    assert_eq!(g.output_len(), 1);
    let p = ready(&g, 1);
    assert_eq!(p.kind(), WorkKind::Decode);
    assert_eq!(p.tokens(), &[TokenId(10)]);
    assert_eq!(p.sample_position(), 1);
}

#[test]
fn forced_runs_respect_credit_and_keep_the_final_token_pending() {
    let mut g = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![TokenId(10), TokenId(11), TokenId(12), TokenId(99)],
        reject: None,
    })));
    let p = ready(&g, 20);
    assert_eq!(p.forced(), &[TokenId(10)]);
    assert!(!p.needs_sample());
    finish(&mut g, p, None);
    g.take(2).unwrap();
    let p = ready(&g, 20);
    assert_eq!(p.forced(), &[TokenId(11), TokenId(12)]);
    assert_eq!(p.tokens(), &[TokenId(10), TokenId(11)]);
    finish(&mut g, p, None);
    assert_eq!(g.processed(), 4);
    assert_eq!(g.constraint_position(), Some(3));
    g.take(2).unwrap();
    let p = ready(&g, 20);
    assert!(p.forced().is_empty());
    assert!(p.needs_sample());
    assert_eq!(p.tokens(), &[TokenId(12)]);
}

#[test]
fn proposals_cannot_attach_to_peers_or_survive_accepted_progress() {
    let mut first = generation(None);
    let mut second = generation(None);
    let p = ready(&first, 2);
    assert!(second.attach(p.clone(), Box::new(row(Some(10)))).is_err());
    finish(&mut first, p.clone(), Some(10));
    assert!(first.attach(p, Box::new(row(Some(11)))).is_err());
    let p = ready(&second, 2);
    finish(&mut second, p, Some(12));
    assert_eq!(second.generated(), &[TokenId(12)]);
}
