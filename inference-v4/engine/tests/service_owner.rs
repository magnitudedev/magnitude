use seismic_engine::{
    generation::{FinishReason, Generation, Options, Sampling},
    inputs::{InputLayout, TokenId},
    models::sequence::Advance,
    service::{
        owner::{Completion, Executor, Owner, PrepareError, Status, Step, Submitted, Work},
        policy::{Limits, RequestId},
    },
};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    rc::Rc,
};
#[derive(Default)]
struct Control {
    input_pending: RefCell<BTreeSet<RequestId>>,
    input_calls: Cell<usize>,
    input_committed: RefCell<Vec<RequestId>>,
    input_dropped: RefCell<Vec<RequestId>>,
    input_bad_token: RefCell<BTreeSet<RequestId>>,
    complete: Cell<bool>,
    calls: Cell<usize>,
    fatal: Cell<bool>,
    wrong_rows: Cell<bool>,
    selected: RefCell<Vec<Option<u32>>>,
    committed: RefCell<Vec<RequestId>>,
    closed: RefCell<Vec<RequestId>>,
    capacity: Cell<bool>,
    token_limit: Cell<Option<usize>>,
    released: Cell<u64>,
    events: RefCell<Vec<String>>,
    shape_quantum: Cell<usize>,
    reclaim_group: RefCell<Vec<RequestId>>,
    evictions: RefCell<Vec<Vec<RequestId>>>,
    restorations: RefCell<Vec<RequestId>>,
    refuse_eviction: Cell<bool>,
}
struct Batch(Rc<Control>);
impl Completion for Batch {
    fn notify(&mut self, wake: seismic_engine::service::worker::CompletionWake) {
        assert!(
            self.is_complete(),
            "direct owner fixture has no asynchronous notifier"
        );
        wake.complete();
    }
    fn is_complete(&self) -> bool {
        self.0.complete.get()
    }
    fn result(&mut self) -> Result<(), String> {
        assert!(self.is_complete());
        if self.0.fatal.get() {
            Err("device failed".into())
        } else {
            Ok(())
        }
    }
}
struct Row {
    control: Rc<Control>,
    id: RequestId,
    selected: Option<TokenId>,
}
impl Advance for Row {
    fn is_complete(&self) -> bool {
        self.control.complete.get()
    }
    fn selected(&mut self) -> Result<Option<TokenId>, String> {
        assert!(self.is_complete());
        Ok(self.selected)
    }
    fn commit(&mut self) -> Result<(), String> {
        assert!(self.is_complete());
        self.control.committed.borrow_mut().push(self.id);
        Ok(())
    }
}
struct InputRow {
    control: Rc<Control>,
    id: RequestId,
}
impl Advance for InputRow {
    fn is_complete(&self) -> bool {
        self.control.complete.get()
    }
    fn selected(&mut self) -> Result<Option<TokenId>, String> {
        assert!(self.is_complete());
        Ok(self
            .control
            .input_bad_token
            .borrow()
            .contains(&self.id)
            .then_some(TokenId(7)))
    }
    fn commit(&mut self) -> Result<(), String> {
        assert!(self.is_complete());
        self.control.input_committed.borrow_mut().push(self.id);
        self.control.input_pending.borrow_mut().remove(&self.id);
        Ok(())
    }
}
impl Drop for InputRow {
    fn drop(&mut self) {
        self.control.input_dropped.borrow_mut().push(self.id);
    }
}
struct Model(Rc<Control>);
impl Executor for Model {
    type Checkpoint = ();
    fn checkpoint(
        &self,
        _: RequestId,
    ) -> Result<seismic_engine::service::owner::NumericalCheckpoint<()>, String> {
        Err("fixture does not implement numerical checkpoints".into())
    }
    fn open_checkpoint(&mut self, _: RequestId, _: &()) -> Result<(), String> {
        Err("fixture does not implement numerical checkpoints".into())
    }

    fn input_pending(&self, request: RequestId) -> Result<bool, String> {
        Ok(self.0.input_pending.borrow().contains(&request))
    }
    fn input_preparation_identity(&self, requests: &[RequestId]) -> Result<String, String> {
        Ok(format!("input:{requests:?}"))
    }
    fn prepare_inputs(&mut self, requests: &[RequestId]) -> Result<Submitted, PrepareError> {
        self.0.input_calls.set(self.0.input_calls.get() + 1);
        if self.0.capacity.get() {
            return Err(PrepareError::Capacity {
                required: 100,
                available: 50,
            });
        }
        Ok(Submitted {
            completion: Box::new(Batch(self.0.clone())),
            rows: requests
                .iter()
                .map(|&id| {
                    Box::new(InputRow {
                        control: self.0.clone(),
                        id,
                    }) as Box<dyn Advance>
                })
                .collect(),
        })
    }
    fn reclaimable(&self, requests: &[RequestId]) -> Result<u64, String> {
        let group = self.0.reclaim_group.borrow();
        Ok(
            if !group.is_empty() && group.iter().all(|id| requests.contains(id)) {
                100
            } else {
                0
            },
        )
    }
    fn evict(&mut self, requests: &[RequestId]) -> Result<u64, String> {
        if self.0.refuse_eviction.get() {
            return Ok(0);
        }
        let released = self.reclaimable(requests)?;
        if released > 0 {
            self.0.evictions.borrow_mut().push(requests.to_vec());
            self.0.reclaim_group.borrow_mut().clear();
            self.0.capacity.set(false);
        }
        Ok(released)
    }
    fn preparation_identity(&self, work: &[Work]) -> Result<String, String> {
        let quantum = self.0.shape_quantum.get().max(1);
        Ok(format!(
            "{:?}",
            work.iter()
                .map(|w| w.proposal.tokens().len().div_ceil(quantum))
                .collect::<Vec<_>>()
        ))
    }
    fn reclaim_idle(&mut self) -> Result<u64, String> {
        self.0.events.borrow_mut().push("reclaim".into());
        let released = self.0.released.replace(0);
        if released > 0 {
            self.0.capacity.set(false);
        }
        Ok(released)
    }
    fn open(&mut self, _: RequestId) -> Result<(), String> {
        Ok(())
    }
    fn close(&mut self, id: RequestId) -> Result<(), String> {
        self.0.closed.borrow_mut().push(id);
        Ok(())
    }
    fn prepare(&mut self, work: &[Work]) -> Result<Submitted, PrepareError> {
        self.0.calls.set(self.0.calls.get() + 1);
        let shape = work
            .iter()
            .map(|w| w.proposal.tokens().len())
            .collect::<Vec<_>>();
        self.0
            .events
            .borrow_mut()
            .push(format!("prepare {shape:?}"));
        if self.0.capacity.get()
            || self
                .0
                .token_limit
                .get()
                .is_some_and(|limit| shape.iter().sum::<usize>() > limit)
        {
            return Err(PrepareError::Capacity {
                required: 100,
                available: 50,
            });
        }
        let rows = if self.0.wrong_rows.get() {
            vec![]
        } else {
            work.iter()
                .enumerate()
                .map(|(i, w)| {
                    Box::new(Row {
                        control: self.0.clone(),
                        id: w.request,
                        selected: if w.proposal.needs_sample() {
                            self.0
                                .selected
                                .borrow()
                                .get(i)
                                .copied()
                                .unwrap_or(Some(7))
                                .map(TokenId)
                        } else {
                            None
                        },
                    }) as Box<dyn Advance>
                })
                .collect()
        };
        self.0
            .restorations
            .borrow_mut()
            .extend(work.iter().filter(|w| w.restore).map(|w| w.request));
        Ok(Submitted {
            completion: Box::new(Batch(self.0.clone())),
            rows,
        })
    }
}

#[test]
fn shared_victim_sets_release_capacity_and_replay_preserves_accepted_output() {
    let control = Rc::new(Control::default());
    control.complete.set(true);
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 0).unwrap();
    let b = owner.admit(generation(), 0).unwrap();
    owner.step(1).unwrap();
    owner.step(2).unwrap();
    *control.reclaim_group.borrow_mut() = vec![a, b];
    control.capacity.set(true);
    let c = owner.admit(generation(), 3).unwrap();
    assert_eq!(owner.step(4).unwrap(), Step::Submitted);
    assert_eq!(*control.evictions.borrow(), vec![vec![a, b]]);
    // Eviction retains queued outputs. No replay can take their output credit.
    assert_eq!(owner.status(a).unwrap(), Status::OutputBlocked);
    assert_eq!(owner.take(a, 1).unwrap()[0].token, TokenId(7));
    assert_eq!(owner.take(b, 1).unwrap()[0].token, TokenId(7));
    owner.step(5).unwrap();
    owner.cancel(c, true).unwrap();
    owner.retire(c).unwrap();
    assert_eq!(owner.status(a).unwrap(), Status::Preempted);
    assert_eq!(owner.step(6).unwrap(), Step::Submitted);
    owner.step(7).unwrap();
    assert_eq!(*control.restorations.borrow(), vec![a, b]);
    assert!(owner.take(a, 1).unwrap().is_empty());
    assert!(owner.take(b, 1).unwrap().is_empty());
    assert_eq!(owner.status(a).unwrap(), Status::Runnable);
    // Decode resumes from the retained pending token after unsampled replay.
    owner.step(8).unwrap();
    owner.step(9).unwrap();
    assert_eq!(owner.take(a, 1).unwrap()[0].index, 1);
}

#[test]
fn recovery_stays_protected_until_progress_beyond_the_evicted_boundary() {
    let control = Rc::new(Control::default());
    control.complete.set(true);
    let mut owner = Owner::new(
        Model(control.clone()),
        Limits {
            max_requests: 8,
            max_batch: 2,
            prefill_tokens: 1,
            decode_tokens: 1,
            decode_share: 0.5,
            locality_seconds: 0.0,
        },
    )
    .unwrap();
    let a = owner.admit(generation(), 0).unwrap();
    let d = owner.admit(generation(), 0).unwrap();
    for time in 1..=4 {
        owner.step(time).unwrap();
    }
    assert_eq!(owner.status(a).unwrap(), Status::OutputBlocked);
    assert_eq!(owner.status(d).unwrap(), Status::OutputBlocked);
    *control.reclaim_group.borrow_mut() = vec![a];
    control.capacity.set(true);
    let c = owner.admit(generation(), 5).unwrap();
    assert_eq!(owner.step(6).unwrap(), Step::Submitted);
    owner.step(7).unwrap();
    owner.cancel(c, true).unwrap();
    owner.retire(c).unwrap();
    owner.take(a, 1).unwrap();
    assert_eq!(owner.step(8).unwrap(), Step::Submitted);
    owner.step(9).unwrap();
    assert_eq!(*control.restorations.borrow(), vec![a]);

    // A has replayed only one of its two accepted inputs. D's decode must
    // wait even though A is the only positively priced resident victim.
    owner.take(d, 1).unwrap();
    *control.reclaim_group.borrow_mut() = vec![a];
    control.capacity.set(true);
    assert_eq!(owner.step(10).unwrap(), Step::Waiting);
    assert_eq!(*control.evictions.borrow(), vec![vec![a]]);
    control.capacity.set(false);
    assert_eq!(owner.step(11).unwrap(), Step::Submitted);
    owner.step(12).unwrap();
    // Reaching the old boundary is still only replay, not new progress.
    control.capacity.set(true);
    assert_eq!(owner.step(13).unwrap(), Step::Waiting);
    assert_eq!(*control.evictions.borrow(), vec![vec![a]]);
    control.capacity.set(false);
    assert_eq!(owner.step(14).unwrap(), Step::Submitted);
    owner.step(15).unwrap();
    assert_eq!(owner.status(a).unwrap(), Status::OutputBlocked);
    // A has now accepted a fresh decode token, so it can be evicted again.
    control.capacity.set(true);
    assert_eq!(owner.step(16).unwrap(), Step::Submitted);
    assert_eq!(*control.evictions.borrow(), vec![vec![a], vec![a]]);
    owner.step(17).unwrap();
    assert_eq!(owner.take(a, 1).unwrap()[0].index, 1);
}

#[test]
fn failed_restoration_keeps_replay_pending_and_cancellation_can_retire_it() {
    let control = Rc::new(Control::default());
    control.complete.set(true);
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 0).unwrap();
    owner.step(1).unwrap();
    owner.step(2).unwrap();
    *control.reclaim_group.borrow_mut() = vec![a];
    control.capacity.set(true);
    let c = owner.admit(generation(), 3).unwrap();
    owner.step(4).unwrap();
    owner.step(5).unwrap();
    owner.step(6).unwrap();
    owner.step(7).unwrap();
    assert_eq!(owner.status(c).unwrap(), Status::OutputBlocked);
    owner.take(a, 1).unwrap();
    control.capacity.set(true);
    assert_eq!(owner.step(8).unwrap(), Step::Waiting);
    assert!(control.restorations.borrow().is_empty());
    let calls = control.calls.get();
    assert_eq!(owner.step(9).unwrap(), Step::Idle);
    assert_eq!(control.calls.get(), calls);
    owner.cancel(a, true).unwrap();
    owner.retire(a).unwrap();
    assert!(control.closed.borrow().contains(&a));
    assert_eq!(owner.status(c).unwrap(), Status::OutputBlocked);
}

#[test]
fn zero_charged_release_does_not_mark_a_victim_evicted() {
    let control = Rc::new(Control::default());
    control.complete.set(true);
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 0).unwrap();
    owner.step(1).unwrap();
    owner.step(2).unwrap();
    *control.reclaim_group.borrow_mut() = vec![a];
    control.capacity.set(true);
    control.refuse_eviction.set(true);
    owner.admit(generation(), 3).unwrap();
    assert_eq!(owner.step(4).unwrap(), Step::Waiting);
    assert!(control.evictions.borrow().is_empty());
    owner.take(a, 1).unwrap();
    assert_eq!(owner.status(a).unwrap(), Status::Runnable);
}
fn generation() -> Generation {
    Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        Options {
            max_tokens: 4,
            output_capacity: 1,
            context_limit: 16,
            vocabulary: 16,
            stop_tokens: BTreeSet::from([TokenId(15)]),
            sampling: Sampling::Greedy,
            seed: 0,
            forced_quantum: 0,
        },
        None,
    )
    .unwrap()
}
fn owner(control: Rc<Control>) -> Owner<Model> {
    Owner::new(
        Model(control),
        Limits {
            max_requests: 8,
            max_batch: 2,
            prefill_tokens: 8,
            decode_tokens: 2,
            decode_share: 0.5,
            locality_seconds: 0.0,
        },
    )
    .unwrap()
}

#[test]
fn shared_completion_precedes_submission_and_peer_acceptance_is_independent() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 10).unwrap();
    let b = owner.admit(generation(), 10).unwrap();
    *control.selected.borrow_mut() = vec![Some(99), Some(7)];
    assert_eq!(owner.step(20).unwrap(), Step::Submitted);
    assert_eq!(owner.step(30).unwrap(), Step::Waiting);
    assert_eq!(control.calls.get(), 1);
    assert!(control.committed.borrow().is_empty());
    control.complete.set(true);
    assert_eq!(owner.step(50).unwrap(), Step::Reconciled);
    assert_eq!(owner.completed_service_ns(), 30);
    assert_eq!(
        owner.status(a).unwrap(),
        Status::Terminal(FinishReason::Failed)
    );
    assert!(owner.error(a).unwrap().contains("outside vocabulary"));
    assert_eq!(*control.committed.borrow(), vec![b]);
    assert_eq!(owner.status(b).unwrap(), Status::OutputBlocked);
    assert_eq!(owner.step(60).unwrap(), Step::Idle);
    assert_eq!(control.calls.get(), 1);
    assert_eq!(owner.take(b, 1).unwrap()[0].token, TokenId(7));
    assert_eq!(owner.step(70).unwrap(), Step::Submitted);
}
#[test]
fn cancellation_and_disconnect_wait_for_completion_and_retirement_drains_output() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let id = owner.admit(generation(), 0).unwrap();
    owner.step(1).unwrap();
    owner.cancel(id, true).unwrap();
    assert!(owner.retire(id).is_err());
    assert_eq!(owner.status(id).unwrap(), Status::AwaitingCompletion);
    assert_eq!(owner.step(2).unwrap(), Step::Waiting);
    control.complete.set(true);
    owner.step(3).unwrap();
    assert!(control.committed.borrow().is_empty());
    owner.retire(id).unwrap();
    let id = owner.admit(generation(), 4).unwrap();
    owner.step(5).unwrap();
    owner.step(6).unwrap();
    owner.cancel(id, false).unwrap();
    assert!(owner.retire(id).is_err());
    assert_eq!(owner.take(id, 1).unwrap().len(), 1);
    owner.retire(id).unwrap();
    assert_eq!(control.closed.borrow().len(), 2);
}
#[test]
fn fatal_completion_fails_peers_and_queued_requests_and_rejects_admission() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let ids = (0..3)
        .map(|_| owner.admit(generation(), 0).unwrap())
        .collect::<Vec<_>>();
    owner.step(1).unwrap();
    control.fatal.set(true);
    control.complete.set(true);
    owner.step(2).unwrap();
    assert_eq!(owner.fatal_error(), Some("device failed"));
    for id in ids {
        assert_eq!(
            owner.status(id).unwrap(),
            Status::Terminal(FinishReason::Failed)
        );
        owner.retire(id).unwrap();
    }
    assert!(control.committed.borrow().is_empty());
    assert!(owner.admit(generation(), 3).is_err());
}
#[test]
fn malformed_batch_is_retained_until_shared_completion() {
    let control = Rc::new(Control::default());
    control.wrong_rows.set(true);
    let mut owner = owner(control.clone());
    let id = owner.admit(generation(), 0).unwrap();
    owner.step(1).unwrap();
    assert!(owner.fatal_error().is_some());
    assert!(owner.retire(id).is_err());
    assert_eq!(owner.step(2).unwrap(), Step::Waiting);
    control.complete.set(true);
    assert_eq!(owner.step(3).unwrap(), Step::Reconciled);
    owner.retire(id).unwrap();
}
#[test]
fn capacity_wait_is_epoch_gated_and_clock_is_monotonic() {
    let control = Rc::new(Control::default());
    control.capacity.set(true);
    let mut owner = owner(control.clone());
    let id = owner.admit(generation(), 0).unwrap();
    let retired_peer = owner.admit(generation(), 0).unwrap();
    owner.cancel(retired_peer, true).unwrap();
    assert_eq!(owner.step(1).unwrap(), Step::Waiting);
    assert_eq!(
        owner.status(id).unwrap(),
        Status::CapacityBlocked {
            required: 100,
            available: 50
        }
    );
    assert_eq!(owner.step(2).unwrap(), Step::Idle);
    assert_eq!(control.calls.get(), 2);
    assert!(owner.step(1).is_err());
    let peer = owner.admit(generation(), 3).unwrap();
    control.capacity.set(false);
    assert_eq!(owner.step(4).unwrap(), Step::Submitted);
    assert_eq!(control.calls.get(), 3);
    owner.cancel(peer, true).unwrap();
}

#[test]
fn capacity_reclaims_then_drops_members_then_reduces_legal_tokens() {
    let control = Rc::new(Control::default());
    control.token_limit.set(Some(1));
    let mut owner = owner(control.clone());
    owner.admit(generation(), 0).unwrap();
    owner.admit(generation(), 0).unwrap();
    assert_eq!(owner.step(1).unwrap(), Step::Submitted);
    assert_eq!(
        *control.events.borrow(),
        vec!["prepare [2, 2]", "reclaim", "prepare [2]", "prepare [1]"]
    );
    control.complete.set(true);
    owner.step(2).unwrap();
    assert_eq!(control.committed.borrow().len(), 1);
}
#[test]
fn successful_idle_reclamation_retries_the_original_batch_before_shrinking() {
    let control = Rc::new(Control::default());
    control.capacity.set(true);
    control.released.set(50);
    let mut owner = owner(control.clone());
    owner.admit(generation(), 0).unwrap();
    owner.admit(generation(), 0).unwrap();
    assert_eq!(owner.step(1).unwrap(), Step::Submitted);
    assert_eq!(
        *control.events.borrow(),
        vec!["prepare [2, 2]", "reclaim", "prepare [2, 2]"]
    );
}
#[test]
fn indivisible_work_does_not_retry_the_same_shape_and_lonely_infeasibility_fails() {
    use seismic_engine::inputs::{BoundaryRule, InputSpan};
    let control = Rc::new(Control::default());
    control.capacity.set(true);
    let mut owner = owner(control.clone());
    let generation = Generation::new(
        vec![TokenId(1); 4],
        InputLayout::new(
            4,
            vec![InputSpan {
                start: 0,
                end: 4,
                identity: "image".into(),
                boundaries: BoundaryRule::Indivisible,
                language_history: false,
            }],
        )
        .unwrap(),
        Options {
            max_tokens: 4,
            output_capacity: 1,
            context_limit: 16,
            vocabulary: 16,
            stop_tokens: BTreeSet::from([TokenId(15)]),
            sampling: Sampling::Greedy,
            seed: 0,
            forced_quantum: 0,
        },
        None,
    )
    .unwrap();
    let id = owner.admit(generation, 0).unwrap();
    assert_eq!(owner.step(1).unwrap(), Step::Progress);
    assert_eq!(*control.events.borrow(), vec!["prepare [4]", "reclaim"]);
    assert_eq!(
        owner.status(id).unwrap(),
        Status::Terminal(FinishReason::Failed)
    );
    assert!(owner
        .error(id)
        .unwrap()
        .contains("required 100 bytes, available 50 bytes"));
}

#[test]
fn different_legal_token_counts_in_the_same_capacity_class_are_not_retried() {
    let control = Rc::new(Control::default());
    control.capacity.set(true);
    control.shape_quantum.set(8);
    let mut owner = owner(control.clone());
    let id = owner.admit(generation(), 0).unwrap();
    assert_eq!(owner.step(1).unwrap(), Step::Progress);
    assert_eq!(*control.events.borrow(), vec!["prepare [2]", "reclaim"]);
    assert_eq!(
        owner.status(id).unwrap(),
        Status::Terminal(FinishReason::Failed)
    );
}

#[test]
fn input_completion_yields_before_decoder_consumption_and_keeps_usage_unchanged() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let id = owner.admit(generation(), 10).unwrap();
    control.input_pending.borrow_mut().insert(id);
    assert_eq!(owner.step(20).unwrap(), Step::Submitted);
    assert_eq!(owner.status(id).unwrap(), Status::AwaitingCompletion);
    assert!(owner
        .checkpoint(id)
        .unwrap_err()
        .contains("submitted input"));
    assert_eq!(control.calls.get(), 0);
    assert_eq!(owner.step(25).unwrap(), Step::Waiting);
    control.complete.set(true);
    assert_eq!(owner.step(30).unwrap(), Step::Reconciled);
    assert_eq!(*control.input_committed.borrow(), vec![id]);
    assert_eq!(owner.usage(id).unwrap().completion_tokens, 0);
    assert_eq!(owner.output_len(id).unwrap(), 0);
    assert_eq!(owner.completed_service_ns(), 10);
    assert_eq!(control.calls.get(), 0);
    assert_eq!(owner.step(40).unwrap(), Step::Submitted);
    assert_eq!(control.calls.get(), 1);
    assert!(control.events.borrow().iter().any(|e| e == "prepare [2]"));
}
#[test]
fn disconnected_input_waits_for_completion_and_never_publishes_features() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 10).unwrap();
    let b = owner.admit(generation(), 10).unwrap();
    control.input_pending.borrow_mut().insert(a);
    assert_eq!(owner.step(20).unwrap(), Step::Submitted);
    assert_eq!(owner.status(b).unwrap(), Status::Runnable);
    owner.release(a).unwrap();
    assert!(control.closed.borrow().is_empty());
    assert!(control.input_dropped.borrow().is_empty());
    assert_eq!(owner.step(25).unwrap(), Step::Waiting);
    control.complete.set(true);
    assert_eq!(owner.step(30).unwrap(), Step::Reconciled);
    assert!(control.input_committed.borrow().is_empty());
    assert_eq!(*control.input_dropped.borrow(), vec![a]);
    assert_eq!(*control.closed.borrow(), vec![a]);
    assert_eq!(owner.step(40).unwrap(), Step::Submitted);
    assert_eq!(control.calls.get(), 1);
}
#[test]
fn input_preparation_rejects_token_results_without_losing_peer_acceptance() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 10).unwrap();
    let b = owner.admit(generation(), 10).unwrap();
    control.input_pending.borrow_mut().extend([a, b]);
    control.input_bad_token.borrow_mut().insert(a);
    assert_eq!(owner.step(20).unwrap(), Step::Submitted);
    control.complete.set(true);
    assert_eq!(owner.step(30).unwrap(), Step::Reconciled);
    assert_eq!(*control.input_committed.borrow(), vec![b]);
    assert_eq!(
        owner.status(a).unwrap(),
        Status::Terminal(FinishReason::Failed)
    );
    assert_eq!(owner.status(b).unwrap(), Status::Runnable);
    assert_eq!(owner.usage(b).unwrap().completion_tokens, 0);
    assert_eq!(control.calls.get(), 0);
}
#[test]
fn input_capacity_drops_members_but_never_retries_by_shrinking_decoder_tokens() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 10).unwrap();
    let b = owner.admit(generation(), 10).unwrap();
    control.input_pending.borrow_mut().extend([a, b]);
    control.capacity.set(true);
    assert_eq!(owner.step(20).unwrap(), Step::Waiting);
    assert_eq!(control.input_calls.get(), 2);
    assert_eq!(control.calls.get(), 0);
    assert_eq!(
        owner.status(a).unwrap(),
        Status::CapacityBlocked {
            required: 100,
            available: 50
        }
    );
    assert_eq!(owner.status(b).unwrap(), Status::Runnable);
}

#[test]
fn failed_input_completion_discards_every_unpublished_peer() {
    let control = Rc::new(Control::default());
    let mut owner = owner(control.clone());
    let a = owner.admit(generation(), 10).unwrap();
    let b = owner.admit(generation(), 10).unwrap();
    control.input_pending.borrow_mut().extend([a, b]);
    assert_eq!(owner.step(20).unwrap(), Step::Submitted);
    control.fatal.set(true);
    control.complete.set(true);
    assert_eq!(owner.step(30).unwrap(), Step::Reconciled);
    assert!(control.input_committed.borrow().is_empty());
    assert_eq!(*control.input_dropped.borrow(), vec![a, b]);
    assert_eq!(
        owner.status(a).unwrap(),
        Status::Terminal(FinishReason::Failed)
    );
    assert_eq!(
        owner.status(b).unwrap(),
        Status::Terminal(FinishReason::Failed)
    );
    assert_eq!(control.calls.get(), 0);
}
