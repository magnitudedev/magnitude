use magnitude_generation::{
    BoundaryRule, Constraint, Demand, FinishReason, Generation, InputLayout, InputSpan, Method,
    MethodCheckpoint, MethodCheckpointError, MethodChoice, MethodEffects, MethodRequirements,
    MethodState, Mtp, Options, Propose, RequestId, RoundStart, Sampling, SelectSpec, Shaping,
    TokenId, Verification, WaitReason, WorkKind,
};
use magnitude_model_executor::{
    FeatureRef, FeatureRetainer, FeatureSpan, Operation, Outcome, ResourceDomainId,
    RetainedFeatureSpan, Selected,
};
use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

fn feature(id: u64) -> FeatureRef {
    thread_local! { static FEATURES: RefCell<HashMap<u64, FeatureRef>> = RefCell::new(HashMap::new()); }
    FEATURES.with(|features| {
        features
            .borrow_mut()
            .entry(id)
            .or_insert_with(|| {
                FeatureRef::logical(
                    ResourceDomainId::new(format!("generation-{id}")).unwrap(),
                    32,
                    4,
                )
                .unwrap()
            })
            .clone()
    })
}

fn options() -> Options {
    Options {
        max_tokens: 8,
        output_capacity: 4,
        context_limit: 32,
        vocabulary: 100,
        stop_tokens: BTreeSet::from([TokenId(99)]),
        sampling: Sampling::Greedy,
        shaping: Shaping {
            temperature: 0.0,
            ..Default::default()
        },
        seed: 42,
        forced_quantum: 4,
        method: MethodChoice::Plain,
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

struct UnusedRetainer;

impl FeatureRetainer for UnusedRetainer {
    fn retain(&mut self, _span: FeatureSpan) -> Result<RetainedFeatureSpan, String> {
        Err("test method does not retain features".into())
    }
}

fn start(generation: &mut Generation, allowance: usize) -> magnitude_generation::RoundForward {
    assert_eq!(
        generation.start_round(RequestId(1), allowance).unwrap(),
        RoundStart::Target
    );
    generation.round_forward().unwrap().clone()
}

fn resolve(
    generation: &mut Generation,
    samples: &[u32],
    features: Option<FeatureRef>,
) -> Vec<Operation> {
    let samples = samples.iter().copied().map(TokenId).collect::<Vec<_>>();
    generation.resolve_round(&samples).unwrap();
    generation
        .commit_round(RequestId(1), features)
        .unwrap()
        .operations
}

fn prefill(generation: &mut Generation, selected: u32) {
    let round = start(generation, generation.prompt().len());
    assert_eq!(round.kind, WorkKind::Prefill);
    assert_eq!(round.tokens, [TokenId(1), TokenId(2)]);
    assert_eq!(round.selects.len(), 1);
    assert!(resolve(generation, &[selected], None).is_empty());
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
    fn mask(&self) -> Result<Arc<[u32]>, String> {
        Ok(vec![u32::MAX; 4].into())
    }
    fn position(&self) -> usize {
        self.accepted.len()
    }
    fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String> {
        if tokens.iter().any(|token| Some(*token) == self.reject) {
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
fn prefill_chunks_and_decode_share_the_round_protocol() {
    let mut generation = generation(None);
    let first = start(&mut generation, 1);
    assert_eq!(first.kind, WorkKind::Prefill);
    assert!(first.selects.is_empty());
    resolve(&mut generation, &[], None);
    assert_eq!(generation.resident_position(), 1);
    assert!(generation.generated().is_empty());

    let final_prefill = start(&mut generation, 1);
    assert_eq!(final_prefill.selects.len(), 1);
    resolve(&mut generation, &[10], None);
    let decode = start(&mut generation, 4);
    assert_eq!(decode.kind, WorkKind::Decode);
    assert_eq!(decode.tokens, [TokenId(10)]);
    resolve(&mut generation, &[11], None);
    assert_eq!(generation.generated(), [TokenId(10), TokenId(11)]);
}

#[test]
fn forced_prefill_and_decode_are_committed_rounds_without_selection() {
    let mut generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![TokenId(10), TokenId(11), TokenId(12)],
        reject: None,
    })));
    let prefill = start(&mut generation, 2);
    assert!(prefill.selects.is_empty());
    resolve(&mut generation, &[], None);
    generation.take(4).unwrap();
    let forced = start(&mut generation, 4);
    assert_eq!(forced.kind, WorkKind::Decode);
    assert!(forced.selects.is_empty());
    assert_eq!(forced.tokens, [TokenId(10), TokenId(11)]);
    resolve(&mut generation, &[], None);
    assert_eq!(
        generation.generated(),
        [TokenId(10), TokenId(11), TokenId(12)]
    );
}

#[test]
fn forced_runs_include_the_first_stop_and_never_commit_rows_past_it() {
    let mut generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![TokenId(10), TokenId(99), TokenId(12)],
        reject: None,
    })));
    start(&mut generation, 2);
    resolve(&mut generation, &[], None);
    generation.take(4).unwrap();

    let forced = start(&mut generation, 4);
    assert_eq!(forced.tokens, [TokenId(10)]);
    assert_eq!(forced.committed, 1);
    resolve(&mut generation, &[], None);
    assert_eq!(generation.generated(), [TokenId(10), TokenId(99)]);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(generation.resident_position(), 3);
}

#[test]
fn output_credit_stop_and_wait_reasons_are_round_native() {
    let mut generation = generation(None);
    prefill(&mut generation, 10);
    for token in [11, 12, 13] {
        start(&mut generation, 1);
        resolve(&mut generation, &[token], None);
    }
    assert_eq!(generation.wait_reason(), Some(WaitReason::Output));
    assert!(generation.start_round(RequestId(1), 1).is_err());
    assert_eq!(generation.take(4).unwrap().len(), 4);
    start(&mut generation, 1);
    resolve(&mut generation, &[99], None);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(generation.wait_reason(), Some(WaitReason::Finished));
}

#[test]
fn grammar_failure_fails_before_logical_publication() {
    let mut generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![],
        reject: Some(TokenId(12)),
    })));
    start(&mut generation, 2);
    assert!(generation.resolve_round(&[TokenId(12)]).is_err());
    assert_eq!(generation.finish_reason(), Some(FinishReason::Failed));
    assert_eq!(generation.resident_position(), 0);
    assert!(generation.generated().is_empty());
}

#[test]
fn prepared_transition_does_not_change_live_generation_until_commit() {
    let mut generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![],
        reject: Some(TokenId(12)),
    })));
    start(&mut generation, 2);
    assert!(
        generation
            .prepare_round_transition(RequestId(1), &[TokenId(12)], None, &mut UnusedRetainer)
            .is_err()
    );
    assert_eq!(generation.finish_reason(), None);
    assert_eq!(generation.resident_position(), 0);
    assert!(generation.awaiting_completion());

    let prepared = generation
        .prepare_round_transition(RequestId(1), &[TokenId(10)], None, &mut UnusedRetainer)
        .unwrap();
    assert_eq!(prepared.decision().accepted_rows, 2);
    drop(prepared);
    assert_eq!(generation.resident_position(), 0);
    assert!(generation.generated().is_empty());
    assert!(generation.awaiting_completion());

    let prepared = generation
        .prepare_round_transition(RequestId(1), &[TokenId(10)], None, &mut UnusedRetainer)
        .unwrap();
    let effects = generation.commit_transition(prepared);
    assert!(effects.operations.is_empty());
    assert_eq!(generation.generated(), [TokenId(10)]);
    assert_eq!(generation.resident_position(), 2);
    assert!(!generation.awaiting_completion());
}

#[test]
fn cancellation_reconciles_the_round_without_committing_or_publishing_it() {
    let mut generation = generation(None);
    start(&mut generation, 2);
    generation.cancel();
    let acceptance = generation.resolve_round(&[TokenId(10)]).unwrap();
    assert_eq!(acceptance.committed_rows, 0);
    generation.commit_round(RequestId(1), None).unwrap();
    assert_eq!(generation.resident_position(), 0);
    assert!(generation.generated().is_empty());
    assert_eq!(generation.finish_reason(), Some(FinishReason::Cancelled));
}

#[test]
fn eviction_discards_suspended_work_and_replays_through_rounds() {
    let mut generation = generation(None);
    prefill(&mut generation, 10);
    generation.take(4).unwrap();
    generation.credit_cached_tokens(1).unwrap();
    start(&mut generation, 1);
    generation.evicted().unwrap();
    assert_eq!(generation.wait_reason(), Some(WaitReason::Residency));
    generation.restored().unwrap();
    for _ in 0..2 {
        let replay = start(&mut generation, 1);
        assert_eq!(replay.kind, WorkKind::Replay);
        resolve(&mut generation, &[], None);
    }
    assert_eq!(generation.resident_position(), 2);
    assert_eq!(generation.generated(), [TokenId(10)]);
}

struct DraftMethod;
impl Method for DraftMethod {
    fn identity(&self) -> &str {
        "mtp:fixture:2"
    }
    fn requires(&self) -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::FEATURES,
            verify_demand: Demand::FEATURES,
            head: true,
        }
    }
    fn create(&self, _checkpoint: Option<&MethodCheckpoint>) -> Box<dyn MethodState> {
        Box::new(DraftState)
    }
}

struct PrimeMethod {
    calls: Arc<Mutex<Vec<Vec<TokenId>>>>,
}
impl Method for PrimeMethod {
    fn identity(&self) -> &str {
        "mtp:prime-fixture:1"
    }
    fn requires(&self) -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::FEATURES,
            verify_demand: Demand::FEATURES,
            head: true,
        }
    }
    fn create(&self, _checkpoint: Option<&MethodCheckpoint>) -> Box<dyn MethodState> {
        Box::new(PrimeState {
            calls: self.calls.clone(),
        })
    }
}
#[derive(Clone)]
struct PrimeState {
    calls: Arc<Mutex<Vec<Vec<TokenId>>>>,
}
impl MethodState for PrimeState {
    fn fork_transition(&self) -> Result<Box<dyn MethodState>, String> {
        Ok(Box::new(self.clone()))
    }
    fn prime(
        &mut self,
        _: RequestId,
        tokens: &[TokenId],
        _features: FeatureRef,
    ) -> Result<MethodEffects, String> {
        self.calls.lock().unwrap().push(tokens.to_vec());
        Ok(MethodEffects::default())
    }
    fn propose(
        &mut self,
        _: RequestId,
        _context: &[TokenId],
        _limit: usize,
        _: SelectSpec,
    ) -> Propose {
        Propose::Tokens(Vec::new())
    }
    fn observe(&mut self, _verification: Verification<'_>) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn reconcile(
        &mut self,
        _: &Operation,
        _: Outcome,
        _: Option<SelectSpec>,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn checkpoint(
        &self,
        _retainer: &mut dyn FeatureRetainer,
    ) -> Result<MethodCheckpoint, MethodCheckpointError> {
        Ok(MethodCheckpoint::Plain)
    }
    fn evict(&mut self) {}
    fn restore(&mut self) {}
    fn reclaimable(&self) -> u64 {
        0
    }
}

#[test]
fn every_prefill_chunk_primes_the_method_with_its_target_features() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut configured = options();
    configured.method = MethodChoice::Mtp { proposals: 1 };
    let mut generation = Generation::new_with_method(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        configured,
        None,
        Arc::new(PrimeMethod {
            calls: calls.clone(),
        }),
    )
    .unwrap();

    start(&mut generation, 1);
    resolve(&mut generation, &[], Some(feature(1)));
    start(&mut generation, 1);
    resolve(&mut generation, &[10], Some(feature(2)));
    assert_eq!(
        *calls.lock().unwrap(),
        vec![vec![TokenId(1)], vec![TokenId(2)]]
    );
}

#[test]
fn prefill_prime_head_reconciles_without_a_proposal_preview() {
    let request = RequestId(9);
    let mut configured = options();
    configured.method = MethodChoice::Mtp { proposals: 2 };
    let mut generation = Generation::new_with_method(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        configured,
        None,
        Arc::new(Mtp::new("fixture", 2).unwrap()),
    )
    .unwrap();
    assert_eq!(
        generation.start_round(request, 2).unwrap(),
        RoundStart::Target
    );
    generation.resolve_round(&[TokenId(10)]).unwrap();
    let priming = generation.commit_round(request, Some(feature(50))).unwrap();
    let [head @ Operation::Head { conditioning, .. }] = priming.operations.as_slice() else {
        panic!("two-row prefill must produce one shifted priming head")
    };
    assert_eq!(
        *conditioning,
        FeatureSpan {
            features: feature(50),
            start: 0,
            count: 1
        }
    );
    let effects = generation
        .reconcile_method(
            head,
            Outcome::Head {
                features: feature(51),
            },
        )
        .unwrap();
    assert_eq!(effects.head_prefix, Some(1));
    assert!(effects.operations.is_empty());
}
#[derive(Clone)]
struct DraftState;
impl MethodState for DraftState {
    fn fork_transition(&self) -> Result<Box<dyn MethodState>, String> {
        Ok(Box::new(self.clone()))
    }
    fn prime(
        &mut self,
        _: RequestId,
        _tokens: &[TokenId],
        _features: FeatureRef,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn propose(
        &mut self,
        _: RequestId,
        _context: &[TokenId],
        limit: usize,
        _: SelectSpec,
    ) -> Propose {
        Propose::Tokens([TokenId(11), TokenId(12)][..limit.min(2)].to_vec())
    }
    fn observe(&mut self, _verification: Verification<'_>) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn reconcile(
        &mut self,
        _: &Operation,
        _: Outcome,
        _: Option<SelectSpec>,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn checkpoint(
        &self,
        _retainer: &mut dyn FeatureRetainer,
    ) -> Result<MethodCheckpoint, MethodCheckpointError> {
        Ok(MethodCheckpoint::Plain)
    }
    fn evict(&mut self) {}
    fn restore(&mut self) {}
    fn reclaimable(&self) -> u64 {
        0
    }
}

#[test]
fn proposal_acceptance_and_usage_stay_inside_the_round_transcript() {
    let mut configured = options();
    configured.method = MethodChoice::Mtp { proposals: 2 };
    let mut generation = Generation::new_with_method(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        configured,
        None,
        Arc::new(DraftMethod),
    )
    .unwrap();
    start(&mut generation, 2);
    resolve(&mut generation, &[10], Some(feature(1)));
    generation.take(4).unwrap();
    let verify = start(&mut generation, 3);
    assert_eq!(verify.kind, WorkKind::Verify);
    assert_eq!(verify.tokens, [TokenId(10), TokenId(11), TokenId(12)]);
    resolve(&mut generation, &[11, 20, 21], Some(feature(2)));
    assert_eq!(
        generation.generated(),
        [TokenId(10), TokenId(11), TokenId(20)]
    );
    assert_eq!(generation.detailed_usage().draft_n, 2);
    assert_eq!(generation.detailed_usage().draft_n_accepted, 1);
}

struct ChainedMethod;
impl Method for ChainedMethod {
    fn identity(&self) -> &str {
        "mtp:chain:2"
    }
    fn requires(&self) -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::FEATURES,
            verify_demand: Demand::FEATURES,
            head: true,
        }
    }
    fn create(&self, _: Option<&MethodCheckpoint>) -> Box<dyn MethodState> {
        Box::new(ChainedState {
            proposed: Vec::new(),
            feature: None,
        })
    }
}

#[derive(Clone)]
struct ChainedState {
    proposed: Vec<TokenId>,
    feature: Option<FeatureRef>,
}
impl MethodState for ChainedState {
    fn fork_transition(&self) -> Result<Box<dyn MethodState>, String> {
        Ok(Box::new(self.clone()))
    }
    fn prime(
        &mut self,
        _: RequestId,
        _: &[TokenId],
        _: FeatureRef,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn propose(
        &mut self,
        request: RequestId,
        context: &[TokenId],
        limit: usize,
        _: SelectSpec,
    ) -> Propose {
        if !self.proposed.is_empty() {
            return Propose::Tokens(self.proposed.clone());
        }
        Propose::Pending(vec![Operation::Head {
            request,
            tokens: context[context.len() - 1..].to_vec(),
            conditioning: FeatureSpan {
                features: feature(70),
                start: 0,
                count: 1,
            },
            position: context.len() - 1,
            demand: Demand::FEATURES
                & if limit > 0 {
                    Demand::FEATURES
                } else {
                    Demand::NONE
                },
        }])
    }
    fn observe(&mut self, _: Verification<'_>) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn reconcile(
        &mut self,
        operation: &Operation,
        outcome: Outcome,
        next_select: Option<SelectSpec>,
    ) -> Result<MethodEffects, String> {
        let request = operation.request();
        match outcome {
            Outcome::Head { features } => self.feature = Some(features),
            Outcome::Project { selected } => self.proposed.push(selected[0].token),
            _ => return Err("wrong chained outcome".into()),
        }
        Ok(MethodEffects {
            operations: next_select
                .map(|select| Operation::Project {
                    request,
                    features: self.feature.clone().unwrap(),
                    select,
                })
                .into_iter()
                .collect(),
            head_prefix: None,
        })
    }
    fn checkpoint(
        &self,
        _retainer: &mut dyn FeatureRetainer,
    ) -> Result<MethodCheckpoint, MethodCheckpointError> {
        Ok(MethodCheckpoint::Plain)
    }
    fn evict(&mut self) {}
    fn restore(&mut self) {}
    fn reclaimable(&self) -> u64 {
        0
    }
}

fn ready_chained_generation(request: RequestId) -> Generation {
    let mut configured = options();
    configured.method = MethodChoice::Mtp { proposals: 2 };
    let mut generation = Generation::new_with_method(
        vec![TokenId(1)],
        InputLayout::new(1, vec![]).unwrap(),
        configured,
        Some(Box::new(Grammar {
            accepted: vec![],
            forced: vec![],
            reject: None,
        })),
        Arc::new(ChainedMethod),
    )
    .unwrap();
    assert_eq!(
        generation.start_round(request, 1).unwrap(),
        RoundStart::Target
    );
    generation.resolve_round(&[TokenId(10)]).unwrap();
    generation.commit_round(request, Some(feature(60))).unwrap();
    generation.take(4).unwrap();
    generation
}

#[test]
fn method_project_chain_uses_generation_owned_preview_selection() {
    let request = RequestId(44);
    let mut generation = ready_chained_generation(request);

    let RoundStart::Method(head) = generation.start_round(request, 3).unwrap() else {
        panic!("method must start with head work")
    };
    assert!(
        generation
            .fork_at(generation.resident_position(), &mut UnusedRetainer)
            .is_err()
    );
    let first = generation
        .reconcile_method(
            &head[0],
            Outcome::Head {
                features: feature(71),
            },
        )
        .unwrap();
    let Operation::Project { select, .. } = &first.operations[0] else {
        panic!("head must yield first projection")
    };
    assert_eq!(select.position, 1);
    assert_eq!(select.domain, 1);

    let second = generation
        .reconcile_method(
            &first.operations[0],
            Outcome::Project {
                selected: vec![Selected {
                    token: TokenId(11),
                    status: 0,
                }],
            },
        )
        .unwrap();
    let Operation::Project { select, .. } = &second.operations[0] else {
        panic!("first projection must yield second projection")
    };
    assert_eq!(select.position, 2);

    assert!(
        generation
            .reconcile_method(
                &second.operations[0],
                Outcome::Project {
                    selected: vec![Selected {
                        token: TokenId(12),
                        status: 0,
                    }],
                },
            )
            .unwrap()
            .operations
            .is_empty()
    );
    assert_eq!(
        generation.start_round(request, 3).unwrap(),
        RoundStart::Target
    );
    assert_eq!(
        generation.round_forward().unwrap().tokens,
        [TokenId(10), TokenId(11), TokenId(12)]
    );
}

#[test]
fn method_preview_is_cleared_by_cancellation_and_reconciliation_error() {
    let request = RequestId(45);
    let mut cancelled = ready_chained_generation(request);
    assert!(matches!(
        cancelled.start_round(request, 3).unwrap(),
        RoundStart::Method(_)
    ));
    cancelled.cancel();
    assert!(
        cancelled
            .fork_at(cancelled.resident_position(), &mut UnusedRetainer)
            .is_ok()
    );

    let mut failed = ready_chained_generation(request);
    let RoundStart::Method(head) = failed.start_round(request, 3).unwrap() else {
        panic!("method must start with head work")
    };
    let project = failed
        .reconcile_method(
            &head[0],
            Outcome::Head {
                features: feature(72),
            },
        )
        .unwrap();
    assert!(
        failed
            .reconcile_method(
                &project.operations[0],
                Outcome::Project {
                    selected: vec![Selected {
                        token: TokenId(11),
                        status: 1,
                    }],
                },
            )
            .is_err()
    );
    assert_eq!(failed.finish_reason(), Some(FinishReason::Failed));
    assert_eq!(failed.constraint_position(), Some(1));
    assert!(
        failed
            .fork_at(failed.resident_position(), &mut UnusedRetainer)
            .is_ok()
    );
}

#[test]
fn mtp_choice_requires_an_injected_factory() {
    let mut configured = options();
    configured.method = MethodChoice::Mtp { proposals: 2 };
    assert!(
        Generation::new(
            vec![TokenId(1), TokenId(2)],
            InputLayout::new(2, vec![]).unwrap(),
            configured,
            None,
        )
        .is_err()
    );
}

#[test]
fn retained_prefix_restores_position_into_a_fresh_extended_prompt() {
    let prompt = (0..64).map(TokenId).collect::<Vec<_>>();
    let mut configured = options();
    configured.context_limit = 128;
    configured.vocabulary = 256;
    let grammar = || {
        Box::new(Grammar {
            accepted: vec![],
            forced: vec![],
            reject: None,
        }) as Box<dyn Constraint>
    };
    let mut source = Generation::new(
        prompt.clone(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        configured.clone(),
        Some(grammar()),
    )
    .unwrap();
    assert_eq!(start(&mut source, prompt.len()).tokens, prompt);
    resolve(&mut source, &[70], None);
    assert_eq!(source.resident_position(), 64);
    assert_eq!(source.generated(), [TokenId(70)]);
    assert_eq!(source.constraint_position(), Some(1));
    assert_eq!(source.output_len(), 1);
    let checkpoint = source.method_checkpoint(&mut UnusedRetainer).unwrap();

    let mut extended = prompt;
    extended.extend([TokenId(80), TokenId(81)]);
    let mut fresh = Generation::new(
        extended.clone(),
        InputLayout::new(extended.len(), vec![]).unwrap(),
        configured,
        Some(grammar()),
    )
    .unwrap();
    fresh.restore_prefix(64, &checkpoint).unwrap();

    assert_eq!(fresh.prompt(), extended);
    assert_eq!(fresh.resident_position(), 64);
    assert_eq!(fresh.accepted_position(), 64);
    assert_eq!(fresh.detailed_usage().cached_tokens, 64);
    assert!(fresh.generated().is_empty());
    assert_eq!(fresh.output_len(), 0);
    assert_eq!(fresh.constraint_position(), Some(0));
    assert_eq!(fresh.method_identity(), "plain");
}

#[test]
fn retained_prefix_requires_an_exact_layout_boundary() {
    let prompt = (0..66).map(TokenId).collect::<Vec<_>>();
    let layout = InputLayout::new(
        prompt.len(),
        vec![InputSpan {
            start: 63,
            end: 66,
            identity: "image".into(),
            boundaries: BoundaryRule::Indivisible,
            language_history: false,
        }],
    )
    .unwrap();
    let mut configured = options();
    configured.context_limit = 128;
    configured.vocabulary = 256;
    let mut fresh = Generation::new(prompt, layout, configured, None).unwrap();
    assert!(fresh.restore_prefix(64, &MethodCheckpoint::Plain).is_err());
    assert_eq!(fresh.resident_position(), 0);
    assert_eq!(fresh.detailed_usage().cached_tokens, 0);
}

#[test]
fn causal_reconciliation_commits_the_final_sample_without_new_output() {
    let prompt = (0..64).map(TokenId).collect::<Vec<_>>();
    let mut configured = options();
    configured.max_tokens = 1;
    configured.context_limit = 128;
    configured.vocabulary = 256;
    let mut generation = Generation::new(
        prompt.clone(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        configured,
        None,
    )
    .unwrap();
    start(&mut generation, 64);
    resolve(&mut generation, &[TokenId(70).0], None);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
    assert_eq!(generation.resident_position(), 64);
    assert_eq!(generation.generated(), [TokenId(70)]);
    assert_eq!(
        generation.pending_reconciliation().unwrap(),
        magnitude_generation::PendingReconciliation { start: 64, end: 65 }
    );

    generation.start_reconciliation(1).unwrap();
    let reconciliation = generation.round_forward().unwrap();
    assert_eq!(reconciliation.kind, WorkKind::Replay);
    assert_eq!(reconciliation.tokens, [TokenId(70)]);
    assert!(reconciliation.selects.is_empty());
    generation.resolve_round(&[]).unwrap();
    generation.commit_round(RequestId(1), None).unwrap();

    assert_eq!(generation.resident_position(), 65);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
    assert_eq!(generation.generated(), [TokenId(70)]);
    assert_eq!(generation.output_len(), 1);
    assert!(generation.pending_reconciliation().is_none());
}

#[test]
fn staged_causal_reconciliation_advances_after_finished_prefill() {
    let prompt = (0..64).map(TokenId).collect::<Vec<_>>();
    let mut configured = options();
    configured.max_tokens = 1;
    configured.context_limit = 128;
    configured.vocabulary = 256;
    let mut generation = Generation::new(
        prompt.clone(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        configured,
        None,
    )
    .unwrap();
    start(&mut generation, 64);
    let transition = generation
        .prepare_round_transition(RequestId(1), &[TokenId(70)], None, &mut UnusedRetainer)
        .unwrap();
    assert_eq!(transition.decision().accepted_rows, 64);
    generation.commit_transition(transition);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
    assert_eq!(generation.pending_reconciliation().unwrap().end, 65);

    generation.start_reconciliation(1).unwrap();
    let transition = generation
        .prepare_round_transition(RequestId(1), &[], None, &mut UnusedRetainer)
        .unwrap();
    assert_eq!(transition.decision().accepted_rows, 1);
    generation.commit_transition(transition);
    assert_eq!(generation.resident_position(), 65);
    assert!(generation.pending_reconciliation().is_none());
    assert_eq!(generation.generated(), [TokenId(70)]);
}
