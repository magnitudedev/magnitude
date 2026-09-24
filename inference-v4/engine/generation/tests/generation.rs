use magnitude_generation::{
    BoundaryRule, Constraint, Demand, FinishReason, Generation, InputLayout, InputSpan, Method,
    MethodCheckpoint, MethodCheckpointError, MethodChoice, MethodEffects, MethodRequirements,
    MethodState, Mtp, Options, Propose, RequestId, RoundStart, Sampling, SelectSpec, Shaping,
    TokenId, Verification, WaitReason, WorkKind,
};
use magnitude_model_executor::{
    FeatureReader, FeatureRef, FeatureRows, FeatureSpan, Operation, Outcome, ResourceDomainId,
    Selected,
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

/// Reads row `r` of feature `id` as the four bytes `[id, r, 0, 0]`.
struct Rows;

impl FeatureReader for Rows {
    fn read(&mut self, span: &FeatureSpan) -> Result<FeatureRows, String> {
        let id = span
            .features
            .domain()
            .as_str()
            .strip_prefix("generation-")
            .and_then(|id| id.parse::<u8>().ok())
            .ok_or("unknown test feature")?;
        let bytes = (span.start..span.start + span.count)
            .flat_map(|row| [id, row as u8, 0, 0])
            .collect::<Vec<_>>();
        FeatureRows::new(bytes.into(), span.count).map_err(|error| error.to_string())
    }
}

/// The (feature id, row) of every row, in order.
fn row_ids(rows: &FeatureRows) -> Vec<(u8, u8)> {
    rows.bytes()
        .chunks_exact(4)
        .map(|row| (row[0], row[1]))
        .collect()
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
    let transition = generation
        .prepare_round_transition(RequestId(1), &samples, features, &mut Rows)
        .unwrap();
    generation.commit_transition(transition).operations
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
fn prepared_transition_does_not_change_live_generation_until_commit() {
    let mut generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![],
        reject: Some(TokenId(12)),
    })));
    start(&mut generation, 2);
    assert!(
        generation
            .prepare_round_transition(RequestId(1), &[TokenId(12)], None, &mut Rows)
            .is_err()
    );
    assert_eq!(generation.finish_reason(), None);
    assert_eq!(generation.resident_position(), 0);
    assert!(generation.awaiting_completion());

    let prepared = generation
        .prepare_round_transition(RequestId(1), &[TokenId(10)], None, &mut Rows)
        .unwrap();
    assert_eq!(prepared.decision().accepted_rows, 2);
    drop(prepared);
    assert_eq!(generation.resident_position(), 0);
    assert!(generation.generated().is_empty());
    assert!(generation.awaiting_completion());

    let prepared = generation
        .prepare_round_transition(RequestId(1), &[TokenId(10)], None, &mut Rows)
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
    assert!(!generation.awaiting_completion());
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

#[test]
fn eviction_resumes_from_a_retained_prefix_and_replays_the_rest() {
    let mut generation = generation(None);
    prefill(&mut generation, 10);
    let checkpoint = generation.method_checkpoint().unwrap();
    generation.evicted().unwrap();
    // Accepted input: the prompt and the sampled token, three rows.
    assert!(generation.restored_at(4, &checkpoint).is_err());
    generation.restored_at(1, &checkpoint).unwrap();
    assert!(generation.restored_at(1, &checkpoint).is_err());
    assert_eq!(generation.resident_position(), 1);
    let replay = start(&mut generation, 4);
    assert_eq!(replay.kind, WorkKind::Replay);
    assert_eq!(replay.tokens, [TokenId(2), TokenId(10)]);
    resolve(&mut generation, &[], None);
    assert_eq!(generation.resident_position(), 3);
    assert_eq!(generation.generated(), [TokenId(10)]);
}

fn mtp_generation(prompt: &[u32], proposals: u8) -> Generation {
    let mut configured = options();
    configured.method = MethodChoice::Mtp { proposals };
    Generation::new_with_method(
        prompt.iter().copied().map(TokenId).collect(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        configured,
        None,
        Arc::new(Mtp::new("fixture", usize::from(proposals)).unwrap()),
    )
    .unwrap()
}

fn head_parts(operation: &Operation) -> (Vec<TokenId>, Vec<(u8, u8)>, usize, Vec<SelectSpec>) {
    let Operation::Head {
        tokens,
        conditioning,
        position,
        proposals,
        ..
    } = operation
    else {
        panic!("expected a head transaction")
    };
    (
        tokens.clone(),
        row_ids(conditioning),
        *position,
        proposals.clone(),
    )
}

/// Reconcile a head transaction as the service does.
fn reconcile_head(generation: &mut Generation, operation: &Operation, proposals: &[(u32, u8)]) {
    let outcome = Outcome::Head {
        proposals: proposals
            .iter()
            .map(|&(token, status)| Selected {
                token: TokenId(token),
                status,
            })
            .collect(),
    };
    let transition = generation
        .prepare_method_transition(operation, &outcome)
        .unwrap();
    let Operation::Head { tokens, .. } = operation else {
        unreachable!()
    };
    assert_eq!(transition.decision().accepted_rows, tokens.len());
    generation.commit_method_transition(transition);
}

#[test]
fn prefill_chunks_enter_target_conditioned_pairs_and_keep_the_anchor() {
    let request = RequestId(1);
    let mut generation = mtp_generation(&[1, 2, 3], 2);
    // First chunk: rows 1, 2. Its pairs are (2, f5.0); f5.1 waits for 3.
    start(&mut generation, 2);
    let effects = resolve(&mut generation, &[], Some(feature(5)));
    let [head] = effects.as_slice() else {
        panic!("a non-final chunk enters its complete pairs")
    };
    let (tokens, rows, position, proposals) = head_parts(head);
    assert_eq!((tokens, rows, position), (vec![TokenId(2)], vec![(5, 0)], 0));
    assert!(proposals.is_empty());
    reconcile_head(&mut generation, head, &[]);
    // Final chunk: row 3 selects 10. Pairs (3, f5.1) are entered; the anchor
    // (10, f6.0) waits for the first draft.
    start(&mut generation, 1);
    let effects = resolve(&mut generation, &[10], Some(feature(6)));
    let [head] = effects.as_slice() else {
        panic!("the final chunk enters all but the anchor")
    };
    let (tokens, rows, position, _) = head_parts(head);
    assert_eq!((tokens, rows, position), (vec![TokenId(3)], vec![(5, 1)], 1));
    reconcile_head(&mut generation, head, &[]);
    generation.take(4).unwrap();
    let RoundStart::Method(draft) = generation.start_round(request, 4).unwrap() else {
        panic!("decode drafts first")
    };
    let (tokens, rows, position, proposals) = head_parts(&draft[0]);
    assert_eq!((tokens, rows, position), (vec![TokenId(10)], vec![(6, 0)], 2));
    // Proposal selections share the target's keys at the same output rows.
    assert_eq!(
        proposals
            .iter()
            .map(|select| (select.position, select.domain))
            .collect::<Vec<_>>(),
        [(1, 0), (2, 0)]
    );
}

#[test]
fn verification_accepts_the_matching_prefix_and_re_enters_accepted_rows() {
    let request = RequestId(1);
    let mut generation = mtp_generation(&[1, 2], 3);
    start(&mut generation, 2);
    let effects = resolve(&mut generation, &[10], Some(feature(1)));
    reconcile_head(&mut generation, &effects[0], &[]);
    generation.take(4).unwrap();
    let RoundStart::Method(draft) = generation.start_round(request, 4).unwrap() else {
        panic!("decode drafts first")
    };
    reconcile_head(&mut generation, &draft[0], &[(11, 0), (12, 0), (13, 0)]);
    let verify = start(&mut generation, 4);
    assert_eq!(verify.kind, WorkKind::Verify);
    assert_eq!(
        verify.tokens,
        [TokenId(10), TokenId(11), TokenId(12), TokenId(13)]
    );
    // The target agrees on 11 and 12 and samples 20 after them.
    assert!(resolve(&mut generation, &[11, 12, 20, 30], Some(feature(2))).is_empty());
    assert_eq!(
        generation.generated(),
        [TokenId(10), TokenId(11), TokenId(12), TokenId(20)]
    );
    assert_eq!(generation.resident_position(), 5);
    assert_eq!(generation.detailed_usage().draft_n, 3);
    assert_eq!(generation.detailed_usage().draft_n_accepted, 2);
    generation.take(4).unwrap();
    // The next draft enters every accepted row with its target feature and
    // anchors on the bonus token: (11, f2.0), (12, f2.1), (20, f2.2).
    let RoundStart::Method(draft) = generation.start_round(request, 4).unwrap() else {
        panic!("decode drafts again")
    };
    let (tokens, rows, position, _) = head_parts(&draft[0]);
    assert_eq!(tokens, [TokenId(11), TokenId(12), TokenId(20)]);
    assert_eq!(rows, [(2, 0), (2, 1), (2, 2)]);
    assert_eq!(position, 2);
}

#[test]
fn proposals_stop_at_a_failed_selection_or_a_stop_token() {
    let request = RequestId(1);
    let mut generation = mtp_generation(&[1, 2], 3);
    start(&mut generation, 2);
    let effects = resolve(&mut generation, &[10], Some(feature(1)));
    reconcile_head(&mut generation, &effects[0], &[]);
    generation.take(4).unwrap();
    let RoundStart::Method(draft) = generation.start_round(request, 4).unwrap() else {
        panic!("decode drafts first")
    };
    reconcile_head(&mut generation, &draft[0], &[(11, 0), (99, 0), (13, 0)]);
    assert_eq!(start(&mut generation, 4).tokens, [TokenId(10), TokenId(11)]);

    let mut failed = mtp_generation(&[1, 2], 3);
    start(&mut failed, 2);
    let effects = resolve(&mut failed, &[10], Some(feature(1)));
    reconcile_head(&mut failed, &effects[0], &[]);
    failed.take(4).unwrap();
    let RoundStart::Method(draft) = failed.start_round(request, 4).unwrap() else {
        panic!("decode drafts first")
    };
    reconcile_head(&mut failed, &draft[0], &[(11, 0), (0, 1), (13, 0)]);
    assert_eq!(start(&mut failed, 4).tokens, [TokenId(10), TokenId(11)]);
}

#[test]
fn checkpoints_carry_host_rows_and_restore_at_their_target_boundary() {
    let request = RequestId(1);
    let mut source = mtp_generation(&[1, 2], 2);
    start(&mut source, 2);
    let effects = resolve(&mut source, &[10], Some(feature(3)));
    // A checkpoint needs reconciled method work.
    assert!(source.method_checkpoint().is_err());
    reconcile_head(&mut source, &effects[0], &[]);
    let checkpoint = source.method_checkpoint().unwrap();
    assert_eq!(checkpoint.retained_bytes(), 4);
    let fork = source.fork_at(source.resident_position()).unwrap();
    assert_eq!(fork.method_checkpoint().unwrap(), checkpoint);
    let MethodCheckpoint::Mtp(state) = &checkpoint else {
        panic!("MTP checkpoint")
    };
    // Head row 0 entered, the anchor pending: two target rows.
    assert_eq!((state.position(), state.target_rows()), (1, 2));
    let mut fresh = mtp_generation(&[1, 2, 7], 2);
    assert!(fresh.restore_prefix(1, &checkpoint).is_err());
    fresh.restore_prefix(2, &checkpoint).unwrap();
    assert_eq!(fresh.resident_position(), 2);
    let _ = request;
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

struct PrimeMethod {
    calls: Arc<Mutex<Vec<(Vec<TokenId>, Option<TokenId>)>>>,
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
    fn proposals(&self) -> usize {
        1
    }
    fn create(
        &self,
        _checkpoint: Option<&MethodCheckpoint>,
    ) -> Result<Box<dyn MethodState>, String> {
        Ok(Box::new(PrimeState {
            calls: self.calls.clone(),
        }))
    }
}
#[derive(Clone)]
struct PrimeState {
    calls: Arc<Mutex<Vec<(Vec<TokenId>, Option<TokenId>)>>>,
}
impl MethodState for PrimeState {
    fn fork_transition(&self) -> Box<dyn MethodState> {
        Box::new(self.clone())
    }
    fn prime(
        &mut self,
        _: RequestId,
        tokens: &[TokenId],
        next: Option<TokenId>,
        _features: FeatureRef,
        _: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        self.calls.lock().unwrap().push((tokens.to_vec(), next));
        Ok(MethodEffects::default())
    }
    fn propose(&mut self, _: RequestId, _: &[SelectSpec]) -> Propose {
        Propose::Tokens(Vec::new())
    }
    fn observe(
        &mut self,
        _: RequestId,
        _: Verification<'_>,
        _: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        Ok(MethodEffects::default())
    }
    fn reconcile(&mut self, _: &Operation, _: Outcome) -> Result<(), String> {
        Ok(())
    }
    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError> {
        Ok(MethodCheckpoint::Plain)
    }
    fn evict(&mut self) {}
    fn reclaimable(&self) -> u64 {
        0
    }
}

#[test]
fn every_prefill_chunk_primes_the_method_with_its_selected_successor() {
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
        vec![
            (vec![TokenId(1)], None),
            (vec![TokenId(2)], Some(TokenId(10)))
        ]
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
    let checkpoint = source.method_checkpoint().unwrap();
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
        .prepare_round_transition(RequestId(1), &[TokenId(70)], None, &mut Rows)
        .unwrap();
    assert_eq!(transition.decision().accepted_rows, 64);
    generation.commit_transition(transition);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
    assert_eq!(generation.pending_reconciliation().unwrap().end, 65);
    generation.start_reconciliation(1).unwrap();
    let reconciliation = generation.round_forward().unwrap();
    assert_eq!(reconciliation.kind, WorkKind::Replay);
    assert_eq!(reconciliation.tokens, [TokenId(70)]);
    assert!(reconciliation.selects.is_empty());
    let transition = generation
        .prepare_round_transition(RequestId(1), &[], None, &mut Rows)
        .unwrap();
    assert_eq!(transition.decision().accepted_rows, 1);
    generation.commit_transition(transition);
    assert_eq!(generation.resident_position(), 65);
    assert!(generation.pending_reconciliation().is_none());
    assert_eq!(generation.generated(), [TokenId(70)]);
    assert_eq!(generation.output_len(), 1);
}
