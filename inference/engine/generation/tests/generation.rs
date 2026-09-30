use magnitude_executor::{
    FeatureReader, FeatureRef, FeatureRows, FeatureSpan, Operation, Outcome, ResourceDomainId,
    Selected,
};
use magnitude_generation::{
    BoundaryRule, Constraint, DFlash, Demand, EndOfGeneration, FinishReason, Generation,
    InputLayout, InputSpan, Method, MethodCheckpoint, MethodCheckpointError, MethodChoice,
    MethodEffects, MethodRequirements, MethodState, Mtp, Options, Propose, ReasoningBudget,
    RequestId, RoundForward, RoundStart, Sampling, SelectSpec, Shaping, StartedRound, TokenId,
    Verification, WorkKind,
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
        suppressed_tokens: BTreeSet::new(),
        sampling: Sampling::Greedy,
        shaping: Shaping {
            temperature: 0.0,
            ..Default::default()
        },
        seed: 42,
        forced_quantum: 4,
        method: MethodChoice::Plain,
        end_of_generation: EndOfGeneration::Stop,
        reasoning_budget: None,
    }
}

/// A generation made resident on fresh numerical state.
fn resident(mut generation: Generation) -> Generation {
    generation.resume_at(None).unwrap();
    generation
}

fn generation(constraint: Option<Box<dyn Constraint>>) -> Generation {
    resident(
        Generation::new(
            vec![TokenId(1), TokenId(2)],
            InputLayout::new(2, vec![]).unwrap(),
            options(),
            constraint,
        )
        .unwrap(),
    )
}

/// Start a target round, checking it against the bound computed before it
/// started.
fn start(generation: Generation, allowance: usize) -> StartedRound {
    let bound = generation.publication_bound(allowance).unwrap();
    match generation.start_round(RequestId(1), allowance) {
        Ok(RoundStart::Target(round)) => {
            assert!(round.round_publication_bound() <= bound);
            round
        }
        Ok(RoundStart::Method(..)) => panic!("expected a target round"),
        Err((_, error)) => panic!("{error}"),
    }
}

/// Start a round that returns method (drafter) work.
fn draft(generation: Generation, allowance: usize) -> (Generation, Vec<Operation>) {
    match generation.start_round(RequestId(1), allowance) {
        Ok(RoundStart::Method(generation, operations)) => (generation, operations),
        Ok(RoundStart::Target(_)) => panic!("expected method work"),
        Err((_, error)) => panic!("{error}"),
    }
}

fn resolve(
    round: StartedRound,
    samples: &[u32],
    features: Option<FeatureRef>,
) -> (Generation, Vec<Operation>) {
    let samples = samples.iter().copied().map(TokenId).collect::<Vec<_>>();
    let transition = round
        .prepare_round_transition(RequestId(1), &samples, features, &mut Rows)
        .unwrap();
    let (generation, effects) = round.commit(transition);
    (generation, effects.operations)
}

fn forward(round: &StartedRound) -> RoundForward {
    round.round_forward().clone()
}

fn prefill(generation: Generation, selected: u32) -> Generation {
    let allowance = generation.prompt().len();
    let round = start(generation, allowance);
    assert_eq!(round.round_forward().kind, WorkKind::Prefill);
    assert_eq!(round.round_forward().tokens, [TokenId(1), TokenId(2)]);
    assert_eq!(round.round_forward().selects.len(), 1);
    let (generation, effects) = resolve(round, &[selected], None);
    assert!(effects.is_empty());
    generation
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
    let first = start(generation(None), 1);
    assert_eq!(first.round_forward().kind, WorkKind::Prefill);
    assert!(first.round_forward().selects.is_empty());
    let (generation, _) = resolve(first, &[], None);
    assert_eq!(generation.resident_position(), 1);
    assert!(generation.generated().is_empty());

    let final_prefill = start(generation, 1);
    assert_eq!(final_prefill.round_forward().selects.len(), 1);
    let (generation, _) = resolve(final_prefill, &[10], None);
    let decode = start(generation, 4);
    assert_eq!(decode.round_forward().kind, WorkKind::Decode);
    assert_eq!(decode.round_forward().tokens, [TokenId(10)]);
    let (generation, _) = resolve(decode, &[11], None);
    assert_eq!(generation.generated(), [TokenId(10), TokenId(11)]);
}

#[test]
fn forced_prefill_and_decode_are_committed_rounds_without_selection() {
    let generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![TokenId(10), TokenId(11), TokenId(12)],
        reject: None,
    })));
    let prefill = start(generation, 2);
    assert!(prefill.round_forward().selects.is_empty());
    let (mut generation, _) = resolve(prefill, &[], None);
    generation.take(4).unwrap();
    let forced = start(generation, 4);
    assert_eq!(forced.round_forward().kind, WorkKind::Decode);
    assert!(forced.round_forward().selects.is_empty());
    assert_eq!(forced.round_forward().tokens, [TokenId(10), TokenId(11)]);
    let (generation, _) = resolve(forced, &[], None);
    assert_eq!(
        generation.generated(),
        [TokenId(10), TokenId(11), TokenId(12)]
    );
}

#[test]
fn forced_runs_include_the_first_stop_and_never_commit_rows_past_it() {
    let generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![TokenId(10), TokenId(99), TokenId(12)],
        reject: None,
    })));
    let (mut generation, _) = resolve(start(generation, 2), &[], None);
    generation.take(4).unwrap();

    let forced = start(generation, 4);
    assert_eq!(forced.round_forward().tokens, [TokenId(10)]);
    assert_eq!(forced.round_forward().committed, 1);
    let (generation, _) = resolve(forced, &[], None);
    assert_eq!(generation.generated(), [TokenId(10), TokenId(99)]);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(generation.resident_position(), 3);
}

#[test]
fn output_credit_and_stop_refuse_the_next_round() {
    let mut generation = prefill(generation(None), 10);
    for token in [11, 12, 13] {
        (generation, _) = resolve(start(generation, 1), &[token], None);
    }
    // Full output refuses a round, and the refusal returns the generation.
    assert!(generation.publication_bound(1).is_err());
    let Err((mut generation, _)) = generation.start_round(RequestId(1), 1) else {
        panic!("full output refuses a round")
    };
    assert_eq!(generation.take(4).unwrap().len(), 4);
    let (generation, _) = resolve(start(generation, 1), &[99], None);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Stop));
    assert!(generation.publication_bound(1).is_err());
    assert!(generation.start_round(RequestId(1), 1).is_err());
}

#[test]
fn prepared_transition_does_not_change_live_generation_until_commit() {
    let generation = generation(Some(Box::new(Grammar {
        accepted: vec![],
        forced: vec![],
        reject: Some(TokenId(12)),
    })));
    let round = start(generation, 2);
    assert!(round
        .prepare_round_transition(RequestId(1), &[TokenId(12)], None, &mut Rows)
        .is_err());
    assert_eq!(round.generation().finish_reason(), None);
    assert_eq!(round.generation().resident_position(), 0);

    let prepared = round
        .prepare_round_transition(RequestId(1), &[TokenId(10)], None, &mut Rows)
        .unwrap();
    assert_eq!(prepared.decision().accepted_rows, 2);
    drop(prepared);
    assert_eq!(round.generation().resident_position(), 0);
    assert!(round.generation().generated().is_empty());

    let prepared = round
        .prepare_round_transition(RequestId(1), &[TokenId(10)], None, &mut Rows)
        .unwrap();
    let (generation, effects) = round.commit(prepared);
    assert!(effects.operations.is_empty());
    assert_eq!(generation.generated(), [TokenId(10)]);
    assert_eq!(generation.resident_position(), 2);
}

/// A started round round-trips: start, prepare and commit return the
/// generation, which starts the next round from the committed state.
#[test]
fn a_started_round_commits_back_into_its_generation() {
    let round = start(generation(None), 2);
    assert_eq!(round.generation().resident_position(), 0);
    let prepared = round
        .prepare_round_transition(RequestId(1), &[TokenId(10)], None, &mut Rows)
        .unwrap();
    let (generation, effects) = round.commit(prepared);
    assert!(effects.operations.is_empty());
    assert_eq!(generation.resident_position(), 2);
    assert_eq!(generation.accepted_position(), 3);
    assert_eq!(generation.generated(), [TokenId(10)]);
    assert_eq!(generation.output_len(), 1);
    assert!(generation.method_checkpoint().is_ok());
    let decode = start(generation, 1);
    assert_eq!(decode.round_forward().kind, WorkKind::Decode);
    assert_eq!(decode.round_forward().tokens, [TokenId(10)]);
}

#[test]
fn cancellation_reconciles_the_round_without_committing_or_publishing_it() {
    let generation = start(generation(None), 2).cancel();
    assert_eq!(generation.resident_position(), 0);
    assert!(generation.generated().is_empty());
    assert_eq!(generation.finish_reason(), Some(FinishReason::Cancelled));
    assert!(generation.start_round(RequestId(1), 1).is_err());
}

#[test]
fn failure_discards_the_round_and_finishes_failed() {
    let generation = start(prefill(generation(None), 10), 1).fail();
    assert_eq!(generation.resident_position(), 2);
    assert_eq!(generation.generated(), [TokenId(10)]);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Failed));
}

#[test]
fn eviction_discards_suspended_work_and_replays_through_rounds() {
    let mut generation = prefill(generation(None), 10);
    generation.take(4).unwrap();
    let generation = start(generation, 1).evicted().unwrap();
    assert!(!generation.is_resident());
    let Err((mut generation, _)) = generation.start_round(RequestId(1), 1) else {
        panic!("a non-resident generation starts no round")
    };
    generation.resume_at(None).unwrap();
    for _ in 0..2 {
        let replay = start(generation, 1);
        assert_eq!(replay.round_forward().kind, WorkKind::Replay);
        (generation, _) = resolve(replay, &[], None);
    }
    assert_eq!(generation.resident_position(), 2);
    assert_eq!(generation.generated(), [TokenId(10)]);
    let decode = start(generation, 1);
    assert_eq!(decode.round_forward().kind, WorkKind::Decode);
}

/// Evicting a started round rewinds exactly as evicting its generation
/// between rounds: to the numerical prefix held before the round, keeping
/// accepted history and output.
#[test]
fn evicting_a_started_round_rewinds_to_its_numerical_prefix() {
    let between = {
        let mut generation = prefill(generation(None), 10);
        generation.evicted().unwrap();
        generation
    };
    let started = start(prefill(generation(None), 10), 1).evicted().unwrap();
    for generation in [&between, &started] {
        assert!(!generation.is_resident());
        assert_eq!(generation.resident_position(), 2);
        assert_eq!(generation.accepted_position(), 3);
        assert_eq!(generation.resume_bound(), 3);
        assert_eq!(generation.generated(), [TokenId(10)]);
        assert_eq!(generation.output_len(), 1);
        assert_eq!(generation.finish_reason(), None);
    }
    let mut generation = started;
    generation.resume_at(None).unwrap();
    // Replay stops at the two consumed prompt rows; the accepted successor
    // remains the next decode's input.
    let replay = start(generation, 4);
    assert_eq!(replay.round_forward().kind, WorkKind::Replay);
    assert_eq!(replay.round_forward().tokens, [TokenId(1), TokenId(2)]);
    let (generation, _) = resolve(replay, &[], None);
    assert_eq!(forward(&start(generation, 1)).tokens, [TokenId(10)]);
}

#[test]
fn eviction_resumes_from_a_retained_prefix_and_replays_the_rest() {
    let mut generation = prefill(generation(None), 10);
    let checkpoint = generation.method_checkpoint().unwrap();
    generation.evicted().unwrap();
    // Accepted input includes the sampled successor, while the numerical
    // prefix still ends at the two consumed prompt rows. The row the next
    // decode samples from bounds a resumed prefix.
    assert_eq!(generation.resume_bound(), 3);
    assert!(generation.resume_at(Some((3, &checkpoint))).is_err());
    generation.resume_at(Some((1, &checkpoint))).unwrap();
    assert!(generation.resume_at(Some((1, &checkpoint))).is_err());
    assert_eq!(generation.resident_position(), 1);
    assert_eq!(generation.detailed_usage().cached_tokens, 0);
    let replay = start(generation, 4);
    assert_eq!(replay.round_forward().kind, WorkKind::Replay);
    assert_eq!(replay.round_forward().tokens, [TokenId(2)]);
    let (generation, _) = resolve(replay, &[], None);
    assert_eq!(generation.resident_position(), 2);
    assert_eq!(generation.generated(), [TokenId(10)]);
    let decode = start(generation, 1);
    assert_eq!(decode.round_forward().kind, WorkKind::Decode);
}

/// An admitted MTP generation that is not yet resident.
fn mtp_fresh(prompt: &[u32], proposals: u8) -> Generation {
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

fn mtp_generation(prompt: &[u32], proposals: u8) -> Generation {
    resident(mtp_fresh(prompt, proposals))
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
    let generation = mtp_generation(&[1, 2, 3], 2);
    // First chunk: rows 1, 2. Its pairs are (2, f5.0); f5.1 waits for 3.
    let (mut generation, effects) = resolve(start(generation, 2), &[], Some(feature(5)));
    let [head] = effects.as_slice() else {
        panic!("a non-final chunk enters its complete pairs")
    };
    assert!(matches!(
        head,
        Operation::Head {
            phase: magnitude_executor::HeadPhase::Priming { .. },
            ..
        }
    ));
    let (tokens, rows, position, proposals) = head_parts(head);
    assert_eq!(
        (tokens, rows, position),
        (vec![TokenId(2)], vec![(5, 0)], 0)
    );
    assert!(proposals.is_empty());
    reconcile_head(&mut generation, head, &[]);
    // Final chunk: row 3 selects 10. Pairs (3, f5.1) are entered; the anchor
    // (10, f6.0) waits for the first draft.
    let (mut generation, effects) = resolve(start(generation, 1), &[10], Some(feature(6)));
    let [head] = effects.as_slice() else {
        panic!("the final chunk enters all but the anchor")
    };
    assert!(matches!(
        head,
        Operation::Head {
            phase: magnitude_executor::HeadPhase::Priming { .. },
            ..
        }
    ));
    let (tokens, rows, position, _) = head_parts(head);
    assert_eq!(
        (tokens, rows, position),
        (vec![TokenId(3)], vec![(5, 1)], 1)
    );
    reconcile_head(&mut generation, head, &[]);
    generation.take(4).unwrap();
    let (_, draft) = draft(generation, 4);
    let (tokens, rows, position, proposals) = head_parts(&draft[0]);
    assert_eq!(
        (tokens, rows, position),
        (vec![TokenId(10)], vec![(6, 0)], 2)
    );
    // Proposal selections share the target's keys at the same output rows.
    assert_eq!(
        proposals
            .iter()
            .map(|select| (select.position, select.domain))
            .collect::<Vec<_>>(),
        [(1, 0), (2, 0)]
    );
}

/// An MTP generation after its prefill, with its first draft started.
fn mtp_drafting(proposals: u8) -> (Generation, Operation) {
    let generation = mtp_generation(&[1, 2], proposals);
    let (mut generation, effects) = resolve(start(generation, 2), &[10], Some(feature(1)));
    reconcile_head(&mut generation, &effects[0], &[]);
    generation.take(4).unwrap();
    let (generation, mut draft) = draft(generation, 4);
    (generation, draft.remove(0))
}

#[test]
fn verification_accepts_the_matching_prefix_and_re_enters_accepted_rows() {
    let (mut generation, first) = mtp_drafting(3);
    reconcile_head(&mut generation, &first, &[(11, 0), (12, 0), (13, 0)]);
    let verify = start(generation, 4);
    assert_eq!(verify.round_forward().kind, WorkKind::Verify);
    assert_eq!(
        verify.round_forward().tokens,
        [TokenId(10), TokenId(11), TokenId(12), TokenId(13)]
    );
    // The target agrees on 11 and 12 and samples 20 after them.
    let (mut generation, effects) = resolve(verify, &[11, 12, 20, 30], Some(feature(2)));
    assert!(effects.is_empty());
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
    let (_, next) = draft(generation, 4);
    let (tokens, rows, position, _) = head_parts(&next[0]);
    assert_eq!(tokens, [TokenId(11), TokenId(12), TokenId(20)]);
    assert_eq!(rows, [(2, 0), (2, 1), (2, 2)]);
    assert_eq!(position, 2);
}

#[test]
fn proposals_stop_at_a_failed_selection_or_a_stop_token() {
    let (mut generation, first) = mtp_drafting(3);
    reconcile_head(&mut generation, &first, &[(11, 0), (99, 0), (13, 0)]);
    assert_eq!(
        start(generation, 4).round_forward().tokens,
        [TokenId(10), TokenId(11)]
    );

    let (mut failed, first) = mtp_drafting(3);
    reconcile_head(&mut failed, &first, &[(11, 0), (0, 1), (13, 0)]);
    assert_eq!(
        start(failed, 4).round_forward().tokens,
        [TokenId(10), TokenId(11)]
    );
}

#[test]
fn checkpoints_carry_host_rows_and_restore_at_their_target_boundary() {
    let source = mtp_generation(&[1, 2], 2);
    let (mut source, effects) = resolve(start(source, 2), &[10], Some(feature(3)));
    // A checkpoint needs reconciled method work.
    assert!(source.method_checkpoint().is_err());
    reconcile_head(&mut source, &effects[0], &[]);
    let checkpoint = source.method_checkpoint().unwrap();
    assert_eq!(checkpoint.retained_bytes(), 4);
    let MethodCheckpoint::Mtp(state) = &checkpoint else {
        panic!("MTP checkpoint")
    };
    // Head row 0 entered, the anchor pending: two target rows.
    assert_eq!((state.position(), state.target_rows()), (1, 2));
    let mut fresh = mtp_fresh(&[1, 2, 7], 2);
    assert!(fresh.resume_at(Some((1, &checkpoint))).is_err());
    fresh.resume_at(Some((2, &checkpoint))).unwrap();
    assert_eq!(fresh.resident_position(), 2);
}

#[test]
fn mtp_choice_requires_an_injected_factory() {
    let mut configured = options();
    configured.method = MethodChoice::Mtp { proposals: 2 };
    assert!(Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        configured,
        None,
    )
    .is_err());
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
    fn priming_position(&self) -> Option<usize> {
        None
    }
    fn prime(
        &mut self,
        _: RequestId,
        tokens: &[TokenId],
        next: Option<TokenId>,
        _features: FeatureRef,
        _draft_from: usize,
        _primed: usize,
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
    generation.resume_at(None).unwrap();

    let (generation, _) = resolve(start(generation, 1), &[], Some(feature(1)));
    resolve(start(generation, 1), &[10], Some(feature(2)));
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
    let source = resident(
        Generation::new(
            prompt.clone(),
            InputLayout::new(prompt.len(), vec![]).unwrap(),
            configured.clone(),
            Some(grammar()),
        )
        .unwrap(),
    );
    let round = start(source, prompt.len());
    assert_eq!(round.round_forward().tokens, prompt);
    let (source, _) = resolve(round, &[70], None);
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
    fresh.resume_at(Some((64, &checkpoint))).unwrap();
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
    assert!(fresh
        .resume_at(Some((64, &MethodCheckpoint::Plain)))
        .is_err());
    assert!(!fresh.is_resident());
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
    generation.resume_at(None).unwrap();
    let round = start(generation, 64);
    let transition = round
        .prepare_round_transition(RequestId(1), &[TokenId(70)], None, &mut Rows)
        .unwrap();
    assert_eq!(transition.decision().accepted_rows, 64);
    let (generation, _) = round.commit(transition);
    assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
    assert_eq!(generation.pending_reconciliation().unwrap().end, 65);
    let Ok(reconciliation) = generation.start_reconciliation(1) else {
        panic!("pending reconciliation starts")
    };
    let forward = reconciliation.round_forward();
    assert_eq!(forward.kind, WorkKind::Replay);
    assert_eq!(forward.tokens, [TokenId(70)]);
    assert!(forward.selects.is_empty());
    assert_eq!(reconciliation.round_publication_bound(), 0);
    let transition = reconciliation
        .prepare_round_transition(RequestId(1), &[], None, &mut Rows)
        .unwrap();
    assert_eq!(transition.decision().accepted_rows, 1);
    let (generation, _) = reconciliation.commit(transition);
    assert_eq!(generation.resident_position(), 65);
    assert!(generation.pending_reconciliation().is_none());
    assert_eq!(generation.generated(), [TokenId(70)]);
    assert_eq!(generation.output_len(), 1);
}

#[test]
fn ignored_end_of_generation_masks_stop_tokens_from_every_selection() {
    let mut configured = options();
    configured.end_of_generation = EndOfGeneration::Suppress;
    let mut generation = Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        configured,
        None,
    )
    .unwrap();
    generation.resume_at(None).unwrap();
    let round = start(generation, 2);
    let mask = round.round_forward().selects[0]
        .mask
        .clone()
        .expect("suppression masks selection");
    assert_eq!(mask[99 / 32] & (1 << (99 % 32)), 0);
    assert_ne!(mask[98 / 32] & (1 << (98 % 32)), 0);
    assert!(round
        .prepare_round_transition(RequestId(1), &[TokenId(99)], None, &mut Rows)
        .is_err());
}

#[test]
fn spent_reasoning_budget_selects_only_the_end_tag() {
    let mut configured = options();
    configured.reasoning_budget = Some(ReasoningBudget {
        tokens: 1,
        start: vec![TokenId(10)],
        end: vec![TokenId(11)],
        open: true,
    });
    let mut generation = Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        configured,
        None,
    )
    .unwrap();
    generation.resume_at(None).unwrap();
    let generation = prefill(generation, 5);
    assert_eq!(
        generation.selection_mask().unwrap().unwrap()[0],
        1 << 11,
        "after one reasoning token only the end tag is selectable"
    );
}

/// An admitted separate-draft generation that is not yet resident.
fn dflash_fresh(prompt: &[u32], proposals: u8) -> Generation {
    let mut configured = options();
    configured.method = MethodChoice::DFlash { proposals };
    Generation::new_with_method(
        prompt.iter().copied().map(TokenId).collect(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        configured,
        None,
        Arc::new(DFlash::new("fixture", usize::from(proposals)).unwrap()),
    )
    .unwrap()
}

fn dflash_generation(prompt: &[u32], proposals: u8) -> Generation {
    resident(dflash_fresh(prompt, proposals))
}

/// A separate draft's generation after its prefill: the prefill's entry
/// transaction reconciled and the first block draft started.
fn dflash_drafting(proposals: u8) -> (Generation, Operation) {
    let generation = dflash_generation(&[1, 2], proposals);
    let (mut generation, effects) = resolve(start(generation, 2), &[10], Some(feature(1)));
    reconcile_head(&mut generation, &effects[0], &[]);
    generation.take(4).unwrap();
    let (generation, draft) = draft(generation, 4);
    let [draft] = <[Operation; 1]>::try_from(draft).unwrap();
    let Operation::Head { form, phase, .. } = &draft else {
        panic!("a draft is a head transaction")
    };
    assert_eq!(*form, magnitude_executor::DraftForm::Block);
    assert_eq!(*phase, magnitude_executor::HeadPhase::Generation);
    (generation, draft)
}

/// Zero, partial and full acceptance of a separate draft's block: the
/// verified prefix plus the target's bonus token is published, and the next
/// block draft enters exactly the accepted rows with their target features
/// before anchoring on the bonus token.
#[test]
fn dflash_verification_publishes_the_accepted_prefix_and_re_enters_it() {
    for (samples, accepted, entered) in [
        // Zero: the target rejects 11 and samples 21.
        (vec![21, 30, 31, 32], 0, vec![21]),
        // Partial: 11 and 12 agree; 20 follows them.
        (vec![11, 12, 20, 30], 2, vec![11, 12, 20]),
        // Full: every proposal agrees; 40 is the bonus token.
        (vec![11, 12, 13, 40], 3, vec![11, 12, 13, 40]),
    ] {
        let (mut generation, first) = dflash_drafting(3);
        reconcile_head(&mut generation, &first, &[(11, 0), (12, 0), (13, 0)]);
        let verify = start(generation, 4);
        assert_eq!(verify.round_forward().kind, WorkKind::Verify);
        assert_eq!(
            verify.round_forward().tokens,
            [TokenId(10), TokenId(11), TokenId(12), TokenId(13)]
        );
        let (mut generation, effects) = resolve(verify, &samples, Some(feature(2)));
        assert!(effects.is_empty());
        let mut generated = vec![TokenId(10)];
        generated.extend(entered.iter().copied().map(TokenId));
        assert_eq!(generation.generated(), generated, "{accepted} accepted");
        assert_eq!(generation.resident_position(), 2 + accepted + 1);
        assert_eq!(generation.detailed_usage().draft_n, 3);
        assert_eq!(generation.detailed_usage().draft_n_accepted, accepted);
        generation.take(8).unwrap();
        let (_, next) = draft(generation, 4);
        let (tokens, rows, position, _) = head_parts(&next[0]);
        assert_eq!(
            tokens,
            entered.iter().copied().map(TokenId).collect::<Vec<_>>()
        );
        assert_eq!(
            rows,
            (0..=accepted as u8).map(|row| (2, row)).collect::<Vec<_>>()
        );
        assert_eq!(position, 2);
    }
}

/// A separate draft's checkpoint is its own kind, is refused while a draft
/// transaction is unreconciled, restores at its target boundary, and never
/// restores another method's state.
#[test]
fn dflash_checkpoints_wait_for_reconciled_drafts_and_keep_their_kind() {
    let source = dflash_generation(&[1, 2], 2);
    let (mut source, effects) = resolve(start(source, 2), &[10], Some(feature(3)));
    assert!(source.method_checkpoint().is_err());
    reconcile_head(&mut source, &effects[0], &[]);
    let checkpoint = source.method_checkpoint().unwrap();
    let MethodCheckpoint::DFlash(state) = &checkpoint else {
        panic!("DFlash checkpoint")
    };
    assert_eq!((state.position(), state.target_rows()), (1, 2));
    let mut fresh = dflash_fresh(&[1, 2, 7], 2);
    assert!(fresh.resume_at(Some((1, &checkpoint))).is_err());
    fresh.resume_at(Some((2, &checkpoint))).unwrap();
    assert_eq!(fresh.resident_position(), 2);
    // An MTP generation never adopts a separate draft's state.
    let mut mtp = mtp_fresh(&[1, 2, 7], 2);
    assert!(mtp.resume_at(Some((2, &checkpoint))).is_err());
}

/// Cancelling while a block draft is in flight finishes the generation
/// without publishing its proposals.
#[test]
fn dflash_cancellation_discards_the_in_flight_draft() {
    let (mut generation, _draft) = dflash_drafting(3);
    generation.cancel();
    assert_eq!(generation.finish_reason(), Some(FinishReason::Cancelled));
    assert_eq!(generation.generated(), [TokenId(10)]);
    assert_eq!(generation.detailed_usage().draft_n, 0);
}

/// The bound computed before a round starts covers the started round's bound:
/// exactly for replay, prompt chunks and forced runs, and as the proposal
/// width for decode and verification, whose method may propose fewer.
#[test]
fn publication_bound_before_start_covers_the_started_round() {
    fn bounds(generation: Generation, allowance: usize) -> (usize, usize, Generation) {
        let before = generation.publication_bound(allowance).unwrap();
        let round = start(generation, allowance);
        let started = round.round_publication_bound();
        let samples = vec![10; round.round_forward().selects.len()];
        let (generation, _) = resolve(round, &samples, None);
        (before, started, generation)
    }

    // Plain: a prompt chunk that does not end the prompt, the chunk that
    // selects the first token, then a decode.
    let (before, started, generation) = bounds(generation(None), 1);
    assert_eq!((before, started), (0, 0));
    let (before, started, mut generation) = bounds(generation, 1);
    assert_eq!((before, started), (1, 1));
    generation.take(4).unwrap();
    let (before, started, mut generation) = bounds(generation, 4);
    assert_eq!((before, started), (1, 1));

    // Replay after eviction publishes nothing.
    generation.evicted().unwrap();
    generation.resume_at(None).unwrap();
    let (before, started, _) = bounds(generation, 4);
    assert_eq!((before, started), (0, 0));

    // Forced: the first token is forced by the final prompt chunk, the rest
    // by one forced run.
    let forced = generation_with(Grammar {
        accepted: vec![],
        forced: vec![TokenId(10), TokenId(11), TokenId(12)],
        reject: None,
    });
    let (before, started, mut forced) = bounds(forced, 2);
    assert_eq!((before, started), (1, 1));
    forced.take(4).unwrap();
    let (before, started, _) = bounds(forced, 4);
    assert_eq!((before, started), (2, 2));

    // Speculative: drafter work starts no target round; the verification of
    // a full proposal publishes at most its width, and a proposal cut at a
    // stop token publishes less.
    for (proposals, width) in [
        (vec![(11, 0), (12, 0), (13, 0)], 4),
        (vec![(11, 0), (99, 0), (13, 0)], 2),
    ] {
        let (mut generation, first) = mtp_drafting(3);
        reconcile_head(&mut generation, &first, &proposals);
        let before = generation.publication_bound(4).unwrap();
        let round = start(generation, 4);
        assert_eq!((before, round.round_publication_bound()), (4, width));
    }
}

fn generation_with(grammar: Grammar) -> Generation {
    generation(Some(Box::new(grammar)))
}
