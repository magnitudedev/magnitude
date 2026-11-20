use magnitude_generation::{Generation, MethodEffects, PreparedGenerationTransition, StartedRound};
use magnitude_executor::{
    ConditioningRef, DomainError, ExecutorDomain, FeatureRef, Operation, Outcome,
    PendingOperationOutcome, ProgramFamily, RequestId, RowResult,
};

pub enum RoundError {
    Logical(String),
    Physical(DomainError),
    /// The round's step was interrupted (a late beat of its load's pulse),
    /// for this reason: its outcome was aborted and the generation returned
    /// unfinished, to run the round again (resident) or to re-prefill (its
    /// state released with a chain it was a chunk of). Not a failure.
    Interrupted(String),
}

impl std::fmt::Display for RoundError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Logical(error) => formatter.write_str(error),
            Self::Physical(error) => error.fmt(formatter),
            Self::Interrupted(reason) => formatter.write_str(reason),
        }
    }
}

impl std::fmt::Debug for RoundError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}

impl std::error::Error for RoundError {}

pub fn lower_round(
    round: &StartedRound,
    request: RequestId,
    numerical_position: usize,
    conditioning: Option<ConditioningRef>,
) -> Result<Operation, String> {
    round
        .round_forward()
        .clone()
        .into_operation(request, numerical_position, conditioning)
}

/// Reconcile a started round's physical outcome. Success commits the round
/// back into its generation; a failure aborts the outcome and fails the
/// round, returning the failed generation with the cause.
pub fn reconcile_forward<F: ProgramFamily>(
    round: StartedRound,
    domain: &mut ExecutorDomain<F>,
    request: RequestId,
    pending: PendingOperationOutcome,
) -> Result<(Generation, MethodEffects), (Generation, RoundError)> {
    // The round's step was interrupted: its outputs are discarded and its
    // accepted state returns; the round runs again from it.
    if let Outcome::Interrupted { reason } = pending.outcome() {
        let reason = reason.clone();
        if let Err(physical) = domain.abort(pending) {
            return Err((round.fail(), RoundError::Physical(physical)));
        }
        return Err((round.abandon(), RoundError::Interrupted(reason)));
    }
    // The sequence's table rows could not be read: its outputs are
    // discarded and the round fails with the typed cause.
    if let Outcome::WeightReadFailed(error) = pending.outcome() {
        let error = error.clone();
        let error = match domain.abort(pending) {
            Ok(()) => RoundError::Physical(DomainError::WeightRead(error)),
            Err(physical) => RoundError::Physical(physical),
        };
        return Err((round.fail(), error));
    }
    let transition = match stage_forward(&round, domain, request, &pending) {
        Ok(transition) => transition,
        Err(error) => {
            let error = match domain.abort(pending) {
                Ok(()) => RoundError::Logical(error),
                Err(physical) => RoundError::Physical(physical),
            };
            return Err((round.fail(), error));
        }
    };
    round
        .reconcile(transition, |decision| domain.reconcile(pending, decision))
        .map_err(|(round, error)| (round.fail(), RoundError::Physical(error)))
}

fn stage_forward<F: ProgramFamily>(
    round: &StartedRound,
    domain: &mut ExecutorDomain<F>,
    request: RequestId,
    pending: &PendingOperationOutcome,
) -> Result<PreparedGenerationTransition, String> {
    stage_outcome(round, domain, request, pending.outcome())
}

fn stage_outcome(
    round: &StartedRound,
    reader: &mut dyn magnitude_executor::FeatureReader,
    request: RequestId,
    outcome: &Outcome,
) -> Result<PreparedGenerationTransition, String> {
    let forward = round.round_forward();
    let features_required = forward
        .demand
        .contains(magnitude_executor::Demand::FEATURES);
    let Outcome::Forward { rows, guesses } = outcome else {
        return Err("target round returned a non-forward outcome".into());
    };
    let samples = forward.samples(rows)?;
    if std::env::var_os("MAGNITUDE_TRACE_TARGET").is_some() {
        eprintln!(
            "target round {:?} input_rows={} selections={} samples={:?}",
            forward.kind,
            rows.len(),
            forward.selects.len(),
            samples
        );
    }
    let features = common_features(rows, features_required)?;
    round.prepare_round_transition(request, &samples, guesses.clone(), features, reader)
}

/// A forward may attach one aggregate `[rows, D]` lease to exactly one row, or
/// repeat that same aggregate lease on every row. Partial repetition and
/// distinct per-row leases are ambiguous above the executor boundary.
fn common_features(rows: &[RowResult], required: bool) -> Result<Option<FeatureRef>, String> {
    let present = rows
        .iter()
        .filter_map(|row| row.features.clone())
        .collect::<Vec<_>>();
    if present.is_empty() {
        return if required {
            Err("feature-demanding forward returned no feature lease".into())
        } else {
            Ok(None)
        };
    }
    let first = present[0].clone();
    if present.iter().any(|feature| feature != &first)
        || (present.len() != 1 && present.len() != rows.len())
    {
        return Err("forward rows returned an ambiguous feature-lease layout".into());
    }
    Ok(Some(first))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{
        Demand, FeatureReader, FeatureRows, FeatureSpan, FrontGrant, Guess, GuessKind,
        ResourceDomainId, Sampling, Selected, Shaping, TokenId,
    };
    use magnitude_generation::{EndOfGeneration, InputLayout, MethodChoice, Mtp, Options, RoundStart};
    use std::collections::BTreeSet;
    use std::sync::Arc;

    struct DummyFeatures;

    impl FeatureReader for DummyFeatures {
        fn read(&mut self, span: &FeatureSpan) -> Result<FeatureRows, String> {
            let bytes = vec![0u8; span.count * 4];
            FeatureRows::new(bytes.into(), span.count).map_err(|error| error.to_string())
        }
    }

    fn lease(rows: usize) -> FeatureRef {
        FeatureRef::logical(ResourceDomainId::new("round-driver-test").unwrap(), rows, 4).unwrap()
    }

    fn selected(token: u32) -> Selected {
        Selected { token: TokenId(token), status: 0 }
    }

    #[test]
    fn chained_decode_uses_device_only_features_while_verify_requires_host_lease() {
        let options = Options {
            max_tokens: 16,
            output_capacity: 16,
            context_limit: 64,
            vocabulary: 100,
            stop_tokens: BTreeSet::from([TokenId(99)]),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping { temperature: 0.0, ..Default::default() },
            seed: 42,
            forced_quantum: 4,
            method: MethodChoice::Mtp { proposals: 3 },
            end_of_generation: EndOfGeneration::Stop,
            reasoning_budget: None,
        };
        let prompt = (1..=17).map(TokenId).collect::<Vec<_>>();
        let layout = InputLayout::new(17, vec![]).unwrap();
        let method = Arc::new(Mtp::new("fixture", 3).unwrap().fronting().unwrap());
        let mut generation = Generation::new_with_method(prompt, layout, options, None, method).unwrap();
        generation.resume_at(None).unwrap();

        let request = RequestId(1);
        let mut reader = DummyFeatures;

        let start = |generation: Generation, allowance: usize, grant: FrontGrant| match generation
            .start_round(request, allowance, grant)
        {
            Ok(RoundStart::Target(round)) => round,
            _ => panic!("expected target round"),
        };

        // 1. Closing prefill chunk demands host-read FEATURES.
        let closing = start(generation, 32, FrontGrant::UpTo(0));
        assert!(closing.round_forward().demand.contains(Demand::FEATURES));
        let mut prefill_rows = vec![RowResult::default(); 17];
        prefill_rows[16].selected = Some(selected(10));
        let no_lease = Outcome::Forward {
            rows: prefill_rows.clone(),
            guesses: vec![Guess { token: TokenId(11), kind: GuessKind::Head }],
        };
        assert_eq!(
            stage_outcome(&closing, &mut reader, request, &no_lease).err().as_deref(),
            Some("feature-demanding forward returned no feature lease"),
        );
        prefill_rows[0].features = Some(lease(17));
        let with_lease = Outcome::Forward {
            rows: prefill_rows,
            guesses: vec![Guess { token: TokenId(11), kind: GuessKind::Head }],
        };
        let transition = stage_outcome(&closing, &mut reader, request, &with_lease).unwrap();
        let (generation, _) = closing.commit(transition);

        // 2. Chained front-row decode demands DEVICE_FEATURES (not host FEATURES)
        //    and succeeds in stage_outcome with no host feature lease.
        let chained = start(generation, 4, FrontGrant::UpTo(1));
        assert_eq!(chained.round_forward().demand, Demand::SELECT | Demand::DEVICE_FEATURES);
        assert!(chained.round_forward().chain.is_some());
        lower_round(&chained, request, 17, None).unwrap();
        let chained_outcome = Outcome::Forward {
            rows: vec![RowResult { selected: Some(selected(11)), features: None, logits: None }],
            guesses: vec![Guess { token: TokenId(12), kind: GuessKind::Chained }],
        };
        let transition = stage_outcome(&chained, &mut reader, request, &chained_outcome)
            .expect("chained decode with device-only features needs no host lease");
        let (generation, _) = chained.commit(transition);

        // 3. Verify pass demands host-read FEATURES: missing lease fails,
        //    present lease succeeds.
        let verify = start(generation, 4, FrontGrant::Verify(1));
        assert_eq!(verify.round_forward().demand, Demand::SELECT | Demand::FEATURES);
        assert!(verify.round_forward().chain.is_none());
        let verify_no_lease = Outcome::Forward {
            rows: vec![
                RowResult { selected: Some(selected(12)), features: None, logits: None },
                RowResult { selected: Some(selected(13)), features: None, logits: None },
            ],
            guesses: Vec::new(),
        };
        assert_eq!(
            stage_outcome(&verify, &mut reader, request, &verify_no_lease).err().as_deref(),
            Some("feature-demanding forward returned no feature lease"),
        );
        let verify_with_lease = Outcome::Forward {
            rows: vec![
                RowResult { selected: Some(selected(12)), features: Some(lease(2)), logits: None },
                RowResult { selected: Some(selected(13)), features: None, logits: None },
            ],
            guesses: Vec::new(),
        };
        stage_outcome(&verify, &mut reader, request, &verify_with_lease)
            .expect("verify pass with host feature lease succeeds");
    }
}
