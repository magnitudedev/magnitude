use magnitude_generation::{Generation, MethodEffects, PreparedGenerationTransition, StartedRound};
use magnitude_executor::{
    ConditioningRef, DomainError, ExecutorDomain, FeatureRef, Operation, Outcome,
    PendingOperationOutcome, PhysicalDecision, ProgramFamily, RequestId, RowResult,
};

pub enum RoundError {
    Logical(String),
    Physical(DomainError),
}

impl std::fmt::Display for RoundError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Logical(error) => formatter.write_str(error),
            Self::Physical(error) => error.fmt(formatter),
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
    let accepted_rows = transition.decision().accepted_rows;
    if let Err(error) = domain.reconcile(pending, PhysicalDecision { accepted_rows }) {
        return Err((round.fail(), RoundError::Physical(error)));
    }
    Ok(round.commit(transition))
}

fn stage_forward<F: ProgramFamily>(
    round: &StartedRound,
    domain: &mut ExecutorDomain<F>,
    request: RequestId,
    pending: &PendingOperationOutcome,
) -> Result<PreparedGenerationTransition, String> {
    let forward = round.round_forward();
    let expected_rows = forward.tokens.len();
    let features_required = forward
        .demand
        .contains(magnitude_executor::Demand::FEATURES);
    let Outcome::Forward { rows } = pending.outcome() else {
        return Err("target round returned a non-forward outcome".into());
    };
    if rows.len() != expected_rows {
        return Err("target round returned the wrong number of rows".into());
    }
    let samples = if !forward.selects.is_empty() {
        let selected_rows = rows
            .iter()
            .filter(|row| row.selected.is_some())
            .collect::<Vec<_>>();
        if selected_rows.len() != forward.selects.len() {
            return Err("target round selection count differs from its rows".into());
        }
        selected_rows
            .into_iter()
            .map(|row| match row.selected {
                Some(selected) if selected.status == 0 => Ok(selected.token),
                Some(selected) if selected.status == 1 => Err("sampling distribution is empty"),
                Some(selected) if selected.status == 2 => Err("sampling distribution is nonfinite"),
                Some(_) => Err("sampling returned an unknown status"),
                None => Err("target verification row has no selected token"),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(str::to_owned)?
    } else {
        if rows.iter().any(|row| row.selected.is_some()) {
            return Err("non-selecting target round returned an unexpected sample".into());
        }
        Vec::new()
    };
    if std::env::var_os("MAGNITUDE_TRACE_TARGET").is_some() {
        eprintln!(
            "target round {:?} input_rows={} selections={} samples={:?}",
            forward.kind,
            rows.len(),
            forward.selects.len(),
            samples
        );
    }
    let features = common_features(&rows, features_required)?;
    round.prepare_round_transition(request, &samples, features, domain)
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
