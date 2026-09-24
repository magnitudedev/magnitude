use magnitude_generation::{Generation, MethodEffects, PreparedGenerationTransition, RoundForward};
use magnitude_model_executor::{
    ConditioningRef, DomainError, ExecutorDomain, FeatureRef, Operation, Outcome,
    PendingOperationOutcome, PhysicalDecision, PhysicalResolution, ProgramFamily, RequestId,
    RowResult,
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

pub enum RoundReconcile {
    Committed {
        effects: MethodEffects,
    },
    Repair {
        operation: Operation,
        transition: PreparedGenerationTransition,
    },
}

pub fn lower_round(
    generation: &Generation,
    request: RequestId,
    numerical_position: usize,
    conditioning: Option<ConditioningRef>,
) -> Result<Operation, String> {
    generation
        .round_forward()
        .ok_or_else(|| "generation has no suspended target round".to_owned())?
        .clone()
        .into_operation(request, numerical_position, conditioning)
}

pub fn reconcile_forward<F: ProgramFamily>(
    generation: &mut Generation,
    domain: &mut ExecutorDomain<F>,
    request: RequestId,
    pending: PendingOperationOutcome,
) -> Result<RoundReconcile, RoundError> {
    let staged = stage_forward(generation, domain, request, &pending);
    let transition = match staged {
        Ok(transition) => transition,
        Err(error) => {
            domain.abort(pending).map_err(RoundError::Physical)?;
            return Err(RoundError::Logical(error));
        }
    };
    let committed_rows = transition.decision().accepted_rows;
    match domain
        .reconcile(
            pending,
            PhysicalDecision {
                accepted_rows: committed_rows,
            },
        )
        .map_err(RoundError::Physical)?
    {
        PhysicalResolution::Committed => {
            let effects = generation.commit_transition(transition);
            Ok(RoundReconcile::Committed { effects })
        }
        PhysicalResolution::Repair {
            request: repair_request,
            rows,
        } if repair_request == request && rows == committed_rows => Ok(RoundReconcile::Repair {
            operation: Operation::Repair { request, rows },
            transition,
        }),
        PhysicalResolution::Repair { .. } => Err(RoundError::Logical(
            "executor returned an invalid prefix repair".into(),
        )),
    }
}

fn stage_forward<F: ProgramFamily>(
    generation: &Generation,
    domain: &mut ExecutorDomain<F>,
    request: RequestId,
    pending: &PendingOperationOutcome,
) -> Result<PreparedGenerationTransition, String> {
    let expected_rows = generation
        .round_forward()
        .map(|forward| forward.tokens.len())
        .ok_or("generation has no suspended target round")?;
    let forward = generation
        .round_forward()
        .ok_or("generation has no suspended target round")?;
    let features_required = generation.round_forward().is_some_and(|forward| {
        forward
            .demand
            .contains(magnitude_model_executor::Demand::FEATURES)
    });
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
    generation.prepare_round_transition(request, &samples, features, domain)
}

pub fn reconcile_repair(
    generation: &mut Generation,
    request: RequestId,
    transition: PreparedGenerationTransition,
) -> Result<MethodEffects, String> {
    if transition.request() != request {
        return Err("prefix repair belongs to another prepared request".into());
    }
    Ok(generation.commit_transition(transition))
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

pub fn round_forward(generation: &Generation) -> Option<&RoundForward> {
    generation.round_forward()
}
