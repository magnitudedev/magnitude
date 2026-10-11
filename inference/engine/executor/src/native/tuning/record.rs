//! A unit's stored record, as its two readers take it.
//!
//! A search's result ([`seismic::TuningResult`]) holds every candidate and
//! every measurement the search made: megabytes for one unit of a model
//! whose search measured thousands of candidates. A load serves the unit's
//! choice and holds that choice to the points it serves; it needs nothing
//! of the other candidates. So a record is stored as its [`StoredChoice`],
//! which every load decodes, and its evidence, the result itself, which is
//! decoded only where it is stated or chosen from again
//! ([`crate::kernel_cache`]): by a tuning trace, on the trace's own thread,
//! and by a load under a numerical requirement other than the one the
//! record was searched under. A load that is asked for no trace, under the
//! requirement its records were searched under, never reads the evidence.

use super::PointShape;
use crate::kernel_cache::StoredEvidence;
use seismic::{
    Configuration, CoverageCheck, NumericalRequirement, SearchStop, TuningDisposition,
    TuningMethod, TuningResult,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// What a load needs of a unit's record: the choice, what it was chosen
/// under, how its search stood, and what the evidence holds of the choice
/// itself. Derived from the result when the record is published
/// ([`StoredChoice::of`]) and stored beside it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct StoredChoice {
    /// The chosen configuration, for the load to prepare.
    pub overall: Configuration,
    pub disposition: TuningDisposition,
    /// What `overall` was chosen under: a load under another requirement
    /// chooses again from the evidence ([`seismic::reselect`]).
    pub requirement: NumericalRequirement,
    /// Whether the result's search reached its own end.
    pub stop: Option<SearchStop>,
    pub coverage: Option<CoverageCheck>,
    /// Configurations the result records as measured.
    pub measured: usize,
    pub excluded: usize,
    /// Configurations rejected during formation or by the requirement.
    pub rejections: usize,
    /// The first rejection's configuration and reason.
    pub first_rejection: Option<String>,
    pub chosen: ChosenEvidence,
}

/// What a unit's evidence holds of its chosen configuration: what a load
/// holds the choice to before it serves it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ChosenEvidence {
    /// The evidence records the chosen configuration as excluded.
    pub excluded: bool,
    /// The chosen configuration's measurement, when it was measured.
    pub measured: Option<MeasuredChoice>,
    /// The search served it: its default, or a promotion that was timed,
    /// confirmed and covered.
    pub promoted: bool,
}

/// The measurements of a chosen configuration against the unit's default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct MeasuredChoice {
    /// The first constraint of the record's requirement its measurements
    /// violate, if any.
    pub violation: Option<String>,
    /// The points it holds a measurement at.
    pub observed: Vec<String>,
}

impl ChosenEvidence {
    pub(crate) fn of(result: &TuningResult) -> Self {
        let chosen = || {
            result
                .configurations
                .iter()
                .filter(|record| record.configuration == result.overall)
        };
        Self {
            excluded: chosen().any(|record| matches!(record.outcome, seismic::Outcome::Excluded(_))),
            measured: chosen()
                .find(|record| matches!(record.outcome, seismic::Outcome::Measured { .. }))
                .map(|record| MeasuredChoice {
                    violation: result
                        .admitted(record)
                        .err()
                        .map(|violation| violation.to_string()),
                    observed: record
                        .observations
                        .iter()
                        .map(|observation| result.observations[*observation].point.clone())
                        .collect(),
                }),
            promoted: result.served.contains(&result.overall),
        }
    }

    /// A recorded canonical fallback is usable without evidence of its own,
    /// unless the canonical artifact itself was rejected.
    pub(crate) fn usable_fallback(&self) -> Result<(), String> {
        if self.excluded {
            return Err("canonical artifact was rejected; an unqualified fallback cannot override a fidelity failure".into());
        }
        Ok(())
    }

    /// A qualified choice was measured, was served by its search, holds a
    /// measurement against the default at every one of `shapes`, and every
    /// one of its measurements is admitted by the requirement the record is
    /// served under.
    pub(crate) fn qualified_at(&self, shapes: &[PointShape]) -> Result<(), String> {
        let measured = self
            .measured
            .as_ref()
            .ok_or_else(|| "selected configuration was not measured".to_owned())?;
        if !self.promoted {
            return Err("selected configuration was never promoted".into());
        }
        if let Some(violation) = &measured.violation {
            return Err(violation.clone());
        }
        for shape in shapes {
            if !measured.observed.contains(&shape.label) {
                return Err(format!(
                    "selected artifact has no measurement against the native default at {}",
                    shape.label
                ));
            }
        }
        Ok(())
    }
}

impl StoredChoice {
    /// The choice of `result`, as it stands under the requirement `result`
    /// states.
    pub(crate) fn of(result: &TuningResult) -> Self {
        let measured = result
            .configurations
            .iter()
            .filter(|record| matches!(record.outcome, seismic::Outcome::Measured { .. }))
            .count();
        let first_rejection = result.rejections().next().map(|record| {
            let reason = match (&record.outcome, result.admitted(record)) {
                (seismic::Outcome::Excluded(exclusion), _) => format!("{exclusion:?}"),
                // A survey's record: measured and timed, and not admitted.
                (seismic::Outcome::Measured { .. }, Err(violation)) => violation.to_string(),
                (seismic::Outcome::Measured { .. }, Ok(())) => {
                    unreachable!("a rejection was not formed or is not admitted")
                }
            };
            format!("{:?}: {reason}", record.configuration.params)
        });
        Self {
            overall: result.overall.clone(),
            disposition: result.disposition.clone(),
            requirement: result.requirement.clone(),
            stop: match &result.method {
                TuningMethod::Search { complete, .. } => Some(if *complete {
                    SearchStop::Exhausted
                } else {
                    SearchStop::Expired
                }),
                TuningMethod::Survey { .. } | TuningMethod::Census => None,
            },
            coverage: result.coverage.clone(),
            measured,
            excluded: result.configurations.len() - measured,
            rejections: result.rejections().count(),
            first_rejection,
            chosen: ChosenEvidence::of(result),
        }
    }

    /// The choice of a unit whose remaining pieces did not fit the memory
    /// room when its load's optional work stopped: that, not the budget,
    /// is what stopped its search. The evidence cannot say so (the refusal
    /// is the allocator's, and a unit still at its census has no search to
    /// state a stop for), so the choice is told.
    pub(crate) fn without_room(self) -> Self {
        Self {
            stop: Some(SearchStop::NoRoom),
            ..self
        }
    }
}

/// The evidence of a unit's choice: every candidate and measurement of its
/// search. A search holds its own; a stored unit's stays in the store,
/// named and unread, until something states it.
#[derive(Clone, Debug, PartialEq)]
pub enum Receipt {
    Held(Arc<TuningResult>),
    Stored(StoredEvidence),
}

impl Receipt {
    /// The evidence, read from the store if that is where it is.
    pub fn read(&self) -> Result<Arc<TuningResult>, String> {
        match self {
            Self::Held(result) => Ok(result.clone()),
            Self::Stored(evidence) => evidence.read().map(Arc::new),
        }
    }
}
