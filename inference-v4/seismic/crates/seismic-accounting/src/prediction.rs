//! Explicit service/concurrency/dependency prediction. Inputs must come from a
//! physical realization mapping and applicable calibration, not portable FLOP or
//! allocation-size counters. Ranges express input variation, not confidence bounds.
use crate::{
    quantity::Count,
    resource::{Demand, DemandMeaning, Rate, RateEvidence, Service},
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    lower: f64,
    upper: f64,
}
impl Range {
    pub fn new(lower: f64, upper: f64) -> Result<Self, String> {
        if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || upper < lower {
            return Err("invalid nonnegative finite prediction range".into());
        }
        Ok(Self { lower, upper })
    }
    pub fn exact(value: f64) -> Result<Self, String> {
        Self::new(value, value)
    }
    pub fn endpoints(self) -> (f64, f64) {
        (self.lower, self.upper)
    }
    fn add(self, other: Self) -> Result<Self, String> {
        Self::new(self.lower + other.lower, self.upper + other.upper)
    }
    fn max(self, other: Self) -> Self {
        Self {
            lower: self.lower.max(other.lower),
            upper: self.upper.max(other.upper),
        }
    }
}
#[derive(Clone, Debug)]
pub struct Evidence {
    pub source: String,
    pub conditions: String,
}
impl Evidence {
    fn validate(&self) -> Result<(), String> {
        if self.source.is_empty() || self.conditions.is_empty() {
            Err("prediction evidence needs a source and applicability conditions".into())
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug)]
pub struct Latency {
    pub seconds: Range,
    pub evidence: Evidence,
}
#[derive(Clone, Debug)]
pub struct Behavior {
    /// This empirical saturated service must match the demand's unit and pool.
    pub saturated: Rate,
    /// Optional latency-coverage model; outstanding service units must then be
    /// derived from independent resident work, not the allocation size.
    pub latency: Option<Latency>,
}
#[derive(Clone, Debug)]
pub struct PlannedService {
    pub demand: Demand,
    pub mapping: Evidence,
    pub outstanding: Option<(Count, Evidence)>,
}
#[derive(Clone, Debug)]
pub struct Stage {
    pub name: String,
    pub services: Vec<PlannedService>,
    /// Accounted dependency path within the stage, in seconds.
    pub dependency: Latency,
    /// Justifies ideal overlap of distinct pools. Terms sharing a pool add.
    pub overlap: Evidence,
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub identity: String,
    pub profile_identity: String,
    pub workload_identity: String,
    pub boundary_identity: String,
    /// Stages are separated by actual non-overlapping execution boundaries.
    pub stages: Vec<Stage>,
    pub sequencing: Evidence,
    /// Counted once for the modeled invocation boundary, not once per pool.
    pub submission: Latency,
    /// Material unmapped effects prohibit a complete duration/ranking.
    pub unavailable: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct Profile {
    pub identity: String,
    pub behaviors: Vec<Behavior>,
}
#[derive(Clone, Debug)]
pub struct Constraint {
    pub stage: String,
    pub pool: String,
    pub origin: String,
    pub effective_units_per_second: Range,
    pub service_seconds: Range,
    pub mapping: Evidence,
    pub rate_evidence: RateEvidence,
    pub latency_evidence: Option<Evidence>,
    pub outstanding_evidence: Option<Evidence>,
}
#[derive(Clone, Debug)]
pub struct StagePrediction {
    pub name: String,
    pub seconds: Option<Range>,
    pub pool_seconds: BTreeMap<String, Range>,
    pub dependency: Latency,
    pub overlap: Evidence,
}
#[derive(Clone, Debug)]
pub struct Prediction {
    pub plan_identity: String,
    pub profile_identity: String,
    pub workload_identity: String,
    pub boundary_identity: String,
    pub seconds: Option<Range>,
    pub stages: Vec<StagePrediction>,
    pub constraints: Vec<Constraint>,
    pub unavailable: Vec<String>,
    pub submission: Latency,
    pub sequencing: Evidence,
}
fn range(count: &Count) -> Result<Range, String> {
    let (lo, hi) = count
        .bounds()
        .ok_or_else(|| format!("unavailable count: {count:?}"))?;
    Range::new(lo as f64, hi as f64)
}
fn constraint(
    stage: &str,
    planned: &PlannedService,
    behavior: &Behavior,
) -> Result<Constraint, String> {
    planned.mapping.validate()?;
    if planned.demand.meaning != DemandMeaning::Realization {
        return Err(
            "prediction requires selected-realization demand, not a portable obligation".into(),
        );
    }
    if planned.demand.service != behavior.saturated.service {
        return Err("physical demand and calibrated service differ".into());
    }
    if !matches!(
        behavior.saturated.evidence,
        RateEvidence::Calibration { .. }
    ) {
        return Err("prediction needs applicable calibrated behavior; capacity alone is not sustainable service".into());
    }
    let demand = range(&planned.demand.count)?;
    let saturated = behavior.saturated.per_second();
    let mut rate = Range::exact(saturated)?;
    if let Some(latency) = &behavior.latency {
        latency.evidence.validate()?;
        if latency.seconds.lower <= 0.0 {
            return Err("service latency must be positive".into());
        }
        let (outstanding, evidence) = planned
            .outstanding
            .as_ref()
            .ok_or("latency coverage needs derived outstanding service units")?;
        evidence.validate()?;
        let outstanding = range(outstanding)?;
        rate = Range::new(
            saturated.min(outstanding.lower / latency.seconds.upper),
            saturated.min(outstanding.upper / latency.seconds.lower),
        )?;
    }
    let seconds = if demand.upper == 0.0 {
        Range::exact(0.0)?
    } else {
        if rate.lower <= 0.0 {
            return Err("nonzero demand has no positive proven modeled concurrency".into());
        }
        Range::new(demand.lower / rate.upper, demand.upper / rate.lower)?
    };
    Ok(Constraint {
        stage: stage.into(),
        pool: planned.demand.service.pool.clone(),
        origin: planned.demand.origin.clone(),
        effective_units_per_second: rate,
        service_seconds: seconds,
        mapping: planned.mapping.clone(),
        rate_evidence: behavior.saturated.evidence.clone(),
        latency_evidence: behavior.latency.as_ref().map(|l| l.evidence.clone()),
        outstanding_evidence: planned.outstanding.as_ref().map(|(_, e)| e.clone()),
    })
}
impl Profile {
    fn behavior(&self, service: &Service) -> Option<&Behavior> {
        self.behaviors
            .iter()
            .find(|b| &b.saturated.service == service)
    }
    fn validate(&self) -> Result<(), String> {
        if self.identity.is_empty() {
            return Err("profile identity missing".into());
        }
        for (i, behavior) in self.behaviors.iter().enumerate() {
            if self.behaviors[..i]
                .iter()
                .any(|b| b.saturated.service == behavior.saturated.service)
            {
                return Err("duplicate profile service".into());
            }
        }
        Ok(())
    }
}
pub fn predict(plan: &Plan, profile: &Profile) -> Result<Prediction, String> {
    profile.validate()?;
    plan.sequencing.validate()?;
    plan.submission.evidence.validate()?;
    if plan.identity.is_empty()
        || plan.workload_identity.is_empty()
        || plan.boundary_identity.is_empty()
        || plan.profile_identity != profile.identity
    {
        return Err("missing plan identity or incompatible profile identity".into());
    }
    let mut result = Prediction {
        plan_identity: plan.identity.clone(),
        profile_identity: profile.identity.clone(),
        workload_identity: plan.workload_identity.clone(),
        boundary_identity: plan.boundary_identity.clone(),
        seconds: None,
        stages: Vec::new(),
        constraints: Vec::new(),
        unavailable: plan.unavailable.clone(),
        submission: plan.submission.clone(),
        sequencing: plan.sequencing.clone(),
    };
    let mut total = plan.submission.seconds;
    for stage in &plan.stages {
        stage.overlap.validate()?;
        stage.dependency.evidence.validate()?;
        if stage.name.is_empty() {
            return Err("unnamed prediction stage".into());
        }
        let mut pools = BTreeMap::<String, Range>::new();
        let mut complete = true;
        for service in &stage.services {
            let term = profile
                .behavior(&service.demand.service)
                .ok_or_else(|| format!("missing calibration for {:?}", service.demand.service))
                .and_then(|behavior| constraint(&stage.name, service, behavior));
            match term {
                Ok(term) => {
                    let sum = pools
                        .get(&term.pool)
                        .copied()
                        .unwrap_or(Range::exact(0.0)?)
                        .add(term.service_seconds)?;
                    pools.insert(term.pool.clone(), sum);
                    result.constraints.push(term);
                }
                Err(error) => {
                    complete = false;
                    result.unavailable.push(format!(
                        "{} / {}: {error}",
                        stage.name, service.demand.origin
                    ));
                }
            }
        }
        let seconds = complete.then(|| {
            pools
                .values()
                .fold(stage.dependency.seconds, |a, b| a.max(*b))
        });
        if let Some(seconds) = seconds {
            total = total.add(seconds)?;
        }
        result.stages.push(StagePrediction {
            name: stage.name.clone(),
            seconds,
            pool_seconds: pools,
            dependency: stage.dependency.clone(),
            overlap: stage.overlap.clone(),
        });
    }
    if result.unavailable.is_empty() {
        result.seconds = Some(total)
    }
    Ok(result)
}
/// A model range dominates only when its upper endpoint is no larger than every
/// alternative's lower endpoint. Overlap requires better evidence, not a magic score.
/// This claim concerns the supplied finite set and this model, never global optimality.
pub fn unambiguous_choice(predictions: &[Prediction]) -> Result<Option<usize>, String> {
    let Some(first) = predictions.first() else {
        return Ok(None);
    };
    if predictions.iter().any(|p| {
        p.profile_identity != first.profile_identity
            || p.workload_identity != first.workload_identity
            || p.boundary_identity != first.boundary_identity
    }) {
        return Err("cannot compare incompatible profiles".into());
    }
    if predictions.iter().any(|p| p.seconds.is_none()) {
        return Ok(None);
    }
    for (i, candidate) in predictions.iter().enumerate() {
        let upper = candidate.seconds.unwrap().upper;
        if predictions
            .iter()
            .enumerate()
            .all(|(j, p)| i == j || upper <= p.seconds.unwrap().lower)
        {
            return Ok(Some(i));
        }
    }
    Ok(None)
}
