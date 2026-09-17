//! Dimension-checked service constraints. A service pool may be shared by several
//! operation classes; their service times add within that pool, not across pools.

use crate::quantity::Count;
use seismic_lang::types::DType;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Add,
    Multiply,
    Divide,
    Fma,
    Compare,
    Convert { from: DType },
    Exp,
    ExpFast,
    Log,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Min,
    Max,
    Abs,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    Bytes,
    Operations { operation: Operation, dtype: DType },
    Instructions { intrinsic: String },
}

/// Identity includes scope and direction (when the path distinguishes direction).
/// It is supplied by a backend/profile, not inferred from a label like "memory".
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Service {
    pub pool: String,
    pub unit: Unit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RateEvidence {
    /// A justified maximum under stated operating conditions; not a sampled maximum.
    CapacityBound { source: String, conditions: String },
    /// Sustainable service observed in an identified probe/conditioning environment.
    Calibration { probe: String, environment: String },
}

#[derive(Clone, Debug)]
pub struct Rate {
    pub service: Service,
    per_second: f64,
    pub evidence: RateEvidence,
}

impl Rate {
    pub fn new(service: Service, per_second: f64, evidence: RateEvidence) -> Result<Self, String> {
        if !per_second.is_finite() || per_second <= 0.0 {
            return Err("service rate must be finite and positive".into());
        }
        if service.pool.is_empty() {
            return Err("service rate needs a resource pool identity".into());
        }
        let identified = match &evidence {
            RateEvidence::CapacityBound { source, conditions } => {
                !source.is_empty() && !conditions.is_empty()
            }
            RateEvidence::Calibration { probe, environment } => {
                !probe.is_empty() && !environment.is_empty()
            }
        };
        if !identified {
            return Err("service rate needs provenance and applicability conditions".into());
        }
        Ok(Self {
            service,
            per_second,
            evidence,
        })
    }

    pub fn per_second(&self) -> f64 {
        self.per_second
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemandMeaning {
    /// A necessary obligation under an explicit semantic/algorithmic condition.
    Obligation { justification: String },
    /// Consumption of one selected realization, not a universal minimum.
    Realization,
}

#[derive(Clone, Debug)]
pub struct Demand {
    pub service: Service,
    pub count: Count,
    pub meaning: DemandMeaning,
    pub origin: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceTime {
    pool: String,
    lower_seconds: f64,
    upper_seconds: f64,
    /// True only when both demand and supply justify a latency floor.
    justified_floor: bool,
    origin: String,
}

impl ServiceTime {
    pub fn pool(&self) -> &str {
        &self.pool
    }
    pub fn seconds(&self) -> (f64, f64) {
        (self.lower_seconds, self.upper_seconds)
    }
    pub fn is_justified_floor(&self) -> bool {
        self.justified_floor
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
}

pub fn service_time(demand: &Demand, rate: &Rate) -> Result<ServiceTime, String> {
    if demand.service != rate.service {
        return Err(format!(
            "incompatible demand/rate units or resource scope: {:?} versus {:?}",
            demand.service, rate.service
        ));
    }
    let (lower, upper) = demand.count.bounds().ok_or_else(|| {
        format!(
            "unavailable demand at {}: {:?}",
            demand.origin, demand.count
        )
    })?;
    let justified_floor = match (&demand.meaning, &rate.evidence) {
        (DemandMeaning::Obligation { justification }, RateEvidence::CapacityBound { .. }) => {
            if justification.is_empty() {
                return Err("necessary demand requires a justification".into());
            }
            true
        }
        _ => false,
    };
    let lower_seconds = lower as f64 / rate.per_second;
    let upper_seconds = upper as f64 / rate.per_second;
    if !lower_seconds.is_finite() || !upper_seconds.is_finite() {
        return Err("service duration exceeds finite numeric range".into());
    }
    Ok(ServiceTime {
        pool: demand.service.pool.clone(),
        lower_seconds,
        upper_seconds,
        justified_floor,
        origin: demand.origin.clone(),
    })
}

/// A partial floor from independently justified constraints. This does not infer
/// overlap, sequential stages or a complete kernel roofline from flat demand terms.
pub fn partial_floor(terms: &[ServiceTime]) -> Option<f64> {
    terms
        .iter()
        .filter(|t| t.justified_floor)
        .map(|t| t.lower_seconds)
        .reduce(f64::max)
}

/// Explicit stage assumption: each pool serializes its listed service demands, while
/// different pools can overlap fully. Call only when that assumption is applicable.
/// Mixed-operation rates must be calibrated for the same sharing regime by the caller.
pub fn overlapped_stage(terms: &[ServiceTime]) -> Result<(f64, f64), String> {
    let mut pools: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for term in terms {
        let total = pools.entry(&term.pool).or_default();
        total.0 += term.lower_seconds;
        total.1 += term.upper_seconds;
        if !total.0.is_finite() || !total.1.is_finite() {
            return Err("stage service duration exceeds finite numeric range".into());
        }
    }
    Ok(pools
        .values()
        .fold((0.0_f64, 0.0_f64), |a, b| (a.0.max(b.0), a.1.max(b.1))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> Service {
        Service {
            pool: "device-memory/read".into(),
            unit: Unit::Bytes,
        }
    }
    fn bound() -> RateEvidence {
        RateEvidence::CapacityBound {
            source: "test specification".into(),
            conditions: "fixed clock".into(),
        }
    }
    fn demand() -> Demand {
        Demand {
            service: memory(),
            count: Count::Exact(1024),
            origin: "input region".into(),
            meaning: DemandMeaning::Obligation {
                justification: "cold input must cross this boundary".into(),
            },
        }
    }

    #[test]
    fn measured_bandwidth_does_not_create_a_roofline_bound() {
        let rate = Rate::new(
            memory(),
            512.0,
            RateEvidence::Calibration {
                probe: "stream-read-v1".into(),
                environment: "test device, uncached".into(),
            },
        )
        .unwrap();
        let time = service_time(&demand(), &rate).unwrap();
        assert_eq!(time.lower_seconds, 2.0);
        assert_eq!(partial_floor(&[time]), None);
        let rate = Rate::new(memory(), 512.0, bound()).unwrap();
        assert_eq!(
            partial_floor(&[service_time(&demand(), &rate).unwrap()]),
            Some(2.0)
        );
    }

    #[test]
    fn realized_work_is_not_a_necessary_obligation() {
        let mut d = demand();
        d.meaning = DemandMeaning::Realization;
        let t = service_time(&d, &Rate::new(memory(), 512.0, bound()).unwrap()).unwrap();
        assert!(!t.justified_floor);
    }

    #[test]
    fn rejects_wrong_precision_path_and_unavailable_counts() {
        let rate = Rate::new(memory(), 512.0, bound()).unwrap();
        let mut d = demand();
        d.service.pool = "shared-memory/read".into();
        assert!(service_time(&d, &rate).is_err());
        d.service = memory();
        d.service.unit = Unit::Operations {
            operation: Operation::Exp,
            dtype: DType::F32,
        };
        assert!(service_time(&d, &rate).is_err());
        d.service = memory();
        d.count = Count::unknown("runtime-visible history");
        assert!(service_time(&d, &rate).is_err());
    }

    #[test]
    fn shared_service_adds_before_overlap() {
        let term = |pool: &str, n| ServiceTime {
            pool: pool.into(),
            lower_seconds: n,
            upper_seconds: n,
            justified_floor: false,
            origin: "test".into(),
        };
        assert_eq!(
            overlapped_stage(&[term("alu", 3.0), term("alu", 4.0), term("memory", 5.0)]).unwrap(),
            (7.0, 7.0)
        );
    }

    #[test]
    fn invalid_profile_values_fail() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(Rate::new(memory(), value, bound()).is_err());
        }
    }
}
