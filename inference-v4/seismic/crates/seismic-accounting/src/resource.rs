//! Dimension-checked service estimates for diagnostics and model qualification.
//!
//! The floating-point values and descriptive provenance in this module establish
//! neither necessary demand nor a physical duration bound. Execution constraints
//! and their consistency checks belong to the analyses that derive them.
//! A service pool may be shared by several operation classes; the overlap estimate
//! assumes that their service times add within that pool, not across pools.

use crate::quantity::Count;
use seismic_lang::types::DType;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
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
#[derive(serde::Serialize, serde::Deserialize)]
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
pub enum RateSource {
    /// An assumed sustainable rate in a supplied model. The description does not
    /// establish that this rate applies to any physical device.
    Model { source: String, conditions: String },
    /// Service observed in an identified probe/conditioning environment. An
    /// observation does not establish a universal service ceiling.
    Calibration { probe: String, environment: String },
}

#[derive(Clone, Debug)]
pub struct Rate {
    pub service: Service,
    per_second: f64,
    pub source: RateSource,
}

impl Rate {
    pub fn new(service: Service, per_second: f64, source: RateSource) -> Result<Self, String> {
        if !per_second.is_finite() || per_second <= 0.0 {
            return Err("service rate must be finite and positive".into());
        }
        if service.pool.is_empty() {
            return Err("service rate needs a resource pool identity".into());
        }
        let identified = match &source {
            RateSource::Model { source, conditions } => {
                !source.is_empty() && !conditions.is_empty()
            }
            RateSource::Calibration { probe, environment } => {
                !probe.is_empty() && !environment.is_empty()
            }
        };
        if !identified {
            return Err("service estimate needs descriptive provenance and conditions".into());
        }
        Ok(Self {
            service,
            per_second,
            source,
        })
    }

    pub fn per_second(&self) -> f64 {
        self.per_second
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemandMeaning {
    /// Operations or accesses in the source computation; no physical mapping or
    /// necessity claim follows from this classification.
    Algorithm,
    /// Consumption attributed to one selected realization by a supplied mapping.
    /// This label does not verify that mapping or establish a universal minimum.
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
pub struct ServiceEstimate {
    pool: String,
    lower_seconds: f64,
    upper_seconds: f64,
    origin: String,
}

impl ServiceEstimate {
    pub fn pool(&self) -> &str {
        &self.pool
    }
    pub fn seconds(&self) -> (f64, f64) {
        (self.lower_seconds, self.upper_seconds)
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
}

/// Divide a diagnostic count by a supplied rate. Endpoints describe the supplied
/// count range, with ordinary floating-point rounding; neither is a proof bound.
pub fn estimate_service(demand: &Demand, rate: &Rate) -> Result<ServiceEstimate, String> {
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
    let lower_seconds = lower as f64 / rate.per_second;
    let upper_seconds = upper as f64 / rate.per_second;
    if !lower_seconds.is_finite() || !upper_seconds.is_finite() {
        return Err("service duration exceeds finite numeric range".into());
    }
    Ok(ServiceEstimate {
        pool: demand.service.pool.clone(),
        lower_seconds,
        upper_seconds,
        origin: demand.origin.clone(),
    })
}

/// Explicit stage assumption: each pool serializes its listed service demands, while
/// different pools can overlap fully. Call only when that assumption is applicable.
/// Supplied mixed-operation rates must describe the same sharing regime. This
/// calculation does not check that assumption or certify a completion interval.
pub fn estimate_overlapped_stage(terms: &[ServiceEstimate]) -> Result<(f64, f64), String> {
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
    fn model() -> RateSource {
        RateSource::Model {
            source: "synthetic service model".into(),
            conditions: "fixed modeled rate".into(),
        }
    }
    fn demand() -> Demand {
        Demand {
            service: memory(),
            count: Count::Exact(1024),
            origin: "input access account".into(),
            meaning: DemandMeaning::Algorithm,
        }
    }

    #[test]
    fn provenance_is_descriptive_and_does_not_change_the_estimate() {
        let calibrated = Rate::new(
            memory(),
            512.0,
            RateSource::Calibration {
                probe: "stream-read-v1".into(),
                environment: "test device, uncached".into(),
            },
        )
        .unwrap();
        let supplied = Rate::new(memory(), 512.0, model()).unwrap();
        // Both are only ServiceEstimate values. There is no conversion from a
        // source description, demand label, or estimate into a verified bound.
        assert_eq!(
            estimate_service(&demand(), &calibrated).unwrap(),
            estimate_service(&demand(), &supplied).unwrap()
        );
        assert_eq!(
            estimate_service(&demand(), &supplied).unwrap().seconds(),
            (2.0, 2.0)
        );
        let mut realized = demand();
        realized.meaning = DemandMeaning::Realization;
        assert_eq!(
            estimate_service(&realized, &supplied).unwrap().seconds(),
            (2.0, 2.0)
        );
    }

    #[test]
    fn rejects_wrong_precision_path_and_unavailable_counts() {
        let rate = Rate::new(memory(), 512.0, model()).unwrap();
        let mut d = demand();
        d.service.pool = "shared-memory/read".into();
        assert!(estimate_service(&d, &rate).is_err());
        d.service = memory();
        d.service.unit = Unit::Operations {
            operation: Operation::Exp,
            dtype: DType::F32,
        };
        assert!(estimate_service(&d, &rate).is_err());
        d.service = memory();
        d.count = Count::unknown("runtime-visible history");
        assert!(estimate_service(&d, &rate).is_err());
    }

    #[test]
    fn shared_service_adds_before_overlap() {
        let term = |pool: &str, n| ServiceEstimate {
            pool: pool.into(),
            lower_seconds: n,
            upper_seconds: n,
            origin: "test".into(),
        };
        assert_eq!(
            estimate_overlapped_stage(&[term("alu", 3.0), term("alu", 4.0), term("memory", 5.0)])
                .unwrap(),
            (7.0, 7.0)
        );
    }

    #[test]
    fn invalid_profile_values_fail() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(Rate::new(memory(), value, model()).is_err());
        }
        assert!(Rate::new(
            memory(),
            1.0,
            RateSource::Model {
                source: String::new(),
                conditions: "arbitrary label".into(),
            }
        )
        .is_err());
    }
}
