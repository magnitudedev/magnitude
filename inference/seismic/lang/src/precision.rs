//! Caller-owned numerical policy types. Compiler assessment, transfer,
//! numerical analysis and comparison belong outside the language crate.

use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Limit(f64);
impl Limit {
    pub const ZERO: Self = Self(0.0);
    pub fn new(value: f64) -> Result<Self, String> {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "precision limit must be finite and nonnegative, got {value}"
            ));
        }
        Ok(Self(if value == 0.0 { 0.0 } else { value }))
    }
    pub fn get(self) -> f64 {
        self.0
    }
    /// A limit from a positive finite constant, for tables: `new` for
    /// values known when the program is written.
    pub const fn from_finite(value: f64) -> Self {
        assert!(value > 0.0 && value < f64::INFINITY);
        Self(value)
    }
}
impl TryFrom<f64> for Limit {
    type Error = String;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}
impl From<Limit> for f64 {
    fn from(value: Limit) -> Self {
        value.get()
    }
}
impl PartialEq for Limit {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Limit {}
impl PartialOrd for Limit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Limit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl Hash for Limit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Finite(f64);
impl Finite {
    pub fn new(value: f64) -> Result<Self, String> {
        value
            .is_finite()
            .then_some(Self(if value == 0.0 { 0.0 } else { value }))
            .ok_or_else(|| format!("range endpoint must be finite, got {value}"))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}
impl TryFrom<f64> for Finite {
    type Error = String;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}
impl From<Finite> for f64 {
    fn from(value: Finite) -> Self {
        value.get()
    }
}
impl PartialEq for Finite {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Finite {}
impl Hash for Finite {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Tolerance {
    pub absolute: Limit,
    pub relative: Limit,
    pub relative_floor: Limit,
    pub ulps: Option<u64>,
}
impl Tolerance {
    pub const EXACT: Self = Self {
        absolute: Limit::ZERO,
        relative: Limit::ZERO,
        relative_floor: Limit::ZERO,
        ulps: Some(0),
    };
    pub fn envelope(self, reference: f64) -> f64 {
        self.absolute.get() + self.relative.get() * reference.abs().max(self.relative_floor.get())
    }
}
impl Default for Tolerance {
    fn default() -> Self {
        Self::EXACT
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SpecialPolicy {
    pub nan: bool,
    pub infinity: bool,
    pub signed_zero: bool,
    pub subnormal: bool,
}
impl SpecialPolicy {
    pub const PRESERVE: Self = Self {
        nan: true,
        infinity: true,
        signed_zero: true,
        subnormal: true,
    };
}
impl Default for SpecialPolicy {
    fn default() -> Self {
        Self::PRESERVE
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "RangeEndpoints")]
pub struct InputRange {
    pub minimum: Finite,
    pub maximum: Finite,
}
#[derive(serde::Deserialize)]
struct RangeEndpoints {
    minimum: Finite,
    maximum: Finite,
}
impl TryFrom<RangeEndpoints> for InputRange {
    type Error = String;
    fn try_from(value: RangeEndpoints) -> Result<Self, Self::Error> {
        Self::new(value.minimum.get(), value.maximum.get())
    }
}
impl InputRange {
    pub fn new(minimum: f64, maximum: f64) -> Result<Self, String> {
        let (minimum, maximum) = (Finite::new(minimum)?, Finite::new(maximum)?);
        if minimum.get() > maximum.get() {
            return Err(format!(
                "input range minimum {} exceeds maximum {}",
                minimum.get(),
                maximum.get()
            ));
        }
        Ok(Self { minimum, maximum })
    }
    pub fn magnitude(maximum: f64) -> Result<Self, String> {
        let maximum = Limit::new(maximum)?.get();
        Self::new(-maximum, maximum)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PrecisionPolicy {
    Exact,
    Bounded {
        default: Tolerance,
        outputs: BTreeMap<String, Tolerance>,
        specials: SpecialPolicy,
        inputs: BTreeMap<String, InputRange>,
    },
    Unconstrained,
}
impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::Exact
    }
}
impl PrecisionPolicy {
    pub fn bounded(default: Tolerance) -> Self {
        Self::Bounded {
            default,
            outputs: BTreeMap::new(),
            specials: SpecialPolicy::PRESERVE,
            inputs: BTreeMap::new(),
        }
    }
    pub fn tolerance(&self, output: &str) -> Option<Tolerance> {
        match self {
            Self::Exact => Some(Tolerance::EXACT),
            Self::Bounded {
                default, outputs, ..
            } => Some(outputs.get(output).copied().unwrap_or(*default)),
            Self::Unconstrained => None,
        }
    }
}

/// The bound on one error class's deviation from the reference, per floating
/// result or state subject: the root-mean-square error relative to the
/// reference's root mean square, and the largest gap between published
/// values' rounding cells in units of the reference's root mean square.
/// The first uses raw published errors; the second rejects concentrated
/// error without charging final storage rounding as kernel error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ErrorEnvelope {
    pub relative_rms: Limit,
    pub peak: Limit,
}
impl ErrorEnvelope {
    /// Part of both numerical evidence and persistent native tuning keys.
    pub const COMPARISON_VERSION: &'static str = "rounding-cells-v1";

    /// The envelope that admits what either admits.
    pub fn widest(self, other: Self) -> Self {
        Self {
            relative_rms: self.relative_rms.max(other.relative_rms),
            peak: self.peak.max(other.peak),
        }
    }
}

/// What empirical native tuning holds its candidates to: `policy` for a
/// configuration in no error class, and for one in declared error classes
/// the envelopes of the classes the caller admits. A configuration in a
/// class that is not admitted is never formed.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TuningPrecision {
    pub policy: PrecisionPolicy,
    pub admitted: BTreeMap<String, ErrorEnvelope>,
}
impl From<PrecisionPolicy> for TuningPrecision {
    /// No error class admitted.
    fn from(policy: PrecisionPolicy) -> Self {
        Self {
            policy,
            admitted: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;
    #[test]
    fn policy_round_trip_preserves_all_limits_and_subjects() {
        let mut policy = PrecisionPolicy::bounded(Tolerance {
            absolute: Limit::new(1e-5).unwrap(),
            relative: Limit::new(1e-4).unwrap(),
            relative_floor: Limit::new(0.25).unwrap(),
            ulps: Some(7),
        });
        if let PrecisionPolicy::Bounded {
            outputs, inputs, ..
        } = &mut policy
        {
            outputs.insert("i4".into(), Tolerance::EXACT);
            inputs.insert("i0".into(), InputRange::new(-2., 4.).unwrap());
        }
        let bytes = postcard::to_allocvec(&policy).unwrap();
        assert_eq!(
            postcard::from_bytes::<PrecisionPolicy>(&bytes).unwrap(),
            policy
        );
    }
    #[test]
    fn decoding_rejects_invalid_limits_and_ranges() {
        for value in [-1., f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let bytes = postcard::to_allocvec(&value).unwrap();
            assert!(postcard::from_bytes::<Limit>(&bytes).is_err());
        }
        let invalid = InputRange {
            minimum: Finite::new(2.).unwrap(),
            maximum: Finite::new(1.).unwrap(),
        };
        assert!(
            postcard::from_bytes::<InputRange>(&postcard::to_allocvec(&invalid).unwrap()).is_err()
        );
        assert_eq!(Limit::new(-0.).unwrap(), Limit::ZERO);
    }
}
