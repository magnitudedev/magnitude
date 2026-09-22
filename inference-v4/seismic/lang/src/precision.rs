//! Caller-owned numerical policy types. Compiler assessment, transfer,
//! qualification and comparison belong outside the language crate.

use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceRequirement {
    #[default]
    Proven,
    Qualified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputRange {
    pub minimum: Finite,
    pub maximum: Finite,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrecisionPolicy {
    Exact,
    Bounded {
        default: Tolerance,
        outputs: BTreeMap<String, Tolerance>,
        evidence: EvidenceRequirement,
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
            evidence: EvidenceRequirement::Proven,
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
