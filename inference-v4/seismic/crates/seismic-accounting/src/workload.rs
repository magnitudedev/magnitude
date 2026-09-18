//! Invocation storage bindings and finite derivation budgets shared by backend models.
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocation {
    pub id: u64,
    pub bytes: u64,
    pub alignment: u64,
    /// Bytes constrained by this workload. Other external bytes are unknown, not
    /// zero. Only values needed to determine control or addressing must be known.
    pub known_bytes: BTreeMap<u64, u8>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub allocation: u64,
    pub offset: u64,
    pub bytes: u64,
}
/// One finite integer progression. Every admitted value is `min + n * stride`
/// and at most `max`; endpoints are inclusive. This describes runtime inputs,
/// not an average or worst-case timing objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerRange {
    pub min: i128,
    pub max: i128,
    pub stride: u64,
}
impl IntegerRange {
    pub fn contains(self, value: i128) -> bool {
        self.stride != 0 && self.min <= value && value <= self.max
            && value.checked_sub(self.min).is_some_and(|offset| offset % i128::from(self.stride) == 0)
    }
    pub fn covers(self, other: Self) -> bool {
        other.stride != 0 && other.min <= other.max && self.contains(other.min)
            && self.contains(other.max)
            && (other.min == other.max || other.stride % self.stride == 0)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntegerInput {
    /// Index into the shared eight-byte scalar slots.
    Scalar { slot: usize },
    /// Byte offset into an actual allocation, preserving alias identity.
    Allocation { allocation: u64, offset: u64 },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegerDomain {
    pub input: IntegerInput,
    pub bytes: u8,
    pub signed: bool,
    pub range: IntegerRange,
}
impl IntegerDomain {
    pub fn decode(&self, bytes: &[u8]) -> Option<i128> {
        if !matches!(self.bytes, 1 | 2 | 4 | 8) || bytes.len() != usize::from(self.bytes) { return None; }
        let mut raw = [0u8; 8];
        raw[..bytes.len()].copy_from_slice(bytes);
        let value = u64::from_le_bytes(raw);
        let bits = u32::from(self.bytes) * 8;
        Some(if self.signed && value & (1u64 << (bits - 1)) != 0 {
            i128::from(value) - (1i128 << bits)
        } else { i128::from(value) })
    }
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        self.decode(bytes).is_some_and(|value| self.range.contains(value))
    }
    fn valid(&self) -> bool {
        if !matches!(self.bytes, 1 | 2 | 4 | 8) || self.range.stride == 0 || self.range.min > self.range.max { return false; }
        let bits = u32::from(self.bytes) * 8;
        let (min, max) = if self.signed { (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1) }
            else { (0, (1i128 << bits) - 1) };
        self.range.min >= min && self.range.max <= max && self.range.contains(self.range.max)
    }
    fn span(&self) -> Option<(Option<u64>, u64, u64)> {
        let (allocation, start) = match self.input {
            IntegerInput::Scalar { slot } => (None, u64::try_from(slot).ok()?.checked_mul(8)?),
            IntegerInput::Allocation { allocation, offset } => (Some(allocation), offset),
        };
        Some((allocation, start, start.checked_add(u64::from(self.bytes))?))
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarWorkload {
    pub identity: String,
    pub allocations: Vec<Allocation>,
    /// Ordered exactly as ScalarProgram::buffers. Equal allocation IDs denote
    /// actual shared storage even when parameter names differ.
    pub buffers: Vec<BufferBinding>,
    /// The shared eight-byte-slot scalar ABI, checked against its typed schema.
    /// A domain field retains its canonical minimum here for ABI validation;
    /// these bytes must never be interpreted as its exact runtime value.
    pub scalars: Vec<u8>,
    /// Independently varying input values. A completed model must hold uniformly
    /// over this entire domain. Unknown control or nonuniform cost is unresolved.
    pub integer_domains: Vec<IntegerDomain>,
}
impl ScalarWorkload {
    pub fn validate(&self) -> Result<(), String> {
        let ids: std::collections::BTreeSet<_> = self.allocations.iter().map(|a| a.id).collect();
        if ids.len() != self.allocations.len() || self.allocations.iter().any(|a|
            !a.alignment.is_power_of_two() || a.known_bytes.keys().any(|offset| *offset >= a.bytes))
            || self.buffers.iter().any(|b| !self.allocations.iter().find(|a| a.id == b.allocation)
                .is_some_and(|a| b.offset.checked_add(b.bytes).is_some_and(|end| end <= a.bytes))) {
            return Err("invalid workload allocation or binding".into());
        }
        let mut spans = Vec::new();
        for domain in &self.integer_domains {
            if !domain.valid() { return Err("invalid finite integer input domain".into()); }
            let span @ (allocation, start, end) = domain.span().ok_or("integer domain offset overflow")?;
            if spans.iter().any(|&(a, s, e)| a == allocation && start < e && s < end) {
                return Err("overlapping integer input domains".into());
            }
            match allocation {
                None => {
                    let bytes = self.scalars.get(start as usize..end as usize).ok_or("integer domain exceeds scalar ABI")?;
                    if domain.decode(bytes) != Some(domain.range.min) { return Err("integer scalar domain needs its canonical minimum".into()); }
                }
                Some(id) => {
                    let a = self.allocations.iter().find(|a| a.id == id).ok_or("integer domain allocation missing")?;
                    if end > a.bytes || !self.buffers.iter().any(|b| b.allocation == id && b.offset <= start
                        && b.offset.checked_add(b.bytes).is_some_and(|bound| end <= bound)) {
                        return Err("integer domain exceeds bound allocation views".into());
                    }
                    if a.known_bytes.range(start..end).next().is_some() {
                        return Err("integer domain overlaps exact known bytes".into());
                    }
                }
            }
            spans.push(span);
        }
        Ok(())
    }
    pub fn conditions_allocation(&self, allocation: u64) -> bool {
        self.allocations.iter().any(|a| a.id == allocation && !a.known_bytes.is_empty())
            || self.integer_domains.iter().any(|d| matches!(d.input, IntegerInput::Allocation { allocation: id, .. } if id == allocation))
    }
    /// Check actual scalar bytes against the exact fields and finite domains.
    pub fn accepts_scalars(&self, actual: &[u8]) -> bool {
        if actual.len() != self.scalars.len() { return false; }
        for domain in &self.integer_domains {
            if let Some((None, start, end)) = domain.span() {
                if !actual.get(start as usize..end as usize).is_some_and(|bytes| domain.accepts(bytes)) { return false; }
            }
        }
        self.scalars.iter().zip(actual).enumerate().all(|(offset, (a, b))| a == b || self.integer_domains.iter().any(|d|
            d.span().is_some_and(|(allocation, start, end)| allocation.is_none() && start <= offset as u64 && (offset as u64) < end)))
    }
    /// Whether a more specific invocation establishes this workload's facts.
    /// Program, hardware, objective, and execution-family identities remain the
    /// owner's responsibility. Retained search still requires exact equality.
    pub fn covers(&self, invocation: &Self) -> bool {
        if self.identity != invocation.identity || self.buffers != invocation.buffers
            || self.allocations.len() != invocation.allocations.len()
            || self.validate().is_err() || invocation.validate().is_err()
            || !self.accepts_scalars(&invocation.scalars) { return false; }
        // No scalar that was exact in the original selection may become variable.
        if invocation.integer_domains.iter().any(|actual| matches!(actual.input, IntegerInput::Scalar { .. })
            && !self.integer_domains.iter().any(|d| d.input == actual.input && d.bytes == actual.bytes && d.signed == actual.signed)) { return false; }
        if !self.allocations.iter().all(|expected| {
            invocation.allocations.iter().find(|actual| actual.id == expected.id).is_some_and(|actual|
                expected.bytes == actual.bytes && actual.alignment >= expected.alignment
                && expected.known_bytes.iter().all(|(offset, byte)| actual.known_bytes.get(offset) == Some(byte)))
        }) { return false; }
        self.integer_domains.iter().all(|expected| {
            if let Some(actual) = invocation.integer_domains.iter().find(|d| d.input == expected.input) {
                return expected.bytes == actual.bytes && expected.signed == actual.signed && expected.range.covers(actual.range);
            }
            let Some((allocation, start, end)) = expected.span() else { return false; };
            match allocation {
                None => invocation.scalars.get(start as usize..end as usize).is_some_and(|bytes| expected.accepts(bytes)),
                Some(id) => invocation.allocations.iter().find(|a| a.id == id).and_then(|a|
                    (start..end).map(|offset| a.known_bytes.get(&offset).copied()).collect::<Option<Vec<_>>>())
                    .is_some_and(|bytes| expected.accepts(&bytes)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn workload() -> ScalarWorkload {
        ScalarWorkload { integer_domains: Vec::new(), identity: "entry".into(), scalars: vec![0; 8],
            allocations: vec![Allocation { id: 0, bytes: 32, alignment: 8,
                known_bytes: BTreeMap::from([(4, 7)]) }],
            buffers: vec![BufferBinding { allocation: 0, offset: 0, bytes: 16 }] }
    }
    #[test]
    fn workload_coverage_accepts_only_established_conditions() {
        let expected = workload();
        let mut actual = expected.clone();
        actual.allocations[0].alignment = 16;
        actual.allocations[0].known_bytes.insert(8, 23);
        assert!(expected.covers(&actual));
        assert!(!actual.covers(&expected));
        for mutate in [
            |w: &mut ScalarWorkload| { w.scalars[0] = 1; },
            |w: &mut ScalarWorkload| { w.buffers[0].offset = 8; },
            |w: &mut ScalarWorkload| { w.buffers[0].allocation = 1; },
            |w: &mut ScalarWorkload| { w.allocations[0].alignment = 4; },
            |w: &mut ScalarWorkload| { w.allocations[0].alignment = 24; },
            |w: &mut ScalarWorkload| { w.allocations[0].bytes = 64; },
            |w: &mut ScalarWorkload| { w.allocations[0].known_bytes.remove(&4); },
            |w: &mut ScalarWorkload| { w.allocations[0].known_bytes.insert(4, 8); },
        ] {
            let mut incompatible = actual.clone();
            mutate(&mut incompatible);
            assert!(!expected.covers(&incompatible));
        }
    }
    #[test]
    fn integer_domain_implication_preserves_extent_residue_and_exact_fields() {
        let mut expected = workload();
        expected.scalars = [0i32.to_le_bytes(), [0; 4], 7i32.to_le_bytes(), [0; 4]].concat();
        expected.integer_domains = vec![
            IntegerDomain { input: IntegerInput::Scalar { slot: 0 }, bytes: 4, signed: true,
                range: IntegerRange { min: 0, max: 12, stride: 4 } },
            IntegerDomain { input: IntegerInput::Allocation { allocation: 0, offset: 8 }, bytes: 4, signed: true,
                range: IntegerRange { min: -2, max: 6, stride: 2 } },
        ];
        expected.validate().unwrap();
        let mut exact = expected.clone();
        exact.integer_domains.clear();
        exact.scalars[..4].copy_from_slice(&8i32.to_le_bytes());
        for (i, byte) in 4i32.to_le_bytes().into_iter().enumerate() {
            exact.allocations[0].known_bytes.insert(8 + i as u64, byte);
        }
        assert!(expected.covers(&exact));
        assert!(!exact.covers(&expected));
        let mut narrower = expected.clone();
        narrower.integer_domains[0].range = IntegerRange { min: 4, max: 12, stride: 8 };
        narrower.scalars[..4].copy_from_slice(&4i32.to_le_bytes());
        assert!(expected.covers(&narrower));
        assert!(!narrower.covers(&expected));
        for value in [2i32, 16, -4] {
            exact.scalars[..4].copy_from_slice(&value.to_le_bytes());
            assert!(!expected.covers(&exact));
        }
        exact.scalars[..4].copy_from_slice(&8i32.to_le_bytes());
        exact.scalars[8] = 8;
        assert!(!expected.covers(&exact), "unmodeled scalar field remains exact");
        let mut invalid = expected.clone();
        invalid.integer_domains.push(expected.integer_domains[1].clone());
        assert!(invalid.validate().is_err());
        invalid.integer_domains.last_mut().unwrap().input = IntegerInput::Allocation { allocation: 0, offset: 9 };
        assert!(invalid.validate().is_err(), "partial domain overlaps must also be rejected");
        invalid = expected.clone();
        invalid.allocations[0].known_bytes.insert(9, 0);
        assert!(invalid.validate().is_err(), "range may not retain a representative exact byte");
        invalid = expected.clone();
        invalid.integer_domains[1].range = IntegerRange { min: 0, max: i128::from(u32::MAX), stride: 1 };
        assert!(invalid.validate().is_err(), "domain must fit signed storage");
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DerivationLimits {
    pub instructions: u64,
    pub operations: usize,
}

/// A construction limit does not constrain the execution's legal domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivationLimit {
    Instructions(u64),
    Operations(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DerivationError {
    Exhausted(DerivationLimit),
    /// A legal execution needs analysis the current model cannot establish.
    /// Selection must retain its region, not treat it as infeasible.
    Unsupported(String),
    Analysis(String),
}
impl std::fmt::Display for DerivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted(DerivationLimit::Instructions(limit)) => {
                write!(f, "model derivation instruction budget exhausted ({limit})")
            }
            Self::Exhausted(DerivationLimit::Operations(limit)) => {
                write!(f, "model derivation operation budget exhausted ({limit})")
            }
            Self::Unsupported(message) | Self::Analysis(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for DerivationError {}
impl From<String> for DerivationError {
    fn from(message: String) -> Self {
        Self::Analysis(message)
    }
}
impl From<&str> for DerivationError {
    fn from(message: &str) -> Self {
        Self::Analysis(message.into())
    }
}
impl From<DerivationError> for String {
    fn from(error: DerivationError) -> Self {
        error.to_string()
    }
}
