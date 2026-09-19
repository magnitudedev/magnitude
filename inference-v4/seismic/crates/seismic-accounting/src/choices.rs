//! Indexed access to the actual choices owned by lowering and execution.
//! Retained construction domains; strings are diagnostic labels only.
use std::{any::Any, fmt::Debug, sync::Arc};

/// Implemented by the existing choice owner, not by a second decision registry.
/// Legal alternatives are derived by the implementation itself.
pub trait Choices: Clone + Debug + PartialEq + 'static {
    type Alternative: Debug;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<Self::Alternative>;
}
trait Erased: Debug {
    fn value(&self) -> &dyn Any;
    fn same(&self, other: &dyn Erased) -> bool;
    fn len(&self) -> usize;
    fn label(&self, index: usize) -> Option<String>;
}
impl<T: Choices> Erased for T {
    fn value(&self) -> &dyn Any {
        self
    }
    fn same(&self, other: &dyn Erased) -> bool {
        other.value().downcast_ref::<T>() == Some(self)
    }
    fn len(&self) -> usize {
        Choices::len(self)
    }
    fn label(&self, index: usize) -> Option<String> {
        self.get(index).map(|a| format!("{a:?}"))
    }
}
#[derive(Clone, Debug)]
pub struct Domain(Arc<dyn Erased>);
impl PartialEq for Domain {
    fn eq(&self, other: &Self) -> bool {
        self.0.same(other.0.as_ref())
    }
}
impl Domain {
    pub fn new<T: Choices>(owner: T) -> Result<Self, String> {
        let domain = Self(Arc::new(owner));
        domain.validate()?;
        Ok(domain)
    }
    pub fn owner<T: Choices>(&self) -> Option<&T> {
        self.0.value().downcast_ref()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn label(&self, index: usize) -> Option<String> {
        self.0.label(index)
    }
    fn validate(&self) -> Result<(), String> {
        if self.is_empty() {
            return Err("implementation choice has an empty legal domain".into());
        }
        Ok(())
    }
}

/// A typed numeric implementation choice. Bounds and identity are retained;
/// cardinality never allocates an entry for every legal integer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegerRange<K> {
    pub decision: K,
    first: u64,
    last: u64,
    count: usize,
}
impl<K> IntegerRange<K> {
    pub fn new(decision: K, first: u64, last: u64) -> Result<Self, String> {
        let count = first
            .abs_diff(last)
            .checked_add(1)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("choice interval cardinality overflow")?;
        Ok(Self {
            decision,
            first,
            last,
            count,
        })
    }
    pub fn first(&self) -> u64 {
        self.first
    }
    pub fn last(&self) -> u64 {
        self.last
    }
    pub fn index(&self, value: u64) -> Option<usize> {
        if value < self.first.min(self.last) || value > self.first.max(self.last) {
            None
        } else {
            usize::try_from(self.first.abs_diff(value)).ok()
        }
    }
}
impl<K: Clone + Debug + PartialEq + 'static> Choices for IntegerRange<K> {
    type Alternative = u64;
    fn len(&self) -> usize {
        self.count
    }
    fn get(&self, index: usize) -> Option<u64> {
        if index >= self.count {
            None
        } else if self.first <= self.last {
            Some(self.first + index as u64)
        } else {
            Some(self.first - index as u64)
        }
    }
}
impl Choices for seismic_lang::lowered_ir::Decision {
    type Alternative = seismic_lang::lowered_ir::Alternative;
    fn len(&self) -> usize {
        self.alternatives.len()
    }
    fn get(&self, index: usize) -> Option<Self::Alternative> {
        self.alternatives.get(index)
    }
}

impl Choices for seismic_lang::lower::alternatives::LoweringChoice {
    type Alternative = seismic_lang::lowered_ir::Alternative;
    fn len(&self) -> usize {
        self.decision().alternatives.len()
    }
    fn get(&self, index: usize) -> Option<Self::Alternative> {
        self.decision().alternatives.get(index)
    }
}
impl Choices for seismic_lang::normalize::loads::Choice {
    type Alternative = seismic_lang::ir::LoadMode;
    fn len(&self) -> usize {
        self.modes().len()
    }
    fn get(&self, index: usize) -> Option<Self::Alternative> {
        self.modes().get(index).copied()
    }
}
