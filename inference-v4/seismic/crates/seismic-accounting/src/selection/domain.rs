//! Indexed access to the actual choices owned by lowering and execution.
//! The optimizer retains the typed owner intact; strings are only display.
use std::{any::Any, fmt::Debug, sync::{Arc, Weak}};

/// Implemented by the existing choice owner, not by a second decision registry.
/// Legal alternatives are derived there; search only partitions their indices.
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
/// Historical decisions must not keep an otherwise retired prepared IR alive.
#[derive(Clone, Debug)]
pub(super) struct WeakDomain(Weak<dyn Erased>);
impl WeakDomain {
    pub(super) fn upgrade(&self) -> Option<Domain> { self.0.upgrade().map(Domain) }
}
impl PartialEq for Domain {
    fn eq(&self, other: &Self) -> bool {
        self.0.same(other.0.as_ref())
    }
}
impl Domain {
    pub(super) fn downgrade(&self) -> WeakDomain { WeakDomain(Arc::downgrade(&self.0)) }
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
    pub(super) fn validate(&self) -> Result<(), String> {
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

/// Exact unresolved subsets of the implementation's own typed domain. The
/// bound is private and can only be strengthened by the owning search after
/// deriving resource demand for every retained alternative.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    parent: Vec<usize>,
    children: Option<(Domain, std::ops::Range<usize>)>,
    lower_bound: u64,
}
/// A completed exclusion keeps indexed coverage and its derived bound. Its
/// execution owner has no remaining construction or relaxation responsibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExcludedRegion {
    parent: Vec<usize>,
    indices: Option<std::ops::Range<usize>>,
    lower_bound: u64,
}
impl ExcludedRegion {
    pub fn len(&self) -> usize { self.indices.as_ref().map_or(1, |indices| indices.len()) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn lower_bound(&self) -> u64 { self.lower_bound }
    pub fn paths(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        (0..self.len()).map(move |offset| {
            let mut path = self.parent.clone();
            if let Some(indices) = &self.indices { path.push(indices.start + offset); }
            path
        })
    }
}
impl Region {
    pub(super) fn exclude(self) -> ExcludedRegion {
        ExcludedRegion {
            parent: self.parent,
            indices: self.children.map(|(_, indices)| indices),
            lower_bound: self.lower_bound,
        }
    }
    pub(super) fn root() -> Self {
        Self {
            parent: Vec::new(),
            children: None,
            lower_bound: 0,
        }
    }
    pub(super) fn children(parent: Vec<usize>, domain: Domain, lower_bound: u64) -> Self {
        let indices = 0..domain.len();
        Self {
            parent,
            children: Some((domain, indices)),
            lower_bound,
        }
    }
    pub fn len(&self) -> usize {
        self.children
            .as_ref()
            .map_or(1, |(_, indices)| indices.len())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn lower_bound(&self) -> u64 {
        self.lower_bound
    }
    pub fn paths(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        (0..self.len()).map(move |offset| {
            let mut path = self.parent.clone();
            if let Some((_, indices)) = &self.children {
                path.push(indices.start + offset);
            }
            path
        })
    }
    pub(super) fn first_path(&self) -> Vec<usize> {
        let mut path = self.parent.clone();
        if let Some((_, indices)) = &self.children {
            path.push(indices.start);
        }
        path
    }
    pub fn alternatives(&self) -> Option<(&Domain, std::ops::Range<usize>)> {
        self.children
            .as_ref()
            .map(|(domain, indices)| (domain, indices.clone()))
    }
    pub(super) fn strengthen(&mut self, demand: &crate::schedule::Demand) -> Result<(), String> {
        self.lower_bound = self.lower_bound.max(demand.lower_bound()?);
        Ok(())
    }
    /// Disjoint, exhaustive bisection of this owner's indexed alternatives.
    /// No implementation is constructed and no candidate is silently discarded.
    pub(super) fn split(self) -> Result<(Self, Self), Self> {
        let Some((domain, indices)) = &self.children else {
            return Err(self);
        };
        if indices.len() < 2 {
            return Err(self);
        }
        let middle = indices.start + indices.len() / 2;
        let mut left = self.clone();
        left.children = Some((domain.clone(), indices.start..middle));
        let mut right = self.clone();
        right.children = Some((domain.clone(), middle..indices.end));
        Ok((left, right))
    }
}
