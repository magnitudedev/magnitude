//! One instantiated entry for one backend and one shape binding: what
//! `instantiate` produces and a backend realization consumes.
use super::ir::{Stmt, Var};
use super::types::Ty;
use crate::sym::Sym;
use crate::types::Elem;
use std::collections::{BTreeSet, HashMap};

/// A function after instantiation for one backend and one shape binding.
#[derive(Clone, Debug, PartialEq)]
pub struct LoweredIr {
    pub name: String,
    pub backend: String,
    pub ownership: Ownership,
    /// Source parallel binding requirements.
    pub alias_requirements: Vec<AliasRequirement>,
    pub params: Vec<(String, Ty)>,
    /// Number of source-authored parameters at the start of `params`. Remaining parameters are
    /// compiler-owned hidden destinations for flattened owned result leaves.
    pub source_param_count: usize,
    pub result: Ty,
    pub result_bindings: Vec<ResultBinding>,
    pub index_params: Vec<(String, Sym)>,
    pub range_params: Vec<RangeParameter>,
    pub vars: Vec<Var>,
    pub body: Vec<Stmt>,
    /// Shape parameters and their concrete values.
    pub shapes: HashMap<String, i64>,
}

/// One logical `range[N]` parameter's stable two-scalar invocation representation.
#[derive(Clone, Debug, PartialEq)]
pub struct RangeParameter {
    pub name: String,
    pub start: String,
    pub end: String,
    pub bound: Sym,
}

/// One owned tensor leaf of the source result, realized as a hidden destination parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultBinding {
    pub path: Vec<u32>,
    pub parameter: usize,
}

/// Invocation-owned, nonescaping roots. Admission must establish that their
/// allocations are independent of all other arguments. Stores still round.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ownership {
    pub intermediates: BTreeSet<String>,
    pub results: BTreeSet<String>,
}
impl Ownership {
    pub fn validate(&self, f: &LoweredIr) -> Result<(), String> {
        for name in &self.intermediates {
            if !f.params.iter().any(|(n, t)| {
                n == name && matches!(t,Ty::Tensor(s) if matches!(s.elem,Elem::Dtype(_)))
            }) {
                return Err(format!(
                    "composition intermediate {name} must name a dense tensor parameter"
                ));
            }
        }
        for name in &self.results {
            if !f.params.iter().any(|(n, t)| {
                n == name
                    && matches!(t, Ty::Tensor(s) if matches!(s.elem, Elem::Dtype(_) | Elem::Repr(_)))
            }) {
                return Err(format!(
                    "owned result {name} must name a concrete hidden tensor destination"
                ));
            }
        }
        if self
            .intermediates
            .iter()
            .any(|name| self.results.contains(name))
        {
            return Err("an ABI binding cannot be both an intermediate and a result".into());
        }
        Ok(())
    }
}

/// Parameter ordinals whose storage must be disjoint unless their exact typed
/// per-item regions were proved equal in the source computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasRequirement {
    pub left: usize,
    pub right: usize,
    pub exact_allowed: bool,
}
