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
    pub index_params: Vec<(String, Sym)>,
    pub vars: Vec<Var>,
    pub body: Vec<Stmt>,
    /// Shape parameters and their concrete values.
    pub shapes: HashMap<String, i64>,
}

/// Invocation-owned, nonescaping roots. Admission must establish that their
/// allocations are independent of all other arguments. Stores still round.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ownership {
    pub intermediates: BTreeSet<String>,
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
