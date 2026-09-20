//! Applicability of one definition under one binding: predicates over concrete shapes are
//! decided now; predicates over a structurally bound shape parameter become requirements
//! on the bound site (R1: they constrain the slice capacity).

use super::super::{Requirement, SiteId};
use super::walk;
use crate::sir::{Body, ExprKind, Predicate, StmtKind, VarId};
use crate::sym::{Atom, Sym};
use crate::types::{Elem, Ty};
use std::collections::BTreeMap;

/// How a definition's shape and element parameters are bound at one occurrence.
/// `structural` follows the definition's shape-parameter order.
pub struct Binding {
    pub shapes: BTreeMap<String, i64>,
    pub structural: Vec<(String, SiteId)>,
    /// Shape parameters bound to a runtime-valued semantic extent of the caller.
    pub dynamic: Vec<String>,
    pub elems: BTreeMap<String, Elem>,
}

/// What construction knows about a site when a requirement is derived.
#[derive(Clone, Copy)]
pub struct SiteExtent {
    pub extent: i64,
    /// `extent` is a static upper bound of a runtime-valued domain.
    pub bounded: bool,
}

pub fn describe(predicate: &Predicate) -> String {
    match predicate {
        Predicate::NonNegative(e) => format!("{e} >= 0"),
        Predicate::Zero(e) => format!("{e} == 0"),
        Predicate::NonZero(e) => format!("{e} != 0"),
        Predicate::Full(p) => format!("full({p})"),
    }
}

const UNDECIDABLE: &str = "predicate not decidable over a structural extent";
const RUNTIME: &str = "predicate not decidable over a runtime extent";

/// Requirements of `predicates` under `binding`, or the reason the definition is inapplicable.
pub fn requirements(
    predicates: &[Predicate],
    binding: &Binding,
    site: &dyn Fn(SiteId) -> SiteExtent,
) -> Result<Vec<Requirement>, String> {
    let mut out = Vec::new();
    for predicate in predicates {
        let e = match predicate {
            Predicate::Full(param) => {
                // Vacuous for a semantic binding, static or dynamic.
                if let Some((_, id)) = binding.structural.iter().find(|(name, _)| name == param) {
                    let known = site(*id);
                    if known.bounded {
                        return Err(format!("`full({param})` over a runtime extent"));
                    }
                    out.push(Requirement::Divides {
                        site: *id,
                        extent: known.extent,
                    });
                }
                continue;
            }
            Predicate::NonNegative(e) | Predicate::Zero(e) | Predicate::NonZero(e) => e,
        };
        let params = e.params();
        if binding.dynamic.iter().any(|name| params.contains(name)) {
            return Err(format!("{RUNTIME}: `{}`", describe(predicate)));
        }
        let bound: Vec<&(String, SiteId)> = binding
            .structural
            .iter()
            .filter(|(name, _)| params.contains(name))
            .collect();
        match bound.as_slice() {
            [] => {
                let value = e
                    .eval(&|name| binding.shapes.get(name).copied())
                    .ok_or_else(|| {
                        format!(
                            "predicate `{}` is not evaluable over the bound shapes",
                            describe(predicate)
                        )
                    })?;
                let holds = match predicate {
                    Predicate::NonNegative(_) => value >= 0,
                    Predicate::Zero(_) => value == 0,
                    _ => value != 0,
                };
                if !holds {
                    return Err(format!("predicate `{}` does not hold", describe(predicate)));
                }
            }
            [(param, id)] => out.push(structural(predicate, e, param, *id, binding)?),
            _ => return Err(format!("{UNDECIDABLE}: `{}`", describe(predicate))),
        }
    }
    Ok(out)
}

/// `predicate` over the single structurally bound parameter `param`.
fn structural(
    predicate: &Predicate,
    e: &Sym,
    param: &str,
    site: SiteId,
    binding: &Binding,
) -> Result<Requirement, String> {
    let undecidable = || format!("{UNDECIDABLE}: `{}`", describe(predicate));
    let residual = binding.shapes.iter().fold(e.clone(), |r, (name, value)| {
        r.subst(&Atom::Param(name.clone()), &Sym::constant(*value))
    });
    if let Some((c, rest)) = residual.linear_in(&Atom::Param(param.to_string())) {
        let k = rest.as_constant().ok_or_else(undecidable)?;
        // c * P + k
        return match predicate {
            Predicate::NonNegative(_) if c > 0 => Ok(Requirement::AtLeast {
                site,
                value: -k.div_euclid(c),
            }),
            Predicate::NonNegative(_) => Ok(Requirement::AtMost {
                site,
                value: k.div_euclid(-c),
            }),
            Predicate::Zero(_) if k % c == 0 => Ok(Requirement::Equal {
                site,
                value: -k / c,
            }),
            Predicate::Zero(_) => Err(format!("predicate `{}` cannot hold", describe(predicate))),
            _ => Err(undecidable()),
        };
    }
    // k * (P % unit) == 0
    if let (Predicate::Zero(_), [Atom::Rem(num, den)]) = (predicate, residual.atoms().as_slice()) {
        let unit = den.eval(&|name| binding.shapes.get(name).copied());
        let multiple = residual
            .linear_in(&Atom::Rem(num.clone(), den.clone()))
            .is_some_and(|(_, rest)| rest.is_zero());
        if let (true, true, Some(unit)) = (multiple, **num == Sym::param(param), unit) {
            return Ok(Requirement::Multiple { site, unit });
        }
    }
    Err(undecidable())
}

/// Static extents of authored domains. Loop-scoped symbols are bounded by their binder's
/// declared range, so a runtime-valued domain gets its static upper bound when one exists.
pub struct Bounds<'a> {
    shapes: &'a BTreeMap<String, i64>,
    /// Loop-scoped symbol -> `[lo, hi)` of its binder.
    symbols: BTreeMap<String, (Sym, Sym)>,
}

impl<'a> Bounds<'a> {
    pub fn new(body: &Body, shapes: &'a BTreeMap<String, i64>) -> Bounds<'a> {
        let mut ranges: BTreeMap<VarId, (Sym, Sym)> = BTreeMap::new();
        walk::stmts(&body.block, &mut |s| {
            if let StmtKind::Range { var, lo, hi, .. } = &s.kind {
                if let (Some(lo), Some(hi)) = (&lo.sym, &hi.sym) {
                    ranges.insert(*var, (lo.clone(), hi.clone()));
                }
            }
        });
        let mut symbols = BTreeMap::new();
        walk::block(&body.block, true, &mut |e| {
            let (ExprKind::Var(var), Some(sym)) = (&e.kind, &e.sym) else {
                return;
            };
            let params = sym.params();
            let [name] = params.as_slice() else { return };
            if *sym != Sym::param(name) || shapes.contains_key(name) {
                return;
            }
            let range = match body.vars.get(*var).map(|v| &v.ty) {
                Some(Ty::Index(bound)) => Some((Sym::constant(0), bound.clone())),
                _ => ranges.get(var).cloned(),
            };
            if let Some(range) = range {
                symbols.insert(name.clone(), range);
            }
        });
        Bounds { shapes, symbols }
    }

    fn range(&self, name: &str, depth: usize) -> Option<(i64, i64)> {
        if let Some(value) = self.shapes.get(name) {
            return Some((*value, *value));
        }
        let (lo, hi) = self.symbols.get(name)?;
        let depth = depth.checked_sub(1)?;
        let lo = lo.eval_interval(&|n| self.range(n, depth))?.0;
        let hi = hi.eval_interval(&|n| self.range(n, depth))?.1;
        (hi > lo).then_some((lo, hi - 1))
    }

    /// Extent of `lo..hi`: exact under the concrete shapes, else its static upper bound.
    /// An empty domain is never visited; its site keeps the unit domain.
    pub fn extent(&self, lo: &Sym, hi: &Sym) -> Option<SiteExtent> {
        let length = hi.sub(lo);
        if let Some(extent) = length.eval(&|name| self.shapes.get(name).copied()) {
            return Some(SiteExtent {
                extent: extent.max(1),
                bounded: false,
            });
        }
        let (_, upper) = length.eval_interval(&|name| self.range(name, self.symbols.len() + 1))?;
        Some(SiteExtent {
            extent: upper.max(1),
            bounded: true,
        })
    }
}
