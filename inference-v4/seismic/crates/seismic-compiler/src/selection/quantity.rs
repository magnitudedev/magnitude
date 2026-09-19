//! Derived quantities: arithmetic expressions over numerical site values. Piece counts,
//! visits, byte counts and work counts are never solver decisions. Target-neutral: a
//! backend states a rule of its own mapping as a `Quantity::Rule` over other quantities.
use seismic_lang::family::SiteId;
use seismic_lang::sym::Sym;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// A semantic extent expression over workload constants and ranged names: structural
/// parameters and slices (exact site widths), and runtime values (dynamic shape parameters,
/// `@dyn#n` range lengths, index variables) known only by their static range. Evaluates to
/// the expression's static upper (or lower) bound by interval arithmetic; exact when every
/// name is.
#[derive(Debug)]
pub struct SymbolicExtent {
    pub sym: Sym,
    pub lower: bool,
    pub constants: Arc<BTreeMap<String, i64>>,
    pub ranges: Arc<BTreeMap<String, (Quantity, Quantity)>>,
}

#[derive(Clone, Debug)]
pub enum Quantity {
    Constant(u64),
    /// The selected value of a site.
    Site(SiteId),
    /// `ceil(extent / site)`: pieces or visits of one binder.
    Pieces { extent: u64, site: SiteId },
    Symbolic(Arc<SymbolicExtent>),
    Product(Vec<Quantity>),
    Sum(Vec<Quantity>),
    Max(Vec<Quantity>),
    /// `n - 1`, saturating: the last coordinate of an axis of extent `n`.
    Predecessor(Box<Quantity>),
    /// `ceil(value / divisor)`: fractional rates carried as a scaled integer.
    Quotient(Box<Quantity>, u64),
    /// `value` rounded up to a multiple of `unit` (allocation alignment).
    Aligned(Box<Quantity>, u64),
    /// A backend's deterministic rule over the values of `arguments` (a storage share, a
    /// reduction algorithm's work): a pure function, stated where the backend documents it.
    Rule(Rule, Vec<Quantity>),
    /// `exact` when derivable, else its static upper `bound` (runtime trip counts).
    Bounded { exact: Box<Quantity>, bound: Box<Quantity> },
    /// No supported derivation: evaluation fails, never zero.
    Unknown(String),
}

/// A named pure function of evaluated quantities.
#[derive(Clone)]
pub struct Rule {
    pub name: &'static str,
    pub apply: Arc<dyn Fn(&[u64]) -> Result<u64, String> + Send + Sync>,
}

impl std::fmt::Debug for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

impl Quantity {
    pub fn one() -> Quantity {
        Quantity::Constant(1)
    }

    /// `above` when `value` exceeds `limit`, else `within`.
    pub fn exceeds(value: Quantity, limit: u64, above: u64, within: u64) -> Quantity {
        let apply = move |values: &[u64]| match values {
            [value] => Ok(if *value > limit { above } else { within }),
            _ => Err("rule `exceeds` reads one quantity".to_string()),
        };
        Quantity::Rule(Rule { name: "exceeds", apply: Arc::new(apply) }, vec![value])
    }

    pub fn product(factors: impl IntoIterator<Item = Quantity>) -> Quantity {
        Quantity::Product(factors.into_iter().collect())
    }

    pub fn eval(&self, site: &dyn Fn(SiteId) -> Option<i64>) -> Result<u64, String> {
        let value = |id: SiteId| -> Result<u64, String> {
            site(id).and_then(|v| u64::try_from(v).ok()).filter(|&v| v > 0).ok_or_else(|| format!("site {} has no positive value in scope", id.0))
        };
        let overflow = || "derived quantity overflows u64".to_string();
        match self {
            Quantity::Constant(c) => Ok(*c),
            Quantity::Site(id) => value(*id),
            Quantity::Pieces { extent, site } => Ok(extent.div_ceil(value(*site)?)),
            Quantity::Symbolic(s) => {
                let mut ranges = BTreeMap::new();
                for name in s.sym.params() {
                    let range = match (s.constants.get(&name), s.ranges.get(&name)) {
                        (Some(c), _) => (*c, *c),
                        (None, Some((lo, hi))) => {
                            let bound = |q: &Quantity| q.eval(site).and_then(|v| i64::try_from(v).map_err(|_| "derived quantity exceeds i64".to_string()));
                            (bound(lo)?, bound(hi)?)
                        }
                        (None, None) => return Err(format!("extent `{}` is not static: `{name}` has no static range", s.sym)),
                    };
                    ranges.insert(name, range);
                }
                let value = if ranges.values().all(|(lo, hi)| lo == hi) {
                    s.sym.eval(&|name: &str| ranges.get(name).map(|r| r.0))
                } else {
                    s.sym.eval_interval(&|name: &str| ranges.get(name).copied()).map(|(lo, hi)| if s.lower { lo } else { hi })
                };
                value.map(|v| u64::try_from(v).unwrap_or(0)).ok_or_else(|| format!("extent `{}` has no static {} bound", s.sym, if s.lower { "lower" } else { "upper" }))
            }
            Quantity::Product(items) => items.iter().try_fold(1u64, |acc, q| acc.checked_mul(q.eval(site)?).ok_or_else(overflow)),
            Quantity::Sum(items) => items.iter().try_fold(0u64, |acc, q| acc.checked_add(q.eval(site)?).ok_or_else(overflow)),
            Quantity::Max(items) => items.iter().try_fold(0u64, |acc, q| Ok(acc.max(q.eval(site)?))),
            Quantity::Predecessor(n) => Ok(n.eval(site)?.saturating_sub(1)),
            Quantity::Quotient(value, divisor) => Ok(value.eval(site)?.div_ceil((*divisor).max(1))),
            Quantity::Aligned(value, unit) => {
                let unit = (*unit).max(1);
                value.eval(site)?.div_ceil(unit).checked_mul(unit).ok_or_else(overflow)
            }
            Quantity::Rule(rule, arguments) => (rule.apply)(&arguments.iter().map(|q| q.eval(site)).collect::<Result<Vec<_>, _>>()?),
            Quantity::Bounded { exact, bound } => exact.eval(site).or_else(|_| bound.eval(site)),
            Quantity::Unknown(reason) => Err(reason.clone()),
        }
    }

    pub fn sites(&self, out: &mut BTreeSet<SiteId>) {
        match self {
            Quantity::Constant(_) | Quantity::Unknown(_) => {}
            Quantity::Site(id) | Quantity::Pieces { site: id, .. } => {
                out.insert(*id);
            }
            Quantity::Symbolic(s) => {
                for name in s.sym.params() {
                    if let Some((lo, hi)) = s.ranges.get(&name) {
                        lo.sites(out);
                        hi.sites(out);
                    }
                }
            }
            Quantity::Product(items) | Quantity::Sum(items) | Quantity::Max(items) | Quantity::Rule(_, items) => items.iter().for_each(|q| q.sites(out)),
            Quantity::Predecessor(n) | Quantity::Quotient(n, _) | Quantity::Aligned(n, _) => n.sites(out),
            Quantity::Bounded { exact, bound } => {
                exact.sites(out);
                bound.sites(out);
            }
        }
    }
}

/// Scope of several quantities, ascending, and a lookup from the tabulated value slice.
pub fn scope<'a>(quantities: impl IntoIterator<Item = &'a Quantity>) -> Vec<SiteId> {
    let mut out = BTreeSet::new();
    quantities.into_iter().for_each(|q| q.sites(&mut out));
    out.into_iter().collect()
}

pub fn lookup<'a>(scope: &'a [SiteId], values: &'a [i64]) -> impl Fn(SiteId) -> Option<i64> + 'a {
    move |id| scope.binary_search(&id).ok().and_then(|i| values.get(i).copied())
}
