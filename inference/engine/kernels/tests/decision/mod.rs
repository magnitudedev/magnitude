//! Whether one implementation's decisions agree with a reference's: the one
//! owner of every check of a decision output across implementations (a
//! native against a host model or its portable body, a backend against
//! another). A test binary that checks one declares this module beside
//! `common`. Its own tests run once, in `shape_rows` (`decision_rules`),
//! not in every binary that declares it.
//!
//! A decision (a token kept or removed, an expert or a token chosen) is a
//! discontinuous function of a computed quantity, and two implementations
//! round that quantity differently: a token whose weight lies at its cut can
//! fall on either side in either. Bit identity is not a requirement
//! (`docs/precision.md`), so a decision may differ from the reference's only
//! where its deciding quantity lies within its rounding scale of the boundary
//! it is decided against. Everywhere else it is the reference's: a token far
//! from every cut that flips, or a lower-valued token kept while a
//! higher-valued one is removed beyond rounding, is a defect.
//!
//! Each check derives its quantities and their scales from its own
//! reference: the rounding-scale estimate the precision document allows
//! kernel tests, never admission's `r <= α` and never a declaration of a
//! backend's math. Tuning admission is no such check: it compares a candidate
//! with its default on one device, where every decision is identical by
//! construction and a difference is a defect.

#![allow(dead_code)]

use crate::common::bound::{Float, gamma};

/// A deciding quantity as the reference computes it, with its rounding
/// scale: the most two implementations' values of it may differ by rounding
/// alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantity {
    pub value: f64,
    pub scale: f64,
}

impl Quantity {
    /// An item no implementation may choose: it is close to nothing.
    pub const INELIGIBLE: Quantity = Quantity {
        value: f64::NEG_INFINITY,
        scale: 0.0,
    };

    /// A value every implementation holds exactly: a parameter, an input.
    pub fn exact(value: f64) -> Quantity {
        Quantity { value, scale: 0.0 }
    }

    /// A finite `value` each implementation forms from exact inputs by
    /// `operations` F32 operations, each rounding a result of about its
    /// magnitude ([`operation`]). A value not finite is every
    /// implementation's alike.
    pub fn rounded(value: f64, operations: usize) -> Quantity {
        match operations {
            0 => Quantity::exact(value),
            _ => Quantity::carried(value, (operations - 1) as f64 * operation(value)),
        }
    }

    /// A finite `value` formed by one F32 operation from inputs whose
    /// scales move it by `carried`: their scales through the operation's
    /// slope. A value not finite is every implementation's alike.
    pub fn carried(value: f64, carried: f64) -> Quantity {
        if value.is_finite() {
            Quantity {
                value,
                scale: carried + operation(value),
            }
        } else {
            Quantity::exact(value)
        }
    }

    /// Whether rounding can order `self` and `other` either way: they lie
    /// within their scales together. Two values held exactly are close to
    /// nothing, each other included, and no value is close to one not
    /// finite.
    pub fn close(self, other: Quantity) -> bool {
        (self.value - other.value).abs() < self.scale + other.scale
    }
}

/// The F32 rounding step at `value`: the gap between the representable
/// values around it (at most `2u|value|`), and the subnormal step near zero.
pub fn step(value: f64) -> f64 {
    2.0 * (Float::F32.unit() * value.abs() + Float::F32.underflow())
}

/// The scale one F32 operation adds to its result `value`, a step on each
/// side: a correctly rounded operation moves it by half a step, a library
/// function (`exp`, `log`) by about one.
pub fn operation(value: f64) -> f64 {
    2.0 * step(value)
}

/// The scale of an F32 sum of `count` terms whose magnitudes sum to
/// `magnitude` between two summation orders: each is within `γ_count`
/// times the magnitude of the exact sum ([`gamma`]).
pub fn summed(count: usize, magnitude: f64) -> f64 {
    2.0 * gamma(count) * magnitude
}

/// One cut an item is decided against: it passes while its quantity stays
/// on the reference's side of the boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cut {
    pub quantity: Quantity,
    pub boundary: Quantity,
}

impl Cut {
    /// Whether rounding can put the item on either side of this cut.
    pub fn close(self) -> bool {
        self.quantity.close(self.boundary)
    }
}

/// The items `actual` selects otherwise than `reference` beyond rounding.
/// An item is selected while it passes the cuts the reference decided it by
/// (`cuts(item)`: each stage's it reached, the one that removed it last), so
/// its selection may differ only where one of them is close; any other
/// difference is returned.
pub fn selected(reference: &[bool], actual: &[bool], cuts: impl Fn(usize) -> Vec<Cut>) -> Vec<usize> {
    assert_eq!(reference.len(), actual.len(), "one selection per item");
    (0..reference.len())
        .filter(|&item| reference[item] != actual[item] && !cuts(item).into_iter().any(Cut::close))
        .collect()
}

/// The positions `actual` chooses otherwise than `reference` beyond
/// rounding: position j may hold another item only where its score
/// (`scores`) is close to the reference's choice's. An item outside
/// `scores` (a negative index read as `usize::MAX`) is close to nothing.
pub fn chosen(scores: &[Quantity], reference: &[usize], actual: &[usize]) -> Vec<usize> {
    assert_eq!(reference.len(), actual.len(), "one choice per position");
    let score = |item: usize| scores.get(item).copied().unwrap_or(Quantity::INELIGIBLE);
    (0..reference.len())
        .filter(|&position| {
            let (expected, chose) = (reference[position], actual[position]);
            chose != expected && !score(chose).close(score(expected))
        })
        .collect()
}

/// Whether the `k` largest of `quantities` are settled: the k-th and the
/// (k + 1)-th are not close, so every implementation selects the same
/// items and a check may require them exactly.
pub fn settled(quantities: &[Quantity], k: usize) -> bool {
    if quantities.len() <= k || k == 0 {
        return true;
    }
    let mut order = quantities.to_vec();
    order.sort_by(|a, b| b.value.total_cmp(&a.value));
    !order[k - 1].close(order[k])
}
