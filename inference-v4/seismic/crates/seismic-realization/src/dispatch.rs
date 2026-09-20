//! The one common linear/runtime iteration geometry.
//!
//! Independent axes use the universal `LinearIterationMap`: retained extents,
//! an overflow-checked row-major total (the exact runtime product retained
//! for runtime domains, with the checked capacity bound), the physical linear
//! participant count (a planning expression), one-pass or grid-stride
//! traversal, delinearization for every logical axis, and a tail mask.
//! Logical rank is not native grid rank. Ordered axes are ascending serial
//! loops inside each independent point. Zero work is a retained launch
//! condition skipped by the runtime; zero native grids are never submitted.
//! Authored tile authority is deleted; only this mapping exists.

use seismic_lang::{
    logical::{RuntimeExtent, RuntimeScalarExpr},
    sym::Sym,
    types::{ExtentExpr, RuntimeExtentId},
};
use std::collections::BTreeMap;

/// How participants traverse the linear domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Traversal {
    /// Participant count equals the total; each participant visits exactly
    /// one linear coordinate (no tail).
    OnePass,
    /// A fixed participant count covers the domain by grid stride; the tail
    /// mask skips participants beyond the total.
    GridStride,
}

/// The overflow-checked row-major total of the retained extents.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearTotal {
    /// Every axis extent is static: the exact total.
    Static(u64),
    Runtime {
        /// Retained runtime expression: the exact semantic total.
        product: RuntimeScalarExpr,
        /// Checked product of the axis capacities; the semantic total never
        /// exceeds it, and its fit was proved during construction.
        capacity: u64,
        /// The workload's expected total, when every runtime axis states an
        /// expected extent. Cost only; never geometry, resources, or semantics.
        expected: Option<u64>,
    },
}

impl LinearTotal {
    /// The exact total, when every axis is static.
    pub fn as_static(&self) -> Option<u64> {
        match self {
            LinearTotal::Static(n) => Some(*n),
            LinearTotal::Runtime { .. } => None,
        }
    }

    /// The resource/tuning bound of the total: exact for static domains, the
    /// checked capacity product for runtime domains (semantics always use
    /// the retained runtime product, never this bound).
    pub fn bound(&self) -> u64 {
        match self {
            LinearTotal::Static(n) => *n,
            LinearTotal::Runtime { capacity, .. } => *capacity,
        }
    }

    /// The total to price work by: exact for static domains, the workload's
    /// expected total for runtime domains that state one, else the bound.
    pub fn expected(&self) -> u64 {
        match self {
            LinearTotal::Static(n) => *n,
            LinearTotal::Runtime {
                expected, capacity, ..
            } => expected.unwrap_or(*capacity),
        }
    }
}

/// The retained launch condition of one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchCondition {
    /// The domain is statically empty: no native dispatch is submitted.
    AlwaysSkip,
    Execute,
}

/// Why a linear iteration map could not be constructed. Overflow makes the
/// alternative infeasible; it never wraps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearMapError {
    /// A static row-major product overflowed.
    StaticOverflow,
    /// The checked capacity product overflowed, so the runtime total cannot
    /// be bounded.
    CapacityOverflow,
    /// An unresolved symbolic extent survived specialization.
    UnresolvedSymbol,
}

impl std::fmt::Display for LinearMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearMapError::StaticOverflow => {
                f.write_str("static row-major iteration total overflows")
            }
            LinearMapError::CapacityOverflow => {
                f.write_str("runtime extent capacities overflow the iteration total")
            }
            LinearMapError::UnresolvedSymbol => {
                f.write_str("an unresolved symbolic extent survived specialization")
            }
        }
    }
}
impl std::error::Error for LinearMapError {}

/// The universal arbitrary-rank iteration map.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearIterationMap {
    /// Retained logical axis extents, outermost first (last axis fastest).
    pub extents: Vec<ExtentExpr>,
    /// Overflow-checked row-major total.
    pub total: LinearTotal,
    pub traversal: Traversal,
    /// Whether participants beyond the total must be masked.
    pub tail_mask: bool,
    /// The containing independent domain is serialized (one participant):
    /// the universal atomic-add strategy.
    pub serialized: bool,
    /// The physical linear participant count: a planning expression over
    /// tuning parameters. Strategies set it after `linear()`/`serialized()`.
    pub participants: Sym,
}

impl LinearIterationMap {
    /// The universal map over `extents` (outermost first). The total is
    /// overflow-checked; a runtime domain retains its exact runtime product
    /// together with the checked capacity bound. Participants default to one
    /// (the serial participant domain); grid-stride strategies set the
    /// planning expression with `with_participants`.
    pub fn linear(
        extents: &[ExtentExpr],
        runtime_extents: &BTreeMap<RuntimeExtentId, RuntimeExtent>,
    ) -> Result<LinearIterationMap, LinearMapError> {
        let mut product: Option<RuntimeScalarExpr> = None;
        let mut capacity = 1u64;
        let mut expected: Option<u64> = Some(1);
        let mut all_static = true;
        for extent in extents {
            let (value_expr, factor_capacity, factor_expected, factor_is_static) = match extent {
                ExtentExpr::Static(n) => {
                    let n = *n;
                    (RuntimeScalarExpr::Const(n as i64), n, Some(n), true)
                }
                ExtentExpr::Runtime(id) => {
                    let runtime = runtime_extents
                        .get(id)
                        .ok_or(LinearMapError::UnresolvedSymbol)?;
                    (
                        RuntimeScalarExpr::Extent(*id),
                        runtime.capacity,
                        runtime.expected,
                        false,
                    )
                }
                ExtentExpr::Sym(sym) => {
                    let constant = sym.as_constant().ok_or(LinearMapError::UnresolvedSymbol)?;
                    let n = u64::try_from(constant).map_err(|_| LinearMapError::StaticOverflow)?;
                    (RuntimeScalarExpr::Const(constant), n, Some(n), true)
                }
            };
            product = Some(match product {
                // Canonical form: no leading unit factor.
                None => value_expr,
                Some(acc) => RuntimeScalarExpr::Mul(Box::new(acc), Box::new(value_expr)),
            });
            // The expected total exists only when every runtime axis states one;
            // an overflowing expectation is no expectation.
            expected = match (expected, factor_expected) {
                (Some(acc), Some(factor)) => acc.checked_mul(factor),
                _ => None,
            };
            capacity = capacity.checked_mul(factor_capacity).ok_or_else(|| {
                // An overflow of static factors is a static overflow; a
                // runtime factor makes the capacity bound unprovable.
                if all_static && factor_is_static {
                    LinearMapError::StaticOverflow
                } else {
                    LinearMapError::CapacityOverflow
                }
            })?;
            all_static &= factor_is_static;
        }
        let total = if all_static {
            LinearTotal::Static(capacity)
        } else {
            LinearTotal::Runtime {
                product: product.unwrap_or(RuntimeScalarExpr::Const(1)),
                capacity,
                expected,
            }
        };
        Ok(LinearIterationMap {
            extents: extents.to_vec(),
            total,
            traversal: Traversal::GridStride,
            tail_mask: true,
            serialized: false,
            participants: Sym::constant(1),
        })
    }

    /// A serialized domain: exactly one participant traverses everything
    /// (the universal atomic-add strategy serializes the containing
    /// independent domain). Built from an already-checked map of the same
    /// extents.
    pub fn serialized(map: &LinearIterationMap) -> LinearIterationMap {
        LinearIterationMap {
            extents: map.extents.clone(),
            total: map.total.clone(),
            traversal: Traversal::OnePass,
            tail_mask: false,
            serialized: true,
            participants: Sym::constant(1),
        }
    }

    /// The empty serial map: one participant, one visit (the participant
    /// domain of a single mapped node whose extents are governed by the
    /// enclosing structure).
    pub fn serial() -> LinearIterationMap {
        LinearIterationMap {
            extents: Vec::new(),
            total: LinearTotal::Static(1),
            traversal: Traversal::OnePass,
            tail_mask: false,
            serialized: true,
            participants: Sym::constant(1),
        }
    }

    /// Set the physical linear participant count (a planning expression).
    pub fn with_participants(mut self, participants: Sym) -> LinearIterationMap {
        self.participants = participants;
        self
    }

    /// The retained launch condition: zero work is skipped by the runtime and
    /// zero native grids are never submitted.
    pub fn launch_condition(&self) -> LaunchCondition {
        match self.total.as_static() {
            Some(0) => LaunchCondition::AlwaysSkip,
            _ => LaunchCondition::Execute,
        }
    }

    /// Delinearize one row-major coordinate into per-axis logical
    /// coordinates (outermost first, last axis fastest). `None` when the
    /// coordinate lies outside a static total (runtime totals delinearize
    /// against their runtime values at execution).
    pub fn delinearize(&self, linear: u64) -> Option<Vec<u64>> {
        let total = self.total.as_static()?;
        if linear >= total {
            return None;
        }
        let rank = self.extents.len();
        let mut coordinates = vec![0u64; rank];
        let mut rest = linear;
        for (axis, coordinate) in coordinates.iter_mut().enumerate() {
            let stride: u64 = self.extents[axis + 1..]
                .iter()
                .map(|extent| extent.as_static())
                .try_fold(1u64, |acc, n| Some(acc.checked_mul(n?)?))?;
            *coordinate = rest / stride;
            rest %= stride;
        }
        Some(coordinates)
    }

    /// The work-item total as a planning expression for geometry and
    /// resources: exact for static domains, the checked capacity bound for
    /// runtime domains (the resolved launch retains the exact runtime product).
    pub fn total_symbol(&self) -> Sym {
        match i64::try_from(self.total.bound()) {
            Ok(bound) => Sym::constant(bound),
            Err(_) => Sym::constant(i64::MAX),
        }
    }

    /// The work-item total to price cost by: the workload's expected total
    /// for a runtime domain that states one, else `total_symbol`. Never used
    /// for geometry or resources.
    pub fn cost_symbol(&self) -> Sym {
        match i64::try_from(self.total.expected()) {
            Ok(expected) => Sym::constant(expected),
            Err(_) => Sym::constant(i64::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delinearizes_arbitrary_rank() {
        let map = LinearIterationMap::linear(
            &[
                ExtentExpr::Static(4),
                ExtentExpr::Static(3),
                ExtentExpr::Static(5),
            ],
            &BTreeMap::new(),
        )
        .expect("the map builds");
        assert_eq!(map.delinearize(0).unwrap(), vec![0, 0, 0]);
        // Row-major: the last axis is fastest.
        assert_eq!(map.delinearize(7).unwrap(), vec![0, 1, 2]);
        assert_eq!(map.delinearize(59).unwrap(), vec![3, 2, 4]);
        assert_eq!(map.delinearize(60), None);
    }

    #[test]
    fn grid_stride_and_one_pass_cover_the_same_domain() {
        let extents = vec![ExtentExpr::Static(7), ExtentExpr::Static(3)];
        let grid_stride = LinearIterationMap::linear(&extents, &BTreeMap::new())
            .expect("the map builds")
            .with_participants(Sym::constant(8));
        assert_eq!(grid_stride.traversal, Traversal::GridStride);
        assert!(grid_stride.tail_mask);
        let mut strided = Vec::new();
        for participant in 0..8u64 {
            let mut linear = participant;
            while linear < 21 {
                strided.push(grid_stride.delinearize(linear).unwrap());
                linear += 8;
            }
        }
        let one_pass = LinearIterationMap::serialized(&grid_stride);
        assert_eq!(one_pass.traversal, Traversal::OnePass);
        assert!(!one_pass.tail_mask);
        assert!(one_pass.serialized);
        let mut ascending = (0..21)
            .map(|l| one_pass.delinearize(l).unwrap())
            .collect::<Vec<_>>();
        let mut sorted = strided.clone();
        sorted.sort();
        ascending.sort();
        assert_eq!(sorted, ascending);
    }

    #[test]
    fn zero_work_is_a_retained_launch_condition() {
        let map = LinearIterationMap::linear(&[ExtentExpr::Static(0)], &BTreeMap::new())
            .expect("the map builds");
        assert_eq!(map.launch_condition(), LaunchCondition::AlwaysSkip);
        assert_eq!(map.delinearize(0), None);
        let map = LinearIterationMap::linear(&[ExtentExpr::Static(4)], &BTreeMap::new())
            .expect("the map builds");
        assert_eq!(map.launch_condition(), LaunchCondition::Execute);
    }

    #[test]
    fn runtime_domains_retain_values_and_bound_capacities() {
        let id = RuntimeExtentId(0);
        let runtime = RuntimeExtent {
            id,
            value: RuntimeScalarExpr::Extent(id),
            capacity: 4096,
            expected: None,
        };
        let map = LinearIterationMap::linear(
            &[ExtentExpr::Runtime(id)],
            &BTreeMap::from([(id, runtime)]),
        )
        .expect("the map builds");
        // Planning/resource accounting sees the checked capacity bound…
        assert_eq!(map.total_symbol().as_constant(), Some(4096));
        // …while the retained total stays the exact runtime product.
        assert!(matches!(map.total, LinearTotal::Runtime { .. }));
        // Delinearization of a runtime domain is an execution-time act.
        assert_eq!(map.delinearize(0), None);
    }

    #[test]
    fn overflow_is_an_error_never_a_wrap() {
        let huge = ExtentExpr::Static(u64::MAX);
        let map = LinearIterationMap::linear(&[huge, ExtentExpr::Static(2)], &BTreeMap::new());
        assert_eq!(map.unwrap_err(), LinearMapError::StaticOverflow);
        let id = RuntimeExtentId(0);
        let runtime = RuntimeExtent {
            id,
            value: RuntimeScalarExpr::Extent(id),
            capacity: u64::MAX,
            expected: None,
        };
        let map = LinearIterationMap::linear(
            &[ExtentExpr::Runtime(id), ExtentExpr::Runtime(id)],
            &BTreeMap::from([(id, runtime)]),
        );
        assert_eq!(map.unwrap_err(), LinearMapError::CapacityOverflow);
        let map = LinearIterationMap::linear(
            &[ExtentExpr::Sym(Sym::param("unresolved"))],
            &BTreeMap::new(),
        );
        assert_eq!(map.unwrap_err(), LinearMapError::UnresolvedSymbol);
    }
}
