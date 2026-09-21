//! Total route tables (package D1).
//!
//! A `RouteTable` maps every non-local semantic leaf of a strategy to
//! exactly one exhaustive route. Kernel-local SSA values do not appear. A
//! computed tensor crossing a cut receives one compiler-owned spill
//! residence; a tensor view derives its route from its canonical storage
//! residence plus its transform. Aliasing is two routes naming one
//! `ResidenceId`; there is no alias map.

use crate::ids::{CanonicalLeafId, ExecutorScalarSlotId, ResidenceId, ResultFieldIx};
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::logical::boundary::BoundaryLeaf;
use seismic_lang::logical::Access;
use seismic_lang::types::ExtentExpr;
use std::collections::BTreeMap;

/// One scalar transported between schedule components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarRoute {
    /// A scalar of the root ABI (input leaf, with range endpoint).
    RootAbi { leaf: BoundaryLeaf, endpoint: Option<RangeEndpoint> },
    /// A planned executor scalar slot written by a preceding launch.
    ExecutorSlot(ExecutorScalarSlotId),
    /// A field of the compiler-owned result scalar block.
    ResultField { leaf: BoundaryLeaf, endpoint: Option<RangeEndpoint> },
}

/// One tensor route: the residence, access, and complete view transform in
/// residence coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRoute {
    pub residence: ResidenceId,
    pub access: Access,
    pub transform: ViewTransformTemplate,
}

/// The exhaustive route of one semantic leaf. The route table is keyed by
/// canonical leaf, and the canonical leaf registry (O1) already expands
/// tuples and ranges into leaves, so no tuple route exists. A void value has
/// no leaf; `Void` is retained for exhaustive consumers but D1 never routes a
/// void leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueRoute {
    Void,
    Scalar(ScalarRoute),
    Tensor(TensorRoute),
}

/// A composed view transform with occurrence-qualified dynamic endpoints
/// resolved to canonical leaves. Steps apply outermost first: `steps[0]`
/// is applied to the residence's own coordinates, and every later step to
/// the result of the previous one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewTransformTemplate {
    pub steps: Vec<ViewStepTemplate>,
}

impl ViewTransformTemplate {
    /// The identity transform: the value is the residence itself.
    pub fn identity() -> ViewTransformTemplate {
        ViewTransformTemplate { steps: Vec::new() }
    }

    /// `outer` applied to the residence first, then `inner` to its result
    /// (the caller's argument view followed by the callee's local view).
    pub fn compose(outer: &ViewTransformTemplate, inner: &ViewTransformTemplate) -> ViewTransformTemplate {
        ViewTransformTemplate {
            steps: outer.steps.iter().chain(&inner.steps).cloned().collect(),
        }
    }

    /// Every dynamic slice endpoint of the transform, in step and axis
    /// order. Each is a scalar canonical leaf with its own route.
    pub fn dynamic_endpoints(&self) -> Vec<CanonicalLeafId> {
        let mut out = Vec::new();
        for step in &self.steps {
            match &step.kind {
                ViewStepKind::Reshape | ViewStepKind::Transpose { .. } => {}
                ViewStepKind::Slice { axes } => {
                    for axis in axes {
                        match axis {
                            SliceAxisTemplate::Full => {}
                            SliceAxisTemplate::Point(leaf) => out.push(*leaf),
                            SliceAxisTemplate::Range { start, end } => {
                                out.extend(start.iter().copied());
                                out.extend(end.iter().copied());
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewStepTemplate {
    pub source_shape: Vec<ExtentExpr>,
    pub kind: ViewStepKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewStepKind {
    Reshape,
    Transpose { permutation: Vec<u32> },
    Slice { axes: Vec<SliceAxisTemplate> },
}

/// Slice axes are full, point, or optional-start/optional-end range. Every
/// dynamic endpoint is a canonical scalar leaf with its own route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SliceAxisTemplate {
    Full,
    Point(CanonicalLeafId),
    Range { start: Option<CanonicalLeafId>, end: Option<CanonicalLeafId> },
}

/// The total route table of one strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteTable {
    routes: BTreeMap<CanonicalLeafId, ValueRoute>,
}

impl RouteTable {
    pub(crate) fn seal(routes: BTreeMap<CanonicalLeafId, ValueRoute>) -> RouteTable {
        RouteTable { routes }
    }

    /// Total over the strategy's non-local leaves. D1 guarantees
    /// `domain(RouteTable) = every nonlocal semantic leaf required by the
    /// shape`; a lookup outside that domain is a defect of the caller.
    pub fn route(&self, leaf: CanonicalLeafId) -> &ValueRoute {
        &self.routes[&leaf]
    }

    /// Whether `leaf` is a non-local leaf of the strategy (routed). A
    /// kernel-local SSA leaf answers `false`; this is the one legitimate
    /// membership query (K1 uses it to separate interface values from SSA).
    pub fn contains(&self, leaf: CanonicalLeafId) -> bool {
        self.routes.contains_key(&leaf)
    }

    pub fn len(&self) -> usize {
        self.routes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (CanonicalLeafId, &ValueRoute)> + '_ {
        self.routes.iter().map(|(leaf, route)| (*leaf, route))
    }
}

/// Where a published scalar result lands after the physical seal (dense).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultFieldRef(pub ResultFieldIx);
