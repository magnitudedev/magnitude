//! Logical initialization regions and checked function transfers.
//!
//! The source checker constructs function contracts. Compiler construction
//! applies those same contracts to its actual bindings. This module owns the
//! one coordinate calculus; neither consumer may invent whole-root permission.
use crate::check::{prove, xfer};
use crate::expr::{AnyExpr, BoolExpr, ExprArena, IntExpr, NodeView, SymbolId};
use crate::intrinsics::AtomicOp;
use crate::span::Span;
use crate::syntax::ast::BinaryOp;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ParameterPath {
    pub(crate) parameter: usize,
    pub(crate) fields: Vec<usize>,
}
impl ParameterPath {
    pub(crate) fn root(parameter: usize) -> Self {
        Self {
            parameter,
            fields: vec![],
        }
    }
    pub(crate) fn child(&self, field: usize) -> Self {
        let mut path = self.clone();
        path.fields.push(field);
        path
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Condition {
    Constant(bool),
    Parameter(ParameterPath),
    Version(u64, Vec<SymbolId>),
    Actual(BoolExpr, Vec<SymbolId>),
    Compare(BinaryOp, IntExpr, IntExpr),
    Not(Box<Condition>),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
}
pub(crate) type Path = Vec<(Condition, bool)>;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Bound {
    pub(crate) symbol: SymbolId,
    pub(crate) start: IntExpr,
    pub(crate) end: IntExpr,
}

/// Elements of one storage root. Images bind their coordinates jointly; the
/// row-major forms are derived from the root's checked axes, never from
/// matching byte sizes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Region {
    Empty,
    Full,
    /// Root elements whose logical root coordinates are `coordinates` (one
    /// per root axis) for every assignment of `domain`. Every access builds
    /// one; a reshape view's coordinates carry its quotient/remainder
    /// decomposition.
    Image {
        domain: Vec<Bound>,
        coordinates: Vec<IntExpr>,
    },
    /// Row-major element interval `[start, end)` of the root.
    Linear(IntExpr, IntExpr),
    /// Row-major root addresses for every assignment of `domain`.
    LinearImage {
        domain: Vec<Bound>,
        address: IntExpr,
    },
    Union(Vec<Region>),
    Intersection(Vec<Region>),
    Bind(Bound, Box<Region>),
    Guard(Path, Box<Region>),
}
impl Region {
    pub(crate) fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Full, _) | (_, Self::Full) => Self::Full,
            (Self::Empty, b) => b,
            (a, Self::Empty) => a,
            (a, b) if a == b => a,
            (Self::Union(mut a), Self::Union(b)) => {
                a.extend(b);
                Self::Union(a)
            }
            (Self::Union(mut a), b) => {
                a.push(b);
                Self::Union(a)
            }
            (a, Self::Union(mut b)) => {
                b.insert(0, a);
                Self::Union(b)
            }
            (a, b) => Self::Union(vec![a, b]),
        }
    }
    pub(crate) fn intersection(self, other: Self) -> Self {
        match (self, other) {
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
            (Self::Full, b) => b,
            (a, Self::Full) => a,
            (a, b) if a == b => a,
            (a, b) => Self::Intersection(vec![a, b]),
        }
    }
    fn guarded(path: Path, region: Self) -> Self {
        match region {
            Self::Empty => Self::Empty,
            region if path.is_empty() => region,
            region => Self::Guard(path, Box::new(region)),
        }
    }
    /// The top-level members of a union.
    fn members(self) -> Vec<Self> {
        match self {
            Self::Empty => vec![],
            Self::Union(parts) => parts,
            other => vec![other],
        }
    }
    /// The region after a join on `condition`: `then` where it held and `els`
    /// where it did not. Members common to both arms stay unguarded, so the
    /// region grows linearly in the number of joins.
    pub(crate) fn branch(condition: &Condition, then: Self, els: Self) -> Self {
        if then == els {
            return then;
        }
        let then_members = then.members();
        let else_members = els.members();
        let (common, then_only): (Vec<_>, Vec<_>) = then_members
            .into_iter()
            .partition(|member| else_members.contains(member));
        let else_only = else_members
            .into_iter()
            .filter(|member| !common.contains(member))
            .collect();
        let arm = |truth: bool, members: Vec<Region>| {
            Region::guarded(
                vec![(condition.clone(), truth)],
                members.into_iter().fold(Region::Empty, Region::union),
            )
        };
        common
            .into_iter()
            .fold(Region::Empty, Region::union)
            .union(arm(true, then_only))
            .union(arm(false, else_only))
    }
}

/// Ordered-loop visit separation, L25 (I1)-(I3): accesses of distinct visits
/// to storage live at loop entry are separated whenever one writes, no local
/// live at entry is rebound, and no access is atomic. `Unrecorded` exists
/// only while checking: the recording pass meets every analysed entry world
/// into it. A loop left `Unrecorded` is dead under path facts and lowers as
/// `Unproven`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisitSeparation {
    Unrecorded,
    Separated,
    Unproven,
}
impl VisitSeparation {
    /// Meet one analysed entry world's proof into the loop's fact.
    pub(crate) fn record(&mut self, separated: bool) {
        *self = match (*self, separated) {
            (Self::Unproven, _) | (_, false) => Self::Unproven,
            (Self::Unrecorded | Self::Separated, true) => Self::Separated,
        };
    }
}

pub(crate) trait RegionOps {
    fn arena(&mut self) -> &mut ExprArena;
    fn arena_ref(&self) -> &ExprArena;
    /// A fresh integer variable of this arena: a proof variable in a
    /// definition arena, a region binder in an entry arena.
    fn fresh_variable(&mut self) -> SymbolId;
    fn substitute(&mut self, value: IntExpr, map: &HashMap<SymbolId, IntExpr>) -> IntExpr {
        prove::substitute(self.arena(), value, &|s| map.get(&s).copied())
    }
    fn same(&self, a: IntExpr, b: IntExpr) -> bool {
        prove::same(self.arena_ref(), a, b)
    }
    fn le(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        prove::le(self.arena(), facts, a, b)
    }
    fn lt(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        prove::lt(self.arena(), facts, a, b)
    }

    fn normalize(&mut self, region: Region, facts: &prove::Facts) -> Region {
        match region {
            Region::Image {
                domain,
                coordinates,
            } => self.normalize_image(domain, coordinates, facts),
            Region::LinearImage { domain, address } => {
                self.normalize_linear(domain, address, facts)
            }
            Region::Guard(path, inner) => Region::guarded(path, self.normalize(*inner, facts)),
            Region::Linear(start, end) if self.le(facts, end, start) => Region::Empty,
            Region::Union(parts) => {
                let joined = parts.into_iter().fold(Region::Empty, |a, b| {
                    let b = self.normalize(b, facts);
                    a.union(b)
                });
                let Region::Union(mut parts) = joined else {
                    return joined;
                };
                // Join adjacent/overlapping intervals before a coverage
                // query. A zero-length loop prefix can meet an initialized
                // seed without requiring a spurious strict inequality.
                let mut i = 0;
                while i < parts.len() {
                    let mut j = i + 1;
                    while j < parts.len() {
                        if let (Region::Linear(a, b), Region::Linear(c, d)) = (&parts[i], &parts[j])
                        {
                            let (a, b, c, d) = (*a, *b, *c, *d);
                            if self.le(facts, a, c) && self.le(facts, c, b) && self.le(facts, b, d)
                            {
                                parts[i] = Region::Linear(a, d);
                                parts.remove(j);
                                j = i + 1;
                                continue;
                            }
                            if self.le(facts, c, a) && self.le(facts, a, d) && self.le(facts, d, b)
                            {
                                parts[i] = Region::Linear(c, b);
                                parts.remove(j);
                                j = i + 1;
                                continue;
                            }
                        }
                        j += 1;
                    }
                    i += 1;
                }
                self.join_images(&mut parts, facts);
                merge_complementary_guards(&mut parts);
                if parts.len() == 1 {
                    parts.pop().unwrap()
                } else {
                    Region::Union(parts)
                }
            }
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |a, b| {
                let b = self.normalize(b, facts);
                a.intersection(b)
            }),
            Region::Bind(bound, body) => {
                if self.le(facts, bound.end, bound.start) {
                    return Region::Empty;
                }
                let inner = self.normalize(*body, facts);
                match inner {
                    Region::Empty => Region::Empty,
                    Region::Union(parts) => {
                        let parts = parts
                            .into_iter()
                            .map(|p| {
                                self.normalize(Region::Bind(bound.clone(), Box::new(p)), facts)
                            })
                            .collect();
                        Region::Union(parts)
                    }
                    Region::Image {
                        mut domain,
                        coordinates,
                    } => {
                        domain.insert(0, bound);
                        self.normalize_image(domain, coordinates, facts)
                    }
                    Region::LinearImage {
                        mut domain,
                        address,
                    } => {
                        domain.insert(0, bound);
                        self.normalize_linear(domain, address, facts)
                    }
                    Region::Linear(start, end) => {
                        let (symbol, index) = self.fresh_integer();
                        let zero = self.arena().int(0);
                        let width = self.arena().int_sub(end, start);
                        let address = self.arena().int_add(start, index);
                        self.normalize_linear(
                            vec![
                                bound,
                                Bound {
                                    symbol,
                                    start: zero,
                                    end: width,
                                },
                            ],
                            address,
                            facts,
                        )
                    }
                    Region::Guard(path, inner)
                        if path
                            .iter()
                            .all(|(c, _)| !self.condition_mentions(c, bound.symbol)) =>
                    {
                        Region::Guard(
                            path,
                            Box::new(self.normalize(Region::Bind(bound, inner), facts)),
                        )
                    }
                    Region::Full if self.lt(facts, bound.start, bound.end) => Region::Full,
                    other => Region::Bind(bound, Box::new(other)),
                }
            }
            other => other,
        }
    }
    /// Normalize an image axis by axis. When every coordinate mentions its
    /// own domain symbols only, the image is the product of the coordinate
    /// images, and a coordinate that enumerates a dense range becomes one
    /// symbol over that range: loop completion of `b + off` over
    /// `b in [lo, hi)` is the axis range `[lo + off, hi + off)`.
    fn normalize_image(
        &mut self,
        domain: Vec<Bound>,
        coordinates: Vec<IntExpr>,
        facts: &prove::Facts,
    ) -> Region {
        if domain.iter().any(|d| self.le(facts, d.end, d.start)) {
            return Region::Empty;
        }
        let normalized = self.rebased_image(domain.clone(), coordinates.clone(), facts);
        // Rebasing names fresh symbols. An image that is already normal stays
        // the identical region, so equal members remain equal across joins.
        match normalized {
            Region::Image {
                domain: rebased,
                coordinates: rebased_coordinates,
            } if self.same_image(&rebased, &rebased_coordinates, &domain, &coordinates) => {
                Region::Image {
                    domain,
                    coordinates,
                }
            }
            other => other,
        }
    }
    /// Whether two images enumerate the same coordinates, up to the names
    /// of their domain symbols.
    fn same_image(
        &mut self,
        domain: &[Bound],
        coordinates: &[IntExpr],
        other_domain: &[Bound],
        other_coordinates: &[IntExpr],
    ) -> bool {
        if domain.len() != other_domain.len() || coordinates.len() != other_coordinates.len() {
            return false;
        }
        let mut map = HashMap::new();
        for (bound, other) in domain.iter().zip(other_domain) {
            let start = self.substitute(bound.start, &map);
            let end = self.substitute(bound.end, &map);
            if !self.same(start, other.start) || !self.same(end, other.end) {
                return false;
            }
            let renamed = self.arena().int_symbol(other.symbol);
            map.insert(bound.symbol, renamed);
        }
        coordinates.iter().zip(other_coordinates).all(|(x, y)| {
            let x = self.substitute(*x, &map);
            self.same(x, *y)
        })
    }
    fn rebased_image(
        &mut self,
        domain: Vec<Bound>,
        coordinates: Vec<IntExpr>,
        facts: &prove::Facts,
    ) -> Region {
        // Rebase every symbol to start at zero: `s in [a, b)` is `a + r` for
        // a fresh `r in [0, b - a)`, so no symbol takes a new meaning. A
        // width-one symbol is its start.
        let mut substitutions = HashMap::new();
        let mut dimensions = vec![];
        for d in domain {
            let start = self.substitute(d.start, &substitutions);
            let end = self.substitute(d.end, &substitutions);
            let width = self.arena().int_sub(end, start);
            let width = prove::canonical(self.arena(), width);
            if prove::constant(self.arena(), width) == Some(1) {
                substitutions.insert(d.symbol, start);
            } else {
                let symbol = if prove::is_zero(self.arena_ref(), start) {
                    d.symbol
                } else {
                    let (symbol, offset) = self.fresh_integer();
                    let rebased = self.arena().int_add(start, offset);
                    substitutions.insert(d.symbol, rebased);
                    symbol
                };
                let zero = self.arena().int(0);
                dimensions.push(Bound {
                    symbol,
                    start: zero,
                    end: width,
                });
            }
        }
        let coordinates = coordinates
            .into_iter()
            .map(|c| {
                let c = self.substitute(c, &substitutions);
                prove::recompose_address(self.arena(), c)
            })
            .collect::<Vec<_>>();
        let mentioned = coordinates
            .iter()
            .map(|c| {
                (0..dimensions.len())
                    .filter(|d| prove::mentions(self.arena_ref(), *c, dimensions[*d].symbol))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut owner = vec![None; dimensions.len()];
        // A width that depends on another symbol makes the domain triangular,
        // not a product.
        let mut joint = dimensions.iter().any(|d| {
            dimensions
                .iter()
                .any(|other| prove::mentions(self.arena_ref(), d.end, other.symbol))
        });
        for (axis, uses) in mentioned.iter().enumerate() {
            for d in uses {
                joint |= owner[*d].replace(axis).is_some();
            }
        }
        // A symbol no coordinate mentions only matters when its range may be
        // empty, which empties the whole image.
        let mut kept = vec![];
        for (d, bound) in dimensions.iter().enumerate() {
            if owner[d].is_none() && !self.lt(facts, bound.start, bound.end) {
                kept.push(bound.clone());
            }
        }
        if joint {
            kept.extend(
                (0..dimensions.len())
                    .filter(|d| owner[*d].is_some())
                    .map(|d| dimensions[d].clone()),
            );
            return Region::Image {
                domain: kept,
                coordinates,
            };
        }
        let mut axes = Vec::with_capacity(coordinates.len());
        for (axis, coordinate) in coordinates.into_iter().enumerate() {
            let own = mentioned[axis]
                .iter()
                .map(|d| dimensions[*d].clone())
                .collect::<Vec<_>>();
            if own.is_empty() {
                axes.push(coordinate);
                continue;
            }
            match self.dense_range(coordinate, &own, facts) {
                Some((start, end)) => {
                    // A coordinate that is its own symbol keeps it; any other
                    // dense range is enumerated by a fresh symbol.
                    let symbol = match self.arena_ref().view(AnyExpr::Int(coordinate)) {
                        NodeView::Symbol(symbol) if own.len() == 1 && own[0].symbol == symbol => {
                            symbol
                        }
                        _ => self.fresh_integer().0,
                    };
                    kept.push(Bound { symbol, start, end });
                    axes.push(self.arena().int_symbol(symbol));
                }
                None => {
                    kept.extend(own);
                    axes.push(coordinate);
                }
            }
        }
        Region::Image {
            domain: kept,
            coordinates: axes,
        }
    }
    /// `[start, end)` when `value` enumerates exactly that dense range as the
    /// mixed-radix combination of `dimensions`, which it alone mentions.
    /// `dimensions` is reordered from the least significant stride.
    fn dense_range(
        &mut self,
        value: IntExpr,
        dimensions: &[Bound],
        facts: &prove::Facts,
    ) -> Option<(IntExpr, IntExpr)> {
        let mut terms = vec![];
        let mut start_map = HashMap::new();
        for d in dimensions {
            let coefficient = prove::linear_coefficient(self.arena(), value, d.symbol)?;
            if dimensions
                .iter()
                .any(|other| prove::mentions(self.arena_ref(), coefficient, other.symbol))
                || prove::is_zero(self.arena(), coefficient)
            {
                return None;
            }
            terms.push((d.clone(), coefficient));
            start_map.insert(d.symbol, d.start);
        }
        let start = self.substitute(value, &start_map);
        let mut stride = self.arena().int(1);
        while !terms.is_empty() {
            let index = terms
                .iter()
                .position(|(_, coefficient)| self.same(*coefficient, stride))?;
            let (d, _) = terms.remove(index);
            let width = self.arena().int_sub(d.end, d.start);
            if !prove::nonneg(self.arena(), facts, width) {
                return None;
            }
            stride = self.arena().int_mul(stride, width);
        }
        let end = self.arena().int_add(start, stride);
        Some((
            prove::canonical(self.arena(), start),
            prove::canonical(self.arena(), end),
        ))
    }
    fn normalize_linear(
        &mut self,
        domain: Vec<Bound>,
        address: IntExpr,
        facts: &prove::Facts,
    ) -> Region {
        if domain.iter().any(|d| self.le(facts, d.end, d.start)) {
            return Region::Empty;
        }
        let mut substitutions = HashMap::new();
        let mut dimensions = vec![];
        for d in &domain {
            let width = self.arena().int_sub(d.end, d.start);
            if prove::constant(self.arena(), width) == Some(1) {
                substitutions.insert(d.symbol, d.start);
            } else {
                dimensions.push(d.clone());
            }
        }
        let address = self.substitute(address, &substitutions);
        let address = prove::recompose_address(self.arena(), address);
        let mut used = vec![];
        for d in &dimensions {
            if prove::mentions(self.arena_ref(), address, d.symbol) {
                used.push(d.clone());
            } else if !self.lt(facts, d.start, d.end) {
                return Region::LinearImage { domain, address };
            }
        }
        match self.dense_range(address, &used, facts) {
            Some((start, end)) => Region::Linear(start, end),
            None => Region::LinearImage { domain, address },
        }
    }
    /// The row-major address of root `coordinates` over the root `extents`.
    fn linear_address(&mut self, coordinates: &[IntExpr], extents: &[IntExpr]) -> IntExpr {
        assert_eq!(
            coordinates.len(),
            extents.len(),
            "region coordinates match the root rank"
        );
        let mut address = self.arena().int(0);
        for (coordinate, extent) in coordinates.iter().zip(extents) {
            address = self.arena().int_mul(address, *extent);
            address = self.arena().int_add(address, *coordinate);
        }
        address
    }
    /// The same elements in row-major form over the root `extents`.
    fn linearize(&mut self, region: Region, extents: &[IntExpr], facts: &prove::Facts) -> Region {
        match region {
            Region::Image {
                domain,
                coordinates,
            } => {
                let address = self.linear_address(&coordinates, extents);
                self.normalize_linear(domain, address, facts)
            }
            Region::Union(parts) => parts.into_iter().fold(Region::Empty, |a, b| {
                let b = self.linearize(b, extents, facts);
                a.union(b)
            }),
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |a, b| {
                let b = self.linearize(b, extents, facts);
                a.intersection(b)
            }),
            Region::Bind(bound, inner) => {
                let inner = self.linearize(*inner, extents, facts);
                self.normalize(Region::Bind(bound, Box::new(inner)), facts)
            }
            Region::Guard(path, inner) => {
                Region::Guard(path, Box::new(self.linearize(*inner, extents, facts)))
            }
            other => other,
        }
    }
    /// Join union members that are boxes equal on every axis but one, where
    /// their ranges meet or overlap: the box image of the joined range.
    fn join_images(&mut self, parts: &mut Vec<Region>, facts: &prove::Facts) {
        let mut ranges = parts
            .iter()
            .map(|part| match part {
                Region::Image {
                    domain,
                    coordinates,
                } => self.image_ranges(domain, coordinates, facts, true),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut i = 0;
        while i < parts.len() {
            let mut j = i + 1;
            while j < parts.len() {
                let joined = match (&ranges[i], &ranges[j]) {
                    (Some(left), Some(right)) if left.len() == right.len() => {
                        self.joined_axis(left, right, facts)
                    }
                    _ => None,
                };
                let Some((axis, start, end)) = joined else {
                    j += 1;
                    continue;
                };
                let Region::Image {
                    mut domain,
                    mut coordinates,
                } = parts[i].clone()
                else {
                    unreachable!("only images have box ranges")
                };
                let bound = match self.arena_ref().view(AnyExpr::Int(coordinates[axis])) {
                    NodeView::Symbol(symbol) => domain.iter().position(|d| d.symbol == symbol),
                    _ => None,
                };
                match bound {
                    Some(d) => {
                        domain[d].start = start;
                        domain[d].end = end;
                    }
                    None => {
                        // The replaced coordinate enumerated its axis alone:
                        // a domain symbol only it used goes with it, or an
                        // empty range of that symbol would be assumed
                        // nonempty wherever the domain becomes facts.
                        let replaced = coordinates[axis];
                        domain.retain(|d| {
                            !prove::mentions(self.arena_ref(), replaced, d.symbol)
                                || coordinates.iter().enumerate().any(|(other, c)| {
                                    other != axis && prove::mentions(self.arena_ref(), *c, d.symbol)
                                })
                        });
                        let (symbol, value) = self.fresh_integer();
                        domain.push(Bound { symbol, start, end });
                        coordinates[axis] = value;
                    }
                }
                let mut joined = ranges[i].take().expect("joined box has ranges");
                joined[axis] = (start, end);
                ranges[i] = Some(joined);
                parts[i] = Region::Image {
                    domain,
                    coordinates,
                };
                parts.remove(j);
                ranges.remove(j);
                j = i + 1;
            }
            i += 1;
        }
    }
    /// The one axis on which two boxes differ, with their joined range, when
    /// the two ranges on it meet or overlap.
    fn joined_axis(
        &mut self,
        left: &[(IntExpr, IntExpr)],
        right: &[(IntExpr, IntExpr)],
        facts: &prove::Facts,
    ) -> Option<(usize, IntExpr, IntExpr)> {
        let differing = (0..left.len())
            .filter(|k| !(self.same(left[*k].0, right[*k].0) && self.same(left[*k].1, right[*k].1)))
            .collect::<Vec<_>>();
        let [axis] = differing[..] else {
            return None;
        };
        let ((a, b), (c, d)) = (left[axis], right[axis]);
        if self.le(facts, a, c) && self.le(facts, c, b) && self.le(facts, b, d) {
            return Some((axis, a, d));
        }
        if self.le(facts, c, a) && self.le(facts, a, d) && self.le(facts, d, b) {
            return Some((axis, c, b));
        }
        None
    }
    /// Per-axis half-open ranges of an image that is the product of them.
    /// An `available` image additionally needs every unmentioned domain
    /// symbol to have a nonempty range, or it may hold no element at all.
    fn image_ranges(
        &mut self,
        domain: &[Bound],
        coordinates: &[IntExpr],
        facts: &prove::Facts,
        available: bool,
    ) -> Option<Vec<(IntExpr, IntExpr)>> {
        let dependent = domain.iter().any(|bound| {
            domain.iter().any(|other| {
                prove::mentions(self.arena_ref(), bound.start, other.symbol)
                    || prove::mentions(self.arena_ref(), bound.end, other.symbol)
            })
        });
        if dependent {
            return None;
        }
        let mut owned = vec![false; domain.len()];
        let mut ranges = vec![];
        for coordinate in coordinates {
            // A normalized image enumerates each axis range by one symbol.
            if let NodeView::Symbol(symbol) = self.arena_ref().view(AnyExpr::Int(*coordinate)) {
                if let Some(d) = domain.iter().position(|bound| bound.symbol == symbol) {
                    if std::mem::replace(&mut owned[d], true) {
                        return None;
                    }
                    ranges.push((domain[d].start, domain[d].end));
                    continue;
                }
            }
            let uses = (0..domain.len())
                .filter(|d| prove::mentions(self.arena_ref(), *coordinate, domain[*d].symbol))
                .collect::<Vec<_>>();
            match uses.as_slice() {
                [] => {
                    let one = self.arena().int(1);
                    let end = self.arena().int_add(*coordinate, one);
                    ranges.push((*coordinate, end));
                }
                [d] if !std::mem::replace(&mut owned[*d], true) => {
                    let bound = &domain[*d];
                    let (symbol, start, end) = (bound.symbol, bound.start, bound.end);
                    let coefficient = prove::linear_coefficient(self.arena(), *coordinate, symbol)?;
                    if prove::constant(self.arena(), coefficient) != Some(1) {
                        return None;
                    }
                    let zero = self.arena().int(0);
                    let offset = self.substitute(*coordinate, &HashMap::from([(symbol, zero)]));
                    let start = self.arena().int_add(start, offset);
                    let end = self.arena().int_add(end, offset);
                    ranges.push((start, end));
                }
                _ => return None,
            }
        }
        if available
            && (0..domain.len())
                .filter(|d| !owned[*d])
                .any(|d| !self.lt(facts, domain[d].start, domain[d].end))
        {
            return None;
        }
        Some(ranges)
    }
    /// Structural coverage of a required image, axis by axis.
    fn covered_image(
        &mut self,
        available: &Region,
        domain: &[Bound],
        coordinates: &[IntExpr],
        facts: &prove::Facts,
    ) -> bool {
        if let Region::Image {
            domain: a,
            coordinates: x,
        } = available
        {
            if self.same_image(a, x, domain, coordinates) {
                return true;
            }
        }
        if let Some(required) = self.image_ranges(domain, coordinates, facts, false) {
            return self.covered_ranges(available, &required, facts);
        }
        // Otherwise every element is the point at its coordinates, for every
        // assignment of the domain.
        let facts = self.domain_facts(facts, domain.iter());
        let one = self.arena().int(1);
        let points = coordinates
            .iter()
            .map(|c| (*c, self.arena().int_add(*c, one)))
            .collect::<Vec<_>>();
        self.covered_ranges(available, &points, &facts)
    }
    /// Whether the images of `available` contain the product of `required`
    /// per-axis ranges: one box does, or boxes tile the required range along
    /// one axis while containing it along every other axis.
    fn covered_ranges(
        &mut self,
        available: &Region,
        required: &[(IntExpr, IntExpr)],
        facts: &prove::Facts,
    ) -> bool {
        let parts = match available {
            Region::Union(parts) => parts.iter().collect::<Vec<_>>(),
            other => vec![other],
        };
        let mut boxes = vec![];
        for part in parts {
            if let Region::Image {
                domain,
                coordinates,
            } = part
            {
                if coordinates.len() == required.len() {
                    if let Some(ranges) = self.image_ranges(domain, coordinates, facts, true) {
                        boxes.push(ranges);
                    }
                }
            }
        }
        if required.is_empty() {
            // A rank-zero root has one element, held by any nonempty image.
            return !boxes.is_empty();
        }
        let contains =
            |owner: &mut Self, outer: &(IntExpr, IntExpr), inner: &(IntExpr, IntExpr)| {
                owner.le(facts, outer.0, inner.0) && owner.le(facts, inner.1, outer.1)
            };
        for axis in 0..required.len() {
            let mut tiles = boxes
                .iter()
                .filter(|ranges| {
                    (0..required.len())
                        .filter(|other| *other != axis)
                        .all(|other| contains(self, &ranges[other], &required[other]))
                })
                .map(|ranges| ranges[axis])
                .collect::<Vec<_>>();
            let (mut cursor, end) = required[axis];
            loop {
                if self.le(facts, end, cursor) {
                    return true;
                }
                let Some(index) = tiles
                    .iter()
                    .position(|(a, b)| self.le(facts, *a, cursor) && self.lt(facts, cursor, *b))
                else {
                    break;
                };
                cursor = tiles.remove(index).1;
            }
        }
        false
    }
    fn covered_linear(
        &mut self,
        available: &Region,
        required: &Region,
        facts: &prove::Facts,
    ) -> bool {
        if matches!(available, Region::Full)
            || matches!(required, Region::Empty)
            || available == required
        {
            return true;
        }
        if let Region::Union(parts) = required {
            return parts
                .iter()
                .all(|p| self.covered_linear(available, p, facts));
        }
        if let Region::Intersection(parts) = available {
            return parts
                .iter()
                .all(|p| self.covered_linear(p, required, facts));
        }
        match (available, required) {
            (Region::Linear(a, b), Region::Linear(c, d)) => {
                self.le(facts, *a, *c) && self.le(facts, *d, *b)
            }
            (Region::Linear(a, b), Region::LinearImage { domain, address }) => {
                let mut facts = facts.clone();
                let one = self.arena().int(1);
                for bound in domain {
                    let upper = self.arena().int_sub(bound.end, one);
                    facts.set_range(bound.symbol, bound.start, upper);
                }
                self.le(&facts, *a, *address) && self.lt(&facts, *address, *b)
            }
            (
                Region::LinearImage {
                    domain: a,
                    address: x,
                },
                Region::LinearImage {
                    domain: b,
                    address: y,
                },
            ) if a.len() == b.len() => {
                let mut map = HashMap::new();
                for (a, b) in a.iter().zip(b) {
                    map.insert(a.symbol, self.arena().int_symbol(b.symbol));
                    if !self.same(a.start, b.start) || !self.same(a.end, b.end) {
                        return false;
                    }
                }
                let x = self.substitute(*x, &map);
                self.same(x, *y)
            }
            (Region::Union(parts), _) => {
                if parts
                    .iter()
                    .any(|part| self.covered_linear(part, required, facts))
                {
                    return true;
                }
                let Region::Linear(start, end) = required else {
                    return false;
                };
                let mut cursor = *start;
                let mut remaining = parts.iter().collect::<Vec<_>>();
                loop {
                    if self.le(facts, *end, cursor) {
                        return true;
                    }
                    let Some(index) = remaining.iter().position(|part| match part {
                        Region::Linear(a, b) => {
                            self.le(facts, *a, cursor) && self.lt(facts, cursor, *b)
                        }
                        _ => false,
                    }) else {
                        return false;
                    };
                    let Region::Linear(_, next) = remaining.remove(index) else {
                        unreachable!()
                    };
                    cursor = *next;
                }
            }
            (_, Region::Intersection(parts)) => parts
                .iter()
                .any(|p| self.covered_linear(available, p, facts)),
            _ => false,
        }
    }

    fn condition_value(
        &mut self,
        condition: &Condition,
        path: &Path,
        facts: &prove::Facts,
    ) -> Option<bool> {
        if let Some((_, value)) = path.iter().rev().find(|(c, _)| c == condition) {
            return Some(*value);
        }
        match condition {
            Condition::Constant(value) => Some(*value),
            Condition::Not(c) => self.condition_value(c, path, facts).map(|v| !v),
            Condition::And(a, b) => match (
                self.condition_value(a, path, facts),
                self.condition_value(b, path, facts),
            ) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            Condition::Or(a, b) => match (
                self.condition_value(a, path, facts),
                self.condition_value(b, path, facts),
            ) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            Condition::Compare(op, a, b) => {
                use BinaryOp::*;
                match op {
                    Eq if self.same(*a, *b) => Some(true),
                    Ne if self.same(*a, *b) => Some(false),
                    Eq if self.lt(facts, *a, *b) || self.lt(facts, *b, *a) => Some(false),
                    Ne if self.lt(facts, *a, *b) || self.lt(facts, *b, *a) => Some(true),
                    Lt if self.lt(facts, *a, *b) => Some(true),
                    Lt if self.le(facts, *b, *a) => Some(false),
                    Le if self.le(facts, *a, *b) => Some(true),
                    Le if self.lt(facts, *b, *a) => Some(false),
                    Gt => self.condition_value(&Condition::Compare(Lt, *b, *a), path, facts),
                    Ge => self.condition_value(&Condition::Compare(Le, *b, *a), path, facts),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    fn assume(
        &mut self,
        path: &mut Path,
        facts: &mut prove::Facts,
        condition: Condition,
        value: bool,
    ) -> bool {
        if let Some(actual) = self.condition_value(&condition, path, facts) {
            return actual == value;
        }
        match &condition {
            Condition::Not(c) => return self.assume(path, facts, *c.clone(), !value),
            Condition::And(a, b) if value => {
                if !self.assume(path, facts, *a.clone(), true)
                    || !self.assume(path, facts, *b.clone(), true)
                {
                    return false;
                }
            }
            Condition::Or(a, b) if !value => {
                if !self.assume(path, facts, *a.clone(), false)
                    || !self.assume(path, facts, *b.clone(), false)
                {
                    return false;
                }
            }
            Condition::Compare(op, a, b) => {
                let (a, b, strict, equality) = match (op, value) {
                    (BinaryOp::Lt, true) | (BinaryOp::Ge, false) => (*b, *a, true, false),
                    (BinaryOp::Le, true) | (BinaryOp::Gt, false) => (*b, *a, false, false),
                    (BinaryOp::Gt, true) | (BinaryOp::Le, false) => (*a, *b, true, false),
                    (BinaryOp::Ge, true) | (BinaryOp::Lt, false) => (*a, *b, false, false),
                    (BinaryOp::Eq, true) | (BinaryOp::Ne, false) => (*a, *b, false, true),
                    _ => {
                        path.push((condition, value));
                        return true;
                    }
                };
                let mut difference = self.arena().int_sub(a, b);
                if strict {
                    let one = self.arena().int(1);
                    difference = self.arena().int_sub(difference, one);
                }
                let zero = self.arena().int(0);
                let negative = self.arena().int_sub(zero, difference);
                if equality || prove::nonneg(self.arena_ref(), facts, negative) {
                    // A nonpositive checked extent on an empty path is zero,
                    // including a quotient such as H/KV. Keep that equality
                    // in the same index facts even when it is not a linear
                    // bound on one named dimension.
                    facts.assume_zero(self.arena(), difference);
                    facts.assume_zero(self.arena(), negative);
                    facts.assume_nonnegative(self.arena(), negative);
                }
                facts.assume_nonnegative(self.arena(), difference);
            }
            _ => {}
        }
        path.push((condition, value));
        true
    }
    fn unknown_guard(
        &mut self,
        region: &Region,
        path: &Path,
        facts: &prove::Facts,
    ) -> Option<Condition> {
        match region {
            Region::Guard(guard, inner) => {
                for (condition, value) in guard {
                    match self.condition_value(condition, path, facts) {
                        Some(actual) if actual != *value => return None,
                        None => return Some(condition.clone()),
                        _ => {}
                    }
                }
                self.unknown_guard(inner, path, facts)
            }
            Region::Union(parts) | Region::Intersection(parts) => parts
                .iter()
                .find_map(|p| self.unknown_guard(p, path, facts)),
            Region::Bind(_, inner) => self.unknown_guard(inner, path, facts),
            _ => None,
        }
    }
    fn active_region(&mut self, region: Region, path: &Path, facts: &prove::Facts) -> Region {
        match region {
            Region::Guard(guard, inner) => {
                if guard
                    .iter()
                    .all(|(c, v)| self.condition_value(c, path, facts) == Some(*v))
                {
                    self.active_region(*inner, path, facts)
                } else {
                    Region::Empty
                }
            }
            Region::Union(parts) => parts.into_iter().fold(Region::Empty, |a, b| {
                a.union(self.active_region(b, path, facts))
            }),
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |a, b| {
                a.intersection(self.active_region(b, path, facts))
            }),
            other => other,
        }
    }
    /// Whether `available` contains `required`, both elements of one root
    /// with `extents`, on every execution with `path` and `facts`.
    fn covered(
        &mut self,
        available: &Region,
        required: &Region,
        path: &Path,
        facts: &prove::Facts,
        extents: &[IntExpr],
    ) -> bool {
        if let Some(condition) = self
            .unknown_guard(available, path, facts)
            .or_else(|| self.unknown_guard(required, path, facts))
        {
            return [false, true].into_iter().all(|value| {
                let mut path = path.clone();
                let mut facts = facts.clone();
                !self.assume(&mut path, &mut facts, condition.clone(), value)
                    || self.covered(available, required, &path, &facts, extents)
            });
        }
        let available = self.active_region(available.clone(), path, facts);
        let required = self.active_region(required.clone(), path, facts);
        let available = self.normalize(available, facts);
        let required = self.normalize(required, facts);
        self.covered_plain(&available, &required, facts, extents)
    }
    fn covered_plain(
        &mut self,
        available: &Region,
        required: &Region,
        facts: &prove::Facts,
        extents: &[IntExpr],
    ) -> bool {
        if matches!(available, Region::Full)
            || matches!(required, Region::Empty)
            || available == required
        {
            return true;
        }
        if let Region::Union(parts) = required {
            return parts
                .iter()
                .all(|p| self.covered_plain(available, p, facts, extents));
        }
        if let Region::Intersection(parts) = available {
            return parts
                .iter()
                .all(|p| self.covered_plain(p, required, facts, extents));
        }
        if let Region::Intersection(parts) = required {
            if parts
                .iter()
                .any(|p| self.covered_plain(available, p, facts, extents))
            {
                return true;
            }
        }
        if matches!(available, Region::Empty) {
            return matches!(required, Region::Full)
                && extents.iter().any(|extent| {
                    let zero = self.arena().int(0);
                    self.le(facts, *extent, zero)
                });
        }
        let structural = match required {
            // Coverage normalization: images spanning every root axis are
            // the whole root.
            Region::Full => {
                let zero = self.arena().int(0);
                let root = extents
                    .iter()
                    .map(|extent| (zero, *extent))
                    .collect::<Vec<_>>();
                self.covered_ranges(available, &root, facts)
            }
            Region::Image {
                domain,
                coordinates,
            } => self.covered_image(available, domain, coordinates, facts),
            _ => false,
        };
        // Images of a root of rank at most one are already row-major.
        if structural || extents.len() <= 1 && !row_major(available) && !row_major(required) {
            return structural;
        }
        let available = self.linearize(available.clone(), extents, facts);
        let available = self.normalize(available, facts);
        let required = match required {
            Region::Full => {
                let zero = self.arena().int(0);
                let elements = extents
                    .iter()
                    .fold(self.arena().int(1), |p, e| self.arena().int_mul(p, *e));
                Region::Linear(zero, elements)
            }
            required => {
                let required = self.linearize(required.clone(), extents, facts);
                self.normalize(required, facts)
            }
        };
        self.covered_linear(&available, &required, facts)
    }
    /// Whether no element of `left` is an element of `right`, both elements
    /// of one root with `extents`. Images are separated when some axis has
    /// separated coordinate intervals; any pair involving a row-major form
    /// compares row-major intervals.
    fn separated_regions(
        &mut self,
        left: Region,
        right: Region,
        facts: &prove::Facts,
        extents: &[IntExpr],
    ) -> bool {
        let left = self.normalize(left, facts);
        let right = self.normalize(right, facts);
        match (left, right) {
            (Region::Empty, _) | (_, Region::Empty) => true,
            (Region::Union(parts), right) => parts
                .into_iter()
                .all(|part| self.separated_regions(part, right.clone(), facts, extents)),
            (left, Region::Union(parts)) => parts
                .into_iter()
                .all(|part| self.separated_regions(left.clone(), part, facts, extents)),
            (Region::Intersection(parts), right) => parts
                .into_iter()
                .any(|part| self.separated_regions(part, right.clone(), facts, extents)),
            (left, Region::Intersection(parts)) => parts
                .into_iter()
                .any(|part| self.separated_regions(left.clone(), part, facts, extents)),
            (Region::Guard(_, inner), right) => {
                self.separated_regions(*inner, right, facts, extents)
            }
            (left, Region::Guard(_, inner)) => self.separated_regions(left, *inner, facts, extents),
            (Region::Full, _) | (_, Region::Full) => false,
            (
                Region::Image {
                    domain: a,
                    coordinates: x,
                },
                Region::Image {
                    domain: b,
                    coordinates: y,
                },
            ) if {
                let facts = self.domain_facts(facts, a.iter().chain(b.iter()));
                x.iter()
                    .zip(y.iter())
                    .any(|(x, y)| self.lt(&facts, *x, *y) || self.lt(&facts, *y, *x))
            } =>
            {
                true
            }
            (left, right) => {
                // Images of a root of rank at most one are already row-major.
                if extents.len() <= 1 && !row_major(&left) && !row_major(&right) {
                    return false;
                }
                let left = self.linearize(left, extents, facts);
                let right = self.linearize(right, extents, facts);
                let (Some(left), Some(right)) = (self.linear_span(&left), self.linear_span(&right))
                else {
                    return false;
                };
                let facts = self.domain_facts(facts, left.domain.iter().chain(right.domain));
                self.le(&facts, left.end, right.start) || self.le(&facts, right.end, left.start)
            }
        }
    }
    /// `facts` with every symbol of `domain` bounded by its range.
    fn domain_facts<'b>(
        &mut self,
        facts: &prove::Facts,
        domain: impl Iterator<Item = &'b Bound>,
    ) -> prove::Facts {
        let mut facts = facts.clone();
        let one = self.arena().int(1);
        for bound in domain {
            let upper = self.arena().int_sub(bound.end, one);
            facts.set_range(bound.symbol, bound.start, upper);
        }
        facts
    }
    /// A row-major form as `(domain, [start, end))`.
    fn linear_span<'r>(&mut self, region: &'r Region) -> Option<LinearSpan<'r>> {
        match region {
            Region::Linear(start, end) => Some(LinearSpan {
                domain: &[],
                start: *start,
                end: *end,
            }),
            Region::LinearImage { domain, address } => {
                let one = self.arena().int(1);
                let end = self.arena().int_add(*address, one);
                Some(LinearSpan {
                    domain,
                    start: *address,
                    end,
                })
            }
            _ => None,
        }
    }
    fn condition_mentions(&self, condition: &Condition, symbol: SymbolId) -> bool {
        match condition {
            Condition::Version(_, binders) | Condition::Actual(_, binders) => {
                binders.contains(&symbol)
            }
            Condition::Compare(_, a, b) => {
                prove::mentions(self.arena_ref(), *a, symbol)
                    || prove::mentions(self.arena_ref(), *b, symbol)
            }
            Condition::Not(c) => self.condition_mentions(c, symbol),
            Condition::And(a, b) | Condition::Or(a, b) => {
                self.condition_mentions(a, symbol) || self.condition_mentions(b, symbol)
            }
            _ => false,
        }
    }
    fn view_region(&mut self, place: &InitializationView) -> Region {
        let zero = self.arena().int(0);
        Region::Image {
            domain: place
                .coordinates
                .iter()
                .zip(&place.axes)
                .map(|(symbol, end)| Bound {
                    symbol: *symbol,
                    start: zero,
                    end: *end,
                })
                .collect(),
            coordinates: place.root.clone(),
        }
    }
    fn select_view(
        &mut self,
        place: &InitializationView,
        selections: &[(Option<IntExpr>, Option<IntExpr>, bool)],
    ) -> InitializationView {
        let mut map = HashMap::new();
        let mut axes = vec![];
        let mut coordinates = vec![];
        let zero = self.arena().int(0);
        for (axis, (&coordinate, &extent)) in place.coordinates.iter().zip(&place.axes).enumerate()
        {
            let (start, end, point) = selections.get(axis).copied().unwrap_or((None, None, false));
            let start = start.unwrap_or(zero);
            if point {
                map.insert(coordinate, start);
            } else {
                let end = end.unwrap_or(extent);
                let width = self.arena().int_sub(end, start);
                let (symbol, index) = self.fresh_integer();
                coordinates.push(symbol);
                axes.push(width);
                let value = self.arena().int_add(start, index);
                map.insert(coordinate, value);
            }
        }
        InitializationView {
            axes,
            coordinates,
            root: place
                .root
                .iter()
                .map(|value| self.substitute(*value, &map))
                .collect(),
            extents: place.extents.clone(),
        }
    }
    fn reshape_view(&mut self, place: &InitializationView, axes: &[IntExpr]) -> InitializationView {
        let mut coordinates = vec![];
        let mut linear = self.arena().int(0);
        for &extent in axes {
            let (symbol, index) = self.fresh_integer();
            coordinates.push(symbol);
            linear = self.arena().int_mul(linear, extent);
            linear = self.arena().int_add(linear, index);
        }
        InitializationView {
            axes: axes.to_vec(),
            coordinates,
            root: self.view_coordinates_at(place, linear),
            extents: place.extents.clone(),
        }
    }
    /// Root coordinates of the view element at row-major view offset `linear`.
    fn view_coordinates_at(
        &mut self,
        place: &InitializationView,
        mut linear: IntExpr,
    ) -> Vec<IntExpr> {
        let mut map = HashMap::new();
        for (axis, (&symbol, &extent)) in
            place.coordinates.iter().zip(&place.axes).enumerate().rev()
        {
            // The flat coordinate is already checked against the complete
            // view domain. Its most significant coordinate needs no modulo.
            let coordinate = if axis == 0 {
                linear
            } else {
                self.arena().int_rem(linear, extent)
            };
            map.insert(symbol, coordinate);
            linear = self.arena().int_div(linear, extent);
        }
        place
            .root
            .iter()
            .map(|value| self.substitute(*value, &map))
            .collect()
    }
    fn root_view(&mut self, axes: &[IntExpr]) -> InitializationView {
        let mut coordinates = vec![];
        let mut root = vec![];
        for _ in axes {
            let (symbol, index) = self.fresh_integer();
            coordinates.push(symbol);
            root.push(index);
        }
        InitializationView {
            axes: axes.to_vec(),
            coordinates,
            root,
            extents: axes.to_vec(),
        }
    }
    fn boundary_condition(&self, condition: &Condition, allowed: &[SymbolId]) -> bool {
        match condition {
            Condition::Version(..) => false,
            Condition::Actual(_, binders) => binders.iter().all(|binder| allowed.contains(binder)),
            Condition::Compare(_, a, b) => [a, b].into_iter().all(|e| {
                prove::symbols(self.arena_ref(), *e)
                    .iter()
                    .all(|s| allowed.contains(s))
            }),
            Condition::Not(c) => self.boundary_condition(c, allowed),
            Condition::And(a, b) | Condition::Or(a, b) => {
                self.boundary_condition(a, allowed) && self.boundary_condition(b, allowed)
            }
            _ => true,
        }
    }
    fn boundary_paths(&self, path: &Path, allowed: &[SymbolId]) -> Vec<Path> {
        fn combine(left: Vec<Path>, right: Vec<Path>) -> Vec<Path> {
            left.into_iter()
                .flat_map(|a| {
                    right.iter().map(move |b| {
                        let mut p = a.clone();
                        p.extend(b.clone());
                        p
                    })
                })
                .collect()
        }
        fn condition<T: RegionOps + ?Sized>(
            owner: &T,
            c: &Condition,
            truth: bool,
            allowed: &[SymbolId],
        ) -> Vec<Path> {
            match (c, truth) {
                (Condition::Not(c), truth) => condition(owner, c, !truth, allowed),
                (Condition::And(a, b), true) | (Condition::Or(a, b), false) => combine(
                    condition(owner, a, truth, allowed),
                    condition(owner, b, truth, allowed),
                ),
                (Condition::And(a, b), false) | (Condition::Or(a, b), true) => {
                    let mut paths = condition(owner, a, truth, allowed);
                    paths.extend(condition(owner, b, truth, allowed));
                    paths
                }
                _ if owner.boundary_condition(c, allowed) => vec![vec![(c.clone(), truth)]],
                _ => vec![vec![]],
            }
        }
        path.iter().fold(vec![vec![]], |paths, (c, v)| {
            combine(paths, condition(self, c, *v, allowed))
        })
    }
    fn boundary_region(&mut self, region: Region, allowed: &[SymbolId], required: bool) -> Region {
        let unknown = if required {
            Region::Full
        } else {
            Region::Empty
        };
        let known = |owner: &Self, e: IntExpr, allowed: &[SymbolId]| {
            prove::symbols(owner.arena_ref(), e)
                .iter()
                .all(|s| allowed.contains(s))
        };
        let domain_known = |owner: &Self, domain: &[Bound], scope: &mut Vec<SymbolId>| {
            for bound in domain {
                if !known(owner, bound.start, scope) || !known(owner, bound.end, scope) {
                    return false;
                }
                scope.push(bound.symbol);
            }
            true
        };
        match region {
            Region::Linear(a, b) if !known(self, a, allowed) || !known(self, b, allowed) => unknown,
            Region::Image {
                domain,
                coordinates,
            } => {
                let mut scope = allowed.to_vec();
                if domain_known(self, &domain, &mut scope)
                    && coordinates.iter().all(|c| known(self, *c, &scope))
                {
                    Region::Image {
                        domain,
                        coordinates,
                    }
                } else {
                    unknown
                }
            }
            Region::LinearImage { domain, address } => {
                let mut scope = allowed.to_vec();
                if domain_known(self, &domain, &mut scope) && known(self, address, &scope) {
                    Region::LinearImage { domain, address }
                } else {
                    unknown
                }
            }
            Region::Bind(bound, inner) => {
                if !known(self, bound.start, allowed) || !known(self, bound.end, allowed) {
                    return unknown;
                }
                let mut scope = allowed.to_vec();
                scope.push(bound.symbol);
                Region::Bind(
                    bound,
                    Box::new(self.boundary_region(*inner, &scope, required)),
                )
            }
            Region::Guard(path, inner) => {
                let inner = self.boundary_region(*inner, allowed, required);
                if required {
                    self.boundary_paths(&path, allowed)
                        .into_iter()
                        .fold(Region::Empty, |state, path| {
                            state.union(Region::Guard(path, Box::new(inner.clone())))
                        })
                } else if path
                    .iter()
                    .all(|(c, _)| self.boundary_condition(c, allowed))
                {
                    Region::Guard(path, Box::new(inner))
                } else {
                    Region::Empty
                }
            }
            Region::Union(parts) => parts.into_iter().fold(Region::Empty, |s, r| {
                s.union(self.boundary_region(r, allowed, required))
            }),
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |s, r| {
                s.intersection(self.boundary_region(r, allowed, required))
            }),
            other => other,
        }
    }
    fn close_transfer(
        &mut self,
        contract: InitializationContract,
        allowed: &[SymbolId],
    ) -> InitializationContract {
        let mut accesses = vec![];
        for access in contract.accesses {
            // Accesses are a may-set: losing a private coordinate or predicate
            // must widen the footprint, never erase a possible conflict.
            let region = self.boundary_region(access.region, allowed, true);
            for path in self.boundary_paths(&access.path, allowed) {
                accesses.push(ParameterAccess {
                    parameter: access.parameter.clone(),
                    region: region.clone(),
                    path,
                    write: access.write,
                    atomic: access.atomic,
                });
            }
        }
        let mut requirements = vec![];
        for requirement in contract.requirements {
            let region = self.boundary_region(requirement.region, &allowed, true);
            for path in self.boundary_paths(&requirement.path, &allowed) {
                requirements.push(Requirement {
                    parameter: requirement.parameter.clone(),
                    region: region.clone(),
                    path,
                    span: requirement.span,
                });
            }
        }
        let mut writes = HashMap::<ParameterPath, Region>::new();
        for exit in contract.exits {
            let paths = self.boundary_paths(&exit.path, &allowed);
            for (parameter, region) in exit.written {
                let region = self.boundary_region(region, &allowed, false);
                for path in &paths {
                    // Every private successful outcome compatible with this
                    // public path must guarantee the exported write. Outside
                    // its public path this outcome imposes no restriction.
                    let mut implication = Region::Guard(path.clone(), Box::new(region.clone()));
                    for (condition, truth) in path {
                        implication = implication.union(Region::Guard(
                            vec![(condition.clone(), !*truth)],
                            Box::new(Region::Full),
                        ));
                    }
                    let prior = writes.remove(&parameter).unwrap_or(Region::Full);
                    writes.insert(parameter.clone(), prior.intersection(implication));
                }
            }
        }
        let mut written = writes.into_iter().collect::<Vec<_>>();
        written.sort_by(|a, b| a.0.cmp(&b.0));
        InitializationContract {
            requirements,
            exits: vec![Exit {
                path: vec![],
                written,
            }],
            symbols: contract.symbols,
            accesses,
        }
    }
    /// A region of the root a view denotes, in the root coordinates of that
    /// view's own storage root.
    fn map_view_region(&mut self, region: Region, view: &InitializationView) -> Region {
        match region {
            Region::Empty => Region::Empty,
            Region::Full => self.view_region(view),
            Region::Image {
                domain,
                coordinates,
            } => {
                assert_eq!(
                    coordinates.len(),
                    view.coordinates.len(),
                    "region coordinates match the view rank"
                );
                let map = view
                    .coordinates
                    .iter()
                    .copied()
                    .zip(coordinates)
                    .collect::<HashMap<_, _>>();
                Region::Image {
                    domain,
                    coordinates: view
                        .root
                        .iter()
                        .map(|value| self.substitute(*value, &map))
                        .collect(),
                }
            }
            Region::Linear(start, end) => {
                let (symbol, index) = self.fresh_integer();
                Region::Image {
                    domain: vec![Bound { symbol, start, end }],
                    coordinates: self.view_coordinates_at(view, index),
                }
            }
            Region::LinearImage { domain, address } => Region::Image {
                domain,
                coordinates: self.view_coordinates_at(view, address),
            },
            Region::Union(parts) => Region::Union(
                parts
                    .into_iter()
                    .map(|r| self.map_view_region(r, view))
                    .collect(),
            ),
            Region::Intersection(parts) => Region::Intersection(
                parts
                    .into_iter()
                    .map(|r| self.map_view_region(r, view))
                    .collect(),
            ),
            Region::Bind(bound, inner) => {
                Region::Bind(bound, Box::new(self.map_view_region(*inner, view)))
            }
            Region::Guard(path, inner) => {
                Region::Guard(path, Box::new(self.map_view_region(*inner, view)))
            }
        }
    }
    fn shift_region(&mut self, region: Region, offset: IntExpr) -> Region {
        match region {
            Region::Linear(a, b) => Region::Linear(
                self.arena().int_sub(a, offset),
                self.arena().int_sub(b, offset),
            ),
            Region::LinearImage { domain, address } => Region::LinearImage {
                domain,
                address: self.arena().int_sub(address, offset),
            },
            Region::Image { .. } => unreachable!("only row-major regions are shifted"),
            Region::Union(parts) => Region::Union(
                parts
                    .into_iter()
                    .map(|r| self.shift_region(r, offset))
                    .collect(),
            ),
            Region::Intersection(parts) => Region::Intersection(
                parts
                    .into_iter()
                    .map(|r| self.shift_region(r, offset))
                    .collect(),
            ),
            Region::Bind(bound, inner) => {
                Region::Bind(bound, Box::new(self.shift_region(*inner, offset)))
            }
            Region::Guard(path, inner) => {
                Region::Guard(path, Box::new(self.shift_region(*inner, offset)))
            }
            other => other,
        }
    }
    /// A root region seen through a view, in the view's own row-major
    /// coordinates. Exact for a whole view and for a view that is one
    /// contiguous row-major window of its root; otherwise nothing is known.
    fn project_view_region(
        &mut self,
        region: Region,
        view: &InitializationView,
        path: &Path,
        facts: &prove::Facts,
    ) -> Region {
        let domain = self.view_region(view);
        if self.covered(&region, &domain, path, facts, &view.extents) {
            return Region::Full;
        }
        let (symbol, linear) = self.fresh_integer();
        let coordinates = self.view_coordinates_at(view, linear);
        let address = self.linear_address(&coordinates, &view.extents);
        let address = prove::recompose_address(self.arena(), address);
        let Some(coefficient) = prove::linear_coefficient(self.arena(), address, symbol) else {
            return Region::Empty;
        };
        if prove::constant(self.arena_ref(), coefficient) != Some(1) {
            return Region::Empty;
        }
        let zero = self.arena().int(0);
        let offset = self.substitute(address, &HashMap::from([(symbol, zero)]));
        let region = self.linearize(region, &view.extents, facts);
        self.shift_region(region, offset)
    }
    fn fresh_integer(&mut self) -> (SymbolId, IntExpr) {
        let symbol = self.fresh_variable();
        let value = self.arena().int_symbol(symbol);
        (symbol, value)
    }
}

/// A row-major form: the addresses `[start, end)` for every assignment of
/// `domain`.
pub(crate) struct LinearSpan<'r> {
    domain: &'r [Bound],
    start: IntExpr,
    end: IntExpr,
}

/// Whether a region mentions a row-major form.
fn row_major(region: &Region) -> bool {
    match region {
        Region::Linear(..) | Region::LinearImage { .. } => true,
        Region::Union(parts) | Region::Intersection(parts) => parts.iter().any(row_major),
        Region::Bind(_, inner) | Region::Guard(_, inner) => row_major(inner),
        Region::Empty | Region::Full | Region::Image { .. } => false,
    }
}

/// Coverage normalization: `Guard(p ∧ c, R) ∪ Guard(p ∧ ¬c, R)` is
/// `Guard(p, R)`, and a guarded copy of an unguarded member adds nothing.
fn merge_complementary_guards(parts: &mut Vec<Region>) {
    fn guard(region: &Region) -> (&[(Condition, bool)], &Region) {
        match region {
            Region::Guard(path, inner) => (path, inner),
            other => (&[], other),
        }
    }
    fn complement(left: &[(Condition, bool)], right: &[(Condition, bool)]) -> Option<Path> {
        if left.len() != right.len() {
            return None;
        }
        for (index, (condition, truth)) in left.iter().enumerate() {
            let Some(opposite) = right.iter().position(|(c, t)| c == condition && t != truth)
            else {
                continue;
            };
            let mut rest = left.to_vec();
            rest.remove(index);
            let mut other = right.to_vec();
            other.remove(opposite);
            if rest.len() == other.len() && rest.iter().all(|entry| other.contains(entry)) {
                return Some(rest);
            }
        }
        None
    }
    let mut changed = true;
    while changed {
        changed = false;
        'search: for i in 0..parts.len() {
            for j in 0..parts.len() {
                if i == j {
                    continue;
                }
                let (left_path, left) = guard(&parts[i]);
                let (right_path, right) = guard(&parts[j]);
                if left != right {
                    continue;
                }
                if left_path.is_empty() {
                    parts.remove(j);
                    changed = true;
                    break 'search;
                }
                if let Some(path) = complement(left_path, right_path) {
                    parts[i] = Region::guarded(path, left.clone());
                    parts.remove(j);
                    changed = true;
                    break 'search;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Requirement {
    pub(crate) parameter: ParameterPath,
    pub(crate) region: Region,
    pub(crate) path: Path,
    pub(crate) span: Span,
}
/// A may-access of a formal tensor. Unlike an initialization requirement or
/// guaranteed exit write, this includes reads of already initialized data and
/// writes on any reachable source path. Calls instantiate it at their actual
/// argument views when checking independent visits.
#[derive(Clone, Debug)]
pub(crate) struct ParameterAccess {
    pub(crate) parameter: ParameterPath,
    pub(crate) region: Region,
    pub(crate) path: Path,
    pub(crate) write: bool,
    /// The combining operation of an atomic access. Only same-operation
    /// atomic accesses commute (L12).
    pub(crate) atomic: Option<AtomicOp>,
}
#[derive(Clone, Debug)]
pub(crate) enum ParameterPart {
    Integer(ParameterPath),
    Start(ParameterPath),
    End(ParameterPath),
}
#[derive(Clone, Debug)]
pub(crate) struct Exit {
    pub(crate) path: Path,
    pub(crate) written: Vec<(ParameterPath, Region)>,
}
#[derive(Clone, Debug)]
pub struct InitializationContract {
    pub(crate) requirements: Vec<Requirement>,
    pub(crate) exits: Vec<Exit>,
    pub(crate) symbols: Vec<(SymbolId, ParameterPart)>,
    pub(crate) accesses: Vec<ParameterAccess>,
}

impl InitializationContract {
    pub(crate) fn empty() -> Self {
        Self {
            requirements: vec![],
            exits: vec![],
            symbols: vec![],
            accesses: vec![],
        }
    }
}

/// The checked loop's captured writes and owned carry invariants. Both are
/// derived by source checking and instantiated on the loop's actual parameters.
#[derive(Clone, Debug)]
pub struct LoopInitialization {
    pub(crate) transfer: InitializationContract,
    pub(crate) carried: Vec<(ParameterPath, Region)>,
    pub(crate) binder: ParameterPath,
}

impl LoopInitialization {
    pub(crate) fn empty(binder: ParameterPath) -> Self {
        Self {
            transfer: InitializationContract::empty(),
            carried: Vec::new(),
            binder,
        }
    }
    pub(crate) fn remap<'a>(
        &self,
        source: &'a ExprArena,
        target: &mut ExprArena,
        leaves: &[(usize, Vec<usize>, usize)],
        map: &mut xfer::SymbolMap<'a>,
    ) -> Self {
        let mut mapping = EntryMapping {
            source,
            target,
            map,
            bound: HashMap::new(),
            leaves,
        };
        for (symbol, _) in &self.transfer.symbols {
            mapping.binder(*symbol);
        }
        Self {
            transfer: mapping.contract(&self.transfer),
            carried: self
                .carried
                .iter()
                .map(|(parameter, region)| (mapping.parameter(parameter), mapping.region(region)))
                .collect(),
            binder: mapping.parameter(&self.binder),
        }
    }
}

/// A checked logical view's coordinate map into its original storage root.
/// Storage identity remains on the actual compiler binding.
#[derive(Clone, Debug, PartialEq)]
pub struct InitializationView {
    pub(crate) axes: Vec<IntExpr>,
    pub(crate) coordinates: Vec<SymbolId>,
    /// The root coordinates of the view element at `coordinates`, one per
    /// root axis.
    pub(crate) root: Vec<IntExpr>,
    /// The axes of the storage root.
    pub(crate) extents: Vec<IntExpr>,
}
/// Initialized logical coordinates of one actual storage root.
#[derive(Clone, Debug)]
pub struct InitializationState(Region);
impl InitializationState {
    pub fn empty() -> Self {
        Self(Region::Empty)
    }
    pub fn full() -> Self {
        Self(Region::Full)
    }
    pub fn union(&self, other: &Self) -> Self {
        Self(self.0.clone().union(other.0.clone()))
    }
    pub fn intersection(&self, other: &Self) -> Self {
        Self(self.0.clone().intersection(other.0.clone()))
    }
}
#[derive(Clone, Debug)]
pub enum InitializationArgument {
    Tensor {
        state: InitializationState,
        view: InitializationView,
    },
    Integer(IntExpr),
    Range {
        start: IntExpr,
        end: IntExpr,
    },
    Predicate {
        value: BoolExpr,
        binders: Vec<SymbolId>,
    },
    Boolean(bool),
    Unknown,
}
/// Call construction could not establish a required initialized region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitializationFailure {
    pub parameter: usize,
    pub phase: InitializationPhase,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitializationPhase {
    Input,
    Output,
}
/// Reuses the language coverage rules in an existing semantic expression arena.
pub struct InitializationContext<'a> {
    arena: &'a mut ExprArena,
    path: Path,
    facts: prove::Facts,
}
impl RegionOps for InitializationContext<'_> {
    fn arena(&mut self) -> &mut ExprArena {
        self.arena
    }
    fn arena_ref(&self) -> &ExprArena {
        self.arena
    }
    fn fresh_variable(&mut self) -> SymbolId {
        self.arena.loop_binder().1
    }
}
impl<'a> InitializationContext<'a> {
    pub fn new(arena: &'a mut ExprArena) -> Self {
        Self {
            arena,
            path: vec![],
            facts: prove::Facts::new(),
        }
    }
    pub fn assume(&mut self, value: BoolExpr, truth: bool, binders: &[SymbolId]) {
        self.path
            .push((Condition::Actual(value, binders.to_vec()), truth));
    }
    pub fn root(&mut self, axes: &[IntExpr]) -> InitializationView {
        self.root_view(axes)
    }
    pub fn slice(
        &mut self,
        view: &InitializationView,
        selections: &[(Option<IntExpr>, Option<IntExpr>, bool)],
    ) -> InitializationView {
        self.select_view(view, selections)
    }
    pub fn transpose(&mut self, view: &InitializationView) -> InitializationView {
        let mut view = view.clone();
        view.axes.reverse();
        view.coordinates.reverse();
        view
    }
    pub fn permute(
        &mut self,
        view: &InitializationView,
        permutation: &[u32],
    ) -> InitializationView {
        assert_eq!(view.axes.len(), permutation.len());
        let mut seen = vec![false; permutation.len()];
        for &axis in permutation {
            assert!(
                !std::mem::replace(&mut seen[axis as usize], true),
                "view permutation repeats an axis"
            );
        }
        InitializationView {
            axes: permutation
                .iter()
                .map(|axis| view.axes[*axis as usize])
                .collect(),
            coordinates: permutation
                .iter()
                .map(|axis| view.coordinates[*axis as usize])
                .collect(),
            ..view.clone()
        }
    }
    pub fn reshape(&mut self, view: &InitializationView, axes: &[IntExpr]) -> InitializationView {
        self.reshape_view(view, axes)
    }
    pub fn write(
        &mut self,
        state: &InitializationState,
        view: &InitializationView,
    ) -> InitializationState {
        let region = self.view_region(view);
        InitializationState(
            state
                .0
                .clone()
                .union(self.normalize(region, &self.facts.clone())),
        )
    }
    pub fn readable(&mut self, state: &InitializationState, view: &InitializationView) -> bool {
        let required = self.view_region(view);
        self.covered(
            &state.0,
            &required,
            &self.path.clone(),
            &self.facts.clone(),
            &view.extents,
        )
    }
    pub fn project(
        &mut self,
        state: &InitializationState,
        view: &InitializationView,
    ) -> InitializationState {
        InitializationState(self.project_view_region(
            state.0.clone(),
            view,
            &self.path.clone(),
            &self.facts.clone(),
        ))
    }
    /// Transfer initialized contents through a whole-value copy. The result is
    /// in destination root coordinates; callers replace a complete destination
    /// allocation rather than retaining bytes overwritten by this operation.
    pub fn relocate(
        &mut self,
        state: &InitializationState,
        source: &InitializationView,
        destination: &InitializationView,
    ) -> InitializationState {
        let logical = self.project(state, source);
        InitializationState(self.map_view_region(logical.0, destination))
    }

    pub fn completed_loop(
        &mut self,
        before: &InitializationState,
        iteration: &InitializationState,
        binder: SymbolId,
        start: IntExpr,
        end: IntExpr,
    ) -> InitializationState {
        let completed = Region::Bind(
            Bound {
                symbol: binder,
                start,
                end,
            },
            Box::new(iteration.0.clone()),
        );
        InitializationState(self.normalize(before.0.clone().union(completed), &self.facts.clone()))
    }
    /// The state after an `if` join (see [`Region::branch`]).
    pub fn branch(
        &mut self,
        condition: BoolExpr,
        binders: &[SymbolId],
        then_state: &InitializationState,
        else_state: &InitializationState,
    ) -> InitializationState {
        InitializationState(Region::branch(
            &Condition::Actual(condition, binders.to_vec()),
            then_state.0.clone(),
            else_state.0.clone(),
        ))
    }
}

pub(crate) trait RegionMapping {
    fn integer(&mut self, value: IntExpr) -> IntExpr;
    fn binder(&mut self, symbol: SymbolId) -> SymbolId;
    fn parameter(&mut self, path: &ParameterPath) -> ParameterPath;
    fn predicate(&mut self, path: &ParameterPath) -> Condition {
        Condition::Parameter(self.parameter(path))
    }
    fn condition(&mut self, c: &Condition) -> Condition {
        match c {
            Condition::Compare(op, a, b) => {
                Condition::Compare(*op, self.integer(*a), self.integer(*b))
            }
            Condition::Not(c) => Condition::Not(Box::new(self.condition(c))),
            Condition::And(a, b) => {
                Condition::And(Box::new(self.condition(a)), Box::new(self.condition(b)))
            }
            Condition::Or(a, b) => {
                Condition::Or(Box::new(self.condition(a)), Box::new(self.condition(b)))
            }
            Condition::Parameter(path) => self.predicate(path),
            Condition::Version(..) => panic!("private initialization predicate escaped checking"),
            other => other.clone(),
        }
    }
    fn path(&mut self, path: &Path) -> Path {
        path.iter().map(|(c, v)| (self.condition(c), *v)).collect()
    }
    fn bound(&mut self, bound: &Bound) -> Bound {
        Bound {
            symbol: self.binder(bound.symbol),
            start: self.integer(bound.start),
            end: self.integer(bound.end),
        }
    }
    fn region(&mut self, region: &Region) -> Region {
        match region {
            Region::Empty => Region::Empty,
            Region::Full => Region::Full,
            Region::Linear(a, b) => Region::Linear(self.integer(*a), self.integer(*b)),
            Region::Image {
                domain,
                coordinates,
            } => Region::Image {
                domain: domain.iter().map(|b| self.bound(b)).collect(),
                coordinates: coordinates.iter().map(|c| self.integer(*c)).collect(),
            },
            Region::LinearImage { domain, address } => Region::LinearImage {
                domain: domain.iter().map(|b| self.bound(b)).collect(),
                address: self.integer(*address),
            },
            Region::Union(parts) => Region::Union(parts.iter().map(|r| self.region(r)).collect()),
            Region::Intersection(parts) => {
                Region::Intersection(parts.iter().map(|r| self.region(r)).collect())
            }
            Region::Bind(bound, inner) => {
                Region::Bind(self.bound(bound), Box::new(self.region(inner)))
            }
            Region::Guard(path, inner) => {
                Region::Guard(self.path(path), Box::new(self.region(inner)))
            }
        }
    }
    fn contract(&mut self, contract: &InitializationContract) -> InitializationContract {
        InitializationContract {
            accesses: contract
                .accesses
                .iter()
                .map(|a| ParameterAccess {
                    parameter: self.parameter(&a.parameter),
                    region: self.region(&a.region),
                    path: self.path(&a.path),
                    write: a.write,
                    atomic: a.atomic,
                })
                .collect(),
            requirements: contract
                .requirements
                .iter()
                .map(|r| Requirement {
                    parameter: self.parameter(&r.parameter),
                    region: self.region(&r.region),
                    path: self.path(&r.path),
                    span: r.span,
                })
                .collect(),
            exits: contract
                .exits
                .iter()
                .map(|e| Exit {
                    path: self.path(&e.path),
                    written: e
                        .written
                        .iter()
                        .map(|(p, r)| (self.parameter(p), self.region(r)))
                        .collect(),
                })
                .collect(),
            symbols: contract
                .symbols
                .iter()
                .map(|(s, p)| {
                    (
                        self.binder(*s),
                        match p {
                            ParameterPart::Integer(p) => ParameterPart::Integer(self.parameter(p)),
                            ParameterPart::Start(p) => ParameterPart::Start(self.parameter(p)),
                            ParameterPart::End(p) => ParameterPart::End(self.parameter(p)),
                        },
                    )
                })
                .collect(),
        }
    }
}
struct EntryMapping<'a, 'b, 'c> {
    source: &'a ExprArena,
    target: &'b mut ExprArena,
    map: &'b mut xfer::SymbolMap<'a>,
    bound: HashMap<SymbolId, IntExpr>,
    leaves: &'c [(usize, Vec<usize>, usize)],
}
impl RegionMapping for EntryMapping<'_, '_, '_> {
    fn integer(&mut self, value: IntExpr) -> IntExpr {
        let bound = &self.bound;
        let map = &mut self.map;
        xfer::transfer_int(self.source, value, self.target, &mut |s, a| {
            bound
                .get(&s)
                .copied()
                .map(AnyExpr::Int)
                .unwrap_or_else(|| map(s, a))
        })
    }
    fn binder(&mut self, symbol: SymbolId) -> SymbolId {
        let expression = *self.bound.entry(symbol).or_insert_with(|| {
            let s = self.target.loop_binder().1;
            self.target.int_symbol(s)
        });
        match self.target.view(AnyExpr::Int(expression)) {
            NodeView::Symbol(s) => s,
            _ => unreachable!("formal binder is a symbol"),
        }
    }
    fn parameter(&mut self, path: &ParameterPath) -> ParameterPath {
        let parameter = self
            .leaves
            .iter()
            .find(|(p, fields, _)| *p == path.parameter && fields == &path.fields)
            .map(|(_, _, ordinal)| *ordinal)
            .expect("checked parameter leaf exists in semantic function");
        ParameterPath::root(parameter)
    }
}
impl InitializationContract {
    /// Entry instantiation remaps all contract coordinates into the same arena
    /// as the semantic body, and uses its existing canonical parameter leaves.
    pub(crate) fn remap<'a>(
        &self,
        source: &'a ExprArena,
        target: &mut ExprArena,
        leaves: &'a [(usize, Vec<usize>, usize)],
        map: &mut xfer::SymbolMap<'a>,
    ) -> Self {
        let mut mapping = EntryMapping {
            source,
            target,
            map,
            bound: HashMap::new(),
            leaves,
        };
        for (symbol, _) in &self.symbols {
            mapping.binder(*symbol);
        }
        mapping.contract(self)
    }
}
struct ApplicationMapping<'a, 'b> {
    arena: &'a mut ExprArena,
    arguments: &'b [InitializationArgument],
    symbols: HashMap<SymbolId, IntExpr>,
}
impl RegionMapping for ApplicationMapping<'_, '_> {
    fn integer(&mut self, value: IntExpr) -> IntExpr {
        prove::substitute(self.arena, value, &|s| self.symbols.get(&s).copied())
    }
    fn binder(&mut self, symbol: SymbolId) -> SymbolId {
        symbol
    }
    fn parameter(&mut self, path: &ParameterPath) -> ParameterPath {
        assert!(path.fields.is_empty());
        path.clone()
    }
    fn predicate(&mut self, path: &ParameterPath) -> Condition {
        match &self.arguments[path.parameter] {
            InitializationArgument::Predicate { value, binders } => {
                Condition::Actual(*value, binders.clone())
            }
            InitializationArgument::Boolean(value) => Condition::Constant(*value),
            _ => Condition::Version(path.parameter as u64, vec![]),
        }
    }
}
impl InitializationContext<'_> {
    fn instantiate(
        &mut self,
        contract: &InitializationContract,
        arguments: &[InitializationArgument],
    ) -> InitializationContract {
        let mut symbols = HashMap::new();
        let mut missing = vec![];
        for (symbol, part) in &contract.symbols {
            let value = match part {
                ParameterPart::Integer(p) => match arguments.get(p.parameter) {
                    Some(InitializationArgument::Integer(v)) => Some(*v),
                    _ => None,
                },
                ParameterPart::Start(p) => match arguments.get(p.parameter) {
                    Some(InitializationArgument::Range { start, .. }) => Some(*start),
                    _ => None,
                },
                ParameterPart::End(p) => match arguments.get(p.parameter) {
                    Some(InitializationArgument::Range { end, .. }) => Some(*end),
                    _ => None,
                },
            };
            if let Some(value) = value {
                symbols.insert(*symbol, value);
            } else {
                missing.push(*symbol);
            }
        }
        let mut mapping = ApplicationMapping {
            arena: self.arena,
            arguments,
            symbols,
        };
        let mapped = mapping.contract(contract);
        let allowed = self
            .arena
            .symbols()
            .filter(|s| !missing.contains(s))
            .collect::<Vec<_>>();
        self.close_transfer(mapped, &allowed)
    }
    /// Apply the checked transfer to the actual argument views. Tensor outputs
    /// are root-relative and already include their incoming initialization.
    pub fn apply(
        &mut self,
        contract: &InitializationContract,
        arguments: &[InitializationArgument],
    ) -> Result<Vec<Option<InitializationState>>, InitializationFailure> {
        let contract = self.instantiate(contract, arguments);
        for required in contract.requirements {
            let parameter = required.parameter.parameter;
            let Some(InitializationArgument::Tensor { state, view }) = arguments.get(parameter)
            else {
                return Err(InitializationFailure {
                    parameter,
                    phase: InitializationPhase::Input,
                });
            };
            let region = self.map_view_region(required.region, view);
            let region = Region::Guard(required.path, Box::new(region));
            if !self.covered(
                &state.0,
                &region,
                &self.path.clone(),
                &self.facts.clone(),
                &view.extents,
            ) {
                return Err(InitializationFailure {
                    parameter,
                    phase: InitializationPhase::Input,
                });
            }
        }
        let mut outputs = arguments
            .iter()
            .map(|argument| match argument {
                InitializationArgument::Tensor { state, .. } => Some(state.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        for exit in contract.exits {
            for (parameter, region) in exit.written {
                let parameter = parameter.parameter;
                let Some(InitializationArgument::Tensor { view, .. }) = arguments.get(parameter)
                else {
                    return Err(InitializationFailure {
                        parameter,
                        phase: InitializationPhase::Output,
                    });
                };
                let region = self.map_view_region(region, view);
                let region = Region::Guard(exit.path.clone(), Box::new(region));
                let state = outputs[parameter]
                    .as_mut()
                    .expect("tensor input has initialization state");
                state.0 = self.normalize(state.0.clone().union(region), &self.facts.clone());
            }
        }
        Ok(outputs)
    }
    /// Instantiate completed captured writes at an actual loop header or exit.
    /// `end` is the current binder at a header, and the actual bound at exit.
    pub fn loop_completed(
        &mut self,
        contract: &LoopInitialization,
        arguments: &[InitializationArgument],
        start: IntExpr,
        end: IntExpr,
    ) -> Result<Vec<Option<InitializationState>>, InitializationFailure> {
        let (symbol, value) = self.fresh_integer();
        let mut iteration = arguments.to_vec();
        iteration[contract.binder.parameter] = InitializationArgument::Integer(value);
        let produced = self.apply(&contract.transfer, &iteration)?;
        Ok(arguments
            .iter()
            .zip(produced)
            .map(|(argument, output)| match (argument, output) {
                (InitializationArgument::Tensor { state, .. }, Some(output)) => {
                    Some(self.completed_loop(state, &output, symbol, start, end))
                }
                _ => None,
            })
            .collect())
    }

    /// Owned carries receive the already checked invariant in their own logical
    /// coordinates. Replaced allocations never inherit another root's history.
    pub fn loop_carried(
        &mut self,
        contract: &LoopInitialization,
        arguments: &[InitializationArgument],
    ) -> Vec<(usize, InitializationState)> {
        let mut symbolic = contract.transfer.clone();
        symbolic.exits = vec![Exit {
            path: Vec::new(),
            written: contract.carried.clone(),
        }];
        let instantiated = self.instantiate(&symbolic, arguments);
        instantiated
            .exits
            .into_iter()
            .flat_map(|exit| exit.written)
            .filter_map(|(parameter, region)| {
                let InitializationArgument::Tensor { view, .. } = &arguments[parameter.parameter]
                else {
                    return None;
                };
                Some((
                    parameter.parameter,
                    InitializationState(self.map_view_region(region, view)),
                ))
            })
            .collect()
    }

    /// An authored alternative is usable only if it accepts actual incoming
    /// state and preserves the reference call's initialized outgoing state.
    pub fn applicable(
        &mut self,
        candidate: &InitializationContract,
        reference: &InitializationContract,
        arguments: &[InitializationArgument],
    ) -> Result<Vec<Option<InitializationState>>, InitializationFailure> {
        let candidate = self.apply(candidate, arguments)?;
        let reference = self.apply(reference, arguments)?;
        for (parameter, (candidate, reference)) in candidate.iter().zip(&reference).enumerate() {
            if let (Some(candidate), Some(reference), InitializationArgument::Tensor { view, .. }) =
                (candidate, reference, &arguments[parameter])
            {
                if !self.covered(
                    &candidate.0,
                    &reference.0,
                    &self.path.clone(),
                    &self.facts.clone(),
                    &view.extents,
                ) {
                    return Err(InitializationFailure {
                        parameter,
                        phase: InitializationPhase::Output,
                    });
                }
            }
        }
        Ok(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::CmpOp;

    fn size(region: &Region) -> usize {
        match region {
            Region::Union(parts) | Region::Intersection(parts) => {
                1 + parts.iter().map(size).sum::<usize>()
            }
            Region::Bind(_, inner) | Region::Guard(_, inner) => 1 + size(inner),
            _ => 1,
        }
    }

    #[test]
    fn sequential_joins_grow_linearly() {
        let mut arena = ExprArena::new();
        let sixteen = arena.int(16);
        let (_, _, selector) = arena.loop_binder();
        let conditions = (0..12)
            .map(|k| {
                let k = arena.int(k);
                arena.int_cmp(CmpOp::Lt, k, selector)
            })
            .collect::<Vec<_>>();
        let points = (0..12).map(|k| arena.int(k)).collect::<Vec<_>>();
        let always = arena.bool(true);
        let mut context = InitializationContext::new(&mut arena);
        let root = context.root(&[sixteen]);
        let first = context.slice(&root, &[(Some(points[0]), None, true)]);
        let mut state = context.write(&InitializationState::empty(), &first);
        for (condition, point) in conditions.into_iter().zip(points) {
            let element = context.slice(&root, &[(Some(point), None, true)]);
            let written = context.write(&state, &element);
            state = context.branch(condition, &[], &written, &state);
        }
        let Region::Union(members) = &state.0 else {
            panic!("joins keep a union: {:?}", state.0);
        };
        assert!(members.len() <= 13, "{} members", members.len());
        assert!(size(&state.0) <= 3 * 13, "size {}", size(&state.0));
        let same = context.branch(always, &[], &state, &state);
        assert_eq!(size(&same.0), size(&state.0));
    }

    #[test]
    fn joined_image_forgets_the_symbol_of_its_replaced_coordinate() {
        // `s + 1` for `s in [lo, hi)` joined with the point `hi + 1`, where
        // only `hi >= lo` is known, so `s`'s range may be empty. The joined
        // box is `[lo + 1, hi + 2)`; `s` must not stay in its domain, or
        // `lo <= s <= hi - 1` would become a fact about the free `s` below.
        let mut arena = ExprArena::new();
        let (_, _, lo) = arena.loop_binder();
        let (_, _, hi) = arena.loop_binder();
        let (_, s, s_value) = arena.loop_binder();
        let mut facts = prove::Facts::new();
        let width = arena.int_sub(hi, lo);
        facts.assume_nonnegative(&mut arena, width);
        let one = arena.int(1);
        let two = arena.int(2);
        let shifted = arena.int_add(s_value, one);
        let last = arena.int_add(hi, one);
        // A point that overlaps the joined box for `s = lo - 1`.
        let other = arena.int_add(s_value, two);
        let other = arena.int_add(other, width);
        let extent = arena.int(64);
        let mut context = InitializationContext::new(&mut arena);
        let mut parts = vec![
            Region::Image {
                domain: vec![Bound {
                    symbol: s,
                    start: lo,
                    end: hi,
                }],
                coordinates: vec![shifted],
            },
            Region::Image {
                domain: vec![],
                coordinates: vec![last],
            },
        ];
        context.join_images(&mut parts, &facts);
        let [Region::Image { domain, .. }] = parts.as_slice() else {
            panic!("the two boxes join: {parts:?}");
        };
        assert!(domain.iter().all(|bound| bound.symbol != s), "{domain:?}");
        let joined = parts.pop().unwrap();
        let point = Region::Image {
            domain: vec![],
            coordinates: vec![other],
        };
        assert!(!context.separated_regions(joined, point, &facts, &[extent]));
    }

    #[test]
    fn image_rebasing_names_fresh_symbols_and_is_idempotent() {
        // `s + 1` over `s in [2, 9)` is the range `[3, 10)`. Its enumerating
        // symbol is fresh: the binder `s`, still bounded by `[2, 8]` in the
        // facts, never takes the meaning `s + 1`.
        let mut arena = ExprArena::new();
        let (_, s, s_value) = arena.loop_binder();
        let (one, two, eight, nine) = (arena.int(1), arena.int(2), arena.int(8), arena.int(9));
        let shifted = arena.int_add(s_value, one);
        let mut facts = prove::Facts::new();
        facts.set_range(s, two, eight);
        let mut context = InitializationContext::new(&mut arena);
        let image = Region::Image {
            domain: vec![Bound {
                symbol: s,
                start: two,
                end: nine,
            }],
            coordinates: vec![shifted],
        };
        let normalized = context.normalize(image, &facts);
        let Region::Image { domain, .. } = &normalized else {
            panic!("an image stays an image: {normalized:?}");
        };
        assert!(
            domain.iter().all(|bound| bound.symbol != s),
            "the binder is not reused with a rebased meaning: {domain:?}"
        );
        let again = context.normalize(normalized.clone(), &facts);
        assert_eq!(again, normalized);
    }
}
