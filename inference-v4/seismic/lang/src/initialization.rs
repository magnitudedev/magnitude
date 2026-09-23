//! Logical initialization regions and checked function transfers.
//!
//! The source checker constructs function contracts. Compiler construction
//! applies those same contracts to its actual bindings. This module owns the
//! one coordinate calculus; neither consumer may invent whole-root permission.
use crate::check::{prove, xfer};
use crate::expr::{AnyExpr, BoolExpr, ExprArena, IntExpr, SymbolId};
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

/// Images bind their coordinates jointly. Logical linearization is a derived
/// bijection from the checked axes, never a claim based on matching byte sizes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Region {
    Empty,
    Full,
    Interval(IntExpr, IntExpr),
    Image {
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
}

pub(crate) trait RegionOps {
    fn arena(&mut self) -> &mut ExprArena;
    fn arena_ref(&self) -> &ExprArena;
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
            Region::Image { domain, address } => self.normalize_image(domain, address, facts),
            Region::Guard(path, inner) if path.is_empty() => self.normalize(*inner, facts),
            Region::Interval(start, end) if self.le(facts, end, start) => Region::Empty,
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
                        if let (Region::Interval(a, b), Region::Interval(c, d)) =
                            (&parts[i], &parts[j])
                        {
                            let (a, b, c, d) = (*a, *b, *c, *d);
                            if self.le(facts, a, c) && self.le(facts, c, b) && self.le(facts, b, d)
                            {
                                parts[i] = Region::Interval(a, d);
                                parts.remove(j);
                                j = i + 1;
                                continue;
                            }
                            if self.le(facts, c, a) && self.le(facts, a, d) && self.le(facts, d, b)
                            {
                                parts[i] = Region::Interval(c, b);
                                parts.remove(j);
                                j = i + 1;
                                continue;
                            }
                        }
                        j += 1;
                    }
                    i += 1;
                }
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
                        address,
                    } => {
                        domain.insert(0, bound);
                        self.normalize_image(domain, address, facts)
                    }
                    Region::Interval(start, end) => {
                        let (symbol, index) = self.fresh_integer();
                        let zero = self.arena().int(0);
                        let width = self.arena().int_sub(end, start);
                        let address = self.arena().int_add(start, index);
                        self.normalize_image(
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
    fn normalize_image(
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
        let mut terms = vec![];
        let mut start_map = HashMap::new();
        for d in &dimensions {
            let Some(coefficient) = prove::linear_coefficient(self.arena(), address, d.symbol)
            else {
                return Region::Image { domain, address };
            };
            if dimensions
                .iter()
                .any(|other| prove::mentions(self.arena_ref(), coefficient, other.symbol))
            {
                return Region::Image { domain, address };
            }
            if prove::is_zero(self.arena(), coefficient) {
                if !self.lt(facts, d.start, d.end) {
                    return Region::Image { domain, address };
                }
            } else {
                terms.push((d.clone(), coefficient));
            }
            start_map.insert(d.symbol, d.start);
        }
        let start = self.substitute(address, &start_map);
        let mut stride = self.arena().int(1);
        while !terms.is_empty() {
            let Some(index) = terms
                .iter()
                .position(|(_, coefficient)| self.same(*coefficient, stride))
            else {
                return Region::Image { domain, address };
            };
            let (d, _) = terms.remove(index);
            let width = self.arena().int_sub(d.end, d.start);
            if !prove::nonneg(self.arena(), facts, width) {
                return Region::Image { domain, address };
            }
            stride = self.arena().int_mul(stride, width);
        }
        let end = self.arena().int_add(start, stride);
        Region::Interval(start, end)
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
    fn assume_nonnegative(&mut self, facts: &mut prove::Facts, value: IntExpr) {
        for symbol in prove::symbols(self.arena(), value) {
            match prove::linear_in(self.arena(), value, symbol) {
                Some((1, rest)) => {
                    let zero = self.arena().int(0);
                    let lower = self.arena().int_sub(zero, rest);
                    facts.add_lower(symbol, lower);
                }
                Some((-1, rest)) => facts.add_upper(symbol, rest),
                _ => {}
            }
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
                    self.assume_nonnegative(facts, negative);
                }
                self.assume_nonnegative(facts, difference);
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
    fn covered(
        &mut self,
        available: &Region,
        required: &Region,
        path: &Path,
        facts: &prove::Facts,
    ) -> bool {
        if let Some(condition) = self
            .unknown_guard(available, path, facts)
            .or_else(|| self.unknown_guard(required, path, facts))
        {
            return [false, true].into_iter().all(|value| {
                let mut path = path.clone();
                let mut facts = facts.clone();
                !self.assume(&mut path, &mut facts, condition.clone(), value)
                    || self.covered(available, required, &path, &facts)
            });
        }
        let available = self.active_region(available.clone(), path, facts);
        let required = self.active_region(required.clone(), path, facts);
        let available = self.normalize(available, facts);
        let required = self.normalize(required, facts);
        self.covered_plain(&available, &required, facts)
    }
    fn covered_plain(
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
                .all(|p| self.covered_plain(available, p, facts));
        }
        if let Region::Intersection(parts) = available {
            return parts.iter().all(|p| self.covered_plain(p, required, facts));
        }
        match (available, required) {
            (Region::Interval(a, b), Region::Interval(c, d)) => {
                self.le(facts, *a, *c) && self.le(facts, *d, *b)
            }
            (Region::Interval(a, b), Region::Image { domain, address }) => {
                let mut facts = facts.clone();
                let one = self.arena().int(1);
                for bound in domain {
                    let upper = self.arena().int_sub(bound.end, one);
                    facts.set_range(bound.symbol, bound.start, upper);
                }
                self.le(&facts, *a, *address) && self.lt(&facts, *address, *b)
            }
            (
                Region::Image {
                    domain: a,
                    address: x,
                },
                Region::Image {
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
                    .any(|part| self.covered_plain(part, required, facts))
                {
                    return true;
                }
                let Region::Interval(start, end) = required else {
                    return false;
                };
                let mut cursor = *start;
                let mut remaining = parts.iter().collect::<Vec<_>>();
                loop {
                    if self.le(facts, *end, cursor) {
                        return true;
                    }
                    let Some(index) = remaining.iter().position(|part| match part {
                        Region::Interval(a, b) => {
                            self.le(facts, *a, cursor) && self.lt(facts, cursor, *b)
                        }
                        _ => false,
                    }) else {
                        return false;
                    };
                    let Region::Interval(_, next) = remaining.remove(index) else {
                        unreachable!()
                    };
                    cursor = *next;
                }
            }
            (_, Region::Intersection(parts)) => parts
                .iter()
                .any(|p| self.covered_plain(available, p, facts)),
            _ => false,
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
            address: place.address,
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
            address: self.substitute(place.address, &map),
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
        let address = self.view_address_at(place, linear);
        InitializationView {
            axes: axes.to_vec(),
            coordinates,
            address,
        }
    }
    fn view_address_at(&mut self, place: &InitializationView, mut linear: IntExpr) -> IntExpr {
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
        self.substitute(place.address, &map)
    }
    fn root_view(&mut self, axes: &[IntExpr]) -> InitializationView {
        let mut coordinates = vec![];
        let mut address = self.arena().int(0);
        for &extent in axes {
            let (symbol, index) = self.fresh_integer();
            coordinates.push(symbol);
            address = self.arena().int_mul(address, extent);
            address = self.arena().int_add(address, index);
        }
        InitializationView {
            axes: axes.to_vec(),
            coordinates,
            address,
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
        match region {
            Region::Interval(a, b) if !known(self, a, allowed) || !known(self, b, allowed) => {
                unknown
            }
            Region::Image { domain, address } => {
                let mut scope = allowed.to_vec();
                for bound in &domain {
                    if !known(self, bound.start, &scope) || !known(self, bound.end, &scope) {
                        return unknown;
                    }
                    scope.push(bound.symbol);
                }
                if known(self, address, &scope) {
                    Region::Image { domain, address }
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
    fn map_view_region(&mut self, region: Region, view: &InitializationView) -> Region {
        match region {
            Region::Empty => Region::Empty,
            Region::Full => self.view_region(view),
            Region::Interval(start, end) => {
                let (symbol, index) = self.fresh_integer();
                Region::Image {
                    domain: vec![Bound { symbol, start, end }],
                    address: self.view_address_at(view, index),
                }
            }
            Region::Image { domain, address } => Region::Image {
                domain,
                address: self.view_address_at(view, address),
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
            Region::Interval(a, b) => Region::Interval(
                self.arena().int_sub(a, offset),
                self.arena().int_sub(b, offset),
            ),
            Region::Image { domain, address } => Region::Image {
                domain,
                address: self.arena().int_sub(address, offset),
            },
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
    fn project_view_region(
        &mut self,
        region: Region,
        view: &InitializationView,
        path: &Path,
        facts: &prove::Facts,
    ) -> Region {
        let domain = self.view_region(view);
        if self.covered(&region, &domain, path, facts) {
            return Region::Full;
        }
        let (symbol, linear) = self.fresh_integer();
        let address = self.view_address_at(view, linear);
        let address = prove::recompose_address(self.arena(), address);
        let Some(coefficient) = prove::linear_coefficient(self.arena(), address, symbol) else {
            return Region::Empty;
        };
        if prove::constant(self.arena_ref(), coefficient) != Some(1) {
            return Region::Empty;
        }
        let zero = self.arena().int(0);
        let offset = self.substitute(address, &HashMap::from([(symbol, zero)]));
        self.shift_region(region, offset)
    }
    fn fresh_integer(&mut self) -> (SymbolId, IntExpr) {
        let symbol = self.arena().loop_binder().1;
        let value = self.arena().int_symbol(symbol);
        (symbol, value)
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
    pub(crate) atomic: bool,
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
#[derive(Clone, Debug)]
pub struct InitializationView {
    pub(crate) axes: Vec<IntExpr>,
    pub(crate) coordinates: Vec<SymbolId>,
    pub(crate) address: IntExpr,
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
        self.covered(&state.0, &required, &self.path.clone(), &self.facts.clone())
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
    pub fn branch(
        &mut self,
        condition: BoolExpr,
        binders: &[SymbolId],
        then_state: &InitializationState,
        else_state: &InitializationState,
    ) -> InitializationState {
        InitializationState(
            Region::Guard(
                vec![(Condition::Actual(condition, binders.to_vec()), true)],
                Box::new(then_state.0.clone()),
            )
            .union(Region::Guard(
                vec![(Condition::Actual(condition, binders.to_vec()), false)],
                Box::new(else_state.0.clone()),
            )),
        )
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
            Region::Interval(a, b) => Region::Interval(self.integer(*a), self.integer(*b)),
            Region::Image { domain, address } => Region::Image {
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
            crate::expr::NodeView::Symbol(s) => s,
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
            if !self.covered(&state.0, &region, &self.path.clone(), &self.facts.clone()) {
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
            if let (Some(candidate), Some(reference)) = (candidate, reference) {
                if !self.covered(
                    &candidate.0,
                    &reference.0,
                    &self.path.clone(),
                    &self.facts.clone(),
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
