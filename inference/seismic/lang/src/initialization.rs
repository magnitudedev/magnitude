//! Logical initialization regions and checked function transfers.
//!
//! The source checker constructs function contracts. Compiler construction
//! applies those same contracts to its actual bindings. This module owns the
//! one coordinate calculus; neither consumer may invent whole-root permission.
use crate::check::{prove, xfer};
use crate::expr::{
    AnyExpr, BoolExpr, ErasedScalarExpr, ExprArena, IntExpr, NodeView, SymbolId, SymbolSort,
};
use crate::intrinsics::AtomicOp;
use crate::span::Span;
use crate::syntax::ast::BinaryOp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Exact opaque Boolean premises implied by the whole condition. Unknown
/// alternatives contribute none; a disjunction never selects one alternative.
fn guaranteed_actuals(condition: &Condition, truth: bool) -> Vec<(BoolExpr, bool)> {
    match condition {
        Condition::Actual(value, _) => vec![(*value, truth)],
        Condition::Not(inner) => guaranteed_actuals(inner, !truth),
        Condition::And(a, b) | Condition::Or(a, b) => {
            let mut left = guaranteed_actuals(a, truth);
            let right = guaranteed_actuals(b, truth);
            let both = matches!(condition, Condition::And(_, _)) == truth;
            if both {
                for premise in right {
                    if !left.contains(&premise) {
                        left.push(premise);
                    }
                }
            } else {
                left.retain(|premise| right.contains(premise));
            }
            left
        }
        _ => vec![],
    }
}

/// Successful return selects one reached exit; it does not assert all exits.
pub(crate) fn successful_paths(paths: &[Path]) -> Condition {
    paths
        .iter()
        .map(|path| {
            path.iter()
                .map(|(condition, truth)| {
                    if *truth {
                        condition.clone()
                    } else {
                        Condition::Not(Box::new(condition.clone()))
                    }
                })
                .reduce(|a, b| Condition::And(Box::new(a), Box::new(b)))
                .unwrap_or(Condition::Constant(true))
        })
        .reduce(|a, b| Condition::Or(Box::new(a), Box::new(b)))
        .unwrap_or(Condition::Constant(false))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct Bound {
    pub(crate) symbol: SymbolId,
    pub(crate) start: IntExpr,
    pub(crate) end: IntExpr,
}

/// Elements of one storage root. Images bind their coordinates jointly; the
/// row-major forms are derived from the root's checked axes, never from
/// matching byte sizes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
                for member in b {
                    if !a.contains(&member) {
                        a.push(member);
                    }
                }
                Self::Union(a)
            }
            (Self::Union(mut a), b) => {
                if !a.contains(&b) {
                    a.push(b);
                }
                Self::Union(a)
            }
            (a, Self::Union(mut b)) => {
                if !b.contains(&a) {
                    b.insert(0, a);
                }
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
    pub(crate) fn guarded(path: Path, region: Self) -> Self {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

/// One normalization invocation owns its lexical fact scopes. Only an exact
/// premise under the same parent can share a child; no scope survives the batch.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NormalizationPremise {
    Bind(Bound),
    Guard(Path),
}

pub(crate) struct NormalizationBatch {
    scopes: Vec<Rc<prove::Facts>>,
    children: HashMap<(usize, NormalizationPremise), Option<usize>>,
}

impl NormalizationBatch {
    fn new(facts: &prove::Facts) -> Self {
        Self {
            scopes: vec![Rc::new(facts.clone())],
            children: HashMap::new(),
        }
    }

    fn facts(&self, scope: usize) -> Rc<prove::Facts> {
        self.scopes[scope].clone()
    }

    fn insert(
        &mut self,
        parent: usize,
        premise: NormalizationPremise,
        facts: Option<prove::Facts>,
    ) -> Option<usize> {
        let child = facts.map(|facts| {
            let id = self.scopes.len();
            self.scopes.push(Rc::new(facts));
            id
        });
        self.children.insert((parent, premise), child);
        child
    }
}

pub(crate) trait RegionOps {
    fn arena(&mut self) -> &mut ExprArena;
    fn arena_ref(&self) -> &ExprArena;
    /// A fresh integer variable of this arena: a checker variable in a
    /// definition arena, a region binder in an entry arena.
    fn fresh_variable(&mut self) -> SymbolId;
    /// The integer value of a domain symbol. A domain binds `Int` symbols it
    /// mints and the region binders its caller supplies, which may be `Nat`.
    fn symbol_value(&mut self, symbol: SymbolId) -> IntExpr {
        crate::expr::poly::intern_atom(self.arena(), &crate::expr::poly::Atom::Symbol(symbol))
    }
    fn substitute(&mut self, value: IntExpr, map: &HashMap<SymbolId, IntExpr>) -> IntExpr {
        prove::substitute(self.arena(), value, &|s| map.get(&s).copied())
    }
    fn same(&self, a: IntExpr, b: IntExpr) -> bool {
        prove::same(self.arena_ref(), a, b)
    }
    fn le(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        prove::derive_le(self.arena(), facts, a, b).is_established()
    }
    fn lt(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        prove::derive_lt(self.arena(), facts, a, b).is_established()
    }

    /// Preserve a guarded loop image exactly. Simple integer comparisons
    /// constrain this binder's half-open interval; other conditions stay
    /// inside the binder, where their private coordinate remains meaningful.
    fn clip_bound_guard(
        &mut self,
        mut bound: Bound,
        path: Path,
        batch: &mut NormalizationBatch,
        scope: usize,
    ) -> (Bound, Path) {
        let binder_scope = self.normalization_bind_scope(batch, scope, &bound);
        let binder_facts = batch.facts(binder_scope);
        let one = self.arena().int(1);
        let coordinate = self.symbol_value(bound.symbol);
        let mut pending = path;
        let mut remaining = Vec::new();
        while let Some((condition, truth)) = pending.pop() {
            match condition {
                Condition::Not(inner) => {
                    pending.push((*inner, !truth));
                }
                Condition::And(a, b) if truth => {
                    pending.push((*b, true));
                    pending.push((*a, true));
                }
                Condition::Or(a, b) if !truth => {
                    pending.push((*b, false));
                    pending.push((*a, false));
                }
                Condition::Compare(op, a, b) => {
                    let a = prove::canonical_with_facts(self.arena(), &binder_facts, a);
                    let b = prove::canonical_with_facts(self.arena(), &binder_facts, b);
                    let op = if truth {
                        op
                    } else {
                        match op {
                            BinaryOp::Lt => BinaryOp::Ge,
                            BinaryOp::Le => BinaryOp::Gt,
                            BinaryOp::Gt => BinaryOp::Le,
                            BinaryOp::Ge => BinaryOp::Lt,
                            BinaryOp::Eq => BinaryOp::Ne,
                            BinaryOp::Ne => BinaryOp::Eq,
                            _ => {
                                remaining.push((Condition::Compare(op, a, b), truth));
                                continue;
                            }
                        }
                    };
                    let (op, limit) = if self.same(a, coordinate)
                        && !prove::mentions(self.arena_ref(), b, bound.symbol)
                    {
                        (op, b)
                    } else if self.same(b, coordinate)
                        && !prove::mentions(self.arena_ref(), a, bound.symbol)
                    {
                        let reversed = match op {
                            BinaryOp::Lt => BinaryOp::Gt,
                            BinaryOp::Le => BinaryOp::Ge,
                            BinaryOp::Gt => BinaryOp::Lt,
                            BinaryOp::Ge => BinaryOp::Le,
                            other => other,
                        };
                        (reversed, a)
                    } else {
                        remaining.push((Condition::Compare(op, a, b), true));
                        continue;
                    };
                    // Inclusive upper bounds and strict lower bounds need a
                    // representable successor. Unknown/overflowing limits
                    // retain the original guard instead of changing meaning.
                    let successor = if matches!(op, BinaryOp::Le | BinaryOp::Gt | BinaryOp::Eq) {
                        let maximum = self.arena().int(i64::MAX - 1);
                        self.optional_le(&binder_facts, limit, maximum)
                            .then(|| self.arena().int_add(limit, one))
                    } else {
                        None
                    };
                    match op {
                        BinaryOp::Lt => bound.end = self.arena().int_min(bound.end, limit),
                        BinaryOp::Ge => bound.start = self.arena().int_max(bound.start, limit),
                        BinaryOp::Le if successor.is_some() => {
                            bound.end = self.arena().int_min(bound.end, successor.unwrap());
                        }
                        BinaryOp::Gt if successor.is_some() => {
                            bound.start = self.arena().int_max(bound.start, successor.unwrap());
                        }
                        BinaryOp::Eq if successor.is_some() => {
                            bound.start = self.arena().int_max(bound.start, limit);
                            bound.end = self.arena().int_min(bound.end, successor.unwrap());
                        }
                        _ => remaining.push((Condition::Compare(op, coordinate, limit), true)),
                    }
                }
                other => remaining.push((other, truth)),
            }
        }
        remaining.reverse();
        (bound, remaining)
    }

    /// Install a lexical binder once per exact parent and reached interval.
    fn normalization_bind_scope(
        &mut self,
        batch: &mut NormalizationBatch,
        parent: usize,
        bound: &Bound,
    ) -> usize {
        let premise = NormalizationPremise::Bind(bound.clone());
        if let Some(child) = batch.children.get(&(parent, premise.clone())) {
            return child.expect("a binder always has a child scope");
        }
        let mut scoped = (*batch.facts(parent)).clone();
        let one = self.arena().int(1);
        let last = self.arena().int_sub(bound.end, one);
        scoped.set_range(bound.symbol, bound.start, last);
        batch.insert(parent, premise, Some(scoped)).unwrap()
    }

    /// A retained guard owns its justified premises. Opposite or differently
    /// scoped guards never share assumptions, even if their regions coincide.
    fn normalization_guard_scope(
        &mut self,
        batch: &mut NormalizationBatch,
        parent: usize,
        path: &Path,
    ) -> Option<usize> {
        if path.is_empty() {
            return Some(parent);
        }
        let premise = NormalizationPremise::Guard(path.clone());
        if let Some(child) = batch.children.get(&(parent, premise.clone())) {
            return *child;
        }
        let mut scoped = (*batch.facts(parent)).clone();
        let mut justified = vec![];
        for (condition, truth) in path {
            if !self.assume(&mut justified, &mut scoped, condition.clone(), *truth) {
                return batch.insert(parent, premise, None);
            }
        }
        batch.insert(parent, premise, Some(scoped))
    }

    /// Endpoints use their parent scope; coordinates use the completed domain.
    /// Domain prefixes are the same lexical scopes as retained Bind regions.
    fn reached_domain(
        &mut self,
        domain: Vec<Bound>,
        batch: &mut NormalizationBatch,
        mut scope: usize,
    ) -> (Vec<Bound>, usize) {
        let mut reached = Vec::with_capacity(domain.len());
        for mut bound in domain {
            let facts = batch.facts(scope);
            let endpoints =
                prove::canonical_many_with_facts(self.arena(), &facts, &[bound.start, bound.end]);
            bound.start = endpoints[0];
            bound.end = endpoints[1];
            scope = self.normalization_bind_scope(batch, scope, &bound);
            reached.push(bound);
        }
        (reached, scope)
    }
    fn reached_condition(&mut self, condition: Condition, facts: &prove::Facts) -> Condition {
        match condition {
            Condition::Compare(op, a, b) => {
                let values = prove::canonical_many_with_facts(self.arena(), facts, &[a, b]);
                Condition::Compare(op, values[0], values[1])
            }
            Condition::Not(c) => Condition::Not(Box::new(self.reached_condition(*c, facts))),
            Condition::And(a, b) => Condition::And(
                Box::new(self.reached_condition(*a, facts)),
                Box::new(self.reached_condition(*b, facts)),
            ),
            Condition::Or(a, b) => Condition::Or(
                Box::new(self.reached_condition(*a, facts)),
                Box::new(self.reached_condition(*b, facts)),
            ),
            other => other,
        }
    }
    /// Reach scalar expressions once in their source scopes, then reduce
    /// the resulting geometry. Enclosing binders accumulate into an image's
    /// domain before rebasing; geometry must not re-enter the reached walk.
    fn normalize(&mut self, region: Region, facts: &prove::Facts) -> Region {
        self.normalize_many(vec![region], facts).pop().unwrap()
    }
    /// Preserve separate regions and their caller-owned effect metadata while
    /// sharing only their exact lexical fact scopes throughout both phases.
    fn normalize_many(&mut self, regions: Vec<Region>, facts: &prove::Facts) -> Vec<Region> {
        let mut batch = NormalizationBatch::new(facts);
        let reached = regions
            .into_iter()
            .map(|region| self.reach_region(region, &mut batch, 0))
            .collect::<Vec<_>>();
        reached
            .into_iter()
            .map(|region| self.reduce_reached_region(region, &mut batch, 0))
            .collect()
    }
    fn reach_region(
        &mut self,
        region: Region,
        batch: &mut NormalizationBatch,
        scope: usize,
    ) -> Region {
        crate::cancellation::check();
        let facts = batch.facts(scope);
        match region {
            Region::Image {
                domain,
                coordinates,
            } => self.reach_image(domain, coordinates, batch, scope),
            Region::LinearImage { domain, address } => {
                self.reach_linear(domain, address, batch, scope)
            }
            Region::Guard(_, inner) if matches!(inner.as_ref(), Region::Empty) => Region::Empty,
            Region::Bind(_, inner) if matches!(inner.as_ref(), Region::Empty) => Region::Empty,
            Region::Guard(path, inner) => {
                // A condition the normalization's own facts already decide adds
                // nothing to the guard; one they refute empties the region,
                // exactly as assuming it would. Only undecided conditions stay
                // guards. A binder scope's private range decides nothing here:
                // a guard can state exactly that range's nonemptiness.
                let base = batch.facts(0);
                let mut reached = Path::with_capacity(path.len());
                for (condition, truth) in path {
                    let condition = self.reached_condition(condition, &facts);
                    match self.condition_value(&condition, &reached, &base) {
                        Some(actual) if actual == truth => {}
                        Some(_) => return Region::Empty,
                        None => reached.push((condition, truth)),
                    }
                }
                let Some(child) = self.normalization_guard_scope(batch, scope, &reached) else {
                    return Region::Empty;
                };
                Region::guarded(reached, self.reach_region(*inner, batch, child))
            }
            Region::Linear(start, end) => {
                let values = prove::canonical_many_with_facts(self.arena(), &facts, &[start, end]);
                if self.optional_le(&facts, values[1], values[0]) {
                    Region::Empty
                } else {
                    Region::Linear(values[0], values[1])
                }
            }
            Region::Union(mut parts) => {
                merge_complementary_guards(&mut parts);
                parts.into_iter().fold(Region::Empty, |a, b| {
                    a.union(self.reach_region(b, batch, scope))
                })
            }
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |a, b| {
                let b = self.reach_region(b, batch, scope);
                a.intersection(b)
            }),
            Region::Bind(mut bound, body) => {
                let endpoints = prove::canonical_many_with_facts(
                    self.arena(),
                    &facts,
                    &[bound.start, bound.end],
                );
                bound.start = endpoints[0];
                bound.end = endpoints[1];
                if self.optional_le(&facts, bound.end, bound.start) {
                    return Region::Empty;
                }
                let child = self.normalization_bind_scope(batch, scope, &bound);
                let inner = self.reach_region(*body, batch, child);
                self.finalize_reached_bind(bound, inner, batch, scope)
            }
            other => other,
        }
    }
    fn reduce_reached_region(
        &mut self,
        region: Region,
        batch: &mut NormalizationBatch,
        scope: usize,
    ) -> Region {
        crate::cancellation::check();
        let facts = batch.facts(scope);
        match region {
            Region::Image {
                domain,
                coordinates,
            } => self.reduce_reached_image(domain, coordinates, &facts),
            Region::LinearImage { domain, address } => {
                self.reduce_reached_linear(domain, address, &facts)
            }
            Region::Guard(_, inner) if matches!(inner.as_ref(), Region::Empty) => Region::Empty,
            Region::Bind(_, inner) if matches!(inner.as_ref(), Region::Empty) => Region::Empty,
            Region::Guard(path, inner) => {
                let Some(child) = self.normalization_guard_scope(batch, scope, &path) else {
                    return Region::Empty;
                };
                Region::guarded(path, self.reduce_reached_region(*inner, batch, child))
            }
            Region::Union(mut parts) => {
                // Remove repeated paths before recursively normalizing their
                // identical regions; loop transfers can repeat many exits.
                merge_complementary_guards(&mut parts);
                let joined = parts.into_iter().fold(Region::Empty, |a, b| {
                    let b = self.reduce_reached_region(b, batch, scope);
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
                            if self.optional_le(&facts, a, c)
                                && self.optional_le(&facts, c, b)
                                && self.optional_le(&facts, b, d)
                            {
                                parts[i] = Region::Linear(a, d);
                                parts.remove(j);
                                j = i + 1;
                                continue;
                            }
                            if self.optional_le(&facts, c, a)
                                && self.optional_le(&facts, a, d)
                                && self.optional_le(&facts, d, b)
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
                self.join_images(&mut parts, &facts);
                merge_complementary_guards(&mut parts);
                if parts.len() == 1 {
                    parts.pop().unwrap()
                } else {
                    Region::Union(parts)
                }
            }
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |a, b| {
                a.intersection(self.reduce_reached_region(b, batch, scope))
            }),
            Region::Bind(bound, inner) => {
                let child = self.normalization_bind_scope(batch, scope, &bound);
                let body = self.reduce_reached_region(*inner, batch, child);
                let finalized = self.finalize_reached_bind(bound, body, batch, scope);
                // A retained binder guards an unsupported shape whose child
                // geometry is already reduced. Newly exposed images/unions
                // still need their newly combined domains reduced.
                match finalized {
                    retained @ Region::Bind(_, _) => retained,
                    other => self.reduce_reached_region(other, batch, scope),
                }
            }
            other => other,
        }
    }
    /// Finish a binder whose body has already been reached under its scoped
    /// interval (and any retained guards). Distribution is structural: walking
    /// those bodies again repeats both scalar canonicalization and fact setup
    /// at every enclosing binder. Each union member clips its own bound, while
    /// lifted guards remain around the body that justified its normalization.
    fn finalize_reached_bind(
        &mut self,
        mut bound: Bound,
        mut inner: Region,
        batch: &mut NormalizationBatch,
        scope: usize,
    ) -> Region {
        let facts = batch.facts(scope);
        if let Region::Guard(path, guarded) = inner {
            let (clipped, remaining) = self.clip_bound_guard(bound, path, batch, scope);
            bound = clipped;
            // Clipping forms new min/max endpoints after the source-scope
            // reached walk. Reach only those new values in the finalizer's
            // outer world; installing this binder would give its own domain
            // authority over the extent being decided.
            let endpoints =
                prove::canonical_many_with_facts(self.arena(), &facts, &[bound.start, bound.end]);
            bound.start = endpoints[0];
            bound.end = endpoints[1];
            inner = Region::guarded(remaining, *guarded);
            if self.optional_le(&facts, bound.end, bound.start) {
                return Region::Empty;
            }
        }
        match inner {
            Region::Empty => Region::Empty,
            Region::Union(parts) => {
                let parts = parts
                    .into_iter()
                    .map(|p| self.finalize_reached_bind(bound.clone(), p, batch, scope))
                    .collect();
                Region::Union(parts)
            }
            Region::Image {
                mut domain,
                coordinates,
            } => {
                domain.insert(0, bound);
                Region::Image {
                    domain,
                    coordinates,
                }
            }
            Region::LinearImage {
                mut domain,
                address,
            } => {
                domain.insert(0, bound);
                Region::LinearImage { domain, address }
            }
            Region::Linear(start, end) => {
                let (symbol, index) = self.fresh_integer();
                let zero = self.arena().int(0);
                let width = self.arena().int_sub(end, start);
                let address = self.arena().int_add(start, index);
                self.reach_linear(
                    vec![
                        bound,
                        Bound {
                            symbol,
                            start: zero,
                            end: width,
                        },
                    ],
                    address,
                    batch,
                    scope,
                )
            }
            Region::Guard(path, inner)
                if path
                    .iter()
                    .all(|(c, _)| !self.condition_mentions(c, bound.symbol)) =>
            {
                let Some(child) = self.normalization_guard_scope(batch, scope, &path) else {
                    return Region::Empty;
                };
                Region::guarded(
                    path,
                    self.finalize_reached_bind(bound, *inner, batch, child),
                )
            }
            Region::Full if self.optional_lt(&facts, bound.start, bound.end) => Region::Full,
            other => Region::Bind(bound, Box::new(other)),
        }
    }
    /// Reach image endpoints in outer-to-inner order, and coordinates only
    /// after installing every domain. Geometry reduction is a separate phase.
    fn reach_image(
        &mut self,
        domain: Vec<Bound>,
        coordinates: Vec<IntExpr>,
        batch: &mut NormalizationBatch,
        scope: usize,
    ) -> Region {
        let (domain, child) = self.reached_domain(domain, batch, scope);
        let scoped = batch.facts(child);
        let coordinates = prove::canonical_many_with_facts(self.arena(), &scoped, &coordinates);
        Region::Image {
            domain,
            coordinates,
        }
    }
    /// Consume an already reached image. Its endpoints were reached in
    /// outer-to-inner order and coordinates under every retained domain.
    fn reduce_reached_image(
        &mut self,
        domain: Vec<Bound>,
        coordinates: Vec<IntExpr>,
        facts: &prove::Facts,
    ) -> Region {
        if domain
            .iter()
            .any(|d| self.optional_le(facts, d.end, d.start))
        {
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
            let renamed = self.symbol_value(other.symbol);
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
            let width = prove::canonical_with_facts(self.arena(), facts, width);
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
            if owner[d].is_none() && !self.optional_lt(facts, bound.start, bound.end) {
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
                    axes.push(self.symbol_value(symbol));
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
            if !self.optional_nonnegative(facts, width) {
                return None;
            }
            stride = self.arena().int_mul(stride, width);
        }
        let end = self.arena().int_add(start, stride);
        Some((
            prove::canonical_with_facts(self.arena(), facts, start),
            prove::canonical_with_facts(self.arena(), facts, end),
        ))
    }
    fn reach_linear(
        &mut self,
        domain: Vec<Bound>,
        address: IntExpr,
        batch: &mut NormalizationBatch,
        scope: usize,
    ) -> Region {
        let (domain, child) = self.reached_domain(domain, batch, scope);
        let scoped = batch.facts(child);
        let address = prove::canonical_with_facts(self.arena(), &scoped, address);
        Region::LinearImage { domain, address }
    }
    fn reduce_reached_linear(
        &mut self,
        domain: Vec<Bound>,
        address: IntExpr,
        facts: &prove::Facts,
    ) -> Region {
        if domain
            .iter()
            .any(|d| self.optional_le(facts, d.end, d.start))
        {
            return Region::Empty;
        }
        let mut substitutions = HashMap::new();
        let mut dimensions = vec![];
        for d in &domain {
            let width = self.arena().int_sub(d.end, d.start);
            let width = prove::canonical_with_facts(self.arena(), facts, width);
            if prove::constant(self.arena(), width) == Some(1) {
                substitutions.insert(d.symbol, d.start);
            } else {
                let end = self.arena().int_add(d.start, width);
                dimensions.push(Bound {
                    symbol: d.symbol,
                    start: d.start,
                    end,
                });
            }
        }
        let address = self.substitute(address, &substitutions);
        let address = prove::recompose_address(self.arena(), address);
        let mut used = vec![];
        for d in &dimensions {
            if prove::mentions(self.arena_ref(), address, d.symbol) {
                used.push(d.clone());
            } else if !self.optional_lt(facts, d.start, d.end) {
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
    /// Form the same elements in row-major coordinates without reaching their
    /// scalar expressions again. The consumer normalizes the complete result
    /// under its own facts, retaining every binder and guard scope.
    fn linearize(&mut self, region: Region, extents: &[IntExpr]) -> Region {
        match region {
            Region::Image {
                domain,
                coordinates,
            } => {
                let address = self.linear_address(&coordinates, extents);
                Region::LinearImage { domain, address }
            }
            Region::Union(parts) => parts.into_iter().fold(Region::Empty, |a, b| {
                let b = self.linearize(b, extents);
                a.union(b)
            }),
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |a, b| {
                let b = self.linearize(b, extents);
                a.intersection(b)
            }),
            Region::Bind(bound, inner) => {
                let inner = self.linearize(*inner, extents);
                Region::Bind(bound, Box::new(inner))
            }
            Region::Guard(path, inner) => {
                Region::Guard(path, Box::new(self.linearize(*inner, extents)))
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
                        // One complete prefix suffices. Do not construct bounds
                        // for unrelated partial writes before trying it.
                        if ranges.iter().zip(required).all(|(outer, inner)| {
                            self.le(facts, outer.0, inner.0) && self.le(facts, inner.1, outer.1)
                        }) {
                            return true;
                        }
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
                    map.insert(a.symbol, self.symbol_value(b.symbol));
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

    fn optional_same(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        let difference = self.arena().int_sub(a, b);
        prove::derive_zero(self.arena(), facts, difference).is_established()
    }
    fn optional_nonnegative(&mut self, facts: &prove::Facts, e: IntExpr) -> bool {
        prove::established_nonnegative(self.arena_ref(), facts, e)
    }
    fn optional_le(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        let d = self.arena().int_sub(b, a);
        self.optional_nonnegative(facts, d)
    }
    fn optional_lt(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        let d = self.arena().int_sub(b, a);
        let one = self.arena().int(1);
        let d = self.arena().int_sub(d, one);
        self.optional_nonnegative(facts, d)
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
            Condition::And(a, b) => {
                let left = self.condition_value(a, path, facts);
                match (left, self.condition_value(b, path, facts)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            Condition::Or(a, b) => {
                let left = self.condition_value(a, path, facts);
                match (left, self.condition_value(b, path, facts)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
            Condition::Compare(op, a, b) => {
                use BinaryOp::*;
                if matches!(op, Gt | Ge) {
                    let flipped = if *op == Gt { Lt } else { Le };
                    return self.condition_value(&Condition::Compare(flipped, *b, *a), path, facts);
                }
                if let Some(known) = facts.comparison(condition) {
                    return known;
                }
                // The operands' reached values in this world are equal to
                // them, so their comparison decides this one.
                let value = self.compare_value(*op, *a, *b, facts).or_else(|| {
                    let reached = prove::canonical_many_with_facts(self.arena(), facts, &[*a, *b]);
                    (reached[0] != *a || reached[1] != *b)
                        .then(|| self.compare_value(*op, reached[0], reached[1], facts))
                        .flatten()
                });
                facts.remember_comparison(condition.clone(), value);
                value
            }
            _ => None,
        }
    }
    /// A comparison of `Lt`, `Le`, `Eq` or `Ne` that `facts` decide directly.
    fn compare_value(
        &mut self,
        op: BinaryOp,
        a: IntExpr,
        b: IntExpr,
        facts: &prove::Facts,
    ) -> Option<bool> {
        use BinaryOp::*;
        match op {
            Eq if self.optional_same(facts, a, b) => Some(true),
            Ne if self.optional_same(facts, a, b) => Some(false),
            Eq if self.optional_lt(facts, a, b) || self.optional_lt(facts, b, a) => Some(false),
            Ne if self.optional_lt(facts, a, b) || self.optional_lt(facts, b, a) => Some(true),
            Lt if self.optional_lt(facts, a, b) => Some(true),
            Lt if self.optional_le(facts, b, a) => Some(false),
            Le if self.optional_le(facts, a, b) => Some(true),
            Le if self.optional_lt(facts, b, a) => Some(false),
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
        // Retain only premises guaranteed by every applicable alternative.
        // In particular, successful exit OR can imply a shared continuation
        // without granting any branch-specific premise.
        for (actual, truth) in guaranteed_actuals(&condition, value) {
            let actual = if truth {
                actual
            } else {
                self.arena().not(actual)
            };
            facts.assume_boolean(actual);
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
                if matches!((op, value), (BinaryOp::Eq, false) | (BinaryOp::Ne, true)) {
                    let difference = self.arena().int_sub(*a, *b);
                    facts.assume_nonzero(self.arena_ref(), difference);
                    let zero = self.arena().int(0);
                    let negative = self.arena().int_sub(zero, difference);
                    let positive = if self.optional_nonnegative(facts, difference) {
                        Some(difference)
                    } else if self.optional_nonnegative(facts, negative) {
                        Some(negative)
                    } else {
                        None
                    };
                    if let Some(positive) = positive {
                        let one = self.arena().int(1);
                        let strict = self.arena().int_sub(positive, one);
                        facts.assume_nonnegative(self.arena(), strict);
                    }
                    path.push((condition, value));
                    return true;
                }
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
                if equality || prove::directly_nonnegative(self.arena_ref(), facts, negative) {
                    // A nonpositive checked extent on an empty path is zero,
                    // including a quotient such as H/KV. Keep that equality
                    // when its opposite already has direct evidence. This
                    // optional strengthening must not search speculative
                    // arithmetic for every imported access predicate.
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
            // A guard on this bound coordinate is not a global branch. Exact
            // interval normalization handles supported comparisons; remaining
            // private guards must not become assumptions about every visit.
            Region::Bind(bound, inner) => self
                .unknown_guard(inner, path, facts)
                .filter(|condition| !self.condition_mentions(condition, bound.symbol)),
            _ => None,
        }
    }
    /// `region` without the members whose global guards `facts` leave
    /// undecided: a subset of it on every execution. Guards on a bound
    /// coordinate are private to their domain and stay as they are.
    fn settled_region(
        &mut self,
        region: Region,
        path: &Path,
        facts: &prove::Facts,
        bound: &mut Vec<SymbolId>,
    ) -> Region {
        match region {
            Region::Guard(guard, inner) => {
                let global = guard.iter().any(|(condition, value)| {
                    !bound
                        .iter()
                        .any(|symbol| self.condition_mentions(condition, *symbol))
                        && self.condition_value(condition, path, facts) != Some(*value)
                });
                if global {
                    Region::Empty
                } else {
                    Region::guarded(guard, self.settled_region(*inner, path, facts, bound))
                }
            }
            Region::Union(parts) => parts.into_iter().fold(Region::Empty, |a, b| {
                a.union(self.settled_region(b, path, facts, bound))
            }),
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |a, b| {
                a.intersection(self.settled_region(b, path, facts, bound))
            }),
            Region::Bind(domain, inner) => {
                bound.push(domain.symbol);
                let inner = self.settled_region(*inner, path, facts, bound);
                bound.pop();
                match inner {
                    Region::Empty => Region::Empty,
                    inner => Region::Bind(domain, Box::new(inner)),
                }
            }
            other => other,
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
        // These containments hold under every activation of their guards.
        // Discharging them before splitting unknown conditions avoids building
        // Boolean worlds whose coverage result is already unconditional.
        if matches!(available, Region::Full)
            || matches!(required, Region::Empty)
            || available == required
        {
            return true;
        }
        // A required union is covered member by member, and a guarded
        // requirement only where its guard holds: assume it there instead of
        // splitting every world on it.
        match required {
            Region::Union(parts) => {
                return parts
                    .iter()
                    .all(|part| self.covered(available, part, path, facts, extents));
            }
            Region::Guard(guard, inner) => {
                let mut path = path.clone();
                let mut facts = facts.clone();
                if !guard
                    .iter()
                    .all(|(condition, truth)| self.assume(&mut path, &mut facts, condition.clone(), *truth))
                {
                    return true;
                }
                return self.covered(available, inner, &path, &facts, extents);
            }
            _ => {}
        }
        // Nothing available: containment is exactly the emptiness of the
        // requirement, whose guards that proof assumes one scope at a time.
        if matches!(available, Region::Empty) {
            let required = self.normalize(required.clone(), facts);
            return self.required_region_empty(&required, path, facts, extents);
        }
        // A sufficient available member proves containment without activating
        // unrelated partial overwrites. Keep the original required proof for
        // that member; failure still reaches the full path-aware union below.
        if let Region::Union(parts) = available {
            for part in parts {
                if self.unknown_guard(part, path, facts).is_none()
                    && self.covered(part, required, path, facts, extents)
                {
                    return true;
                }
            }
        }
        if let Some(condition) = self
            .unknown_guard(available, path, facts)
            .or_else(|| self.unknown_guard(required, path, facts))
        {
            // The writes this world already decides may suffice together.
            // Dropping undecided members only shrinks what is available, so
            // a containment found here holds on every split below.
            if self.unknown_guard(required, path, facts).is_none() {
                let settled = self.settled_region(available.clone(), path, facts, &mut vec![]);
                if settled != *available
                    && self.unknown_guard(&settled, path, facts).is_none()
                    && self.covered(&settled, required, path, facts, extents)
                {
                    return true;
                }
            }
            return [false, true].into_iter().all(|value| {
                let mut path = path.clone();
                let mut facts = facts.clone();
                !self.assume(&mut path, &mut facts, condition.clone(), value)
                    || self.covered(available, required, &path, &facts, extents)
            });
        }
        let available = self.active_region(available.clone(), path, facts);
        let required = self.active_region(required.clone(), path, facts);
        let mut regions = self.normalize_many(vec![available, required], facts).into_iter();
        let available = regions.next().expect("normalized available region");
        let required = regions.next().expect("normalized required region");
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
            // Optional normalization may leave an empty quantified/guarded
            // image intact. Required admission owns its full emptiness proof.
            return self.required_region_empty(required, &Vec::new(), facts, extents);
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
        let available = self.linearize(available.clone(), extents);
        let required = match required {
            Region::Full => {
                let zero = self.arena().int(0);
                let elements = extents
                    .iter()
                    .fold(self.arena().int(1), |p, e| self.arena().int_mul(p, *e));
                Region::Linear(zero, elements)
            }
            required => {
                self.linearize(required.clone(), extents)
            }
        };
        let mut regions = self.normalize_many(vec![available, required], facts).into_iter();
        let available = regions.next().expect("normalized row-major availability");
        let required = regions.next().expect("normalized row-major requirement");
        self.covered_linear(&available, &required, facts)
    }

    /// Establish that a required region has no instance, with each private
    /// domain and guard scoped to its own body. Emptiness uses direct
    /// evidence only (a directly contradictory world, an established empty
    /// range); an undetected empty region stays required, which is sound.
    /// This grants no produced initialization.
    fn required_region_empty(
        &mut self,
        region: &Region,
        path: &Path,
        facts: &prove::Facts,
        extents: &[IntExpr],
    ) -> bool {
        if prove::directly_contradictory(self.arena_ref(), facts) {
            return true;
        }
        let zero = self.arena().int(0);
        match region {
            Region::Empty => true,
            Region::Full => extents
                .iter()
                .any(|extent| self.established_le(facts, *extent, zero)),
            Region::Linear(start, end) => self.established_le(facts, *end, *start),
            Region::Image { domain, .. } | Region::LinearImage { domain, .. } => {
                domain
                    .iter()
                    .any(|bound| self.established_le(facts, bound.end, bound.start))
                    || {
                        let scoped = self.domain_facts(facts, domain.iter());
                        prove::directly_contradictory(self.arena_ref(), &scoped)
                    }
            }
            Region::Bind(bound, inner) => {
                if self.established_le(facts, bound.end, bound.start) {
                    return true;
                }
                let scoped = self.domain_facts(facts, std::iter::once(bound));
                self.required_region_empty(inner, path, &scoped, extents)
            }
            Region::Guard(guard, inner) => {
                let mut path = path.clone();
                let mut scoped = facts.clone();
                if !guard.iter().all(|(condition, truth)| {
                    self.assume(&mut path, &mut scoped, condition.clone(), *truth)
                }) {
                    return true;
                }
                self.required_region_empty(inner, &path, &scoped, extents)
            }
            Region::Union(parts) => parts
                .iter()
                .all(|part| self.required_region_empty(part, path, facts, extents)),
            Region::Intersection(parts) => parts
                .iter()
                .any(|part| self.required_region_empty(part, path, facts, extents)),
        }
    }
    /// `a <= b` by established evidence, without a derivation search.
    fn established_le(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        let difference = self.arena().int_sub(b, a);
        prove::established_nonnegative(self.arena_ref(), facts, difference)
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
        let mut regions = self.normalize_many(vec![left, right], facts).into_iter();
        let left = regions.next().expect("normalized left footprint");
        let right = regions.next().expect("normalized right footprint");
        self.separated_normalized_regions(left, right, facts, extents)
    }
    /// Recursive comparison consumes the reached footprints. Descending into
    /// a union or intersection does not reconstruct its scalar scopes.
    fn separated_normalized_regions(
        &mut self,
        left: Region,
        right: Region,
        facts: &prove::Facts,
        extents: &[IntExpr],
    ) -> bool {
        match (left, right) {
            (Region::Empty, _) | (_, Region::Empty) => true,
            (Region::Union(parts), right) => parts
                .into_iter()
                .all(|part| self.separated_normalized_regions(part, right.clone(), facts, extents)),
            (left, Region::Union(parts)) => parts
                .into_iter()
                .all(|part| self.separated_normalized_regions(left.clone(), part, facts, extents)),
            (Region::Intersection(parts), right) => parts
                .into_iter()
                .any(|part| self.separated_normalized_regions(part, right.clone(), facts, extents)),
            (left, Region::Intersection(parts)) => parts
                .into_iter()
                .any(|part| self.separated_normalized_regions(left.clone(), part, facts, extents)),
            (Region::Guard(_, inner), right) => {
                self.separated_normalized_regions(*inner, right, facts, extents)
            }
            (left, Region::Guard(_, inner)) => {
                self.separated_normalized_regions(left, *inner, facts, extents)
            }
            // A retained binder is one more domain coordinate of the images
            // beneath it; a guard beneath it only narrows that footprint.
            (Region::Bind(bound, inner), right) => match bind_domain(bound, *inner) {
                Some(left) => self.separated_normalized_regions(left, right, facts, extents),
                None => false,
            },
            (left, Region::Bind(bound, inner)) => match bind_domain(bound, *inner) {
                Some(right) => self.separated_normalized_regions(left, right, facts, extents),
                None => false,
            },
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
                let left = self.linearize(left, extents);
                let right = self.linearize(right, extents);
                let mut regions = self.normalize_many(vec![left, right], facts).into_iter();
                let left = regions.next().expect("normalized left row-major footprint");
                let right = regions.next().expect("normalized right row-major footprint");
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
    /// `boundary_paths` of `path` when it has at most one, and otherwise the
    /// one path of their disjunction: the same public condition, without
    /// enumerating (and copying an effect into) every combination.
    fn closed_paths(&self, path: &Path, allowed: &[SymbolId]) -> Vec<Path> {
        fn count<T: RegionOps + ?Sized>(
            owner: &T,
            c: &Condition,
            truth: bool,
            allowed: &[SymbolId],
        ) -> usize {
            match (c, truth) {
                (Condition::Not(c), truth) => count(owner, c, !truth, allowed),
                (Condition::And(a, b), true) | (Condition::Or(a, b), false) => {
                    count(owner, a, truth, allowed).saturating_mul(count(owner, b, truth, allowed))
                }
                (Condition::And(a, b), false) | (Condition::Or(a, b), true) => {
                    count(owner, a, truth, allowed).saturating_add(count(owner, b, truth, allowed))
                }
                _ => 1,
            }
        }
        /// The public projection of one literal; `None` is unconstrained.
        fn projected<T: RegionOps + ?Sized>(
            owner: &T,
            c: &Condition,
            truth: bool,
            allowed: &[SymbolId],
        ) -> Option<Condition> {
            match (c, truth) {
                (Condition::Not(c), truth) => projected(owner, c, !truth, allowed),
                (Condition::And(a, b), true) | (Condition::Or(a, b), false) => {
                    match (
                        projected(owner, a, truth, allowed),
                        projected(owner, b, truth, allowed),
                    ) {
                        (Some(a), Some(b)) => Some(Condition::And(Box::new(a), Box::new(b))),
                        (one, None) | (None, one) => one,
                    }
                }
                (Condition::And(a, b), false) | (Condition::Or(a, b), true) => {
                    let a = projected(owner, a, truth, allowed)?;
                    let b = projected(owner, b, truth, allowed)?;
                    Some(Condition::Or(Box::new(a), Box::new(b)))
                }
                _ if owner.boundary_condition(c, allowed) => Some(if truth {
                    c.clone()
                } else {
                    Condition::Not(Box::new(c.clone()))
                }),
                _ => None,
            }
        }
        let combinations = path.iter().fold(1usize, |total, (c, truth)| {
            total.saturating_mul(count(self, c, *truth, allowed))
        });
        if combinations <= 1 {
            return self.boundary_paths(path, allowed);
        }
        let condition = path
            .iter()
            .filter_map(|(c, truth)| projected(self, c, *truth, allowed))
            .reduce(|a, b| Condition::And(Box::new(a), Box::new(b)));
        vec![condition.map_or_else(Vec::new, |condition| vec![(condition, true)])]
    }
    fn boundary_region(
        &mut self,
        region: Region,
        allowed: &[SymbolId],
        required: bool,
        axes: Option<&[IntExpr]>,
    ) -> Region {
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
                    // A may-read/access can forget a private coordinate without
                    // forgetting the other coordinates of this same image.
                    // The replacement range comes from this parameter's actual
                    // logical geometry, never from a guessed physical width.
                    let Some(axes) = axes.filter(|axes| {
                        required
                            && axes.len() == coordinates.len()
                            && axes.iter().all(|axis| known(self, *axis, allowed))
                    }) else {
                        return unknown;
                    };
                    let mut scope = allowed.to_vec();
                    let mut projected_domain = Vec::new();
                    for bound in domain {
                        if known(self, bound.start, &scope) && known(self, bound.end, &scope) {
                            scope.push(bound.symbol);
                            projected_domain.push(bound);
                        }
                    }
                    let mut projected_coordinates = Vec::new();
                    for (coordinate, extent) in coordinates.into_iter().zip(axes) {
                        if known(self, coordinate, &scope) {
                            projected_coordinates.push(coordinate);
                        } else {
                            let symbol = self.fresh_variable();
                            let start = self.arena().int(0);
                            projected_domain.push(Bound {
                                symbol,
                                start,
                                end: *extent,
                            });
                            projected_coordinates.push(self.symbol_value(symbol));
                        }
                    }
                    Region::Image {
                        domain: projected_domain,
                        coordinates: projected_coordinates,
                    }
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
                    return if required {
                        // Existentially forgetting this unsupported private
                        // domain widens only coordinates that depend on it.
                        self.boundary_region(*inner, allowed, true, axes)
                    } else {
                        unknown
                    };
                }
                let mut scope = allowed.to_vec();
                scope.push(bound.symbol);
                Region::Bind(
                    bound,
                    Box::new(self.boundary_region(*inner, &scope, required, axes)),
                )
            }
            Region::Guard(path, inner) => {
                let inner = self.boundary_region(*inner, allowed, required, axes);
                if required {
                    self.closed_paths(&path, allowed)
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
                s.union(self.boundary_region(r, allowed, required, axes))
            }),
            Region::Intersection(parts) => parts.into_iter().fold(Region::Full, |s, r| {
                s.intersection(self.boundary_region(r, allowed, required, axes))
            }),
            other => other,
        }
    }
    fn close_transfer(
        &mut self,
        contract: InitializationContract,
        allowed: &[SymbolId],
        parameter_axes: &HashMap<ParameterPath, Vec<IntExpr>>,
    ) -> InitializationContract {
        let mut accesses = vec![];
        for access in contract.accesses {
            // Accesses are a may-set: losing a private coordinate or predicate
            // must widen the footprint, never erase a possible conflict.
            let region = self.boundary_region(
                access.region,
                allowed,
                true,
                parameter_axes.get(&access.parameter).map(Vec::as_slice),
            );
            for path in self.closed_paths(&access.path, allowed) {
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
            let region = self.boundary_region(
                requirement.region,
                allowed,
                true,
                parameter_axes
                    .get(&requirement.parameter)
                    .map(Vec::as_slice),
            );
            for path in self.closed_paths(&requirement.path, allowed) {
                requirements.push(Requirement {
                    parameter: requirement.parameter.clone(),
                    region: region.clone(),
                    path,
                    span: requirement.span,
                });
            }
        }
        let mut writes = HashMap::<ParameterPath, Region>::new();
        let mut returned = vec![];
        for exit in contract.exits {
            let paths = self.closed_paths(&exit.path, allowed);
            returned.extend(paths.iter().cloned());
            for (parameter, region) in exit.written {
                let region = self.boundary_region(
                    region,
                    allowed,
                    false,
                    parameter_axes.get(&parameter).map(Vec::as_slice),
                );
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
                // Writes remain the intersection of all compatible private
                // outcomes; successful return retains its projected domain.
                path: vec![(successful_paths(&returned), true)],
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
        // Pullback preserves constant sets without a coverage or vacuity query.
        // Partial regions still need whole-view promotion before private facts
        // disappear at export; an exact address shift alone cannot replace it.
        if matches!(&region, Region::Empty | Region::Full) {
            return region;
        }
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
        let region = self.linearize(region, &view.extents);
        let region = self.normalize(region, facts);
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
/// The footprint of `bound` over `inner` with its binder made a domain
/// coordinate of each image beneath it and inner guards dropped (a superset),
/// or `None` for a shape without that image form.
fn bind_domain(bound: Bound, inner: Region) -> Option<Region> {
    match inner {
        Region::Empty => Some(Region::Empty),
        Region::Guard(_, inner) => bind_domain(bound, *inner),
        Region::Image {
            mut domain,
            coordinates,
        } => {
            domain.insert(0, bound);
            Some(Region::Image {
                domain,
                coordinates,
            })
        }
        Region::LinearImage {
            mut domain,
            address,
        } => {
            domain.insert(0, bound);
            Some(Region::LinearImage { domain, address })
        }
        Region::Bind(inner_bound, body) => {
            let body = bind_domain(inner_bound, *body)?;
            bind_domain(bound, body)
        }
        Region::Union(parts) => parts
            .into_iter()
            .map(|part| bind_domain(bound.clone(), part))
            .collect::<Option<Vec<_>>>()
            .map(Region::Union),
        Region::Full | Region::Linear(..) | Region::Intersection(_) => None,
    }
}

fn merge_complementary_guards(parts: &mut Vec<Region>) {
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
    // Only paths with identical underlying regions can merge. Index those
    // regions once instead of comparing every pair after every merge.
    let mut indices = HashMap::new();
    let mut groups: Vec<(Region, Vec<Path>)> = Vec::new();
    for part in std::mem::take(parts) {
        let (path, region) = match part {
            Region::Guard(path, inner) => (path, *inner),
            region => (Vec::new(), region),
        };
        let index = *indices.entry(region.clone()).or_insert_with(|| {
            groups.push((region, Vec::new()));
            groups.len() - 1
        });
        if !groups[index].1.contains(&path) {
            groups[index].1.push(path);
        }
    }
    for (region, mut paths) in groups {
        loop {
            if paths.iter().any(Vec::is_empty) {
                paths = vec![Vec::new()];
                break;
            }
            let mut merged = false;
            'search: for i in 0..paths.len() {
                for j in i + 1..paths.len() {
                    if let Some(path) = complement(&paths[i], &paths[j]) {
                        paths[i] = path;
                        paths.remove(j);
                        merged = true;
                        break 'search;
                    }
                }
            }
            if !merged {
                break;
            }
        }
        parts.extend(
            paths
                .into_iter()
                .map(|path| Region::guarded(path, region.clone())),
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ParameterAccess {
    pub(crate) parameter: ParameterPath,
    pub(crate) region: Region,
    pub(crate) path: Path,
    pub(crate) write: bool,
    /// The combining operation of an atomic access. Only same-operation
    /// atomic accesses commute (L12).
    pub(crate) atomic: Option<AtomicOp>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum ParameterPart {
    Integer(ParameterPath),
    Scalar(ParameterPath),
    Start(ParameterPath),
    End(ParameterPath),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Exit {
    pub(crate) path: Path,
    pub(crate) written: Vec<(ParameterPath, Region)>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
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

    /// Both closed contracts' outcomes: success on either success path, and
    /// each write guaranteed by every outcome (a write one side omits is
    /// unrestricted there). `self` may be the empty contract.
    pub(crate) fn conjoin_closed(self, other: Self) -> Self {
        let Some(prior) = self.exits.into_iter().next() else {
            return other;
        };
        let mut exits = other.exits.into_iter();
        let next = exits.next().expect("a closed contract has one exit");
        let success = |exit: &Exit| match exit.path.as_slice() {
            [(condition, true)] => condition.clone(),
            _ => unreachable!("a closed contract exit is one success condition"),
        };
        let path = vec![(
            Condition::Or(Box::new(success(&prior)), Box::new(success(&next))),
            true,
        )];
        let mut written = prior.written;
        for (parameter, region) in next.written {
            match written.iter_mut().find(|(known, _)| *known == parameter) {
                Some((_, known)) => {
                    let known_region = std::mem::replace(known, Region::Empty);
                    *known = known_region.intersection(region);
                }
                None => written.push((parameter, region)),
            }
        }
        written.sort_by(|a, b| a.0.cmp(&b.0));
        let mut requirements = self.requirements;
        requirements.extend(other.requirements);
        let mut accesses = self.accesses;
        accesses.extend(other.accesses);
        Self {
            requirements,
            exits: vec![Exit { path, written }],
            symbols: other.symbols,
            accesses,
        }
    }
}

/// The checked loop's captured writes and owned carry invariants. Both are
/// derived by source checking and instantiated on the loop's actual parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
            mapping.value_symbol(*symbol);
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
    Scalar(ErasedScalarExpr),
    Range {
        start: IntExpr,
        end: IntExpr,
    },
    Predicate {
        scalar: ErasedScalarExpr,
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
        let condition = self.actual_condition(value, binders);
        // The same assumption the checker makes on an arm: a comparison
        // becomes a fact over its operands. A contradictory assumption is an
        // unreached arm; its path stays as recorded.
        let (mut path, mut facts) = (std::mem::take(&mut self.path), self.facts.clone());
        if RegionOps::assume(self, &mut path, &mut facts, condition.clone(), truth) {
            self.path = path;
            self.facts = facts;
        } else {
            self.path = path;
            self.path.push((condition, truth));
        }
    }
    /// An exact integer comparison keeps its meaning (so path facts can
    /// establish regions over its operands); any other predicate is opaque.
    fn actual_condition(&mut self, value: BoolExpr, binders: &[SymbolId]) -> Condition {
        use crate::expr::CmpOp;
        match self.arena.view(AnyExpr::Bool(value)) {
            NodeView::Cmp { op, lhs, rhs } => {
                let (a, b) = match (lhs, rhs) {
                    (AnyExpr::Int(a), AnyExpr::Int(b)) => (a, b),
                    (AnyExpr::Nat(a), AnyExpr::Nat(b)) => {
                        (self.arena.int_from_nat(a), self.arena.int_from_nat(b))
                    }
                    _ => return Condition::Actual(value, binders.to_vec()),
                };
                let op = match op {
                    CmpOp::Eq => BinaryOp::Eq,
                    CmpOp::Ne => BinaryOp::Ne,
                    CmpOp::Lt => BinaryOp::Lt,
                    CmpOp::Le => BinaryOp::Le,
                    CmpOp::Gt => BinaryOp::Gt,
                    CmpOp::Ge => BinaryOp::Ge,
                };
                Condition::Compare(op, a, b)
            }
            NodeView::Unary {
                op: crate::expr::UnaryOp::Not,
                operand: AnyExpr::Bool(inner),
            } => Condition::Not(Box::new(self.actual_condition(inner, binders))),
            _ => Condition::Actual(value, binders.to_vec()),
        }
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
        let Some(completed) = Self::completed_loop_region(before, iteration, binder, start, end)
        else {
            return before.clone();
        };
        InitializationState(self.normalize(completed, &self.facts.clone()))
    }
    /// Form a loop effect without installing the synthetic iteration's facts
    /// in its caller. Both single-state and multi-output consumers use this
    /// same formation rule, then normalize in their parent context.
    fn completed_loop_region(
        before: &InitializationState,
        iteration: &InitializationState,
        binder: SymbolId,
        start: IntExpr,
        end: IntExpr,
    ) -> Option<Region> {
        // An unchanged initialization region needs no loop-binder expansion.
        // This is common for captured inputs and preserves the empty-loop case.
        if before.0 == iteration.0 {
            return None;
        }
        // Only what an iteration adds is expanded over the binder. A member
        // the iteration carries unchanged from before the loop was formed
        // outside it, so it cannot mention the binder: the union over every
        // iteration contains it exactly when `before` does. Expanding it
        // again would double the region at every enclosing loop.
        let carried = before
            .0
            .clone()
            .members()
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let mut added = iteration
            .0
            .clone()
            .members()
            .into_iter()
            .filter(|member| !carried.contains(member))
            .collect::<Vec<_>>();
        let added = match added.len() {
            0 => return None,
            1 => added.pop().expect("one added member"),
            _ => Region::Union(added),
        };
        let completed = Region::Bind(
            Bound {
                symbol: binder,
                start,
                end,
            },
            Box::new(added),
        );
        Some(before.0.clone().union(completed))
    }
    /// The state after an `if` join (see [`Region::branch`]).
    pub fn branch(
        &mut self,
        condition: BoolExpr,
        binders: &[SymbolId],
        then_state: &InitializationState,
        else_state: &InitializationState,
    ) -> InitializationState {
        // The join's guard has the same meaning an arm's assumption of the
        // condition records (`assume`): an exact comparison stays a
        // comparison of its operands, so path facts decide it.
        let condition = self.actual_condition(condition, binders);
        InitializationState(Region::branch(
            &condition,
            then_state.0.clone(),
            else_state.0.clone(),
        ))
    }
}

pub(crate) trait RegionMapping {
    fn integer(&mut self, value: IntExpr) -> IntExpr;
    fn binder(&mut self, symbol: SymbolId) -> SymbolId;
    /// A parameter value is not a geometric binder; retain its actual sort.
    fn value_symbol(&mut self, symbol: SymbolId) -> SymbolId {
        self.binder(symbol)
    }
    fn parameter(&mut self, path: &ParameterPath) -> ParameterPath;
    fn predicate(&mut self, path: &ParameterPath) -> Condition {
        Condition::Parameter(self.parameter(path))
    }
    fn boolean(&mut self, value: BoolExpr) -> (BoolExpr, Vec<SymbolId>);
    fn condition(&mut self, c: &Condition) -> Condition {
        match c {
            Condition::Actual(value, _) => {
                let (value, binders) = self.boolean(*value);
                Condition::Actual(value, binders)
            }
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
                        self.value_symbol(*s),
                        match p {
                            ParameterPart::Integer(p) => ParameterPart::Integer(self.parameter(p)),
                            ParameterPart::Scalar(p) => ParameterPart::Scalar(self.parameter(p)),
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
    bound: HashMap<SymbolId, AnyExpr>,
    leaves: &'c [(usize, Vec<usize>, usize)],
}
impl RegionMapping for EntryMapping<'_, '_, '_> {
    fn boolean(&mut self, value: BoolExpr) -> (BoolExpr, Vec<SymbolId>) {
        let bound = &self.bound;
        let map = &mut self.map;
        let value = xfer::transfer_bool(self.source, value, self.target, &mut |s, a| {
            bound.get(&s).copied().unwrap_or_else(|| map(s, a))
        });
        (value, self.target.free_symbols(value.into()))
    }
    fn integer(&mut self, value: IntExpr) -> IntExpr {
        let bound = &self.bound;
        let map = &mut self.map;
        xfer::transfer_int(self.source, value, self.target, &mut |s, a| {
            bound.get(&s).copied().unwrap_or_else(|| map(s, a))
        })
    }
    fn binder(&mut self, symbol: SymbolId) -> SymbolId {
        let expression = *self.bound.entry(symbol).or_insert_with(|| {
            let s = self.target.loop_binder().1;
            AnyExpr::Int(self.target.int_symbol(s))
        });
        match self.target.view(expression) {
            NodeView::Symbol(s) => s,
            _ => unreachable!("formal binder is a symbol"),
        }
    }
    fn value_symbol(&mut self, symbol: SymbolId) -> SymbolId {
        let expression =
            *self
                .bound
                .entry(symbol)
                .or_insert_with(|| match self.source.symbol_sort(symbol) {
                    SymbolSort::Int => {
                        let s = self.target.loop_binder().1;
                        AnyExpr::Int(self.target.int_symbol(s))
                    }
                    SymbolSort::Nat => {
                        let s = self.target.checker_variable(SymbolSort::Nat);
                        AnyExpr::Nat(self.target.nat_symbol(s))
                    }
                    SymbolSort::Scalar(dtype) => {
                        let s = self.target.checker_variable(SymbolSort::Scalar(dtype));
                        AnyExpr::Scalar(self.target.scalar_symbol_erased(s))
                    }
                });
        match self.target.view(expression) {
            NodeView::Symbol(s) => s,
            _ => unreachable!("formal parameter maps to its canonical value symbol"),
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
    /// `symbol` is the value of the integer semantic parameter `ordinal`.
    pub(crate) fn bind_integer_parameter(&mut self, symbol: SymbolId, ordinal: usize) {
        assert!(
            self.symbols.iter().all(|(prior, _)| *prior != symbol),
            "contract symbol bound twice"
        );
        self.symbols
            .push((symbol, ParameterPart::Integer(ParameterPath::root(ordinal))));
    }
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
        // Each contract symbol keeps its sort: an integer parameter's value
        // is an integer symbol, a typed scalar parameter's a scalar symbol.
        for (symbol, _) in &self.symbols {
            mapping.value_symbol(*symbol);
        }
        mapping.contract(self)
    }
}
struct ApplicationMapping<'a, 'b> {
    arena: &'a mut ExprArena,
    arguments: &'b [InitializationArgument],
    symbols: HashMap<SymbolId, AnyExpr>,
}
impl RegionMapping for ApplicationMapping<'_, '_> {
    fn boolean(&mut self, value: BoolExpr) -> (BoolExpr, Vec<SymbolId>) {
        let replacements = self
            .symbols
            .iter()
            .map(|(s, v)| (*s, *v))
            .collect::<Vec<_>>();
        let value = match self
            .arena
            .substitute_any(AnyExpr::Bool(value), &replacements)
        {
            AnyExpr::Bool(value) => value,
            _ => unreachable!("predicate substitution preserves Bool sort"),
        };
        (value, self.arena.free_symbols(value.into()))
    }
    fn integer(&mut self, value: IntExpr) -> IntExpr {
        let replacements = self
            .symbols
            .iter()
            .map(|(s, v)| (*s, *v))
            .collect::<Vec<_>>();
        match self
            .arena
            .substitute_any(AnyExpr::Int(value), &replacements)
        {
            AnyExpr::Int(value) => value,
            _ => unreachable!("integer substitution preserves its result sort"),
        }
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
            InitializationArgument::Predicate { value, binders, .. } => {
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
                    Some(InitializationArgument::Integer(v)) => Some(AnyExpr::Int(*v)),
                    _ => None,
                },
                ParameterPart::Scalar(p) => match arguments.get(p.parameter) {
                    Some(InitializationArgument::Scalar(v))
                    | Some(InitializationArgument::Predicate { scalar: v, .. }) => {
                        Some(AnyExpr::Scalar(*v))
                    }
                    Some(InitializationArgument::Boolean(v)) => Some(AnyExpr::Scalar(
                        self.arena
                            .scalar_const_erased(crate::reference_math::ReferenceScalar::Bool(*v)),
                    )),
                    _ => None,
                },
                ParameterPart::Start(p) => match arguments.get(p.parameter) {
                    Some(InitializationArgument::Range { start, .. }) => Some(AnyExpr::Int(*start)),
                    _ => None,
                },
                ParameterPart::End(p) => match arguments.get(p.parameter) {
                    Some(InitializationArgument::Range { end, .. }) => Some(AnyExpr::Int(*end)),
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
        let parameter_axes = arguments
            .iter()
            .enumerate()
            .filter_map(|(ordinal, argument)| match argument {
                InitializationArgument::Tensor { view, .. } => {
                    Some((ParameterPath::root(ordinal), view.axes.clone()))
                }
                _ => None,
            })
            .collect();
        self.close_transfer(mapped, &allowed, &parameter_axes)
    }
    /// Apply the checked transfer to the actual argument views. Tensor outputs
    /// are root-relative and already include their incoming initialization.
    pub fn apply(
        &mut self,
        contract: &InitializationContract,
        arguments: &[InitializationArgument],
    ) -> Result<Vec<Option<InitializationState>>, InitializationFailure> {
        let mut transition = InitializationContext {
            arena: &mut *self.arena,
            path: self.path.clone(),
            facts: self.facts.clone(),
        };
        let outputs = transition.apply_transition(contract, arguments)?;
        // Publish only the complete successful transition. An input or output
        // failure retains the exact incoming continuation of the caller.
        self.path = transition.path;
        self.facts = transition.facts;
        Ok(outputs)
    }

    fn apply_transition(
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
            let (path, facts) = (self.path.clone(), self.facts.clone());
            // An initialized whole root supplies every requirement on it,
            // whatever quantified shape the requirement takes.
            if !self.covered(&state.0, &region, &path, &facts, &view.extents)
                && !self.covered(&state.0, &Region::Full, &path, &facts, &view.extents)
            {
                return Err(InitializationFailure {
                    parameter,
                    phase: InitializationPhase::Input,
                });
            }
        }
        // Requirements are checked in the caller's pre-return context. Only
        // then is the callee's successful continuation available.
        let returned = successful_paths(
            &contract
                .exits
                .iter()
                .map(|exit| exit.path.clone())
                .collect::<Vec<_>>(),
        );
        let (mut path, mut facts) = (std::mem::take(&mut self.path), self.facts.clone());
        if !RegionOps::assume(self, &mut path, &mut facts, returned.clone(), true) {
            path.push((returned, true));
        }
        self.path = path;
        self.facts = facts;
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
                state.0 = std::mem::replace(&mut state.0, Region::Empty).union(region);
            }
        }
        // All exits contribute a union under the same facts. Normalize once
        // after collecting them, rather than repeatedly revisiting every
        // preceding exit for each additional path.
        let regions = outputs
            .iter_mut()
            .flatten()
            .map(|state| std::mem::replace(&mut state.0, Region::Empty))
            .collect();
        let regions = self.normalize_many(regions, &self.facts.clone());
        for (state, region) in outputs.iter_mut().flatten().zip(regions) {
            state.0 = region;
        }
        Ok(outputs)
    }
    /// A hypothetical transfer starts from the current incoming context and
    /// owns its successful-return premises. It never publishes them to the
    /// enclosing call, alternative comparison, or another loop iteration.
    fn preview_apply(
        &mut self,
        contract: &InitializationContract,
        arguments: &[InitializationArgument],
    ) -> Result<Vec<Option<InitializationState>>, InitializationFailure> {
        let mut hypothesis = InitializationContext {
            arena: &mut *self.arena,
            path: self.path.clone(),
            facts: self.facts.clone(),
        };
        hypothesis.apply_transition(contract, arguments)
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
        let produced = self.preview_apply(&contract.transfer, &iteration)?;
        let mut outputs = Vec::with_capacity(arguments.len());
        let mut changed_slots = Vec::new();
        let mut regions = Vec::new();
        for (slot, (argument, output)) in arguments.iter().zip(produced).enumerate() {
            match (argument, output) {
                (InitializationArgument::Tensor { state, .. }, Some(output)) => {
                    if let Some(region) =
                        Self::completed_loop_region(state, &output, symbol, start, end)
                    {
                        changed_slots.push(slot);
                        regions.push(region);
                    }
                    outputs.push(Some(state.clone()));
                }
                _ => outputs.push(None),
            }
        }
        // The synthetic iteration was an isolated hypothesis. Only its
        // effects are quantified here; normalization uses the parent facts.
        let regions = self.normalize_many(regions, &self.facts.clone());
        for (slot, region) in changed_slots.into_iter().zip(regions) {
            outputs[slot] = Some(InitializationState(region));
        }
        Ok(outputs)
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
        let candidate = self.preview_apply(candidate, arguments)?;
        let reference = self.preview_apply(reference, arguments)?;
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

    #[test]
    fn normalization_batch_reuses_exact_scopes_and_isolates_guard_premises() {
        let mut arena = ExprArena::new();
        let (_, row, r) = arena.loop_binder();
        let (_, count, n) = arena.loop_binder();
        let zero = arena.int(0);
        let two = arena.int(2);
        let three = arena.int(3);
        let four = arena.int(4);
        let column = arena.int_min(n, two);
        let bound = Bound {
            symbol: row,
            start: zero,
            end: two,
        };
        let at_zero = vec![(Condition::Compare(BinaryOp::Eq, n, zero), true)];
        let at_four = vec![
            (Condition::Compare(BinaryOp::Eq, n, zero), false),
            (Condition::Compare(BinaryOp::Eq, n, four), true),
        ];
        let mut facts = prove::Facts::new();
        facts.set_range(count, zero, four);
        let mut context = InitializationContext::new(&mut arena);
        let mut batch = NormalizationBatch::new(&facts);
        let child = context.normalization_bind_scope(&mut batch, 0, &bound);
        assert_eq!(
            child,
            context.normalization_bind_scope(&mut batch, 0, &bound)
        );
        let first = context
            .normalization_guard_scope(&mut batch, child, &at_zero)
            .unwrap();
        assert_eq!(
            first,
            context
                .normalization_guard_scope(&mut batch, child, &at_zero)
                .unwrap()
        );
        let other = context
            .normalization_guard_scope(&mut batch, child, &at_four)
            .unwrap();
        assert_ne!(first, other);
        let first_facts = batch.facts(first);
        let other_facts = batch.facts(other);
        assert_eq!(
            prove::canonical_with_facts(context.arena(), &first_facts, column),
            zero
        );
        assert_eq!(
            prove::canonical_with_facts(context.arena(), &other_facts, column),
            two
        );
        assert_eq!(
            prove::canonical_with_facts(context.arena(), &facts, column),
            column
        );

        let make = |path: Path| {
            Region::Bind(
                bound.clone(),
                Box::new(Region::guarded(
                    path,
                    Region::Image {
                        domain: vec![],
                        coordinates: vec![r, column],
                    },
                )),
            )
        };
        let first_input = make(at_zero.clone());
        let other_input = make(at_four.clone());
        let results = context.normalize_many(
            vec![
                first_input.clone(),
                other_input.clone(),
                first_input.clone(),
            ],
            &facts,
        );
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], results[2]);
        assert_eq!(results[0], context.normalize(first_input, &facts));
        assert_eq!(results[1], context.normalize(other_input, &facts));
        let zero_column = Region::Image {
            domain: vec![],
            coordinates: vec![zero, zero],
        };
        let two_column = Region::Image {
            domain: vec![],
            coordinates: vec![zero, two],
        };
        // A reached caller world owns both its path and the corresponding
        // numeric premises. Do not use the normalization batch's private
        // binder facts or a raw path paired with the unassumed parent facts.
        let mut zero_path = vec![];
        let mut zero_facts = facts.clone();
        for (condition, truth) in &at_zero {
            assert!(RegionOps::assume(
                &mut context,
                &mut zero_path,
                &mut zero_facts,
                condition.clone(),
                *truth,
            ));
        }
        let mut four_path = vec![];
        let mut four_facts = facts.clone();
        for (condition, truth) in &at_four {
            assert!(RegionOps::assume(
                &mut context,
                &mut four_path,
                &mut four_facts,
                condition.clone(),
                *truth,
            ));
        }
        assert!(context.covered(
            &results[0],
            &zero_column,
            &zero_path,
            &zero_facts,
            &[two, three]
        ));
        assert!(!context.covered(
            &results[0],
            &two_column,
            &zero_path,
            &zero_facts,
            &[two, three]
        ));
        assert!(context.covered(
            &results[1],
            &two_column,
            &four_path,
            &four_facts,
            &[two, three]
        ));
        assert!(!context.covered(
            &results[1],
            &zero_column,
            &four_path,
            &four_facts,
            &[two, three]
        ));
        assert!(!context.covered(&results[0], &zero_column, &vec![], &facts, &[two, three]));
        assert_eq!(
            prove::canonical_with_facts(context.arena(), &facts, column),
            column
        );
    }

    #[test]
    fn normalization_batch_keeps_clipped_members_and_dependent_endpoints_separate() {
        let mut arena = ExprArena::new();
        let (_, row, r) = arena.loop_binder();
        let (_, column, c) = arena.loop_binder();
        let (_, _, unknown_extent) = arena.loop_binder();
        let zero = arena.int(0);
        let one = arena.int(1);
        let two = arena.int(2);
        let three = arena.int(3);
        let four = arena.int(4);
        let row_end = arena.int_add(r, one);
        let dependent_end = arena.int_min(row_end, four);
        let bound = Bound {
            symbol: row,
            start: zero,
            end: four,
        };
        let point = Region::Image {
            domain: vec![],
            coordinates: vec![r],
        };
        let clipped = |op, limit| {
            Region::Bind(
                bound.clone(),
                Box::new(Region::guarded(
                    vec![(Condition::Compare(op, r, limit), true)],
                    point.clone(),
                )),
            )
        };
        let low = clipped(BinaryOp::Lt, two);
        let high = clipped(BinaryOp::Ge, three);
        let triangular = Region::Bind(
            bound.clone(),
            Box::new(Region::Bind(
                Bound {
                    symbol: column,
                    start: zero,
                    end: dependent_end,
                },
                Box::new(Region::Image {
                    domain: vec![],
                    coordinates: vec![r, c],
                }),
            )),
        );
        let empty = Region::Bind(bound.clone(), Box::new(Region::Empty));
        let facts = prove::Facts::new();
        let mut context = InitializationContext::new(&mut arena);
        let inputs = vec![low, high, triangular, empty];
        let results = context.normalize_many(inputs.clone(), &facts);
        for (input, result) in inputs.into_iter().zip(&results) {
            assert_eq!(*result, context.normalize(input, &facts));
            assert_eq!(*result, context.normalize(result.clone(), &facts));
        }
        let gap = Region::Image {
            domain: vec![],
            coordinates: vec![two],
        };
        let union = results[0].clone().union(results[1].clone());
        assert!(!context.covered(&union, &gap, &vec![], &facts, &[four]));
        // Normalization preserves the dependent domain exactly. Coverage's
        // rectangular-image rules do not invert triangular domains to points.
        let Region::Image {
            domain,
            coordinates,
        } = &results[2]
        else {
            panic!(
                "triangular batch member retains its domains: {:?}",
                results[2]
            );
        };
        assert!(context.same_image(
            domain,
            coordinates,
            &[
                bound.clone(),
                Bound {
                    symbol: column,
                    start: zero,
                    end: row_end,
                },
            ],
            &[r, c],
        ));
        let forbidden = Region::Image {
            domain: vec![],
            coordinates: vec![zero, one],
        };
        assert!(!context.covered(&results[2], &forbidden, &vec![], &facts, &[four, four]));
        assert_eq!(results[3], Region::Empty);

        // Empty does not install a binder, while Full still needs a nonempty
        // domain. Its own interval cannot justify an outer endpoint.
        let zero_bound = Bound {
            symbol: row,
            start: zero,
            end: zero,
        };
        let constants = context.normalize_many(
            vec![
                Region::Bind(zero_bound.clone(), Box::new(Region::Full)),
                Region::Bind(zero_bound, Box::new(Region::Empty)),
                Region::Guard(
                    vec![(Condition::Constant(false), true)],
                    Box::new(Region::Empty),
                ),
                Region::Bind(bound.clone(), Box::new(Region::Full)),
                Region::Bind(
                    Bound {
                        symbol: row,
                        start: zero,
                        end: unknown_extent,
                    },
                    Box::new(Region::Full),
                ),
            ],
            &facts,
        );
        assert_eq!(
            &constants[..4],
            &[Region::Empty, Region::Empty, Region::Empty, Region::Full]
        );
        assert!(
            matches!(&constants[4], Region::Bind(_, body) if matches!(body.as_ref(), Region::Full))
        );
    }

    #[test]
    fn view_projection_preserves_empty_domain_and_private_full_promotion() {
        let mut arena = ExprArena::new();
        let zero = arena.int(0);
        let two = arena.int(2);
        let three = arena.int(3);
        let mut context = InitializationContext::new(&mut arena);
        let empty_view = context.root(&[zero, three]);
        let empty = context.project(&InitializationState::empty(), &empty_view);
        assert!(matches!(empty.0, Region::Empty));
        assert!(context.readable(&empty, &empty_view));

        let root = context.root(&[two, three]);
        let transposed = context.transpose(&root);
        let full = context.project(&InitializationState(Region::Full), &transposed);
        assert!(matches!(full.0, Region::Full));
        assert!(context.readable(&full, &transposed));

        // Both contiguous and noncontiguous views must discharge a reached
        // private condition before contract export forgets it. A contiguous
        // address shift alone would retain the unavailable guard.
        let private = Condition::Version(900, vec![]);
        for view in [&root, &transposed] {
            context.path.push((private.clone(), true));
            let guarded = InitializationState(Region::Guard(
                vec![(private.clone(), true)],
                Box::new(Region::Full),
            ));
            let projected = context.project(&guarded, view);
            let parameter = ParameterPath::root(0);
            let contract = InitializationContract {
                requirements: vec![],
                exits: vec![Exit {
                    path: vec![],
                    written: vec![(parameter.clone(), projected.0)],
                }],
                symbols: vec![],
                accesses: vec![],
            };
            let axes = HashMap::from([(parameter, view.axes.clone())]);
            let closed = context.close_transfer(contract, &[], &axes);
            context.path.clear();
            let outputs = context
                .apply(
                    &closed,
                    &[InitializationArgument::Tensor {
                        state: InitializationState::empty(),
                        view: view.clone(),
                    }],
                )
                .unwrap();
            assert!(context.readable(outputs[0].as_ref().unwrap(), view));
        }
    }

    #[test]
    fn successful_exit_alternatives_retain_only_common_actual_premises() {
        let mut arena = ExprArena::new();
        let shared = arena.bool(true);
        let private = arena.bool(false);
        let actual = |v| Condition::Actual(v, vec![]);
        let paths = vec![
            vec![(actual(shared), true), (actual(private), true)],
            vec![(actual(shared), true), (actual(private), false)],
        ];
        assert_eq!(
            guaranteed_actuals(&successful_paths(&paths), true),
            vec![(shared, true)]
        );
        let with_unknown = Condition::Or(
            Box::new(actual(shared)),
            Box::new(Condition::Version(3, vec![])),
        );
        assert!(guaranteed_actuals(&with_unknown, true).is_empty());
        assert_eq!(
            guaranteed_actuals(&with_unknown, false),
            vec![(shared, false)]
        );
        assert!(matches!(successful_paths(&[]), Condition::Constant(false)));
    }

    #[test]
    fn partial_helper_contract_return_domain_survives_closure_and_application() {
        use crate::expr::ScalarOperand;
        use crate::reference_math::ScalarOp;
        use crate::types::DType;
        let mut arena = ExprArena::new();
        let denominator = arena.checker_variable(SymbolSort::Int);
        let denominator_value = arena.int_symbol(denominator);
        let one = arena.int(1);
        let four = arena.int(4);
        let recipe = arena.scalar_integer(
            ScalarOp::Binary(BinaryOp::Div),
            &[
                ScalarOperand::Word {
                    dtype: DType::I32,
                    value: one,
                },
                ScalarOperand::Word {
                    dtype: DType::I32,
                    value: denominator_value,
                },
            ],
        );
        let defined = arena.scalar_integer_defined(recipe);
        let success = Condition::Actual(defined, vec![denominator]);
        let branch = Condition::Parameter(ParameterPath::root(2));
        let tensor = ParameterPath::root(1);
        let exits = [true, false]
            .into_iter()
            .map(|truth| Exit {
                path: vec![(success.clone(), true), (branch.clone(), truth)],
                written: vec![(tensor.clone(), Region::Full)],
            })
            .collect::<Vec<_>>();
        let contract = InitializationContract {
            requirements: vec![],
            exits,
            symbols: vec![(denominator, ParameterPart::Integer(ParameterPath::root(0)))],
            accesses: vec![],
        };
        let mut context = InitializationContext::new(&mut arena);
        context.facts.assume_type(
            context.arena,
            denominator,
            &crate::types::ValueType::Scalar(DType::I32),
        );
        let view = context.root(&[four]);
        let arguments = vec![
            InitializationArgument::Integer(denominator_value),
            InitializationArgument::Tensor {
                state: InitializationState::empty(),
                view: view.clone(),
            },
            InitializationArgument::Boolean(false),
        ];
        let axes = HashMap::from([(tensor.clone(), vec![four])]);
        let closed = context.close_transfer(contract.clone(), &[denominator], &axes);
        let output = context.apply(&closed, &arguments).unwrap();
        assert!(context.readable(output[1].as_ref().unwrap(), &view));
        // No caller-side recipe evaluation occurred. Return-domain transfer
        // alone must provide the partial operation's scoped continuation.
        let minimum = context.arena.int(i64::from(i32::MIN));
        let facts = context.facts.clone();
        assert!(prove::derive_le(context.arena, &facts, minimum, recipe).is_established());

        let mut conditional = contract.clone();
        conditional.exits[1].path = vec![(branch, false)];
        conditional.exits[1].written = vec![(tensor.clone(), Region::Empty)];
        let conditional = context.close_transfer(conditional, &[denominator], &axes);
        let mut negative = InitializationContext::new(context.arena);
        negative.facts.assume_type(
            negative.arena,
            denominator,
            &crate::types::ValueType::Scalar(DType::I32),
        );
        let output = negative.apply(&conditional, &arguments).unwrap();
        assert!(!negative.readable(output[1].as_ref().unwrap(), &view));

        // A malformed output is rejected after the temporary return-domain
        // calculation; neither path nor its partial-operation premise commits.
        let mut malformed = contract.clone();
        for exit in &mut malformed.exits {
            exit.written = vec![(ParameterPath::root(0), Region::Full)];
        }
        let before = negative.path.clone();
        let before_facts = negative.facts.clone();
        let prior =
            prove::derive_le(negative.arena, &before_facts, minimum, recipe).is_established();
        let failure = negative.apply(&malformed, &arguments).unwrap_err();
        assert_eq!(failure.phase, InitializationPhase::Output);
        assert_eq!(negative.path, before);
        let after_facts = negative.facts.clone();
        assert_eq!(
            prove::derive_le(negative.arena, &after_facts, minimum, recipe).is_established(),
            prior
        );

        let mut reads_first = contract;
        reads_first.requirements.push(Requirement {
            parameter: tensor,
            path: vec![(success, true)],
            region: Region::Full,
            span: Span::default(),
        });
        let reads_first = negative.close_transfer(reads_first, &[denominator], &axes);
        // The hypothetical candidate may return successfully, but must not
        // publish its definedness into the reference's earlier input reads.
        let before = negative.path.clone();
        let failure = negative
            .applicable(&closed, &reads_first, &arguments)
            .unwrap_err();
        assert_eq!(failure.phase, InitializationPhase::Input);
        assert_eq!(negative.path, before);
        let failure = negative.apply(&reads_first, &arguments).unwrap_err();
        assert_eq!(failure.phase, InitializationPhase::Input);
    }

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
    fn complementary_guards_merge_only_identical_regions() {
        let mut arena = ExprArena::new();
        let a = Condition::Version(0, vec![]);
        let b = Condition::Version(1, vec![]);
        let regions = (0..64)
            .map(|i| Region::Linear(arena.int(i), arena.int(i + 1)))
            .collect::<Vec<_>>();
        let mut parts = Vec::new();
        // Interleave many distinct regions, including duplicate paths and
        // reversed conjunction order. Each region is covered in all four worlds.
        for left in [false, true] {
            for right in [false, true] {
                for region in &regions {
                    let path = if left {
                        vec![(a.clone(), left), (b.clone(), right)]
                    } else {
                        vec![(b.clone(), right), (a.clone(), left)]
                    };
                    parts.push(Region::guarded(path.clone(), region.clone()));
                    parts.push(Region::guarded(path, region.clone()));
                }
            }
        }
        let distinct = Region::Linear(arena.int(100), arena.int(101));
        parts.push(Region::guarded(vec![(a.clone(), true)], distinct.clone()));
        merge_complementary_guards(&mut parts);
        let mut expected = regions;
        expected.push(Region::guarded(vec![(a, true)], distinct));
        assert_eq!(parts, expected);
        let once = parts.clone();
        merge_complementary_guards(&mut parts);
        assert_eq!(parts, once);
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

    #[test]
    fn bind_finalization_keeps_member_clips_and_nested_image_domains() {
        let mut arena = ExprArena::new();
        let (_, row, r) = arena.loop_binder();
        let (_, column, c) = arena.loop_binder();
        let zero = arena.int(0);
        let one = arena.int(1);
        let two = arena.int(2);
        let four = arena.int(4);
        let six = arena.int(6);
        let seven = arena.int(7);
        let eight = arena.int(8);
        let image = Region::Image {
            domain: vec![Bound {
                symbol: column,
                start: zero,
                end: two,
            }],
            coordinates: vec![r, c],
        };
        let region = Region::Bind(
            Bound {
                symbol: row,
                start: zero,
                end: eight,
            },
            Box::new(Region::Union(vec![
                Region::guarded(
                    vec![(Condition::Compare(BinaryOp::Lt, r, two), true)],
                    image.clone(),
                ),
                Region::guarded(
                    vec![(Condition::Compare(BinaryOp::Ge, r, six), true)],
                    image,
                ),
                Region::Empty,
            ])),
        );
        let facts = prove::Facts::new();
        let mut context = InitializationContext::new(&mut arena);
        let normalized = context.normalize(region, &facts);
        assert_eq!(context.normalize(normalized.clone(), &facts), normalized);
        for (row, expected) in [(one, true), (seven, true), (four, false)] {
            let point = Region::Image {
                domain: vec![],
                coordinates: vec![row, one],
            };
            assert_eq!(
                context.covered(&normalized, &point, &vec![], &facts, &[eight, two]),
                expected
            );
        }
    }

    #[test]
    fn bind_finalization_retains_independent_nonempty_guard() {
        let mut arena = ExprArena::new();
        let (_, row, _) = arena.loop_binder();
        let (_, _, count) = arena.loop_binder();
        let zero = arena.int(0);
        let one = arena.int(1);
        let extent = arena.int(8);
        let condition = Condition::Compare(BinaryOp::Gt, count, zero);
        let region = Region::Bind(
            Bound {
                symbol: row,
                start: zero,
                end: count,
            },
            Box::new(Region::guarded(
                vec![(condition.clone(), true)],
                Region::Full,
            )),
        );
        let facts = prove::Facts::new();
        let mut context = InitializationContext::new(&mut arena);
        let normalized = context.normalize(region, &facts);
        assert_eq!(
            normalized,
            Region::guarded(vec![(condition.clone(), true)], Region::Full)
        );
        assert_eq!(context.normalize(normalized.clone(), &facts), normalized);
        let point = Region::Linear(zero, one);
        assert!(!context.covered(&normalized, &point, &vec![], &facts, &[extent]));
        assert!(context.covered(
            &normalized,
            &point,
            &vec![(condition, true)],
            &facts,
            &[extent]
        ));
        let empty = context.normalize(
            Region::Bind(
                Bound {
                    symbol: row,
                    start: zero,
                    end: zero,
                },
                Box::new(Region::Full),
            ),
            &facts,
        );
        assert_eq!(empty, Region::Empty);
        let singleton = context.normalize(
            Region::Bind(
                Bound {
                    symbol: row,
                    start: zero,
                    end: one,
                },
                Box::new(Region::Full),
            ),
            &facts,
        );
        assert_eq!(singleton, Region::Full);
    }

    #[test]
    fn a_natural_region_binder_is_a_domain_symbol() {
        // A completed loop's binder may be a `Nat` symbol of the caller's
        // arena. Its image `n` over `n in [0, 4)` is already normal and stays
        // the identical region.
        let mut arena = ExprArena::new();
        let (_, n, natural) = arena.nat_loop_binder();
        let n_value = arena.int_from_nat(natural);
        let (zero, four) = (arena.int(0), arena.int(4));
        let facts = prove::Facts::new();
        let mut context = InitializationContext::new(&mut arena);
        let image = Region::Image {
            domain: vec![Bound {
                symbol: n,
                start: zero,
                end: four,
            }],
            coordinates: vec![n_value],
        };
        assert_eq!(context.normalize(image.clone(), &facts), image);
    }

    #[test]
    fn reached_geometry_keeps_triangular_domain_and_scoped_widths() {
        let mut arena = ExprArena::new();
        let (_, row, r) = arena.loop_binder();
        let (_, column, c) = arena.loop_binder();
        let zero = arena.int(0);
        let one = arena.int(1);
        let four = arena.int(4);
        let row_end = arena.int_add(r, one);
        let clipped_end = arena.int_min(row_end, four);
        let region = Region::Bind(
            Bound {
                symbol: row,
                start: zero,
                end: four,
            },
            Box::new(Region::Bind(
                Bound {
                    symbol: column,
                    start: zero,
                    end: clipped_end,
                },
                Box::new(Region::Image {
                    domain: vec![],
                    coordinates: vec![r, c],
                }),
            )),
        );
        let facts = prove::Facts::new();
        let mut context = InitializationContext::new(&mut arena);
        let normalized = context.normalize(region, &facts);
        let Region::Image {
            domain,
            coordinates,
        } = &normalized
        else {
            panic!("triangular image retains its domains: {normalized:?}");
        };
        assert!(context.same_image(
            domain,
            coordinates,
            &[
                Bound {
                    symbol: row,
                    start: zero,
                    end: four
                },
                Bound {
                    symbol: column,
                    start: zero,
                    end: row_end
                }
            ],
            &[r, c],
        ));
        assert_eq!(context.normalize(normalized.clone(), &facts), normalized);
        let forbidden = Region::Image {
            domain: vec![],
            coordinates: vec![zero, one],
        };
        assert!(!context.covered(&normalized, &forbidden, &vec![], &facts, &[four, four]));
    }

    #[test]
    fn reached_geometry_preserves_nested_guarded_union_coverage() {
        let mut arena = ExprArena::new();
        let (_, row, r) = arena.loop_binder();
        let (_, column, c) = arena.loop_binder();
        let (_, _, count) = arena.loop_binder();
        let zero = arena.int(0);
        let two = arena.int(2);
        let three = arena.int(3);
        let four = arena.int(4);
        let positive = Condition::Compare(BinaryOp::Gt, count, zero);
        let point = Region::Image {
            domain: vec![],
            coordinates: vec![r, c],
        };
        let region = Region::Bind(
            Bound {
                symbol: row,
                start: zero,
                end: four,
            },
            Box::new(Region::Bind(
                Bound {
                    symbol: column,
                    start: zero,
                    end: four,
                },
                Box::new(Region::guarded(
                    vec![(positive.clone(), true)],
                    Region::Union(vec![
                        Region::guarded(
                            vec![(Condition::Compare(BinaryOp::Lt, c, r), true)],
                            point.clone(),
                        ),
                        Region::Union(vec![
                            Region::guarded(
                                vec![(Condition::Compare(BinaryOp::Ge, r, three), true)],
                                point,
                            ),
                            Region::Empty,
                        ]),
                    ]),
                )),
            )),
        );
        let facts = prove::Facts::new();
        let mut context = InitializationContext::new(&mut arena);
        let normalized = context.normalize(region, &facts);
        assert_eq!(context.normalize(normalized.clone(), &facts), normalized);
        let included = Region::Image {
            domain: vec![],
            coordinates: vec![three, three],
        };
        let forbidden = Region::Image {
            domain: vec![],
            coordinates: vec![two, three],
        };
        assert!(!context.covered(&normalized, &included, &vec![], &facts, &[four, four]));
        assert!(context.covered(
            &normalized,
            &included,
            &vec![(positive.clone(), true)],
            &facts,
            &[four, four]
        ));
        assert!(!context.covered(
            &normalized,
            &forbidden,
            &vec![(positive, true)],
            &facts,
            &[four, four]
        ));
    }
}
