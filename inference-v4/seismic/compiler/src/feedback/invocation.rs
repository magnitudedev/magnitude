//! Bounded witness construction over the entry's authoritative predicates.
//! Cells are navigation bookkeeping; only domain evaluation admits a point.
mod intervals;

use super::{
    evolution::Random, options::InvocationParameter, CaseArgument, InvocationScope, ObservationCase,
};
use crate::candidate_domain::CandidateDomain;
use seismic_lang::entry::ParameterKind;
use seismic_lang::expr::compiled::{CompiledPredicate, InvocationValues};
use seismic_lang::expr::{
    self, AnyExpr, BigInt, BigUint, BoolExpr, CmpOp, ExprArena, NaryOp, NodeView,
    PartialAssignment, SymbolId, SymbolSort, SymbolValue,
};
use seismic_lang::types::DType;
use seismic_target::TargetFamily;
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Debug)]
pub(super) struct Axis {
    pub symbol: SymbolId,
    pub sort: SymbolSort,
    low: u64,
    high: u64,
    bands: Vec<(u64, u64)>,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Point(pub Vec<u64>);
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Cell {
    bounds: Vec<(u64, u64)>,
}

pub(super) struct Navigator {
    pub axes: Vec<Axis>,
    aliases: Vec<(SymbolId, SymbolId)>,
    predicate: CompiledPredicate,
    intervals: intervals::Program,
    resource_limit: usize,
    pub resource_limited: bool,
    pub unresolved: u64,
    /// This navigator searches single-word ranks, not the whole mathematical
    /// Nat/Int domain. Exhausting those ranks cannot prove the entry empty.
    rank_limited: bool,
    cursor: WitnessCursor,
}

/// Persistent remaining search space, including unresolved subdivisions.
/// Resuming a query never restarts rejected sampling.
pub(super) struct WitnessCursor {
    coarse: VecDeque<Vec<usize>>,
    queued: HashSet<Vec<usize>>,
    refinement: VecDeque<Cell>,
    focused: VecDeque<Cell>,
    focused_seen: HashSet<Cell>,
    turn: u64,
    visited: HashSet<Point>,
    enumerate: Option<Vec<u64>>,
    enumerate_done: bool,
    alternate: bool,
}

/// Outcomes concern the remaining, unvisited search space. ProvenEmpty is
/// returned only after exhaustive traversal or sound pruning, never a timeout.
pub(super) enum WitnessResult<'a> {
    Found(Point),
    ProvenEmpty,
    Unresolved(&'a WitnessCursor),
}

/// Ranks are navigation bookkeeping over the finite single-word slice. An
/// exact mathematical value outside that slice has no rank; it is never
/// truncated into a different witness.
fn encode(value: SymbolValue) -> Option<(SymbolSort, u64)> {
    Some(match value {
        SymbolValue::Nat(v) => (SymbolSort::Nat, u64::try_from(v).ok()?),
        SymbolValue::Int(v) => (SymbolSort::Int, (i64::try_from(v).ok()? as u64) ^ (1 << 63)),
        SymbolValue::I32(v) => (
            SymbolSort::Scalar(DType::I32),
            ((v as u32) ^ (1 << 31)) as u64,
        ),
        SymbolValue::U32(v) => (SymbolSort::Scalar(DType::U32), v as u64),
        SymbolValue::Bool(v) => (SymbolSort::Scalar(DType::Bool), v as u64),
        SymbolValue::F32(v) => (
            SymbolSort::Scalar(DType::F32),
            float_order(v.to_bits() as u64, 32),
        ),
        SymbolValue::F16(v) => (SymbolSort::Scalar(DType::F16), float_order(v as u64, 16)),
        SymbolValue::BF16(v) => (SymbolSort::Scalar(DType::BF16), float_order(v as u64, 16)),
    })
}
fn float_order(bits: u64, width: u32) -> u64 {
    let sign = 1u64 << (width - 1);
    if bits & sign != 0 {
        (!bits) & ((1u64 << width) - 1)
    } else {
        bits ^ sign
    }
}
fn float_bits(rank: u64, width: u32) -> u64 {
    let sign = 1u64 << (width - 1);
    if rank & sign != 0 {
        rank ^ sign
    } else {
        (!rank) & ((1u64 << width) - 1)
    }
}
fn decode(sort: SymbolSort, rank: u64) -> SymbolValue {
    match sort {
        SymbolSort::Nat => SymbolValue::Nat(BigUint::from(rank)),
        SymbolSort::Int => SymbolValue::Int(BigInt::from((rank ^ (1 << 63)) as i64)),
        SymbolSort::Scalar(DType::I32) => SymbolValue::I32(((rank as u32) ^ (1 << 31)) as i32),
        SymbolSort::Scalar(DType::U32) => SymbolValue::U32(rank as u32),
        SymbolSort::Scalar(DType::Bool) => SymbolValue::Bool(rank != 0),
        SymbolSort::Scalar(DType::F32) => {
            SymbolValue::F32(f32::from_bits(float_bits(rank, 32) as u32))
        }
        SymbolSort::Scalar(DType::F16) => SymbolValue::F16(float_bits(rank, 16) as u16),
        SymbolSort::Scalar(DType::BF16) => SymbolValue::BF16(float_bits(rank, 16) as u16),
    }
}
fn range(sort: SymbolSort) -> (u64, u64) {
    match sort {
        SymbolSort::Nat | SymbolSort::Int => (0, u64::MAX),
        SymbolSort::Scalar(DType::I32 | DType::U32) => (0, u32::MAX as u64),
        SymbolSort::Scalar(DType::Bool) => (0, 1),
        SymbolSort::Scalar(DType::F32) => (0, u32::MAX as u64),
        SymbolSort::Scalar(DType::F16 | DType::BF16) => (0, u16::MAX as u64),
    }
}
impl Axis {
    fn new(symbol: SymbolId, sort: SymbolSort) -> Self {
        let (low, high) = range(sort);
        Self {
            symbol,
            sort,
            low,
            high,
            bands: Vec::new(),
        }
    }
    fn positive_integer(&self, rank: u64) -> Option<u64> {
        match decode(self.sort, rank) {
            SymbolValue::Nat(v) => u64::try_from(v).ok().filter(|v| *v > 0),
            SymbolValue::Int(v) => u64::try_from(v).ok().filter(|v| *v > 0),
            SymbolValue::U32(v) if v > 0 => Some(v as u64),
            SymbolValue::I32(v) if v > 0 => Some(v as u64),
            _ => None,
        }
    }
    fn normalized_span(&self, low: u64, high: u64) -> f64 {
        match (self.positive_integer(low), self.positive_integer(high)) {
            (Some(a), Some(b)) => {
                ((b as f64).ln() - (a as f64).ln()) / (64.0 * std::f64::consts::LN_2)
            }
            _ => (high - low) as f64 / (self.high - self.low).max(1) as f64,
        }
    }
    fn midpoint(&self, low: u64, high: u64) -> u64 {
        if let (Some(a), Some(b)) = (self.positive_integer(low), self.positive_integer(high)) {
            let value = ((a as f64).sqrt() * (b as f64).sqrt()).round() as u64;
            return low
                .saturating_add(value.saturating_sub(a))
                .clamp(low, high - 1);
        }
        low + (high - low) / 2
    }

    fn make_bands(&mut self) {
        if self.low == self.high {
            self.bands.push((self.low, self.high));
            return;
        }
        // Rank bands cover sign/exponent strata for floats and magnitude strata
        // for integers. Signed values get independent positive/negative bands.
        let mut add = |a: u64, b: u64| {
            let low = a.max(self.low);
            let high = b.min(self.high);
            if low <= high {
                self.bands.push((low, high));
            }
        };
        match self.sort {
            SymbolSort::Scalar(DType::F32 | DType::F16 | DType::BF16) => {
                let (width, mantissa, exponents) = match self.sort {
                    SymbolSort::Scalar(DType::F32) => (32, 23, 255),
                    SymbolSort::Scalar(DType::F16) => (16, 10, 31),
                    _ => (16, 7, 255),
                };
                add(float_order(0, width), float_order(0, width));
                add(
                    float_order(1 << (width - 1), width),
                    float_order(1 << (width - 1), width),
                );
                // Infinities and NaN payloads are distinct legal strata. The
                // authoritative domain still decides whether any witness is legal.
                let infinity = (exponents as u64) << mantissa;
                let sign = 1u64 << (width - 1);
                for sign in [0, sign] {
                    let inf = float_order(infinity | sign, width);
                    add(inf, inf);
                    let first = float_order((infinity + 1) | sign, width);
                    let last = float_order((infinity + ((1 << mantissa) - 1)) | sign, width);
                    add(first.min(last), first.max(last));
                }
                for exponent in 0..exponents {
                    let a = (exponent as u64) << mantissa;
                    let b = a + ((1 << mantissa) - 1);
                    add(float_order(a, width), float_order(b, width));
                    add(
                        float_order(b | (1 << (width - 1)), width),
                        float_order(a | (1 << (width - 1)), width),
                    );
                }
            }
            SymbolSort::Int | SymbolSort::Scalar(DType::I32) => {
                let sign = if self.sort == SymbolSort::Int {
                    1u64 << 63
                } else {
                    1u64 << 31
                };
                add(sign, sign);
                for bit in 0..sign.trailing_zeros() {
                    let a = 1u64 << bit;
                    let b = (a.saturating_mul(2) - 1).min(sign - 1);
                    add(sign + a, sign + b);
                    add(sign - b, sign - a);
                }
                add(0, 0);
            }
            _ => {
                add(0, 0);
                for bit in 0..64 {
                    let a = 1u64 << bit;
                    let b = a.saturating_mul(2).saturating_sub(1);
                    add(a, if bit == 63 { u64::MAX } else { b });
                }
            }
        }
        self.bands.sort_by_key(|&(a, b)| (b - a, a));
        self.bands.dedup();
    }
}

impl Navigator {
    pub fn new<T: TargetFamily>(
        domain: &CandidateDomain<'_, T>,
        scope: Option<&InvocationScope>,
        resource_limit: usize,
    ) -> Result<Self, String> {
        let schema = domain.schema();
        let arena = domain.arena();
        let mut axes = schema
            .dimensions()
            .iter()
            .map(|d| Axis::new(d.symbol, SymbolSort::Nat))
            .collect::<Vec<_>>();
        for parameter in schema.parameters() {
            match parameter.kind {
                ParameterKind::Tensor { .. } => {}
                ParameterKind::Scalar { symbol, .. } | ParameterKind::Index { symbol, .. } => {
                    axes.push(Axis::new(symbol, arena.symbol_sort(symbol)))
                }
                ParameterKind::Range { start, end, .. } => {
                    axes.push(Axis::new(start, arena.symbol_sort(start)));
                    axes.push(Axis::new(end, arena.symbol_sort(end)));
                }
            }
        }
        narrow(
            &arena,
            domain.target_domain().predicate().node().into(),
            &mut axes,
        )?;
        let mut rank_scoped = HashSet::new();
        if let Some(scope) = scope {
            if scope.entry != domain.entry() {
                return Err("optimization range belongs to another entry".into());
            }
            for constraint in &scope.constraints {
                let symbol =
                    match constraint.parameter {
                        InvocationParameter::Dimension(i) => {
                            schema.dimensions().get(i as usize).map(|d| d.symbol)
                        }
                        InvocationParameter::Scalar(i) => schema
                            .parameters()
                            .get(i as usize)
                            .and_then(|p| match p.kind {
                                ParameterKind::Scalar { symbol, .. }
                                | ParameterKind::Index { symbol, .. } => Some(symbol),
                                _ => None,
                            }),
                        InvocationParameter::RangeStart(i) => schema
                            .parameters()
                            .get(i as usize)
                            .and_then(|p| match p.kind {
                                ParameterKind::Range { start, .. } => Some(start),
                                _ => None,
                            }),
                        InvocationParameter::RangeEnd(i) => schema
                            .parameters()
                            .get(i as usize)
                            .and_then(|p| match p.kind {
                                ParameterKind::Range { end, .. } => Some(end),
                                _ => None,
                            }),
                    }
                    .ok_or("optimization range does not match the entry schema")?;
                let axis = axes.iter_mut().find(|axis| axis.symbol == symbol).unwrap();
                let (low_sort, low) = encode(constraint.lower.clone())
                    .ok_or("optimization range endpoint exceeds finite witness navigation")?;
                let (high_sort, high) = encode(constraint.upper.clone())
                    .ok_or("optimization range endpoint exceeds finite witness navigation")?;
                if low_sort != axis.sort || high_sort != axis.sort || low > high {
                    return Err("invalid optimization range type or endpoints".into());
                }
                rank_scoped.insert(symbol);
                axis.low = axis.low.max(low);
                axis.high = axis.high.min(high);
                if axis.low > axis.high {
                    return Err("optimization range has a proven empty intersection".into());
                }
            }
        }
        let aliases = collapse_equalities(
            &arena,
            domain.target_domain().predicate().node().into(),
            &mut axes,
        )?;
        let intervals = intervals::Program::new(
            &arena,
            domain.target_domain().predicate().node().into(),
            &axes,
            &aliases,
            domain.constants().bindings(),
        );
        let mut region = Cell {
            bounds: axes.iter().map(|a| (a.low, a.high)).collect(),
        };
        if !intervals.contract(&axes, &mut region) {
            return Err("optimization range has a proven empty intersection".into());
        }
        for (axis, (low, high)) in axes.iter_mut().zip(region.bounds) {
            axis.low = low;
            axis.high = high;
        }
        let finite = axes
            .iter()
            .try_fold(1u64, |n, a| {
                n.checked_mul(a.high.checked_sub(a.low)?.checked_add(1)?)
            })
            .filter(|n| *n <= 4096);
        for axis in &mut axes {
            axis.make_bands();
        }
        let initial = vec![0; axes.len()];
        let mut fixed = PartialAssignment::new();
        for (symbol, value) in domain.constants().bindings() {
            fixed.bind(*symbol, value.clone());
        }
        let predicate = arena.compile_bool_with(domain.target_domain().predicate().node(), &fixed);
        if axes.iter().all(|axis| axis.low == axis.high) {
            let mut values = InvocationValues::new();
            for axis in &axes {
                values.bind(axis.symbol, decode(axis.sort, axis.low));
            }
            for (alias, root) in &aliases {
                values.bind(*alias, values.get(*root).expect("alias root is bound"));
            }
            // A singleton is exhaustively decided by the authoritative
            // predicate, including unsupported interval expressions.
            if !predicate.evaluate(&values).unwrap_or(false) {
                return Err("optimization range has a proven empty intersection".into());
            }
        }

        let rank_limited = axes.iter().any(|axis| {
            matches!(axis.sort, SymbolSort::Nat | SymbolSort::Int)
                && !rank_scoped.contains(&axis.symbol)
        });
        Ok(Self {
            predicate,
            intervals,
            aliases,
            resource_limit,
            resource_limited: false,
            rank_limited,
            cursor: WitnessCursor {
                enumerate: finite.map(|_| axes.iter().map(|a| a.low).collect()),
                coarse: VecDeque::from([initial.clone()]),
                queued: HashSet::from([initial]),
                refinement: VecDeque::new(),
                focused: VecDeque::new(),
                focused_seen: HashSet::new(),
                turn: 0,
                visited: HashSet::new(),
                enumerate_done: false,
                alternate: false,
            },
            axes,
            unresolved: 0,
        })
    }
    pub fn values(&self, point: &Point) -> InvocationValues {
        let mut values = InvocationValues::new();
        for (axis, rank) in self.axes.iter().zip(&point.0) {
            values.bind(axis.symbol, decode(axis.sort, *rank));
        }
        for (alias, root) in &self.aliases {
            values.bind(*alias, values.get(*root).expect("alias root is bound"));
        }
        values
    }
    /// Disagreement gives a region extra opportunities, without removing any
    /// broad-coverage cell or treating endpoint agreement as a proof.
    pub fn prioritize_between(&mut self, left: &Point, right: &Point) {
        if self.cursor.enumerate.is_some() {
            return;
        }
        let cell = Cell {
            bounds: left
                .0
                .iter()
                .zip(&right.0)
                .map(|(a, b)| ((*a).min(*b), (*a).max(*b)))
                .collect(),
        };
        if cell.bounds.iter().all(|(lo, hi)| lo == hi) {
            return;
        }
        if self.cursor.focused_seen.len() < self.resource_limit / 4
            && self.cursor.focused_seen.insert(cell.clone())
        {
            self.cursor.focused.push_back(cell);
        }
    }

    /// Probe a candidate's existing applicability authority. A change gives
    /// both sides focused work; it never establishes monotonicity or discards
    /// the remainder of the domain. Unknown or illegal probes give no fact.
    pub fn prioritize_boundary(
        &mut self,
        origin: &Point,
        eligibility: impl Fn(&InvocationValues) -> Option<bool>,
    ) {
        if self.cursor.enumerate.is_some() {
            return;
        }
        let Some(origin_eligible) = eligibility(&self.values(origin)) else {
            return;
        };
        let mut allowance = 32usize;
        for axis in 0..self.axes.len() {
            for endpoint in [self.axes[axis].low, self.axes[axis].high] {
                if allowance == 0 {
                    return;
                }
                allowance -= 1;
                let mut other = origin.clone();
                other.0[axis] = endpoint;
                let values = self.values(&other);
                if !self.predicate.evaluate(&values).unwrap_or(false) {
                    continue;
                }
                let Some(other_eligible) = eligibility(&values) else {
                    continue;
                };
                if other_eligible == origin_eligible {
                    continue;
                }
                self.prioritize_between(origin, &other);
                let (mut left, mut right) = if origin.0[axis] < endpoint {
                    (origin.clone(), other)
                } else {
                    (other, origin.clone())
                };
                let left_eligible = if origin.0[axis] < endpoint {
                    origin_eligible
                } else {
                    other_eligible
                };
                for _ in 0..8 {
                    if allowance == 0 || right.0[axis] - left.0[axis] <= 1 {
                        break;
                    }
                    allowance -= 1;
                    let mut middle = left.clone();
                    middle.0[axis] = self.axes[axis].midpoint(left.0[axis], right.0[axis]);
                    let values = self.values(&middle);
                    if !self.predicate.evaluate(&values).unwrap_or(false) {
                        break;
                    }
                    let Some(mid_eligible) = eligibility(&values) else {
                        break;
                    };
                    // Preserve both subdivisions even when endpoint states agree.
                    self.prioritize_between(&left, &middle);
                    self.prioritize_between(&middle, &right);
                    if mid_eligible == left_eligible {
                        left = middle;
                    } else {
                        right = middle;
                    }
                }
                for point in [left, right] {
                    let cell = Cell {
                        bounds: point.0.iter().map(|v| (*v, *v)).collect(),
                    };
                    if !self.cursor.visited.contains(&point)
                        && self.cursor.focused_seen.len() < self.resource_limit / 4
                        && self.cursor.focused_seen.insert(cell.clone())
                    {
                        self.cursor.focused.push_front(cell);
                    }
                }
            }
        }
    }

    pub fn next(&mut self, random: &mut Random) -> Option<Point> {
        match self.query(random, 64) {
            WitnessResult::Found(point) => Some(point),
            WitnessResult::ProvenEmpty | WitnessResult::Unresolved(_) => None,
        }
    }

    pub fn query(&mut self, random: &mut Random, allowance: usize) -> WitnessResult<'_> {
        match self.search(random, allowance) {
            Some(point) => WitnessResult::Found(point),
            None if self.exhausted() && !self.rank_limited => WitnessResult::ProvenEmpty,
            None => WitnessResult::Unresolved(&self.cursor),
        }
    }

    fn search(&mut self, random: &mut Random, allowance: usize) -> Option<Point> {
        for _ in 0..allowance {
            if self
                .cursor
                .visited
                .len()
                .saturating_add(self.cursor.queued.len())
                .saturating_add(self.cursor.refinement.len())
                .saturating_add(self.cursor.focused.len())
                .saturating_add(self.cursor.focused_seen.len())
                .saturating_add(self.axes.len())
                .saturating_add(2)
                > self.resource_limit
            {
                self.resource_limited = true;
                return None;
            }
            let point = if let Some(cursor) = self.cursor.enumerate.as_mut() {
                if self.cursor.enumerate_done {
                    return None;
                }
                let point = Point(cursor.clone());
                let mut advanced = false;
                for i in (0..cursor.len()).rev() {
                    if cursor[i] < self.axes[i].high {
                        cursor[i] += 1;
                        for j in i + 1..cursor.len() {
                            cursor[j] = self.axes[j].low;
                        }
                        advanced = true;
                        break;
                    }
                }
                self.cursor.enumerate_done = !advanced;
                point
            } else {
                self.cursor.alternate = !self.cursor.alternate;
                self.cursor.turn += 1;
                let mut cell = if self.cursor.turn % 3 == 0 && !self.cursor.focused.is_empty() {
                    self.cursor.focused.pop_front()?
                } else if self.cursor.alternate && !self.cursor.refinement.is_empty() {
                    self.cursor.refinement.pop_front()?
                } else if let Some(indices) = self.cursor.coarse.pop_front() {
                    for i in 0..indices.len() {
                        let mut next = indices.clone();
                        next[i] += 1;
                        if next[i] < self.axes[i].bands.len()
                            && self.cursor.queued.insert(next.clone())
                        {
                            self.cursor.coarse.push_back(next);
                        }
                    }
                    Cell {
                        bounds: indices
                            .iter()
                            .zip(&self.axes)
                            .map(|(i, a)| a.bands[*i])
                            .collect(),
                    }
                } else {
                    self.cursor
                        .refinement
                        .pop_front()
                        .or_else(|| self.cursor.focused.pop_front())?
                };
                if !self.intervals.contract(&self.axes, &mut cell) {
                    continue;
                }
                // Bind one coordinate at a time and propagate before drawing
                // the next. Coupled equalities therefore construct witnesses
                // instead of depending on independent draws coinciding.
                let mut proposal = cell.clone();
                let mut possible = true;
                for axis in 0..self.axes.len() {
                    let (low, high) = proposal.bounds[axis];
                    let rank = match (high - low).checked_add(1) {
                        Some(width) => low + random.next() % width,
                        None => random.next(),
                    };
                    proposal.bounds[axis] = (rank, rank);
                    if !self.intervals.contract(&self.axes, &mut proposal) {
                        possible = false;
                        break;
                    }
                }
                let mut split = None;
                let mut largest = 0.0;
                for (index, &(low, high)) in cell.bounds.iter().enumerate() {
                    if low == high {
                        continue;
                    }
                    let span = self.axes[index].normalized_span(low, high);
                    if split.is_none() || span > largest {
                        largest = span;
                        split = Some((index, low, high));
                    }
                }
                if let Some((axis, low, high)) = split {
                    let mid = self.axes[axis].midpoint(low, high);
                    let mut left = cell.clone();
                    let mut right = cell;
                    left.bounds[axis] = (low, mid);
                    right.bounds[axis] = (mid + 1, high);
                    self.cursor.refinement.push_back(left);
                    self.cursor.refinement.push_back(right);
                }
                if !possible {
                    continue;
                }
                Point(proposal.bounds.into_iter().map(|(rank, _)| rank).collect())
            };
            if !self.cursor.visited.insert(point.clone()) {
                continue;
            }
            let values = self.values(&point);
            if self.predicate.evaluate(&values).unwrap_or(false) {
                return Some(point);
            }
        }
        if !self.exhausted() {
            self.unresolved += 1;
        }
        None
    }
    pub fn exhausted(&self) -> bool {
        self.cursor.enumerate_done
            || (self.cursor.coarse.is_empty()
                && self.cursor.refinement.is_empty()
                && self.cursor.focused.is_empty())
    }
    pub fn case<T: TargetFamily>(
        &self,
        domain: &CandidateDomain<'_, T>,
        point: &Point,
        seed: u64,
    ) -> Result<ObservationCase, String> {
        let values = self.values(point);
        let mut fixed = PartialAssignment::new();
        for (symbol, value) in domain.constants().bindings() {
            fixed.bind(*symbol, value.clone());
        }
        let nat = |e| {
            domain
                .arena()
                .compile_nat_with(e, &fixed)
                .evaluate_u64(&values)
                .map_err(|e| format!("case geometry: {e:?}"))
        };
        let index = |s| -> Result<BigUint, String> {
            match values.get(s) {
                Some(SymbolValue::Nat(v)) => Ok(v),
                Some(SymbolValue::Int(v)) => {
                    BigUint::try_from(v).map_err(|_| "case index is negative".into())
                }
                _ => Err("invalid case index".into()),
            }
        };
        let arguments = domain
            .schema()
            .parameters()
            .iter()
            .map(|p| {
                Ok(match &p.kind {
                    ParameterKind::Tensor {
                        representation,
                        axes,
                        ..
                    } => CaseArgument::Tensor {
                        representation: *representation,
                        extents: axes.iter().map(|a| nat(*a)).collect::<Result<_, _>>()?,
                    },
                    ParameterKind::Scalar { symbol, .. } => {
                        CaseArgument::Scalar(values.get(*symbol).ok_or("unbound case scalar")?)
                    }
                    ParameterKind::Index { symbol, .. } => CaseArgument::Index(index(*symbol)?),
                    ParameterKind::Range { start, end, .. } => CaseArgument::Range {
                        start: index(*start)?,
                        end: index(*end)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(ObservationCase {
            values,
            arguments,
            seed,
        })
    }
    pub fn predicate(&self, arena: &mut ExprArena, point: &Point) -> Option<BoolExpr> {
        let terms = self
            .axes
            .iter()
            .zip(&point.0)
            .map(|(axis, rank)| compare(arena, axis.symbol, CmpOp::Eq, decode(axis.sort, *rank)))
            .collect::<Option<Vec<_>>>()?;
        Some(arena.all(&terms))
    }
}

/// Equal integral metadata parameters are one search axis, not independent
/// draws followed by nearly-certain rejection. The complete domain predicate
/// still authorizes the resulting point.
fn collapse_equalities(
    arena: &ExprArena,
    predicate: AnyExpr,
    axes: &mut Vec<Axis>,
) -> Result<Vec<(SymbolId, SymbolId)>, String> {
    fn collect(arena: &ExprArena, root: AnyExpr, output: &mut Vec<(SymbolId, SymbolId)>) {
        let mut pending = vec![root];
        let mut seen = HashSet::new();
        while let Some(node) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            match arena.view(node) {
                NodeView::Nary {
                    op: NaryOp::All,
                    operands,
                } => pending.extend(operands),
                NodeView::Binary {
                    op: expr::BinaryOp::And,
                    lhs,
                    rhs,
                } => pending.extend([lhs, rhs]),
                NodeView::Cmp {
                    op: CmpOp::Eq,
                    lhs,
                    rhs,
                } => {
                    if let (NodeView::Symbol(a), NodeView::Symbol(b)) =
                        (arena.view(lhs), arena.view(rhs))
                    {
                        output.push((a, b));
                    }
                }
                _ => {}
            }
        }
    }
    let mut relations = Vec::new();
    collect(arena, predicate, &mut relations);
    let mut roots = (0..axes.len()).collect::<Vec<_>>();
    for (a, b) in relations {
        let (Some(a), Some(b)) = (
            axes.iter().position(|axis| axis.symbol == a),
            axes.iter().position(|axis| axis.symbol == b),
        ) else {
            continue;
        };
        if axes[a].sort != axes[b].sort
            || matches!(
                axes[a].sort,
                SymbolSort::Scalar(DType::F32 | DType::F16 | DType::BF16)
            )
        {
            continue;
        }
        let low = roots[a].min(roots[b]);
        let high = roots[a].max(roots[b]);
        for root in &mut roots {
            if *root == high {
                *root = low;
            }
        }
    }
    let mut aliases = Vec::new();
    for i in 0..axes.len() {
        let root = roots[i];
        if i != root {
            axes[root].low = axes[root].low.max(axes[i].low);
            axes[root].high = axes[root].high.min(axes[i].high);
            if axes[root].low > axes[root].high {
                return Err("equal invocation parameters have disjoint ranges".into());
            }
            aliases.push((axes[i].symbol, axes[root].symbol));
        }
    }
    let mut index = 0;
    axes.retain(|_| {
        let keep = roots[index] == index;
        index += 1;
        keep
    });
    Ok(aliases)
}

fn narrow(arena: &ExprArena, node: AnyExpr, axes: &mut [Axis]) -> Result<(), String> {
    let mut pending = vec![node];
    let mut seen = HashSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        match arena.view(node) {
            NodeView::BoolConst(false) => return Err("entry domain is empty".into()),
            NodeView::Nary {
                op: NaryOp::All,
                operands,
            } => {
                pending.extend(operands);
            }
            NodeView::Binary {
                op: expr::BinaryOp::And,
                lhs,
                rhs,
            } => {
                pending.extend([lhs, rhs]);
            }
            NodeView::Cmp { op, lhs, rhs } => {
                let pair = match (arena.view(lhs), arena.view(rhs)) {
                    (NodeView::Symbol(symbol), NodeView::NatConst(value)) => {
                        Some((symbol, op, SymbolValue::Nat(value.into())))
                    }
                    (NodeView::Symbol(symbol), NodeView::IntConst(value)) => {
                        Some((symbol, op, SymbolValue::Int(value.into())))
                    }
                    (NodeView::NatConst(value), NodeView::Symbol(symbol)) => {
                        Some((symbol, reverse(op), SymbolValue::Nat(value.into())))
                    }
                    (NodeView::IntConst(value), NodeView::Symbol(symbol)) => {
                        Some((symbol, reverse(op), SymbolValue::Int(value.into())))
                    }
                    _ => None,
                };
                if let Some((symbol, op, value)) = pair {
                    if let Some(axis) = axes.iter_mut().find(|a| a.symbol == symbol) {
                        let Some((sort, value)) = encode(value) else {
                            // A bound outside this finite navigation slice
                            // cannot be used to prune it by lossy conversion.
                            continue;
                        };
                        if axis.sort == sort {
                            match op {
                                CmpOp::Eq => {
                                    axis.low = axis.low.max(value);
                                    axis.high = axis.high.min(value);
                                }
                                CmpOp::Le => axis.high = axis.high.min(value),
                                CmpOp::Ge => axis.low = axis.low.max(value),
                                CmpOp::Lt => {
                                    axis.high = axis
                                        .high
                                        .min(value.checked_sub(1).ok_or("empty upper bound")?)
                                }
                                CmpOp::Gt => {
                                    axis.low = axis
                                        .low
                                        .max(value.checked_add(1).ok_or("empty lower bound")?)
                                }
                                CmpOp::Ne => {}
                            }
                            if axis.low > axis.high {
                                return Err("entry bounds are empty".into());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
fn reverse(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
        other => other,
    }
}
pub(super) fn compare(
    arena: &mut ExprArena,
    symbol: SymbolId,
    op: CmpOp,
    value: SymbolValue,
) -> Option<BoolExpr> {
    macro_rules! scalar {
        ($sort:ty,$value:expr) => {{
            let lhs = arena.scalar_symbol::<$sort>(symbol);
            let rhs = arena.scalar_const::<$sort>($value);
            arena.scalar_cmp(op, lhs, rhs)
        }};
    }
    Some(match value {
        SymbolValue::Nat(v) => {
            let a = arena.nat_symbol(symbol);
            let b = arena.nat_exact(v);
            arena.nat_cmp(op, a, b)
        }
        SymbolValue::Int(v) => {
            let a = arena.int_symbol(symbol);
            let b = arena.int_exact(v);
            arena.int_cmp(op, a, b)
        }
        SymbolValue::F32(v) => {
            if !v.is_finite() || v == 0.0 {
                return None;
            }
            scalar!(expr::F32, v)
        }
        SymbolValue::F16(v) => {
            if v & 0x7fff == 0 || v & 0x7c00 == 0x7c00 {
                return None;
            }
            scalar!(expr::F16, v)
        }
        SymbolValue::BF16(v) => {
            if v & 0x7fff == 0 || v & 0x7f80 == 0x7f80 {
                return None;
            }
            scalar!(expr::BF16, v)
        }
        SymbolValue::I32(v) => scalar!(expr::I32, v),
        SymbolValue::U32(v) => scalar!(expr::U32, v),
        SymbolValue::Bool(v) => scalar!(expr::BoolScalar, v),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_navigator(
        arena: &ExprArena,
        predicate: BoolExpr,
        mut axes: Vec<Axis>,
        bounds: Vec<(u64, u64)>,
    ) -> Navigator {
        for (axis, (lo, hi)) in axes.iter_mut().zip(&bounds) {
            axis.low = *lo;
            axis.high = *hi;
            axis.make_bands();
        }
        Navigator {
            intervals: intervals::Program::new(arena, predicate.into(), &axes, &[], &[]),
            predicate: arena.compile_bool_with(predicate, &PartialAssignment::new()),
            axes,
            aliases: Vec::new(),
            resource_limit: 100_000,
            resource_limited: false,
            // This helper treats `bounds` as the complete test domain.
            rank_limited: false,
            unresolved: 0,
            cursor: WitnessCursor {
                coarse: VecDeque::new(),
                queued: HashSet::new(),
                refinement: VecDeque::from([Cell { bounds }]),
                focused: VecDeque::new(),
                focused_seen: HashSet::new(),
                turn: 0,
                visited: HashSet::new(),
                enumerate: None,
                enumerate_done: false,
                alternate: false,
            },
        }
    }
    #[test]
    fn sparse_affine_witness_is_constructed_and_cursor_resumes() {
        let mut arena = ExprArena::new();
        let (_, sx) = arena.target_constant(SymbolSort::Nat);
        let (_, sy) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(sx);
        let y = arena.nat_symbol(sy);
        let sum = arena.nat_add(x, y);
        let million = arena.nat(1_000_000);
        let predicate = arena.nat_cmp(CmpOp::Eq, sum, million);
        let mut nav = test_navigator(
            &arena,
            predicate,
            vec![
                Axis::new(sx, SymbolSort::Nat),
                Axis::new(sy, SymbolSort::Nat),
            ],
            vec![(0, 1_000_000), (0, 1_000_000)],
        );
        let mut random = Random(19);
        let WitnessResult::Unresolved(cursor) = nav.query(&mut random, 0) else {
            panic!("zero allowance is not proof of emptiness")
        };
        assert_eq!(cursor.refinement.len(), 1);
        let WitnessResult::Found(first) = nav.query(&mut random, 1) else {
            panic!("affine propagation should construct a witness")
        };
        assert_eq!(first.0.iter().sum::<u64>(), 1_000_000);
        let WitnessResult::Found(second) = nav.query(&mut random, 8) else {
            panic!("retained subdivisions should continue")
        };
        assert_ne!(first, second);
        assert_eq!(second.0.iter().sum::<u64>(), 1_000_000);
    }
    #[test]
    fn bounded_nonlinear_search_distinguishes_unresolved_from_exhausted() {
        let mut arena = ExprArena::new();
        let (_, sx) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(sx);
        let two = arena.nat(2);
        let one = arena.nat(1);
        let remainder = arena.nat_rem(x, two);
        let predicate = arena.nat_cmp(CmpOp::Eq, remainder, one);
        let mut nav = test_navigator(
            &arena,
            predicate,
            vec![Axis::new(sx, SymbolSort::Nat)],
            vec![(0, 0)],
        );
        let mut random = Random(42);
        assert!(matches!(
            nav.query(&mut random, 0),
            WitnessResult::Unresolved(_)
        ));
        assert!(matches!(
            nav.query(&mut random, 1),
            WitnessResult::ProvenEmpty
        ));
    }
    #[test]
    fn exact_quantities_outside_navigation_rank_are_not_truncated_or_proven_empty() {
        assert!(encode(SymbolValue::Nat(BigUint::from(u64::MAX) + 1u32)).is_none());
        assert!(encode(SymbolValue::Int(BigInt::from(i64::MAX) + 1u32)).is_none());
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let value = arena.nat_symbol(symbol);
        let one = arena.nat(1);
        let predicate = arena.nat_cmp(CmpOp::Eq, value, one);
        let mut nav = test_navigator(
            &arena,
            predicate,
            vec![Axis::new(symbol, SymbolSort::Nat)],
            vec![(0, 0)],
        );
        nav.rank_limited = true;
        let mut random = Random(42);
        assert!(matches!(
            nav.query(&mut random, 1),
            WitnessResult::Unresolved(_)
        ));
    }
    #[test]
    fn guard_boundary_probes_both_sides_without_losing_broad_search() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(symbol);
        let upper = arena.nat(256);
        let domain = arena.nat_cmp(CmpOp::Le, x, upper);
        let mut nav = test_navigator(
            &arena,
            domain,
            vec![Axis::new(symbol, SymbolSort::Nat)],
            vec![(1, 256)],
        );
        nav.prioritize_boundary(&Point(vec![16]), |values| match values.get(symbol) {
            Some(SymbolValue::Nat(v)) => Some(v < BigUint::from(64u32)),
            _ => None,
        });
        assert!(nav
            .cursor
            .focused
            .iter()
            .any(|cell| cell.bounds == vec![(63, 63)]));
        assert!(nav
            .cursor
            .focused
            .iter()
            .any(|cell| cell.bounds == vec![(64, 64)]));
        assert_eq!(
            nav.cursor.refinement.front().unwrap().bounds,
            vec![(1, 256)]
        );
    }
    #[test]
    fn equal_parameters_share_one_axis_and_intersect_partial_scopes() {
        let mut arena = ExprArena::new();
        let (_, a) = arena.target_constant(SymbolSort::Nat);
        let (_, b) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(a);
        let y = arena.nat_symbol(b);
        let predicate = arena.nat_cmp(CmpOp::Eq, x, y);
        let mut axes = vec![Axis::new(a, SymbolSort::Nat), Axis::new(b, SymbolSort::Nat)];
        axes[0].low = 4;
        axes[0].high = 8;
        axes[1].low = 6;
        let aliases = collapse_equalities(&arena, predicate.into(), &mut axes).unwrap();
        assert_eq!(aliases, vec![(b, a)]);
        assert_eq!(axes.len(), 1);
        assert_eq!((axes[0].low, axes[0].high), (6, 8));
        let mut axes = vec![Axis::new(a, SymbolSort::Nat), Axis::new(b, SymbolSort::Nat)];
        axes[0].high = 4;
        axes[1].low = 5;
        assert!(collapse_equalities(&arena, predicate.into(), &mut axes).is_err());
    }
    #[test]
    fn bands_cover_zero_signs_and_integer_boundaries_without_holes() {
        let mut arena = ExprArena::new();
        for sort in [SymbolSort::Nat, SymbolSort::Int] {
            let (_, symbol) = arena.target_constant(sort);
            let mut axis = Axis::new(symbol, sort);
            (axis.low, axis.high) = if sort == SymbolSort::Nat {
                (0, 64)
            } else {
                (
                    encode(SymbolValue::Int((-32).into())).unwrap().1,
                    encode(SymbolValue::Int(32.into())).unwrap().1,
                )
            };
            axis.make_bands();
            for point in axis.low..=axis.high {
                assert!(axis.bands.iter().any(|(a, b)| *a <= point && point <= *b));
            }
        }
    }
    #[test]
    fn focused_integer_split_uses_log_magnitude() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let axis = Axis::new(symbol, SymbolSort::Nat);
        assert_eq!(axis.midpoint(16, 256), 64);
        assert_eq!(axis.midpoint(64, 256), 128);
        assert_eq!(axis.midpoint(1, 2), 1);
        let mid = axis.midpoint(1 << 63, u64::MAX);
        assert!(mid >= 1 << 63 && mid < u64::MAX);
    }
    #[test]
    fn float_bands_cover_every_half_bit_pattern_including_exceptions() {
        let mut arena = ExprArena::new();
        for dtype in [DType::F16, DType::BF16] {
            let sort = SymbolSort::Scalar(dtype);
            let (_, symbol) = arena.target_constant(sort);
            let mut axis = Axis::new(symbol, sort);
            axis.make_bands();
            for rank in 0..=u16::MAX as u64 {
                assert!(
                    axis.bands.iter().any(|(a, b)| *a <= rank && rank <= *b),
                    "missing {dtype:?} rank {rank}"
                );
            }
        }
    }
    #[test]
    fn float_coordinates_preserve_signed_zero_and_nan_payloads() {
        for bits in [0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc00001] {
            let value = SymbolValue::F32(f32::from_bits(bits));
            let (sort, rank) = encode(value).unwrap();
            let SymbolValue::F32(value) = decode(sort, rank) else {
                panic!()
            };
            assert_eq!(value.to_bits(), bits);
        }
        assert_ne!(
            encode(SymbolValue::F32(0.0)).unwrap().1,
            encode(SymbolValue::F32(-0.0)).unwrap().1
        );
    }
}
