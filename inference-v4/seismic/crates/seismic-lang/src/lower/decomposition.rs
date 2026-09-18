//! Logical contraction domains derived from the portable body's value flow.
//! This analysis does not choose an implementation. It establishes when invoking
//! the same operation on contiguous input domains carries exactly its state.
use crate::{
    ast::AssignOp,
    ir::*,
    sym::{Atom, Sym},
    types::Ty,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub(super) struct Domain {
    pub parameter: String,
    pub axes: Vec<Option<usize>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Value {
    Input(usize, Vec<Sym>),
    Fold(Box<Value>),
    Other {
        accesses: Vec<(usize, Vec<Sym>)>,
        sensitive: bool,
        folded: bool,
    },
    Unknown,
}
impl Value {
    fn facts(&self) -> (Vec<(usize, Vec<Sym>)>, bool, bool) {
        match self {
            Self::Input(v, c) => (vec![(*v, c.clone())], false, false),
            Self::Fold(v) => {
                let (a, b, _) = v.facts();
                (a, b, true)
            }
            Self::Other {
                accesses,
                sensitive,
                folded,
            } => (accesses.clone(), *sensitive, *folded),
            Self::Unknown => (Vec::new(), true, true),
        }
    }
    fn subst(&self, map: &[(Atom, Sym)]) -> Self {
        match self {
            Self::Input(v, coords) => Self::Input(
                *v,
                coords
                    .iter()
                    .map(|s| map.iter().fold(s.clone(), |s, (a, b)| s.subst(a, b)))
                    .collect(),
            ),
            Self::Fold(v) => Self::Fold(Box::new(v.subst(map))),
            Self::Other {
                accesses,
                sensitive,
                folded,
            } => Self::Other {
                accesses: accesses
                    .iter()
                    .map(|(v, c)| {
                        (
                            *v,
                            c.iter()
                                .map(|s| map.iter().fold(s.clone(), |s, (a, b)| s.subst(a, b)))
                                .collect(),
                        )
                    })
                    .collect(),
                sensitive: *sensitive,
                folded: *folded,
            },
            other => other.clone(),
        }
    }
}
#[derive(Clone)]
struct Cell {
    coordinates: Vec<Atom>,
    value: Value,
}
struct Analysis<'a> {
    vars: &'a [Var],
    dimension: &'a str,
    axes: &'a [Option<usize>],
    values: HashMap<usize, Cell>,
    bounds: HashMap<Atom, Sym>,
    written: HashSet<usize>,
    reads: HashSet<usize>,
    serial: usize,
}
impl Analysis<'_> {
    fn coordinates(&mut self, rank: usize) -> Vec<Atom> {
        let n = self.serial;
        self.serial += 1;
        (0..rank)
            .map(|axis| Atom::Param(format!("domain#{n}#{axis}")))
            .collect()
    }
    fn read(&self, e: &Expr, coords: &[Sym]) -> Value {
        if matches!(e.ty, Ty::Scalar(_))
            && e.sym
                .as_ref()
                .is_some_and(|s| s.params().contains(&self.dimension.to_string()))
        {
            return Value::Other {
                accesses: Vec::new(),
                sensitive: true,
                folded: false,
            };
        }
        match &e.kind {
            ExprKind::Var(v) => self
                .values
                .get(v)
                .map(|c| {
                    c.value.subst(
                        &c.coordinates
                            .iter()
                            .cloned()
                            .zip(coords.iter().cloned())
                            .collect::<Vec<_>>(),
                    )
                })
                .unwrap_or_else(|| match self.vars[*v].kind {
                    VarKind::Param(_) => Value::Input(*v, coords.to_vec()),
                    VarKind::Index(ref a) => Value::Other {
                        accesses: Vec::new(),
                        sensitive: self
                            .bounds
                            .get(a)
                            .is_some_and(|n| n.params().contains(&self.dimension.to_string())),
                        folded: false,
                    },
                    _ => Value::Unknown,
                }),
            ExprKind::Index { base, indices } => {
                let Some(shape) = base.ty.shaped().map(|s| &s.shape) else {
                    return Value::Unknown;
                };
                let mut next = 0;
                let mut mapped = Vec::new();
                for axis in 0..shape.len() {
                    match indices.get(axis) {
                        Some(Index::Point(e)) => {
                            let Some(s) = &e.sym else {
                                return Value::Unknown;
                            };
                            mapped.push(s.clone());
                        }
                        index => {
                            let Some(c) = coords.get(next) else {
                                return Value::Unknown;
                            };
                            next += 1;
                            let start = match index {
                                Some(Index::Slice { start: Some(e), .. }) => match &e.sym {
                                    Some(s) => s.clone(),
                                    None => return Value::Unknown,
                                },
                                _ => Sym::constant(0),
                            };
                            mapped.push(start.add(c));
                        }
                    }
                }
                self.read(base, &mapped)
            }
            ExprKind::Transpose(base) => {
                let mut c = coords.to_vec();
                c.reverse();
                self.read(base, &c)
            }
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            } => {
                let Some(shape) = e.ty.shaped().map(|s| &s.shape) else {
                    return Value::Unknown;
                };
                let mut linear = Sym::constant(0);
                for (c, n) in coords.iter().zip(shape) {
                    linear = linear.mul(n).add(c);
                }
                let Some(source) = args.first() else {
                    return Value::Unknown;
                };
                let Some(s) = source.ty.shaped() else {
                    return Value::Unknown;
                };
                let mut c = Vec::new();
                for n in s.shape.iter().rev() {
                    c.push(linear.rem(n));
                    linear = linear.quot(n);
                }
                c.reverse();
                self.read(source, &c)
            }
            ExprKind::Cast { dtype, expr } if expr.ty == Ty::Scalar(*dtype) => {
                self.read(expr, coords)
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => Value::Other {
                accesses: Vec::new(),
                sensitive: e
                    .sym
                    .as_ref()
                    .is_some_and(|s| s.params().contains(&self.dimension.to_string())),
                folded: false,
            },
            ExprKind::ShapeParam(_) => Value::Other {
                accesses: Vec::new(),
                sensitive: e
                    .sym
                    .as_ref()
                    .is_some_and(|s| s.params().contains(&self.dimension.to_string())),
                folded: false,
            },
            ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => {
                self.combine(&[expr], coords)
            }
            ExprKind::Binary { lhs, rhs, .. } => self.combine(&[lhs, rhs], coords),
            ExprKind::Builtin { name, args }
                if !matches!(
                    name,
                    Builtin::Store | Builtin::Atomic | Builtin::Reduce | Builtin::Load
                ) =>
            {
                self.combine(&args.iter().collect::<Vec<_>>(), coords)
            }
            _ => Value::Unknown,
        }
    }
    fn combine(&self, args: &[&Expr], coords: &[Sym]) -> Value {
        let mut accesses = Vec::new();
        let (mut sensitive, mut folded) = (false, false);
        for e in args {
            let (a, b, c) = self.read(e, coords).facts();
            accesses.extend(a);
            sensitive |= b;
            folded |= c;
        }
        Value::Other {
            accesses,
            sensitive,
            folded,
        }
    }
    fn write(&mut self, target: &Expr, value: Value) -> Option<()> {
        match &target.kind {
            ExprKind::Var(v) => {
                if matches!(self.vars[*v].kind, VarKind::Param(_)) {
                    self.written.insert(*v);
                }
                let coords = self.coordinates(target.ty.shaped().map_or(0, |s| s.shape.len()));
                self.values.insert(
                    *v,
                    Cell {
                        coordinates: coords,
                        value,
                    },
                );
                Some(())
            }
            ExprKind::Index { base, indices } => {
                let ExprKind::Var(v) = base.kind else {
                    return None;
                };
                let shape = &base.ty.shaped()?.shape;
                if indices.len() != shape.len() {
                    return None;
                }
                let mut atoms = Vec::new();
                for (ix, n) in indices.iter().zip(shape) {
                    let Index::Point(e) = ix else { return None };
                    let sym = e.sym.as_ref()?;
                    let atom = self
                        .bounds
                        .keys()
                        .find(|a| *sym == Sym::atom((*a).clone()))?
                        .clone();
                    if self.bounds.get(&atom) != Some(n) {
                        return None;
                    }
                    atoms.push(atom);
                }
                if matches!(self.vars[v].kind, VarKind::Param(_)) {
                    self.written.insert(v);
                }
                self.values.insert(
                    v,
                    Cell {
                        coordinates: atoms,
                        value,
                    },
                );
                Some(())
            }
            _ => None,
        }
    }
    fn block(&mut self, body: &[Stmt]) -> Option<()> {
        for statement in body {
            match &statement.kind {
                StmtKind::Assign {
                    target,
                    op: AssignOp::Assign,
                    value,
                } => {
                    if matches!(value.kind, ExprKind::TileAlloc { .. }) {
                        let ExprKind::Var(v) = target.kind else {
                            return None;
                        };
                        self.values.remove(&v);
                        continue;
                    }
                    let coords = self.coordinates(target.ty.shaped().map_or(0, |s| s.shape.len()));
                    let coordinates = coords.iter().cloned().map(Sym::atom).collect::<Vec<_>>();
                    let result = self.read(value, &coordinates);
                    if let ExprKind::Var(v) = target.kind {
                        if matches!(self.vars[v].kind, VarKind::Param(_)) {
                            self.written.insert(v);
                        }
                        self.values.insert(
                            v,
                            Cell {
                                coordinates: coords,
                                value: result,
                            },
                        );
                    } else {
                        self.write(target, result)?;
                    }
                }
                StmtKind::Owned { vars, tile, body } => {
                    for (v, n) in vars.iter().zip(&tile.ty.shaped()?.shape) {
                        let VarKind::Index(a) = &self.vars[*v].kind else {
                            return None;
                        };
                        self.bounds.insert(a.clone(), n.clone());
                    }
                    self.block(body)?;
                }
                StmtKind::Reduction(r) => {
                    if r.extent() != &Sym::param(self.dimension) {
                        return None;
                    }
                    let position = Sym::atom(self.coordinates(1).remove(0));
                    for input in &r.inputs {
                        let mut coords = self
                            .coordinates(input.ty.shaped()?.shape.len())
                            .into_iter()
                            .map(Sym::atom)
                            .collect::<Vec<_>>();
                        coords[r.axis] = position.clone();
                        let (reads, sensitive, folded) = self.read(input, &coords).facts();
                        if sensitive || folded {
                            return None;
                        }
                        for (root, coordinates) in reads {
                            if let Some(axis) = self.axes.get(root).copied().flatten() {
                                if coordinates.get(axis) != Some(&position) {
                                    return None;
                                }
                            }
                            self.reads.insert(root);
                        }
                    }
                    for state in &r.state {
                        let ExprKind::Var(v) = state.kind else {
                            return None;
                        };
                        let coords = self.coordinates(state.ty.shaped()?.shape.len());
                        let value = self.read(
                            state,
                            &coords.iter().cloned().map(Sym::atom).collect::<Vec<_>>(),
                        );
                        if value.facts().2 {
                            return None;
                        }
                        if matches!(self.vars[v].kind, VarKind::Param(_)) {
                            self.written.insert(v);
                        }
                        self.values.insert(
                            v,
                            Cell {
                                coordinates: coords,
                                value: Value::Fold(Box::new(value)),
                            },
                        );
                    }
                }
                _ => return None,
            }
        }
        Some(())
    }
}

pub(super) fn domains(f: &Function, vars: &[Var], body: &[Stmt]) -> Vec<Domain> {
    let mut domains = Vec::new();
    for dimension in &f.shape_params {
        let mut axes = Vec::new();
        let mut valid = true;
        for (_, ty) in &f.params {
            let found = ty
                .shaped()
                .map(|s| {
                    s.shape
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| n.params().contains(dimension))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if found.is_empty() {
                axes.push(None);
                continue;
            }
            if found.len() != 1
                || found[0].1 != &Sym::param(dimension)
                || !matches!(ty, Ty::Tile(_))
            {
                valid = false;
                break;
            }
            axes.push(Some(found[0].0));
        }
        if !valid || !axes.iter().any(Option::is_some) {
            continue;
        }
        let mut analysis = Analysis {
            vars,
            dimension,
            axes: &axes,
            values: HashMap::new(),
            bounds: HashMap::new(),
            written: HashSet::new(),
            reads: HashSet::new(),
            serial: 0,
        };
        if analysis.block(body).is_none()
            || analysis.written.is_empty()
            || !analysis.written.is_disjoint(&analysis.reads)
        {
            continue;
        }
        let valid=analysis.written.iter().all(|v|{
            let Some(cell)=analysis.values.get(v)else{return false};
            matches!(&cell.value,Value::Fold(seed) if **seed==Value::Input(*v,cell.coordinates.iter().cloned().map(Sym::atom).collect()))
        });
        if !valid {
            continue;
        }
        if analysis.written.iter().any(|v| axes[*v].is_some()) {
            continue;
        }
        domains.push(Domain {
            parameter: dimension.clone(),
            axes,
        });
    }
    domains
}

/// Retained ordered state permits contiguous input decomposition without a new
/// arithmetic tree. Reassociated trees retain their separately selected shape.
pub(super) fn select(
    function: &mut crate::lowered_ir::LoweredIr,
    options: &super::Options,
    select: &mut dyn FnMut(
        &crate::lowered_ir::Decision,
    ) -> Result<crate::lowered_ir::Alternative, String>,
) -> Result<(), String> {
    select_body(&mut function.body, &mut function.vars, &[], options, select)
}
pub(super) fn select_body(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    context: &[Stmt],
    options: &super::Options,
    select: &mut dyn FnMut(
        &crate::lowered_ir::Decision,
    ) -> Result<crate::lowered_ir::Alternative, String>,
) -> Result<(), String> {
    let mut bounds = HashMap::<Sym, (i64, IterationDomain)>::new();
    for statement in context {
        direct_bounds(statement, &mut bounds);
    }
    block(body, vars, &bounds, options, select)
}
pub(super) fn collect(e: &Expr, bounds: &mut HashMap<Sym, (i64, IterationDomain)>) {
    if let Ty::Tensor(shape) = &e.ty {
        for (axis, extent) in shape.shape.iter().enumerate() {
            if let Ok(capacity) = super::view_axis_capacity(e, axis) {
                bounds
                    .entry(extent.clone())
                    .and_modify(|n| {
                        if capacity < n.0 {
                            *n = (
                                capacity,
                                IterationDomain {
                                    view: e.clone(),
                                    axis,
                                },
                            );
                        }
                    })
                    .or_insert((
                        capacity,
                        IterationDomain {
                            view: e.clone(),
                            axis,
                        },
                    ));
            }
        }
    }
    match &e.kind {
        ExprKind::Index { base, indices } => {
            collect(base, bounds);
            for index in indices {
                match index {
                    Index::Point(e) => collect(e, bounds),
                    Index::Slice { start, end } => {
                        for e in start.iter().chain(end) {
                            collect(e, bounds)
                        }
                    }
                }
            }
        }
        ExprKind::Load { view: e, .. }
        | ExprKind::Transpose(e)
        | ExprKind::Accessor { base: e, .. }
        | ExprKind::Lanes { base: e, .. }
        | ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. } => collect(e, bounds),
        ExprKind::Builtin { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Tuple(args) => {
            for e in args {
                collect(e, bounds)
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect(lhs, bounds);
            collect(rhs, bounds);
        }
        _ => {}
    }
}
// A value definition carries both a bound and an already evaluated shape.
// Following parent axes here proves capacity without replaying their endpoint
// expressions at the later reduction site.
fn value_axis_capacity(
    value: &Expr,
    axis: usize,
    bounds: &HashMap<Sym, (i64, IterationDomain)>,
) -> Option<i64> {
    let extent = value.ty.shaped()?.shape.get(axis)?;
    if let Some(capacity) = extent.as_constant() {
        return (capacity >= 0).then_some(capacity);
    }
    if let Some((capacity, _)) = bounds.get(extent) {
        return Some(*capacity);
    }
    match &value.kind {
        ExprKind::Index { base, indices } => {
            let parent = (0..base.ty.shaped()?.shape.len())
                .filter(|i| !matches!(indices.get(*i), Some(Index::Point(_))))
                .nth(axis)?;
            value_axis_capacity(base, parent, bounds)
        }
        ExprKind::Transpose(base) => {
            value_axis_capacity(base, value.ty.shaped()?.shape.len() - 1 - axis, bounds)
        }
        ExprKind::Load { view, .. } => value_axis_capacity(view, axis, bounds),
        ExprKind::Builtin { name: Builtin::Load, args } => {
            value_axis_capacity(args.first()?, axis, bounds)
        }
        _ => None,
    }
}

pub(super) fn direct_bounds(s: &Stmt, bounds: &mut HashMap<Sym, (i64, IterationDomain)>) {
    match &s.kind {
        StmtKind::Assign { target, value, .. } => {
            collect(target, bounds);
            collect(value, bounds);
            if matches!(target.kind, ExprKind::Var(_))
                && matches!(target.ty, Ty::Tile(_))
            {
                // Logical shape provenance applies to packed snapshots too;
                // geometry-only storage elimination has separate eligibility.
                for (axis, extent) in target.ty.shaped().unwrap().shape.iter().enumerate() {
                    if let Some(capacity) = value_axis_capacity(value, axis, bounds) {
                        let domain = IterationDomain {
                            // Assignment fixes a logical value version. Its
                            // metadata remains valid if raw endpoints change.
                            view: target.clone(),
                            axis,
                        };
                        bounds.entry(extent.clone())
                            .and_modify(|(known, source)| {
                                *known = (*known).min(capacity);
                                // Keep the first captured value for an equal
                                // extent. A later computed producer need not
                                // stay live merely to supply the same geometry.
                                if !matches!(source.view.kind, ExprKind::Var(_))
                                    || !matches!(source.view.ty, Ty::Tile(_))
                                {
                                    *source = domain.clone();
                                }
                            })
                            .or_insert((capacity, domain));
                    }
                }
            }
        }
        StmtKind::Expr(e) => collect(e, bounds),
        StmtKind::Reduction(r) => {
            for e in r.operands() {
                collect(e, bounds);
            }
        }
        StmtKind::Owned { tile, .. } => collect(tile, bounds),
        StmtKind::LoadLoop { domain, views, .. } => {
            collect(&domain.view, bounds);
            for e in views {
                collect(e, bounds);
            }
        }
        StmtKind::If { cond, .. } => collect(cond, bounds),
        _ => {}
    }
}

fn block(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    bounds: &HashMap<Sym, (i64, IterationDomain)>,
    options: &super::Options,
    select: &mut dyn FnMut(
        &crate::lowered_ir::Decision,
    ) -> Result<crate::lowered_ir::Alternative, String>,
) -> Result<(), String> {
    use crate::{
        lowered_ir::{Alternative, Alternatives, Decision, DecisionKind},
        reduction::structured::Tree,
    };
    let mut result = Vec::new();
    let mut visible_bounds = bounds.clone();
    for mut statement in std::mem::take(body) {
        if let StmtKind::Reduction(reduction) = &mut statement.kind {
            // A piece must not take a new snapshot after an earlier piece has
            // mutated the reduction state. Capture once at the source site.
            let captured = reduction.capture_inputs(vars);
            for binding in &captured {
                direct_bounds(binding, &mut visible_bounds);
            }
            result.extend(captured);
        }
        direct_bounds(&statement, &mut visible_bounds);
        let bounds = &visible_bounds;
        match &mut statement.kind {
            StmtKind::Reduction(r) => {
                for m in r.implementations_mut() {
                    block(&mut m.body, vars, bounds, options, select)?;
                }
            }
            StmtKind::Parallel { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. }
            | StmtKind::LoadLoop { body, .. } => block(body, vars, bounds, options, select)?,
            StmtKind::If { then, els, .. } => {
                block(then, vars, bounds, options, select)?;
                block(els, vars, bounds, options, select)?;
            }
            _ => {}
        }
        if let StmtKind::Reduction(r) = &statement.kind {
            if r.tree == Some(Tree::Ordered) {
                let extent = r.extent().clone();
                if let Some(maximum) = extent
                    .as_constant()
                    .or_else(|| bounds.get(&extent).map(|b| b.0))
                    .filter(|n| *n > 0)
                {
                    let span = statement.span;
                    let piece = Atom::Param(format!("reduction_piece#{}", vars.len()));
                    let domain = Decision {
                        kind: DecisionKind::Stream {
                            piece: piece.clone(),
                            extent: extent.clone(),
                            maximum,
                        },
                        alternatives: match options.piece {
                            Some(n) => vec![Alternative::StreamCapacity(n.min(maximum))].into(),
                            None => Alternatives::stream_capacities(maximum)?,
                        },
                    };
                    let Alternative::StreamCapacity(capacity) = select(&domain)? else {
                        return Err("reduction decomposition requires a capacity".into());
                    };
                    if !domain
                        .alternatives
                        .contains(&Alternative::StreamCapacity(capacity))
                    {
                        return Err("reduction capacity outside logical bound".into());
                    }
                    if let Some(n) = extent.as_constant() {
                        if capacity < n {
                            let index = vars.len();
                            let atom = Atom::Param(format!("reduction_partition#{index}"));
                            vars.push(Var {
                                name: format!("reduction_partition_{index}"),
                                ty: Ty::Scalar(crate::types::DType::I32),
                                kind: VarKind::Index(atom.clone()),
                                span,
                            });
                            for (start, count, repeated) in [
                                (Sym::atom(atom).scale(capacity), capacity, true),
                                (Sym::constant(n / capacity * capacity), n % capacity, false),
                            ] {
                                if count == 0 {
                                    continue;
                                }
                                let mut inputs = Vec::new();
                                let mut chunk = Vec::new();
                                for input in &r.inputs {
                                    let (copy, value) = slice_input(
                                        input,
                                        r.axis,
                                        start.clone(),
                                        Sym::constant(count),
                                        vars,
                                    )?;
                                    chunk.push(copy);
                                    inputs.push(value);
                                }
                                let mut reduction = r.clone();
                                reduction.inputs = inputs;
                                chunk.push(Stmt {
                                    id: None,
                                    span,
                                    kind: StmtKind::Reduction(reduction),
                                });
                                if repeated {
                                    result.push(Stmt {
                                        id: None,
                                        span,
                                        kind: StmtKind::Range {
                                            var: index,
                                            lo: Sym::constant(0),
                                            hi: Sym::constant(n / capacity),
                                            body: chunk,
                                        },
                                    })
                                } else {
                                    result.extend(chunk);
                                }
                            }
                            continue;
                        }
                    } else {
                        let mut inputs = Vec::new();
                        let mut bindings = Vec::new();
                        let offset = vars.len();
                        vars.push(Var {
                            name: format!("logical_start_{offset}"),
                            ty: Ty::Scalar(crate::types::DType::I32),
                            kind: VarKind::Index(Atom::Param(format!("logical_start#{offset}"))),
                            span,
                        });
                        for input in &r.inputs {
                            let mut shape = input
                                .ty
                                .shaped()
                                .ok_or("reduction stream operand shape missing")?
                                .clone();
                            shape.shape[r.axis] = Sym::atom(piece.clone());
                            let v = vars.len();
                            let ty = Ty::Tile(shape);
                            vars.push(Var {
                                name: format!("reduction_input_{v}"),
                                ty: ty.clone(),
                                span,
                                kind: VarKind::Local,
                            });
                            bindings.push(v);
                            inputs.push(Expr {
                                kind: ExprKind::Var(v),
                                ty,
                                sym: None,
                                span,
                            });
                        }
                        let mut reduction = r.clone();
                        reduction.inputs = inputs;
                        result.push(Stmt {
                            id: None,
                            span,
                            kind: StmtKind::LoadLoop {
                                domain: bounds
                                    .get(&extent)
                                    .ok_or("dynamic iteration domain has no logical source")?
                                    .1
                                    .clone(),
                                offset: Some(offset),
                                vars: bindings,
                                views: r.inputs.clone(),
                                axes: vec![r.axis; r.inputs.len()],
                                piece,
                                capacity: Some(capacity),
                                modes: None,
                                body: vec![Stmt {
                                    id: None,
                                    span,
                                    kind: StmtKind::Reduction(reduction),
                                }],
                            },
                        });
                        continue;
                    }
                }
            }
        }
        result.push(statement);
    }
    *body = result;
    Ok(())
}
pub(super) fn slice_input(
    input: &Expr,
    axis: usize,
    start: Sym,
    count: Sym,
    vars: &mut Vec<Var>,
) -> Result<(Stmt, Expr), String> {
    let span = input.span;
    let mut shape = input
        .ty
        .shaped()
        .ok_or("decomposition input must be shaped")?
        .clone();
    let rank = shape.shape.len();
    shape.shape[axis] = count.clone();
    let ty = Ty::Tile(shape);
    let expression = |s: Sym| Expr {
        kind: ExprKind::ShapeParam(s.to_string()),
        ty: Ty::Scalar(crate::types::DType::I32),
        sym: Some(s),
        span,
    };
    let indices = (0..rank)
        .map(|a| {
            if a == axis {
                Index::Slice {
                    start: Some(expression(start.clone())),
                    end: Some(expression(start.add(&count))),
                }
            } else {
                Index::Slice {
                    start: None,
                    end: None,
                }
            }
        })
        .collect();
    let value = Expr {
        kind: ExprKind::Index {
            base: Box::new(input.clone()),
            indices,
        },
        ty: ty.clone(),
        sym: None,
        span,
    };
    let id = vars.len();
    vars.push(Var {
        name: format!("partition_input_{id}"),
        ty: ty.clone(),
        span,
        kind: VarKind::Local,
    });
    let target = Expr {
        kind: ExprKind::Var(id),
        ty,
        sym: None,
        span,
    };
    Ok((
        Stmt {
            id: None,
            span,
            kind: StmtKind::Assign {
                target: target.clone(),
                op: AssignOp::Assign,
                value,
            },
        },
        target,
    ))
}
