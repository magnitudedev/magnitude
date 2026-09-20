//! Statements, blocks with their selected covers, stage chains and call inlining.
use super::context::*;
use super::expr::{Coordinate, Reads};
use super::walk;
use crate::exec::ir::{Builtin, Expr, ExprKind, Stmt, StmtKind};
use crate::exec::types::Ty;
use crate::family::CandidateRef;
use crate::sir;
use crate::syntax::ast::AssignOp;
use crate::sir::Mode;
use crate::types as st;
use std::cell::RefCell;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub(super) struct Item<'a> {
    pub stmt: &'a sir::Stmt,
    pub block: usize,
    pub index: usize,
}

fn items_of(block: &sir::Block) -> Vec<Item<'_>> {
    let key = block_key(block);
    block
        .iter()
        .enumerate()
        .map(|(index, stmt)| Item {
            stmt,
            block: key,
            index,
        })
        .collect()
}

impl<'a> Instantiation<'a> {
    pub fn block(
        &mut self,
        f: &mut Frame<'a>,
        block: &'a sir::Block,
        root: bool,
    ) -> Result<Vec<Stmt>, String> {
        self.items(f, &items_of(block), root)
    }

    fn items(
        &mut self,
        f: &mut Frame<'a>,
        items: &[Item<'a>],
        root: bool,
    ) -> Result<Vec<Stmt>, String> {
        let saved = f.root;
        let mut out = Vec::new();
        let mut i = 0;
        while i < items.len() {
            let item = items[i];
            f.root = root;
            if let Some(fused) = f.fused_at(item.block, item.index).cloned() {
                let members: Vec<Item<'a>> = items[i..]
                    .iter()
                    .copied()
                    .take_while(|m| m.block == item.block && m.index <= fused.last)
                    .collect();
                if members.last().map(|m| m.index) != Some(fused.last) {
                    return Err(format!(
                        "a fused interval of `{}` is split by terminal control flow",
                        f.definition.name
                    ));
                }
                match fused.kind {
                    FusedKind::Elementwise => self.fused_elementwise(f, &members, &mut out)?,
                    FusedKind::Region => self.fused_regions(f, &members, root, &mut out)?,
                }
                i += members.len();
                continue;
            }
            if let sir::StmtKind::If { cond, then, els } = &item.stmt.kind {
                if walk::exits(then) || walk::exits(els) {
                    // A terminal `yield`/`return` ends its boundary: what follows the `if`
                    // runs only on the paths that did not end.
                    let cond = self.scalar(f, cond, &mut out)?;
                    let rest = &items[i + 1..];
                    let mut arms = Vec::with_capacity(2);
                    // Bindings made on one path are not bindings of the other.
                    let bindings = f.vars.clone();
                    for arm in [then, els] {
                        let mut path = items_of(arm);
                        if !walk::terminates(arm) {
                            path.extend_from_slice(rest);
                        }
                        f.conditional += 1;
                        let owner = self.launch_owner.take();
                        let body = self.items(f, &path, false);
                        self.launch_owner = owner;
                        f.conditional -= 1;
                        f.vars = bindings.clone();
                        arms.push(body?);
                    }
                    let els = arms.pop().unwrap_or_default();
                    let then = arms.pop().unwrap_or_default();
                    out.push(stmt(StmtKind::If { cond, then, els }, item.stmt.span));
                    f.root = saved;
                    return Ok(out);
                }
            }
            self.statement(f, item.stmt, root, &mut out)?;
            i += 1;
        }
        f.root = saved;
        Ok(out)
    }

    fn statement(
        &mut self,
        f: &mut Frame<'a>,
        s: &'a sir::Stmt,
        root: bool,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        // Element loops and branches are scalar control of the current owner: a region inside
        // them is never an inner owner region of the launch.
        if matches!(
            s.kind,
            sir::StmtKind::Range { .. }
                | sir::StmtKind::Coordinates { .. }
                | sir::StmtKind::Members { .. }
                | sir::StmtKind::If { .. }
        ) {
            let owner = self.launch_owner.take();
            let result = self.statement_inner(f, s, root, out);
            self.launch_owner = owner;
            return result;
        }
        self.statement_inner(f, s, root, out)
    }

    fn statement_inner(
        &mut self,
        f: &mut Frame<'a>,
        s: &'a sir::Stmt,
        root: bool,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        match &s.kind {
            sir::StmtKind::Bind { pattern, value } => {
                if let sir::Pattern::Var(v) = pattern {
                    if f.folded.contains(v) {
                        f.vars[*v] = Some(Value::Deferred(value));
                        return Ok(());
                    }
                }
                let value = self.value(f, value, out)?;
                self.bind(f, pattern, value, out)
            }
            sir::StmtKind::Assign { target, op, value } => {
                self.assign_to(f, target, *op, value, out)
            }
            sir::StmtKind::Region(region) => self.region(f, region, root, out).map(|_| ()),
            sir::StmtKind::Stages(stages) => self.stages(f, stages, root, out),
            sir::StmtKind::Range {
                kind,
                var,
                lo,
                hi,
                value,
                body,
            } => {
                let (lo, hi) = if let Some(value) = value {
                    let Value::Range(lo, hi) = self.value(f, value, out)? else {
                        return Err(format!(
                            "loop source in `{}` is not a range",
                            f.definition.name
                        ));
                    };
                    (lo, hi)
                } else {
                    (self.scalar(f, lo, out)?, self.scalar(f, hi, out)?)
                };
                let lo = self.symbolic(lo, out)?;
                let hi = self.symbolic(hi, out)?;
                let (id, _, index) = self.index(f.name(*var), s.span);
                f.vars[*var] = Some(Value::Scalar(index));
                f.loops += 1;
                let body = self.block(f, body, false);
                f.loops -= 1;
                out.push(stmt(
                    StmtKind::Range {
                        independent: *kind == sir::LoopKind::Parallel,
                        var: id,
                        lo,
                        hi,
                        body: body?,
                    },
                    s.span,
                ));
                Ok(())
            }
            sir::StmtKind::Coordinates {
                vars,
                of,
                axes,
                body,
            } => {
                let tile = self.reference(f, of, out)?;
                let shape = tile_shape(&tile)?.shape.clone();
                if vars.len() != axes.len() || axes.iter().any(|a| *a >= shape.len()) {
                    return Err(format!(
                        "coordinate loop in `{}` disagrees with the rank of its operand",
                        f.definition.name
                    ));
                }
                let mut ids = Vec::with_capacity(vars.len());
                for var in vars {
                    let (id, _, index) = self.index(f.name(*var), s.span);
                    f.vars[*var] = Some(Value::Scalar(index));
                    ids.push(id);
                }
                let every_axis = axes.iter().copied().eq(0..shape.len());
                let element_domain = every_axis
                    && matches!(tile.ty, Ty::Tile(_))
                    && walk::coordinate_local(body, f.body);
                f.loops += 1;
                let inner = self.block(f, body, false);
                f.loops -= 1;
                let mut inner = inner?;
                if element_domain {
                    out.push(stmt(
                        StmtKind::Owned {
                            vars: ids,
                            tile,
                            body: inner,
                        },
                        s.span,
                    ));
                } else {
                    // Ordered scalar control over the listed axes, first binder outermost.
                    for (id, axis) in ids.into_iter().zip(axes).rev() {
                        inner = vec![stmt(
                            StmtKind::Range {
                                independent: false,
                                var: id,
                                lo: crate::sym::Sym::constant(0),
                                hi: shape[*axis].clone(),
                                body: inner,
                            },
                            s.span,
                        )];
                    }
                    out.extend(inner);
                }
                Ok(())
            }
            sir::StmtKind::Members { var, slice, body } => {
                let bound = f.slice(*slice)?.clone();
                let (id, _, index) = self.index(f.name(*var), s.span);
                f.vars[*var] = Some(Value::Scalar(index));
                f.loops += 1;
                let body = self.block(f, body, false);
                f.loops -= 1;
                out.push(stmt(
                    StmtKind::Range {
                        independent: false,
                        var: id,
                        lo: bound.lo,
                        hi: bound.hi,
                        body: body?,
                    },
                    s.span,
                ));
                Ok(())
            }
            sir::StmtKind::If { cond, then, els } => {
                let cond = self.scalar(f, cond, out)?;
                let then = self.block(f, then, false)?;
                let els = self.block(f, els, false)?;
                out.push(stmt(StmtKind::If { cond, then, els }, s.span));
                Ok(())
            }
            sir::StmtKind::Publish { value, destination } => {
                let destination = self.reference(f, destination, out)?;
                match destination.ty {
                    Ty::Tensor(_) => {
                        let tile = self.tile(f, value, out)?;
                        if let ExprKind::Var(id) = tile.kind {
                            self.temporaries.remove(&id);
                        }
                        let store = Expr {
                            kind: ExprKind::Builtin {
                                name: Builtin::Store,
                                args: vec![tile, destination],
                            },
                            ty: Ty::Void,
                            sym: None,
                            span: s.span,
                        };
                        out.push(stmt(StmtKind::Expr(store), s.span));
                        Ok(())
                    }
                    _ => self.assign_tile(f, destination, AssignOp::Assign, value, None, out),
                }
            }
            sir::StmtKind::Yield(values) => {
                let values = values
                    .iter()
                    .map(|v| self.value(f, v, out))
                    .collect::<Result<Vec<_>, _>>()?;
                let Some(sink) = f.yields.pop() else {
                    return Err(format!(
                        "`yield` in `{}` has no structural consumer",
                        f.definition.name
                    ));
                };
                let (sink, result) = match sink {
                    YieldSink::Slots(mut slots) => {
                        let result = self.fill(f, &mut slots, values, out);
                        (YieldSink::Slots(slots), result)
                    }
                    YieldSink::Result {
                        mut member,
                        pieces,
                        loops,
                    } => {
                        let result = if f.loops != loops {
                            Err(format!(
                                "`yield` inside a loop of a result-producing visit in `{}`",
                                f.definition.name
                            ))
                        } else {
                            let value = if values.len() == 1 {
                                values.into_iter().next().unwrap_or(Value::Void)
                            } else {
                                Value::Tuple(values)
                            };
                            self.store_member(&mut member, &pieces, value, out)
                        };
                        (
                            YieldSink::Result {
                                member,
                                pieces,
                                loops,
                            },
                            result,
                        )
                    }
                };
                f.yields.push(sink);
                result
            }
            sir::StmtKind::Return(values) => {
                let values = values
                    .iter()
                    .map(|v| self.value(f, v, out))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut slots = std::mem::take(&mut f.returns);
                let result = self.fill(f, &mut slots, values, out);
                f.returns = slots;
                result
            }
            sir::StmtKind::Expr(e) => self.value(f, e, out).map(|_| ()),
        }
    }

    pub fn bind(
        &mut self,
        f: &mut Frame<'a>,
        pattern: &sir::Pattern,
        value: Value<'a>,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        match pattern {
            sir::Pattern::Var(v) => {
                let declared = f.declared(*v)?;
                // An alias of a slice is the same slice.
                if let (st::Ty::Slice(id), Value::Slice(slice)) = (&declared.ty, &value) {
                    f.slices.insert(*id, slice.clone());
                }
                let bound = if declared.kind == sir::VarKind::State {
                    self.own(value, &declared.name, out)?
                } else {
                    self.snapshot(value, &declared.name, out)?
                };
                f.vars[*v] = Some(bound);
                Ok(())
            }
            sir::Pattern::Tuple(patterns) => match value {
                Value::Tuple(values) if values.len() == patterns.len() => patterns
                    .iter()
                    .zip(values)
                    .try_for_each(|(p, v)| self.bind(f, p, v, out)),
                _ => Err(format!(
                    "tuple pattern in `{}` does not match its value",
                    f.definition.name
                )),
            },
        }
    }

    // ---- boundaries ----

    /// Deliver the values of a `yield`/`return`. On the unconditional path they are the
    /// boundary's values directly; on conditional paths they are assigned to locals that
    /// the owner declares before the branch.
    fn fill(
        &mut self,
        f: &Frame<'a>,
        slots: &mut Slots<'a>,
        values: Vec<Value<'a>>,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        if f.loops != slots.loops {
            return Err(format!(
                "`yield`/`return` inside a loop in `{}` has no execution form",
                f.definition.name
            ));
        }
        if f.conditional == slots.conditional && slots.locals.is_none() && slots.direct.is_none() {
            slots.direct = Some(
                values
                    .into_iter()
                    .map(|v| self.snapshot(v, "port", out))
                    .collect::<Result<_, _>>()?,
            );
            return Ok(());
        }
        if slots.locals.is_none() {
            let mut prologue = Vec::new();
            let locals = values
                .iter()
                .map(|v| self.declare(v, &mut prologue))
                .collect::<Result<Vec<_>, _>>()?;
            slots.prologue.extend(prologue);
            slots.locals = Some(locals);
        }
        let locals = slots.locals.clone().unwrap_or_default();
        if locals.len() != values.len() {
            return Err(format!(
                "paths of `{}` yield different arities",
                f.definition.name
            ));
        }
        locals
            .iter()
            .zip(values)
            .try_for_each(|(local, value)| self.write_local(local, value, out))
    }

    fn declare(
        &mut self,
        value: &Value<'a>,
        prologue: &mut Vec<Stmt>,
    ) -> Result<Value<'a>, String> {
        match value {
            Value::Scalar(e) => {
                let Ty::Scalar(dtype) = e.ty else {
                    return Err("a non-scalar value is yielded as a scalar".into());
                };
                let target = self.local("port", e.ty.clone(), e.span);
                prologue.push(assign(target.clone(), literal(dtype, 0.0, e.span)));
                if let ExprKind::Var(id) = target.kind {
                    self.mutable.insert(id);
                }
                Ok(Value::Scalar(target))
            }
            Value::Shaped(e) if matches!(e.ty, Ty::Tile(_) | Ty::Tensor(_)) => {
                let shaped = tile_shape(e)?.clone();
                if shaped.shape.iter().any(|d| d.as_constant().is_none())
                    || !matches!(shaped.elem, crate::types::Elem::Dtype(_))
                {
                    return Err("a conditionally yielded tile needs a static dense shape".into());
                }
                let target = self.allocate("port", shaped, e.span, prologue);
                if let ExprKind::Var(id) = target.kind {
                    self.temporaries.remove(&id);
                    self.mutable.insert(id);
                }
                Ok(Value::Shaped(target))
            }
            Value::Tuple(items) => Ok(Value::Tuple(
                items
                    .iter()
                    .map(|v| self.declare(v, prologue))
                    .collect::<Result<_, _>>()?,
            )),
            Value::Void => Ok(Value::Void),
            _ => Err(
                "a conditionally yielded view, slice or region result has no execution form".into(),
            ),
        }
    }

    fn write_local(
        &mut self,
        local: &Value<'a>,
        value: Value<'a>,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        match (local, value) {
            (Value::Scalar(target), Value::Scalar(e)) => {
                let Ty::Scalar(dtype) = target.ty else {
                    return Err("scalar port lost its type".into());
                };
                out.push(assign(target.clone(), conform(e, dtype)));
                Ok(())
            }
            (Value::Shaped(target), Value::Shaped(e)) => {
                let source = if matches!(e.ty, Ty::Tensor(_)) {
                    self.load(e, out)?
                } else {
                    e
                };
                if let ExprKind::Var(id) = source.kind {
                    self.temporaries.remove(&id);
                }
                out.push(assign(target.clone(), source));
                Ok(())
            }
            (Value::Tuple(locals), Value::Tuple(values)) if locals.len() == values.len() => locals
                .iter()
                .zip(values)
                .try_for_each(|(l, v)| self.write_local(l, v, out)),
            (Value::Void, Value::Void) => Ok(()),
            _ => Err("conditional paths yield different value schemas".into()),
        }
    }

    fn stages(
        &mut self,
        f: &mut Frame<'a>,
        stages: &'a [sir::Stage],
        root: bool,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        let mut ports: Vec<Value<'a>> = Vec::new();
        for (k, stage) in stages.iter().enumerate() {
            if stage.ports.len() != ports.len() {
                return Err(format!(
                    "stage `{}` of `{}` binds {} ports but its predecessor yields {}",
                    stage.name,
                    f.definition.name,
                    stage.ports.len(),
                    ports.len()
                ));
            }
            for (var, value) in stage.ports.iter().zip(std::mem::take(&mut ports)) {
                f.vars[*var] = Some(value);
            }
            if k + 1 == stages.len() {
                // The terminal stage yields to the enclosing boundary.
                out.extend(self.block(f, &stage.body, root)?);
                continue;
            }
            f.yields.push(YieldSink::Slots(Slots {
                conditional: f.conditional,
                loops: f.loops,
                ..Slots::default()
            }));
            let body = self.block(f, &stage.body, root);
            let Some(YieldSink::Slots(slots)) = f.yields.pop() else {
                return Err(format!(
                    "stage `{}` of `{}` lost its port boundary",
                    stage.name, f.definition.name
                ));
            };
            let body = body?;
            out.extend(slots.prologue);
            out.extend(body);
            ports = slots.locals.or(slots.direct).unwrap_or_default();
        }
        Ok(())
    }

    // ---- state updates ----

    fn assign_to(
        &mut self,
        f: &mut Frame<'a>,
        target: &'a sir::Expr,
        op: AssignOp,
        value: &'a sir::Expr,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        match &target.kind {
            sir::ExprKind::Tuple(targets) => {
                if op != AssignOp::Assign {
                    return Err(format!(
                        "compound tuple assignment in `{}`",
                        f.definition.name
                    ));
                }
                // Every right-hand side reads the old versions first.
                let values = match self.value(f, value, out)? {
                    Value::Tuple(values) if values.len() == targets.len() => values,
                    _ => {
                        return Err(format!(
                            "tuple assignment in `{}` does not match its value",
                            f.definition.name
                        ))
                    }
                };
                let mut held = Vec::with_capacity(values.len());
                for v in values {
                    held.push(self.own(v, "old", out)?);
                }
                targets
                    .iter()
                    .zip(held)
                    .try_for_each(|(t, v)| self.assign_value(f, t, v, out))
            }
            sir::ExprKind::Var(_) | sir::ExprKind::Index { .. } => {
                let place = match self.value(f, target, out)? {
                    Value::Scalar(place) | Value::Shaped(place) => place,
                    _ => {
                        return Err(format!(
                            "assignment target in `{}` has no storage",
                            f.definition.name
                        ))
                    }
                };
                match place.ty {
                    Ty::Scalar(dtype) => {
                        let value = self.scalar(f, value, out)?;
                        out.push(assign_op(place, op, conform(value, dtype)));
                        Ok(())
                    }
                    _ => {
                        let var = match &target.kind {
                            sir::ExprKind::Var(v) => Some(*v),
                            _ => None,
                        };
                        self.assign_tile(f, place, op, value, var, out)
                    }
                }
            }
            _ => Err(format!(
                "assignment target in `{}` has no storage",
                f.definition.name
            )),
        }
    }

    fn assign_value(
        &mut self,
        f: &mut Frame<'a>,
        target: &'a sir::Expr,
        value: Value<'a>,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        if let (sir::ExprKind::Tuple(targets), Value::Tuple(values)) = (&target.kind, &value) {
            if targets.len() == values.len() {
                return targets
                    .iter()
                    .zip(values.clone())
                    .try_for_each(|(t, v)| self.assign_value(f, t, v, out));
            }
        }
        let place = match self.value(f, target, out)? {
            Value::Scalar(place) | Value::Shaped(place) => place,
            _ => {
                return Err(format!(
                    "assignment target in `{}` has no storage",
                    f.definition.name
                ))
            }
        };
        match (place.ty.clone(), value) {
            (Ty::Scalar(dtype), Value::Scalar(e)) => out.push(assign(place, conform(e, dtype))),
            (Ty::Tile(_), Value::Shaped(e)) => self.copy_into(place, AssignOp::Assign, e, out)?,
            _ => {
                return Err(format!(
                    "assignment in `{}` changes the kind of its target",
                    f.definition.name
                ))
            }
        }
        Ok(())
    }

    /// `place op= source` between complete tiles.
    pub(super) fn copy_into(
        &mut self,
        place: Expr,
        op: AssignOp,
        source: Expr,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        let source = if matches!(source.ty, Ty::Tensor(_)) {
            self.load(source, out)?
        } else {
            source
        };
        if let ExprKind::Var(id) = source.kind {
            self.temporaries.remove(&id);
        }
        if op == AssignOp::Assign
            && matches!(place.kind, ExprKind::Var(_))
            && matches!(place.ty, Ty::Tile(_))
        {
            out.push(assign(place, source));
            return Ok(());
        }
        let span = source.span;
        let (vars, at) = self.coordinates(tile_shape(&source)?.shape.len(), span);
        let body = vec![assign_op(
            points(place, &at)?,
            op,
            points(source.clone(), &at)?,
        )];
        out.push(stmt(
            StmtKind::Owned {
                vars,
                tile: source,
                body,
            },
            span,
        ));
        Ok(())
    }

    /// Tile state update or publication into a tile view, computed elementwise in place
    /// where the value is an elementwise expression.
    pub fn assign_tile(
        &mut self,
        f: &mut Frame<'a>,
        place: Expr,
        op: AssignOp,
        value: &'a sir::Expr,
        target: Option<sir::VarId>,
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        let whole = matches!(
            value.kind,
            sir::ExprKind::Var(_)
                | sir::ExprKind::Call { .. }
                | sir::ExprKind::Region(_)
                | sir::ExprKind::Member { .. }
                | sir::ExprKind::Field { .. }
                | sir::ExprKind::Index { .. }
                | sir::ExprKind::TileAlloc
                | sir::ExprKind::Reduce { .. }
                | sir::ExprKind::Load(_)
        ) || !matches!(
            value.ty,
            st::Ty::Tile(_) | st::Ty::View(_) | st::Ty::Tensor(_)
        );
        // An in-place element loop may read its target only at the current coordinate.
        fn remaps(f: &Frame<'_>, e: &sir::Expr, target: sir::VarId, under: bool) -> bool {
            match &e.kind {
                sir::ExprKind::Var(v) if *v == target => under,
                sir::ExprKind::Var(v) => {
                    matches!(f.vars.get(*v), Some(Some(Value::Deferred(d))) if remaps(f, d, target, under))
                }
                sir::ExprKind::Load(_)
                | sir::ExprKind::Decode(_)
                | sir::ExprKind::Reduce { .. } => false,
                sir::ExprKind::Cast { expr, .. }
                    if matches!(expr.ty, st::Ty::View(_) | st::Ty::Tensor(_)) =>
                {
                    false
                }
                sir::ExprKind::Transpose(_) | sir::ExprKind::Index { .. } => walk::children(e)
                    .into_iter()
                    .any(|c| remaps(f, c, target, true)),
                _ => walk::children(e)
                    .into_iter()
                    .any(|c| remaps(f, c, target, under)),
            }
        }
        let hazard = target.is_some_and(|v| remaps(f, value, v, false));
        if whole || hazard {
            if !matches!(
                value.ty,
                st::Ty::Tile(_) | st::Ty::View(_) | st::Ty::Tensor(_)
            ) {
                // Scalar broadcast into every element.
                let scalar = self.scalar(f, value, out)?;
                let (vars, at) = self.coordinates(tile_shape(&place)?.shape.len(), value.span);
                let body = vec![assign_op(points(place.clone(), &at)?, op, scalar)];
                out.push(stmt(
                    StmtKind::Owned {
                        vars,
                        tile: place,
                        body,
                    },
                    value.span,
                ));
                return Ok(());
            }
            let source = self.tile(f, value, out)?;
            return self.copy_into(place, op, source, out);
        }
        let (vars, at) = self.coordinates(tile_shape(&place)?.shape.len(), value.span);
        let group = BTreeSet::new();
        let reads = RefCell::new(Reads::default());
        let element = self.element(
            f,
            value,
            &Coordinate {
                at: &at,
                group: &group,
                reads: &reads,
                reversed: false,
            },
            out,
        )?;
        let mut body = reads.into_inner().body;
        body.push(assign_op(points(place.clone(), &at)?, op, element));
        out.push(stmt(
            StmtKind::Owned {
                vars,
                tile: place,
                body,
            },
            value.span,
        ));
        Ok(())
    }

    // ---- fused elementwise interval ----

    /// One element loop computing every unit of the interval in authored order per
    /// coordinate. A binding read again inside the interval is computed once per coordinate.
    fn fused_elementwise(
        &mut self,
        f: &mut Frame<'a>,
        members: &[Item<'a>],
        out: &mut Vec<Stmt>,
    ) -> Result<(), String> {
        let definition: &'a sir::Definition = f.definition;
        let unsupported = || {
            format!("a fused elementwise interval of `{}` contains a statement that is not a tile binding or tile state update", definition.name)
        };
        let mut entries = Vec::new();
        let mut group = BTreeSet::new();
        for member in members {
            match &member.stmt.kind {
                sir::StmtKind::Bind {
                    pattern: sir::Pattern::Var(v),
                    value,
                } if f.folded.contains(v) => f.vars[*v] = Some(Value::Deferred(value)),
                sir::StmtKind::Bind {
                    pattern: sir::Pattern::Var(v),
                    value,
                } if matches!(value.ty, st::Ty::Tile(_)) => {
                    group.insert(*v);
                    entries.push(member.stmt);
                }
                sir::StmtKind::Assign { target, value, .. }
                    if matches!(target.kind, sir::ExprKind::Var(_))
                        && matches!(target.ty, st::Ty::Tile(_))
                        && matches!(value.ty, st::Ty::Tile(_)) =>
                {
                    entries.push(member.stmt)
                }
                _ => return Err(unsupported()),
            }
        }
        let mut shape = None;
        let mut tiles: Vec<Option<Expr>> = Vec::with_capacity(entries.len());
        for entry in &entries {
            let sir::StmtKind::Bind {
                pattern: sir::Pattern::Var(v),
                value,
            } = &entry.kind
            else {
                tiles.push(None);
                continue;
            };
            let st::Ty::Tile(declared) = &value.ty else {
                return Err(unsupported());
            };
            let shaped = self.shaped(f, declared)?;
            if shape.get_or_insert_with(|| shaped.shape.clone()) != &shaped.shape {
                return Err(format!(
                    "a fused elementwise interval of `{}` spans different element domains",
                    f.definition.name
                ));
            }
            let inside: usize = entries
                .iter()
                .flat_map(|s| walk::direct_exprs(s))
                .map(|e| {
                    let mut n = 0;
                    walk::each_expr(e, &mut |x| {
                        n += usize::from(matches!(x.kind, sir::ExprKind::Var(u) if u == *v))
                    });
                    n
                })
                .sum();
            let state = f.declared(*v)?.kind == sir::VarKind::State;
            let escapes = state || f.uses.get(*v).copied().unwrap_or(0) > inside;
            tiles.push(if escapes {
                Some(self.allocate(f.name(*v), shaped, value.span, out))
            } else {
                None
            });
        }
        let mut domain = tiles.iter().flatten().next().cloned();
        for entry in &entries {
            if let sir::StmtKind::Assign { target, .. } = &entry.kind {
                let place = self.reference(f, target, out)?;
                if shape.get_or_insert_with(|| {
                    place
                        .ty
                        .shaped()
                        .map(|s| s.shape.clone())
                        .unwrap_or_default()
                }) != &tile_shape(&place)?.shape
                {
                    return Err(format!(
                        "a fused elementwise interval of `{}` spans different element domains",
                        f.definition.name
                    ));
                }
                domain.get_or_insert(place);
            }
        }
        let domain = match domain {
            Some(domain) => domain,
            None => {
                let Some(sir::StmtKind::Bind { value, .. }) = entries.first().map(|s| &s.kind)
                else {
                    return Ok(());
                };
                let st::Ty::Tile(declared) = &value.ty else {
                    return Err(unsupported());
                };
                let shaped = self.shaped(f, declared)?;
                let tile = self.allocate("fused", shaped, value.span, out);
                tiles[0] = Some(tile.clone());
                tile
            }
        };
        let (vars, at) = self.coordinates(tile_shape(&domain)?.shape.len(), domain.span);
        let mut pre = Vec::new();
        let mut body = Vec::new();
        let reads = RefCell::new(Reads::default());
        for (entry, tile) in entries.iter().zip(&tiles) {
            let c = Coordinate {
                at: &at,
                group: &group,
                reads: &reads,
                reversed: false,
            };
            match &entry.kind {
                sir::StmtKind::Bind {
                    pattern: sir::Pattern::Var(v),
                    value,
                } => {
                    let element = self.element(f, value, &c, &mut pre)?;
                    body.append(&mut reads.borrow_mut().body);
                    let local = self.local(f.name(*v), element.ty.clone(), value.span);
                    body.push(assign(local.clone(), element));
                    if let Some(tile) = tile {
                        body.push(assign(points(tile.clone(), &at)?, local.clone()));
                    }
                    if f.declared(*v)?.kind != sir::VarKind::State {
                        f.overrides.insert(*v, local);
                    }
                }
                sir::StmtKind::Assign { target, op, value } => {
                    let element = self.element(f, value, &c, &mut pre)?;
                    body.append(&mut reads.borrow_mut().body);
                    let place = self.reference(f, target, &mut pre)?;
                    if let Some(id) = root(&place) {
                        reads.borrow_mut().written(id);
                    }
                    body.push(assign_op(points(place, &at)?, *op, element));
                }
                _ => return Err(unsupported()),
            }
            // A state binding of the interval is readable by the units after it.
            if let (
                sir::StmtKind::Bind {
                    pattern: sir::Pattern::Var(v),
                    ..
                },
                Some(tile),
            ) = (&entry.kind, tile)
            {
                if let ExprKind::Var(id) = tile.kind {
                    self.temporaries.remove(&id);
                    if f.declared(*v)?.kind == sir::VarKind::State {
                        self.mutable.insert(id);
                    }
                }
                f.vars[*v] = Some(Value::Shaped(tile.clone()));
            }
        }
        for v in &group {
            f.overrides.remove(v);
        }
        out.extend(pre);
        out.push(stmt(
            StmtKind::Owned {
                vars,
                tile: domain.clone(),
                body,
            },
            domain.span,
        ));
        Ok(())
    }

    // ---- calls ----

    /// Inline the witness-selected candidate of a call occurrence.
    pub fn call(
        &mut self,
        f: &mut Frame<'a>,
        call: sir::CallId,
        args: &'a [sir::Expr],
        out: &mut Vec<Stmt>,
    ) -> Result<Value<'a>, String> {
        if self.depth >= 64 {
            return Err(format!(
                "call nesting under `{}` exceeds the instantiation limit",
                f.definition.name
            ));
        }
        let parent = self.candidate(f.candidate)?;
        let occurrence = parent
            .children
            .iter()
            .filter_map(|id| self.family.occurrences.get(id.0 as usize))
            .find(|o| o.call == Some(call) && o.parent == Some(f.candidate))
            .ok_or_else(|| {
                format!(
                    "call {} of `{}` has no occurrence under the selected candidate",
                    call.0, f.definition.name
                )
            })?;
        let choice = *self.witness.choices.get(&occurrence.id).ok_or_else(|| {
            format!(
                "witness has no choice for active occurrence {} in `{}`",
                occurrence.id.0, f.definition.name
            )
        })?;
        let selected = CandidateRef {
            occurrence: occurrence.id,
            candidate: choice,
        };
        let mut callee = self.frame(selected)?;
        let via = self.candidate(selected)?.via;
        let site = f.body.calls.get(call.0 as usize).ok_or_else(|| {
            format!(
                "call {} is outside the body of `{}`",
                call.0, f.definition.name
            )
        })?;
        let binding = site
            .bindings
            .iter()
            .find(|b| b.definition == callee.definition.id)
            .or_else(|| site.bindings.iter().find(|b| b.definition == via))
            .ok_or_else(|| {
                format!(
                    "call to `{}` in `{}` has no argument binding for the selected definition",
                    callee.definition.name, f.definition.name
                )
            })?;
        let values = args
            .iter()
            .map(|a| self.value(f, a, out))
            .collect::<Result<Vec<_>, _>>()?;
        for (ordinal, param) in callee.definition.params.iter().enumerate() {
            let value = binding
                .arg_order
                .get(ordinal)
                .and_then(|a| values.get(*a))
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "parameter `{}` of `{}` has no argument in `{}`",
                        param.name, callee.definition.name, f.definition.name
                    )
                })?;
            let bound = match (param.mode, value) {
                (Mode::In, value @ Value::Scalar(_)) => self.snapshot(value, &param.name, out)?,
                (Mode::In, Value::Shaped(e)) => {
                    let e = if matches!(param.ty, st::Ty::Tile(_)) && matches!(e.ty, Ty::Tensor(_))
                    {
                        self.load(e, out)?
                    } else {
                        e
                    };
                    if let ExprKind::Var(id) = e.kind {
                        self.temporaries.remove(&id);
                    }
                    Value::Shaped(e)
                }
                (Mode::Out | Mode::Inout, Value::Shaped(e)) if root(&e).is_some() => {
                    Value::Shaped(e)
                }
                (Mode::Out | Mode::Inout, _) => {
                    return Err(format!(
                        "`{}` argument `{}` of `{}` is not a place",
                        if param.mode == Mode::Out {
                            "out"
                        } else {
                            "inout"
                        },
                        param.name,
                        callee.definition.name
                    ));
                }
                (_, other) => other,
            };
            // A slice passed to a helper keeps its geometry under the helper's identity.
            if let (st::Ty::Slice(id), Value::Slice(slice)) = (&param.ty, &bound) {
                callee.slices.insert(*id, slice.clone());
            }
            let slot = callee.vars.get_mut(param.var).ok_or_else(|| {
                format!(
                    "parameter `{}` of `{}` has no variable",
                    param.name, callee.definition.name
                )
            })?;
            *slot = Some(bound);
        }
        // A shape parameter bound to a runtime extent of the caller takes the extent of the
        // argument axis that determines it.
        let template = self.template(self.candidate(selected)?)?;
        for name in &template.dynamic {
            let extent = callee.definition.params.iter().find_map(|param| {
                let axis = param
                    .ty
                    .shaped()?
                    .axes
                    .iter()
                    .position(|a| a.semantic() == Some(&crate::sym::Sym::param(name)))?;
                match callee.vars.get(param.var) {
                    Some(Some(Value::Shaped(e))) => e.ty.shaped()?.shape.get(axis).cloned(),
                    _ => None,
                }
            });
            let extent = extent.ok_or_else(|| format!("runtime shape parameter `{name}` of `{}` is not the extent of any argument axis in `{}`", callee.definition.name, f.definition.name))?;
            callee.dynamic.insert(name.clone(), extent);
        }
        let first_local = self.vars.len();
        let body: &'a sir::Body = callee.body;
        self.depth += 1;
        let stmts = self.block(&mut callee, &body.block, f.root);
        self.depth -= 1;
        let stmts = stmts?;
        let returns = std::mem::take(&mut callee.returns);
        out.extend(returns.prologue);
        out.extend(stmts);
        let mut values = returns.locals.or(returns.direct).unwrap_or_default();
        // Tiles created by the callee have no other owner once it has returned.
        fn release(value: &Value<'_>, first: usize, temporaries: &mut BTreeSet<usize>) {
            match value {
                Value::Shaped(Expr {
                    kind: ExprKind::Var(id),
                    ty: Ty::Tile(_),
                    ..
                }) if *id >= first => {
                    temporaries.insert(*id);
                }
                Value::Tuple(items) => items.iter().for_each(|v| release(v, first, temporaries)),
                _ => {}
            }
        }
        values
            .iter()
            .for_each(|v| release(v, first_local, &mut self.temporaries));
        Ok(match (&callee.definition.result, values.len()) {
            (st::Ty::Void, _) => Value::Void,
            (_, 1) => values.swap_remove(0),
            (_, 0) => {
                return Err(format!(
                    "`{}` returns no value on the instantiated path",
                    callee.definition.name
                ))
            }
            _ => Value::Tuple(values),
        })
    }
}
