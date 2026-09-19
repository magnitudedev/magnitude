//! Expressions: scalar computation maps 1:1; tile-valued computation becomes element
//! loops; views stay symbolic references into their storage.
use super::context::*;
use super::walk;
use crate::syntax::ast::BinaryOp;
use crate::exec::ir::{self, Builtin, Expr, ExprKind, LoadMode, Stmt, StmtKind};
use crate::exec::types::{Shaped, Ty};
use crate::span::Span;
use crate::sir;
use crate::types as st;
use crate::sym::Sym;
use crate::types::{DType, Elem};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// The coordinate at which a tile-valued expression is evaluated, and the bindings
/// computed inside the same loop (their tiles are incomplete while it runs).
pub(super) struct Coordinate<'c> {
    pub at: &'c [Expr],
    pub group: &'c BTreeSet<sir::VarId>,
    pub reads: &'c RefCell<Reads>,
    pub reversed: bool,
}

/// Element reads of tile variables at the loop's coordinate: each is one scalar local of
/// the loop body, however often the unit mentions the tile.
#[derive(Default)]
pub(super) struct Reads {
    cached: BTreeMap<(ir::VarId, bool), Expr>,
    /// Statements of the loop body that precede the consumer of the returned element.
    pub body: Vec<Stmt>,
}

impl Reads {
    pub fn written(&mut self, tile: ir::VarId) {
        self.cached.retain(|(id, _), _| *id != tile);
    }
}

fn math(op: sir::Math) -> Builtin {
    match op {
        sir::Math::Fma => Builtin::Fma,
        sir::Math::Exp => Builtin::Exp,
        sir::Math::ExpFast => Builtin::ExpFast,
        sir::Math::Rsqrt => Builtin::Rsqrt,
        sir::Math::Sqrt => Builtin::Sqrt,
        sir::Math::Log => Builtin::Log,
        sir::Math::Sin => Builtin::Sin,
        sir::Math::Cos => Builtin::Cos,
        sir::Math::Abs => Builtin::Abs,
        sir::Math::Max => Builtin::Max,
        sir::Math::Min => Builtin::Min,
    }
}

fn shaped_type(ty: &st::Ty) -> bool {
    matches!(ty, st::Ty::Tensor(_) | st::Ty::View(_))
}

impl<'a> Instantiation<'a> {
    pub fn value(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        match &e.kind {
            sir::ExprKind::Var(v) => {
                let deferred = match f.var(*v)? {
                    Value::Deferred(deferred) => Some(*deferred),
                    _ => None,
                };
                if let Some(deferred) = deferred {
                    let tile = self.materialize(f, deferred, out)?;
                    f.vars[*v] = Some(Value::Shaped(tile));
                }
                Ok(f.var(*v)?.clone())
            }
            sir::ExprKind::Tuple(items) => Ok(Value::Tuple(items.iter().map(|i| self.value(f, i, out)).collect::<Result<_, _>>()?)),
            sir::ExprKind::Field { base, index } => match self.value(f, base, out)? {
                Value::Tuple(mut items) if *index < items.len() => Ok(items.swap_remove(*index)),
                _ => Err(format!("field {index} of a non-tuple value in `{}`", f.definition.name)),
            },
            sir::ExprKind::Call { call, args } => self.call(f, *call, args, out),
            sir::ExprKind::Region(region) => self.region(f, region, f.root, out),
            sir::ExprKind::Member { result, slices } => self.member(f, result, slices, out),
            sir::ExprKind::Index { base, indices } => {
                let indexed = self.indexed(f, base, indices, out)?;
                Ok(if matches!(indexed.ty, Ty::Scalar(_)) { Value::Scalar(indexed) } else { Value::Shaped(indexed) })
            }
            sir::ExprKind::Transpose(base) => {
                let base = self.reference(f, base, out)?;
                let source = tile_shape(&base)?.clone();
                if source.shape.len() != 2 {
                    return Err(format!("transpose of a rank-{} value in `{}`", source.shape.len(), f.definition.name));
                }
                let shaped = Shaped { shape: vec![source.shape[1].clone(), source.shape[0].clone()], elem: source.elem, packed_axis: source.packed_axis.map(|a| 1 - a) };
                let ty = if matches!(base.ty, Ty::Tensor(_)) { Ty::Tensor(shaped) } else { Ty::Tile(shaped) };
                Ok(Value::Shaped(Expr { kind: ExprKind::Transpose(Box::new(base)), ty, sym: None, span: e.span }))
            }
            sir::ExprKind::Reshape { base, axes } => {
                let base = self.reference(f, base, out)?;
                let Ty::Tensor(source) = &base.ty else {
                    return Err(format!("reshape of a {} in `{}`: the execution IR reshapes tensor views only", base.ty, f.definition.name));
                };
                let shape = axes.iter().map(|a| self.extent(f, a)).collect::<Result<Vec<_>, _>>()?;
                let ty = Ty::Tensor(Shaped { packed_axis: source.packed_axis.map(|_| shape.len().saturating_sub(1)), elem: source.elem.clone(), shape: shape.clone() });
                let mut args = vec![base];
                args.extend(shape.into_iter().map(|s| symbol(s, e.span)));
                Ok(Value::Shaped(Expr { kind: ExprKind::Builtin { name: Builtin::Reshape, args }, ty, sym: None, span: e.span }))
            }
            sir::ExprKind::Accessor { base, name } => {
                let base = self.reference(f, base, out)?;
                let ty = accessor_type(&base, name)?;
                Ok(Value::Shaped(Expr { kind: ExprKind::Accessor { base: Box::new(base), name: name.clone() }, ty, sym: None, span: e.span }))
            }
            _ => match &e.ty {
                st::Ty::Scalar(_) | st::Ty::Index(_) | st::Ty::Coord(_) => Ok(Value::Scalar(self.compute(f, e, out)?)),
                st::Ty::Tile(_) => Ok(Value::Shaped(self.materialize(f, e, out)?)),
                st::Ty::Native(native) => self.native(f, e, native, out),
                st::Ty::Void => match &e.kind {
                    sir::ExprKind::Intrinsic { op, args } => {
                        let args = args.iter().map(|a| self.operand(f, a, out)).collect::<Result<Vec<_>, _>>()?;
                        out.push(stmt(StmtKind::Expr(Expr { kind: ExprKind::Intrinsic { op: *op, args }, ty: Ty::Void, sym: None, span: e.span }), e.span));
                        Ok(Value::Void)
                    }
                    sir::ExprKind::Atomic { .. } => Err(format!(
                        "`atomic` in `{}`: ir::Builtin::Atomic carries no operation, so it cannot be instantiated faithfully",
                        f.definition.name
                    )),
                    _ => Err(format!("void expression in `{}` has no execution form", f.definition.name)),
                },
                other => Err(format!("a computed `{other}` in `{}` has no execution form", f.definition.name)),
            },
        }
    }

    pub fn scalar(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        match self.value(f, e, out)? {
            Value::Scalar(x) => Ok(x),
            _ => Err(format!("`{}` in `{}` is used as a scalar", e.ty, f.definition.name)),
        }
    }

    /// A reference to tensor, view, tile or fragment storage.
    pub fn reference(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        match self.value(f, e, out)? {
            Value::Shaped(x) => Ok(x),
            _ => Err(format!("`{}` in `{}` is used as shaped storage", e.ty, f.definition.name)),
        }
    }

    /// A tile variable holding the value: tensor views are snapshotted.
    pub fn tile(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        let reference = self.reference(f, e, out)?;
        match reference.ty {
            Ty::Tensor(_) => self.load(reference, out),
            _ => Ok(reference),
        }
    }

    fn operand(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        match self.value(f, e, out)? {
            Value::Scalar(x) | Value::Shaped(x) => Ok(x),
            _ => Err(format!("`{}` in `{}` is not an intrinsic operand", e.ty, f.definition.name)),
        }
    }

    fn native(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, native: &st::NativeTy, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        let sir::ExprKind::Intrinsic { op, .. } = &e.kind else {
            return Err(format!("native `{}.{}` in `{}` is not produced by an intrinsic", native.target, native.name, f.definition.name));
        };
        if *op != crate::intrinsics::Operation::Matrix {
            return Err(format!("intrinsic `{op}` in `{}` does not declare a native value", f.definition.name));
        }
        let dtype = match native.elem.as_ref().map(|e| self.elem(f, e)).transpose()? {
            Some(Elem::Dtype(d)) => d,
            _ => return Err(format!("native `{}` in `{}` needs a dense element type", native.name, f.definition.name)),
        };
        let ty = crate::intrinsics::frag8x8(dtype);
        let target = self.local(&native.name, ty.clone(), e.span);
        let tag = Expr { kind: ExprKind::Int(0), ty: Ty::Scalar(dtype), sym: None, span: e.span };
        out.push(assign(target.clone(), Expr { kind: ExprKind::Intrinsic { op: *op, args: vec![tag] }, ty, sym: None, span: e.span }));
        Ok(Value::Shaped(target))
    }

    /// Scalar computation, 1:1 onto the execution IR.
    fn compute(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        let dtype = self.dtype(f, &e.ty)?;
        let ty = Ty::Scalar(dtype);
        let span = e.span;
        Ok(match &e.kind {
            sir::ExprKind::Int(n) if dtype.is_int() => Expr { kind: ExprKind::Int(*n), ty, sym: Some(Sym::constant(*n)), span },
            sir::ExprKind::Int(n) => literal(dtype, *n as f64, span),
            sir::ExprKind::Float(x) => literal(dtype, *x, span),
            sir::ExprKind::Bool(b) => literal(DType::Bool, f64::from(u8::from(*b)), span),
            sir::ExprKind::ShapeParam(name) => coerce(symbol(self.resolve(f, &Sym::param(name))?, span), dtype),
            sir::ExprKind::Cast { dtype, expr } => coerce(self.scalar(f, expr, out)?, *dtype),
            sir::ExprKind::Unary { op, expr } => {
                let inner = self.scalar(f, expr, out)?;
                let sym = match op {
                    crate::syntax::ast::UnaryOp::Neg if dtype.is_int() => inner.sym.as_ref().map(Sym::neg),
                    _ => None,
                };
                Expr { kind: ExprKind::Unary { op: *op, expr: Box::new(inner) }, ty, sym, span }
            }
            sir::ExprKind::Binary { op, lhs, rhs } => binary(*op, self.scalar(f, lhs, out)?, self.scalar(f, rhs, out)?, ty),
            sir::ExprKind::Math { op, args } => {
                let args = args.iter().map(|a| self.scalar(f, a, out)).collect::<Result<Vec<_>, _>>()?;
                Expr { kind: ExprKind::Builtin { name: math(*op), args }, ty, sym: None, span }
            }
            sir::ExprKind::Select { cond, then, els } => {
                let args = vec![self.scalar(f, cond, out)?, coerce(self.scalar(f, then, out)?, dtype), coerce(self.scalar(f, els, out)?, dtype)];
                Expr { kind: ExprKind::Builtin { name: Builtin::Select, args }, ty, sym: None, span }
            }
            sir::ExprKind::Reduce { value, axis, op, unordered } => {
                let operand = self.tile(f, value, out)?;
                self.reduce(operand, *axis, *op, *unordered, span, out)?
            }
            sir::ExprKind::CoordOf(v) => {
                let coordinate = match f.var(*v)? {
                    Value::Scalar(x) => x.clone(),
                    _ => return Err(format!("`coord({})` in `{}` is not over a coordinate", f.name(*v), f.definition.name)),
                };
                let origin = match &f.declared(*v)?.ty {
                    st::Ty::Coord(slice) => f.slice(*slice)?.lo.clone(),
                    st::Ty::Index(bound) => self.structural_origin(f, bound),
                    _ => Sym::constant(0),
                };
                if origin.is_zero() { coordinate } else { binary(BinaryOp::Add, symbol(origin, span), coordinate, ty) }
            }
            sir::ExprKind::ExtentOf { base, axis } | sir::ExprKind::Geometry { base, axis, .. } => {
                // Only dividing widths are instantiated, so capacity and valid extent coincide.
                let base = self.reference(f, base, out)?;
                let extent = tile_shape(&base)?.shape.get(*axis).cloned().ok_or_else(|| format!("axis {axis} is outside its operand in `{}`", f.definition.name))?;
                coerce(symbol(extent, span), dtype)
            }
            sir::ExprKind::Intrinsic { op, args } => {
                let args = args.iter().map(|a| self.operand(f, a, out)).collect::<Result<Vec<_>, _>>()?;
                Expr { kind: ExprKind::Intrinsic { op: *op, args }, ty, sym: None, span }
            }
            other => return Err(format!("scalar expression {other:?} in `{}` has no execution form", f.definition.name)),
        })
    }

    /// Semantic start of the axis a shape expression names, when it is structural.
    fn structural_origin(&self, f: &Frame<'a>, extent: &Sym) -> Sym {
        match extent.params().as_slice() {
            [name] if *extent == Sym::param(name) => f.structural.get(name).map_or(Sym::constant(0), |s| s.lo.clone()),
            _ => Sym::constant(0),
        }
    }

    fn axis_origin(&self, f: &Frame<'a>, base: &sir::Expr, axis: usize) -> Result<Sym, String> {
        Ok(match base.ty.shaped().and_then(|s| s.axes.get(axis)) {
            Some(st::Extent::Structural(slice)) => f.slice(*slice)?.lo.clone(),
            Some(st::Extent::Semantic(extent)) => self.structural_origin(f, extent),
            None => Sym::constant(0),
        })
    }

    /// `base[indices]`. Storage over a structural axis is addressed relative to the start
    /// of that axis' slice; semantic member coordinates are rebased, tile coordinates are not.
    pub fn indexed(&mut self, f: &mut Frame<'a>, base: &'a sir::Expr, indices: &'a [sir::Index], out: &mut Vec<Stmt>) -> Result<Expr, String> {
        let storage = self.reference(f, base, out)?;
        let mut lowered = Vec::with_capacity(indices.len());
        let mut extents = vec![None; indices.len()];
        for (axis, index) in indices.iter().enumerate() {
            let origin = self.axis_origin(f, base, axis)?;
            let relative = |sym: Sym| sym.sub(&origin);
            lowered.push(match index {
                sir::Index::Coord(v) => match f.var(*v)? {
                    Value::Scalar(x) => ir::Index::Point(x.clone()),
                    _ => return Err(format!("`{}` in `{}` is not a tile coordinate", f.name(*v), f.definition.name)),
                },
                sir::Index::Point(point) => {
                    let mut semantic = false;
                    walk::each_expr(point, &mut |x| {
                        semantic |= matches!(x.kind, sir::ExprKind::Var(v) if matches!(f.body.vars.get(v).map(|v| &v.kind), Some(sir::VarKind::SliceMember(_))));
                    });
                    let value = self.scalar(f, point, out)?;
                    let value = coerce(value, DType::I32);
                    ir::Index::Point(if semantic && !origin.is_zero() {
                        binary(BinaryOp::Sub, value, symbol(origin.clone(), point.span), Ty::Scalar(DType::I32))
                    } else {
                        value
                    })
                }
                sir::Index::Slice(slice) => {
                    let bound = f.slice(*slice)?;
                    let (start, end) = (relative(bound.lo.clone()), relative(bound.hi.clone()));
                    let whole = start.is_zero() && tile_shape(&storage)?.shape.get(axis) == Some(&end);
                    if whole {
                        ir::Index::Slice { start: None, end: None }
                    } else {
                        ir::Index::Slice { start: Some(symbol(start, base.span)), end: Some(symbol(end, base.span)) }
                    }
                }
                sir::Index::Range { start, end } => {
                    let start = self.bound(f, start, &origin, out)?;
                    let end = self.bound(f, end, &origin, out)?;
                    if start.iter().chain(&end).any(|e| e.sym.is_none()) {
                        // A runtime-bounded window: clamped bounds, extent named by an atom.
                        let parent = tile_shape(&storage)?.shape.get(axis).cloned().ok_or_else(|| format!("index beyond the rank of its operand in `{}`", f.definition.name))?;
                        let key = (start.as_ref().map(crate::exec::normalize::value_identity), end.as_ref().map(crate::exec::normalize::value_identity));
                        let known = self.windows.iter().find(|(s, e, p, _)| *s == key.0 && *e == key.1 && *p == parent).map(|w| w.3.clone());
                        let atom = match known {
                            Some(atom) => atom,
                            None => {
                                let atom = crate::sym::Atom::Param(format!("dyn#{}", self.fresh()));
                                self.windows.push((key.0, key.1, parent, atom.clone()));
                                atom
                            }
                        };
                        extents[axis] = Some(Sym::atom(atom));
                    }
                    ir::Index::Slice { start, end }
                }
            });
        }
        index_with(storage, lowered, &extents, base.span).map_err(|e| format!("{e} in `{}`", f.definition.name))
    }

    fn bound(&mut self, f: &mut Frame<'a>, e: &'a Option<sir::Expr>, origin: &Sym, out: &mut Vec<Stmt>) -> Result<Option<Expr>, String> {
        let Some(e) = e else { return Ok(None) };
        let value = coerce(self.scalar(f, e, out)?, DType::I32);
        Ok(Some(match &value.sym {
            Some(sym) => symbol(sym.sub(origin), e.span),
            None if origin.is_zero() => value,
            None => binary(BinaryOp::Sub, value, symbol(origin.clone(), e.span), Ty::Scalar(DType::I32)),
        }))
    }

    // ---- tiles ----

    pub fn allocate(&mut self, name: &str, shaped: Shaped, span: Span, out: &mut Vec<Stmt>) -> Expr {
        let target = self.local(name, Ty::Tile(shaped.clone()), span);
        let value = Expr { kind: ExprKind::TileAlloc { shape: shaped.shape, dtype: shaped.elem }, ty: target.ty.clone(), sym: None, span };
        out.push(assign(target.clone(), value));
        if let ExprKind::Var(id) = target.kind {
            self.temporaries.insert(id);
        }
        target
    }

    /// Value snapshot of a view in its own representation.
    pub fn load(&mut self, view: Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        let shaped = tile_shape(&view)?.clone();
        let span = view.span;
        let target = self.local("loaded", Ty::Tile(shaped), span);
        let value = Expr { kind: ExprKind::Load { view: Box::new(view), mode: LoadMode::Materialize }, ty: target.ty.clone(), sym: None, span };
        out.push(assign(target.clone(), value));
        if let ExprKind::Var(id) = target.kind {
            self.temporaries.insert(id);
        }
        Ok(target)
    }

    /// An independent dense tile with the same elements.
    pub fn copy(&mut self, source: Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        if matches!(source.ty, Ty::Tensor(_)) {
            return self.load(source, out);
        }
        let shaped = tile_shape(&source)?.clone();
        let span = source.span;
        let target = self.local("copy", Ty::Tile(shaped), span);
        out.push(assign(target.clone(), source));
        if let ExprKind::Var(id) = target.kind {
            self.temporaries.insert(id);
        }
        Ok(target)
    }

    pub fn coordinates(&mut self, rank: usize, span: Span) -> (Vec<ir::VarId>, Vec<Expr>) {
        (0..rank).map(|_| self.index("c", span)).map(|(id, _, e)| (id, e)).unzip()
    }

    /// The fourth argument of the execution IR's reduction is its numerical contract: `true`
    /// keeps the authored ascending order, `false` (only from `unordered=true` inside an
    /// `admit fn`) permits the backend to reassociate.
    fn reduce(&mut self, operand: Expr, axis: usize, op: sir::ReduceOp, unordered: bool, span: Span, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        // The backend reduces tile variables only.
        let operand = if matches!(operand.kind, ExprKind::Var(_)) { operand } else { self.copy(operand, out)? };
        let source = tile_shape(&operand)?.clone();
        if axis >= source.shape.len() {
            return Err(format!("reduction axis {axis} is outside rank {}", source.shape.len()));
        }
        let tag = match op {
            sir::ReduceOp::Sum => ir::ReduceOp::Sum,
            sir::ReduceOp::Max => ir::ReduceOp::Max,
            sir::ReduceOp::Min => ir::ReduceOp::Min,
            sir::ReduceOp::Argmax => ir::ReduceOp::Argmax,
        };
        let dtype = if tag == ir::ReduceOp::Argmax { DType::I32 } else { source.elem.read_dtype().ok_or("reduction of an unbound element type")? };
        let mut shape = source.shape.clone();
        shape.remove(axis);
        let ty = if shape.is_empty() { Ty::Scalar(dtype) } else { Ty::Tile(Shaped::new(shape, Elem::Dtype(dtype))) };
        let target = self.local("reduced", ty.clone(), span);
        let args = vec![
            operand,
            int(axis as i64, span),
            Expr { kind: ExprKind::Int(tag as i64), ty: Ty::Scalar(DType::I32), sym: None, span },
            Expr { kind: ExprKind::Bool(!unordered), ty: Ty::Scalar(DType::Bool), sym: None, span },
        ];
        out.push(assign(target.clone(), Expr { kind: ExprKind::Builtin { name: Builtin::Reduce, args }, ty, sym: None, span }));
        Ok(target)
    }

    /// A fresh tile variable holding a tile-valued expression: one element loop.
    pub fn materialize(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, out: &mut Vec<Stmt>) -> Result<Expr, String> {
        let st::Ty::Tile(declared) = &e.ty else {
            return Err(format!("`{}` in `{}` is not tile-valued", e.ty, f.definition.name));
        };
        let same_representation = match &e.kind {
            sir::ExprKind::Cast { expr, .. } if shaped_type(&expr.ty) => expr.ty.shaped().map(|s| self.elem(f, &s.elem)).transpose()? == Some(self.elem(f, &declared.elem)?),
            _ => false,
        };
        // Snapshots and reductions take their shape from their operand, which may have a
        // runtime extent the declared type cannot name here.
        match &e.kind {
            sir::ExprKind::Load(view) => {
                let view = self.reference(f, view, out)?;
                return self.load(view, out);
            }
            sir::ExprKind::Cast { expr, .. } if same_representation => {
                let view = self.reference(f, expr, out)?;
                return self.load(view, out);
            }
            sir::ExprKind::Reduce { value, axis, op, unordered } => {
                let operand = self.tile(f, value, out)?;
                return self.reduce(operand, *axis, *op, *unordered, e.span, out);
            }
            _ => {}
        }
        let shaped = self.shaped(f, declared)?;
        if matches!(e.kind, sir::ExprKind::TileAlloc) {
            return Ok(self.allocate("tile", shaped, e.span, out));
        }
        if matches!(shaped.elem, Elem::Repr(_)) {
            return Err(format!("`{}` computes a packed tile; encoding needs an explicit operation", f.definition.name));
        }
        let (vars, at) = self.coordinates(shaped.shape.len(), e.span);
        let group = BTreeSet::new();
        let reads = RefCell::new(Reads::default());
        let value = self.element(f, e, &Coordinate { at: &at, group: &group, reads: &reads, reversed: false }, out)?;
        let target = self.allocate("tile", shaped, e.span, out);
        let mut body = reads.into_inner().body;
        body.push(assign(points(target.clone(), &at)?, value));
        out.push(stmt(StmtKind::Owned { vars, tile: target.clone(), body }, e.span));
        Ok(target)
    }

    fn element_dtype(&self, f: &Frame<'a>, ty: &st::Ty) -> Result<DType, String> {
        match ty {
            st::Ty::Tile(s) | st::Ty::View(s) | st::Ty::Tensor(s) => self.elem(f, &s.elem)?.read_dtype().ok_or_else(|| "unbound element type".to_string()),
            other => self.dtype(f, other),
        }
    }

    /// The element of tile-valued `e` at a coordinate. Operands that are not elementwise
    /// (snapshots, reductions, calls) are evaluated once into `pre`, before the loop.
    pub fn element(&mut self, f: &mut Frame<'a>, e: &'a sir::Expr, c: &Coordinate<'_>, pre: &mut Vec<Stmt>) -> Result<Expr, String> {
        if !matches!(e.ty, st::Ty::Tile(_) | st::Ty::View(_) | st::Ty::Tensor(_)) {
            let mut inside = false;
            walk::each_expr(e, &mut |x| inside |= matches!(x.kind, sir::ExprKind::Var(v) if c.group.contains(&v)));
            if inside {
                return Err(format!("a fused interval of `{}` has a non-elementwise dependency between its units", f.definition.name));
            }
            return self.scalar(f, e, pre);
        }
        if let sir::ExprKind::Var(v) = &e.kind {
            if let Some(value) = f.overrides.get(v) {
                return Ok(value.clone());
            }
            let deferred = match f.var(*v)? {
                Value::Deferred(deferred) => Some(*deferred),
                _ => None,
            };
            if let Some(deferred) = deferred {
                return self.element(f, deferred, c, pre);
            }
        }
        let dtype = self.element_dtype(f, &e.ty)?;
        let ty = Ty::Scalar(dtype);
        let span = e.span;
        Ok(match &e.kind {
            sir::ExprKind::Filled { value, .. } => literal(dtype, *value, span),
            sir::ExprKind::Cast { dtype, expr } => {
                let inner = if shaped_type(&expr.ty) { self.snapshot_element(f, expr, c, pre)? } else { self.element(f, expr, c, pre)? };
                coerce(inner, *dtype)
            }
            sir::ExprKind::Load(view) => self.snapshot_element(f, view, c, pre)?,
            sir::ExprKind::Decode(view) => coerce(self.snapshot_element(f, view, c, pre)?, DType::F32),
            sir::ExprKind::Unary { op, expr } => Expr { kind: ExprKind::Unary { op: *op, expr: Box::new(self.element(f, expr, c, pre)?) }, ty, sym: None, span },
            sir::ExprKind::Binary { op, lhs, rhs } => binary(*op, self.element(f, lhs, c, pre)?, self.element(f, rhs, c, pre)?, ty),
            sir::ExprKind::Math { op, args } => {
                let args = args.iter().map(|a| self.element(f, a, c, pre)).collect::<Result<Vec<_>, _>>()?;
                Expr { kind: ExprKind::Builtin { name: math(*op), args }, ty, sym: None, span }
            }
            sir::ExprKind::Select { cond, then, els } => {
                let args = vec![self.element(f, cond, c, pre)?, coerce(self.element(f, then, c, pre)?, dtype), coerce(self.element(f, els, c, pre)?, dtype)];
                Expr { kind: ExprKind::Builtin { name: Builtin::Select, args }, ty, sym: None, span }
            }
            sir::ExprKind::Transpose(inner) => {
                // A transposed operand is read at another coordinate: it must be complete.
                self.complete(f, inner, c, true)?;
                let reversed: Vec<Expr> = c.at.iter().rev().cloned().collect();
                self.element(f, inner, &Coordinate { at: &reversed, group: c.group, reads: c.reads, reversed: !c.reversed }, pre)?
            }
            _ => {
                self.complete(f, e, c, false)?;
                let storage = self.reference(f, e, pre)?;
                match storage.kind {
                    ExprKind::Var(id) => {
                        let cached = c.reads.borrow().cached.get(&(id, c.reversed)).cloned();
                        match cached {
                            Some(read) => read,
                            None => {
                                let read = points(storage, c.at)?;
                                let local = self.local("element", read.ty.clone(), span);
                                let mut reads = c.reads.borrow_mut();
                                reads.body.push(assign(local.clone(), read));
                                reads.cached.insert((id, c.reversed), local.clone());
                                local
                            }
                        }
                    }
                    _ => points(storage, c.at)?,
                }
            }
        })
    }

    /// Operands read at other coordinates, or as a whole, must not be computed by the
    /// loop that reads them.
    fn complete(&self, f: &Frame<'a>, e: &sir::Expr, c: &Coordinate<'_>, remapped: bool) -> Result<(), String> {
        let whole = remapped || !matches!(e.kind, sir::ExprKind::Var(_));
        let mut inside = false;
        walk::each_expr(e, &mut |x| inside |= matches!(x.kind, sir::ExprKind::Var(v) if c.group.contains(&v)));
        if whole && inside {
            return Err(format!("a fused interval of `{}` has a non-elementwise dependency between its units", f.definition.name));
        }
        Ok(())
    }

    fn snapshot_element(&mut self, f: &mut Frame<'a>, view: &'a sir::Expr, c: &Coordinate<'_>, pre: &mut Vec<Stmt>) -> Result<Expr, String> {
        self.complete(f, view, c, true)?;
        let view = self.reference(f, view, pre)?;
        let tile = self.load(view, pre)?;
        if let ExprKind::Var(id) = tile.kind {
            self.temporaries.remove(&id);
        }
        points(tile, c.at)
    }

    // ---- value semantics ----

    fn mutable_root(&self, e: &Expr) -> bool {
        matches!(e.kind, ExprKind::Var(v) if self.mutable.contains(&v))
    }

    /// An immutable binding: aliases stable values, snapshots anything a later statement
    /// may change.
    pub fn snapshot(&mut self, value: Value<'a>, name: &str, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        Ok(match value {
            Value::Scalar(e) => {
                let stable = e.sym.is_some()
                    || matches!(e.kind, ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_))
                    || (matches!(e.kind, ExprKind::Var(_)) && !self.mutable_root(&e));
                if stable {
                    Value::Scalar(e)
                } else {
                    let target = self.local(name, e.ty.clone(), e.span);
                    out.push(assign(target.clone(), e));
                    Value::Scalar(target)
                }
            }
            Value::Shaped(e) => match e.kind {
                ExprKind::Var(id) if self.temporaries.remove(&id) => Value::Shaped(e),
                ExprKind::Var(_) if matches!(e.ty, Ty::Tile(_)) && self.mutable_root(&e) => {
                    let copy = self.copy(e, out)?;
                    if let ExprKind::Var(id) = copy.kind {
                        self.temporaries.remove(&id);
                    }
                    Value::Shaped(copy)
                }
                _ => Value::Shaped(e),
            },
            Value::Tuple(items) => Value::Tuple(items.into_iter().map(|v| self.snapshot(v, name, out)).collect::<Result<_, _>>()?),
            other => other,
        })
    }

    /// Mutable state: storage owned by the binding alone.
    pub fn own(&mut self, value: Value<'a>, name: &str, out: &mut Vec<Stmt>) -> Result<Value<'a>, String> {
        Ok(match value {
            Value::Scalar(e) => {
                let target = self.local(name, e.ty.clone(), e.span);
                out.push(assign(target.clone(), e));
                if let ExprKind::Var(id) = target.kind {
                    self.mutable.insert(id);
                }
                Value::Scalar(target)
            }
            Value::Shaped(e) if matches!(e.ty, Ty::Tile(_) | Ty::Tensor(_)) => {
                let owned = match e.kind {
                    ExprKind::Var(id) if self.temporaries.remove(&id) => e,
                    _ => {
                        let copy = self.copy(e, out)?;
                        if let ExprKind::Var(id) = copy.kind {
                            self.temporaries.remove(&id);
                        }
                        copy
                    }
                };
                if let ExprKind::Var(id) = owned.kind {
                    self.mutable.insert(id);
                }
                Value::Shaped(owned)
            }
            Value::Tuple(items) => Value::Tuple(items.into_iter().map(|v| self.own(v, name, out)).collect::<Result<_, _>>()?),
            other => other,
        })
    }
}

fn accessor_type(base: &Expr, name: &str) -> Result<Ty, String> {
    let Ty::Tile(shaped) = &base.ty else {
        return Err(format!("`.{name}` needs a packed tile, found {}", base.ty));
    };
    let Elem::Repr(repr) = &shaped.elem else {
        return Err(format!("`.{name}` needs a packed tile, found element type {}", shaped.elem));
    };
    let repr = crate::repr::lookup(repr).ok_or_else(|| format!("unknown representation `{repr}`"))?;
    let axis = shaped.packed_axis.ok_or("this packed tile has no packet axis left")?;
    let extent = &shaped.shape[axis];
    let (extent, dtype) = if name == "scale" || name == "bias" {
        if name == "bias" && !repr.has_bias() {
            return Err(format!("`{}` has no bias", repr.name));
        }
        (repr.groups_extent(extent), repr.coefficient_dtype())
    } else {
        let plane = repr.plane(name).ok_or_else(|| format!("`{}` has no physical plane `{name}`", repr.name))?;
        (plane.extent(extent), plane.dtype())
    };
    let mut shape = shaped.shape.clone();
    shape[axis] = extent;
    Ok(Ty::Tile(Shaped::new(shape, Elem::Dtype(dtype))))
}
