//! Expression evaluation and shaped storage access.
use super::exec::Frame;
use super::scalar;
use super::value::{Backing, Flow, Piece, Shaped, Value, S};
use super::{round_to, Interpreter, TensorData};
use crate::sir::{Expr, ExprKind, Index, Math, ReduceOp};
use crate::syntax::ast::{AssignOp, BinaryOp};
use crate::types::{DType, Elem};
use crate::types::{Extent, Ty};

/// An elementwise operand: a broadcast scalar or row-major tile data.
pub(super) enum Operand {
    Scalar(S),
    Tile {
        dtype: DType,
        shape: Vec<usize>,
        data: Vec<f64>,
    },
}

pub(super) fn hint(ty: &Ty) -> Option<DType> {
    match ty {
        Ty::Scalar(d) => Some(*d),
        Ty::Index(_) => Some(DType::I32),
        Ty::Tensor(s) | Ty::View(s) | Ty::Tile(s) => match s.elem {
            Elem::Dtype(d) => Some(d),
            _ => None,
        },
        _ => None,
    }
}

impl<'a> Interpreter<'a> {
    // ----- storage ---------------------------------------------------------------------

    pub(super) fn dtype_of(&self, s: &Shaped) -> DType {
        match &s.backing {
            Backing::Tensor(id) => match &self.tensors[*id] {
                TensorData::Dense { dtype, .. } => *dtype,
                TensorData::Packed { .. } => DType::F32,
            },
            Backing::Owned(rc) => rc.borrow().dtype,
        }
    }

    pub(super) fn elem_of(&self, s: &Shaped) -> Elem {
        match &s.backing {
            Backing::Tensor(id) => match &self.tensors[*id] {
                TensorData::Dense { dtype, .. } => Elem::Dtype(*dtype),
                TensorData::Packed { repr, .. } => Elem::Repr(repr.name.into()),
            },
            Backing::Owned(rc) => Elem::Dtype(rc.borrow().dtype),
        }
    }

    pub(super) fn read_flat(&self, s: &Shaped, flat: usize) -> Result<f64, String> {
        match &s.backing {
            Backing::Tensor(id) => {
                let t = &self.tensors[*id];
                if flat >= t.shape().iter().product::<usize>() {
                    return Err("read outside tensor bounds".into());
                }
                Ok(t.get(flat))
            }
            Backing::Owned(rc) => {
                let dense = rc.borrow();
                match dense.init.get(flat) {
                    Some(true) => Ok(dense.data[flat]),
                    Some(false) => Err("read of an uninitialized tile element".into()),
                    None => Err("read outside tile bounds".into()),
                }
            }
        }
    }

    pub(super) fn write_flat(&mut self, s: &Shaped, flat: usize, value: S) -> Result<(), String> {
        match &s.backing {
            Backing::Tensor(id) => match &mut self.tensors[*id] {
                TensorData::Packed { .. } => Err("cannot store into a packed tensor".into()),
                TensorData::Dense { dtype, data, .. } => {
                    let slot = data.get_mut(flat).ok_or("write outside tensor bounds")?;
                    *slot = scalar::cast(*dtype, value).1;
                    Ok(())
                }
            },
            Backing::Owned(rc) => {
                let mut dense = rc.borrow_mut();
                if flat >= dense.data.len() {
                    return Err("write outside tile bounds".into());
                }
                dense.data[flat] = scalar::cast(dense.dtype, value).1;
                dense.init[flat] = true;
                Ok(())
            }
        }
    }

    /// Row-major decoded values of a selection; every element must be initialized.
    pub(super) fn gather(&self, s: &Shaped) -> Result<Vec<f64>, String> {
        s.flats()
            .into_iter()
            .map(|flat| self.read_flat(s, flat))
            .collect()
    }

    pub(super) fn operand(&self, v: &Value) -> Result<Operand, String> {
        match v {
            Value::Scalar(d, x) => Ok(Operand::Scalar((*d, *x))),
            Value::Tile(s) | Value::View(s) => Ok(Operand::Tile {
                dtype: self.dtype_of(s),
                shape: s.shape.clone(),
                data: self.gather(s)?,
            }),
            other => Err(format!("{} is not a numerical operand", other.kind())),
        }
    }

    pub(super) fn elementwise(
        &self,
        operands: &[Operand],
        fallback: DType,
        fun: &mut dyn FnMut(&[S]) -> Result<S, String>,
    ) -> Result<Value, String> {
        let mut shape: Option<&Vec<usize>> = None;
        for o in operands {
            if let Operand::Tile { shape: s, .. } = o {
                if shape.is_some_and(|p| p != s) {
                    return Err(format!(
                        "elementwise shape mismatch: {:?} vs {s:?}",
                        shape.unwrap_or(s)
                    ));
                }
                shape = Some(s);
            }
        }
        let at = |i: usize| -> Vec<S> {
            operands
                .iter()
                .map(|o| match o {
                    Operand::Scalar(s) => *s,
                    Operand::Tile { dtype, data, .. } => (*dtype, data[i]),
                })
                .collect()
        };
        let Some(shape) = shape else {
            return fun(&at(0)).map(Value::scalar);
        };
        let n: usize = shape.iter().product();
        let mut dtype = fallback;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let (d, x) = fun(&at(i))?;
            dtype = d;
            data.push(x);
        }
        Ok(Value::Tile(Shaped::owned(dtype, shape.clone(), data)))
    }

    /// Write `value` through `dst` (`dst op= value`), rounding to the destination dtype.
    /// A scalar broadcasts. Sources are read completely before the first write.
    pub(super) fn write(
        &mut self,
        dst: &Shaped,
        op: AssignOp,
        value: &Value,
    ) -> Result<(), String> {
        let source = self.operand(value)?;
        if let Operand::Tile { shape, .. } = &source {
            if shape != &dst.shape {
                return Err(format!(
                    "shape mismatch: value {shape:?} vs destination {:?}",
                    dst.shape
                ));
            }
        }
        let dtype = self.dtype_of(dst);
        let flats = dst.flats();
        let current = if op == AssignOp::Assign {
            Vec::new()
        } else {
            self.gather(dst)?
        };
        for (i, flat) in flats.into_iter().enumerate() {
            let v = match &source {
                Operand::Scalar(s) => *s,
                Operand::Tile { dtype, data, .. } => (*dtype, data[i]),
            };
            let v = if op == AssignOp::Assign {
                v
            } else {
                scalar::assign(op, (dtype, current[i]), v)?
            };
            self.write_flat(dst, flat, v)?;
        }
        Ok(())
    }

    /// Value semantics at a binding, yield, port or return: an owned tile that anything else
    /// can still reach is copied. Views stay borrows; packed snapshots are immutable.
    pub(super) fn snapshot(&self, v: Value) -> Result<Value, String> {
        Ok(match v {
            Value::Tile(s) if matches!(s.backing, Backing::Owned(_)) && !s.exclusive() => {
                Value::Tile(Shaped::owned(
                    self.dtype_of(&s),
                    s.shape.clone(),
                    self.gather(&s)?,
                ))
            }
            Value::Tuple(items) => Value::Tuple(
                items
                    .into_iter()
                    .map(|i| self.snapshot(i))
                    .collect::<Result<_, _>>()?,
            ),
            other => other,
        })
    }

    // ----- symbols and geometry --------------------------------------------------------

    pub(super) fn piece(
        &self,
        f: &Frame<'a>,
        slice: crate::types::SliceId,
    ) -> Result<Piece, String> {
        f.slices
            .get(slice.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("slice#{} is not bound here", slice.0))
    }

    pub(super) fn extent(&self, e: &Extent, f: &Frame<'a>) -> Result<usize, String> {
        let n = match e {
            Extent::Semantic(s) => f.sym(s)?,
            Extent::Structural(s) => self.piece(f, *s)?.extent(),
        };
        usize::try_from(n).map_err(|_| format!("negative extent {n}"))
    }

    /// Semantic coordinate of position zero of `axis` of a value of static type `ty`.
    fn axis_base(&self, ty: &Ty, axis: usize, f: &Frame<'a>) -> Result<i64, String> {
        match ty.shaped().and_then(|s| s.axes.get(axis)) {
            Some(Extent::Structural(s)) => Ok(self.piece(f, *s)?.lo),
            _ => Ok(0),
        }
    }

    // ----- expressions -----------------------------------------------------------------

    pub(super) fn scalar(&mut self, e: &'a Expr, f: &mut Frame<'a>) -> Result<S, String> {
        match self.expr(e, f)? {
            Value::Scalar(d, x) => Ok((d, x)),
            other => Err(format!("expected a scalar, found {}", other.kind())),
        }
    }

    pub(super) fn int(&mut self, e: &'a Expr, f: &mut Frame<'a>) -> Result<i64, String> {
        match self.scalar(e, f)? {
            (d, x) if d.is_int() => Ok(x as i64),
            (d, _) => Err(format!("expected an integer, found {}", d.name())),
        }
    }

    /// The storage an expression designates, without reading it.
    pub(super) fn place(&mut self, e: &'a Expr, f: &mut Frame<'a>) -> Result<Shaped, String> {
        match &e.kind {
            ExprKind::Range { .. } => return Err("a range value is only consumed by a loop".into()),
            ExprKind::Index { base, indices } => {
                let b = self.place(base, f)?;
                self.index(b, &base.ty, &e.ty, indices, f)
            }
            ExprKind::Transpose(inner) => self.place(inner, f)?.transposed(),
            _ => match self.expr(e, f)? {
                Value::Tile(s) | Value::View(s) => Ok(s),
                other => Err(format!("{} is not storage", other.kind())),
            },
        }
    }

    /// The checker's atom for the extent of a runtime-bounded range (`@dyn#n`), if `axis` of
    /// `ty` is exactly one.
    fn dynamic_atom(ty: &Ty, axis: usize) -> Option<String> {
        let Extent::Semantic(sym) = ty.shaped()?.axes.get(axis)? else {
            return None;
        };
        sym.atoms().into_iter().find_map(|a| match a {
            crate::sym::Atom::Param(p)
                if p.starts_with('@') && sym == &crate::sym::Sym::param(&p) =>
            {
                Some(p)
            }
            _ => None,
        })
    }

    fn index(
        &mut self,
        b: Shaped,
        bt: &Ty,
        rt: &Ty,
        indices: &'a [Index],
        f: &mut Frame<'a>,
    ) -> Result<Shaped, String> {
        if indices.len() > b.shape.len() {
            return Err(format!(
                "{} indices into a rank-{} value",
                indices.len(),
                b.shape.len()
            ));
        }
        let mut out = Shaped {
            backing: b.backing.clone(),
            shape: Vec::new(),
            strides: Vec::new(),
            offset: b.offset,
        };
        for (axis, extent) in b.shape.iter().copied().enumerate() {
            let stride = b.strides[axis];
            let Some(index) = indices.get(axis) else {
                out.shape.push(extent);
                out.strides.push(stride);
                continue;
            };
            let base = self.axis_base(bt, axis, f)?;
            let point = |v: i64| -> Result<usize, String> {
                usize::try_from(v - base).ok().filter(|p| *p < extent).ok_or_else(|| format!("index {v} outside axis {axis} of extent {extent} (first coordinate {base})"))
            };
            let (lo, hi) = match index {
                Index::Point(p) => {
                    out.offset += point(self.int(p, f)?)? * stride;
                    continue;
                }
                Index::Coord(var) => {
                    let v = match f.vars.get(*var).and_then(|v| v.as_ref()) {
                        Some(Value::Scalar(_, x)) => *x as i64,
                        _ => return Err("tile coordinate is not bound".into()),
                    };
                    out.offset += point(v)? * stride;
                    continue;
                }
                Index::Slice(s) => {
                    let p = self.piece(f, *s)?;
                    (p.lo - base, p.hi - base)
                }
                Index::Range { start, end } => {
                    let lo = match start {
                        Some(x) => self.int(x, f)? - base,
                        None => 0,
                    };
                    let hi = match end {
                        Some(x) => self.int(x, f)? - base,
                        None => extent as i64,
                    };
                    (lo, hi)
                }
            };
            // A runtime-bounded range is clamped to its axis; its realized length is the value
            // of the checker's extent atom from here on.
            let dynamic = if matches!(index, Index::Range { .. }) {
                Self::dynamic_atom(rt, out.shape.len())
            } else {
                None
            };
            let (lo, hi) = match &dynamic {
                Some(_) => {
                    let hi = hi.clamp(0, extent as i64);
                    (lo.clamp(0, hi), hi)
                }
                None => (lo, hi),
            };
            if let Some(atom) = dynamic {
                f.dynamic.insert(atom, hi - lo);
            }
            if lo < 0 || hi < lo || hi > extent as i64 {
                return Err(format!("selection {}..{} outside axis {axis} of extent {extent} (first coordinate {base})", lo + base, hi + base));
            }
            out.offset += lo as usize * stride;
            out.shape.push((hi - lo) as usize);
            out.strides.push(stride);
        }
        Ok(out)
    }

    pub(super) fn expr(&mut self, e: &'a Expr, f: &mut Frame<'a>) -> Result<Value, String> {
        match &e.kind {
            ExprKind::Int(v) => {
                let d = hint(&e.ty).unwrap_or(DType::I32);
                Ok(Value::Scalar(
                    d,
                    if d.is_float() {
                        round_to(d, *v as f64)
                    } else {
                        *v as f64
                    },
                ))
            }
            ExprKind::Float(v) => {
                let d = hint(&e.ty).filter(|d| d.is_float()).unwrap_or(DType::F32);
                Ok(Value::Scalar(d, round_to(d, *v)))
            }
            ExprKind::Bool(b) => Ok(Value::Scalar(DType::Bool, u8::from(*b) as f64)),
            ExprKind::Var(id) => f.vars.get(*id).and_then(|v| v.clone()).ok_or_else(|| {
                format!("`{}` is read before it has a value", f.body.vars[*id].name)
            }),
            ExprKind::ShapeParam(name) => f
                .shapes
                .get(name)
                .map(|v| Value::int(*v))
                .ok_or_else(|| format!("shape parameter {name} is unbound")),
            ExprKind::Tuple(items) => Ok(Value::Tuple(
                items
                    .iter()
                    .map(|i| self.expr(i, f))
                    .collect::<Result<_, _>>()?,
            )),
            ExprKind::Range { .. } => Err("a range value is only consumed by a loop".into()),
            ExprKind::Field { base, index } => match self.expr(base, f)? {
                Value::Tuple(mut items) if *index < items.len() => Ok(items.swap_remove(*index)),
                other => Err(format!("component {index} of {}", other.kind())),
            },
            ExprKind::TileAlloc => {
                let Ty::Tile(s) = &e.ty else {
                    return Err("tile allocation without a tile type".into());
                };
                let shape = s
                    .axes
                    .iter()
                    .map(|a| self.extent(a, f))
                    .collect::<Result<Vec<_>, _>>()?;
                match f.elem(&s.elem) {
                    Elem::Dtype(d) => Ok(Value::Tile(Shaped::uninit(d, shape))),
                    other => Err(format!(
                        "local tile allocation requires a dense dtype, found {other}"
                    )),
                }
            }
            ExprKind::Filled { like, value } => {
                let like = self.place(like, f)?;
                let dtype = match e.ty.shaped().map(|s| f.elem(&s.elem)) {
                    Some(Elem::Dtype(d)) => d,
                    _ => self.dtype_of(&like),
                };
                let n = like.count();
                Ok(Value::Tile(Shaped::owned(
                    dtype,
                    like.shape,
                    vec![round_to(dtype, *value); n],
                )))
            }
            ExprKind::Index { base, indices } => {
                let b = self.place(base, f)?;
                let element = indices.len() == b.shape.len()
                    && indices
                        .iter()
                        .all(|i| matches!(i, Index::Point(_) | Index::Coord(_)));
                let s = self.index(b, &base.ty, &e.ty, indices, f)?;
                if element {
                    Ok(Value::Scalar(
                        self.dtype_of(&s),
                        self.read_flat(&s, s.offset)?,
                    ))
                } else {
                    Ok(Value::View(s))
                }
            }
            ExprKind::Member { result, slices } => {
                let Value::Result(r) = self.expr(result, f)? else {
                    return Err("member selection on a value that is not a region result".into());
                };
                let at = slices
                    .iter()
                    .map(|s| self.piece(f, *s))
                    .collect::<Result<Vec<_>, _>>()?;
                r.member(&at).cloned()
            }
            ExprKind::Transpose(inner) => Ok(match self.expr(inner, f)? {
                Value::Tile(s) => Value::Tile(s.transposed()?),
                Value::View(s) => Value::View(s.transposed()?),
                other => return Err(format!("transpose of {}", other.kind())),
            }),
            ExprKind::Reshape { base, axes } => {
                let v = self.expr(base, f)?;
                let Some(s) = v.shaped() else {
                    return Err(format!("reshape of {}", v.kind()));
                };
                if matches!(self.elem_of(s), Elem::Repr(_)) {
                    return Err("reshape requires dense storage".into());
                }
                let target = axes
                    .iter()
                    .map(|a| self.extent(a, f).map(|n| n as i64))
                    .collect::<Result<Vec<_>, _>>()?;
                let signed = |v: &[usize]| {
                    v.iter()
                        .map(|n| {
                            i64::try_from(*n).map_err(|_| "reshape extent overflow".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()
                };
                let strides = crate::layout::reshape_strides(
                    &signed(&s.shape)?,
                    &signed(&s.strides)?,
                    &target,
                )?;
                let out = Shaped {
                    backing: s.backing.clone(),
                    shape: target.into_iter().map(|n| n as usize).collect(),
                    strides: strides.into_iter().map(|n| n as usize).collect(),
                    offset: s.offset,
                };
                Ok(if matches!(v, Value::Tile(_)) {
                    Value::Tile(out)
                } else {
                    Value::View(out)
                })
            }
            ExprKind::Load(view) => {
                let s = self.place(view, f)?;
                if matches!(self.elem_of(&s), Elem::Repr(_)) {
                    return Ok(Value::Tile(s));
                }
                Ok(Value::Tile(Shaped::owned(
                    self.dtype_of(&s),
                    s.shape.clone(),
                    self.gather(&s)?,
                )))
            }
            ExprKind::Decode(view) => {
                let s = self.place(view, f)?;
                let data = self
                    .gather(&s)?
                    .into_iter()
                    .map(|x| round_to(DType::F32, x))
                    .collect();
                Ok(Value::Tile(Shaped::owned(DType::F32, s.shape, data)))
            }
            ExprKind::Cast { dtype, expr } => {
                let v = self.expr(expr, f)?;
                let operand = self.operand(&v)?;
                self.elementwise(&[operand], *dtype, &mut |a| Ok(scalar::cast(*dtype, a[0])))
            }
            ExprKind::Unary { op, expr } => {
                let v = self.expr(expr, f)?;
                let operand = self.operand(&v)?;
                self.elementwise(&[operand], hint(&e.ty).unwrap_or(DType::F32), &mut |a| {
                    scalar::unary(*op, a[0])
                })
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.expr(lhs, f)?;
                // Scalar logic short-circuits, so a guard protects its right operand.
                if let (BinaryOp::And | BinaryOp::Or, Value::Scalar(DType::Bool, x)) = (op, &l) {
                    if (*x != 0.0) == (*op == BinaryOp::Or) {
                        return Ok(l);
                    }
                }
                let r = self.expr(rhs, f)?;
                let h = hint(&e.ty);
                let operands = [self.operand(&l)?, self.operand(&r)?];
                self.elementwise(&operands, h.unwrap_or(DType::F32), &mut |a| {
                    scalar::binary(*op, a[0], a[1], h)
                })
            }
            ExprKind::Math { op, args } => self.math(*op, args, e, f),
            ExprKind::Select { cond, then, els } => {
                let c = self.expr(cond, f)?;
                let t = self.expr(then, f)?;
                let n = self.expr(els, f)?;
                if let Value::Scalar(d, x) = &c {
                    if *d != DType::Bool {
                        return Err("select requires a bool condition".into());
                    }
                    if !matches!(t, Value::Scalar(..)) || !matches!(n, Value::Scalar(..)) {
                        return Ok(if *x != 0.0 { t } else { n });
                    }
                }
                let operands = [self.operand(&c)?, self.operand(&t)?, self.operand(&n)?];
                self.elementwise(&operands, hint(&e.ty).unwrap_or(DType::F32), &mut |a| {
                    if a[0].0 != DType::Bool {
                        return Err("select requires a bool mask".into());
                    }
                    let d = DType::promote(a[1].0, a[2].0).unwrap_or(a[1].0);
                    Ok(scalar::cast(d, if a[0].1 != 0.0 { a[1] } else { a[2] }))
                })
            }
            ExprKind::Reduce {
                value, axis, op, ..
            } => {
                let s = self.place(value, f)?;
                self.reduce(&s, *axis, *op)
            }
            ExprKind::CoordOf(var) => match f.vars.get(*var).and_then(|v| v.as_ref()) {
                Some(Value::Scalar(_, x)) => Ok(Value::int(*x as i64)),
                _ => Err("coord() of an unbound tile coordinate".into()),
            },
            ExprKind::ExtentOf { base, axis } => {
                let s = self.place(base, f)?;
                s.shape
                    .get(*axis)
                    .map(|n| Value::int(*n as i64))
                    .ok_or_else(|| {
                        format!("extent of axis {axis} of a rank-{} value", s.shape.len())
                    })
            }
            ExprKind::Geometry { base, axis, valid } => {
                let s = self.place(base, f)?;
                let extent = *s.shape.get(*axis).ok_or_else(|| {
                    format!("geometry of axis {axis} of a rank-{} value", s.shape.len())
                })? as i64;
                if *valid {
                    return Ok(Value::int(extent));
                }
                let capacity = match base.ty.shaped().and_then(|t| t.axes.get(*axis)) {
                    Some(Extent::Structural(slice)) => self.piece(f, *slice)?.width,
                    Some(Extent::Semantic(sym)) => f
                        .caps
                        .iter()
                        .find(|(p, _)| sym == &crate::sym::Sym::param(p))
                        .map(|(_, piece)| piece.width)
                        .unwrap_or(extent),
                    None => extent,
                };
                Ok(Value::int(capacity.max(extent)))
            }
            ExprKind::Call { call, args } => self.call(*call, args, f),
            ExprKind::Region(region) => match self.region(region, f)? {
                Flow::Yield(v) => Ok(v),
                _ => Err("region expression produced no value".into()),
            },
            ExprKind::Intrinsic { op, args } => self.intrinsic(*op, args, e, f),
            ExprKind::Accessor { base, name } => {
                let s = self.place(base, f)?;
                let axis = base
                    .ty
                    .shaped()
                    .and_then(|t| t.packed_axis)
                    .unwrap_or(s.shape.len().saturating_sub(1));
                self.accessor(&s, axis, name)
            }
            ExprKind::Atomic { op, place, value } => {
                let dst = self.place(place, f)?;
                if !dst.shape.is_empty() {
                    return Err("atomic update of a place that is not one element".into());
                }
                let v = self.scalar(value, f)?;
                let d = self.dtype_of(&dst);
                let current = (d, self.read_flat(&dst, dst.offset)?);
                let updated = scalar::binary(*op, current, v, Some(d))?;
                self.write_flat(&dst, dst.offset, updated)?;
                Ok(Value::Void)
            }
        }
    }

    fn math(
        &mut self,
        op: Math,
        args: &'a [Expr],
        e: &'a Expr,
        f: &mut Frame<'a>,
    ) -> Result<Value, String> {
        let mut operands = Vec::with_capacity(args.len());
        for a in args {
            let v = self.expr(a, f)?;
            operands.push(self.operand(&v)?);
        }
        self.elementwise(&operands, hint(&e.ty).unwrap_or(DType::F32), &mut |a| {
            scalar::math(op, a)
        })
    }

    /// Reduction along one axis in ascending index order. Sums round every step to the
    /// accumulation dtype; max/min/argmax keep the smaller index on ties.
    fn reduce(&self, s: &Shaped, axis: usize, op: ReduceOp) -> Result<Value, String> {
        if axis >= s.shape.len() {
            return Err(format!(
                "reduce along axis {axis} of a rank-{} value",
                s.shape.len()
            ));
        }
        // Floating reductions are carried in f32 whatever the operand's element type.
        let input = if self.dtype_of(s).is_float() {
            DType::F32
        } else {
            self.dtype_of(s)
        };
        let data = self.gather(s)?;
        let operation = match op {
            ReduceOp::Sum => crate::exec::ir::ReduceOp::Sum,
            ReduceOp::Max => crate::exec::ir::ReduceOp::Max,
            ReduceOp::Min => crate::exec::ir::ReduceOp::Min,
            ReduceOp::Argmax => crate::exec::ir::ReduceOp::Argmax,
        };
        let contract = crate::exec::reduction::Contract::new(operation, input, true);
        let extent = s.shape[axis];
        if extent == 0 && !contract.allows_empty_axis() {
            return Err("argmax requires a nonempty axis".into());
        }
        let outer: usize = s.shape[..axis].iter().product();
        let inner: usize = s.shape[axis + 1..].iter().product();
        let mut out = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = contract.identity().value();
                let mut arg = 0usize;
                for k in 0..extent {
                    let x = data[(o * extent + k) * inner + i];
                    match op {
                        ReduceOp::Sum => acc = round_to(input, acc + x),
                        ReduceOp::Max => acc = acc.max(x),
                        ReduceOp::Min => acc = acc.min(x),
                        ReduceOp::Argmax => {
                            if x > acc {
                                acc = x;
                                arg = k;
                            }
                        }
                    }
                }
                out.push(if op == ReduceOp::Argmax {
                    arg as f64
                } else {
                    acc
                });
            }
        }
        let mut shape = s.shape.clone();
        shape.remove(axis);
        let dtype = contract.output();
        Ok(if shape.is_empty() {
            Value::Scalar(dtype, out[0])
        } else {
            Value::Tile(Shaped::owned(dtype, shape, out))
        })
    }
}
